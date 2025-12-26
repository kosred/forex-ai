import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist  # DDP Support
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..core.system import thread_limits
from .base import EarlyStopper, ExpertModel, dataframe_to_float32_numpy
from .device import select_device

logger = logging.getLogger(__name__)


class NBeatsBlock(nn.Module):
    def __init__(self, hidden_dim: int, theta_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.theta_b = nn.Linear(hidden_dim, theta_dim, bias=False)
        self.theta_f = nn.Linear(hidden_dim, theta_dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        h = self.act(self.fc4(h))
        return self.theta_b(h), self.theta_f(h)


class NBeatsExpert(ExpertModel):
    """
    N-BEATS Implementation (flexible device).
    Features: Quantization, JIT, Thread Limiting (if CPU).
    """

    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 256

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_blocks: int = 2,
        lr: float = 1e-3,
        max_time_sec: int = 1800,
        device: str = "auto",
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        self.model = None
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.lr = lr
        self.max_time_sec = max_time_sec
        self.device = select_device(device)
        self.batch_size = batch_size
        self.input_dim = 0

    def _build_model(self) -> nn.Module:
        class NBeatsNet(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([NBeatsBlock(hidden_dim, hidden_dim) for _ in range(3)])
                self.final = nn.Linear(hidden_dim, output_dim)
                self.embed = nn.Linear(input_dim, hidden_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.embed(x)
                forecast = torch.zeros_like(x)
                for block in self.blocks:
                    backcast, fore = block(x)
                    x = x - backcast
                    forecast = forecast + fore
                return self.final(forecast)

        return NBeatsNet(self.input_dim, self.hidden_dim).to(self.device)

    def fit(self, X: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        # Robust Normalization
        x_np = dataframe_to_float32_numpy(X)
        self.mean_ = np.mean(x_np, axis=0)
        self.scale_ = np.maximum(np.std(x_np, axis=0), 1e-3)
        x_norm = (x_np - self.mean_) / self.scale_

        self.input_dim = x_norm.shape[1]
        self.model = self._build_model()

        # Use thread limits only if on CPU
        is_cpu = str(self.device) == "cpu"

        split = int(len(x_norm) * 0.85)
        # Use normalized data
        X_train_norm, X_val_norm = x_norm[:split], x_norm[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        X_t = torch.as_tensor(X_train_norm, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_train.values + 1, dtype=torch.long, device=self.device)
        X_v = torch.as_tensor(X_val_norm, dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(y_val.values + 1, dtype=torch.long, device=self.device)

        dataset = TensorDataset(X_t, y_t)

        # DDP Sampler
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        pin_mem = str(self.device) != "cpu"
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            pin_memory=pin_mem
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        early_stopper = EarlyStopper(patience=8, min_delta=0.001)

        import time

        start = time.time()
        self.model.train()

        # Context manager helper
        import multiprocessing
        from contextlib import nullcontext

        ctx = thread_limits(blas_threads=max(1, multiprocessing.cpu_count() - 1)) if is_cpu else nullcontext()

        with ctx:
            for epoch in range(100):
                if sampler:
                    sampler.set_epoch(epoch)

                if time.time() - start > self.max_time_sec:
                    break

                for bx, by in loader:
                    optimizer.zero_grad()
                    out = self.model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

                if len(X_val_norm) > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(X_v)
                        val_loss = criterion(val_out, y_v).item()
                    self.model.train()
                    if early_stopper(val_loss):
                        break

        # Post-Training Optimization for Inference (CPU only)
        if is_cpu:
            self._optimize_for_cpu_inference()

    def _optimize_for_cpu_inference(self):
        if self.model is None:
            return
        try:
            self.model.eval()
            dummy_input = torch.randn(1, self.input_dim)
            traced = torch.jit.trace(self.model, dummy_input, check_trace=False, strict=False)
            self.model = torch.jit.freeze(traced)
            logger.info("JIT Compiled NBeats for CPU.")
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(X), 3))
        self.model.eval()

        x_np = dataframe_to_float32_numpy(X)
        # Apply normalization
        if hasattr(self, "mean_") and self.mean_ is not None:
            x_norm = (x_np - self.mean_) / self.scale_
        else:
            x_norm = x_np

        outs = []
        inf_batch = self.batch_size * 4
        with torch.no_grad():
            for start in range(0, len(X), inf_batch):
                xb_chunk = x_norm[start : start + inf_batch]
                xb = torch.as_tensor(xb_chunk, dtype=torch.float32, device=self.device)
                logits = self.model(xb)
                outs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if not outs:
            return np.zeros((0, 3))
        probs = np.vstack(outs)
        # Training encodes labels as y+1 where y ∈ {-1,0,1} -> class indices {0,1,2}.
        # Reorder to project convention [neutral, buy, sell].
        return probs[:, [1, 2, 0]]

    def save(self, path: str) -> None:
        if self.model:
            # Save JIT optimized model if available
            if isinstance(self.model, torch.jit.ScriptModule):
                torch.jit.save(self.model, Path(path) / "nbeats_cpu.pt")
            else:
                torch.save(self.model.state_dict(), Path(path) / "nbeats.pt")
            joblib.dump({"input_dim": self.input_dim, "hidden": self.hidden_dim}, Path(path) / "nbeats_meta.pkl")

    def load(self, path: str) -> None:
        p_jit = Path(path) / "nbeats_cpu.pt"
        p_std = Path(path) / "nbeats.pt"
        m = Path(path) / "nbeats_meta.pkl"

        if not m.exists():
            return
        meta = joblib.load(m)
        self.input_dim = meta["input_dim"]
        self.hidden_dim = meta["hidden"]

        if p_jit.exists():
            try:
                self.model = torch.jit.load(str(p_jit), map_location="cpu")
                logger.info("Loaded JIT Optimized NBeats")
                return
            except Exception:
                logger.warning("Failed to load JIT model, falling back to state dict")

        if p_std.exists():
            self.model = self._build_model()
            try:
                state = torch.load(p_std, map_location="cpu", weights_only=True)
            except TypeError as exc:
                raise RuntimeError("PyTorch too old for weights_only load; upgrade for security.") from exc
            self.model.load_state_dict(state)
            self._optimize_for_cpu_inference()

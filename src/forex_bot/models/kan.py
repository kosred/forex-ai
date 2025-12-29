import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..core.system import thread_limits
from .base import EarlyStopper, ExpertModel, dataframe_to_float32_numpy, get_early_stop_params
from .device import select_device

logger = logging.getLogger(__name__)

# Optional full KAN (efficient-kan)
try:  # pragma: no cover
    from efficient_kan import KAN as EfficientKAN  # type: ignore

    _KAN_AVAILABLE = True
except Exception:  # pragma: no cover
    EfficientKAN = None  # type: ignore
    _KAN_AVAILABLE = False


class KANLayerBSpline(nn.Module):
    """
    Lite KAN-style layer using linear + GELU approximation.
    Faster on CPU than true B-splines.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KANExpert(ExpertModel):
    """
    CPU-Optimized KAN (Lite Version).
    Approximates KAN behavior for speed.
    """

    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 256

    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_time_sec: int = 36000,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.model = None
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_time_sec = max_time_sec
        self.device = select_device(device)
        self.input_dim = 0
        # Mixed precision on CUDA for faster training
        self._use_amp = str(self.device).startswith("cuda")

    def _build_model(self) -> nn.Module:
        if _KAN_AVAILABLE:
            try:
                # Full KAN with spline activations (efficient-kan)
                return EfficientKAN(
                    layer_sizes=[self.input_dim, max(self.hidden_dim, 256), max(self.hidden_dim, 256), 3],
                    grid_size=16,
                    spline_order=3,
                ).to(self.device)
            except Exception as exc:
                logger.warning(f"efficient-kan available but init failed, falling back to Lite KAN: {exc}")

        class KANNet(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3) -> None:
                super().__init__()
                self.layer1 = KANLayerBSpline(input_dim, hidden_dim)
                self.layer2 = KANLayerBSpline(hidden_dim, hidden_dim)
                self.final = nn.Linear(hidden_dim, output_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.layer1(x)
                x = self.layer2(x)
                return self.final(x)

        return KANNet(self.input_dim, self.hidden_dim).to(self.device)

    def fit(self, X: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        # Robust Normalization
        x_np = dataframe_to_float32_numpy(X)
        self.mean_ = np.mean(x_np, axis=0)
        self.scale_ = np.maximum(np.std(x_np, axis=0), 1e-3)
        x_norm = (x_np - self.mean_) / self.scale_

        # Align device to local rank if distributed
        if dist.is_available() and dist.is_initialized() and torch.cuda.is_available():
            try:
                local_rank = dist.get_rank() % torch.cuda.device_count()
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f"cuda:{local_rank}")
            except Exception:
                pass

        self.input_dim = x_norm.shape[1]
        self.model = self._build_model()

        import multiprocessing

        split = int(len(x_norm) * 0.85)
        # Use normalized data
        X_train_norm, X_val_norm = x_norm[:split], x_norm[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        X_t = torch.as_tensor(X_train_norm, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_train.values + 1, dtype=torch.long, device=self.device)
        X_v = torch.as_tensor(X_val_norm, dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(y_val.values + 1, dtype=torch.long, device=self.device)

        dataset = TensorDataset(X_t, y_t)
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            pin_memory=str(self.device).startswith("cuda"),
        )

        train_model = self.model
        if dist.is_available() and dist.is_initialized() and world_size > 1 and str(self.device).startswith("cuda"):
            try:
                local_rank = torch.cuda.current_device()
                train_model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
                )
            except Exception as e:
                logger.warning(f"DDP wrap failed for KAN, continuing without: {e}")

        optimizer = optim.Adam(train_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        es_pat, es_delta = get_early_stop_params(20, 0.001)
        early_stopper = EarlyStopper(patience=es_pat, min_delta=es_delta)

        import time

        start = time.time()
        train_model.train()

        if str(self.device) == "cpu":
            try:
                cpu_threads = int(os.environ.get("FOREX_BOT_CPU_THREADS", "0") or 0)
            except Exception:
                cpu_threads = 0
            if cpu_threads <= 0:
                cpu_threads = max(1, multiprocessing.cpu_count() - 1)
            with thread_limits(blas_threads=cpu_threads):
                self._train_loop(loader, optimizer, criterion, start, early_stopper, X_val_norm, X_v, y_v, sampler, train_model)
        else:
            self._train_loop(loader, optimizer, criterion, start, early_stopper, X_val_norm, X_v, y_v, sampler, train_model)

        if str(self.device) == "cpu" and not _KAN_AVAILABLE:
            # JIT not applied to efficient-kan (not TorchScript friendly)
            self._optimize_for_cpu_inference()

    def _train_loop(self, loader, optimizer, criterion, start, early_stopper, X_val, X_v, y_v, sampler, train_model):
        import time

        for _epoch in range(100):
            if time.time() - start > self.max_time_sec:
                break
            if sampler is not None:
                sampler.set_epoch(_epoch)

            for bx, by in loader:
                optimizer.zero_grad()
                if self._use_amp:
                    with torch.autocast(device_type=str(self.device), dtype=torch.float16):
                        out = train_model(bx)
                        loss = criterion(out, by)
                    loss.backward()
                else:
                    out = train_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                optimizer.step()

            if len(X_val) > 0:
                train_model.eval()
                with torch.no_grad():
                    val_out = train_model(X_v)
                    val_loss = criterion(val_out, y_v).item()
                train_model.train()
                if early_stopper(val_loss):
                    break

    def _optimize_for_cpu_inference(self):
        if self.model is None:
            return
        try:
            self.model.eval()
            dummy_input = torch.randn(1, self.input_dim)
            traced = torch.jit.trace(self.model, dummy_input, check_trace=False, strict=False)
            self.model = torch.jit.freeze(traced)
            logger.info("JIT Compiled KAN (Lite) for CPU.")
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
        # Dynamic inference batch size (usually can be larger than train)
        inf_batch = self.batch_size * 4
        with torch.no_grad():
            for start in range(0, len(X), inf_batch):
                # Use sliced numpy array directly
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
            if isinstance(self.model, torch.jit.ScriptModule):
                torch.jit.save(self.model, Path(path) / "kan_cpu.pt")
            else:
                torch.save(self.model.state_dict(), Path(path) / "kan.pt")
            joblib.dump({"input_dim": self.input_dim, "hidden": self.hidden_dim, "full_kan": _KAN_AVAILABLE}, Path(path) / "kan_meta.pkl")

    def load(self, path: str) -> None:
        # Similar fallback logic: prefer GPU model if available?
        # GPU KAN (efficient-kan) structure != CPU KAN (Lite).
        # We CANNOT load GPU weights into CPU Lite model directly because architectures differ.
        # Strategy: If on CPU, we must use CPU trained model OR we need 'efficient-kan' installed on CPU
        # to run the GPU weights (slowly).
        # If user trained on Cloud (GPU), they have 'kan_gpu.pt'.
        # If we are on PC (CPU), we check for 'kan_gpu.pt'. If found, try to load it using 'efficient-kan' if installed?
        # Or just warn that re-training is needed for Lite model.

        p_jit = Path(path) / "kan_cpu.pt"
        p_std = Path(path) / "kan.pt"
        m = Path(path) / "kan_meta.pkl"

        if not m.exists():
            return
        meta = joblib.load(m)
        self.input_dim = meta["input_dim"]
        self.hidden_dim = meta["hidden"]  # 'hidden_dim' key exists in both

        if p_jit.exists():
            try:
                self.model = torch.jit.load(str(p_jit), map_location="cpu")
                return
            except Exception:
                pass

        if p_std.exists():
            self.model = self._build_model()
            try:
                state = torch.load(p_std, map_location=str(self.device), weights_only=True)
            except TypeError as exc:
                raise RuntimeError("PyTorch too old for weights_only load; upgrade for security.") from exc
            self.model.load_state_dict(state)
            if str(self.device) == "cpu" and not _KAN_AVAILABLE:
                self._optimize_for_cpu_inference()

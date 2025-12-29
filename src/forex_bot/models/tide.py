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


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.fc(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.norm(out + res)
        return out


class TiDEExpert(ExpertModel):
    """
    TiDE-inspired classifier (deeper than Lite). Supports CPU/GPU.
    """

    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 256

    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_epochs: int = 300,
        max_time_sec: int = 36000,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.model = None
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_time_sec = max_time_sec
        self.device = select_device(device)
        self.input_dim = 0

    def _build_model(self):
        class TiDENet(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3, dropout: float = 0.2) -> None:
                super().__init__()
                self.feature_proj = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.encoder = nn.Sequential(
                    ResidualBlock(hidden_dim, dropout),
                    ResidualBlock(hidden_dim, dropout),
                    ResidualBlock(hidden_dim, dropout),
                    ResidualBlock(hidden_dim, dropout),
                )
                self.decoder = nn.Sequential(
                    ResidualBlock(hidden_dim, dropout),
                    ResidualBlock(hidden_dim, dropout),
                    ResidualBlock(hidden_dim, dropout),
                    ResidualBlock(hidden_dim, dropout),
                )
                self.temporal_link = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.final = nn.Linear(hidden_dim, output_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x_emb = self.feature_proj(x)
                enc_out = self.encoder(x_emb)
                dec_in = self.temporal_link(enc_out)
                dec_out = self.decoder(dec_in)
                return self.final(dec_out)

        return TiDENet(self.input_dim, self.hidden_dim).to(self.device)

    def fit(self, X: pd.DataFrame, y: pd.Series, tensorboard_writer=None) -> None:
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
            batch_size=max(32, int(self.batch_size)),
            shuffle=(sampler is None),
            pin_memory=str(self.device).startswith("cuda"),
            sampler=sampler,
        )

        train_model = self.model
        # Wrap with DDP if distributed and CUDA
        if dist.is_available() and dist.is_initialized() and world_size > 1 and str(self.device).startswith("cuda"):
            try:
                local_rank = torch.cuda.current_device()
                train_model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
                )
            except Exception as e:
                logger.warning(f"DDP wrap failed for TiDE, continuing without: {e}")

        optimizer = optim.Adam(train_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        es_pat, es_delta = get_early_stop_params(20, 0.0005)
        early_stopper = EarlyStopper(patience=es_pat, min_delta=es_delta)

        import time

        start = time.time()
        self.model.train()

        torch.backends.cudnn.benchmark = True

        try:
            cpu_threads = int(os.environ.get("FOREX_BOT_CPU_THREADS", "0") or 0)
        except Exception:
            cpu_threads = 0
        if cpu_threads <= 0:
            cpu_threads = max(1, multiprocessing.cpu_count() - 1)
        with thread_limits(blas_threads=cpu_threads):
            for _epoch in range(self.max_epochs):
                if time.time() - start > self.max_time_sec:
                    break

                for bx, by in loader:
                    optimizer.zero_grad(set_to_none=True)
                    out = train_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

                if len(X_val_norm) > 0:
                    train_model.eval()
                    with torch.no_grad():
                        val_out = train_model(X_v)
                        val_loss = criterion(val_out, y_v).item()
                    train_model.train()
                    if early_stopper(val_loss):
                        break

        self._optimize_for_cpu_inference()

    def _optimize_for_cpu_inference(self):
        if self.model is None:
            return
        try:
            self.model.eval()
            dummy_input = torch.randn(1, self.input_dim)
            traced = torch.jit.trace(self.model, dummy_input, check_trace=False, strict=False)
            self.model = torch.jit.freeze(traced)
            logger.info("JIT Compiled TiDE for CPU.")
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TiDE model not loaded")
        self.model.eval()

        x_np = dataframe_to_float32_numpy(x)
        # Apply normalization
        if hasattr(self, "mean_") and self.mean_ is not None:
            x_norm = (x_np - self.mean_) / self.scale_
        else:
            x_norm = x_np

        outs: list[np.ndarray] = []
        batch_size = self.batch_size * 4
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                # Use sliced numpy array directly
                xb_chunk = x_norm[start : start + batch_size]
                xb = torch.as_tensor(xb_chunk, dtype=torch.float32, device=self.device)
                logits = self.model(xb)
                outs.append(torch.softmax(logits, dim=1).cpu().float().numpy())
        if not outs:
            raise RuntimeError("TiDE inference returned no batches")
        probs = np.vstack(outs)
        # Training encodes labels as y+1 where y ∈ {-1,0,1} -> class indices {0,1,2}.
        # Reorder to project convention [neutral, buy, sell].
        return probs[:, [1, 2, 0]]

    def save(self, path: str) -> None:
        if self.model:
            if isinstance(self.model, torch.jit.ScriptModule):
                torch.jit.save(self.model, Path(path) / "tide_cpu.pt")
            else:
                torch.save(self.model.state_dict(), Path(path) / "tide.pt")
            joblib.dump({"input_dim": self.input_dim, "hidden": self.hidden_dim}, Path(path) / "tide_meta.pkl")

    def load(self, path: str) -> None:
        p_jit = Path(path) / "tide_cpu.pt"
        p_std = Path(path) / "tide.pt"
        m = Path(path) / "tide_meta.pkl"

        if not m.exists():
            return
        meta = joblib.load(m)
        self.input_dim = meta["input_dim"]
        self.hidden_dim = meta["hidden"]

        if p_jit.exists():
            try:
                self.model = torch.jit.load(str(p_jit), map_location="cpu")
                return
            except Exception:
                pass

        if p_std.exists():
            self.model = self._build_model()
            try:
                state = torch.load(p_std, map_location="cpu", weights_only=True)
            except TypeError as exc:
                raise RuntimeError("PyTorch too old for weights_only load; upgrade for security.") from exc
            self.model.load_state_dict(state)
            self._optimize_for_cpu_inference()

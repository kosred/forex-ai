from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import (
    EarlyStopper,
    ExpertModel,
    compute_class_weights,
    get_early_stop_params,
    time_series_train_val_split,
    validate_time_ordering,
)
from .device import maybe_init_distributed, select_device, tune_torch_backend, wrap_ddp

logger = logging.getLogger(__name__)


class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, n_layers)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPExpert(ExpertModel):
    """Simple PyTorch MLP for tabular signals (CPU/GPU)."""

    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 8192

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        lr: float = 1e-3,
        max_time_sec: int = 36000,
        device: str = "cpu",
        batch_size: int = 4096,
        **_: Any,
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.max_time_sec = int(max_time_sec)
        self.device = select_device(device)
        self.batch_size = int(batch_size)
        self.model: nn.Module | None = None
        self.input_dim: int | None = None
        self._ddp = False

    def _build(self, input_dim: int) -> nn.Module:
        net = MLPNet(input_dim, self.hidden_dim, self.n_layers, self.dropout)
        return net

    def _reshape_data(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        # No special reshape needed; exists to let ONNX exporter target self.model.
        return x

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if x is None or len(x) == 0:
            return
        try:
            validate_time_ordering(x, context="MLPExpert.fit")
        except Exception:
            pass

        x_arr = x.astype(np.float32).to_numpy()
        y_arr = np.asarray(y, dtype=int)
        y_arr = np.where(y_arr == -1, 2, y_arr).astype(int, copy=False)

        max_rows = getattr(self, "max_train_rows", self.MAX_TRAIN_ROWS)
        if max_rows and len(x_arr) > max_rows:
            x_arr = x_arr[-max_rows:]
            y_arr = y_arr[-max_rows:]

        x_train, x_val, y_train, y_val = time_series_train_val_split(
            x,
            pd.Series(y_arr, index=x.index),
            val_ratio=0.15,
            min_train_samples=100,
            embargo_samples=max(10, int(len(x) * 0.005)),
        )
        x_train_arr = x_train.astype(np.float32).to_numpy()
        x_val_arr = x_val.astype(np.float32).to_numpy()
        y_train_arr = np.asarray(y_train, dtype=int)
        y_val_arr = np.asarray(y_val, dtype=int)

        self.input_dim = int(x_train_arr.shape[1])
        self.model = self._build(self.input_dim)

        dev = torch.device(self.device)
        tune_torch_backend(dev)
        maybe_init_distributed()

        self.model.to(dev)
        if isinstance(self.model, nn.Module):
            # wrap_ddp returns (model, device, ddp_active, rank, world_size)
            self.model, dev, ddp_active, rank, world_size = wrap_ddp(self.model, dev)
            self._ddp = hasattr(self.model, "module")

        # PyTorch 2.x fix: ensure arrays are writable to avoid VRAM spikes (2025 best practice)
        # Source: https://github.com/invoke-ai/InvokeAI/pull/8329
        x_train_writable = x_train_arr.copy() if not x_train_arr.flags.writeable else x_train_arr
        y_train_writable = y_train_arr.astype(np.int64)
        x_val_writable = x_val_arr.copy() if not x_val_arr.flags.writeable else x_val_arr
        y_val_writable = y_val_arr.astype(np.int64)

        train_ds = TensorDataset(
            torch.from_numpy(x_train_writable),
            torch.from_numpy(y_train_writable),
        )
        val_ds = TensorDataset(
            torch.from_numpy(x_val_writable),
            torch.from_numpy(y_val_writable),
        )

        pin_mem = dev.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=pin_mem,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1024, self.batch_size),
            shuffle=False,
            pin_memory=pin_mem,
        )

        weights = compute_class_weights(y_train_arr)
        weight_tensor = torch.ones(3, dtype=torch.float32)
        for cls, w in weights.items():
            if 0 <= int(cls) < 3:
                weight_tensor[int(cls)] = float(w)
        weight_tensor = weight_tensor.to(dev)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        patience, min_delta = get_early_stop_params(15, 0.0)
        early = EarlyStopper(patience=patience, min_delta=min_delta)

        # PyTorch 2.4+ API: torch.cuda.amp.GradScaler deprecated
        # Source: https://docs.pytorch.org/docs/stable/amp.html
        device_type = "cuda" if dev.type == "cuda" else "cpu"
        scaler = torch.amp.GradScaler(device_type, enabled=(dev.type == "cuda"))
        start = time.time()
        epoch = 0
        while time.time() - start < self.max_time_sec:
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(dev, non_blocking=True)
                yb = yb.to(dev, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                # PyTorch 2.4+ API: torch.cuda.amp.autocast deprecated
                with torch.amp.autocast(device_type, enabled=(dev.type == "cuda")):
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Validation
            self.model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(dev, non_blocking=True)
                    yb = yb.to(dev, non_blocking=True)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_loss += float(loss.item()) * len(xb)
                    n_val += len(xb)
            val_loss = val_loss / max(1, n_val)

            epoch += 1
            if early(val_loss):
                break

        if self._ddp and hasattr(self.model, "module"):
            self.model = self.model.module  # unwrap for saving

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(x), 3))
        dev = torch.device(self.device)
        self.model.to(dev)
        self.model.eval()
        x_arr = x.astype(np.float32).to_numpy()
        probs: list[np.ndarray] = []
        bs = self.PRED_BATCH_SIZE
        with torch.no_grad():
            for start in range(0, len(x_arr), bs):
                xb = torch.from_numpy(x_arr[start : start + bs]).to(dev, non_blocking=True)
                logits = self.model(xb)
                p = torch.softmax(logits, dim=1).detach().cpu().numpy()
                probs.append(p)
        if not probs:
            return np.zeros((len(x), 3))
        out = np.vstack(probs)
        # Unified Protocol: [Neutral, Buy, Sell] already in indices [0, 1, 2]
        return out[:, :3]

    def save(self, path: str) -> None:
        if self.model is None:
            return
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        state_path = p / "mlp.pt"
        meta_path = p / "mlp_meta.pkl"
        torch.save(self.model.state_dict(), state_path)
        joblib.dump(
            {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
            },
            meta_path,
        )

    def load(self, path: str) -> None:
        p = Path(path)
        state_path = p / "mlp.pt"
        meta_path = p / "mlp_meta.pkl"
        if not state_path.exists():
            return
        try:
            meta = joblib.load(meta_path) if meta_path.exists() else {}
            self.input_dim = int(meta.get("input_dim", self.input_dim or 0) or 0)
            if self.input_dim <= 0:
                return
            self.hidden_dim = int(meta.get("hidden_dim", self.hidden_dim))
            self.n_layers = int(meta.get("n_layers", self.n_layers))
            self.dropout = float(meta.get("dropout", self.dropout))
            self.model = self._build(self.input_dim)
            dev = torch.device(self.device)
            self.model.to(dev)
            state = torch.load(state_path, map_location=dev)
            self.model.load_state_dict(state)
        except Exception as exc:
            logger.warning(f"Failed to load MLP model: {exc}")

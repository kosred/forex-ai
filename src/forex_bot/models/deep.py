"""
Deep Learning Expert Models for Trading Signal Classification.

This module provides:
- NBeatsExpert: N-BEATS architecture for time series
- TiDEExpert: Time-series Dense Encoder
- TabNetExpert: Attention-based tabular learning
- KANExpert: Kolmogorov-Arnold Network style model

All models support:
- Time-series aware train/val splits (no look-ahead bias)
- Class-weighted loss for imbalanced data
- Mixed precision training (AMP)
- Distributed training (DDP)
- JIT optimization for CPU inference
"""

import logging
import os
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
    stratified_downsample,
    time_series_train_val_split,
    validate_time_ordering,
)
from .device import gpu_supports_bf16, maybe_init_distributed, select_device, tune_torch_backend, wrap_ddp

logger = logging.getLogger(__name__)


def _compute_class_weight_tensor(y: np.ndarray, device: torch.device, num_classes: int = 3) -> torch.Tensor:
    """
    Compute class weights as a tensor for CrossEntropyLoss.

    Parameters
    ----------
    y : np.ndarray
        Labels (already shifted to 0-indexed: 0, 1, 2)
    device : torch.device
        Target device
    num_classes : int
        Number of classes

    Returns
    -------
    torch.Tensor
        Weight tensor of shape (num_classes,)
    """
    weights = compute_class_weights(y)

    # Create weight tensor in class order (0, 1, 2)
    weight_tensor = torch.ones(num_classes, dtype=torch.float32)
    for cls, weight in weights.items():
        if 0 <= cls < num_classes:
            weight_tensor[cls] = weight

    return weight_tensor.to(device)


class NBeatsBlock(nn.Module):
    """N-BEATS building block with fully connected layers."""

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
    N-BEATS Expert Model for classification.

    Adapted from the N-BEATS forecasting architecture for
    multi-class signal classification.
    """

    MAX_TRAIN_ROWS = 0  # 0 = no cap
    PRED_BATCH_SIZE = 2048

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_blocks: int = 2,
        lr: float = 1e-3,
        max_time_sec: int = 1800,
        device: str = "cpu",
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
        self.is_ddp = False
        self.rank = 0
        self.world_size = 1
        tune_torch_backend(str(self.device))

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

        ddp_device = maybe_init_distributed()
        if ddp_device:
            self.device = ddp_device
        model = NBeatsNet(self.input_dim, self.hidden_dim).to(self.device)
        model, self.device, self.is_ddp, self.rank, self.world_size = wrap_ddp(model, self.device)
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        # Validate time ordering
        validate_time_ordering(X, context="NBeatsExpert.fit")

        # Downsample if needed (preserves class balance and temporal order)
        max_rows = getattr(self, "max_train_rows", self.MAX_TRAIN_ROWS)
        if max_rows and len(X) > max_rows:
            X, y = stratified_downsample(X, y, max_rows)

        self.input_dim = X.shape[1]
        self.model = self._build_model()

        try:
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                from .device import maybe_compile
                self.model = maybe_compile(self.model)
        except Exception as e:
            logger.warning(f"NBeats torch.compile failed: {e}")

        # Time-series aware split with embargo
        try:
            X_train, X_val, y_train, y_val = time_series_train_val_split(
                X, y, val_ratio=0.15, min_train_samples=100, embargo_samples=10
            )
        except ValueError as e:
            logger.warning(f"Time-series split failed, using simple split: {e}")
            split = int(len(X) * 0.85)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

        # Convert labels: {-1, 0, 1} -> {0, 1, 2} for CrossEntropyLoss
        y_train_shifted = y_train.values + 1
        y_val_shifted = y_val.values + 1

        X_t = torch.as_tensor(X_train.values, dtype=torch.float32, device="cpu")
        y_t = torch.as_tensor(y_train_shifted, dtype=torch.long, device="cpu")
        X_v = torch.as_tensor(X_val.values, dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(y_val_shifted, dtype=torch.long, device=self.device)

        # Compute class weights for balanced loss
        class_weights = _compute_class_weight_tensor(y_train_shifted, self.device, num_classes=3)

        dataset = TensorDataset(X_t, y_t)
        is_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        num_workers = max(0, min(os.cpu_count() or 0, 4) - 1) if is_cuda else 0
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=is_cuda,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        early_stopper = EarlyStopper(patience=8, min_delta=0.001)

        use_amp = is_cuda
        amp_dtype = torch.bfloat16 if (use_amp and gpu_supports_bf16(str(self.device))) else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024.0)

        start = time.time()
        self.model.train()
        stop_event = getattr(self, "stop_event", None)

        for epoch in range(100):
            if (stop_event is not None and getattr(stop_event, "is_set", lambda: False)()) or (
                time.time() - start > self.max_time_sec
            ):
                break

            epoch_loss = 0.0
            steps = 0
            for bx, by in loader:
                if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                    break
                if is_cuda:
                    bx = bx.to(self.device, non_blocking=True)
                    by = by.to(self.device, non_blocking=True)
                
                # Volatility Weighting (Prop Firm Optimization)
                # Feature 0 is 'returns'. Penalize errors on big moves to reduce drawdown.
                with torch.no_grad():
                    # Scale: 1% move => weight +1.0. 
                    vol_weight = 1.0 + (torch.abs(bx[:, 0]) * 100.0)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    out = self.model(bx)
                    # Use functional CE to apply per-sample weights dynamically
                    raw_loss = nn.functional.cross_entropy(out, by, weight=class_weights, reduction='none')
                    loss = (raw_loss * vol_weight).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                steps += 1

            avg_train_loss = epoch_loss / max(1, steps)
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/train_NBeats", avg_train_loss, epoch)

            if len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_v)
                    val_loss = criterion(val_out, y_v).item()
                self.model.train()
                if tensorboard_writer:
                    tensorboard_writer.add_scalar("Loss/val_NBeats", val_loss, epoch)
                if early_stopper(val_loss):
                    logger.info(f"N-BEATS Early stopping at epoch {epoch}, val_loss={val_loss:.4f}")
                    break

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NBeats model not loaded")
        self.model.eval()
        outs: list[np.ndarray] = []
        batch_size = self.PRED_BATCH_SIZE
        use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
        with torch.inference_mode():
            for start in range(0, len(X), batch_size):
                xb = torch.as_tensor(
                    X.iloc[start : start + batch_size].values, dtype=torch.float32, device=self.device
                )
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self.model(xb)
                outs.append(torch.softmax(logits, dim=1).cpu().float().numpy())
        if not outs:
            raise RuntimeError("NBeats inference returned no batches")
        return np.vstack(outs)

    def save(self, path: str) -> None:
        if self.model and (not self.is_ddp or self.rank == 0):
            target = getattr(self.model, "module", self.model)
            if hasattr(target, "_orig_mod"):
                target = target._orig_mod
            torch.save(target.state_dict(), Path(path) / "nbeats.pt")
            meta = {"input_dim": self.input_dim, "hidden": self.hidden_dim}
            joblib.dump(meta, Path(path) / "nbeats_meta.pkl")

    def load(self, path: str) -> None:
        p = Path(path) / "nbeats.pt"
        m = Path(path) / "nbeats_meta.pkl"
        if p.exists() and m.exists():
            meta = joblib.load(m)
            self.input_dim = meta["input_dim"]
            self.hidden_dim = meta["hidden"]
            self.model = self._build_model()
            target = getattr(self.model, "module", self.model)
            target.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))

            if str(self.device) == "cpu":
                self._optimize_for_cpu_inference()

    def _optimize_for_cpu_inference(self) -> None:
        if self.model is None:
            return
        try:
            self.model.eval()
            dummy_input = torch.randn(1, self.input_dim)
            traced = torch.jit.trace(self.model, dummy_input, check_trace=False, strict=False)
            self.model = torch.jit.freeze(traced)
            logger.info(f"JIT Compiled {self.__class__.__name__} for CPU.")
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""

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
    TiDE Expert Model (Time-series Dense Encoder).

    Efficient architecture combining dense encoding with residual connections.
    """

    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 2048

    def __init__(
        self,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_time_sec: int = 1800,
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
        self.is_ddp = False
        self.rank = 0
        self.world_size = 1

    def _build_model(self) -> nn.Module:
        class TiDENet(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3) -> None:
                super().__init__()
                self.feature_proj = nn.Linear(input_dim, hidden_dim)
                self.encoder = nn.Sequential(ResidualBlock(hidden_dim), ResidualBlock(hidden_dim))
                self.decoder = nn.Sequential(ResidualBlock(hidden_dim), ResidualBlock(hidden_dim))
                self.temporal_link = nn.Linear(hidden_dim, hidden_dim)
                self.final = nn.Linear(hidden_dim, output_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x_emb = self.feature_proj(x)
                enc_out = self.encoder(x_emb)
                dec_in = self.temporal_link(enc_out)
                dec_out = self.decoder(dec_in)
                return self.final(dec_out)

        ddp_device = maybe_init_distributed()
        if ddp_device:
            self.device = ddp_device
        model = TiDENet(self.input_dim, self.hidden_dim).to(self.device)
        model, self.device, self.is_ddp, self.rank, self.world_size = wrap_ddp(model, self.device)
        return model

    def fit(self, x: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        validate_time_ordering(x, context="TiDEExpert.fit")

        max_rows = getattr(self, "max_train_rows", self.MAX_TRAIN_ROWS)
        if max_rows and len(x) > max_rows:
            x, y = stratified_downsample(x, y, max_rows)

        self.input_dim = x.shape[1]
        self.model = self._build_model()

        try:
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                from .device import maybe_compile
                self.model = maybe_compile(self.model)
        except Exception as e:
            logger.warning(f"TiDE torch.compile failed: {e}")

        try:
            X_train, X_val, y_train, y_val = time_series_train_val_split(
                x, y, val_ratio=0.15, min_train_samples=100, embargo_samples=10
            )
        except ValueError as e:
            logger.warning(f"Time-series split failed, using simple split: {e}")
            split = int(len(x) * 0.85)
            X_train, X_val = x.iloc[:split], x.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

        y_train_shifted = y_train.values + 1
        y_val_shifted = y_val.values + 1

        X_t = torch.as_tensor(X_train.values, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_train_shifted, dtype=torch.long, device=self.device)
        X_v = torch.as_tensor(X_val.values, dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(y_val_shifted, dtype=torch.long, device=self.device)

        class_weights = _compute_class_weight_tensor(y_train_shifted, self.device, num_classes=3)

        dataset = TensorDataset(X_t, y_t)
        sampler = None
        if self.is_ddp:
            from torch.utils.data import DistributedSampler

            sampler = DistributedSampler(dataset, shuffle=True)

        pin_mem = torch.cuda.is_available()
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=(sampler is None), sampler=sampler, pin_memory=pin_mem
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        early_stopper = EarlyStopper(patience=8, min_delta=0.001)

        use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start = time.time()
        self.model.train()

        for epoch in range(100):
            if time.time() - start > self.max_time_sec:
                break
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_loss = 0.0
            steps = 0
            for bx, by in loader:
                # Volatility Weighting (Prop Firm Optimization)
                with torch.no_grad():
                    vol_weight = 1.0 + (torch.abs(bx[:, 0]) * 100.0)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = self.model(bx)
                    raw_loss = nn.functional.cross_entropy(out, by, weight=class_weights, reduction='none')
                    loss = (raw_loss * vol_weight).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                steps += 1

            avg_train_loss = epoch_loss / max(1, steps)
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/train_TiDE", avg_train_loss, epoch)

            if len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_v)
                    val_loss = criterion(val_out, y_v).item()
                self.model.train()

                if tensorboard_writer:
                    tensorboard_writer.add_scalar("Loss/val_TiDE", val_loss, epoch)

                if early_stopper(val_loss):
                    logger.info(f"TiDE Early stopping at epoch {epoch}, val_loss={val_loss:.4f}")
                    break

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TiDE model not loaded")
        self.model.eval()
        outs: list[np.ndarray] = []
        batch_size = self.PRED_BATCH_SIZE
        use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = torch.as_tensor(
                    x.iloc[start : start + batch_size].values, dtype=torch.float32, device=self.device
                )
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self.model(xb)
                outs.append(torch.softmax(logits, dim=1).cpu().float().numpy())
        if not outs:
            raise RuntimeError("TiDE inference returned no batches")
        return np.vstack(outs)

    def save(self, path: str) -> None:
        if self.model and (not self.is_ddp or self.rank == 0):
            target = getattr(self.model, "module", self.model)
            if hasattr(target, "_orig_mod"):
                target = target._orig_mod
            torch.save(target.state_dict(), Path(path) / "tide.pt")
            joblib.dump({"input_dim": self.input_dim, "hidden": self.hidden_dim}, Path(path) / "tide_meta.pkl")

    def load(self, path: str) -> None:
        p = Path(path) / "tide.pt"
        m = Path(path) / "tide_meta.pkl"
        if p.exists() and m.exists():
            meta = joblib.load(m)
            self.input_dim = meta["input_dim"]
            self.hidden_dim = meta["hidden"]
            self.model = self._build_model()
            target = getattr(self.model, "module", self.model)
            target.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))


class AttentiveTransformer(nn.Module):
    """Attention mechanism for TabNet feature selection."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = x * prior
        return self.activation(x)


class TabNetExpert(ExpertModel):
    """
    TabNet Expert Model for tabular data.

    Uses attention-based feature selection for interpretable predictions.
    """

    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 2048

    def __init__(
        self,
        hidden_dim: int = 64,
        n_steps: int = 3,
        lr: float = 2e-3,
        batch_size: int = 64,
        max_time_sec: int = 1800,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.model = None
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.lr = lr
        self.batch_size = batch_size
        self.max_time_sec = max_time_sec
        self.device = select_device(device)
        self.input_dim = 0
        self.is_ddp = False
        self.rank = 0
        self.world_size = 1

    def _build_model(self) -> nn.Module:
        class TabNetReal(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3, n_steps: int = 3) -> None:
                super().__init__()
                self.n_steps = n_steps
                self.hidden_dim = hidden_dim
                self.feat_transform = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim * 2), nn.GLU(), nn.Linear(hidden_dim, hidden_dim * 2), nn.GLU()
                )
                self.attentive = AttentiveTransformer(hidden_dim, input_dim)
                self.final = nn.Linear(hidden_dim, output_dim)
                self.norm = nn.BatchNorm1d(input_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.norm(x)
                prior = torch.ones_like(x)
                out_accum = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
                for _ in range(self.n_steps):
                    feat = self.feat_transform(x)
                    mask = self.attentive(feat, prior)
                    prior = torch.clamp(prior * (1.5 - mask), 0, 1)
                    masked_x = x * mask
                    step_out = self.feat_transform(masked_x)
                    out_accum = out_accum + step_out
                return self.final(out_accum)

        ddp_device = maybe_init_distributed()
        if ddp_device:
            self.device = ddp_device
        model = TabNetReal(self.input_dim, self.hidden_dim, n_steps=self.n_steps).to(self.device)
        model, self.device, self.is_ddp, self.rank, self.world_size = wrap_ddp(model, self.device)
        return model

    def fit(self, x: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        validate_time_ordering(x, context="TabNetExpert.fit")

        max_rows = getattr(self, "max_train_rows", self.MAX_TRAIN_ROWS)
        if max_rows and len(x) > max_rows:
            x, y = stratified_downsample(x, y, max_rows)

        self.input_dim = x.shape[1]
        self.model = self._build_model()

        try:
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                from .device import maybe_compile
                self.model = maybe_compile(self.model)
        except Exception as e:
            logger.warning(f"TabNet torch.compile failed: {e}")

        try:
            x_train, x_val, y_train, y_val = time_series_train_val_split(
                x, y, val_ratio=0.15, min_train_samples=100, embargo_samples=10
            )
        except ValueError as e:
            logger.warning(f"Time-series split failed, using simple split: {e}")
            split = int(len(x) * 0.85)
            x_train, x_val = x.iloc[:split], x.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

        y_train_shifted = y_train.values + 1
        y_val_shifted = y_val.values + 1

        x_t = torch.as_tensor(x_train.values, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_train_shifted, dtype=torch.long, device=self.device)
        x_v = torch.as_tensor(x_val.values, dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(y_val_shifted, dtype=torch.long, device=self.device)

        class_weights = _compute_class_weight_tensor(y_train_shifted, self.device, num_classes=3)

        dataset = TensorDataset(x_t, y_t)
        sampler = None
        if self.is_ddp:
            from torch.utils.data import DistributedSampler

            sampler = DistributedSampler(dataset, shuffle=True)

        pin_mem = torch.cuda.is_available()
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=(sampler is None), sampler=sampler, pin_memory=pin_mem
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        early_stopper = EarlyStopper(patience=8, min_delta=0.001)

        use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start = time.time()
        self.model.train()

        for epoch in range(100):
            if time.time() - start > self.max_time_sec:
                break
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_loss = 0.0
            steps = 0
            for bx, by in loader:
                # Volatility Weighting (Prop Firm Optimization)
                with torch.no_grad():
                    vol_weight = 1.0 + (torch.abs(bx[:, 0]) * 100.0)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self.model(bx)
                    raw_loss = nn.functional.cross_entropy(logits, by, weight=class_weights, reduction='none')
                    loss = (raw_loss * vol_weight).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                steps += 1

            avg_train_loss = epoch_loss / max(1, steps)
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/train_TabNet", avg_train_loss, epoch)

            if len(x_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(x_v)
                    val_loss = criterion(val_out, y_v).item()
                self.model.train()

                if tensorboard_writer:
                    tensorboard_writer.add_scalar("Loss/val_TabNet", val_loss, epoch)

                if early_stopper(val_loss):
                    logger.info(f"TabNet Early stopping at epoch {epoch}, val_loss={val_loss:.4f}")
                    break

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TabNet model not loaded")
        self.model.eval()
        outs: list[np.ndarray] = []
        batch_size = self.PRED_BATCH_SIZE
        use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = torch.as_tensor(
                    x.iloc[start : start + batch_size].values, dtype=torch.float32, device=self.device
                )
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self.model(xb)
                outs.append(torch.softmax(logits, dim=1).cpu().float().numpy())
        if not outs:
            raise RuntimeError("TabNet inference returned no batches")
        return np.vstack(outs)

    def save(self, path: str) -> None:
        if self.model and (not self.is_ddp or self.rank == 0):
            target = getattr(self.model, "module", self.model)
            if hasattr(target, "_orig_mod"):
                target = target._orig_mod
            torch.save(target.state_dict(), Path(path) / "tabnet.pt")
            joblib.dump(
                {"input_dim": self.input_dim, "hidden": self.hidden_dim, "n_steps": self.n_steps},
                Path(path) / "tabnet_meta.pkl",
            )

    def load(self, path: str) -> None:
        p = Path(path) / "tabnet.pt"
        m = Path(path) / "tabnet_meta.pkl"
        if p.exists() and m.exists():
            meta = joblib.load(m)
            self.input_dim = meta["input_dim"]
            self.hidden_dim = meta["hidden"]
            self.n_steps = meta.get("n_steps", 3)
            self.model = self._build_model()
            target = getattr(self.model, "module", self.model)
            target.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))


class KANLayerBSpline(nn.Module):
    """KAN-style layer with learnable activation functions."""

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
    KAN Expert Model (Kolmogorov-Arnold Network style).

    Uses learnable activation functions for expressive representation.
    """

    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 2048

    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_time_sec: int = 1800,
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
        self.is_ddp = False
        self.rank = 0
        self.world_size = 1

    def _build_model(self) -> nn.Module:
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

        ddp_device = maybe_init_distributed()
        if ddp_device:
            self.device = ddp_device
        model = KANNet(self.input_dim, self.hidden_dim).to(self.device)
        model, self.device, self.is_ddp, self.rank, self.world_size = wrap_ddp(model, self.device)
        return model

    def fit(self, x: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        validate_time_ordering(x, context="KANExpert.fit")

        max_rows = getattr(self, "max_train_rows", self.MAX_TRAIN_ROWS)
        if max_rows and len(x) > max_rows:
            x, y = stratified_downsample(x, y, max_rows)

        self.input_dim = x.shape[1]
        self.model = self._build_model()

        try:
            if torch.cuda.is_available() and hasattr(torch, "compile"):
                from .device import maybe_compile
                self.model = maybe_compile(self.model)
        except Exception as e:
            logger.warning(f"KAN torch.compile failed: {e}")

        try:
            x_train, x_val, y_train, y_val = time_series_train_val_split(
                x, y, val_ratio=0.15, min_train_samples=100, embargo_samples=10
            )
        except ValueError as e:
            logger.warning(f"Time-series split failed, using simple split: {e}")
            split = int(len(x) * 0.85)
            x_train, x_val = x.iloc[:split], x.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

        y_train_shifted = y_train.values + 1
        y_val_shifted = y_val.values + 1

        x_t = torch.as_tensor(x_train.values, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_train_shifted, dtype=torch.long, device=self.device)
        x_v = torch.as_tensor(x_val.values, dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(y_val_shifted, dtype=torch.long, device=self.device)

        class_weights = _compute_class_weight_tensor(y_train_shifted, self.device, num_classes=3)

        dataset = TensorDataset(x_t, y_t)
        sampler = None
        if self.is_ddp:
            from torch.utils.data import DistributedSampler

            sampler = DistributedSampler(dataset, shuffle=True)

        pin_mem = torch.cuda.is_available()
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=(sampler is None), sampler=sampler, pin_memory=pin_mem
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        early_stopper = EarlyStopper(patience=8, min_delta=0.001)

        use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start = time.time()
        self.model.train()

        for epoch in range(100):
            if time.time() - start > self.max_time_sec:
                break
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_loss = 0.0
            steps = 0
            for bx, by in loader:
                # Volatility Weighting (Prop Firm Optimization)
                with torch.no_grad():
                    vol_weight = 1.0 + (torch.abs(bx[:, 0]) * 100.0)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = self.model(bx)
                    raw_loss = nn.functional.cross_entropy(out, by, weight=class_weights, reduction='none')
                    loss = (raw_loss * vol_weight).mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                steps += 1

            avg_train_loss = epoch_loss / max(1, steps)
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/train_KAN", avg_train_loss, epoch)

            if len(x_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(x_v)
                    val_loss = criterion(val_out, y_v).item()
                self.model.train()

                if tensorboard_writer:
                    tensorboard_writer.add_scalar("Loss/val_KAN", val_loss, epoch)

                if early_stopper(val_loss):
                    logger.info(f"KAN Early stopping at epoch {epoch}, val_loss={val_loss:.4f}")
                    break

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("KAN model not loaded")
        self.model.eval()
        outs: list[np.ndarray] = []
        batch_size = self.PRED_BATCH_SIZE
        use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = torch.as_tensor(
                    x.iloc[start : start + batch_size].values, dtype=torch.float32, device=self.device
                )
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = self.model(xb)
                outs.append(torch.softmax(logits, dim=1).cpu().float().numpy())
        if not outs:
            raise RuntimeError("KAN inference returned no batches")
        return np.vstack(outs)

    def save(self, path: str) -> None:
        if self.model and (not self.is_ddp or self.rank == 0):
            target = getattr(self.model, "module", self.model)
            if hasattr(target, "_orig_mod"):
                target = target._orig_mod
            torch.save(target.state_dict(), Path(path) / "kan.pt")
            joblib.dump({"input_dim": self.input_dim, "hidden": self.hidden_dim}, Path(path) / "kan_meta.pkl")

    def load(self, path: str) -> None:
        p = Path(path) / "kan.pt"
        m = Path(path) / "kan_meta.pkl"
        if p.exists() and m.exists():
            meta = joblib.load(m)
            self.input_dim = meta["input_dim"]
            self.hidden_dim = meta["hidden"]
            self.model = self._build_model()
            target = getattr(self.model, "module", self.model)
            target.load_state_dict(torch.load(p, map_location=self.device, weights_only=True))

            if str(self.device) == "cpu":
                self._optimize_for_cpu_inference()

    def _optimize_for_cpu_inference(self) -> None:
        if self.model is None:
            return
        try:
            self.model.eval()
            dummy_input = torch.randn(1, self.input_dim)
            traced = torch.jit.trace(self.model, dummy_input, check_trace=False, strict=False)
            self.model = torch.jit.freeze(traced)
            logger.info(f"JIT Compiled {self.__class__.__name__} for CPU.")
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")

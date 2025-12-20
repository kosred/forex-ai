import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..core.system import thread_limits
from .base import EarlyStopper, ExpertModel, dataframe_to_float32_numpy

TabNetClassifier = None  # type: ignore


def _try_import_tabnet() -> Any | None:
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetClassifier  # type: ignore

        return _TabNetClassifier
    except Exception:
        return None


logger = logging.getLogger(__name__)


class AttentiveTransformer(nn.Module):
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
    CPU-Optimized TabNet (Custom Lite Implementation).
    Optimized for low latency inference via JIT.
    """

    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 256

    @staticmethod
    def _pad_probs(probs: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
        """
        Normalize probability outputs to shape (n,3) ordered as [neutral, buy, sell].

        If `classes` is provided, it is interpreted as the class label for each column and used to reorder/map.
        Supported label conventions:
          - {-1, 0, 1}: -1=sell, 0=neutral, 1=buy
          - {0, 1, 2}: 0=neutral, 1=buy, 2=sell
        """
        if probs is None or len(probs) == 0:
            return np.zeros((0, 3), dtype=float)
        arr = np.asarray(probs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n = arr.shape[0]

        out = np.zeros((n, 3), dtype=float)
        if classes and len(classes) == arr.shape[1]:
            for col, cls in enumerate(classes):
                if cls == 0:
                    out[:, 0] = arr[:, col]
                elif cls == 1:
                    out[:, 1] = arr[:, col]
                elif cls in (-1, 2):
                    out[:, 2] = arr[:, col]
            return out

        if arr.shape[1] >= 3:
            return arr[:, :3]
        if arr.shape[1] == 2:
            out[:, 0] = arr[:, 0]
            out[:, 1] = arr[:, 1]
            return out
        out[:, 0] = 1.0 - arr[:, 0]
        out[:, 1] = arr[:, 0]
        return out

    def __init__(
        self,
        hidden_dim: int = 64,
        n_steps: int = 3,
        lr: float = 2e-3,
        max_time_sec: int = 1800,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.model = None
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.lr = lr
        try:
            self.batch_size = int(kwargs.get("batch_size", 64) or 64)
        except Exception:
            self.batch_size = 64
        self.max_time_sec = max_time_sec
        self.device = torch.device("cpu")
        self.input_dim = 0
        self._lib_model: Any | None = None

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
                out_accum = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
                for _ in range(self.n_steps):
                    feat = self.feat_transform(x)
                    mask = self.attentive(feat, prior)
                    prior = torch.clamp(prior * (1.5 - mask), 0, 1)
                    masked_x = x * mask
                    step_out = self.feat_transform(masked_x)
                    out_accum = out_accum + step_out
                return self.final(out_accum)

        return TabNetReal(self.input_dim, self.hidden_dim, n_steps=self.n_steps).to(self.device)

    def fit(self, X: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        self.input_dim = X.shape[1]
        self.model = self._build_model()

        import multiprocessing

        split = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        X_t = torch.as_tensor(dataframe_to_float32_numpy(X_train), dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y_train.values + 1, dtype=torch.long, device=self.device)
        X_v = torch.as_tensor(dataframe_to_float32_numpy(X_val), dtype=torch.float32, device=self.device)
        y_v = torch.as_tensor(y_val.values + 1, dtype=torch.long, device=self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=max(8, int(self.batch_size)), shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        early_stopper = EarlyStopper(patience=20, min_delta=0.001)

        import time

        start = time.time()
        self.model.train()

        with thread_limits(blas_threads=max(1, multiprocessing.cpu_count() - 1)):
            for _epoch in range(100):
                if time.time() - start > self.max_time_sec:
                    break

                for bx, by in loader:
                    optimizer.zero_grad()
                    out = self.model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

                if len(X_val) > 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_out = self.model(X_v)
                        val_loss = criterion(val_out, y_v).item()
                    self.model.train()
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
            logger.info("JIT Compiled TabNet for CPU.")
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self._lib_model is not None:
            raw = self._lib_model.predict_proba(dataframe_to_float32_numpy(x))
            classes: list[int] | None = None
            try:
                classes = [int(c) for c in list(getattr(self._lib_model, "classes_", []))]
            except Exception:
                classes = None
            return self._pad_probs(raw, classes=classes)

        if self.model is None:
            raise RuntimeError("TabNet model not loaded")
        self.model.eval()
        outs: list[np.ndarray] = []
        batch_size = max(32, int(getattr(self, "batch_size", self.PRED_BATCH_SIZE)) * 4)
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb_df = x.iloc[start : start + batch_size]
                xb = torch.as_tensor(dataframe_to_float32_numpy(xb_df), dtype=torch.float32, device=self.device)
                logits = self.model(xb)
                outs.append(torch.softmax(logits, dim=1).cpu().float().numpy())
        if not outs:
            raise RuntimeError("TabNet inference returned no batches")
        probs = np.vstack(outs)
        # Training encodes labels as y+1 where y ∈ {-1,0,1} -> class indices {0,1,2}.
        # Reorder to project convention [neutral, buy, sell].
        return probs[:, [1, 2, 0]]

    def save(self, path: str) -> None:
        if self.model:
            if isinstance(self.model, torch.jit.ScriptModule):
                torch.jit.save(self.model, Path(path) / "tabnet_cpu.pt")
            else:
                torch.save(self.model.state_dict(), Path(path) / "tabnet.pt")
            joblib.dump({"input_dim": self.input_dim, "hidden": self.hidden_dim}, Path(path) / "tabnet_meta.pkl")

    def load(self, path: str) -> None:
        # Try to load GPU model first (if trained on cloud)
        # pytorch-tabnet saves as .zip. We need to check if we can load it?
        # Loading 'pytorch-tabnet' zip into custom 'TabNetReal' is HARD.
        # Strategy: CPU inference prefers 'tabnet.pt' (custom).
        # If user trains on cloud, they should use 'TabNetExpert' wrapper which saves .pt if possible
        # OR we just assume GPU training saves .zip and we must use pytorch-tabnet for inference if available.

        p_gpu = Path(path) / "tabnet_gpu.zip"
        if p_gpu.exists():
            tabnet_cls = _try_import_tabnet()
            if tabnet_cls is not None:
                # Use official lib for inference if available (slower but works).
                # Keep it separate from the custom torch model to avoid API mismatch.
                self._lib_model = tabnet_cls(device_name="auto", verbose=0)
                self._lib_model.load_model(str(p_gpu))
                return

        # Fallback to custom CPU model
        p_jit = Path(path) / "tabnet_cpu.pt"
        p_std = Path(path) / "tabnet.pt"
        m = Path(path) / "tabnet_meta.pkl"

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
            self.model.load_state_dict(torch.load(p_std, map_location="cpu"))
            self._optimize_for_cpu_inference()

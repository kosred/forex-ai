import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import ExpertModel
from .device import select_device, tune_torch_backend

logger = logging.getLogger(__name__)

TabNetClassifier = None  # type: ignore


def _try_import_tabnet() -> Any | None:
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetClassifier  # type: ignore

        return _TabNetClassifier
    except Exception:
        return None


class TabNetExpert(ExpertModel):
    """
    GPU-Optimized TabNet using official 'pytorch-tabnet' library.
    Benefits: Self-supervised pretraining, Sparsemax, Ghost Batch Normalization.
    """

    MAX_TRAIN_ROWS = 0

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
            # Default: treat as [neutral, buy]
            out[:, 0] = arr[:, 0]
            out[:, 1] = arr[:, 1]
            return out
        out[:, 0] = 1.0 - arr[:, 0]
        out[:, 1] = arr[:, 0]
        return out

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        lambda_sparse: float = 1e-4,
        lr: float = 2e-2,
        batch_size: int = 4096,
        max_time_sec: int = 3600,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        self.model = None
        self.hyperparams = {
            "n_d": n_d,
            "n_a": n_a,
            "n_steps": n_steps,
            "gamma": gamma,
            "lambda_sparse": lambda_sparse,
            "optimizer_params": {"lr": lr},
            "mask_type": "entmax",  # More rigorous than sparsemax
        }
        self.batch_size = batch_size
        self.max_time_sec = max_time_sec
        self.device = select_device(device)
        self.input_dim = 0
        tune_torch_backend(str(self.device))

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        tabnet_cls = _try_import_tabnet()
        if tabnet_cls is None:
            logger.warning("pytorch-tabnet not available; skipping TabNet GPU training.")
            return

        self.input_dim = int(X.shape[1])
        device_name = "cuda" if str(self.device).startswith("cuda") else "cpu"

        # Instantiate on-demand so registry construction doesn't require pytorch-tabnet at import time.
        if self.model is None:
            self.model = tabnet_cls(device_name=device_name, verbose=0, **self.hyperparams)

        x = X.to_numpy(dtype=np.float32, copy=False)
        if not x.flags.writeable:
            x = x.copy()
        y_arr = y.to_numpy(dtype=int, copy=False)

        split = int(len(x) * 0.85)
        split = max(0, min(len(x), split))
        x_train, y_train = x[:split], y_arr[:split]
        x_val, y_val = x[split:], y_arr[split:]

        # Increase patience and epochs for deeper training
        max_epochs = 2000
        patience = 50

        self.model.fit(
            X_train=x_train,
            y_train=y_train,
            eval_set=[(x_val, y_val)] if len(x_val) else [(x_train, y_train)],
            eval_name=["val"] if len(x_val) else ["train"],
            eval_metric=["logloss"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=self.batch_size,
            virtual_batch_size=min(self.batch_size, 128),
            num_workers=0,
            drop_last=False,
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TabNet GPU model not loaded")
        try:
            x = X.to_numpy(dtype=np.float32, copy=False)
            if not x.flags.writeable:
                x = x.copy()
            raw = self.model.predict_proba(x)
            classes: list[int] | None = None
            try:
                classes = [int(c) for c in list(getattr(self.model, "classes_", []))]
            except Exception:
                classes = None
            return self._pad_probs(raw, classes=classes)
        except Exception as exc:
            logger.error(f"TabNet GPU inference failed: {exc}", exc_info=True)
            raise

    def save(self, path: str) -> None:
        if self.model is not None:
            # pytorch-tabnet has built-in save/load using zip
            self.model.save_model(str(Path(path) / "tabnet_gpu"))

    def load(self, path: str) -> None:
        p = Path(path) / "tabnet_gpu.zip"
        tabnet_cls = _try_import_tabnet()
        if p.exists() and tabnet_cls is not None:
            device_name = "cuda" if str(self.device).startswith("cuda") else "cpu"
            self.model = tabnet_cls(device_name=device_name, verbose=0, **self.hyperparams)
            self.model.load_model(str(p))

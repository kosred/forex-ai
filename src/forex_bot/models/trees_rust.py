from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import ExpertModel

logger = logging.getLogger(__name__)

_BINDINGS_ERROR: Exception | None = None
try:
    import forex_bindings as _fb
except Exception as exc:  # pragma: no cover - import error depends on build env
    _fb = None
    _BINDINGS_ERROR = exc


def _require_bindings() -> None:
    if _fb is None:
        raise ImportError(
            "forex_bindings is not available; build the Rust bindings to use "
            "Rust tree models."
        ) from _BINDINGS_ERROR


def _remap_labels_to_contiguous(y: pd.Series | np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.int64)
    remapped = np.zeros_like(y_arr)
    remapped[y_arr == -1] = 0
    remapped[y_arr == 0] = 1
    remapped[y_arr == 1] = 2
    return remapped


def _to_float64_contig(df: pd.DataFrame) -> np.ndarray:
    arr = df.to_numpy(dtype=np.float64, copy=False)
    if not arr.flags.writeable or not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


class _RustTreeBase(ExpertModel):
    _model_cls = None

    def __init__(self, params: dict[str, Any] | None = None, idx: int = 1) -> None:
        _require_bindings()
        if self._model_cls is None:
            raise RuntimeError("Rust model class not configured")
        self._model = self._model_cls(idx=idx, params=params or None)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        try:
            x_arr = _to_float64_contig(x)
            y_arr = _remap_labels_to_contiguous(y)
            self._model.fit(x_arr, y_arr)
        except Exception as exc:
            logger.error("Rust tree model training failed: %s", exc)
            raise

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        try:
            x_arr = _to_float64_contig(x)
            return np.asarray(self._model.predict_proba(x_arr), dtype=np.float32)
        except Exception as exc:
            logger.error("Rust tree model prediction failed: %s", exc)
            return np.zeros((len(x), 3), dtype=np.float32)

    def save(self, path: str) -> None:
        self._model.save(path)

    def load(self, path: str) -> None:
        self._model.load(path)


class RustLightGBMExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "LightGBMModel", None)


class RustXGBoostExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "XGBoostModel", None)


class RustXGBoostRFExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "XGBoostRFModel", None)


class RustXGBoostDARTExpert(_RustTreeBase):
    _model_cls = getattr(_fb, "XGBoostDARTModel", None)


# Aliases for drop-in compatibility
LightGBMExpert = RustLightGBMExpert
XGBoostExpert = RustXGBoostExpert
XGBoostRFExpert = RustXGBoostRFExpert
XGBoostDARTExpert = RustXGBoostDARTExpert

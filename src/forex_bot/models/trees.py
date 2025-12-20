from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import ExpertModel, validate_time_ordering, time_series_train_val_split

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except Exception:
    lgb = None
    LGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    RandomForestClassifier = None
    SGDClassifier = None
    train_test_split = None
    make_pipeline = None
    StandardScaler = None
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except Exception:
    xgb = None
    XGB_AVAILABLE = False

try:
    import catboost as cb

    CAT_AVAILABLE = True
except Exception:
    cb = None
    CAT_AVAILABLE = False

import joblib

logger = logging.getLogger(__name__)


def _get_model_classes(model: Any) -> list[int] | None:
    """
    Best-effort extraction of class labels in the same order as predict_proba columns.
    Returns labels as ints when possible.
    """
    if model is None:
        return None
    with_classes = getattr(model, "classes_", None)
    if with_classes is not None:
        try:
            return [int(c) for c in list(with_classes)]
        except Exception:
            return None
    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        for key in ("classifier", "model"):
            step = named_steps.get(key)
            if step is not None and getattr(step, "classes_", None) is not None:
                try:
                    return [int(c) for c in list(step.classes_)]
                except Exception:
                    return None
    steps = getattr(model, "steps", None)
    if steps:
        try:
            _, last = steps[-1]
            if getattr(last, "classes_", None) is not None:
                return [int(c) for c in list(last.classes_)]
        except Exception:
            return None
    return None


def _reorder_to_neutral_buy_sell(probs: np.ndarray, classes: list[int] | None) -> np.ndarray:
    """
    Ensure probabilities are shape (n,3) and ordered as [neutral, buy, sell].
    If `classes` is provided, it must match `probs.shape[1]`.
    """
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]

    if arr.shape[1] >= 3:
        arr = arr[:, :3]
        # If we know classes, reorder.
        if classes and len(classes) >= 3:
            classes = classes[:3]
            out = np.zeros((n, 3), dtype=float)
            for col, cls in enumerate(classes):
                if cls == 0:
                    out[:, 0] = arr[:, col]
                elif cls == 1:
                    out[:, 1] = arr[:, col]
                elif cls == -1:
                    out[:, 2] = arr[:, col]
                elif cls == 2:
                    out[:, 2] = arr[:, col]
            return out
        return arr

    out = np.zeros((n, 3), dtype=float)
    if classes and len(classes) == arr.shape[1]:
        for col, cls in enumerate(classes):
            if cls == 0:
                out[:, 0] = arr[:, col]
            elif cls == 1:
                out[:, 1] = arr[:, col]
            elif cls == -1:
                out[:, 2] = arr[:, col]
            elif cls == 2:
                out[:, 2] = arr[:, col]
        return out

    # Fallback: assume columns are [neutral, buy] or [p_buy]
    if arr.shape[1] == 2:
        out[:, 0] = arr[:, 0]
        out[:, 1] = arr[:, 1]
    else:
        out[:, 0] = 1.0 - arr[:, 0]
        out[:, 1] = arr[:, 0]
    return out


class LightGBMExpert(ExpertModel):
    def __init__(self, params: dict[str, Any] = None) -> None:
        self.model = None
        self.params = params or {
            "n_estimators": 500,
            "num_leaves": 64,
            "learning_rate": 0.05,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if not LGBM_AVAILABLE:
            logger.warning("LightGBM not available.")
            return

        try:
            uniq, counts = np.unique(y, return_counts=True)
            class_weight = {
                int(cls): float(len(y) / (len(uniq) * cnt)) for cls, cnt in zip(uniq, counts, strict=False) if cnt > 0
            }

            params = self.params.copy()
            params["class_weight"] = class_weight if class_weight else None
            import torch

            has_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0

            if has_gpu:
                params.setdefault("device_type", "gpu")
                params.setdefault("max_bin", 63)  # Critical for GPU speedup
                params.setdefault("gpu_use_dp", False)  # FP32 is faster
                if "gpu_device_id" not in params:
                    params["gpu_device_id"] = 0
            else:
                # Force CPU if no usable CUDA device even if caller requested GPU
                params["device_type"] = "cpu"
                params.pop("gpu_use_dp", None)
                params.pop("gpu_device_id", None)
                params.pop("max_bin", None)

            binary = len(uniq) <= 2
            if binary:
                params["objective"] = "binary"
                params.pop("num_class", None)
                eval_metric = "binary_logloss"
            else:
                params["objective"] = "multiclass"
                params["num_class"] = params.get("num_class", len(uniq))
                eval_metric = "multi_logloss"

            # Use time-series aware split to prevent look-ahead bias
            if len(y) > 500:
                try:
                    validate_time_ordering(x, context="LightGBMExpert.fit")
                    x_train, x_val, y_train, y_val = time_series_train_val_split(
                        x, y, val_ratio=0.15, min_train_samples=100, embargo_samples=0
                    )
                    eval_set = [(x_val, y_val)]
                except ValueError:
                    # Fall back to simple positional split if validation fails
                    split_idx = int(len(x) * 0.85)
                    x_train, x_val = x.iloc[:split_idx], x.iloc[split_idx:]
                    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
                    eval_set = [(x_val, y_val)]
            else:
                x_train, y_train = x, y
                eval_set = [(x, y)]

            self.model = lgb.LGBMClassifier(**params)
            try:
                self.model.fit(
                    x_train,
                    y_train,
                    eval_set=eval_set,
                    eval_metric=eval_metric,
                    callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
                )
            except Exception as e:
                if params.get("device_type") == "gpu":
                    logger.warning(f"LightGBM GPU training failed ({e}), falling back to CPU.")
                    params["device_type"] = "cpu"
                    params.pop("max_bin", None)
                    params.pop("gpu_use_dp", None)
                    self.model = lgb.LGBMClassifier(**params)
                    self.model.fit(
                        x_train,
                        y_train,
                        eval_set=eval_set,
                        eval_metric=eval_metric,
                        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)],
                    )
                else:
                    raise e
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBM model not loaded")
        try:
            probs = self.model.predict_proba(x)
            classes = _get_model_classes(self.model)
            return _reorder_to_neutral_buy_sell(probs, classes)
        except Exception as exc:
            logger.error(f"LightGBM inference failed: {exc}", exc_info=True)
            raise

    def save(self, path: str) -> None:
        if self.model:
            joblib.dump(self.model, Path(path) / "lightgbm.joblib")

    def load(self, path: str) -> None:
        p = Path(path) / "lightgbm.joblib"
        if p.exists():
            self.model = joblib.load(p)


class RandomForestExpert(ExpertModel):
    def __init__(self, params: dict[str, Any] = None) -> None:
        self.model = None
        self.params = params or {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 10,
            "n_jobs": -1,
            "random_state": 42,
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        # Try CuML first for GPU acceleration
        try:
            import cudf
            from cuml.ensemble import RandomForestClassifier as CuRF

            # Convert to cudf/cupy for best performance
            x_gpu = cudf.DataFrame.from_pandas(x)
            y_gpu = cudf.Series.from_pandas(y)

            logger.info(f"Training RandomForest using RAPIDS (GPU) on {len(x)} samples...")
            self.model = CuRF(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 10),
                min_samples_split=self.params.get("min_samples_split", 10),
                n_streams=1,  # Sync
                verbose=False,
            )
            self.model.fit(x_gpu, y_gpu)
            return
        except ImportError:
            pass  # Fallthrough to sklearn
        except Exception as e:
            logger.warning(f"RAPIDS RF failed ({e}), falling back to CPU.")

        if not SKLEARN_AVAILABLE:
            return
        try:
            uniq, counts = np.unique(y, return_counts=True)
            class_weight = {
                int(cls): float(len(y) / (len(uniq) * cnt)) for cls, cnt in zip(uniq, counts, strict=False) if cnt > 0
            }
            params = self.params.copy()
            params["class_weight"] = class_weight if class_weight else None

            self.model = RandomForestClassifier(**params)
            self.model.fit(x, y)
        except Exception as e:
            logger.error(f"RF training failed: {e}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(x), 3))
        probs = self.model.predict_proba(x)
        classes = _get_model_classes(self.model)
        return _reorder_to_neutral_buy_sell(probs, classes)

    def save(self, path: str) -> None:
        if self.model:
            joblib.dump(self.model, Path(path) / "rf.joblib")

    def load(self, path: str) -> None:
        p = Path(path) / "rf.joblib"
        if p.exists():
            self.model = joblib.load(p)


class ElasticNetExpert(ExpertModel):
    """
    Robust Linear Baseline.
    Uses SGD with ElasticNet penalty (L1+L2) to handle high-dimensional features
    and select relevant ones, avoiding overfitting better than simple Logistic Regression.
    """

    def __init__(self, params: dict[str, Any] = None) -> None:
        self.model = None
        self.params = params or {
            "loss": "log_loss",
            "penalty": "elasticnet",
            "alpha": 0.001,
            "l1_ratio": 0.5,
            "max_iter": 1000,
            "tol": 1e-3,
            "n_jobs": -1,
            "random_state": 42,
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if not SKLEARN_AVAILABLE:
            return
        try:
            uniq, counts = np.unique(y, return_counts=True)
            class_weight = {
                int(cls): float(len(y) / (len(uniq) * cnt)) for cls, cnt in zip(uniq, counts, strict=False) if cnt > 0
            }
            from pathlib import Path
            from tempfile import gettempdir

            from sklearn.pipeline import Pipeline

            cache_dir = Path(gettempdir()) / "sklearn_cache"
            cache_dir.mkdir(exist_ok=True)

            self.model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", SGDClassifier(**self.params, class_weight=class_weight if class_weight else None)),
                ],
                memory=str(cache_dir),
            )
            self.model.fit(x, y)
        except Exception as e:
            logger.error(f"ElasticNet training failed: {e}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained")
        try:
            probs = self.model.predict_proba(x)
            classes = _get_model_classes(self.model)
            return _reorder_to_neutral_buy_sell(probs, classes)
        except Exception as e:
            logger.error(f"ElasticNet prediction failed: {e}", exc_info=True)
            raise

    def save(self, path: str) -> None:
        if self.model:
            joblib.dump(self.model, Path(path) / "elasticnet.joblib")

    def load(self, path: str) -> None:
        p = Path(path) / "elasticnet.joblib"
        if p.exists():
            self.model = joblib.load(p)


class XGBoostExpert(ExpertModel):
    def __init__(self, params: dict[str, Any] = None) -> None:
        self.model = None

        has_gpu = False
        try:
            import torch

            has_gpu = torch.cuda.is_available()
        except ImportError:
            pass

        self.params = params or {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        if has_gpu:
            self.params.setdefault("tree_method", "hist")
            self.params.setdefault("device", "cuda")

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available")
            return
        try:
            # XGBoost's sklearn wrapper expects multiclass labels to be contiguous: {0,1,2,...}.
            # Our project uses {-1,0,1} where -1=sell, 0=neutral, 1=buy.
            y_arr = np.asarray(y, dtype=int)
            # Backward compat: some older pipelines used 2=sell.
            y_arr = np.where(y_arr == -1, 2, y_arr).astype(int, copy=False)

            uniq, counts = np.unique(y, return_counts=True)
            sample_weight = np.ones(len(y))
            for cls, cnt in zip(uniq, counts, strict=False):
                if cnt > 0:
                    sample_weight[y == cls] = len(y) / (len(uniq) * cnt)

            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(x, y_arr, sample_weight=sample_weight)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(x), 3))
        try:
            probs = self.model.predict_proba(x)
            classes = _get_model_classes(self.model)
            return _reorder_to_neutral_buy_sell(probs, classes)
        except Exception:
            return np.zeros((len(x), 3))

    def save(self, path: str) -> None:
        if self.model:
            joblib.dump(self.model, Path(path) / "xgboost.joblib")

    def load(self, path: str) -> None:
        p = Path(path) / "xgboost.joblib"
        if p.exists():
            self.model = joblib.load(p)


class CatBoostExpert(ExpertModel):
    def __init__(self, params: dict[str, Any] = None) -> None:
        self.model = None

        has_gpu = False
        try:
            import torch

            has_gpu = torch.cuda.is_available()
        except ImportError:
            pass

        self.params = params or {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "MultiClass",
            "random_state": 42,
            "verbose": False,
            "thread_count": -1,
        }

        if has_gpu:
            self.params.setdefault("task_type", "GPU")
            self.params.setdefault("devices", "0")
            self.params.setdefault("border_count", 32)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if not CAT_AVAILABLE:
            logger.warning("CatBoost not available")
            return
        try:
            uniq, counts = np.unique(y, return_counts=True)
            class_weights = {
                int(cls): float(len(y) / (len(uniq) * cnt)) for cls, cnt in zip(uniq, counts, strict=False) if cnt > 0
            }

            params = self.params.copy()
            params["class_weights"] = list(class_weights.values()) if class_weights else None

            self.model = cb.CatBoostClassifier(**params)
            self.model.fit(x, y)
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(x), 3))
        try:
            probs = self.model.predict_proba(x)
            classes = _get_model_classes(self.model)
            return _reorder_to_neutral_buy_sell(probs, classes)
        except Exception:
            return np.zeros((len(x), 3))

    def save(self, path: str) -> None:
        if self.model:
            # CatBoost has its own efficient format
            self.model.save_model(str(Path(path) / "catboost.cbm"))

    def load(self, path: str) -> None:
        if not CAT_AVAILABLE:
            return
        p = Path(path) / "catboost.cbm"
        if p.exists():
            try:
                self.model = cb.CatBoostClassifier()
                self.model.load_model(str(p))
            except Exception as e:
                logger.warning(f"Failed to load CatBoost: {e}")

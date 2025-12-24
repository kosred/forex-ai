from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import ExpertModel, time_series_train_val_split, validate_time_ordering

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except Exception:
    lgb = None
    LGBM_AVAILABLE = False

try:
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    ExtraTreesClassifier = None
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


def _cpu_threads_hint() -> int:
    try:
        return max(0, int(os.environ.get("FOREX_BOT_CPU_THREADS", "0") or 0))
    except Exception:
        return 0


def _tree_device_preference() -> str:
    """
    Controls whether tree models should run on GPU or CPU.

    Env:
      - FOREX_BOT_TREE_DEVICE=auto|gpu|cpu  (default: auto)
    """
    raw = str(os.environ.get("FOREX_BOT_TREE_DEVICE", "auto")).strip().lower()
    if raw in {"cpu", "gpu", "auto"}:
        return raw
    if raw in {"0", "false", "no", "off"}:
        return "cpu"
    if raw in {"1", "true", "yes", "on"}:
        return "gpu"
    return "auto"


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


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


def _augment_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight lag/volatility features for tree models when raw close is available.
    If 'close' is absent, returns the input unchanged.
    """
    if "close" not in df.columns:
        return df
    try:
        out = df.copy()
        close = pd.to_numeric(out["close"], errors="coerce")
        ret1 = close.pct_change().fillna(0.0)
        out["ret1"] = ret1
        out["ret1_lag1"] = ret1.shift(1).fillna(0.0)
        out["ret1_lag2"] = ret1.shift(2).fillna(0.0)
        out["ret1_lag5"] = ret1.shift(5).fillna(0.0)
        out["ret1_lag8"] = ret1.shift(8).fillna(0.0)
        out["vol14"] = ret1.rolling(14).std().fillna(0.0)
        out["vol50"] = ret1.rolling(50).std().fillna(0.0)
        out["mom5"] = close.diff(5).fillna(0.0)
        out["mom15"] = close.diff(15).fillna(0.0)
        return out
    except Exception:
        return df


def _remap_labels_to_contiguous(y: pd.Series | np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """
    Map {-1,0,1} style labels to contiguous {0,1,2} expected by some tree libs.
    Returns remapped array and mapping dict.
    """
    y_arr = np.asarray(y, dtype=int)
    mapping: dict[int, int] = {}
    # Unique in stable order
    uniq = list(dict.fromkeys(int(v) for v in y_arr))
    for new_idx, old in enumerate(uniq):
        mapping[old] = new_idx
    remapped = np.array([mapping[int(v)] for v in y_arr], dtype=int)
    return remapped, mapping


class LightGBMExpert(ExpertModel):
    def __init__(self, params: dict[str, Any] = None) -> None:
        self.model = None
        self.params = params or {
            "n_estimators": 800,
            "num_leaves": 64,
            "learning_rate": 0.03,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
            # Reduce overfitting, improve smoothness/extrapolation
            "min_data_in_leaf": 50,
            "feature_fraction": 0.6,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "path_smooth": 10,
            "linear_tree": True,
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> bool:
        if not LGBM_AVAILABLE:
            logger.warning("LightGBM not available.")
            return False

        try:
            x = _augment_time_features(x)
            # Enforce time ordering if index is datetime and not monotonic
            if isinstance(x.index, pd.DatetimeIndex) and not x.index.is_monotonic_increasing:
                order = np.argsort(x.index.view("int64"))
                x = x.iloc[order]
                y = y.iloc[order]

            params = self.params.copy()

            # Respect worker thread partitioning when set (prevents oversubscription on large CPUs).
            cpu_threads = _cpu_threads_hint()
            if cpu_threads > 0:
                params["n_jobs"] = cpu_threads

            pref = _tree_device_preference()
            has_cuda = _torch_cuda_available()
            requested = str(params.get("device_type", "")).strip().lower()

            if requested == "cpu" or pref == "cpu":
                use_gpu = False
            elif requested == "gpu" or pref == "gpu":
                use_gpu = has_cuda
                if not has_cuda:
                    logger.warning("LightGBM GPU requested but no CUDA devices detected; falling back to CPU.")
            else:
                # auto: prefer GPU if available
                use_gpu = has_cuda

            if use_gpu:
                params["device_type"] = "gpu"
                params.setdefault("max_bin", 63)  # critical for GPU speed
                params.setdefault("gpu_use_dp", False)
                params.setdefault("gpu_device_id", 0)
            else:
                params["device_type"] = "cpu"
                params.pop("gpu_use_dp", None)
                params.pop("gpu_device_id", None)
                # max_bin=255 for better accuracy on CPU
                params.setdefault("max_bin", 255)

            # Use time-series aware split to prevent look-ahead bias
            if len(y) > 500:
                try:
                    validate_time_ordering(x, context="LightGBMExpert.fit")
                    y_arr, mapping = _remap_labels_to_contiguous(y)
                    embargo = max(24, int(len(y_arr) * 0.01))
                    x_train, x_val, y_train, y_val = time_series_train_val_split(
                        x,
                        pd.Series(y_arr, index=y.index),
                        val_ratio=0.15,
                        min_train_samples=100,
                        embargo_samples=embargo,
                    )
                    eval_set = [(x_val, y_val)]
                except ValueError:
                    # Fall back to simple positional split if validation fails
                    split_idx = int(len(x) * 0.85)
                    y_arr, mapping = _remap_labels_to_contiguous(y)
                    x_train, x_val = x.iloc[:split_idx], x.iloc[split_idx:]
                    y_train, y_val = y_arr[:split_idx], y_arr[split_idx:]
                    eval_set = [(x_val, y_val)]
            else:
                y_arr, mapping = _remap_labels_to_contiguous(y)
                x_train, y_train = x, y_arr
                eval_set = [(x, y_arr)]

            # Build class weights on remapped labels so keys align with LightGBM class map.
            uniq, counts = np.unique(y_train, return_counts=True)
            class_weight = {
                int(cls): float(len(y_train) / (len(uniq) * cnt))
                for cls, cnt in zip(uniq, counts, strict=False)
                if cnt > 0
            }
            params["class_weight"] = class_weight if class_weight else None

            # Decide binary vs multiclass based on remapped labels
            binary = len(uniq) <= 2
            if binary:
                params["objective"] = "binary"
                params.pop("num_class", None)
                eval_metric = "binary_logloss"
            else:
                params["objective"] = "multiclass"
                params["num_class"] = params.get("num_class", len(uniq))
                eval_metric = "multi_logloss"

            # Decide binary vs multiclass based on remapped labels
            binary = len(uniq) <= 2
            if binary:
                params["objective"] = "binary"
                params.pop("num_class", None)
                eval_metric = "binary_logloss"
            else:
                params["objective"] = "multiclass"
                params["num_class"] = params.get("num_class", len(uniq))
                eval_metric = "multi_logloss"

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
                    logger.warning(f"LightGBM GPU training failed ({e}), falling back to CPU.", exc_info=True)
                    params["device_type"] = "cpu"
                    params.pop("gpu_use_dp", None)
                    params.pop("gpu_device_id", None)
                    params.pop("max_bin", None)
                    params.setdefault("max_bin", 255)
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
            return True
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}", exc_info=True)
            self.model = None
            return False

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBM model not loaded")
        try:
            x = _augment_time_features(x)
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
            "n_estimators": 400,
            "max_depth": 16,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": 0.7,
            "bootstrap": True,
            "n_jobs": -1,
            "random_state": 42,
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> bool:
        # Ensure chronological order to reduce look-ahead bias in bootstrap samples
        try:
            if isinstance(x.index, pd.DatetimeIndex) and not x.index.is_monotonic_increasing:
                order = np.argsort(x.index.view("int64"))
                x = x.iloc[order]
                y = y.iloc[order]
        except Exception:
            pass

        x = _augment_time_features(x)

        # Try CuML first for GPU acceleration
        try:
            import cupy as cp
            import cudf
            from cuml.ensemble import RandomForestClassifier as CuRF

            # Select specific GPU if specified in params
            gpu_id = self.params.get("gpu_device_id", 0)
            with cp.cuda.Device(gpu_id):
                # Convert to cudf/cupy for best performance
                x_gpu = cudf.DataFrame.from_pandas(x)
                y_gpu = cudf.Series.from_pandas(y)

                logger.info(f"Training RandomForest using RAPIDS (GPU {gpu_id}) on {len(x)} samples...")
                gpu_params = {
                    "n_estimators": self.params.get("n_estimators", 400),
                    "max_depth": self.params.get("max_depth", 16),
                    "min_samples_split": self.params.get("min_samples_split", 2),
                    "min_samples_leaf": self.params.get("min_samples_leaf", 1),
                    "max_features": self.params.get("max_features", 1.0),
                    "bootstrap": bool(self.params.get("bootstrap", True)),
                    "n_streams": 1,  # Sync
                    "verbose": False,
                }
                self.model = CuRF(**gpu_params)
                self.model.fit(x_gpu, y_gpu)
            return True
        except ImportError:
            pass  # Fallthrough to sklearn
        except Exception as e:
            logger.warning(f"RAPIDS RF failed ({e}), falling back to CPU.")

        if not SKLEARN_AVAILABLE:
            return False
        try:
            uniq, counts = np.unique(y, return_counts=True)
            class_weight = {
                int(cls): float(len(y) / (len(uniq) * cnt)) for cls, cnt in zip(uniq, counts, strict=False) if cnt > 0
            }
            params = self.params.copy()
            params["class_weight"] = class_weight if class_weight else "balanced_subsample"

            self.model = RandomForestClassifier(**params)
            self.model.fit(x, y)
            return True
        except Exception as e:
            logger.error(f"RF training failed: {e}")
            self.model = None
            return False

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


class ExtraTreesExpert(ExpertModel):
    """
    Extremely Randomized Trees (Geurts et al., 2006).
    Uses heavy randomization of split thresholds to reduce variance and overfitting,
    while staying fast on CPU and producing calibrated class probabilities.
    """

    def __init__(self, params: dict[str, Any] = None) -> None:
        self.model = None
        self.params = params or {
            "n_estimators": 500,
            "max_depth": 18,
            "min_samples_split": 2,
            "min_samples_leaf": 2,
            "max_features": 0.6,
            "bootstrap": False,
            "n_jobs": -1,
            "random_state": 42,
        }

    def fit(self, x: pd.DataFrame, y: pd.Series) -> bool:
        if not SKLEARN_AVAILABLE or ExtraTreesClassifier is None:
            return False
        try:
            x = _augment_time_features(x)
            if isinstance(x.index, pd.DatetimeIndex) and not x.index.is_monotonic_increasing:
                order = np.argsort(x.index.view("int64"))
                x = x.iloc[order]
                y = y.iloc[order]

            uniq, counts = np.unique(y, return_counts=True)
            class_weight = {
                int(cls): float(len(y) / (len(uniq) * cnt)) for cls, cnt in zip(uniq, counts, strict=False) if cnt > 0
            }

            params = self.params.copy()
            params["class_weight"] = class_weight if class_weight else "balanced_subsample"
            self.model = ExtraTreesClassifier(**params)
            self.model.fit(x, y)
            return True
        except Exception as e:
            logger.error(f"ExtraTrees training failed: {e}")
            self.model = None
            return False

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.zeros((len(x), 3))
        x_aug = _augment_time_features(x)
        probs = self.model.predict_proba(x_aug)
        classes = _get_model_classes(self.model)
        return _reorder_to_neutral_buy_sell(probs, classes)

    def save(self, path: str) -> None:
        if self.model:
            joblib.dump(self.model, Path(path) / "extratrees.joblib")

    def load(self, path: str) -> None:
        p = Path(path) / "extratrees.joblib"
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

        self.params = params or {
            "n_estimators": 800,
            "max_depth": 8,
            "learning_rate": 0.05,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "eval_metric": "mlogloss",
            "tree_method": "hist",  # default; overridden to gpu_hist if GPU
        }

        cpu_threads = _cpu_threads_hint()
        if cpu_threads > 0 and int(self.params.get("n_jobs", -1) or -1) < 0:
            self.params["n_jobs"] = cpu_threads

        pref = _tree_device_preference()
        has_cuda = _torch_cuda_available()
        requested = str(self.params.get("device", "")).strip().lower()

        if requested.startswith("cuda"):
            use_gpu = has_cuda
        elif pref == "cpu":
            use_gpu = False
        elif pref == "gpu":
            use_gpu = has_cuda
        else:
            # auto: prefer GPU if available
            use_gpu = has_cuda

        if use_gpu:
            # Newer XGBoost prefers `tree_method=hist` + `device=cuda`.
            self.params.setdefault("device", "cuda")
            self.params["tree_method"] = "gpu_hist"
        else:
            self.params.pop("device", None)
            self.params["tree_method"] = "hist"

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available")
            return
        try:
            if isinstance(x.index, pd.DatetimeIndex) and not x.index.is_monotonic_increasing:
                order = np.argsort(x.index.view("int64"))
                x = x.iloc[order]
                y = y.iloc[order]

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

        pref = _tree_device_preference()
        has_cuda = _torch_cuda_available()

        self.params = params or {
            "iterations": 800,
            "depth": 8,
            "learning_rate": 0.05,
            "loss_function": "MultiClass",
            "random_seed": 42,
            "verbose": False,
            "thread_count": -1,
        }

        cpu_threads = _cpu_threads_hint()
        if cpu_threads > 0 and int(self.params.get("thread_count", -1) or -1) < 0:
            self.params["thread_count"] = cpu_threads

        requested = str(self.params.get("task_type", "")).strip().lower()
        if requested == "gpu":
            use_gpu = has_cuda
        elif pref == "cpu":
            use_gpu = False
        elif pref == "gpu":
            use_gpu = has_cuda
        else:
            # auto: prefer GPU if available
            use_gpu = has_cuda

        if use_gpu:
            self.params.setdefault("task_type", "GPU")
            self.params.setdefault("devices", "0")
            self.params.setdefault("border_count", 32)
        else:
            # Ensure we don't accidentally try to run CatBoost in GPU mode without devices.
            if str(self.params.get("task_type", "")).strip().lower() == "gpu":
                self.params.pop("task_type", None)
                self.params.pop("devices", None)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if not CAT_AVAILABLE:
            logger.warning("CatBoost not available")
            return
        try:
            if isinstance(x.index, pd.DatetimeIndex) and not x.index.is_monotonic_increasing:
                order = np.argsort(x.index.view("int64"))
                x = x.iloc[order]
                y = y.iloc[order]

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

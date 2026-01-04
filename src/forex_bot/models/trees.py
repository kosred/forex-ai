from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import ExpertModel, get_early_stop_params, time_series_train_val_split, validate_time_ordering

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except Exception:
    lgb = None
    LGBM_AVAILABLE = False

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

def _gpu_only_mode() -> bool:
    raw = str(os.environ.get("FOREX_BOT_GPU_ONLY", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


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
    HPC PROTOCOL: Force output to [Neutral, Buy, Sell].
    Standard indices: 0=Neutral, 1=Buy, 2=Sell.
    """
    arr = np.asarray(probs, dtype=float)
    n = arr.shape[0]
    out = np.zeros((n, 3), dtype=float)
    
    # If model is binary (Neutral vs Signal)
    if arr.shape[1] == 2:
        out[:, 0] = arr[:, 0] # Neutral
        out[:, 1] = arr[:, 1] # Buy (Default assumption for 2-class)
        return out
        
    # If model is multiclass
    if classes is not None:
        for col, cls_val in enumerate(classes):
            # Map based on our contiguous mapping: 0=Sell, 1=Neutral, 2=Buy
            # (Wait, let's match the remap_labels we just made: -1->0, 0->1, 1->2)
            if cls_val == 0: out[:, 2] = arr[:, col] # Sell
            elif cls_val == 1: out[:, 0] = arr[:, col] # Neutral
            elif cls_val == 2: out[:, 1] = arr[:, col] # Buy
    else:
        # Fallback to direct mapping if classes missing
        return arr # Assume [Neutral, Buy, Sell]
        
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
    HPC FIX: Hardcoded Deterministic Mapping.
    Prevents 'Label Drift' where Buy/Sell columns are swapped depending on data.
    Order: -1 -> 0 (Sell), 0 -> 1 (Neutral), 1 -> 2 (Buy)
    """
    y_arr = np.asarray(y, dtype=int)
    mapping = {-1: 0, 0: 1, 1: 2}
    
    # Fast vectorized mapping
    remapped = np.zeros_like(y_arr)
    remapped[y_arr == -1] = 0
    remapped[y_arr == 0] = 1
    remapped[y_arr == 1] = 2
    
    return remapped, mapping


class LightGBMExpert(ExpertModel):
    def __init__(self, params: dict[str, Any] = None, idx: int = 1) -> None:
        self.model = None
        self._gpu_only_disabled = False
        self.idx = idx # Store index for GPU distribution
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

            # Clean inf values (replace with nan)
            x = x.replace([np.inf, -np.inf], np.nan)

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
            if _gpu_only_mode() and not use_gpu:
                logger.warning("GPU-only mode: LightGBM GPU unavailable; skipping LightGBM.")
                self._gpu_only_disabled = True
                self.model = None
                return False

            if use_gpu:
                params["device_type"] = "gpu"
                params.setdefault("max_bin", 63)
                params.setdefault("gpu_use_dp", False)
                # HPC: Spread models across all 8 GPUs
                import torch
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
                params["gpu_device_id"] = (self.idx - 1) % gpu_count
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
                    if _gpu_only_mode():
                        logger.warning(f"GPU-only mode: LightGBM GPU training failed ({e}); skipping LightGBM.")
                        self._gpu_only_disabled = True
                        self.model = None
                        return False
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
        if self._gpu_only_disabled:
            raise RuntimeError("GPU-only mode: LightGBM skipped.")
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


class XGBoostExpert(ExpertModel):
    def __init__(self, params: dict[str, Any] = None, idx: int = 1) -> None:
        self.model = None
        self._gpu_only_disabled = False
        self.idx = idx

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
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            gpu_id = (self.idx - 1) % gpu_count
            self.params.setdefault("device", f"cuda:{gpu_id}")
            self.params["tree_method"] = "gpu_hist"
        else:
            self.params.pop("device", None)
            self.params["tree_method"] = "hist"
            if _gpu_only_mode():
                self._gpu_only_disabled = True

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if self._gpu_only_disabled:
            logger.warning("GPU-only mode: XGBoost GPU unavailable; skipping.")
            self.model = None
            return
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available")
            return
        try:
            if isinstance(x.index, pd.DatetimeIndex) and not x.index.is_monotonic_increasing:
                order = np.argsort(x.index.view("int64"))
                x = x.iloc[order]
                y = y.iloc[order]

            # XGBoost can't handle inf values - replace with nan (which XGBoost handles natively)
            x = x.replace([np.inf, -np.inf], np.nan)

            # XGBoost's sklearn wrapper expects multiclass labels to be contiguous: {0,1,2,...}.
            # Our project uses {-1,0,1} where -1=sell, 0=neutral, 1=buy.
            y_arr = np.asarray(y, dtype=int)
            # Backward compat: some older pipelines used 2=sell.
            y_arr = np.where(y_arr == -1, 2, y_arr).astype(int, copy=False)

            x_train = x
            y_train = y_arr
            eval_set = None
            if len(y_arr) > 500:
                try:
                    validate_time_ordering(x, context="XGBoostExpert.fit")
                    embargo = max(24, int(len(y_arr) * 0.01))
                    x_train, x_val, y_train_s, y_val_s = time_series_train_val_split(
                        x,
                        pd.Series(y_arr, index=y.index),
                        val_ratio=0.15,
                        min_train_samples=100,
                        embargo_samples=embargo,
                    )
                    y_train = y_train_s.to_numpy(dtype=int)
                    y_val = y_val_s.to_numpy(dtype=int)
                    eval_set = [(x_val, y_val)]
                except ValueError:
                    split_idx = int(len(x) * 0.85)
                    x_train, x_val = x.iloc[:split_idx], x.iloc[split_idx:]
                    y_train = y_arr[:split_idx]
                    y_val = y_arr[split_idx:]
                    eval_set = [(x_val, y_val)]

            uniq, counts = np.unique(y_train, return_counts=True)
            sample_weight = np.ones(len(y_train))
            for cls, cnt in zip(uniq, counts, strict=False):
                if cnt > 0:
                    sample_weight[y_train == cls] = len(y_train) / (len(uniq) * cnt)

            # XGBoost 2.0+: early_stopping_rounds moved from fit() to constructor
            params = self.params.copy()
            fit_kwargs = {}
            if eval_set:
                es_pat, _ = get_early_stop_params(50, 0.0)
                if es_pat > 0:
                    params["early_stopping_rounds"] = es_pat
                    fit_kwargs["eval_set"] = eval_set
                    fit_kwargs["verbose"] = False

            self.model = xgb.XGBClassifier(**params)
            self.model.fit(x_train, y_train, sample_weight=sample_weight, **fit_kwargs)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self._gpu_only_disabled:
            raise RuntimeError("GPU-only mode: XGBoost skipped.")
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
    def __init__(self, params: dict[str, Any] = None, idx: int = 1) -> None:
        self.model = None
        self._gpu_only_disabled = False
        self.idx = idx

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
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
            gpu_id = (self.idx - 1) % gpu_count
            self.params.setdefault("devices", str(gpu_id))
            self.params.setdefault("border_count", 32)
        else:
            # Ensure we don't accidentally try to run CatBoost in GPU mode without devices.
            if str(self.params.get("task_type", "")).strip().lower() == "gpu":
                self.params.pop("task_type", None)
                self.params.pop("devices", None)
            if _gpu_only_mode():
                self._gpu_only_disabled = True

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if self._gpu_only_disabled:
            logger.warning("GPU-only mode: CatBoost GPU unavailable; skipping.")
            self.model = None
            return
        if not CAT_AVAILABLE:
            logger.warning("CatBoost not available")
            return
        try:
            if isinstance(x.index, pd.DatetimeIndex) and not x.index.is_monotonic_increasing:
                order = np.argsort(x.index.view("int64"))
                x = x.iloc[order]
                y = y.iloc[order]

            # Clean inf values (replace with nan)
            x = x.replace([np.inf, -np.inf], np.nan)

            uniq, counts = np.unique(y, return_counts=True)
            class_weights = {
                int(cls): float(len(y) / (len(uniq) * cnt)) for cls, cnt in zip(uniq, counts, strict=False) if cnt > 0
            }

            params = self.params.copy()
            params["class_weights"] = list(class_weights.values()) if class_weights else None

            x_train = x
            y_train = y
            eval_set = None
            if len(y) > 500:
                try:
                    validate_time_ordering(x, context="CatBoostExpert.fit")
                    embargo = max(24, int(len(y) * 0.01))
                    x_train, x_val, y_train_s, y_val_s = time_series_train_val_split(
                        x,
                        y,
                        val_ratio=0.15,
                        min_train_samples=100,
                        embargo_samples=embargo,
                    )
                    y_train = y_train_s
                    y_val = y_val_s
                    eval_set = (x_val, y_val)
                except ValueError:
                    split_idx = int(len(x) * 0.85)
                    x_train, x_val = x.iloc[:split_idx], x.iloc[split_idx:]
                    y_train = y.iloc[:split_idx]
                    y_val = y.iloc[split_idx:]
                    eval_set = (x_val, y_val)

            if eval_set is not None:
                es_pat, _ = get_early_stop_params(50, 0.0)
                if es_pat > 0:
                    params.setdefault("od_type", "Iter")
                    params.setdefault("od_wait", int(es_pat))
                    params.setdefault("use_best_model", True)

            self.model = cb.CatBoostClassifier(**params)
            if eval_set is not None:
                self.model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
            else:
                self.model.fit(x_train, y_train)
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self._gpu_only_disabled:
            raise RuntimeError("GPU-only mode: CatBoost skipped.")
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


class CatBoostAltExpert(CatBoostExpert):
    """Alternate CatBoost preset (CPU/GPU capable) to diversify tree ensemble."""

    def __init__(self, params: dict[str, Any] = None) -> None:
        defaults = {
            "iterations": 900,
            "depth": 10,
            "learning_rate": 0.03,
            "loss_function": "MultiClass",
            "random_seed": 7,
            "verbose": False,
            "thread_count": -1,
            "l2_leaf_reg": 6.0,
            "random_strength": 1.5,
        }
        if params:
            defaults.update(params)
        super().__init__(params=defaults)


class XGBoostRFExpert(XGBoostExpert):
    """Random-forest flavored XGBoost (CPU/GPU via XGBoost)."""

    def __init__(self, params: dict[str, Any] = None) -> None:
        defaults = {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.3,
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "colsample_bytree": 0.8,
            "num_parallel_tree": 8,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
        }
        if params:
            defaults.update(params)
        super().__init__(params=defaults)


class XGBoostDARTExpert(XGBoostExpert):
    """DART (dropout) XGBoost variant (CPU/GPU via XGBoost)."""

    def __init__(self, params: dict[str, Any] = None) -> None:
        defaults = {
            "n_estimators": 600,
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
            "booster": "dart",
            "rate_drop": 0.10,
            "skip_drop": 0.50,
            "sample_type": "uniform",
            "normalize_type": "tree",
            "tree_method": "hist",
        }
        if params:
            defaults.update(params)
        super().__init__(params=defaults)

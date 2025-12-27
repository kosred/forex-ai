import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import ax  # type: ignore
    AX_AVAILABLE = True
except ImportError:
    AX_AVAILABLE = False

try:
    from ax.service.ax_client import AxClient  # type: ignore
    AX_CLIENT_AVAILABLE = True
except Exception:
    AxClient = None
    AX_CLIENT_AVAILABLE = False

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.ax import AxSearch

    RAY_AVAILABLE = True
except Exception:
    ray = None  # type: ignore
    tune = None  # type: ignore
    ASHAScheduler = None  # type: ignore
    ConcurrencyLimiter = None  # type: ignore
    AxSearch = None  # type: ignore
    RAY_AVAILABLE = False



from ..core.config import Settings
from ..core.system import normalize_device_preference
from ..models.device import get_available_gpus, select_device
from ..models.transformers import TransformerExpertTorch
from ..models.trees import (
    CatBoostAltExpert,
    LightGBMExpert,
    RandomForestExpert,
    XGBoostDARTExpert,
    XGBoostRFExpert,
)
from ..models.mlp import MLPExpert
from ..strategy.fast_backtest import (
    fast_evaluate_strategy,
    infer_pip_metrics,
    infer_sl_tp_pips_auto,
)

# Dynamic GPU/CPU model selection - use GPU versions when available for HPO
_gpus = get_available_gpus()
if _gpus:
    from ..models.kan_gpu import KANExpert
    from ..models.nbeats_gpu import NBeatsExpert
    from ..models.tabnet_gpu import TabNetExpert
    from ..models.tide_gpu import TiDEExpert
else:
    from ..models.kan import KANExpert
    from ..models.nbeats import NBeatsExpert
    from ..models.tabnet import TabNetExpert
    from ..models.tide import TiDEExpert

logger = logging.getLogger(__name__)
class HyperparameterOptimizer:
    """
    Auto-tunes model hyperparameters using Ray Tune + Ax/BoTorch (default).
    Objective:
      - If OHLC metadata is available: prop-aware composite score (profit-like backtest + small accuracy term).
      - Otherwise: fallback to accuracy only (trial values will look like ~0.xx).
    """

    def __init__(self, settings: Settings, cache_dir: str = "cache") -> None:
        self.settings = settings
        self.hpo_backend = str(getattr(self.settings.models, "hpo_backend", "ax") or "ax").lower()
        hpo_trials = int(getattr(self.settings.models, "hpo_trials", 0) or 0)
        self.n_trials = max(1, hpo_trials if hpo_trials > 0 else 25)
        self.cache_dir = Path(cache_dir) / "hpo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ray_dir = Path(cache_dir) / "ray_tune"
        self.ray_dir.mkdir(parents=True, exist_ok=True)
        self.available = RAY_AVAILABLE and AX_AVAILABLE
        self.meta_df: pd.DataFrame | None = None
        self.ray_max_concurrency = max(
            1, int(getattr(self.settings.models, "ray_tune_max_concurrency", 1) or 1)
        )

        # FIX: Sync with Risk Manager's threshold to prevent "Training/Trading Gap"
        # If we train for 0.55 but trade at 0.70, we select models that fail in prod.
        risk_thresh = float(getattr(self.settings.risk, "min_confidence_threshold", 0.55))
        model_thresh = float(getattr(self.settings.models, "prop_conf_threshold", 0.55))
        self.prop_conf_threshold = max(risk_thresh, model_thresh)

        self.prop_min_trades = int(getattr(self.settings.models, "prop_min_trades", 0))
        self.prop_weight = float(getattr(self.settings.models, "prop_metric_weight", 1.0))
        self.acc_weight = float(getattr(self.settings.models, "prop_accuracy_weight", 0.1))
        dev_pref = normalize_device_preference(getattr(self.settings.system, "enable_gpu_preference", "auto"))
        self.device_pool = get_available_gpus() if dev_pref != "cpu" else []
        if dev_pref == "gpu" and not self.device_pool:
            logger.warning("GPU requested but no CUDA devices detected; falling back to CPU.")
        self.default_device = select_device("cpu" if dev_pref == "cpu" else "auto")
        self.stop_event: Any | None = None
        self._hpo_cpu_threads: int = 0

        if not (RAY_AVAILABLE and AX_AVAILABLE):
            logger.warning("Ray Tune/Ax not available. Skipping optimization.")
        elif self.hpo_backend != "none":
            if AX_CLIENT_AVAILABLE:
                logger.info("HPO backend: Ax/BoTorch via AxClient.")
            else:
                logger.info("HPO backend: AxSearch (AxClient unavailable).")

    def _should_stop(self) -> bool:
        try:
            return bool(self.stop_event and getattr(self.stop_event, "is_set", lambda: False)())
        except Exception:
            return False

    def _ray_trials_for_model(self, name: str) -> int:
        trials = int(self.n_trials)
        if name in {"TabNet", "N-BEATS", "TiDE", "KAN"}:
            trials = max(5, trials // 2)
        if name == "Transformer":
            trials = max(3, trials // 3)
        if (str(self.default_device) == "cpu" and not self.device_pool) and name in {"TabNet", "N-BEATS", "TiDE", "KAN", "Transformer"}:
            trials = min(trials, 3)
        return max(1, int(trials))

    def _ax_search_space(self, name: str, y_train: pd.Series) -> list[dict[str, Any]]:
        if name == "LightGBM":
            return [
                {"name": "n_estimators", "type": "range", "bounds": [100, 1000], "value_type": "int"},
                {"name": "num_leaves", "type": "range", "bounds": [20, 150], "value_type": "int"},
                {"name": "learning_rate", "type": "range", "bounds": [0.01, 0.3], "log_scale": True},
                {"name": "feature_fraction", "type": "range", "bounds": [0.5, 1.0]},
                {"name": "bagging_fraction", "type": "range", "bounds": [0.5, 1.0]},
                {"name": "min_child_samples", "type": "range", "bounds": [10, 100], "value_type": "int"},
            ]
        if name == "RandomForest":
            return [
                {"name": "n_estimators", "type": "range", "bounds": [100, 300], "value_type": "int"},
                {"name": "max_depth", "type": "range", "bounds": [8, 16], "value_type": "int"},
                {"name": "min_samples_split", "type": "range", "bounds": [2, 10], "value_type": "int"},
                {"name": "min_samples_leaf", "type": "range", "bounds": [1, 5], "value_type": "int"},
                {"name": "max_features", "type": "range", "bounds": [0.5, 1.0]},
                {"name": "bootstrap", "type": "choice", "values": [True, False]},
            ]
        if name == "XGBoostRF":
            return [
                {"name": "n_estimators", "type": "range", "bounds": [200, 800], "value_type": "int"},
                {"name": "max_depth", "type": "range", "bounds": [4, 10], "value_type": "int"},
                {"name": "learning_rate", "type": "range", "bounds": [0.05, 0.5], "log_scale": True},
                {"name": "subsample", "type": "range", "bounds": [0.6, 1.0]},
                {"name": "colsample_bynode", "type": "range", "bounds": [0.6, 1.0]},
                {"name": "num_parallel_tree", "type": "range", "bounds": [4, 16], "value_type": "int"},
            ]
        if name == "XGBoostDART":
            return [
                {"name": "n_estimators", "type": "range", "bounds": [300, 900], "value_type": "int"},
                {"name": "max_depth", "type": "range", "bounds": [4, 12], "value_type": "int"},
                {"name": "learning_rate", "type": "range", "bounds": [0.01, 0.1], "log_scale": True},
                {"name": "rate_drop", "type": "range", "bounds": [0.05, 0.3]},
                {"name": "skip_drop", "type": "range", "bounds": [0.1, 0.7]},
            ]
        if name == "CatBoostAlt":
            return [
                {"name": "iterations", "type": "range", "bounds": [400, 1200], "value_type": "int"},
                {"name": "depth", "type": "range", "bounds": [6, 12], "value_type": "int"},
                {"name": "learning_rate", "type": "range", "bounds": [0.01, 0.2], "log_scale": True},
                {"name": "l2_leaf_reg", "type": "range", "bounds": [1.0, 10.0]},
                {"name": "random_strength", "type": "range", "bounds": [0.5, 3.0]},
            ]
        if name == "MLP":
            return [
                {"name": "hidden_dim", "type": "range", "bounds": [128, 512], "value_type": "int"},
                {"name": "n_layers", "type": "range", "bounds": [2, 5], "value_type": "int"},
                {"name": "dropout", "type": "range", "bounds": [0.0, 0.3]},
                {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
                {"name": "batch_size", "type": "range", "bounds": [1024, 8192], "value_type": "int"},
            ]
        if name == "TabNet":
            return [
                {"name": "hidden_dim", "type": "range", "bounds": [32, 128], "value_type": "int"},
                {"name": "n_steps", "type": "range", "bounds": [3, 7], "value_type": "int"},
                {"name": "gamma", "type": "range", "bounds": [1.0, 2.0]},
                {"name": "lambda_sparse", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
                {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
                {"name": "batch_size", "type": "range", "bounds": [512, 8192], "value_type": "int"},
            ]
        if name == "N-BEATS":
            return [
                {"name": "hidden_dim", "type": "range", "bounds": [64, 256], "value_type": "int"},
                {"name": "n_layers", "type": "range", "bounds": [2, 6], "value_type": "int"},
                {"name": "n_blocks", "type": "range", "bounds": [1, 4], "value_type": "int"},
                {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
                {"name": "batch_size", "type": "range", "bounds": [512, 4096], "value_type": "int"},
            ]
        if name == "TiDE":
            return [
                {"name": "hidden_dim", "type": "range", "bounds": [64, 512], "value_type": "int"},
                {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
                {"name": "batch_size", "type": "range", "bounds": [1024, 8192], "value_type": "int"},
            ]
        if name == "KAN":
            return [
                {"name": "hidden_dim", "type": "range", "bounds": [32, 128], "value_type": "int"},
                {"name": "grid_size", "type": "range", "bounds": [3, 8], "value_type": "int"},
                {"name": "spline_order", "type": "range", "bounds": [2, 4], "value_type": "int"},
                {"name": "scale_noise", "type": "range", "bounds": [0.05, 0.2]},
                {"name": "scale_base", "type": "range", "bounds": [0.5, 2.0]},
                {"name": "scale_spline", "type": "range", "bounds": [0.5, 2.0]},
                {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
                {"name": "batch_size", "type": "range", "bounds": [512, 8192], "value_type": "int"},
            ]
        if name == "Transformer":
            return [
                {"name": "d_model", "type": "choice", "values": [64, 128, 256]},
                {"name": "n_heads", "type": "choice", "values": [4, 8]},
                {"name": "n_layers", "type": "range", "bounds": [2, 4], "value_type": "int"},
                {"name": "lr", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True},
                {"name": "batch_size", "type": "range", "bounds": [32, 256], "value_type": "int"},
            ]
        return []

    def _ray_pick_device(self, name: str) -> str:
        try:
            if ray and ray.is_initialized():
                gpu_ids = ray.get_gpu_ids()
                if gpu_ids:
                    return f"cuda:{int(gpu_ids[0])}"
        except Exception:
            pass
        if self.device_pool:
            return self.device_pool[0]
        return "cpu"

    def _ray_create_model(self, name: str, params: dict[str, Any], device: str, y_train: pd.Series):
        params = params.copy()
        if name == "TabNet":
            hidden = params.pop("hidden_dim", None)
            if hidden is not None:
                params.setdefault("n_d", int(hidden))
                params.setdefault("n_a", int(hidden))
        if name == "LightGBM":
            params["objective"] = "binary" if y_train.nunique() == 2 else "multiclass"
            params["num_class"] = int(y_train.nunique())
            params["n_jobs"] = int(self._hpo_cpu_threads) if self._hpo_cpu_threads > 0 else -1
            if isinstance(device, str) and device.startswith("cuda:"):
                try:
                    params["device_type"] = "gpu"
                    params["gpu_device_id"] = int(device.split(":")[1])
                    params.setdefault("max_bin", 63)
                except Exception:
                    pass
            return LightGBMExpert(params=params)
        if name == "RandomForest":
            if isinstance(device, str) and device.startswith("cuda:"):
                try:
                    params["gpu_device_id"] = int(device.split(":")[1])
                except Exception:
                    pass
            params.setdefault("random_state", 42)
            return RandomForestExpert(params=params)
        if name == "XGBoostRF":
            params.setdefault("random_state", 42)
            return XGBoostRFExpert(params=params)
        if name == "XGBoostDART":
            params.setdefault("random_state", 42)
            return XGBoostDARTExpert(params=params)
        if name == "CatBoostAlt":
            params.setdefault("random_seed", 42)
            return CatBoostAltExpert(params=params)
        if name == "N-BEATS":
            params.setdefault("max_time_sec", int(self.settings.models.nbeats_train_seconds))
            params.setdefault("device", device)
            return NBeatsExpert(**self._filter_kwargs(NBeatsExpert, params))
        if name == "TiDE":
            params.setdefault("max_time_sec", int(self.settings.models.tide_train_seconds))
            params.setdefault("device", device)
            return TiDEExpert(**self._filter_kwargs(TiDEExpert, params))
        if name == "KAN":
            params.setdefault("max_time_sec", int(self.settings.models.kan_train_seconds))
            params.setdefault("device", device)
            return KANExpert(**self._filter_kwargs(KANExpert, params))
        if name == "Transformer":
            params.setdefault("max_time_sec", int(self.settings.models.transformer_train_seconds))
            params.setdefault("device", device)
            return TransformerExpertTorch(**self._filter_kwargs(TransformerExpertTorch, params))
        if name == "TabNet":
            params.setdefault("max_time_sec", int(self.settings.models.tabnet_train_seconds))
            params.setdefault("device", device)
            return TabNetExpert(**self._filter_kwargs(TabNetExpert, params))
        if name == "MLP":
            params.setdefault("max_time_sec", int(self.settings.models.transformer_train_seconds))
            params.setdefault("device", device)
            return MLPExpert(**self._filter_kwargs(MLPExpert, params))
        return None

    def _filter_kwargs(self, cls: type, params: dict[str, Any]) -> dict[str, Any]:
        try:
            sig = inspect.signature(cls.__init__)
            valid = set(sig.parameters.keys())
            return {k: v for k, v in params.items() if k in valid}
        except Exception:
            return params

    def _infer_meta_trading_days(self, meta: pd.DataFrame | None) -> float | None:
        if meta is None:
            return None
        try:
            ts: pd.DatetimeIndex | None
            idx = getattr(meta, "index", None)
            if isinstance(idx, pd.DatetimeIndex):
                ts = idx
            else:
                ts = None
                for col in ("timestamp", "time", "datetime", "date"):
                    if col in meta.columns:
                        ts = pd.to_datetime(meta[col], utc=True, errors="coerce")
                        break
            if ts is None:
                return None
            ts = pd.DatetimeIndex(ts)
            if ts.tz is not None:
                ts = ts.tz_convert("UTC")
            ts = ts[~ts.isna()]
            if len(ts) == 0:
                return None
            mask = ts.weekday < 5
            if not np.any(mask):
                return None
            days = pd.unique(ts[mask].normalize())
            return float(len(days))
        except Exception:
            return None

    def _prop_required_trades(self, meta_val: pd.DataFrame | None) -> float:
        base = float(self.prop_min_trades)
        if base <= 0:
            return 0.0
        if not bool(getattr(self.settings.risk, "prop_firm_rules", False)):
            return base
        if meta_val is None:
            return base
        trading_days = self._infer_meta_trading_days(meta_val)
        if trading_days is None:
            return base
        days_per_month = 21.0
        try:
            days_per_month = float(os.environ.get("FOREX_BOT_TRADING_DAYS_PER_MONTH", "21") or 21.0)
        except Exception:
            days_per_month = 21.0
        days_per_month = max(1.0, float(days_per_month))
        required = base * (trading_days / days_per_month)
        return max(0.0, float(required))

    @staticmethod
    def _pad_probs(probs: np.ndarray) -> np.ndarray:
        """
        HPC UNIFIED PROTOCOL: Force output to [Neutral, Buy, Sell].
        """
        if probs is None or len(probs) == 0:
            return np.zeros((0, 3), dtype=float)
        arr = np.asarray(probs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n = arr.shape[0]

        out = np.zeros((n, 3), dtype=float)
        if arr.shape[1] == 3:
            return arr
        elif arr.shape[1] == 2:
            out[:, 0] = arr[:, 0]
            out[:, 1] = arr[:, 1]
            return out

        out[:, 0] = 1.0 - arr[:, 0]
        out[:, 1] = arr[:, 0]
        return out

    def _objective_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        meta_val: pd.DataFrame | None,
    ) -> dict[str, float]:
        y_true_arr = np.asarray(y_true, dtype=int)
        y_true_arr = np.where(y_true_arr == 2, -1, y_true_arr).astype(int, copy=False)

        probs = self._pad_probs(y_pred_proba)
        n = min(len(y_true_arr), len(probs))
        if n <= 0:
            return {
                "prop_score": 0.0,
                "drawdown": 0.0,
                "daily_dd": 0.0,
                "accuracy": 0.0,
                "trades": 0.0,
            }
        y_true_arr = y_true_arr[:n]
        probs = probs[:n]
        if meta_val is not None and len(meta_val) != n:
            try:
                meta_val = meta_val.iloc[:n]
            except Exception:
                meta_val = None

        p_buy = probs[:, 1]
        p_sell = probs[:, 2]
        trade_prob = np.maximum(p_buy, p_sell)
        direction = np.where(p_buy >= p_sell, 1, -1).astype(np.int8, copy=False)
        signals = np.where(trade_prob >= self.prop_conf_threshold, direction, 0).astype(np.int8, copy=False)
        acc = float(np.mean(signals.astype(int) == y_true_arr)) if len(y_true_arr) else 0.0

        if meta_val is None or not {"close", "high", "low"}.issubset(set(meta_val.columns)):
            return {
                "prop_score": float(acc),
                "drawdown": 0.0,
                "daily_dd": 0.0,
                "accuracy": float(acc),
                "trades": 0.0,
            }

        sl_cfg = getattr(self.settings.risk, "meta_label_sl_pips", None)
        tp_cfg = getattr(self.settings.risk, "meta_label_tp_pips", None)
        rr = float(getattr(self.settings.risk, "min_risk_reward", 2.0))
        max_hold = int(getattr(self.settings.risk, "triple_barrier_max_bars", 0) or 0)
        trailing_enabled = bool(getattr(self.settings.risk, "trailing_enabled", False))
        trailing_mult = float(getattr(self.settings.risk, "trailing_atr_multiplier", 1.0) or 1.0)
        trailing_trigger_r = float(getattr(self.settings.risk, "trailing_be_trigger_r", 1.0) or 1.0)
        spread = float(getattr(self.settings.risk, "backtest_spread_pips", 1.5))
        commission = float(getattr(self.settings.risk, "commission_per_lot", 7.0))
        dd_limit = float(getattr(self.settings.risk, "total_drawdown_limit", 0.08))
        daily_dd_limit = float(getattr(self.settings.risk, "daily_drawdown_limit", 0.04) or 0.04)

        close_all = meta_val["close"].to_numpy()
        high_all = meta_val["high"].to_numpy()
        low_all = meta_val["low"].to_numpy()
        atr_all = meta_val["atr"].to_numpy() if "atr" in meta_val.columns else None

        month_indices_all = np.zeros(len(meta_val), dtype=np.int64)
        day_indices_all = np.zeros(len(meta_val), dtype=np.int64)
        try:
            ts: pd.DatetimeIndex | None
            idx = getattr(meta_val, "index", None)
            if isinstance(idx, pd.DatetimeIndex):
                ts = idx
            else:
                ts = None
                for col in ("timestamp", "time", "datetime", "date"):
                    if col in meta_val.columns:
                        ts = pd.DatetimeIndex(pd.to_datetime(meta_val[col], utc=True, errors="coerce"))
                        break
            if ts is not None and len(ts) == len(meta_val):
                if ts.tz is not None:
                    ts = ts.tz_convert("UTC")
                raw = ts.year.to_numpy() * 12 + ts.month.to_numpy()
                month_indices_all = np.nan_to_num(raw, nan=0.0).astype(np.int64, copy=False)
                day_raw = (
                    ts.year.to_numpy() * 10000
                    + ts.month.to_numpy() * 100
                    + ts.day.to_numpy()
                )
                day_indices_all = np.nan_to_num(day_raw, nan=0.0).astype(np.int64, copy=False)
        except Exception:
            month_indices_all = np.zeros(len(meta_val), dtype=np.int64)
            day_indices_all = np.zeros(len(meta_val), dtype=np.int64)

        try:
            days_per_month = float(os.environ.get("FOREX_BOT_TRADING_DAYS_PER_MONTH", "21") or 21.0)
        except Exception:
            days_per_month = 21.0
        tdays = self._infer_meta_trading_days(meta_val)
        months = max(1e-6, (tdays / days_per_month)) if tdays else 1.0
        min_monthly = float(getattr(self.settings.risk, "monthly_profit_target_pct", 0.04) or 0.04)

        def _invalid(drawdown: float, daily_dd: float) -> dict[str, float]:
            return {
                "prop_score": -1e9,
                "drawdown": float(drawdown),
                "daily_dd": float(daily_dd),
                "accuracy": float(acc),
                "trades": 0.0,
            }

        def _penalty_factor(drawdown: float, daily_dd: float, monthly_ret: float) -> float:
            penalty = 1.0
            if dd_limit > 0 and drawdown > dd_limit:
                excess = max(0.0, drawdown - dd_limit)
                penalty *= max(0.1, 1.0 - (excess / dd_limit))
            if daily_dd_limit > 0 and daily_dd > daily_dd_limit:
                excess = max(0.0, daily_dd - daily_dd_limit)
                penalty *= max(0.1, 1.0 - (excess / daily_dd_limit))
            if min_monthly > 0 and monthly_ret < min_monthly:
                penalty *= max(0.1, monthly_ret / min_monthly)
            return float(penalty)

        sym_series = meta_val["symbol"] if "symbol" in meta_val.columns else None
        if sym_series is not None:
            sym_arr = np.asarray(sym_series)
            uniq_syms = pd.unique(sym_arr)
            total_trades = 0.0
            weighted_score = 0.0
            weighted_monthly = 0.0
            weighted_sortino = 0.0
            weighted_calmar = 0.0
            weighted_pf = 0.0
            max_dd = 0.0
            max_daily = 0.0
            for sym in uniq_syms:
                mask = sym_arr == sym
                if not np.any(mask):
                    continue
                close = close_all[mask]
                high = high_all[mask]
                low = low_all[mask]
                sig = signals[mask]
                month_idx = month_indices_all[mask]
                day_idx = day_indices_all[mask]
                pip_size, pip_value_per_lot = infer_pip_metrics(str(sym))
                if sl_cfg is None or float(sl_cfg) <= 0:
                    atr_vals = atr_all[mask] if atr_all is not None else None
                    auto = infer_sl_tp_pips_auto(
                        open_prices=meta_val["open"].to_numpy(dtype=np.float64)[mask]
                        if "open" in meta_val.columns
                        else close,
                        high_prices=high,
                        low_prices=low,
                        close_prices=close,
                        atr_values=atr_vals,
                        pip_size=pip_size,
                        atr_mult=float(getattr(self.settings.risk, "atr_stop_multiplier", 1.5)),
                        min_rr=rr,
                        min_dist=float(getattr(self.settings.risk, "meta_label_min_dist", 0.0)),
                        settings=self.settings,
                    )
                    if auto is None:
                        continue
                    sl_pips, tp_pips = auto
                else:
                    sl_pips = float(sl_cfg)
                    if tp_cfg is None or float(tp_cfg) <= 0:
                        tp_pips = sl_pips * rr
                    else:
                        tp_pips = max(float(tp_cfg), sl_pips * rr)
                metrics = fast_evaluate_strategy(
                    close_prices=close,
                    high_prices=high,
                    low_prices=low,
                    signals=sig,
                    month_indices=month_idx,
                    day_indices=day_idx,
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    max_hold_bars=max_hold,
                    trailing_enabled=trailing_enabled,
                    trailing_atr_multiplier=trailing_mult,
                    trailing_be_trigger_r=trailing_trigger_r,
                    pip_value=pip_size,
                    spread_pips=spread,
                    commission_per_trade=commission,
                    pip_value_per_lot=pip_value_per_lot,
                )

                net_profit, _sharpe, sortino, dd, _win_rate, profit_factor, _exp, _sqn, trades, _r2, daily_dd = metrics
                max_dd = max(max_dd, float(dd))
                max_daily = max(max_daily, float(daily_dd))
                if trades <= 0:
                    continue
                init_balance = float(getattr(self.settings.risk, "initial_balance", 100000.0) or 100000.0)
                monthly_ret = (net_profit / init_balance) / months
                calmar = (monthly_ret / dd) if dd > 1e-9 else 0.0
                prop_score = (
                    monthly_ret * 100.0
                    + 0.6 * sortino
                    + 0.4 * calmar
                    + 0.2 * profit_factor
                    - 50.0 * max(0.0, dd - dd_limit)
                )
                prop_score *= _penalty_factor(float(dd), float(daily_dd), float(monthly_ret))
                total_trades += float(trades)
                weighted_score += float(trades) * float(prop_score)
                weighted_monthly += float(trades) * float(monthly_ret)
                weighted_sortino += float(trades) * float(sortino)
                weighted_calmar += float(trades) * float(calmar)
                weighted_pf += float(trades) * float(profit_factor)

            required = float(self._prop_required_trades(meta_val))
            if required > 0 and total_trades < required:
                return _invalid(max_dd, max_daily)
            denom = max(total_trades, 1.0)
            return {
                "prop_score": float(self.prop_weight * (weighted_score / denom) + self.acc_weight * acc),
                "drawdown": float(max_dd),
                "daily_dd": float(max_daily),
                "accuracy": float(acc),
                "trades": float(total_trades),
                "monthly_return": float(weighted_monthly / denom),
                "sortino": float(weighted_sortino / denom),
                "calmar": float(weighted_calmar / denom),
                "profit_factor": float(weighted_pf / denom),
            }

        pip_size, pip_value_per_lot = infer_pip_metrics(getattr(self.settings.system, "symbol", ""))
        if sl_cfg is None or float(sl_cfg) <= 0:
            auto = infer_sl_tp_pips_auto(
                open_prices=meta_val["open"].to_numpy(dtype=np.float64)
                if "open" in meta_val.columns
                else close_all,
                high_prices=high_all,
                low_prices=low_all,
                close_prices=close_all,
                atr_values=atr_all,
                pip_size=pip_size,
                atr_mult=float(getattr(self.settings.risk, "atr_stop_multiplier", 1.5)),
                min_rr=rr,
                min_dist=float(getattr(self.settings.risk, "meta_label_min_dist", 0.0)),
                settings=self.settings,
            )
            if auto is None:
                return _invalid(1.0, 1.0)
            sl_pips, tp_pips = auto
        else:
            sl_pips = float(sl_cfg)
            if tp_cfg is None or float(tp_cfg) <= 0:
                tp_pips = sl_pips * rr
            else:
                tp_pips = max(float(tp_cfg), sl_pips * rr)
        metrics = fast_evaluate_strategy(
            close_prices=close_all,
            high_prices=high_all,
            low_prices=low_all,
            signals=signals,
            month_indices=month_indices_all,
            day_indices=day_indices_all,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            max_hold_bars=max_hold,
            trailing_enabled=trailing_enabled,
            trailing_atr_multiplier=trailing_mult,
            trailing_be_trigger_r=trailing_trigger_r,
            pip_value=pip_size,
            spread_pips=spread,
            commission_per_trade=commission,
            pip_value_per_lot=pip_value_per_lot,
        )

        net_profit, _sharpe, sortino, dd, win_rate, profit_factor, _exp, _sqn, trades, _r2, daily_dd = metrics
        required = float(self._prop_required_trades(meta_val))
        if required > 0 and trades < required:
            return _invalid(dd, daily_dd)

        init_balance = float(getattr(self.settings.risk, "initial_balance", 100000.0) or 100000.0)
        monthly_ret = (net_profit / init_balance) / months
        calmar = (monthly_ret / dd) if dd > 1e-9 else 0.0
        prop_score = (
            monthly_ret * 100.0
            + 0.6 * sortino
            + 0.4 * calmar
            + 0.2 * profit_factor
            + 0.1 * (win_rate * 100.0)
            - 50.0 * max(0.0, dd - dd_limit)
        )
        prop_score *= _penalty_factor(float(dd), float(daily_dd), float(monthly_ret))
        return {
            "prop_score": float(self.prop_weight * prop_score + self.acc_weight * acc),
            "drawdown": float(dd),
            "daily_dd": float(daily_dd),
            "accuracy": float(acc),
            "trades": float(trades),
            "monthly_return": float(monthly_ret),
            "sortino": float(sortino),
            "calmar": float(calmar),
            "profit_factor": float(profit_factor),
        }

    def _objective_base(self, y_true: np.ndarray, y_pred_proba: np.ndarray, meta_val: pd.DataFrame | None) -> float:
        metrics = self._objective_metrics(y_true, y_pred_proba, meta_val)
        return float(metrics.get("prop_score", 0.0))

    def _ray_trainable(
        self,
        config: dict[str, Any],
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None,
    ) -> None:
        if self._should_stop():
            tune.report(prop_score=-1e9, drawdown=1.0, daily_dd=1.0, accuracy=0.0, trades=0.0)
            return
        device = self._ray_pick_device(model_name)
        model = self._ray_create_model(model_name, config, device, y_train)
        if model is None:
            tune.report(prop_score=-1e9, drawdown=1.0, daily_dd=1.0, accuracy=0.0, trades=0.0)
            return
        try:
            fit_ok = model.fit(X_train, y_train)
            if isinstance(fit_ok, bool) and not fit_ok:
                tune.report(prop_score=-1e9, drawdown=1.0, daily_dd=1.0, accuracy=0.0, trades=0.0)
                return
            preds = model.predict_proba(X_val)
            metrics = self._objective_metrics(y_val, preds, meta_val)
            tune.report(**metrics)
        except Exception as exc:
            logger.warning(f"Ray Tune trial failed for {model_name}: {exc}", exc_info=True)
            tune.report(prop_score=-1e9, drawdown=1.0, daily_dd=1.0, accuracy=0.0, trades=0.0)

    def _ray_tune_model(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None,
    ) -> dict[str, Any]:
        if not (RAY_AVAILABLE and AX_AVAILABLE):
            return {}
        space = self._ax_search_space(name, y_train)
        if not space:
            return {}

        if name == "Transformer":
            max_rows = int(getattr(self.settings.models, "hpo_max_rows", 200_000) or 0)
            if max_rows > 0 and len(X_train) > max_rows:
                idx_train = np.linspace(0, len(X_train) - 1, max_rows, dtype=int)
                X_train = X_train.iloc[idx_train]
                y_train = y_train.iloc[idx_train]
            if max_rows > 0 and len(X_val) > max_rows:
                idx_val = np.linspace(0, len(X_val) - 1, max_rows, dtype=int)
                X_val = X_val.iloc[idx_val]
                y_val = y_val.iloc[idx_val]
                if meta_val is not None:
                    try:
                        meta_val = meta_val.iloc[idx_val]
                    except Exception:
                        meta_val = None

        dd_limit = float(getattr(self.settings.risk, "total_drawdown_limit", 0.08) or 0.08)
        daily_dd_limit = float(getattr(self.settings.risk, "daily_drawdown_limit", 0.04) or 0.04)
        outcome_constraints = []
        strict_constraints = str(os.environ.get("FOREX_BOT_HPO_STRICT_CONSTRAINTS", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        if strict_constraints and meta_val is not None and {"close", "high", "low"}.issubset(set(meta_val.columns)):
            outcome_constraints = [
                f"drawdown <= {dd_limit}",
                f"daily_dd <= {daily_dd_limit}",
            ]

        ax_search = None
        if AX_CLIENT_AVAILABLE and AxClient is not None:
            try:
                ax_client = AxClient()
                ax_client.create_experiment(
                    name=f"HPO_{name}",
                    parameters=space,
                    objective_name="prop_score",
                    minimize=False,
                    outcome_constraints=outcome_constraints or None,
                )
                logger.info(
                    "AxClient experiment ready for %s (BoTorch). Constraints=%s",
                    name,
                    outcome_constraints or "none",
                )
                ax_search = AxSearch(
                    ax_client=ax_client,
                    metric="prop_score",
                    mode="max",
                )
            except Exception as exc:
                logger.warning(f"AxClient init failed, falling back to AxSearch: {exc}")
                ax_search = None

        if ax_search is None:
            logger.info("AxSearch fallback for %s (no AxClient).", name)
            ax_search = AxSearch(
                space=space,
                metric="prop_score",
                mode="max",
                outcome_constraints=outcome_constraints or None,
            )

        try:
            if hasattr(ax_search, "set_search_properties"):
                sig = inspect.signature(ax_search.set_search_properties)
                kwargs = {}
                if "metric" in sig.parameters:
                    kwargs["metric"] = "prop_score"
                if "mode" in sig.parameters:
                    kwargs["mode"] = "max"
                if "config" in sig.parameters:
                    kwargs["config"] = space
                elif "parameters" in sig.parameters:
                    kwargs["parameters"] = space
                elif "space" in sig.parameters:
                    kwargs["space"] = space
                if kwargs:
                    ax_search.set_search_properties(**kwargs)
        except Exception as exc:
            logger.debug(f"AxSearch set_search_properties failed: {exc}")
        max_concurrent = self.ray_max_concurrency
        if name in {"TabNet", "N-BEATS", "TiDE", "KAN", "Transformer"} and self.device_pool:
            max_concurrent = 1
        if ConcurrencyLimiter is not None:
            ax_search = ConcurrencyLimiter(ax_search, max_concurrent=max_concurrent)

        scheduler = ASHAScheduler(metric="prop_score", mode="max") if ASHAScheduler else None

        resources = {"cpu": 1, "gpu": 0}
        if self.device_pool and name in {"TabNet", "N-BEATS", "TiDE", "KAN", "Transformer"}:
            resources["gpu"] = 1
        if name in {"LightGBM", "RandomForest", "XGBoostRF", "XGBoostDART", "CatBoostAlt"} and self.device_pool:
            resources["gpu"] = 1

        try:
            if ray and not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)
        except Exception:
            pass

        trials = self._ray_trials_for_model(name)
        trainable = tune.with_parameters(
            self._ray_trainable,
            model_name=name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            meta_val=meta_val,
        )

        analysis = tune.run(
            trainable,
            name=f"HPO_{name}",
            search_alg=ax_search,
            num_samples=trials,
            resources_per_trial=resources,
            scheduler=scheduler,
            local_dir=str(self.ray_dir),
            log_to_file=False,
            verbose=1,
            raise_on_failed_trial=False,
        )

        try:
            best_config = analysis.get_best_config(metric="prop_score", mode="max")
        except Exception:
            return {}

        int_keys = {
            "n_estimators",
            "num_leaves",
            "min_child_samples",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "n_layers",
            "n_blocks",
            "n_steps",
            "hidden_dim",
            "grid_size",
            "spline_order",
            "batch_size",
            "d_model",
            "n_heads",
        }
        for key in int_keys:
            if key in best_config:
                try:
                    best_config[key] = int(best_config[key])
                except Exception:
                    pass

        if name == "TabNet" and "hidden_dim" in best_config:
            hidden = int(best_config.get("hidden_dim", 64))
            best_config["n_d"] = hidden
            best_config["n_a"] = hidden
            best_config.pop("hidden_dim", None)

        if name == "LightGBM":
            best_config = best_config.copy()
            best_config["objective"] = "binary" if y_train.nunique() == 2 else "multiclass"
            best_config["num_class"] = int(y_train.nunique())
        return best_config

    async def _optimize_all_ray_tune_async(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None,
    ) -> dict[str, dict[str, Any]]:
        logger.info(f"HPC HPO: Launching parallel tuning for all models across {len(self.device_pool)} GPUs...")
        best_params: dict[str, dict[str, Any]] = {}
        
        all_models = [
            "LightGBM", "RandomForest", "XGBoostRF", "XGBoostDART", "CatBoostAlt", "MLP",
            "TabNet", "N-BEATS", "TiDE", "KAN", "Transformer"
        ]

        async def _tune_single(name):
            if self._should_stop(): return name, {}
            try:
                # Run the heavy ray tune in a thread to keep event loop responsive
                result = await asyncio.to_thread(
                    self._ray_tune_model, name, X_train, y_train, X_val, y_val, meta_val
                )
                return name, result
            except Exception as e:
                logger.error(f"HPO failed for {name}: {e}")
                return name, {}

        # HPC BLITZ: Launch every model study at the same time
        results = await asyncio.gather(*[_tune_single(m) for m in all_models])
        
        for name, params in results:
            if params:
                best_params[name] = params

        self._save_params(best_params)
        return best_params

    def _optimize_all_ray_tune(self, *args, **kwargs):
        # Compatibility wrapper for sync callers
        return asyncio.run(self._optimize_all_ray_tune_async(*args, **kwargs))

    def optimize_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta_df: pd.DataFrame | None = None,  # noqa: N803
        stop_event: Any | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run optimization for all supported models."""
        if self.hpo_backend == "none":
            logger.info("HPO backend disabled; skipping hyperparameter optimization.")
            return {}
        if not (RAY_AVAILABLE and AX_AVAILABLE):
            logger.warning("Ray Tune/Ax not installed; hyperparameter optimization skipped.")
            return {}
        self.stop_event = stop_event
        self.meta_df = meta_df

        # Cap dataset size to keep trials fast on CPU/limited nodes
        cap = int(getattr(self.settings.models, "hpo_max_rows", 0) or 0)
        if cap > 0 and len(X) > cap:
            idx = np.arange(len(X) - cap, len(X))
            X = X.iloc[idx]
            y = y.iloc[idx]
            if self.meta_df is not None:
                try:
                    self.meta_df = self.meta_df.iloc[idx]
                except Exception:
                    self.meta_df = None

        split_idx = int(len(X) * 0.8)

        # FIX: Add embargo buffer to prevent leakage from overlapping labels
        # Labels often look forward by max_hold_bars (e.g. 50-100 bars).
        # We must skip these bars in the validation set.
        try:
            max_hold = int(getattr(self.settings.risk, "meta_label_max_hold_bars", 100) or 100)
        except Exception:
            max_hold = 100

        embargo_size = max(50, max_hold)
        val_start = min(len(X), split_idx + embargo_size)

        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

        X_val = X.iloc[val_start:]
        y_val = y.iloc[val_start:]

        if len(X_val) < 100:
            logger.info("Validation set too small after embargo. Reducing embargo/split.")
            # Fallback: just use standard split if dataset is tiny
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]

        meta_val = None
        if self.meta_df is not None:
            try:
                meta_val = (
                    self.meta_df.iloc[val_start:]
                    if len(X_val) == (len(X) - val_start)
                    else self.meta_df.iloc[split_idx:]
                )
            except Exception:
                meta_val = None

        if meta_val is None:
            logger.info(
                "HPO objective: accuracy-only (no OHLC metadata). "
                "Trial values will not reflect profitability in this mode."
            )
        elif not {"close", "high", "low"}.issubset(set(meta_val.columns)):
            logger.info(
                "HPO objective: accuracy-only (metadata missing OHLC columns). "
                f"Columns={sorted(set(meta_val.columns))}"
            )
        else:
            dd_limit = float(getattr(self.settings.risk, "total_drawdown_limit", 0.08))
            required = float(self._prop_required_trades(meta_val))
            tdays = self._infer_meta_trading_days(meta_val)
            span_msg = f" val_trading_days={tdays:.0f}" if tdays is not None else ""
            logger.info(
                "HPO objective: prop-aware (OHLC available) "
                f"| conf>={self.prop_conf_threshold:.2f} min_trades>={required:.1f}{span_msg} "
                f"dd_limit={dd_limit:.2%}"
            )

        best_params = {}

        if self._should_stop():
            logger.info("Stop requested before optimization; skipping all HPO studies.")
            return best_params

        best_params = self._optimize_all_ray_tune(
            X_train, y_train, X_val, y_val, meta_val
        )
        return best_params

    def _save_params(self, params: dict[str, Any]) -> None:
        with open(self.cache_dir / "best_params.json", "w") as f:
            json.dump(params, f, indent=2)

    def load_params(self) -> dict[str, Any]:
        p = self.cache_dir / "best_params.json"
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load best_params.json: {e}")
        return {}

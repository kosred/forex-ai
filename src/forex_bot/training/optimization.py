import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    import redis  # type: ignore

    REDIS_PY_AVAILABLE = True
except Exception:
    redis = None  # type: ignore
    REDIS_PY_AVAILABLE = False

try:
    import optuna
    from optuna.trial import Trial, TrialState

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LGBM_AVAILABLE = False


from ..core.config import Settings
from ..core.system import normalize_device_preference
from ..models.device import get_available_gpus, select_device
from ..models.evolution import EvoExpertCMA
from ..models.rl import RLExpertPPO, RLExpertSAC, _build_continuous_env
from ..models.transformers import TransformerExpertTorch
from ..models.trees import ExtraTreesExpert, LightGBMExpert, RandomForestExpert
from ..strategy.fast_backtest import fast_evaluate_strategy, infer_pip_metrics

# Dynamic GPU/CPU model selection - use GPU versions when available for Optuna
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
_JOURNAL_STORAGE_SENTINEL = "__journal__"

class HyperparameterOptimizer:
    """
    Auto-tunes model hyperparameters using Optuna.
    Objective:
      - If OHLC metadata is available: prop-aware composite score (profit-like backtest + small accuracy term).
      - Otherwise: fallback to accuracy only (trial values will look like ~0.xx).
    """

    def __init__(self, settings: Settings, cache_dir: str = "cache") -> None:
        self.settings = settings
        self.cache_dir = Path(cache_dir) / "optuna"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = max(1, int(getattr(self.settings.models, "optuna_trials", 25)))
        self.available = OPTUNA_AVAILABLE
        self.meta_df: pd.DataFrame | None = None
        self.reuse_study = bool(getattr(self.settings.models, "optuna_reuse_study", False))

        # FIX: Sync with Risk Manager's threshold to prevent "Training/Trading Gap"
        # If we train for 0.55 but trade at 0.70, we select models that fail in prod.
        risk_thresh = float(getattr(self.settings.risk, "min_confidence_threshold", 0.55))
        model_thresh = float(getattr(self.settings.models, "prop_conf_threshold", 0.55))
        self.prop_conf_threshold = max(risk_thresh, model_thresh)

        self.prop_min_trades = int(getattr(self.settings.models, "prop_min_trades", 30))
        self.prop_weight = float(getattr(self.settings.models, "prop_metric_weight", 1.0))
        self.acc_weight = float(getattr(self.settings.models, "prop_accuracy_weight", 0.1))
        dev_pref = normalize_device_preference(getattr(self.settings.system, "enable_gpu_preference", "auto"))
        self.device_pool = get_available_gpus() if dev_pref != "cpu" else []
        if dev_pref == "gpu" and not self.device_pool:
            logger.warning("GPU requested for Optuna but no CUDA devices detected; falling back to CPU.")
        self.default_device = select_device("cpu" if dev_pref == "cpu" else "auto")
        self.stop_event: Any | None = None
        # Set by `_run_study()` to prevent CPU oversubscription when Optuna runs parallel trials.
        self._optuna_cpu_threads: int = 0

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Skipping optimization.")

        self._study_suffix = None

    def _time_series_holdout(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 3, embargo: int = 100
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create a simple time-series aware train/val split with an embargo gap.
        Falls back to the original split if data is too short.
        """
        try:
            if len(X) < max(embargo * 2, 200):
                return X, pd.DataFrame(index=X.index), y, pd.Series(dtype=y.dtype)
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=embargo)
            splits = list(tscv.split(X))
            if not splits:
                return X, pd.DataFrame(index=X.index), y, pd.Series(dtype=y.dtype)
            train_idx, val_idx = splits[-1]
            return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
        except Exception:
            return X, pd.DataFrame(index=X.index), y, pd.Series(dtype=y.dtype)

    def _dataset_signature(self, X: pd.DataFrame, y: pd.Series) -> str:  # noqa: N803
        """Generate a short hash to distinguish datasets/configs for studies."""
        try:
            vc = y.value_counts().to_dict()
        except Exception:
            vc = {}
        try:
            sample_hash = hashlib.sha256(
                pd.util.hash_pandas_object(X.head(500), index=False).values.tobytes()
            ).hexdigest()
        except Exception:
            sample_hash = hashlib.sha256(str((X.shape, tuple(X.columns))).encode()).hexdigest()
        payload = {
            "rows": len(X),
            "cols": tuple(X.columns),
            "y_counts": vc,
            "sample": sample_hash,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def _should_stop(self) -> bool:
        try:
            return bool(self.stop_event and getattr(self.stop_event, "is_set", lambda: False)())
        except Exception:
            return False

    def optimize_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        meta_df: pd.DataFrame | None = None,  # noqa: N803
        stop_event: Any | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run optimization for all supported models."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not installed; hyperparameter optimization skipped.")
            return {}
        self.stop_event = stop_event
        self.meta_df = meta_df

        # Cap dataset size to keep trials fast on CPU/limited nodes
        cap = int(getattr(self.settings.models, "optuna_max_rows", 0) or 0)
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
                "Optuna objective: accuracy-only (no OHLC metadata). "
                "Trial values will not reflect profitability in this mode."
            )
        elif not {"close", "high", "low"}.issubset(set(meta_val.columns)):
            logger.info(
                "Optuna objective: accuracy-only (metadata missing OHLC columns). "
                f"Columns={sorted(set(meta_val.columns))}"
            )
        else:
            dd_limit = float(getattr(self.settings.risk, "total_drawdown_limit", 0.08))
            required = float(self._prop_required_trades(meta_val))
            tdays = self._infer_meta_trading_days(meta_val)
            span_msg = f" val_trading_days={tdays:.0f}" if tdays is not None else ""
            logger.info(
                "Optuna objective: prop-aware (OHLC available) "
                f"| conf>={self.prop_conf_threshold:.2f} min_trades>={required:.1f}{span_msg} "
                f"dd_limit={dd_limit:.2%}"
            )

        self._study_suffix = self._dataset_signature(X, y)

        best_params = {}

        if self._should_stop():
            logger.info("Stop requested before optimization; skipping all Optuna studies.")
            return best_params

        if LGBM_AVAILABLE:
            best_params["LightGBM"] = self._optimize_lgbm(X_train, y_train, X_val, y_val, meta_val)

        if self._should_stop():
            logger.info("Stop requested; skipping remaining optimization studies.")
            return best_params

        best_params["RandomForest"] = self._optimize_random_forest(X_train, y_train, X_val, y_val, meta_val)

        if self._should_stop():
            logger.info("Stop requested; skipping remaining optimization studies.")
            return best_params

        best_params["ExtraTrees"] = self._optimize_extra_trees(X_train, y_train, X_val, y_val, meta_val)

        if self._should_stop():
            logger.info("Stop requested; skipping remaining optimization studies.")
            return best_params

        best_params["TabNet"] = self._optimize_tabnet(X_train, y_train, X_val, y_val, meta_val)
        if self._should_stop():
            logger.info("Stop requested; skipping remaining optimization studies.")
            return best_params

        best_params["N-BEATS"] = self._optimize_nbeats(X_train, y_train, X_val, y_val, meta_val)
        if self._should_stop():
            logger.info("Stop requested; skipping remaining optimization studies.")
            return best_params

        best_params["TiDE"] = self._optimize_tide(X_train, y_train, X_val, y_val, meta_val)
        if self._should_stop():
            logger.info("Stop requested; skipping remaining optimization studies.")
            return best_params

        best_params["KAN"] = self._optimize_kan(X_train, y_train, X_val, y_val, meta_val)
        if self._should_stop():
            logger.info("Stop requested; skipping remaining optimization studies.")
            return best_params

        best_params["Transformer"] = self._optimize_transformer(X_train, y_train, X_val, y_val, meta_val)

        if hasattr(RLExpertPPO, "__init__") and "rl_ppo" in self.settings.models.ml_models:  # Check if PPO is enabled
            best_params["RL_PPO"] = self._optimize_rl_ppo(X_train, y_train, X_val, y_val, meta_val)

        if self._should_stop():
            logger.info("Stop requested; skipping remaining optimization studies.")
            return best_params

        if hasattr(RLExpertSAC, "__init__") and "rl_sac" in self.settings.models.ml_models:  # Check if SAC is enabled
            best_params["RL_SAC"] = self._optimize_rl_sac(X_train, y_train, X_val, y_val, meta_val)

        if self._should_stop():
            logger.info("Stop requested; skipping remaining optimization studies.")
            return best_params

        if EvoExpertCMA:  # Check for import success
            best_params["Neuroevolution"] = self._optimize_neuroevolution(X_train, y_train, X_val, y_val, meta_val)

        self._save_params(best_params)
        return best_params

    def _study_name(self, base: str) -> str:
        if self.reuse_study:
            return base  # Stable name; reuse across runs even if dataset signature shifts
        if self._study_suffix:
            return f"{base}_{self._study_suffix}"
        return base

    def _pick_device(self, trial_index: int) -> str:
        if self.device_pool:
            return self.device_pool[trial_index % len(self.device_pool)]
        return self.default_device or "cpu"

    def _get_study(self, study_name: str) -> optuna.Study:
        env_url = os.environ.get("OPTUNA_STORAGE_URL", "")
        cfg_url = getattr(self.settings.models, "optuna_storage_url", "")
        storage_url = env_url or cfg_url or "redis://localhost:6379/0"
        storage_url = self._ensure_storage_backend(storage_url)
        full_name = self._study_name(study_name)
        pruner_name = str(getattr(self.settings.models, "optuna_pruner", "asha")).lower()
        if pruner_name == "asha":
            pruner = optuna.pruners.HyperbandPruner()
        elif pruner_name == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        else:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)

        sampler_name = str(getattr(self.settings.models, "optuna_sampler", "tpe")).lower()
        if sampler_name == "random":
            sampler = optuna.samplers.RandomSampler()
        elif sampler_name == "cmaes":
            sampler = optuna.samplers.CmaEsSampler()
        else:
            # Avoid Optuna ExperimentalWarning spam from multivariate/group TPESampler options.
            sampler = optuna.samplers.TPESampler()

        if storage_url == _JOURNAL_STORAGE_SENTINEL:
            # Journal storage is a simple, durable fallback that does not require SQL backends.
            try:
                from optuna.storages import JournalStorage
                from optuna.storages.journal import JournalFileBackend

                journal_path = Path(self.cache_dir) / "optuna.journal"
                journal_path.parent.mkdir(parents=True, exist_ok=True)
                storage = JournalStorage(JournalFileBackend(str(journal_path)))
                return optuna.create_study(
                    study_name=full_name,
                    storage=storage,
                    load_if_exists=True,
                    direction="maximize",
                    pruner=pruner,
                    sampler=sampler,
                )
            except Exception as exc:
                logger.warning(f"Optuna journal storage unavailable; falling back to in-memory (no persistence): {exc}")
                return optuna.create_study(
                    direction="maximize",
                    pruner=pruner,
                    sampler=sampler,
                )

        # Try Redis with new JournalRedisBackend (Optuna 3.1+)
        if storage_url.startswith("redis://"):
            try:
                from optuna.storages import JournalStorage
                from optuna.storages.journal import JournalRedisBackend

                redis_backend = JournalRedisBackend(storage_url)
                storage = JournalStorage(redis_backend)
                logger.info(f"Using Redis JournalStorage for Optuna: {storage_url}")
                return optuna.create_study(
                    study_name=full_name,
                    storage=storage,
                    load_if_exists=True,
                    direction="maximize",
                    pruner=pruner,
                    sampler=sampler,
                )
            except Exception as exc:
                logger.info(f"Redis JournalStorage failed: {exc}, trying file journal")

        try:
            return optuna.create_study(
                study_name=full_name,
                storage=storage_url,
                load_if_exists=True,
                direction="maximize",
                pruner=pruner,
                sampler=sampler,
            )
        except Exception as exc:
            logger.info(f"Optuna storage '{storage_url}' unavailable, trying journal storage: {exc}")

            # Journal storage is a simple, durable fallback that does not require SQL backends.
            try:
                from optuna.storages import JournalStorage
                from optuna.storages.journal import JournalFileBackend

                journal_path = Path(self.cache_dir) / "optuna.journal"
                journal_path.parent.mkdir(parents=True, exist_ok=True)
                storage = JournalStorage(JournalFileBackend(str(journal_path)))
                return optuna.create_study(
                    study_name=full_name,
                    storage=storage,
                    load_if_exists=True,
                    direction="maximize",
                    pruner=pruner,
                    sampler=sampler,
                )
            except Exception as exc2:
                logger.warning(
                    f"Optuna journal storage unavailable; falling back to in-memory (no persistence): {exc2}"
                )
                return optuna.create_study(
                    direction="maximize",
                    pruner=pruner,
                    sampler=sampler,
                )

    def _ensure_storage_backend(self, url: str) -> str:
        """
        Best-effort bootstrap of a fast storage backend.
        Priority: Redis (start local daemon if needed) -> DuckDB -> in-memory fallback.
        """
        # Redis path
        if url.startswith("redis://"):
            if not REDIS_PY_AVAILABLE:
                if os.name == "nt":
                    logger.info("redis-py not installed; using Optuna journal storage on Windows.")
                    return _JOURNAL_STORAGE_SENTINEL
                logger.info("redis-py not installed; switching storage to DuckDB.")
                return self._duckdb_url()
            try:
                client = redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
                client.ping()
                return url
            except Exception:
                # Try to start local redis if host is localhost
                try:
                    from urllib.parse import urlparse

                    parsed = urlparse(url)
                    if parsed.hostname in ("127.0.0.1", "localhost"):
                        if shutil.which("redis-server"):
                            logger.info("Starting local redis-server for Optuna storage...")
                            subprocess.check_call(
                                ["redis-server", "--daemonize", "yes", "--save", "", "--appendonly", "no"]
                            )
                            client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
                            client.ping()
                            return url
                        else:
                            # Try apt-get install redis-server (Linux best-effort)
                            if shutil.which("apt-get"):
                                try:
                                    subprocess.check_call(
                                        ["sudo", "-n", "apt-get", "update"],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                    )
                                    subprocess.check_call(
                                        ["sudo", "-n", "apt-get", "install", "-y", "redis-server"],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                    )
                                    if shutil.which("redis-server"):
                                        subprocess.check_call(
                                            ["redis-server", "--daemonize", "yes", "--save", "", "--appendonly", "no"]
                                        )
                                        client = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
                                        client.ping()
                                        return url
                                except Exception:
                                    pass
                            # Windows: attempt to start installed Redis service if present
                            if os.name == "nt":
                                try:
                                    # common Redis installation paths; best effort
                                    possible = [
                                        r"C:\\Program Files\\Redis\\redis-server.exe",
                                        r"C:\\Program Files (x86)\\Redis\\redis-server.exe",
                                    ]
                                    exe = next((p for p in possible if Path(p).exists()), None)
                                    if exe:
                                        logger.info("Starting local redis-server (Windows)...")
                                        subprocess.Popen(
                                            [exe, "--service-run"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                                        )
                                        client = redis.from_url(url, socket_connect_timeout=3, socket_timeout=3)
                                        client.ping()
                                        return url
                                except Exception:
                                    pass
                except Exception:
                    pass
                if os.name == "nt":
                    logger.info("Redis storage unavailable; using Optuna journal storage on Windows.")
                    return _JOURNAL_STORAGE_SENTINEL
                logger.info("Redis storage unavailable; falling back to DuckDB.")
                return self._duckdb_url()

        # DuckDB path
        if url.startswith("duckdb://"):
            return url

        # Any other URL (Postgres/MySQL) – trust user
        if "://" in url:
            return url

        # Default fallback to DuckDB
        if os.name == "nt":
            return _JOURNAL_STORAGE_SENTINEL
        return self._duckdb_url()

    def _duckdb_url(self) -> str:
        db_path = Path(self.cache_dir) / "optuna.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"duckdb:///{db_path}"

    def _run_study(self, study_name: str, objective: Callable[[Trial], float], target_trials: int) -> optuna.Study:
        """
        Warm-start aware study runner. Resumes existing study and only runs remaining trials.
        Runs trials in PARALLEL across all available GPUs for maximum speed.
        """
        study = self._get_study(study_name)

        def _cb(study, trial):
            if self._should_stop():
                study.stop()

        if self._should_stop():
            logger.info(f"Optuna study {study_name}: stop requested before scheduling; skipping.")
            return study
        completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        remaining = max(0, target_trials - completed)
        if remaining > 0:
            # Parallel execution: use threading for Optuna trials
            # Override with OPTUNA_N_JOBS env var for testing (default: use all GPUs)
            n_parallel = int(os.environ.get("OPTUNA_N_JOBS", len(self.device_pool) if self.device_pool else 1))
            n_parallel = max(1, n_parallel)

            cpu_total = max(1, os.cpu_count() or 1)
            cpu_override = 0
            try:
                cpu_override = int(os.environ.get("FOREX_BOT_OPTUNA_CPU_THREADS", "0") or 0)
            except Exception:
                cpu_override = 0

            self._optuna_cpu_threads = cpu_override if cpu_override > 0 else max(1, cpu_total // n_parallel)

            logger.info(
                f"Optuna study {study_name}: running {remaining} new trials (completed={completed}) "
                f"with n_jobs={n_parallel} and cpu_threads_per_trial={self._optuna_cpu_threads}"
            )
            study.optimize(
                objective,
                n_trials=remaining,
                n_jobs=n_parallel,
                callbacks=[_cb],
                catch=(KeyboardInterrupt, MemoryError),
            )
        else:
            logger.info(f"Optuna study {study_name}: using existing {completed} trials; no new trials scheduled.")
        return study

    @staticmethod
    def _infer_meta_trading_days(meta: pd.DataFrame) -> float | None:
        """
        Best-effort estimate of how many trading days (Mon–Fri) are present in a metadata slice.

        This uses timestamps present in the data, so missing holidays/sessions naturally reduce the count.
        """
        try:
            if meta is None or len(meta) < 2:
                return None

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

            # Count unique weekday dates.
            mask = ts.weekday < 5
            if not np.any(mask):
                return None
            days = pd.unique(ts[mask].normalize())
            return float(len(days))
        except Exception:
            return None

    def _prop_required_trades(self, meta_val: pd.DataFrame | None) -> float:
        """
        Interpret `prop_min_trades` as a *monthly* requirement and scale it to the validation window length
        using trading days (Mon–Fri) when datetime metadata is available.

        Override the assumed trading days per month via `FOREX_BOT_TRADING_DAYS_PER_MONTH` (default: 21).
        """
        base = float(self.prop_min_trades)
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
        return max(1.0, float(required))

    def _objective_base(self, y_true: np.ndarray, y_pred_proba: np.ndarray, meta_val: pd.DataFrame | None) -> float:
        """
        Prop-aware composite objective: prop PnL metric + small accuracy term.
        Hard-rejects strategies that breach DD or have too few trades.
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        # Backward compat: some older pipelines used 2=sell.
        y_true_arr = np.where(y_true_arr == 2, -1, y_true_arr).astype(int, copy=False)

        probs = self._pad_probs(y_pred_proba)
        n = min(len(y_true_arr), len(probs))
        if n <= 0:
            return 0.0
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

        if meta_val is not None and {"close", "high", "low"}.issubset(set(meta_val.columns)):
            tp_pips = float(getattr(self.settings.risk, "meta_label_tp_pips", 20.0))
            sl_pips = float(getattr(self.settings.risk, "meta_label_sl_pips", 20.0))
            spread = float(getattr(self.settings.risk, "backtest_spread_pips", 1.5))
            commission = float(getattr(self.settings.risk, "commission_per_lot", 7.0))
            dd_limit = float(getattr(self.settings.risk, "total_drawdown_limit", 0.08))

            close_all = meta_val["close"].to_numpy()
            high_all = meta_val["high"].to_numpy()
            low_all = meta_val["low"].to_numpy()

            # Month indices for consistency tracking (fast_backtest API).
            month_indices_all = np.zeros(len(meta_val), dtype=np.int64)
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
            except Exception:
                month_indices_all = np.zeros(len(meta_val), dtype=np.int64)

            # Multi-symbol pooled validation: score profitability per symbol and aggregate.
            sym_series = meta_val["symbol"] if "symbol" in meta_val.columns else None
            if sym_series is not None:
                sym_arr = np.asarray(sym_series)
                uniq_syms = pd.unique(sym_arr)
                total_trades = 0.0
                weighted = 0.0
                for sym in uniq_syms:
                    mask = sym_arr == sym
                    if not np.any(mask):
                        continue
                    close = close_all[mask]
                    high = high_all[mask]
                    low = low_all[mask]
                    sig = signals[mask]
                    month_idx = month_indices_all[mask]
                    pip_size, pip_value_per_lot = infer_pip_metrics(str(sym))
                    metrics = fast_evaluate_strategy(
                        close_prices=close,
                        high_prices=high,
                        low_prices=low,
                        signals=sig,
                        month_indices=month_idx,
                        sl_pips=sl_pips,
                        tp_pips=tp_pips,
                        pip_value=pip_size,
                        spread_pips=spread,
                        commission_per_trade=commission,
                        pip_value_per_lot=pip_value_per_lot,
                    )

                    net_profit, sharpe, _sortino, max_dd, win_rate, profit_factor, _exp, _sqn, trades, _r2 = metrics
                    if max_dd >= dd_limit:
                        return -1e9
                    if trades <= 0:
                        continue
                    monthly_ret = net_profit / 100000.0  # baseline equity in _fast_backtest
                    prop_score = (
                        monthly_ret * 100.0
                        + 0.5 * sharpe
                        + 0.2 * profit_factor
                        + 0.1 * (win_rate * 100.0)
                        - 50.0 * max(0.0, max_dd - 0.04)
                    )
                    total_trades += float(trades)
                    weighted += float(trades) * float(prop_score)

                required = float(self._prop_required_trades(meta_val))
                if total_trades < required:
                    return -1e9
                agg = weighted / max(total_trades, 1.0)
                return float(self.prop_weight * agg + self.acc_weight * acc)

            pip_size, pip_value_per_lot = infer_pip_metrics(getattr(self.settings.system, "symbol", ""))
            metrics = fast_evaluate_strategy(
                close_prices=close_all,
                high_prices=high_all,
                low_prices=low_all,
                signals=signals,
                month_indices=month_indices_all,
                sl_pips=sl_pips,
                tp_pips=tp_pips,
                pip_value=pip_size,
                spread_pips=spread,
                commission_per_trade=commission,
                pip_value_per_lot=pip_value_per_lot,
            )

            net_profit, sharpe, sortino, max_dd, win_rate, profit_factor, expectancy, sqn, trades, _r2 = metrics
            required = float(self._prop_required_trades(meta_val))
            if trades < required or max_dd >= dd_limit:
                return -1e9

            monthly_ret = net_profit / 100000.0  # baseline equity in _fast_backtest
            prop_score = (
                monthly_ret * 100.0
                + 0.5 * sharpe
                + 0.2 * profit_factor
                + 0.1 * (win_rate * 100.0)
                - 50.0 * max(0.0, max_dd - 0.04)
            )
            return float(self.prop_weight * prop_score + self.acc_weight * acc)

        return float(acc)

    @staticmethod
    def _pad_probs(probs: np.ndarray) -> np.ndarray:
        """
        Normalize probability outputs to shape (n,3) ordered as [neutral, buy, sell].

        This matches the convention used by the live SignalEngine and all ExpertModel implementations.
        """
        if probs is None or len(probs) == 0:
            return np.zeros((0, 3), dtype=float)
        arr = np.asarray(probs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n = arr.shape[0]

        if arr.shape[1] >= 3:
            return arr[:, :3]

        out = np.zeros((n, 3), dtype=float)
        if arr.shape[1] == 2:
            # Default binary mapping: [neutral, buy]
            out[:, 0] = arr[:, 0]
            out[:, 1] = arr[:, 1]
            return out

        # Single-column fallback: treat as p_buy (neutral vs buy)
        out[:, 0] = 1.0 - arr[:, 0]
        out[:, 1] = arr[:, 0]
        return out

    def _decode_predictions(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Map predicted probabilities to labels using the project-wide proba convention.

        Columns are interpreted as [neutral, buy, sell] and mapped to labels {0, 1, -1}.
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        y_true_arr = np.where(y_true_arr == 2, -1, y_true_arr).astype(int, copy=False)

        p = self._pad_probs(y_pred_proba)
        if p.ndim != 2 or p.shape[0] == 0:
            return np.zeros_like(y_true_arr), np.zeros_like(y_true_arr, dtype=float)

        idx = np.argmax(p, axis=1)
        conf = np.max(p, axis=1)

        mapped = np.zeros(len(idx), dtype=int)
        mapped[idx == 1] = 1
        mapped[idx == 2] = -1

        # Avoid emitting unseen labels in binary edge cases (e.g. only {0,1} present).
        present = set(np.unique(y_true_arr).astype(int).tolist())
        if 1 not in present:
            mapped[mapped == 1] = 0
        if -1 not in present:
            mapped[mapped == -1] = 0
        return mapped, conf

    def _optimize_lgbm(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            device = self._pick_device(trial.number)
            device_id = None
            if isinstance(device, str) and device.startswith("cuda:"):
                try:
                    device_id = int(device.split(":")[1])
                except Exception:
                    device_id = 0
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "objective": "binary" if y_train.nunique() == 2 else "multiclass",  # Adapt for Meta-Labeling
                "num_class": y_train.nunique(),  # Adapt for Meta-Labeling
                "n_jobs": int(self._optuna_cpu_threads) if self._optuna_cpu_threads > 0 else -1,
                "verbosity": -1,
            }
            tree_pref = normalize_device_preference(os.environ.get("FOREX_BOT_TREE_DEVICE", "cpu"))
            if tree_pref == "gpu" and device_id is not None:
                params["device_type"] = "gpu"
                params["gpu_device_id"] = device_id
                params.setdefault("max_bin", 63)
            else:
                params["device_type"] = "cpu"
            model = LightGBMExpert(params=params)
            x_tr, x_va, y_tr, y_va = self._time_series_holdout(x_train, y_train, n_splits=3, embargo=100)
            model.fit(x_tr, y_tr)
            if len(x_va) == 0:
                return 0.0
            preds = model.predict_proba(x_va)
            return self._objective_base(y_va, preds, None)

        study = self._run_study("LightGBM_Opt", objective, self.n_trials)
        logger.info(f"Optuna LightGBM completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})")
        return study.best_params

    def _optimize_random_forest(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 15),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
                "n_jobs": int(self._optuna_cpu_threads) if self._optuna_cpu_threads > 0 else -1,
                "random_state": 42,
            }
            model = RandomForestExpert(params=params)
            x_tr, x_va, y_tr, y_va = self._time_series_holdout(x_train, y_train, n_splits=3, embargo=100)
            model.fit(x_tr, y_tr)
            if len(x_va) == 0:
                return 0.0
            preds = model.predict_proba(x_va)
            return self._objective_base(y_va, preds, None)

        study = self._run_study("RandomForest_Opt", objective, self.n_trials)
        logger.info(f"Optuna RandomForest completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})")
        return study.best_params

    def _optimize_extra_trees(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 10, 24),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
                "max_features": trial.suggest_float("max_features", 0.4, 0.9),
                "bootstrap": trial.suggest_categorical("bootstrap", [False, True]),
                "n_jobs": int(self._optuna_cpu_threads) if self._optuna_cpu_threads > 0 else -1,
                "random_state": 42,
            }
            model = ExtraTreesExpert(params=params)
            x_tr, x_va, y_tr, y_va = self._time_series_holdout(x_train, y_train, n_splits=3, embargo=100)
            model.fit(x_tr, y_tr)
            if len(x_va) == 0:
                return 0.0
            preds = model.predict_proba(x_va)
            return self._objective_base(y_va, preds, None)

        study = self._run_study("ExtraTrees_Opt", objective, self.n_trials)
        logger.info(f"Optuna ExtraTrees completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})")
        return study.best_params

    def _optimize_tabnet(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            try:
                params = {
                    "hidden_dim": trial.suggest_int("hidden_dim", 32, 128),
                    "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                    "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                }
                device = self._pick_device(trial.number)
                model = TabNetExpert(**params, max_time_sec=self.settings.models.tabnet_train_seconds, device=device)
                model.fit(x_train, y_train)
                preds = model.predict_proba(x_val)
                return self._objective_base(y_val, preds, meta_val)
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.warning(f"TabNet Optuna trial {trial.number} failed: {exc}", exc_info=True)
                raise optuna.TrialPruned(str(exc)) from exc

        target_trials = max(5, self.n_trials // 2)
        if self.default_device == "cpu" and not self.device_pool:
            target_trials = min(target_trials, 3)
        study = self._run_study("TabNet_Opt", objective, target_trials)
        logger.info(f"Optuna TabNet completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})")
        try:
            return study.best_params
        except Exception as exc:
            logger.warning(f"Optuna TabNet produced no completed trials; using defaults: {exc}")
            return {}

    def _optimize_nbeats(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            try:
                params = {
                    "hidden_dim": trial.suggest_int("hidden_dim", 64, 256),
                    "n_layers": trial.suggest_int("n_layers", 2, 6),
                    "n_blocks": trial.suggest_int("n_blocks", 1, 4),
                    "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                }
                device = self._pick_device(trial.number)
                model = NBeatsExpert(**params, max_time_sec=self.settings.models.nbeats_train_seconds, device=device)
                model.fit(x_train, y_train)
                preds = model.predict_proba(x_val)
                return self._objective_base(y_val, preds, meta_val)
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.warning(f"N-BEATS Optuna trial {trial.number} failed: {exc}", exc_info=True)
                raise optuna.TrialPruned(str(exc)) from exc

        target_trials = max(5, self.n_trials // 2)
        if self.default_device == "cpu" and not self.device_pool:
            target_trials = min(target_trials, 3)
        study = self._run_study("NBEATS_Opt", objective, target_trials)
        logger.info(f"Optuna N-BEATS completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})")
        try:
            return study.best_params
        except Exception as exc:
            logger.warning(f"Optuna N-BEATS produced no completed trials; using defaults: {exc}")
            return {}

    def _optimize_tide(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            try:
                params = {
                    "hidden_dim": trial.suggest_int("hidden_dim", 64, 512),
                    "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                    "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                }
                device = self._pick_device(trial.number)
                model = TiDEExpert(**params, max_time_sec=self.settings.models.tide_train_seconds, device=device)
                model.fit(x_train, y_train)
                preds = model.predict_proba(x_val)
                score = self._objective_base(y_val, preds, meta_val)

                del model
                import gc

                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return score
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.warning(f"TiDE Optuna trial {trial.number} failed: {exc}", exc_info=True)
                raise optuna.TrialPruned(str(exc)) from exc

        target_trials = max(5, self.n_trials // 2)
        if self.default_device == "cpu" and not self.device_pool:
            target_trials = min(target_trials, 3)
        study = self._run_study("TiDE_Opt", objective, target_trials)
        logger.info(f"Optuna TiDE completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})")
        try:
            return study.best_params
        except Exception as exc:
            logger.warning(f"Optuna TiDE produced no completed trials; using defaults: {exc}")
            return {}

    def _optimize_kan(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            try:
                params = {
                    "hidden_dim": trial.suggest_int("hidden_dim", 32, 128),
                    "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                    "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                }
                device = self._pick_device(trial.number)
                model = KANExpert(**params, max_time_sec=self.settings.models.kan_train_seconds, device=device)
                model.fit(x_train, y_train)
                preds = model.predict_proba(x_val)
                return self._objective_base(y_val, preds, meta_val)
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.warning(f"KAN Optuna trial {trial.number} failed: {exc}", exc_info=True)
                raise optuna.TrialPruned(str(exc)) from exc

        target_trials = max(5, self.n_trials // 2)
        if self.default_device == "cpu" and not self.device_pool:
            target_trials = min(target_trials, 3)
        study = self._run_study("KAN_Opt", objective, target_trials)
        logger.info(f"Optuna KAN completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})")
        try:
            return study.best_params
        except Exception as exc:
            logger.warning(f"Optuna KAN produced no completed trials; using defaults: {exc}")
            return {}

    def _optimize_transformer(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        max_rows = int(getattr(self.settings.models, "optuna_max_rows", 200_000))
        if max_rows > 0 and len(x_train) > max_rows:
            idx_train = np.linspace(0, len(x_train) - 1, max_rows, dtype=int)
            x_train = x_train.iloc[idx_train]
            y_train = y_train.iloc[idx_train]
        if max_rows > 0 and len(x_val) > max_rows:
            idx_val = np.linspace(0, len(x_val) - 1, max_rows, dtype=int)
            x_val = x_val.iloc[idx_val]
            y_val = y_val.iloc[idx_val]
            if meta_val is not None:
                try:
                    meta_val = meta_val.iloc[idx_val]
                except Exception:
                    meta_val = None

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            try:
                params = {
                    "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
                    "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
                    "n_layers": trial.suggest_int("n_layers", 2, 4),
                    "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                }
                device = self._pick_device(trial.number)
                model = TransformerExpertTorch(
                    **params,
                    max_time_sec=min(self.settings.models.transformer_train_seconds, 1800),  # hard cap per trial
                    device=device,
                )
                model.fit(x_train, y_train)
                preds = model.predict_proba(x_val)
                score = self._objective_base(y_val, preds, meta_val)

                # Cleanup to prevent OOM
                del model
                import gc

                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

                return score
            except optuna.TrialPruned:
                raise
            except Exception as exc:
                logger.warning(f"Transformer Optuna trial {trial.number} failed: {exc}", exc_info=True)
                raise optuna.TrialPruned(str(exc)) from exc

        target_trials = max(3, int(getattr(self.settings.models, "optuna_trials", 25)) // 3)
        if self.default_device == "cpu" and not self.device_pool:
            target_trials = min(target_trials, 2)
        study = self._run_study("Transformer_Opt", objective, target_trials)
        logger.info(f"Optuna Transformer completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})")
        try:
            return study.best_params
        except Exception as exc:
            logger.warning(f"Optuna Transformer produced no completed trials; using defaults: {exc}")
            return {}

    def _optimize_rl_ppo(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
            net_arch = [64, 64] if net_arch_type == "small" else [128, 128]
            device = self._pick_device(trial.number)

            model = RLExpertPPO(timesteps=10_000, max_time_sec=300, network_arch=net_arch, device=device)
            model.fit(x_train, y_train)
            preds = model.predict_proba(x_val)
            return self._objective_base(y_val, preds, meta_val)

        target_trials = max(3, self.n_trials // 3)
        study = self._run_study("RL_PPO_Opt", objective, target_trials)
        logger.info(f"Optuna RL PPO completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})")

        params = study.best_params
        if params.get("net_arch") == "small":
            params["rl_network_arch"] = [64, 64]
        elif params.get("net_arch") == "medium":
            params["rl_network_arch"] = [128, 128]
        if "net_arch" in params:
            del params["net_arch"]
        return params

    def _optimize_rl_sac(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "buffer_size": trial.suggest_categorical("buffer_size", [10000, 50000, 100000]),
                "gamma": trial.suggest_float("gamma", 0.9, 0.999),
                "tau": trial.suggest_float("tau", 0.005, 0.02),
                "train_freq": trial.suggest_categorical("train_freq", [(1, "step"), (10, "step"), (1, "episode")]),
                "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 5, 10]),
            }
            device = self._pick_device(trial.number)
            model = RLExpertSAC(
                timesteps=int(self.settings.models.rl_timesteps / 10),  # short budget for HPO
                network_arch=list(getattr(self.settings.models, "rl_network_arch", [256, 256])),
                device=device,
                parallel_envs=int(max(1, getattr(self.settings.models, "rl_parallel_envs", 1))),
                **params,
            )

            def env_fn() -> Any:
                return _build_continuous_env(x_train, y_train)

            model.fit(x_train, y_train, env_fn=env_fn)
            preds = model.predict_proba(x_val)
            return self._objective_base(y_val, preds, meta_val)

        target_trials = max(3, self.n_trials // 3)
        study = self._run_study("RL_SAC_Opt", objective, target_trials)
        logger.info(f"Optuna RL SAC completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})")
        return study.best_params

    def _optimize_neuroevolution(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        meta_val: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        def objective(trial: Trial) -> float:
            if self._should_stop():
                raise optuna.TrialPruned("stop requested")
            params = {
                "hidden_size": trial.suggest_int("hidden_size", 32, 128),
                "population": trial.suggest_int("population", 16, 64),
                "num_islands": trial.suggest_int("num_islands", 2, 8),
            }
            device = self._pick_device(trial.number)
            hpo_seconds = int(getattr(self.settings.models, "evo_optuna_train_seconds", 0) or 0)
            if hpo_seconds <= 0:
                hpo_seconds = min(int(getattr(self.settings.models, "evo_train_seconds", 3600)), 900)
            max_rows = int(getattr(self.settings.models, "optuna_max_rows", 0) or 0)
            model = EvoExpertCMA(
                **params,
                max_time_sec=hpo_seconds,
                device=device,
                MAX_TRAIN_ROWS=max_rows,
                evo_multiproc_per_gpu=bool(getattr(self.settings.system, "evo_multiproc_per_gpu", True)),
            )
            try:
                model.fit(x_train, y_train)
                preds = model.predict_proba(x_val)
                return self._objective_base(y_val, preds, meta_val)
            except MemoryError as exc:
                raise optuna.TrialPruned(f"Evo OOM: {exc}") from exc
            except Exception as exc:
                logger.warning(f"Evo Optuna trial failed: {exc}", exc_info=True)
                raise optuna.TrialPruned(f"Evo failed: {exc}") from exc

        target_trials = max(5, self.n_trials // 2)
        study = self._run_study("Evo_Opt", objective, target_trials)
        logger.info(
            f"Optuna Neuroevolution completed in {time.perf_counter() - start:.1f}s (trials={len(study.trials)})"
        )
        return study.best_params

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

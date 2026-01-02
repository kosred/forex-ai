import hashlib
import json
import logging
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import concurrent.futures
import multiprocessing

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# Suppress pandas nanops RuntimeWarnings (invalid value in subtract/add/etc)
# We clean data before calculations, but pandas internals may still emit warnings
warnings.filterwarnings("ignore", message="invalid value encountered in subtract", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in add", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in multiply", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

from ..core.config import Settings
from ..core.system import normalize_device_preference
from ..domain.events import PreparedDataset
from ..models.device import get_available_gpus

# Modular Imports
from .cpu.indicators import (
    compute_adx_numba,
    compute_cci_numba,
    compute_stochastic_numba,
    compute_vwap_numba,
    rsi_pandas,
    rsi_vectorized_numba,
    vwap_pandas,
)
from .cpu.labeling import compute_meta_labels_atr_numba
from .gpu.indicators import CUPY_AVAILABLE, atr_wicks_cupy, rsi_cupy, talib_gpu
from .prep_utils import have_cudf
from .smc import NUMBA_AVAILABLE, njit
from .specialized import add_smc_features, compute_obi_features, compute_volume_profile_features
from .talib_mixer import TALIB_AVAILABLE, talib, ALL_INDICATORS, abstract

if CUPY_AVAILABLE:
    import cupy as cp
else:
    cp = None

VOLUME_TALIB_INDICATORS = {"AD", "ADOSC", "OBV", "MFI"}

logger = logging.getLogger(__name__)
pd.options.mode.copy_on_write = True
EPSILON = 1e-10
FEATURE_CACHE_VERSION = 5
# Backwards-compatible alias used in tests
_compute_adx_numba = compute_adx_numba


def to_numpy_safe(data: Any) -> np.ndarray:
    """
    Safely convert CuPy/CuDF/Pandas data to a NumPy array.
    Prevents 'Implicit conversion to a NumPy array is not allowed' errors.
    """
    if data is None:
        return np.array([])
    
    # 1. Handle CuPy
    if cp is not None and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    
    # 2. Handle cuDF Series/DataFrame
    if hasattr(data, "to_numpy"):
        try:
            return data.to_numpy()
        except Exception:
            pass
            
    if hasattr(data, "get"):
        try:
            return data.get()
        except Exception:
            pass
            
    # 3. Fallback to standard NumPy conversion
    return np.asarray(data)


def _feature_cpu_budget() -> int:
    """Best-effort CPU budget per rank/process for feature work."""
    local_rank = os.environ.get("LOCAL_RANK")
    per_gpu = os.environ.get("FOREX_BOT_CPU_THREADS_PER_GPU")
    if local_rank is not None:
        if per_gpu:
            try:
                return max(1, int(per_gpu))
            except Exception:
                pass
        for key in ("FOREX_BOT_CPU_BUDGET", "FOREX_BOT_CPU_THREADS"):
            val = os.environ.get(key)
            if val:
                try:
                    return max(1, int(val))
                except Exception:
                    pass
        return 20
    for key in ("FOREX_BOT_CPU_BUDGET", "FOREX_BOT_CPU_THREADS"):
        val = os.environ.get(key)
        if val:
            try:
                # If explicitly set, respect it (user knows what they're doing)
                return max(1, int(val))
            except Exception:
                pass
    # Default: use all cores minus reserve.
    cpu_count = os.cpu_count() or 1
    try:
        reserve = int(os.environ.get("FOREX_BOT_CPU_RESERVE", "1") or 1)
    except Exception:
        reserve = 1
    reserve = max(0, reserve)
    return max(1, cpu_count - reserve)


def _talib_chunk_worker(payload):
    """Compute a chunk of TA-Lib indicators for parallel feature injection."""
    inputs, indicators, use_volume = payload
    out: dict[str, np.ndarray] = {}
    for ind_name in indicators:
        try:
            if (not use_volume) and ind_name in VOLUME_TALIB_INDICATORS:
                continue
            if ind_name in ["SMA", "EMA", "RSI", "BBANDS", "ADX", "CCI", "MOM", "ROC", "WILLR", "ATR"]:
                continue
            func = abstract.Function(ind_name)
            res = func(inputs)
            if isinstance(res, (list, tuple)):
                for i, arr in enumerate(res):
                    out[f"ta_{ind_name.lower()}_{i}"] = arr.astype(np.float32)
            elif isinstance(res, (pd.Series, np.ndarray)):
                val = res.values if isinstance(res, pd.Series) else res
                out[f"ta_{ind_name.lower()}"] = val.astype(np.float32)
        except Exception:
            continue
    return out


class FeatureEngineer:
    """Feature engineering pipeline with autotuned optimization."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._indices_cache: pd.DataFrame | None = None
        self.cache_enabled = bool(getattr(settings.system, "cache_enabled", True))
        self.cache_dir = Path(getattr(settings.system, "cache_dir", "cache")) / "features"
        self.cache_ttl_minutes = int(getattr(settings.system, "cache_max_age_minutes", 60))
        self.use_volume_features = bool(getattr(settings.system, "use_volume_features", False))
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Feature caching enabled. Cache directory: {self.cache_dir}")
            self._prune_feature_cache()
        else:
            logger.info("Feature caching disabled.")
        self._symbol = None

        self._gpu_pool = get_available_gpus()
        pref = normalize_device_preference(getattr(self.settings.system, "enable_gpu_preference", "auto"))
        self._gpu_enabled = bool(self._gpu_pool) and pref in {"auto", "gpu"}

        # cuDF pandas mode is powerful but invasive (it monkey-patches pandas globally), so keep it opt-in.
        if self._gpu_enabled and have_cudf() and bool(int(os.environ.get("FOREX_BOT_CUDF_PANDAS", "0") or 0)):
            self._enable_cudf_pandas()

    def _enable_cudf_pandas(self) -> None:
        """Enable cudf.pandas mode for automatic GPU acceleration."""
        try:
            import cudf.pandas

            cudf.pandas.install()
            logger.info("✓ GPU acceleration enabled via cudf.pandas.")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to enable cudf.pandas: {e}")

    def _prune_feature_cache(self) -> None:
        """
        Best-effort cleanup for the feature cache directory.

        The cache uses TTL for reads, but old entries can accumulate on disk; prune them
        at startup so disk usage doesn't grow unbounded.
        """
        if not self.cache_enabled:
            return
        try:
            now = datetime.now(UTC)
            ttl = int(self.cache_ttl_minutes)
            removed = 0

            for p in self.cache_dir.glob("*.pkl"):
                try:
                    # Drop caches from older feature schema versions.
                    if not p.name.endswith(f"_{FEATURE_CACHE_VERSION}.pkl"):
                        p.unlink(missing_ok=True)
                        removed += 1
                        continue

                    # Drop stale caches beyond TTL (if TTL enabled).
                    if ttl > 0:
                        age_minutes = (now - datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)).total_seconds() / 60.0
                        if age_minutes > ttl:
                            p.unlink(missing_ok=True)
                            removed += 1
                except Exception:
                    continue

            if removed:
                logger.info(f"Pruned {removed} expired feature cache file(s).")
        except Exception:
            return

    def _clean_garbage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.replace([np.inf, -np.inf], np.nan)
        threshold = len(df) * 0.5
        df = df.dropna(axis=1, thresh=threshold)
        df = df.fillna(0.0)
        try:
            preserve = [c for c in df.columns if str(c).startswith("symbol_")]
            std = df.std()
            cols_to_drop = std[std < 1e-9].index
            cols_to_drop = [c for c in cols_to_drop if c not in preserve]
            if len(cols_to_drop) > 0:
                df = df.drop(columns=cols_to_drop)
        except Exception:
            pass
        return df

    def prepare(
        self,
        frames: dict[str, DataFrame],
        news_features: DataFrame | None = None,
        order_book_features: DataFrame | None = None,
        symbol: str | None = None,
    ) -> PreparedDataset:
        self._symbol = symbol or getattr(self.settings.system, "symbol", None)
        base_tf = self.settings.system.base_timeframe
        use_volume = self.use_volume_features

        # Safe lookup for base dataframe (prevent "ambiguous truth value" error)
        base = frames.get(base_tf)
        if base is None or (isinstance(base, pd.DataFrame) and base.empty):
            base = frames.get("M5")
        if base is None or (isinstance(base, pd.DataFrame) and base.empty):
            base = frames.get("M1")

        if base is None or (isinstance(base, pd.DataFrame) and base.empty):
            raise ValueError(f"No data available for timeframe {base_tf}")

        features_pref = normalize_device_preference(os.environ.get("FOREX_BOT_FEATURES_DEVICE", "auto"))
        use_gpu = False
        if features_pref == "gpu":
            use_gpu = self._gpu_enabled and CUPY_AVAILABLE
        elif features_pref == "auto":
            try:
                min_rows = int(os.environ.get("FOREX_BOT_FEATURES_GPU_MIN_ROWS", "200000") or 200000)
            except Exception:
                min_rows = 200000
            use_gpu = self._gpu_enabled and CUPY_AVAILABLE and len(base) >= min_rows
        smc_cuda_enabled = bool(getattr(self.settings.system, "smc_use_cuda", True))

        df = base.copy()

        # Optional row cap to prevent runaway memory on very long histories
        def _parse_int_env(name: str) -> int | None:
            raw = os.environ.get(name)
            if raw is None:
                return None
            try:
                return int(str(raw).strip())
            except Exception:
                return None

        full_data = str(os.environ.get("FOREX_BOT_FULL_DATA", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        max_rows = 0 if full_data else int(getattr(self.settings.system, "max_training_rows_per_tf", 0) or 0)
        env_max_rows = _parse_int_env("FOREX_BOT_MAX_TRAINING_ROWS")
        if env_max_rows is not None:
            max_rows = 0 if env_max_rows <= 0 else env_max_rows
        if max_rows > 0 and len(df) > max_rows:
            logger.info(f"Capping {symbol or base_tf} rows from {len(df):,} -> {max_rows:,}")
            df = df.tail(max_rows)

        # Clean inf/NaN from raw OHLC data BEFORE any calculations (prevents RuntimeWarnings)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                # Forward-fill NaN for OHLC (missing ticks), but don't create new data
                if col != "volume":
                    df[col] = df[col].ffill().bfill()
                else:
                    # Volume NaN = 0 (no trading)
                    df[col] = df[col].fillna(0.0)

        # Normalize timestamps
        try:
            if hasattr(df, "to_pandas"):
                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp").set_index("timestamp")
                df = df.to_pandas()

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                df = df.sort_values("timestamp").set_index("timestamp")
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df.sort_index()
        except Exception as exc:
            logger.warning(f"Timestamp normalization failed: {exc}")

        cache_meta = None
        if self.cache_enabled:
            try:
                cache_meta = self._compute_cache_meta(df, frames, symbol, news_features, order_book_features)
                cached = self._load_cached_dataset(cache_meta)
                if cached:
                    logger.info(f"Loaded features from cache: {cache_meta.get('cache_key')}")
                    return PreparedDataset(
                        X=cached["X"],
                        y=cached["y"],
                        index=cached["index"],
                        feature_names=list(cached.get("feature_names", [])),
                        metadata=cached.get("metadata"),
                        labels=cached.get("labels"),
                    )
            except Exception:
                pass

        # === High-Performance HPC Dispatch (252-Core Saturation) ===
        cpu_budget = _feature_cpu_budget()
        logger.info(f"Engineering features using {cpu_budget} cores...")

        # 1. Standard Indicators (Parallelized)
        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_budget) as executor:
            f1 = executor.submit(self._compute_basic_features, df, use_gpu=False)
            f2 = executor.submit(self._compute_adx_safe, df)
            f3 = executor.submit(self._compute_stochastic_safe, df)
            f4 = executor.submit(self._compute_cci_safe, df)
            
            df = f1.result()
            df["adx"] = f2.result()
            df["stoch_k"], df["stoch_d"] = f3.result()
            df["cci"] = f4.result()

        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()

        # 2. Comprehensive TA-Lib Injection (Deep Learning Fuel)
        if TALIB_AVAILABLE:
            df = self._compute_comprehensive_talib_features(df)
        else:
            df["mfi14"] = 50.0

        df = self._merge_indices(df)


        # 3. Multi-Timeframe Features (Optimized "Collect then Concat")
        # Use config-driven timeframes for multi-TF features; fall back to frames if empty.
        cfg_tfs = list(getattr(self.settings.system, "higher_timeframes", []) or [])
        if not cfg_tfs:
            cfg_tfs = list(getattr(self.settings.system, "required_timeframes", []) or [])
        target_tfs = [base_tf] + cfg_tfs
        target_tfs = [tf for tf in dict.fromkeys(target_tfs) if tf in frames]
        # Include any additional frames we have on disk/runtime (e.g. custom TFs)
        for tf in frames.keys():
            if tf not in target_tfs:
                target_tfs.append(tf)
        if not target_tfs:
            target_tfs = [base_tf] + [tf for tf in frames.keys() if tf != base_tf]
        htf_cols = []
        htf_feature_blocks = [] # Collect blocks here
        
        import gc 

        try:
            htf_parallel = str(os.environ.get("FOREX_BOT_HTF_PARALLEL", "1") or "1").strip().lower() in {
                "1", "true", "yes", "on"
            }
        except Exception:
            htf_parallel = True
        try:
            htf_workers_env = os.environ.get("FOREX_BOT_HTF_WORKERS")
            htf_workers = max(1, int(htf_workers_env)) if htf_workers_env else _feature_cpu_budget()
        except Exception:
            htf_workers = _feature_cpu_budget()
        htf_workers = max(1, min(htf_workers, max(1, len(target_tfs) - 1)))
        htf_use_gpu = use_gpu and str(os.environ.get("FOREX_BOT_HTF_USE_GPU", "0") or "0").strip().lower() in {
            "1", "true", "yes", "on"
        }

        def _build_htf_block(tf_name: str):
            if tf_name == base_tf:
                return None
            htf = frames.get(tf_name)
            if htf is None or htf.empty:
                return None
            try:
                local = htf.copy()
                if "timestamp" in local.columns:
                    local["timestamp"] = pd.to_datetime(local["timestamp"], utc=True, errors="coerce")
                    local = local.set_index("timestamp")
                elif not isinstance(local.index, pd.DatetimeIndex):
                    local.index = pd.to_datetime(local.index, utc=True, errors="coerce")
                local = local.sort_index()

                local = self._compute_basic_features(local, use_gpu=htf_use_gpu)
                if not isinstance(local, pd.DataFrame):
                    local = local.to_pandas()

                local = self._compute_volatility_features(local)
                if use_volume:
                    local = self._compute_volume_profile_features(local)
                    local = self._compute_obi_features(local, htf_use_gpu)

                key_feats = [
                    "rsi",
                    "macd",
                    "macd_hist",
                    "atr14",
                    "bb_width",
                ]
                if use_volume:
                    key_feats += [
                        "dist_to_poc",
                        "in_value_area",
                        "vol_imbalance",
                    ]
                subset = local[[c for c in key_feats if c in local.columns]].copy()
                # Prevent lookahead leakage: align to the last *completed* HTF candle.
                subset = subset.shift(1)
                subset.columns = [f"{tf_name}_{c}" for c in subset.columns]

                merged = subset.reindex(df.index).ffill().fillna(0.0)
                merged = merged.astype(np.float32)
                return merged
            except Exception as e:
                logger.warning(f"Failed to generate HTF features for {tf_name}: {e}")
                return None
            finally:
                del htf
                gc.collect()

        if htf_parallel and len(target_tfs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=htf_workers) as executor:
                futures = {
                    executor.submit(_build_htf_block, tf_name): tf_name
                    for tf_name in target_tfs
                    if tf_name != base_tf
                }
                for future in concurrent.futures.as_completed(futures):
                    merged = future.result()
                    if merged is None:
                        continue
                    htf_feature_blocks.append(merged)
                    htf_cols.extend(merged.columns.tolist())
        else:
            for tf_name in target_tfs:
                if tf_name == base_tf:
                    continue
                merged = _build_htf_block(tf_name)
                if merged is None:
                    continue
                htf_feature_blocks.append(merged)
                htf_cols.extend(merged.columns.tolist())

        # Merge all HTF blocks at once
        if htf_feature_blocks:
            df = pd.concat([df] + htf_feature_blocks, axis=1, copy=False)
            del htf_feature_blocks
            gc.collect()
            # HPC SHIELD: Limit memory fragmentation on high-RAM machines
            import ctypes
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0) 
            except Exception:
                pass

        # MTF confluence: average directional agreement across higher TFs.
        tf_votes = []
        for tf_name in target_tfs:
            if tf_name == base_tf:
                continue
            col = f"{tf_name}_macd_hist"
            if col in df.columns:
                try:
                    tf_votes.append(np.sign(df[col].to_numpy(dtype=np.float32, copy=False)))
                except Exception:
                    tf_votes.append(np.sign(df[col].to_numpy(dtype=np.float32)))
        if tf_votes:
            votes = np.vstack(tf_votes)
            df["mtf_confluence"] = votes.mean(axis=0).astype(np.float32)
        else:
            df["mtf_confluence"] = 0.0

        # 4. News Features
        if news_features is not None:
            try:
                nf = news_features
                if not isinstance(nf, pd.DataFrame):
                    nf = pd.DataFrame(nf)
                if len(nf) > 0:
                    if not isinstance(nf.index, pd.DatetimeIndex):
                        nf.index = pd.to_datetime(nf.index, utc=True, errors="coerce")
                    nf = nf.sort_index().reindex(df.index).ffill().fillna(0.0)
                    for col in ["news_sentiment", "news_confidence", "news_count", "news_recency_minutes"]:
                        if col in nf.columns:
                            df[col] = nf[col].astype(np.float32)
            except Exception:
                pass

        for col in ["news_sentiment", "news_confidence", "news_count", "news_recency_minutes"]:
            if col not in df.columns:
                df[col] = 0.0

        # 5. Volatility & Other Features
        df = self._compute_volatility_features(df, as_pandas=True)

        df["vwap"] = self._compute_vwap(df, use_gpu=use_gpu)
        df["dist_to_vwap"] = np.where(df["vwap"] > 0, (df["close"] - df["vwap"]) / df["vwap"], 0.0)

        # ATR / Wicks
        if use_gpu and CUPY_AVAILABLE:
            try:
                atr_res = atr_wicks_cupy(
                    df["high"].to_numpy(), df["low"].to_numpy(), df["open"].to_numpy(), df["close"].to_numpy()
                )
                df["atr"] = atr_res["atr"]
                df["upper_wick"] = atr_res["upper_wick"]
                df["lower_wick"] = atr_res["lower_wick"]
            except Exception:
                self._compute_atr_wicks_cpu(df)
        else:
            self._compute_atr_wicks_cpu(df)

        # SMC features (CPU by default; CUDA only when explicitly allowed)
        df = self._add_smc_features(df, use_gpu=(use_gpu and smc_cuda_enabled))

        if use_volume:
            df = self._compute_volume_profile_features(df)
        if use_volume:
            df = self._compute_obi_features(df, use_gpu)

        hours = df.index.hour
        df["session_asia"] = ((hours >= 0) & (hours < 8)).astype(int)
        df["session_london"] = ((hours >= 7) & (hours < 16)).astype(int)
        df["session_ny"] = ((hours >= 12) & (hours < 21)).astype(int)

        # HPC FIX: Deterministic Symbol Mapping
        # Prevents ID drift between shards/processes
        if symbol:
            known_symbols = ["AUDUSD", "EURGBP", "EURJPY", "EURUSD", "GBPJPY", "GBPUSD", "XAUUSD"]
            try:
                sym_id = known_symbols.index(symbol.upper())
            except ValueError:
                sym_id = 99 # Fallback
            df["symbol_id"] = np.int16(sym_id)

        # Meta-label outcomes... (remaining logic)

        symbol_cols: list[str] = []
        symbol_df: pd.DataFrame | None = None
        if symbol:
            try:
                bucket_dim = int(getattr(self.settings.models, "symbol_hash_buckets", 32) or 32)
            except Exception:
                bucket_dim = 32
            bucket_dim = int(max(2, min(bucket_dim, 256)))
            stable_id = int(hashlib.sha256(str(symbol).upper().encode("utf-8")).hexdigest()[:12], 16)
            bucket = int(stable_id % bucket_dim)
            symbol_cols = [f"symbol_hash_{i}" for i in range(bucket_dim)]
            # Construct in one shot to avoid DataFrame fragmentation from repeated inserts.
            sym = np.zeros((len(df), bucket_dim), dtype=np.float32)
            if len(df) > 0:
                sym[:, bucket] = 1.0
            symbol_df = pd.DataFrame(sym, index=df.index, columns=symbol_cols)

        base_cols = [
            "returns",
            "ema_fast",
            "ema_slow",
            "rsi",
            "realized_vol",
            "ewma_vol",
            "boll_width",
            "zscore_close",
            "upper_wick",
            "lower_wick",
            "session_asia",
            "session_london",
            "session_ny",
            "fvg_direction",
            "fvg_size",
            "market_structure_trend",
            "bos_event",
            "choch_event",
            "liquidity_sweep",
            "order_block",
            "smc_confluence",
            "dist_liquidity",
            "dist_fvg",
            "premium_discount",
            "inducement",
            "adx",
            "stoch_k",
            "stoch_d",
            "cci",
            "dist_to_vwap",
            "dist_to_poc",
            "in_value_area",
            "vol_imbalance",
            "adl_slope",
            "news_sentiment",
            "news_confidence",
            "news_count",
            "news_recency_minutes",
            "idx_dxy_ret",
            "idx_eur_ret",
            "mtf_confluence",
            "base_signal",
        ]
        if not use_volume:
            base_cols = [
                c
                for c in base_cols
                if c not in {"dist_to_poc", "in_value_area", "vol_imbalance", "adl_slope"}
            ]

        # Dynamically collect all generated features (Base + TA-Lib + SMC + HTF)
        # This ensures our 'Unsupervised' approach actually passes all data to the models.
        all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # We prioritize our core columns but include everything starting with ta_ or smc_
        dynamic_features = [c for c in all_numeric_cols if c.startswith("ta_") or c.startswith("smc_") or c.startswith("idx_")]
        
        cols = base_cols + htf_cols + dynamic_features
        # Deduplicate while preserving order
        cols = list(dict.fromkeys(cols))
        # Ensure symbol_hash columns are added last
        cols = [c for c in cols if not c.startswith("symbol_hash_")] + symbol_cols

        # Base signals (ensure column exists before feature selection)
        base_sigs = self._generate_base_signals(df, frames)
        try:
            df["base_signal"] = base_sigs.astype(np.int8, copy=False)
        except Exception:
            df["base_signal"] = base_sigs

        # Final Feature Matrix Construction
        valid_cols = [c for c in cols if c in df.columns]
        X = df[valid_cols].fillna(0.0)
        
        # Labels & Metadata (HPC: Vectorized Generation)
        labels = self._meta_label_outcomes(df, base_sigs)
        
        meta = df[["close", "high", "low", "open"]].copy()
        if "volume" in df.columns:
            meta["volume"] = df["volume"]
        if "atr" in df.columns:
            meta["atr"] = df["atr"]
        try:
            if symbol:
                meta.attrs["symbol"] = symbol
        except Exception:
            pass

        # Merged dataset finalized
        final_features = list(X.columns)
        logger.info(
            f"[TELEMETRY] Pipeline: Engineered {len(final_features)} features "
            f"across {len(target_tfs)} timeframes. Data Shape: {X.shape}"
        )

        if self.cache_enabled and cache_meta:
            self._save_cached_dataset(cache_meta, X, labels, df.index, final_features, meta)

        return PreparedDataset(
            X=X.astype(np.float32),
            y=labels.astype(int),
            index=df.index,
            feature_names=final_features,
            metadata=meta,
            labels=labels,
        )

    def _compute_comprehensive_talib_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamically injects the massive TA-Lib indicator set with HYPERSPACE EXPANSION.
        Optimized to collect features and concat once to avoid memory fragmentation.
        """
        if not TALIB_AVAILABLE:
            return df
            
        try:
            inputs = {
                'open': df["open"].values.astype(np.float64),
                'high': df["high"].values.astype(np.float64),
                'low': df["low"].values.astype(np.float64),
                'close': df["close"].values.astype(np.float64),
            }
            if self.use_volume_features and "volume" in df.columns:
                inputs['volume'] = df["volume"].values.astype(np.float64)

            new_features = {}

            # 1. Standard "Kitchen Sink" (Defaults) - parallelized
            indicator_names = []
            for ind_name in ALL_INDICATORS:
                if (not self.use_volume_features) and ind_name in VOLUME_TALIB_INDICATORS:
                    continue
                if ind_name in ["SMA", "EMA", "RSI", "BBANDS", "ADX", "CCI", "MOM", "ROC", "WILLR", "ATR"]:
                    continue
                indicator_names.append(ind_name)

            if indicator_names:
                workers_env = os.environ.get("FOREX_BOT_TALIB_WORKERS")
                if workers_env:
                    try:
                        workers = max(1, int(workers_env))
                    except Exception:
                        workers = _feature_cpu_budget()
                else:
                    workers = _feature_cpu_budget()
                workers = max(1, min(workers, len(indicator_names)))
                if workers == 1:
                    try:
                        new_features.update(
                            _talib_chunk_worker((inputs, indicator_names, self.use_volume_features))
                        )
                    except Exception:
                        pass
                else:
                    # HPC OPTIMIZATION: Use more chunks to fill all 252 cores
                    num_chunks = max(workers, 128) 
                    chunk_size = max(1, (len(indicator_names) + num_chunks - 1) // num_chunks)
                    chunks = [
                        indicator_names[i:i + chunk_size]
                        for i in range(0, len(indicator_names), chunk_size)
                    ]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = [
                            executor.submit(_talib_chunk_worker, (inputs, chunk, self.use_volume_features))
                            for chunk in chunks
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                new_features.update(future.result())
                            except Exception:
                                continue

            # 2. Hyperspace Variations
            simple_vars = {
                "RSI": [9, 14, 21, 34],
                "ADX": [14, 30],
                "ATR": [14, 30],
                "CCI": [14, 34, 50],
                "MOM": [10, 20],
                "ROC": [10, 20],
                "WILLR": [14, 34],
                "SMA": [20, 50, 100, 200],
                "EMA": [9, 21, 50, 200],
                "WMA": [20, 50],
                "T3": [5, 10],
                "CMO": [9, 14]
            }
            
            for name, periods in simple_vars.items():
                try:
                    func = abstract.Function(name)
                    for p in periods:
                        res = func(inputs, timeperiod=p)
                        col = f"ta_{name.lower()}_{p}"
                        if isinstance(res, (pd.Series, np.ndarray)):
                            val = res.values if isinstance(res, pd.Series) else res
                            new_features[col] = val.astype(np.float32)
                except Exception:
                    pass

            # Complex Variations (BBANDS)
            bb_vars = [(20, 2.0, 2.0), (20, 2.5, 2.5), (50, 2.0, 2.0)]
            try:
                func = abstract.Function("BBANDS")
                for p, dev_up, dev_dn in bb_vars:
                    u, m, l = func(inputs, timeperiod=p, nbdevup=dev_up, nbdevdn=dev_dn)
                    suffix = f"{p}_{int(dev_up*10)}"
                    new_features[f"ta_bb_upper_{suffix}"] = u.astype(np.float32)
                    new_features[f"ta_bb_lower_{suffix}"] = l.astype(np.float32)
            except Exception:
                pass
            
            if new_features:
                new_df = pd.DataFrame(new_features, index=df.index)
                df = pd.concat([df, new_df], axis=1)
                return df.copy()

        except Exception as e:
            logger.warning(f"TA-Lib comprehensive dynamic injection failed: {e}")
        
        return df.fillna(0.0)


            

        

    

    def _compute_atr_wicks_cpu(self, df):
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        close_prev = np.concatenate([[close[0]], close[:-1]])
        tr = np.maximum.reduce([high - low, np.abs(high - close_prev), np.abs(low - close_prev)])
        df["atr"] = pd.Series(tr, index=df.index).rolling(14).mean().fillna(0.0)
        rng = (df["high"] - df["low"]).replace(0.0, EPSILON)
        df["upper_wick"] = (df["high"] - np.maximum(df["open"], df["close"])) / rng
        df["lower_wick"] = (np.minimum(df["open"], df["close"]) - df["low"]) / rng

    def _compute_adx_safe(self, df: DataFrame, period: int = 14) -> Series:
        if NUMBA_AVAILABLE:
            high = to_numpy_safe(df["high"].values).astype(np.float64)
            low = to_numpy_safe(df["low"].values).astype(np.float64)
            close = to_numpy_safe(df["close"].values).astype(np.float64)
            return pd.Series(
                compute_adx_numba(high, low, close, period), index=df.index
            )
        return pd.Series(0.0, index=df.index)

    def _compute_stochastic_safe(self, df: DataFrame, period: int = 14) -> tuple[Series, Series]:
        if NUMBA_AVAILABLE:
            high = to_numpy_safe(df["high"].values).astype(np.float64)
            low = to_numpy_safe(df["low"].values).astype(np.float64)
            close = to_numpy_safe(df["close"].values).astype(np.float64)
            k, d = compute_stochastic_numba(high, low, close, period)
            return pd.Series(k, index=df.index), pd.Series(d, index=df.index)
        return pd.Series(50.0, index=df.index), pd.Series(50.0, index=df.index)

    def _compute_cci_safe(self, df: DataFrame, period: int = 20) -> Series:
        if NUMBA_AVAILABLE:
            high = to_numpy_safe(df["high"].values).astype(np.float64)
            low = to_numpy_safe(df["low"].values).astype(np.float64)
            close = to_numpy_safe(df["close"].values).astype(np.float64)
            return pd.Series(
                compute_cci_numba(high, low, close, period), index=df.index
            )
        if TALIB_AVAILABLE:
            try:
                return talib.CCI(df["high"], df["low"], df["close"], timeperiod=period).fillna(0.0)
            except Exception:
                pass
        return pd.Series(0.0, index=df.index)

    def _compute_vwap(self, df: DataFrame, use_gpu: bool = False) -> Series:
        # TODO: Add GPU VWAP if needed, currently CPU Numba is very fast
        if not self.use_volume_features or "volume" not in df.columns:
            return pd.Series(df["close"].to_numpy(dtype=np.float64), index=df.index)
        if NUMBA_AVAILABLE:
            idx = df.index
            if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            dates_arr = idx.to_numpy(dtype="datetime64[D]").astype(np.int64)
            
            high = to_numpy_safe(df["high"].values).astype(np.float64)
            low = to_numpy_safe(df["low"].values).astype(np.float64)
            close = to_numpy_safe(df["close"].values).astype(np.float64)
            vol = to_numpy_safe(df["volume"].values).astype(np.float64)
            
            return pd.Series(
                compute_vwap_numba(
                    high, low, close, vol, dates_arr
                ),
                index=df.index,
            )
        return vwap_pandas(df)

    def _compute_basic_features(self, df: DataFrame, *, use_gpu: bool = False, as_pandas: bool = True) -> DataFrame:
        try:
            if use_gpu and have_cudf() and bool(int(os.environ.get("GPU_FEATURE_CUDF", "1"))):
                import cudf
                df = cudf.from_pandas(df)
            
            # HPC FIX: Causal Indicators (Shift by 1 to prevent look-ahead bias)
            # Decisions at time T must only use data from T-1
            close_lag = df["close"].shift(1)
            
            df["returns"] = df["close"].pct_change().shift(1).fillna(0.0)
            df["ema_fast"] = close_lag.ewm(span=12, adjust=False).mean()
            df["ema_slow"] = close_lag.ewm(span=26, adjust=False).mean()

            if use_gpu and CUPY_AVAILABLE:
                try:
                    vals = rsi_cupy(cp.asarray(close_lag.values), period=14)
                    df["rsi"] = pd.Series(cp.asnumpy(vals), index=df.index)
                except Exception:
                    close_np = to_numpy_safe(close_lag.values).astype(np.float64)
                    df["rsi"] = (
                        rsi_vectorized_numba(close_np, period=14)
                        if NUMBA_AVAILABLE
                        else rsi_pandas(close_lag, period=14)
                    )
            elif NUMBA_AVAILABLE:
                close_np = to_numpy_safe(close_lag.values).astype(np.float64)
                df["rsi"] = rsi_vectorized_numba(close_np, period=14)
            else:
                df["rsi"] = rsi_pandas(close_lag, period=14)

            macd = df["ema_fast"] - df["ema_slow"]
            signal = macd.ewm(span=9, adjust=False).mean()
            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = macd - signal

            # Volatility must also be causal
            high_lag = df["high"].shift(1)
            low_lag = df["low"].shift(1)
            
            df["atr14"] = (high_lag - low_lag).rolling(14).mean()
            bb_mid = close_lag.rolling(20).mean()
            bb_std = close_lag.rolling(20).std()
            df["bb_width"] = (2 * bb_std) / (bb_mid + EPSILON)

            if use_gpu and have_cudf() and hasattr(df, "to_pandas"):
                df = df.to_pandas()

        except Exception as exc:
            logger.warning(f"Basic feature computation failed: {exc}")
            for col in [
                "returns",
                "ema_fast",
                "ema_slow",
                "rsi",
                "macd",
                "macd_signal",
                "macd_hist",
                "atr14",
                "bb_width",
            ]:
                if col not in df.columns:
                    df[col] = 0.0
        return df

    def _compute_volatility_features(self, df: DataFrame, *, as_pandas: bool = True) -> DataFrame:
        try:
            gpu_mode = hasattr(df, "to_pandas") and df.__class__.__module__.startswith("cudf")
            if gpu_mode and hasattr(df, "to_pandas"):
                pass  # type: ignore
            
            new_feats = {}
            new_feats["realized_vol"] = df["returns"].rolling(30).std().fillna(0.0)
            new_feats["ewma_vol"] = df["returns"].ewm(span=30, adjust=False).std().fillna(0.0)

            rolling_20 = df["close"].rolling(20)
            ma20 = rolling_20.mean()
            sd20 = rolling_20.std().fillna(0.0)
            # Use safe division
            new_feats["boll_width"] = np.where(ma20.abs() > 1e-6, 2.0 * sd20 / ma20.abs(), 0.0)

            rolling_50 = df["close"].rolling(50)
            ma50 = rolling_50.mean()
            sd50 = rolling_50.std().fillna(0.0)
            # HPC FIX: Robust Z-Score with Clipping
            new_feats["zscore_close"] = np.clip(np.where(sd50 > 1e-6, (df["close"] - ma50) / sd50, 0.0), -3.0, 3.0)

            if gpu_mode and hasattr(df, "to_pandas"):
                df = df.to_pandas()
            
            # Efficient Merge & Defragmentation (2025 Standard)
            vol_df = pd.DataFrame(new_feats, index=df.index)
            df = pd.concat([df, vol_df], axis=1)
            return df.copy()
            
        except Exception:
            # Fallback
            for col in ["realized_vol", "ewma_vol", "boll_width", "zscore_close"]:
                if col not in df.columns:
                    df[col] = 0.0
        return df

    def _compute_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.use_volume_features or "volume" not in df.columns:
            return df
        return compute_volume_profile_features(df)

    def _compute_obi_features(self, df: pd.DataFrame, use_gpu: bool) -> pd.DataFrame:
        if not self.use_volume_features or "volume" not in df.columns:
            df["vol_imbalance"] = 0.0
            df["adl_slope"] = 0.0
            return df
        return compute_obi_features(df, prefer_gpu=use_gpu)

    def _add_smc_features(self, df: pd.DataFrame, use_gpu: bool) -> pd.DataFrame:
        dev = "cuda" if use_gpu else "cpu"
        if NUMBA_AVAILABLE:
            try:
                import numba
                smc_threads_env = os.environ.get("FOREX_BOT_SMC_THREADS")
                if smc_threads_env:
                    smc_threads = max(1, int(smc_threads_env))
                else:
                    smc_threads = _feature_cpu_budget()
                # Only set if numba is actually importable
                if hasattr(numba, 'set_num_threads'):
                    numba.set_num_threads(smc_threads)
            except Exception:
                pass
        freshness, atr_disp, max_levels = self._auto_smc_params(df)
        return add_smc_features(
            df,
            prefer_gpu=use_gpu,
            device=dev,
            freshness_limit=freshness,
            atr_displacement=atr_disp,
            max_levels=max_levels,
        )

    def _auto_smc_params(self, df: pd.DataFrame) -> tuple[int, float, int]:
        """Derive SMC params from data length/vol unless user overrides."""
        user_fresh = int(getattr(self.settings.system, "smc_freshness_limit", 0) or 0)
        user_disp = float(getattr(self.settings.system, "smc_atr_displacement", 0.0) or 0.0)
        user_levels = int(getattr(self.settings.system, "smc_max_levels", 0) or 0)

        n = len(df)
        # Freshness: shorter for high-frequency, longer for sparse
        fresh_auto = int(min(2000, max(100, n // 1000))) if n > 0 else 200
        freshness = user_fresh if user_fresh > 0 else fresh_auto

        # ATR displacement: base on median ATR vs price to avoid tiny gaps
        atr_series = df["atr"] if "atr" in df.columns else None
        if atr_series is not None and len(atr_series) > 0:
            atr_med = float(np.nanmedian(atr_series))
            price_med = float(np.nanmedian(df["close"])) if "close" in df.columns else 1.0
            # scale to ~0.8–1.6 range depending on vol
            disp_auto = float(np.clip(atr_med / max(price_med, 1e-6) * 5_000, 0.8, 1.6))
        else:
            disp_auto = 1.2
        atr_disp = user_disp if user_disp > 0 else disp_auto

        # Levels to track: more volatility -> more levels
        vol_proxy = float(df["returns"].std()) if "returns" in df.columns else 0.0
        levels_auto = 2 if vol_proxy < 0.002 else 3 if vol_proxy < 0.005 else 4
        max_levels = user_levels if user_levels > 0 else levels_auto

        return freshness, atr_disp, max_levels

    def _meta_label_outcomes(self, df: DataFrame, base_signals: Series) -> Series:
        """
        HPC FIX: Volatility-Normalized Meta Labeling (Unit-Normal).
        """
        close = to_numpy_safe(df["close"].values).astype(np.float64)
        high = to_numpy_safe(df["high"].values).astype(np.float64)
        low = to_numpy_safe(df["low"].values).astype(np.float64)
        atr = to_numpy_safe(df["atr"].values).astype(np.float64)
        
        # Ensure ATR is valid
        atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nanmedian(atr) + 1e-9)
        sigs = to_numpy_safe(base_signals.values).astype(np.int8)

        # 1. Calculate Target Distance in ATR units (Institutional Standard)
        # Instead of generic 20 pips, we use 2 ATR for TP and 1 ATR for SL
        sl_dist = atr * float(getattr(self.settings.risk, "stop_k_vol", 1.0) or 1.0)
        tp_dist = sl_dist * float(getattr(self.settings.risk, "min_risk_reward", 2.0) or 2.0)
        
        horizon = int(getattr(self.settings.risk, "meta_label_max_hold_bars", 100) or 100)
        n = len(close)
        outcomes = np.zeros(n, dtype=np.int8)

        # 2. Optimized Numba Triple-Barrier Search
        @njit(cache=True, fastmath=True)
        def _get_outcomes_numba(c, h, l, s, sl, tp, hrz):
            res = np.zeros(len(c), dtype=np.int8)
            for i in range(len(c) - hrz - 1):
                if s[i] == 0: continue
                
                entry = c[i]
                for j in range(1, hrz + 1):
                    idx = i + j
                    # LONG
                    if s[i] == 1:
                        if l[idx] <= (entry - sl[i]): # Hit SL
                            res[i] = -1
                            break
                        if h[idx] >= (entry + tp[i]): # Hit TP
                            res[i] = 1
                            break
                    # SHORT
                    elif s[i] == -1:
                        if h[idx] >= (entry + sl[i]): # Hit SL
                            res[i] = -1
                            break
                        if l[idx] <= (entry - tp[i]): # Hit TP
                            res[i] = 1
                            break
            return res

        raw_outcomes = _get_outcomes_numba(close, high, low, sigs, sl_dist, tp_dist, horizon)
        return pd.Series(raw_outcomes, index=df.index)

    def _generate_base_signals(self, df: DataFrame, frames: dict[str, DataFrame]) -> Series:
        """
        Generates base signals using an ensemble of evolved experts.
        Prioritizes the new GPU-Native Tensor Experts.
        """
        tensor_path = self.cache_dir.parent / "tensor_knowledge.pt"
        
        # 1. Try New Tensor Experts (GPU-Native)
        if tensor_path.exists():
            try:
                import torch
                from ..strategy.discovery_tensor import TensorDiscoveryEngine
                
                experts_data = torch.load(tensor_path, map_location="cuda")
                genomes = experts_data["genomes"] # (N, Dim)
                timeframes = experts_data["timeframes"]
                
                # HPC FIX: Multi-Timeframe Signal Reconstruction
                # We need features for ALL timeframes used during discovery
                all_signals = torch.zeros((len(df),), device="cuda")
                
                frames_local = frames or {}
                for t_idx, tf_name in enumerate(timeframes):
                    tf_df = frames_local.get(tf_name, df).reindex(df.index).ffill().fillna(0.0)
                    num_df = tf_df.select_dtypes(include=[np.number]).copy()
                    num_df = num_df.shift(1).fillna(0.0)
                    f_vals = num_df.to_numpy(dtype=np.float32)
                    f_tens = torch.from_numpy(f_vals).to("cuda")
                    
                    # Correct slice from genomes for logic weights
                    tf_count = len(timeframes)
                    feat_count = f_tens.shape[1]
                    
                    logic_weights = genomes[:, tf_count : tf_count + feat_count]
                    tf_weights = torch.softmax(genomes[:, :tf_count], dim=1)
                    
                    # Weighted contribution of this TF to the signal
                    tf_sig = torch.matmul(f_tens, logic_weights.t())
                    sig_std = torch.std(tf_sig, dim=0) + 1e-6
                    norm_sig = tf_sig / sig_std
                    
                    all_signals += torch.mean(norm_sig * tf_weights[:, t_idx], dim=1)
                
                final_sig = torch.tanh(all_signals)
                df["signal_strength"] = torch.abs(final_sig).cpu().numpy()
                
                try:
                    sig_thresh = float(os.environ.get("FOREX_BOT_TENSOR_SIGNAL_THRESHOLD", "0.2") or 0.2)
                except Exception:
                    sig_thresh = 0.2
                out = torch.where(torch.abs(final_sig) > sig_thresh, torch.sign(final_sig), 0.0)
                return pd.Series(out.cpu().numpy().astype(np.int8), index=df.index)
                
            except Exception as e:
                logger.warning(f"Tensor Expert signal generation failed: {e}")

        # 2. Fallback to Old Evolution (talib_knowledge.json)
        knowledge_path = self.cache_dir / "talib_knowledge.json"
        try:
            sym = str(getattr(self.settings.system, "symbol", "") or "")
        except Exception:
            sym = ""
        if not sym:
            try:
                sym = str(df.attrs.get("symbol", "") or "")
            except Exception:
                sym = ""
        if sym:
            sym_tag = "".join(c for c in sym if c.isalnum() or c in ("-", "_"))
            sym_path = self.cache_dir / f"talib_knowledge_{sym_tag}.json"
            if sym_path.exists():
                knowledge_path = sym_path
        if knowledge_path.exists():
            try:
                import json
                from ..strategy.genes.factory import GeneFactory
                from ..features.talib_mixer import TALibStrategyMixer
                
                data = json.loads(knowledge_path.read_text())
                genes_data = data.get("best_genes", [])
                
                if genes_data:
                    mixer = TALibStrategyMixer(use_volume_features=self.use_volume_features)
                    all_signals = []
                    frames_local = frames or {}
                    base_tf = str(getattr(self.settings.system, "base_timeframe", "M1") or "M1")

                    genes_by_tf: dict[str, list] = {}
                    for g in genes_data:
                        tf_name = str(g.get("timeframe") or base_tf or base_tf)
                        genes_by_tf.setdefault(tf_name, []).append(GeneFactory.from_dict(g))

                    for tf_name, genes in genes_by_tf.items():
                        tf_df = frames_local.get(tf_name)
                        if tf_df is None or tf_df.empty:
                            tf_df = df
                        else:
                            tf_df = tf_df.copy()
                            if "timestamp" in tf_df.columns:
                                tf_df["timestamp"] = pd.to_datetime(tf_df["timestamp"], utc=True, errors="coerce")
                                tf_df = tf_df.set_index("timestamp")
                            elif not isinstance(tf_df.index, pd.DatetimeIndex):
                                tf_df.index = pd.to_datetime(tf_df.index, utc=True, errors="coerce")
                            tf_df = tf_df.sort_index()

                        cache = mixer.bulk_calculate_indicators(tf_df, genes)
                        for gene in genes:
                            sig = mixer.compute_signals(tf_df, gene, cache=cache)
                            if tf_df is not df:
                                sig = sig.reindex(df.index).ffill().fillna(0.0)
                            all_signals.append(sig.to_numpy(dtype=np.float32))
                    
                    if all_signals:
                        signal_matrix = np.vstack(all_signals)
                        consensus = np.mean(signal_matrix, axis=0)
                        active_experts = np.sum(signal_matrix != 0, axis=0)
                        agreement = np.sum(np.sign(signal_matrix) == np.sign(consensus), axis=0) / np.maximum(active_experts, 1)
                        df["signal_strength"] = agreement.astype(np.float32)
                        final_sig = np.where(np.abs(consensus) > 0.2, np.sign(consensus), 0).astype(np.int8)
                        return pd.Series(final_sig, index=df.index)
            except Exception as e:
                logger.warning(f"Failed to load evolved experts, falling back to RSI: {e}")

        # 3. Final Fallback (Old RSI/EMA cross)
        signals = pd.Series(0, index=df.index, dtype=np.int8)
        try:
            if NUMBA_AVAILABLE:
                rsi_vals = rsi_vectorized_numba(df["close"].values, 2)
            else:
                rsi_vals = rsi_pandas(df["close"], 2)

            rsi = pd.Series(rsi_vals, index=df.index)
            signals.loc[rsi < 15] = 1
            signals.loc[rsi > 85] = -1

            ema_f = df["ema_fast"]
            ema_s = df["ema_slow"]
            signals.loc[(ema_f.shift(1) < ema_s.shift(1)) & (ema_f > ema_s)] = 1
            signals.loc[(ema_f.shift(1) > ema_s.shift(1)) & (ema_f < ema_s)] = -1
        except Exception as e:
            logger.error(f"Final signal fallback failed: {e}")
        
        df["signal_strength"] = 0.5 # Neutral strength for fallback
        return signals


    def _compute_cache_meta(self, df, frames, symbol, news_features, order_book_features):
        sym = symbol or getattr(self.settings.system, "symbol", "UNKNOWN")
        base_tf = getattr(self.settings.system, "base_timeframe", "M5")
        try:
            h = self._fingerprint_df(df)
        except Exception:
            h = str(len(df))

        # Include a lightweight signature of the provided news feature frame so rescoring/backfills
        # invalidate cached feature sets (otherwise stale cached X can ignore updated news values).
        news_sig = None
        if news_features is not None:
            try:
                nf = news_features
                if hasattr(nf, "to_pandas"):
                    nf = nf.to_pandas()
                if not isinstance(nf, pd.DataFrame):
                    nf = pd.DataFrame(nf)
                if nf is not None and len(nf) > 0:
                    if not isinstance(nf.index, pd.DatetimeIndex):
                        nf = nf.copy()
                        nf.index = pd.to_datetime(nf.index, utc=True, errors="coerce")
                    nf = nf.sort_index()
                    keep = [
                        c
                        for c in ["news_sentiment", "news_confidence", "news_count", "news_recency_minutes"]
                        if c in nf.columns
                    ]
                    if keep:
                        nf = nf[keep]
                    news_sig = self._fingerprint_df(nf)[:12]
            except Exception:
                news_sig = None

        subset = {
            "base_tf": base_tf,
            "req_tfs": getattr(self.settings.system, "required_timeframes", []),
            "labels": {
                "tp": getattr(self.settings.risk, "meta_label_tp_pips", None),
                "sl": getattr(self.settings.risk, "meta_label_sl_pips", None),
                "max_hold": getattr(self.settings.risk, "meta_label_max_hold_bars", None),
            },
            "news_enabled": bool(getattr(self.settings.news, "enable_news", False)),
            "enhanced_features": bool(getattr(self.settings.risk, "enhanced_features", False)),
            "news_sig": news_sig,
        }
        settings_sig = hashlib.sha256(json.dumps(subset, sort_keys=True).encode("utf-8")).hexdigest()
        return {
            "cache_key": f"{sym}_{base_tf}_{len(df)}_{h[:12]}_{settings_sig[:10]}_{FEATURE_CACHE_VERSION}.pkl",
            "version": FEATURE_CACHE_VERSION,
        }

    def _load_cached_dataset(self, cache_meta: dict) -> dict | None:
        if not self.cache_enabled or not cache_meta:
            return None
        cache_key = cache_meta.get("cache_key")
        if not cache_key:
            return None
        cache_file = self.cache_dir / cache_key
        if not cache_file.exists():
            return None
        try:
            if self.cache_ttl_minutes > 0:
                age_minutes = (
                    datetime.now(UTC) - datetime.fromtimestamp(cache_file.stat().st_mtime, tz=UTC)
                ).total_seconds() / 60.0
                if age_minutes > self.cache_ttl_minutes:
                    return None
            cached = joblib.load(cache_file)
            if cached.get("version") != FEATURE_CACHE_VERSION:
                return None
            return cached
        except Exception:
            return None

    def _save_cached_dataset(
        self,
        cache_meta: dict,
        X: pd.DataFrame,
        labels: pd.Series,
        index: pd.DatetimeIndex,
        feature_names: list,
        metadata: pd.DataFrame,
    ) -> None:
        if not self.cache_enabled or not cache_meta:
            return
        cache_key = cache_meta.get("cache_key")
        if not cache_key:
            return
        cache_file = self.cache_dir / cache_key
        try:
            cached_data = {
                "version": FEATURE_CACHE_VERSION,
                "X": X,
                "y": labels,
                "index": index,
                "feature_names": feature_names,
                "metadata": metadata,
                "labels": labels,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            joblib.dump(cached_data, cache_file, compress=3)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _merge_indices(self, df):
        return df

    def _fingerprint_df(self, df: pd.DataFrame) -> str:
        """
        Lightweight, deterministic fingerprint to avoid hashing full frames (RAM heavy).
        Uses shape, first/last timestamps, and a small numeric sample.
        """
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(str(len(df)).encode("utf-8"))
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            hasher.update(str(df.index[0]).encode("utf-8"))
            hasher.update(str(df.index[-1]).encode("utf-8"))
        hasher.update("|".join(df.columns).encode("utf-8"))
        numeric = df.select_dtypes(include=["number"])
        if not numeric.empty:
            head = numeric.head(32)
            tail = numeric.tail(32)
            sample = pd.concat([head, tail])
            hasher.update(sample.to_numpy(dtype=np.float64, copy=False).tobytes())
        return hasher.hexdigest()

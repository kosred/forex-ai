import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

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
from .smc import NUMBA_AVAILABLE
from .specialized import add_smc_features, compute_obi_features, compute_volume_profile_features
from .talib_mixer import TALIB_AVAILABLE, talib, ALL_INDICATORS, abstract

if CUPY_AVAILABLE:
    import cupy as cp
else:
    cp = None

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


class FeatureEngineer:
    """Feature engineering pipeline with autotuned optimization."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._indices_cache: pd.DataFrame | None = None
        self.cache_enabled = bool(getattr(settings.system, "cache_enabled", True))
        self.cache_dir = Path(getattr(settings.system, "cache_dir", "cache")) / "features"
        self.cache_ttl_minutes = int(getattr(settings.system, "cache_max_age_minutes", 60))
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
        max_rows = int(getattr(self.settings.system, "max_training_rows_per_tf", 0) or 0)
        if max_rows > 0 and len(df) > max_rows:
            df = df.tail(max_rows)

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

        # === Feature Engineering Dispatch (Stability Fix: Numba for accuracy, GPU for Discovery) ===

        # 1. Basic Features (Always use Numba/CPU for 100% TA-Lib accuracy and Stability)
        df = self._compute_basic_features(df, use_gpu=False)
        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()

        # 2. Advanced Indicators (ADX, Stoch, CCI, MFI)
        # Numba is preferred for speed/safety, but we also now inject full TA-Lib suite 
        # to ensure the Deep Learning models have maximum information.
        df["adx"] = self._compute_adx_safe(df)
        df["stoch_k"], df["stoch_d"] = self._compute_stochastic_safe(df)
        df["cci"] = self._compute_cci_safe(df)
        
        # Comprehensive TA-Lib Injection (Deep Learning Fuel)
        if TALIB_AVAILABLE:
            df = self._compute_comprehensive_talib_features(df)
        else:
            # Fallback for minimal viable features if TA-Lib missing
            try:
                if TALIB_AVAILABLE:
                    df["mfi14"] = talib.MFI(df["high"], df["low"], df["close"], df["volume"], timeperiod=14)
                else:
                    df["mfi14"] = 50.0
            except Exception:
                df["mfi14"] = 50.0

        df = self._merge_indices(df)


        # 3. Multi-Timeframe Features (Optimized "Collect then Concat")
        target_tfs = ["M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "H12", "D1", "W1", "MN1"]
        htf_cols = []
        htf_feature_blocks = [] # Collect blocks here
        
        import gc 

        for tf_name in target_tfs:
            if tf_name in frames and tf_name != base_tf:
                htf = frames[tf_name].copy()
                if not htf.empty:
                    try:
                        if "timestamp" in htf.columns:
                            htf["timestamp"] = pd.to_datetime(htf["timestamp"], utc=True, errors="coerce")
                            htf = htf.set_index("timestamp")
                        elif not isinstance(htf.index, pd.DatetimeIndex):
                            htf.index = pd.to_datetime(htf.index, utc=True, errors="coerce")
                        htf = htf.sort_index()

                        htf = self._compute_basic_features(htf, use_gpu=use_gpu)
                        if not isinstance(htf, pd.DataFrame):
                            htf = htf.to_pandas()

                        htf = self._compute_volatility_features(htf)
                        htf = self._compute_volume_profile_features(htf)
                        htf = self._compute_obi_features(htf, use_gpu)

                        key_feats = [
                            "rsi",
                            "macd",
                            "macd_hist",
                            "atr14",
                            "bb_width",
                            "dist_to_poc",
                            "in_value_area",
                            "vol_imbalance",
                        ]
                        subset = htf[[c for c in key_feats if c in htf.columns]].copy()
                        # Prevent lookahead leakage: align to the last *completed* HTF candle.
                        subset = subset.shift(1)
                        subset.columns = [f"{tf_name}_{c}" for c in subset.columns]
                        
                        # Reindex is still heavy, but now we store just the small subset, not the full df
                        merged = subset.reindex(df.index, method="ffill").fillna(0.0)
                        
                        # Optimization: Downcast to float32 to save RAM
                        merged = merged.astype(np.float32)
                        
                        htf_feature_blocks.append(merged)
                        htf_cols.extend(merged.columns.tolist())
                        
                    except Exception as e:
                        logger.warning(f"Failed to generate HTF features for {tf_name}: {e}")
                    finally:
                        # Aggressive cleanup
                        del htf
                        if 'subset' in locals(): del subset
                        if 'merged' in locals(): del merged
                        gc.collect()

        # Merge all HTF blocks at once
        if htf_feature_blocks:
            df = pd.concat([df] + htf_feature_blocks, axis=1, copy=False)
            del htf_feature_blocks
            gc.collect()

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
                    nf = nf.sort_index().reindex(df.index, method="ffill").fillna(0.0)
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

        df = self._compute_volume_profile_features(df)
        df = self._compute_obi_features(df, use_gpu)

        hours = df.index.hour
        df["session_asia"] = ((hours >= 0) & (hours < 8)).astype(int)
        df["session_london"] = ((hours >= 7) & (hours < 16)).astype(int)
        df["session_ny"] = ((hours >= 12) & (hours < 21)).astype(int)

        df = self._add_smc_features(df, use_gpu and smc_cuda_enabled)
        base_signals = self._generate_base_signals(df)
        # Meta-label outcomes are binary (win/loss) for the base signal direction.
        # Convert to directional action labels in {-1, 0, 1}:
        #   -1 = take short (base short wins)
        #    0 = no trade / filter out (loss or no signal)
        #    1 = take long (base long wins)
        meta_outcomes = self._meta_label_outcomes(df, base_signals).astype(int)
        labels = (base_signals.astype(int) * meta_outcomes).astype(int)

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

        # Build X using the complete dynamic column set
        X = df.reindex(columns=[c for c in cols if c in df.columns], fill_value=0.0)
        X["base_signal"] = base_signals.to_numpy(dtype=np.float32, copy=False)
        if symbol_df is not None:
            X = pd.concat([X, symbol_df], axis=1)

        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X = self._clean_garbage_features(X)
        cols = list(X.columns)

        meta = df[["open", "high", "low", "close", "volume"]].copy()

        if self.cache_enabled and cache_meta:
            self._save_cached_dataset(cache_meta, X, labels, df.index, cols, meta)

        return PreparedDataset(
            X=X.astype(np.float32),
            y=labels.astype(int),
            index=df.index,
            feature_names=cols,
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

                        # Inputs (float64 for precision)

                        inputs = {

                            'open': df["open"].values.astype(np.float64),

                            'high': df["high"].values.astype(np.float64),

                            'low': df["low"].values.astype(np.float64),

                            'close': df["close"].values.astype(np.float64),

                            'volume': df["volume"].values.astype(np.float64) if "volume" in df.columns else np.random.random(len(df))

                        }

            

                        # Collect new columns here instead of inserting into df one-by-one

                        new_features = {}

            

                        # 1. Standard "Kitchen Sink" (Defaults)

                        for ind_name in ALL_INDICATORS:

                            try:

                                if ind_name in ["SMA", "EMA", "RSI", "BBANDS", "ADX", "CCI", "MOM", "ROC", "WILLR", "ATR"]:

                                    continue

                                    

                                func = abstract.Function(ind_name)

                                res = func(inputs)

                                

                                if isinstance(res, (list, tuple)):

                                    for i, arr in enumerate(res):

                                        new_features[f"ta_{ind_name.lower()}_{i}"] = arr.astype(np.float32)

                                elif isinstance(res, (pd.Series, np.ndarray)):

                                    val = res.values if isinstance(res, pd.Series) else res

                                    new_features[f"ta_{ind_name.lower()}"] = val.astype(np.float32)

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

                        

                        # Efficient Merge & Defragmentation (2025 Standard)

                        if new_features:

                            new_df = pd.DataFrame(new_features, index=df.index)

                            df = pd.concat([df, new_df], axis=1)

                            # Force a consolidated memory layout to stop PerformanceWarnings and speed up access

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
                import cudf  # type: ignore

                df = cudf.from_pandas(df)
            df["returns"] = df["close"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
            df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
            df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()

            if use_gpu and CUPY_AVAILABLE:
                try:
                    vals = rsi_cupy(cp.asarray(df["close"].values), period=14)
                    df["rsi"] = pd.Series(cp.asnumpy(vals), index=df.index)
                except Exception:
                    close_np = to_numpy_safe(df["close"].values).astype(np.float64)
                    df["rsi"] = (
                        rsi_vectorized_numba(close_np, period=14)
                        if NUMBA_AVAILABLE
                        else rsi_pandas(df["close"], period=14)
                    )
            elif NUMBA_AVAILABLE:
                close_np = to_numpy_safe(df["close"].values).astype(np.float64)
                df["rsi"] = rsi_vectorized_numba(close_np, period=14)
            else:
                df["rsi"] = rsi_pandas(df["close"], period=14)

            macd = df["ema_fast"] - df["ema_slow"]
            signal = macd.ewm(span=9, adjust=False).mean()
            df["macd"] = macd
            df["macd_signal"] = signal
            df["macd_hist"] = macd - signal

            hl2 = (df["high"] + df["low"]) / 2.0
            atr_period = 14
            df["atr14"] = (df["high"] - df["low"]).rolling(atr_period).mean()
            bb_mid = hl2.rolling(20).mean()
            bb_std = hl2.rolling(20).std()
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
            new_feats["zscore_close"] = np.where(sd50 > 1e-6, (df["close"] - ma50) / sd50, 0.0)

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
        return compute_volume_profile_features(df)

    def _compute_obi_features(self, df: pd.DataFrame, use_gpu: bool) -> pd.DataFrame:
        return compute_obi_features(df, prefer_gpu=use_gpu)

    def _add_smc_features(self, df: pd.DataFrame, use_gpu: bool) -> pd.DataFrame:
        dev = "cuda" if use_gpu else "cpu"
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
        close = to_numpy_safe(df["close"].values).astype(np.float64)
        high = to_numpy_safe(df["high"].values).astype(np.float64)
        low = to_numpy_safe(df["low"].values).astype(np.float64)
        atr = to_numpy_safe(df["atr"].values).astype(np.float64)
        sigs = to_numpy_safe(base_signals.values).astype(np.int8)
        
        labels_arr = compute_meta_labels_atr_numba(
            close,
            high,
            low,
            atr,
            sigs,
            int(self.settings.risk.meta_label_max_hold_bars),
            float(self.settings.risk.atr_stop_multiplier),
            float(self.settings.risk.min_risk_reward),
            0.0005,
            0.0020,
            0.0040,
        )
        return pd.Series(labels_arr, index=df.index)

    def _generate_base_signals(self, df: DataFrame) -> Series:
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
                
                discovery = TensorDiscoveryEngine(device="cuda")
                experts_data = torch.load(tensor_path, map_location="cuda")
                genomes = experts_data["genomes"] # (20, Dim)
                
                # Prepare data slice for signal generation
                # We need all features used during discovery
                features = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                f_tensor = torch.from_numpy(features).to("cuda")
                
                # Decode Genomes (Simplified for signal path)
                tf_count = len(experts_data["timeframes"])
                feat_count = experts_data["n_features"]
                
                # logic_weights = genomes[:, tf_count : tf_count+feat_count]
                # thresholds = genomes[:, -4:-2]
                
                # Calculate Consensus
                # We assume signals are calculated on the CURRENT dataframe timeframe
                # (Actual discovery uses multi-tf, but for base_signal we use current TF logic)
                raw_signals = torch.matmul(f_tensor, genomes[:, tf_count : tf_count+feat_count].t())
                consensus = torch.mean(raw_signals, dim=1)
                
                # Signal Strength (Expert Agreement)
                strength = torch.sum(torch.sign(raw_signals) == torch.sign(consensus.unsqueeze(1)), dim=1).float() / genomes.shape[0]
                df["signal_strength"] = strength.cpu().numpy()
                
                # Final Action
                final_sig = torch.where(torch.abs(consensus) > 0.3, torch.sign(consensus), 0.0)
                return pd.Series(final_sig.cpu().numpy().astype(np.int8), index=df.index)
                
            except Exception as e:
                logger.warning(f"Tensor Expert signal generation failed: {e}")

        # 2. Fallback to Old Evolution (talib_knowledge.json)
        knowledge_path = self.cache_dir / "talib_knowledge.json"
        if knowledge_path.exists():
            try:
                import json
                from ..strategy.genes.factory import GeneFactory
                from ..features.talib_mixer import TALibStrategyMixer
                
                data = json.loads(knowledge_path.read_text())
                genes_data = data.get("best_genes", [])
                
                if genes_data:
                    mixer = TALibStrategyMixer()
                    all_signals = []
                    genes = [GeneFactory.from_dict(g) for g in genes_data]
                    cache = mixer.bulk_calculate_indicators(df, genes)
                    for gene in genes:
                        sig = mixer.compute_signals(df, gene, cache=cache)
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

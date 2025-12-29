import logging

import numpy as np
import pandas as pd

from .smc import (
    CUDA_AVAILABLE,
    NUMBA_AVAILABLE,
    _compute_smc_cuda,
    _compute_smc_numba,
)
from .talib_mixer import TALIB_AVAILABLE, talib

try:
    import cupy as cp  # GPU path if available

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

if NUMBA_AVAILABLE:
    from numba import njit
else:
    njit = None

if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
        n = len(arr)
        out = np.zeros(n, dtype=np.float64)
        if window <= 1:
            for i in range(n):
                out[i] = arr[i]
            return out
        cumsum = 0.0
        for i in range(n):
            cumsum += arr[i]
            if i >= window:
                cumsum -= arr[i - window]
            out[i] = cumsum
        return out


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, parallel=True)
    def _compute_daily_profile_numba(low, high, close, volume, day_ids):
        n = len(close)
        poc_out = np.zeros(n, dtype=np.float32)
        vah_out = np.zeros(n, dtype=np.float32)
        val_out = np.zeros(n, dtype=np.float32)
        
        # 1. Identify Day Start/End Indices
        day_changes = np.where(day_ids[1:] != day_ids[:-1])[0] + 1
        day_starts = np.concatenate([[0], day_changes])
        day_ends = np.concatenate([day_changes, [n]])
        
        # 2. Parallel Profile Calculation
        for d in prange(len(day_starts)):
            start, end = day_starts[d], day_ends[d]
            if end - start < 5: continue
            
            p_min, p_max = np.min(low[start:end]), np.max(high[start:end])
            if p_max == p_min: continue
            
            bin_size = (p_max - p_min) / 50.0 # Standard 50-bin profile
            bins = np.zeros(50, dtype=np.float32)
            
            for i in range(start, end):
                b_idx = int((close[i] - p_min) / bin_size)
                if b_idx >= 50: b_idx = 49
                bins[b_idx] += volume[i]
            
            poc_bin = np.argmax(bins)
            poc_price = p_min + (poc_bin * bin_size)
            
            # Simple 70% Value Area
            total_v = np.sum(bins)
            target_v = total_v * 0.70
            
            # Sort bins by volume
            sorted_bins = np.argsort(bins)[::-1]
            cum_v = 0.0
            min_b, max_b = poc_bin, poc_bin
            
            for b in sorted_bins:
                cum_v += bins[b]
                min_b = min(min_b, b)
                max_b = max(max_b, b)
                if cum_v >= target_v: break
                
            # Propagation: Today's profile features become tomorrow's inputs (prevent leakage)
            if d + 1 < len(day_starts):
                next_start, next_end = day_starts[d+1], day_ends[d+1]
                poc_out[next_start:next_end] = poc_price
                vah_out[next_start:next_end] = p_min + (max_b * bin_size)
                val_out[next_start:next_end] = p_min + (min_b * bin_size)
                
        return poc_out, vah_out, val_out

def compute_volume_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """HPC Optimized: Parallel Volume Profile."""
    if "close" not in df.columns or "volume" not in df.columns:
        return df
    try:
        l, h, c, v = df["low"].values, df["high"].values, df["close"].values, df["volume"].values
        day_ids = df.index.dayofyear.values.astype(np.int32)
        
        poc, vah, val = _compute_daily_profile_numba(l, h, c, v, day_ids)
        
        df["poc_daily"], df["vah_daily"], df["val_daily"] = poc, vah, val
        df["dist_to_poc"] = df["close"] - df["poc_daily"]
        df["in_value_area"] = ((df["close"] >= df["val_daily"]) & (df["close"] <= df["vah_daily"])).astype(np.int8)
    except Exception as e:
        logger.warning(f"Volume profile failed: {e}")
    return df


def compute_obi_features(df: pd.DataFrame, prefer_gpu: bool = False) -> pd.DataFrame:
    """Compute Order Book Imbalance (OBI) proxy. GPU-accelerated when CuPy is available."""
    try:
        if prefer_gpu and CUPY_AVAILABLE:
            close_vals = cp.asarray(df["close"].to_numpy(dtype=np.float64))
            vol_vals = cp.asarray(df["volume"].to_numpy(dtype=np.float64))

            close_change = cp.empty_like(close_vals)
            close_change[0] = 0.0
            close_change[1:] = close_vals[1:] - close_vals[:-1]

            vol_up = cp.where(close_change > 0, vol_vals, 0.0)
            vol_down = cp.where(close_change < 0, vol_vals, 0.0)

            kernel = cp.ones(5, dtype=cp.float64)
            vol_up_smooth = cp.convolve(vol_up, kernel, mode="full")[: len(vol_up)]
            vol_down_smooth = cp.convolve(vol_down, kernel, mode="full")[: len(vol_down)]

            total_vol = vol_up_smooth + vol_down_smooth
            vol_imbalance = (vol_up_smooth - vol_down_smooth) / (total_vol + 1e-9)
            df["vol_imbalance"] = cp.asnumpy(vol_imbalance)
        else:
            close_vals = df["close"].to_numpy(dtype=np.float64)
            vol_vals = df["volume"].to_numpy(dtype=np.float64)

            close_change = np.empty_like(close_vals)
            close_change[0] = 0.0
            close_change[1:] = close_vals[1:] - close_vals[:-1]

            vol_up = np.where(close_change > 0, vol_vals, 0.0)
            vol_down = np.where(close_change < 0, vol_vals, 0.0)

            if NUMBA_AVAILABLE:
                vol_up_smooth = _rolling_sum(vol_up, 5)
                vol_down_smooth = _rolling_sum(vol_down, 5)
            else:
                vol_up_smooth = pd.Series(vol_up).rolling(5, min_periods=1).sum().to_numpy()
                vol_down_smooth = pd.Series(vol_down).rolling(5, min_periods=1).sum().to_numpy()

            total_vol = vol_up_smooth + vol_down_smooth
            df["vol_imbalance"] = (vol_up_smooth - vol_down_smooth) / (total_vol + 1e-9)

        if talib and TALIB_AVAILABLE:
            df["adl"] = talib.AD(df["high"], df["low"], df["close"], df["volume"])
            df["adl_slope"] = df["adl"].diff(5) / (df["adl"].abs().rolling(20).mean() + 1e-9)
        else:
            mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"] + 1e-9)
            mf_volume = mf_multiplier * df["volume"]
            adl = mf_volume.cumsum()
            df["adl_slope"] = adl.diff(5)

    except Exception as e:
        logger.warning(f"OBI feature calc failed: {e}")
        df["vol_imbalance"] = 0.0
        df["adl_slope"] = 0.0

    return df


def add_smc_features(
    df: pd.DataFrame,
    prefer_gpu: bool = False,
    device: str = "cpu",
    freshness_limit: int = 200,
    atr_displacement: float = 1.2,
    max_levels: int = 3,
) -> pd.DataFrame:
    """
    Add Smart Money Concepts (SMC) features to dataframe.

    NOTE: CPU (Numba) is the default and preferred path for accuracy/stability.
    CUDA is optional and will fall back to CPU if unavailable or unstable.
    """
    try:
        open_Arr = df["open"].values.astype(np.float32)
        high_Arr = df["high"].values.astype(np.float32)
        low_Arr = df["low"].values.astype(np.float32)
        close_Arr = df["close"].values.astype(np.float32)
        atr_Arr = df["atr"].values.astype(np.float32) if "atr" in df.columns else np.zeros_like(close_Arr)

        use_cuda = prefer_gpu and CUDA_AVAILABLE
        if use_cuda:
            try:
                fvg, sweep, ob, bos, choch, trend, dist_liq, dist_fvg, pd_zone, inducement = _compute_smc_cuda(
                    open_Arr,
                    high_Arr,
                    low_Arr,
                    close_Arr,
                    atr_Arr,
                    lookback=5,
                    freshness_limit=freshness_limit,
                    atr_displacement=atr_displacement,
                    max_levels=max_levels,
                )
            except Exception as exc:
                logger.warning(f"SMC CUDA failed, falling back to CPU: {exc}")
                use_cuda = False

        if not use_cuda and NUMBA_AVAILABLE:
            fvg, sweep, ob, bos, choch, trend, dist_liq, dist_fvg, pd_zone, inducement = _compute_smc_numba(
                open_Arr,
                high_Arr,
                low_Arr,
                close_Arr,
                atr_Arr,
                lookback=5,
                freshness_limit=freshness_limit,
                atr_displacement=atr_displacement,
                max_levels=max_levels,
            )
        elif not use_cuda:
            logger.warning("Numba not available, SMC features will be zeros")
            fvg = sweep = ob = bos = choch = trend = dist_liq = dist_fvg = pd_zone = inducement = np.zeros_like(
                close_Arr
            )

        df["fvg_direction"], df["fvg_size"], df["liquidity_sweep"], df["order_block"] = fvg, np.abs(fvg), sweep, ob
        df["bos_event"], df["choch_event"], df["market_structure_trend"] = bos, choch, trend
        df["dist_liquidity"], df["dist_fvg"], df["premium_discount"], df["inducement"] = (
            dist_liq,
            dist_fvg,
            pd_zone,
            inducement,
        )
        df["smc_confluence"] = (
            (df["order_block"] != 0).astype(int)
            + (df["fvg_direction"] != 0).astype(int)
            + (df["liquidity_sweep"] != 0).astype(int)
        )
    except Exception as e:
        logger.warning(f"SMC feature generation failed: {e}")
    return df

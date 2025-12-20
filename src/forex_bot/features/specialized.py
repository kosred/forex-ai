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


def compute_volume_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Session Volume Profile features: POC, VAH, VAL."""
    if "close" not in df.columns or "volume" not in df.columns:
        return df

    try:

        def calc_day_profile(group):
            price_min = group["low"].min()
            price_max = group["high"].max()
            if price_max == price_min:
                return pd.Series(0, index=["poc", "vah", "val"])

            bin_size = (price_max - price_min) / 100.0
            if bin_size == 0:
                bin_size = 0.0001

            bins = ((group["close"] - price_min) / bin_size).astype(int)
            vol_by_bin = group.groupby(bins)["volume"].sum().sort_index()

            if vol_by_bin.empty:
                return pd.Series(0, index=["poc", "vah", "val"])

            poc_bin = vol_by_bin.idxmax()
            poc_price = price_min + (poc_bin * bin_size)

            total_vol = vol_by_bin.sum()
            target_vol = total_vol * 0.70

            sorted_vol = vol_by_bin.sort_values(ascending=False)
            cum_vol = 0
            va_bins = []
            for b, v in sorted_vol.items():
                cum_vol += v
                va_bins.append(b)
                if cum_vol >= target_vol:
                    break

            if not va_bins:
                vah = val = poc_price
            else:
                vah = price_min + (max(va_bins) * bin_size)
                val = price_min + (min(va_bins) * bin_size)

            return pd.Series({"poc": poc_price, "vah": vah, "val": val})

        daily_stats = df.groupby(df.index.date).apply(calc_day_profile)
        daily_stats = daily_stats.shift(1)

        daily_stats.index = pd.to_datetime(daily_stats.index)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            if getattr(daily_stats.index, "tz", None) is None:
                if str(df.index.tz) == "UTC":
                    daily_stats.index = daily_stats.index.tz_localize("UTC")
                else:
                    daily_stats.index = daily_stats.index.tz_localize(df.index.tz)

        temp_df = df.copy()
        temp_df["date"] = temp_df.index.normalize()
        merged = temp_df.merge(daily_stats, left_on="date", right_index=True, how="left")

        df["poc_daily"] = merged["poc"].ffill().fillna(0.0)
        df["vah_daily"] = merged["vah"].ffill().fillna(0.0)
        df["val_daily"] = merged["val"].ffill().fillna(0.0)

        df["dist_to_poc"] = df["close"] - df["poc_daily"]
        df["in_value_area"] = ((df["close"] >= df["val_daily"]) & (df["close"] <= df["vah_daily"])).astype(int)

    except Exception as e:
        logger.warning(f"Volume profile calc failed: {e}")
        df["dist_to_poc"] = 0.0
        df["in_value_area"] = 0

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

    NOTE: CUDA implementation has been removed as it was slower than CPU.
    Always uses Numba CPU parallelization which is faster for typical data sizes.
    """
    try:
        open_Arr = df["open"].values.astype(np.float32)
        high_Arr = df["high"].values.astype(np.float32)
        low_Arr = df["low"].values.astype(np.float32)
        close_Arr = df["close"].values.astype(np.float32)
        atr_Arr = df["atr"].values.astype(np.float32) if "atr" in df.columns else np.zeros_like(close_Arr)

        if prefer_gpu and CUDA_AVAILABLE:
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
                fvg = sweep = ob = bos = choch = trend = dist_liq = dist_fvg = pd_zone = inducement = np.zeros_like(
                    close_Arr
                )
        elif NUMBA_AVAILABLE:
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
        else:
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

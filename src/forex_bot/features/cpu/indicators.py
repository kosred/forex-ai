import numpy as np
import pandas as pd

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*, cache: bool = False, fastmath: bool = False):  # type: ignore[no-redef]
        def _decorator(func):
            return func

        return _decorator

    prange = range

EPSILON = 1e-10

# --- Numba Optimized Kernels ---

if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def rsi_vectorized_numba(series_arr: np.ndarray, period: int = 14) -> np.ndarray:
        n = len(series_arr)
        if n < period + 1:
            return np.full(n, 0.0)

        diff = np.diff(series_arr)
        gain = np.zeros(len(diff))
        loss = np.zeros(len(diff))

        for i in prange(len(diff)):
            if diff[i] > 0:
                gain[i] = diff[i]
            else:
                loss[i] = -diff[i]

        avg_gain = np.zeros(n - 1)
        avg_loss = np.zeros(n - 1)

        avg_gain[period - 1] = np.mean(gain[:period])
        avg_loss[period - 1] = np.mean(loss[:period])

        for i in prange(period, len(diff)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

        rs = np.zeros(n - 1)
        rsi = np.zeros(n)

        for i in prange(period - 1, n - 1):
            if avg_loss[i] < 1e-10:
                rsi[i + 1] = 100.0
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs[i]))

        return rsi

    @njit(cache=True, fastmath=True, parallel=True)
    def compute_adx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        n = len(close)
        adx_out = np.full(n, 0.0)
        if n < 2 * period + 1:
            return adx_out

        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        true_range = np.zeros(n)

        for i in prange(1, n):
            move_up = high[i] - high[i - 1]
            move_down = low[i - 1] - low[i]

            if move_up > move_down and move_up > 0:
                plus_dm[i] = move_up
            else:
                plus_dm[i] = 0.0

            if move_down > move_up and move_down > 0:
                minus_dm[i] = move_down
            else:
                minus_dm[i] = 0.0

            true_range[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

        alpha = 1.0 / period
        smooth_plus_dm = np.zeros(n)
        smooth_minus_dm = np.zeros(n)
        smooth_true_range = np.zeros(n)

        if n >= period:
            smooth_plus_dm[period - 1] = np.mean(plus_dm[:period])
            smooth_minus_dm[period - 1] = np.mean(minus_dm[:period])
            smooth_true_range[period - 1] = np.mean(true_range[:period])

        for i in prange(n):
            if i < period:
                continue
            elif i == period:
                smooth_plus_dm[i] = smooth_plus_dm[i - 1] * (1 - alpha) + plus_dm[i] * alpha
                smooth_minus_dm[i] = smooth_minus_dm[i - 1] * (1 - alpha) + minus_dm[i] * alpha
                smooth_true_range[i] = smooth_true_range[i - 1] * (1 - alpha) + true_range[i] * alpha
            else:
                smooth_plus_dm[i] = smooth_plus_dm[i - 1] * (1 - alpha) + plus_dm[i] * alpha
                smooth_minus_dm[i] = smooth_minus_dm[i - 1] * (1 - alpha) + minus_dm[i] * alpha
                smooth_true_range[i] = smooth_true_range[i - 1] * (1 - alpha) + true_range[i] * alpha

        plus_di = np.zeros(n)
        minus_di = np.zeros(n)

        for i in prange(n):
            if smooth_true_range[i] > EPSILON:
                plus_di[i] = 100 * (smooth_plus_dm[i] / smooth_true_range[i])
                minus_di[i] = 100 * (smooth_minus_dm[i] / smooth_true_range[i])
            else:
                plus_di[i] = 0.0
                minus_di[i] = 0.0

        dx = np.zeros(n)
        for i in prange(n):
            sum_di = plus_di[i] + minus_di[i]
            if sum_di > EPSILON:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / sum_di
            else:
                dx[i] = 0.0

        for i in prange(n):
            if i == 0:
                adx_out[i] = dx[i] * alpha
            else:
                adx_out[i] = adx_out[i - 1] * (1 - alpha) + dx[i] * alpha

        return adx_out

    @njit(cache=True, fastmath=True, parallel=True)
    def compute_stochastic_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(close)
        stoch_k = np.full(n, 50.0)
        stoch_d = np.full(n, 50.0)
        if n < period:
            return stoch_k, stoch_d

        highest_high = np.zeros(n)
        lowest_low = np.zeros(n)

        for i in prange(n):
            if i < period - 1:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i - period + 1 : i + 1])
                lowest_low[i] = np.min(low[i - period + 1 : i + 1])

            if (highest_high[i] - lowest_low[i]) > EPSILON:
                stoch_k[i] = 100.0 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])

        for i in prange(n):
            if i < period + 1:
                stoch_d[i] = np.nan
            else:
                stoch_d[i] = (stoch_k[i] + stoch_k[i - 1] + stoch_k[i - 2]) / 3.0

        return np.nan_to_num(stoch_k, nan=50.0), np.nan_to_num(stoch_d, nan=50.0)

    @njit(cache=True, fastmath=True, parallel=True)
    def compute_cci_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
        n = len(close)
        cci_out = np.full(n, 0.0)
        if n < period:
            return cci_out

        typical_price = (high + low + close) / 3.0

        sma_tp = np.zeros(n)
        for i in prange(period - 1, n):
            sma_tp[i] = np.mean(typical_price[i - period + 1 : i + 1])

        mean_deviation = np.zeros(n)
        for i in prange(period - 1, n):
            mean_deviation[i] = np.mean(np.abs(typical_price[i - period + 1 : i + 1] - sma_tp[i]))

        for i in prange(period - 1, n):
            if mean_deviation[i] > EPSILON:
                cci_out[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mean_deviation[i])
            else:
                cci_out[i] = 0.0

        return cci_out

    @njit(cache=True, fastmath=True, parallel=True)
    def compute_vwap_numba(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, dates_epoch_days: np.ndarray
    ) -> np.ndarray:
        n = len(close)
        vwap_out = np.full(n, 0.0)

        if n == 0:
            return vwap_out

        tp_v_arr = ((high + low + close) / 3.0) * volume

        current_daily_cum_tp_v = 0.0
        current_daily_cum_vol = 0.0

        for i in prange(n):
            if i > 0 and dates_epoch_days[i] != dates_epoch_days[i - 1]:
                current_daily_cum_tp_v = 0.0
                current_daily_cum_vol = 0.0

            current_daily_cum_tp_v += tp_v_arr[i]
            current_daily_cum_vol += volume[i]

            if current_daily_cum_vol > EPSILON:
                vwap_out[i] = current_daily_cum_tp_v / current_daily_cum_vol
            else:
                vwap_out[i] = vwap_out[i - 1] if i > 0 else close[i]

        return vwap_out

# --- Pandas Fallbacks ---


def rsi_pandas(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + EPSILON)
    return (100 - (100 / (1 + rs))).fillna(0.0)


def vwap_pandas(df: pd.DataFrame) -> pd.Series:
    df_tmp = df.copy()
    df_tmp["tp_v"] = ((df_tmp["high"] + df_tmp["low"] + df_tmp["close"]) / 3.0) * df_tmp["volume"]
    grouped = df_tmp.groupby(df_tmp.index.date)
    cum_tp_v = grouped["tp_v"].cumsum()
    cum_vol = grouped["volume"].cumsum().replace(0, 1.0)
    vwap = cum_tp_v / cum_vol
    return vwap.ffill().fillna(0.0)

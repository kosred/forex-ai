import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Optional Numba acceleration for signal combination
try:  # pragma: no cover
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    njit = None  # type: ignore
    NUMBA_AVAILABLE = False

# Optional GPU backend (CuPy ≥13 supports Python 3.13 & CUDA 12.8+)
try:  # pragma: no cover
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    import cupyx.scipy.signal as cpx_signal

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    cpx_signal = None
    cpx_ndimage = None
    CUPY_AVAILABLE = False

# Optional cuDF (RAPIDS) for GPU rolling ops
try:  # pragma: no cover
    import cudf

    CUDA_AVAILABLE = True
except Exception:  # pragma: no cover
    cudf = None
    CUDA_AVAILABLE = False

try:
    import talib
    from talib import abstract

    TALIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    talib = None
    abstract = None
    TALIB_AVAILABLE = False

logger = logging.getLogger(__name__)

TALIB_INDICATORS = {
    "trend": [
        "SMA",
        "EMA",
        "WMA",
        "DEMA",
        "TEMA",
        "TRIMA",
        "KAMA",
        "MAMA",
        "T3",
        "ADX",
        "ADXR",
        "APO",
        "AROON",
        "AROONOSC",
        "BOP",
        "CCI",
        "DX",
        "MACD",
        "MACDEXT",
        "MACDFIX",
        "MOM",
        "PPO",
        "ROC",
        "ROCP",
        "ROCR",
        "ROCR100",
        "RSI",
        "TRIX",
        "ULTOSC",
        "WILLR",
        "HT_TRENDLINE",
        "HT_TRENDMODE",
    ],
    "volatility": ["ATR", "NATR", "TRANGE", "BBANDS", "STDDEV", "VAR", "HT_DCPERIOD", "HT_DCPHASE"],
    "momentum": [
        "ADX",
        "ADXR",
        "APO",
        "AROON",
        "AROONOSC",
        "BOP",
        "CCI",
        "CMO",
        "DX",
        "MACD",
        "MFI",
        "MINUS_DI",
        "MINUS_DM",
        "MOM",
        "PLUS_DI",
        "PLUS_DM",
        "PPO",
        "ROC",
        "ROCP",
        "ROCR",
        "ROCR100",
        "RSI",
        "STOCH",
        "STOCHF",
        "STOCHRSI",
        "TRIX",
        "ULTOSC",
        "WILLR",
    ],
    "reversal": ["RSI", "STOCH", "STOCHF", "STOCHRSI", "WILLR", "CCI", "MFI", "BBANDS", "SAR", "SAREXT"],
    "volume": ["AD", "ADOSC", "OBV", "MFI"],
    "price_transform": ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE", "HT_TRENDLINE"],
}
ALL_INDICATORS = list({ind for cat in TALIB_INDICATORS.values() for ind in cat})


@dataclass(slots=True)
class TALibStrategyGene:
    indicators: list[str]
    params: dict[str, dict[str, Any]]
    combination_method: str = "weighted_vote"
    long_threshold: float = 0.6
    short_threshold: float = -0.6
    weights: dict[str, float] = None
    preferred_regime: str = "any"
    fitness: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    strategy_id: str = ""
    use_ob: bool = False  # Require Order Block for entry
    use_fvg: bool = False  # Require Fair Value Gap for entry
    use_liq_sweep: bool = False  # Require Liquidity Sweep for entry
    mtf_confirmation: bool = False  # Require HTF Trend Confirmation
    use_premium_discount: bool = False  # Require Discount for Buy, Premium for Sell
    use_inducement: bool = False  # Require Inducement near entry
    tp_pips: float = 40.0
    sl_pips: float = 20.0


class TALibStrategyMixer:
    def __init__(self, device: str = "cpu", use_volume_features: bool = False) -> None:
        """
        Args:
            device: 'cpu' or 'cuda'. When 'cuda' and CuPy is available, a
                    fast GPU path is used for a subset of indicators.
        """
        self.indicator_synergy_matrix: dict[tuple[str, str], float] = {}
        self.regime_performance: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.available_indicators = []
        self.use_volume_features = use_volume_features
        self._volume_indicators = set(TALIB_INDICATORS.get("volume", [])) | {"VWAP"}

        # GPU flag is explicit to avoid surprise memory usage
        self.device = device.lower()
        self.use_gpu = self.device in ("cuda", "gpu") and CUPY_AVAILABLE

        if self.use_gpu and not CUPY_AVAILABLE:
            logger.warning("CUDA requested but CuPy not available; falling back to CPU.")
            self.use_gpu = False
        # Batch GPU mode: process indicators in groups to reduce kernel launches
        self.gpu_batch = bool(int(os.environ.get("GPU_FEATURE_BATCH", "1"))) if self.use_gpu else False
        self.use_cudf = self.use_gpu and CUDA_AVAILABLE and bool(int(os.environ.get("GPU_FEATURE_CUDF", "1")))
        self.parity_check = bool(int(os.environ.get("GPU_PARITY_CHECK", "1")))
        self.parity_tol = float(os.environ.get("GPU_PARITY_TOL", "5e-4"))
        self._gpu_blacklist: set[str] = set()

        if TALIB_AVAILABLE:
            for ind in ALL_INDICATORS:
                if (not self.use_volume_features) and ind in self._volume_indicators:
                    continue
                try:
                    _ = abstract.Function(ind).info
                    self.available_indicators.append(ind)
                except Exception as e:
                    logger.warning(f"TA-Lib indicator calculation failed: {e}", exc_info=True)
            if not self.available_indicators:
                logger.warning("TA-Lib installed but no indicators found available via abstract API.")
        else:
            logger.warning("TA-Lib not installed; mixer will be inactive.")

        # Subset we can compute on GPU (expanded)
        self._gpu_supported = {
            # Moving averages & momentum
            "SMA",
            "EMA",
            "DEMA",
            "TEMA",
            "WMA",
            "TRIMA",
            "KAMA",
            "T3",
            "MOM",
            "ROC",
            "ROCP",
            "ROCR",
            "ROCR100",
            "RSI",
            "STOCHRSI",
            "CMO",
            "PPO",
            "APO",
            "MACD",
            "TRIX",
            "ULTOSC",
            # Volatility
            "ATR",
            "NATR",
            "TRANGE",
            "STDDEV",
            "VAR",
            "BBANDS",
            "SAR",
            "SAREXT",
            # Trend/strength
            "ADX",
            "ADXR",
            "DX",
            "PLUS_DI",
            "MINUS_DI",
            "PLUS_DM",
            "MINUS_DM",
            "AROON",
            "AROONOSC",
            "MFI",
            "STOCH",
            "STOCHF",
            "CCI",
            "WILLR",
            "HT_TRENDLINE",
            "HT_TRENDMODE",
            "HT_DCPERIOD",
            "HT_DCPHASE",
            # Volume/price transforms
            "OBV",
            "AD",
            "ADOSC",
            "AVGPRICE",
            "MEDPRICE",
            "TYPPRICE",
            "WCLPRICE",
            "BOP",
            "VWAP",
            # Candlestick patterns (approximate parity; auto-fallback on mismatch)
            "CDL_DOJI",
            "CDL_HAMMER",
            "CDL_SHOOTINGSTAR",
            "CDL_ENGULFING",
            "CDL_MARUBOZU",
            "CDL_SPINNINGTOP",
        }
        if not self.use_volume_features:
            self._gpu_supported.difference_update(self._volume_indicators)

    # --------------------------- GPU helpers --------------------------- #
    def _ema_gpu(self, arr: "cp.ndarray", period: int) -> "cp.ndarray":
        """EMA via IIR filter on GPU; keeps numeric parity close to TA-Lib."""
        alpha = 2.0 / (period + 1.0)
        if cpx_signal is not None:
            # y[n] = alpha*x[n] + (1-alpha)*y[n-1]
            b = cp.array([alpha], dtype=cp.float32)
            a = cp.array([1.0, -(1.0 - alpha)], dtype=cp.float32)
            return cpx_signal.lfilter(b, a, arr)
        # Fallback: manual recurrence (still on GPU)
        out = cp.empty_like(arr, dtype=cp.float32)
        if arr.size == 0:
            return out
        out[0] = arr[0]
        for i in range(1, arr.size):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return out

    def _wma_gpu(self, arr: "cp.ndarray", period: int) -> "cp.ndarray":
        if period <= 1:
            return arr
        weights = cp.arange(1, period + 1, dtype=cp.float32)
        kernel = weights / weights.sum()
        if cpx_signal is not None:
            return cpx_signal.convolve(arr, kernel[::-1], mode="same")
        return cp.convolve(arr, kernel[::-1], mode="same")

    def _sma_gpu(self, arr: "cp.ndarray", period: int) -> "cp.ndarray":
        if period <= 1:
            return arr
        kernel = cp.ones(period, dtype=cp.float32) / period
        if cpx_signal is not None:
            return cpx_signal.convolve(arr, kernel, mode="same")
        return cp.convolve(arr, kernel, mode="same")

    def _rolling_minmax_gpu(self, arr: "cp.ndarray", period: int, mode: str) -> "cp.ndarray":
        if period <= 1:
            return arr
        if cpx_ndimage is not None:
            if mode == "min":
                return cpx_ndimage.minimum_filter1d(arr, period, mode="nearest")
            return cpx_ndimage.maximum_filter1d(arr, period, mode="nearest")
        # Fallback: stride trick with cp (approx)
        rolled = cp.stack([cp.roll(arr, -k) for k in range(period)], axis=0)
        return cp.min(rolled, axis=0) if mode == "min" else cp.max(rolled, axis=0)

    def _calc_indicator_gpu(
        self, ind: str, params: dict[str, Any], df: pd.DataFrame, preloaded: dict | None = None
    ) -> np.ndarray | None:
        """Compute a limited set of indicators on GPU; return numpy array or None."""
        if (not self.use_volume_features) and ind in self._volume_indicators:
            return None
        if not self.use_gpu or ind not in self._gpu_supported or ind in self._gpu_blacklist:
            return None
        try:
            if preloaded:
                close = preloaded.get("close")
                high = preloaded.get("high") or close
                low = preloaded.get("low") or close
                open_ = preloaded.get("open") or close
                vol = preloaded.get("volume") if self.use_volume_features else None
            else:
                close = cp.asarray(df["close"].to_numpy(), dtype=cp.float32)
                high = cp.asarray(df["high"].to_numpy(), dtype=cp.float32) if "high" in df else close
                low = cp.asarray(df["low"].to_numpy(), dtype=cp.float32) if "low" in df else close
                open_ = cp.asarray(df["open"].to_numpy(), dtype=cp.float32) if "open" in df else close
                vol = (
                    cp.asarray(df["volume"].to_numpy(), dtype=cp.float32)
                    if self.use_volume_features and "volume" in df
                    else None
                )

            tp = int(params.get("timeperiod", params.get("period", 14)))
            tp = max(2, min(tp, 500))

            if ind in ("SMA",):
                return cp.asnumpy(self._sma_gpu(close, tp))
            if ind in ("WMA",):
                return cp.asnumpy(self._wma_gpu(close, tp))
            if ind == "TRIMA":
                # triangular moving average: SMA of SMA
                sma1 = self._sma_gpu(close, tp)
                return cp.asnumpy(self._sma_gpu(sma1, tp))
            if ind == "KAMA":
                # Kaufman adaptive MA
                er_period = tp
                change = cp.abs(close - cp.roll(close, er_period))
                vol = cp.abs(close - cp.roll(close, 1))
                vol[:er_period] = 0
                sum_vol = cp.convolve(vol, cp.ones(er_period), mode="same")
                er = cp.where(sum_vol > 1e-9, change / sum_vol, 0.0)
                fast = 2.0 / (2 + 1)
                slow = 2.0 / (30 + 1)
                sc = cp.square(er * (fast - slow) + slow)
                kama = cp.empty_like(close)
                kama[0] = close[0]
                for i in range(1, close.size):
                    kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
                return cp.asnumpy(kama)
            if ind == "T3":
                v = float(params.get("vfactor", 0.7))
                e1 = self._ema_gpu(close, tp)
                e2 = self._ema_gpu(e1, tp)
                e3 = self._ema_gpu(e2, tp)
                e4 = self._ema_gpu(e3, tp)
                e5 = self._ema_gpu(e4, tp)
                e6 = self._ema_gpu(e5, tp)
                c1 = -(v**3)
                c2 = 3 * v**2 + 3 * v**3
                c3 = -6 * v**2 - 3 * v - 3 * v**3
                c4 = 1 + 3 * v + v**3 + 3 * v**2
                t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
                return cp.asnumpy(t3)
            if ind in ("EMA", "DEMA", "TEMA"):
                ema = self._ema_gpu(close, tp)
                if ind == "DEMA":
                    ema2 = self._ema_gpu(ema, tp)
                    out = 2 * ema - ema2
                elif ind == "TEMA":
                    ema2 = self._ema_gpu(ema, tp)
                    ema3 = self._ema_gpu(ema2, tp)
                    out = 3 * (ema - ema2) + ema3
                else:
                    out = ema
                return cp.asnumpy(out)
            if ind == "RSI":
                diff = cp.diff(cp.pad(close, (1, 0), mode="edge"))
                gain = cp.maximum(diff, 0.0)
                loss = cp.maximum(-diff, 0.0)
                avg_gain = self._sma_gpu(gain, tp)
                avg_loss = self._sma_gpu(loss, tp)
                rs = cp.where(avg_loss > 1e-9, avg_gain / avg_loss, 0.0)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                rsi[:tp] = 50.0
                return cp.asnumpy(rsi)
            if ind in ("MOM",):
                shifted = cp.roll(close, tp)
                mom = close - shifted
                mom[:tp] = 0.0
                return cp.asnumpy(mom)
            if ind in ("ROC", "ROCP", "ROCR", "ROCR100"):
                shifted = cp.roll(close, tp)
                shifted[:tp] = close[:tp]
                if ind == "ROC":
                    out = (close - shifted) / shifted * 100.0
                elif ind == "ROCP":
                    out = (close / shifted) - 1.0
                elif ind == "ROCR":
                    out = close / shifted
                else:
                    out = (close / shifted) * 100.0
                out[:tp] = 0.0
                return cp.asnumpy(out)
            if ind in ("PPO", "APO"):
                fast = int(params.get("fastperiod", 12))
                slow = int(params.get("slowperiod", 26))
                ema_fast = self._ema_gpu(close, fast)
                ema_slow = self._ema_gpu(close, slow)
                if ind == "PPO":
                    out = cp.where(cp.abs(ema_slow) > 1e-9, (ema_fast - ema_slow) / ema_slow * 100.0, 0.0)
                else:
                    out = ema_fast - ema_slow
                return cp.asnumpy(out)
            if ind == "ATR":
                prev_close = cp.concatenate([close[:1], close[:-1]])
                tr = cp.maximum(cp.maximum(high - low, cp.abs(high - prev_close)), cp.abs(low - prev_close))
                atr = self._sma_gpu(tr, tp)
                return cp.asnumpy(atr)
            if ind == "SAR" or ind == "SAREXT":
                # Parabolic SAR port aligned with TA-Lib default params
                acc = float(params.get("acceleration", 0.02))
                acc_max = float(params.get("maximum", 0.2))
                n = close.size
                sar = cp.empty(n, dtype=cp.float32)
                # Initialization: use first two bars to set trend
                long = high[1] + low[1] >= high[0] + low[0]
                ep = high[0] if long else low[0]
                sar[0] = low[0] if long else high[0]
                af = acc
                for i in range(1, n):
                    psar = sar[i - 1] + af * (ep - sar[i - 1])
                    if long:
                        psar = cp.minimum(psar, low[i - 1])
                        if i > 1:
                            psar = cp.minimum(psar, low[i - 2])
                        if high[i] > ep:
                            ep = high[i]
                            af = cp.minimum(af + acc, acc_max)
                        if low[i] < psar:
                            long = False
                            psar = ep
                            ep = low[i]
                            af = acc
                    else:
                        psar = cp.maximum(psar, high[i - 1])
                        if i > 1:
                            psar = cp.maximum(psar, high[i - 2])
                        if low[i] < ep:
                            ep = low[i]
                            af = cp.minimum(af + acc, acc_max)
                        if high[i] > psar:
                            long = True
                            psar = ep
                            ep = high[i]
                            af = acc
                    sar[i] = psar
                return cp.asnumpy(sar)
            if ind == "VWAP":
                if vol is None:
                    return None
                typical = (high + low + close) / 3.0
                cum_vol = cp.cumsum(vol)
                cum_tp_vol = cp.cumsum(typical * vol)
                vwap = cp.where(cum_vol > 1e-6, cum_tp_vol / cum_vol, typical)
                return cp.asnumpy(vwap)
            if ind == "TRANGE":
                prev_close = cp.concatenate([close[:1], close[:-1]])
                tr = cp.maximum(cp.maximum(high - low, cp.abs(high - prev_close)), cp.abs(low - prev_close))
                return cp.asnumpy(tr)
            if ind == "NATR":
                prev_close = cp.concatenate([close[:1], close[:-1]])
                tr = cp.maximum(cp.maximum(high - low, cp.abs(high - prev_close)), cp.abs(low - prev_close))
                atr = self._sma_gpu(tr, tp)
                natr = cp.where(close > 1e-9, (atr / close) * 100.0, 0.0)
                return cp.asnumpy(natr)
            if ind in ("STDDEV", "VAR"):
                mean = self._sma_gpu(close, tp)
                mean_sq = self._sma_gpu(close * close, tp)
                var = cp.maximum(mean_sq - mean * mean, 0.0)
                return cp.asnumpy(cp.sqrt(var) if ind == "STDDEV" else var)
            if ind == "BBANDS":
                nbup = float(params.get("nbdevup", 2.0))
                mean = self._sma_gpu(close, tp)
                mean_sq = self._sma_gpu(close * close, tp)
                std = cp.sqrt(cp.maximum(mean_sq - mean * mean, 0.0))
                upper = mean + nbup * std
                # Match Abstract API first output (upper band)
                return cp.asnumpy(upper)
            if ind == "ADX":
                plus_dm = cp.zeros_like(close)
                minus_dm = cp.zeros_like(close)
                plus_dm[1:] = cp.maximum(high[1:] - high[:-1], 0)
                minus_dm[1:] = cp.maximum(low[:-1] - low[1:], 0)
                prev_close = cp.concatenate([close[:1], close[:-1]])
                tr = cp.maximum(cp.maximum(high - low, cp.abs(high - prev_close)), cp.abs(low - prev_close))
                smooth_plus = self._sma_gpu(plus_dm, tp)
                smooth_minus = self._sma_gpu(minus_dm, tp)
                smooth_tr = self._sma_gpu(tr, tp)
                plus_di = cp.where(smooth_tr > 1e-9, 100 * smooth_plus / smooth_tr, 0.0)
                minus_di = cp.where(smooth_tr > 1e-9, 100 * smooth_minus / smooth_tr, 0.0)
                dx = cp.where((plus_di + minus_di) > 1e-9, 100 * cp.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
                adx = self._sma_gpu(dx, tp)
                return cp.asnumpy(adx)
            if ind == "DX":
                plus_dm = cp.zeros_like(close)
                minus_dm = cp.zeros_like(close)
                plus_dm[1:] = cp.maximum(high[1:] - high[:-1], 0)
                minus_dm[1:] = cp.maximum(low[:-1] - low[1:], 0)
                prev_close = cp.concatenate([close[:1], close[:-1]])
                tr = cp.maximum(cp.maximum(high - low, cp.abs(high - prev_close)), cp.abs(low - prev_close))
                smooth_plus = self._sma_gpu(plus_dm, tp)
                smooth_minus = self._sma_gpu(minus_dm, tp)
                smooth_tr = self._sma_gpu(tr, tp)
                plus_di = cp.where(smooth_tr > 1e-9, 100 * smooth_plus / smooth_tr, 0.0)
                minus_di = cp.where(smooth_tr > 1e-9, 100 * smooth_minus / smooth_tr, 0.0)
                dx = cp.where((plus_di + minus_di) > 1e-9, 100 * cp.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
                return cp.asnumpy(dx)
            if ind == "ADXR":
                # reuse DX then average ADX with lag
                # calculate ADX same as above
                plus_dm = cp.zeros_like(close)
                minus_dm = cp.zeros_like(close)
                plus_dm[1:] = cp.maximum(high[1:] - high[:-1], 0)
                minus_dm[1:] = cp.maximum(low[:-1] - low[1:], 0)
                prev_close = cp.concatenate([close[:1], close[:-1]])
                tr = cp.maximum(cp.maximum(high - low, cp.abs(high - prev_close)), cp.abs(low - prev_close))
                smooth_plus = self._sma_gpu(plus_dm, tp)
                smooth_minus = self._sma_gpu(minus_dm, tp)
                smooth_tr = self._sma_gpu(tr, tp)
                plus_di = cp.where(smooth_tr > 1e-9, 100 * smooth_plus / smooth_tr, 0.0)
                minus_di = cp.where(smooth_tr > 1e-9, 100 * smooth_minus / smooth_tr, 0.0)
                dx = cp.where((plus_di + minus_di) > 1e-9, 100 * cp.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)
                adx = self._sma_gpu(dx, tp)
                adxr = (adx + cp.roll(adx, tp)) / 2.0
                return cp.asnumpy(adxr)
            if ind in ("PLUS_DI", "MINUS_DI", "PLUS_DM", "MINUS_DM"):
                plus_dm = cp.zeros_like(close)
                minus_dm = cp.zeros_like(close)
                plus_dm[1:] = cp.maximum(high[1:] - high[:-1], 0)
                minus_dm[1:] = cp.maximum(low[:-1] - low[1:], 0)
                prev_close = cp.concatenate([close[:1], close[:-1]])
                tr = cp.maximum(cp.maximum(high - low, cp.abs(high - prev_close)), cp.abs(low - prev_close))
                smooth_tr = self._sma_gpu(tr, tp)
                if ind == "PLUS_DM":
                    return cp.asnumpy(plus_dm)
                if ind == "MINUS_DM":
                    return cp.asnumpy(minus_dm)
                if ind == "PLUS_DI":
                    plus_di = cp.where(smooth_tr > 1e-9, 100 * self._sma_gpu(plus_dm, tp) / smooth_tr, 0.0)
                    return cp.asnumpy(plus_di)
                minus_di = cp.where(smooth_tr > 1e-9, 100 * self._sma_gpu(minus_dm, tp) / smooth_tr, 0.0)
                return cp.asnumpy(minus_di)
            if ind in ("AROON", "AROONOSC"):
                period = tp
                n = close.size
                aroon_up = cp.zeros(n, dtype=cp.float32)
                aroon_down = cp.zeros(n, dtype=cp.float32)
                for i in range(period - 1, n):
                    window_high = high[i - period + 1 : i + 1]
                    window_low = low[i - period + 1 : i + 1]
                    days_since_high = period - 1 - cp.argmax(window_high)
                    days_since_low = period - 1 - cp.argmin(window_low)
                    aroon_up[i] = 100.0 * (period - days_since_high) / period
                    aroon_down[i] = 100.0 * (period - days_since_low) / period
                if ind == "AROON":
                    return cp.asnumpy(cp.stack([aroon_down, aroon_up], axis=1))
                return cp.asnumpy(aroon_up - aroon_down)
            if ind == "MFI":
                if vol is None:
                    return None
                typical_price = (high + low + close) / 3.0
                mf = typical_price * vol
                tp_delta = cp.diff(cp.pad(typical_price, (1, 0), mode="edge"))
                pos_flow = cp.where(tp_delta > 0, mf, 0.0)
                neg_flow = cp.where(tp_delta < 0, mf, 0.0)
                pos_sum = self._sma_gpu(pos_flow, tp) * tp
                neg_sum = self._sma_gpu(neg_flow, tp) * tp
                mfi = cp.where((pos_sum + neg_sum) > 1e-9, 100 * pos_sum / (pos_sum + neg_sum), 50.0)
                return cp.asnumpy(mfi)
            if ind in ("STOCH", "STOCHF"):
                k_period = int(params.get("fastk_period", params.get("k_period", 14)))
                d_period = int(params.get("slowd_period", params.get("d_period", 3)))
                lowest = self._rolling_minmax_gpu(low, k_period, "min")
                highest = self._rolling_minmax_gpu(high, k_period, "max")
                stoch_k = cp.where((highest - lowest) > 1e-9, 100 * (close - lowest) / (highest - lowest), 50.0)
                stoch_d = self._sma_gpu(stoch_k, d_period)
                return cp.asnumpy(stoch_d if ind == "STOCHF" else stoch_k)
            if ind == "STOCHRSI":
                rsi_vals = self._calc_indicator_gpu("RSI", {"timeperiod": tp}, df, preloaded=preloaded)
                if rsi_vals is None:
                    return None
                rsi_cp = cp.asarray(rsi_vals)
                lowest = self._rolling_minmax_gpu(rsi_cp, tp, "min")
                highest = self._rolling_minmax_gpu(rsi_cp, tp, "max")
                stoch_rsi = cp.where((highest - lowest) > 1e-9, (rsi_cp - lowest) / (highest - lowest), 0.5)
                return cp.asnumpy(stoch_rsi * 100.0)
            if ind == "CCI":
                typical = (high + low + close) / 3.0
                sma = self._sma_gpu(typical, tp)
                md = self._sma_gpu(cp.abs(typical - sma), tp)
                cci = cp.where(md > 1e-9, (typical - sma) / (0.015 * md), 0.0)
                return cp.asnumpy(cci)
            if ind == "WILLR":
                highest = self._rolling_minmax_gpu(high, tp, "max")
                lowest = self._rolling_minmax_gpu(low, tp, "min")
                out = cp.where((highest - lowest) > 1e-9, (highest - close) / (highest - lowest) * -100.0, 0.0)
                return cp.asnumpy(out)
            if ind == "MACD":
                fast = int(params.get("fastperiod", 12))
                slow = int(params.get("slowperiod", 26))
                signal = int(params.get("signalperiod", 9))
                ema_fast = self._ema_gpu(close, fast)
                ema_slow = self._ema_gpu(close, slow)
                macd = ema_fast - ema_slow
                macd_signal = self._ema_gpu(macd, signal)
                macd_hist = macd - macd_signal
                return cp.asnumpy(macd_hist)
            if ind == "TRIX":
                ema1 = self._ema_gpu(close, tp)
                ema2 = self._ema_gpu(ema1, tp)
                ema3 = self._ema_gpu(ema2, tp)
                shifted = cp.roll(ema3, 1)
                trix = cp.where(cp.abs(shifted) > 1e-9, (ema3 - shifted) / shifted * 100.0, 0.0)
                trix[: tp * 3] = 0.0
                return cp.asnumpy(trix)
            if ind == "CMO":
                diff = cp.diff(cp.pad(close, (1, 0), mode="edge"))
                pos = cp.maximum(diff, 0.0)
                neg = cp.maximum(-diff, 0.0)
                sum_pos = self._sma_gpu(pos, tp) * tp
                sum_neg = self._sma_gpu(neg, tp) * tp
                cmo = cp.where((sum_pos + sum_neg) > 1e-9, 100 * (sum_pos - sum_neg) / (sum_pos + sum_neg), 0.0)
                return cp.asnumpy(cmo)
            if ind == "ULTOSC":
                # periods fixed 7,14,28
                bp = close - cp.minimum(low, cp.concatenate([close[:1], close[:-1]]))
                tr = cp.maximum(
                    cp.maximum(high - low, cp.abs(high - cp.concatenate([close[:1], close[:-1]]))),
                    cp.abs(low - cp.concatenate([close[:1], close[:-1]])),
                )

                def avg(arr, period):
                    return self._sma_gpu(arr, period) * period

                avg7 = avg(bp, 7)
                tr7 = avg(tr, 7)
                avg14 = avg(bp, 14)
                tr14 = avg(tr, 14)
                avg28 = avg(bp, 28)
                tr28 = avg(tr, 28)
                ult = (4 * avg7 / tr7 + 2 * avg14 / tr14 + avg28 / tr28) / 7 * 100
                return cp.asnumpy(cp.nan_to_num(ult, nan=50.0))
            # Hilbert Transform indicators (ported from TA-Lib ht_trendline/ht_dc* reference)
            if ind in ("HT_TRENDLINE", "HT_TRENDMODE", "HT_DCPERIOD", "HT_DCPHASE"):
                n = close.size
                if n < 64:
                    return None
                detrender = cp.zeros(n, dtype=cp.float32)
                Q1 = cp.zeros(n, dtype=cp.float32)
                I1 = cp.zeros(n, dtype=cp.float32)
                jI = cp.zeros(n, dtype=cp.float32)
                jQ = cp.zeros(n, dtype=cp.float32)
                I2 = cp.zeros(n, dtype=cp.float32)
                Q2 = cp.zeros(n, dtype=cp.float32)
                Re = cp.zeros(n, dtype=cp.float32)
                Im = cp.zeros(n, dtype=cp.float32)
                period = cp.zeros(n, dtype=cp.float32)
                smooth = cp.zeros(n, dtype=cp.float32)

                # smooth price
                smooth[:] = close
                # constants from TA-Lib
                rad2deg = 180.0 / cp.pi
                deg2rad = cp.pi / 180.0
                const = cp.array([0.0962, 0.5769], dtype=cp.float32)

                # seed
                for i in range(6, n):
                    smooth[i] = (4 * close[i] + 3 * close[i - 1] + 2 * close[i - 2] + close[i - 3]) / 10.0

                for i in range(6, n):
                    detrender[i] = (
                        const[0] * smooth[i]
                        + const[1] * smooth[i - 2]
                        - const[1] * smooth[i - 4]
                        - const[0] * smooth[i - 6]
                    )
                # quadrature and in-phase
                for i in range(0, n):
                    if i >= 3:
                        Q1[i] = (
                            const[0] * detrender[i]
                            + const[1] * detrender[i - 2]
                            - const[1] * detrender[i - 4]
                            - const[0] * detrender[i - 6]
                        )
                        I1[i] = detrender[i - 3]
                # advance the phase of I1 by 90 degrees
                for i in range(n):
                    if i >= 2:
                        jI[i] = const[0] * I1[i] + const[1] * I1[i - 2] - const[1] * I1[i - 4] - const[0] * I1[i - 6]
                        jQ[i] = const[0] * Q1[i] + const[1] * Q1[i - 2] - const[1] * Q1[i - 4] - const[0] * Q1[i - 6]
                # Homodyne discriminator
                for i in range(n):
                    I2[i] = I1[i] - jQ[i]
                    Q2[i] = Q1[i] + jI[i]
                for i in range(n):
                    Re[i] = I2[i] * I2[i - 1] + Q2[i] * Q2[i - 1] if i > 0 else 0.0
                    Im[i] = I2[i] * Q2[i - 1] - Q2[i] * I2[i - 1] if i > 0 else 0.0
                for i in range(n):
                    if Re[i] != 0.0:
                        period[i] = 360.0 / (rad2deg * cp.arctan(Im[i] / Re[i]))
                    else:
                        period[i] = period[i - 1] if i > 0 else 0.0
                    period[i] = cp.clip(period[i], 6.0, 50.0)
                    if i > 0:
                        period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
                # dominant cycle period
                dcperiod = period
                # phase
                dcphase = cp.zeros(n, dtype=cp.float32)
                for i in range(n):
                    dcphase[i] = dcphase[i - 1] + 360.0 / period[i] if i > 0 and period[i] > 0 else 0.0
                    if dcphase[i] > 360.0:
                        dcphase[i] -= 360.0
                # trendline (inst trend)
                trendline = cp.zeros(n, dtype=cp.float32)
                for i in range(n):
                    trendline[i] = 0.0
                    if i >= 3:
                        trendline[i] = (4 * close[i] + 3 * close[i - 1] + 2 * close[i - 2] + close[i - 3]) / 10.0

                if ind == "HT_TRENDLINE":
                    return cp.asnumpy(trendline)
                if ind == "HT_DCPERIOD":
                    return cp.asnumpy(dcperiod)
                if ind == "HT_DCPHASE":
                    return cp.asnumpy(dcphase)
                # trendmode: 1 trending, 0 cycling heuristic
                trendmode = cp.where(cp.abs(trendline - close) < 0.01 * close, 1.0, 0.0)
                return cp.asnumpy(trendmode)
            # -------- Candlestick patterns (vectorized approximations) -------- #
            if ind.startswith("CDL_"):
                body = cp.abs(close - open_)
                range_ = cp.maximum(high - low, 1e-9)
                upper = high - cp.maximum(close, open_)
                lower = cp.minimum(close, open_) - low

                if ind == "CDL_DOJI":
                    out = cp.where(body <= 0.1 * range_, 100, 0)
                    return cp.asnumpy(out.astype(cp.int16))

                if ind == "CDL_HAMMER":
                    cond = (lower >= 2.0 * body) & (upper <= 0.1 * body)
                    sign = cp.where(close >= open_, 100, -100)
                    out = cp.where(cond, sign, 0)
                    return cp.asnumpy(out.astype(cp.int16))

                if ind == "CDL_SHOOTINGSTAR":
                    cond = (upper >= 2.0 * body) & (lower <= 0.1 * body)
                    sign = cp.where(close <= open_, -100, 100)
                    out = cp.where(cond, sign, 0)
                    return cp.asnumpy(out.astype(cp.int16))

                if ind == "CDL_MARUBOZU":
                    cond = (upper <= 0.02 * range_) & (lower <= 0.02 * range_) & (body >= 0.9 * range_)
                    sign = cp.where(close >= open_, 100, -100)
                    out = cp.where(cond, sign, 0)
                    return cp.asnumpy(out.astype(cp.int16))

                if ind == "CDL_SPINNINGTOP":
                    cond = (body <= 0.25 * range_) & (upper >= 0.25 * range_) & (lower >= 0.25 * range_)
                    sign = cp.where(close >= open_, 100, -100)
                    out = cp.where(cond, sign, 0)
                    return cp.asnumpy(out.astype(cp.int16))

                if ind == "CDL_ENGULFING":
                    # Need previous candle; shift by one
                    prev_open = cp.concatenate([open_[:1], open_[:-1]])
                    prev_close = cp.concatenate([close[:1], close[:-1]])
                    prev_body = prev_close - prev_open
                    curr_body = close - open_
                    bull = (curr_body > 0) & (prev_body < 0) & (curr_body > cp.abs(prev_body))
                    bear = (curr_body < 0) & (prev_body > 0) & (cp.abs(curr_body) > prev_body)
                    out = cp.zeros_like(close, dtype=cp.int16)
                    out = cp.where(bull, 100, out)
                    out = cp.where(bear, -100, out)
                    return cp.asnumpy(out)

                # Unknown candlestick -> fall back
                return None
            if ind == "OBV":
                if vol is None:
                    return None
                direction = cp.sign(cp.diff(cp.pad(close, (1, 0), mode="edge")))
                obv = cp.cumsum(direction * vol)
                return cp.asnumpy(obv)
            if ind == "AD":
                if vol is None:
                    return None
                mf_mult = cp.where((high - low) > 1e-9, ((close - low) - (high - close)) / (high - low), 0.0)
                ad = cp.cumsum(mf_mult * vol)
                return cp.asnumpy(ad)
            if ind == "ADOSC":
                if vol is None:
                    return None
                fast = int(params.get("fastperiod", 3))
                slow = int(params.get("slowperiod", 10))
                mf_mult = cp.where((high - low) > 1e-9, ((close - low) - (high - close)) / (high - low), 0.0)
                ad = cp.cumsum(mf_mult * vol)
                ad_fast = self._ema_gpu(ad, fast)
                ad_slow = self._ema_gpu(ad, slow)
                return cp.asnumpy(ad_fast - ad_slow)
            if ind == "AVGPRICE":
                return cp.asnumpy((open_ + high + low + close) / 4.0)
            if ind == "MEDPRICE":
                return cp.asnumpy((high + low) / 2.0)
            if ind == "TYPPRICE":
                return cp.asnumpy((high + low + close) / 3.0)
            if ind == "WCLPRICE":
                return cp.asnumpy((high + low + close * 2.0) / 4.0)
            if ind == "BOP":
                denom = cp.maximum(high - low, 1e-9)
                bop = (close - open_) / denom
                return cp.asnumpy(bop)
        except Exception as e:  # pragma: no cover - GPU optional
            logger.debug(f"GPU indicator {ind} failed, fallback to CPU: {e}")
            return None

        out = locals().get("out", None)
        if out is None:
            return None

        # Parity check against TA-Lib; blacklist indicator if mismatch beyond tolerance
        if self.parity_check and TALIB_AVAILABLE:
            try:
                func = abstract.Function(ind)
                cpu_res = func(df, **params)
                if isinstance(cpu_res, pd.DataFrame):
                    cpu_vals = cpu_res.iloc[:, 0].values
                elif isinstance(cpu_res, pd.Series):
                    cpu_vals = cpu_res.values
                else:
                    cpu_vals = np.asarray(cpu_res)
                cpu_vals = np.nan_to_num(cpu_vals, nan=0.0).astype(np.float32)
                gpu_vals = np.asarray(out, dtype=np.float32)
                if gpu_vals.shape != cpu_vals.shape:
                    raise ValueError("shape mismatch")
                diff = float(np.max(np.abs(gpu_vals - cpu_vals)))
                if diff > self.parity_tol:
                    self._gpu_blacklist.add(ind)
                    logger.debug(f"GPU parity failed for {ind}: diff={diff:.4e}, blacklisting GPU path")
                    return None
            except Exception as e:
                self._gpu_blacklist.add(ind)
                logger.debug(f"Parity check failed for {ind}, blacklisting GPU: {e}")
                return None

        return out

    def _randomize_indicator_params(self, indicator: str, default_params: dict[str, Any]) -> dict[str, Any]:
        """
        Auto-tune indicator parameters by exploring optimal ranges.

        Common parameter ranges based on market research:
        - timeperiod: 5-200 (most indicators)
        - fastperiod: 5-50
        - slowperiod: 20-200
        - signalperiod: 3-20
        - nbdevup/nbdevdn: 1.5-3.0 (Bollinger Bands)
        """
        tuned = {}

        for param_name, default_value in default_params.items():
            if "period" in param_name.lower():
                if "fast" in param_name.lower():
                    tuned[param_name] = random.randint(5, 50)
                elif "slow" in param_name.lower():
                    tuned[param_name] = random.randint(20, 200)
                elif "signal" in param_name.lower():
                    tuned[param_name] = random.randint(3, 20)
                else:
                    tuned[param_name] = random.choice([5, 7, 10, 14, 20, 21, 30, 50, 100, 200])

            elif "dev" in param_name.lower() or "nbdev" in param_name.lower():
                tuned[param_name] = round(random.uniform(1.5, 3.0), 1)

            elif "accel" in param_name.lower():
                tuned[param_name] = round(random.uniform(0.01, 0.05), 2)

            elif "max" in param_name.lower():
                tuned[param_name] = round(random.uniform(0.1, 0.3), 2)

            elif "matype" in param_name.lower():
                tuned[param_name] = random.randint(0, 8)

            else:
                tuned[param_name] = default_value

        return tuned

    def generate_random_strategy(self, regime: str = "any", max_indicators: int = 20) -> TALibStrategyGene:
        if not self.available_indicators:
            return TALibStrategyGene(indicators=[], params={}, strategy_id="empty")

        num_inds = random.randint(1, min(max_indicators, len(self.available_indicators)))
        inds = random.sample(self.available_indicators, k=num_inds)
        params = {}
        weights = {}
        for ind in inds:
            try:
                info = abstract.Function(ind).info
                default_params = info.get("parameters", {})
                params[ind] = self._randomize_indicator_params(ind, default_params)
            except Exception:
                params[ind] = {}
            weights[ind] = random.random()
        gene = TALibStrategyGene(
            indicators=inds,
            params=params,
            combination_method=random.choice(["weighted_vote", "threshold"]),
            long_threshold=random.uniform(0.4, 0.7),
            short_threshold=-random.uniform(0.4, 0.7),
            weights=weights,
            preferred_regime=regime,
            strategy_id=f"gene_{random.randint(0, 1_000_000)}",
            use_ob=random.choice([True, False]),
            use_fvg=random.choice([True, False]),
            use_liq_sweep=random.choice([True, False]),
            mtf_confirmation=random.choice([True, False]),
            use_premium_discount=random.choice([True, False]),
            use_inducement=random.choice([True, False]),
            tp_pips=random.uniform(10.0, 100.0),  # Randomize TP
            sl_pips=random.uniform(5.0, 50.0),  # Randomize SL
        )
        return gene

    def bulk_calculate_indicators(self, df: pd.DataFrame, population: list[TALibStrategyGene]) -> dict[str, np.ndarray]:
        """
        Pre-calculate all unique indicators required by the population in memory-safe chunks.
        Returns a cache dict: {"Indicator_ParamsHash": np.array}
        """
        if not TALIB_AVAILABLE or df.empty:
            return {}

        # 1. Identify unique indicator configurations
        unique_tasks = {}
        for gene in population:
            for ind in gene.indicators:
                if (not self.use_volume_features) and ind in self._volume_indicators:
                    continue
                params = gene.params.get(ind, {})
                # Create a stable key for caching
                param_key = json.dumps(params, sort_keys=True)
                task_key = f"{ind}_{param_key}"
                if task_key not in unique_tasks:
                    unique_tasks[task_key] = (ind, params)

        task_list = list(unique_tasks.items())
        total_tasks = len(task_list)
        logger.info(f"Bulk calculating {total_tasks} unique indicators for {len(population)} genes...")

        cache = {}

        # Memory Safety: Calculate in chunks to avoid spiking RAM with thousands of result arrays at once
        # before they can be processed or if we were parallelizing too aggressively.
        chunk_size = 100

        # Helper for calculation (prefers GPU when available & supported)
        def _calc_safe(key, ind, p, dataframe_ref):
            if self.use_gpu:
                gpu_vals = self._calc_indicator_gpu(ind, p, dataframe_ref)
                if gpu_vals is not None:
                    return key, gpu_vals.astype(np.float32, copy=False)
            try:
                func = abstract.Function(ind)
                # TA-Lib abstract API works on dict of Series or DataFrame
                res = func(dataframe_ref, **p)
                if isinstance(res, (pd.Series, pd.DataFrame)):
                    vals = res.values if isinstance(res, pd.Series) else res.iloc[:, 0].values
                else:
                    vals = res
                # Downcast to float32 immediately to save RAM
                return key, np.nan_to_num(vals, nan=0.0).astype(np.float32)
            except Exception:
                return key, np.zeros(len(dataframe_ref), dtype=np.float32)

        import concurrent.futures

        import psutil

        # Dynamic worker limit based on RAM
        mem = psutil.virtual_memory()
        # Conservative: leave 20% headroom. Each indicator is roughly float32 * rows.
        # 50k rows * 4 bytes = 200KB (negligible).
        # 10M rows * 4 bytes = 40MB.
        # Parallelism is fine for computation speed, bottleneck is usually GIL for light ops
        # or pure CPU for heavy ops. Threading works for TA-Lib (C-ext).

        env_workers = os.environ.get("FOREX_BOT_TALIB_WORKERS") or os.environ.get(
            "FOREX_BOT_CPU_THREADS"
        )
        if env_workers:
            try:
                max_workers = max(1, int(env_workers))
            except Exception:
                max_workers = min(32, (os.cpu_count() or 1) * 2)
        else:
            try:
                cpu_budget = int(os.environ.get("FOREX_BOT_CPU_BUDGET", "0") or 0)
            except Exception:
                cpu_budget = 0
            if cpu_budget > 0:
                max_workers = max(1, min(cpu_budget, os.cpu_count() or 1))
            else:
                max_workers = min(32, (os.cpu_count() or 1) * 2)
        if mem.percent > 85:
            max_workers = max(1, max_workers // 2)
            logger.warning(f"High RAM ({mem.percent}%), throttling indicator calc to {max_workers} workers.")

        # If GPU batch mode enabled, try to pre-pull numpy columns to cupy once
        gpu_arrays = None
        if self.use_gpu and self.gpu_batch:
            try:
                gpu_arrays = {
                    "close": cp.asarray(df["close"].to_numpy(dtype=np.float32)),
                    "high": cp.asarray(df["high"].to_numpy(dtype=np.float32)) if "high" in df else None,
                    "low": cp.asarray(df["low"].to_numpy(dtype=np.float32)) if "low" in df else None,
                    "open": cp.asarray(df["open"].to_numpy(dtype=np.float32)) if "open" in df else None,
                    "volume": cp.asarray(df["volume"].to_numpy(dtype=np.float32))
                    if self.use_volume_features and "volume" in df
                    else None,
                }
            except Exception:
                gpu_arrays = None

        # Process in chunks
        for i in range(0, total_tasks, chunk_size):
            chunk = task_list[i : i + chunk_size]

            # Check RAM before dispatching next chunk
            if psutil.virtual_memory().percent > 90:
                logger.warning("Critical RAM usage (>90%). Switching to sequential processing for safety.")
                max_workers = 1

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for k, (idx, p) in chunk:
                    if self.use_gpu and gpu_arrays is not None:
                        futures.append(executor.submit(self._calc_indicator_gpu, idx, p, df, gpu_arrays))
                    else:
                        futures.append(executor.submit(_calc_safe, k, idx, p, df))

                for fut, item in zip(concurrent.futures.as_completed(futures), chunk, strict=False):
                    k = item[0]
                    try:
                        res = fut.result()
                        if isinstance(res, tuple) and len(res) == 2:
                            _, vals = res
                        else:
                            vals = res
                        if vals is None:
                            _, vals = _calc_safe(k, item[1][0], item[1][1], df)
                        cache[k] = np.asarray(vals, dtype=np.float32)
                    except Exception:
                        cache[k] = np.zeros(len(df), dtype=np.float32)

            if self.use_gpu and cp is not None:
                try:
                    # Prevent CuPy memory pool fragmentation during long-running genetic searches.
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass

            # Optional: Aggressive GC if needed, though usually automatic

        return cache

    def compute_signals(
        self, df: pd.DataFrame, strategy: TALibStrategyGene, cache: dict[str, np.ndarray] | None = None
    ) -> pd.Series:
        if not TALIB_AVAILABLE or df.empty:
            return pd.Series(0, index=df.index)

        n_samples = len(df)
        n_indicators = len(strategy.indicators)
        if n_indicators == 0:
            return pd.Series(0, index=df.index)

        signal_matrix = np.zeros((n_samples, n_indicators), dtype=np.float32)

        for i, ind in enumerate(strategy.indicators):
            params = strategy.params.get(ind, {})

            # Try cache first
            vals = None
            if cache:
                param_key = json.dumps(params, sort_keys=True)
                task_key = f"{ind}_{param_key}"
                vals = cache.get(task_key)

            # Fallback to on-demand calc
            if vals is None:
                # GPU first if enabled
                if self.use_gpu:
                    vals = self._calc_indicator_gpu(ind, params, df)
                if vals is None:
                    try:
                        func = abstract.Function(ind)
                        res = func(df, **params)
                        if isinstance(res, (pd.Series, pd.DataFrame)):
                            vals = res.values if isinstance(res, pd.Series) else res.iloc[:, 0].values
                        else:
                            vals = res
                        vals = np.nan_to_num(vals, nan=0.0)
                    except Exception:
                        vals = np.zeros(n_samples)

            signal_matrix[:, i] = vals

        weights = strategy.weights or {}
        weight_vec = np.array([weights.get(ind, 1.0) for ind in strategy.indicators], dtype=np.float32)
        weight_sum = np.sum(weight_vec) + 1e-9
        norm_weights = weight_vec / weight_sum

        if _combine_signals_numba is not None:
            final_signal = _combine_signals_numba(
                signal_matrix.astype(np.float32, copy=False),
                norm_weights.astype(np.float32, copy=False),
                float(strategy.long_threshold),
                float(strategy.short_threshold),
            )
        else:
            combined = signal_matrix @ norm_weights
            final_signal = np.zeros(n_samples, dtype=np.int8)
            final_signal[combined >= strategy.long_threshold] = 1
            final_signal[combined <= strategy.short_threshold] = -1

        # ... (SMC Logic remains the same)
        cols = set(df.columns)

        if strategy.use_ob and "order_block" in cols:
            ob = df["order_block"].values
            mask_long = (final_signal == 1) & (ob != 1)
            mask_short = (final_signal == -1) & (ob != -1)
            final_signal[mask_long] = 0
            final_signal[mask_short] = 0

        if strategy.use_fvg and "fvg_direction" in cols:
            fvg = df["fvg_direction"].values
            mask_long = (final_signal == 1) & (fvg != 1)
            mask_short = (final_signal == -1) & (fvg != -1)
            final_signal[mask_long] = 0
            final_signal[mask_short] = 0

        if strategy.use_liq_sweep and "liquidity_sweep" in cols:
            sweep = df["liquidity_sweep"].values
            mask_long = (final_signal == 1) & (sweep != 1)
            mask_short = (final_signal == -1) & (sweep != -1)
            final_signal[mask_long] = 0
            final_signal[mask_short] = 0

        if strategy.mtf_confirmation and "H1_trend" in cols:
            htf = df["H1_trend"].values
            mask_long = (final_signal == 1) & (htf != 1)
            mask_short = (final_signal == -1) & (htf != -1)
            final_signal[mask_long] = 0
            final_signal[mask_short] = 0

        if strategy.use_premium_discount and "premium_discount" in cols:
            pd_zone = df["premium_discount"].values
            mask_long = (final_signal == 1) & (pd_zone != 1)
            mask_short = (final_signal == -1) & (pd_zone != -1)
            final_signal[mask_long] = 0
            final_signal[mask_short] = 0

        if strategy.use_inducement and "inducement" in cols:
            ind = df["inducement"].values
            mask_entry = (final_signal != 0) & (ind != 1)
            final_signal[mask_entry] = 0

        return pd.Series(final_signal, index=df.index)


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _combine_signals_numba(
        signal_matrix: np.ndarray, weights: np.ndarray, long_th: float, short_th: float
    ) -> np.ndarray:
        n_samples, n_indicators = signal_matrix.shape
        out = np.zeros(n_samples, dtype=np.int8)
        for i in range(n_samples):
            acc = 0.0
            for j in range(n_indicators):
                acc += signal_matrix[i, j] * weights[j]
            if acc >= long_th:
                out[i] = 1
            elif acc <= short_th:
                out[i] = -1
        return out

else:
    _combine_signals_numba = None

    def save_knowledge(self, path: Path) -> None:
        data = {
            "synergy_matrix": {f"{k[0]}_{k[1]}": v for k, v in self.indicator_synergy_matrix.items()},
            "regime_performance": {reg: dict(perf) for reg, perf in self.regime_performance.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load_knowledge(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text())
        for key, value in data.get("synergy_matrix", {}).items():
            ind1, ind2 = key.split("_")
            self.indicator_synergy_matrix[tuple(sorted([ind1, ind2]))] = value
        for regime, perf in data.get("regime_performance", {}).items():
            self.regime_performance[regime] = defaultdict(float, perf)

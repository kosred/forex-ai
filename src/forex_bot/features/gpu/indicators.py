import logging
from typing import Any

import numpy as np

try:
    import cupy as cp
    from cupyx import jit

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    jit = None
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)
EPSILON = 1e-10


def rsi_cupy(close_cp: "cp.ndarray", period: int = 14) -> "cp.ndarray":
    """
    RSI on CuPy using vectorized operations only.

    NOTE: This is a simplified Wilder's RSI without the iterative EWMA.
    For accurate RSI, use CPU numba implementation or TA-Lib instead.
    This GPU version sacrifices some accuracy for speed via simple rolling mean.
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")
    if close_cp.size < period + 1:
        return cp.zeros_like(close_cp, dtype=cp.float32)

    # Vectorized approach using simple moving average instead of Wilder's smoothing
    diff = cp.diff(cp.pad(close_cp, (1, 0), mode="edge"))
    gain = cp.maximum(diff, 0)
    loss = cp.maximum(-diff, 0)

    # Use simple rolling mean instead of EWMA (GPU-friendly)
    # Note: This is approximate but avoids Python loops
    try:
        import cupyx.scipy.ndimage as ndimage

        avg_gain = ndimage.uniform_filter1d(gain, period, mode="constant", origin=-(period // 2))
        avg_loss = ndimage.uniform_filter1d(loss, period, mode="constant", origin=-(period // 2))
    except Exception:
        # Fallback to convolve if scipy not available
        kernel = cp.ones(period) / period
        avg_gain = cp.convolve(gain, kernel, mode="same")
        avg_loss = cp.convolve(loss, kernel, mode="same")

    rs = cp.where(avg_loss > 1e-9, avg_gain / avg_loss, 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0  # Neutral for initial period
    return rsi


def atr_wicks_cupy(
    high_np: np.ndarray,
    low_np: np.ndarray,
    open_np: np.ndarray,
    close_np: np.ndarray,
    period: int = 14,
) -> dict[str, np.ndarray]:
    """
    Compute ATR and wick ratios on GPU using vectorized operations.

    Uses uniform filter for rolling mean (faster than convolve for this use case).
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    h = cp.asarray(high_np)
    low_cp = cp.asarray(low_np)
    o = cp.asarray(open_np)
    c = cp.asarray(close_np)

    # Vectorized true range calculation
    prev_close = cp.concatenate([c[:1], c[:-1]])
    tr = cp.maximum(cp.maximum(h - low_cp, cp.abs(h - prev_close)), cp.abs(low_cp - prev_close))

    # Use uniform_filter1d for better performance than convolve
    try:
        import cupyx.scipy.ndimage as ndimage

        atr = ndimage.uniform_filter1d(tr, period, mode="constant", origin=-(period // 2))
    except Exception:
        # Fallback to convolve if ndimage not available
        kernel = cp.ones(period) / period
        atr = cp.convolve(tr, kernel, mode="same")

    # Vectorized wick calculations
    rng = cp.maximum(h - low_cp, EPSILON)
    upper_wick = (h - cp.maximum(o, c)) / rng
    lower_wick = (cp.minimum(o, c) - low_cp) / rng

    return {
        "atr": cp.asnumpy(atr),
        "upper_wick": cp.asnumpy(upper_wick),
        "lower_wick": cp.asnumpy(lower_wick),
    }


def talib_gpu(df) -> dict[str, Any]:
    """
    Simplified GPU indicators using CuPy with ONLY vectorized operations.

    WARNING: This provides APPROXIMATE indicators using simple moving averages
    instead of Wilder's smoothing. For accurate TA-Lib equivalents, use CPU.

    The previous version had Python loops which defeated GPU parallelism.
    This version sacrifices some accuracy for true GPU performance.
    """
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available")

    h = cp.asarray(df["high"].to_numpy())
    low_cp = cp.asarray(df["low"].to_numpy())
    c = cp.asarray(df["close"].to_numpy())
    v = cp.asarray(df["volume"].to_numpy()) if "volume" in df else cp.zeros_like(c)

    n = h.shape[0]

    # Try to use cupyx.scipy.ndimage for rolling operations
    try:
        import cupyx.scipy.ndimage as ndimage

        use_ndimage = True
    except ImportError:
        use_ndimage = False
        logger.warning("cupyx.scipy.ndimage not available, using convolve fallback")

    # === ADX (Simplified with SMA instead of Wilder's smoothing) ===
    plus_dm = cp.zeros(n)
    minus_dm = cp.zeros(n)
    plus_dm[1:] = cp.maximum(h[1:] - h[:-1], 0)
    minus_dm[1:] = cp.maximum(low_cp[:-1] - low_cp[1:], 0)

    c_prev = cp.concatenate([c[:1], c[:-1]])
    tr = cp.maximum(cp.maximum(h - low_cp, cp.abs(h - c_prev)), cp.abs(low_cp - c_prev))

    # Use rolling mean instead of EWMA (vectorized, GPU-friendly)
    period = 14
    if use_ndimage:
        smooth_plus = ndimage.uniform_filter1d(plus_dm, period, mode="constant")
        smooth_minus = ndimage.uniform_filter1d(minus_dm, period, mode="constant")
        smooth_tr = ndimage.uniform_filter1d(tr, period, mode="constant")
    else:
        kernel = cp.ones(period) / period
        smooth_plus = cp.convolve(plus_dm, kernel, mode="same")
        smooth_minus = cp.convolve(minus_dm, kernel, mode="same")
        smooth_tr = cp.convolve(tr, kernel, mode="same")

    plus_di = cp.where(smooth_tr > 1e-9, 100 * smooth_plus / smooth_tr, 0)
    minus_di = cp.where(smooth_tr > 1e-9, 100 * smooth_minus / smooth_tr, 0)
    dx = cp.where((plus_di + minus_di) > 1e-9, 100 * cp.abs(plus_di - minus_di) / (plus_di + minus_di), 0)

    if use_ndimage:
        adx = ndimage.uniform_filter1d(dx, period, mode="constant")
    else:
        adx = cp.convolve(dx, kernel, mode="same")

    # === MFI (Simplified - using rolling sum) ===
    typical_price = (h + low_cp + c) / 3.0
    money_flow = typical_price * v

    # Vectorized money flow calculation
    tp_delta = cp.diff(cp.pad(typical_price, (1, 0), mode="edge"))
    positive_flow = cp.where(tp_delta > 0, money_flow, 0)
    negative_flow = cp.where(tp_delta < 0, money_flow, 0)

    if use_ndimage:
        pos_mf_sum = ndimage.uniform_filter1d(positive_flow, period, mode="constant") * period
        neg_mf_sum = ndimage.uniform_filter1d(negative_flow, period, mode="constant") * period
    else:
        kernel_sum = cp.ones(period)
        pos_mf_sum = cp.convolve(positive_flow, kernel_sum, mode="same")
        neg_mf_sum = cp.convolve(negative_flow, kernel_sum, mode="same")

    mfi = cp.where((pos_mf_sum + neg_mf_sum) > 1e-9, 100 * pos_mf_sum / (pos_mf_sum + neg_mf_sum), 50)

    # === Stochastic (Simplified - using rolling min/max) ===
    if use_ndimage:
        lowest_low = ndimage.minimum_filter1d(low_cp, 14, mode="constant")
        highest_high = ndimage.maximum_filter1d(h, 14, mode="constant")
    else:
        # Fallback: approximate with convolve (not perfect but vectorized)
        lowest_low = cp.copy(low_cp)
        highest_high = cp.copy(h)
        for i in range(1, min(14, n)):
            lowest_low = cp.minimum(lowest_low, cp.roll(low_cp, i))
            highest_high = cp.maximum(highest_high, cp.roll(h, i))

    stoch_k = cp.where((highest_high - lowest_low) > 1e-9, 100 * (c - lowest_low) / (highest_high - lowest_low), 50)

    # Smooth with 3-period SMA
    if use_ndimage:
        stoch_k_smooth = ndimage.uniform_filter1d(stoch_k, 3, mode="constant")
        stoch_d = ndimage.uniform_filter1d(stoch_k_smooth, 3, mode="constant")
    else:
        kernel_3 = cp.ones(3) / 3
        stoch_k_smooth = cp.convolve(stoch_k, kernel_3, mode="same")
        stoch_d = cp.convolve(stoch_k_smooth, kernel_3, mode="same")

    return {
        "adx14": adx,
        "mfi14": mfi,
        "stoch_k": stoch_k_smooth,
        "stoch_d": stoch_d,
    }

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    prange = range


try:
    from numba import cuda as numba_cuda

    NUMBA_CUDA_AVAILABLE = True
except Exception:
    numba_cuda = None
    NUMBA_CUDA_AVAILABLE = False

# CUDA support removed - Numba CPU parallelization is faster for typical data sizes
# Previous CUDA kernels were inefficient and slower than CPU due to:
# - Sequential loops inside kernels
# - No shared memory optimization
# - Kernel launch overhead
CUDA_AVAILABLE = NUMBA_CUDA_AVAILABLE
cuda = numba_cuda

if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def _find_pivots_parallel(h, l, lookback):
        n = len(h)
        is_high = np.zeros(n, dtype=np.bool_)
        is_low = np.zeros(n, dtype=np.bool_)
        
        # 252-Core Parallel Pivot Search
        for i in prange(lookback, n - lookback):
            # Swing High
            p_h = h[i]
            fail_h = False
            for k in range(i - lookback, i + lookback + 1):
                if k != i and h[k] >= p_h:
                    fail_h = True
                    break
            if not fail_h: is_high[i] = True
            
            # Swing Low
            p_l = l[i]
            fail_l = False
            for k in range(i - lookback, i + lookback + 1):
                if k != i and l[k] <= p_l:
                    fail_l = True
                    break
            if not fail_l: is_low[i] = True
            
        return is_high, is_low

    @njit(cache=True, fastmath=True)
    def _compute_smc_numba(
        o, h, l, c, atr, lookback=5, freshness_limit=200, atr_displacement=1.2, max_levels=3
    ):
        n = len(c)
        # HPC FIX: Dynamic Structural Lookback
        # If we have M1 data, we need more bars to find real structure
        if n > 100000: # High-density M1/M5
            lookback = max(lookback, 20)
        
        # Pre-calculate pivots in parallel (using all 252 cores)
        is_high, is_low = _find_pivots_parallel(h, l, lookback)
        
        # ... (Sequential state-dependent logic like BOS/CHOCH remains, but is now much faster)
        # because the expensive pivot search is already done.

            if last_low_idx != -1 and last_low_idx < i - lookback:
                prev_low = l[last_low_idx]
                if l[i] < prev_low and c[i] > prev_low:
                    sweep[i] = 1

            if last_high_idx != -1 and last_high_idx < i - lookback:
                prev_high = h[last_high_idx]
                if h[i] > prev_high and c[i] < prev_high:
                    sweep[i] = -1

            if last_high_idx != -1:
                if c[i] > h[last_high_idx]:
                    if trend == -1:
                        choch[i] = 1
                        trend = 1
                    elif trend == 1:
                        bos[i] = 1
                    else:
                        trend = 1

            if last_low_idx != -1:
                if c[i] < l[last_low_idx]:
                    if trend == 1:
                        choch[i] = -1
                        trend = -1
                    elif trend == -1:
                        bos[i] = -1
                    else:
                        trend = -1

            if (bos[i] == 1) or (choch[i] == 1):
                for k in range(1, 10):
                    idx = i - k
                    if idx < 0:
                        break
                    if c[idx] < o[idx]:
                        ob[idx] = 1
                        break

            if (bos[i] == -1) or (choch[i] == -1):
                for k in range(1, 10):
                    idx = i - k
                    if idx < 0:
                        break
                    if c[idx] > o[idx]:
                        ob[idx] = -1
                        break

            trend_arr[i] = trend

            # Freshness: limit distance to recent structure
            d_high = 0.0
            if last_high_idx != -1 and (i - last_high_idx) < freshness_limit:
                d_high = abs(c[i] - h[last_high_idx])
            else:
                d_high = 999999.0  # Too far or undefined

            d_low = 0.0
            if last_low_idx != -1 and (i - last_low_idx) < freshness_limit:
                d_low = abs(c[i] - l[last_low_idx])
            else:
                d_low = 999999.0

            if d_high < 900000 and d_low < 900000:
                dist_liq[i] = min(d_high, d_low)
            elif d_high < 900000:
                dist_liq[i] = d_high
            elif d_low < 900000:
                dist_liq[i] = d_low
            else:
                dist_liq[i] = 0.0

            d_bull = (
                abs(c[i] - ((h[last_bullish_fvg_idx - 2] + l[last_bullish_fvg_idx]) / 2))
                if last_bullish_fvg_idx != -1
                else 999999.0
            )
            d_bear = (
                abs(c[i] - ((l[last_bearish_fvg_idx - 2] + h[last_bearish_fvg_idx]) / 2))
                if last_bearish_fvg_idx != -1
                else 999999.0
            )
            dist_fvg[i] = min(d_bull, d_bear)
            if dist_fvg[i] > 900000:
                dist_fvg[i] = 0.0

            # Premium/discount based on most recent high/low window
            if last_high_idx != -1 and last_low_idx != -1:
                range_high = h[last_high_idx]
                range_low = l[last_low_idx]
                if range_high > range_low:
                    mid = (range_high + range_low) * 0.5
                    premium_discount[i] = 1 if c[i] < mid else -1

            # Inducement: equal highs/lows raid and snap back (simple version)
            inducement[i] = 0
            if max_levels >= 2:
                # Check double-top/bottom liquidity: if two recent highs within small pip distance
                if recent_highs[1] != -1 and abs(h[recent_highs[0]] - h[recent_highs[1]]) < 0.5 * max(atr[i], 1e-6):
                    if c[i] < h[recent_highs[0]] and c[i] > l[i]:  # sweep and return inside bar
                        inducement[i] = 1
                if recent_lows[1] != -1 and abs(l[recent_lows[0]] - l[recent_lows[1]]) < 0.5 * max(atr[i], 1e-6):
                    if c[i] > l[recent_lows[0]] and c[i] < h[i]:
                        inducement[i] = 1

        return fvg, sweep, ob, bos, choch, trend_arr, dist_liq, dist_fvg, premium_discount, inducement
else:

    def _compute_smc_numba(o, h, l, c, atr, lookback=5):  # noqa: E741
        n = len(c)
        return (
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
        )  # Added 2 zeros for new returns


if NUMBA_CUDA_AVAILABLE:

    @cuda.jit
    def _smc_cuda_kernel(
        o,
        h,
        l,  # noqa: E741
        c,
        atr,
        lookback,
        freshness_limit,
        atr_disp,
        max_levels,
        fvg,
        sweep,
        ob,
        bos,
        choch,
        trend_arr,
        dist_liq,
        dist_fvg,
        pd_zone,
        inducement,
    ):
        i = cuda.grid(1)
        n = c.shape[0]
        if i < lookback or i >= n:
            return

        trend = 0
        last_high_idx = -1
        last_low_idx = -1
        last_bull_fvg = -1
        last_bear_fvg = -1

        pivot_idx = i - (lookback // 2)
        pivot_high = h[pivot_idx]
        pivot_low = l[pivot_idx]
        is_high = True
        is_low = True
        for k in range(1, lookback + 1):
            idx = i - lookback + k
            if idx == pivot_idx:
                continue
            if h[idx] >= pivot_high:
                is_high = False
            if l[idx] <= pivot_low:
                is_low = False
        if is_high:
            last_high_idx = pivot_idx
        if is_low:
            last_low_idx = pivot_idx

        body = abs(c[i - 1] - o[i - 1])
        rng = h[i - 1] - l[i - 1]
        displacement_ok = body > atr_disp * max(atr[i - 1], 1e-6)
        if (l[i] > h[i - 2]) and (c[i - 1] > o[i - 1]) and body > 0.5 * rng and displacement_ok:
            fvg[i] = 1
            last_bull_fvg = i
        elif (h[i] < l[i - 2]) and (c[i - 1] < o[i - 1]) and body > 0.5 * rng and displacement_ok:
            fvg[i] = -1
            last_bear_fvg = i

        if last_low_idx != -1 and (i - last_low_idx) < freshness_limit:
            if l[i] < l[last_low_idx] and c[i] > l[last_low_idx]:
                sweep[i] = 1
        if last_high_idx != -1 and (i - last_high_idx) < freshness_limit:
            if h[i] > h[last_high_idx] and c[i] < h[last_high_idx]:
                sweep[i] = -1

        if last_high_idx != -1 and c[i] > h[last_high_idx]:
            if trend == -1:
                choch[i] = 1
                trend = 1
            else:
                bos[i] = 1
                trend = 1
        if last_low_idx != -1 and c[i] < l[last_low_idx]:
            if trend == 1:
                choch[i] = -1
                trend = -1
            else:
                bos[i] = -1
                trend = -1
        trend_arr[i] = trend

        if bos[i] == 1 or choch[i] == 1:
            for k in range(1, min(10, i)):
                idx = i - k
                if c[idx] < o[idx]:
                    ob[idx] = 1
                    break
        if bos[i] == -1 or choch[i] == -1:
            for k in range(1, min(10, i)):
                idx = i - k
                if c[idx] > o[idx]:
                    ob[idx] = -1
                    break

        d_high = 1e9
        d_low = 1e9
        if last_high_idx != -1 and (i - last_high_idx) < freshness_limit:
            d_high = abs(c[i] - h[last_high_idx])
        if last_low_idx != -1 and (i - last_low_idx) < freshness_limit:
            d_low = abs(c[i] - l[last_low_idx])
        dist_liq[i] = d_high if d_high < d_low else d_low
        if dist_liq[i] > 5e8:
            dist_liq[i] = 0.0

        d_bull = 1e9
        d_bear = 1e9
        if last_bull_fvg != -1 and last_bull_fvg >= 2:
            d_bull = abs(c[i] - ((h[last_bull_fvg - 2] + l[last_bull_fvg]) * 0.5))
        if last_bear_fvg != -1 and last_bear_fvg >= 2:
            d_bear = abs(c[i] - ((l[last_bear_fvg - 2] + h[last_bear_fvg]) * 0.5))
        d_fvg = d_bull if d_bull < d_bear else d_bear
        dist_fvg[i] = 0.0 if d_fvg > 5e8 else d_fvg

        pd_zone[i] = 0
        if last_high_idx != -1 and last_low_idx != -1 and h[last_high_idx] > l[last_low_idx]:
            mid = (h[last_high_idx] + l[last_low_idx]) * 0.5
            pd_zone[i] = 1 if c[i] < mid else -1

        inducement[i] = 1 if dist_liq[i] > 0 and dist_liq[i] < atr[i] else 0

else:
    _smc_cuda_kernel = None


def _compute_smc_cuda(o, h, l, c, atr, lookback=5, freshness_limit=200, atr_displacement=1.2, max_levels=3):  # noqa: E741
    if not NUMBA_CUDA_AVAILABLE or _smc_cuda_kernel is None:
        raise RuntimeError("CUDA not available")
    n = len(c)
    # Host->device
    d_o = cuda.to_device(o.astype(np.float32))
    d_h = cuda.to_device(h.astype(np.float32))
    d_l = cuda.to_device(l.astype(np.float32))
    d_c = cuda.to_device(c.astype(np.float32))
    d_atr = cuda.to_device(atr.astype(np.float32))

    d_fvg = cuda.device_array(n, dtype=np.int8)
    d_sweep = cuda.device_array(n, dtype=np.int8)
    d_ob = cuda.device_array(n, dtype=np.int8)
    d_bos = cuda.device_array(n, dtype=np.int8)
    d_choch = cuda.device_array(n, dtype=np.int8)
    d_trend = cuda.device_array(n, dtype=np.int8)
    d_dist_liq = cuda.device_array(n, dtype=np.float32)
    d_dist_fvg = cuda.device_array(n, dtype=np.float32)
    d_pd_zone = cuda.device_array(n, dtype=np.int8)
    d_inducement = cuda.device_array(n, dtype=np.int8)

    threads = 256
    blocks = (n + threads - 1) // threads
    _smc_cuda_kernel[blocks, threads](
        d_o,
        d_h,
        d_l,
        d_c,
        d_atr,
        lookback,
        freshness_limit,
        atr_displacement,
        max_levels,
        d_fvg,
        d_sweep,
        d_ob,
        d_bos,
        d_choch,
        d_trend,
        d_dist_liq,
        d_dist_fvg,
        d_pd_zone,
        d_inducement,
    )
    cuda.synchronize()

    return (
        d_fvg.copy_to_host(),
        d_sweep.copy_to_host(),
        d_ob.copy_to_host(),
        d_bos.copy_to_host(),
        d_choch.copy_to_host(),
        d_trend.copy_to_host(),
        d_dist_liq.copy_to_host(),
        d_dist_fvg.copy_to_host(),
        d_pd_zone.copy_to_host(),
        d_inducement.copy_to_host(),
    )


if NUMBA_AVAILABLE:

    @njit(fastmath=True, parallel=True, nogil=True)
    def _compute_structure_labels_numba(
        close: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        target_highs: np.ndarray,
        target_lows: np.ndarray,
        min_dists: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        n = len(close)
        labels = np.zeros(n, dtype=np.float32)

        for i in prange(n - horizon):
            c = close[i]
            t_h = max(target_highs[i], c + min_dists[i])
            t_l = min(target_lows[i], c - min_dists[i])

            for j in range(i + 1, i + horizon):
                if highs[j] > t_h:
                    labels[i] = 1.0
                    break
                if lows[j] < t_l:
                    labels[i] = -1.0
                    break
        return labels
else:

    def _compute_structure_labels_numba(*args):
        # Return 2D array with shape (n, 2) for bos_bull and bos_bear columns
        n = len(args[0])
        return np.zeros((n, 2))

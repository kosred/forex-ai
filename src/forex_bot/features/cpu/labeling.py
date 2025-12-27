"""
Triple Barrier Labeling for Financial Time Series.

This module implements the triple barrier method for labeling trading signals:
- Upper barrier (take profit): Price target for profit
- Lower barrier (stop loss): Price target for loss protection
- Vertical barrier (time horizon): Maximum holding period

Labels:
- 1 = Buy (long) - price hit upper barrier first
- -1 = Sell (short) - price hit lower barrier first
- 0 = Neutral - neither barrier hit within horizon
"""

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


if NUMBA_AVAILABLE:

    @njit(fastmath=True, parallel=True, nogil=True, cache=True)
    def compute_triple_barrier_labels(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        horizon: np.ndarray,
        tp_mult: float,
        sl_mult: float,
        min_move: float,
    ) -> np.ndarray:
        """
        Compute triple barrier labels with dynamic horizon based on volatility.

        Parameters
        ----------
        close : np.ndarray
            Close prices
        high : np.ndarray
            High prices
        low : np.ndarray
            Low prices
        atr : np.ndarray
            ATR values for dynamic barrier sizing
        horizon : np.ndarray
            Dynamic horizon per bar (adjusted for volatility)
        tp_mult : float
            ATR multiplier for take profit distance
        sl_mult : float
            ATR multiplier for stop loss distance
        min_move : float
            Minimum price move to consider (filters noise)

        Returns
        -------
        np.ndarray
            Labels: 1 (buy), -1 (sell), 0 (neutral)
        """
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)

        for i in prange(n):
            curr_atr = atr[i]
            curr_horizon = horizon[i]

            if curr_horizon <= 0 or i + curr_horizon >= n:
                continue

            if curr_atr <= 0:
                continue

            entry_price = close[i]

            # Dynamic barriers based on ATR
            tp_dist = curr_atr * tp_mult
            sl_dist = curr_atr * sl_mult

            # Ensure minimum movement threshold
            if tp_dist < min_move:
                tp_dist = min_move
            if sl_dist < min_move:
                sl_dist = min_move

            # Upper and lower barriers (absolute price levels)
            upper_barrier = entry_price + tp_dist
            lower_barrier = entry_price - sl_dist

            # Track which barrier is hit first
            upper_hit_idx = -1
            lower_hit_idx = -1

            end_idx = min(i + curr_horizon + 1, n)

            for j in range(i + 1, end_idx):
                h = high[j]
                lo = low[j]

                # Check upper barrier (potential long profit)
                if upper_hit_idx < 0 and h >= upper_barrier:
                    upper_hit_idx = j

                # Check lower barrier (potential short profit)
                if lower_hit_idx < 0 and lo <= lower_barrier:
                    lower_hit_idx = j

                # Both hit - determine which was first
                if upper_hit_idx > 0 and lower_hit_idx > 0:
                    break

            # Determine label based on which barrier hit first
            if upper_hit_idx > 0 and lower_hit_idx < 0:
                # Only upper barrier hit -> Buy signal would have profited
                labels[i] = 1
            elif lower_hit_idx > 0 and upper_hit_idx < 0:
                # Only lower barrier hit -> Sell signal would have profited
                labels[i] = -1
            elif upper_hit_idx > 0 and lower_hit_idx > 0:
                # Both hit - which was first?
                if upper_hit_idx < lower_hit_idx:
                    labels[i] = 1
                elif lower_hit_idx < upper_hit_idx:
                    labels[i] = -1
                else:
                    # Same bar - check which extreme was hit first using OHLC order
                    # Assume worst case for trading (price moved against first)
                    labels[i] = 0
            else:
                # Neither barrier hit within horizon
                labels[i] = 0

        return labels

    @njit(fastmath=True, parallel=True, nogil=True, cache=True)
    def compute_meta_labels_atr_numba(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        signals: np.ndarray,
        max_hold: int,
        atr_mult: float,
        rr: float,
        min_dist: float,
        fixed_sl: float,
        fixed_tp: float,
    ) -> np.ndarray:
        """
        Compute meta-labels for existing signals (validates if signal would have been profitable).

        This is used for meta-labeling: given a primary signal, would taking that trade
        have resulted in hitting TP before SL?

        Parameters
        ----------
        close, high, low : np.ndarray
            OHLC price data
        atr : np.ndarray
            ATR values for dynamic sizing
        signals : np.ndarray
            Primary model signals: 1 (buy), -1 (sell), 0 (no trade)
        max_hold : int
            Maximum bars to hold position
        atr_mult : float
            ATR multiplier for stop loss
        rr : float
            Risk-reward ratio (TP = SL * rr)
        min_dist : float
            Minimum distance for SL/TP
        fixed_sl, fixed_tp : float
            Retained for API compatibility; when ATR is zero the label is skipped.

        Returns
        -------
        np.ndarray
            Meta-labels: 1 (signal was correct), 0 (signal was wrong or uncertain)
        """
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)

        for i in prange(n):
            sig = signals[i]
            if sig == 0:
                continue

            entry_price = close[i]
            curr_atr = atr[i]

            if curr_atr > 0:
                dist_sl = curr_atr * atr_mult
                dist_tp = dist_sl * rr
            else:
                # No reliable distance -> skip labeling for this bar.
                continue

            if dist_sl < min_dist:
                dist_sl = min_dist
            if dist_tp < min_dist:
                dist_tp = min_dist

            tp_price = 0.0
            sl_price = 0.0

            if sig == 1:  # Long signal
                tp_price = entry_price + dist_tp
                sl_price = entry_price - dist_sl
            else:  # Short signal (sig == -1)
                tp_price = entry_price - dist_tp
                sl_price = entry_price + dist_sl

            end_idx = min(i + max_hold + 1, n)
            hit_tp = False
            hit_sl = False
            tp_idx = n + 1
            sl_idx = n + 1

            for j in range(i + 1, end_idx):
                h = high[j]
                low_j = low[j]

                if sig == 1:  # Long position
                    if not hit_tp and h >= tp_price:
                        hit_tp = True
                        tp_idx = j
                    if not hit_sl and low_j <= sl_price:
                        hit_sl = True
                        sl_idx = j
                else:  # Short position
                    if not hit_tp and low_j <= tp_price:
                        hit_tp = True
                        tp_idx = j
                    if not hit_sl and h >= sl_price:
                        hit_sl = True
                        sl_idx = j

                if hit_tp and hit_sl:
                    break

            # Determine if the signal was correct
            if hit_tp and not hit_sl:
                labels[i] = 1  # Signal was correct
            elif hit_sl and not hit_tp:
                labels[i] = 0  # Signal was wrong
            elif hit_tp and hit_sl:
                # Both hit - TP first means signal was correct
                labels[i] = 1 if tp_idx < sl_idx else 0
            else:
                # Neither hit - uncertain, mark as wrong for conservative labeling
                labels[i] = 0

        return labels

    @njit(fastmath=True, parallel=True, nogil=True, cache=True)
    def compute_dynamic_horizon(
        atr: np.ndarray,
        base_horizon: int,
        vol_scale: float,
    ) -> np.ndarray:
        """
        HPC Optimized: Parallel dynamic horizon calculation.
        """
        n = len(atr)
        horizon = np.zeros(n, dtype=np.int32)

        # Compute mean ATR for fast parallel normalization
        # (Median is hard to parallelize, Mean is near-instant)
        sum_atr = 0.0
        count = 0
        for i in prange(n):
            if atr[i] > 0:
                sum_atr += atr[i]
                count += 1
        
        if count == 0:
            for i in prange(n): horizon[i] = base_horizon
            return horizon

        avg_atr = sum_atr / count

        for i in prange(n):
            curr_atr = atr[i]
            if curr_atr <= 0:
                horizon[i] = base_horizon
                continue

            vol_ratio = curr_atr / avg_atr
            adj_factor = 1.0 / (vol_ratio ** vol_scale)

            if adj_factor < 0.25: adj_factor = 0.25
            elif adj_factor > 4.0: adj_factor = 4.0

            horizon[i] = max(5, int(base_horizon * adj_factor))

        return horizon

    # Legacy function for backward compatibility
    @njit(fastmath=True, parallel=True, nogil=True, cache=True)
    def compute_labels_numba(
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        profit_target: np.ndarray,
        stop_loss: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """
        Legacy triple barrier labeling (kept for backward compatibility).

        NOTE: This function has limited functionality. Use compute_triple_barrier_labels
        for new code.
        """
        n = len(prices)
        labels = np.zeros(n, dtype=np.float32)

        for i in prange(n):
            if i + horizon >= n:
                continue

            entry = prices[i]
            pt = profit_target[i]
            sl = stop_loss[i]

            # Determine direction based on barrier positions
            is_long = pt > entry and sl < entry
            is_short = pt < entry and sl > entry

            if not is_long and not is_short:
                # Invalid barrier configuration
                labels[i] = 0.0
                continue

            hit_label = 0.0

            for j in range(i + 1, min(i + horizon + 1, n)):
                h = highs[j]
                low_j = lows[j]

                if is_long:
                    # Check stop loss first (conservative)
                    if low_j <= sl:
                        hit_label = 0.0
                        break
                    if h >= pt:
                        hit_label = 1.0
                        break
                else:  # is_short
                    # Check stop loss first (conservative)
                    if h >= sl:
                        hit_label = 0.0
                        break
                    if low_j <= pt:
                        hit_label = 1.0
                        break

            labels[i] = hit_label

        return labels

else:
    # Python fallbacks (slower but functional)

    def compute_triple_barrier_labels(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        horizon: np.ndarray,
        tp_mult: float,
        sl_mult: float,
        min_move: float,
    ) -> np.ndarray:
        """Python fallback for triple barrier labels."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)

        for i in range(n):
            curr_atr = atr[i]
            curr_horizon = int(horizon[i])

            if curr_horizon <= 0 or i + curr_horizon >= n:
                continue

            if curr_atr <= 0:
                continue

            entry_price = close[i]

            tp_dist = max(curr_atr * tp_mult, min_move)
            sl_dist = max(curr_atr * sl_mult, min_move)

            upper_barrier = entry_price + tp_dist
            lower_barrier = entry_price - sl_dist

            upper_hit_idx = -1
            lower_hit_idx = -1

            end_idx = min(i + curr_horizon + 1, n)

            for j in range(i + 1, end_idx):
                h = high[j]
                lo = low[j]

                if upper_hit_idx < 0 and h >= upper_barrier:
                    upper_hit_idx = j

                if lower_hit_idx < 0 and lo <= lower_barrier:
                    lower_hit_idx = j

                if upper_hit_idx > 0 and lower_hit_idx > 0:
                    break

            if upper_hit_idx > 0 and lower_hit_idx < 0:
                labels[i] = 1
            elif lower_hit_idx > 0 and upper_hit_idx < 0:
                labels[i] = -1
            elif upper_hit_idx > 0 and lower_hit_idx > 0:
                if upper_hit_idx < lower_hit_idx:
                    labels[i] = 1
                elif lower_hit_idx < upper_hit_idx:
                    labels[i] = -1
                else:
                    labels[i] = 0
            else:
                labels[i] = 0

        return labels

    def compute_meta_labels_atr_numba(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        signals: np.ndarray,
        max_hold: int,
        atr_mult: float,
        rr: float,
        min_dist: float,
        fixed_sl: float,
        fixed_tp: float,
    ) -> np.ndarray:
        """Python fallback for meta-labels."""
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)

        for i in range(n):
            sig = signals[i]
            if sig == 0:
                continue

            entry_price = close[i]
            curr_atr = atr[i]

            if curr_atr > 0:
                dist_sl = curr_atr * atr_mult
                dist_tp = dist_sl * rr
            else:
                # No reliable distance -> skip labeling for this bar.
                continue

            dist_sl = max(dist_sl, min_dist)
            dist_tp = max(dist_tp, min_dist)

            if sig == 1:
                tp_price = entry_price + dist_tp
                sl_price = entry_price - dist_sl
            else:
                tp_price = entry_price - dist_tp
                sl_price = entry_price + dist_sl

            end_idx = min(i + max_hold + 1, n)
            hit_tp = False
            hit_sl = False
            tp_idx = n + 1
            sl_idx = n + 1

            for j in range(i + 1, end_idx):
                h = high[j]
                low_j = low[j]

                if sig == 1:
                    if not hit_tp and h >= tp_price:
                        hit_tp = True
                        tp_idx = j
                    if not hit_sl and low_j <= sl_price:
                        hit_sl = True
                        sl_idx = j
                else:
                    if not hit_tp and low_j <= tp_price:
                        hit_tp = True
                        tp_idx = j
                    if not hit_sl and h >= sl_price:
                        hit_sl = True
                        sl_idx = j

                if hit_tp and hit_sl:
                    break

            if hit_tp and not hit_sl:
                labels[i] = 1
            elif hit_sl and not hit_tp:
                labels[i] = 0
            elif hit_tp and hit_sl:
                labels[i] = 1 if tp_idx < sl_idx else 0
            else:
                labels[i] = 0

        return labels

    def compute_dynamic_horizon(
        atr: np.ndarray,
        base_horizon: int,
        vol_scale: float,
    ) -> np.ndarray:
        """Python fallback for dynamic horizon."""
        n = len(atr)
        horizon = np.zeros(n, dtype=np.int32)

        valid_atr = atr[atr > 0]
        if len(valid_atr) == 0:
            horizon[:] = base_horizon
            return horizon

        median_atr = np.median(valid_atr)
        if median_atr <= 0:
            horizon[:] = base_horizon
            return horizon

        for i in range(n):
            curr_atr = atr[i]
            if curr_atr <= 0:
                horizon[i] = base_horizon
                continue

            vol_ratio = curr_atr / median_atr
            adj_factor = 1.0 / (vol_ratio ** vol_scale)
            adj_factor = max(0.25, min(4.0, adj_factor))
            horizon[i] = max(5, int(base_horizon * adj_factor))

        return horizon

    def compute_labels_numba(
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        profit_target: np.ndarray,
        stop_loss: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Legacy Python fallback."""
        n = len(prices)
        labels = np.zeros(n, dtype=np.float32)

        for i in range(n):
            if i + horizon >= n:
                continue

            entry = prices[i]
            pt = profit_target[i]
            sl = stop_loss[i]

            is_long = pt > entry and sl < entry
            is_short = pt < entry and sl > entry

            if not is_long and not is_short:
                labels[i] = 0.0
                continue

            hit_label = 0.0

            for j in range(i + 1, min(i + horizon + 1, n)):
                h = highs[j]
                low_j = lows[j]

                if is_long:
                    if low_j <= sl:
                        hit_label = 0.0
                        break
                    if h >= pt:
                        hit_label = 1.0
                        break
                else:
                    if h >= sl:
                        hit_label = 0.0
                        break
                    if low_j <= pt:
                        hit_label = 1.0
                        break

            labels[i] = hit_label

        return labels

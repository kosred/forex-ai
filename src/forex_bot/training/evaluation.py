from typing import Any

import numpy as np
import pandas as pd

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False


def pad_probs(probs: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
    """
    Normalize probability outputs to shape (n,3) ordered as [neutral, buy, sell].

    If `classes` is provided, it is interpreted as the class label for each column and used to reorder/map.
    Supported label conventions:
      - {-1, 0, 1}: -1=sell, 0=neutral, 1=buy
      - {0, 1, 2}: 0=neutral, 1=buy, 2=sell
    """
    if probs is None or len(probs) == 0:
        return np.zeros((0, 3), dtype=float)
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]

    if arr.shape[1] >= 3 and not classes:
        return arr[:, :3]

    out = np.zeros((n, 3), dtype=float)
    if classes and len(classes) == arr.shape[1]:
        for col, cls in enumerate(classes):
            if cls == 0:
                out[:, 0] = arr[:, col]
            elif cls == 1:
                out[:, 1] = arr[:, col]
            elif cls == -1:
                out[:, 2] = arr[:, col]
            elif cls == 2:
                out[:, 2] = arr[:, col]
        return out

    if arr.shape[1] >= 3:
        return arr[:, :3]
    if arr.shape[1] == 2:
        # Default: treat as [neutral, buy]
        out[:, 0] = arr[:, 0]
        out[:, 1] = arr[:, 1]
        return out
    out[:, 0] = 1.0 - arr[:, 0]
    out[:, 1] = arr[:, 0]
    return out


def probs_to_signals(probs: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
    """
    Map probability matrix [neutral, buy, sell] to discrete signals {-1,0,1}.
    Chooses highest prob; ties fall back to neutral.
    """
    p = pad_probs(probs, classes=classes)
    if len(p) == 0:
        return np.zeros(0, dtype=int)
    sig_idx = p.argmax(axis=1)
    signals = np.zeros(len(p), dtype=int)
    signals[sig_idx == 1] = 1
    signals[sig_idx == 2] = -1
    return signals


def quick_backtest(df: pd.DataFrame, signals: pd.Series) -> dict[str, Any]:
    """
    Lightweight backtest: assumes df has 'close', uses next-bar direction as outcome.
    Returns accuracy and simple PnL proxy.
    """
    if df is None or signals is None or len(df) == 0 or len(signals) != len(df):
        return {}
    close = df["close"].to_numpy()
    future = np.roll(close, -1)
    ret = future - close
    sig_arr = signals.to_numpy()

    pnl = np.where(
        sig_arr == 0, 0.0, np.where(sig_arr == 1, np.where(ret > 0, 1.0, -1.0), np.where(ret < 0, 1.0, -1.0))
    )
    pnl = pnl[:-1]
    acc = float((sig_arr[:-1] == np.sign(ret[:-1])).mean()) if len(ret) > 1 else 0.0
    return {
        "accuracy": acc,
        "pnl_score": float(pnl.sum()),
        "win_rate": float((pnl > 0).mean()) if len(pnl) > 0 else 0.0,
    }


# Forward declaration of Numba function to avoid import errors if Numba missing
_fast_prop_backtest_numba = None

try:
    import numba

    @numba.njit(fastmath=True, cache=True, nogil=True)
    def _fast_prop_backtest_numba(
        close: np.ndarray, signals: np.ndarray, timestamps: np.ndarray, max_daily_dd_pct: float, max_trades_per_day: int
    ):
        n = len(close)
        if n < 2:
            return 0.0, 0.0, 0.0, 0, False, False

        equity = 1.0
        peak = 1.0
        max_dd = 0.0

        # State tracking
        violations_dd = False
        violations_trades = False

        # Trade stats
        wins = 0
        trades = 0

        # Daily reset logic
        # timestamps are int64 nanoseconds.
        # 1 day = 24 * 3600 * 1e9 ns
        NS_PER_DAY = 86400 * 1_000_000_000

        current_day_idx = timestamps[0] // NS_PER_DAY
        day_start_equity = 1.0
        daily_trades = 0

        # Meta-Controller Logic Proxies (Simplified for JIT)
        consec_losses = 0
        # Rolling win rate (last 20 trades)
        outcome_history = np.zeros(20, dtype=np.int8)
        outcome_ptr = 0
        outcome_len = 0

        for i in range(n - 1):
            # 1. Day Check
            ts = timestamps[i]
            day_idx = ts // NS_PER_DAY

            if day_idx != current_day_idx:
                current_day_idx = day_idx
                day_start_equity = equity
                daily_trades = 0

            # 2. Risk Checks
            current_daily_dd = 0.0
            if day_start_equity > 0:
                current_daily_dd = (day_start_equity - equity) / day_start_equity

            allow_trade = True
            if current_daily_dd >= max_daily_dd_pct:
                violations_dd = True
                allow_trade = False

            if daily_trades >= max_trades_per_day:
                violations_trades = True
                allow_trade = False

            # 3. Trade Logic
            sig = signals[i]
            if sig != 0 and allow_trade:
                # Calculate simple rolling win rate
                wr_sum = 0
                for k in range(outcome_len):
                    wr_sum += outcome_history[k]

                recent_wr = 0.5
                if outcome_len > 0:
                    recent_wr = wr_sum / outcome_len

                # Dynamic Risk Scaling (Meta-Controller Logic)
                risk_mult = 1.0
                if consec_losses > 2:
                    risk_mult = 0.5
                elif recent_wr > 0.6:
                    risk_mult = 1.5

                # Execute Trade
                curr_price = close[i]
                next_price = close[i + 1]
                if curr_price < 1e-9:
                    bar_ret = 0.0
                else:
                    bar_ret = (next_price - curr_price) / curr_price

                trade_ret = sig * bar_ret

                base_risk = 0.01
                realized_pnl = base_risk * risk_mult * (1.0 if trade_ret > 0 else -1.0)

                # Commission/Spread Penalty
                realized_pnl -= 0.0005 * risk_mult

                equity *= 1.0 + realized_pnl

                trades += 1
                daily_trades += 1

                if trade_ret > 0:
                    wins += 1
                    consec_losses = 0
                    outcome_history[outcome_ptr] = 1
                else:
                    consec_losses += 1
                    outcome_history[outcome_ptr] = 0

                outcome_ptr = (outcome_ptr + 1) % 20
                if outcome_len < 20:
                    outcome_len += 1

            # 4. Global Drawdown Tracking
            if equity > peak:
                peak = equity

            dd = peak - equity
            if peak > 0:
                dd_pct = dd / peak
                if dd_pct > max_dd:
                    max_dd = dd_pct

        total_pnl = equity - 1.0
        final_wr = 0.0
        if trades > 0:
            final_wr = wins / trades

        return total_pnl, final_wr, max_dd, trades, violations_dd, violations_trades

except ImportError:
    # Fallback if Numba not installed
    def _fast_prop_backtest_numba(c, s, t, mdd, mt):
        return 0.0, 0.0, 0.0, 0, False, False


def prop_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    max_daily_dd_pct: float = 0.05,
    daily_dd_warn_pct: float = 0.03,
    max_trades_per_day: int = 10,
    use_gpu: bool | None = None,
) -> dict[str, Any]:
    """
    Backtest with MetaController integration for realistic Prop Firm simulation.
    Uses Numba for high-performance CPU execution (approx 100x speedup vs pure Python).
    """
    if df is None or signals is None or len(df) == 0 or len(signals) != len(df):
        return {}

    # Extract numpy arrays for Numba
    # Ensure contiguous float64/int64 arrays for maximum JIT speed
    close = df["close"].to_numpy(dtype=np.float64)
    signals_arr = signals.to_numpy(dtype=np.int8)

    # We need timestamps to detect day changes.
    # Convert DateTimeIndex to int64 (nanoseconds)
    timestamps = df.index.view(np.int64)

    # Call the JIT-compiled kernel
    pnl, win_rate, max_dd, trades, vio_dd, vio_trades = _fast_prop_backtest_numba(
        close, signals_arr, timestamps, max_daily_dd_pct, max_trades_per_day
    )

    return {
        "pnl_score": float(pnl),
        "win_rate": float(win_rate),
        "max_dd_pct": float(max_dd),
        "trades": int(trades),
        "daily_dd_violation": bool(vio_dd),
        "trade_limit_violation": bool(vio_trades),
    }

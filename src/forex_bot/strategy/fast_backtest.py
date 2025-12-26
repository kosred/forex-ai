import numpy as np
from numba import njit
from typing import Any


def infer_pip_metrics(symbol: str) -> tuple[float, float]:
    """Heuristic pip_size/pip_value_per_lot for offline backtests."""
    sym = (symbol or "").upper()
    if sym.startswith(("XAU", "XAG")):
        return 0.1, 10.0
    if "BTC" in sym or "ETH" in sym or "LTC" in sym:
        return 1.0, 1.0
    if sym.endswith("JPY") or sym.startswith("JPY"):
        return 0.01, 9.0
    return 0.0001, 10.0


def infer_sl_tp_pips_auto(
    *,
    open_prices: np.ndarray | None = None,
    high_prices: np.ndarray | None,
    low_prices: np.ndarray | None,
    close_prices: np.ndarray | None = None,
    atr_values: np.ndarray | None,
    pip_size: float,
    atr_mult: float,
    min_rr: float,
    min_dist: float,
    settings: Any | None = None,
) -> tuple[float, float] | None:
    """
    Derive SL/TP pips from market data when fixed pip targets are disabled.

    Returns None if no reliable distance can be inferred.
    """
    if settings is not None and open_prices is not None and close_prices is not None:
        try:
            import pandas as pd
            from .stop_target import infer_stop_target_pips

            df = pd.DataFrame(
                {
                    "open": open_prices,
                    "high": high_prices if high_prices is not None else close_prices,
                    "low": low_prices if low_prices is not None else close_prices,
                    "close": close_prices,
                }
            )
            res = infer_stop_target_pips(df, settings=settings, pip_size=pip_size)
            if res is not None:
                sl_pips, tp_pips, _rr = res
                return float(sl_pips), float(tp_pips)
        except Exception:
            pass

    base_dist = 0.0
    if atr_values is not None:
        try:
            atr_vals = atr_values[np.isfinite(atr_values) & (atr_values > 0)]
            if atr_vals.size > 0:
                base_dist = float(np.nanmedian(atr_vals)) * max(atr_mult, 0.0)
        except Exception:
            base_dist = 0.0

    if base_dist <= 0.0 and high_prices is not None and low_prices is not None:
        try:
            tr = high_prices - low_prices
            tr_vals = tr[np.isfinite(tr) & (tr > 0)]
            if tr_vals.size > 0:
                base_dist = float(np.nanmedian(tr_vals))
        except Exception:
            base_dist = 0.0

    if base_dist <= 0.0:
        return None

    if min_dist and min_dist > 0:
        base_dist = max(base_dist, float(min_dist))

    pip_size = max(float(pip_size), 1e-9)
    sl_pips = base_dist / pip_size

    rr = float(min_rr) if min_rr and min_rr > 0 else 1.0
    tp_pips = sl_pips * rr
    return float(sl_pips), float(tp_pips)


@njit(cache=True, fastmath=True)
def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Calculate Max Drawdown from cumulative returns array."""
    max_peak = cumulative_returns[0]
    max_dd = 0.0
    for i in range(1, len(cumulative_returns)):
        if cumulative_returns[i] > max_peak:
            max_peak = cumulative_returns[i]
        dd = (max_peak - cumulative_returns[i]) / max_peak if max_peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


@njit(cache=True, fastmath=True)
def fast_evaluate_strategy(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    signals: np.ndarray,
    month_indices: np.ndarray,  # New: integer encoding of month (e.g. 2023*12 + 1)
    sl_pips: float,
    tp_pips: float,
    pip_value: float = 0.0001,
    spread_pips: float = 1.5,
    commission_per_trade: float = 0.0,  # Flat cost
    pip_value_per_lot: float = 10.0,
) -> np.ndarray:
    """
    Numba-optimized backtester with Monthly Consistency Tracking.
    Returns metrics array:
    [NetProfit, Sharpe, Sortino, MaxDD, WinRate, ProfitFactor, Expectancy, SQN,
     Trades, ConsistencyScore]
    """
    n = len(close_prices)
    equity = 100000.0
    peak_equity = 100000.0

    returns = np.zeros(n)
    trade_count = 0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    last_month_val = -1
    current_month_pnl = 0.0
    monthly_pnls = np.zeros(240)
    month_ptr = -1

    in_position = 0
    entry_price = 0.0

    cash_per_pip = pip_value_per_lot

    for i in range(1, n):
        m_val = month_indices[i]
        if m_val != last_month_val:
            if last_month_val != -1:
                month_ptr += 1
                if month_ptr < 240:
                    monthly_pnls[month_ptr] = current_month_pnl
            current_month_pnl = 0.0
            last_month_val = m_val

        if in_position != 0:
            current_low = low_prices[i]
            current_high = high_prices[i]

            sl_price = 0.0
            tp_price = 0.0

            pnl = 0.0
            exit_signal = False

            if in_position == 1:  # Long
                sl_price = entry_price - (sl_pips * pip_value)
                tp_price = entry_price + (tp_pips * pip_value)

                if current_low <= sl_price:
                    pnl = (sl_price - entry_price) / pip_value * cash_per_pip
                    exit_signal = True
                elif current_high >= tp_price:
                    pnl = (tp_price - entry_price) / pip_value * cash_per_pip
                    exit_signal = True
                elif signals[i] == -1:
                    pnl = (close_prices[i] - entry_price) / pip_value * cash_per_pip
                    exit_signal = True

            elif in_position == -1:  # Short
                sl_price = entry_price + (sl_pips * pip_value)
                tp_price = entry_price - (tp_pips * pip_value)

                if current_high >= sl_price:
                    pnl = (entry_price - sl_price) / pip_value * cash_per_pip
                    exit_signal = True
                elif current_low <= tp_price:
                    pnl = (entry_price - tp_price) / pip_value * cash_per_pip
                    exit_signal = True
                elif signals[i] == 1:
                    pnl = (entry_price - close_prices[i]) / pip_value * cash_per_pip
                    exit_signal = True

            if exit_signal:
                cost = (spread_pips * cash_per_pip) + commission_per_trade
                pnl -= cost

                equity += pnl
                current_month_pnl += pnl
                returns[trade_count] = pnl
                trade_count += 1
                in_position = 0

                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    losses += 1
                    gross_loss += abs(pnl)

                if equity > peak_equity:
                    peak_equity = equity

        if in_position == 0:
            sig = signals[i]
            if sig == 1:
                in_position = 1
                entry_price = close_prices[i] + (spread_pips * pip_value)
            elif sig == -1:
                in_position = -1
                entry_price = close_prices[i]

    if month_ptr < 239:
        monthly_pnls[month_ptr + 1] = current_month_pnl

    if trade_count == 0:
        return np.zeros(10)

    net_profit = equity - 100000.0
    win_rate = wins / trade_count
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0

    trade_rets = returns[:trade_count]
    avg_trade = np.mean(trade_rets)
    std_trade = np.std(trade_rets)

    sharpe = (avg_trade / std_trade) * np.sqrt(trade_count) if std_trade > 0 else 0.0

    downside = trade_rets[trade_rets < 0]
    std_down = np.std(downside) if len(downside) > 0 else 1.0
    sortino = (avg_trade / std_down) * np.sqrt(trade_count) if std_down > 0 else 0.0

    equity_curve = np.zeros(trade_count + 1)
    equity_curve[0] = 100000.0
    curr = 100000.0
    max_dd = 0.0
    peak = 100000.0
    for k in range(trade_count):
        curr += trade_rets[k]
        equity_curve[k + 1] = curr
        if curr > peak:
            peak = curr
        dd = (peak - curr) / peak
        if dd > max_dd:
            max_dd = dd

    expectancy = avg_trade
    sqn = (avg_trade / std_trade) * np.sqrt(trade_count) if std_trade > 0 else 0.0

    r_squared = 0.0
    if trade_count > 2:
        x = np.arange(trade_count + 1, dtype=np.float32)
        y = equity_curve[: trade_count + 1]
        n_points = float(len(x))
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        sum_yy = np.sum(y * y)
        numerator = (n_points * sum_xy) - (sum_x * sum_y)
        denominator_x = (n_points * sum_xx) - (sum_x * sum_x)
        denominator_y = (n_points * sum_yy) - (sum_y * sum_y)
        if denominator_x > 0 and denominator_y > 0:
            r = numerator / np.sqrt(denominator_x * denominator_y)
            r_squared = r * r

    active_months = monthly_pnls[: month_ptr + 2]  # +2 because ptr is 0-based and we added last one

    negative_months = np.sum(active_months < 0)

    consistency_score = r_squared
    if negative_months > 0:
        consistency_score *= 1.0 - (negative_months * 0.1)  # Reduce by 10% per losing month
        consistency_score = max(0.0, consistency_score)
    else:
        if len(active_months) > 1:  # Valid streak
            consistency_score += 0.2

    return np.array(
        [
            net_profit,
            sharpe,
            sortino,
            max_dd,
            win_rate,
            profit_factor,
            expectancy,
            sqn,
            float(trade_count),
            consistency_score,
        ]
    )

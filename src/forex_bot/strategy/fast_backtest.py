import numpy as np
from typing import Any
import logging

logger = logging.getLogger(__name__)

try:
    from numba import njit, prange
except ImportError:
    logger.warning("Numba not available or incompatible. Using slow pure-Python fallback.")
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    prange = range


def infer_pip_metrics(symbol: str) -> tuple[float, float]:
    """
    HPC FIX: Precision Pip Metrics.
    Uses institutional defaults but prefers live-broker metadata when available.
    """
    sym = (symbol or "").upper()
    # Gold/Silver (XAU/XAG) - Standard 2-decimal or 3-decimal
    if sym.startswith(("XAU", "XAG")):
        return 0.01, 1.0 # 1 cent move = 1 dollar per lot
    
    # JPY Pairs - Standard 2-decimal or 3-decimal
    if sym.endswith("JPY") or sym.startswith("JPY"):
        return 0.01, 6.5 # Approx dollar value per pip for JPY
        
    # Crypto
    if any(c in sym for c in ["BTC", "ETH", "LTC"]):
        return 1.0, 1.0
        
    # Standard FX - 4/5-decimal
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


@njit(cache=True, fastmath=True, parallel=True)
def batch_evaluate_strategies(
    close_prices_batch: np.ndarray,
    high_prices_batch: np.ndarray,
    low_prices_batch: np.ndarray,
    signals_batch: np.ndarray,
    month_indices_batch: np.ndarray,
    day_indices_batch: np.ndarray,
    sl_pips: float,
    tp_pips: float,
    max_hold_bars: int = 0,
    pip_value: float = 0.0001,
    spread_pips: float = 1.5,
    commission: float = 0.0,
    pip_val_lot: float = 10.0,
) -> np.ndarray:
    """
    HPC BATCH EVALUATOR: Runs multiple backtests in parallel across 252 cores.
    Inputs are 2D arrays (num_strategies, num_bars).
    Returns 2D metrics (num_strategies, 11).
    """
    n_strats = close_prices_batch.shape[0]
    out_metrics = np.zeros((n_strats, 11))
    
    for s in prange(n_strats):
        out_metrics[s] = fast_evaluate_strategy(
            close_prices_batch[s],
            high_prices_batch[s],
            low_prices_batch[s],
            signals_batch[s],
            month_indices_batch[s],
            day_indices_batch[s],
            sl_pips,
            tp_pips,
            max_hold_bars,
            False, 1.0, 1.0, # trailing defaults for speed
            pip_value,
            spread_pips,
            commission,
            pip_val_lot
        )
    return out_metrics

@njit(cache=True, fastmath=True)
def fast_evaluate_strategy(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    signals: np.ndarray,
    month_indices: np.ndarray,  # New: integer encoding of month (e.g. 2023*12 + 1)
    day_indices: np.ndarray,  # Integer encoding of day (e.g. 20240130)
    sl_pips: float,
    tp_pips: float,
    max_hold_bars: int = 0,
    trailing_enabled: bool = False,
    trailing_atr_multiplier: float = 1.0,
    trailing_be_trigger_r: float = 1.0,
    pip_value: float = 0.0001,
    spread_pips: float = 1.5,
    commission_per_trade: float = 0.0,  # Flat cost
    pip_value_per_lot: float = 10.0,
) -> np.ndarray:
    """
    Numba-optimized backtester with Monthly Consistency Tracking.
    Implements triple-barrier exits: SL/TP plus a max-hold time barrier, with optional trailing.
    Returns metrics array:
    [NetProfit, Sharpe, Sortino, MaxDD, WinRate, ProfitFactor, Expectancy, SQN,
     Trades, ConsistencyScore, MaxDailyDD]
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

    last_day_val = -1
    day_peak = equity
    day_low = equity
    max_daily_dd = 0.0

    in_position = 0
    entry_price = 0.0
    entry_index = -1
    trail_price = 0.0

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

        d_val = day_indices[i] if i < len(day_indices) else last_day_val
        if d_val != last_day_val:
            if last_day_val != -1 and day_peak > 0:
                dd = (day_peak - day_low) / day_peak
                if dd > max_daily_dd:
                    max_daily_dd = dd
            last_day_val = d_val
            day_peak = equity
            day_low = equity

        if in_position != 0:
            current_low = low_prices[i]
            current_high = high_prices[i]
            
            # HPC FIX: Real-time Intraday Floating PnL Tracking
            if in_position == 1:
                floating_pnl = (current_low - entry_price) / pip_value * cash_per_pip
            else:
                floating_pnl = (entry_price - current_high) / pip_value * cash_per_pip
            
            current_floating_equity = equity + floating_pnl
            if current_floating_equity < day_low:
                day_low = current_floating_equity

            sl_price = 0.0
            tp_price = 0.0

            pnl = 0.0
            exit_signal = False

            if in_position == 1:  # Long
                sl_price = entry_price - (sl_pips * pip_value)
                tp_price = entry_price + (tp_pips * pip_value)

                if trailing_enabled:
                    move = current_high - entry_price
                    if move >= (trailing_be_trigger_r * sl_pips * pip_value):
                        trail_dist = trailing_atr_multiplier * sl_pips * pip_value
                        candidate = current_high - trail_dist
                        if trail_price == 0.0 or candidate > trail_price:
                            trail_price = candidate
                        if trail_price > sl_price:
                            sl_price = trail_price

                if current_low <= sl_price:
                    pnl = (sl_price - entry_price) / pip_value * cash_per_pip
                    exit_signal = True
                elif current_high >= tp_price:
                    pnl = (tp_price - entry_price) / pip_value * cash_per_pip
                    exit_signal = True

            elif in_position == -1:  # Short
                sl_price = entry_price + (sl_pips * pip_value)
                tp_price = entry_price - (tp_pips * pip_value)

                if trailing_enabled:
                    move = entry_price - current_low
                    if move >= (trailing_be_trigger_r * sl_pips * pip_value):
                        trail_dist = trailing_atr_multiplier * sl_pips * pip_value
                        candidate = current_low + trail_dist
                        if trail_price == 0.0 or candidate < trail_price:
                            trail_price = candidate
                        if trail_price < sl_price:
                            sl_price = trail_price

                if current_high >= sl_price:
                    pnl = (entry_price - sl_price) / pip_value * cash_per_pip
                    exit_signal = True
                elif current_low <= tp_price:
                    pnl = (entry_price - tp_price) / pip_value * cash_per_pip
                    exit_signal = True

            if not exit_signal and max_hold_bars > 0 and entry_index >= 0:
                if (i - entry_index) >= max_hold_bars:
                    if in_position == 1:
                        pnl = (close_prices[i] - entry_price) / pip_value * cash_per_pip
                    else:
                        pnl = (entry_price - close_prices[i]) / pip_value * cash_per_pip
                    exit_signal = True

            if not exit_signal:
                if in_position == 1 and signals[i] == -1:
                    pnl = (close_prices[i] - entry_price) / pip_value * cash_per_pip
                    exit_signal = True
                elif in_position == -1 and signals[i] == 1:
                    pnl = (entry_price - close_prices[i]) / pip_value * cash_per_pip
                    exit_signal = True

            if exit_signal:
                # HPC FIX: Lot-based commission (Standard Prop Firm Rule)
                # commission_per_trade is now treated as 'Commission Per Side Per Lot'
                lot_size = 1.0 # Default lot size for score normalization
                cost_in_pips = spread_pips
                commission_in_cash = commission_per_trade * lot_size * 2.0 # Round trip
                
                total_cost = (cost_in_pips * cash_per_pip) + commission_in_cash
                pnl -= total_cost

                equity += pnl
                current_month_pnl += pnl
                returns[trade_count] = pnl
                trade_count += 1
                in_position = 0
                entry_index = -1
                trail_price = 0.0

                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                else:
                    losses += 1
                    gross_loss += abs(pnl)

                if equity > peak_equity:
                    peak_equity = equity
                if equity > day_peak:
                    day_peak = equity
                if equity < day_low:
                    day_low = equity

        if in_position == 0:
            sig = signals[i]
            if sig == 1:
                in_position = 1
                entry_price = close_prices[i] + (spread_pips * pip_value)
                entry_index = i
                trail_price = 0.0
            elif sig == -1:
                in_position = -1
                entry_price = close_prices[i]
                entry_index = i
                trail_price = 0.0

    if month_ptr < 239:
        monthly_pnls[month_ptr + 1] = current_month_pnl

    if last_day_val != -1 and day_peak > 0:
        dd = (day_peak - day_low) / day_peak
        if dd > max_daily_dd:
            max_daily_dd = dd

    if trade_count == 0:
        return np.zeros(11)

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
            max_daily_dd,
        ]
    )


def detailed_backtest(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    signals: np.ndarray,
    timestamps: np.ndarray | None,
    month_indices: np.ndarray | None,
    day_indices: np.ndarray | None,
    sl_pips: float,
    tp_pips: float,
    max_hold_bars: int = 0,
    trailing_enabled: bool = False,
    trailing_atr_multiplier: float = 1.0,
    trailing_be_trigger_r: float = 1.0,
    pip_value: float = 0.0001,
    spread_pips: float = 1.5,
    commission_per_trade: float = 0.0,
    pip_value_per_lot: float = 10.0,
    month_label_map: dict[int, str] | None = None,
    base_balance: float = 100000.0,
) -> dict[str, Any]:
    """
    Detailed backtest with per-month metrics and trade log.
    Mirrors fast_evaluate_strategy logic but returns richer diagnostics.
    """
    close = np.asarray(close_prices)
    high = np.asarray(high_prices)
    low = np.asarray(low_prices)
    sigs = np.asarray(signals)
    n = len(close)
    if n == 0:
        return {
            "summary": {
                "net_profit": 0.0,
                "max_dd": 0.0,
                "trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "sqn": 0.0,
                "consistency": 0.0,
                "max_daily_dd": 0.0,
            },
            "months": [],
            "monthly": {},
            "trades": [],
        }

    if month_indices is None or len(month_indices) != n:
        month_indices = np.zeros(n, dtype=np.int64)
    if day_indices is None or len(day_indices) != n:
        day_indices = np.zeros(n, dtype=np.int64)
    if timestamps is None or len(timestamps) != n:
        timestamps = None

    month_label_map = month_label_map or {}
    cash_per_pip = float(pip_value_per_lot)

    equity = float(base_balance)
    peak_equity = float(base_balance)

    trade_pnls: list[float] = []
    trade_log: list[dict[str, Any]] = []

    # Monthly tracking
    monthly: dict[str, dict[str, Any]] = {}
    months: list[str] = []
    current_month_val = int(month_indices[0]) if n > 0 else 0
    month_start_equity = equity
    month_peak_equity = equity
    month_low_equity = equity
    month_pnl = 0.0
    month_trades = 0
    month_wins = 0
    month_losses = 0
    month_gross_profit = 0.0
    month_gross_loss = 0.0
    month_trade_pnls: list[float] = []
    month_hold_bars_total = 0
    month_bars_in_pos = 0
    month_bars_total = 0
    month_max_dd = 0.0

    last_day_val = -1
    day_peak = equity
    day_low = equity
    max_daily_dd = 0.0

    month_last_day_val = -1
    month_day_peak = equity
    month_day_low = equity
    month_max_daily_dd = 0.0

    def _month_label(month_val: int) -> str:
        label = month_label_map.get(int(month_val))
        if label:
            return label
        # month_val encodes year*12 + month
        year = (month_val - 1) // 12
        month = ((month_val - 1) % 12) + 1
        return f"{int(year):04d}-{int(month):02d}"

    def _finalize_month() -> None:
        nonlocal month_pnl, month_trades, month_wins, month_losses
        nonlocal month_gross_profit, month_gross_loss, month_trade_pnls
        nonlocal month_hold_bars_total, month_bars_in_pos, month_bars_total
        nonlocal month_peak_equity, month_low_equity, month_max_dd
        nonlocal month_max_daily_dd, month_start_equity, equity

        label = _month_label(current_month_val)
        if label in monthly:
            return

        if month_trades > 0:
            win_rate = month_wins / month_trades
            profit_factor = (
                month_gross_profit / month_gross_loss if month_gross_loss > 0 else 99.0
            )
            avg_trade = float(np.mean(month_trade_pnls)) if month_trade_pnls else 0.0
            std_trade = float(np.std(month_trade_pnls)) if len(month_trade_pnls) > 1 else 0.0
            sharpe = (avg_trade / std_trade) * np.sqrt(month_trades) if std_trade > 0 else 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_trade = 0.0
            sharpe = 0.0

        exposure_pct = (month_bars_in_pos / month_bars_total) if month_bars_total > 0 else 0.0
        return_pct = (month_pnl / month_start_equity) * 100.0 if month_start_equity > 0 else 0.0
        base_return_pct = (month_pnl / base_balance) * 100.0 if base_balance > 0 else 0.0

        monthly[label] = {
            "month": label,
            "net_profit": float(month_pnl),
            "return_pct": float(return_pct),
            "base_return_pct": float(base_return_pct),
            "trades": int(month_trades),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_trade": float(avg_trade),
            "max_dd": float(month_max_dd),
            "max_daily_dd": float(month_max_daily_dd),
            "gross_profit": float(month_gross_profit),
            "gross_loss": float(month_gross_loss),
            "avg_hold_bars": float(month_hold_bars_total / month_trades) if month_trades > 0 else 0.0,
            "exposure_pct": float(exposure_pct),
            "start_equity": float(month_start_equity),
            "end_equity": float(equity),
            "sharpe": float(sharpe),
        }
        months.append(label)

    in_position = 0
    entry_price = 0.0
    entry_index = -1
    trail_price = 0.0
    entry_time = None

    for i in range(1, n):
        m_val = int(month_indices[i])
        if m_val != current_month_val:
            _finalize_month()
            current_month_val = m_val
            month_start_equity = equity
            month_peak_equity = equity
            month_low_equity = equity
            month_pnl = 0.0
            month_trades = 0
            month_wins = 0
            month_losses = 0
            month_gross_profit = 0.0
            month_gross_loss = 0.0
            month_trade_pnls = []
            month_hold_bars_total = 0
            month_bars_in_pos = 0
            month_bars_total = 0
            month_max_dd = 0.0
            month_last_day_val = -1
            month_day_peak = equity
            month_day_low = equity
            month_max_daily_dd = 0.0

        month_bars_total += 1
        if in_position != 0:
            month_bars_in_pos += 1

        d_val = int(day_indices[i]) if i < len(day_indices) else last_day_val
        if d_val != last_day_val:
            if last_day_val != -1 and day_peak > 0:
                dd = (day_peak - day_low) / day_peak
                if dd > max_daily_dd:
                    max_daily_dd = dd
            last_day_val = d_val
            day_peak = equity
            day_low = equity

        if d_val != month_last_day_val:
            if month_last_day_val != -1 and month_day_peak > 0:
                dd = (month_day_peak - month_day_low) / month_day_peak
                if dd > month_max_daily_dd:
                    month_max_daily_dd = dd
            month_last_day_val = d_val
            month_day_peak = equity
            month_day_low = equity

        if in_position != 0:
            current_low = low[i]
            current_high = high[i]

            # Floating PnL for daily DD tracking
            if in_position == 1:
                floating_pnl = (current_low - entry_price) / pip_value * cash_per_pip
            else:
                floating_pnl = (entry_price - current_high) / pip_value * cash_per_pip
            floating_equity = equity + floating_pnl
            if floating_equity < day_low:
                day_low = floating_equity
            if floating_equity < month_day_low:
                month_day_low = floating_equity

            sl_price = 0.0
            tp_price = 0.0
            pnl = 0.0
            exit_signal = False
            exit_reason = ""
            exit_price = close[i]

            if in_position == 1:
                sl_price = entry_price - (sl_pips * pip_value)
                tp_price = entry_price + (tp_pips * pip_value)

                if trailing_enabled:
                    move = current_high - entry_price
                    if move >= (trailing_be_trigger_r * sl_pips * pip_value):
                        trail_dist = trailing_atr_multiplier * sl_pips * pip_value
                        candidate = current_high - trail_dist
                        if trail_price == 0.0 or candidate > trail_price:
                            trail_price = candidate
                        if trail_price > sl_price:
                            sl_price = trail_price

                if current_low <= sl_price:
                    pnl = (sl_price - entry_price) / pip_value * cash_per_pip
                    exit_signal = True
                    exit_reason = "sl"
                    exit_price = sl_price
                elif current_high >= tp_price:
                    pnl = (tp_price - entry_price) / pip_value * cash_per_pip
                    exit_signal = True
                    exit_reason = "tp"
                    exit_price = tp_price

            else:
                sl_price = entry_price + (sl_pips * pip_value)
                tp_price = entry_price - (tp_pips * pip_value)

                if trailing_enabled:
                    move = entry_price - current_low
                    if move >= (trailing_be_trigger_r * sl_pips * pip_value):
                        trail_dist = trailing_atr_multiplier * sl_pips * pip_value
                        candidate = current_low + trail_dist
                        if trail_price == 0.0 or candidate < trail_price:
                            trail_price = candidate
                        if trail_price < sl_price:
                            sl_price = trail_price

                if current_high >= sl_price:
                    pnl = (entry_price - sl_price) / pip_value * cash_per_pip
                    exit_signal = True
                    exit_reason = "sl"
                    exit_price = sl_price
                elif current_low <= tp_price:
                    pnl = (entry_price - tp_price) / pip_value * cash_per_pip
                    exit_signal = True
                    exit_reason = "tp"
                    exit_price = tp_price

            if not exit_signal and max_hold_bars > 0 and entry_index >= 0:
                if (i - entry_index) >= max_hold_bars:
                    if in_position == 1:
                        pnl = (close[i] - entry_price) / pip_value * cash_per_pip
                    else:
                        pnl = (entry_price - close[i]) / pip_value * cash_per_pip
                    exit_signal = True
                    exit_reason = "max_hold"
                    exit_price = close[i]

            if not exit_signal:
                if in_position == 1 and sigs[i] == -1:
                    pnl = (close[i] - entry_price) / pip_value * cash_per_pip
                    exit_signal = True
                    exit_reason = "reverse"
                    exit_price = close[i]
                elif in_position == -1 and sigs[i] == 1:
                    pnl = (entry_price - close[i]) / pip_value * cash_per_pip
                    exit_signal = True
                    exit_reason = "reverse"
                    exit_price = close[i]

            if exit_signal:
                lot_size = 1.0
                cost_in_pips = spread_pips
                commission_in_cash = commission_per_trade * lot_size * 2.0
                total_cost = (cost_in_pips * cash_per_pip) + commission_in_cash
                pnl_gross = pnl
                pnl -= total_cost

                equity += pnl
                trade_pnls.append(pnl)
                month_trade_pnls.append(pnl)
                month_pnl += pnl
                month_trades += 1
                month_hold_bars_total += (i - entry_index)

                if pnl > 0:
                    month_wins += 1
                    month_gross_profit += pnl
                else:
                    month_losses += 1
                    month_gross_loss += abs(pnl)

                if equity > peak_equity:
                    peak_equity = equity
                if equity > month_peak_equity:
                    month_peak_equity = equity
                if equity < month_low_equity:
                    month_low_equity = equity
                if month_peak_equity > 0:
                    dd = (month_peak_equity - equity) / month_peak_equity
                    if dd > month_max_dd:
                        month_max_dd = dd

                if equity > day_peak:
                    day_peak = equity
                if equity < day_low:
                    day_low = equity
                if equity > month_day_peak:
                    month_day_peak = equity
                if equity < month_day_low:
                    month_day_low = equity

                trade_log.append(
                    {
                        "entry_time": timestamps[entry_index] if timestamps is not None else entry_index,
                        "exit_time": timestamps[i] if timestamps is not None else i,
                        "direction": "long" if in_position == 1 else "short",
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "pnl": float(pnl),
                        "pnl_gross": float(pnl_gross),
                        "cost": float(total_cost),
                        "pnl_pips": float(pnl / cash_per_pip) if cash_per_pip > 0 else 0.0,
                        "bars_held": int(i - entry_index),
                        "exit_reason": exit_reason,
                        "sl_price": float(sl_price),
                        "tp_price": float(tp_price),
                        "equity_after": float(equity),
                        "month": _month_label(current_month_val),
                    }
                )

                in_position = 0
                entry_index = -1
                entry_price = 0.0
                entry_time = None
                trail_price = 0.0

        if in_position == 0:
            sig = sigs[i]
            if sig == 1:
                in_position = 1
                entry_price = close[i] + (spread_pips * pip_value)
                entry_index = i
                entry_time = timestamps[i] if timestamps is not None else i
                trail_price = 0.0
            elif sig == -1:
                in_position = -1
                entry_price = close[i]
                entry_index = i
                entry_time = timestamps[i] if timestamps is not None else i
                trail_price = 0.0

    _finalize_month()

    if last_day_val != -1 and day_peak > 0:
        dd = (day_peak - day_low) / day_peak
        if dd > max_daily_dd:
            max_daily_dd = dd
    if month_last_day_val != -1 and month_day_peak > 0:
        dd = (month_day_peak - month_day_low) / month_day_peak
        if dd > month_max_daily_dd:
            month_max_daily_dd = dd
            # update last month record if present
            label = _month_label(current_month_val)
            if label in monthly:
                monthly[label]["max_daily_dd"] = float(month_max_daily_dd)

    trade_count = len(trade_pnls)
    if trade_count == 0:
        return {
            "summary": {
                "net_profit": 0.0,
                "max_dd": 0.0,
                "trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "sqn": 0.0,
                "consistency": 0.0,
                "max_daily_dd": float(max_daily_dd),
            },
            "months": months,
            "monthly": monthly,
            "trades": trade_log,
        }

    trade_rets = np.asarray(trade_pnls, dtype=np.float64)
    net_profit = equity - base_balance
    wins = float(np.sum(trade_rets > 0))
    losses = float(np.sum(trade_rets <= 0))
    win_rate = wins / trade_count if trade_count > 0 else 0.0
    gross_profit = float(np.sum(trade_rets[trade_rets > 0])) if wins > 0 else 0.0
    gross_loss = float(np.sum(np.abs(trade_rets[trade_rets < 0]))) if losses > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0

    avg_trade = float(np.mean(trade_rets)) if trade_count > 0 else 0.0
    std_trade = float(np.std(trade_rets)) if trade_count > 1 else 0.0
    sharpe = (avg_trade / std_trade) * np.sqrt(trade_count) if std_trade > 0 else 0.0

    downside = trade_rets[trade_rets < 0]
    std_down = float(np.std(downside)) if downside.size > 0 else 1.0
    sortino = (avg_trade / std_down) * np.sqrt(trade_count) if std_down > 0 else 0.0

    equity_curve = np.zeros(trade_count + 1, dtype=np.float64)
    equity_curve[0] = base_balance
    curr = base_balance
    max_dd = 0.0
    peak = base_balance
    for k in range(trade_count):
        curr += trade_rets[k]
        equity_curve[k + 1] = curr
        if curr > peak:
            peak = curr
        dd = (peak - curr) / peak if peak > 0 else 0.0
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
            r_squared = float(r * r)

    active_months = [m.get("net_profit", 0.0) for m in monthly.values()]
    negative_months = sum(1 for p in active_months if p < 0)
    consistency_score = r_squared
    if negative_months > 0:
        consistency_score *= 1.0 - (negative_months * 0.1)
        consistency_score = max(0.0, consistency_score)
    else:
        if len(active_months) > 1:
            consistency_score += 0.2

    summary = {
        "net_profit": float(net_profit),
        "max_dd": float(max_dd),
        "trades": int(trade_count),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "sqn": float(sqn),
        "consistency": float(consistency_score),
        "max_daily_dd": float(max_daily_dd),
    }

    return {
        "summary": summary,
        "months": months,
        "monthly": monthly,
        "trades": trade_log,
    }

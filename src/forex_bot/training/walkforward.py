import logging
import os
from typing import Any

import numpy as np
import pandas as pd

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)


def embargoed_walkforward_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    train_ratio: float = 0.7,
    n_splits: int = 5,
    embargo_minutes: int = 120,
    timeframe_minutes: int = 5,
    max_daily_loss_pct: float = 0.05,
    max_daily_profit_pct: float = 0.50,
    min_trading_days: int = 3,
    max_trades_per_day: int = 15,
    use_gpu: bool | None = None,
) -> dict[str, Any]:
    """
    Walk-forward backtest with embargo and dynamic Meta-Controller simulation.
    Mimics live execution behavior including risk throttling.
    """
    n = len(df)
    if n == 0 or signals is None or len(signals) != n:
        return {"walk_forward_splits": 0}

    window = max(1, n // n_splits)
    embargo_bars = max(0, embargo_minutes // max(1, timeframe_minutes))
    results = []

    for i in range(n_splits):
        start = i * window
        end = min(n, (i + 1) * window)
        if end - start < 80:
            break

        train_end = start + int(window * train_ratio)
        test_start = train_end + embargo_bars
        if test_start >= end or (train_end - start) < 40 or (end - test_start) < 40:
            continue

        df_test = df.iloc[test_start:end]
        sig = signals.iloc[test_start:end]

        # GPU toggle auto
        if use_gpu is None:
            use_gpu = CUPY_AVAILABLE and bool(int(os.environ.get("GPU_BACKTEST", "1")))

        close = df_test["close"].values
        future = np.roll(close, -1)
        ret = (future - close) / close
        if len(ret) > 0:
            ret[-1] = 0.0

        if use_gpu and CUPY_AVAILABLE:
            try:
                c_close = cp.asarray(close, dtype=cp.float32)
                c_future = cp.roll(c_close, -1)
                c_ret = (c_future - c_close) / c_close
                c_ret[-1] = 0.0
                c_ret = cp.nan_to_num(c_ret, nan=0.0, posinf=0.0, neginf=0.0)
                c_sig = cp.asarray(sig.to_numpy(), dtype=cp.int8)
                c_idx = cp.asarray(df_test.index.astype("datetime64[ns]").view("int64"))
                days = (c_idx // (24 * 60 * 60 * 1_000_000_000)).astype(cp.int64)

                equity = cp.float32(1.0)
                peak = cp.float32(1.0)
                max_dd = cp.float32(0.0)
                max_consec_losses = 0
                consec_losses = 0
                daily_stats = {}
                day_trades = {}
                daily_dd_vals = []
                rolling = cp.zeros(20, dtype=cp.float32)
                rolling_len = 0

                last_day = days[0]
                day_pnl = cp.float32(0.0)
                max_daily_loss = cp.float32(0.0)

                pnl = []

                for i in range(len(c_close) - 1):
                    day = days[i]
                    if day != last_day:
                        daily_dd_vals.append(float(day_pnl))
                        day_pnl = cp.float32(0.0)
                        last_day = day

                    # crude day start equity storage
                    ds = daily_stats.setdefault(
                        int(day), {"equity_start": float(equity), "equity_end": float(equity), "trades": 0}
                    )
                    day_start_eq = ds["equity_start"]
                    current_daily_dd = (day_start_eq - float(equity)) / day_start_eq if day_start_eq > 0 else 0.0
                    current_daily_dd = max(0.0, current_daily_dd)
                    # risk mult proxy
                    risk_mult = 1.0
                    allow_trade = True
                    if current_daily_dd >= max_daily_loss_pct or ds["trades"] >= max_trades_per_day:
                        allow_trade = False

                    trade_pnl = 0.0
                    sigv = int(c_sig[i])
                    bar_ret = float(c_ret[i])
                    if sigv != 0 and allow_trade:
                        trade_ret = sigv * bar_ret
                        base_risk = 0.01
                        realized = base_risk * risk_mult * (1.0 if trade_ret > 0 else -1.0)
                        realized -= 0.0005 * risk_mult
                        trade_pnl = realized
                        ds["trades"] += 1
                        day_trades[int(day)] = ds["trades"]
                        if trade_ret > 0:
                            consec_losses = 0
                            if rolling_len < 20:
                                rolling[rolling_len] = 1.0
                                rolling_len += 1
                            else:
                                rolling[:-1] = rolling[1:]
                                rolling[-1] = 1.0
                        else:
                            consec_losses += 1
                            max_consec_losses = max(max_consec_losses, consec_losses)
                            if rolling_len < 20:
                                rolling[rolling_len] = 0.0
                                rolling_len += 1
                            else:
                                rolling[:-1] = rolling[1:]
                                rolling[-1] = 0.0
                    pnl.append(trade_pnl)
                    equity *= 1.0 + trade_pnl
                    peak = max(peak, equity)
                    if peak > 0:
                        max_dd = max(max_dd, (peak - equity) / peak)
                    ds["equity_end"] = float(equity)
                    day_pnl += trade_pnl
                    max_daily_loss = min(max_daily_loss, day_pnl)

                # finalize day stats
                if day_pnl != 0:
                    daily_dd_vals.append(float(day_pnl))
                pnl_arr = np.array(pnl[:-1]) if len(pnl) > 1 else np.array(pnl)
                trades = int(np.count_nonzero(sig))

                daily_loss_breach = any(v <= -max_daily_loss_pct for v in daily_dd_vals)
                consistency_violation = any(v >= max_daily_profit_pct for v in daily_dd_vals)
                trade_limit_violation = any(v["trades"] > max_trades_per_day for v in daily_stats.values())
                days_with_trades = sum(1 for d in daily_stats.values() if d["trades"] > 0)
                min_days_ok = days_with_trades >= min_trading_days
                daily_min_dd = float(min(daily_dd_vals)) if daily_dd_vals else 0.0
                max_daily_loss_val = float(max_daily_loss)
                score = {
                    "split": i + 1,
                    "trades": trades,
                    "pnl": float(pnl_arr.sum()),
                    "win_rate": float((pnl_arr > 0).mean()) if len(pnl_arr) > 0 else 0.0,
                    "max_dd": float(max_dd),
                    "max_consec_losses": int(max_consec_losses),
                    "daily_min_dd": daily_min_dd,
                    "max_daily_loss": max_daily_loss_val,
                    "daily_loss_breach": bool(daily_loss_breach),
                    "consistency_violation": bool(consistency_violation),
                    "trade_limit_violation": bool(trade_limit_violation),
                    "min_trading_days_ok": bool(min_days_ok),
                    "daily_returns": daily_dd_vals,
                }
                if equity > 0:
                    score["max_daily_dd_pct"] = float(max_daily_loss_val / float(equity))
                    score["prop_compliant"] = bool(max_daily_loss_val > -(0.05 * float(equity)))
                results.append(score)
                continue
            except Exception as exc:
                use_gpu = False
                logger.warning(f"GPU walkforward backtest failed, fallback to CPU: {exc}")

        # CPU path
        pnl: list[float] = []
        equity = 1.0
        peak = equity
        max_dd = 0.0
        max_consec_losses = 0
        consec_losses = 0
        daily_stats: dict[object, dict[str, float | int]] = {}

        for idx in range(max(0, len(close) - 1)):
            ts = df_test.index[idx]
            day_key = ts.date()
            stats = daily_stats.get(day_key)
            if stats is None:
                stats = {"equity_start": equity, "equity_end": equity, "trades": 0}
                daily_stats[day_key] = stats

            sig_val = int(sig.iloc[idx])
            pnl_i = float(sig_val) * float(ret[idx])
            pnl.append(pnl_i)

            equity *= 1.0 + pnl_i
            stats["equity_end"] = equity
            if sig_val != 0:
                stats["trades"] = int(stats.get("trades", 0)) + 1

            if equity > peak:
                peak = equity
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)

            if pnl_i < 0:
                consec_losses += 1
                max_consec_losses = max(max_consec_losses, consec_losses)
            elif pnl_i > 0:
                consec_losses = 0

        pnl_arr = np.array(pnl, dtype=float)
        trades = int(np.count_nonzero(sig.to_numpy()[:-1]))

        daily_loss_breach = False
        consistency_violation = False
        trade_limit_violation = False
        daily_returns: list[float] = []
        daily_dd_list: list[float] = []
        for stats in daily_stats.values():
            eq_start = float(stats["equity_start"])
            eq_end = float(stats["equity_end"])
            daily_ret = (eq_end - eq_start) / max(eq_start, 1e-9)
            daily_returns.append(daily_ret)
            daily_dd_list.append(min(0.0, daily_ret))
            if daily_ret <= -max_daily_loss_pct:
                daily_loss_breach = True
            if daily_ret >= max_daily_profit_pct:
                consistency_violation = True
            if int(stats["trades"]) > max_trades_per_day:
                trade_limit_violation = True

        days_with_trades = sum(1 for stats in daily_stats.values() if int(stats["trades"]) > 0)
        daily_min_dd = float(min(daily_dd_list)) if daily_dd_list else 0.0
        min_days_ok = days_with_trades >= min_trading_days
        max_daily_loss = float(min(daily_returns)) if daily_returns else 0.0

        score = {
            "split": i + 1,
            "trades": trades,
            "pnl": float(pnl_arr.sum()),
            "win_rate": float((pnl_arr > 0).mean()) if len(pnl_arr) > 0 else 0.0,
            "max_dd": float(max_dd),
            "max_consec_losses": int(max_consec_losses),
            "daily_min_dd": daily_min_dd,
            "max_daily_loss": max_daily_loss,
            "daily_loss_breach": bool(daily_loss_breach),
            "consistency_violation": bool(consistency_violation),
            "trade_limit_violation": bool(trade_limit_violation),
            "min_trading_days_ok": bool(min_days_ok),
            "daily_returns": daily_returns,
        }
        if equity > 0:
            score["max_daily_dd_pct"] = float(max_daily_loss / equity)
            score["prop_compliant"] = bool(max_daily_loss > -(0.05 * equity))
        results.append(score)

    if not results:
        return {"walk_forward_splits": 0}

    agg = {
        "walk_forward_splits": len(results),
        "avg_pnl": float(np.mean([r["pnl"] for r in results])),
        "avg_win_rate": float(np.mean([r["win_rate"] for r in results])),
        "avg_max_dd": float(np.mean([r["max_dd"] for r in results])),
        "avg_max_consec_losses": float(np.mean([r["max_consec_losses"] for r in results])),
        "avg_daily_min_dd": float(np.mean([r["daily_min_dd"] for r in results])),
        "avg_max_daily_loss": float(np.mean([r["max_daily_loss"] for r in results])),
        "any_daily_loss_breach": any(r["daily_loss_breach"] for r in results),
        "any_consistency_violation": any(r["consistency_violation"] for r in results),
        "any_trade_limit_violation": any(r["trade_limit_violation"] for r in results),
        "all_min_trading_days_ok": all(r["min_trading_days_ok"] for r in results),
        "splits": results,
    }
    return agg

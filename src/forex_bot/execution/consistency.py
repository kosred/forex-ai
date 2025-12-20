import logging
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConsistencyMetrics:
    score: float
    daily_profit_consistency: float
    daily_trade_consistency: float
    daily_risk_consistency: float
    weekly_profit_consistency: float
    weekly_drawdown_consistency: float
    trade_size_consistency: float
    hold_time_consistency: float
    win_rate_rolling: float
    grade: str


class ConsistencyTracker:
    """
    Tracks trading consistency over a rolling window (prop-firm friendly).
    """

    def __init__(self, cache_dir: Path, lookback_days: int = 30):
        self.cache_dir = cache_dir
        self.lookback_days = lookback_days
        self.daily_pnl: dict[date, float] = {}
        self.daily_trades: dict[date, int] = {}
        self.daily_risk: dict[date, float] = {}
        self._max_hist = 500
        self.trade_sizes: deque[float] = deque(maxlen=self._max_hist)
        self.hold_times: deque[float] = deque(maxlen=self._max_hist)
        self.trade_outcomes: deque[int] = deque(maxlen=self._max_hist)  # 1 win, 0 loss

    def update(self, trade_event: dict) -> None:
        """
        trade_event expected keys: entry_time (iso), pnl, risk_pct, size, hold_minutes, win(bool/int)
        """
        try:
            ts = datetime.fromisoformat(trade_event.get("entry_time"))
        except Exception:
            logger.warning("ConsistencyTracker dropped trade with invalid entry_time: %s", trade_event)
            return
        d = ts.date()
        pnl = float(trade_event.get("pnl", 0.0))
        risk_pct = float(trade_event.get("risk_pct", 0.0))
        size = float(trade_event.get("size", 0.0))
        hold_minutes = float(trade_event.get("hold_minutes", 0.0))
        win_flag = int(trade_event.get("win", pnl > 0))

        self.daily_pnl[d] = self.daily_pnl.get(d, 0.0) + pnl
        self.daily_trades[d] = self.daily_trades.get(d, 0) + 1
        self.daily_risk[d] = self.daily_risk.get(d, 0.0) + risk_pct
        self.trade_sizes.append(size)
        self.hold_times.append(hold_minutes)
        self.trade_outcomes.append(win_flag)

        from datetime import timedelta

        cutoff_date = d - timedelta(days=self.lookback_days * 2)

        for key in list(self.daily_pnl.keys()):
            if key < cutoff_date:
                del self.daily_pnl[key]

        for key in list(self.daily_trades.keys()):
            if key < cutoff_date:
                del self.daily_trades[key]

        for key in list(self.daily_risk.keys()):
            if key < cutoff_date:
                del self.daily_risk[key]

    def get_metrics(self) -> ConsistencyMetrics:
        if not self.daily_pnl:
            return ConsistencyMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "F")

        days = sorted(self.daily_pnl.keys())[-self.lookback_days :]
        pnls = [self.daily_pnl[d] for d in days]
        daily_profit_consistency = float(sum(1 for p in pnls if p > 0) / max(1, len(pnls)))

        trades = np.array([self.daily_trades.get(d, 0) for d in days], dtype=float)
        trade_var = float(np.var(trades)) if len(trades) > 1 else 0.0
        daily_trade_consistency = float(1.0 / (1.0 + trade_var))

        risks = np.array([self.daily_risk.get(d, 0.0) for d in days], dtype=float)
        risk_var = float(np.var(risks)) if len(risks) > 1 else 0.0
        daily_risk_consistency = float(1.0 / (1.0 + risk_var))

        weekly = []
        for i in range(0, len(days), 5):
            weekly.append(sum(pnls[i : i + 5]))
        weekly_profit_consistency = float(sum(1 for w in weekly if w > 0) / max(1, len(weekly)))
        weekly_dd_consistency = float(1.0 / (1.0 + np.std(weekly))) if len(weekly) > 1 else 1.0

        size_var = float(np.var(self.trade_sizes)) if len(self.trade_sizes) > 1 else 0.0
        trade_size_consistency = float(1.0 / (1.0 + size_var))

        hold_var = float(np.var(self.hold_times)) if len(self.hold_times) > 1 else 0.0
        hold_time_consistency = float(1.0 / (1.0 + hold_var))

        win_rate = float(np.mean(self.trade_outcomes[-30:])) if self.trade_outcomes else 0.0

        score = (
            0.25 * daily_profit_consistency
            + 0.2 * daily_trade_consistency
            + 0.15 * daily_risk_consistency
            + 0.1 * weekly_profit_consistency
            + 0.1 * weekly_dd_consistency
            + 0.1 * trade_size_consistency
            + 0.05 * hold_time_consistency
            + 0.05 * win_rate
        ) * 100.0

        grade = "F"
        if score >= 90:
            grade = "A+"
        elif score >= 80:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 60:
            grade = "C"
        elif score >= 50:
            grade = "D"

        return ConsistencyMetrics(
            score=score,
            daily_profit_consistency=daily_profit_consistency,
            daily_trade_consistency=daily_trade_consistency,
            daily_risk_consistency=daily_risk_consistency,
            weekly_profit_consistency=weekly_profit_consistency,
            weekly_drawdown_consistency=weekly_dd_consistency,
            trade_size_consistency=trade_size_consistency,
            hold_time_consistency=hold_time_consistency,
            win_rate_rolling=win_rate,
            grade=grade,
        )

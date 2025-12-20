import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChallengeTarget:
    total_profit_target: float = 0.10
    daily_target: float = 0.10 / 20  # Assuming 20 trading days/month
    max_daily_dd: float = 0.045  # 4.5% strict
    max_total_dd: float = 0.10
    min_trading_days: int = 5
    max_trading_days: int = 60


class ChallengeOptimizer:
    """
    Prop firm challenge optimization logic.
    Dynamically adjusts risk to meet profit targets while strictly avoiding drawdown limits.
    """

    def __init__(self, target: ChallengeTarget | None = None) -> None:
        self.target = target if target else ChallengeTarget()

    def optimize_risk(self, current_profit: float, days_left: int) -> float:
        """Legacy placeholder wrapper."""
        return self.optimize_risk_allocation(current_profit, days_left, 0.0, 0.5, 2.0)

    def optimize_risk_allocation(
        self,
        current_profit: float,
        days_left: int,
        current_drawdown: float,
        win_rate: float,
        avg_risk_reward: float,
        daily_loss_pct: float = 0.0,
    ) -> float:
        """
        Calculate optimal risk % per trade.

        Args:
            current_profit: Current profit percentage (e.g. 0.02 for 2%)
            days_left: Trading days remaining in challenge
            current_drawdown: Current TOTAL drawdown percentage from peak (positive value, e.g. 0.03)
            win_rate: Historical win rate (0.0 to 1.0)
            avg_risk_reward: Average R:R ratio
            daily_loss_pct: Current DAILY loss percentage (positive value, e.g. 0.01 for -1% PnL today)
        """
        dd_room = max(0.0, self.target.max_total_dd - current_drawdown)
        if dd_room <= 0:
            return 0.0

        safe_total_cap = dd_room * 0.20

        daily_room = max(0.0, self.target.max_daily_dd - daily_loss_pct)

        if daily_room <= 0.005:  # Less than 0.5% room left? Stop trading.
            return 0.0

        safe_daily_cap = daily_room * 0.25

        safety_limit = min(safe_total_cap, safe_daily_cap)

        if days_left <= 0:
            base_risk = self._kelly_criterion(win_rate, avg_risk_reward)
            return min(base_risk, safety_limit)

        remaining_target = self.target.total_profit_target - current_profit

        if remaining_target <= 0:
            return 0.0025  # Maintenance mode

        est_trades = max(1, days_left * 3)
        expectancy = (win_rate * avg_risk_reward) - (1.0 - win_rate)

        if expectancy <= 0.1:
            required_risk = 0.01
        else:
            required_risk = remaining_target / (est_trades * expectancy)

        kelly = self._kelly_criterion(win_rate, avg_risk_reward)
        kelly_limit = kelly * 0.5

        optimal_risk = min(required_risk, kelly_limit, safety_limit)

        optimal_risk = max(0.0025, min(0.02, optimal_risk))

        if optimal_risk > safety_limit:
            optimal_risk = safety_limit

        return optimal_risk

    def _kelly_criterion(self, win_rate: float, rw_ratio: float) -> float:
        """Calculate Kelly fraction."""
        if rw_ratio <= 0:
            return 0.0
        kelly = win_rate - ((1.0 - win_rate) / rw_ratio)
        return max(0.0, kelly)

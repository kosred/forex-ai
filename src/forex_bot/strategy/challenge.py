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
        realized_trades_per_day: float = 2.0,
    ) -> float:
        """
        Calculate optimal risk % per trade.
        """
        # ...
        remaining_target = self.target.total_profit_target - current_profit

        if remaining_target <= 0:
            return 0.0025  # Maintenance mode

        # HPC FIX: Dynamic Expectancy-Based Sizing
        est_trades = max(1, int(days_left * realized_trades_per_day))
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

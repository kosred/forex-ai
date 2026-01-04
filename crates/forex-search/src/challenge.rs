#[derive(Debug, Clone, Copy)]
pub struct ChallengeTarget {
    pub total_profit_target: f64,
    pub daily_target: f64,
    pub max_daily_dd: f64,
    pub max_total_dd: f64,
    pub min_trading_days: i32,
    pub max_trading_days: i32,
}

impl Default for ChallengeTarget {
    fn default() -> Self {
        let total_profit_target = 0.10;
        Self {
            total_profit_target,
            daily_target: total_profit_target / 20.0,
            max_daily_dd: 0.045,
            max_total_dd: 0.10,
            min_trading_days: 5,
            max_trading_days: 60,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChallengeOptimizer {
    pub target: ChallengeTarget,
}

impl ChallengeOptimizer {
    pub fn new(target: Option<ChallengeTarget>) -> Self {
        Self {
            target: target.unwrap_or_default(),
        }
    }

    pub fn optimize_risk(&self, current_profit: f64, days_left: i32) -> f64 {
        self.optimize_risk_allocation(current_profit, days_left, 0.0, 0.55, 2.0, 0.0, 2.0)
    }

    pub fn optimize_risk_allocation(
        &self,
        current_profit: f64,
        days_left: i32,
        current_drawdown: f64,
        win_rate: f64,
        avg_risk_reward: f64,
        daily_loss_pct: f64,
        realized_trades_per_day: f64,
    ) -> f64 {
        let remaining_target = self.target.total_profit_target - current_profit;
        if remaining_target <= 0.0 {
            return 0.0025;
        }

        let est_trades = (days_left.max(1) as f64 * realized_trades_per_day).max(1.0);
        let expectancy = (win_rate * avg_risk_reward) - (1.0 - win_rate);

        let required_risk = if expectancy <= 0.1 {
            0.01
        } else {
            remaining_target / (est_trades * expectancy.max(1e-6))
        };

        let kelly = self.kelly_criterion(win_rate, avg_risk_reward);
        let kelly_limit = kelly * 0.5;

        let daily_room = (self.target.max_daily_dd - daily_loss_pct).max(0.0);
        let total_room = (self.target.max_total_dd - current_drawdown).max(0.0);
        let safety_limit = daily_room.min(total_room).max(0.0);

        let mut optimal_risk = required_risk.min(kelly_limit).min(safety_limit);
        optimal_risk = optimal_risk.clamp(0.0025, 0.02);
        if optimal_risk > safety_limit {
            optimal_risk = safety_limit;
        }
        optimal_risk
    }

    fn kelly_criterion(&self, win_rate: f64, rw_ratio: f64) -> f64 {
        if rw_ratio <= 0.0 {
            return 0.0;
        }
        let kelly = win_rate - ((1.0 - win_rate) / rw_ratio);
        kelly.max(0.0)
    }
}

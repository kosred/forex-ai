use crate::genetic::{month_day_indices, signals_for_gene, Gene};
use forex_data::{FeatureFrame, Ohlcv};

#[derive(Debug, Clone)]
pub struct GauntletConfig {
    pub min_win_rate: f64,
    pub min_profit_factor: f64,
    pub max_drawdown_pct: f64,
    pub max_daily_dd: f64,
    pub warn_only: bool,
    pub max_hold_bars: usize,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub pip_value: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub pip_value_per_lot: f64,
}

impl Default for GauntletConfig {
    fn default() -> Self {
        Self {
            min_win_rate: 0.55,
            min_profit_factor: 1.2,
            max_drawdown_pct: 0.07,
            max_daily_dd: 0.04,
            warn_only: false,
            max_hold_bars: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            pip_value: 0.0001,
            spread_pips: 1.5,
            commission_per_trade: 0.0,
            pip_value_per_lot: 10.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StrategyGauntlet {
    pub config: GauntletConfig,
}

impl StrategyGauntlet {
    pub fn new(config: Option<GauntletConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
        }
    }

    pub fn run(&self, features: &FeatureFrame, ohlcv: &Ohlcv, gene: &Gene) -> bool {
        if features.data.nrows() == 0 {
            return true;
        }
        if ohlcv.close.len() != features.data.nrows() {
            return false;
        }

        let signals = signals_for_gene(features, gene);
        let (months, days) = month_day_indices(&features.timestamps);

        let metrics = crate::eval::fast_evaluate_strategy_core(
            &ohlcv.close,
            &ohlcv.high,
            &ohlcv.low,
            &signals,
            &months,
            &days,
            gene.sl_pips,
            gene.tp_pips,
            self.config.max_hold_bars,
            self.config.trailing_enabled,
            self.config.trailing_atr_multiplier,
            self.config.trailing_be_trigger_r,
            self.config.pip_value,
            self.config.spread_pips,
            self.config.commission_per_trade,
            self.config.pip_value_per_lot,
        );

        let net_profit = metrics[0];
        let max_dd = metrics[3];
        let win_rate = metrics[4];
        let profit_factor = metrics[5];
        let max_daily_dd = metrics[10];

        if win_rate < self.config.min_win_rate {
            return self.config.warn_only;
        }
        if max_dd > self.config.max_drawdown_pct {
            return self.config.warn_only;
        }
        if max_daily_dd > self.config.max_daily_dd {
            return self.config.warn_only;
        }
        if profit_factor <= self.config.min_profit_factor {
            return self.config.warn_only;
        }
        if net_profit <= 0.0 {
            return self.config.warn_only;
        }
        true
    }
}

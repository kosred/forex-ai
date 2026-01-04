use chrono::{Datelike, TimeZone, Utc};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub entry_time: i64,
    pub exit_time: Option<i64>,
    pub pnl: f64,
    pub pnl_pct: Option<f64>,
    pub duration_hours: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub strategy_id: String,
    pub total_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub total_return_pct: f64,
    pub avg_win_pct: f64,
    pub avg_loss_pct: f64,
    pub largest_win_pct: f64,
    pub largest_loss_pct: f64,
    pub max_drawdown_pct: f64,
    pub avg_drawdown_pct: f64,
    pub longest_losing_streak: usize,
    pub longest_winning_streak: usize,
    pub expectancy: f64,
    pub kelly_fraction: f64,
    pub statistical_significance: f64,
    pub monthly_win_rate: f64,
    pub positive_months: usize,
    pub negative_months: usize,
    pub avg_monthly_return_pct: f64,
    pub profit_per_trade: f64,
    pub avg_trade_duration_hours: f64,
    pub trades_per_month: f64,
    pub quality_score: f64,
    pub has_edge: bool,
    pub recommendation: String,
}

#[derive(Debug, Clone)]
pub struct StrategyQualityAnalyzer {
    pub min_sharpe: f64,
    pub min_sortino: f64,
    pub min_calmar: f64,
    pub min_profit_factor: f64,
    pub min_win_rate: f64,
    pub min_trades: usize,
    pub max_dd_acceptable: f64,
    pub min_monthly_return_pct: f64,
    pub edge_significance_pvalue: f64,
}

impl Default for StrategyQualityAnalyzer {
    fn default() -> Self {
        Self {
            min_sharpe: 1.2,
            min_sortino: 1.2,
            min_calmar: 1.0,
            min_profit_factor: 1.5,
            min_win_rate: 0.50,
            min_trades: 0,
            max_dd_acceptable: 0.15,
            min_monthly_return_pct: 0.04,
            edge_significance_pvalue: 0.01,
        }
    }
}

impl StrategyQualityAnalyzer {
    pub fn analyze_strategy(&self, strategy_id: &str, trades: &[Trade], initial_balance: f64) -> StrategyMetrics {
        if trades.is_empty() {
            return empty_metrics(strategy_id);
        }

        let mut pnls = Vec::with_capacity(trades.len());
        let mut returns = Vec::with_capacity(trades.len());
        let mut durations = Vec::with_capacity(trades.len());

        for trade in trades {
            let pnl_pct = trade.pnl_pct.unwrap_or(trade.pnl / initial_balance);
            pnls.push(trade.pnl);
            returns.push(pnl_pct);
            if let Some(dur) = trade.duration_hours {
                durations.push(dur);
            } else if let Some(exit) = trade.exit_time {
                if trade.entry_time > 0 && exit >= trade.entry_time {
                    let hours = (exit - trade.entry_time) as f64 / 3_600_000.0;
                    durations.push(hours);
                }
            }
        }

        let total_trades = returns.len();
        let wins: Vec<f64> = returns.iter().cloned().filter(|v| *v > 0.0).collect();
        let losses: Vec<f64> = returns.iter().cloned().filter(|v| *v <= 0.0).collect();

        let win_rate = if total_trades > 0 {
            wins.len() as f64 / total_trades as f64
        } else {
            0.0
        };

        let avg_win_pct = if !wins.is_empty() { mean(&wins) } else { 0.0 };
        let losses_cleaned: Vec<f64> = if losses.iter().any(|v| *v < 0.0) {
            losses.iter().cloned().filter(|v| *v < 0.0).collect()
        } else {
            returns.iter().map(|v| -v.abs()).collect()
        };
        let avg_loss_pct = if !losses_cleaned.is_empty() { mean(&losses_cleaned) } else { 0.0 };
        let avg_loss_mag = avg_loss_pct.abs();

        let gross_profit: f64 = pnls.iter().cloned().filter(|v| *v > 0.0).sum();
        let gross_loss: f64 = pnls.iter().cloned().filter(|v| *v < 0.0).map(|v| v.abs()).sum();
        let eps = 1e-7;
        let mut profit_factor = (gross_profit + eps) / (gross_loss + eps);
        if profit_factor > 100.0 {
            profit_factor = 100.0;
        }

        let mut equity = initial_balance;
        let mut peak = initial_balance;
        let mut drawdowns = Vec::with_capacity(total_trades);
        for pnl in &pnls {
            equity += *pnl;
            if equity > peak {
                peak = equity;
            }
            let dd = if peak > 0.0 { (peak - equity) / peak } else { 0.0 };
            drawdowns.push(dd);
        }
        let max_dd = drawdowns.iter().cloned().fold(0.0, f64::max);
        let avg_dd = if !drawdowns.is_empty() { mean(&drawdowns) } else { 0.0 };

        let sharpe = calculate_sharpe(&returns);
        let sortino = calculate_sortino(&returns);

        let total_return = pnls.iter().sum::<f64>();
        let total_return_pct = total_return / initial_balance;
        let calmar = if max_dd > 1e-9 { total_return_pct / max_dd } else { 0.0 };

        let longest_win_streak = longest_streak(&pnls, true);
        let longest_loss_streak = longest_streak(&pnls, false);

        let expectancy = (win_rate * avg_win_pct) - ((1.0 - win_rate) * avg_loss_mag);
        let kelly = calculate_kelly(win_rate, avg_win_pct, avg_loss_mag);
        let p_value = test_statistical_significance(&returns);

        let monthly_metrics = analyze_monthly_consistency(trades, initial_balance);
        let monthly_win_rate = monthly_metrics.monthly_win_rate;
        let avg_monthly_return_pct = monthly_metrics.avg_return_pct;

        let avg_duration = if durations.is_empty() { 0.0 } else { mean(&durations) };
        let trades_per_month = calculate_trade_frequency(trades);

        let mut metrics = StrategyMetrics {
            strategy_id: strategy_id.to_string(),
            total_trades,
            win_rate,
            profit_factor,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            calmar_ratio: calmar,
            total_return_pct,
            avg_win_pct,
            avg_loss_pct,
            largest_win_pct: returns.iter().cloned().fold(0.0, f64::max),
            largest_loss_pct: returns.iter().cloned().fold(0.0, f64::min),
            max_drawdown_pct: max_dd,
            avg_drawdown_pct: avg_dd,
            longest_losing_streak: longest_loss_streak,
            longest_winning_streak: longest_win_streak,
            expectancy,
            kelly_fraction: kelly,
            statistical_significance: p_value,
            monthly_win_rate,
            positive_months: monthly_metrics.positive,
            negative_months: monthly_metrics.negative,
            avg_monthly_return_pct,
            profit_per_trade: if !pnls.is_empty() { mean(&pnls) } else { 0.0 },
            avg_trade_duration_hours: avg_duration,
            trades_per_month,
            quality_score: 0.0,
            has_edge: false,
            recommendation: String::new(),
        };

        score_strategy(self, &mut metrics);
        metrics
    }
}

#[derive(Debug, Clone)]
struct MonthlyMetrics {
    monthly_win_rate: f64,
    positive: usize,
    negative: usize,
    avg_return_pct: f64,
}

fn analyze_monthly_consistency(trades: &[Trade], initial_balance: f64) -> MonthlyMetrics {
    if trades.is_empty() {
        return MonthlyMetrics {
            monthly_win_rate: 0.0,
            positive: 0,
            negative: 0,
            avg_return_pct: 0.0,
        };
    }

    let mut monthly_pnl: HashMap<i64, f64> = HashMap::new();
    for trade in trades {
        if trade.entry_time <= 0 {
            continue;
        }
        if let Some(dt) = Utc.timestamp_millis_opt(trade.entry_time).single() {
            let key = (dt.year() as i64) * 12 + dt.month() as i64;
            *monthly_pnl.entry(key).or_insert(0.0) += trade.pnl;
        }
    }

    if monthly_pnl.is_empty() {
        return MonthlyMetrics {
            monthly_win_rate: 0.0,
            positive: 0,
            negative: 0,
            avg_return_pct: 0.0,
        };
    }

    let mut positive = 0;
    let mut negative = 0;
    let mut sum = 0.0;
    for val in monthly_pnl.values() {
        sum += *val;
        if *val > 0.0 {
            positive += 1;
        } else {
            negative += 1;
        }
    }
    let total = monthly_pnl.len();
    let avg_return_pct = if total > 0 { (sum / total as f64) / initial_balance } else { 0.0 };

    MonthlyMetrics {
        monthly_win_rate: if total > 0 { positive as f64 / total as f64 } else { 0.0 },
        positive,
        negative,
        avg_return_pct,
    }
}

fn calculate_trade_frequency(trades: &[Trade]) -> f64 {
    if trades.is_empty() {
        return 0.0;
    }

    let mut days = std::collections::HashSet::new();
    for trade in trades {
        if trade.entry_time <= 0 {
            continue;
        }
        if let Some(dt) = Utc.timestamp_millis_opt(trade.entry_time).single() {
            if dt.weekday().num_days_from_monday() < 5 {
                let day_key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
                days.insert(day_key);
            }
        }
    }

    if days.is_empty() {
        return 0.0;
    }

    let trading_days = days.len() as f64;
    let days_per_month = std::env::var("FOREX_BOT_TRADING_DAYS_PER_MONTH")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(21.0)
        .max(1.0);
    let months = (trading_days / days_per_month).max(1e-6);
    trades.len() as f64 / months
}

fn calculate_sharpe(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean_ret = mean(returns);
    let std_ret = stddev_sample(returns, mean_ret);
    if std_ret < 1e-9 {
        return 0.0;
    }
    (mean_ret / std_ret) * 252_f64.sqrt()
}

fn calculate_sortino(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean_ret = mean(returns);
    let downside: Vec<f64> = returns.iter().cloned().filter(|v| *v < 0.0).collect();
    if downside.len() < 2 {
        return 0.0;
    }
    let std_down = stddev_sample(&downside, mean(&downside));
    if std_down < 1e-9 {
        return 0.0;
    }
    (mean_ret / std_down) * 252_f64.sqrt()
}

fn longest_streak(pnls: &[f64], win: bool) -> usize {
    let mut max_streak = 0;
    let mut current = 0;
    for pnl in pnls {
        let is_win = *pnl > 0.0;
        if (win && is_win) || (!win && !is_win) {
            current += 1;
            if current > max_streak {
                max_streak = current;
            }
        } else {
            current = 0;
        }
    }
    max_streak
}

fn calculate_kelly(win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
    if avg_loss < 1e-6 || win_rate <= 0.0 || win_rate >= 1.0 {
        return 0.0;
    }
    let b = avg_win / avg_loss;
    let p = win_rate;
    let q = 1.0 - p;
    let mut kelly = (p * b - q) / b;
    kelly = kelly.clamp(0.0, 1.0);
    kelly * 0.25
}

fn test_statistical_significance(returns: &[f64]) -> f64 {
    if returns.len() < 10 {
        return 1.0;
    }
    let mean_ret = mean(returns);
    let std_ret = stddev_sample(returns, mean_ret);
    if std_ret <= 0.0 {
        return 1.0;
    }
    let n = returns.len() as f64;
    let t_stat = mean_ret / (std_ret / n.sqrt());
    if t_stat <= 0.0 {
        return 1.0;
    }
    let df = n - 1.0;
    let dist = StudentsT::new(0.0, 1.0, df).unwrap();
    1.0 - dist.cdf(t_stat)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev_sample(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for v in values {
        let d = *v - mean;
        sum += d * d;
    }
    (sum / (values.len() as f64 - 1.0)).sqrt()
}

fn score_strategy(analyzer: &StrategyQualityAnalyzer, metrics: &mut StrategyMetrics) {
    let mut score = 0.0;
    if metrics.sortino_ratio >= 3.0 {
        score += 30.0;
    } else if metrics.sortino_ratio >= 2.0 {
        score += 24.0;
    } else if metrics.sortino_ratio >= 1.5 {
        score += 18.0;
    } else if metrics.sortino_ratio >= 1.2 {
        score += 12.0;
    }

    if metrics.profit_factor >= 2.5 {
        score += 20.0;
    } else if metrics.profit_factor >= 2.0 {
        score += 15.0;
    } else if metrics.profit_factor >= 1.8 {
        score += 12.0;
    } else if metrics.profit_factor >= 1.5 {
        score += 10.0;
    } else if metrics.profit_factor >= 1.3 {
        score += 5.0;
    }

    if metrics.win_rate >= 0.65 {
        score += 15.0;
    } else if metrics.win_rate >= 0.60 {
        score += 12.0;
    } else if metrics.win_rate >= 0.55 {
        score += 10.0;
    } else if metrics.win_rate >= 0.50 {
        score += 8.0;
    }

    if metrics.calmar_ratio >= 2.0 {
        score += 20.0;
    } else if metrics.calmar_ratio >= 1.5 {
        score += 15.0;
    } else if metrics.calmar_ratio >= 1.0 {
        score += 10.0;
    }

    if metrics.max_drawdown_pct <= 0.08 {
        score += 15.0;
    } else if metrics.max_drawdown_pct <= 0.10 {
        score += 13.0;
    } else if metrics.max_drawdown_pct <= 0.12 {
        score += 10.0;
    } else if metrics.max_drawdown_pct <= 0.15 {
        score += 7.0;
    }

    if metrics.statistical_significance <= 0.01 {
        score += 10.0;
    } else if metrics.statistical_significance <= 0.05 {
        score += 7.0;
    }

    if metrics.monthly_win_rate >= 0.70 {
        score += 10.0;
    } else if metrics.monthly_win_rate >= 0.60 {
        score += 7.0;
    } else if metrics.monthly_win_rate >= 0.50 {
        score += 5.0;
    }

    if metrics.avg_monthly_return_pct >= analyzer.min_monthly_return_pct {
        score += 10.0;
    }

    metrics.quality_score = score.min(100.0);

    metrics.has_edge = metrics.sortino_ratio >= analyzer.min_sortino
        && metrics.calmar_ratio >= analyzer.min_calmar
        && metrics.profit_factor >= analyzer.min_profit_factor
        && metrics.win_rate >= analyzer.min_win_rate
        && metrics.max_drawdown_pct <= analyzer.max_dd_acceptable
        && metrics.avg_monthly_return_pct >= analyzer.min_monthly_return_pct
        && metrics.statistical_significance <= analyzer.edge_significance_pvalue
        && (analyzer.min_trades == 0 || metrics.total_trades >= analyzer.min_trades);

    metrics.recommendation = if metrics.quality_score >= 80.0 {
        "EXCELLENT"
    } else if metrics.quality_score >= 70.0 {
        "GOOD"
    } else if metrics.quality_score >= 60.0 {
        "ACCEPTABLE"
    } else {
        "POOR"
    }
    .to_string();
}

fn empty_metrics(strategy_id: &str) -> StrategyMetrics {
    StrategyMetrics {
        strategy_id: strategy_id.to_string(),
        total_trades: 0,
        win_rate: 0.0,
        profit_factor: 0.0,
        sharpe_ratio: 0.0,
        sortino_ratio: 0.0,
        calmar_ratio: 0.0,
        total_return_pct: 0.0,
        avg_win_pct: 0.0,
        avg_loss_pct: 0.0,
        largest_win_pct: 0.0,
        largest_loss_pct: 0.0,
        max_drawdown_pct: 0.0,
        avg_drawdown_pct: 0.0,
        longest_losing_streak: 0,
        longest_winning_streak: 0,
        expectancy: 0.0,
        kelly_fraction: 0.0,
        statistical_significance: 1.0,
        monthly_win_rate: 0.0,
        positive_months: 0,
        negative_months: 0,
        avg_monthly_return_pct: 0.0,
        profit_per_trade: 0.0,
        avg_trade_duration_hours: 0.0,
        trades_per_month: 0.0,
        quality_score: 0.0,
        has_edge: false,
        recommendation: String::new(),
    }
}

pub struct StrategyRanker {
    pub analyzer: StrategyQualityAnalyzer,
    pub strategy_metrics: HashMap<String, StrategyMetrics>,
}

impl StrategyRanker {
    pub fn new(analyzer: Option<StrategyQualityAnalyzer>) -> Self {
        Self {
            analyzer: analyzer.unwrap_or_default(),
            strategy_metrics: HashMap::new(),
        }
    }

    pub fn evaluate_strategies(
        &mut self,
        strategies: &HashMap<String, Vec<Trade>>,
        initial_balance: f64,
    ) -> Vec<StrategyMetrics> {
        let mut results = Vec::new();
        for (strategy_id, trades) in strategies {
            let metrics = self.analyzer.analyze_strategy(strategy_id, trades, initial_balance);
            self.strategy_metrics.insert(strategy_id.clone(), metrics.clone());
            results.push(metrics);
        }
        results.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn get_top_strategies(&self, n: usize, min_quality: f64) -> Vec<String> {
        let mut ranked: Vec<_> = self.strategy_metrics.iter().collect();
        ranked.sort_by(|a, b| b.1.quality_score.partial_cmp(&a.1.quality_score).unwrap_or(std::cmp::Ordering::Equal));
        ranked
            .into_iter()
            .filter(|(_, m)| m.quality_score >= min_quality && m.has_edge)
            .take(n)
            .map(|(sid, _)| sid.clone())
            .collect()
    }

    pub fn save_rankings(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut rankings = Vec::new();
        for m in self.strategy_metrics.values() {
            rankings.push(serde_json::json!({
                "strategy_id": m.strategy_id,
                "quality_score": m.quality_score,
                "has_edge": m.has_edge,
                "recommendation": m.recommendation,
            }));
        }
        let payload = serde_json::to_string_pretty(&rankings).unwrap_or_else(|_| "[]".to_string());
        fs::write(path, payload)
    }
}

use crate::genetic::{evolve_search, signals_for_gene, Gene};
use anyhow::Result;
use chrono::{Datelike, TimeZone, Utc};
use forex_data::{FeatureFrame, Ohlcv};
use serde::Serialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub population: usize,
    pub generations: usize,
    pub max_indicators: usize,
    pub candidate_count: usize,
    pub portfolio_size: usize,
    pub corr_threshold: f64,
    pub min_trades_per_day: f64,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            population: 100,
            generations: 5,
            max_indicators: 12,
            candidate_count: 200,
            portfolio_size: 100,
            corr_threshold: 0.7,
            min_trades_per_day: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    pub portfolio: Vec<Gene>,
    pub candidates: Vec<Gene>,
}

#[derive(Debug, Serialize)]
struct GeneExport<'a> {
    strategy_id: &'a str,
    indicators: Vec<&'a str>,
    indices: Vec<usize>,
    weights: Vec<f32>,
    long_threshold: f32,
    short_threshold: f32,
    fitness: f64,
    sharpe_ratio: f64,
    win_rate: f64,
    tp_pips: f64,
    sl_pips: f64,
}

pub fn run_discovery_cycle(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<DiscoveryResult> {
    let search = evolve_search(
        features,
        ohlcv,
        config.population,
        config.generations,
        config.max_indicators,
    )?;

    let mut candidates = search.genes;
    candidates.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
    let max_candidates = config.candidate_count.max(10).min(candidates.len());
    candidates.truncate(max_candidates);

    let min_trades = min_trades_required(&features.timestamps, config.min_trades_per_day, features.data.nrows());
    let mut filtered = Vec::new();
    let mut signals_map = Vec::new();
    for gene in &candidates {
        let sig = signals_for_gene(features, gene);
        let trade_count = sig.iter().filter(|v| **v != 0).count() as f64;
        if trade_count >= min_trades as f64 {
            filtered.push(gene.clone());
            signals_map.push(sig);
        }
    }

    let mut portfolio = Vec::new();
    let mut portfolio_signals: Vec<Vec<i8>> = Vec::new();
    for (gene, sig) in filtered.into_iter().zip(signals_map.into_iter()) {
        if portfolio.len() >= config.portfolio_size {
            break;
        }
        let mut ok = true;
        for existing in &portfolio_signals {
            let corr = pearson_corr_i8(&sig, existing);
            if corr.abs() >= config.corr_threshold {
                ok = false;
                break;
            }
        }
        if ok {
            portfolio_signals.push(sig);
            portfolio.push(gene);
        }
    }

    Ok(DiscoveryResult { portfolio, candidates })
}

fn min_trades_required(timestamps: &[i64], min_trades_per_day: f64, n_rows: usize) -> usize {
    if timestamps.is_empty() {
        let days = (n_rows as f64 / 1440.0).max(1.0);
        return (days * min_trades_per_day).ceil() as usize;
    }
    let mut days = HashSet::new();
    for ts in timestamps {
        if let Some(dt) = Utc.timestamp_millis_opt(*ts).single() {
            if dt.weekday().num_days_from_monday() < 5 {
                let key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
                days.insert(key);
            }
        }
    }
    let day_count = days.len().max(1) as f64;
    (day_count * min_trades_per_day).ceil() as usize
}

fn pearson_corr_i8(a: &[i8], b: &[i8]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for i in 0..n {
        sum_a += a[i] as f64;
        sum_b += b[i] as f64;
    }
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;
    let mut num = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    for i in 0..n {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if denom_a <= 1e-12 || denom_b <= 1e-12 {
        return 0.0;
    }
    num / (denom_a.sqrt() * denom_b.sqrt())
}

pub fn save_portfolio_json(
    path: impl AsRef<Path>,
    portfolio: &[Gene],
    feature_names: &[String],
) -> Result<()> {
    let mut exports = Vec::new();
    for gene in portfolio {
        let mut names = Vec::new();
        for idx in &gene.indices {
            if let Some(name) = feature_names.get(*idx) {
                names.push(name.as_str());
            }
        }
        exports.push(GeneExport {
            strategy_id: &gene.strategy_id,
            indicators: names,
            indices: gene.indices.clone(),
            weights: gene.weights.clone(),
            long_threshold: gene.long_threshold,
            short_threshold: gene.short_threshold,
            fitness: gene.fitness,
            sharpe_ratio: gene.sharpe_ratio,
            win_rate: gene.win_rate,
            tp_pips: gene.tp_pips,
            sl_pips: gene.sl_pips,
        });
    }
    let payload = serde_json::to_string_pretty(&exports)?;
    fs::write(path, payload)?;
    Ok(())
}

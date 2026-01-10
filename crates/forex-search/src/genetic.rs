use anyhow::{bail, Result};
use chrono::{Datelike, TimeZone, Utc};
use forex_data::{FeatureFrame, Ohlcv};
use ndarray::Array2;
use rand::seq::index::sample;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gene {
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
    pub long_threshold: f32,
    pub short_threshold: f32,
    pub fitness: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub trades_count: usize,
    pub generation: usize,
    pub strategy_id: String,
    pub use_ob: bool,
    pub use_fvg: bool,
    pub use_liq_sweep: bool,
    pub mtf_confirmation: bool,
    pub use_premium_discount: bool,
    pub use_inducement: bool,
    pub tp_pips: f64,
    pub sl_pips: f64,
    pub slice_pass_rate: f64,
}

impl Gene {
    fn normalize(&mut self, n_indicators: usize, min_indicators: usize) {
        if self.indices.is_empty() {
            self.indices.push(0);
        }
        if self.weights.len() != self.indices.len() {
            self.weights = vec![1.0; self.indices.len()];
        }
        for idx in &mut self.indices {
            if *idx >= n_indicators {
                *idx = idx.saturating_sub(n_indicators.saturating_sub(1));
            }
        }
        let min_indicators = min_indicators.min(n_indicators.max(1));
        if self.indices.len() < min_indicators {
            let mut rng = rand::thread_rng();
            let mut seen = std::collections::HashSet::new();
            for idx in &self.indices {
                seen.insert(*idx);
            }
            while self.indices.len() < min_indicators {
                let idx = rng.gen_range(0..n_indicators.max(1));
                if seen.insert(idx) {
                    self.indices.push(idx);
                    self.weights.push(rng.gen_range(0.1..1.0));
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub genes: Vec<Gene>,
    pub metrics: Vec<[f64; 11]>,
}

#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub max_hold_bars: usize,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub pip_value: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub pip_value_per_lot: f64,
    pub smc_gate_threshold: f32,
    pub smc_weight_ob: f32,
    pub smc_weight_fvg: f32,
    pub smc_weight_liq: f32,
    pub smc_weight_mtf: f32,
    pub smc_weight_premium: f32,
    pub smc_weight_inducement: f32,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            max_hold_bars: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            pip_value: 0.0001,
            spread_pips: 1.5,
            commission_per_trade: 0.0,
            pip_value_per_lot: 10.0,
            smc_gate_threshold: 0.0,
            smc_weight_ob: 1.0,
            smc_weight_fvg: 1.0,
            smc_weight_liq: 1.0,
            smc_weight_mtf: 1.0,
            smc_weight_premium: 1.0,
            smc_weight_inducement: 1.0,
        }
    }
}

pub fn month_day_indices(timestamps: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let mut months = Vec::with_capacity(timestamps.len());
    let mut days = Vec::with_capacity(timestamps.len());
    for ts in timestamps {
        let dt = Utc.timestamp_millis_opt(*ts).single();
        if let Some(dt) = dt {
            let month_key = (dt.year() as i64) * 12 + dt.month() as i64;
            let day_key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
            months.push(month_key);
            days.push(day_key);
        } else {
            months.push(0);
            days.push(0);
        }
    }
    (months, days)
}

fn build_gene_arrays(genes: &[Gene]) -> (Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut offsets = Vec::with_capacity(genes.len() + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let mut long_thr = Vec::with_capacity(genes.len());
    let mut short_thr = Vec::with_capacity(genes.len());
    offsets.push(0);
    for gene in genes {
        long_thr.push(gene.long_threshold);
        short_thr.push(gene.short_threshold);
        for (idx, weight) in gene.indices.iter().zip(gene.weights.iter()) {
            indices.push(*idx as i32);
            weights.push(*weight);
        }
        offsets.push(indices.len() as i32);
    }
    (offsets, indices, weights, long_thr, short_thr)
}

fn transpose_features(frame: &FeatureFrame) -> Array2<f32> {
    frame.data.t().to_owned()
}

fn new_random_gene(n_indicators: usize, max_indicators: usize, generation: usize) -> Gene {
    let mut rng = rand::thread_rng();
    let min_indicators = 4.min(n_indicators.max(1));
    let max_indicators = max_indicators.max(min_indicators).min(n_indicators.max(1));
    let count = rng.gen_range(min_indicators..=max_indicators);
    let sample = sample(&mut rng, n_indicators.max(1), count);
    let indices: Vec<usize> = sample.iter().collect();
    let weights: Vec<f32> = (0..count).map(|_| rng.gen_range(0.1..1.0)).collect();
    let long_threshold = rng.gen_range(0.4..0.8);
    let short_threshold = -rng.gen_range(0.4..0.8);
    let strategy_id = format!("gene_{}_{}", rng.gen_range(0..1_000_000u64), generation);
    Gene {
        indices,
        weights,
        long_threshold,
        short_threshold,
        fitness: 0.0,
        sharpe_ratio: 0.0,
        win_rate: 0.0,
        max_drawdown: 0.0,
        profit_factor: 0.0,
        expectancy: 0.0,
        trades_count: 0,
        generation,
        strategy_id,
        use_ob: false,
        use_fvg: false,
        use_liq_sweep: false,
        mtf_confirmation: true,
        use_premium_discount: false,
        use_inducement: false,
        tp_pips: 40.0,
        sl_pips: 20.0,
        slice_pass_rate: 0.0,
    }
}

pub fn generate_random_genes(
    n_genes: usize,
    n_indicators: usize,
    max_indicators: usize,
    generation: usize,
) -> Vec<Gene> {
    (0..n_genes)
        .map(|_| new_random_gene(n_indicators, max_indicators, generation))
        .collect()
}

pub fn signals_for_gene(features: &FeatureFrame, gene: &Gene) -> Vec<i8> {
    let n_samples = features.data.nrows();
    let mut combined = vec![0.0_f32; n_samples];
    for (idx, weight) in gene.indices.iter().zip(gene.weights.iter()) {
        if *idx >= features.data.ncols() {
            continue;
        }
        let col = features.data.column(*idx);
        for (i, v) in col.iter().enumerate() {
            combined[i] += *weight * *v;
        }
    }
    let mut signals = vec![0_i8; n_samples];
    for i in 0..n_samples {
        let v = combined[i];
        if v >= gene.long_threshold {
            signals[i] = 1;
        } else if v <= gene.short_threshold {
            signals[i] = -1;
        }
    }
    signals
}

pub fn evaluate_genes(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    genes: &[Gene],
    config: &EvaluationConfig,
) -> Result<Vec<[f64; 11]>> {
    if features.data.nrows() == 0 || features.data.ncols() == 0 {
        bail!("empty feature matrix");
    }
    let n_samples = features.data.nrows();
    if ohlcv.close.len() != n_samples {
        bail!("ohlcv length does not match feature rows");
    }

    let indicators = transpose_features(features);
    let (offsets, indices, weights, long_thr, short_thr) = build_gene_arrays(genes);
    let sl_pips: Vec<f64> = genes.iter().map(|g| g.sl_pips).collect();
    let tp_pips: Vec<f64> = genes.iter().map(|g| g.tp_pips).collect();
    let (months, days) = month_day_indices(&features.timestamps);

    let zeros_samples = vec![0_i8; n_samples];
    let use_ob: Vec<i8> = genes.iter().map(|g| if g.use_ob { 1 } else { 0 }).collect();
    let use_fvg: Vec<i8> = genes.iter().map(|g| if g.use_fvg { 1 } else { 0 }).collect();
    let use_liq: Vec<i8> = genes.iter().map(|g| if g.use_liq_sweep { 1 } else { 0 }).collect();
    let use_mtf: Vec<i8> = genes.iter().map(|g| if g.mtf_confirmation { 1 } else { 0 }).collect();
    let use_premium: Vec<i8> = genes.iter().map(|g| if g.use_premium_discount { 1 } else { 0 }).collect();
    let use_inducement: Vec<i8> = genes.iter().map(|g| if g.use_inducement { 1 } else { 0 }).collect();

    let metrics = crate::eval::evaluate_population_core(
        &ohlcv.close,
        &ohlcv.high,
        &ohlcv.low,
        indicators.view(),
        &offsets,
        &indices,
        &weights,
        &long_thr,
        &short_thr,
        &months,
        &days,
        &sl_pips,
        &tp_pips,
        &zeros_samples,
        &zeros_samples,
        &zeros_samples,
        &zeros_samples,
        &zeros_samples,
        &zeros_samples,
        &use_ob,
        &use_fvg,
        &use_liq,
        &use_mtf,
        &use_premium,
        &use_inducement,
        config.smc_gate_threshold,
        config.smc_weight_ob,
        config.smc_weight_fvg,
        config.smc_weight_liq,
        config.smc_weight_mtf,
        config.smc_weight_premium,
        config.smc_weight_inducement,
        config.max_hold_bars,
        config.trailing_enabled,
        config.trailing_atr_multiplier,
        config.trailing_be_trigger_r,
        config.pip_value,
        config.spread_pips,
        config.commission_per_trade,
        config.pip_value_per_lot,
    )
    .map_err(|e| anyhow::anyhow!(e))?;

    Ok(metrics)
}

fn score_from_metrics(metrics: &[f64; 11]) -> f64 {
    let net_profit = metrics[0];
    let sharpe = metrics[1];
    let max_dd = metrics[3];
    let profit_factor = metrics[5];
    let dd_cap = 0.07;
    let pfloor = 1.0;
    let dd_penalty = 10.0 * (max_dd - dd_cap).max(0.0);
    let pf_penalty = if profit_factor <= pfloor { 5.0 } else { 0.0 };
    sharpe + (net_profit / 10_000.0) - dd_penalty - pf_penalty
}

fn apply_metrics(genes: &mut [Gene], metrics: &[[f64; 11]]) {
    for (gene, m) in genes.iter_mut().zip(metrics.iter()) {
        gene.fitness = score_from_metrics(m);
        gene.sharpe_ratio = m[1];
        gene.max_drawdown = m[3];
        gene.win_rate = m[4];
        gene.profit_factor = m[5];
        gene.expectancy = m[6];
        gene.trades_count = m[8].max(0.0) as usize;
        gene.slice_pass_rate = 1.0;
    }
}

fn crossover(a: &Gene, b: &Gene, generation: usize) -> Gene {
    let mut rng = rand::thread_rng();
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    let split_a = a.indices.len() / 2;
    let split_b = b.indices.len() / 2;
    indices.extend_from_slice(&a.indices[..split_a]);
    indices.extend_from_slice(&b.indices[split_b..]);
    weights.extend_from_slice(&a.weights[..split_a]);
    weights.extend_from_slice(&b.weights[split_b..]);
    if indices.is_empty() {
        indices.push(*a.indices.first().unwrap_or(&0));
        weights.push(*a.weights.first().unwrap_or(&1.0));
    }
    let long_threshold = if rng.gen_bool(0.5) { a.long_threshold } else { b.long_threshold };
    let short_threshold = if rng.gen_bool(0.5) { a.short_threshold } else { b.short_threshold };
    let strategy_id = format!("gene_{}_{}", rng.gen_range(0..1_000_000u64), generation);
    Gene {
        indices,
        weights,
        long_threshold,
        short_threshold,
        fitness: 0.0,
        sharpe_ratio: 0.0,
        win_rate: 0.0,
        max_drawdown: 0.0,
        profit_factor: 0.0,
        expectancy: 0.0,
        trades_count: 0,
        generation,
        strategy_id,
        use_ob: a.use_ob,
        use_fvg: a.use_fvg,
        use_liq_sweep: a.use_liq_sweep,
        mtf_confirmation: a.mtf_confirmation,
        use_premium_discount: a.use_premium_discount,
        use_inducement: a.use_inducement,
        tp_pips: a.tp_pips,
        sl_pips: a.sl_pips,
        slice_pass_rate: 0.0,
    }
}

fn mutate(gene: &Gene, n_indicators: usize, max_indicators: usize, generation: usize) -> Gene {
    let mut rng = rand::thread_rng();
    let mut mutated = gene.clone();
    let mutation_type = rng.gen_range(0..4);
    match mutation_type {
        0 => {
            if !mutated.indices.is_empty() && rng.gen_bool(0.5) {
                let idx = rng.gen_range(0..mutated.indices.len());
                mutated.indices[idx] = rng.gen_range(0..n_indicators.max(1));
                mutated.weights[idx] = rng.gen_range(0.1..1.0);
            } else {
                let min_indicators = 4.min(n_indicators.max(1));
                let max_indicators = max_indicators.max(min_indicators).min(n_indicators.max(1));
                let count = rng.gen_range(min_indicators..=max_indicators);
                let sample = sample(&mut rng, n_indicators.max(1), count);
                mutated.indices = sample.iter().collect();
                mutated.weights = (0..count).map(|_| rng.gen_range(0.1..1.0)).collect();
            }
        }
        1 => {
            mutated.long_threshold = (mutated.long_threshold * rng.gen_range(0.9..1.1)).clamp(0.4, 0.9);
            mutated.short_threshold = (mutated.short_threshold * rng.gen_range(0.9..1.1)).clamp(-0.9, -0.4);
        }
        2 => {
            mutated.tp_pips = (mutated.tp_pips * rng.gen_range(0.8..1.2)).clamp(10.0, 100.0);
            mutated.sl_pips = (mutated.sl_pips * rng.gen_range(0.8..1.2)).clamp(5.0, 50.0);
        }
        _ => {
            mutated.use_ob = rng.gen_bool(0.5);
            mutated.use_fvg = rng.gen_bool(0.5);
            mutated.use_liq_sweep = rng.gen_bool(0.5);
            mutated.mtf_confirmation = rng.gen_bool(0.5);
            mutated.use_premium_discount = rng.gen_bool(0.5);
            mutated.use_inducement = rng.gen_bool(0.5);
        }
    }
    mutated.strategy_id = format!("gene_{}_{}", rng.gen_range(0..1_000_000u64), generation);
    mutated.generation = generation;
    mutated
}

pub fn random_search(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    n_genes: usize,
    max_indicators: usize,
) -> Result<SearchResult> {
    let n_indicators = features.data.ncols();
    let mut genes = generate_random_genes(n_genes, n_indicators, max_indicators, 0);
    for gene in genes.iter_mut() {
        gene.normalize(n_indicators, 4);
    }
    let metrics = evaluate_genes(features, ohlcv, &genes, &EvaluationConfig::default())?;
    Ok(SearchResult { genes, metrics })
}

pub fn evolve_search(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    population: usize,
    generations: usize,
    max_indicators: usize,
) -> Result<SearchResult> {
    if population == 0 {
        bail!("population must be > 0");
    }
    let n_indicators = features.data.ncols();
    let mut genes = generate_random_genes(population, n_indicators, max_indicators, 0);
    for gene in genes.iter_mut() {
        gene.normalize(n_indicators, 4);
    }
    let mut best_metrics = Vec::new();

    if generations == 0 {
        let metrics = evaluate_genes(features, ohlcv, &genes, &EvaluationConfig::default())?;
        apply_metrics(&mut genes, &metrics);
        return Ok(SearchResult { genes, metrics });
    }

    for gen in 0..generations {
        let metrics = evaluate_genes(features, ohlcv, &genes, &EvaluationConfig::default())?;
        apply_metrics(&mut genes, &metrics);

        let mut scored: Vec<(f64, Gene, [f64; 11])> = genes
            .iter()
            .cloned()
            .zip(metrics.into_iter())
            .map(|(g, m)| (g.fitness, g, m))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let elite_count = (population.max(2) as f32 * 0.2) as usize;
        let elite_count = elite_count.max(2).min(scored.len());
        let elites: Vec<Gene> = scored.iter().take(elite_count).map(|(_, g, _)| g.clone()).collect();
        best_metrics = scored.iter().take(elite_count).map(|(_, _, m)| *m).collect();

        if gen + 1 == generations {
            return Ok(SearchResult {
                genes: elites,
                metrics: best_metrics,
            });
        }

        let mut next = Vec::with_capacity(population);
        next.extend(elites.clone());
        while next.len() < population {
            let a = &elites[rand::thread_rng().gen_range(0..elites.len())];
            let b = &elites[rand::thread_rng().gen_range(0..elites.len())];
            let child = mutate(&crossover(a, b, gen + 1), n_indicators, max_indicators, gen + 1);
            next.push(child);
        }
        genes = next;
    }

    Ok(SearchResult {
        genes,
        metrics: best_metrics,
    })
}

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub symbol: String,
    pub weight: f64,
    pub kelly_size: f64,
    pub risk_budget: f64,
    pub correlation_score: f64,
    pub sharpe: f64,
}

#[derive(Debug, Clone)]
pub struct SymbolMetrics {
    pub returns: Vec<f64>,
    pub sharpe: f64,
    pub win_rate: f64,
}

#[derive(Debug, Clone)]
pub struct PortfolioOptimizer {
    pub lookback_days: usize,
    pub max_weight: f64,
    pub kelly_fraction: f64,
}

impl Default for PortfolioOptimizer {
    fn default() -> Self {
        Self {
            lookback_days: 30,
            max_weight: 0.35,
            kelly_fraction: 0.25,
        }
    }
}

impl PortfolioOptimizer {
    pub fn new(lookback_days: usize, max_weight: f64, kelly_fraction: f64) -> Self {
        Self {
            lookback_days,
            max_weight,
            kelly_fraction,
        }
    }

    pub fn get_optimal_allocation(
        &self,
        symbols: &[String],
        metrics_map: &HashMap<String, SymbolMetrics>,
    ) -> HashMap<String, AllocationResult> {
        if symbols.is_empty() {
            return HashMap::new();
        }

        let mut rets = Vec::new();
        let mut names = Vec::new();
        for s in symbols {
            if let Some(metrics) = metrics_map.get(s) {
                if metrics.returns.len() > 5 {
                    rets.push(metrics.returns.clone());
                    names.push(s.clone());
                }
            }
        }

        let mut weights = HashMap::new();
        let min_corr_samples = 30usize;

        if rets.len() >= 2 {
            let min_len = rets.iter().map(|r| r.len()).min().unwrap_or(0);
            if min_len < min_corr_samples {
                let w = 1.0 / names.len().max(1) as f64;
                for s in names {
                    weights.insert(
                        s.clone(),
                        AllocationResult {
                            symbol: s,
                            weight: w,
                            kelly_size: 0.0,
                            risk_budget: w,
                            correlation_score: 0.0,
                            sharpe: 0.0,
                        },
                    );
                }
                return weights;
            }

            for r in &mut rets {
                if r.len() > min_len {
                    *r = r[(r.len() - min_len)..].to_vec();
                }
            }

            let n_assets = names.len();
            let mut means = vec![0.0; n_assets];
            let mut vols = vec![0.0; n_assets];
            for (i, r) in rets.iter().enumerate() {
                let mean = mean(r);
                let std = stddev(r, mean);
                means[i] = mean;
                vols[i] = std.max(1e-6);
            }

            let mut corr = vec![vec![0.0; n_assets]; n_assets];
            for i in 0..n_assets {
                for j in 0..n_assets {
                    if i == j {
                        corr[i][j] = 1.0;
                    } else {
                        let c = cov(&rets[i], means[i], &rets[j], means[j]);
                        corr[i][j] = c / (vols[i] * vols[j]).max(1e-9);
                    }
                }
            }

            let mut sharpe_map = HashMap::new();
            let mut win_map = HashMap::new();
            for s in &names {
                if let Some(m) = metrics_map.get(s) {
                    sharpe_map.insert(s.clone(), m.sharpe);
                    win_map.insert(s.clone(), m.win_rate);
                }
            }

            let mut avg_corr = vec![0.0; n_assets];
            for i in 0..n_assets {
                let mut sum = 0.0;
                for j in 0..n_assets {
                    if i != j {
                        sum += corr[i][j];
                    }
                }
                avg_corr[i] = sum / (n_assets.saturating_sub(1).max(1) as f64);
            }

            let mut raw = vec![0.0; n_assets];
            for i in 0..n_assets {
                let s = &names[i];
                let sharpe = *sharpe_map.get(s).unwrap_or(&0.0);
                let div_score = 1.0 / (1.0 + avg_corr[i].abs());
                raw[i] = (1.0 / vols[i]) * sharpe.max(0.01) * div_score;
            }

            if raw.iter().all(|v| *v <= 0.0) {
                raw = vec![1.0; n_assets];
            }
            let sum_raw: f64 = raw.iter().sum();
            let mut norm: Vec<f64> = raw.iter().map(|v| v / sum_raw).collect();

            for w in &mut norm {
                *w = w.clamp(0.0, self.max_weight);
            }
            let sum_capped: f64 = norm.iter().sum();
            let norm = if sum_capped > 0.0 {
                norm.iter().map(|v| v / sum_capped).collect::<Vec<f64>>()
            } else {
                vec![1.0 / n_assets as f64; n_assets]
            };

            for (i, s) in names.iter().enumerate() {
                let win_rate = *win_map.get(s).unwrap_or(&0.5);
                let kelly_val = self.kelly_fraction * win_rate;
                weights.insert(
                    s.clone(),
                    AllocationResult {
                        symbol: s.clone(),
                        weight: norm[i],
                        kelly_size: kelly_val,
                        risk_budget: norm[i],
                        correlation_score: avg_corr[i],
                        sharpe: *sharpe_map.get(s).unwrap_or(&0.0),
                    },
                );
            }
        } else {
            let w = 1.0 / symbols.len().max(1) as f64;
            for s in symbols {
                let sharpe = metrics_map.get(s).map(|m| m.sharpe).unwrap_or(0.0);
                weights.insert(
                    s.clone(),
                    AllocationResult {
                        symbol: s.clone(),
                        weight: w,
                        kelly_size: 0.0,
                        risk_budget: w,
                        correlation_score: 0.0,
                        sharpe,
                    },
                );
            }
        }

        weights
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], mean: f64) -> f64 {
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

fn cov(a: &[f64], mean_a: f64, b: &[f64], mean_b: f64) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let n = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..n {
        sum += (a[i] - mean_a) * (b[i] - mean_b);
    }
    sum / (n as f64 - 1.0)
}

use anyhow::{bail, Result};
use forex_data::{compute_talib_feature_frame, FeatureCache, FeatureFrame, Ohlcv, SymbolDataset};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Kind, Tensor};

#[derive(Debug, Clone)]
pub struct GpuDiscoveryConfig {
    pub population: usize,
    pub generations: usize,
    pub elite_fraction: f64,
    pub sigma: f64,
    pub crossover_rate: f64,
    pub threshold_scale: f64,
    pub threshold_margin: f64,
    pub threshold_clip: f64,
    pub window_bars: usize,
    pub segments: usize,
    pub min_trades_per_day: f64,
    pub trade_penalty: f64,
    pub dd_limit: f64,
    pub dd_penalty: f64,
    pub robust_weight: f64,
    pub pos_window_fraction: f64,
    pub pos_penalty: f64,
    pub chunk_size: usize,
    pub devices: Vec<i64>,
}

impl Default for GpuDiscoveryConfig {
    fn default() -> Self {
        Self {
            population: 24000,
            generations: 200,
            elite_fraction: 0.05,
            sigma: 0.5,
            crossover_rate: 0.35,
            threshold_scale: 0.10,
            threshold_margin: 0.02,
            threshold_clip: 0.30,
            window_bars: 1440 * 22 * 6,
            segments: 4,
            min_trades_per_day: 1.0,
            trade_penalty: 25.0,
            dd_limit: 0.04,
            dd_penalty: 200.0,
            robust_weight: 0.2,
            pos_window_fraction: 0.5,
            pos_penalty: 15.0,
            chunk_size: 2048,
            devices: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuDiscoveryResult {
    pub genomes: Vec<Vec<f32>>,
    pub fitness: Vec<f32>,
    pub feature_names: Vec<String>,
    pub timeframes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct GenomeExport<'a> {
    fitness: f32,
    genome: &'a [f32],
}

pub fn save_gpu_genomes(path: impl AsRef<Path>, result: &GpuDiscoveryResult) -> Result<()> {
    let mut payload = Vec::new();
    for (g, f) in result.genomes.iter().zip(result.fitness.iter()) {
        payload.push(GenomeExport { fitness: *f, genome: g });
    }
    let json = serde_json::to_string_pretty(&payload)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn build_feature_cube(
    dataset: &SymbolDataset,
    base_tf: &str,
    timeframes: &[&str],
    cache: Option<&FeatureCache>,
) -> Result<(Vec<FeatureFrame>, Vec<String>, Ohlcv)> {
    let base_tf = if dataset.frames.contains_key(base_tf) {
        base_tf.to_string()
    } else if dataset.frames.contains_key("M5") {
        "M5".to_string()
    } else if dataset.frames.contains_key("M1") {
        "M1".to_string()
    } else {
        dataset
            .frames
            .keys()
            .next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no timeframes available"))?
    };

    let base_ohlcv = dataset
        .frames
        .get(&base_tf)
        .ok_or_else(|| anyhow::anyhow!("base timeframe missing"))?;

    let base_key = format!("{}_{}_base", dataset.symbol, base_tf);
    let base_frame = if let Some(cache) = cache {
        if let Some(frame) = cache.load(&base_key)? {
            frame
        } else {
            let frame = compute_talib_feature_frame(base_ohlcv, true)?;
            cache.store(&base_key, &frame)?;
            frame
        }
    } else {
        compute_talib_feature_frame(base_ohlcv, true)?
    };

    let base_ts = base_frame.timestamps.clone();
    let base_names = base_frame.names.clone();
    let base_aligned = base_frame.data.clone();

    let mut targets: Vec<String> = if timeframes.is_empty() {
        dataset
            .frames
            .keys()
            .filter(|tf| *tf != &base_tf)
            .cloned()
            .collect()
    } else {
        timeframes.iter().map(|tf| tf.to_string()).collect()
    };
    targets.sort();

    let mut frames = Vec::new();
    frames.push(FeatureFrame {
        timestamps: base_ts.clone(),
        names: base_names.clone(),
        data: base_aligned,
    });

    for tf in targets.iter() {
        if tf == &base_tf {
            continue;
        }
        let htf = match dataset.frames.get(tf) {
            Some(v) => v,
            None => continue,
        };
        let key = format!("{}_{}_htf", dataset.symbol, tf);
        let htf_frame = if let Some(cache) = cache {
            if let Some(frame) = cache.load(&key)? {
                frame
            } else {
                let frame = compute_talib_feature_frame(htf, true)?;
                cache.store(&key, &frame)?;
                frame
            }
        } else {
            compute_talib_feature_frame(htf, true)?
        };

        let aligned = align_features(&base_ts, &htf_frame.timestamps, &htf_frame.data);
        let mapped = map_feature_columns(&base_names, &htf_frame.names, &aligned);
        frames.push(FeatureFrame {
            timestamps: base_ts.clone(),
            names: base_names.clone(),
            data: mapped,
        });
    }

    Ok((frames, base_names, base_ohlcv.clone()))
}

pub fn run_gpu_discovery(
    frames: &[FeatureFrame],
    base_ohlcv: &Ohlcv,
    config: &GpuDiscoveryConfig,
) -> Result<GpuDiscoveryResult> {
    if frames.is_empty() {
        bail!("no feature frames supplied");
    }

    let tf_count = frames.len();
    let n_samples = frames[0].data.nrows();
    let n_features = frames[0].data.ncols();
    if n_samples == 0 || n_features == 0 {
        bail!("empty feature frame");
    }

    let data_cube = build_data_cube(frames)?;
    let ohlc_cube = build_ohlc_cube(base_ohlcv, tf_count)?;

    let device_ids = if config.devices.is_empty() {
        let count = tch::Cuda::device_count();
        (0..count).collect::<Vec<_>>()
    } else {
        config.devices.clone()
    };
    if device_ids.is_empty() {
        bail!("no CUDA devices available for GPU discovery");
    }

    let dim = tf_count + n_features + 2;
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut genomes: Vec<Vec<f32>> = (0..config.population)
        .map(|_| {
            (0..dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect::<Vec<f32>>()
        })
        .collect();

    let mut best_genomes = Vec::new();
    let mut best_scores = Vec::new();

    for gen in 0..config.generations {
        let fitness = evaluate_population_multi_gpu(
            &data_cube,
            &ohlc_cube,
            &genomes,
            config,
            &device_ids,
        )?;

        let mut scored: Vec<(f32, Vec<f32>)> = genomes
            .into_iter()
            .zip(fitness.into_iter())
            .map(|(g, f)| (f, g))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let elite_count = ((config.population as f64) * config.elite_fraction)
            .round()
            .max(2.0) as usize;
        let elite_count = elite_count.min(scored.len());
        let elites: Vec<Vec<f32>> = scored.iter().take(elite_count).map(|(_, g)| g.clone()).collect();
        let elite_scores: Vec<f32> = scored.iter().take(elite_count).map(|(f, _)| *f).collect();

        if gen + 1 == config.generations {
            best_genomes = elites.clone();
            best_scores = elite_scores.clone();
            break;
        }

        let mu = mean_vector(&elites);
        let std = std_vector(&elites, &mu);

        let mut next = elites.clone();
        while next.len() < config.population {
            let use_cross = rng.gen_bool(config.crossover_rate);
            let mut child = vec![0.0_f32; dim];
            if use_cross && elites.len() >= 2 {
                let a = &elites[rng.gen_range(0..elites.len())];
                let b = &elites[rng.gen_range(0..elites.len())];
                for i in 0..dim {
                    let base = 0.5 * (a[i] + b[i]);
                    let noise = std[i] as f64 * normal.sample(&mut rng) * config.sigma;
                    child[i] = (base as f64 + noise).clamp(-1.0, 1.0) as f32;
                }
            } else {
                for i in 0..dim {
                    let noise = std[i] as f64 * normal.sample(&mut rng) * config.sigma;
                    child[i] = (mu[i] as f64 + noise).clamp(-1.0, 1.0) as f32;
                }
            }
            next.push(child);
        }
        genomes = next;
    }

    Ok(GpuDiscoveryResult {
        genomes: best_genomes,
        fitness: best_scores,
        feature_names: frames[0].names.clone(),
        timeframes: (0..tf_count).map(|idx| format!("tf_{idx}")).collect(),
    })
}

fn build_data_cube(frames: &[FeatureFrame]) -> Result<Tensor> {
    let tf_count = frames.len();
    let n_samples = frames[0].data.nrows();
    let n_features = frames[0].data.ncols();
    let mut buf = vec![0.0_f32; tf_count * n_samples * n_features];

    for (t, frame) in frames.iter().enumerate() {
        let shifted = shift_down(&frame.data);
        let standardized = zscore(&shifted);
        for i in 0..n_samples {
            for j in 0..n_features {
                let idx = (t * n_samples * n_features) + (i * n_features) + j;
                buf[idx] = standardized[(i, j)];
            }
        }
    }

    Ok(Tensor::from_slice(&buf).reshape(&[tf_count as i64, n_samples as i64, n_features as i64]))
}

fn build_ohlc_cube(base: &Ohlcv, tf_count: usize) -> Result<Tensor> {
    let n_samples = base.close.len();
    let mut buf = vec![0.0_f32; tf_count * n_samples * 4];
    for t in 0..tf_count {
        for i in 0..n_samples {
            let idx = (t * n_samples * 4) + (i * 4);
            buf[idx] = base.open[i] as f32;
            buf[idx + 1] = base.high[i] as f32;
            buf[idx + 2] = base.low[i] as f32;
            buf[idx + 3] = base.close[i] as f32;
        }
    }
    Ok(Tensor::from_slice(&buf).reshape(&[tf_count as i64, n_samples as i64, 4]))
}

fn evaluate_population_multi_gpu(
    data_cube: &Tensor,
    ohlc_cube: &Tensor,
    genomes: &[Vec<f32>],
    config: &GpuDiscoveryConfig,
    device_ids: &[i64],
) -> Result<Vec<f32>> {
    let mut results = vec![0.0_f32; genomes.len()];
    let mut offset = 0usize;
    while offset < genomes.len() {
        let end = (offset + config.chunk_size).min(genomes.len());
        let chunk = &genomes[offset..end];
        let mut chunk_buf = Vec::with_capacity(chunk.len() * chunk[0].len());
        for g in chunk {
            chunk_buf.extend_from_slice(g);
        }
        let chunk_tensor = Tensor::from_slice(&chunk_buf)
            .reshape(&[chunk.len() as i64, chunk[0].len() as i64]);

        let mut per_device = Vec::new();
        let split = split_tensor(&chunk_tensor, device_ids.len());
        for (i, part) in split.into_iter().enumerate() {
            let device = Device::Cuda(device_ids[i]);
            let fit = evaluate_population_gpu(data_cube, ohlc_cube, &part, config, device)?;
            per_device.push(fit);
        }

        let mut idx = offset;
        for fit in per_device {
            let vec: Vec<f32> = Vec::<f32>::from(&fit);
            for v in vec {
                results[idx] = v;
                idx += 1;
            }
        }
        offset = end;
    }
    Ok(results)
}

fn split_tensor(t: &Tensor, parts: usize) -> Vec<Tensor> {
    let n = t.size()[0] as usize;
    if parts <= 1 || n <= 1 {
        return vec![t.shallow_clone()];
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    for i in 0..parts {
        let remaining = n - start;
        let take = if i == parts - 1 {
            remaining
        } else {
            (n / parts).max(1)
        };
        let len = take.min(remaining);
        if len == 0 {
            break;
        }
        out.push(t.narrow(0, start as i64, len as i64));
        start += len;
    }
    out
}

fn evaluate_population_gpu(
    data_cube: &Tensor,
    ohlc_cube: &Tensor,
    genomes: &Tensor,
    config: &GpuDiscoveryConfig,
    device: Device,
) -> Result<Tensor> {
    let tf_count = data_cube.size()[0];
    let n_samples = data_cube.size()[1];
    let n_features = data_cube.size()[2];
    let pop = genomes.size()[0];

    let data = data_cube.to_device(device).to_kind(Kind::Float);
    let ohlc = ohlc_cube.to_device(device).to_kind(Kind::Float);
    let genomes = genomes.to_device(device).to_kind(Kind::Float);

    let tf_weights = genomes
        .narrow(1, 0, tf_count)
        .softmax(-1, Kind::Float);
    let logic_weights = genomes.narrow(1, tf_count, n_features);
    let thresholds = genomes
        .narrow(1, tf_count + n_features, 2)
        .clamp(-config.threshold_clip, config.threshold_clip)
        * (config.threshold_scale as f32);
    let buy_th = thresholds
        .select(1, 0)
        .maximum(&thresholds.select(1, 1))
        + config.threshold_margin as f32;
    let sell_th = thresholds
        .select(1, 0)
        .minimum(&thresholds.select(1, 1))
        - config.threshold_margin as f32;

    let segments = build_segments(n_samples as usize, config.window_bars, config.segments);

    let mut fitness_sum = Tensor::zeros([pop], (Kind::Float, device));
    let mut min_fitness = Tensor::full([pop], 1e9, (Kind::Float, device));
    let mut pos_windows = Tensor::zeros([pop], (Kind::Float, device));

    for (start, len) in segments {
        let data_slice = data.narrow(1, start as i64, len as i64);
        let ohlc_slice = ohlc.narrow(1, start as i64, len as i64);
        let mut all_signals = Tensor::zeros([pop, len as i64], (Kind::Float, device));
        for t in 0..tf_count {
            let tf_data = data_slice.get(t);
            let tf_sig = tf_data.matmul(&logic_weights.transpose(0, 1));
            let std = tf_sig.std_dim(&[0], false, Kind::Float) + 1e-6;
            let tf_sig = tf_sig / std.unsqueeze(0);
            let weight = tf_weights.select(1, t).unsqueeze(1);
            all_signals += tf_sig.transpose(0, 1) * weight;
        }
        all_signals = all_signals.tanh();

        let actions = all_signals.gt(&buy_th.unsqueeze(1)).to_kind(Kind::Float)
            - all_signals.lt(&sell_th.unsqueeze(1)).to_kind(Kind::Float);

        let open_p = ohlc_slice.get(0).select(1, 0);
        let close_p = ohlc_slice.get(0).select(1, 3);
        let open_next = open_p.narrow(0, 1, (len - 1) as i64);
        let close_next = close_p.narrow(0, 1, (len - 1) as i64);
        let rets = (close_next - open_next) / open_next.clamp_min(1e-6);
        let actions_slice = actions.narrow(1, 0, (len - 1) as i64);
        let batch_rets = &actions_slice * rets.unsqueeze(0) - actions_slice.abs() * 0.0002;

        let equity = batch_rets.cumsum(1, Kind::Float);
        let peaks = equity.cummax(1, false).0;
        let max_dd = (&peaks - &equity).max_dim(1, false).0;

        let mean_ret = batch_rets.mean_dim(&[1], false, Kind::Float);
        let downside = batch_rets.minimum(&Tensor::zeros([1], (Kind::Float, device)));
        let downside_std = downside.pow(2).mean_dim(&[1], false, Kind::Float).sqrt() + 1e-9;
        let sortino = &mean_ret / downside_std;

        let steps = Tensor::arange((len - 1) as i64, (Kind::Float, device));
        let equity_mean = equity.mean_dim(&[1], true, Kind::Float);
        let steps_mean = steps.mean(Kind::Float);
        let num = ((&equity - &equity_mean) * (&steps - steps_mean)).sum_dim_intlist(&[1], false, Kind::Float);
        let den = ((&equity - &equity_mean).pow(2).sum_dim_intlist(&[1], false, Kind::Float)
            * (&steps - steps_mean).pow(2).sum(Kind::Float))
            .sqrt();
        let consistency = num / (den + 1e-9);

        let trade_count = actions.abs().sum_dim_intlist(&[1], false, Kind::Float);
        let expected = (len as f64 / 1440.0) * config.min_trades_per_day;
        let freq_penalty = (Tensor::from(expected as f32).to_device(device) - &trade_count)
            .clamp_min(0.0)
            * config.trade_penalty as f32;
        let dd_penalty = (max_dd - config.dd_limit as f32)
            .clamp_min(0.0)
            * config.dd_penalty as f32;

        let mut window_fit = sortino * 10.0 + consistency * 5.0 - freq_penalty - dd_penalty;
        let profit_pct = equity.select(1, (len - 2) as i64);
        window_fit += profit_pct.clamp_max(0.10) * 100.0;

        fitness_sum += &window_fit;
        min_fitness = min_fitness.minimum(&window_fit);

        let pos = profit_pct.gt(0.0) * trade_count.ge(expected as f32);
        pos_windows += pos.to_kind(Kind::Float);
    }

    let avg_fit = fitness_sum / (segments.len() as f64);
    let min_pos = (segments.len() as f64 * config.pos_window_fraction).ceil();
    let pos_penalty = (Tensor::from(min_pos as f32).to_device(device) - pos_windows)
        .clamp_min(0.0)
        * config.pos_penalty as f32;
    let final_fit = avg_fit + min_fitness * config.robust_weight as f32 - pos_penalty;
    Ok(final_fit.to_device(Device::Cpu))
}

fn build_segments(n_samples: usize, window: usize, segments: usize) -> Vec<(usize, usize)> {
    if n_samples <= window + 2 {
        return vec![(0, n_samples)];
    }
    let mut rng = rand::thread_rng();
    let mut out = Vec::new();
    let start_recent = n_samples.saturating_sub(window + 1);
    out.push((start_recent, window));
    let segs = segments.saturating_sub(1);
    for _ in 0..segs {
        let start = rng.gen_range(0..(n_samples - window - 1));
        out.push((start, window));
    }
    out
}

fn align_features(base_ts: &[i64], htf_ts: &[i64], htf_data: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let n_base = base_ts.len();
    let n_htf = htf_ts.len();
    let n_cols = htf_data.ncols();
    let mut out = ndarray::Array2::<f32>::zeros((n_base, n_cols));
    if n_htf == 0 || n_base == 0 {
        return out;
    }
    let mut j = 0usize;
    for i in 0..n_base {
        let target = base_ts[i];
        while j + 1 < n_htf && htf_ts[j + 1] <= target {
            j += 1;
        }
        if htf_ts[j] > target {
            continue;
        }
        if j == 0 {
            continue;
        }
        let src = j - 1;
        for c in 0..n_cols {
            out[(i, c)] = htf_data[(src, c)];
        }
    }
    out
}

fn map_feature_columns(
    base_names: &[String],
    htf_names: &[String],
    aligned: &ndarray::Array2<f32>,
) -> ndarray::Array2<f32> {
    let n_rows = aligned.nrows();
    let n_cols = base_names.len();
    let mut out = ndarray::Array2::<f32>::zeros((n_rows, n_cols));
    let mut index_map = HashMap::new();
    for (idx, name) in htf_names.iter().enumerate() {
        index_map.insert(name.as_str(), idx);
    }
    for (col_idx, name) in base_names.iter().enumerate() {
        if let Some(src_idx) = index_map.get(name.as_str()) {
            for row in 0..n_rows {
                out[(row, col_idx)] = aligned[(row, *src_idx)];
            }
        }
    }
    out
}

fn shift_down(data: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let (rows, cols) = data.dim();
    let mut out = ndarray::Array2::<f32>::zeros((rows, cols));
    for r in 1..rows {
        for c in 0..cols {
            out[(r, c)] = data[(r - 1, c)];
        }
    }
    out
}

fn zscore(data: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let (rows, cols) = data.dim();
    let mut out = ndarray::Array2::<f32>::zeros((rows, cols));
    for c in 0..cols {
        let mut sum = 0.0_f64;
        let mut sumsq = 0.0_f64;
        for r in 0..rows {
            let v = data[(r, c)] as f64;
            sum += v;
            sumsq += v * v;
        }
        let mean = sum / rows.max(1) as f64;
        let var = (sumsq / rows.max(1) as f64) - mean * mean;
        let std = var.max(1e-12).sqrt();
        for r in 0..rows {
            out[(r, c)] = ((data[(r, c)] as f64 - mean) / std) as f32;
        }
    }
    out
}

fn mean_vector(elites: &[Vec<f32>]) -> Vec<f32> {
    let dim = elites[0].len();
    let mut out = vec![0.0_f32; dim];
    for e in elites {
        for i in 0..dim {
            out[i] += e[i];
        }
    }
    let n = elites.len().max(1) as f32;
    for v in &mut out {
        *v /= n;
    }
    out
}

fn std_vector(elites: &[Vec<f32>], mean: &[f32]) -> Vec<f32> {
    let dim = elites[0].len();
    let mut out = vec![0.0_f32; dim];
    for e in elites {
        for i in 0..dim {
            let d = e[i] - mean[i];
            out[i] += d * d;
        }
    }
    let n = elites.len().max(2) as f32;
    for v in &mut out {
        *v = (*v / (n - 1.0)).sqrt();
        if *v < 1e-6 {
            *v = 1e-6;
        }
    }
    out
}

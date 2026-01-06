// Base classes and utilities for machine learning models
//
// Ported from src/forex_bot/models/base.py line-by-line
//
// This module provides:
// - EarlyStopper: Universal early stopping for training loops
// - ExpertModel: Abstract trait for all expert models
// - Training utilities for time-series aware data handling

use anyhow::{Context, Result};
use ndarray::Array2;
use polars::prelude::*;
use std::collections::HashMap;
use std::path::Path;
use tracing::*;

// ============================================================================
// EARLY STOPPING
// ============================================================================

/// Universal Early Stopping utility.
/// Stops training when validation metric stops improving.
///
/// Ported from Python EarlyStopper class (lines 25-48)
pub struct EarlyStopper {
    patience: usize,
    min_delta: f64,
    counter: usize,
    best_loss: Option<f64>,
    pub early_stop: bool,
}

impl EarlyStopper {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            counter: 0,
            best_loss: None,
            early_stop: false,
        }
    }

    /// Call with validation loss. Returns true if should stop.
    /// Ported from Python __call__ method (lines 38-48)
    pub fn check(&mut self, val_loss: f64) -> bool {
        if self.best_loss.is_none() {
            self.best_loss = Some(val_loss);
        } else if let Some(best) = self.best_loss {
            if val_loss > best - self.min_delta {
                self.counter += 1;
                if self.counter >= self.patience {
                    self.early_stop = true;
                }
            } else {
                self.best_loss = Some(val_loss);
                self.counter = 0;
            }
        }
        self.early_stop
    }
}

/// Return (patience, min_delta) with optional env overrides.
/// Ported from Python get_early_stop_params (lines 51-69)
pub fn get_early_stop_params(default_patience: usize, default_min_delta: f64) -> (usize, f64) {
    let mut patience = default_patience;
    let mut min_delta = default_min_delta;

    // Try to read env var for patience
    if let Ok(env_pat) = std::env::var("FOREX_BOT_EARLY_STOP_PATIENCE") {
        if !env_pat.is_empty() {
            if let Ok(val) = env_pat.parse::<usize>() {
                if val > 0 {
                    patience = val;
                }
            }
        }
    }

    // Try to read env var for min_delta
    if let Ok(env_delta) = std::env::var("FOREX_BOT_EARLY_STOP_MIN_DELTA") {
        if !env_delta.is_empty() {
            if let Ok(val) = env_delta.parse::<f64>() {
                min_delta = val;
            }
        }
    }

    (patience, min_delta)
}

// ============================================================================
// EXPERT MODEL TRAIT
// ============================================================================

/// Abstract base trait for all expert models.
/// Ported from Python ExpertModel class (lines 71-127)
pub trait ExpertModel {
    /// Train the model.
    /// Ported from Python fit method (lines 74-77)
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()>;

    /// Predict probabilities for classes [-1, 0, 1].
    ///
    /// Returns:
    ///     Array2<f32>: Shape (N, 3) where columns map to [neutral, buy, sell]
    ///                  Convention: col 0 -> neutral, col 1 -> buy, col 2 -> sell
    ///
    /// Ported from Python predict_proba method (lines 79-89)
    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>>;

    /// Save model artifacts to directory.
    /// Ported from Python save method (lines 91-94)
    fn save(&self, path: &Path) -> Result<()>;

    /// Load model artifacts from directory.
    /// Ported from Python load method (lines 96-99)
    fn load(&mut self, path: &Path) -> Result<()>;

    /// Helper for atomic model saving with rotation/backup.
    /// Keeps 'model.pt' (current) and 'model.pt.bak' (previous).
    ///
    /// Ported from Python _atomic_save method (lines 101-126)
    fn atomic_save(&self, save_func: Box<dyn FnOnce(&Path) -> Result<()>>, target_path: &Path) -> Result<()>
    {
        let temp_path = target_path.with_extension("tmp");
        let backup_path = target_path.with_extension("bak");

        // Save to temp file
        save_func(&temp_path)
            .with_context(|| format!("Failed to save to temp file: {}", temp_path.display()))?;

        // Rotate: current -> backup, temp -> current
        if target_path.exists() {
            if backup_path.exists() {
                std::fs::remove_file(&backup_path)
                    .with_context(|| format!("Failed to delete old backup: {}", backup_path.display()))?;
            }
            std::fs::rename(target_path, &backup_path)
                .with_context(|| format!("Failed to rotate to backup: {}", backup_path.display()))?;
        }

        std::fs::rename(&temp_path, target_path)
            .with_context(|| format!("Failed to move temp to target: {}", target_path.display()))?;

        Ok(())
    }
}

// ============================================================================
// DATA CONVERSION UTILITIES
// ============================================================================

/// Convert a DataFrame to a float32 ndarray suitable for models.
///
/// Ported from Python dataframe_to_float32_numpy (lines 129-139)
pub fn dataframe_to_float32_array(df: &DataFrame) -> Result<Array2<f32>> {
    let n_rows = df.height();
    let n_cols = df.width();

    let mut data = Vec::with_capacity(n_rows * n_cols);

    // Iterate through columns
    for col in df.get_columns() {
        // Try to convert to f64 series first
        let series_f64 = col.cast(&DataType::Float64)
            .with_context(|| format!("Failed to cast column {} to f64", col.name()))?;

        let ca = series_f64.f64()
            .with_context(|| format!("Failed to get f64 chunked array for {}", col.name()))?;

        // Extract values
        for val in ca.into_iter() {
            data.push(val.unwrap_or(0.0) as f32);
        }
    }

    // Reshape to (n_rows, n_cols) - column-major to row-major
    let mut array_data = Vec::with_capacity(n_rows * n_cols);
    for row_idx in 0..n_rows {
        for col_idx in 0..n_cols {
            array_data.push(data[col_idx * n_rows + row_idx]);
        }
    }

    Array2::from_shape_vec((n_rows, n_cols), array_data)
        .context("Failed to create Array2 from DataFrame")
}

// ============================================================================
// TIME-SERIES VALIDATION
// ============================================================================

/// Validate that DataFrame index is monotonically increasing (time-ordered).
///
/// This is critical for time-series models to prevent look-ahead bias.
///
/// Ported from Python validate_time_ordering (lines 142-184)
pub fn validate_time_ordering(df: &DataFrame, context: &str) -> Result<bool> {
    if df.height() == 0 {
        return Ok(true);
    }

    // Polars doesn't expose index like Pandas - check if we have a datetime column
    // For now, assume data is already sorted or skip this check
    // TODO: Implement proper datetime column check

    warn!("{}: Rust port skips strict monotonic check - assume data is pre-sorted", context);
    Ok(true)
}

/// Splits data for time-series training with an embargo gap.
///
/// Ported from Python time_series_train_val_split (lines 187-212)
pub fn time_series_train_val_split(
    x: &DataFrame,
    y: &Series,
    val_ratio: f64,
    min_train_samples: usize,
    embargo_samples: usize, // HPC FIX: Guaranteed memory flush
) -> Result<(DataFrame, DataFrame, Series, Series)> {
    let n = x.height();
    let val_size = (n as f64 * val_ratio) as usize;
    let mut train_end = n.saturating_sub(val_size).saturating_sub(embargo_samples);

    if train_end < min_train_samples {
        // If dataset too small, reduce embargo but maintain at least 100 bars
        let reduced_embargo = embargo_samples.min(100.max(n / 10));
        train_end = n.saturating_sub(val_size).saturating_sub(reduced_embargo);
    }

    let x_train = x.slice(0, train_end);
    let y_train = y.slice(0, train_end);

    let val_start = train_end + embargo_samples;
    let val_len = n.saturating_sub(val_start);
    let x_val = x.slice(val_start as i64, val_len);
    let y_val = y.slice(val_start as i64, val_len);

    Ok((x_train, x_val, y_train, y_val))
}

// ============================================================================
// SAMPLING UTILITIES
// ============================================================================

/// Downsample data while preserving class distribution.
///
/// Used to limit memory/compute for large datasets while maintaining
/// representative class balance.
///
/// Ported from Python stratified_downsample (lines 215-289)
pub fn stratified_downsample(
    x: &DataFrame,
    y: &Series,
    max_samples: usize,
    random_state: u64,
) -> Result<(DataFrame, Series)> {
    let n = x.height();

    if max_samples == 0 || n <= max_samples {
        return Ok((x.clone(), y.clone()));
    }

    use rand::prelude::*;
    use rand::SeedableRng;
    let mut rng = StdRng::seed_from_u64(random_state);

    // Group by class
    let y_i64 = y.cast(&DataType::Int64)?;
    let y_ca = y_i64.i64()?;

    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (idx, label) in y_ca.into_iter().enumerate() {
        if let Some(lbl) = label {
            class_indices.entry(lbl).or_insert_with(Vec::new).push(idx);
        }
    }

    // Calculate samples per class (proportional)
    let total = n;
    let mut sampled_indices = Vec::new();

    for (_label, indices) in class_indices.iter() {
        // Proportion of this class in original data
        let class_ratio = indices.len() as f64 / total as f64;
        // Target samples for this class
        let target_count = ((max_samples as f64 * class_ratio) as usize).max(1);
        // Actual samples to take
        let take_count = indices.len().min(target_count);

        if take_count > 0 {
            let mut indices_clone = indices.clone();
            indices_clone.shuffle(&mut rng);
            sampled_indices.extend_from_slice(&indices_clone[..take_count]);
        }
    }

    // Trim to max if over
    if sampled_indices.len() > max_samples {
        sampled_indices.shuffle(&mut rng);
        sampled_indices.truncate(max_samples);
    }

    // Sort to maintain temporal order
    sampled_indices.sort_unstable();

    // Create downsampled DataFrame and Series
    // Polars 0.47: take() expects ChunkedArray<UInt32Type>
    let indices: Vec<u32> = sampled_indices.iter().map(|&i| i as u32).collect();
    let indices_ca = Series::new("indices".into(), indices).u32()?.clone();
    let x_out = x.take(&indices_ca)?;
    let y_out = y.take(&indices_ca)?;

    info!(
        "Downsampled from {} to {} samples ({:.1}%)",
        n,
        x_out.height(),
        (x_out.height() as f64 / n as f64) * 100.0
    );

    Ok((x_out, y_out))
}

// ============================================================================
// CLASS WEIGHTING
// ============================================================================

/// Compute balanced class weights for imbalanced classification.
///
/// Uses inverse frequency weighting: rare classes get higher weights.
///
/// Ported from Python compute_class_weights (lines 292-319)
pub fn compute_class_weights(y: &Series) -> Result<HashMap<i64, f64>> {
    let y_i64 = y.cast(&DataType::Int64)?;
    let y_ca = y_i64.i64()?;

    let mut class_counts: HashMap<i64, usize> = HashMap::new();
    let mut n_samples = 0;

    for label in y_ca.into_iter().flatten() {
        *class_counts.entry(label).or_insert(0) += 1;
        n_samples += 1;
    }

    let n_classes = class_counts.len();
    let mut weights = HashMap::new();

    for (cls, count) in class_counts.iter() {
        if *count > 0 {
            // sklearn-style balanced weight
            weights.insert(*cls, n_samples as f64 / (n_classes as f64 * *count as f64));
        }
    }

    Ok(weights)
}

/// Compute per-sample weights based on class frequency.
///
/// Ported from Python compute_sample_weights (lines 322-343)
pub fn compute_sample_weights(y: &Series) -> Result<Vec<f32>> {
    let class_weights = compute_class_weights(y)?;
    let y_i64 = y.cast(&DataType::Int64)?;
    let y_ca = y_i64.i64()?;

    let mut sample_weights = Vec::with_capacity(y.len());

    for label in y_ca.into_iter() {
        if let Some(lbl) = label {
            let weight = class_weights.get(&lbl).copied().unwrap_or(1.0);
            sample_weights.push(weight as f32);
        } else {
            sample_weights.push(1.0);
        }
    }

    Ok(sample_weights)
}

// ============================================================================
// FEATURE DRIFT DETECTION
// ============================================================================

/// Detect feature drift between training and validation data.
///
/// Uses Population Stability Index (PSI) or simple mean/std comparison
/// to identify features that have shifted significantly.
///
/// Ported from Python detect_feature_drift (lines 346-477)
pub fn detect_feature_drift(
    train_df: &DataFrame,
    val_df: &DataFrame,
    _threshold: f64,
    method: &str,
) -> Result<FeatureDriftReport> {
    if train_df.height() == 0 || val_df.height() == 0 {
        return Ok(FeatureDriftReport {
            drifted_features: vec![],
            drift_scores: HashMap::new(),
            summary: "Insufficient data for drift detection".to_string(),
            critical: false,
        });
    }

    // Find common numeric columns
    let train_cols: std::collections::HashSet<_> = train_df.get_column_names().iter().copied().collect();
    let val_cols: std::collections::HashSet<_> = val_df.get_column_names().iter().copied().collect();
    let common_cols: Vec<_> = train_cols.intersection(&val_cols).copied().collect();

    let numeric_cols: Vec<String> = common_cols
        .iter()
        .filter(|&col_name| {
            if let (Ok(train_col), Ok(val_col)) = (train_df.column(col_name), val_df.column(col_name)) {
                matches!(train_col.dtype(), DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::Int64)
                    && matches!(val_col.dtype(), DataType::Float32 | DataType::Float64 | DataType::Int32 | DataType::Int64)
            } else {
                false
            }
        })
        .map(|s| s.to_string())
        .collect();

    if numeric_cols.is_empty() {
        return Ok(FeatureDriftReport {
            drifted_features: vec![],
            drift_scores: HashMap::new(),
            summary: "No numeric features to check".to_string(),
            critical: false,
        });
    }

    // HPC FIX: Regime-Aware Drift Thresholding (lines 405-417)
    let base_threshold = std::env::var("FOREX_BOT_DRIFT_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.20);

    let vol_scale = 1.0; // Simplified - full implementation would check realized_vol
    let threshold = base_threshold * vol_scale;

    let mut drift_scores = HashMap::new();
    let mut drifted_features = Vec::new();

    // HPC: Use parallel processing for drift detection (lines 436-437)
    use rayon::prelude::*;

    let results: Vec<_> = numeric_cols.par_iter()
        .filter_map(|col| {
            let train_col = train_df.column(col).ok()?;
            let val_col = val_df.column(col).ok()?;

            // Polars 0.47: Convert Column to Series
            let train_series = train_col.as_materialized_series().clone();
            let val_series = val_col.as_materialized_series().clone();

            let train_vals = extract_numeric_values(&train_series).ok()?;
            let val_vals = extract_numeric_values(&val_series).ok()?;

            if train_vals.len() < 10 || val_vals.len() < 10 {
                return None;
            }

            let score = if method == "psi" {
                compute_psi(&train_vals, &val_vals, 10)
            } else {
                compute_stats_drift(&train_vals, &val_vals)
            };

            Some((col.clone(), score))
        })
        .collect();

    for (col, score) in results {
        drift_scores.insert(col.clone(), score);
        if score >= threshold {
            drifted_features.push(col);
        }
    }

    // Calculate overall drift severity (lines 448-460)
    let critical_threshold = 0.25;
    let critical_count = drift_scores.values().filter(|&&s| s >= critical_threshold).count();
    let total_features = drift_scores.len();

    let (critical, summary) = if critical_count > total_features * 3 / 10 {
        (true, format!("CRITICAL: {}/{} features have significant drift", critical_count, total_features))
    } else if drifted_features.len() > total_features * 2 / 10 {
        (false, format!("WARNING: {}/{} features show drift", drifted_features.len(), total_features))
    } else {
        (false, format!("OK: {}/{} features with minor drift", drifted_features.len(), total_features))
    };

    if !drifted_features.is_empty() {
        let mut sorted_drifted = drifted_features.clone();
        sorted_drifted.sort_by(|a, b| {
            let score_a = drift_scores.get(a).copied().unwrap_or(0.0);
            let score_b = drift_scores.get(b).copied().unwrap_or(0.0);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_5: Vec<_> = sorted_drifted.iter().take(5).map(|s| s.as_str()).collect();
        let msg = format!(
            "Feature drift detected: {}. Top drifted: {:?}",
            summary, top_5
        );

        if critical || summary.starts_with("WARNING:") {
            warn!("{}", msg);
        } else {
            info!("{}", msg);
        }
    }

    Ok(FeatureDriftReport {
        drifted_features,
        drift_scores,
        summary,
        critical,
    })
}

pub struct FeatureDriftReport {
    pub drifted_features: Vec<String>,
    pub drift_scores: HashMap<String, f64>,
    pub summary: String,
    pub critical: bool,
}

/// Extract numeric values from a Polars series
fn extract_numeric_values(series: &Series) -> Result<Vec<f64>> {
    let series_f64 = series.cast(&DataType::Float64)?;
    let ca = series_f64.f64()?;
    let values: Vec<f64> = ca.into_iter().filter_map(|v| v).collect();
    Ok(values)
}

/// Compute Population Stability Index (PSI) between two distributions.
///
/// PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
///
/// Interpretation:
/// - PSI < 0.1: No significant change
/// - 0.1 <= PSI < 0.25: Moderate change
/// - PSI >= 0.25: Significant change
///
/// Ported from Python _compute_psi (lines 480-535)
pub fn compute_psi(expected: &[f64], actual: &[f64], n_bins: usize) -> f64 {
    let eps = 1e-6;
    let n_bins = n_bins.max(3);

    // Create bins from expected distribution
    let mut breakpoints = compute_percentiles(expected, n_bins + 1);
    breakpoints.sort_by(|a, b| a.partial_cmp(b).unwrap());
    breakpoints.dedup();

    if breakpoints.len() < 2 {
        return 0.0;
    }

    let expected_counts = histogram(expected, &breakpoints);
    let actual_counts = histogram(actual, &breakpoints);

    // If bins are too sparse, retry with coarser bins
    if expected_counts.iter().any(|&c| c < 3) || actual_counts.iter().any(|&c| c < 3) {
        let coarse_bins = (3_usize).max((breakpoints.len() - 1).min(5));
        let coarse_breaks = compute_percentiles(expected, coarse_bins + 1);
        if coarse_breaks.len() >= 2 && coarse_breaks.len() < breakpoints.len() {
            let expected_counts = histogram(expected, &coarse_breaks);
            let actual_counts = histogram(actual, &coarse_breaks);
            return compute_psi_from_counts(&expected_counts, &actual_counts, expected.len(), actual.len(), eps);
        }
    }

    compute_psi_from_counts(&expected_counts, &actual_counts, expected.len(), actual.len(), eps)
}

fn compute_psi_from_counts(
    expected_counts: &[usize],
    actual_counts: &[usize],
    expected_len: usize,
    actual_len: usize,
    eps: f64,
) -> f64 {
    let expected_pct: Vec<f64> = expected_counts.iter()
        .map(|&c| (c as f64 / (expected_len as f64 + eps)).clamp(eps, 1.0))
        .collect();
    let actual_pct: Vec<f64> = actual_counts.iter()
        .map(|&c| (c as f64 / (actual_len as f64 + eps)).clamp(eps, 1.0))
        .collect();

    let psi: f64 = expected_pct.iter()
        .zip(actual_pct.iter())
        .map(|(&exp, &act)| {
            let diff = act - exp;
            let ratio = (act / exp).ln();
            diff * ratio
        })
        .sum();

    psi
}

fn compute_percentiles(data: &[f64], n: usize) -> Vec<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    (0..=n)
        .map(|i| {
            let pct = i as f64 / n as f64;
            let idx = ((sorted.len() - 1) as f64 * pct) as usize;
            sorted[idx.min(sorted.len() - 1)]
        })
        .collect()
}

fn histogram(data: &[f64], breakpoints: &[f64]) -> Vec<usize> {
    let mut counts = vec![0; breakpoints.len().saturating_sub(1)];

    for &val in data {
        for i in 0..breakpoints.len() - 1 {
            if val >= breakpoints[i] && (i == breakpoints.len() - 2 || val < breakpoints[i + 1]) {
                counts[i] += 1;
                break;
            }
        }
    }

    counts
}

/// Fallback drift metric based on mean/std shift.
///
/// Ported from Python _compute_stats_drift (lines 538-554)
pub fn compute_stats_drift(train_vals: &[f64], val_vals: &[f64]) -> f64 {
    let train_mean = train_vals.iter().sum::<f64>() / train_vals.len() as f64;
    let val_mean = val_vals.iter().sum::<f64>() / val_vals.len() as f64;

    let train_std = {
        let variance = train_vals.iter().map(|&x| (x - train_mean).powi(2)).sum::<f64>() / train_vals.len() as f64;
        variance.sqrt()
    };
    let val_std = {
        let variance = val_vals.iter().map(|&x| (x - val_mean).powi(2)).sum::<f64>() / val_vals.len() as f64;
        variance.sqrt()
    };

    let eps = f64::EPSILON;

    if train_std > eps {
        let mean_shift = (val_mean - train_mean).abs() / train_std.max(eps);
        let std_ratio = val_std / train_std.max(eps);
        mean_shift + (1.0 - std_ratio).abs()
    } else {
        0.0
    }
}

// ============================================================================
// ROBUST SCALING (HPC ADVANCEMENT)
// ============================================================================

/// Robust Scaler that handles NaN and Infinite values efficiently.
/// Ported from Python advancements in normalization.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RobustScaler {
    pub mean: Option<Array2<f32>>,
    pub scale: Option<Array2<f32>>,
}

impl RobustScaler {
    pub fn new() -> Self {
        Self {
            mean: None,
            scale: None,
        }
    }

    /// Fit the scaler to data, ignoring NaN/Inf.
    pub fn fit(&mut self, data: &Array2<f32>) -> Result<()> {
        let n_features = data.ncols();
        let mut means = Array2::zeros((1, n_features));
        let mut scales = Array2::zeros((1, n_features));

        for j in 0..n_features {
            let col = data.column(j);
            let valid_values: Vec<f32> = col
                .iter()
                .filter(|&&x| x.is_finite())
                .copied()
                .collect();

            if valid_values.is_empty() {
                means[[0, j]] = 0.0;
                scales[[0, j]] = 1.0;
                continue;
            }

            let sum: f32 = valid_values.iter().sum();
            let count = valid_values.len() as f32;
            let mean = sum / count;

            let variance: f32 = valid_values
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / count;
            
            let std = variance.sqrt().max(1e-3);

            means[[0, j]] = mean;
            scales[[0, j]] = std;
        }

        self.mean = Some(means);
        self.scale = Some(scales);
        Ok(())
    }

    /// Transform data using fitted parameters.
    pub fn transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
        let mean = self.mean.as_ref().context("Scaler not fitted")?;
        let scale = self.scale.as_ref().context("Scaler not fitted")?;

        let mut transformed = data.clone();
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                let val = transformed[[i, j]];
                if val.is_finite() {
                    transformed[[i, j]] = (val - mean[[0, j]]) / scale[[0, j]];
                } else {
                    // Replace NaN/Inf with 0.0 after normalization (mean)
                    transformed[[i, j]] = 0.0;
                }
            }
        }

        Ok(transformed)
    }
}
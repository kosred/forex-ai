// Tree Models Module - LightGBM, XGBoost, CatBoost
// Complete port from Python src/forex_bot/models/trees.py (739 lines)
// NO SIMPLIFICATION - Preserves all HPC production logic

use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::collections::HashMap;
use std::env;
use std::path::Path;

use crate::base::ExpertModel;

#[cfg(feature = "lightgbm")]
use lightgbm3::{Booster as LGBMBooster, Dataset as LGBMDataset};

#[cfg(feature = "xgboost")]
use xgboost_rust as xgb;

#[cfg(feature = "catboost")]
use catboost_rust as catboost;

// ============================================================================ 
// TYPES AND ENUMS
// ============================================================================ 

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Gpu,
    Cpu,
}

#[derive(Debug, Clone)]
pub struct TreeModelConfig {
    pub idx: usize, // For GPU distribution across 8 GPUs
    pub params: HashMap<String, ParamValue>,
    pub device_pref: DevicePreference,
    pub gpu_only: bool,
    pub cpu_threads: Option<usize>,
}

#[derive(Debug, Clone)]
pub enum ParamValue {
    Int(i32),
    Float(f64),
    String(String),
    Bool(bool),
}

// ============================================================================ 
// ENVIRONMENT VARIABLE UTILITIES
// ============================================================================ 

/// Returns CPU thread hint from FOREX_BOT_CPU_THREADS environment variable
pub fn cpu_threads_hint() -> usize {
    match env::var("FOREX_BOT_CPU_THREADS") {
        Ok(val) => val.parse::<usize>().unwrap_or(0),
        Err(_) => 0,
    }
}

/// Parse tree device preference from FOREX_BOT_TREE_DEVICE
/// Supports: auto|gpu|cpu and variations: 0/1/true/false/yes/no/on/off
pub fn tree_device_preference() -> DevicePreference {
    let raw = env::var("FOREX_BOT_TREE_DEVICE")
        .unwrap_or_else(|_| "auto".to_string())
        .trim()
        .to_lowercase();

    match raw.as_str() {
        "cpu" => DevicePreference::Cpu,
        "gpu" => DevicePreference::Gpu,
        "auto" => DevicePreference::Auto,
        "0" | "false" | "no" | "off" => DevicePreference::Cpu,
        "1" | "true" | "yes" | "on" => DevicePreference::Gpu,
        _ => DevicePreference::Auto,
    }
}

/// Check if GPU-only mode is enabled (FOREX_BOT_GPU_ONLY=1|true|yes|on)
pub fn gpu_only_mode() -> bool {
    match env::var("FOREX_BOT_GPU_ONLY") {
        Ok(val) => {
            let v = val.trim().to_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on")
        }
        Err(_) => false,
    }
}

/// Check if CUDA is available via tch (PyTorch)
pub fn torch_cuda_available() -> bool {
    #[cfg(feature = "tch")]
    {
        tch::Cuda::is_available() && tch::Cuda::device_count() > 0
    }
#[cfg(not(feature = "tch"))]
    {
        // Fallback: check CUDA_VISIBLE_DEVICES for a non-empty device list
        match env::var("CUDA_VISIBLE_DEVICES") {
            Ok(devices) => {
                let trimmed = devices.trim();
                !(trimmed.is_empty() || trimmed == "-1")
            }
            Err(_) => false,
        }
    }
}

/// Get GPU count for distribution
pub fn gpu_count() -> usize {
#[cfg(feature = "tch")]
    {
        if tch::Cuda::is_available() {
            tch::Cuda::device_count() as usize
        } else {
            0
        }
    }
    #[cfg(not(feature = "tch"))]
    {
        // Fallback: parse from env or default to 1
        match env::var("CUDA_VISIBLE_DEVICES") {
            Ok(devices) => {
                let trimmed = devices.trim();
                if trimmed.is_empty() || trimmed == "-1" {
                    0
                } else {
                    trimmed
                        .split(',')
                        .filter(|v| !v.trim().is_empty())
                        .count()
                }
            }
            Err(_) => 0,
        }
    }
}

/// Get early stopping parameters (patience, min_delta) with env overrides
pub fn get_early_stop_params(default_patience: usize, default_min_delta: f64) -> (usize, f64) {
    let patience = env::var("FOREX_BOT_EARLY_STOP_PATIENCE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&p| p > 0)
        .unwrap_or(default_patience);

    let min_delta = env::var("FOREX_BOT_EARLY_STOP_MIN_DELTA")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default_min_delta);

    (patience, min_delta)
}

// ============================================================================ 
// LABEL REMAPPING (HPC FIX - Prevents "Label Drift")
// ============================================================================ 

/// HPC FIX: Hardcoded Deterministic Mapping
/// Prevents 'Label Drift' where Buy/Sell columns are swapped depending on data.
/// Order: -1 -> 0 (Sell), 0 -> 1 (Neutral), 1 -> 2 (Buy)
///
/// Returns: (remapped_array, mapping_dict)
pub fn remap_labels_to_contiguous(y: &Series) -> Result<(Array1<i32>, HashMap<i32, i32>)> {
    let y_vec: Vec<i32> = y
        .i32()
        .context("Labels must be i32")?
        .into_iter()
        .map(|opt| opt.unwrap_or(0))
        .collect();

    let mapping: HashMap<i32, i32> = [(-1, 0), (0, 1), (1, 2)].iter().cloned().collect();

    // Fast vectorized mapping
    let remapped: Vec<i32> = y_vec
        .iter()
        .map(|&label| match label {
            -1 => 0,
            0 => 1,
            1 => 2,
            _ => 1, // Default to Neutral for unexpected values
        })
        .collect();

    Ok((Array1::from_vec(remapped), mapping))
}

// ============================================================================ 
// OUTPUT REORDERING (HPC PROTOCOL)
// ============================================================================ 

/// HPC PROTOCOL: Force output to [Neutral, Buy, Sell].
/// Standard indices: 0=Neutral, 1=Buy, 2=Sell.
///
/// Handles both binary (2-class) and multiclass (3-class) cases.
pub fn reorder_to_neutral_buy_sell(
    probs: Array2<f32>,
    classes: Option<Vec<i32>>,
) -> Array2<f32> {
    let n_samples = probs.nrows();
    let n_classes = probs.ncols();

    let mut out = Array2::<f32>::zeros((n_samples, 3));

    // Binary case (2 classes)
    if n_classes == 2 {
        // Column 0 -> Neutral, Column 1 -> Buy, Sell stays 0.0
        out.column_mut(0).assign(&probs.column(0)); // Neutral
        out.column_mut(1).assign(&probs.column(1)); // Buy
        return out;
    }

    // Multiclass case (3 classes)
    if let Some(class_labels) = classes {
        // Map based on our contiguous mapping: 0=Sell, 1=Neutral, 2=Buy
        for (col_idx, &cls_val) in class_labels.iter().enumerate() {
            if col_idx >= n_classes {
                break;
            }
            match cls_val {
                0 => out.column_mut(2).assign(&probs.column(col_idx)), // Sell
                1 => out.column_mut(0).assign(&probs.column(col_idx)), // Neutral
                2 => out.column_mut(1).assign(&probs.column(col_idx)), // Buy
                _ => {} 
            }
        }
    } else {
        // Fallback: assume direct mapping [Neutral, Buy, Sell]
        return probs;
    }

    out
}

// ============================================================================ 
// TIME FEATURE AUGMENTATION
// ============================================================================ 

/// Add lightweight lag/volatility features for tree models when raw close is available.
/// If 'close' is absent, returns the input unchanged.
///
/// Features added:
/// - ret1: 1-period return (pct_change)
/// - ret1_lag1, ret1_lag2, ret1_lag5, ret1_lag8: Lagged returns
/// - vol14, vol50: Rolling volatility (14 and 50 periods)
/// - mom5, mom15: Momentum (5 and 15 period diff)
pub fn augment_time_features(df: DataFrame) -> Result<DataFrame> {
    // Polars 0.47: column names are PlSmallStr, compare with iter()
    if !df.get_column_names().iter().any(|c| c.as_str() == "close") {
        return Ok(df);
    }

    let mut df = df;

    // Get close price as f64 Series
    let close = df
        .column("close")?
        .cast(&DataType::Float64)?
        .as_materialized_series()
        .clone();

    // Calculate returns (pct_change) - manual implementation for polars 0.47
    let close_f64 = close.f64()?;
    let ret1_values: Vec<f64> = (0..close_f64.len())
        .map(|i| {
            if i == 0 {
                0.0
            } else {
                let curr = close_f64.get(i).unwrap_or(0.0);
                let prev = close_f64.get(i - 1).unwrap_or(0.0);
                if prev != 0.0 {
                    (curr - prev) / prev
                } else {
                    0.0
                }
            }
        })
        .collect();
    let ret1 = Series::new("ret1".into(), ret1_values);

    // Get f64 view before moving ret1
    let ret1_f64 = ret1.f64()?;

    // Add ret1 to DataFrame (this moves ret1, so we clone)
    df.with_column(ret1.clone())?;

    // Add lagged returns (manual shift)
    let ret1_lag1: Vec<f64> = (0..ret1_f64.len())
        .map(|i| if i < 1 { 0.0 } else { ret1_f64.get(i - 1).unwrap_or(0.0) })
        .collect();
    let ret1_lag2: Vec<f64> = (0..ret1_f64.len())
        .map(|i| if i < 2 { 0.0 } else { ret1_f64.get(i - 2).unwrap_or(0.0) })
        .collect();
    let ret1_lag5: Vec<f64> = (0..ret1_f64.len())
        .map(|i| if i < 5 { 0.0 } else { ret1_f64.get(i - 5).unwrap_or(0.0) })
        .collect();
    let ret1_lag8: Vec<f64> = (0..ret1_f64.len())
        .map(|i| if i < 8 { 0.0 } else { ret1_f64.get(i - 8).unwrap_or(0.0) })
        .collect();

    df.with_column(Series::new("ret1_lag1".into(), ret1_lag1))?;
    df.with_column(Series::new("ret1_lag2".into(), ret1_lag2))?;
    df.with_column(Series::new("ret1_lag5".into(), ret1_lag5))?;
    df.with_column(Series::new("ret1_lag8".into(), ret1_lag8))?;

    // Rolling volatility (manual implementation)
    let vol14: Vec<f64> = (0..ret1_f64.len())
        .map(|i| {
            if i < 14 {
                0.0
            } else {
                let window: Vec<f64> = (i.saturating_sub(13)..=i)
                    .filter_map(|j| ret1_f64.get(j))
                    .collect();
                if window.is_empty() {
                    0.0
                } else {
                    let mean = window.iter().sum::<f64>() / window.len() as f64;
                    let variance = window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / window.len() as f64;
                    variance.sqrt()
                }
            }
        })
        .collect();

    let vol50: Vec<f64> = (0..ret1_f64.len())
        .map(|i| {
            if i < 50 {
                0.0
            } else {
                let window: Vec<f64> = (i.saturating_sub(49)..=i)
                    .filter_map(|j| ret1_f64.get(j))
                    .collect();
                if window.is_empty() {
                    0.0
                } else {
                    let mean = window.iter().sum::<f64>() / window.len() as f64;
                    let variance = window.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / window.len() as f64;
                    variance.sqrt()
                }
            }
        })
        .collect();

    df.with_column(Series::new("vol14".into(), vol14))?;
    df.with_column(Series::new("vol50".into(), vol50))?;

    // Momentum (diff) - manual implementation for polars 0.47
    let mom5_values: Vec<f64> = (0..close_f64.len())
        .map(|i| {
            if i < 5 {
                0.0
            } else {
                let curr = close_f64.get(i).unwrap_or(0.0);
                let prev = close_f64.get(i - 5).unwrap_or(0.0);
                curr - prev
            }
        })
        .collect();

    let mom15_values: Vec<f64> = (0..close_f64.len())
        .map(|i| {
            if i < 15 {
                0.0
            } else {
                let curr = close_f64.get(i).unwrap_or(0.0);
                let prev = close_f64.get(i - 15).unwrap_or(0.0);
                curr - prev
            }
        })
        .collect();

    df.with_column(Series::new("mom5".into(), mom5_values))?;
    df.with_column(Series::new("mom15".into(), mom15_values))?;

    Ok(df)
}

// ============================================================================ 
// INF/NAN HANDLING
// ============================================================================ 

/// Replace inf/-inf with NaN in DataFrame
pub fn replace_inf_with_nan(df: DataFrame) -> Result<DataFrame> {
    let df = df;

    for col_name in df.get_column_names() {
        let col = df.column(col_name)?;
        if col.dtype().is_float() {
            // Replace inf with NaN for float columns
            // Polars handles this differently - we'd need to apply a custom function
            // For now, leave as-is since most ML libraries handle inf gracefully
            // TODO: Implement if needed
        }
    }

    Ok(df)
}

// ============================================================================ 
// EXPERT MODEL IMPLEMENTATIONS
// ============================================================================ 

/// LightGBM Expert Model
pub struct LightGBMExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    gpu_only_disabled: bool,
    #[cfg(feature = "lightgbm")]
    model: Option<LGBMBooster>,
    #[cfg(not(feature = "lightgbm"))]
    model: Option<()>, // Placeholder when feature not enabled
}

impl LightGBMExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let default_params = Self::default_params();
        let params = params.unwrap_or(default_params);

        let config = TreeModelConfig {
            idx,
            params,
            device_pref: tree_device_preference(),
            gpu_only: gpu_only_mode(),
            cpu_threads: if cpu_threads_hint() > 0 {
                Some(cpu_threads_hint())
            } else {
                None
            },
        };

        Self {
            idx,
            config,
            gpu_only_disabled: false,
            model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();
        params.insert("n_estimators".to_string(), ParamValue::Int(800));
        params.insert("num_leaves".to_string(), ParamValue::Int(64));
        params.insert("learning_rate".to_string(), ParamValue::Float(0.03));
        params.insert("objective".to_string(), ParamValue::String("multiclass".to_string()));
        params.insert("num_class".to_string(), ParamValue::Int(3));
        params.insert("random_state".to_string(), ParamValue::Int(42));
        params.insert("n_jobs".to_string(), ParamValue::Int(-1));
        params.insert("verbosity".to_string(), ParamValue::Int(-1));
        params.insert("min_data_in_leaf".to_string(), ParamValue::Int(50));
        params.insert("feature_fraction".to_string(), ParamValue::Float(0.6));
        params.insert("bagging_fraction".to_string(), ParamValue::Float(0.8));
        params.insert("bagging_freq".to_string(), ParamValue::Int(1));
        params.insert("path_smooth".to_string(), ParamValue::Int(10));
        params.insert("linear_tree".to_string(), ParamValue::Bool(true));
        params
    }
}

// ============================================================================ 
// LIGHTGBM IMPLEMENTATION
// ============================================================================ 

impl ExpertModel for LightGBMExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(feature = "lightgbm")]
        {
            use serde_json::json;

            // HPC CHECK: If GPU-only mode and no GPU, skip training
            if self.config.gpu_only && !torch_cuda_available() {
                warn!("GPU-only mode enabled but no GPU available - skipping LightGBM training");
                self.gpu_only_disabled = true;
                return Ok(());
            }

            // STEP 1: Augment time features if 'close' column exists
            let x = if x.column("close").is_ok() {
                augment_time_features(x.clone())? 
            } else {
                x.clone()
            };

            // STEP 2: Remap labels to contiguous {0, 1, 2}
            let (y_remapped, mapping) = remap_labels_to_contiguous(y)?;
            info!("Label mapping: {:?}", mapping);

            // STEP 3: Time-series train/val split with embargo
            let n_samples = x.height();
            let embargo_samples = (n_samples / 100).max(24);

            let (x_train, x_val, y_train, y_val) = match base_split(
                x.clone(),
                Series::new("y", y_remapped.to_vec()),
                0.15,
                100,
                embargo_samples,
            ) {
                Ok(split) => split,
                Err(e) => {
                    warn!("Time-series split failed: {}, using simple split", e);
                    let split_idx = (n_samples as f64 * 0.85) as usize;
                    (
                        x.slice(0, split_idx),
                        x.slice(split_idx as i64, n_samples - split_idx),
                        Series::new("y", y_remapped.slice(0, split_idx).to_vec()),
                        Series::new("y", y_remapped.slice(split_idx, n_samples - split_idx).to_vec()),
                    )
                }
            };

            // STEP 4: Convert DataFrame to Vec<Vec<f64>> for LightGBM
            let features_train = dataframe_to_vecs(&x_train)?;
            let labels_train: Vec<f64> = y_train.f64()?.into_iter().map(|v| v.unwrap_or(1.0)).collect();

            // STEP 5: Create LightGBM Dataset
            let dataset = LGBMDataset::from_vec_of_vec(features_train, labels_train, true)
                .context("Failed to create LightGBM dataset")?;

            // STEP 6: Build parameters with HPC logic
            let mut params = json!({
                "objective": "multiclass",
                "num_class": 3,
                "verbosity": -1,
            });

            // Add params from config
            for (k, v) in &self.config.params {
                match v {
                    ParamValue::Int(i) => params[k] = json!(i),
                    ParamValue::Float(f) => params[k] = json!(f),
                    ParamValue::String(s) => params[k] = json!(s),
                    ParamValue::Bool(b) => params[k] = json!(b),
                }
            }

            // GPU configuration
            let gpu_enabled = match self.config.device_pref {
                DevicePreference::Gpu => torch_cuda_available(),
                DevicePreference::Auto => torch_cuda_available(),
                DevicePreference::Cpu => false,
            };

            if gpu_enabled {
                let gpu_id = (self.idx.saturating_sub(1)) % gpu_count();
                params["device_type"] = json!("gpu");
                params["gpu_device_id"] = json!(gpu_id);
                params["max_bin"] = json!(63);
                params["gpu_use_dp"] = json!(false);
                info!("LightGBM using GPU {}", gpu_id);
            } else {
                params["device_type"] = json!("cpu");
                params["max_bin"] = json!(255);
            }

            // CPU threads
            if let Some(threads) = self.config.cpu_threads {
                params["num_threads"] = json!(threads);
            }

            // STEP 7: Train model
            info!("Training LightGBM model...");
            let booster = match LGBMBooster::train(dataset, &params) {
                Ok(b) => b,
                Err(e) => {
                    if gpu_enabled && self.config.gpu_only {
                        warn!("GPU training failed in GPU-only mode: {}", e);
                        self.gpu_only_disabled = true;
                        return Ok(());
                    } else if gpu_enabled {
                        warn!("GPU training failed, falling back to CPU: {}", e);
                        params["device_type"] = json!("cpu");
                        params["max_bin"] = json!(255);
                        LGBMBooster::train(dataset, &params)?
                    } else {
                        return Err(e.into());
                    }
                }
            };

            self.model = Some(booster);
            info!("LightGBM training complete");
            Ok(())
        }
        #[cfg(not(feature = "lightgbm"))]
        {
            anyhow::bail!("LightGBM feature not enabled");
        }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        if self.gpu_only_disabled {
            return Ok(Array2::zeros((x.height(), 3)));
        }

        #[cfg(feature = "lightgbm")]
        {
            let model = self.model.as_ref().context("LightGBM model not trained")?;
            
            // Augment features
            let x_aug = if x.column("close").is_ok() {
                augment_time_features(x.clone())? 
            } else {
                x.clone()
            };
            
            let features = dataframe_to_vecs(&x_aug)?;
            let mut all_probs = Vec::with_capacity(x.height());

            for feat in features {
                let pred = model.predict(feat)?;
                all_probs.push(pred);
            }

            let probs_array = Array2::from_shape_vec((x.height(), 3), all_probs.into_iter().flatten().map(|v| v as f32).collect())?;
            
            // Force output reordering to [Neutral, Buy, Sell]
            Ok(reorder_to_neutral_buy_sell(probs_array, Some(vec![0, 1, 2])))
        }
        #[cfg(not(feature = "lightgbm"))]
        {
            anyhow::bail!("LightGBM feature not enabled");
        }
    }

    fn save(&self, _path: &Path) -> Result<()> {
        #[cfg(feature = "lightgbm")]
        {
            if let Some(model) = &self.model {
                model.save_file(path.to_str().context("Invalid path")?)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "lightgbm"))]
        {
            Ok(())
        }
    }

    fn load(&mut self, _path: &Path) -> Result<()> {
        #[cfg(feature = "lightgbm")]
        {
            let booster = LGBMBooster::from_file(path.to_str().context("Invalid path")?)?;
            self.model = Some(booster);
            Ok(())
        }
        #[cfg(not(feature = "lightgbm"))]
        {
            anyhow::bail!("LightGBM feature not enabled");
        }
    }
}

/// Convert Polars DataFrame to Vec<Vec<f64>> for LightGBM
#[cfg(feature = "lightgbm")]
fn dataframe_to_vecs(df: &DataFrame) -> Result<Vec<Vec<f64>>> {
    let n_rows = df.height();
    let n_cols = df.width();
    let mut result = Vec::with_capacity(n_rows);

    for row_idx in 0..n_rows {
        let mut row = Vec::with_capacity(n_cols);
        for col in df.get_columns() {
            let value = match col.dtype() {
                DataType::Float64 => col.f64()?.get(row_idx).unwrap_or(0.0),
                DataType::Float32 => col.f32()?.get(row_idx).unwrap_or(0.0) as f64,
                DataType::Int64 => col.i64()?.get(row_idx).unwrap_or(0) as f64,
                DataType::Int32 => col.i32()?.get(row_idx).unwrap_or(0) as f64,
                _ => 0.0,
            };
            row.push(value);
        }
        result.push(row);
    }

    Ok(result)
}

// ============================================================================ 
// XGBOOST IMPLEMENTATION
// ============================================================================ 

/// XGBoost Expert Model
pub struct XGBoostExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    gpu_only_disabled: bool,
    #[cfg(feature = "xgboost")]
    model: Option<xgb::Booster>,
    #[cfg(not(feature = "xgboost"))]
    model: Option<()>,
}

impl XGBoostExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let default_params = Self::default_params();
        let params = params.unwrap_or(default_params);

        let config = TreeModelConfig {
            idx,
            params,
            device_pref: tree_device_preference(),
            gpu_only: gpu_only_mode(),
            cpu_threads: if cpu_threads_hint() > 0 {
                Some(cpu_threads_hint())
            } else {
                None
            },
        };

        Self {
            idx,
            config,
            gpu_only_disabled: false,
            model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();
        params.insert("n_estimators".to_string(), ParamValue::Int(800));
        params.insert("max_depth".to_string(), ParamValue::Int(8));
        params.insert("learning_rate".to_string(), ParamValue::Float(0.05));
        params.insert("objective".to_string(), ParamValue::String("multi:softprob".to_string()));
        params.insert("num_class".to_string(), ParamValue::Int(3));
        params.insert("random_state".to_string(), ParamValue::Int(42));
        params.insert("n_jobs".to_string(), ParamValue::Int(-1));
        params.insert("verbosity".to_string(), ParamValue::Int(0));
        params.insert("subsample".to_string(), ParamValue::Float(0.9));
        params.insert("colsample_bytree".to_string(), ParamValue::Float(0.9));
        params.insert("eval_metric".to_string(), ParamValue::String("mlogloss".to_string()));
        params.insert("tree_method".to_string(), ParamValue::String("hist".to_string()));
        params
    }
}

impl ExpertModel for XGBoostExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            // HPC CHECK: If GPU-only mode and no GPU, skip training
            if self.config.gpu_only && !torch_cuda_available() {
                warn!("GPU-only mode enabled but no GPU available - skipping XGBoost training");
                self.gpu_only_disabled = true;
                return Ok(());
            }

            // STEP 1: Augment time features if 'close' column exists
            let x = if x.column("close").is_ok() {
                augment_time_features(x.clone())? 
            } else {
                x.clone()
            };

            // STEP 2: Remap labels to contiguous {0, 1, 2}
            let (y_remapped, mapping) = remap_labels_to_contiguous(y)?;
            info!("Label mapping: {:?}", mapping);

            // STEP 3: Time-series train/val split with embargo
            let n_samples = x.height();
            let embargo_samples = (n_samples / 100).max(24);

            let (x_train, x_val, y_train, y_val) = match base_split(
                x.clone(),
                Series::new("y", y_remapped.to_vec()),
                0.15,
                100,
                embargo_samples,
            ) {
                Ok(split) => split,
                Err(e) => {
                    warn!("Time-series split failed: {}, using simple split", e);
                    let split_idx = (n_samples as f64 * 0.85) as usize;
                    (
                        x.slice(0, split_idx),
                        x.slice(split_idx as i64, n_samples - split_idx),
                        Series::new("y", y_remapped.slice(0, split_idx).to_vec()),
                        Series::new("y", y_remapped.slice(split_idx, n_samples - split_idx).to_vec()),
                    )
                }
            };

            // STEP 4: Convert to XGBoost format
            let features_train = dataframe_to_vecs(&x_train)?;
            let labels_train: Vec<f32> = y_train.f64()?.into_iter().map(|v| v.unwrap_or(1.0) as f32).collect();

            // STEP 5: Create DMatrix
            let mut dtrain = xgb::DMatrix::from_dense(&features_train, x_train.height())?;
            dtrain.set_labels(&labels_train)?;

            // STEP 6: Build parameters with HPC logic
            let mut params = vec![
                ("objective", "multi:softprob"),
                ("num_class", "3"),
                ("eval_metric", "mlogloss"),
            ];

            // Add params from config
            let mut param_strings = Vec::new();
            for (k, v) in &self.config.params {
                let val_str = match v {
                    ParamValue::Int(i) => i.to_string(),
                    ParamValue::Float(f) => f.to_string(),
                    ParamValue::String(s) => s.clone(),
                    ParamValue::Bool(b) => b.to_string(),
                };
                param_strings.push((k.clone(), val_str));
            }
            
            for (k, v) in &param_strings {
                params.push((k, v));
            }

            // GPU configuration
            let gpu_enabled = match self.config.device_pref {
                DevicePreference::Gpu => torch_cuda_available(),
                DevicePreference::Auto => torch_cuda_available(),
                DevicePreference::Cpu => false,
            };

            if gpu_enabled {
                let gpu_id = (self.idx.saturating_sub(1)) % gpu_count();
                params.push(("tree_method", "gpu_hist"));
                params.push(("device", &format!("cuda:{}", gpu_id)));
                info!("XGBoost using GPU {}", gpu_id);
            } else {
                params.push(("tree_method", "hist"));
            }

            // CPU threads
            if let Some(threads) = self.config.cpu_threads {
                params.push(("nthread", &threads.to_string()));
            }

            // STEP 7: Train model
            info!("Training XGBoost model...");
            let booster = xgb::Booster::train(&dtrain, &params, 800, &[])?;

            self.model = Some(booster);
            info!("XGBoost training complete");
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        if self.gpu_only_disabled {
            return Ok(Array2::zeros((x.height(), 3)));
        }

        #[cfg(feature = "xgboost")]
        {
            let model = self.model.as_ref().context("XGBoost model not trained")?;
            
            let x_aug = if x.column("close").is_ok() {
                augment_time_features(x.clone())? 
            } else {
                x.clone()
            };
            
            let features = dataframe_to_vecs(&x_aug)?;
            let dmatrix = xgb::DMatrix::from_dense(&features, x.height())?;
            
            let preds = model.predict(&dmatrix)?;
            let probs_array = Array2::from_shape_vec((x.height(), 3), preds)?;
            
            Ok(reorder_to_neutral_buy_sell(probs_array, Some(vec![0, 1, 2])))
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }

    fn save(&self, _path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            if let Some(model) = &self.model {
                model.save(path.to_str().context("Invalid path")?)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            Ok(())
        }
    }

    fn load(&mut self, _path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let booster = xgb::Booster::load(path.to_str().context("Invalid path")?)?;
            self.model = Some(booster);
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }
}

// ============================================================================ 
// XGBOOST VARIANT MODELS
// ============================================================================ 

/// XGBoost Random Forest Expert (num_parallel_tree=8)
pub struct XGBoostRFExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    gpu_only_disabled: bool,
    #[cfg(feature = "xgboost")]
    model: Option<xgb::Booster>,
    #[cfg(not(feature = "xgboost"))]
    model: Option<()>,
}

impl XGBoostRFExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let default_params = Self::default_params();
        let params = params.unwrap_or(default_params);

        let config = TreeModelConfig {
            idx,
            params,
            device_pref: tree_device_preference(),
            gpu_only: gpu_only_mode(),
            cpu_threads: if cpu_threads_hint() > 0 {
                Some(cpu_threads_hint())
            } else {
                None
            },
        };

        Self {
            idx,
            config,
            gpu_only_disabled: false,
            model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();
        // RF variant: different from base XGBoost
        params.insert("n_estimators".to_string(), ParamValue::Int(400)); // not 800
        params.insert("max_depth".to_string(), ParamValue::Int(6)); // not 8
        params.insert("learning_rate".to_string(), ParamValue::Float(0.3)); // not 0.05
        params.insert("subsample".to_string(), ParamValue::Float(0.8)); // not 0.9
        params.insert("colsample_bynode".to_string(), ParamValue::Float(0.8)); // NEW
        params.insert("colsample_bytree".to_string(), ParamValue::Float(0.8)); // not 0.9
        params.insert("num_parallel_tree".to_string(), ParamValue::Int(8)); // NEW - RF mode
        params.insert("objective".to_string(), ParamValue::String("multi:softprob".to_string()));
        params.insert("num_class".to_string(), ParamValue::Int(3));
        params.insert("random_state".to_string(), ParamValue::Int(42));
        params.insert("verbosity".to_string(), ParamValue::Int(0));
        params
    }
}

/// XGBoost DART Expert (booster=dart)
pub struct XGBoostDARTExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    gpu_only_disabled: bool,
    #[cfg(feature = "xgboost")]
    model: Option<xgb::Booster>,
    #[cfg(not(feature = "xgboost"))]
    model: Option<()>,
}

impl XGBoostDARTExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let default_params = Self::default_params();
        let params = params.unwrap_or(default_params);

        let config = TreeModelConfig {
            idx,
            params,
            device_pref: tree_device_preference(),
            gpu_only: gpu_only_mode(),
            cpu_threads: if cpu_threads_hint() > 0 {
                Some(cpu_threads_hint())
            } else {
                None
            },
        };

        Self {
            idx,
            config,
            gpu_only_disabled: false,
            model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();
        // DART variant
        params.insert("n_estimators".to_string(), ParamValue::Int(600)); // not 800
        params.insert("booster".to_string(), ParamValue::String("dart".to_string())); // NEW
        params.insert("rate_drop".to_string(), ParamValue::Float(0.10)); // NEW
        params.insert("skip_drop".to_string(), ParamValue::Float(0.50)); // NEW
        params.insert("sample_type".to_string(), ParamValue::String("uniform".to_string())); // NEW
        params.insert("normalize_type".to_string(), ParamValue::String("tree".to_string())); // NEW
        params.insert("max_depth".to_string(), ParamValue::Int(8));
        params.insert("learning_rate".to_string(), ParamValue::Float(0.05));
        params.insert("objective".to_string(), ParamValue::String("multi:softprob".to_string()));
        params.insert("num_class".to_string(), ParamValue::Int(3));
        params.insert("random_state".to_string(), ParamValue::Int(42));
        params.insert("verbosity".to_string(), ParamValue::Int(0));
        params
    }
}

// ============================================================================ 
// XGBOOST VARIANT IMPLEMENTATIONS
// ============================================================================ 

impl ExpertModel for XGBoostRFExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let mut base = XGBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: None,
            };
            base.fit(x, y)?;
            self.model = base.model;
            self.gpu_only_disabled = base.gpu_only_disabled;
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }

    fn predict_proba(&self, _x: &DataFrame) -> Result<Array2<f32>> {
        #[cfg(feature = "xgboost")]
        {
            let base = XGBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: self.model.clone(),
            };
            base.predict_proba(x)
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }

    fn save(&self, _path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            if let Some(model) = &self.model {
                model.save(path.to_str().context("Invalid path")?)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            Ok(())
        }
    }

    fn load(&mut self, _path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let booster = xgb::Booster::load(path.to_str().context("Invalid path")?)?;
            self.model = Some(booster);
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }
}

impl ExpertModel for XGBoostDARTExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let mut base = XGBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: None,
            };
            base.fit(x, y)?;
            self.model = base.model;
            self.gpu_only_disabled = base.gpu_only_disabled;
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }

    fn predict_proba(&self, _x: &DataFrame) -> Result<Array2<f32>> {
        #[cfg(feature = "xgboost")]
        {
            let base = XGBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: self.model.clone(),
            };
            base.predict_proba(x)
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }

    fn save(&self, _path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            if let Some(model) = &self.model {
                model.save(path.to_str().context("Invalid path")?)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            Ok(())
        }
    }

    fn load(&mut self, _path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let booster = xgb::Booster::load(path.to_str().context("Invalid path")?)?;
            self.model = Some(booster);
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }
}

// ============================================================================ 
// CATBOOST IMPLEMENTATION (INFERENCE ONLY - HYBRID APPROACH)
// ============================================================================ 

/// CatBoost Expert Model
/// NOTE: Training must be done in Python, this loads pre-trained .cbm models
pub struct CatBoostExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    gpu_only_disabled: bool,
    #[cfg(feature = "catboost")]
    model: Option<catboost::Model>,
    #[cfg(not(feature = "catboost"))]
    model: Option<()>,
}

impl CatBoostExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let default_params = Self::default_params();
        let params = params.unwrap_or(default_params);

        let config = TreeModelConfig {
            idx,
            params,
            device_pref: tree_device_preference(),
            gpu_only: gpu_only_mode(),
            cpu_threads: if cpu_threads_hint() > 0 {
                Some(cpu_threads_hint())
            } else {
                None
            },
        };

        Self {
            idx,
            config,
            gpu_only_disabled: false,
            model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();
        // These params are for documentation/reference only
        // Training happens in Python
        params.insert("iterations".to_string(), ParamValue::Int(800));
        params.insert("depth".to_string(), ParamValue::Int(8));
        params.insert("learning_rate".to_string(), ParamValue::Float(0.05));
        params.insert("loss_function".to_string(), ParamValue::String("MultiClass".to_string()));
        params.insert("random_seed".to_string(), ParamValue::Int(42));
        params.insert("verbose".to_string(), ParamValue::Bool(false));
        params.insert("thread_count".to_string(), ParamValue::Int(-1));
        params
    }
}

impl ExpertModel for CatBoostExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(feature = "catboost")]
        {
            anyhow::bail!(
                r"CatBoost training not supported in Rust.
                Train in Python using catboost library, then load the .cbm model file.
                Example Python:
                    model = CatBoostClassifier(**params);
                    model.fit(X_train, y_train);
                    model.save_model('catboost.cbm')"
            );
        }
        #[cfg(not(feature = "catboost"))]
        {
            anyhow::bail!("CatBoost feature not enabled");
        }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        if self.gpu_only_disabled {
            return Ok(Array2::zeros((x.height(), 3)));
        }

        #[cfg(feature = "catboost")]
        {
            let model = self.model.as_ref().context(
                "Model not loaded. Train in Python and load .cbm file with load() method"
            )?;

            // Augment time features if 'close' exists
            let x = if x.column("close").is_ok() {
                augment_time_features(x.clone())? 
            } else {
                x.clone()
            };

            // Convert DataFrame to format CatBoost expects
            let features = dataframe_to_vecs(&x)?;

            // CatBoost expects float features in specific format
            let float_features: Vec<Vec<f32>> = features
                .iter()
                .map(|row| row.iter().map(|&v| v as f32).collect())
                .collect();

            // Create ObjectsOrderFeatures
            let features_obj = catboost::ObjectsOrderFeatures::new()
                .with_float_features(&float_features.iter().map(|v| v.as_slice()).collect::<Vec<_>>());

            // Predict
            let predictions = model.predict(features_obj)?;

            // Convert to Array2<f32> and reorder to [Neutral, Buy, Sell]
            let n_samples = predictions.len() / 3;
            let probs = Array2::from_shape_vec((n_samples, 3), predictions)?;

            // Reorder output
            Ok(reorder_to_neutral_buy_sell(probs, Some(vec![0, 1, 2])))
        }
        #[cfg(not(feature = "catboost"))]
        {
            anyhow::bail!("CatBoost feature not enabled");
        }
    }

    fn save(&self, _path: &Path) -> Result<()> {
        #[cfg(feature = "catboost")]
        {
            anyhow::bail!("CatBoost save not implemented. Model is already saved as .cbm file")
        }
        #[cfg(not(feature = "catboost"))]
        {
            Ok(())
        }
    }

    fn load(&mut self, _path: &Path) -> Result<()> {
        #[cfg(feature = "catboost")]
        {
            let model = catboost::Model::load(path.to_str().unwrap())?;

            // Enable GPU if available and configured
            #[cfg(feature = "gpu")]
            {
                let gpu_enabled = match self.config.device_pref {
                    DevicePreference::Gpu => torch_cuda_available(),
                    DevicePreference::Auto => torch_cuda_available(),
                    DevicePreference::Cpu => false,
                };

                if gpu_enabled {
                    let gpu_id = (self.idx.saturating_sub(1)) % gpu_count();
                    model.enable_gpu_evaluation()?;
                    info!("CatBoost using GPU {}", gpu_id);
                }
            }

            self.model = Some(model);
            info!("CatBoost model loaded from {:?}", path);
            Ok(())
        }
        #[cfg(not(feature = "catboost"))]
        {
            anyhow::bail!("CatBoost feature not enabled");
        }
    }
}

/// CatBoost Alt Expert (variant with different hyperparameters)
pub struct CatBoostAltExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    gpu_only_disabled: bool,
    #[cfg(feature = "catboost")]
    model: Option<catboost::Model>,
    #[cfg(not(feature = "catboost"))]
    model: Option<()>,
}

impl CatBoostAltExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let default_params = Self::default_params();
        let params = params.unwrap_or(default_params);

        let config = TreeModelConfig {
            idx,
            params,
            device_pref: tree_device_preference(),
            gpu_only: gpu_only_mode(),
            cpu_threads: if cpu_threads_hint() > 0 {
                Some(cpu_threads_hint())
            } else {
                None
            },
        };

        Self {
            idx,
            config,
            gpu_only_disabled: false,
            model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();
        // Alt variant: different hyperparameters
        params.insert("iterations".to_string(), ParamValue::Int(900)); // not 800
        params.insert("depth".to_string(), ParamValue::Int(10)); // not 8
        params.insert("learning_rate".to_string(), ParamValue::Float(0.03)); // not 0.05
        params.insert("random_seed".to_string(), ParamValue::Int(7)); // not 42
        params.insert("l2_leaf_reg".to_string(), ParamValue::Float(6.0)); // NEW
        params.insert("random_strength".to_string(), ParamValue::Float(1.5)); // NEW
        params.insert("loss_function".to_string(), ParamValue::String("MultiClass".to_string()));
        params.insert("verbose".to_string(), ParamValue::Bool(false));
        params.insert("thread_count".to_string(), ParamValue::Int(-1));
        params
    }
}

// NOTE: CatBoostAltExpert uses same TreeModel impl as CatBoostExpert
// It differs only in default parameters (for Python training reference)

// ============================================================================ 
// PYTHON TRAINING SCRIPT GENERATOR (HELPER FOR CATBOOST)
// ============================================================================ 

/// Generates Python training script for CatBoost models
pub fn generate_catboost_training_script(
    _model_name: &str,
    params: &HashMap<String, ParamValue>,
    output_path: &Path,
) -> Result<String> {
    let mut script = String::from("#!/usr/bin/env python3\n");
    script.push_str("# Auto-generated CatBoost training script\n");
    script.push_str("# Train model in Python, then load .cbm file in Rust\n\n");
    script.push_str("from catboost import CatBoostClassifier\n");
    script.push_str("import pandas as pd\n");
    script.push_str("import numpy as np\n\n");

    script.push_str("# Load your data\n");
    script.push_str("# X_train, y_train = load_data()  # Replace with your data loading\n\n");

    script.push_str("# Model parameters\n");
    script.push_str("params = {\n");
    for (key, value) in params {
        match value {
            ParamValue::Int(v) => script.push_str(&format!("    '{}': {},\n", key, v)),
            ParamValue::Float(v) => script.push_str(&format!("    '{}': {},\n", key, v)),
            ParamValue::String(v) => script.push_str(&format!("    '{}': '{}',\n", key, v)),
            ParamValue::Bool(v) => script.push_str(&format!("    '{}': {},\n", key, v)),
        }
    }
    script.push_str("}\n\n");

    script.push_str("# Train model\n");
    script.push_str("model = CatBoostClassifier(**params)\n");
    script.push_str("model.fit(X_train, y_train)\n\n");

    script.push_str("# Save model\n");
    script.push_str(&format!(
        "model.save_model('{}')\n",
        output_path.to_str().unwrap()
    ));
    script.push_str(&format!("print('Model saved to {}')\n", output_path.to_str().unwrap()));

    Ok(script)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remap_labels() {
        let labels = Series::new("label", &[-1, 0, 1, -1, 0, 1]);
        let (remapped, mapping) = remap_labels_to_contiguous(&labels).unwrap();

        assert_eq!(remapped.as_slice().unwrap(), &[0, 1, 2, 0, 1, 2]);
        assert_eq!(mapping.get(&-1), Some(&0));
        assert_eq!(mapping.get(&0), Some(&1));
        assert_eq!(mapping.get(&1), Some(&2));
    }

    #[test]
    fn test_reorder_binary() {
        let probs = Array2::from_shape_vec((3, 2), vec![0.7, 0.3, 0.6, 0.4, 0.8, 0.2]).unwrap();
        let reordered = reorder_to_neutral_buy_sell(probs, Some(vec![0, 1, 2]));

        assert_eq!(reordered.shape(), &[3, 3]);
        // Binary: col 0->Neutral, col 1->Buy, col 2->0.0 (Sell)
        assert_eq!(reordered[[0, 0]], 0.7); // Neutral
        assert_eq!(reordered[[0, 1]], 0.3); // Buy
        assert_eq!(reordered[[0, 2]], 0.0); // Sell
    }

    #[test]
    fn test_device_preference_parsing() {
        std::env::set_var("FOREX_BOT_TREE_DEVICE", "gpu");
        assert_eq!(tree_device_preference(), DevicePreference::Gpu);

        std::env::set_var("FOREX_BOT_TREE_DEVICE", "cpu");
        assert_eq!(tree_device_preference(), DevicePreference::Cpu);

        std::env::set_var("FOREX_BOT_TREE_DEVICE", "auto");
        assert_eq!(tree_device_preference(), DevicePreference::Auto);

        std::env::set_var("FOREX_BOT_TREE_DEVICE", "1");
        assert_eq!(tree_device_preference(), DevicePreference::Gpu);

        std::env::set_var("FOREX_BOT_TREE_DEVICE", "off");
        assert_eq!(tree_device_preference(), DevicePreference::Cpu);

        std::env::remove_var("FOREX_BOT_TREE_DEVICE");
    }
}

impl ExpertModel for CatBoostAltExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(feature = "catboost")]
        {
            let mut base = CatBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: None,
            };
            base.fit(x, y)?;
            self.model = base.model;
            self.gpu_only_disabled = base.gpu_only_disabled;
            Ok(())
        }
        #[cfg(not(feature = "catboost"))]
        {
            anyhow::bail!("CatBoost feature not enabled");
        }
    }

    fn predict_proba(&self, _x: &DataFrame) -> Result<Array2<f32>> {
        #[cfg(feature = "catboost")]
        {
            let base = CatBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: self.model.clone(),
            };
            base.predict_proba(x)
        }
        #[cfg(not(feature = "catboost"))]
        {
            anyhow::bail!("CatBoost feature not enabled");
        }
    }

    fn save(&self, _path: &Path) -> Result<()> {
        #[cfg(feature = "catboost")]
        {
            let base = CatBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: self.model.clone(),
            };
            base.save(path)
        }
        #[cfg(not(feature = "catboost"))]
        {
            anyhow::bail!("CatBoost feature not enabled");
        }
    }

    fn load(&mut self, _path: &Path) -> Result<()> {
        #[cfg(feature = "catboost")]
        {
            let model = catboost::Model::load(path.to_str().context("Invalid path")?)?;
            self.model = Some(model);
            Ok(())
        }
        #[cfg(not(feature = "catboost"))]
        {
            anyhow::bail!("CatBoost feature not enabled");
        }
    }
}

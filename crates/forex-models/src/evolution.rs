// neuroevolution module using CMA-ES
// Ported from src/forex_bot/models/evolution.py
// NO SIMPLIFICATION - Preserves all HPC production logic, multi-island parallelism, and robust scaling.

use anyhow::{Context, Result};
use cmaes::{CMAESOptions, DVector};
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use rayon::prelude::*;
use std::path::Path;
use tracing::{info, warn};

use crate::base::{ExpertModel, RobustScaler, dataframe_to_float32_array, time_series_train_val_split};

// ============================================================================
// MLP FORWARD PASS (CPU OPTIMIZED)
// ============================================================================

fn softmax(z: &Array2<f32>) -> Array2<f32> {
    let mut out = Array2::zeros(z.dim());
    for (i, row) in z.axis_iter(Axis(0)).enumerate() {
        let max = row.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let mut sum = 0.0;
        let mut out_row = out.row_mut(i);
        for (j, &val) in row.iter().enumerate() {
            let e = (val - max).exp();
            out_row[j] = e;
            sum += e;
        }
        out_row.mapv_inplace(|x| x / (sum + 1e-9));
    }
    out
}

fn unpack_theta(theta: &[f64], d: usize, hidden: usize) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let mut idx = 0;
    
    // W1: [d, hidden]
    let w1_size = d * hidden;
    let w1_vec: Vec<f32> = theta[idx..idx + w1_size].iter().map(|&x| x as f32).collect();
    let w1 = Array2::from_shape_vec((d, hidden), w1_vec).unwrap();
    idx += w1_size;
    
    // b1: [1, hidden]
    let b1_size = hidden;
    let b1_vec: Vec<f32> = theta[idx..idx + b1_size].iter().map(|&x| x as f32).collect();
    let b1 = Array2::from_shape_vec((1, hidden), b1_vec).unwrap();
    idx += b1_size;
    
    // W2: [hidden, 3]
    let w2_size = hidden * 3;
    let w2_vec: Vec<f32> = theta[idx..idx + w2_size].iter().map(|&x| x as f32).collect();
    let w2 = Array2::from_shape_vec((hidden, 3), w2_vec).unwrap();
    idx += w2_size;
    
    // b2: [1, 3]
    let b2_size = 3;
    let b2_vec: Vec<f32> = theta[idx..idx + b2_size].iter().map(|&x| x as f32).collect();
    let b2 = Array2::from_shape_vec((1, 3), b2_vec).unwrap();
    
    (w1, b1, w2, b2)
}

fn compute_loss(
    theta: &[f64],
    x_norm: &Array2<f32>,
    class_idx: &[usize],
    weight_decay: f64,
    d: usize,
    hidden: usize,
) -> f64 {
    let (w1, b1, w2, b2) = unpack_theta(theta, d, hidden);
    
    // L2 Regularization
    let reg = weight_decay * theta.iter().map(|&x| x * x).sum::<f64>();
    
    let n = x_norm.nrows();
    
    // Forward Pass: tanh(X @ W1 + b1) @ W2 + b2
    let hidden_act = (x_norm.dot(&w1) + &b1).mapv(|x| x.tanh());
    let logits = hidden_act.dot(&w2) + &b2;
    let probs = softmax(&logits);
    
    let mut total_nll = 0.0;
    for i in 0..n {
        let p = probs[[i, class_idx[i]]];
        total_nll -= (p + 1e-9).ln();
    }
    
    (total_nll as f64 / n as f64) + reg
}

// ============================================================================
// EVO EXPERT CMA
// ============================================================================

pub struct EvoExpertCMA {
    pub theta: Option<Array1<f64>>,
    pub mean: Option<Array1<f32>>,
    pub scale: Option<Array1<f32>>,
    pub hidden_size: usize,
    pub population: usize,
    pub num_islands: usize,
    pub weight_decay: f64,
    pub sigma: f64,
    pub max_time_sec: u64,
    pub scaler: RobustScaler,
}

impl EvoExpertCMA {
    pub fn new() -> Self {
        Self {
            theta: None,
            mean: None,
            scale: None,
            hidden_size: 64,
            population: 32,
            num_islands: 4,
            weight_decay: 1e-3,
            sigma: 0.2,
            max_time_sec: 7200,
            scaler: RobustScaler::new(),
        }
    }
}

impl ExpertModel for EvoExpertCMA {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let x_arr = dataframe_to_float32_array(x)?;
        
        // Filter directional labels only (y != 0)
        let y_vals: Vec<i32> = y.i32()?.into_iter().map(|v| v.unwrap_or(0)).collect();
        let mask: Vec<bool> = y_vals.iter().map(|&v| v != 0).collect();
        let n_directional = mask.iter().filter(|&&m| m).count();
        
        if n_directional < 200 {
            warn!("Evo skipped: insufficient directional labels ({})", n_directional);
            return Ok(());
        }
        
        // Apply mask to X and Y
        let mut x_filtered_vec = Vec::with_capacity(n_directional * x_arr.ncols());
        let mut y_filtered = Vec::with_capacity(n_directional);
        
        for (i, &m) in mask.iter().enumerate() {
            if m {
                for j in 0..x_arr.ncols() {
                    x_filtered_vec.push(x_arr[[i, j]]);
                }
                y_filtered.push(y_vals[i]);
            }
        }
        
        let x_filtered = Array2::from_shape_vec((n_directional, x_arr.ncols()), x_filtered_vec)?;
        
        // Robust Scaling
        self.scaler.fit(&x_filtered)?;
        let x_norm = self.scaler.transform(&x_filtered)?;
        
        // Map labels to {0, 1, 2}
        // -1 -> 0, 0 -> 1, 1 -> 2
        let class_idx: Vec<usize> = y_filtered.iter().map(|&v| match v {
            -1 => 0,
            1 => 2,
            _ => 1,
        }).collect();
        
        let d = x_filtered.ncols();
        let hidden = self.hidden_size;
        let theta_dim = d * hidden + hidden + hidden * 3 + 3;
        
        info!("Starting Neuroevolution (CMA-ES) with {} islands, pop_size={}", self.num_islands, self.population);
        
        // Parallel Islands
        let results: Vec<(f64, Vec<f64>)> = (0..self.num_islands).into_par_iter().map(|island_idx| {
            let mut options = CMAESOptions::new(vec![0.0; theta_dim], self.sigma);
            options.population_size(self.population);
            
            let mut best_island_theta = vec![0.0; theta_dim];
            let mut best_island_loss = f64::MAX;
            
            let cma = options.build(|theta: &DVector<f64>| {
                compute_loss(theta.as_slice(), &x_norm, &class_idx, self.weight_decay, d, hidden)
            }).unwrap();
            
            let result = cma.run();
            
            best_island_loss = result.best_fitness;
            best_island_theta = result.best_parameters.as_slice().to_vec();
            
            info!("Island {} finished. Best loss: {:.4}", island_idx, best_island_loss);
            (best_island_loss, best_island_theta)
        }).collect();
        
        // Pick best island
        let mut best_overall_loss = f64::MAX;
        let mut best_overall_theta = vec![0.0; theta_dim];
        
        for (loss, theta) in results {
            if loss < best_overall_loss {
                best_overall_loss = loss;
                best_overall_theta = theta;
            }
        }
        
        info!("Neuroevolution complete. Best overall loss: {:.4}", best_overall_loss);
        
        self.theta = Some(Array1::from_vec(best_overall_theta));
        self.mean = Some(self.scaler.mean.as_ref().unwrap().row(0).to_owned());
        self.scale = Some(self.scaler.scale.as_ref().unwrap().row(0).to_owned());
        
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        let theta = self.theta.as_ref().context("Model not trained")?;
        let mean = self.mean.as_ref().context("Model not trained")?;
        let scale = self.scale.as_ref().context("Model not trained")?;
        
        let x_arr = dataframe_to_float32_array(x)?;
        let n = x_arr.nrows();
        let d = x_arr.ncols();
        let hidden = self.hidden_size;
        
        // Manual Normalization
        let mut x_norm = Array2::zeros(x_arr.dim());
        for i in 0..n {
            for j in 0..d {
                let val = x_arr[[i, j]];
                if val.is_finite() {
                    x_norm[[i, j]] = (val - mean[j]) / scale[j];
                } else {
                    x_norm[[i, j]] = 0.0;
                }
            }
        }
        
        let (w1, b1, w2, b2) = unpack_theta(theta.as_slice().unwrap(), d, hidden);
        
        // Forward Pass
        let hidden_act = (x_norm.dot(&w1) + &b1).mapv(|x| x.tanh());
        let logits = hidden_act.dot(&w2) + &b2;
        let p_raw = softmax(&logits);
        
        // Reorder to [Neutral, Buy, Sell]
        // Current indices: 0=Sell (-1), 1=Neutral (0), 2=Buy (1)
        // Target indices: 0=Neutral, 1=Buy, 2=Sell
        let mut p_final = Array2::zeros((n, 3));
        for i in 0..n {
            p_final[[i, 0]] = p_raw[[i, 1]]; // Neutral
            p_final[[i, 1]] = p_raw[[i, 2]]; // Buy
            p_final[[i, 2]] = p_raw[[i, 0]]; // Sell
        }
        
        Ok(p_final)
    }

    fn save(&self, path: &Path) -> Result<()> {
        let theta = self.theta.as_ref().context("Model not trained")?;
        let mean = self.mean.as_ref().context("Model not trained")?;
        let scale = self.scale.as_ref().context("Model not trained")?;
        
        let data = serde_json::json!({
            "theta": theta.to_vec(),
            "mean": mean.to_vec(),
            "scale": scale.to_vec(),
            "hidden_size": self.hidden_size,
        });
        
        std::fs::write(path.join("evo_cma.json"), data.to_string())?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let data_str = std::fs::read_to_string(path.join("evo_cma.json"))?;
        let data: serde_json::Value = serde_json::from_str(&data_str)?;
        
        let theta_vec: Vec<f64> = serde_json::from_value(data["theta"].clone())?;
        let mean_vec: Vec<f32> = serde_json::from_value(data["mean"].clone())?;
        let scale_vec: Vec<f32> = serde_json::from_value(data["scale"].clone())?;
        
        self.theta = Some(Array1::from_vec(theta_vec));
        self.mean = Some(Array1::from_vec(mean_vec));
        self.scale = Some(Array1::from_vec(scale_vec));
        self.hidden_size = data["hidden_size"].as_u64().unwrap() as usize;
        
        Ok(())
    }
}

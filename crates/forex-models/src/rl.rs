// Reinforcement Learning Module
// Ported from src/forex_bot/models/rl.py and rllib_agent.py
// Implements ExpertModel for PPO and SAC agents via ONNX inference.
// Training is delegated to Python (Stable-Baselines3 / RLlib).

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
#[cfg(feature = "onnx")]
use ort::{inputs, Session, SessionBuilder, Value};
use std::path::Path;
use tracing::{info, warn};

use crate::base::{ExpertModel, RobustScaler, dataframe_to_float32_array, validate_time_ordering};

// ============================================================================ 
// RL EXPERT (PPO/SAC) - Stable Baselines 3
// ============================================================================ 

pub struct RLExpert {
    pub name: String,
    #[cfg(feature = "onnx")]
    pub session: Option<Session>,
    pub scaler: RobustScaler,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl RLExpert {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            #[cfg(feature = "onnx")]
            session: None,
            scaler: RobustScaler::new(),
            input_dim: 0,
            output_dim: 3, // Neural, Buy, Sell
        }
    }

    #[cfg(feature = "onnx")]
    fn load_onnx_session(&mut self, path: &Path) -> Result<()> {
        let session = Session::builder()? 
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(path)?;
        
        self.session = Some(session);
        info!("Loaded ONNX model for {} from {:?}", self.name, path);
        Ok(())
    }
}

impl ExpertModel for RLExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        anyhow::bail!(
            "RL training (PPO/SAC) must be done in Python using Stable-Baselines3. \
            Train the model, export to ONNX, then load it here."
        );
    }

    #[cfg(feature = "onnx")]
    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        let session = self.session.as_ref().context("ONNX session not loaded")?;
        
        // 1. Preprocess
        // Ensure data is float32
        let x_arr = dataframe_to_float32_array(x)?;
        
        // Apply scaling if fitted (loaded from metadata)
        // RL usually expects raw or normalized states. 
        // If scaler is empty, we assume input is already prepared or scaler not needed.
        let x_norm = if self.scaler.mean.is_some() {
            self.scaler.transform(&x_arr)?
        } else {
            x_arr
        };

        // 2. Inference
        // ONNX input shape: [batch_size, input_dim]
        let input_tensor = Value::from_array(x_norm.clone())?;
        
        // SB3 ONNX export usually outputs (action, value) or just action probabilities
        // We assume the ONNX model outputs logits or probabilities for 3 actions
        let outputs = session.run(inputs![input_tensor]?)?;
        
        // Assume first output is what we want
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let output_array = output_tensor.into_owned();
        
        // If output is actions (discrete), we need to one-hot encode
        // If output is logits/probs, we use them
        // SB3 "MlpPolicy" export behavior depends on exporter.
        // Standard assumption: Output is [batch, n_actions] (logits/probs)
        
        // Check shape
        let shape = output_array.shape();
        if shape.len() == 2 && shape[1] == 3 {
            // Already probabilities/logits
            // Apply softmax if they look like logits (can check range, but usually safe to apply)
            // Or just return as is if they are probs.
            // Let's assume logits and apply softmax
            let mut probs = output_array.into_dimensionality::<2>()?;
            for mut row in probs.axis_iter_mut(Axis(0)) {
                let max = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let sum = row.mapv(|v| (v - max).exp()).sum();
                row.mapv_inplace(|v| (v - max).exp() / sum);
            }
            Ok(probs)
        } else if shape.len() == 1 || (shape.len() == 2 && shape[1] == 1) {
            // Discrete actions [0, 1, 2]
            let actions = output_array.into_dimensionality::<2>()?;
            let n = actions.len();
            let mut probs = Array2::zeros((n, 3));
            for (i, &action) in actions.iter().enumerate() {
                let a = action as usize;
                if a < 3 {
                    probs[[i, a]] = 1.0;
                }
            }
            Ok(probs)
        } else {
            anyhow::bail!("Unexpected ONNX output shape: {:?}", shape);
        }
    }

    #[cfg(not(feature = "onnx"))]
    fn predict_proba(&self, _x: &DataFrame) -> Result<Array2<f32>> {
        anyhow::bail!("ONNX feature not enabled");
    }

    fn save(&self, _path: &Path) -> Result<()> {
        warn!("RL save not implemented (inference only)");
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        // Load scaler metadata if exists
        if path.join("scaler.json").exists() {
            let data = std::fs::read_to_string(path.join("scaler.json"))?;
            self.scaler = serde_json::from_str(&data)?;
        }

        // Load ONNX model
        #[cfg(feature = "onnx")]
        {
            let model_path = path.join(format!("{}.onnx", self.name));
            if model_path.exists() {
                self.load_onnx_session(&model_path)?;
            } else {
                warn!("ONNX model not found at {:?}", model_path);
            }
        }
        
        Ok(())
    }
}

// ============================================================================ 
// RLlib AGENTS
// ============================================================================ 

// RLlib agents follow similar pattern
// We reuse RLExpert struct but could specialize if needed
pub type RLlibExpert = RLExpert;

// Transformer Models Module
// Ported from src/forex_bot/models/transformers.py
// Implements TransformerExpert using Burn

use anyhow::{Context, Result};
use burn::prelude::*;
use burn::nn::transformer::{TransformerEncoder, TransformerEncoderConfig};
use burn::tensor::backend::AutodiffBackend;
use ndarray::Array2;
use polars::prelude::*;
use std::path::Path;

use crate::base::{ExpertModel, RobustScaler, dataframe_to_float32_array};
use crate::neural_networks::{TrainingConfig, train_classification, ModelBackend};

// ============================================================================
// TRANSFORMER MODEL
// ============================================================================

#[derive(Config, Debug)]
pub struct TransformerConfig {
    pub input_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
    #[config(default = 4)]
    pub n_heads: usize,
    #[config(default = 2)]
    pub n_layers: usize,
    pub output_dim: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct TransformerModel<B: Backend> {
    embed: nn::Linear<B>,
    encoder: TransformerEncoder<B>,
    final_layer: nn::Linear<B>,
    dropout: nn::Dropout,
}

impl<B: Backend> TransformerModel<B> {
    pub fn new(config: &TransformerConfig, device: &B::Device) -> Self {
        let embed = nn::LinearConfig::new(config.input_dim, config.hidden_dim).init(device);
        
        let encoder_config = TransformerEncoderConfig::new(
            config.hidden_dim, 
            config.n_heads, 
            config.hidden_dim * 4, // feed forward dim usually 4x
            config.n_layers
        )
        .with_dropout(config.dropout);
        
        let encoder = encoder_config.init(device);
        let final_layer = nn::LinearConfig::new(config.hidden_dim, config.output_dim).init(device);
        let dropout = nn::DropoutConfig::new(config.dropout).init();

        Self {
            embed,
            encoder,
            final_layer,
            dropout,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        // Input: [batch, seq_len, features]
        let x = self.embed.forward(input.clone());
        let x = self.dropout.forward(x);
        
        // Transformer Encoder
        // Create mask (optional, here we assume full context)
        // Burn 0.19 TransformerEncoder takes (input, mask_pad)
        // Mask pad: [batch, seq_len] bool tensor. True = pad.
        // We assume no padding for fixed window sliding.
        let x = self.encoder.forward(x, None);
        
        // Take last time step
        // x: [batch, seq_len, hidden]
        let [batch, seq, hidden] = x.dims();
        let last_x = x.slice([0..batch, (seq-1)..seq, 0..hidden]).reshape([batch, hidden]);
        
        self.final_layer.forward(last_x)
    }

    pub fn predict_proba(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(input);
        burn::tensor::activation::softmax(logits, 1)
    }
}

// ============================================================================
// EXPERT WRAPPER
// ============================================================================

pub struct TransformerExpert<B: Backend> {
    pub model: Option<TransformerModel<B>>,
    pub config: TransformerConfig,
    pub training_config: TrainingConfig,
    pub scaler: RobustScaler,
    pub device: B::Device,
}

impl<B: Backend> TransformerExpert<B> {
    pub fn new(config: TransformerConfig, training_config: TrainingConfig, device: B::Device) -> Self {
        Self {
            model: None,
            config,
            training_config,
            scaler: RobustScaler::new(),
            device,
        }
    }
}

// Helper to implement Forward trait for generic training loop compatibility
use crate::neural_networks::Forward;
impl<B: Backend> Forward<B, 2> for TransformerModel<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Reshape 2D [batch, features] to 3D [batch, 1, features]
        let [batch, features] = input.dims();
        self.forward(input.reshape([batch, 1, features]))
    }
}

impl<B: AutodiffBackend> ExpertModel for TransformerExpert<B> {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let x_arr = dataframe_to_float32_array(x)?;
        self.scaler.fit(&x_arr)?;
        let x_norm = self.scaler.transform(&x_arr)?;
        
        let y_i32: Vec<i32> = y.cast(&DataType::Int32)?
            .i32()?
            .into_iter()
            .map(|val: Option<i32>| match val.unwrap_or(0) {
                -1 => 2,
                1 => 1,
                _ => 0,
            })
            .collect();

        let model = TransformerModel::new(&self.config, &self.device);
        
        // Use generic classification training
        let train_data = vec![(x_norm, y_i32)];
        let trained_model = train_classification(model, train_data, self.training_config.clone(), &self.device)?;
        
        self.model = Some(trained_model);
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        let model = self.model.as_ref().context("Model not trained")?;
        
        let x_arr = dataframe_to_float32_array(x)?;
        let x_norm = self.scaler.transform(&x_arr)?;
        
        let (n, f) = x_norm.dim();
        // Reshape for 3D input: [batch, 1, features]
        let input = Tensor::<B, 3>::from_floats(x_norm.as_slice().unwrap(), &self.device).reshape([n, 1, f]);
        let probs = model.predict_proba(input);
        
        let data = probs.into_data();
        let flat_probs: Vec<f32> = data.as_slice::<f32>().unwrap().to_vec();
        
        Array2::from_shape_vec((x.height(), 3), flat_probs)
            .context("Failed to create prediction array")
    }

    fn save(&self, path: &Path) -> Result<()> {
        use crate::neural_networks::save_model;
        let model = self.model.as_ref().context("Model not trained")?;
        save_model(model, &path.join("model.mpk"))?;
        // Save config
        std::fs::write(path.join("meta.json"), serde_json::to_string(&self.config)?)?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        use crate::neural_networks::load_model;
        let model = TransformerModel::new(&self.config, &self.device);
        let loaded_model = load_model(model, &path.join("model.mpk"), &self.device)?;
        self.model = Some(loaded_model);
        Ok(())
    }
}

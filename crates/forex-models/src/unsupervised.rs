// Unsupervised Learning Module - Market Regime Classification
// Ported from src/forex_bot/models/unsupervised.py
// Uses linfa for GMM clustering

use anyhow::{Context, Result};
use linfa::prelude::*;
use linfa_clustering::GaussianMixtureModel;
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use std::path::Path;
use tracing::{info, warn};

use crate::base::{ExpertModel, RobustScaler, dataframe_to_float32_array};

pub struct MarketRegimeClassifier {
    pub n_regimes: usize,
    pub model: Option<GaussianMixtureModel<f64>>,
    pub scaler: RobustScaler,
    pub is_fitted: bool,
}

impl MarketRegimeClassifier {
    pub fn new(n_regimes: usize) -> Self {
        Self {
            n_regimes,
            model: None,
            scaler: RobustScaler::new(),
            is_fitted: false,
        }
    }

    fn extract_features(&self, df: &DataFrame) -> Result<Array2<f32>> {
        // HPC Optimized: Feature extraction
        // Log Returns (Vectorized)
        let close = df.column("close")?.cast(&DataType::Float64)?.f64()?.clone();
        let close_vals: Vec<f64> = close.into_iter().map(|v| v.unwrap_or(0.0)).collect();
        
        if close_vals.len() < 2 {
            return Ok(Array2::zeros((0, 0)));
        }

        let mut returns = Vec::with_capacity(close_vals.len());
        returns.push(0.0);
        for i in 0..close_vals.len() - 1 {
            let r = (close_vals[i+1] / close_vals[i]).ln();
            returns.push(r as f32);
        }

        // Rolling Volatility (20 periods)
        let mut volatility = vec![0.0f32; returns.len()];
        let window = 20;
        for i in window..returns.len() {
            let slice = &returns[i-window+1..=i];
            let mean = slice.iter().sum::<f32>() / window as f32;
            let variance = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / window as f32;
            volatility[i] = variance.sqrt();
        }

        // ADX approximation (vol * 100) if not present
        let adx: Vec<f32> = if let Ok(col) = df.column("adx") {
            col.cast(&DataType::Float32)?.f32()?.into_iter().map(|v| v.unwrap_or(0.0)).collect()
        } else {
            volatility.iter().map(|&v| v * 100.0).collect()
        };

        // Combine into Array2 [n, 3]
        // Shift by 1 bar to avoid look-ahead leakage (done in Python by shift(1).dropna())
        // We simulate shift by taking [0..n-1] as features for [1..n]
        // But for training we align. 
        // Python code: data = data.shift(1).dropna()
        // So row `i` contains features from `i-1`.
        
        let n = returns.len();
        if n < 2 {
             return Ok(Array2::zeros((0, 3)));
        }

        let mut features = Vec::with_capacity((n - 1) * 3);
        for i in 1..n {
            features.push(returns[i-1]);
            features.push(volatility[i-1]);
            features.push(adx[i-1]);
        }

        Array2::from_shape_vec((n - 1, 3), features).context("Failed to create feature array")
    }
}

impl ExpertModel for MarketRegimeClassifier {
    fn fit(&mut self, x: &DataFrame, _y: &Series) -> Result<()> {
        let features_f32 = self.extract_features(x)?;
        if features_f32.is_empty() {
            warn!("Empty features for GMM fit");
            return Ok(());
        }

        // Robust Scaler expects Array2<f32>, linfa GMM expects f64
        self.scaler.fit(&features_f32)?;
        let x_norm = self.scaler.transform(&features_f32)?;
        
        let x_f64 = x_norm.mapv(|v| v as f64);
        let dataset = DatasetBase::from(x_f64);

        let gmm = GaussianMixtureModel::params(self.n_regimes)
            .n_runs(10)
            .tolerance(1e-4)
            .fit(&dataset)
            .map_err(|e| anyhow::anyhow!("GMM fit failed: {}", e))?;

        self.model = Some(gmm);
        self.is_fitted = true;
        info!("Unsupervised GMM fitted: {} latent regimes discovered.", self.n_regimes);
        Ok(())
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        // Mock implementation for ExpertModel compatibility
        // Returns neutral probabilities
        let n = x.height();
        let mut probs = Array2::zeros((n, 3));
        for i in 0..n {
            probs[[i, 0]] = 1.0; // Neutral
        }
        Ok(probs)
    }

    fn save(&self, _path: &Path) -> Result<()> {
        // Saving linfa models is complex (serde support might be partial)
        // For now, skip or implement manual serialization if needed
        warn!("GMM save not fully implemented");
        Ok(())
    }

    fn load(&mut self, _path: &Path) -> Result<()> {
        warn!("GMM load not fully implemented");
        Ok(())
    }
}

/// Helper for predicting regime distribution
impl MarketRegimeClassifier {
    pub fn predict_regime_distribution(&self, df: &DataFrame) -> Result<Vec<f64>> {
        if !self.is_fitted || self.model.is_none() {
            return Ok(vec![0.0; self.n_regimes]);
        }

        // Process last 50 rows for stability
        let tail_len = 50.min(df.height());
        let tail_df = df.tail(Some(tail_len));
        
        let features_f32 = self.extract_features(&tail_df)?;
        if features_f32.is_empty() {
            return Ok(vec![0.0; self.n_regimes]);
        }

        let x_norm = self.scaler.transform(&features_f32)?;
        let x_f64 = x_norm.mapv(|v| v as f64);
        let dataset = DatasetBase::from(x_f64);

        let model = self.model.as_ref().unwrap();
        let probs = model.predict_probabilities(&dataset); // Returns Array2
        
        // Return the latest posterior (last row)
        let last_row = probs.row(probs.nrows() - 1);
        Ok(last_row.to_vec())
    }
}

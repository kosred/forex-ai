use anyhow::Result;
use crate::base::ExpertModel;
use crate::tree_models::{LightGBMExpert, XGBoostExpert, XGBoostRFExpert, XGBoostDARTExpert, CatBoostExpert, CatBoostAltExpert};

#[cfg(feature = "burn")]
use crate::neural_networks::{MLPExpert, LSTMExpert, NBeatsExpert, TiDEExpert, TabNetExpert, KANExpert, MLPConfig, LSTMConfig, NBeatsConfig, TiDEConfig, TabNetConfig, KANConfig, TrainingConfig};

#[cfg(feature = "burn")]
use crate::transformers::{TransformerExpert, TransformerConfig};

/// Create a new expert model by name.
pub fn create_model(name: &str) -> Result<Box<dyn ExpertModel>> {
    match name {
        "lightgbm" => Ok(Box::new(LightGBMExpert::new(0, None))),
        "xgboost" => Ok(Box::new(XGBoostExpert::new(0, None))),
        "xgboost_rf" => Ok(Box::new(XGBoostRFExpert::new(0, None))),
        "xgboost_dart" => Ok(Box::new(XGBoostDARTExpert::new(0, None))),
        "catboost" => Ok(Box::new(CatBoostExpert::new(0, None))),
        "catboost_alt" => Ok(Box::new(CatBoostAltExpert::new(0, None))),
        
        #[cfg(feature = "burn")]
        "mlp" => {
            let config = MLPConfig::new(10, vec![64, 32], 3);
            let train_config = TrainingConfig::new();
            let device = Default::default(); 
            Ok(Box::new(MLPExpert::new(config, train_config, device)))
        },
        
        #[cfg(feature = "burn")]
        "lstm" => {
            let config = LSTMConfig::new(10, 64, 3);
            let train_config = TrainingConfig::new();
            let device = Default::default();
            Ok(Box::new(LSTMExpert::new(config, train_config, device)))
        },

        #[cfg(feature = "burn")]
        "transformer" => {
            let config = TransformerConfig::new(10, 3);
            let train_config = TrainingConfig::new();
            let device = Default::default();
            Ok(Box::new(TransformerExpert::new(config, train_config, device)))
        },

        #[cfg(feature = "burn")]
        "nbeats" => {
            let config = NBeatsConfig::new(10, 3);
            let train_config = TrainingConfig::new();
            let device = Default::default();
            Ok(Box::new(NBeatsExpert::new(config, train_config, device)))
        },

        #[cfg(feature = "burn")]
        "tide" => {
            let config = TiDEConfig::new(10, 3);
            let train_config = TrainingConfig::new();
            let device = Default::default();
            Ok(Box::new(TiDEExpert::new(config, train_config, device)))
        },

        #[cfg(feature = "burn")]
        "tabnet" => {
            let config = TabNetConfig::new(10, 3);
            let train_config = TrainingConfig::new();
            let device = Default::default();
            Ok(Box::new(TabNetExpert::new(config, train_config, device)))
        },

        #[cfg(feature = "burn")]
        "kan" => {
            let config = KANConfig::new(10, 3);
            let train_config = TrainingConfig::new();
            let device = Default::default();
            Ok(Box::new(KANExpert::new(config, train_config, device)))
        },

        _ => Err(anyhow::anyhow!("Model '{}' not found in registry", name)),
    }
}
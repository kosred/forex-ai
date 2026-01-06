// Core configuration structures for Forex trading system
// Port of src/forex_bot/core/config.py

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// System-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemConfig {
    pub symbol: String,
    pub symbols: Vec<String>,
    pub data_dir: PathBuf,
    pub indices_path: String,
    pub use_online_indices: bool,
    pub mt5_dxy_symbol: String,
    pub mt5_eur_symbol: String,
    pub base_timeframe: String,
    pub use_volume_features: bool,
    pub higher_timeframes: Vec<String>,
    pub required_timeframes: Vec<String>,
    pub enable_level2: bool,
    pub level2_depth_levels: usize,
    pub history_years: usize,
    pub trading_session_start: String,
    pub trading_session_end: String,
    pub session_timezone: String,
    pub broker_backend: String,
    pub mt5_required: bool,
    pub mt5_terminal_path: PathBuf,
    pub mt5_login: i64,
    pub mt5_password: String,
    pub mt5_server: String,
    pub mt5_timeout_seconds: u64,
    pub mt5_timezone_offset_hours: i32,
    pub poll_interval_seconds: u64,
    pub metrics_logging_enabled: bool,
    pub metrics_db_path: PathBuf,
    pub risk_ledger_enabled: bool,
    pub risk_ledger_max_events: usize,
    pub strategy_ledger_path: PathBuf,
    pub enable_dashboard: bool,
    pub cache_enabled: bool,
    pub cache_dir: PathBuf,
    pub cache_max_age_minutes: u64,
    pub deep_purge_mode: String,
    pub deep_purge_on_train: bool,
    pub n_jobs: usize,
    pub enable_gpu_preference: String,
    pub discovery_auto_cap: bool,
    pub discovery_max_rows: usize,
    pub discovery_stream: bool,
    pub enable_gpu: bool,
    pub num_gpus: usize,
    pub device: String,
    pub evo_multiproc_per_gpu: bool,
    pub cache_training_frames: bool,
    pub training_cache_max_bytes: usize,
    pub max_training_rows_per_tf: usize,
    pub downcast_training_float32: bool,
    pub parquet_memory_map: bool,
    pub smc_freshness_limit: usize,
    pub smc_atr_displacement: f64,
    pub smc_max_levels: usize,
    pub smc_use_cuda: bool,
}

impl Default for SystemConfig {
    fn default() -> Self {
        let n_jobs = std::thread::available_parallelism()
            .map(|n| (n.get() - 1).max(1))
            .unwrap_or(1);

        Self {
            symbol: "EURUSD".to_string(),
            symbols: vec!["EURUSD".to_string()],
            data_dir: PathBuf::from("data"),
            indices_path: String::new(),
            use_online_indices: false,
            mt5_dxy_symbol: "USDX".to_string(),
            mt5_eur_symbol: "EXY".to_string(),
            base_timeframe: "M1".to_string(),
            use_volume_features: true,
            higher_timeframes: vec![
                "M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1", "MN1",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            required_timeframes: vec![
                "M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1", "MN1",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            enable_level2: false,
            level2_depth_levels: 10,
            history_years: 10,
            trading_session_start: "00:05".to_string(),
            trading_session_end: "23:55".to_string(),
            session_timezone: "UTC".to_string(),
            broker_backend: "mt5_local".to_string(),
            mt5_required: cfg!(windows),
            mt5_terminal_path: PathBuf::new(),
            mt5_login: 0,
            mt5_password: String::new(),
            mt5_server: String::new(),
            mt5_timeout_seconds: 30,
            mt5_timezone_offset_hours: 0,
            poll_interval_seconds: 60,
            metrics_logging_enabled: true,
            metrics_db_path: PathBuf::from("metrics.sqlite"),
            risk_ledger_enabled: true,
            risk_ledger_max_events: 1000,
            strategy_ledger_path: PathBuf::from("strategy_ledger.sqlite"),
            enable_dashboard: true,
            cache_enabled: true,
            cache_dir: PathBuf::from("cache"),
            cache_max_age_minutes: 60,
            deep_purge_mode: "off".to_string(),
            deep_purge_on_train: true,
            n_jobs,
            enable_gpu_preference: "auto".to_string(),
            discovery_auto_cap: true,
            discovery_max_rows: 0,
            discovery_stream: false,
            enable_gpu: false,
            num_gpus: 0,
            device: "cpu".to_string(),
            evo_multiproc_per_gpu: true,
            cache_training_frames: false,
            training_cache_max_bytes: 2_000_000_000,
            max_training_rows_per_tf: 0,
            downcast_training_float32: true,
            parquet_memory_map: true,
            smc_freshness_limit: 0,
            smc_atr_displacement: 0.0,
            smc_max_levels: 0,
            smc_use_cuda: false,
        }
    }
}

/// Risk management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RiskConfig {
    pub initial_balance: f64,
    pub monthly_profit_target_pct: f64,
    pub min_risk_per_trade: f64,
    pub max_risk_per_trade: f64,
    pub risk_per_trade: f64,
    pub daily_drawdown_limit: f64,
    pub total_drawdown_limit: f64,
    pub min_risk_reward: f64,
    pub spread_guard_multiplier: f64,
    pub slippage_guard_multiplier: f64,
    pub challenge_mode: bool,
    pub challenge_phase: String,
    pub prop_firm_rules: bool,
    pub max_daily_risk_pct: f64,
    pub base_risk_per_trade: f64,
    pub daily_risk_budget: f64,
    pub consistency_tracking: bool,
    pub min_confidence_threshold: f64,
    pub kill_zones_enabled: bool,
    pub enhanced_features: bool,
    pub uncertainty_quantification: bool,
    pub max_trades_per_day: usize,
    pub daily_profit_stop_pct: f64,
    pub recovery_mode_enabled: bool,
    pub feature_drift_threshold: f64,
    pub high_quality_confidence: f64,
    pub high_quality_risk_pct: f64,
    pub high_quality_rr: f64,
    pub atr_period: usize,
    pub atr_stop_multiplier: f64,
    pub triple_barrier_max_bars: usize,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub kelly_lambda: f64,
    pub slippage_pips: f64,
    pub commission_per_lot: f64,
    pub backtest_spread_pips: f64,
    pub cost_penalty_r: f64,
    pub gate_trade_prob: f64,
    pub daily_hard_stop_pct: f64,
    pub conformal_alpha: f64,
    pub volatility_stop_sigma: f64,
    pub volatility_lookback: usize,
    pub meta_label_tp_pips: Option<f64>,
    pub meta_label_sl_pips: Option<f64>,
    pub meta_label_max_hold_bars: usize,
    pub meta_label_min_prob_threshold: f64,
    pub meta_label_min_dist: f64,
    pub meta_label_fixed_sl: f64,
    pub meta_label_fixed_tp: f64,
    pub stop_target_mode: String,
    pub vol_estimator: String,
    pub vol_ensemble_weights: HashMap<String, f64>,
    pub vol_ensemble_weights_trend: Option<HashMap<String, f64>>,
    pub vol_ensemble_weights_range: Option<HashMap<String, f64>>,
    pub vol_ensemble_weights_neutral: Option<HashMap<String, f64>>,
    pub vol_window: usize,
    pub ewma_lambda: f64,
    pub ewma_lambda_by_timeframe: HashMap<String, f64>,
    pub vol_horizon_bars: usize,
    pub tail_window: usize,
    pub tail_alpha: f64,
    pub tail_step: usize,
    pub tail_max_bars: usize,
    pub stop_k_vol: f64,
    pub stop_k_tail: f64,
    pub rr_trend: f64,
    pub rr_range: f64,
    pub rr_neutral: f64,
    pub regime_adx_trend: f64,
    pub regime_adx_range: f64,
    pub hurst_window: usize,
    pub hurst_trend: f64,
    pub hurst_range: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        let mut vol_ensemble_weights = HashMap::new();
        vol_ensemble_weights.insert("yang_zhang".to_string(), 1.0);
        vol_ensemble_weights.insert("garman_klass".to_string(), 1.0);
        vol_ensemble_weights.insert("rogers_satchell".to_string(), 1.0);
        vol_ensemble_weights.insert("parkinson".to_string(), 1.0);

        let mut ewma_lambda_by_timeframe = HashMap::new();
        for (tf, lambda) in [
            ("M1", 0.90),
            ("M5", 0.92),
            ("M15", 0.94),
            ("M30", 0.95),
            ("H1", 0.96),
            ("H4", 0.97),
            ("D1", 0.98),
            ("W1", 0.985),
            ("MN1", 0.99),
        ] {
            ewma_lambda_by_timeframe.insert(tf.to_string(), lambda);
        }

        Self {
            initial_balance: 10_000.0,
            monthly_profit_target_pct: 0.04,
            min_risk_per_trade: 0.0,
            max_risk_per_trade: 0.030,
            risk_per_trade: 0.030,
            daily_drawdown_limit: 0.04,
            total_drawdown_limit: 0.07,
            min_risk_reward: 2.0,
            spread_guard_multiplier: 2.5,
            slippage_guard_multiplier: 2.0,
            challenge_mode: false,
            challenge_phase: "phase_1".to_string(),
            prop_firm_rules: true,
            max_daily_risk_pct: 0.04,
            base_risk_per_trade: 0.03,
            daily_risk_budget: 0.040,
            consistency_tracking: true,
            min_confidence_threshold: 0.55,
            kill_zones_enabled: true,
            enhanced_features: true,
            uncertainty_quantification: true,
            max_trades_per_day: 8,
            daily_profit_stop_pct: 0.0,
            recovery_mode_enabled: true,
            feature_drift_threshold: 0.30,
            high_quality_confidence: 0.65,
            high_quality_risk_pct: 0.030,
            high_quality_rr: 2.0,
            atr_period: 14,
            atr_stop_multiplier: 1.5,
            triple_barrier_max_bars: 35,
            trailing_enabled: true,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            kelly_lambda: 1.0,
            slippage_pips: 0.5,
            commission_per_lot: 7.0,
            backtest_spread_pips: 1.5,
            cost_penalty_r: 0.0,
            gate_trade_prob: 0.55,
            daily_hard_stop_pct: 0.04,
            conformal_alpha: 0.10,
            volatility_stop_sigma: 0.02,
            volatility_lookback: 50,
            meta_label_tp_pips: None,
            meta_label_sl_pips: None,
            meta_label_max_hold_bars: 100,
            meta_label_min_prob_threshold: 0.55,
            meta_label_min_dist: 0.0005,
            meta_label_fixed_sl: 0.0020,
            meta_label_fixed_tp: 0.0040,
            stop_target_mode: "blend".to_string(),
            vol_estimator: "ensemble".to_string(),
            vol_ensemble_weights,
            vol_ensemble_weights_trend: None,
            vol_ensemble_weights_range: None,
            vol_ensemble_weights_neutral: None,
            vol_window: 50,
            ewma_lambda: 0.94,
            ewma_lambda_by_timeframe,
            vol_horizon_bars: 5,
            tail_window: 100,
            tail_alpha: 0.975,
            tail_step: 5,
            tail_max_bars: 300_000,
            stop_k_vol: 1.0,
            stop_k_tail: 1.25,
            rr_trend: 2.5,
            rr_range: 1.5,
            rr_neutral: 2.0,
            regime_adx_trend: 25.0,
            regime_adx_range: 20.0,
            hurst_window: 100,
            hurst_trend: 0.55,
            hurst_range: 0.45,
        }
    }
}

/// Models and training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    pub ml_models: Vec<String>,
    pub use_rl_agent: bool,
    pub use_sac_agent: bool,
    pub use_rllib_agent: bool,
    pub rllib_num_workers: usize,
    pub auto_enable_rllib: bool,
    pub use_neuroevolution: bool,
    pub rl_population_size: usize,
    pub rl_timesteps: usize,
    pub rl_eval_episodes: usize,
    pub rl_network_arch: Vec<usize>,
    pub rl_parallel_envs: usize,
    pub rl_train_seconds: u64,
    pub evo_train_seconds: u64,
    pub evo_hidden_size: usize,
    pub evo_population: usize,
    pub evo_islands: usize,
    pub prop_search_enabled: bool,
    pub prop_search_population: usize,
    pub prop_search_generations: usize,
    pub prop_search_max_hours: f64,
    pub prop_search_max_rows: usize,
    pub prop_search_max_rows_by_tf: HashMap<String, usize>,
    pub prop_search_portfolio_size: usize,
    pub prop_search_checkpoint: PathBuf,
    pub prop_search_device: String,
    pub prop_search_train_years: usize,
    pub prop_search_val_years: usize,
    pub prop_search_val_candidates: usize,
    pub prop_search_val_min_positive_months: usize,
    pub prop_search_val_min_trades_per_month: usize,
    pub prop_search_val_min_trades_per_day: f64,
    pub prop_search_val_min_monthly_profit_pct: f64,
    pub prop_search_val_log_trades: bool,
    pub prop_search_val_trade_log_max: usize,
    pub prop_search_async: bool,
    pub prop_search_async_wait: bool,
    pub tree_device_preference: String,
    pub prop_search_opportunistic_enabled: bool,
    pub prop_search_opportunistic_min_positive_months: usize,
    pub prop_search_opportunistic_min_trades_per_month: usize,
    pub prop_search_opportunistic_min_trade_return_pct: f64,
    pub prop_search_opportunistic_max_dd: f64,
    pub prop_search_use_opportunistic: bool,
    pub train_batch_size: usize,
    pub inference_batch_size: usize,
    pub enable_transformer_expert: bool,
    pub transformer_heads: usize,
    pub transformer_layers: usize,
    pub transformer_hidden_dim: usize,
    pub transformer_dropout: f64,
    pub transformer_seq_len: usize,
    pub transformer_train_seconds: u64,
    pub nbeats_train_seconds: u64,
    pub tide_train_seconds: u64,
    pub tabnet_train_seconds: u64,
    pub kan_train_seconds: u64,
    pub mlp_train_seconds: u64,
    pub num_transformers: usize,
    pub hpo_backend: String,
    pub hpo_trials: usize,
    pub hpo_trials_by_model: HashMap<String, usize>,
    pub hpo_max_rows: usize,
    pub max_epochs_by_model: HashMap<String, usize>,
    pub ray_tune_max_concurrency: usize,
    pub export_onnx: bool,
    pub filter_to_base_signal: bool,
    pub global_max_rows: usize,
    pub global_max_rows_per_symbol: usize,
    pub symbol_hash_buckets: usize,
    pub global_train_ratio: f64,
    pub train_holdout_pct: f64,
    pub walkforward_splits: usize,
    pub embargo_minutes: usize,
    pub prop_metric_weight: f64,
    pub prop_accuracy_weight: f64,
    pub prop_min_trades: usize,
    pub prop_conf_threshold: f64,
    pub enable_cpcv: bool,
    pub cpcv_n_splits: usize,
    pub cpcv_n_test_groups: usize,
    pub cpcv_embargo_pct: f64,
    pub cpcv_purge_pct: f64,
    pub cpcv_min_phi: f64,
    pub cpcv_max_rows: usize,
    pub enable_ddp: bool,
    pub enable_fsdp: bool,
    pub ddp_world_size: usize,
    pub transformer_d_model: usize,
    pub transformer_n_heads: usize,
    pub transformer_n_layers: usize,
    pub nf_hidden_dim: usize,
    pub tide_hidden_dim: usize,
    pub nbeats_hidden_dim: usize,
    pub kan_hidden_dim: usize,
    pub tabnet_hidden_dim: usize,
    pub phase5_filter_meta_blender: bool,
    pub phase5_core_models: Vec<String>,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        let mut hpo_trials_by_model = HashMap::new();
        for (model, trials) in [
            ("lightgbm", 8),
            ("xgboost", 8),
            ("xgboost_rf", 6),
            ("xgboost_dart", 6),
            ("catboost", 8),
            ("catboost_alt", 6),
            ("mlp", 6),
            ("tabnet", 6),
            ("nbeats", 6),
            ("tide", 6),
            ("kan", 6),
            ("transformer", 6),
        ] {
            hpo_trials_by_model.insert(model.to_string(), trials);
        }

        Self {
            ml_models: vec![
                "lightgbm",
                "xgboost",
                "xgboost_rf",
                "xgboost_dart",
                "catboost",
                "catboost_alt",
                "mlp",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
            use_rl_agent: true,
            use_sac_agent: true,
            use_rllib_agent: false,
            rllib_num_workers: 0,
            auto_enable_rllib: true,
            use_neuroevolution: true,
            rl_population_size: 5,
            rl_timesteps: 10_000_000,
            rl_eval_episodes: 15,
            rl_network_arch: vec![4096, 4096, 4096, 2048, 1024],
            rl_parallel_envs: 1,
            rl_train_seconds: 3600,
            evo_train_seconds: 3600,
            evo_hidden_size: 64,
            evo_population: 32,
            evo_islands: 4,
            prop_search_enabled: false,
            prop_search_population: 100,
            prop_search_generations: 20,
            prop_search_max_hours: 2.0,
            prop_search_max_rows: 0,
            prop_search_max_rows_by_tf: HashMap::new(),
            prop_search_portfolio_size: 3000,
            prop_search_checkpoint: PathBuf::from("models/strategy_evo_checkpoint.json"),
            prop_search_device: "cpu".to_string(),
            prop_search_train_years: 0,
            prop_search_val_years: 0,
            prop_search_val_candidates: 0,
            prop_search_val_min_positive_months: 0,
            prop_search_val_min_trades_per_month: 0,
            prop_search_val_min_trades_per_day: 0.0,
            prop_search_val_min_monthly_profit_pct: 0.0,
            prop_search_val_log_trades: false,
            prop_search_val_trade_log_max: 20,
            prop_search_async: false,
            prop_search_async_wait: false,
            tree_device_preference: "auto".to_string(),
            prop_search_opportunistic_enabled: true,
            prop_search_opportunistic_min_positive_months: 3,
            prop_search_opportunistic_min_trades_per_month: 10,
            prop_search_opportunistic_min_trade_return_pct: 4.0,
            prop_search_opportunistic_max_dd: 0.025,
            prop_search_use_opportunistic: true,
            train_batch_size: 32,
            inference_batch_size: 32,
            enable_transformer_expert: true,
            transformer_heads: 16,
            transformer_layers: 12,
            transformer_hidden_dim: 1200,
            transformer_dropout: 0.20,
            transformer_seq_len: 64,
            transformer_train_seconds: 3600,
            nbeats_train_seconds: 3600,
            tide_train_seconds: 3600,
            tabnet_train_seconds: 3600,
            kan_train_seconds: 3600,
            mlp_train_seconds: 3600,
            num_transformers: 2,
            hpo_backend: "ax".to_string(),
            hpo_trials: 8,
            hpo_trials_by_model,
            hpo_max_rows: 1_000_000,
            max_epochs_by_model: HashMap::new(),
            ray_tune_max_concurrency: 1,
            export_onnx: false,
            filter_to_base_signal: true,
            global_max_rows: 0,
            global_max_rows_per_symbol: 0,
            symbol_hash_buckets: 32,
            global_train_ratio: 0.8,
            train_holdout_pct: 0.2,
            walkforward_splits: 20,
            embargo_minutes: 120,
            prop_metric_weight: 1.0,
            prop_accuracy_weight: 0.1,
            prop_min_trades: 0,
            prop_conf_threshold: 0.55,
            enable_cpcv: true,
            cpcv_n_splits: 5,
            cpcv_n_test_groups: 2,
            cpcv_embargo_pct: 0.01,
            cpcv_purge_pct: 0.02,
            cpcv_min_phi: 0.80,
            cpcv_max_rows: 0,
            enable_ddp: false,
            enable_fsdp: false,
            ddp_world_size: 1,
            transformer_d_model: 256,
            transformer_n_heads: 8,
            transformer_n_layers: 4,
            nf_hidden_dim: 256,
            tide_hidden_dim: 256,
            nbeats_hidden_dim: 256,
            kan_hidden_dim: 256,
            tabnet_hidden_dim: 64,
            phase5_filter_meta_blender: true,
            phase5_core_models: vec!["transformer", "nbeats", "tide", "tabnet", "kan"]
                .into_iter()
                .map(String::from)
                .collect(),
        }
    }
}

/// News and LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct NewsConfig {
    pub news_decay_minutes: usize,
    pub news_kill_window_min: usize,
    pub news_confidence_threshold: f64,
    pub news_lookahead_minutes: usize,
    pub news_trade_on_event: bool,
    pub news_trade_confidence_threshold: f64,
    pub enable_news: bool,
    pub enable_llm_helper: bool,
    pub llm_helper_enabled: bool,
    pub llm_sentiment_positive_threshold: f64,
    pub llm_sentiment_negative_threshold: f64,
    pub news_backfill_enabled: bool,
    pub news_backfill_days: usize,
    pub news_local_glob: String,
    pub openai_model: String,
    pub openai_api_key_env: String,
    pub openai_max_tokens: usize,
    pub openai_max_events_per_fetch: usize,
    pub openai_news_enabled: bool,
    pub perplexity_enabled: bool,
    pub perplexity_api_key_env: String,
    pub perplexity_model: String,
    pub perplexity_num_results: usize,
    pub perplexity_timeframe_hours: usize,
    pub strategist_enabled: bool,
    pub strategist_interval_minutes: usize,
    pub auto_rescore_enabled: bool,
    pub auto_rescore_days: usize,
    pub auto_rescore_max_events: usize,
    pub auto_rescore_only_missing: bool,
}

impl Default for NewsConfig {
    fn default() -> Self {
        Self {
            news_decay_minutes: 120,
            news_kill_window_min: 30,
            news_confidence_threshold: 0.65,
            news_lookahead_minutes: 60,
            news_trade_on_event: false,
            news_trade_confidence_threshold: 0.90,
            enable_news: true,
            enable_llm_helper: true,
            llm_helper_enabled: true,
            llm_sentiment_positive_threshold: 0.2,
            llm_sentiment_negative_threshold: -0.2,
            news_backfill_enabled: true,
            news_backfill_days: 30,
            news_local_glob: String::new(),
            openai_model: "gpt-5-nano-2025-08-07".to_string(),
            openai_api_key_env: "OPENAI_API_KEY".to_string(),
            openai_max_tokens: 256,
            openai_max_events_per_fetch: 50,
            openai_news_enabled: true,
            perplexity_enabled: true,
            perplexity_api_key_env: "PPLX_API_KEY".to_string(),
            perplexity_model: "sonar".to_string(),
            perplexity_num_results: 10,
            perplexity_timeframe_hours: 24,
            strategist_enabled: false,
            strategist_interval_minutes: 30,
            auto_rescore_enabled: false,
            auto_rescore_days: 30,
            auto_rescore_max_events: 200,
            auto_rescore_only_missing: true,
        }
    }
}

/// Main settings structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub system: SystemConfig,
    pub risk: RiskConfig,
    pub models: ModelsConfig,
    pub news: NewsConfig,
    pub secrets_file: PathBuf,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            risk: RiskConfig::default(),
            models: ModelsConfig::default(),
            news: NewsConfig::default(),
            secrets_file: PathBuf::from("keys.txt"),
        }
    }
}

impl Settings {
    /// Load settings from YAML config file
    pub fn from_yaml(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let settings: Settings = serde_yaml_ng::from_str(&content)?;
        Ok(settings)
    }

    /// Load settings from environment variable CONFIG_FILE or default config.yaml
    pub fn load() -> anyhow::Result<Self> {
        let config_file =
            std::env::var("CONFIG_FILE").unwrap_or_else(|_| "config.yaml".to_string());
        Self::from_yaml(&config_file)
    }

    /// Load settings with environment variable overrides
    pub fn load_with_env() -> anyhow::Result<Self> {
        let mut settings = Self::load()?;

        // Apply environment variable overrides
        if let Ok(symbol) = std::env::var("FOREX_BOT_SYMBOL") {
            settings.system.symbol = symbol;
        }

        if let Ok(device) = std::env::var("FOREX_BOT_DEVICE") {
            settings.system.device = device;
        }

        if let Ok(tree_device) = std::env::var("FOREX_BOT_TREE_DEVICE") {
            settings.models.tree_device_preference = tree_device;
        }

        Ok(settings)
    }

    /// Save settings to YAML file
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let yaml = serde_yaml_ng::to_string(self)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert_eq!(settings.system.symbol, "EURUSD");
        assert_eq!(settings.risk.initial_balance, 10_000.0);
        assert!(!settings.models.ml_models.is_empty());
    }

    #[test]
    fn test_serialize_deserialize() {
        let settings = Settings::default();
        let yaml = serde_yaml_ng::to_string(&settings).unwrap();
        let deserialized: Settings = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(deserialized.system.symbol, settings.system.symbol);
    }
}

import os

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    def get_field_value(self, field: object, field_name: str) -> tuple[object, str, bool]:
        return super().get_field_value(field, field_name)

    def __call__(self) -> dict[str, object]:
        config_file = os.environ.get("CONFIG_FILE", "config.yaml")
        if not os.path.exists(config_file):
            return {}
        with open(config_file) as f:
            return yaml.safe_load(f) or {}


class SystemConfig(BaseModel):
    symbol: str = "EURUSD"
    symbols: list[str] = Field(default_factory=lambda: ["EURUSD"])
    data_dir: str = "data"
    indices_path: str = ""  # Optional path to cross-market indices (e.g., DXY/EUR index parquet/csv)
    use_online_indices: bool = False  # If true, best-effort API fetch for DXY/EUR; otherwise skip
    mt5_dxy_symbol: str = "USDX"  # MT5 symbol for DXY (e.g., USDX, DXY, or broker-specific)
    mt5_eur_symbol: str = "EXY"  # MT5 symbol for EUR index if available
    base_timeframe: str = "M1"  # 1-minute base for multi-timeframe analysis
    higher_timeframes: list[str] = Field(
        default_factory=lambda: [
            "M1",
            "M3",
            "M5",
            "M15",
            "M30",
            "H1",
            "H2",
            "H4",
            "D1",
            "W1",
            "MN1",
        ]
    )
    required_timeframes: list[str] = Field(
        default_factory=lambda: [
            "M1",
            "M3",
            "M5",
            "M15",
            "M30",
            "H1",
            "H2",
            "H4",
            "D1",
            "W1",
            "MN1",
        ]
    )
    enable_level2: bool = False
    level2_depth_levels: int = 10
    history_years: int = 10
    trading_session_start: str = "00:05"
    trading_session_end: str = "23:55"
    session_timezone: str = "UTC"
    broker_backend: str = "mt5_local"
    mt5_required: bool = Field(default_factory=lambda: os.name == "nt")  # Auto-disable on Linux/Mac
    mt5_terminal_path: str = ""
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_timeout_seconds: int = 30
    mt5_timezone_offset_hours: int = 0  # Offset of Broker Server Time from UTC (e.g. 2 or 3). Auto-detected if 0.
    poll_interval_seconds: int = 60
    metrics_logging_enabled: bool = True
    metrics_db_path: str = "metrics.sqlite"
    risk_ledger_enabled: bool = True
    risk_ledger_max_events: int = 1000
    strategy_ledger_path: str = "strategy_ledger.sqlite"
    enable_dashboard: bool = True
    cache_enabled: bool = True
    cache_dir: str = "cache"
    cache_max_age_minutes: int = 60
    n_jobs: int = 2
    enable_gpu_preference: str = "auto"
    enable_gpu: bool = False
    num_gpus: int = 0
    device: str = "cpu"
    evo_multiproc_per_gpu: bool = True
    cache_training_frames: bool = False  # optionally cache training frames per symbol to save I/O (may use RAM)
    training_cache_max_bytes: int = 2_000_000_000  # 2GB cap for cached frames
    max_training_rows_per_tf: int = 0  # if >0, limit loaded rows per timeframe (tail) to save memory
    downcast_training_float32: bool = True  # reduce memory by downcasting numeric float columns to float32
    parquet_memory_map: bool = True  # mmap parquet reads to reduce RAM spikes
    smc_freshness_limit: int = 0  # 0 = auto-tune based on data length
    smc_atr_displacement: float = 0.0  # 0 = auto (use ATR quantiles), otherwise fixed multiplier
    smc_max_levels: int = 0  # 0 = auto (2-4 based on volatility), else fixed
    smc_use_cuda: bool = True  # try CUDA SMC when GPU + numba-cuda available


class RiskConfig(BaseModel):
    # Fallback balance used before MT5 fetch or when offline
    initial_balance: float = 10_000.0
    monthly_profit_target_pct: float = 0.04
    min_risk_per_trade: float = 0.005
    max_risk_per_trade: float = 0.030
    risk_per_trade: float = 0.015
    daily_drawdown_limit: float = 0.045
    total_drawdown_limit: float = 0.07
    min_risk_reward: float = 2.0
    spread_guard_multiplier: float = 2.5
    slippage_guard_multiplier: float = 2.0
    challenge_mode: bool = False
    challenge_phase: str = "phase_1"
    prop_firm_rules: bool = True
    max_daily_risk_pct: float = 0.04
    base_risk_per_trade: float = 0.01
    daily_risk_budget: float = 0.040
    consistency_tracking: bool = True
    min_confidence_threshold: float = 0.55
    kill_zones_enabled: bool = True
    enhanced_features: bool = True
    uncertainty_quantification: bool = True
    max_trades_per_day: int = 8
    daily_profit_stop_pct: float = 0.025
    recovery_mode_enabled: bool = True
    feature_drift_threshold: float = 0.30  # fraction of features allowed to drift beyond 3 std
    high_quality_confidence: float = 0.65
    high_quality_risk_pct: float = 0.030
    high_quality_rr: float = 2.0
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    triple_barrier_max_bars: int = 35
    trailing_enabled: bool = True
    trailing_atr_multiplier: float = 1.0
    trailing_be_trigger_r: float = 1.0
    kelly_lambda: float = 1.0
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0
    backtest_spread_pips: float = 1.5
    cost_penalty_r: float = 0.0
    gate_trade_prob: float = 0.55
    daily_hard_stop_pct: float = 0.04
    conformal_alpha: float = 0.10
    volatility_stop_sigma: float = 0.02  # if recent returns std exceeds, pause trading
    volatility_lookback: int = 50  # bars to compute realized volatility
    meta_label_tp_pips: float = 20.0
    meta_label_sl_pips: float = 20.0
    meta_label_max_hold_bars: int = 100  # Equivalent to 100 * 5 min = ~8 hours on M5
    meta_label_min_prob_threshold: float = 0.55  # Only meta-label signals with this probability


class ModelsConfig(BaseModel):
    ml_models: list[str] = Field(
        default_factory=lambda: [
            "lightgbm",
            "xgboost",
            "catboost",
            "random_forest",
            "extra_trees",
        ]
    )
    use_rl_agent: bool = True
    use_sac_agent: bool = True
    use_rllib_agent: bool = False
    rllib_num_workers: int = 0
    auto_enable_rllib: bool = True
    use_neuroevolution: bool = True
    rl_population_size: int = 5
    rl_timesteps: int = 10000000
    rl_eval_episodes: int = 15
    rl_network_arch: list[int] = Field(default_factory=lambda: [4096, 4096, 4096, 2048, 1024])
    rl_parallel_envs: int = 1
    rl_train_seconds: int = 14400
    evo_train_seconds: int = 14400
    # Shorter per-trial budget for Optuna HPO; keeps Evo_Opt from running for hours per trial.
    evo_optuna_train_seconds: int = 900
    evo_hidden_size: int = 64
    evo_population: int = 32
    evo_islands: int = 4
    train_batch_size: int = 32
    inference_batch_size: int = 32
    enable_transformer_expert: bool = True
    transformer_heads: int = 16
    transformer_layers: int = 12
    transformer_hidden_dim: int = 1200
    transformer_dropout: float = 0.20
    transformer_seq_len: int = 64
    transformer_train_seconds: int = 7200
    nbeats_train_seconds: int = 3600
    tide_train_seconds: int = 1800
    tabnet_train_seconds: int = 3600  # Increased for deeper training
    kan_train_seconds: int = 3600  # Increased for deeper training
    num_transformers: int = 2
    optuna_trials: int = 25
    optuna_reuse_study: bool = False
    optuna_max_rows: int = 0  # 0 = no cap; allow full-row trials
    optuna_storage_url: str = (
        "redis://localhost:6379/0"  # default Redis for speed; override for your Redis/RDB endpoint
    )
    optuna_sampler: str = "tpe"  # tpe | random | cmaes
    optuna_pruner: str = "asha"  # asha | median | hyperband
    export_onnx: bool = False  # ONNX export is optional; disable by default for clean, portable runs
    symbol_hash_buckets: int = 32  # fixed-dim hashed one-hot for multi-symbol/global models
    global_train_ratio: float = 0.8  # time-based split for pooled multi-symbol training
    train_holdout_pct: float = 0.2  # holdout for per-model OOS eval + stacking
    walkforward_splits: int = 20
    embargo_minutes: int = 120
    prop_metric_weight: float = 1.0
    prop_accuracy_weight: float = 0.1
    prop_min_trades: int = 30
    prop_conf_threshold: float = 0.55
    enable_cpcv: bool = True
    cpcv_n_splits: int = 5
    cpcv_n_test_groups: int = 2
    cpcv_embargo_pct: float = 0.01
    cpcv_purge_pct: float = 0.02
    cpcv_min_phi: float = 0.80
    # Guardrail: CPCV is memory-heavy (sklearn casts to float64). Cap rows to keep local runs stable.
    cpcv_max_rows: int = 0  # 0 = no cap; use full data


class NewsConfig(BaseModel):
    news_decay_minutes: int = 120
    news_kill_window_min: int = 30
    news_confidence_threshold: float = 0.65
    news_lookahead_minutes: int = 60
    news_trade_on_event: bool = False
    news_trade_confidence_threshold: float = 0.90
    enable_news: bool = True
    enable_llm_helper: bool = True
    llm_helper_enabled: bool = True
    llm_sentiment_positive_threshold: float = 0.2
    llm_sentiment_negative_threshold: float = -0.2
    news_backfill_enabled: bool = True
    news_backfill_days: int = 30  # limit backfill window when enabled
    news_local_glob: str = ""  # Optional local CSV glob for news archives (e.g., data/news/*.csv)
    openai_model: str = "gpt-5-nano-2025-08-07"
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_max_tokens: int = 256
    openai_max_events_per_fetch: int = 50
    openai_news_enabled: bool = True
    perplexity_enabled: bool = True
    perplexity_api_key_env: str = "PPLX_API_KEY"
    perplexity_model: str = "sonar"
    perplexity_num_results: int = 10
    perplexity_timeframe_hours: int = 24
    strategist_enabled: bool = False
    strategist_interval_minutes: int = 30

    # Optional: automatically rescore existing (archive/DB) events with OpenAI to replace 0/low-quality scores.
    # Kept conservative by default to avoid unexpected cost.
    auto_rescore_enabled: bool = False
    auto_rescore_days: int = 30  # lookback from "end" timestamp; 0 = full span
    auto_rescore_max_events: int = 200
    auto_rescore_only_missing: bool = True


class Settings(BaseSettings):
    system: SystemConfig = Field(default_factory=SystemConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    news: NewsConfig = Field(default_factory=NewsConfig)
    secrets_file: str = "keys.txt"  # Optional local secrets file (git-ignored) to load API keys

    model_config = SettingsConfigDict(env_nested_delimiter="__", env_file=".env", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


settings = Settings()

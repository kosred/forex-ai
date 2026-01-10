pub mod challenge;
pub mod discovery;
pub mod discovery_gpu;
pub mod eval;
pub mod gauntlet;
pub mod genetic;
pub mod portfolio;
pub mod quality;
pub mod stop_target;

pub use challenge::{ChallengeOptimizer, ChallengeTarget};
pub use discovery::{run_discovery_cycle, save_portfolio_json, DiscoveryConfig, DiscoveryResult};
pub use discovery_gpu::{
    build_feature_cube, run_gpu_discovery, save_gpu_genomes, GpuDiscoveryConfig, GpuDiscoveryResult,
};
pub use eval::{evaluate_population_core, fast_evaluate_strategy_core};
pub use gauntlet::{GauntletConfig, StrategyGauntlet};
pub use genetic::{evaluate_genes, evolve_search, random_search, signals_for_gene, EvaluationConfig, Gene, SearchResult};
pub use portfolio::{AllocationResult, PortfolioOptimizer, SymbolMetrics};
pub use quality::{StrategyMetrics, StrategyQualityAnalyzer, StrategyRanker, Trade};
pub use stop_target::{compute_stop_distance_series, infer_stop_target_pips, StopTargetSettings};

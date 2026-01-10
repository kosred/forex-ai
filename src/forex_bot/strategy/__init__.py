"""Strategy helpers (Rust-accelerated backtesting)."""

from .fast_backtest import fast_evaluate_strategy, batch_evaluate_strategies, infer_pip_metrics, infer_sl_tp_pips_auto
from .stop_target import infer_stop_target_pips
from .discovery import AutonomousDiscoveryEngine
from .discovery_tensor import TensorDiscoveryEngine

__all__ = [
    "fast_evaluate_strategy",
    "batch_evaluate_strategies",
    "infer_pip_metrics",
    "infer_sl_tp_pips_auto",
    "infer_stop_target_pips",
    "AutonomousDiscoveryEngine",
    "TensorDiscoveryEngine",
]

# Lazy imports to avoid circular dependency with training.ensemble
def __getattr__(name: str):
    if name == "SignalEngine":
        from .engine import SignalEngine
        return SignalEngine
    elif name == "FeatureEngineer":
        from .pipeline import FeatureEngineer
        return FeatureEngineer
    elif name == "_compute_smc_numba":
        from .smc import _compute_smc_numba
        return _compute_smc_numba
    elif name == "_compute_structure_labels_numba":
        from .smc import _compute_structure_labels_numba
        return _compute_structure_labels_numba
    elif name == "AtrOnlineTuner":
        from .tuners import AtrOnlineTuner
        return AtrOnlineTuner
    elif name == "RiskOnlineTuner":
        from .tuners import RiskOnlineTuner
        return RiskOnlineTuner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SignalEngine",
    "FeatureEngineer",
    "_compute_smc_numba",
    "_compute_structure_labels_numba",
    "AtrOnlineTuner",
    "RiskOnlineTuner",
]

"""
Lazy-Loading Model Registry.
Thread-safe implementation for HPC environments.
"""

import logging
import os
import threading
import importlib
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from .base import ExpertModel

logger = logging.getLogger(__name__)

# HPC FIX: Thread Lock for Registry
_REGISTRY_LOCK = threading.Lock()
_CLASS_CACHE: Dict[str, Type['ExpertModel']] = {}

# Registry mapping: name -> (module_path, class_name)
MODEL_MAPPING = {
    "lightgbm": ("trees", "LightGBMExpert"),
    "xgboost": ("trees", "XGBoostExpert"),
    "xgboost_rf": ("trees", "XGBoostRFExpert"),
    "xgboost_dart": ("trees", "XGBoostDARTExpert"),
    "catboost": ("trees", "CatBoostExpert"),
    "catboost_alt": ("trees", "CatBoostAltExpert"),
    "mlp": ("mlp", "MLPExpert"),
    "transformer": ("transformers", "TransformerExpertTorch"),
    "kan": ("kan_gpu", "KANExpert"),
    "nbeats": ("nbeats_gpu", "NBeatsExpert"),
    "tabnet": ("tabnet_gpu", "TabNetExpert"),
    "tide": ("tide_gpu", "TiDEExpert"),
    "rl_ppo": ("rl", "RLExpertPPO"),
    "rl_sac": ("rl", "RLExpertSAC"),
    "rllib_ppo": ("rllib_agent", "RLlibPPOAgent"),
    "rllib_sac": ("rllib_agent", "RLlibSACAgent"),
    "evolution": ("evolution", "EvoExpertCMA"),
    "genetic": ("genetic", "GeneticStrategyExpert"),
    "unsupervised": ("unsupervised", "ClusterExpert"),
}

_RUST_TREE_MAPPING = {
    "lightgbm": "RustLightGBMExpert",
    "xgboost": "RustXGBoostExpert",
    "xgboost_rf": "RustXGBoostRFExpert",
    "xgboost_dart": "RustXGBoostDARTExpert",
}


def _use_rust_tree_models() -> bool:
    """Return True if Rust tree bindings should be preferred."""
    raw = os.environ.get("FOREX_BOT_TREE_BACKEND", "auto").strip().lower()
    if raw in {"rust", "1", "true", "yes", "on"}:
        return True
    if raw in {"python", "py", "0", "false", "no", "off"}:
        return False
    # auto: try to detect bindings
    try:
        import forex_bindings  # noqa: F401

        return True
    except Exception:
        return False

def register_model(name: str, module_path: str, class_name: str) -> None:
    """Dynamically registers a new model type."""
    with _REGISTRY_LOCK:
        MODEL_MAPPING[name] = (module_path, class_name)
        # Clear cache if overwriting
        if name in _CLASS_CACHE:
            del _CLASS_CACHE[name]
        logger.info(f"Registered new model: {name} -> {module_path}.{class_name}")

def get_model_class(name: str, prefer_gpu: bool = False) -> Type['ExpertModel']:
    """Thread-safe lazy-imports the requested model class."""
    with _REGISTRY_LOCK:
        if name in _CLASS_CACHE:
            return _CLASS_CACHE[name]
        
        if name not in MODEL_MAPPING:
            raise ValueError(f"Model '{name}' not found in registry.")
        
        module_name, class_name = MODEL_MAPPING[name]
        rust_requested = False

        if name in _RUST_TREE_MAPPING and _use_rust_tree_models():
            module_name = "trees_rust"
            class_name = _RUST_TREE_MAPPING[name]
            rust_requested = True

        # Handle CPU fallback for GPU models if needed
        if not prefer_gpu and name in {"kan", "nbeats", "tabnet", "tide"}:
            module_name = module_name.replace("_gpu", "")

        try:
            # Import with package context
            module = importlib.import_module(f".{module_name}", package="forex_bot.models")
            cls = getattr(module, class_name)
            if rust_requested and getattr(cls, "_model_cls", None) is None:
                raise ImportError(f"Rust bindings missing class for {name}")
            _CLASS_CACHE[name] = cls
            return cls
        except Exception as e:
            if rust_requested:
                # Rust bindings failed; fall back to Python implementation.
                base_module, base_class = MODEL_MAPPING[name]
                try:
                    module = importlib.import_module(f".{base_module}", package="forex_bot.models")
                    cls = getattr(module, base_class)
                    _CLASS_CACHE[name] = cls
                    logger.warning(
                        "Rust bindings unavailable for '%s' (error: %s); using Python model.",
                        name,
                        e,
                    )
                    return cls
                except Exception as py_exc:
                    logger.error(
                        "Python fallback import failed for '%s' after Rust error: %s",
                        name,
                        py_exc,
                    )
                    raise ImportError(f"Could not load model {name}") from py_exc

            # If GPU module import fails, try CPU implementation as fallback.
            if name in {"kan", "nbeats", "tabnet", "tide"} and module_name.endswith("_gpu"):
                try:
                    cpu_module = module_name.replace("_gpu", "")
                    module = importlib.import_module(f".{cpu_module}", package="forex_bot.models")
                    cls = getattr(module, class_name)
                    _CLASS_CACHE[name] = cls
                    logger.warning(
                        "Falling back to CPU model for '%s' after GPU import failure: %s",
                        name,
                        e,
                    )
                    return cls
                except Exception as cpu_exc:
                    logger.error(
                        "CPU fallback import failed for '%s' after GPU import error: %s",
                        name,
                        cpu_exc,
                    )
                    raise ImportError(f"Could not load model {name}") from cpu_exc
            logger.error(f"Failed to lazy-import model '{name}': {e}")
            raise ImportError(f"Could not load model {name}") from e

# Keep for backward compatibility with existing code
MODEL_REGISTRY = MODEL_MAPPING

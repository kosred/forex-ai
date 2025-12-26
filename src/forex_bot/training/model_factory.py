import inspect
import logging
from pathlib import Path

from ..core.config import Settings
from ..core.system import normalize_device_preference
from ..models.base import ExpertModel
from ..models.device import get_available_gpus
from ..models.registry import get_model_class

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Handles instantiation, configuration, and warm-starting of ExpertModels.
    Uses settings + hardware availability for device selection.
    """

    def __init__(self, settings: Settings, models_dir: Path):
        self.settings = settings
        self.models_dir = models_dir

        pref = normalize_device_preference(getattr(self.settings.system, "enable_gpu_preference", "auto"))
        self.available_gpus = get_available_gpus() if pref != "cpu" else []
        self.prefer_gpu = bool(self.available_gpus) and pref in {"auto", "gpu"}

    def create_model(self, model_name: str, best_params: dict, idx: int) -> ExpertModel:
        """Create and configure a model instance."""
        prefer_gpu = self.prefer_gpu and bool(self.available_gpus)
        model_cls = get_model_class(model_name, prefer_gpu=prefer_gpu)

        # 1. Resolve Parameters
        params = {}
        # Map config names to HPO key names if needed
        opt_key_map = {
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "catboost": "CatBoost",
            "tabnet": "TabNet",
            "nbeats": "N-BEATS",
            "tide": "TiDE",
            "kan": "KAN",
            "transformer": "Transformer",
            "evolution": "Neuroevolution",
            "extra_trees": "ExtraTrees",
        }
        opt_key = opt_key_map.get(model_name)
        if opt_key and opt_key in best_params:
            params = best_params[opt_key].copy()
        elif model_name in best_params:
            params = best_params[model_name].copy()

        # 2. Batch size (config override if not fixed by HPO params)
        if "batch_size" not in params:
            cfg_bs = getattr(self.settings.models, "train_batch_size", None)
            if cfg_bs and int(cfg_bs) > 0:
                params["batch_size"] = int(cfg_bs)

        # 2b. Size presets (approx ~3M-class defaults) if not already set by HPO/config
        if model_name == "transformer":
            params.setdefault("d_model", int(getattr(self.settings.models, "transformer_d_model", 256) or 256))
            params.setdefault("n_heads", int(getattr(self.settings.models, "transformer_n_heads", 8) or 8))
            params.setdefault("n_layers", int(getattr(self.settings.models, "transformer_n_layers", 4) or 4))
        if model_name in {"patchtst", "timesnet"}:
            params.setdefault("hidden_dim", int(getattr(self.settings.models, "nf_hidden_dim", 256) or 256))
        if model_name in {"tide", "tide_nf"}:
            params.setdefault("hidden_dim", int(getattr(self.settings.models, "tide_hidden_dim", 256) or 256))
        if model_name in {"nbeatsx_nf"}:
            params.setdefault("hidden_dim", int(getattr(self.settings.models, "nbeats_hidden_dim", 256) or 256))
        if model_name == "kan":
            params.setdefault("hidden_dim", int(getattr(self.settings.models, "kan_hidden_dim", 256) or 256))
        if model_name == "tabnet":
            params.setdefault("hidden_dim", int(getattr(self.settings.models, "tabnet_hidden_dim", 64) or 64))

        # 4. Filter Init Kwargs based on Signature
        init_kwargs = params.copy()
        # Tree experts expect a single "params" dict in __init__
        if model_name in {"lightgbm", "random_forest", "extra_trees"}:
            init_kwargs = {"params": params.copy()}

        # Inject Device if supported
        if prefer_gpu and self.available_gpus:
            # Simple round-robin for multi-GPU
            device_str = self.available_gpus[(idx - 1) % len(self.available_gpus)]
        else:
            device_str = "cpu"

        # Explicitly check signature and only pass supported args
        try:
            sig = inspect.signature(model_cls.__init__)

            # 1. Filter out unsupported params from init_kwargs
            valid_keys = set(sig.parameters.keys())
            init_kwargs = {k: v for k, v in init_kwargs.items() if k in valid_keys}

            # 2. Inject device if supported
            if "device" in valid_keys and "device" not in init_kwargs:
                init_kwargs["device"] = device_str

        except Exception as e:
            logger.warning(f"Signature inspection failed for {model_name}: {e}")
            # Fallback: try raw params
            pass

        # 5. Instantiate
        try:
            model = model_cls(**init_kwargs)
        except Exception as e:
            logger.warning(f"Failed to instantiate {model_name} with args, retrying empty: {e}")
            model = model_cls()

        # 6. Post-Init Configuration
        self._configure_instance(model, model_name, device_str, params)
        self._maybe_warm_start(model, model_name)

        return model

    def _configure_instance(self, model: ExpertModel, name: str, device: str, params: dict):
        if hasattr(model, "device"):
            model.device = device

        # Model-specific budget injection (seconds)
        if hasattr(model, "max_time_sec"):
            budget_key = {
                "transformer": "transformer_train_seconds",
                "tabnet": "tabnet_train_seconds",
                "nbeats": "nbeats_train_seconds",
                "tide": "tide_train_seconds",
                "kan": "kan_train_seconds",
                "evolution": "evo_train_seconds",
                "rl_ppo": "rl_train_seconds",
                "rl_sac": "rl_train_seconds",
                "rllib_ppo": "rl_train_seconds",
                "rllib_sac": "rl_train_seconds",
            }.get(name)
            if budget_key:
                try:
                    model.max_time_sec = int(getattr(self.settings.models, budget_key))
                except Exception:
                    pass

        # Special handling
        if name == "evolution":
            if hasattr(model, "hidden_size"):
                model.hidden_size = int(getattr(self.settings.models, "evo_hidden_size", 64))
            if hasattr(model, "population"):
                model.population = self.settings.models.evo_population
            if hasattr(model, "num_islands"):
                model.num_islands = self.settings.models.evo_islands
            if hasattr(model, "evo_multiproc_per_gpu"):
                model.evo_multiproc_per_gpu = bool(getattr(self.settings.system, "evo_multiproc_per_gpu", True))

        if name in {"rllib_ppo", "rllib_sac"}:
            if hasattr(model, "parallel_envs"):
                model.parallel_envs = getattr(self.settings.models, "rllib_num_workers", 1)

        # Apply remaining params
        for k, v in params.items():
            if hasattr(model, k):
                setattr(model, k, v)

    def _maybe_warm_start(self, model: ExpertModel, name: str):
        try:
            if hasattr(model, "load"):
                # Try standard paths
                candidates = [
                    self.models_dir / f"{name}.pkl",
                    self.models_dir / f"{name}.pt",
                    self.models_dir / f"{name}_model.pkl",
                ]
                for p in candidates:
                    if p.exists():
                        model.load(str(self.models_dir))
                        logger.info(f"Warm-started {name}")
                        return
        except Exception as e:
            logger.warning(f"Warm start failed for {name}: {e}")

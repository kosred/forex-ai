import logging

from .base import ExpertModel

logger = logging.getLogger(__name__)

# MODEL_REGISTRY maps model_name -> {"cpu": cls_cpu, "gpu": cls_gpu or cls_cpu}
MODEL_REGISTRY: dict[str, dict[str, type[ExpertModel]]] = {}


def register_model(name: str, cpu_cls: type[ExpertModel], gpu_cls: type[ExpertModel] | None = None) -> None:
    MODEL_REGISTRY[name] = {"cpu": cpu_cls, "gpu": gpu_cls or cpu_cls}


def get_model_class(name: str, prefer_gpu: bool = False) -> type[ExpertModel]:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    entry = MODEL_REGISTRY[name]
    if prefer_gpu and entry.get("gpu"):
        return entry["gpu"]
    return entry["cpu"]


# Tree/linear models (CPU only)
try:
    from .trees import CatBoostExpert, ExtraTreesExpert, LightGBMExpert, RandomForestExpert, XGBoostExpert

    register_model("lightgbm", LightGBMExpert)
    register_model("xgboost", XGBoostExpert)
    register_model("catboost", CatBoostExpert)
    register_model("random_forest", RandomForestExpert)
    register_model("extra_trees", ExtraTreesExpert)
except ImportError as e:
    logger.warning(f"Tree models not fully available: {e}")

# Deep models: always register both CPU and GPU variants when present
try:
    from .nbeats import NBeatsExpert as NBeatsCPU
except ImportError:
    NBeatsCPU = None
try:
    from .nbeats_gpu import NBeatsExpert as NBeatsGPU
except ImportError:
    NBeatsGPU = None
if NBeatsCPU:
    register_model("nbeats", NBeatsCPU, NBeatsGPU)

try:
    from .tabnet import TabNetExpert as TabNetCPU
except ImportError:
    TabNetCPU = None
try:
    from .tabnet_gpu import TabNetExpert as TabNetGPU
except ImportError:
    TabNetGPU = None
if TabNetCPU:
    register_model("tabnet", TabNetCPU, TabNetGPU)

try:
    from .tide import TiDEExpert as TiDECPU
except ImportError:
    TiDECPU = None
try:
    from .tide_gpu import TiDEExpert as TiDEGPU
except ImportError:
    TiDEGPU = None
if TiDECPU:
    register_model("tide", TiDECPU, TiDEGPU)

try:
    from .kan import KANExpert as KANCPU  # noqa: N814
except ImportError:
    KANCPU = None
try:
    from .kan_gpu import KANExpert as KANGPU  # noqa: N814
except ImportError:
    KANGPU = None
if KANCPU:
    register_model("kan", KANCPU, KANGPU)

try:
    from .transformers import TransformerExpertTorch

    register_model("transformer", TransformerExpertTorch, TransformerExpertTorch)  # same class, device handled inside
except ImportError as e:
    logger.warning(f"Transformer models not available: {e}")

# NeuralForecast Transformers (PatchTST / TimesNet)
try:
    from .transformer_nf import PatchTSTExpert, TimesNetExpert

    register_model("patchtst", PatchTSTExpert)
    register_model("timesnet", TimesNetExpert)
except ImportError as e:
    logger.warning(f"NeuralForecast transformers not available: {e}")

# NeuralForecast TiDE / NBEATSx
try:
    from .forecast_nf import TiDENFExpert, NBEATSxNFExpert

    register_model("tide_nf", TiDENFExpert)
    register_model("nbeatsx_nf", NBEATSxNFExpert)
except ImportError as e:
    logger.warning(f"NeuralForecast TiDE/NBEATSx not available: {e}")

try:
    from .rl import RLExpertPPO, RLExpertSAC

    register_model("rl_ppo", RLExpertPPO)
    register_model("rl_sac", RLExpertSAC)
except ImportError as e:
    logger.warning(f"RL models not available: {e}")
try:
    from .rllib_agent import RLlibPPOAgent, RLlibSACAgent

    register_model("rllib_ppo", RLlibPPOAgent)
    register_model("rllib_sac", RLlibSACAgent)
except ImportError as e:
    logger.warning(f"RLlib models not available: {e}")

try:
    from .evolution import EvoExpertCMA

    register_model("evolution", EvoExpertCMA)
except ImportError as e:
    logger.warning(f"Evolution models not available: {e}")

try:
    from .genetic import GeneticStrategyExpert

    register_model("genetic", GeneticStrategyExpert)
except ImportError as e:
    logger.warning(f"Genetic models not available: {e}")

try:
    from .unsupervised import ClusterExpert

    register_model("unsupervised", ClusterExpert)
except ImportError as e:
    logger.warning(f"Unsupervised models not available: {e}")

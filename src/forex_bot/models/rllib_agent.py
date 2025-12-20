import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .base import ExpertModel
from .device import get_available_gpus

logger = logging.getLogger(__name__)

try:
    import ray  # type: ignore
    from ray.rllib.algorithms.ppo import PPOConfig  # type: ignore
    from ray.rllib.algorithms.sac import SACConfig  # type: ignore

    RAY_AVAILABLE = True
except Exception:
    ray = None
    PPOConfig = None
    SACConfig = None
    RAY_AVAILABLE = False


def _default_config(device: str, num_workers: int = 1) -> dict[str, Any]:
    gpu = 1 if isinstance(device, str) and device.startswith("cuda") else 0
    return {
        "num_gpus": gpu,
        "num_cpus_per_worker": 1,
        "num_gpus_per_worker": 0,
        "num_workers": max(0, num_workers - 1),
        "framework": "torch",
    }


def _apply_resources(cfg: Any, device: str, num_workers: int) -> Any:
    """
    Map GPUs to learner/rollout workers.
    - Single GPU: give it to the learner, keep workers on CPU.
    - Multi-GPU: reserve 1 for learner, give 1 per worker up to remaining GPUs.
    """
    import multiprocessing

    gpus = get_available_gpus()
    total_gpus = len(gpus)
    cpu_total = multiprocessing.cpu_count()

    learner_gpus = 1 if total_gpus > 0 else 0

    req_workers = max(0, num_workers)
    if total_gpus > 1:
        max_gpu_workers = max(0, total_gpus - 1)
        worker_count = min(req_workers, max_gpu_workers)
        worker_gpus = 1 if worker_count > 0 else 0
    else:
        worker_count = min(req_workers, max(0, cpu_total - 1))
        worker_gpus = 0

    cfg = cfg.resources(num_gpus=learner_gpus, num_gpus_per_worker=worker_gpus, num_cpus_per_worker=1)
    # Optimize for CPU: Vectorize environments (4 per worker) to amortize overhead
    cfg = cfg.env_runners(num_env_runners=worker_count, num_envs_per_env_runner=4)
    return cfg


def _maybe_init_ray() -> bool:
    if not RAY_AVAILABLE or ray is None:
        return False
    if ray.is_initialized():
        return True
    try:
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        return True
    except Exception as exc:
        logger.info(f"Ray init failed, skipping RLlib: {exc}")
        return False


class _TabularEnv:
    """
    Simple Gym environment wrapper for tabular data (DataFrame/Numpy).
    Must be at module level for Ray/Pickle compatibility.
    """

    def __init__(self, config: dict = None) -> None:
        import gymnasium as gym

        config = config or {}
        self.data = config.get("data")
        self.labels = config.get("labels")

        # Safety check for unpickled/missing data
        if self.data is None or self.labels is None:
            # Fallback for initialization checks
            self.data = np.zeros((10, 10), dtype=np.float32)
            self.labels = np.zeros(10, dtype=np.int64)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.i = 0
        self.n = len(self.labels)

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict]:
        self.i = 0
        return self.data[self.i], {}

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]:
        label = self.labels[self.i]
        reward = 1.0 if (action - 1) == label else -1.0
        self.i = (self.i + 1) % self.n
        done = False
        return self.data[self.i], reward, done, False, {}


@dataclass(slots=True)
class RLlibPPOAgent(ExpertModel):
    timesteps: int = 50_000
    max_time_sec: int = 600
    device: str = "cpu"
    parallel_envs: int = 1
    config_overrides: dict[str, Any] = field(default_factory=dict)
    algo: Any = None

    def fit(self, x: pd.DataFrame, y: pd.Series, env_fn: Any | None = None) -> None:
        if not RAY_AVAILABLE or PPOConfig is None:
            logger.info("RLlib PPO not available; skipping fit.")
            return
        if not _maybe_init_ray():
            return

        # Prepare data for the env
        data_obs = x.to_numpy(dtype=np.float32)
        labels = y.to_numpy(dtype=np.int64)

        # We pass data via env_config
        env_config = {"data": data_obs, "labels": labels}

        if env_fn is None:
            # Use the module-level class
            env_cls = _TabularEnv
        else:
            env_cls = env_fn

        config = PPOConfig().environment(env=env_cls, env_config=env_config)
        config = _apply_resources(config, self.device, self.parallel_envs)

        for k, v in self.config_overrides.items():
            try:
                config = config.update_from_dict({k: v})
            except Exception as e:
                logger.info(f"RLlib config update failed: {e}", exc_info=True)

        self.algo = config.build()
        try:
            self.algo.train()
        except Exception as exc:
            logger.info(f"RLlib PPO training failed: {exc}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.algo is None:
            raise RuntimeError("RLlib PPO not loaded")
        try:
            obs = x.to_numpy(dtype=np.float32)
            actions, _, _ = self.algo.compute_actions(
                observations=obs,
                explore=False,  # Deterministic prediction
            )

            probs = np.zeros((len(actions), 3))
            for i, a in enumerate(actions):
                probs[i, int(a) % 3] = 1.0
            return probs
        except Exception as e:
            logger.error(f"RLlib PPO prediction failed: {e}", exc_info=True)
            raise

    def save(self, path: str) -> None:
        return

    def load(self, path: str) -> None:
        return


@dataclass(slots=True)
class RLlibSACAgent(ExpertModel):
    timesteps: int = 50_000
    max_time_sec: int = 600
    device: str = "cpu"
    parallel_envs: int = 1
    config_overrides: dict[str, Any] = field(default_factory=dict)
    algo: Any = None

    def fit(self, x: pd.DataFrame, y: pd.Series, env_fn: Any | None = None) -> None:
        if not RAY_AVAILABLE or SACConfig is None:
            logger.info("RLlib SAC not available; skipping fit.")
            return
        if not _maybe_init_ray():
            return

        data_obs = x.to_numpy(dtype=np.float32)
        labels = y.to_numpy(dtype=np.int64)

        env_config = {"data": data_obs, "labels": labels}

        if env_fn is None:
            env_cls = _TabularEnv
        else:
            env_cls = env_fn

        config = SACConfig().environment(env=env_cls, env_config=env_config)
        config = _apply_resources(config, self.device, self.parallel_envs)
        self.algo = config.build()
        try:
            self.algo.train()
        except Exception as exc:
            logger.info(f"RLlib SAC training failed: {exc}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.algo is None:
            raise RuntimeError("RLlib SAC not loaded")
        try:
            obs = x.to_numpy(dtype=np.float32)
            actions, _, _ = self.algo.compute_actions(observations=obs, explore=False)

            probs = np.zeros((len(actions), 3))
            for i, a in enumerate(actions):
                probs[i, int(a) % 3] = 1.0
            return probs
        except Exception as e:
            logger.error(f"RLlib SAC prediction failed: {e}", exc_info=True)
            raise

    def save(self, path: str) -> None:
        return

    def load(self, path: str) -> None:
        return

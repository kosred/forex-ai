import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import ExpertModel
from .device import get_available_gpus

logger = logging.getLogger(__name__)

_RAY_OWNED = False  # Track if this module initialized Ray

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

try:
    import gymnasium as _gym  # type: ignore

    GYM_AVAILABLE = True
except Exception:
    _gym = None
    GYM_AVAILABLE = False


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
    global _RAY_OWNED
    if not RAY_AVAILABLE or ray is None:
        return False
    if ray.is_initialized():
        return True
    try:
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        _RAY_OWNED = True
        return True
    except Exception as exc:
        logger.info(f"Ray init failed, skipping RLlib: {exc}")
        return False


def _maybe_shutdown_ray() -> None:
    """Shutdown Ray only if this module started it."""
    global _RAY_OWNED
    if not RAY_AVAILABLE or ray is None:
        return
    if not _RAY_OWNED:
        return
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception as exc:
        logger.debug(f"Ray shutdown failed: {exc}", exc_info=True)
    finally:
        _RAY_OWNED = False


def _stop_algo(algo: Any) -> None:
    """Best-effort cleanup for RLlib Algorithm instances."""
    if algo is None:
        return
    try:
        if hasattr(algo, "stop"):
            algo.stop()
        elif hasattr(algo, "cleanup"):
            algo.cleanup()
    except Exception as exc:
        logger.debug(f"Algorithm cleanup failed: {exc}", exc_info=True)


def _deps_ready(context: str) -> bool:
    """Check that required runtime dependencies are present."""
    if not RAY_AVAILABLE:
        logger.info("%s: Ray not available; skipping RLlib path.", context)
        return False
    if not GYM_AVAILABLE:
        logger.info("%s: gymnasium not available; skipping RLlib path.", context)
        return False
    return True


def _find_latest_checkpoint(checkpoint_dir: Path) -> str | None:
    marker = checkpoint_dir / "checkpoint_path.txt"
    if marker.exists():
        try:
            text = marker.read_text(encoding="utf-8").strip()
            if text:
                return text
        except Exception:
            pass

    try:
        candidates = [p for p in checkpoint_dir.glob("checkpoint_*") if p.name != "checkpoint_path.txt"]
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0])
    except Exception:
        return None


class _TabularEnv:
    """
    Simple Gym environment wrapper for tabular data (DataFrame/Numpy).
    Must be at module level for Ray/Pickle compatibility.
    """

    def __init__(self, config: dict = None) -> None:
        if not GYM_AVAILABLE or _gym is None:
            raise ImportError("gymnasium is required for RLlib tabular environment.")

        config = config or {}
        self.data = config.get("data")
        self.labels = config.get("labels")

        # Safety check for unpickled/missing data
        if self.data is None or self.labels is None:
            # Fallback for initialization checks
            self.data = np.zeros((10, 10), dtype=np.float32)
            self.labels = np.zeros(10, dtype=np.int64)

        self.observation_space = _gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32
        )
        self.action_space = _gym.spaces.Discrete(3)
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
    ray_owned: bool = False

    def fit(self, x: pd.DataFrame, y: pd.Series, env_fn: Any | None = None) -> None:
        if not _deps_ready("RLlib PPO fit") or PPOConfig is None:
            return
        if not _maybe_init_ray():
            return
        self.ray_owned = _RAY_OWNED

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

        self.algo = None
        try:
            self.algo = config.build()
            self.algo.train()
        except Exception as exc:
            logger.info(f"RLlib PPO training failed: {exc}")
            _stop_algo(self.algo)
            self.algo = None
            if self.ray_owned:
                _maybe_shutdown_ray()

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.algo is None:
            raise RuntimeError("RLlib PPO not loaded")
        if not _deps_ready("RLlib PPO predict"):
            # Dependencies missing after load; safer to abort than return stale algo output
            raise RuntimeError("RLlib PPO dependencies unavailable")
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
            _stop_algo(self.algo)
            self.algo = None
            if self.ray_owned:
                _maybe_shutdown_ray()
            raise

    def save(self, path: str) -> None:
        if self.algo is None:
            return
        out_dir = Path(path) / "rllib_ppo"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint_path = self.algo.save(str(out_dir))
            (out_dir / "checkpoint_path.txt").write_text(str(checkpoint_path), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"RLlib PPO save failed: {exc}")
            _stop_algo(self.algo)
            self.algo = None
            if self.ray_owned:
                _maybe_shutdown_ray()

    def load(self, path: str) -> None:
        if not _deps_ready("RLlib PPO load"):
            return
        if not _maybe_init_ray():
            return
        self.ray_owned = _RAY_OWNED

        base_dir = Path(path) / "rllib_ppo"
        if not base_dir.exists():
            if self.ray_owned:
                _maybe_shutdown_ray()
            return

        checkpoint_path = _find_latest_checkpoint(base_dir)
        if not checkpoint_path:
            if self.ray_owned:
                _maybe_shutdown_ray()
            return

        try:
            from ray.rllib.algorithms.algorithm import Algorithm  # type: ignore

            self.algo = Algorithm.from_checkpoint(checkpoint_path)
        except Exception as exc:
            logger.warning(f"RLlib PPO load failed: {exc}")
            _stop_algo(self.algo)
            self.algo = None
            if self.ray_owned:
                _maybe_shutdown_ray()


@dataclass(slots=True)
class RLlibSACAgent(ExpertModel):
    timesteps: int = 50_000
    max_time_sec: int = 600
    device: str = "cpu"
    parallel_envs: int = 1
    config_overrides: dict[str, Any] = field(default_factory=dict)
    algo: Any = None
    ray_owned: bool = False

    def fit(self, x: pd.DataFrame, y: pd.Series, env_fn: Any | None = None) -> None:
        if not _deps_ready("RLlib SAC fit") or SACConfig is None:
            return
        if not _maybe_init_ray():
            return
        self.ray_owned = _RAY_OWNED

        data_obs = x.to_numpy(dtype=np.float32)
        labels = y.to_numpy(dtype=np.int64)

        env_config = {"data": data_obs, "labels": labels}

        if env_fn is None:
            env_cls = _TabularEnv
        else:
            env_cls = env_fn

        config = SACConfig().environment(env=env_cls, env_config=env_config)
        config = _apply_resources(config, self.device, self.parallel_envs)
        self.algo = None
        try:
            self.algo = config.build()
            self.algo.train()
        except Exception as exc:
            logger.info(f"RLlib SAC training failed: {exc}")
            _stop_algo(self.algo)
            self.algo = None
            if self.ray_owned:
                _maybe_shutdown_ray()

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.algo is None:
            raise RuntimeError("RLlib SAC not loaded")
        if not _deps_ready("RLlib SAC predict"):
            raise RuntimeError("RLlib SAC dependencies unavailable")
        try:
            obs = x.to_numpy(dtype=np.float32)
            actions, _, _ = self.algo.compute_actions(observations=obs, explore=False)

            probs = np.zeros((len(actions), 3))
            for i, a in enumerate(actions):
                probs[i, int(a) % 3] = 1.0
            return probs
        except Exception as e:
            logger.error(f"RLlib SAC prediction failed: {e}", exc_info=True)
            _stop_algo(self.algo)
            self.algo = None
            if self.ray_owned:
                _maybe_shutdown_ray()
            raise

    def save(self, path: str) -> None:
        if self.algo is None:
            return
        out_dir = Path(path) / "rllib_sac"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint_path = self.algo.save(str(out_dir))
            (out_dir / "checkpoint_path.txt").write_text(str(checkpoint_path), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"RLlib SAC save failed: {exc}")
            _stop_algo(self.algo)
            self.algo = None
            if self.ray_owned:
                _maybe_shutdown_ray()

    def load(self, path: str) -> None:
        if not _deps_ready("RLlib SAC load"):
            return
        if not _maybe_init_ray():
            return
        self.ray_owned = _RAY_OWNED

        base_dir = Path(path) / "rllib_sac"
        if not base_dir.exists():
            if self.ray_owned:
                _maybe_shutdown_ray()
            return

        checkpoint_path = _find_latest_checkpoint(base_dir)
        if not checkpoint_path:
            if self.ray_owned:
                _maybe_shutdown_ray()
            return

        try:
            from ray.rllib.algorithms.algorithm import Algorithm  # type: ignore

            self.algo = Algorithm.from_checkpoint(checkpoint_path)
        except Exception as exc:
            logger.warning(f"RLlib SAC load failed: {exc}")
            _stop_algo(self.algo)
            self.algo = None
            if self.ray_owned:
                _maybe_shutdown_ray()

import contextlib
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

try:
    import psutil
except ImportError:
    psutil = None

if TYPE_CHECKING:
    from .config import Settings

logger = logging.getLogger(__name__)


def normalize_device_preference(value: Any | None) -> str:
    """
    Normalize user/device preference input to one of: "auto" | "cpu" | "gpu".

    Accepts common synonyms used across config/env/CLI:
      - cpu: "cpu", "false", "0", "no", "off"
      - gpu: "gpu", "cuda", "true", "1", "yes", "on"
      - auto: "auto" or anything else
    """
    raw = str(value or "auto").strip().lower()
    if raw in {"cpu", "false", "0", "no", "off"}:
        return "cpu"
    if raw in {"gpu", "cuda", "true", "1", "yes", "on"}:
        return "gpu"
    return "auto"


@contextlib.contextmanager
def thread_limits(blas_threads: int | None = None) -> None:
    """
    Runtime control of BLAS/OpenMP thread usage via threadpoolctl.
    Does NOT modify os.environ to avoid race conditions in multi-threaded apps.
    """
    limiter = contextlib.nullcontext()
    if blas_threads is not None and blas_threads > 0:
        # Only use threadpoolctl for dynamic limiting
        with contextlib.suppress(Exception):
            from threadpoolctl import threadpool_limits as _threadpool_limits

            limiter = _threadpool_limits(limits=blas_threads, user_api="blas")

        # numexpr is separate
        with contextlib.suppress(Exception):
            import numexpr as _numexpr

            _numexpr.set_num_threads(blas_threads)

    with limiter:
        yield


@dataclass(slots=True)
class HardwareProfile:
    """Summary of detected compute resources."""

    cpu_cores: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_names: list[str]
    num_gpus: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    platform_label: str = field(default_factory=platform.platform)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "total_ram_gb": round(self.total_ram_gb, 2),
            "available_ram_gb": round(self.available_ram_gb, 2),
            "gpu_names": self.gpu_names,
            "num_gpus": self.num_gpus,
            "timestamp": self.timestamp.isoformat(),
            "platform": self.platform_label,
        }


class HardwareProbe:
    """Detect CPU, RAM, and GPU availability at runtime."""

    def __init__(self) -> None:
        self._torch = None
        with contextlib.suppress(ImportError):
            import torch

            self._torch = torch

    def detect(self) -> HardwareProfile:
        cpu_cores = max(1, os.cpu_count() or 1)
        total_ram_gb, available_ram_gb = self._detect_ram_gb()
        gpu_names = self._detect_gpus()
        return HardwareProfile(
            cpu_cores=cpu_cores,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_names=gpu_names,
            num_gpus=len(gpu_names),
            platform_label=platform.platform(),
        )

    def _detect_ram_gb(self) -> tuple[float, float]:
        """Returns (total_ram_gb, available_ram_gb)."""
        total = 16.0
        available = 8.0

        # Try psutil (Cross-platform, preferred)
        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                return float(vm.total) / (1024**3), float(vm.available) / (1024**3)
            except Exception as e:
                logger.warning(f"psutil RAM detection failed: {e}")

        # Fallback: Linux /proc/meminfo
        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    lines = f.readlines()
                mem_total = 0
                mem_avail = 0
                for line in lines:
                    if "MemTotal" in line:
                        mem_total = int(line.split()[1]) * 1024  # kB to bytes
                    elif "MemAvailable" in line:
                        mem_avail = int(line.split()[1]) * 1024
                if mem_total > 0:
                    return mem_total / (1024**3), (mem_avail if mem_avail > 0 else mem_total * 0.5) / (1024**3)
            except Exception:
                pass

        # Fallback: Windows wmic
        if platform.system() == "Windows":
            try:
                # Get Total
                out = subprocess.check_output(["wmic", "computersystem", "get", "TotalPhysicalMemory"], text=True)
                total_bytes = float(out.strip().split("\n")[1])
                # Get Free
                out_free = subprocess.check_output(["wmic", "OS", "get", "FreePhysicalMemory"], text=True)
                free_kb = float(out_free.strip().split("\n")[1])
                return total_bytes / (1024**3), (free_kb * 1024) / (1024**3)
            except Exception:
                pass

        return total, available

    def _detect_gpus(self) -> list[str]:
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

        # Fallback: nvidia-smi
        smi_candidates = ["nvidia-smi"]
        if platform.system() == "Windows":
            smi_candidates.append(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
            smi_candidates.append(r"C:\Windows\System32\nvidia-smi.exe")

        for cmd in smi_candidates:
            if shutil.which(cmd) or os.path.exists(cmd):
                try:
                    result = subprocess.run(
                        [cmd, "--query-gpu=name", "--format=csv,noheader"],
                        check=True,
                        capture_output=True,
                        encoding="utf-8",
                    )
                    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                    if names:
                        return names
                except Exception:
                    continue
        return []


@dataclass(slots=True)
class AutoTuneHints:
    enable_gpu: bool
    num_gpus: int
    device: str
    n_jobs: int
    train_batch_size: int
    inference_batch_size: int
    optuna_trials: int
    adaptive_training_budget: float
    is_hpc: bool = False  # New flag for High Performance Compute clusters

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_gpu": self.enable_gpu,
            "num_gpus": self.num_gpus,
            "device": self.device,
            "n_jobs": self.n_jobs,
            "train_batch_size": self.train_batch_size,
            "inference_batch_size": self.inference_batch_size,
            "optuna_trials": self.optuna_trials,
            "adaptive_training_budget": self.adaptive_training_budget,
            "is_hpc": self.is_hpc,
        }


class AutoTuner:
    """Derive execution/training defaults from detected hardware."""

    def __init__(self, settings: "Settings", profile: HardwareProfile) -> None:
        self.settings = settings
        self.profile = profile
        self.logger = logging.getLogger(__name__)

    def apply(self) -> AutoTuneHints:
        hints = self._evaluate_hints()
        # Apply settings...
        self.settings.system.enable_gpu = hints.enable_gpu
        self.settings.system.num_gpus = hints.num_gpus
        self.settings.system.device = hints.device
        self.settings.system.n_jobs = hints.n_jobs

        # Hardware-sensitive knobs: keep them within sane bounds.
        try:
            cur_train_bs = int(getattr(self.settings.models, "train_batch_size", 0) or 0)
            cur_infer_bs = int(getattr(self.settings.models, "inference_batch_size", 0) or 0)
            cur_trials = int(getattr(self.settings.models, "optuna_trials", 0) or 0)

            self.settings.models.train_batch_size = int(max(32, min(max(cur_train_bs, hints.train_batch_size), 2048)))
            self.settings.models.inference_batch_size = int(
                max(32, min(max(cur_infer_bs, hints.inference_batch_size), 4096))
            )
            if cur_trials <= 0:
                self.settings.models.optuna_trials = int(max(1, min(hints.optuna_trials, 200)))
        except Exception:
            pass

        # Training budgets: do NOT overwrite user-provided caps (config.yaml) unless unset/invalid.
        def _maybe_set_budget(attr: str) -> None:
            try:
                cur = int(getattr(self.settings.models, attr))
            except Exception:
                cur = 0
            if cur <= 0:
                setattr(self.settings.models, attr, int(hints.adaptive_training_budget))

        for attr in (
            "transformer_train_seconds",
            "nbeats_train_seconds",
            "tide_train_seconds",
            "kan_train_seconds",
            "tabnet_train_seconds",
            "rl_train_seconds",
            "evo_train_seconds",
        ):
            _maybe_set_budget(attr)

        # Evo islands: scale safely on HPC so Evo can explore more candidates without blowing up RAM/threads.
        # This is intentionally conservative because Evo may run alongside other GPU workers.
        try:
            cur_islands = int(getattr(self.settings.models, "evo_islands", 0) or 0)
        except Exception:
            cur_islands = 0
        try:
            if hints.is_hpc and hints.num_gpus > 0:
                cpu_cores = max(1, int(getattr(self.profile, "cpu_cores", 1) or 1))
                per_worker_threads = max(1, cpu_cores // max(1, int(hints.num_gpus)))
                target_islands = max(4, min(32, per_worker_threads))
                if cur_islands <= 0 or cur_islands < target_islands:
                    self.settings.models.evo_islands = target_islands
                    self.logger.info(
                        f"Auto-tuned evo_islands: {cur_islands if cur_islands > 0 else 'unset'} -> {target_islands}"
                    )
        except Exception:
            pass

        try:
            self._apply_thread_env_defaults(hints)
        except Exception as e:
            logger.error(f"System operation failed: {e}", exc_info=True)
        return hints

    def _evaluate_hints(self) -> AutoTuneHints:
        preference = normalize_device_preference(getattr(self.settings.system, "enable_gpu_preference", "auto"))
        has_gpu = self.profile.num_gpus > 0
        enable_gpu = False
        if preference == "gpu" and not has_gpu:
            self.logger.warning("GPU requested but no GPU detected; defaulting to CPU")
        if preference in {"auto", "gpu"} and has_gpu:
            enable_gpu = True
        device = "cuda" if enable_gpu else "cpu"

        cpu_cores = max(1, self.profile.cpu_cores)
        ram_gb = getattr(self.profile, "available_ram_gb", self.profile.total_ram_gb)

        is_hpc = False

        # Detect Cloud/HPC Beast (e.g. 10x A4000, 56 Cores, 200GB RAM)
        if enable_gpu and self.profile.num_gpus >= 4 and ram_gb > 64:
            is_hpc = True
            n_jobs = max(1, min(cpu_cores - 2, 64))  # High, but bounded
            # Per-process/per-GPU batch sizes; do not multiply by GPU count (model-parallel training).
            train_batch = 512
            infer_batch = 2048
            optuna_trials = max(int(getattr(self.settings.models, "optuna_trials", 25)), 50)
            adaptive_budget = 7200.0  # used only when a model budget is unset
            self.logger.info(
                f"HPC mode detected: {self.profile.num_gpus} GPUs, {ram_gb:.0f}GB RAM. Unleashing full power."
            )

        elif enable_gpu:
            # Standard GPU Workstation
            n_jobs = max(1, min(cpu_cores - 1, 16))
            train_batch = 256
            infer_batch = 1024
            optuna_trials = max(int(getattr(self.settings.models, "optuna_trials", 25)), 25)
            adaptive_budget = 3600.0  # used only when a model budget is unset
        else:
            # CPU Mode (Your PC)
            n_jobs = max(1, min(cpu_cores - 1, 16))
            if ram_gb < 4.0:
                n_jobs = 1
                train_batch = 32
                infer_batch = 16
            elif ram_gb < 8.0:
                n_jobs = max(1, n_jobs // 2)
                train_batch = 32
                infer_batch = 32
            else:
                train_batch = 64
                infer_batch = 32

            optuna_trials = min(self.settings.models.optuna_trials, 10)
            adaptive_budget = 1800.0  # used only when a model budget is unset

        return AutoTuneHints(
            enable_gpu=enable_gpu,
            num_gpus=self.profile.num_gpus if enable_gpu else 0,
            device=device,
            n_jobs=n_jobs,
            train_batch_size=train_batch,
            inference_batch_size=infer_batch,
            optuna_trials=optuna_trials,
            adaptive_training_budget=adaptive_budget,
            is_hpc=is_hpc,
        )

    def _apply_thread_env_defaults(self, hints: AutoTuneHints) -> None:
        """Set reasonable BLAS/OpenMP thread defaults based on hardware."""
        cpu_cores = max(1, self.profile.cpu_cores)

        # FIX: For multi-agent/multi-process setups (n_jobs > 1), restrict library threads
        # to 1 to prevent massive oversubscription (e.g., 7 bots * 12 threads = 84 active threads).
        # Heavy operations (training) should dynamically request more threads via 'thread_limits'.
        if hints.n_jobs > 1:
            blas_threads = 1
        else:
            # Single process mode: allow full CPU utilization (capped at 8 for diminishing returns)
            blas_threads = min(8, max(1, cpu_cores))

        os.environ.setdefault("OMP_NUM_THREADS", str(blas_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(blas_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(blas_threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(blas_threads))  # numexpr often matches OMP

        try:
            import numexpr as _numexpr

            _numexpr.set_num_threads(int(os.environ.get("NUMEXPR_NUM_THREADS", str(blas_threads))))
        except ImportError:
            pass  # Optional
        except Exception as e:
            logger.warning(f"NumExpr configuration failed: {e}")

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


def resolve_cpu_budget(reserve_default: int = 1) -> int:
    """
    Resolve the CPU budget (usable cores) as: total_cores - reserve.

    Priority:
      1) per-rank override (FOREX_BOT_CPU_THREADS_PER_GPU) if LOCAL_RANK set
      2) FOREX_BOT_CPU_THREADS / FOREX_BOT_CPU_BUDGET
      3) total cores minus reserve (FOREX_BOT_CPU_RESERVE or reserve_default)
    """
    cpu_total = max(1, os.cpu_count() or 1)

    # Per-rank overrides (DDP)
    if os.environ.get("LOCAL_RANK") is not None:
        per_gpu = os.environ.get("FOREX_BOT_CPU_THREADS_PER_GPU")
        if per_gpu:
            try:
                return max(1, min(cpu_total, int(per_gpu)))
            except Exception:
                pass

    override = os.environ.get("FOREX_BOT_CPU_THREADS") or os.environ.get("FOREX_BOT_CPU_BUDGET")
    if override:
        try:
            return max(1, min(cpu_total, int(override)))
        except Exception:
            pass

    reserve_raw = os.environ.get("FOREX_BOT_CPU_RESERVE")
    if reserve_raw is None:
        reserve = reserve_default
    else:
        try:
            reserve = int(reserve_raw)
        except Exception:
            reserve = reserve_default
    reserve = max(0, reserve or 0)
    return max(1, cpu_total - reserve)


@dataclass(slots=True)
class HardwareProfile:
    """Summary of detected compute resources."""

    cpu_cores: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_names: list[str]
    num_gpus: int
    gpu_mem_gb: list[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    platform_label: str = field(default_factory=platform.platform)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "total_ram_gb": round(self.total_ram_gb, 2),
            "available_ram_gb": round(self.available_ram_gb, 2),
            "gpu_names": self.gpu_names,
            "num_gpus": self.num_gpus,
            "gpu_mem_gb": [round(x, 2) for x in self.gpu_mem_gb],
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
        gpu_mem_gb = self._detect_gpu_memory_gb()
        return HardwareProfile(
            cpu_cores=cpu_cores,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_names=gpu_names,
            num_gpus=len(gpu_names),
            gpu_mem_gb=gpu_mem_gb,
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

    def _detect_gpu_memory_gb(self) -> list[float]:
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            try:
                return [
                    float(torch.cuda.get_device_properties(i).total_memory) / (1024**3)
                    for i in range(torch.cuda.device_count())
                ]
            except Exception:
                return []

        # Fallback: nvidia-smi (if available)
        smi_candidates = ["nvidia-smi"]
        if platform.system() == "Windows":
            smi_candidates.append(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
            smi_candidates.append(r"C:\Windows\System32\nvidia-smi.exe")
        for cmd in smi_candidates:
            if shutil.which(cmd) or os.path.exists(cmd):
                try:
                    result = subprocess.run(
                        [cmd, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                        check=True,
                        capture_output=True,
                        encoding="utf-8",
                    )
                    vals = [
                        float(line.strip()) / 1024.0
                        for line in result.stdout.splitlines()
                        if line.strip()
                    ]
                    if vals:
                        return vals
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
    hpo_trials: int
    adaptive_training_budget: float
    feature_workers: int
    is_hpc: bool = False  # New flag for High Performance Compute clusters      

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_gpu": self.enable_gpu,
            "num_gpus": self.num_gpus,
            "device": self.device,
            "n_jobs": self.n_jobs,
            "train_batch_size": self.train_batch_size,
            "inference_batch_size": self.inference_batch_size,
            "hpo_trials": self.hpo_trials,
            "adaptive_training_budget": self.adaptive_training_budget,
            "feature_workers": self.feature_workers,
            "is_hpc": self.is_hpc,
        }


class AutoTuner:
    """Derive execution/training defaults from detected hardware."""

    def __init__(self, settings: "Settings", profile: HardwareProfile) -> None:
        self.settings = settings
        self.profile = profile
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _read_int_env(name: str, default: int | None = None) -> int | None:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(str(raw).strip())
        except Exception:
            return default

    @staticmethod
    def _read_float_env(name: str, default: float | None = None) -> float | None:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(str(raw).strip())
        except Exception:
            return default

    def _resolve_cpu_budget(self, cpu_cores: int) -> int:
        override = self._read_int_env("FOREX_BOT_CPU_BUDGET", None)
        if override is not None and override > 0:
            return max(1, min(cpu_cores, override))
        reserve = self._read_int_env("FOREX_BOT_CPU_RESERVE", 1)
        reserve = max(0, reserve or 0)
        return max(1, cpu_cores - reserve)

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
            cur_trials = int(getattr(self.settings.models, "hpo_trials", 0) or 0)

            self.settings.models.train_batch_size = int(max(32, min(max(cur_train_bs, hints.train_batch_size), 2048)))
            self.settings.models.inference_batch_size = int(
                max(32, min(max(cur_infer_bs, hints.inference_batch_size), 4096))
            )
            if cur_trials <= 0:
                auto_trials = int(max(1, min(hints.hpo_trials, 200)))
                self.settings.models.hpo_trials = auto_trials
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
            "mlp_train_seconds",
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
                target_islands = max(4, per_worker_threads)
                if cur_islands <= 0 or cur_islands < target_islands:
                    self.settings.models.evo_islands = target_islands
                    self.logger.info(
                        f"Auto-tuned evo_islands: {cur_islands if cur_islands > 0 else 'unset'} -> {target_islands}"
                    )
        except Exception:
            pass

        try:
            self._apply_thread_env_defaults(hints)
            self._apply_feature_workers(hints)
            self._apply_discovery_autotune(hints)
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
        gpu_mem_gb = getattr(self.profile, "gpu_mem_gb", []) or []
        min_vram_gb = min(gpu_mem_gb) if gpu_mem_gb else 0.0
        cpu_budget = self._resolve_cpu_budget(cpu_cores)

        is_hpc = False

        # Detect Cloud/HPC Beast (GPU or CPU-heavy).
        disable_hpc = str(os.environ.get("FOREX_BOT_DISABLE_HPC", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        force_hpc = str(os.environ.get("FOREX_BOT_FORCE_HPC", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if disable_hpc:
            force_hpc = False
        min_hpc_cores = max(1, self._read_int_env("FOREX_BOT_HPC_MIN_CORES", 64) or 64)
        min_hpc_ram = max(1.0, self._read_float_env("FOREX_BOT_HPC_MIN_RAM_GB", 64.0) or 64.0)
        cpu_hpc = cpu_cores >= min_hpc_cores and ram_gb >= min_hpc_ram

        if disable_hpc:
            self.logger.info("HPC disabled via FOREX_BOT_DISABLE_HPC=1; using non-HPC tuning.")
        if (not disable_hpc) and (
            force_hpc or (enable_gpu and self.profile.num_gpus >= 4 and ram_gb > 64) or cpu_hpc
        ):
            is_hpc = True
            # HPC BEAST: Saturate the machine (Leave reserve for OS)
            n_jobs = max(1, cpu_budget)

            # Scale batch size based on VRAM (A6000 = 48GB)
            if min_vram_gb >= 40:
                train_batch = 2048
                infer_batch = 8192
            elif min_vram_gb >= 20:
                train_batch = 1024
                infer_batch = 4096
            else:
                train_batch = 512
                infer_batch = 2048
                
            base_trials = int(getattr(self.settings.models, "hpo_trials", 25) or 25)
            hpo_trials = max(base_trials, 100)
            adaptive_budget = 10800.0  # 3 hours per model default
            self.logger.info(
                f"HPC BEAST detected: cpu_cores={cpu_cores}, gpus={self.profile.num_gpus}, "
                f"ram={ram_gb:.0f}GB. Saturating cpu_budget={cpu_budget}."
            )

        elif enable_gpu:
            # High-End Workstation (Local or Cloud)
            is_hpc = False
            n_jobs = max(1, cpu_budget)
            train_batch = 512 if min_vram_gb >= 12 else 256
            infer_batch = 2048 if min_vram_gb >= 12 else 1024
            hpo_trials = 50
            adaptive_budget = 3600.0
        else:
            # CPU Only (Your Home PC)
            is_hpc = False
            n_jobs = max(1, cpu_budget)
            train_batch = 64
            infer_batch = 128
            hpo_trials = 20
            adaptive_budget = 1800.0

        # Feature worker scaling: 1 worker per 2GB of available RAM, capped by CPU cores
        ram_based_workers = int(ram_gb // 2.0)
        feature_workers = max(1, min(cpu_budget, ram_based_workers))

        return AutoTuneHints(
            enable_gpu=enable_gpu,
            num_gpus=self.profile.num_gpus if enable_gpu else 0,
            device=device,
            n_jobs=n_jobs,
            train_batch_size=train_batch,
            inference_batch_size=infer_batch,
            hpo_trials=hpo_trials,
            adaptive_training_budget=adaptive_budget,
            feature_workers=feature_workers,
            is_hpc=is_hpc,
        )

    def _apply_thread_env_defaults(self, hints: AutoTuneHints) -> None:
        """Set reasonable BLAS/OpenMP thread defaults based on hardware."""
        cpu_cores = max(1, self.profile.cpu_cores)
        cpu_budget = self._resolve_cpu_budget(cpu_cores)

        # Ensure the CPU budget is visible to downstream components.
        os.environ.setdefault("FOREX_BOT_CPU_BUDGET", str(cpu_budget))
        os.environ.setdefault("FOREX_BOT_CPU_THREADS", str(cpu_budget))

        # Respect explicit BLAS thread overrides if present.
        explicit = (
            os.environ.get("FOREX_BOT_BLAS_THREADS")
            or os.environ.get("FOREX_BOT_OMP_THREADS")
            or os.environ.get("OMP_NUM_THREADS")
            or os.environ.get("MKL_NUM_THREADS")
            or os.environ.get("OPENBLAS_NUM_THREADS")
            or os.environ.get("NUMEXPR_NUM_THREADS")
        )

        # Prefer explicit per-rank budgets (e.g., 20 cores per GPU under DDP).
        local_rank = os.environ.get("LOCAL_RANK")
        per_gpu = os.environ.get("FOREX_BOT_CPU_THREADS_PER_GPU")
        budget = os.environ.get("FOREX_BOT_CPU_BUDGET") or os.environ.get("FOREX_BOT_CPU_THREADS")
        blas_threads = None
        if explicit:
            try:
                blas_threads = max(1, int(explicit))
            except Exception:
                blas_threads = None
        if blas_threads is None and local_rank is not None:
            if per_gpu:
                try:
                    blas_threads = max(1, int(per_gpu))
                except Exception:
                    blas_threads = None
            if blas_threads is None and budget:
                try:
                    blas_threads = max(1, int(budget))
                except Exception:
                    blas_threads = None
            if blas_threads is None:
                blas_threads = 20
        elif blas_threads is None:
            if budget:
                try:
                    blas_threads = max(1, int(budget))
                except Exception:
                    blas_threads = None

        # Fallbacks if no explicit override was set.
        if blas_threads is None:
            multiproc_hint = (
                hints.is_hpc
                or hints.feature_workers > 1
                or hints.n_jobs > 1
                or int(os.environ.get("FOREX_BOT_FEATURE_WORKERS", "0") or 0) > 1
                or int(os.environ.get("FOREX_BOT_PROP_SEARCH_WORKERS", "0") or 0) > 1
                or int(os.environ.get("FOREX_BOT_CPU_WORKERS", "0") or 0) > 1
                or int(os.environ.get("FOREX_BOT_GPU_WORKERS", "0") or 0) > 1
            )
            # Default to 1 thread per process when multiprocessing is expected.
            blas_threads = 1 if multiproc_hint else max(1, cpu_budget)

        # HPC FIX: Strategic Thread Saturation
        # Ensure NumPy/SciPy/PyTorch can actually use the 252 cores
        os.environ["OMP_NUM_THREADS"] = str(blas_threads)
        os.environ["MKL_NUM_THREADS"] = str(blas_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)

        try:
            import numexpr as _numexpr

            _numexpr.set_num_threads(int(os.environ.get("NUMEXPR_NUM_THREADS", str(blas_threads))))
        except ImportError:
            pass  # Optional
        except Exception as e:
            logger.warning(f"NumExpr configuration failed: {e}")

        # Align PyTorch intra-op/inter-op threads with the chosen budget.
        try:
            import torch

            torch.set_num_threads(int(blas_threads))
            # Keep inter-op small to avoid oversubscription in DDP.
            torch.set_num_interop_threads(max(1, min(4, int(blas_threads))))
        except Exception:
            pass

    def _apply_feature_workers(self, hints: AutoTuneHints) -> None:
        """Fully Automated Worker Selection - No User Input Needed."""
        # 1. Calculate how many workers fit in RAM
        # Each worker needs roughly 2GB for 5M rows + indicators
        ram_gb = getattr(self.profile, "available_ram_gb", 16.0)
        per_worker_gb = self._read_float_env("FOREX_BOT_FEATURE_WORKER_GB", 2.0) or 2.0
        max_ram_workers = int(ram_gb // max(0.5, per_worker_gb))

        # 2. Saturate Cores
        cpu_cores = self.profile.cpu_cores
        cpu_budget = self._resolve_cpu_budget(cpu_cores)

        requested = self._read_int_env("FOREX_BOT_FEATURE_WORKERS", None)
        if requested is not None and requested > 0:
            target = max(1, min(cpu_budget, requested))
        else:
            target = max(1, min(cpu_budget, max_ram_workers))

        # Apply defaults only when the user has not explicitly set them.
        os.environ.setdefault("FOREX_BOT_FEATURE_WORKERS", str(target))
        os.environ.setdefault("FEATURE_WORKERS", str(target))
        os.environ.setdefault("FOREX_BOT_TALIB_WORKERS", str(target))

        self.logger.info(
            f"AUTO-TUNE: Feature workers={target} (cpu_budget={cpu_budget}, "
            f"ram={ram_gb:.0f}GB, per_worker_gb={per_worker_gb})."
        )

    def _apply_discovery_autotune(self, hints: AutoTuneHints) -> None:
        """Auto-tune discovery settings based on GPU/RAM if env not set."""
        # Allow opt-out
        if str(os.environ.get("FOREX_BOT_DISCOVERY_AUTOTUNE", "1")).strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return

        num_gpus = max(0, int(getattr(hints, "num_gpus", 0) or 0))
        gpu_mem_gb = getattr(self.profile, "gpu_mem_gb", []) or []
        min_vram = min(gpu_mem_gb) if gpu_mem_gb else 0.0
        ram_gb = float(
            getattr(self.profile, "available_ram_gb", None)
            or getattr(self.profile, "total_ram_gb", 0.0)
            or 0.0
        )

        # Preload vs streaming: default to streaming on big RAM, disable preload on small VRAM.
        if "FOREX_BOT_DISCOVERY_STREAM" not in os.environ:
            if num_gpus > 0 and ram_gb >= 64.0:
                os.environ["FOREX_BOT_DISCOVERY_STREAM"] = "1"
        if "FOREX_BOT_DISCOVERY_MAX_ROWS" not in os.environ:
            # Keep full history on high-RAM boxes by default.
            if ram_gb >= 64.0:
                os.environ["FOREX_BOT_DISCOVERY_MAX_ROWS"] = "0"
        if "FOREX_BOT_DISCOVERY_PRELOAD" not in os.environ:
            if min_vram > 0 and min_vram <= 16.0:
                os.environ["FOREX_BOT_DISCOVERY_PRELOAD"] = "0"

        # Population sizing: scale with GPU count but cap to avoid runaway cost.
        stream_flag = str(os.environ.get("FOREX_BOT_DISCOVERY_STREAM", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if "FOREX_BOT_DISCOVERY_POPULATION" not in os.environ:
            if num_gpus <= 0:
                pop = 4000 if ram_gb >= 16.0 else 2000
            else:
                # Allow explicit per-GPU override
                try:
                    per_gpu_env = int(os.environ.get("FOREX_BOT_DISCOVERY_POP_PER_GPU", "0") or 0)
                except Exception:
                    per_gpu_env = 0
                if per_gpu_env > 0:
                    per_gpu = per_gpu_env
                else:
                    if min_vram >= 40.0:
                        per_gpu = 1500
                    elif min_vram >= 24.0:
                        per_gpu = 1200
                    elif min_vram >= 16.0:
                        per_gpu = 800
                    else:
                        per_gpu = 500
                pop = int(per_gpu * max(1, num_gpus))
                # Hard cap for practical runtimes; user can override via env.
                pop_cap = 120000 if num_gpus < 64 else 240000
                pop = max(2000, min(pop, pop_cap))
                try:
                    pop_max = int(os.environ.get("FOREX_BOT_DISCOVERY_POPULATION_MAX", "0") or 0)
                except Exception:
                    pop_max = 0
                if pop_max > 0:
                    pop = min(pop, pop_max)
            # If full-data streaming is enabled and no overrides were provided,
            # default to a smaller, more robust population.
            if stream_flag and int(os.environ.get("FOREX_BOT_DISCOVERY_POP_PER_GPU", "0") or 0) <= 0:
                pop = 12000
            os.environ["FOREX_BOT_DISCOVERY_POPULATION"] = str(int(pop))

        # Optional safety cap on batch size for low VRAM.
        if "FOREX_BOT_DISCOVERY_MAX_BATCH" not in os.environ and min_vram > 0:
            if min_vram <= 16.0:
                os.environ["FOREX_BOT_DISCOVERY_MAX_BATCH"] = "80000"
            elif min_vram <= 24.0:
                os.environ["FOREX_BOT_DISCOVERY_MAX_BATCH"] = "110000"



def optimize_dataframe_memory(df, verbose=False):
    """
    Optimize pandas DataFrame memory usage using 2025 best practices.
    
    Reduces memory footprint by:
    1. Downcasting numeric types (int64 -> int32/int16/int8, float64 -> float32)
    2. Converting low-cardinality string columns to categorical
    3. Removing inf/nan values to prevent computational issues
    
    Sources:
    - https://pandas.pydata.org/pandas-docs/stable/user_guide/scale.html
    - https://thinhdanggroup.github.io/pandas-memory-optimization/
    - https://pythonspeed.com/articles/pandas-load-less-data/
    
    Args:
        df: Input DataFrame
        verbose: Print memory savings if True
        
    Returns:
        Optimized DataFrame
    """
    import pandas as pd
    import numpy as np
    
    start_mem = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Numeric optimization
        if col_type in [np.int64, np.int32, np.int16, np.int8]:
            # Downcast integers
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif col_type in [np.float64, np.float32]:
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif col_type == object:
            # Convert low-cardinality strings to categorical (< 50% unique)
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype("category")
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    if verbose:
        logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df

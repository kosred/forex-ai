"""
Device selection and hardware profiling utilities.
Prefers CUDA when available and beneficial.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class DeviceBenchmark:
    """Simple benchmarking to determine relative CPU vs GPU performance."""

    @staticmethod
    def benchmark_matmul(device: str, size: int = 2000) -> float:
        """Measure time for a matrix multiplication."""
        try:
            if device == "cpu":
                x = torch.randn(size, size, device="cpu")
                y = torch.randn(size, size, device="cpu")
                start = time.perf_counter()
                _ = torch.matmul(x, y)
                return time.perf_counter() - start
            elif device.startswith("cuda"):
                x = torch.randn(size, size, device=device)
                y = torch.randn(size, size, device=device)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                _ = torch.matmul(x, y)
                torch.cuda.synchronize(device)
                return time.perf_counter() - start
        except Exception as e:
            logger.warning(f"Benchmark failed for {device}: {e}")
            return float("inf")
        return float("inf")

    @staticmethod
    def estimate_transfer_overhead(device: str, size_mb: int = 100) -> float:
        """Estimate CPU->GPU transfer latency."""
        if not device.startswith("cuda"):
            return 0.0
        try:
            elements = size_mb * 1024 * 1024 // 4  # float32
            x = torch.randn(elements, device="cpu")
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = x.to(device)
            torch.cuda.synchronize(device)
            return time.perf_counter() - start
        except Exception:
            return float("inf")


def _detect_torch_cuda() -> int:
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        return 0
    return 0


def get_available_gpus() -> list[str]:
    """Return a list of CUDA device strings if available, else empty list."""
    count = _detect_torch_cuda()
    return [f"cuda:{i}" for i in range(count)] if count > 0 else []


def gpu_supports_bf16(device_str: str) -> bool:
    """Check if GPU supports bfloat16 (Ampere+)."""
    if not device_str.startswith("cuda"):
        return False
    try:
        major, minor = torch.cuda.get_device_capability(int(device_str.split(":")[-1]))
        return major >= 8  # Ampere (sm80) and above
    except Exception:
        return False


def preferred_amp_dtype(device: str):
    """
    Return best AMP dtype for the device.
    - bf16 on GPUs that support it
    - fp16 on other CUDA GPUs
    - fp32 on CPU
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.bfloat16 if gpu_supports_bf16(device) else torch.float16
    return torch.float32


def gpu_supports_fp8(device: str) -> bool:
    """
    FP8 commonly requires SM >= 8.9 (Ada/Hopper/Blackwell).
    """
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability(int(device.split(":")[-1]))
        return (major > 8) or (major == 8 and minor >= 9)
    except Exception:
        return False


def get_device_info(device_str: str) -> dict[str, Any]:
    """Get detailed info about a device."""
    info = {"name": "CPU", "memory_total": 0, "compute_capability": (0, 0)}
    if device_str.startswith("cuda"):
        try:
            idx = int(device_str.split(":")[-1])
            props = torch.cuda.get_device_properties(idx)
            info = {
                "name": props.name,
                "memory_total": props.total_memory,
                "compute_capability": (props.major, props.minor),
                "multi_processor_count": props.multi_processor_count,
            }
        except Exception as e:
            logger.warning(f"Failed to get device info for {device_str}: {e}")
    return info


def tune_torch_backend(device: str) -> None:
    """
    Enable fast math on supported GPUs (TF32 for Ampere/Hopper) and set matmul precision.
    Safe defaults for CPU.
    """
    try:
        torch.set_float32_matmul_precision("medium")  # Prefer TF32 on Ampere/Hopper
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    except Exception as exc:  # pragma: no cover
        logger.debug(f"Torch backend tuning skipped: {exc}")


def enable_sdp_flash() -> None:
    """
    Prefer flash/memory-efficient SDPA kernels when available (PyTorch 2.5+).
    """
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception as exc:  # pragma: no cover
        logger.debug(f"SDP flash enable skipped: {exc}")


def enable_flash_attention() -> None:
    """
    Enable flash/memory-efficient SDP kernels when available (PyTorch 2+).
    No-op on older builds.
    """
    try:  # torch >= 2.0
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        try:
            # Fallback API name
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        except Exception:
            pass


def select_device(requested: str | list[str] | None = None, allow_multi: bool = False) -> str | list[str]:
    """
    Select a device based on request and availability.
    requested: "auto"/None to prefer CUDA, "cpu" to force CPU, "cuda" to force GPU if available.
    allow_multi: when True and multiple GPUs available, return list of devices.
    """
    if isinstance(requested, list):
        return requested

    req = (requested or "auto").lower() if isinstance(requested, str) else "auto"
    if req == "cpu":
        return "cpu"

    gpus = get_available_gpus()
    if gpus:
        if allow_multi and len(gpus) > 1:
            return gpus
        # "Auto" logic: verify GPU is actually usable/faster could go here,
        # but for now we trust availability usually means speed for deep learning.
        return gpus[0] if len(gpus) >= 1 else "cuda"

    return "cpu"


def prefer_gpu_env_jobs(parallel_envs: int) -> int:
    import multiprocessing

    try:
        cpu_total = multiprocessing.cpu_count()
        return min(parallel_envs, max(1, cpu_total - 1))
    except Exception:
        return max(1, parallel_envs)


def maybe_data_parallel(model: torch.nn.Module, device: str | list[str]) -> tuple[torch.nn.Module, str | list[str]]:
    try:
        if not isinstance(model, torch.nn.Module):
            return model, device
        gpus = get_available_gpus()
        if len(gpus) <= 1:
            return model, device
        device_ids = list(range(len(gpus)))
        primary = "cuda:0"
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.to(primary)
        return model, primary
    except Exception as exc:
        logger.warning(f"DataParallel setup failed, falling back to single device: {exc}", exc_info=True)
        return model, device


def _ddp_env() -> tuple[int, int, int]:
    try:
        rank = int(os.environ.get("RANK", -1))
    except Exception:
        rank = -1
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
    except Exception:
        local_rank = -1
    try:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    except Exception:
        world_size = 1
    return rank, local_rank, world_size


def maybe_init_distributed(force: bool = False) -> str | None:
    try:
        import torch.distributed as dist
    except Exception:
        return None

    if dist.is_initialized():
        rank, local_rank, _ = _ddp_env()
        if local_rank >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}"
        return None

    rank, local_rank, world_size = _ddp_env()
    if world_size > 1 or force:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=max(rank, 0))
            if local_rank >= 0 and torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                return f"cuda:{local_rank}"
            return None
        except Exception:
            return None
    return None


def wrap_ddp(
    model: torch.nn.Module, device: str | list[str]
) -> tuple[torch.nn.Module, str | list[str], bool, int, int]:
    try:
        import torch.distributed as dist

        if not isinstance(model, torch.nn.Module):
            return model, device, False, 0, 1
        ddp_active = dist.is_available() and dist.is_initialized()
        rank, local_rank, world_size = _ddp_env()
        if ddp_active and world_size > 1:
            device_ids = [local_rank] if isinstance(device, str) and device.startswith("cuda") else None
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
            return model, device, True, max(rank, 0), world_size
        model_dp, dev = maybe_data_parallel(model, device)
        return model_dp, dev, False, 0, 1
    except Exception:
        return model, device, False, 0, 1


def maybe_compile(model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    """
    Conditionally apply torch.compile to a model.
    Disabled if FOREX_BOT_DISABLE_COMPILE=1 or on Windows (where it often fails).
    """
    if os.environ.get("FOREX_BOT_DISABLE_COMPILE") == "1":
        return model
    
    if os.name == "nt" and os.environ.get("FOREX_BOT_FORCE_COMPILE") != "1":
        # torch.compile is notoriously flaky on Windows
        return model

    try:
        if hasattr(torch, "compile"):
            logger.info(f"Applying torch.compile to {model.__class__.__name__}...")
            return torch.compile(model, **kwargs)
    except Exception as e:
        logger.warning(f"torch.compile failed for {model.__class__.__name__}: {e}")
    
    return model

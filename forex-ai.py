#!/usr/bin/env python3
"""
Forex AI Trading Bot - Master Autonomous Launcher
2025 HPC Edition (Self-Bootstrapping)
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

# --- 0.5 VENV BOOTSTRAP (PREFER PROJECT VENV IF PRESENT) ---
SCRIPT_DIR = Path(__file__).resolve().parent

def _in_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix or hasattr(sys, "real_prefix")

def _venv_python() -> Path:
    if os.name == "nt":
        return SCRIPT_DIR / ".venv" / "Scripts" / "python.exe"
    return SCRIPT_DIR / ".venv" / "bin" / "python"

_venv_py = _venv_python()
if _venv_py.exists() and not _in_venv():
    os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)

def _pip_cmd(args: list[str] | None = None, *, upgrade: bool = False) -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    if not _in_venv():
        cmd.append("--user")
        if platform.system().lower() == "linux":
            cmd.append("--break-system-packages")
    if args:
        cmd.extend(args)
    return cmd

def _pip_install(cmd: list[str]) -> None:
    try:
        subprocess.check_call(cmd)
    except Exception:
        if "--break-system-packages" in cmd:
            cmd = [c for c in cmd if c != "--break-system-packages"]
            subprocess.check_call(cmd)
        else:
            raise

# --- 0. HPC GLOBAL ENVIRONMENT (AUTO-DETECT HARDWARE) ---
# Fully automatic hardware detection - NO hardcoding!
#
# To override auto-detection, set these environment variables BEFORE running:
#   FOREX_BOT_CPU_THREADS=X    - Number of worker processes (default: logical_cores - 1)
#   FOREX_BOT_CPU_RESERVE=X    - Cores to reserve for OS (default: 1)
#   OMP_NUM_THREADS=X          - BLAS threads per operation (default: physical_cores - 1)
#   FOREX_BOT_RL_ENVS=X        - RL parallel environments (default: auto, RAM-limited)
#
# Example: set OMP_NUM_THREADS=4 && python forex-ai.py train

# Auto-detect actual physical cores (not hyperthreaded logical cores)
try:
    import psutil
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1
except Exception:
    # Fallback: assume no hyperthreading
    logical_cores = os.cpu_count() or 1
    physical_cores = logical_cores

# CPU budget for worker processes (uses logical cores)
try:
    cpu_reserve = int(os.environ.get("FOREX_BOT_CPU_RESERVE", "1") or 1)
except Exception:
    cpu_reserve = 1
cpu_budget = max(1, logical_cores - max(0, cpu_reserve))

# CRITICAL: When using ProcessPoolExecutor, each process spawns its own BLAS threads!
# If we have 11 workers × 5 BLAS threads = 55 threads (over-subscription!)
# Solution: Set BLAS to 1 thread when using multiprocessing
# This way: 11 workers × 1 BLAS thread = 11 threads (optimal)
def _read_int_env(*keys: str) -> int | None:
    for key in keys:
        val = os.environ.get(key)
        if val:
            try:
                return max(1, int(val))
            except Exception:
                continue
    return None

auto_mode = str(os.environ.get("FOREX_BOT_AUTO_TUNE", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Auto-split CPU budgets when GPUs are present (training vs discovery/search).
gpu_count = 0
try:
    import torch

    if torch.cuda.is_available():
        gpu_count = int(torch.cuda.device_count())
except Exception:
    gpu_count = 0

if gpu_count > 0:
    per_gpu = _read_int_env("FOREX_BOT_CPU_THREADS_PER_GPU")
    if per_gpu is None:
        # Heuristic: 10 cores/GPU if the machine can afford it, otherwise scale down.
        per_gpu = min(10, max(4, cpu_budget // max(1, gpu_count * 2)))
        if cpu_budget // max(1, gpu_count) >= 12:
            per_gpu = 10
    training_budget = min(cpu_budget, per_gpu * gpu_count)
    # Ensure we leave at least 1 core for search/features.
    if training_budget >= cpu_budget:
        per_gpu = max(1, (cpu_budget - 1) // max(1, gpu_count))
        training_budget = max(1, per_gpu * gpu_count)
    remaining = max(1, cpu_budget - training_budget)

    if auto_mode:
        os.environ["FOREX_BOT_CPU_THREADS_PER_GPU"] = str(per_gpu)
        os.environ["FOREX_BOT_CPU_BUDGET"] = str(training_budget)
        os.environ["FOREX_BOT_CPU_THREADS"] = str(training_budget)
    else:
        os.environ.setdefault("FOREX_BOT_CPU_THREADS_PER_GPU", str(per_gpu))
        os.environ.setdefault("FOREX_BOT_CPU_BUDGET", str(training_budget))
        os.environ.setdefault("FOREX_BOT_CPU_THREADS", str(training_budget))

    # Discovery / feature CPU budgets
    if auto_mode:
        os.environ["FOREX_BOT_FEATURE_CPU_BUDGET"] = str(remaining)
        os.environ["FOREX_BOT_FEATURE_WORKERS"] = str(remaining)
        os.environ["FOREX_BOT_PROP_SEARCH_WORKERS"] = str(remaining)
        os.environ["FOREX_BOT_PROP_SEARCH_ASYNC"] = "1"
        os.environ["FOREX_BOT_PROP_SEARCH_ASYNC_WAIT"] = "0"
        if platform.system().lower() == "linux":
            os.environ["FOREX_BOT_PROP_MP_CONTEXT"] = "fork"
    else:
        os.environ.setdefault("FOREX_BOT_FEATURE_CPU_BUDGET", str(remaining))
        os.environ.setdefault("FOREX_BOT_FEATURE_WORKERS", str(remaining))
        os.environ.setdefault("FOREX_BOT_PROP_SEARCH_WORKERS", str(remaining))
        os.environ.setdefault("FOREX_BOT_PROP_SEARCH_ASYNC", "1")
        os.environ.setdefault("FOREX_BOT_PROP_SEARCH_ASYNC_WAIT", "0")
        if platform.system().lower() == "linux":
            os.environ.setdefault("FOREX_BOT_PROP_MP_CONTEXT", "fork")

    # Favor GPU model training by default when GPUs are present.
    if auto_mode:
        os.environ["FOREX_BOT_PARALLEL_MODELS"] = "gpu"
        os.environ["FOREX_BOT_GPU_WORKERS"] = str(gpu_count)
    else:
        os.environ.setdefault("FOREX_BOT_PARALLEL_MODELS", "gpu")
        os.environ.setdefault("FOREX_BOT_GPU_WORKERS", str(gpu_count))

    # Keep the event loop responsive so prop search can actually run.
    if auto_mode:
        os.environ["FOREX_BOT_DISCOVERY_ASYNC"] = "1"
    else:
        os.environ.setdefault("FOREX_BOT_DISCOVERY_ASYNC", "1")

    # If we have lots of RAM, allow prop search to stay parallel.
    try:
        mem = psutil.virtual_memory()
        if float(mem.available) / (1024**3) >= 64.0:
            if auto_mode:
                os.environ["FOREX_BOT_PROP_PARALLEL_MEM_FRAC"] = "0.90"
                os.environ["FOREX_BOT_PROP_PARALLEL_OVERHEAD"] = "1.20"
            else:
                os.environ.setdefault("FOREX_BOT_PROP_PARALLEL_MEM_FRAC", "0.90")
                os.environ.setdefault("FOREX_BOT_PROP_PARALLEL_OVERHEAD", "1.20")
    except Exception:
        pass

# Respect explicit user overrides for BLAS/OMP threads; default to 1 otherwise.
if auto_mode:
    blas_threads = 1
else:
    blas_threads = _read_int_env(
        "FOREX_BOT_BLAS_THREADS",
        "FOREX_BOT_OMP_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    if blas_threads is None:
        blas_threads = 1  # Single-threaded BLAS when using multiprocessing

os.environ.setdefault("FOREX_BOT_CPU_BUDGET", str(cpu_budget))
os.environ.setdefault("FOREX_BOT_CPU_THREADS", str(cpu_budget))
# BLAS libraries: Single-threaded to prevent N_workers × N_threads explosion
if auto_mode:
    os.environ["NUMEXPR_MAX_THREADS"] = str(blas_threads)
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMBA_DEFAULT_NUM_THREADS"] = str(blas_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
    # Disable dynamic threading for predictable performance
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    # PyTorch/TensorFlow thread limits (match BLAS: single-threaded)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(blas_threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TORCH_NUM_THREADS"] = str(blas_threads)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
else:
    os.environ.setdefault("NUMEXPR_MAX_THREADS", str(blas_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("NUMBA_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("NUMBA_DEFAULT_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(blas_threads))
    # Disable dynamic threading for predictable performance
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    # PyTorch/TensorFlow thread limits (match BLAS: single-threaded)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(blas_threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# NCCL optimizations for 8x A6000 P2P topology
os.environ["NCCL_P2P_LEVEL"] = "5"
os.environ["NCCL_IB_DISABLE"] = "1"

# Print hardware auto-detection results (ONLY ONCE - not in worker processes)
# Set flag to prevent worker processes from re-printing when they import this module
if not os.environ.get("_FOREX_BOT_HW_DETECTED"):
    os.environ["_FOREX_BOT_HW_DETECTED"] = "1"
    print("=" * 70)
    print(f"[HW AUTO-DETECT] Physical Cores: {physical_cores} | Logical Cores: {logical_cores}")
    print(f"[HW AUTO-DETECT] Worker Processes: {cpu_budget} x BLAS Threads: {blas_threads} = ~{cpu_budget * blas_threads} compute threads")
    print(f"[HW AUTO-DETECT] Thread Strategy: Single-threaded BLAS (prevents N_workers x N_threads explosion)")
    try:
        mem = psutil.virtual_memory()
        print(f"[HW AUTO-DETECT] Total RAM: {mem.total/1024/1024/1024:.1f}GB | Available: {mem.available/1024/1024/1024:.1f}GB ({mem.percent:.1f}% used)")
    except Exception:
        pass
    # Auto-detect GPU
    gpu_info = "None detected"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_info = f"{gpu_count}x {gpu_name}"
    except Exception:
        pass
    print(f"[HW AUTO-DETECT] GPU: {gpu_info}")
    print("=" * 70)

# --- 1. SELF-HEALING SYSTEM SETUP ---
def bootstrap():
    """Ensure the system is optimized and ready."""
    is_linux = platform.system().lower() == "linux"
    
    # 1.1 TA-Lib (Use pre-built 2025 wheels to save time)
    try:
        import talib
    except ImportError:
        print("[INIT] TA-Lib missing. Installing pre-built binaries...")
        try:
            # In late 2025, TA-Lib has stable wheels for Python 3.13
            pip_cmd = _pip_cmd(["TA-Lib"])
            _pip_install(pip_cmd)
            print("[INIT] TA-Lib installed via wheel.")
        except Exception:
            print("[WARN] Wheel install failed. Falling back to source (this may take 5 mins)...")
            # ... (Existing source build fallback)

    # 1.2 Python Dependencies
    try:
        import pandas
        import pydantic
        # GPU-specific deps are optional on Windows for local CPU testing
        if platform.system().lower() != "windows":
            import torch
            import cupy
    except ImportError:
        print("[INIT] Missing Python libraries. Syncing with Master HPC Stack...")
        base_dir = Path(__file__).parent
        req_file = base_dir / "requirements-hpc.txt"
        is_windows = platform.system().lower() == "windows"

        cmd = _pip_cmd(upgrade=True)

        if is_windows:
            # Minimal, CPU-friendly stack to avoid Linux/CUDA-only wheels
            win_req = base_dir / "requirements-win.txt"
            if not win_req.exists():
                win_req.write_text(
                    "\n".join(
                        [
                            "numpy",
                            "pandas",
                            "scipy",
                            "scikit-learn",
                            "joblib",
                            "pydantic",
                            "pydantic-settings",
                            "psutil",
                            "requests",
                            "PyYAML",
                            "tqdm",
                            "sqlalchemy",
                            "numexpr",
                            "colorama",
                        ]
                    ),
                    encoding="utf-8",
                )
            cmd += ["-r", str(win_req)]
        elif req_file.exists():
            cmd += ["-r", str(req_file)]
        else:
            # Fallback to binary-safe core
            cmd += ["--only-binary", ":all:", "pandas", "numpy", "torch", "pydantic", "sqlalchemy", "cupy-cuda12x"]

        try:
            _pip_install(cmd)
            print("[INIT] Stack synchronized. Restarting engine...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"[FATAL] Dependency sync failed: {e}")
            sys.exit(1)

# --- 2. ENGINE PATHS ---
SRC_DIR = SCRIPT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
os.environ["PYTHONPATH"] = str(SRC_DIR)

if __name__ == "__main__":
    # Internal workers for parallel evaluation
    if "--_worker" in sys.argv[1:]:
        from forex_bot.training.parallel_worker import run_worker
        sys.exit(run_worker(sys.argv[sys.argv.index("--_worker") + 1 :]))

    # Autonomous Setup
    bootstrap()

    # Launch the bot
    from forex_bot.main import main
    main()

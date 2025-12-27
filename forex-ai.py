#!/usr/bin/env python3
"""
Forex AI Trading Bot - Universal Entry Point

Usage:
    ./forex-ai.py              # Auto-detect mode (train if no models, else run)
    ./forex-ai.py --train      # Force training mode
    ./forex-ai.py --run        # Force live trading mode
    ./forex-ai.py --verbose    # Enable debug logging

Works on Windows, Linux, Mac with zero configuration.
Auto-detects HPC environments (CUDA, multiple GPUs) and optimizes accordingly.
"""

import os
import sys
import subprocess
from pathlib import Path

def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _ensure_global_python() -> None:
    """Enforce global Python (no venv/conda) to avoid stale env junk."""
    in_venv = bool(os.environ.get("VIRTUAL_ENV")) or (
        hasattr(sys, "real_prefix") or sys.base_prefix != sys.prefix
    )
    if not in_venv:
        return
    if _is_truthy(os.environ.get("FOREX_BOT_ALLOW_VENV")):
        return
    if _is_truthy(os.environ.get("FOREX_BOT_VENV_REEXEC")):
        print("[FATAL] Virtual env detected and re-exec already attempted.")
        print("        Please run with global/system Python (no .venv).")
        raise SystemExit(2)

    base_prefix = Path(sys.base_prefix)
    if os.name == "nt":
        candidates = [base_prefix / "python.exe"]
    else:
        candidates = [base_prefix / "bin" / "python3", base_prefix / "bin" / "python"]

    base_python = next((p for p in candidates if p.exists()), None)
    if base_python is None:
        print("[FATAL] Virtual env detected but global Python not found.")
        print("        Please run with system Python or set FOREX_BOT_ALLOW_VENV=1.")
        raise SystemExit(2)

    # Prevent infinite recursion and re-exec using the base interpreter.
    os.environ["FOREX_BOT_VENV_REEXEC"] = "1"
    print(f"[INIT] Virtual env detected. Re-launching with {base_python}...")
    os.execv(str(base_python), [str(base_python), *sys.argv])


# --- 1. Path & Environment Bootstrap ---
# 2025 ANTI-VENV POLICY: Purge environment variables that force virtual environments
for env_var in ["VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"]:
    if env_var in os.environ:
        del os.environ[env_var]

_ensure_global_python()

# Fix Python path for direct execution from source tree
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"

# FORCE DOMINANCE: Ensure local /src is ALWAYS index 0, overriding global installs
sys.path.insert(0, str(SRC_DIR))
os.environ["PYTHONPATH"] = str(SRC_DIR)

# Ensure current working directory is project root
os.chdir(str(SCRIPT_DIR))

# --- 2. HPC / Stability Tuning ---
# Disable torch.compile to prevent nvvmAddNVVMContainerToProgram/JIT errors on some drivers
os.environ.setdefault("FOREX_BOT_DISABLE_COMPILE", "1")

# Fix PyTorch Memory Fragmentation (Modern Config)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Tuning for 16GB+ VRAM Cards (A4000/A6000)
os.environ.setdefault("FOREX_BOT_GLOBAL_POOL_MEM_FRAC", "0.20")
os.environ.setdefault("FOREX_BOT_PARALLEL_MODELS", "auto")
if _is_truthy(os.environ.get("FOREX_BOT_DISCOVERY_STREAM")):
    os.environ.setdefault("FOREX_BOT_FULL_DATA", "1")

# Linux-specific: Ensure system CUDA libraries take precedence if present
if sys.platform.startswith("linux"):
    cuda_lib = "/usr/local/cuda/targets/x86_64-linux/lib"
    if os.path.exists(cuda_lib):
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if cuda_lib not in current_ld:
            os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}:{current_ld}"

if __name__ == "__main__":
    if "--_worker" in sys.argv[1:]:
        import logging

        worker_idx = sys.argv.index("--_worker")
        worker_argv = sys.argv[worker_idx + 1 :]
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        from forex_bot.training.parallel_worker import run_worker

        sys.exit(run_worker(worker_argv))

    print("=" * 60)
    try:
        print("?? FOREX AI TRADING BOT (Universal Launcher)")
    except UnicodeEncodeError:
        print("FOREX AI TRADING BOT (Universal Launcher)")
    print("=" * 60)

    # --- 3. Dependency Check ---
    try:
        if os.environ.get("FOREX_BOT_SKIP_DEPS", "0") != "1":
            print("[INIT] Checking and auto-installing dependencies (Global Mode)...", flush=True)
            from forex_bot.core.deps import ensure_dependencies

            ensure_dependencies()
            
            # 2025 SMART LINKING: Force user-level editable install.
            # This ensures that even in global mode, the 'forex_bot' name 
            # points directly to our local /src directory.
            print("[INIT] Syncing global environment with local source code...", flush=True)
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "--no-deps", "-e", "."], cwd=SCRIPT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("[INIT] Global environment synced (pip install --user -e .).")
            except Exception as e:
                print(f"[WARN] Local link failed (non-critical): {e}")
        else:
            print("[INIT] Skipping dependency bootstrap (FOREX_BOT_SKIP_DEPS=1).", flush=True)
    except Exception as dep_err:
        print(f"[WARN] Dependency bootstrap failed: {dep_err}", file=sys.stderr)

    # --- 4. Hardware Detection & DDP Launch ---
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            cpu_cores = os.cpu_count() or 1
            
            print(f"[INIT] Detected {gpu_count} GPUs. Tuning for performance...")
            
            # Use all visible GPUs unless user overrides.
            os.environ.setdefault("FOREX_BOT_MAX_GPUS", str(gpu_count))
            os.environ.setdefault("SYSTEM__NUM_GPUS", str(gpu_count))
            os.environ.setdefault("SYSTEM__ENABLE_GPU_PREFERENCE", "auto")
            os.environ.setdefault("FOREX_BOT_TREE_DEVICE", "auto")
            
            # Cap threads per worker to avoid CPU thrash
            threads_per_worker = max(2, min(16, cpu_cores // max(1, gpu_count)))
            os.environ.setdefault("FOREX_BOT_CPU_THREADS", str(threads_per_worker))
            
            # Feature workers are auto-tuned later in core.system.AutoTuner.

            # Auto-DDP: If > 1 GPU and running on Linux, relaunch with torchrun
            # We skip this if already launched or on Windows (DDP issues)
            if (
                os.name != "nt"
                and gpu_count > 1
                and os.environ.get("FOREX_BOT_DDP_LAUNCHED") != "1"
                and "--train" in sys.argv # Only use DDP for training
            ):
                print(f"[HPC] Relaunching with torch.distributed.run across {gpu_count} GPUs...", flush=True)
                env = os.environ.copy()
                env["FOREX_BOT_DDP_LAUNCHED"] = "1"
                # OMP_NUM_THREADS=1 prevents CPU oversubscription in DDP
                env["OMP_NUM_THREADS"] = "1" 
                
                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    f"--nproc_per_node={gpu_count}",
                    __file__,
                    *sys.argv[1:],
                ]
                subprocess.check_call(cmd, env=env)
                sys.exit(0)
    except Exception:
        pass

    # --- 5. Main Execution ---
    from forex_bot.main import _global_models_exist, main

    has_models = _global_models_exist()
    mode_desc = "LIVE TRADING (models exist)" if has_models else "TRAINING FIRST (no models)"
    if "--train" in sys.argv: mode_desc = "TRAINING (Forced)"
    if "--run" in sys.argv: mode_desc = "LIVE TRADING (Forced)"
    
    print(f"Mode: {mode_desc}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 60)

    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Bot stopped by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

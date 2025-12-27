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

# --- 0. HPC GLOBAL ENVIRONMENT (NO MANUAL EXPORTS NEEDED) ---
os.environ["NUMEXPR_MAX_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# NCCL optimizations for 8x A6000 P2P topology
os.environ["NCCL_P2P_LEVEL"] = "5"
os.environ["NCCL_IB_DISABLE"] = "1"

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
            subprocess.check_call([sys.executable, "-m", "pip", "install", "TA-Lib", "--user", "--break-system-packages"])
            print("[INIT] TA-Lib installed via wheel.")
        except Exception:
            print("[WARN] Wheel install failed. Falling back to source (this may take 5 mins)...")
            # ... (Existing source build fallback)

    # 1.2 Python Dependencies
    try:
        import pandas
        import torch
        import pydantic
        import cupy
    except ImportError:
        print("[INIT] Missing Python libraries. Syncing with Master HPC Stack...")
        req_file = Path(__file__).parent / "requirements-hpc.txt"
        
        cmd = [sys.executable, "-m", "pip", "install", "--user", "--upgrade", "--break-system-packages"]
        if req_file.exists():
            cmd += ["-r", str(req_file)]
        else:
            # Fallback to binary-safe core
            cmd += ["--only-binary", ":all:", "pandas", "numpy", "torch", "pydantic", "sqlalchemy", "cupy-cuda12x"]
            
        try:
            subprocess.check_call(cmd)
            print("[INIT] Stack synchronized. Restarting engine...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as e:
            print(f"[FATAL] Dependency sync failed: {e}")
            sys.exit(1)

# --- 2. ENGINE PATHS ---
SCRIPT_DIR = Path(__file__).resolve().parent
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
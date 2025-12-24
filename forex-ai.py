#!/usr/bin/env python3
"""
Forex AI Trading Bot - Universal Entry Point

Usage:
    ./forex-ai.py              # Auto-detect mode (train if no models, else run)
    ./forex-ai.py --train      # Force training mode
    ./forex-ai.py --run        # Force live trading mode
    ./forex-ai.py --verbose    # Enable debug logging

Works on Windows, Linux, Mac with zero configuration.
"""

import os
import sys
from pathlib import Path

# Fix Python path for direct execution from source tree
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
        print("?? FOREX AI TRADING BOT")
    except UnicodeEncodeError:
        print("FOREX AI TRADING BOT")  # Fallback for Windows terminals
    print("=" * 60)

    # Defer heavy imports (deps auto-install) to normal mode only.
    try:
        if os.environ.get("FOREX_BOT_SKIP_DEPS", "0") != "1":
            print("[INIT] Checking and auto-installing dependencies (Hardware-Aware)...", flush=True)
            from forex_bot.core.deps import ensure_dependencies

            ensure_dependencies()
        else:
            print("[INIT] Skipping dependency bootstrap (FOREX_BOT_SKIP_DEPS=1).", flush=True)
    except Exception as dep_err:
        print(f"[WARN] Dependency bootstrap failed: {dep_err}", file=sys.stderr)

    # Auto-launch DDP when multiple GPUs are present (Linux only) and not already launched.
    if os.name != "nt" and os.environ.get("FOREX_BOT_DDP_LAUNCHED") != "1":
        try:
            import torch
        except Exception:
            torch = None
        if torch is not None and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count and gpu_count > 1:
                import subprocess

                env = os.environ.copy()
                env["FOREX_BOT_DDP_LAUNCHED"] = "1"
                # torchrun will spawn one process per GPU.
                cmd = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    f"--nproc_per_node={gpu_count}",
                    __file__,
                    *sys.argv[1:],
                ]
                print(f"[INIT] Detected {gpu_count} GPUs. Relaunching under torchrun for DDP...", flush=True)
                subprocess.check_call(cmd, env=env)
                sys.exit(0)

    from forex_bot.main import _global_models_exist, main

    has_models = _global_models_exist()
    mode_desc = "LIVE TRADING (models exist)" if has_models else "TRAINING FIRST (no models)"
    print(f"Mode: AUTOMATIC - {mode_desc}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 60)

    # Delegate CLI parsing and logging setup to forex_bot.main
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

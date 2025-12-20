import argparse
import asyncio
import logging
import os
import platform
import signal
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "forex_bot"

try:
    print("[INIT] Checking and auto-installing dependencies (Hardware-Aware)...", flush=True)
    from forex_bot.core.deps import ensure_dependencies

    ensure_dependencies()
except Exception as dep_err:
    print(f"[WARN] Dependency bootstrap failed: {dep_err}", file=sys.stderr)

from forex_bot.core.config import Settings
from forex_bot.core.logging import setup_logging
from forex_bot.data.loader import DataLoader
from forex_bot.execution.bot import ForexBot
from forex_bot.models.device import maybe_init_distributed  # DDP Support


def _global_models_exist(models_dir: Path = Path("models")) -> bool:
    """
    Return True when at least one trained model is registered.

    Notes:
    - `active_models.pkl` can exist but be an empty list (e.g., after a failed run). In that case,
      treat it as "no models" so the bot retrains.
    - `models.joblib` is a best-effort convenience bundle and may be absent on some environments
      (e.g., when some models can't be pickled). We should still consider models "present" if the
      active list is non-empty.
    """
    try:
        p = models_dir / "active_models.pkl"
        bundle = models_dir / "models.joblib"

        try:
            import joblib

            if p.exists():
                active = joblib.load(p)
                if active:
                    return True
        except Exception:
            # If the file exists but can't be read, prefer retraining over assuming models are usable.
            return False

        # Fallback: consider the bundle if present and it contains models.
        if bundle.exists():
            try:
                import joblib

                payload = joblib.load(bundle)
                if isinstance(payload, dict) and payload.get("models"):
                    return True
            except Exception:
                return False
        return False
    except Exception:
        return False


def _export_onnx_from_saved_models(models_dir: Path) -> bool:
    """
    Export already-trained models to ONNX without retraining.

    Uses `run_summary.json` feature columns to create a shape-inference sample input.
    """
    logger = logging.getLogger(__name__)
    try:
        import json

        import numpy as np
        import pandas as pd

        from forex_bot.training.persistence_service import PersistenceService

        models_dir = Path(models_dir)
        onnx_manifest = models_dir / "onnx" / "export_manifest.joblib"
        if onnx_manifest.exists():
            logger.info(f"ONNX manifest already exists at {onnx_manifest}; skipping export.")
            return True

        summary_path = models_dir / "run_summary.json"
        if not summary_path.exists():
            logger.error(f"Cannot export ONNX: missing {summary_path}")
            return False

        try:
            summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error(f"Cannot export ONNX: failed to read {summary_path}: {exc}")
            return False

        cols = summary.get("feature_columns")
        if not isinstance(cols, list) or not cols:
            logger.error("Cannot export ONNX: `feature_columns` missing/empty in run_summary.json")
            return False

        sample = pd.DataFrame(np.zeros((1, len(cols)), dtype=np.float32), columns=[str(c) for c in cols])

        logs_dir = Path(os.environ.get("FOREX_BOT_LOGS_DIR", "logs"))
        ps = PersistenceService(models_dir=models_dir, logs_dir=logs_dir)
        models, blender = ps.load_models()
        if not models:
            logger.error("Cannot export ONNX: no models could be loaded from disk.")
            return False

        exported = ps.export_onnx(models, blender, sample)
        if exported:
            logger.info(f"ONNX export complete ({len(exported)} models).")
            return True

        logger.error("ONNX export produced no models.")
        return False
    except Exception as exc:
        logger.error(f"ONNX export failed: {exc}", exc_info=True)
        return False


async def _run_global_training(base_settings: Settings, symbols: list[str], stop_event: asyncio.Event) -> None:
    """Train a single global model across all symbols."""
    logger = logging.getLogger(__name__)
    if not symbols:
        logger.error("No symbols provided for global training.")
        return
    settings = deepcopy(base_settings)
    settings.system.symbol = symbols[0]
    try:
        bot = ForexBot(settings)
    except Exception:
        logger.error("Failed to initialize ForexBot:", exc_info=True)
        raise
    logger.info(f"[TRAIN-GLOBAL] Training one model across symbols: {symbols}")
    await bot.train_global(symbols=symbols, optimize=True, stop_event=stop_event)


async def _listen_for_escape(stop_event: asyncio.Event) -> None:
    """Listen for DOUBLE ESC key (Windows) to trigger graceful shutdown."""
    if sys.platform != "win32":
        return
    import msvcrt
    from datetime import datetime

    last_esc_time = None
    esc_window_seconds = 2.0  # Must press ESC twice within 2 seconds

    while not stop_event.is_set():
        try:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"\x1b",):  # ESC
                    now = datetime.now()

                    if last_esc_time is None:
                        last_esc_time = now
                        print(
                            "[INFO] ESC pressed once. Press ESC again within 2 seconds to shutdown gracefully...",
                            flush=True,
                        )
                    else:
                        elapsed = (now - last_esc_time).total_seconds()
                        if elapsed <= esc_window_seconds:
                            print("[INFO] DOUBLE ESC detected! Requesting graceful shutdown...", flush=True)
                            stop_event.set()
                            return
                        else:
                            last_esc_time = now
                            print(
                                "[INFO] ESC pressed (too slow). Press ESC again within 2 seconds to shutdown...",
                                flush=True,
                            )

            if last_esc_time and (datetime.now() - last_esc_time).total_seconds() > esc_window_seconds:
                last_esc_time = None

            await asyncio.sleep(0.1)
        except Exception:
            await asyncio.sleep(0.1)


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    """Cross-platform: SIGINT/SIGTERM set the stop_event for graceful shutdown."""
    logger = logging.getLogger(__name__)

    def _cancel_all_tasks():
        for task in asyncio.all_tasks(loop):
            if task is not asyncio.current_task(loop):
                task.cancel()

    def _handler(sig, frame):
        print(f"\n[INFO] Signal {sig} received. Requesting graceful shutdown...", flush=True)
        # Use call_soon_threadsafe to safely interact with the event loop from a signal handler
        stop_event.set()
        loop.call_soon_threadsafe(_cancel_all_tasks)

    # Windows Support
    if platform.system().lower().startswith("win"):
        try:
            # On Windows, we must use the standard signal module for SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)  # SIGTERM might not work typically on Win, but good practice
            if hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, _handler)  # Ctrl+Break in some terminals
            logger.info("Signal handlers installed for Windows (Ctrl+C).")
        except Exception as e:
            logger.warning(f"Windows signal handler setup failed: {e}")
        return

    # Linux/Mac Support (uvloop/standard)
    for sig_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig_name):
            try:
                sig = getattr(signal, sig_name)
                loop.add_signal_handler(sig, lambda sig=sig: _handler(sig, None))
                logger.info(f"Signal handler installed for {sig_name}.")
            except NotImplementedError:
                # Fallback if loop doesn't support add_signal_handler (e.g. some custom loops)
                logger.info(f"Loop doesn't support add_signal_handler for {sig_name}, falling back to signal.signal.")
                try:
                    signal.signal(getattr(signal, sig_name), _handler)
                except Exception:
                    pass
            except RuntimeError as e:
                logger.warning(f"Signal handler setup failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Forex Trading Bot - Fully Automatic")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Train models and exit (no live trading)")
    mode.add_argument("--run", action="store_true", help="Run live trading (skip global pre-train step)")
    parser.add_argument("--symbol", type=str, help="Override symbol from config (comma-separated for multiple)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export existing trained models to ONNX (no retrain), then exit.",
    )
    parser.add_argument(
        "--clean",
        choices=["cache", "models", "all"],
        help="Delete generated artifacts before starting (cache/models/tool caches).",
    )
    parser.add_argument("--clean-only", action="store_true", help="Run cleanup then exit.")
    return parser.parse_args()


def discover_available_symbols(data_dir: str) -> list[str]:
    """Scan data directory for available symbols (format: symbol=EURUSD)."""
    logger = logging.getLogger(__name__)
    p = Path(data_dir)
    if not p.exists():
        logger.warning(f"Data directory {data_dir} not found.")
        return []

    symbols = []
    for path in p.iterdir():
        if path.is_dir() and path.name.startswith("symbol="):
            sym = path.name.split("=")[1]
            symbols.append(sym)
    return sorted(set(symbols))


async def run_bot_instance(settings: Settings, stop_event: asyncio.Event):
    """Run a single bot instance for a specific symbol - auto-trains if no models exist."""
    logger = logging.getLogger(__name__)
    try:
        bot = ForexBot(settings)
        logger.info(f"[AUTO] Starting bot for {settings.system.symbol} (auto-trains if needed, then trades live)...")
        await bot.run(stop_event=stop_event)
    except Exception as e:
        logger.error(f"[ERROR] Bot instance for {settings.system.symbol} failed: {e}", exc_info=True)


async def main_async():
    # Initialize DDP if launched via torchrun (RANK/WORLD_SIZE env vars present)
    ddp_rank = maybe_init_distributed()
    
    args = parse_args()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    if ddp_rank:
        logger.info(f"?? Distributed Training Active (Local Rank: {ddp_rank})")

    if getattr(args, "clean", None):
        try:
            import clean_artifacts as cleaner

            if args.clean == "cache":
                cleaner.clean_artifacts(cache=True)
            elif args.clean == "models":
                cleaner.clean_artifacts(models=True)
            elif args.clean == "all":
                cleaner.clean_artifacts(
                    cache=True,
                    models=True,
                    ruff_cache=True,
                    pytest_cache=True,
                    catboost_info=True,
                    logs=True,
                    pycache=True,
                )
        except Exception as exc:
            logger.warning(f"Cleanup requested but failed: {exc}", exc_info=True)
        if getattr(args, "clean_only", False):
            return

    base_settings = Settings()

    # Intelligent Auto-Configuration (Zero-Config Mode)
    try:
        from forex_bot.core.autoconfig import AutoConfigurator

        # This will automatically tune Hardware (GPU/CPU), Risk (Prop Firm), and Models
        # based on the environment, effectively providing "Zero Config" out of the box.
        base_settings = AutoConfigurator(base_settings).auto_tune()
    except Exception as e:
        logger.warning(f"Auto-configuration failed: {e}")

    # Auto-detect Broker Backend based on OS
    if sys.platform == "win32":
        base_settings.system.broker_backend = "mt5_local"
        base_settings.system.mt5_required = True
        logger.info("[MODE] Operating System: WINDOWS -> Selected 'mt5_local' backend for Trading/Inference.")
    else:
        # Default to simulated/paper for non-Windows unless configured otherwise
        if base_settings.system.broker_backend == "mt5_local":
            logger.info("[MODE] Operating System: LINUX/OTHER -> 'mt5_local' not supported. Switching to 'simulated'.")
            base_settings.system.broker_backend = "simulated"
        else:
            logger.info(
                "[MODE] Operating System: LINUX/OTHER -> Keeping configured backend "
                f"'{base_settings.system.broker_backend}'."
            )

    symbols: list[str] = []
    if args.symbol:
        symbols = [s.strip() for s in args.symbol.split(",")]
    else:
        logger.info(f"Auto-discovering symbols in {base_settings.system.data_dir}...")
        discovered = discover_available_symbols(base_settings.system.data_dir)
        if discovered:
            symbols = discovered
            logger.info(f"Found {len(symbols)} symbols: {symbols}")
        else:
            logger.warning("No symbols discovered in data dir. Falling back to config.")
            symbols = base_settings.system.symbols

    if not symbols:
        logger.error("No symbols configured or found! Check data directory or config.yaml.")
        return

    logger.info(f"Initializing ForexBot for {len(symbols)} symbols (AUTOMATIC MODE)")

    if getattr(args, "export_onnx", False):
        models_dir = Path(os.environ.get("FOREX_BOT_MODELS_DIR", "models"))
        if not _global_models_exist(models_dir):
            logger.error(f"No trained models found in {models_dir}. Train first, then export ONNX.")
            return
        _export_onnx_from_saved_models(models_dir)
        return

    try:
        loader = DataLoader(base_settings)
        logger.info(f"Preflighting data/resampling for symbols: {symbols}")
        await loader.ensure_all_history(symbols)
    except Exception as exc:
        logger.warning(f"Preflight data setup skipped: {exc}")

    stop_event = asyncio.Event()
    _install_signal_handlers(asyncio.get_running_loop(), stop_event)
    listener = asyncio.create_task(_listen_for_escape(stop_event))

    if sys.platform == "win32":
        logger.info("Press ESC twice (within 2 seconds) to shutdown gracefully | Ctrl+C may not work in some terminals")
    else:
        logger.info("Press Ctrl+C to shutdown gracefully")

    if getattr(args, "train", False):
        logger.info("Training-only mode requested (--train).")
        try:
            await _run_global_training(base_settings, symbols, stop_event)
        except asyncio.CancelledError:
            logger.info("Training cancelled by user.")
        return

    if not getattr(args, "run", False) and not _global_models_exist():
        logger.info("No models found -> Training automatically before live trading...")
        try:
            await _run_global_training(base_settings, symbols, stop_event)
        except asyncio.CancelledError:
            logger.info("Global training cancelled by user.")
            return
    else:
        logger.info("Models found -> Starting live trading immediately...")

    # One-time ONNX export from existing models (no retrain), so CPU inference can run via ONNX Runtime.
    if bool(getattr(base_settings.models, "export_onnx", False)):
        models_dir = Path(os.environ.get("FOREX_BOT_MODELS_DIR", "models"))
        onnx_manifest = models_dir / "onnx" / "export_manifest.joblib"
        if not onnx_manifest.exists() and _global_models_exist(models_dir):
            logger.info("ONNX export requested and no manifest found; exporting ONNX models now...")
            await asyncio.to_thread(_export_onnx_from_saved_models, models_dir)

    default_concurrency = min(len(symbols), max(1, getattr(base_settings.system, "n_jobs", 1)))
    sem = asyncio.Semaphore(default_concurrency)

    tasks = []
    for symbol in symbols:
        instance_settings = deepcopy(base_settings)
        instance_settings.system.symbol = symbol

        async def _run_one(s=instance_settings):
            async with sem:
                await run_bot_instance(s, stop_event)

        tasks.append(asyncio.create_task(_run_one()))

    async def _cancel_on_stop():
        await stop_event.wait()
        for t in tasks:
            t.cancel()
        listener.cancel()

    stopper = asyncio.create_task(_cancel_on_stop())

    try:
        await asyncio.gather(*tasks, listener, stopper, return_exceptions=True)
    finally:
        listener.cancel()
        stopper.cancel()


def load_keys(keys_path: str = "keys.txt"):
    """Load secrets from a local keys file into environment variables."""
    path = Path(keys_path)
    if not path.exists():
        print(
            f"[WARN] Keys file {keys_path} not found. Ensure APIs/MT5 credentials are set via env vars.",
            file=sys.stderr,
        )
        return

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
    except Exception as e:
        print(f"[ERROR] Failed to load keys from {keys_path}: {e}", file=sys.stderr)


def main():
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        load_keys()

        asyncio.run(main_async())
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.getLogger(__name__).info("Bot stopped by user.")
    except Exception as e:
        logging.getLogger(__name__).error(f"Fatal error in main: {e!r}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

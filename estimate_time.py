#!/usr/bin/env python3
"""
Estimate total training time across all configured symbols and models
using real feature shapes and current hardware settings.

Usage:
    python estimate_time.py

Outputs per-symbol and total hours based on the estimator in ModelTrainer.
"""

import asyncio

from forex_bot.core.config import Settings
from forex_bot.core.system import AutoTuner, HardwareProbe
from forex_bot.data.loader import DataLoader
from forex_bot.features.pipeline import FeatureEngineer
from forex_bot.training.trainer import ModelTrainer


async def estimate_all(simulate_gpu: str | None = None) -> None:
    """
    Estimate end-to-end training time using current hardware profile.
    Safe to call from an existing event loop (e.g., notebooks) or as a script.
    """
    settings = Settings()

    # Apply hardware auto-detection to set GPU/CPU, batch sizes, etc.
    probe = HardwareProbe()
    profile = probe.detect()
    AutoTuner(settings, profile).apply()

    data_loader = DataLoader(settings)
    feature_engineer = FeatureEngineer(settings)
    trainer = ModelTrainer(settings)

    symbols = settings.system.symbols
    total_sec = 0.0
    original_symbol = settings.system.symbol

    for sym in symbols:
        settings.system.symbol = sym
        await data_loader.ensure_history(sym)
        frames = await data_loader.get_training_data(sym)
        ds = feature_engineer.prepare(frames, symbol=sym)

        est_sec = trainer.estimate_time_for_dataset(ds.X, len(ds.X), context="incremental", simulate_gpu=simulate_gpu)
        note = f" (simulated {simulate_gpu})" if simulate_gpu else ""
        print(f"{sym}: {est_sec / 3600:.2f} hours (n={len(ds.X)}){note}")
        total_sec += est_sec

    settings.system.symbol = original_symbol
    print("=" * 60)
    note = f" (simulated {simulate_gpu})" if simulate_gpu else ""
    print(f"Total estimated time: {total_sec / 3600:.2f} hours for {len(symbols)} symbols{note}")


def _run_coro(coro):
    """Run coroutine safely whether or not a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return asyncio.create_task(coro)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate training time for all symbols.")
    parser.add_argument(
        "--simulate-gpu", type=str, default=None, help="Simulate a GPU type (e.g., A4000) for estimation."
    )
    args = parser.parse_args()

    _run_coro(estimate_all(simulate_gpu=args.simulate_gpu))

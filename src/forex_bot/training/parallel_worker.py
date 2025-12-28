from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import Settings
from ..core.system import AutoTuner, HardwareProbe, thread_limits
from .model_factory import ModelFactory
from .optimization import HyperparameterOptimizer

logger = logging.getLogger(__name__)


def _load_memmap_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    meta_path = dataset_dir / "meta.json"
    cols_path = dataset_dir / "columns.json"
    x_path = dataset_dir / "X.npy"
    y_path = dataset_dir / "y.npy"
    idx_path = dataset_dir / "index.npy"

    if not cols_path.exists() or not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Dataset cache is incomplete: {dataset_dir}")

    meta: dict[str, Any] = {}
    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}

    cols = json.loads(cols_path.read_text(encoding="utf-8"))
    # Copy-on-write keeps the backing file read-only but marks the array writeable, which
    # avoids unnecessary copies in some downstream pipelines that require writeable inputs.
    X_mm = np.load(x_path, mmap_mode="c")
    y_mm = np.load(y_path, mmap_mode="c")

    index = None
    if idx_path.exists():
        try:
            idx_ns = np.load(idx_path, mmap_mode="r")
            if str(meta.get("index_kind", "")).startswith("datetime"):
                index = pd.to_datetime(idx_ns.astype(np.int64), utc=True)
            else:
                index = pd.Index(idx_ns)
        except Exception:
            index = None

    X = pd.DataFrame(X_mm, columns=list(cols), index=index)
    y = pd.Series(y_mm, index=X.index, dtype=np.int8)
    return X, y


def _pad_probs(p: Any) -> np.ndarray:
    if p is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(p)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.shape[1] == 2:
        out = np.zeros((len(arr), 3), dtype=np.float32)
        out[:, 0] = arr[:, 0]
        out[:, 1] = arr[:, 1]
        return out
    return arr[:, :3].astype(np.float32, copy=False)


def run_worker(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Forex AI parallel training worker (internal)")
    parser.add_argument("--dataset-dir", required=True, help="Path to cached memmap dataset directory")
    parser.add_argument("--models", required=True, help="Comma-separated model names to train")
    parser.add_argument("--out-dir", required=True, help="Directory to save trained model artifacts")
    parser.add_argument("--cpu-threads", type=int, default=0, help="BLAS/OMP threads to use (0=auto)")
    args = parser.parse_args(argv)

    models = [m.strip() for m in str(args.models).split(",") if m.strip()]
    if not models:
        raise ValueError("No models provided for worker")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    try:
        profile = HardwareProbe().detect()
        AutoTuner(settings, profile).apply()
    except Exception:
        # Worker can still proceed with safe defaults if probing fails.
        pass
    X, y = _load_memmap_dataset(Path(args.dataset_dir))

    optimizer = HyperparameterOptimizer(settings)
    best_params = optimizer.load_params()
    factory = ModelFactory(settings, out_dir)

    cpu_total = max(1, os.cpu_count() or 1)
    cpu_threads = int(args.cpu_threads or 0)
    if cpu_threads <= 0:
        # Safe default: cap at 8 threads to work on any hardware
        cpu_threads = max(1, min(8, cpu_total, int(os.environ.get("FOREX_BOT_CPU_THREADS", "8"))))

    durations: dict[str, float] = {}
    trained: list[str] = []

    # HPC FIX: Parallel Model Training Within Worker (Thread-Safe)
    # Use ThreadPoolExecutor for cross-platform compatibility
    # NumPy/sklearn release GIL during heavy computation, so threads work well
    max_concurrent_models = min(4, len(models), max(1, cpu_threads // 2))
    threads_per_model = max(1, cpu_threads // max(1, max_concurrent_models))

    logger.info(f"[WORKER] Training {len(models)} models with {max_concurrent_models} concurrent, "
                f"{threads_per_model} threads each (CPU budget: {cpu_threads})")

    def train_single_model(idx: int, name: str) -> tuple[str, float, bool]:
        """Train a single model and return (name, duration, success)"""
        t0 = time.perf_counter()
        try:
            model = factory.create_model(name, best_params, idx)
            with thread_limits(blas_threads=threads_per_model):
                model.fit(X, y)
            model.save(str(out_dir))
            duration = time.perf_counter() - t0
            logger.info(f"[WORKER] Successfully trained {name} in {duration:.1f}s")
            return (name, duration, True)
        except Exception as e:
            logger.error(f"[WORKER] Failed {name}: {e}", exc_info=True)
            return (name, 0.0, False)

    # Use ThreadPoolExecutor for reliable cross-platform parallel training
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[tuple[str, float, bool]] = []

    if max_concurrent_models > 1:
        # Parallel training
        with ThreadPoolExecutor(max_workers=max_concurrent_models) as executor:
            future_to_model = {
                executor.submit(train_single_model, idx, name): name
                for idx, name in enumerate(models, start=1)
            }
            for future in as_completed(future_to_model):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    model_name = future_to_model[future]
                    logger.error(f"[WORKER] Exception training {model_name}: {e}")
                    results.append((model_name, 0.0, False))
    else:
        # Sequential training (fallback for low-resource systems)
        logger.info("[WORKER] Sequential training mode (limited CPU resources)")
        for idx, name in enumerate(models, start=1):
            results.append(train_single_model(idx, name))

    # Collect results
    for name, duration, success in results:
        if success:
            trained.append(name)
            durations[name] = duration

    # Write a small manifest for the coordinator.
    try:
        (out_dir / "worker_manifest.json").write_text(
            json.dumps(
                {
                    "trained": trained,
                    "durations_sec": durations,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    # Ensure artifacts exist for at least one model; otherwise signal failure.
    return 0 if trained else 2

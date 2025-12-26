import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml

from ..core.config import Settings
from ..domain.events import PreparedDataset
from ..models.base import ExpertModel, detect_feature_drift
from ..strategy.fast_backtest import (
    fast_evaluate_strategy,
    infer_pip_metrics,
    infer_sl_tp_pips_auto,
)
from .ensemble import MetaBlender
from .evaluation import probs_to_signals, prop_backtest
from .optimization import HyperparameterOptimizer

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

# Services
from .benchmark_service import BenchmarkService
from .evaluation_service import EvaluationService
from .model_factory import ModelFactory
from .persistence_service import PersistenceService

logger = logging.getLogger(__name__)


def _find_entrypoint_script() -> Path:
    """
    Locate the repo's `forex-ai.py` entrypoint.

    This trainer sometimes runs from installed packages or different working directories, so we
    search upward from the current file rather than assuming a fixed directory depth.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "forex-ai.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot locate entrypoint script `forex-ai.py` from {here}")


# Optional FSDP imports
try:  # pragma: no cover
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
except Exception:  # pragma: no cover
    FSDP = None
    ShardingStrategy = None
    MixedPrecision = None


class ModelTrainer:
    """
    Orchestrates model training using micro-services.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.optimizer = HyperparameterOptimizer(settings)

        self.models_dir = Path(os.environ.get("FOREX_BOT_MODELS_DIR", "models"))
        self.logs_dir = Path(os.environ.get("FOREX_BOT_LOGS_DIR", "logs"))

        # Services
        self.evaluator = EvaluationService(settings, self.models_dir)
        self.factory = ModelFactory(settings, self.models_dir)
        self.persistence = PersistenceService(self.models_dir, self.logs_dir, settings=self.settings)
        self.benchmarker = BenchmarkService()

        # State
        self.models: dict[str, ExpertModel] = {}
        self.meta_blender: MetaBlender | None = None
        self.run_summary: dict[str, Any] = {}
        self.incremental_stats: dict[str, Any] = {}

        # Load history
        self.run_summary = self.persistence.load_run_summary()
        self._historical_durations = self.run_summary.get("train_durations_sec", {})
        self._historical_samples = self.run_summary.get("train_samples", None)
        self._historical_hardware = self.run_summary.get("train_hardware", {})
        self.incremental_stats = self.persistence.load_incremental_stats()
        # Distributed state
        self.distributed_enabled = False
        self.rank = 0
        self.world_size = 1

        self._maybe_init_distributed()

    def _maybe_init_distributed(self) -> None:
        """
        Initialize torch.distributed if enabled via settings.models.enable_ddp/fsdp and world_size > 1.
        Safe no-op if torch or distributed backend is unavailable.
        """
        try:
            use_dist = bool(
                getattr(self.settings.models, "enable_ddp", False) or getattr(self.settings.models, "enable_fsdp", False)
            )
            self.world_size = int(getattr(self.settings.models, "ddp_world_size", 1) or 1)
            if not use_dist or self.world_size <= 1:
                return
            if not dist.is_available():
                logger.warning("torch.distributed not available; running single-process.")
                return
            if dist.is_initialized():
                self.distributed_enabled = True
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                return
            # Default to env:// init (expect MASTER_ADDR/PORT set by launcher)
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            self.distributed_enabled = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            logger.info(f"Distributed initialized: rank {self.rank}/{self.world_size - 1}")
        except Exception as exc:
            logger.warning(f"Failed to initialize distributed; falling back to single process: {exc}")
            self.distributed_enabled = False
            self.rank = 0
            self.world_size = 1

    def estimate_time_for_dataset(
        self,
        X: pd.DataFrame,
        n_samples: int,
        context: str = "incremental",
        simulate_gpu: str | None = None,
    ) -> float:
        """
        Estimate runtime for the current enabled models on a dataset of size n_samples.
        Uses probes, historical stats, and incremental timings. X is only used for probes.
        """
        device = (
            "cuda"
            if bool(getattr(self.settings.system, "enable_gpu", False))
            else str(getattr(self.settings.system, "device", "cpu"))
        )
        bench_result = self.benchmarker.run_micro_benchmark(
            X, pd.Series([0] * min(len(X), 10)), device if simulate_gpu is None else "cuda"
        )
        est_time = self.benchmarker.estimate_time(
            self._get_enabled_models(),
            n_samples,
            bench_result,
            bool(getattr(self.settings.system, "enable_gpu", False)),
            int(getattr(self.settings.system, "num_gpus", 1)),
            context=context,
            historical_durations=self._historical_durations,
            historical_n=self._historical_samples,
            historical_gpu=(
                self._historical_hardware.get("enable_gpu"),
                self._historical_hardware.get("num_gpus", 1),
            ),
            incremental_stats=self.incremental_stats,
            probe_kwargs={
                "X": X.head(10_000),
                "batch_size": int(getattr(self.settings.models, "train_batch_size", 64)),
                "device": device,
                "steps": 6 if context == "incremental" else 8,
            },
            simulate_gpu=simulate_gpu,
        )
        return est_time

    def _parallel_models_enabled(self, enabled_models: list[str]) -> bool:
        mode = str(os.environ.get("FOREX_BOT_PARALLEL_MODELS", "auto")).lower().strip()
        if mode in {"0", "false", "no", "off"}:
            return False
        try:
            if dist.is_available() and dist.is_initialized():
                return False
        except Exception:
            pass
        if os.environ.get("FOREX_BOT_TRAIN_WORKER") == "1":
            return False
        if not bool(getattr(self.settings.system, "enable_gpu", False)):
            return False
        num_gpus = int(getattr(self.settings.system, "num_gpus", 0) or 0)
        if num_gpus <= 1:
            return False
        if len(enabled_models) < 2:
            return False
        # Windows can do this via subprocess, but it's easy to overwhelm local dev boxes.
        if os.name == "nt" and mode not in {"1", "true", "yes", "force"}:
            return False
        return True

    def _parallel_worker_count(self, enabled_models: list[str]) -> int:
        num_gpus = int(getattr(self.settings.system, "num_gpus", 0) or 0)
        requested = str(os.environ.get("FOREX_BOT_MAX_GPUS", "")).strip()
        if requested:
            try:
                num_gpus = max(1, min(num_gpus, int(requested)))
            except Exception:
                pass
        return max(1, min(num_gpus, len(enabled_models)))

    def _write_tuned_config(self, out_path: Path) -> None:
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.settings.model_dump()  # type: ignore[attr-defined]
            out_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to write tuned config for workers: {exc}")

    def _export_memmap_dataset(self, X: pd.DataFrame, y: pd.Series, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cols = list(getattr(X, "columns", []))
        (out_dir / "columns.json").write_text(json.dumps(cols), encoding="utf-8")

        index_kind = "none"
        try:
            if isinstance(X.index, pd.DatetimeIndex):
                index_kind = "datetime_ns"
        except Exception:
            index_kind = "none"

        (out_dir / "meta.json").write_text(json.dumps({"index_kind": index_kind}), encoding="utf-8")

        # Persist index (optional but useful for downstream eval/debug)
        try:
            if len(X.index) > 0:
                if isinstance(X.index, pd.DatetimeIndex):
                    idx_ns = X.index.view("int64")
                else:
                    idx_ns = np.asarray(X.index)
                idx_mm = np.lib.format.open_memmap(
                    out_dir / "index.npy", mode="w+", dtype=np.int64, shape=(len(idx_ns),)
                )
                idx_mm[:] = idx_ns.astype(np.int64, copy=False)
                idx_mm.flush()
        except Exception:
            pass

        # Write X/y in chunks to avoid a second full-size in-memory copy.
        n_rows, n_cols = X.shape
        x_mm = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+", dtype=np.float32, shape=(n_rows, n_cols))
        y_mm = np.lib.format.open_memmap(out_dir / "y.npy", mode="w+", dtype=np.int8, shape=(len(y),))

        chunk = int(os.environ.get("FOREX_BOT_MEMMAP_CHUNK_ROWS", "250000") or 250000)
        chunk = max(10_000, min(chunk, max(10_000, n_rows)))

        for start in range(0, n_rows, chunk):
            end = min(n_rows, start + chunk)
            x_mm[start:end] = X.iloc[start:end].to_numpy(dtype=np.float32, copy=False)
            y_mm[start:end] = y.iloc[start:end].to_numpy(dtype=np.int8, copy=False)

        x_mm.flush()
        y_mm.flush()

    def _schedule_models(self, models: list[str], workers: int) -> list[list[str]]:
        # Rough weights to reduce wall-time skew (heavy models first).
        base = dict(getattr(self.benchmarker, "COMPLEXITY_MAP", {}) or {})
        base.setdefault("random_forest", 0.0015)
        base.setdefault("extra_trees", 0.0012)
        base.setdefault("genetic", 0.002)
        base.setdefault("rllib_ppo", base.get("rl_ppo", 0.005))
        base.setdefault("rllib_sac", base.get("rl_sac", 0.008))

        weights = {m: float(base.get(m, 0.001)) for m in models}
        ordered = sorted(models, key=lambda m: weights.get(m, 0.0), reverse=True)

        buckets: list[list[str]] = [[] for _ in range(workers)]
        loads: list[float] = [0.0 for _ in range(workers)]
        for m in ordered:
            i = int(min(range(workers), key=lambda j: loads[j]))
            buckets[i].append(m)
            loads[i] += weights.get(m, 0.0)
        return buckets

    def _merge_worker_dirs(self, worker_dirs: list[Path]) -> None:
        skip = {
            "active_models.pkl",
            "meta_blender.joblib",
            "run_summary.json",
            "incremental_stats.json",
            "global_incremental_progress.json",
            "worker_manifest.json",
        }
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for wdir in worker_dirs:
            if not wdir.exists():
                continue
            for item in wdir.iterdir():
                if item.name in skip:
                    continue
                dest = self.models_dir / item.name
                try:
                    if item.is_dir():
                        shutil.copytree(item, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dest)
                except Exception as exc:
                    logger.warning(f"Failed to merge worker artifact {item}: {exc}")

    def _train_models_parallel(
        self,
        enabled_models: list[str],
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        *,
        stop_event: asyncio.Event | None,
    ) -> dict[str, float]:
        """
        Train models across available GPUs using subprocess workers pinned via CUDA_VISIBLE_DEVICES.
        Returns per-model durations (best effort).
        """
        worker_count = self._parallel_worker_count(enabled_models)
        buckets = self._schedule_models(enabled_models, worker_count)

        cpu_total = max(1, os.cpu_count() or 1)
        try:
            cpu_budget = int(os.environ.get("FOREX_BOT_CPU_BUDGET", "0") or 0)
        except Exception:
            cpu_budget = 0
        if cpu_budget <= 0:
            cpu_budget = cpu_total  # default: use all available CPU cores
        cpu_budget = max(1, min(cpu_total, cpu_budget))
        base_threads = max(1, cpu_budget // worker_count)
        extra = max(0, cpu_budget % worker_count)

        cache_root = Path(getattr(self.settings.system, "cache_dir", "cache")) / "parallel_training"
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = cache_root / f"run_{run_id}_{os.getpid()}"
        dataset_dir = run_dir / "dataset"
        cfg_path = run_dir / "tuned_config.yaml"
        logs_dir = self.logs_dir / "parallel_workers" / run_dir.name
        workers_root = self.models_dir / "_workers" / run_dir.name

        logs_dir.mkdir(parents=True, exist_ok=True)
        workers_root.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Parallel model training enabled: workers={worker_count}, cpu_budget={cpu_budget}, "
            f"threads/worker={base_threads}+{extra}"
        )

        self._write_tuned_config(cfg_path)
        cfg_exists = cfg_path.exists()
        self._export_memmap_dataset(X_fit, y_fit, dataset_dir)

        entry = _find_entrypoint_script()

        procs: list[tuple[int, Path, subprocess.Popen, Any]] = []
        try:
            for wid, models in enumerate(buckets):
                if not models:
                    continue
                out_dir = workers_root / f"worker_{wid}"
                out_dir.mkdir(parents=True, exist_ok=True)
                log_path = logs_dir / f"worker_{wid}.log"

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                env["FOREX_BOT_TRAIN_WORKER"] = "1"
                if cfg_exists:
                    env["CONFIG_FILE"] = str(cfg_path)
                env["CUDA_VISIBLE_DEVICES"] = str(wid)
                worker_threads = base_threads + (1 if wid < extra else 0)
                env["FOREX_BOT_CPU_THREADS"] = str(worker_threads)
                for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[k] = str(worker_threads)

                cmd = [
                    sys.executable,
                    str(entry),
                    "--_worker",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--models",
                    ",".join(models),
                    "--out-dir",
                    str(out_dir),
                    "--cpu-threads",
                    str(worker_threads),
                ]

                log_f = open(log_path, "w", encoding="utf-8")
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(entry.parent),
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                    )
                except Exception:
                    try:
                        log_f.close()
                    except Exception:
                        pass
                    raise
                procs.append((wid, out_dir, proc, log_f))
                logger.info(f"Worker {wid}: models={models} log={log_path}")

            # Wait for completion / cancellation.
            while True:
                if stop_event and stop_event.is_set():
                    raise KeyboardInterrupt("Stop requested")
                if all(p.poll() is not None for _, _, p, _ in procs):
                    break
                time.sleep(5)

            failed = [
                (wid, int(p.returncode)) for wid, _, p, _ in procs if (p.poll() is not None and p.returncode != 0)
            ]
            if failed:
                logger.error(f"Parallel workers failed: {failed}. See logs under {logs_dir}")
                raise RuntimeError(f"Parallel workers failed: {failed}")
        except Exception:
            for _, _, p, _ in procs:
                try:
                    if p.poll() is None:
                        p.terminate()
                except Exception:
                    pass
            raise
        finally:
            for _, _, _, lf in procs:
                try:
                    lf.close()
                except Exception:
                    pass

        worker_dirs = [out for _, out, _, _ in procs]
        self._merge_worker_dirs(worker_dirs)

        # Aggregate durations from worker manifests.
        durations: dict[str, float] = {}
        trained_models: list[str] = []
        for wdir in worker_dirs:
            try:
                manifest = json.loads((wdir / "worker_manifest.json").read_text(encoding="utf-8"))
                t = manifest.get("trained", [])
                if isinstance(t, list):
                    trained_models.extend([str(x) for x in t if str(x)])
                d = manifest.get("durations_sec", {})
                if isinstance(d, dict):
                    for k, v in d.items():
                        try:
                            durations[str(k)] = float(v)
                        except Exception:
                            pass
            except Exception:
                continue

        # Load full model set into this trainer instance for downstream stacking/eval.
        try:
            import joblib

            if trained_models:
                trained_set = set(trained_models)
                ordered = [m for m in enabled_models if m in trained_set]
            else:
                ordered = list(enabled_models)
            joblib.dump(ordered, self.models_dir / "active_models.pkl")
        except Exception:
            pass

        try:
            self.models, self.meta_blender = self.persistence.load_models()
        except Exception as exc:
            logger.warning(f"Failed to load merged models after parallel training: {exc}", exc_info=True)

        return durations

    def train_all(
        self,
        dataset: PreparedDataset,
        optimize: bool = True,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        logger.info("Starting training cycle...")

        # 1. Setup Logging
        writer = None
        if TENSORBOARD_AVAILABLE:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            writer = SummaryWriter(log_dir=str(self.logs_dir / f"training_{run_id}"))

        # 2. Prepare Data (META-LABELING FILTER)
        # Only train on rows where a base strategy signal exists.
        # This prevents the model from learning to just predict "No Trade" (accuracy padding).
        X_raw = dataset.X
        y_raw = pd.Series(dataset.y) if isinstance(dataset.y, (np.ndarray, list)) else dataset.y

        mask = X_raw["base_signal"] != 0
        if mask.sum() < 100:
            logger.warning("Insufficient base signals for meta-labeling. Training on all data (fallback).")
            X = X_raw
            y = y_raw
            meta_subset = dataset.metadata
        elif int(mask.sum()) == int(len(X_raw)):
            # Avoid an unnecessary full-frame copy when already pre-filtered upstream.
            logger.info(f"Meta-Labeling: Using pre-filtered dataset ({len(X_raw)} active signal events).")
            X = X_raw
            y = y_raw
            meta_subset = dataset.metadata
        else:
            logger.info(f"Meta-Labeling: Filtering to {mask.sum()} active signal events (from {len(X_raw)} rows).")
            X = X_raw.loc[mask].copy()
            y = y_raw.loc[mask].copy()
            meta_subset = dataset.metadata.loc[mask].copy() if dataset.metadata is not None else None

        try:
            self.run_summary["feature_columns"] = list(getattr(X, "columns", []))
        except Exception:
            self.run_summary["feature_columns"] = []

        # Holdout split for out-of-sample evaluation + stacking.
        #
        # IMPORTANT: For pooled multi-symbol datasets, row-order is often NOT time-ordered
        # (e.g., concatenated by symbol). A positional split would leak "future" rows into
        # the fit set and invalidate evaluation. Prefer a timestamp-based split when possible.
        if isinstance(X.index, pd.DatetimeIndex) and not X.index.is_monotonic_increasing:
            order = np.argsort(X.index.view("int64"))
            X = X.iloc[order]
            y = y.iloc[order]
            if meta_subset is not None:
                meta_subset = meta_subset.iloc[order]
        n = len(y)
        holdout_pct = float(getattr(self.settings.models, "train_holdout_pct", 0.2) or 0.2)
        holdout_pct = float(min(0.5, max(0.0, holdout_pct)))
        min_split = 50  # Lower threshold since we filtered

        X_fit = X
        y_fit = y
        X_eval = X.iloc[:0]
        y_eval = y.iloc[:0]
        meta_fit = None
        meta_eval = None
        holdout_cutoff: str | None = None
        split_strategy = "none"

        def _finalize_summary() -> None:
            fit_n = int(len(X_fit))
            eval_n = int(len(X_eval))
            self.run_summary["train_holdout_pct"] = float(holdout_pct if eval_n > 0 else 0.0)
            self.run_summary["train_holdout_n"] = int(eval_n)
            self.run_summary["train_fit_n"] = int(fit_n)
            self.run_summary["train_holdout_split"] = {
                "strategy": split_strategy,
                "cutoff": holdout_cutoff,
            }

        # Time-based split if index is datetime (works even if the frame is not pre-sorted).
        if holdout_pct > 0.0 and isinstance(X.index, pd.DatetimeIndex) and n > 0:
            try:
                idx_ns = X.index.view("int64")
                nat = np.iinfo(np.int64).min
                valid = idx_ns != nat

                if valid.any():
                    times_valid = np.array(idx_ns[valid], copy=True)
                    fit_frac = float(1.0 - holdout_pct)
                    k = int(max(0, min(len(times_valid) - 1, int(len(times_valid) * fit_frac) - 1)))
                    cut_ns = int(np.partition(times_valid, k)[k])
                    cutoff_ts = pd.to_datetime(cut_ns, utc=True)

                    fit_mask = idx_ns <= cut_ns
                    eval_mask = idx_ns > cut_ns
                    fit_n = int(fit_mask.sum())
                    eval_n = int(eval_mask.sum())

                    if fit_n >= min_split and eval_n >= min_split:
                        X_fit = X.loc[fit_mask]
                        y_fit = y.loc[fit_mask]
                        X_eval = X.loc[eval_mask]
                        y_eval = y.loc[eval_mask]
                        if meta_subset is not None:
                            with contextlib.suppress(Exception):
                                meta_fit = meta_subset.loc[fit_mask]
                            with contextlib.suppress(Exception):
                                meta_eval = meta_subset.loc[eval_mask]
                        holdout_cutoff = cutoff_ts.isoformat()
                        split_strategy = "time_cutoff"
                        _finalize_summary()
                    else:
                        split_strategy = "time_cutoff_insufficient"
            except Exception as exc:
                logger.warning(f"Holdout time-based split failed; falling back to positional: {exc}")

        # Fallback: positional split (assumes row-order is time-ordered).
        if split_strategy in {"none", "time_cutoff_insufficient"}:
            holdout_n = int(n * holdout_pct)
            eval_start = max(0, n - holdout_n)
            if eval_start < min_split or (n - eval_start) < min_split:
                eval_start = n  # disable holdout

            X_fit = X.iloc[:eval_start]
            y_fit = y.iloc[:eval_start]
            X_eval = X.iloc[eval_start:]
            y_eval = y.iloc[eval_start:]

            meta_fit = None
            meta_eval = None
            if meta_subset is not None:
                with contextlib.suppress(Exception):
                    meta_fit = meta_subset.iloc[:eval_start]
                with contextlib.suppress(Exception):
                    meta_eval = meta_subset.iloc[eval_start:]

            split_strategy = "positional" if eval_start < n else "positional_disabled"
            _finalize_summary()

        # Feature drift detection between train and eval sets
        if len(X_fit) > 0 and len(X_eval) > 0:
            try:
                drift_result = detect_feature_drift(X_fit, X_eval, threshold=0.1)
                self.run_summary["feature_drift"] = {
                    "summary": drift_result["summary"],
                    "critical": drift_result["critical"],
                    "drifted_count": len(drift_result["drifted_features"]),
                    "top_drifted": drift_result["drifted_features"][:10],
                }
                if drift_result["critical"]:
                    logger.warning(
                        f"Critical feature drift detected! {drift_result['summary']}. "
                        "Model predictions may be unreliable on recent data."
                    )
            except Exception as e:
                logger.debug(f"Feature drift detection failed: {e}")

        # Basic Cast
        try:
            float_cols = [c for c in X.columns if X[c].dtype.kind == "f"]
            if float_cols:
                X[float_cols] = X[float_cols].astype(np.float32)
        except Exception as e:
            logger.debug(f"Float32 casting failed: {e}")

        # 3. Optimization
        best_params = {}
        if optimize and getattr(self.optimizer, "available", False):
            best_params = self.optimizer.optimize_all(X_fit, y_fit, meta_fit, stop_event=stop_event)
        else:
            best_params = self.optimizer.load_params()

        # Preserve metadata even for multi-symbol runs so OHLC-dependent models can train globally.
        # NOTE: downstream models that assume a single continuous series should handle the symbol column explicitly.
        meta_fit_models = meta_fit
        meta_eval_models = meta_eval
        if meta_fit is not None and isinstance(meta_fit, pd.DataFrame) and "symbol" in meta_fit.columns:
            logger.info("Multi-symbol metadata detected; passing through to OHLC-dependent models for global training.")

        # 4. Model Selection
        enabled_models = self._get_enabled_models()
        enabled_models = self._maybe_shard_models(enabled_models)

        # 5. Benchmark
        device = (
            "cuda"
            if bool(getattr(self.settings.system, "enable_gpu", False))
            else str(getattr(self.settings.system, "device", "cpu"))
        )
        bench_result = self.benchmarker.run_micro_benchmark(X_fit, y_fit, device)
        est_time = self.benchmarker.estimate_time(
            enabled_models,
            len(X_fit),
            bench_result,
            bool(getattr(self.settings.system, "enable_gpu", False)),
            int(getattr(self.settings.system, "num_gpus", 1)),
            context="full",
            historical_durations=self._historical_durations,
            historical_n=self._historical_samples,
            historical_gpu=(
                self._historical_hardware.get("enable_gpu"),
                self._historical_hardware.get("num_gpus", 1),
            ),
            incremental_stats=self.incremental_stats,
            probe_kwargs={
                "X": X_fit.head(10_000),
                "batch_size": int(getattr(self.settings.models, "train_batch_size", 64)),
                "device": device,
                "steps": 8,
            },
        )
        logger.info(
            f"Estimated Training Time (ALL {len(enabled_models)} models): {est_time / 3600:.1f} hours "
            f"({est_time / 60:.0f} minutes)"
        )
        if len(enabled_models) > 0:
            logger.info(f"  → Average per model: {est_time / len(enabled_models) / 3600:.2f} hours")

        # 6. Train Loop
        durations = {}
        total = len(enabled_models)

        trained_in_parallel = False
        if self._parallel_models_enabled(enabled_models):
            try:
                durations = self._train_models_parallel(enabled_models, X_fit, y_fit, stop_event=stop_event)
                trained_in_parallel = True
            except Exception as exc:
                logger.warning(f"Parallel model training failed; falling back to sequential: {exc}", exc_info=True)
                durations = {}
                trained_in_parallel = False

        # Device plan (round-robin via factory), just for logging
        if (
            not trained_in_parallel
            and bool(getattr(self.settings.system, "enable_gpu", False))
            and getattr(self.settings.system, "num_gpus", 1) > 1
        ):
            mapping = {}
            for idx, name in enumerate(enabled_models, 1):
                if self.factory.available_gpus:
                    mapping[name] = self.factory.available_gpus[(idx - 1) % len(self.factory.available_gpus)]
            if mapping:
                logger.info(f"GPU assignment: {mapping}")

        for idx, name in enumerate(enabled_models, 1):
            if trained_in_parallel:
                break
            if stop_event and stop_event.is_set():
                break

            # Some experts require OHLC metadata (e.g., GA TA-Lib strategies, prop-firm RL env).
            # In pooled multi-symbol runs, we intentionally disable metadata to avoid single-series leakage.
            # Skip those experts up-front instead of calling fit() with missing metadata,
            # which would otherwise produce noisy warnings.
            # Allow OHLC-dependent models to run with global (multi-symbol) metadata; they should
            # handle the 'symbol' column internally to avoid leakage.
            if name in {"genetic", "rl_ppo", "rl_sac"} and meta_fit_models is None:
                logger.info(f"[{idx}/{total}] Skipping {name}: OHLC metadata unavailable for this training run.")
                durations[name] = 0.0
                continue

            t0 = time.perf_counter()
            try:
                logger.info(f"[{idx}/{total}] Training {name}...")

                # Ensure time-ordered data for models that rely on chronological sequences
                time_order_models = {"genetic", "rl_ppo", "rl_sac", "rllib_ppo", "rllib_sac", "evolution"}
                if name in time_order_models and isinstance(X_fit.index, pd.DatetimeIndex):
                    if not X_fit.index.is_monotonic_increasing:
                        order = np.argsort(X_fit.index.view("int64"))
                        X_fit = X_fit.iloc[order]
                        y_fit = y_fit.iloc[order]
                        if meta_fit_models is not None:
                            meta_fit_models = meta_fit_models.iloc[order]
                        if len(X_eval) > 0 and isinstance(X_eval.index, pd.DatetimeIndex):
                            eval_order = np.argsort(X_eval.index.view("int64"))
                            X_eval = X_eval.iloc[eval_order]
                            y_eval = y_eval.iloc[eval_order]
                            if meta_eval_models is not None:
                                meta_eval_models = meta_eval_models.iloc[eval_order]

                model = self.factory.create_model(name, best_params, idx)

                # Optional FSDP wrapping for torch models when multiple GPUs & USE_FSDP=1
                env_fsdp = os.environ.get("USE_FSDP", "0").lower()
                use_fsdp = env_fsdp in {"1", "true", "yes", "auto"}
                if (
                    use_fsdp
                    and FSDP is not None
                    and dist.is_available()
                    and dist.is_initialized()
                    and dist.get_world_size() > 1
                    and hasattr(model, "model")
                    and isinstance(model.model, torch.nn.Module)
                ):
                    try:
                        # Auto-disable if GPU memory is very small (<16GB)
                        if env_fsdp == "auto":
                            try:
                                if torch.cuda.is_available():
                                    props = torch.cuda.get_device_properties(0)
                                    if props.total_memory < 16 * 1024**3:
                                        use_fsdp = False
                                        logger.info("FSDP auto-disabled: GPU memory <16GB.")
                            except Exception:
                                pass
                        if not use_fsdp:
                            raise RuntimeError("FSDP auto disabled")

                        mp = None
                        if MixedPrecision is not None:
                            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                                mp = MixedPrecision(
                                    param_dtype=torch.bfloat16,
                                    reduce_dtype=torch.float32,
                                    buffer_dtype=torch.bfloat16,
                                )
                            else:
                                mp = MixedPrecision(
                                    param_dtype=torch.float16,
                                    reduce_dtype=torch.float32,
                                    buffer_dtype=torch.float16,
                                )
                        shard = ShardingStrategy.FULL_SHARD if ShardingStrategy else None
                        local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank() % torch.cuda.device_count()))
                        model.model = FSDP(
                            model.model,
                            sharding_strategy=shard,
                            mixed_precision=mp,
                            device_id=local_rank if torch.cuda.is_available() else None,
                        )
                        logger.info(f"FSDP enabled for {name}")
                    except Exception as e:
                        logger.warning(f"FSDP wrap failed for {name}: {e}")

                # Boost Threads
                import multiprocessing

                from ..core.system import thread_limits

                try:
                    cpu_threads = int(os.environ.get("FOREX_BOT_CPU_THREADS", "0") or 0)
                except Exception:
                    cpu_threads = 0
                if cpu_threads <= 0:
                    cpu_threads = multiprocessing.cpu_count()
                with thread_limits(blas_threads=cpu_threads):
                    # Fit logic (simplified here, assume factory configured it well)
                    fit_kwargs = {}
                    if hasattr(model, "fit"):
                        # Inspect if writer needed
                        import inspect

                        sig = inspect.signature(model.fit)
                        has_kwargs = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in sig.parameters.values()  # noqa: B023
                        )
                        if "tensorboard_writer" in sig.parameters:
                            fit_kwargs["tensorboard_writer"] = writer
                        if meta_fit_models is not None and ("metadata" in sig.parameters or has_kwargs):
                            fit_kwargs["metadata"] = meta_fit_models

                        model.fit(X_fit, y_fit, **fit_kwargs)

                # Only persist models that can actually predict (prevents "active but unusable" artifacts).
                try:
                    import inspect

                    pred_kwargs = {}
                    try:
                        psig = inspect.signature(model.predict_proba)
                        has_kwargs = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in psig.parameters.values()  # noqa: B023
                        )
                        if meta_fit_models is not None and ("metadata" in psig.parameters or has_kwargs):
                            pred_kwargs["metadata"] = meta_fit_models
                    except Exception:
                        pass

                    sample = X_fit.iloc[: min(256, len(X_fit))]
                    if len(sample) == 0:
                        raise RuntimeError("No samples available for post-fit check")
                    _ = self._pad_probs(model.predict_proba(sample, **pred_kwargs))
                except Exception as exc:
                    logger.warning(f"{name} trained but is not usable for inference: {exc}")
                    continue

                self.models[name] = model
                model.save(str(self.models_dir))

                # Prop Backtest Metric
                if meta_eval_models is not None and len(X_eval) > 0:
                    try:
                        pred_kwargs = {}
                        try:
                            import inspect

                            psig = inspect.signature(model.predict_proba)
                            has_kwargs = any(
                                p.kind == inspect.Parameter.VAR_KEYWORD
                                for p in psig.parameters.values()  # noqa: B023
                            )
                            if meta_eval_models is not None and ("metadata" in psig.parameters or has_kwargs):
                                pred_kwargs["metadata"] = meta_eval_models
                        except Exception:
                            pass

                        probs_eval = self._pad_probs(model.predict_proba(X_eval, **pred_kwargs))
                        sig_eval = probs_to_signals(probs_eval)
                        sig_eval = pd.Series(sig_eval, index=X_eval.index)
                        prop_metrics = prop_backtest(meta_eval_models, sig_eval)

                        fast_metrics: dict[str, Any] = {}
                        try:
                            if {"close", "high", "low"}.issubset(set(meta_eval_models.columns)):
                                idx = meta_eval_models.index
                                if not isinstance(idx, pd.DatetimeIndex):
                                    idx = pd.to_datetime(idx, utc=True, errors="coerce")
                                month_idx = (idx.year.astype(np.int32) * 12 + idx.month.astype(np.int32)).to_numpy(
                                    dtype=np.int64
                                )

                                pip_size, pip_value_per_lot = infer_pip_metrics(self.settings.system.symbol)
                                sl_cfg = getattr(self.settings.risk, "meta_label_sl_pips", None)
                                tp_cfg = getattr(self.settings.risk, "meta_label_tp_pips", None)
                                rr = float(getattr(self.settings.risk, "min_risk_reward", 2.0))
                                if sl_cfg is None or float(sl_cfg) <= 0:
                                    atr_vals = (
                                        meta_eval_models["atr"].to_numpy(dtype=np.float64)
                                        if "atr" in meta_eval_models.columns
                                        else None
                                    )
                                    auto = infer_sl_tp_pips_auto(
                                        open_prices=meta_eval_models["open"].to_numpy(dtype=np.float64)
                                        if "open" in meta_eval_models.columns
                                        else meta_eval_models["close"].to_numpy(dtype=np.float64),
                                        high_prices=meta_eval_models["high"].to_numpy(dtype=np.float64),
                                        low_prices=meta_eval_models["low"].to_numpy(dtype=np.float64),
                                        close_prices=meta_eval_models["close"].to_numpy(dtype=np.float64),
                                        atr_values=atr_vals,
                                        pip_size=pip_size,
                                        atr_mult=float(getattr(self.settings.risk, "atr_stop_multiplier", 1.5)),
                                        min_rr=rr,
                                        min_dist=float(getattr(self.settings.risk, "meta_label_min_dist", 0.0)),
                                        settings=self.settings,
                                    )
                                    if auto is None:
                                        raise RuntimeError("Cannot infer SL/TP pips from metadata.")
                                    sl_pips, tp_pips = auto
                                else:
                                    sl_pips = float(sl_cfg)
                                    if tp_cfg is None or float(tp_cfg) <= 0:
                                        tp_pips = sl_pips * rr
                                    else:
                                        tp_pips = max(float(tp_cfg), sl_pips * rr)

                                spread = float(getattr(self.settings.risk, "backtest_spread_pips", 1.5))
                                commission = float(getattr(self.settings.risk, "commission_per_lot", 0.0))

                                arr = fast_evaluate_strategy(
                                    close_prices=meta_eval_models["close"].to_numpy(dtype=np.float64),
                                    high_prices=meta_eval_models["high"].to_numpy(dtype=np.float64),
                                    low_prices=meta_eval_models["low"].to_numpy(dtype=np.float64),
                                    signals=sig_eval.to_numpy(dtype=np.int8),
                                    month_indices=month_idx,
                                    sl_pips=sl_pips,
                                    tp_pips=tp_pips,
                                    pip_value=pip_size,
                                    spread_pips=spread,
                                    commission_per_trade=commission,
                                    pip_value_per_lot=pip_value_per_lot,
                                )

                                keys = [
                                    "net_profit",
                                    "sharpe",
                                    "sortino",
                                    "max_dd",
                                    "win_rate",
                                    "profit_factor",
                                    "expectancy",
                                    "sqn",
                                    "trades",
                                    "consistency_score",
                                ]
                                fast_metrics = {k: float(v) for k, v in zip(keys, arr.tolist(), strict=False)}
                        except Exception as e:
                            logger.debug(f"Fast backtest metrics failed for {name}: {e}")

                        self.run_summary.setdefault("model_metrics", {})[name] = {
                            "prop": prop_metrics,
                            "fast": fast_metrics,
                        }
                    except Exception as e:
                        logger.debug(f"Model backtest metrics failed for {name}: {e}")

            except Exception as e:
                logger.error(f"Training {name} failed: {e}")

            durations[name] = time.perf_counter() - t0
            # Cleanup
            import gc

            gc.collect()

        # 7. Meta Blender
        self._prune_low_quality_models()
        if not stop_event or not stop_event.is_set():
            self._train_blender(X_eval, y_eval)

        # 8. Evaluation (enforce walkforward/CPCV)
        if not stop_event or not stop_event.is_set():
            eval_dataset = PreparedDataset(
                X=X_eval,
                y=y_eval,
                index=X_eval.index,
                feature_names=list(getattr(X_eval, "columns", [])),
                metadata=meta_eval,
                labels=y_eval,
            )
            self.run_summary["walkforward"] = self.evaluator.run_walkforward(
                eval_dataset, self.models, self._ensemble_proba, start_index=0
            )
            self.run_summary["cpcv"] = self.evaluator.run_cpcv(eval_dataset, self.models)

        # 9. Persistence
        self.run_summary["train_durations_sec"] = durations
        self.run_summary["train_samples"] = len(X_fit)
        self.run_summary["train_hardware"] = {
            "enable_gpu": bool(getattr(self.settings.system, "enable_gpu", False)),
            "num_gpus": int(getattr(self.settings.system, "num_gpus", 1)),
            "device": str(getattr(self.settings.system, "device", "cpu")),
        }
        if not self.distributed_enabled or self.rank == 0:
            self.persistence.save_run_summary(self.run_summary)
            self.persistence.save_active_models_list(self.models)
            self.persistence.save_models_bundle(self.models)
        if self.distributed_enabled:
            try:
                dist.barrier()
            except Exception:
                pass

        # 10. Export
        sample = X.iloc[:1] if len(X) > 0 else None
        if sample is not None and bool(getattr(self.settings.models, "export_onnx", False)):
            if not self.distributed_enabled or self.rank == 0:
                self.persistence.export_onnx(self.models, self.meta_blender, sample)
            if self.distributed_enabled:
                try:
                    dist.barrier()
                except Exception:
                    pass

        if writer:
            writer.close()
        self.persistence.cleanup_logs()
        logger.info("Training Complete.")

    def _prune_low_quality_models(self, dd_limit: float = 0.10, pfloor: float = 1.0) -> None:
        """
        Remove trained models that violate basic risk/quality thresholds before
        blending/export. This prevents obviously bad experts from dragging the
        ensemble or ONNX export.
        """
        try:
            metrics = self.run_summary.get("model_metrics", {})
        except Exception:
            metrics = {}

        removed: list[str] = []
        for name in list(self.models.keys()):
            m = metrics.get(name, {})
            fast = m.get("fast", m)
            pf = fast.get("profit_factor", None)
            dd = fast.get("max_dd", fast.get("max_dd_pct", None))
            try:
                if pf is not None and float(pf) <= pfloor:
                    removed.append(name)
                    continue
                if dd is not None and float(dd) > dd_limit:
                    removed.append(name)
            except Exception:
                continue

        for name in removed:
            self.models.pop(name, None)

        if removed:
            logger.info(f"Pruned low-quality models (pf<={pfloor} or dd>{dd_limit:.2%}): {removed}")

    def train_incremental(
        self, dataset: PreparedDataset, symbol: str, optimize: bool = False, stop_event: asyncio.Event | None = None
    ) -> None:
        """
        Incremental training wrapper.
        Reuses the factory logic but ensures models are loaded first.
        """
        if not self.models:
            self.load_models()

        X, y = dataset.X, pd.Series(dataset.y)

        # Quick time estimate up front (works for CPU/GPU). Uses the same heuristic as full training.
        try:
            device = (
                "cuda"
                if bool(getattr(self.settings.system, "enable_gpu", False))
                else str(getattr(self.settings.system, "device", "cpu"))
            )
            bench_result = self.benchmarker.run_micro_benchmark(X, y, device)
            est_time = self.benchmarker.estimate_time(
                self._get_enabled_models(),
                len(X),
                bench_result,
                bool(getattr(self.settings.system, "enable_gpu", False)),
                int(getattr(self.settings.system, "num_gpus", 1)),
                context="incremental",
                historical_durations=self._historical_durations,
                historical_n=self._historical_samples,
                historical_gpu=(
                    self._historical_hardware.get("enable_gpu"),
                    self._historical_hardware.get("num_gpus", 1),
                ),
                incremental_stats=self.incremental_stats,
                probe_kwargs={
                    "X": X.head(10_000),
                    "batch_size": int(getattr(self.settings.models, "train_batch_size", 64)),
                    "device": device,
                    "steps": 6,
                },
            )
            logger.info(
                f"Estimated incremental time for {symbol} (ALL {len(self._get_enabled_models())} models): "
                f"{est_time / 3600:.2f} hours ({est_time / 60:.0f} min) for {len(X):,} samples"
            )
            if len(self._get_enabled_models()) > 0:
                logger.info(f"  → Average per model: {est_time / len(self._get_enabled_models()) / 3600:.2f} hours")
            if est_time > 72 * 3600:
                logger.warning(
                    f"Estimate is very high ({est_time / 3600:.1f}h = {est_time / 3600 / 24:.1f} days). "
                    "Check GPU settings (enable_gpu/num_gpus/device)."
                )
        except Exception as e:
            logger.debug(f"Incremental time estimate failed: {e}")

        # Cast
        try:
            float_cols = [c for c in X.columns if X[c].dtype.kind == "f"]
            if float_cols:
                X[float_cols] = X[float_cols].astype(np.float32)
        except Exception as e:
            logger.debug(f"Float32 casting failed in incremental training: {e}")

        enabled = self._get_enabled_models()
        enabled = self._maybe_shard_models(enabled)

        for idx, name in enumerate(enabled, 1):
            if stop_event and stop_event.is_set():
                break
            try:
                logger.info(f"Incremental: {name} on {symbol}")

                # Ensure model exists
                if name not in self.models:
                    self.models[name] = self.factory.create_model(name, {}, idx)

                model = self.models[name]

                import multiprocessing

                from ..core.system import thread_limits

                with thread_limits(blas_threads=multiprocessing.cpu_count()):
                    fit_kwargs = {}
                    # Inspect signature...
                    if hasattr(model, "fit"):
                        import inspect

                        sig = inspect.signature(model.fit)
                        if "metadata" in sig.parameters:
                            fit_kwargs["metadata"] = dataset.metadata
                        t0 = time.perf_counter()
                        model.fit(X, y, **fit_kwargs)
                        duration = time.perf_counter() - t0
                        # Persist incremental timing for future estimates
                        self.incremental_stats[name] = {
                            "duration_sec": duration,
                            "n_samples": len(X),
                            "enable_gpu": bool(getattr(self.settings.system, "enable_gpu", False)),
                            "num_gpus": int(getattr(self.settings.system, "num_gpus", 1)),
                            "device": str(getattr(self.settings.system, "device", "cpu")),
                        }
                        self.persistence.save_incremental_stats(self.incremental_stats)

                model.save(str(self.models_dir))
            except Exception as e:
                logger.warning(f"Inc train {name} failed: {e}")

        self.persistence.save_active_models_list(self.models)

    def load_models(self):
        self.models, self.meta_blender = self.persistence.load_models()

    def _get_enabled_models(self) -> list[str]:
        """
        Hard-enable every model type, ignoring feature flags.
        This guarantees the full stack is always trained/included.
        """
        candidates = list(self.settings.models.ml_models)
        candidates.extend(
            [
                "transformer",
                "evolution",
                "genetic",
                "rl_ppo",
                "rl_sac",
                "rllib_ppo",
                "rllib_sac",
                "nbeats",
                "tide",
                "tabnet",
                "kan",
            ]
        )

        # Preserve order while deduplicating
        seen = set()
        ordered = []
        for m in candidates:
            if m not in seen:
                ordered.append(m)
                seen.add(m)

        # Filter out models that cannot run in the current environment.
        # This keeps the "train everything" philosophy while avoiding known no-op experts
        # (e.g., RLlib on Python 3.13 Windows where Ray wheels do not exist).
        filtered: list[str] = []
        skipped: list[str] = []

        try:
            from ..models.rllib_agent import RAY_AVAILABLE  # type: ignore
        except Exception:
            RAY_AVAILABLE = False  # type: ignore

        try:
            from ..models.rl import SB3_AVAILABLE  # type: ignore
        except Exception:
            SB3_AVAILABLE = False  # type: ignore

        try:
            from ..models.registry import MODEL_REGISTRY

            registry = set(MODEL_REGISTRY.keys())
        except Exception:
            registry = set()

        for name in ordered:
            if name in {"rllib_ppo", "rllib_sac"} and not bool(RAY_AVAILABLE):
                skipped.append(name)
                continue
            if name in {"rl_ppo", "rl_sac"} and not bool(SB3_AVAILABLE):
                skipped.append(name)
                continue
            if registry and name not in registry:
                skipped.append(name)
                continue
            filtered.append(name)

        if skipped:
            logger.info(f"Skipping unavailable models: {skipped}")

        return filtered

    def _maybe_shard_models(self, models: list[str]) -> list[str]:
        """
        Hybrid Parallelism Strategy:
        - Deep Learning (GPU-heavy): Run on ALL ranks (Data Parallel).
        - Machine Learning (CPU-heavy): Shard across ranks (Task Parallel).
        """
        try:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                if world_size > 1:
                    deep_models = {
                        "transformer", "kan", "nbeats", "tide", "tabnet",
                        "rl_ppo", "rl_sac", "rllib_ppo", "rllib_sac",
                        "genetic", "evolution" # GA runs independent islands, so all ranks ok
                    }

                    assigned = []
                    # Pre-compute sorted index map for O(1) lookup instead of O(n) per model
                    sorted_models = sorted(models)
                    index_map = {m: i for i, m in enumerate(sorted_models)}
                    for m in models:
                        if m in deep_models:
                            # Data Parallel: All ranks train this model together
                            # (Assuming model code handles DistributedSampler)
                            assigned.append(m)
                        else:
                            # Task Parallel: Split ML models across ranks
                            # deterministic assignment
                            idx = index_map[m]
                            if idx % world_size == rank:
                                assigned.append(m)

                    logger.info(f"Rank {rank}/{world_size} hybrid assignment: {assigned}")
                    return assigned
        except Exception as e:
            logger.warning(f"Model sharding check failed: {e}")
        return models

    def _train_blender(self, X, y):
        try:
            self.meta_blender = MetaBlender()
            X_m = X
            y_m = y
            if len(X_m) < 200 or len(y_m) < 200:
                return
            feats = pd.DataFrame(index=X_m.index)
            for name, m in self.models.items():
                p = self._pad_probs(m.predict_proba(X_m))
                feats[f"{name}_buy"] = p[:, 1]
                # ...
            feats["label"] = np.asarray(y_m, dtype=int)

            if hasattr(feats.index, "is_monotonic_increasing") and not feats.index.is_monotonic_increasing:
                feats = feats.sort_index(kind="mergesort")
            self.meta_blender.fit(feats)
            self.meta_blender.save(self.models_dir / "meta_blender.joblib")
        except Exception as e:
            logger.error(f"Meta-blender training failed: {e}", exc_info=True)

    def _ensemble_proba(self, X):
        probs = []
        for m in self.models.values():
            try:
                probs.append(self._pad_probs(m.predict_proba(X)))
            except Exception as e:
                logger.debug(f"Model prediction failed in ensemble: {e}")
        if not probs:
            return np.zeros((len(X), 3))
        return np.mean(probs, axis=0)

    @staticmethod
    def _pad_probs(p):
        if p is None:
            return np.zeros((0, 3))
        p = np.asarray(p)
        if p.ndim == 1:
            return np.zeros((0, 3))
        if p.shape[1] == 2:
            z = np.zeros((len(p), 3))
            z[:, 0] = p[:, 0]
            z[:, 1] = p[:, 1]  # Simplification
            return z
        return p[:, :3]

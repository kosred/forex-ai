import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

from ..domain.events import PreparedDataset
from .cpcv import cpcv_backtest

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Handles model evaluation strategies: Walk-Forward Analysis and CPCV.
    """

    def __init__(self, settings: Any, models_dir: Path):
        self.settings = settings
        self.models_dir = models_dir

    def run_walkforward(
        self, dataset: PreparedDataset, models: dict, ensemble_func, start_index: int = 0
    ) -> dict[str, Any]:
        """
        Sequential walk-forward evaluation using the current ensemble (no retraining).

        Args:
            dataset: The full dataset.
            models: Dictionary of trained models.
            ensemble_func: Function to generate predictions.
            start_index: Index to start evaluation from (e.g. end of training set).
                         Prevents leakage by skipping training data.
        """
        try:
            splits = max(1, int(getattr(self.settings.models, "walkforward_splits", 0)))
            if splits < 2:
                return {"walkforward_splits": 0}

            X = dataset.X
            y = pd.Series(dataset.y) if isinstance(dataset.y, (np.ndarray, list)) else dataset.y

            # Slice the dataset to only include data AFTER the start_index (OOS)
            if start_index >= len(X):
                logger.warning("Walk-forward start_index >= dataset length. Skipping.")
                return {"walkforward_splits": 0}

            X_oos = X.iloc[start_index:]
            y_oos = y.iloc[start_index:]
            n = len(y_oos)

            if n < 200:  # Minimum samples needed for meaningful splits
                return {"walkforward_splits": 0, "reason": "not_enough_oos_data"}

            window = max(1, n // splits)

            results = []
            for i in range(splits):
                start = i * window
                end = min(n, (i + 1) * window)
                if end - start < 80:
                    break

                # In this "no-retrain" mode, we treat the OOS data as a sequence of test blocks
                # We don't really have a "train" phase here because the models are static.
                # However, to simulate a rolling window eval, we just predict on the chunk.

                chunk_X = X_oos.iloc[start:end]
                chunk_y = y_oos.iloc[start:end]

                probas = ensemble_func(chunk_X)
                if probas is None or len(probas) == 0:
                    continue
                pred_labels = probas.argmax(axis=1)
                acc = float((pred_labels == chunk_y.to_numpy()).mean())
                results.append({"split": i + 1, "accuracy": acc})

            metrics = {
                "walkforward_splits": len(results),
                "avg_accuracy": float(np.mean([r["accuracy"] for r in results])) if results else 0.0,
                "splits": results,
            }

            # Save
            (self.models_dir / "walkforward_metrics.json").write_text(json.dumps(metrics, indent=2))
            return metrics

        except Exception as e:
            logger.warning(f"Walk-forward eval failed: {e}")
            return {}

    def run_cpcv(self, dataset: PreparedDataset, models: dict) -> dict[str, Any]:
        """Run Combinatorial Purged Cross-Validation."""
        if not self.settings.models.enable_cpcv:
            return {}

        try:
            logger.info("Running CPCV...")
            X = dataset.X
            y = dataset.y
            y = pd.Series(y, index=X.index) if isinstance(y, (np.ndarray, list)) else y
            meta = dataset.metadata

            # CPCV can easily explode memory on multi-million row datasets (scikit-learn casts to float64).
            # Keep it bounded by default; override via config or FOREX_BOT_CPCV_MAX_ROWS.
            max_rows_cfg = int(getattr(self.settings.models, "cpcv_max_rows", 250_000) or 0)
            env_override = str(os.environ.get("FOREX_BOT_CPCV_MAX_ROWS", "")).strip()
            if env_override:
                with contextlib.suppress(Exception):
                    max_rows_cfg = int(env_override)

            if max_rows_cfg > 0 and len(X) > max_rows_cfg:
                before = int(len(X))
                if isinstance(X.index, pd.DatetimeIndex):
                    idx_ns = X.index.view("int64")
                    nat = np.iinfo(np.int64).min
                    valid = idx_ns != nat
                    if valid.any():
                        times_valid = np.array(idx_ns[valid], copy=True)
                        start_k = max(0, min(len(times_valid) - 1, len(times_valid) - max_rows_cfg))
                        cut_ns = int(np.partition(times_valid, start_k)[start_k])
                        keep = idx_ns >= cut_ns
                        X = X.loc[keep]
                        y = y.loc[keep]
                        if meta is not None:
                            meta = meta.loc[keep]
                    else:
                        X = X.iloc[-max_rows_cfg:]
                        y = y.iloc[-max_rows_cfg:]
                        if meta is not None:
                            meta = meta.iloc[-max_rows_cfg:]
                else:
                    X = X.iloc[-max_rows_cfg:]
                    y = y.iloc[-max_rows_cfg:]
                    if meta is not None:
                        meta = meta.iloc[-max_rows_cfg:]

                logger.info(f"CPCV: Capped rows {before:,} -> {len(X):,} (max_rows={max_rows_cfg:,})")

            # Ensure time-ordering for CPCV purging/embargo logic.
            if hasattr(X.index, "is_monotonic_increasing") and not X.index.is_monotonic_increasing:
                X = X.sort_index(kind="mergesort")
                y = y.reindex(X.index)
                if meta is not None:
                    meta = meta.reindex(X.index)

            def model_factory():
                if LogisticRegression:
                    return LogisticRegression(max_iter=200, n_jobs=-1)
                return None

            if meta is not None:
                res = cpcv_backtest(
                    x=X,
                    y=y,
                    metadata=meta,
                    model_factory=model_factory,
                    n_splits=self.settings.models.cpcv_n_splits,
                    n_test_groups=self.settings.models.cpcv_n_test_groups,
                    embargo_pct=self.settings.models.cpcv_embargo_pct,
                    purge_pct=self.settings.models.cpcv_purge_pct,
                    n_jobs=1,
                )
                (self.models_dir / "cpcv_metrics.json").write_text(json.dumps(res, indent=2))
                return res
            return {}
        except Exception as e:
            logger.error(f"CPCV eval failed: {e}")
            return {}

import json
import logging
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

            def model_factory():
                if LogisticRegression:
                    return LogisticRegression(max_iter=200, n_jobs=-1)
                return None

            if hasattr(dataset, "metadata") and dataset.metadata is not None:
                res = cpcv_backtest(
                    x=X,
                    y=y,
                    metadata=dataset.metadata,
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
            logger.warning(f"CPCV eval failed: {e}")
            return {}

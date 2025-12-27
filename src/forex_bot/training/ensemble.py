import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..models.base import validate_time_ordering
from .evaluation import pad_probs

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
except ImportError:
    RandomForestClassifier = None
    GradientBoostingClassifier = None
    LogisticRegression = None

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


logger = logging.getLogger(__name__)


class MetaBlender:
    """Meta-level blending for specialist models (Stacking)."""

    def __init__(self) -> None:
        if XGB_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric="mlogloss", n_jobs=-1, random_state=42
            )
        elif GradientBoostingClassifier:
            self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        elif RandomForestClassifier:
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=10, n_jobs=-1, random_state=42
            )
        elif LogisticRegression:
            self.model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
        else:
            self.model = None

        self.feature_columns: pd.Index | None = None
        self.proba_classes: list[int] | None = None

    def fit(self, frame: pd.DataFrame, val_ratio: float = 0.15) -> dict[str, float]:
        """
        Fit meta-blender using time-series aware validation.

        IMPORTANT: The input frame should contain out-of-fold predictions from
        base models (obtained via cross-validation or time-series split).
        If base predictions were obtained on the same data used here,
        there will be data leakage.

        Parameters
        ----------
        frame : pd.DataFrame
            DataFrame with base model predictions and 'label' column
        val_ratio : float
            Fraction of data for validation (time-series split, not random)

        Returns
        -------
        dict
            Training metrics including validation accuracy
        """
        if self.model is None:
            return {"error": "No sklearn available"}

        dataset = frame.copy()
        if "label" not in dataset:
            raise ValueError("Dataset must include 'label' column")

        # Enforce chronological ordering for time-series splits (prevents look-ahead bias).
        if hasattr(dataset.index, "is_monotonic_increasing") and not dataset.index.is_monotonic_increasing:
            try:
                dataset = dataset.sort_index(kind="mergesort")
            except Exception as exc:
                raise ValueError(
                    "MetaBlender.fit: Data index is NOT monotonically increasing and could not be sorted. "
                    "Time-series models require chronologically ordered data to prevent look-ahead bias."
                ) from exc

        validate_time_ordering(dataset, context="MetaBlender.fit")

        labels_raw = dataset.pop("label").astype(int).to_numpy()
        # Backward compat: some older components used 2=sell; normalize to -1.
        labels_canon = np.where(labels_raw == 2, -1, labels_raw).astype(int, copy=False)
        unique = sorted({int(v) for v in labels_canon.tolist()})
        # Some estimators (notably XGBoost multi-class) require labels 0..K-1.
        to_model = {lab: i for i, lab in enumerate(unique)}
        y_model = np.array([to_model[int(v)] for v in labels_canon], dtype=int)

        if "symbol" in dataset:
            symbols = dataset.pop("symbol")
            symbol_one_hot = pd.get_dummies(symbols, prefix="sym", dtype=float)
            dataset = pd.concat([dataset.reset_index(drop=True), symbol_one_hot.reset_index(drop=True)], axis=1)
            
        # HPC FIX: Unsupervised Multi-Regime Distribution
        # Ingest the [p0...p7] probability vector from the GMM
        regime_cols = [c for c in dataset.columns if c.startswith("regime_p")]
        if regime_cols:
            for col in regime_cols:
                dataset[col] = dataset[col].astype(np.float32)
        elif "regime" in dataset:
            # Fallback for legacy categorical
            dataset["regime"] = dataset["regime"].astype("category")
            if XGB_AVAILABLE and hasattr(self.model, "enable_categorical"):
                self.model.set_params(enable_categorical=True)

        features = dataset.reset_index(drop=True).fillna(0.0)
        self.feature_columns = features.columns

        # Time-series train/val split (no random shuffle!)
        n = len(features)
        val_size = max(10, int(n * val_ratio))
        train_end = n - val_size

        X_train = features.iloc[:train_end].to_numpy()
        y_train = y_model[:train_end]
        X_val = features.iloc[train_end:].to_numpy()
        y_val = y_model[train_end:]

        self.model.fit(X_train, y_train)
        try:
            classes_model = [int(c) for c in list(self.model.classes_)]
            self.proba_classes = [unique[c] for c in classes_model]
        except Exception:
            self.proba_classes = unique

        # Training accuracy (in-sample, for diagnostics only)
        train_preds = self.model.predict(X_train)
        train_accuracy = float(np.mean(train_preds == y_train)) if len(y_train) else 0.0

        # Validation accuracy (out-of-sample, the real metric)
        val_preds = self.model.predict(X_val)
        val_accuracy = float(np.mean(val_preds == y_val)) if len(y_val) else 0.0

        logger.info(
            f"MetaBlender: train_acc={train_accuracy:.3f}, val_acc={val_accuracy:.3f}, "
            f"train_n={len(y_train)}, val_n={len(y_val)}"
        )

        return {
            "samples": int(len(y_model)),
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "accuracy": val_accuracy,  # Report val accuracy as primary metric
            "classes": list(self.proba_classes or []),
        }

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.feature_columns is None:
            raise RuntimeError("MetaBlender has not been fitted")
        
        # HPC Optimization: Direct NumPy Mapping
        # Instead of copying the whole dataframe, we map only the columns we need.
        input_cols = list(frame.columns)
        n_samples = len(frame)
        X_np = np.zeros((n_samples, len(self.feature_columns)), dtype=np.float32)
        
        for i, col in enumerate(self.feature_columns):
            if col in input_cols:
                # Handle categorical regime properly
                if col == "regime":
                    X_np[:, i] = frame[col].astype(np.float32).to_numpy()
                else:
                    X_np[:, i] = frame[col].to_numpy(dtype=np.float32)
            # symbol handling (one-hot)
            elif col.startswith("sym_") and "symbol" in input_cols:
                sym_name = col[4:]
                X_np[:, i] = (frame["symbol"] == sym_name).astype(np.float32)
        
        raw = self.model.predict_proba(X_np)
        return pad_probs(raw, classes=self.proba_classes)

    def save(self, path: Path) -> None:
        payload = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "proba_classes": self.proba_classes,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path) -> "MetaBlender":
        payload = joblib.load(path)
        instance = cls()
        instance.model = payload["model"]
        instance.feature_columns = payload.get("feature_columns")
        instance.proba_classes = payload.get("proba_classes")
        return instance

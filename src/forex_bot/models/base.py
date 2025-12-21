"""
Base classes and utilities for machine learning models.

This module provides:
- EarlyStopper: Universal early stopping for training loops
- ExpertModel: Abstract base class for all expert models
- Training utilities for time-series aware data handling
"""

import abc
import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EarlyStopper:
    """
    Universal Early Stopping utility.
    Stops training when validation metric stops improving.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


class ExpertModel(abc.ABC):
    """Abstract base class for all expert models."""

    @abc.abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass

    @abc.abstractmethod
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for classes [-1, 0, 1].

        Returns:
            np.ndarray: Shape (N, 3) where columns map to [neutral, buy, sell]
                        or [-1, 0, 1] depending on implementation agreement.
                        Convention here: col 0 -> neutral, col 1 -> buy, col 2 -> sell
                        (or whatever signals.py expects, usually mapped by LABEL_FWD)
        """
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Save model artifacts to directory."""
        pass

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load model artifacts from directory."""
        pass

    def _atomic_save(self, save_func: Callable[[str], None], target_path: str) -> None:
        """
        Helper for atomic model saving with rotation/backup.
        Keeps 'model.pt' (current) and 'model.pt.bak' (previous).
        """
        import shutil
        from pathlib import Path

        path_obj = Path(target_path)
        temp_path = path_obj.with_suffix(path_obj.suffix + ".tmp")
        backup_path = path_obj.with_suffix(path_obj.suffix + ".bak")

        try:
            save_func(str(temp_path))

            if path_obj.exists():
                if backup_path.exists():
                    os.remove(backup_path)  # Delete old backup
                shutil.move(str(path_obj), str(backup_path))  # Current -> Backup

            shutil.move(str(temp_path), str(path_obj))  # Temp -> Current

        except Exception as e:
            if temp_path.exists():
                os.remove(temp_path)
            raise RuntimeError(f"Atomic save failed for {target_path}: {e}") from e


def dataframe_to_float32_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Convert a DataFrame to a float32 NumPy array suitable for torch.

    Pandas Copy-on-Write can return read-only views; torch warns about creating tensors
    from non-writable arrays. We only copy when needed.
    """
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr


def validate_time_ordering(df: pd.DataFrame, context: str = "") -> bool:
    """
    Validate that DataFrame index is monotonically increasing (time-ordered).

    This is critical for time-series models to prevent look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex or sortable index
    context : str
        Context string for error messages

    Returns
    -------
    bool
        True if valid, raises ValueError if not

    Raises
    ------
    ValueError
        If index is not monotonically increasing
    """
    if df is None or len(df) == 0:
        return True

    if hasattr(df.index, 'is_monotonic_increasing'):
        if not df.index.is_monotonic_increasing:
            # Check if it's just not strictly monotonic (duplicates OK)
            if hasattr(df.index, 'is_monotonic'):
                if df.index.is_monotonic:
                    logger.warning(
                        f"{context}: Index has duplicate timestamps but is sorted. "
                        "Consider deduplicating for cleaner time-series handling."
                    )
                    return True

            raise ValueError(
                f"{context}: Data index is NOT monotonically increasing. "
                "Time-series models require chronologically ordered data to prevent look-ahead bias. "
                f"Index range: {df.index[0]} to {df.index[-1]}"
            )
    return True


def time_series_train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_ratio: float = 0.15,
    min_train_samples: int = 100,
    embargo_samples: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-series aware train/validation split.

    Unlike random splits, this preserves temporal ordering:
    - Training set: earlier portion of data
    - Validation set: later portion of data
    - Optional embargo: gap between train and val to prevent leakage

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with DatetimeIndex
    y : pd.Series
        Labels aligned with X
    val_ratio : float
        Fraction of data for validation (default 0.15)
    min_train_samples : int
        Minimum samples required in training set
    embargo_samples : int
        Number of samples to skip between train and val (prevents leakage)

    Returns
    -------
    tuple
        (X_train, X_val, y_train, y_val)

    Raises
    ------
    ValueError
        If data is not time-ordered or too small
    """
    validate_time_ordering(X, context="time_series_train_val_split")

    n = len(X)
    if n < min_train_samples + 10:
        raise ValueError(
            f"Dataset too small for time-series split: {n} samples, "
            f"need at least {min_train_samples + 10}"
        )

    # Calculate split point
    val_size = max(10, int(n * val_ratio))
    train_end = n - val_size - embargo_samples

    if train_end < min_train_samples:
        # Reduce embargo to fit
        embargo_samples = max(0, n - val_size - min_train_samples)
        train_end = n - val_size - embargo_samples
        logger.warning(
            f"Reduced embargo to {embargo_samples} samples to maintain minimum training size"
        )

    if train_end < min_train_samples:
        raise ValueError(
            f"Cannot create valid split: train_end={train_end} < min_train_samples={min_train_samples}"
        )

    val_start = train_end + embargo_samples

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[val_start:]
    y_val = y.iloc[val_start:]

    logger.debug(
        f"Time-series split: train={len(X_train)}, embargo={embargo_samples}, val={len(X_val)}"
    )

    return X_train, X_val, y_train, y_val


def stratified_downsample(
    X: pd.DataFrame,
    y: pd.Series,
    max_samples: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Downsample data while preserving class distribution.

    Used to limit memory/compute for large datasets while maintaining
    representative class balance.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Labels
    max_samples : int
        Maximum samples to keep (0 = no limit)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (X_downsampled, y_downsampled)
    """
    if max_samples <= 0 or len(X) <= max_samples:
        return X, y

    rng = np.random.default_rng(random_state)

    # Group by class
    class_indices: dict[Any, list[int]] = {}
    for idx, label in enumerate(y):
        label_key = int(label) if isinstance(label, (int, float, np.integer, np.floating)) else label
        if label_key not in class_indices:
            class_indices[label_key] = []
        class_indices[label_key].append(idx)

    # Calculate samples per class (proportional)
    total = len(y)
    sampled_indices: list[int] = []

    for _label, indices in class_indices.items():
        # Proportion of this class in original data
        class_ratio = len(indices) / total
        # Target samples for this class
        target_count = max(1, int(max_samples * class_ratio))
        # Actual samples to take
        take_count = min(len(indices), target_count)

        if take_count > 0:
            sampled = rng.choice(indices, size=take_count, replace=False)
            sampled_indices.extend(sampled.tolist())

    # Trim to max if over
    if len(sampled_indices) > max_samples:
        sampled_indices = rng.choice(
            sampled_indices, size=max_samples, replace=False
        ).tolist()

    # Sort to maintain temporal order
    sampled_indices.sort()

    X_out = X.iloc[sampled_indices]
    y_out = y.iloc[sampled_indices]

    logger.info(
        f"Downsampled from {len(X)} to {len(X_out)} samples "
        f"({len(X_out)/len(X)*100:.1f}%)"
    )

    return X_out, y_out


def compute_class_weights(y: pd.Series | np.ndarray) -> dict[int, float]:
    """
    Compute balanced class weights for imbalanced classification.

    Uses inverse frequency weighting: rare classes get higher weights.

    Parameters
    ----------
    y : array-like
        Class labels

    Returns
    -------
    dict
        Mapping from class label to weight
    """
    y_arr = np.asarray(y)
    unique, counts = np.unique(y_arr, return_counts=True)
    n_samples = len(y_arr)
    n_classes = len(unique)

    weights = {}
    for cls, count in zip(unique, counts, strict=False):
        if count > 0:
            # sklearn-style balanced weight
            weights[int(cls)] = n_samples / (n_classes * count)

    return weights


def compute_sample_weights(y: pd.Series | np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights based on class frequency.

    Parameters
    ----------
    y : array-like
        Class labels

    Returns
    -------
    np.ndarray
        Weight for each sample
    """
    class_weights = compute_class_weights(y)
    y_arr = np.asarray(y)

    sample_weights = np.ones(len(y_arr), dtype=np.float32)
    for cls, weight in class_weights.items():
        sample_weights[y_arr == cls] = weight

    return sample_weights


def detect_feature_drift(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    threshold: float = 0.1,
    method: str = "psi",
) -> dict[str, Any]:
    """
    Detect feature drift between training and validation data.

    Uses Population Stability Index (PSI) or simple mean/std comparison
    to identify features that have shifted significantly.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training features
    val_df : pd.DataFrame
        Validation/test features
    threshold : float
        Drift threshold (PSI > threshold indicates significant drift)
        - PSI < 0.1: No drift
        - 0.1 <= PSI < 0.25: Moderate drift (warning)
        - PSI >= 0.25: Significant drift (critical)
    method : str
        Detection method: "psi" or "stats"

    Returns
    -------
    dict
        {
            "drifted_features": list of features with drift,
            "drift_scores": dict mapping feature -> drift score,
            "summary": overall drift summary,
            "critical": bool indicating if drift is critical
        }
    """
    if train_df is None or val_df is None or len(train_df) == 0 or len(val_df) == 0:
        return {
            "drifted_features": [],
            "drift_scores": {},
            "summary": "Insufficient data for drift detection",
            "critical": False,
        }

    # Find common numeric columns
    common_cols = list(set(train_df.columns) & set(val_df.columns))
    numeric_cols = [
        c for c in common_cols
        if train_df[c].dtype.kind in ('f', 'i') and val_df[c].dtype.kind in ('f', 'i')
    ]

    if not numeric_cols:
        return {
            "drifted_features": [],
            "drift_scores": {},
            "summary": "No numeric features to check",
            "critical": False,
        }

    drift_scores: dict[str, float] = {}
    drifted_features: list[str] = []

    for col in numeric_cols:
        try:
            train_vals = train_df[col].dropna().values
            val_vals = val_df[col].dropna().values

            if len(train_vals) < 10 or len(val_vals) < 10:
                continue

            if method == "psi":
                score = _compute_psi(train_vals, val_vals)
            else:
                # Simple stats comparison
                train_mean, train_std = np.mean(train_vals), np.std(train_vals)
                val_mean, val_std = np.mean(val_vals), np.std(val_vals)

                if train_std > 1e-9:
                    mean_shift = abs(val_mean - train_mean) / train_std
                    std_ratio = val_std / train_std if train_std > 0 else 1.0
                    score = mean_shift + abs(1.0 - std_ratio)
                else:
                    score = 0.0

            drift_scores[col] = float(score)

            if score >= threshold:
                drifted_features.append(col)

        except Exception:
            continue

    # Calculate overall drift severity
    critical_threshold = 0.25  # PSI >= 0.25 is significant
    critical_count = sum(1 for s in drift_scores.values() if s >= critical_threshold)
    total_features = len(drift_scores)

    if critical_count > total_features * 0.3:  # >30% features have critical drift
        critical = True
        summary = f"CRITICAL: {critical_count}/{total_features} features have significant drift"
    elif len(drifted_features) > total_features * 0.2:  # >20% have moderate drift
        critical = False
        summary = f"WARNING: {len(drifted_features)}/{total_features} features show drift"
    else:
        critical = False
        summary = f"OK: {len(drifted_features)}/{total_features} features with minor drift"

    if drifted_features:
        logger.warning(
            f"Feature drift detected: {summary}. Top drifted: "
            f"{sorted(drifted_features, key=lambda x: drift_scores.get(x, 0), reverse=True)[:5]}"
        )

    return {
        "drifted_features": drifted_features,
        "drift_scores": drift_scores,
        "summary": summary,
        "critical": critical,
    }


def _compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between two distributions.

    PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))

    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change
    - PSI >= 0.25: Significant change
    """
    eps = 1e-6

    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates

    if len(breakpoints) < 2:
        return 0.0

    # Compute bin counts
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to percentages
    expected_pct = expected_counts / (len(expected) + eps)
    actual_pct = actual_counts / (len(actual) + eps)

    # Avoid division by zero
    expected_pct = np.clip(expected_pct, eps, 1.0)
    actual_pct = np.clip(actual_pct, eps, 1.0)

    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)

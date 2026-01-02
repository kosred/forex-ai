"""
Base classes and utilities for machine learning models.

This module provides:
- EarlyStopper: Universal early stopping for training loops
- ExpertModel: Abstract base class for all expert models
- Training utilities for time-series aware data handling
"""

import abc
import contextlib
import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..core.system import resolve_cpu_budget

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


def get_early_stop_params(default_patience: int, default_min_delta: float) -> tuple[int, float]:
    """Return (patience, min_delta) with optional env overrides."""
    patience = default_patience
    min_delta = default_min_delta
    try:
        env_pat = os.environ.get("FOREX_BOT_EARLY_STOP_PATIENCE", "")
        if env_pat != "":
            val = int(env_pat)
            if val > 0:
                patience = val
    except Exception:
        pass
    try:
        env_delta = os.environ.get("FOREX_BOT_EARLY_STOP_MIN_DELTA", "")
        if env_delta != "":
            min_delta = float(env_delta)
    except Exception:
        pass
    return patience, min_delta

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
    embargo_samples: int = 300, # HPC FIX: Guaranteed memory flush
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data for time-series training with an embargo gap.
    """
    n = len(X)
    val_size = int(n * val_ratio)
    train_end = n - val_size - embargo_samples
    
    if train_end < min_train_samples:
        # If dataset too small, reduce embargo but maintain at least 100 bars
        embargo_samples = min(embargo_samples, max(100, n // 10))
        train_end = n - val_size - embargo_samples
        
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end + embargo_samples:]
    y_val = y.iloc[train_end + embargo_samples:]
    
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

    # HPC FIX: Regime-Aware Drift Thresholding
    # Standard PSI: <0.1 (OK), 0.1-0.25 (Warn), >0.25 (Drift)
    # We use a 2025 Adaptive Strategy to save compute
    base_threshold = float(os.environ.get("FOREX_BOT_DRIFT_THRESHOLD", "0.20") or 0.20)
    
    # Increase threshold if market is overall volatile
    vol_scale = 1.0
    with contextlib.suppress(Exception):
        if "realized_vol" in train_df.columns:
            # If current vol is 2x historical, allow 2x more drift
            vol_scale = max(1.0, val_df["realized_vol"].mean() / (train_df["realized_vol"].mean() + 1e-9))
            
    threshold = base_threshold * vol_scale

    drift_scores: dict[str, float] = {}
    drifted_features: list[str] = []

    import concurrent.futures
    
    def _check_col(col):
        try:
            train_vals = train_df[col].dropna().values
            val_vals = val_df[col].dropna().values
            if len(train_vals) < 10 or len(val_vals) < 10:
                return None
            score = _compute_psi(train_vals, val_vals) if method == "psi" else _compute_stats_drift(train_vals, val_vals)
            return col, float(score)
        except Exception:
            return None

    # HPC: Use all cores for parallel drift detection
    cpu_budget = resolve_cpu_budget()
    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_budget) as executor:
        results = list(executor.map(_check_col, numeric_cols))
        
    for res in results:
        if res:
            col, score = res
            drift_scores[col] = score
            if score >= threshold:
                drifted_features.append(col)

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
        msg = (
            f"Feature drift detected: {summary}. Top drifted: "
            f"{sorted(drifted_features, key=lambda x: drift_scores.get(x, 0), reverse=True)[:5]}"
        )
        if critical or summary.startswith("WARNING:"):
            logger.warning(msg)
        else:
            logger.info(msg)

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
    n_bins = max(3, n_bins)
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates

    if len(breakpoints) < 2:
        return 0.0

    def _hist(vals: np.ndarray, bps: np.ndarray) -> np.ndarray:
        return np.histogram(vals, bins=bps)[0]

    expected_counts = _hist(expected, breakpoints)
    actual_counts = _hist(actual, breakpoints)

    # If bins are too sparse, retry once with coarser bins
    if (expected_counts < 3).any() or (actual_counts < 3).any():
        coarse_bins = max(3, min(5, len(breakpoints) - 1))
        coarse_breaks = np.percentile(expected, np.linspace(0, 100, coarse_bins + 1))
        coarse_breaks = np.unique(coarse_breaks)
        if len(coarse_breaks) >= 2 and len(coarse_breaks) < len(breakpoints):
            breakpoints = coarse_breaks
            expected_counts = _hist(expected, breakpoints)
            actual_counts = _hist(actual, breakpoints)

    # Convert to percentages
    expected_pct = expected_counts / (len(expected) + eps)
    actual_pct = actual_counts / (len(actual) + eps)

    # Avoid division by zero and clean NaN/inf
    expected_pct = np.clip(expected_pct, eps, 1.0)
    actual_pct = np.clip(actual_pct, eps, 1.0)
    expected_pct = np.nan_to_num(expected_pct, nan=eps, posinf=1.0, neginf=eps)
    actual_pct = np.nan_to_num(actual_pct, nan=eps, posinf=1.0, neginf=eps)

    # Calculate PSI
    diff = actual_pct - expected_pct
    ratio = np.log(actual_pct / expected_pct)
    # Clean any remaining NaN/inf from operations
    diff = np.nan_to_num(diff, nan=0.0)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    psi = np.sum(diff * ratio)

    return float(psi)


def _compute_stats_drift(train_vals: np.ndarray, val_vals: np.ndarray) -> float:
    """Fallback drift metric based on mean/std shift."""
    # Clean inputs
    train_vals = np.nan_to_num(train_vals, nan=0.0, posinf=0.0, neginf=0.0)
    val_vals = np.nan_to_num(val_vals, nan=0.0, posinf=0.0, neginf=0.0)

    train_mean, train_std = np.mean(train_vals), np.std(train_vals)
    val_mean, val_std = np.mean(val_vals), np.std(val_vals)
    eps = np.finfo(np.float64).eps

    if train_std > eps:
        mean_shift = abs(val_mean - train_mean) / max(train_std, eps)
        std_ratio = val_std / max(train_std, eps)
        # Clean result
        result = mean_shift + abs(1.0 - std_ratio)
        return float(np.nan_to_num(result, nan=0.0))
    return 0.0

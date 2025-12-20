import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class ConceptDriftMonitor:
    """
    Monitors prediction error stream to detect concept drift.
    Uses a simplified ADWIN-like approach: tracking variance in two sub-windows.
    """

    def __init__(self, window_size: int = 100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.error_stream = deque(maxlen=window_size * 2)
        self.mean_error = 0.0
        self.variance = 0.0
        self.drift_detected = False
        self.last_drift_at: datetime | None = None

        # 2025 enhancement: Track drift magnitude and severity
        self.drift_magnitude = 0.0  # 0-1 score of drift severity
        self.drift_method_used = ""  # Which test detected drift
        self.ks_statistic = 0.0
        self.psi_score = 0.0
        self.kl_divergence = 0.0

    def update(self, y_true: int, y_pred_prob: np.ndarray) -> bool:
        """
        Update monitor with new prediction outcome.
        y_true: -1 (Sell), 0 (Neutral), 1 (Buy)
        y_pred_prob: Probability distribution [p_neutral, p_buy, p_sell]
        """
        try:
            y_true = int(y_true)
        except Exception:
            y_true = 0

        if y_true == 0:
            idx = 0
        elif y_true == 1:
            idx = 1
        elif y_true == -1:
            idx = 2
        else:
            idx = 0

        prob_correct = float(np.asarray(y_pred_prob, dtype=float)[idx])
        error = 1.0 - prob_correct

        self.error_stream.append(error)

        try:
            self.mean_error = float(np.mean(self.error_stream))
            self.variance = float(np.var(self.error_stream))
        except Exception as e:
            logger.warning(f"Drift monitoring failed: {e}", exc_info=True)

        if len(self.error_stream) >= self.window_size:
            self.drift_detected = self._check_drift()
            if self.drift_detected:
                self.last_drift_at = datetime.now(UTC)

        return self.drift_detected

    def _check_drift(self) -> bool:
        """
        Multi-method drift detection using modern statistical tests (2025 best practices).
        Uses ensemble of: variance-based, KS test, PSI, and KL divergence.
        """
        n = len(self.error_stream)
        if n < self.window_size * 2:
            return False

        mid = n // 2
        w1 = np.array(list(self.error_stream)[:mid])
        w2 = np.array(list(self.error_stream)[mid:])

        # Method 1: Original variance-based approach
        mu1 = np.mean(w1)
        mu2 = np.mean(w2)
        delta = np.abs(mu1 - mu2)
        sigma = np.sqrt(self.variance) if self.variance > 0 else 0.01
        dynamic_threshold = max(self.threshold, 3.0 * sigma)
        variance_drift = delta > dynamic_threshold

        # Method 2: Kolmogorov-Smirnov test (distribution comparison)
        try:
            ks_stat, ks_pval = ks_2samp(w1, w2)
            self.ks_statistic = float(ks_stat)
            ks_drift = ks_pval < 0.05  # 95% confidence
        except Exception:
            ks_drift = False
            self.ks_statistic = 0.0

        # Method 3: Population Stability Index (PSI) - financial industry standard
        try:
            self.psi_score = self._calculate_psi(w1, w2)
            # PSI thresholds: <0.1 = no change, 0.1-0.25 = moderate, >0.25 = significant
            psi_drift = self.psi_score > 0.25
        except Exception:
            psi_drift = False
            self.psi_score = 0.0

        # Method 4: KL Divergence (distribution distance)
        try:
            self.kl_divergence = self._calculate_kl_divergence(w1, w2)
            # KL > 0.1 indicates significant distribution shift
            kl_drift = self.kl_divergence > 0.1
        except Exception:
            kl_drift = False
            self.kl_divergence = 0.0

        # Ensemble decision: Drift if 2+ methods agree
        drift_votes = sum([variance_drift, ks_drift, psi_drift, kl_drift])
        drift_detected = drift_votes >= 2

        # Calculate drift magnitude (0-1 normalized)
        self.drift_magnitude = min(
            1.0,
            (
                (delta / (dynamic_threshold + 1e-9)) * 0.25
                + self.ks_statistic * 0.25
                + min(self.psi_score / 0.5, 1.0) * 0.25
                + min(self.kl_divergence / 0.3, 1.0) * 0.25
            ),
        )

        # Track which method triggered
        if drift_detected:
            methods = []
            if variance_drift:
                methods.append("variance")
            if ks_drift:
                methods.append("KS")
            if psi_drift:
                methods.append("PSI")
            if kl_drift:
                methods.append("KL")
            self.drift_method_used = "+".join(methods)
            logger.warning(f"Drift detected by {self.drift_method_used} (magnitude={self.drift_magnitude:.3f})")

        return drift_detected

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        PSI measures distribution shift - financial industry standard.
        """
        try:
            # Create bins from expected distribution
            breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)  # Remove duplicates

            if len(breakpoints) < 2:
                return 0.0

            # Count observations in each bin
            expected_counts = np.histogram(expected, bins=breakpoints)[0]
            actual_counts = np.histogram(actual, bins=breakpoints)[0]

            # Convert to percentages with smoothing
            expected_pct = (expected_counts + 1e-6) / (expected_counts.sum() + bins * 1e-6)
            actual_pct = (actual_counts + 1e-6) / (actual_counts.sum() + bins * 1e-6)

            # PSI formula: sum((actual% - expected%) * ln(actual% / expected%))
            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

            return abs(float(psi))
        except Exception as e:
            logger.debug(f"PSI calculation failed: {e}")
            return 0.0

    def _calculate_kl_divergence(self, p_samples: np.ndarray, q_samples: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Kullback-Leibler divergence between two distributions.
        KL(P||Q) measures how much Q diverges from P.
        """
        try:
            # Create histogram-based distributions
            breakpoints = np.linspace(
                min(p_samples.min(), q_samples.min()), max(p_samples.max(), q_samples.max()), bins + 1
            )

            p_counts = np.histogram(p_samples, bins=breakpoints)[0]
            q_counts = np.histogram(q_samples, bins=breakpoints)[0]

            # Normalize to probabilities with smoothing
            p_prob = (p_counts + 1e-9) / (p_counts.sum() + bins * 1e-9)
            q_prob = (q_counts + 1e-9) / (q_counts.sum() + bins * 1e-9)

            # KL divergence formula: sum(P * log(P/Q))
            kl = np.sum(p_prob * np.log(p_prob / q_prob))

            return float(kl)
        except Exception as e:
            logger.debug(f"KL divergence calculation failed: {e}")
            return 0.0

    def should_retrain(self) -> bool:
        """Check if retraining is recommended."""
        return self.drift_detected

    def reset_after_retrain(self) -> None:
        """Reset monitor state after a retraining event."""
        self.drift_detected = False
        self.error_stream.clear()
        self.mean_error = 0.0
        self.variance = 0.0
        self.drift_magnitude = 0.0
        self.drift_method_used = ""
        logger.info("Drift monitor reset after retraining.")

    def initialize_feature_monitor(self, baseline_features: pd.DataFrame, symbol: str):
        """
        Initialize adaptive feature monitor.
        Uses EMA to track shifting mean/std of features (handling non-stationarity).
        """
        self.feature_stats = {}
        self.alpha = 0.01  # Adaptation rate (approx 100-bar memory)

        monitor_cols = [
            c
            for c in baseline_features.columns
            if "rsi" in c or "adx" in c or "atr" in c or "ema" in c or "return" in c
        ]
        if not monitor_cols:
            monitor_cols = baseline_features.columns[:10].tolist()

        for col in monitor_cols:
            try:
                series = baseline_features[col].dropna()
                if len(series) > 10:
                    self.feature_stats[col] = {
                        "mean": float(series.mean()),
                        "std": float(series.std()) + 1e-9,
                        "initialized": True,
                    }
            except Exception as e:
                logger.warning(f"Drift monitoring failed: {e}", exc_info=True)
        logger.info(f"Adaptive Feature Monitor initialized for {symbol}. Tracking {len(self.feature_stats)} features.")

    def check_feature_drift(self, current_features: pd.DataFrame) -> bool:
        """
        Check for drift using Z-score against ADAPTIVE statistics.
        Updates statistics online (EMA) to follow market regimes.
        Returns True if sudden shock detected (drift).
        """
        if not hasattr(self, "feature_stats") or not self.feature_stats:
            return False

        drift_count = 0
        total_checked = 0

        try:
            if len(current_features) == 0:
                return False
            row = current_features.iloc[-1]

            for col, stats in self.feature_stats.items():
                if col not in row:
                    continue

                val = float(row[col])

                z_score = abs(val - stats["mean"]) / stats["std"]

                total_checked += 1
                if z_score > 4.0:  # 4-sigma is a significant shock
                    drift_count += 1

                diff = val - stats["mean"]
                incr = self.alpha * diff
                stats["mean"] += incr

                var_old = stats["std"] ** 2
                var_new = (1 - self.alpha) * var_old + self.alpha * (diff**2)
                stats["std"] = np.sqrt(var_new) + 1e-9

            if total_checked == 0:
                return False

            drift_ratio = drift_count / total_checked
            return drift_ratio > 0.30

        except Exception as e:
            logger.warning(f"Adaptive drift check failed: {e}")
            return False

    def status(self) -> dict[str, Any]:
        return {
            "drift_detected": bool(self.drift_detected),
            "drift_magnitude": float(self.drift_magnitude),
            "drift_method": self.drift_method_used,
            "errors_tracked": len(self.error_stream),
            "last_drift_at": self.last_drift_at.isoformat() if self.last_drift_at else None,
            "ks_statistic": float(self.ks_statistic),
            "psi_score": float(self.psi_score),
            "kl_divergence": float(self.kl_divergence),
        }


def get_drift_monitor(cache_dir):
    return ConceptDriftMonitor()

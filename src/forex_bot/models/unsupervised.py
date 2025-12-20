import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base import ExpertModel  # Import ExpertModel for compatibility

logger = logging.getLogger(__name__)


class MarketRegimeClassifier(ExpertModel):  # Inherit from ExpertModel
    """
    Unsupervised learning component that classifies market conditions
    into discrete regimes (e.g., Low Volatility Bull, High Volatility Bear, Choppy Range).
    Uses K-Means clustering on Returns, Volatility, and ADX.
    """

    def __init__(self, n_regimes: int = 4, lookback: int = 1000, **kwargs):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = KMeans(n_clusters=n_regimes, random_state=42, n_init="auto")
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_map = {}  # Map cluster ID to human-readable label

    def fit(self, df: pd.DataFrame, y=None, **kwargs) -> None:
        """
        Fit the clustering model on historical data.
        Features: Log Returns, Realized Volatility, ADX (Trend Strength).
        Compatible with ExpertModel interface (ignores y).
        """
        if len(df) < self.lookback:
            logger.warning(f"Insufficient data for regime classification. Need {self.lookback}, got {len(df)}.")
            return

        features = self._extract_features(df)
        if features.empty:
            return

        # Scale features
        X = self.scaler.fit_transform(features)

        # Cluster
        self.model.fit(X)
        self.is_fitted = True

        # Analyze clusters to assign labels
        labels = self.model.labels_
        features["cluster"] = labels
        summary = features.groupby("cluster").mean()

        # Heuristic Labeling
        for i in range(self.n_regimes):
            vol = summary.loc[i, "volatility"]
            adx = summary.loc[i, "adx"]
            ret = summary.loc[i, "returns"]

            label = "Normal"
            if vol < -0.5:
                label = "Quiet"
            elif vol > 0.5:
                label = "Volatile"

            if adx > 0.5:
                label += " Trend"
            else:
                label += " Range"

            if ret > 0.2:
                label += " (Bull)"
            elif ret < -0.2:
                label += " (Bear)"

            self.regime_map[i] = label

        logger.info(f"Regime Classifier Fitted. Regimes: {self.regime_map}")

    def predict(self, df: pd.DataFrame) -> str:
        """
        Predict the current market regime based on the latest data.
        Returns: String label (e.g. "Volatile Trend (Bear)")
        """
        if not self.is_fitted:
            return "Unknown"

        try:
            feat = self._extract_features(df.tail(20)).iloc[[-1]]  # Just the last row
            X = self.scaler.transform(feat)
            cluster = self.model.predict(X)[0]
            return self.regime_map.get(cluster, "Unknown")
        except Exception as e:
            logger.warning(f"Regime prediction failed: {e}")
            return "Error"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Mock implementation for ExpertModel compatibility.
        Does not actually predict buy/sell signals, just used for regime detection.
        Returns neutral probabilities.
        """
        n = len(X)
        probs = np.zeros((n, 3))
        probs[:, 0] = 1.0  # All Neutral
        return probs

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute minimal feature set for clustering.
        """
        try:
            # Safe feature extraction
            closes = df["close"]
            returns = np.log(closes / closes.shift(1)).fillna(0.0)
            volatility = returns.rolling(20).std().fillna(0.0)

            # Use 'adx' column if pre-computed, else approx with vol
            adx = df["adx"] if "adx" in df.columns else volatility * 100

            data = pd.DataFrame({"returns": returns, "volatility": volatility, "adx": adx})
            # Drop NaN from rolling windows
            return data.dropna()
        except Exception:
            return pd.DataFrame()

    def save(self, path: str) -> None:
        if self.is_fitted:
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            joblib.dump(self, p / "regime_classifier.joblib")

    @staticmethod
    def load(path: str) -> "MarketRegimeClassifier | None":
        p = Path(path) / "regime_classifier.joblib"
        if p.exists():
            return joblib.load(p)
        return None

    # Alias for registry compatibility
    load_model = load


# Alias for registry
ClusterExpert = MarketRegimeClassifier

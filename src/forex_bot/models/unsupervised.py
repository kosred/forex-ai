import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    prange = range

from .base import ExpertModel

logger = logging.getLogger(__name__)

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, parallel=True)
    def _rolling_std_numba(data, window):
        n = len(data)
        out = np.zeros(n, dtype=np.float32)
        for i in prange(window, n):
            chunk = data[i-window+1 : i+1]
            out[i] = np.std(chunk)
        return out
else:
    def _rolling_std_numba(data, window):
        return pd.Series(data).rolling(window).std().fillna(0.0).values.astype(np.float32)

from sklearn.mixture import GaussianMixture

class MarketRegimeClassifier(ExpertModel):
    """
    HPC FIX: Fully Unsupervised Gaussian Regime Discovery.
    Discovers the latent 'Hidden States' of the market without human labels.
    """

    def __init__(self, n_regimes: int = 8, **kwargs):
        # Increased to 8 regimes to capture more subtle market anomalies
        self.n_regimes = n_regimes
        self.model = GaussianMixture(n_components=n_regimes, covariance_type='full', random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, y=None, **kwargs) -> None:
        features = self._extract_features(df)
        if features.empty: return

        X = self.scaler.fit_transform(features)
        self.model.fit(X)
        self.is_fitted = True
        logger.info(f"Unsupervised GMM fitted: {self.n_regimes} latent regimes discovered.")

    def predict_regime_distribution(self, df: pd.DataFrame) -> np.ndarray:
        """
        HPC FIX: Multi-Regime Posterior Distribution.
        Returns a vector of probabilities [p0, p1, ..., p7].
        """
        if not self.is_fitted:
            return np.zeros(self.n_regimes)
        try:
            # Process entire window to get a stable distribution
            feat = self._extract_features(df.tail(50))
            if feat.empty: return np.zeros(self.n_regimes)
            
            X = self.scaler.transform(feat)
            # Use posteriors (soft assignment)
            probs = self.model.predict_proba(X)
            # Return the latest posterior
            return probs[-1]
        except Exception:
            return np.zeros(self.n_regimes)

    def predict(self, df: pd.DataFrame) -> str:
        """Fallback for legacy components: returns the 'primary' regime ID."""
        dist = self.predict_regime_distribution(df)
        return str(np.argmax(dist))

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
        HPC Optimized: Feature extraction using Numba.
        """
        try:
            closes = df["close"].to_numpy(dtype=np.float32)
            # Log Returns (Vectorized)
            returns = np.log(closes[1:] / closes[:-1])
            returns = np.concatenate([[0.0], returns])
            
            # HPC Rolling Volatility
            volatility = _rolling_std_numba(returns, 20)

            # Use 'adx' column if pre-computed, else approx with vol
            adx = df["adx"].to_numpy(dtype=np.float32) if "adx" in df.columns else volatility * 100

            data = pd.DataFrame({
                "returns": returns, 
                "volatility": volatility, 
                "adx": adx
            }, index=df.index)
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

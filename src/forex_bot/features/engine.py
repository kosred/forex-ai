from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.config import Settings
from ..domain.events import PreparedDataset, SignalResult
from ..training.evaluation import pad_probs, probs_to_signals
from ..training.persistence_service import PersistenceService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ModelProba:
    name: str
    proba: np.ndarray


class SignalEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models_dir = Path(os.environ.get("FOREX_BOT_MODELS_DIR", "models"))
        self.logs_dir = Path(os.environ.get("FOREX_BOT_LOGS_DIR", "logs"))
        self.persistence = PersistenceService(self.models_dir, self.logs_dir, settings=self.settings)
        self.models: dict[str, Any] = {}
        self.meta_blender: Any | None = None
        self._onnx = None

    def load_models(self, models_dir: str | Path | None = None) -> None:
        if models_dir is not None:
            self.models_dir = Path(models_dir)
            self.persistence = PersistenceService(self.models_dir, self.logs_dir, settings=self.settings)
        try:
            self.models, self.meta_blender = self.persistence.load_models()
        except Exception as exc:
            logger.warning("Failed to load models: %s", exc)
            self.models = {}
            self.meta_blender = None
        # Optional ONNX inference path
        use_onnx = str(os.environ.get("FOREX_BOT_USE_ONNX", "")).strip().lower() in {"1", "true", "yes", "on"}
        if use_onnx:
            try:
                from ..models.onnx_exporter import ONNXExporter

                exporter = ONNXExporter(str(self.models_dir))
                exporter.load_models()
                self._onnx = exporter
            except Exception as exc:
                logger.warning("ONNX load failed: %s", exc)
                self._onnx = None

    def _ensemble_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas: list[np.ndarray] = []
        for model in self.models.values():
            try:
                proba = pad_probs(model.predict_proba(X))
                if proba.shape[0] == len(X):
                    probas.append(proba)
            except Exception as exc:
                logger.debug("Model prediction failed in ensemble: %s", exc)
        if not probas:
            return np.zeros((len(X), 3), dtype=float)
        return np.mean(probas, axis=0)

    def _meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=X.index)
        for name, model in self.models.items():
            try:
                proba = pad_probs(model.predict_proba(X))
                if proba.shape[0] != len(X):
                    continue
                feats[f"{name}_buy"] = proba[:, 1]
            except Exception:
                continue
        if feats.empty:
            feats = pd.DataFrame(np.zeros((len(X), 1)), columns=["dummy"], index=X.index)
        return feats

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._onnx is not None:
            try:
                return self._onnx.predict_with_meta_blender(X)
            except Exception:
                pass
        if self.meta_blender is not None:
            try:
                feats = self._meta_features(X)
                return pad_probs(self.meta_blender.predict_proba(feats))
            except Exception:
                pass
        return self._ensemble_proba(X)

    def generate_ensemble_signals(self, dataset: PreparedDataset) -> SignalResult:
        X = dataset.X
        if X is None or len(X) == 0:
            return SignalResult(
                signal=0,
                confidence=0.0,
                model_votes={},
                regime="Normal",
                meta_features={},
                probs=np.zeros(3, dtype=float),
                signals=pd.Series(dtype=int),
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        probas = self._predict_proba(X)
        signals = probs_to_signals(probas)
        last_idx = -1 if len(signals) > 0 else None
        if last_idx is None:
            last_signal = 0
            confidence = 0.0
            last_probs = np.zeros(3, dtype=float)
        else:
            last_signal = int(signals[last_idx])
            last_probs = probas[last_idx]
            confidence = float(np.max(last_probs)) if last_probs is not None else 0.0

        votes: dict[str, float] = {}
        for name, model in self.models.items():
            try:
                proba = pad_probs(model.predict_proba(X))
                if proba.shape[0] == len(X):
                    votes[name] = float(proba[last_idx, 1] - proba[last_idx, 2])
            except Exception:
                continue

        return SignalResult(
            signal=last_signal,
            confidence=confidence,
            model_votes=votes,
            regime="Normal",
            meta_features={},
            probs=np.asarray(last_probs, dtype=float),
            signals=pd.Series(signals, index=X.index),
        )


__all__ = ["SignalEngine"]

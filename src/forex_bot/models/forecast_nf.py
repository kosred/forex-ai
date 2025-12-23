import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .base import ExpertModel, dataframe_to_float32_numpy

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx as NF_NBEATSx
    from neuralforecast.models import TiDE as NF_TiDE

    NF_AVAILABLE = True
except Exception:  # pragma: no cover
    NeuralForecast = None  # type: ignore
    NF_NBEATSx = None  # type: ignore
    NF_TiDE = None  # type: ignore
    NF_AVAILABLE = False

try:  # pragma: no cover
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None  # type: ignore
    SKLEARN_AVAILABLE = False


class _NFClsWrapper(ExpertModel):
    """
    Wraps a neuralforecast regression model (TiDE, NBEATSx) into a 3-class classifier.
    Maps regression outputs to probabilities via a tanh-based logit.
    """

    model_name: str = "tide"

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_steps: int = 400,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.device = device
        self.feature_columns: list[str] = []
        self._nf = None
        self.model = None
        self.calibrator = None

    def _build_nf_model(self, input_dim: int):
        if not NF_AVAILABLE:
            raise RuntimeError("neuralforecast not installed")
        common = dict(
            input_size=input_dim,
            h=1,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            random_seed=42,
        )
        if self.model_name == "tide":
            return NF_TiDE(
                encoder_layers=2,
                decoder_layers=2,
                dropout=self.dropout,
                d_model=self.hidden_dim,
                **common,
            )
        if self.model_name == "nbeatsx":
            return NF_NBEATSx(
                stack_types=("trend", "seasonality"),
                nb_blocks_per_stack=2,
                thetas_dims=(4, 8),
                dropout=self.dropout,
                **common,
            )
        raise ValueError(f"Unknown NF model {self.model_name}")

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if not NF_AVAILABLE:
            raise RuntimeError("neuralforecast not installed")
        x_np = dataframe_to_float32_numpy(x)
        self.feature_columns = list(x.columns)
        df = pd.DataFrame(x_np, columns=self.feature_columns)
        df["y"] = y.to_numpy(dtype=float)
        df["unique_id"] = "fx"
        df["ds"] = np.arange(len(df))

        nf_model = self._build_nf_model(input_dim=x_np.shape[1])
        self._nf = NeuralForecast(models=[nf_model], freq="D")
        self._nf.fit(df=df)
        self.model = nf_model

        # Optional calibration using a holdout split on predicted scalar
        if SKLEARN_AVAILABLE and LogisticRegression is not None:
            try:
                val_size = max(200, int(0.15 * len(df)))
                if val_size < 50:
                    return
                train_df = df.iloc[:-val_size].copy()
                val_df = df.iloc[-val_size:].copy()
                val_y = y.iloc[-val_size:].to_numpy(dtype=int)

                # Predict on holdout
                fcst = self._nf.predict(df=val_df)
                col = self.model.__class__.__name__
                if col not in fcst.columns:
                    col = fcst.columns[-1]
                z = fcst[col].to_numpy(dtype=float)

                # Map labels to contiguous {0,1,2}
                uniq = list(dict.fromkeys(int(v) for v in val_y))
                mapping = {lab: i for i, lab in enumerate(uniq)}
                y_contig = np.array([mapping[int(v)] for v in val_y], dtype=int)

                clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
                clf.fit(z.reshape(-1, 1), y_contig)
                self.calibrator = (clf, mapping)
            except Exception as exc:
                logger.debug(f"Calibration skipped: {exc}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self._nf is None or self.model is None:
            raise RuntimeError("Model not fitted")
        x_np = dataframe_to_float32_numpy(x)
        df = pd.DataFrame(x_np, columns=self.feature_columns)
        df["unique_id"] = "fx"
        df["ds"] = np.arange(len(df))
        fcst = self._nf.predict(df=df)
        col = self.model.__class__.__name__
        if col not in fcst.columns:
            col = fcst.columns[-1]
        val = fcst[col].to_numpy(dtype=float)
        if self.calibrator:
            clf, mapping = self.calibrator
            # Inverse mapping to project back to {-1,0,1} order
            inv = {v: k for k, v in mapping.items()}
            proba = clf.predict_proba(val.reshape(-1, 1))
            classes = [inv.get(i, i) for i in range(proba.shape[1])]
            # Pad/reorder to [neutral(0), buy(1), sell(-1)]
            return self._pad_probs(proba, classes)
        else:
            val_clipped = np.tanh(val)  # map to [-1,1]
            logits = np.stack([0.0 * val_clipped, val_clipped, -val_clipped], axis=1)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
        probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-12)
        return probs.astype(float, copy=False)

    @staticmethod
    def _pad_probs(p: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
        """Pad/reorder to [neutral(0), buy(1), sell(-1)]."""
        arr = np.asarray(p, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n = arr.shape[0]
        out = np.zeros((n, 3), dtype=float)
        if classes and len(classes) == arr.shape[1]:
            for col, cls in enumerate(classes):
                if cls == 0:
                    out[:, 0] = arr[:, col]
                elif cls == 1:
                    out[:, 1] = arr[:, col]
                elif cls == -1 or cls == 2:
                    out[:, 2] = arr[:, col]
            return out
        if arr.shape[1] == 2:
            out[:, 0] = arr[:, 0]
            out[:, 1] = arr[:, 1]
        elif arr.shape[1] >= 3:
            out = arr[:, :3]
        else:
            out[:, 0] = 1.0 - arr[:, 0]
            out[:, 1] = arr[:, 0]
        return out

    def save(self, path: str) -> None:
        if self._nf is None:
            return
        out_dir = Path(path) / f"nf_{self.model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._nf.save(out_dir)
            meta = {
                "model_name": self.model_name,
                "feature_columns": self.feature_columns,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "max_steps": self.max_steps,
                "device": self.device,
            }
            joblib.dump(meta, out_dir / "meta.pkl")
            if self.calibrator is not None:
                joblib.dump(self.calibrator, out_dir / "calibrator.pkl")
        except Exception as e:
            logger.warning(f"Failed to save neuralforecast model {self.model_name}: {e}")

    def load(self, path: str) -> None:
        if not NF_AVAILABLE:
            logger.warning("neuralforecast not installed; cannot load NF models.")
            return
        out_dir = Path(path) / f"nf_{self.model_name}"
        meta_path = out_dir / "meta.pkl"
        if not meta_path.exists():
            return
        meta = joblib.load(meta_path)
        self.model_name = meta.get("model_name", self.model_name)
        self.feature_columns = meta.get("feature_columns", [])
        self.hidden_dim = meta.get("hidden_dim", self.hidden_dim)
        self.dropout = meta.get("dropout", self.dropout)
        self.lr = meta.get("lr", self.lr)
        self.batch_size = meta.get("batch_size", self.batch_size)
        self.max_steps = meta.get("max_steps", self.max_steps)
        self.device = meta.get("device", self.device)
        try:
            nf_model = self._build_nf_model(input_dim=len(self.feature_columns))
            self._nf = NeuralForecast(models=[nf_model], freq="D")
            self._nf.load(out_dir)
            self.model = nf_model
            cal_path = out_dir / "calibrator.pkl"
            if cal_path.exists():
                try:
                    self.calibrator = joblib.load(cal_path)
                except Exception:
                    self.calibrator = None
        except Exception as e:
            logger.warning(f"Failed to load neuralforecast model {self.model_name}: {e}")


class TiDENFExpert(_NFClsWrapper):
    model_name = "tide"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name="tide", **kwargs)


class NBEATSxNFExpert(_NFClsWrapper):
    model_name = "nbeatsx"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name="nbeatsx", **kwargs)

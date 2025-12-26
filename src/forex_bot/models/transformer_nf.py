import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import ExpertModel, dataframe_to_float32_numpy

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST as NF_PatchTST
    from neuralforecast.models import TimesNet as NF_TimesNet

    NF_AVAILABLE = True
except Exception:  # pragma: no cover
    NeuralForecast = None  # type: ignore
    NF_PatchTST = None  # type: ignore
    NF_TimesNet = None  # type: ignore
    NF_AVAILABLE = False

try:  # pragma: no cover
    from sklearn.linear_model import LogisticRegression

    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None  # type: ignore
    SKLEARN_AVAILABLE = False


class _NFWrapper(ExpertModel):
    """
    Wraps a neuralforecast model (PatchTST/TimesNet) into our ExpertModel API.
    """

    def __init__(
        self,
        model_name: str = "patchtst",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        batch_size: int = 128,
        max_steps: int = 400,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name.lower()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.device = device
        self.model = None
        self._nf = None
        self.feature_columns: list[str] = []
        self.calibrator = None
        self.head: nn.Module | None = None

    def _build_nf_model(self, input_dim: int):
        if not NF_AVAILABLE:
            raise RuntimeError("neuralforecast not installed")
        common = dict(
            input_size=input_dim,
            h=1,  # one-step classification framing
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            learning_rate=self.lr,
            random_seed=42,
        )
        if self.model_name == "patchtst":
            return NF_PatchTST(
                d_model=self.hidden_dim,
                dropout=self.dropout,
                **common,
            )
        if self.model_name == "timesnet":
            return NF_TimesNet(
                hidden_size=self.hidden_dim,
                dropout=self.dropout,
                **common,
            )
        raise ValueError(f"Unknown NF model {self.model_name}")

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if not NF_AVAILABLE:
            raise RuntimeError("neuralforecast not installed")
        # Enforce time ordering if index is datetime and not monotonic
        if isinstance(x.index, pd.DatetimeIndex) and not x.index.is_monotonic_increasing:
            order = np.argsort(x.index.view("int64"))
            x = x.iloc[order]
            y = y.iloc[order]

        # Prepare data in neuralforecast format: long df with 'ds','y','unique_id'
        x_np = dataframe_to_float32_numpy(x)
        self.feature_columns = list(x.columns)

        df = pd.DataFrame(x_np, columns=self.feature_columns)
        df["y"] = y.to_numpy(dtype=float)
        df["unique_id"] = "fx"
        df["ds"] = np.arange(len(df))

        nf_model = self._build_nf_model(input_dim=x_np.shape[1])
        self._nf = NeuralForecast(models=[nf_model], freq="D")  # freq placeholder
        self._nf.fit(df=df)
        self.model = nf_model

        # Build classifier head and train on top of NF scalar output
        try:
            # Predict embeddings/scalars from NF on the training set
            fcst = self._nf.predict(df=df)
            col = self.model.__class__.__name__
            if col not in fcst.columns:
                col = fcst.columns[-1]
            z = fcst[col].to_numpy(dtype=np.float32)

            # Map labels to contiguous
            uniq = list(dict.fromkeys(int(v) for v in y.to_numpy(dtype=int)))
            mapping = {lab: i for i, lab in enumerate(uniq)}
            y_contig = np.array([mapping[int(v)] for v in y.to_numpy(dtype=int)], dtype=int)

            z_t = torch.as_tensor(z.reshape(-1, 1), dtype=torch.float32)
            y_t = torch.as_tensor(y_contig, dtype=torch.long)
            dataset = TensorDataset(z_t, y_t)
            loader = DataLoader(dataset, batch_size=max(64, int(self.batch_size)), shuffle=True)

            self.head = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, max(3, len(uniq))),
            )
            self.head.to(torch.device(self.device))

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.head.parameters(), lr=1e-3)

            self.head.train()
            for _ in range(50):
                for zx, zy in loader:
                    zx = zx.to(self.head[0].weight.device)
                    zy = zy.to(self.head[0].weight.device)
                    optimizer.zero_grad()
                    out = self.head(zx)
                    loss = criterion(out, zy)
                    loss.backward()
                    optimizer.step()

            # Save mapping for predict
            self.calibrator = (self.head, mapping)
        except Exception as exc:
            logger.warning(f"Classifier head training failed; falling back to tanh mapping: {exc}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self._nf is None or self.model is None:
            raise RuntimeError("Model not fitted")
        x_np = dataframe_to_float32_numpy(x)
        df = pd.DataFrame(x_np, columns=self.feature_columns)
        df["unique_id"] = "fx"
        df["ds"] = np.arange(len(df))
        # neuralforecast predict returns a df with predictions per model
        fcst = self._nf.predict(df=df)
        # Map regression output to 3-class probabilities via simple logits
        # Use the model name column
        col = self.model.__class__.__name__
        if col not in fcst.columns:
            col = fcst.columns[-1]
        val = fcst[col].to_numpy(dtype=float)
        if self.calibrator:
            head, mapping = self.calibrator
            inv = {v: k for k, v in mapping.items()}
            head.eval()
            with torch.no_grad():
                zx = torch.as_tensor(val.reshape(-1, 1), dtype=torch.float32, device=head[0].weight.device)
                logits = head(zx).cpu().numpy()
            proba = np.exp(logits - logits.max(axis=1, keepdims=True))
            proba = proba / (proba.sum(axis=1, keepdims=True) + 1e-12)
            classes = [inv.get(i, i) for i in range(proba.shape[1])]
            return self._pad_probs(proba, classes)
        # Map to neutral/buy/sell probabilities via tanh-based logits
        val_clipped = np.tanh(val)  # between -1 and 1
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
                try:
                    head, mapping = self.calibrator
                    # Save state_dict + mapping
                    torch.save(head.state_dict(), out_dir / "clf_head.pt")
                    joblib.dump(mapping, out_dir / "calib_mapping.pkl")
                except Exception as exc:
                    logger.warning(f"Failed to save classifier head: {exc}")
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
            try:
                head_path = out_dir / "clf_head.pt"
                map_path = out_dir / "calib_mapping.pkl"
                if head_path.exists() and map_path.exists():
                    head = nn.Sequential(
                        nn.Linear(1, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 3),
                    )
                    try:
                        state = torch.load(head_path, map_location="cpu", weights_only=True)
                    except TypeError as exc:
                        raise RuntimeError("PyTorch too old for weights_only load; upgrade for security.") from exc
                    head.load_state_dict(state)
                    mapping = joblib.load(map_path)
                    self.calibrator = (head, mapping)
            except Exception:
                self.calibrator = None
        except Exception as e:
            logger.warning(f"Failed to load neuralforecast model {self.model_name}: {e}")


class PatchTSTExpert(_NFWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name="patchtst", **kwargs)


class TimesNetExpert(_NFWrapper):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model_name="timesnet", **kwargs)

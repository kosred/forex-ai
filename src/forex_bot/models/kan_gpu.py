import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist  # DDP Support

from .base import ExpertModel, dataframe_to_float32_numpy
from .device import select_device

logger = logging.getLogger(__name__)

try:
    from efficient_kan import KAN

    KAN_LIB_AVAILABLE = True
except ImportError:
    KAN_LIB_AVAILABLE = False
    KAN = None


class KANExpert(ExpertModel):
    """
    GPU-Optimized KAN using 'efficient-kan' library.
    Uses true B-Splines for high accuracy.
    """

    MAX_TRAIN_ROWS = 0

    def __init__(
        self,
        hidden_dim: int = 64,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        lr: float = 1e-3,
        batch_size: int = 4096,
        max_time_sec: int = 3600,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        self.model = None
        self.hyperparams = {
            "hidden_dim": hidden_dim,
            "grid_size": grid_size,
            "spline_order": spline_order,
            "scale_noise": scale_noise,
            "scale_base": scale_base,
            "scale_spline": scale_spline,
        }
        self.lr = lr
        self.batch_size = batch_size
        self.max_time_sec = max_time_sec
        self.device = select_device(device)
        self.input_dim = 0

    def fit(self, X: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        if not KAN_LIB_AVAILABLE:
            logger.error("efficient-kan not installed. GPU Training skipped.")
            return

        self.input_dim = X.shape[1]

        # Convert to tensors
        X_t = torch.as_tensor(dataframe_to_float32_numpy(X), dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y.values + 1, dtype=torch.long, device=self.device)  # -1,0,1 -> 0,1,2

        # KAN library expects [Input -> Hidden -> Output] layers list
        # We construct a simple 2-layer KAN: Input -> Hidden -> 3 (Classes)
        layers_hidden = [self.input_dim] + [self.hyperparams["hidden_dim"]] + [3]

        self.model = KAN(
            layers_hidden=layers_hidden,
            grid_size=self.hyperparams["grid_size"],
            spline_order=self.hyperparams["spline_order"],
            scale_noise=self.hyperparams["scale_noise"],
            scale_base=self.hyperparams["scale_base"],
            scale_spline=self.hyperparams["scale_spline"],
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        from torch.utils.data import DataLoader, TensorDataset, distributed

        dataset = TensorDataset(X_t, y_t)

        # DDP Sampler
        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = distributed.DistributedSampler(dataset)

        # GPU Optimization
        is_cuda = str(self.device).startswith("cuda")
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=4 if is_cuda else 0,
            pin_memory=is_cuda,
            prefetch_factor=2 if is_cuda else None,
            persistent_workers=is_cuda,
        )

        import time

        start = time.time()

        # Training Loop
        self.model.train()
        for epoch in range(1000):
            if sampler:
                sampler.set_epoch(epoch)  # Crucial for shuffling in DDP

            if time.time() - start > self.max_time_sec:
                break

            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/train_KAN", total_loss / len(loader), epoch)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("KAN GPU model not loaded")
        self.model.eval()
        outs = []
        with torch.no_grad():
            # Process in chunks to avoid OOM even on GPU if X is huge
            chunk_size = 8192
            for start in range(0, len(X), chunk_size):
                xb_df = X.iloc[start : start + chunk_size]
                xb = torch.as_tensor(dataframe_to_float32_numpy(xb_df), dtype=torch.float32, device=self.device)
                logits = self.model(xb)
                outs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if not outs:
            raise RuntimeError("KAN GPU inference returned no batches")
        probs = np.vstack(outs)
        # Training encodes labels as y+1 where y ∈ {-1,0,1} -> class indices {0,1,2}.
        # Reorder to project convention [neutral, buy, sell].
        return probs[:, [1, 2, 0]]

    def save(self, path: str) -> None:
        if self.model:
            torch.save(self.model.state_dict(), Path(path) / "kan_gpu.pt")
            # Save config to reconstruct model
            joblib.dump({"input_dim": self.input_dim, "params": self.hyperparams}, Path(path) / "kan_gpu_meta.pkl")

    def load(self, path: str) -> None:
        # Loading GPU KAN on CPU might be hard if library missing.
        # This method assumes we are on GPU if this class is instantiated.
        p = Path(path) / "kan_gpu.pt"
        m = Path(path) / "kan_gpu_meta.pkl"
        if p.exists() and m.exists() and KAN_LIB_AVAILABLE:
            meta = joblib.load(m)
            self.input_dim = meta["input_dim"]
            params = meta["params"]
            layers = [self.input_dim] + [params["hidden_dim"]] + [3]

            self.model = KAN(
                layers_hidden=layers,
                grid_size=params["grid_size"],
                spline_order=params["spline_order"],
                scale_noise=params["scale_noise"],
                scale_base=params["scale_base"],
                scale_spline=params["scale_spline"],
            ).to(self.device)
            self.model.load_state_dict(torch.load(p, map_location=self.device))

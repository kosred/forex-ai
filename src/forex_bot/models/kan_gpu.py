import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist  # DDP Support

from .base import (
    EarlyStopper,
    ExpertModel,
    dataframe_to_float32_numpy,
    get_early_stop_params,
    time_series_train_val_split,
)
from .device import select_device
from ..core.system import resolve_cpu_budget

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
        y_vals = y.to_numpy()
        if not np.all(np.isin(y_vals, [-1, 0, 1])):
            raise ValueError(f"Labels must be in {{-1, 0, 1}}, got: {np.unique(y_vals)}")

        try:
            X_train, X_val, y_train, y_val = time_series_train_val_split(
                X, y, val_ratio=0.15, min_train_samples=100, embargo_samples=0
            )
        except Exception:
            split = int(len(X) * 0.85)
            X_train, X_val = X.iloc[:split], X.iloc[split:]
            y_train, y_val = y.iloc[:split], y.iloc[split:]

        # HPC FIX: Unified Protocol Mapping
        # -1 (Sell) -> 2, 0 (Neutral) -> 0, 1 (Buy) -> 1
        y_mapped = np.where(y_train.to_numpy() == -1, 2, y_train.to_numpy()).astype(int)
        X_t = torch.as_tensor(
            dataframe_to_float32_numpy(X_train),
            dtype=torch.float32,
            device="cpu",
        )
        y_t = torch.as_tensor(y_mapped, dtype=torch.long, device="cpu")

        X_v = None
        y_v = None
        if len(X_val) > 0:
            # ...
            y_v_mapped = np.where(y_val.to_numpy() == -1, 2, y_val.to_numpy()).astype(int)
            X_v = torch.as_tensor(
                dataframe_to_float32_numpy(X_val),
                dtype=torch.float32,
                device="cpu",
            )
            y_v = torch.as_tensor(y_v_mapped, dtype=torch.long, device="cpu")

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

        # GPU Optimization with spawn context to avoid CUDA fork issues
        is_cuda = str(self.device).startswith("cuda")
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        cpu_budget = resolve_cpu_budget()
        if world_size > 1:
            cpu_budget = max(1, cpu_budget // world_size)
        num_workers = max(1, cpu_budget - 1) if is_cuda else 0
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": (sampler is None),
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": is_cuda,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["multiprocessing_context"] = "spawn"
        loader = DataLoader(dataset, **loader_kwargs)
        val_loader = None
        if X_v is not None and y_v is not None:
            val_dataset = TensorDataset(X_v, y_v)
            val_loader = DataLoader(
                val_dataset,
                batch_size=max(1024, int(self.batch_size)),
                shuffle=False,
                num_workers=0,
                pin_memory=is_cuda,
            )

        import time

        start = time.time()

        # HPC: Enable Automatic Mixed Precision (AMP)
        scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
        amp_dtype = torch.bfloat16 if (is_cuda and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

        # Training Loop
        self.model.train()
        es_pat, es_delta = get_early_stop_params(20, 0.001)
        early_stopper = EarlyStopper(patience=es_pat, min_delta=es_delta)
        for epoch in range(1000):
            if sampler:
                sampler.set_epoch(epoch)

            if time.time() - start > self.max_time_sec:
                break

            total_loss = 0
            for bx, by in loader:
                bx = bx.to(self.device, non_blocking=is_cuda)
                by = by.to(self.device, non_blocking=is_cuda)

                optimizer.zero_grad(set_to_none=True)
                
                # AMP Autocast
                with torch.amp.autocast("cuda", enabled=is_cuda, dtype=amp_dtype):
                    logits = self.model(bx)
                    loss = criterion(logits, by)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()

            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/train_KAN", total_loss / len(loader), epoch)

            if val_loader is not None:
                self.model.eval()
                v_loss_sum = 0.0
                v_count = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx = vx.to(self.device, non_blocking=is_cuda)
                        vy = vy.to(self.device, non_blocking=is_cuda)
                        with torch.amp.autocast("cuda", enabled=is_cuda, dtype=amp_dtype):
                            v_logits = self.model(vx)
                            v_loss = criterion(v_logits, vy)
                        v_loss_sum += float(v_loss.item()) * len(vx)
                        v_count += len(vx)
                self.model.train()
                v_loss_avg = v_loss_sum / max(1, v_count)
                if early_stopper(v_loss_avg):
                    logger.info(
                        "KAN GPU early stopping at epoch %s (val_loss=%.6f)",
                        epoch,
                        v_loss_avg,
                    )
                    break

        if is_cuda:
            torch.cuda.empty_cache()

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
        # Unified Protocol: [Neutral, Buy, Sell] already in indices [0, 1, 2]
        return probs[:, :3]

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
            try:
                state = torch.load(p, map_location=self.device, weights_only=True)
            except TypeError as exc:
                raise RuntimeError("PyTorch too old for weights_only load; upgrade for security.") from exc
            self.model.load_state_dict(state)

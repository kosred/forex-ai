import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import EarlyStopper, ExpertModel, dataframe_to_float32_numpy
from .device import (
    maybe_init_distributed,
    preferred_amp_dtype,
    select_device,
    tune_torch_backend,
    wrap_ddp,
)

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.fc(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.norm(out + res)
        return out


class TiDEExpert(ExpertModel):
    """
    GPU-Optimized TiDE (Time-series Dense Encoder).
    Features: AMP, DDP, Large Batches.
    """

    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 4096

    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        batch_size: int = 8192,
        max_time_sec: int = 1800,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        self.model = None
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.max_time_sec = max_time_sec
        self.device = select_device(device)
        self.input_dim = 0
        self.is_ddp = False
        self.rank = 0
        self.world_size = 1
        tune_torch_backend(str(self.device))

    def _build_model(self):
        class TiDENet(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 3) -> None:
                super().__init__()
                self.feature_proj = nn.Linear(input_dim, hidden_dim)
                self.encoder = nn.Sequential(ResidualBlock(hidden_dim), ResidualBlock(hidden_dim))
                self.decoder = nn.Sequential(ResidualBlock(hidden_dim), ResidualBlock(hidden_dim))
                self.temporal_link = nn.Linear(hidden_dim, hidden_dim)
                self.final = nn.Linear(hidden_dim, output_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x_emb = self.feature_proj(x)
                enc_out = self.encoder(x_emb)
                dec_in = self.temporal_link(enc_out)
                dec_out = self.decoder(dec_in)
                return self.final(dec_out)

        ddp_device = maybe_init_distributed()
        if ddp_device:
            self.device = ddp_device
        model = TiDENet(self.input_dim, self.hidden_dim).to(self.device)
        model, self.device, self.is_ddp, self.rank, self.world_size = wrap_ddp(model, self.device)
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        self.input_dim = X.shape[1]
        self.model = self._build_model()

        is_cuda = str(self.device).startswith("cuda")
        compile_flag = os.environ.get("TORCH_COMPILE", "auto").lower()
            try:
                self.model.to(self.device)
                from .device import maybe_compile
                self.model = maybe_compile(self.model, mode="reduce-overhead")

        split = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        # Force CPU initialization to avoid VRAM OOM
        X_t = torch.as_tensor(dataframe_to_float32_numpy(X_train), dtype=torch.float32, device="cpu")
        y_t = torch.as_tensor(y_train.values + 1, dtype=torch.long, device="cpu")
        X_v = torch.as_tensor(dataframe_to_float32_numpy(X_val), dtype=torch.float32, device="cpu")
        y_v = torch.as_tensor(y_val.values + 1, dtype=torch.long, device="cpu")

        # Removed eager move to GPU for validation data
        
        dataset = TensorDataset(X_t, y_t)
        # GPU Optimization with spawn context to avoid CUDA fork issues
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        # Auto-tune batch size based on free GPU memory (safe fallback to provided batch_size)
        eff_batch = self.batch_size
        if is_cuda:
            try:
                mem_free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
                target = max(32, int((mem_free / (0.5 * 1024**3)) * 128 * 0.6))
                eff_batch = max(32, min(self.batch_size, target))
            except Exception:
                pass

        num_workers = max(1, min(os.cpu_count() or 4, 8)) if is_cuda else 0
        if world_size > 1:
            num_workers = max(1, num_workers // world_size)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=max(1, eff_batch // max(1, world_size)),  # Divide batch by GPU count
            shuffle=(world_size == 1),  # Only shuffle if not using DistributedSampler
            num_workers=num_workers,
            pin_memory=is_cuda,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            multiprocessing_context='spawn' if num_workers > 0 else None,
            sampler=sampler,
        )
        fused_ok = is_cuda and hasattr(optim, "AdamW") and "fused" in optim.AdamW.__init__.__code__.co_varnames
        optimizer = (
            optim.AdamW(self.model.parameters(), lr=self.lr, fused=fused_ok)
            if fused_ok
            else optim.Adam(self.model.parameters(), lr=self.lr)
        )
        criterion = nn.CrossEntropyLoss()
        early_stopper = EarlyStopper(patience=8, min_delta=0.001)
        amp_dtype = preferred_amp_dtype(str(self.device))
        scaler = torch.amp.GradScaler("cuda", enabled=is_cuda, init_scale=1024.0)

        import time

        start = time.time()
        self.model.train()

        cg_flag = os.environ.get("CUDA_GRAPH", "auto").lower()
        use_cuda_graph = (cg_flag not in {"0", "false", "no"}) and is_cuda
        static_bx = static_by = graph = None

        def _step(batch_x, batch_y):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=is_cuda, dtype=amp_dtype):
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return loss.detach()

        for epoch in range(100):
            if time.time() - start > self.max_time_sec:
                break
            if sampler is not None:
                sampler.set_epoch(epoch)

            for bx, by in loader:
                bx = bx.to(self.device, non_blocking=is_cuda)
                by = by.to(self.device, non_blocking=is_cuda)

                if use_cuda_graph and graph is None and bx.is_contiguous() and by.is_contiguous():
                    try:
                        static_bx = bx.clone()
                        static_by = by.clone()
                        graph = torch.cuda.CUDAGraph()
                        _ = _step(static_bx, static_by)
                        torch.cuda.synchronize()
                        scaler._lazy_init_scale_growth_tracker()
                        with torch.cuda.graph(graph):
                            _step(static_bx, static_by)
                        logger.info("CUDA Graph captured for TiDE GPU step.")
                    except Exception as exc:
                        logger.warning(f"CUDA Graph capture failed for TiDE GPU: {exc}")
                        graph = None

                if graph is not None and static_bx is not None:
                    static_bx.copy_(bx)
                    static_by.copy_(by)
                    graph.replay()
                else:
                    _step(bx, by)

            if len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    # Move to GPU explicitly for validation
                    X_v_gpu = X_v.to(self.device)
                    y_v_gpu = y_v.to(self.device)
                    val_out = self.model(X_v_gpu)
                    val_loss = criterion(val_out, y_v_gpu).item()
                self.model.train()
                if early_stopper(val_loss):
                    break

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TiDE GPU model not loaded")
        self.model.eval()
        outs = []
        is_cuda = str(self.device).startswith("cuda")
        with torch.inference_mode():
            for start in range(0, len(X), self.PRED_BATCH_SIZE):
                xb_df = X.iloc[start : start + self.PRED_BATCH_SIZE]
                xb = torch.as_tensor(dataframe_to_float32_numpy(xb_df), dtype=torch.float32, device=self.device)
                with torch.amp.autocast("cuda", enabled=is_cuda):
                    logits = self.model(xb)
                outs.append(torch.softmax(logits, dim=1).cpu().float().numpy())
        if not outs:
            raise RuntimeError("TiDE GPU inference returned no batches")
        probs = np.vstack(outs)
        # Training encodes labels as y+1 where y ∈ {-1,0,1} -> class indices {0,1,2}.
        # Reorder to project convention [neutral, buy, sell].
        return probs[:, [1, 2, 0]]

    def save(self, path: str) -> None:
        if self.model and (not self.is_ddp or self.rank == 0):
            target = getattr(self.model, "module", self.model)
            if hasattr(target, "_orig_mod"):
                target = target._orig_mod
            torch.save(target.state_dict(), Path(path) / "tide.pt")
            joblib.dump({"input_dim": self.input_dim, "hidden": self.hidden_dim}, Path(path) / "tide_meta.pkl")

    def load(self, path: str) -> None:
        p = Path(path) / "tide.pt"
        m = Path(path) / "tide_meta.pkl"
        if p.exists() and m.exists():
            meta = joblib.load(m)
            self.input_dim = meta["input_dim"]
            self.hidden_dim = meta["hidden"]
            self.model = self._build_model()
            target = getattr(self.model, "module", self.model)
            target.load_state_dict(torch.load(p, map_location=self.device))

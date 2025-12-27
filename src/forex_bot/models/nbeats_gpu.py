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

from .base import EarlyStopper, ExpertModel, dataframe_to_float32_numpy, get_early_stop_params
from .device import (
    maybe_init_distributed,
    preferred_amp_dtype,
    select_device,
    tune_torch_backend,
    wrap_ddp,
)

logger = logging.getLogger(__name__)

# Re-use Block definition or import?
# For file isolation, better to redefine or import from common 'deep_utils.py'.
# I will redefine to keep files standalone as requested.


class NBeatsBlock(nn.Module):
    def __init__(self, hidden_dim: int, theta_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.theta_b = nn.Linear(hidden_dim, theta_dim, bias=False)
        self.theta_f = nn.Linear(hidden_dim, theta_dim, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        h = self.act(self.fc4(h))
        return self.theta_b(h), self.theta_f(h)


class NBeatsExpert(ExpertModel):
    """
    GPU-Optimized N-BEATS Implementation.
    Features: AMP, DDP, Large Batches.
    """

    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 4096  # Large batch for GPU

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_blocks: int = 2,
        stack_types: list[str] | None = None,
        lr: float = 1e-3,
        max_time_sec: int = 1800,
        device: str = "cuda",
        batch_size: int = 2048,
        **kwargs: Any,
    ) -> None:
        self.model = None
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.max_time_sec = max_time_sec
        self.device = select_device(device)
        self.batch_size = batch_size
        if stack_types:
            self.stack_types = list(stack_types)
        elif n_blocks and n_blocks > 0:
            if n_blocks == 2:
                self.stack_types = ["trend", "seasonality"]
            else:
                self.stack_types = [f"stack_{i}" for i in range(n_blocks)]
        else:
            self.stack_types = ["trend", "seasonality"]
        self.input_dim = 0
        self.is_ddp = False
        self.rank = 0
        self.world_size = 1
        tune_torch_backend(str(self.device))

    def _build_model(self) -> nn.Module:
        class NBeatsNet(nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, stack_types: list[str], output_dim: int = 3) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [NBeatsBlock(hidden_dim, hidden_dim) for _ in range(len(stack_types))]
                )
                self.final = nn.Linear(hidden_dim, output_dim)
                self.embed = nn.Linear(input_dim, hidden_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.embed(x)
                forecast = torch.zeros_like(x)
                for block in self.blocks:
                    backcast, fore = block(x)
                    x = x - backcast
                    forecast = forecast + fore
                return self.final(forecast)

        ddp_device = maybe_init_distributed()
        if ddp_device:
            self.device = ddp_device

        model = NBeatsNet(self.input_dim, self.hidden_dim, self.stack_types).to(self.device)
        model, self.device, self.is_ddp, self.rank, self.world_size = wrap_ddp(model, self.device)
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        self.input_dim = X.shape[1]
        self.model = self._build_model()

        is_cuda = str(self.device).startswith("cuda")
        compile_flag = os.environ.get("TORCH_COMPILE", "auto").lower()
        if compile_flag not in {"0", "false", "no"} and is_cuda:
            try:
                self.model.to(self.device)
                from .device import maybe_compile
                self.model = maybe_compile(self.model, mode="reduce-overhead")

                logger.info("torch.compile enabled for NBeats GPU.")
            except Exception as e:
                logger.warning(f"torch.compile failed for NBeats GPU: {e}")

        y_vals = y.to_numpy()
        if not np.all(np.isin(y_vals, [-1, 0, 1])):
            raise ValueError(f"Labels must be in {{-1, 0, 1}}, got: {np.unique(y_vals)}")

        split = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y_vals[:split], y_vals[split:]

        # HPC FIX: Unified Protocol Mapping
        y_train_mapped = np.where(y_train == -1, 2, y_train).astype(int)
        y_val_mapped = np.where(y_val == -1, 2, y_val).astype(int)

        # Keep data on CPU initially to avoid VRAM OOM
        X_t = torch.as_tensor(dataframe_to_float32_numpy(X_train), dtype=torch.float32, device="cpu")
        y_t = torch.as_tensor(y_train_mapped, dtype=torch.long, device="cpu")
        
        # Validation set can be on GPU if small enough, but safer on CPU for consistency
        X_v = torch.as_tensor(dataframe_to_float32_numpy(X_val), dtype=torch.float32, device="cpu")
        y_v = torch.as_tensor(y_val_mapped, dtype=torch.long, device="cpu")

        dataset = TensorDataset(X_t, y_t)

        # GPU Optimization: Use workers and pinned memory
        # Auto-tune batch size using free memory heuristic
        eff_batch = self.batch_size
        if is_cuda:
            try:
                mem_free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
                target = max(32, int((mem_free / (0.5 * 1024**3)) * 128 * 0.6))
                eff_batch = max(32, min(self.batch_size, target))
            except Exception:
                pass

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None
        # GPU Optimization with spawn context to avoid CUDA fork issues
        num_workers = max(1, min(os.cpu_count() or 4, 8)) if is_cuda else 0
        if world_size > 1:
            num_workers = max(1, num_workers // world_size)
        loader_kwargs = {
            "batch_size": max(1, eff_batch // max(1, world_size)),
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

        fused_ok = is_cuda and hasattr(optim.AdamW, "fused") and "fused" in optim.AdamW.__init__.__code__.co_varnames
        optimizer = (
            optim.AdamW(self.model.parameters(), lr=self.lr, fused=fused_ok)
            if fused_ok
            else optim.AdamW(self.model.parameters(), lr=self.lr)
        )
        criterion = nn.CrossEntropyLoss()
        es_pat, es_delta = get_early_stop_params(8, 0.001)
        early_stopper = EarlyStopper(patience=es_pat, min_delta=es_delta)
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

        val_loader = None
        if len(X_val) > 0:
            val_dataset = TensorDataset(X_v, y_v)
            val_loader = DataLoader(
                val_dataset,
                batch_size=max(1, eff_batch // max(1, world_size)),
                shuffle=False,
                num_workers=0,
                pin_memory=is_cuda,
            )

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
                        if hasattr(scaler, "_lazy_init_scale_growth_tracker"):
                            scaler._lazy_init_scale_growth_tracker()
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph):
                            _step(static_bx, static_by)
                        logger.info("CUDA Graph captured for NBeats GPU step.")
                    except Exception as exc:
                        logger.warning(f"CUDA Graph capture failed for NBeats GPU: {exc}")
                        graph = None

                if graph is not None and static_bx is not None:
                    static_bx.copy_(bx)
                    static_by.copy_(by)
                    graph.replay()
                else:
                    _step(bx, by)

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_losses = []
                    for vbx, vby in val_loader:
                        vbx = vbx.to(self.device, non_blocking=is_cuda)
                        vby = vby.to(self.device, non_blocking=is_cuda)
                        val_out = self.model(vbx)
                        val_losses.append(criterion(val_out, vby).item())
                        del vbx, vby, val_out
                    val_loss = float(np.mean(val_losses)) if val_losses else 0.0
                self.model.train()
                if early_stopper(val_loss):
                    break

        if is_cuda:
            torch.cuda.empty_cache()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NBeats GPU model not loaded")
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
            raise RuntimeError("NBeats GPU inference returned no batches")
        probs = np.vstack(outs)
        # Unified Protocol: [Neutral, Buy, Sell] already in indices [0, 1, 2]
        return probs[:, :3]

    def save(self, path: str) -> None:
        if self.model and (not self.is_ddp or self.rank == 0):
            target = getattr(self.model, "module", self.model)
            if hasattr(target, "_orig_mod"):
                target = target._orig_mod
            # Save as standard .pt for compatibility with CPU loader
            torch.save(target.state_dict(), Path(path) / "nbeats.pt")
            joblib.dump(
                {
                    "input_dim": self.input_dim,
                    "hidden": self.hidden_dim,
                    "stack_types": self.stack_types,
                },
                Path(path) / "nbeats_meta.pkl",
            )

    def load(self, path: str) -> None:
        # GPU loader can just load the standard PT
        p = Path(path) / "nbeats.pt"
        m = Path(path) / "nbeats_meta.pkl"
        if p.exists() and m.exists():
            meta = joblib.load(m)
            self.input_dim = meta["input_dim"]
            self.hidden_dim = meta["hidden"]
            self.stack_types = meta.get("stack_types", self.stack_types)
            self.model = self._build_model()
            target = getattr(self.model, "module", self.model)
            try:
                state = torch.load(p, map_location=self.device, weights_only=True)
            except TypeError as exc:
                raise RuntimeError("PyTorch too old for weights_only load; upgrade for security.") from exc
            target.load_state_dict(state)

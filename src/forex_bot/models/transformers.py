import logging
import os
from pathlib import Path
from typing import Any  # Added this import

import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.optim as optim
from torch.utils.data import DataLoader

from .base import EarlyStopper, ExpertModel
from .device import (
    enable_flash_attention,
    enable_sdp_flash,
    gpu_supports_bf16,
    gpu_supports_fp8,
    select_device,
    tune_torch_backend,
)

logger = logging.getLogger(__name__)


class TimeframeEmbedding(nn.Module):
    """Learns a unique vector for each timeframe (M1...MN1)."""

    def __init__(self, d_model: int, num_timeframes: int = 12) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_timeframes, d_model)

    def forward(self, tf_idx: torch.Tensor) -> torch.Tensor:
        return self.emb(tf_idx)


class OmniFormerBackbone(nn.Module):
    """
    Shared-Weight Transformer that processes multiple timeframes in parallel.
    Input: (Batch, Num_Timeframes, Num_Features)
    Output: (Batch, D_Model) - fused representation
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_head: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        num_timeframes: int = 12,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.tf_embed = TimeframeEmbedding(d_model, num_timeframes)
        self.pos_embed = nn.Parameter(torch.randn(1, num_timeframes, d_model))
        self.n_head = n_head
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            # use PyTorch native SDPA (flash/mem-efficient) by leaving default norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.pool_attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape

        x = self.input_proj(x)  # (b, t, D)

        tf_indices = torch.arange(t, device=x.device).expand(b, t)
        x = x + self.tf_embed(tf_indices)
        x = x + self.pos_embed[:, :t, :]

        if hasattr(self, "_flash_attn") and self._flash_attn is not None:
            # Manual block using flash-attn when dimensions fit
            # reshape to (batch*timeframes, seq=1, d_model)
            qkv = x.reshape(b * t, 1, self.d_model).unsqueeze(2).expand(-1, -1, 3, -1)
            # flash_attn_func expects (B, S, H) fused qkv? We keep fallback if shapes mismatch
            try:
                out = self._flash_attn(qkv, dropout_p=0.0, softmax_scale=None, causal=False)
                x = out.reshape(b, t, self.d_model)
            except Exception:
                x = self.transformer(x)
        else:
            # Use TransformerEncoder (SDPA will choose flash kernels when available)
            x = self.transformer(x)
        x = self.norm(x)

        attn_weights = F.softmax(self.pool_attn(x), dim=1)  # (b, t, 1)
        fused = torch.sum(x * attn_weights, dim=1)  # (b, D)

        return fused


class DeployableOmniFormer(nn.Module):
    """
    Deploy-ready wrapper that includes reshaping logic for ONNX export.
    Accepts flat input, reshapes internally, and runs the model.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, num_timeframes: int, features_per_tf: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_timeframes = num_timeframes
        self.features_per_tf = features_per_tf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x_reshaped = x.view(b, self.num_timeframes, self.features_per_tf)

        feat = self.backbone(x_reshaped)
        return self.head(feat)


class TransformerExpertTorch(ExpertModel):
    """
    Wrapper for the Omni-Timeframe Transformer.
    Handles the complex data shaping (M5 + H1 + ... -> Tensor).
    """

    # 0 = no cap; set via attribute if needed
    MAX_TRAIN_ROWS = 0
    PRED_BATCH_SIZE = 256

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        lr: float = 1e-4,
        max_time_sec: int = 36000,
        device: str = "cpu",
        batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        self.model = None
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.lr = lr
        self.max_time_sec = max_time_sec
        self.device = select_device(device)
        self.batch_size = batch_size
        self.input_dim = 0  # Features per timeframe
        self.num_timeframes = 0
        self.tf_names = []
        self.base_feats = []
        self.use_te_fp8 = False
        self._te = None
        self._flash_attn = None
        tune_torch_backend(str(self.device))
        if str(self.device).startswith("cuda"):
            enable_flash_attention()
            enable_sdp_flash()

    def _build_model(self) -> DeployableOmniFormer:
        class Head(nn.Module):
            def __init__(self, d_model: int, output_dim: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, output_dim)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        backbone = OmniFormerBackbone(
            self.input_dim,
            d_model=self.d_model,
            n_head=self.n_heads,
            n_layers=self.n_layers,
            num_timeframes=self.num_timeframes,
        )
        head = Head(self.d_model, 3)

        return DeployableOmniFormer(backbone, head, self.num_timeframes, self.input_dim).to(self.device)

    def _init_layout(self, x: pd.DataFrame) -> None:
        if self.tf_names and self.base_feats:
            return
        prefixes = set()
        # Known timeframes ordered by duration (smallest to largest)
        known_tfs = ["M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "H12", "D1", "W1", "MN1"]

        # Detect present timeframes
        for c in x.columns:
            parts = c.split("_")
            if parts[0] in known_tfs:
                prefixes.add(parts[0])

        # If no prefixes found, treat entire input as single timeframe
        if not prefixes:
            self.tf_names = ["BASE"]  # Dummy name
            self.input_dim = x.shape[1]
            self.num_timeframes = 1
            self.base_feats = list(x.columns)
            return

        # Sort timeframes by their order in known_tfs
        # Unknown timeframes (if any slipped through) get sorted last
        self.tf_names = sorted(prefixes, key=lambda p: known_tfs.index(p) if p in known_tfs else 999)

        # Collect all unique feature suffixes
        feature_set = set()
        for c in x.columns:
            for tf in self.tf_names:
                if c.startswith(f"{tf}_"):
                    feature_set.add(c[len(tf) + 1 :])
                    break
            else:
                # Column has no known prefix, treat as base feature
                feature_set.add(c)

        self.base_feats = sorted(feature_set)
        self.input_dim = len(self.base_feats)
        self.num_timeframes = len(self.tf_names)

    def _make_dataset(self, x: pd.DataFrame, y: pd.Series | None = None):
        self._init_layout(x)
        tf_names = self.tf_names
        base_feats = self.base_feats
        input_dim = self.input_dim
        num_tf = len(tf_names)
        n_samples = len(x)

        # Vectorized pre-processing: Pre-allocate 3D array (N, T, F)
        # This consumes more RAM upfront but avoids the massive CPU bottleneck of row-wise processing during training.
        X_arr = np.zeros((n_samples, num_tf, input_dim), dtype=np.float32)

        # 1. Map columns to their (t, f) indices
        # We prefer "M5_close" -> t=idx(M5), f=idx(close)
        # Base features (no prefix) map to t=0 (first timeframe)
        
        # Create a mapping of column name -> (t_idx, f_idx)
        col_map = {}
        
        # Pre-compute indices for fast lookup
        tf_idx_map = {name: i for i, name in enumerate(tf_names)}
        feat_idx_map = {name: i for i, name in enumerate(base_feats)}

        for col in x.columns:
            # Try to match "TF_Feature" pattern
            matched = False
            for tf in tf_names:
                if col.startswith(f"{tf}_"):
                    feat_part = col[len(tf)+1:]
                    if feat_part in feat_idx_map:
                        t_idx = tf_idx_map[tf]
                        f_idx = feat_idx_map[feat_part]
                        col_map[col] = (t_idx, f_idx)
                        matched = True
                        break
            
            # Fallback: Treat as base feature for first timeframe if present in base_feats
            if not matched and col in feat_idx_map:
                 # Usually t=0 is the "base" or "fastest" timeframe
                 col_map[col] = (0, feat_idx_map[col])

        # 2. Fill the array using bulk numpy assignment
        # Iterate over columns (fast) instead of rows (slow)
        x_values = x.to_numpy(dtype=np.float32, copy=False)
        col_names = list(x.columns)
        
        for col_idx, col_name in enumerate(col_names):
            if col_name in col_map:
                t_idx, f_idx = col_map[col_name]
                X_arr[:, t_idx, f_idx] = x_values[:, col_idx]

        class _DS(torch.utils.data.Dataset):
            def __init__(self, data_array: np.ndarray, labels: np.ndarray | None) -> None:
                self.data = data_array
                self.labels = labels

            def __len__(self) -> int:
                return len(self.data)

            def __getitem__(self, idx: int):
                # Extremely fast slicing
                return self.data[idx], (self.labels[idx] if self.labels is not None else None)

        y_arr = y.to_numpy(dtype=int) if y is not None else None
        return _DS(X_arr, y_arr)

    def fit(self, x: pd.DataFrame, y: pd.Series, tensorboard_writer: Any | None = None) -> None:
        # Some callers may provide unnamed numeric columns (e.g., DataFrame from a raw numpy array).
        # Normalize to string names so layout parsing is stable.
        if not all(isinstance(c, str) for c in x.columns):
            x = x.copy()
            x.columns = [str(c) for c in x.columns]

        # Align device to local rank if distributed is initialized
        if dist.is_available() and dist.is_initialized() and torch.cuda.is_available():
            try:
                local_rank = dist.get_rank() % torch.cuda.device_count()
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f"cuda:{local_rank}")
            except Exception:
                pass

        # Robust Normalization (Critical for Transformer Stability)
        x_vals = x.select_dtypes(include=np.number).to_numpy(dtype=np.float32)
        self.mean_ = np.mean(x_vals, axis=0)
        self.scale_ = np.maximum(np.std(x_vals, axis=0), 1e-3)

        # Create normalized DataFrame (preserving index/columns for layout parser)
        x_norm = x.copy()
        x_norm[x.select_dtypes(include=np.number).columns] = (x_vals - self.mean_) / self.scale_

        self._init_layout(x_norm)
        self.model = self._build_model()
        stop_event = getattr(self, "stop_event", None)

        # Auto Transformer Engine FP8 when available on FP8-capable GPUs (disable with TE_FP8=0)
        if str(self.device).startswith("cuda") and gpu_supports_fp8(str(self.device)):
            try:
                import transformer_engine.pytorch as te

                te_flag = os.environ.get("TE_FP8", "auto").lower()
                self.use_te_fp8 = te_flag not in {"0", "false", "no"}
                if self.use_te_fp8:
                    self._te = te
                    logger.info("Transformer Engine FP8 enabled (TE_FP8!=0).")
            except Exception as e:
                self.use_te_fp8 = False
                logger.debug(f"Transformer Engine not available or failed to import: {e}")

        # Optional FlashAttention (v2/v3) if installed; falls back silently
        if str(self.device).startswith("cuda"):
            try:
                from flash_attn import flash_attn_func

                self._flash_attn = flash_attn_func
                logger.info("FlashAttention detected; will use when dimensions permit.")
            except Exception:
                self._flash_attn = None

        # Auto torch.compile when on GPU (disable with TORCH_COMPILE=0)
        compile_flag = os.environ.get("TORCH_COMPILE", "auto").lower()
        want_compile = compile_flag not in {"0", "false", "no"}
        if want_compile and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                self.model.to(self.device)
                from .device import maybe_compile
                self.model = maybe_compile(self.model, mode="reduce-overhead")
                logger.info("torch.compile enabled for Transformer (mode=reduce-overhead).")
            except Exception as e:
                logger.warning(f"torch.compile failed for Transformer, continuing without: {e}", exc_info=True)

        split = int(len(x_norm) * 0.85)
        x_train, x_val = x_norm.iloc[:split], x_norm.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        train_ds = self._make_dataset(x_train, y_train)
        val_ds = self._make_dataset(x_val, y_val)

        def _collate(batch):
            arrs, labels = zip(*batch, strict=False)
            tensor = torch.as_tensor(np.stack(arrs), dtype=torch.float32)
            lbl = torch.as_tensor(np.array(labels, dtype=np.int64) + 1, dtype=torch.long)
            return tensor, lbl

        is_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        pin_mem = is_cuda
        # Auto-tune batch size based on free GPU memory (fallback to configured batch_size)
        batch_size = self.batch_size
        if is_cuda:
            try:
                mem_free, mem_total = torch.cuda.mem_get_info(torch.cuda.current_device())
                # target ~0.5 GB per 128 batch of float32; scale accordingly with a safety factor
                target_bs = max(16, int((mem_free / (0.5 * 1024**3)) * 128 * 0.6))
                batch_size = max(16, min(self.batch_size, target_bs))
            except Exception:
                pass
        num_workers = max(0, min(os.cpu_count() or 0, 8) - 1) if is_cuda else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if world_size > 1 else None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(sampler is None),
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=is_cuda,
            prefetch_factor=2 if is_cuda else None,
            collate_fn=_collate,
            sampler=sampler,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=is_cuda,
            prefetch_factor=2 if is_cuda else None,
            collate_fn=_collate,
        )

        fused_ok = torch.cuda.is_available() and str(self.device).startswith("cuda") and hasattr(optim.AdamW, "fused")
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, fused=fused_ok)
        criterion = nn.CrossEntropyLoss()
        early_stopper = EarlyStopper(patience=8, min_delta=0.001)

        use_amp = is_cuda
        amp_dtype = torch.bfloat16 if (use_amp and gpu_supports_bf16(str(self.device))) else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024.0)

        self.model.train()
        train_model = self.model
        # Wrap with DDP if distributed is initialized
        if dist.is_available() and dist.is_initialized() and world_size > 1 and is_cuda:
            try:
                local_rank = torch.cuda.current_device()
                train_model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
                )
            except Exception as e:
                logger.warning(f"DDP wrap failed, continuing without DDP: {e}")
        import time

        start = time.time()
        use_cuda_graph = bool(int(os.environ.get("CUDA_GRAPH", "0"))) and is_cuda
        static_bx = static_by = graph = None

        def _train_step(batch_x, batch_y):
            cm = (
                torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype)
                if not self.use_te_fp8 or self._te is None
                else self._te.fp8_autocast()
            )
            optimizer.zero_grad(set_to_none=True)
            with cm:
                out = train_model(batch_x)
                loss = criterion(out, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return loss.detach()

        for epoch in range(50):
            if (stop_event is not None and getattr(stop_event, "is_set", lambda: False)()) or (
                time.time() - start > self.max_time_sec
            ):
                break
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_loss = 0.0
            steps = 0
            for bx, by in train_loader:
                if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                    break
                if is_cuda:
                    bx = bx.to(self.device, non_blocking=True)
                    by = by.to(self.device, non_blocking=True)
                else:
                    bx = bx.to(self.device)
                    by = by.to(self.device)

                if use_cuda_graph and graph is None and bx.is_contiguous() and by.is_contiguous():
                    try:
                        static_bx = bx.clone()
                        static_by = by.clone()
                        graph = torch.cuda.CUDAGraph()
                        # Warm-up step to set static shapes
                        _ = _train_step(static_bx, static_by)
                        torch.cuda.synchronize()
                        scaler._lazy_init_scale_growth_tracker()
                        with torch.cuda.graph(graph):
                            _train_step(static_bx, static_by)
                        logger.info("CUDA Graph captured for Transformer training step.")
                    except Exception as e:
                        logger.warning(f"CUDA Graph capture failed, falling back: {e}")
                        graph = None

                if graph is not None and static_bx is not None:
                    static_bx.copy_(bx)
                    static_by.copy_(by)
                    graph.replay()
                    loss_val = scaler.get_scale()  # dummy; not exact
                else:
                    loss_val = _train_step(bx, by)

                if isinstance(loss_val, torch.Tensor):
                    epoch_loss += float(loss_val)
                steps += 1

            avg_train_loss = epoch_loss / max(1, steps)
            if tensorboard_writer:
                tensorboard_writer.add_scalar("Loss/train_Transformer", avg_train_loss, epoch)

            if len(val_ds) > 0:
                train_model.eval()
                val_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        if is_cuda:
                            vx = vx.to(self.device, non_blocking=True)
                            vy = vy.to(self.device, non_blocking=True)
                        else:
                            vx = vx.to(self.device)
                            vy = vy.to(self.device)
                        val_out = train_model(vx)
                        val_loss += criterion(val_out, vy).item()
                        val_steps += 1
                val_loss = val_loss / max(1, val_steps)
                train_model.train()

                if tensorboard_writer:
                    tensorboard_writer.add_scalar("Loss/val_Transformer", val_loss, epoch)

                if early_stopper(val_loss):
                    logger.info(f"Transformer early stopping at epoch {epoch}, val_loss={val_loss:.4f}")
                    break

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Transformer model not loaded")
        if not all(isinstance(c, str) for c in x.columns):
            x = x.copy()
            x.columns = [str(c) for c in x.columns]

        self.model.eval()

        # Apply Normalization
        if hasattr(self, "mean_") and self.mean_ is not None:
            x_vals = x.select_dtypes(include=np.number).to_numpy(dtype=np.float32)
            # Safe scale to avoid shape mismatch if columns differ slightly, though usually matched
            try:
                x_norm = x.copy()
                x_norm[x.select_dtypes(include=np.number).columns] = (x_vals - self.mean_) / self.scale_
                x = x_norm
            except Exception as e:
                logger.warning(f"Transformer norm failed at inference: {e}")

        self._init_layout(x)
        ds = self._make_dataset(x, None)

        def _collate_pred(batch):
            arrs, _ = zip(*batch, strict=False)
            tensor = torch.as_tensor(np.stack(arrs), dtype=torch.float32)
            return tensor

        outputs: list[np.ndarray] = []
        is_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        pin_mem = is_cuda
        with torch.inference_mode():
            for bx in DataLoader(
                ds,
                batch_size=self.PRED_BATCH_SIZE,
                shuffle=False,
                pin_memory=pin_mem,
                num_workers=max(0, min(os.cpu_count() or 0, 8) - 1) if is_cuda else 0,
                persistent_workers=is_cuda,
                prefetch_factor=2 if is_cuda else None,
                collate_fn=_collate_pred,
            ):
                if is_cuda:
                    bx = bx.to(self.device, non_blocking=True)
                else:
                    bx = bx.to(self.device)
                logits = self.model(bx)
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        if not outputs:
            raise RuntimeError("Transformer inference returned no batches")
        probs = np.vstack(outputs)
        # Training encodes labels as y+1 where y ∈ {-1,0,1} -> class indices {0,1,2}.
        # Reorder to project convention [neutral, buy, sell].
        return probs[:, [1, 2, 0]]

    def save(self, path: str) -> None:
        if self.model:
            target = getattr(self.model, "module", self.model)
            if hasattr(target, "_orig_mod"):
                target = target._orig_mod
            torch.save(target.state_dict(), Path(path) / "omniformer.pt")
            meta = {
                "d_model": self.d_model,
                "tf_names": self.tf_names,
                "input_dim": self.input_dim,
            }
            joblib.dump(meta, Path(path) / "omniformer_meta.pkl")

    def load(self, path: str) -> None:
        p = Path(path) / "omniformer.pt"
        m = Path(path) / "omniformer_meta.pkl"
        if not (p.exists() and m.exists()):
            return
        try:
            meta = joblib.load(m)
            self.d_model = meta.get("d_model", self.d_model)
            self.tf_names = meta.get("tf_names", [])
            self.input_dim = meta.get("input_dim", self.input_dim)
            self.num_timeframes = len(self.tf_names) or self.num_timeframes

            self.model = self._build_model()
            target = getattr(self.model, "module", self.model)
            try:
                state = torch.load(p, map_location=self.device, weights_only=True)
            except TypeError as exc:
                raise RuntimeError("PyTorch too old for weights_only load; upgrade for security.") from exc
            target.load_state_dict(state)
            self.model = target.to(self.device)

            if str(self.device) == "cpu":
                self._optimize_for_cpu_inference()
        except Exception as exc:
            logger.warning(f"Transformer load failed: {exc}")

    def _optimize_for_cpu_inference(self):
        if self.model is None:
            return
        try:
            self.model.eval()
            # Transformer inputs are flat (Batch, num_timeframes * features_per_tf) and reshaped internally.
            flat_dim = int(self.input_dim) * int(self.num_timeframes or 1)
            dummy_input = torch.randn(1, max(1, flat_dim))
            # Torch's Transformer fast-path can lead to trace check failures (graphs differ across invocations).
            traced = torch.jit.trace(self.model, dummy_input, check_trace=False, strict=False)
            self.model = torch.jit.freeze(traced)
            logger.info(f"JIT Compiled {self.__class__.__name__} for CPU.")
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")

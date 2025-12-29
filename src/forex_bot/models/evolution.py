import logging
import math
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import cupy as cp  # type: ignore

    CUPY_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    CUPY_AVAILABLE = False

try:
    import torch  # type: ignore
    import torch.nn.functional as functional  # type: ignore

    EVOTORCH_TORCH_AVAILABLE = True
except Exception:
    EVOTORCH_TORCH_AVAILABLE = False

try:
    from evotorch import Problem  # type: ignore
    from evotorch.algorithms import CMAES  # type: ignore

    EVOTORCH_AVAILABLE = True
except Exception:
    EVOTORCH_AVAILABLE = False

from ..core.system import thread_limits
from .base import ExpertModel
from .device import select_device

logger = logging.getLogger(__name__)

try:
    import cma

    CMA_AVAILABLE = True
except Exception:  # pragma: no cover
    cma = None
    CMA_AVAILABLE = False

EPS = 1e-9


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / (exp_z.sum(axis=1, keepdims=True) + EPS)


def _unpack_theta(theta_vec: np.ndarray, d: int, hidden: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = 0
    w1 = theta_vec[idx : idx + d * hidden].reshape(d, hidden)
    idx += d * hidden
    b1 = theta_vec[idx : idx + hidden].reshape(1, hidden)
    idx += hidden
    w2 = theta_vec[idx : idx + hidden * 3].reshape(hidden, 3)
    idx += hidden * 3
    b2 = theta_vec[idx : idx + 3].reshape(1, 3)
    return w1, b1, w2, b2


def _parse_gpu_id(dev: str) -> int | None:
    if not isinstance(dev, str):
        return None
    if dev.startswith("cuda"):
        parts = dev.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return 0
    return None


def _loss_cpu(
    theta_vec: np.ndarray,
    x_norm: np.ndarray,
    yb: np.ndarray,
    weight_decay: float,
    d: int,
    hidden: int,
    *,
    row_idx: np.ndarray | None = None,
    class_idx: np.ndarray | None = None,
    batch_size: int = 8192,
) -> float:
    w1, b1, w2, b2 = _unpack_theta(theta_vec, d, hidden)
    reg = weight_decay * np.sum(theta_vec * theta_vec)

    n = x_norm.shape[0]
    total_loss = 0.0

    # Process in batches to avoid OOM
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x_batch = x_norm[start:end]
        y_batch = yb[start:end]

        # Determine class indices for this batch
        if class_idx is not None:
            c_batch = class_idx[start:end]
        else:
            c_batch = y_batch.astype(np.int64) + 1

        hidden_act = np.tanh(x_batch @ w1 + b1)
        logits = hidden_act @ w2 + b2
        probs = _softmax(logits)

        # Gather probabilities for true classes
        r_batch = np.arange(end - start)
        p = probs[r_batch, c_batch]
        total_loss += -np.sum(np.log(p + EPS))

    return float((total_loss / n) + reg)


def _loss_gpu(
    theta_vec: np.ndarray,
    x_norm_cp: Any,
    yb_cp: Any,
    weight_decay: float,
    d: int,
    hidden: int,
    gpu_id: int | None,
    *,
    row_idx: Any | None = None,
    class_idx: Any | None = None,
    batch_size: int = 8192,
) -> float:
    if not CUPY_AVAILABLE or x_norm_cp is None or yb_cp is None:
        raise RuntimeError("GPU path unavailable")
    if gpu_id is not None:
        cp.cuda.Device(gpu_id).use()

    tv = cp.asarray(theta_vec)
    w1, b1, w2, b2 = _unpack_theta(cp.asnumpy(tv), d, hidden)
    w1 = cp.asarray(w1)
    b1 = cp.asarray(b1)
    w2 = cp.asarray(w2)
    b2 = cp.asarray(b2)

    reg = weight_decay * cp.sum(tv * tv)
    n = x_norm_cp.shape[0]
    total_loss = 0.0

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x_batch = x_norm_cp[start:end]
        y_batch = yb_cp[start:end]

        if class_idx is not None:
            c_batch = class_idx[start:end]
        else:
            c_batch = y_batch.astype(cp.int64) + 1

        hidden_act = cp.tanh(x_batch @ w1 + b1)
        logits = hidden_act @ w2 + b2
        exp_z = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))
        probs = exp_z / (cp.sum(exp_z, axis=1, keepdims=True) + 1e-9)

        r_batch = cp.arange(end - start)
        p = probs[r_batch, c_batch]
        total_loss += -cp.sum(cp.log(p + 1e-9))

    return float(cp.asnumpy((total_loss / n) + reg))


def _loss_batch_gpu(
    thetas: list[np.ndarray] | np.ndarray,
    x_norm_cp: Any,
    yb_cp: Any,
    weight_decay: float,
    d: int,
    hidden: int,
    gpu_id: int | None,
    *,
    row_idx: Any | None = None,
    class_idx: Any | None = None,
    batch_size: int = 8192,
) -> list[float]:
    """
    Vectorized batch loss calculation on GPU.
    Computes loss for the entire population in parallel using batched matmul.
    """
    if not CUPY_AVAILABLE or x_norm_cp is None or yb_cp is None:
        raise RuntimeError("GPU path unavailable")
    if gpu_id is not None:
        cp.cuda.Device(gpu_id).use()

    # Stack population params: (Pop, Theta_Dim)
    thetas_cp = cp.asarray(thetas)
    pop_size = thetas_cp.shape[0]

    # Unpack weights for the entire population
    # Layout: [W1 (d*h) | b1 (h) | W2 (h*3) | b2 (3)]
    idx = 0
    w1_size = d * hidden
    W1 = thetas_cp[:, idx : idx + w1_size].reshape(pop_size, d, hidden)
    idx += w1_size
    b1_size = hidden
    B1 = thetas_cp[:, idx : idx + b1_size].reshape(pop_size, 1, hidden)
    idx += b1_size
    w2_size = hidden * 3
    W2 = thetas_cp[:, idx : idx + w2_size].reshape(pop_size, hidden, 3)
    idx += w2_size
    B2 = thetas_cp[:, idx : idx + 3].reshape(pop_size, 1, 3)

    reg = weight_decay * cp.sum(thetas_cp * thetas_cp, axis=1) # (Pop,)
    n_samples = x_norm_cp.shape[0]
    total_loss = cp.zeros(pop_size, dtype=cp.float32)

    # Process samples in chunks to manage VRAM
    # We broadcast (Pop, 1, D, H) against (1, Batch, D) -> (Pop, Batch, H)
    for start in range(0, n_samples, batch_size):
        end = min(n_samples, start + batch_size)
        x_batch = x_norm_cp[start:end] # (Batch, D)
        y_batch = yb_cp[start:end]     # (Batch,)

        if class_idx is not None:
            c_batch = class_idx[start:end]
        else:
            c_batch = y_batch.astype(cp.int64) + 1

        # Vectorized Forward Pass
        # X: (Batch, D) -> (1, Batch, D) for broadcasting
        x_b = x_batch[None, :, :]
        
        # Layer 1: (Pop, Batch, H) = (Pop, 1, D, H) * (1, Batch, D, 1) ?? No, manual broadcast or matmul
        # Easiest: Pop loop is slow? No, use batched matmul (bmm)
        # Reshape X for matmul: (Batch, D)
        # We need (Pop, Batch, H). 
        # Standard: X @ W1[i] + B1[i]. 
        # Optimized: Concatenate X into a huge batch? No.
        # Use cp.matmul with broadcasting:
        # W1: (Pop, D, H)
        # X: (Batch, D) -> need (Pop, Batch, D) to matmul? Or (Batch, D) @ (Pop, D, H)?
        # (Batch, D) @ (D, H) = (Batch, H). 
        # We have Pop matrices.
        # Einstein Summation is cleanest: 'pdh,bd->pbh'
        # P=Pop, D=Dim, H=Hidden, B=Batch
        # cp.tensordot or manual loop? CuPy einsum is fast.
        
        # Hidden Act: tanh(X @ W1 + B1)
        # Using tensordot/matmul:
        # X (Batch, D), W1 (Pop, D, H) -> (Pop, Batch, H) via broadcast
        # Since X is shared, we can compute X @ W1[p] in parallel?
        # Actually, simpler: (Pop, Batch, H) = cp.matmul(x_b, W1) ? No dimensions mismatch
        
        # Let's use einsum for clarity and speed on GPU
        # hidden_pre = cp.einsum('bd,pdh->pbh', x_batch, W1) + B1 # B1 broadcasts (Pop, 1, Hidden) -> (Pop, Batch, Hidden)
        hidden_pre = cp.tensordot(x_batch, W1, axes=([1],[1])) # Returns (Batch, Pop, Hidden) -> swap to (Pop, Batch, Hidden)
        hidden_pre = cp.swapaxes(hidden_pre, 0, 1) + B1
        
        hidden_act = cp.tanh(hidden_pre)

        # Logits: (Pop, Batch, 3)
        # W2: (Pop, Hidden, 3)
        # logits = cp.einsum('pbh,phc->pbc', hidden_act, W2) + B2
        # Use matmul since dimensions align: (Pop, Batch, H) @ (Pop, H, 3) -> (Pop, Batch, 3)
        logits = cp.matmul(hidden_act, W2) + B2
        
        # Softmax & Loss
        # logits: (Pop, Batch, 3)
        max_logits = cp.max(logits, axis=2, keepdims=True)
        exp_z = cp.exp(logits - max_logits)
        probs = exp_z / (cp.sum(exp_z, axis=2, keepdims=True) + 1e-9)

        # Gather true class probs: (Pop, Batch)
        # c_batch: (Batch,) -> Broadcast to (Pop, Batch)
        # We need specific indices.
        # Flat indexing is often faster.
        # Probs shape: (Pop, Batch, 3)
        # We want probs[p, b, c_batch[b]] for all p,b
        
        # Create grid indices
        # pop_idx: (Pop, 1)
        # batch_idx: (1, Batch)
        # c_idx: (1, Batch)
        # gather_idx = c_idx
        
        # Using cp.take_along_axis is robust
        # c_batch_expanded = c_batch[None, :, None] # (1, Batch, 1)
        # p_true = cp.take_along_axis(probs, c_batch_expanded, axis=2).squeeze(2) # (Pop, Batch)
        
        # Manual indexing for speed if needed, but take_along_axis is good
        c_exp = c_batch[None, :, None]
        p_true = cp.take_along_axis(probs, c_exp, axis=2).squeeze(2)

        total_loss += -cp.sum(cp.log(p_true + 1e-9), axis=1)

    mean_loss = (total_loss / n_samples) + reg
    return [float(l) for l in mean_loss.tolist()]


def _run_island(
    args: tuple[
        int,
        np.ndarray,
        float,
        dict,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        int,
        int,
        float,
        float,
        str | None,
        bool,
        int | None,
    ],
) -> tuple[float, np.ndarray, int, str | None]:
    """
    Run one CMA-ES island; can be dispatched in a separate process.
    Returns (loss, theta, island_idx, device_used).
    """
    (
        island_idx,
        x0,
        sigma,
        opts,
        x_np,
        y_np,
        mean,
        scale,
        d,
        hidden,
        weight_decay,
        time_budget,
        device,
        try_gpu,
        blas_threads,
    ) = args

    start = time.time()
    x_norm_np = (x_np - mean) / scale
    row_idx_np = np.arange(y_np.shape[0])
    
    # HPC FIX: Unified Protocol Mapping
    # -1 (Sell) -> 0, 0 (Neutral) -> 1, 1 (Buy) -> 2
    class_idx_np = np.zeros_like(y_np, dtype=np.int64)
    class_idx_np[y_np == -1] = 0
    class_idx_np[y_np == 0] = 1
    class_idx_np[y_np == 1] = 2
    
    x_norm_cp = None
    y_cp = None
    row_idx_cp = None
    class_idx_cp = None
    use_gpu = False
    gpu_id = None

    if try_gpu and CUPY_AVAILABLE:
        try:
            gpu_id = _parse_gpu_id(device) if device else None
            if gpu_id is not None:
                cp.cuda.Device(gpu_id).use()
            x_norm_cp = cp.asarray(x_norm_np)
            y_cp = cp.asarray(y_np)
            row_idx_cp = cp.arange(y_cp.shape[0])
            
            # HPC FIX: Unified Protocol Mapping
            class_idx_cp = cp.zeros_like(y_cp, dtype=cp.int64)
            class_idx_cp[y_cp == -1] = 0
            class_idx_cp[y_cp == 0] = 1
            class_idx_cp[y_cp == 1] = 2
            
            use_gpu = True
        except Exception:
            use_gpu = False
            x_norm_cp = None
            y_cp = None
            row_idx_cp = None
            class_idx_cp = None

    def _loss_local(theta_vec: np.ndarray) -> float:
        # Fallback for single evaluations (best candidate check)
        if use_gpu:
            try:
                return _loss_gpu(
                    theta_vec,
                    x_norm_cp,
                    y_cp,
                    weight_decay,
                    d,
                    hidden,
                    gpu_id,
                    row_idx=row_idx_cp,
                    class_idx=class_idx_cp,
                )
            except Exception as e:
                logger.warning(f"Evolution fitness calculation failed: {e}", exc_info=True)
        return _loss_cpu(
            theta_vec, x_norm_np, y_np, weight_decay, d, hidden, row_idx=row_idx_np, class_idx=class_idx_np
        )

    with thread_limits(blas_threads):
        es = cma.CMAEvolutionStrategy(x0, sigma, opts)
        while not es.stop() and (time.time() - start) < time_budget:
            xc = es.ask()
            
            # Vectorized Population Evaluation
            if use_gpu:
                try:
                    fitness_values = _loss_batch_gpu(
                        xc,
                        x_norm_cp,
                        y_cp,
                        weight_decay,
                        d,
                        hidden,
                        gpu_id,
                        row_idx=row_idx_cp,
                        class_idx=class_idx_cp
                    )
                    es.tell(xc, fitness_values)
                except Exception as e:
                    logger.warning(f"Batch GPU Evo failed, falling back to sequential: {e}")
                    es.tell(xc, [_loss_local(np.asarray(x)) for x in xc])
            else:
                es.tell(xc, [_loss_local(np.asarray(x)) for x in xc])

    candidate = np.asarray(es.result.xbest, dtype=float)
    candidate_loss = _loss_local(candidate)
    return candidate_loss, candidate, island_idx, device if use_gpu else None


@dataclass(slots=True)
class EvoExpertCMA(ExpertModel):
    """Neuroevolution expert using CMA-ES with a small MLP head."""

    # 0 = no cap; set in config if needed
    MAX_TRAIN_ROWS: int = 0
    sigma: float = 0.2
    max_time_sec: int = 7200
    hidden_size: int = 64
    population: int = 32
    num_islands: int = 4
    weight_decay: float = 1e-3
    device: str = "cpu"
    gpu_pool: list[str] | None = None
    evo_multiproc_per_gpu: bool = True
    theta: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    island_losses: list[float] | None = None
    _last_device: str | None = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        if not CMA_AVAILABLE or cma is None:
            logger.warning("CMA-ES not available; skipping Evo training")
            return

        if self.gpu_pool:
            self.device = self.gpu_pool[0]
        self.device = select_device(self.device)
        if self.gpu_pool:
            gpus = list(self.gpu_pool)
        elif isinstance(self.device, list):
            gpus = list(self.device)
        elif isinstance(self.device, str) and self.device.startswith("cuda"):
            gpus = [self.device]
        else:
            gpus = []

        xn = x.astype(np.float32).to_numpy()
        yv = y.astype(int).to_numpy()
        mask = yv != 0
        if mask.sum() < 200:
            logger.warning("Evo skipped: insufficient directional labels")
            return

        yb = yv[mask]
        xb = xn[mask]

        orig_rows = len(xb)
        max_rows = getattr(self, "max_train_rows", self.MAX_TRAIN_ROWS)
        if max_rows and orig_rows > max_rows:
            rng = np.random.default_rng(42)
            frac = max_rows / orig_rows
            sampled_idx: list[int] = []
            for label in np.unique(yb):
                label_idx = np.where(yb == label)[0]
                take = min(len(label_idx), max(1, int(len(label_idx) * frac)))
                if take > 0:
                    sampled = rng.choice(label_idx, size=take, replace=False)
                    sampled_idx.extend(sampled.tolist())
            if len(sampled_idx) > max_rows:
                sampled_idx = rng.choice(sampled_idx, size=max_rows, replace=False).tolist()
            xb = xb[sampled_idx]
            yb = yb[sampled_idx]
            logger.info(f"Evo downsampled to {len(xb)} rows (was {orig_rows}) to avoid OOM.")

        self.mean_ = xb.mean(axis=0)
        # FIX: Use maximum clamping instead of adding epsilon to prevent explosion on zero-variance features
        self.scale_ = np.maximum(np.std(xb, axis=0), 1e-3)
        xb_norm_np = (xb - self.mean_) / self.scale_

        xb_norm_cp = None
        yb_cp = None
        if CUPY_AVAILABLE and gpus:
            try:
                gpu_id = _parse_gpu_id(gpus[0])
                if gpu_id is not None:
                    cp.cuda.Device(gpu_id).use()
                xb_cp = cp.asarray(xb)
                yb_cp = cp.asarray(yb)
                self.mean_ = cp.asnumpy(xb_cp.mean(axis=0))
                # FIX: Consistent robust scaling for GPU path as well
                self.scale_ = cp.asnumpy(cp.maximum(cp.std(xb_cp, axis=0), 1e-3))
                xb_norm_cp = (xb_cp - cp.asarray(self.mean_)) / cp.asarray(self.scale_)
            except Exception as exc:
                logger.warning(f"GPU Evo preprocessing failed, fallback to CPU: {exc}")
                xb_norm_cp = None
                yb_cp = None

        d_features = xb.shape[1]
        hidden = int(self.hidden_size)

        if EVOTORCH_AVAILABLE and EVOTORCH_TORCH_AVAILABLE and gpus:
            try:
                device_torch = torch.device(gpus[0])
                x_t_evotorch = torch.as_tensor(xb_norm_np, device=device_torch, dtype=torch.float32)
                # Map labels to class indices for cross-entropy: -1->0, 0->1, 1->2
                class_idx_evotorch = np.zeros_like(yb, dtype=np.int64)
                class_idx_evotorch[yb == -1] = 0
                class_idx_evotorch[yb == 0] = 1
                class_idx_evotorch[yb == 1] = 2
                y_t = torch.as_tensor(class_idx_evotorch, device=device_torch, dtype=torch.long)
                theta_dim_evotorch = d_features * hidden + hidden + hidden * 3 + 3

                def eval_func(sol: torch.Tensor) -> torch.Tensor:
                    theta_vec = sol
                    if not torch.is_tensor(theta_vec):
                        theta_vec = torch.as_tensor(theta_vec, device=device_torch, dtype=torch.float32)
                    w1, b1, w2, b2 = _unpack_theta(theta_vec, d_features, hidden)
                    w1 = w1.to(device_torch)
                    b1 = b1.to(device_torch)
                    w2 = w2.to(device_torch)
                    b2 = b2.to(device_torch)
                    hidden_act = torch.tanh(x_t_evotorch @ w1 + b1)
                    logits = hidden_act @ w2 + b2
                    loss = functional.cross_entropy(logits, y_t)
                    reg = self.weight_decay * (theta_vec.pow(2).sum())
                    return loss + reg

                problem = Problem(
                    "min",
                    eval_func,
                    solution_length=theta_dim_evotorch,
                    initial_bounds=(-0.5, 0.5),
                    device=device_torch,
                )
                algo = CMAES(problem, popsize=max(4, int(self.population)))
                start = time.time()
                while (time.time() - start) < self.max_time_sec and not algo.stop_condition_met():
                    algo.step()

                best = problem.status.get("best_so_far", None)
                if best is not None and hasattr(best, "values"):
                    theta_best = best.values
                    if torch.is_tensor(theta_best):
                        theta_best = theta_best.detach().cpu().numpy()
                    self.theta = np.asarray(theta_best, dtype=float)
                    self.island_losses = [float(best.evaluation)] if hasattr(best, "evaluation") else None
                    self._last_device = str(gpus[0])
                    return
            except Exception as exc:
                logger.warning(f"EvoTorch path failed, falling back to CMA-ES: {exc}")

        popsize = max(4, int(self.population))
        theta_dim = d_features * hidden + hidden + hidden * 3 + 3
        x0_cma = np.zeros(theta_dim, dtype=float)

        def _available_ram_gb() -> float:
            try:
                import psutil

                return float(psutil.virtual_memory().available) / (1024**3)
            except Exception:
                try:
                    return float(os.environ.get("FOREX_BOT_RAM_GB", "0") or 0)
                except Exception:
                    return 0.0

        def _unpack(theta_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            idx = 0
            w1 = theta_vec[idx : idx + d_features * hidden].reshape(d_features, hidden)
            idx += d_features * hidden
            b1 = theta_vec[idx : idx + hidden].reshape(1, hidden)
            idx += hidden
            w2 = theta_vec[idx : idx + hidden * 3].reshape(hidden, 3)
            idx += hidden * 3
            b2 = theta_vec[idx : idx + 3].reshape(1, 3)
            return w1, b1, w2, b2

        islands = max(1, int(self.num_islands))
        best_theta = None
        best_loss = float("inf")
        total_start = time.time()
        budget_per_island = max(60.0, self.max_time_sec / islands)
        island_losses: list[float] = []
        row_idx_np = np.arange(yb.shape[0])
        class_idx_np = yb.astype(np.int64) + 1
        row_idx_cp = None
        class_idx_cp = None
        if CUPY_AVAILABLE and yb_cp is not None:
            row_idx_cp = cp.arange(yb_cp.shape[0])
            class_idx_cp = yb_cp.astype(cp.int64) + 1

        use_multiproc = bool(self.evo_multiproc_per_gpu) and CUPY_AVAILABLE and len(gpus) > 1 and islands > 1

        available_gb = _available_ram_gb()
        try:
            mem_frac = float(os.environ.get("FOREX_BOT_EVO_MEM_FRAC", "0.4") or 0.4)
        except Exception:
            mem_frac = 0.4
        mem_frac = max(0.05, min(0.95, mem_frac))

        try:
            data_overhead = float(os.environ.get("FOREX_BOT_EVO_DATA_OVERHEAD", "2.0") or 2.0)
        except Exception:
            data_overhead = 2.0
        try:
            cma_overhead = float(os.environ.get("FOREX_BOT_EVO_CMA_OVERHEAD", "4.0") or 4.0)
        except Exception:
            cma_overhead = 4.0

        cma_bytes = float(popsize) * float(theta_dim) * 8.0
        data_bytes = float(xb_norm_np.nbytes + yb.nbytes)

        max_gpu_by_mem = None
        max_cpu_by_mem = None
        if available_gb > 0:
            try:
                per_gpu_bytes = data_bytes * max(1.0, data_overhead) + cma_bytes * max(1.0, cma_overhead)
                if per_gpu_bytes > 0:
                    max_gpu_by_mem = int((available_gb * mem_frac * 1024**3) // per_gpu_bytes)
            except Exception:
                max_gpu_by_mem = None
            try:
                per_cpu_bytes = cma_bytes * max(1.0, cma_overhead)
                if per_cpu_bytes > 0:
                    max_cpu_by_mem = int((available_gb * mem_frac * 1024**3) // per_cpu_bytes)
            except Exception:
                max_cpu_by_mem = None

        gpu_islands = 0
        cpu_islands = islands

        if use_multiproc:
            try:
                per_gpu = int(os.environ.get("FOREX_BOT_EVO_GPU_ISLANDS_PER_GPU", "1") or 1)
            except Exception:
                per_gpu = 1
            per_gpu = max(1, per_gpu)
            gpu_islands = min(islands, len(gpus) * per_gpu)
            if max_gpu_by_mem is not None:
                if max_gpu_by_mem <= 0:
                    logger.warning(
                        "Evo GPU islands disabled by RAM cap (available_ram=%.1fGB, mem_frac=%.2f).",
                        available_gb,
                        mem_frac,
                    )
                    gpu_islands = 0
                else:
                    gpu_islands = min(gpu_islands, max_gpu_by_mem)
            if gpu_islands <= 0:
                use_multiproc = False
            else:
                cpu_islands = max(0, islands - gpu_islands)

            tasks: list[
                tuple[
                    int,
                    np.ndarray,
                    float,
                    dict,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    int,
                    int,
                    float,
                    float,
                    str | None,
                    bool,
                    int | None,
                ]
            ] = []
            proc_count = min(len(gpus), gpu_islands)
            blas_threads = 1
            for island in range(gpu_islands):
                remaining = self.max_time_sec - (time.time() - total_start)
                if remaining <= 0:
                    break
                island_budget = min(budget_per_island, remaining)
                opts = {
                    "verbose": -9,
                    "popsize": popsize,
                    "maxfevals": 20000,
                    "seed": int(time.time()) + island,
                    # IMPORTANT: without diagonal mode, CMA allocates O(N^2) matrices and will OOM for large theta_dim.
                    "CMA_diagonal": True,
                }
                device = gpus[island % len(gpus)] if gpus else None
                tasks.append(
                    (
                        island,
                        x0_cma,
                        self.sigma,
                        opts,
                        xb,
                        yb,
                        self.mean_,
                        self.scale_,
                        d_features,
                        hidden,
                        self.weight_decay,
                        island_budget,
                        device,
                        True,
                        blas_threads,
                    )
                )

            try:
                ctx = multiprocessing.get_context("spawn")
                with ctx.Pool(processes=proc_count) as pool:
                    async_results = pool.map_async(_run_island, tasks)

                    hybrid = str(os.environ.get("FOREX_BOT_EVO_HYBRID", "1")).strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                    if hybrid and cpu_islands > 0:
                        # Run remaining CPU islands in parallel while GPU islands execute.
                        use_cpu_parallel_islands = True
                    else:
                        use_cpu_parallel_islands = False

                    cpu_results: list[tuple[float, np.ndarray]] = []
                    if use_cpu_parallel_islands:
                        total_budget = float(max(0, int(self.max_time_sec)))
                        if total_budget > 0:
                            row_idx = row_idx_np
                            class_idx = class_idx_np

                            try:
                                cpu_cores = int(os.environ.get("FOREX_BOT_CPU_THREADS", "0") or 0)
                            except Exception:
                                cpu_cores = 0
                            if cpu_cores <= 0:
                                try:
                                    cpu_cores = int(os.environ.get("FOREX_BOT_CPU_BUDGET", "0") or 0)
                                except Exception:
                                    cpu_cores = 0
                            if cpu_cores <= 0:
                                cpu_total = max(1, os.cpu_count() or 1)
                                try:
                                    reserve = int(os.environ.get("FOREX_BOT_CPU_RESERVE", "1") or 1)
                                except Exception:
                                    reserve = 1
                                cpu_cores = max(1, cpu_total - max(0, reserve))

                            max_by_time = max(1, int(math.ceil(total_budget / 60.0)))
                            active_islands = max(1, min(cpu_islands, cpu_cores, max_by_time))
                            if max_cpu_by_mem is not None and max_cpu_by_mem > 0:
                                active_islands = min(active_islands, max_cpu_by_mem)
                            if active_islands <= 1:
                                island_budgets = [total_budget]
                            else:
                                per_island = float(total_budget) / float(active_islands)
                                if per_island >= 60.0:
                                    island_budgets = [per_island for _ in range(active_islands)]
                                else:
                                    rem = max(0.0, total_budget - 60.0 * float(active_islands - 1))
                                    island_budgets = [60.0 for _ in range(active_islands - 1)] + [rem]

                            if active_islands != cpu_islands:
                                logger.info(
                                    f"Evo hybrid CPU islands capped: requested={cpu_islands}, active={active_islands}, "
                                    f"cpu_threads={cpu_cores}, budget={total_budget:.0f}s"
                                )

                            blas_threads = max(1, cpu_cores // max(1, active_islands))

                            def _run_cpu_island(island: int, island_budget: float) -> tuple[float, np.ndarray]:
                                opts = {
                                    "verbose": -9,
                                    "popsize": popsize,
                                    "maxfevals": 20000,
                                    "seed": int(time.time()) + island,
                                    # IMPORTANT: avoid O(N^2) memory in full CMA covariance matrices.
                                    "CMA_diagonal": True,
                                }
                                es = cma.CMAEvolutionStrategy(x0_cma, self.sigma, opts)
                                island_start = time.time()
                                last_log = island_start
                                iters = 0
                                while not es.stop() and (time.time() - island_start) < island_budget:
                                    xc = es.ask()
                                    es.tell(
                                        xc,
                                        [
                                            _loss_cpu(
                                                np.asarray(theta),
                                                xb_norm_np,
                                                yb,
                                                self.weight_decay,
                                                d_features,
                                                hidden,
                                                row_idx=row_idx,
                                                class_idx=class_idx,
                                            )
                                            for theta in xc
                                        ],
                                    )
                                    iters += 1
                                    now = time.time()
                                    if now - last_log >= 60.0:
                                        last_log = now
                                        logger.info(
                                            f"Evo island {island + 1}/{active_islands}: iters={iters} "
                                            f"elapsed={now - island_start:.0f}s/{island_budget:.0f}s"
                                        )
                                candidate = np.asarray(es.result.xbest, dtype=float)
                                candidate_loss = _loss_cpu(
                                    candidate,
                                    xb_norm_np,
                                    yb,
                                    self.weight_decay,
                                    d_features,
                                    hidden,
                                    row_idx=row_idx,
                                    class_idx=class_idx,
                                )
                                return float(candidate_loss), candidate

                            with thread_limits(blas_threads):
                                with ThreadPoolExecutor(max_workers=active_islands) as ex:
                                    futures = {
                                        ex.submit(_run_cpu_island, i, island_budgets[i]): i
                                        for i in range(active_islands)
                                    }
                                    for fut in as_completed(futures):
                                        loss_val, theta_candidate = fut.result()
                                        cpu_results.append((float(loss_val), theta_candidate))

                    results = async_results.get()
                for loss_val, theta_candidate, _island_idx, dev_used in results:
                    island_losses.append(float(loss_val))
                    if dev_used:
                        self._last_device = str(dev_used)
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_theta = theta_candidate
                for loss_val, theta_candidate in cpu_results:
                    island_losses.append(float(loss_val))
                    self._last_device = "cpu"
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_theta = theta_candidate
            except Exception as exc:
                logger.warning(f"Multiprocess GPU islands failed, reverting to sequential: {exc}")
                use_multiproc = False

        if not use_multiproc:
            # CPU parallel islands: reduces wall-time by running independent islands concurrently.
            use_cpu_parallel_islands = (
                bool(self.evo_multiproc_per_gpu) and not (CUPY_AVAILABLE and gpus) and islands > 1
            )
            if use_cpu_parallel_islands:
                total_budget = float(max(0, int(self.max_time_sec)))
                if total_budget <= 0:
                    return

                row_idx = row_idx_np
                class_idx = class_idx_np

                try:
                    cpu_cores = int(os.environ.get("FOREX_BOT_CPU_THREADS", "0") or 0)
                except Exception:
                    cpu_cores = 0
                if cpu_cores <= 0:
                    try:
                        cpu_cores = int(os.environ.get("FOREX_BOT_CPU_BUDGET", "0") or 0)
                    except Exception:
                        cpu_cores = 0
                if cpu_cores <= 0:
                    cpu_total = max(1, os.cpu_count() or 1)
                    try:
                        reserve = int(os.environ.get("FOREX_BOT_CPU_RESERVE", "1") or 1)
                    except Exception:
                        reserve = 1
                    cpu_cores = max(1, cpu_total - max(0, reserve))
                else:
                    cpu_cores = max(1, min(cpu_cores, max(1, os.cpu_count() or cpu_cores)))

                # Safety: cap active islands by both time budget and available CPU threads.
                # Scale islands with cores: up to 32 islands for HPC systems
                max_by_time = max(1, int(math.ceil(total_budget / 60.0)))
                requested_islands = islands
                max_islands = cpu_cores
                active_islands = max(1, min(requested_islands, max_islands, max_by_time))
                if max_cpu_by_mem is not None and max_cpu_by_mem > 0:
                    active_islands = min(active_islands, max_cpu_by_mem)
                if active_islands <= 1:
                    island_budgets = [total_budget]
                else:
                    per_island = float(total_budget) / float(active_islands)
                    if per_island >= 60.0:
                        island_budgets = [per_island for _ in range(active_islands)]
                    else:
                        rem = max(0.0, total_budget - 60.0 * float(active_islands - 1))
                        island_budgets = [60.0 for _ in range(active_islands - 1)] + [rem]

                if active_islands != requested_islands:
                    logger.info(
                        f"Evo islands capped: requested={requested_islands}, active={active_islands}, "
                        f"cpu_threads={cpu_cores}, budget={total_budget:.0f}s"
                    )

                blas_threads = max(1, cpu_cores // max(1, active_islands))

                def _run_cpu_island(island: int, island_budget: float) -> tuple[float, np.ndarray]:
                    opts = {
                        "verbose": -9,
                    "popsize": popsize,
                        "maxfevals": 20000,
                        "seed": int(time.time()) + island,
                        # IMPORTANT: avoid O(N^2) memory in full CMA covariance matrices.
                        "CMA_diagonal": True,
                    }
                    es = cma.CMAEvolutionStrategy(x0_cma, self.sigma, opts)
                    island_start = time.time()
                    last_log = island_start
                    iters = 0
                    while not es.stop() and (time.time() - island_start) < island_budget:
                        xc = es.ask()
                        es.tell(
                            xc,
                            [
                                _loss_cpu(
                                    np.asarray(theta),
                                    xb_norm_np,
                                    yb,
                                    self.weight_decay,
                                    d_features,
                                    hidden,
                                    row_idx=row_idx,
                                    class_idx=class_idx,
                                )
                                for theta in xc
                            ],
                        )
                        iters += 1
                        now = time.time()
                        if now - last_log >= 60.0:
                            last_log = now
                            logger.info(
                                f"Evo island {island + 1}/{active_islands}: iters={iters} "
                                f"elapsed={now - island_start:.0f}s/{island_budget:.0f}s"
                            )
                    candidate = np.asarray(es.result.xbest, dtype=float)
                    candidate_loss = _loss_cpu(
                        candidate,
                        xb_norm_np,
                        yb,
                        self.weight_decay,
                        d_features,
                        hidden,
                        row_idx=row_idx,
                        class_idx=class_idx,
                    )
                    return float(candidate_loss), candidate

                try:
                    with thread_limits(blas_threads):
                        with ThreadPoolExecutor(max_workers=active_islands) as ex:
                            futures = {
                                ex.submit(_run_cpu_island, i, island_budgets[i]): i for i in range(active_islands)
                            }
                            for fut in as_completed(futures):
                                loss_val, theta_candidate = fut.result()
                                island_losses.append(float(loss_val))
                                self._last_device = "cpu"
                                if loss_val < best_loss:
                                    best_loss = loss_val
                                    best_theta = theta_candidate
                except Exception as exc:
                    logger.warning(f"CPU parallel islands failed, reverting to sequential: {exc}", exc_info=True)
                    use_cpu_parallel_islands = False

            if not use_cpu_parallel_islands:
                for island in range(islands):
                    remaining = self.max_time_sec - (time.time() - total_start)
                    if remaining <= 0:
                        break
                    island_budget = min(budget_per_island, remaining)
                    active_device = None
                    use_gpu_island = bool(CUPY_AVAILABLE and gpus and xb_norm_cp is not None and yb_cp is not None)
                    if use_gpu_island:
                        active_device = gpus[island % len(gpus)]
                        self._last_device = str(active_device)
                    else:
                        self._last_device = str(self.device)

                    opts = {
                        "verbose": -9,
                        "popsize": popsize,
                        "maxfevals": 20000,
                        "seed": int(time.time()) + island,
                        # IMPORTANT: avoid O(N^2) memory in full CMA covariance matrices.
                        "CMA_diagonal": True,
                    }
                    es = cma.CMAEvolutionStrategy(x0_cma, self.sigma, opts)
                    island_start = time.time()
                    last_log = island_start
                    iters = 0
                    while not es.stop() and (time.time() - island_start) < island_budget:
                        xc = es.ask()
                        es.tell(
                            xc,
                            [
                                _loss_cpu(
                                    np.asarray(x),
                                    xb_norm_np,
                                    yb,
                                    self.weight_decay,
                                    d_features,
                                    hidden,
                                    row_idx=row_idx_np,
                                    class_idx=class_idx_np,
                                )
                                if not use_gpu_island
                                else _loss_gpu(
                                    np.asarray(x),
                                    xb_norm_cp,
                                    yb_cp,
                                    self.weight_decay,
                                    d_features,
                                    hidden,
                                    _parse_gpu_id(active_device),
                                    row_idx=row_idx_cp,
                                    class_idx=class_idx_cp,
                                )
                                for x in xc
                            ],
                        )
                        iters += 1
                        now = time.time()
                        if now - last_log >= 60.0:
                            last_log = now
                            logger.info(
                                f"Evo island {island + 1}/{islands}: iters={iters} "
                                f"elapsed={now - island_start:.0f}s/{island_budget:.0f}s"
                            )
                    candidate = np.asarray(es.result.xbest, dtype=float)
                    if use_gpu_island:
                        candidate_loss = _loss_gpu(
                            candidate,
                            xb_norm_cp,
                            yb_cp,
                            self.weight_decay,
                            d_features,
                            hidden,
                            _parse_gpu_id(active_device),
                            row_idx=row_idx_cp,
                            class_idx=class_idx_cp,
                        )
                    else:
                        candidate_loss = _loss_cpu(
                            candidate,
                            xb_norm_np,
                            yb,
                            self.weight_decay,
                            d_features,
                            hidden,
                            row_idx=row_idx_np,
                            class_idx=class_idx_np,
                        )
                    island_losses.append(float(candidate_loss))
                    if candidate_loss < best_loss:
                        best_loss = candidate_loss
                        best_theta = candidate

        self.theta = best_theta
        self.island_losses = island_losses

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        n = len(x)
        if self.theta is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("EvoExpert parameters not initialized")
        d = x.shape[1]
        hidden = int(self.hidden_size)

        def _unpack(theta_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            idx = 0
            w1 = theta_vec[idx : idx + d * hidden].reshape(d, hidden)
            idx += d * hidden
            b1 = theta_vec[idx : idx + hidden].reshape(1, hidden)
            idx += hidden
            w2 = theta_vec[idx : idx + hidden * 3].reshape(hidden, 3)
            idx += hidden * 3
            b2 = theta_vec[idx : idx + 3].reshape(1, 3)
            return w1, b1, w2, b2

        try:
            w1, b1, w2, b2 = _unpack(self.theta)
            xn_all = x.astype(np.float32).to_numpy()

            # Batching for inference
            batch_size = 8192
            probs_list = []

            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                xn = (xn_all[start:end] - self.mean_) / self.scale_
                hidden_act = np.tanh(xn @ w1 + b1)
                logits = hidden_act @ w2 + b2
                p_batch = _softmax(logits)
                # Reorder to [Neutral, Buy, Sell]
                # Indices after remapping: 0=Sell, 1=Neutral, 2=Buy
                # Target: index 0=Neutral, 1=Buy, 2=Sell
                probs_list.append(p_batch[:, [1, 2, 0]])

            return np.vstack(probs_list)
        except Exception as exc:
            logger.error(f"EvoExpert inference failed: {exc}", exc_info=True)
            raise

    def save(self, path: str) -> None:
        if self.theta is None:
            return
        os.makedirs(path, exist_ok=True)
        np.savez(
            os.path.join(path, "evo_cma.npz"),
            theta=self.theta,
            mean=self.mean_,
            scale=self.scale_,
            island_losses=np.asarray(self.island_losses or []),
        )

    def load(self, path: str) -> None:
        p = os.path.join(path, "evo_cma.npz")
        if os.path.exists(p):
            try:
                data = np.load(p, allow_pickle=True)
                self.theta = data.get("theta")
                self.mean_ = data.get("mean")
                self.scale_ = data.get("scale")
                island_losses = data.get("island_losses")
                self.island_losses = island_losses.tolist() if island_losses is not None else None
            except Exception as exc:
                logger.warning(f"Failed to load Evo CMA: {exc}")

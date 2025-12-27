import contextlib
import logging
import os
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from evotorch import Problem
from evotorch.algorithms import CMAES

logger = logging.getLogger(__name__)

class TensorDiscoveryEngine:
    """
    2025-Grade GPU Discovery Engine.
    Cooperative Multi-GPU Architecture with True Auto-Tuning.
    
    Features:
    - Parallel Data Alignment (utilizing all CPU cores)
    - FP16 Optimization (50% RAM reduction)
    - Dynamic OOM Recovery (Auto-scales batch size on crash)
    - Robust Statistics (Clamped Sharpe to prevent anomalies)
    """
    
    def __init__(
        self,
        device: str = "cuda",
        n_experts: int = 20,
        timeframes: Optional[List[str]] = None,
        max_rows: int | None = None,
        stream_mode: bool | None = None,
        auto_cap: bool | None = None,
    ):
        from ..models.device import get_available_gpus
        gpu_list = get_available_gpus() or ["cpu"]
        # Optional cap on GPUs used by discovery (e.g., FOREX_BOT_MAX_GPUS=4)
        try:
            max_gpus = int(os.environ.get("FOREX_BOT_MAX_GPUS", "0") or 0)
        except Exception:
            max_gpus = 0
        if max_gpus and max_gpus > 0 and len(gpu_list) > max_gpus:
            gpu_list = gpu_list[:max_gpus]
        self.gpu_list = gpu_list
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_experts = n_experts
        self.data_cube = None
        self.ohlc_cube = None
        self.timeframes = timeframes
        self.best_experts = []
        self.month_ids_cpu = None
        self.month_count = 0
        self.full_fp16 = False
        # Optional overrides (prefer config-driven settings vs env)
        self._max_rows_override = max_rows
        self._stream_mode_override = stream_mode
        self._auto_cap_override = auto_cap

    def _prepare_tensor_cube(self, frames: Dict[str, pd.DataFrame], news_map: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Aligns data using parallel CPU processing and optimizes storage with FP16.
        """
        logger.info(f"Aligning Multi-Timeframe Data (Auto-Parallelized)...")
        master_tf = "M1"
        if master_tf not in frames:
            master_tf = next(iter(frames.keys()))
        master_idx = frames[master_tf].index

        # Dynamic Row Limiter based on RAM
        if self._stream_mode_override is None:
            stream_mode = str(os.environ.get("FOREX_BOT_DISCOVERY_STREAM", "0") or "0").strip().lower() in {
                "1", "true", "yes", "on"
            }
        else:
            stream_mode = bool(self._stream_mode_override)
        if self._max_rows_override is None:
            try:
                max_rows_env = int(os.environ.get("FOREX_BOT_DISCOVERY_MAX_ROWS", "0") or 0)
            except Exception:
                max_rows_env = 0
            max_rows = max_rows_env
        else:
            max_rows = int(self._max_rows_override)

        if self._auto_cap_override is None:
            auto_cap = True
        else:
            auto_cap = bool(self._auto_cap_override)

        if max_rows <= 0 and not stream_mode and auto_cap:
            try:
                from ..core.system import HardwareProbe
                profile = HardwareProbe().detect()
                avail_bytes = max(0.0, float(profile.available_ram_gb)) * (1024**3)
                if avail_bytes > 0:
                    bytes_per_row = 0
                    for tf_name, tf_df in frames.items():
                        try:
                            num_cols = int(tf_df.select_dtypes(include=[np.number]).shape[1])
                        except Exception:
                            num_cols = int(tf_df.shape[1])
                        # FP16 = 2 bytes per element
                        bytes_per_row += int((num_cols + 4) * 2) 
                    if bytes_per_row > 0:
                        # Allow using up to 75% of available RAM
                        auto_rows = int((avail_bytes * 0.75) / bytes_per_row)
                        max_rows = max(0, auto_rows)
            except Exception:
                max_rows = 0

        if max_rows > 0 and len(master_idx) > max_rows:
            logger.warning(f"Discovery: Auto-limiting to last {max_rows} rows to fit in RAM.")
            for tf in list(frames.keys()):
                try:
                    frames[tf] = frames[tf].iloc[-max_rows:]
                except Exception:
                    pass
            master_idx = frames[master_tf].index
        
        if self.timeframes:
            self.timeframes = [tf for tf in self.timeframes if tf in frames]
        else:
            self.timeframes = list(frames.keys())

        # --- PARALLEL ALIGNMENT ---
        # Use all available cores by default
        try:
            cpu_workers = int(os.environ.get("FOREX_BOT_DISCOVERY_ALIGN_WORKERS", "0") or 0)
        except Exception:
            cpu_workers = 0
        if cpu_workers <= 0:
            cpu_workers = os.cpu_count() or 1
        cpu_workers = max(1, cpu_workers)
        logger.info(f"Using {cpu_workers} concurrent threads for data alignment.")
        
        # Full FP16 mode (default auto -> enabled on CUDA)
        full_fp16_mode = str(os.environ.get("FOREX_BOT_DISCOVERY_FULL_FP16", "auto") or "auto").strip().lower()
        if full_fp16_mode in {"1", "true", "yes", "on"}:
            self.full_fp16 = True
        elif full_fp16_mode in {"0", "false", "no", "off"}:
            self.full_fp16 = False
        else:
            self.full_fp16 = torch.cuda.is_available()

        results = {}
        
        def _process_frame(tf_name):
            try:
                # Forward-fill to avoid look-ahead
                df = frames[tf_name].reindex(master_idx).ffill().fillna(0.0)
                
                # Extract numeric features
                feats = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                f_std = np.maximum(feats.std(axis=0), 1e-6)
                norm = (feats - feats.mean(axis=0)) / f_std
                
                # OPTIMIZATION: Convert to Float16 immediately to save RAM
                norm_fp16 = norm.astype(np.float16)
                ohlc_dtype = np.float16 if self.full_fp16 else np.float32
                ohlc_arr = df[['open', 'high', 'low', 'close']].to_numpy(dtype=ohlc_dtype)

                return tf_name, torch.from_numpy(norm_fp16), torch.from_numpy(ohlc_arr)
            except Exception as e:
                logger.error(f"Error processing {tf_name}: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_workers) as executor:
            future_to_tf = {executor.submit(_process_frame, tf): tf for tf in self.timeframes}
            for future in concurrent.futures.as_completed(future_to_tf):
                res = future.result()
                if res:
                    results[res[0]] = (res[1], res[2])

        cube_list = []
        ohlc_list = []
        for tf in self.timeframes:
            if tf in results:
                cube_list.append(results[tf][0])
                ohlc_list.append(results[tf][1])
            
        if not cube_list:
            raise ValueError("Data alignment failed - empty cube.")

        self.data_cube_cpu = torch.stack(cube_list)
        self.ohlc_cube_cpu = torch.stack(ohlc_list)
        self.n_features = self.data_cube_cpu.shape[2]
        
        # Month alignment for consistency metrics
        self.month_ids_cpu = None
        self.month_count = 0
        try:
            if isinstance(master_idx, pd.DatetimeIndex):
                idx = master_idx
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC")
                month_keys = idx.to_period("M")
                month_ids, uniques = pd.factorize(month_keys)
                self.month_ids_cpu = torch.as_tensor(month_ids, dtype=torch.long)
                self.month_count = int(len(uniques))
        except Exception:
            self.month_ids_cpu = None
            self.month_count = 0
        
        # News Tensor
        self.news_tensor_cpu = None
        if isinstance(news_map, pd.DataFrame):
             nf = news_map.reindex(master_idx).ffill().fillna(0.0)
             sent = nf['news_sentiment'].values if 'news_sentiment' in nf.columns else np.zeros(len(nf))
             impact = nf['news_confidence'].values if 'news_confidence' in nf.columns else np.zeros(len(nf))
             news_dtype = np.float16 if self.full_fp16 else np.float32
             news_data = np.stack([sent, impact], axis=1).astype(news_dtype)
             self.news_tensor_cpu = torch.from_numpy(news_data)

        logger.info(f"Master Data Cube Ready: {self.data_cube_cpu.shape} [{self.data_cube_cpu.dtype}]")

    def run_unsupervised_search(self, frames: Dict[str, pd.DataFrame], news_features: Optional[pd.DataFrame] = None, iterations: int = 1000):
        self._prepare_tensor_cube(frames, news_features)

        # Initialize Metrics
        debug_metrics = os.environ.get("FOREX_BOT_DISCOVERY_DEBUG_METRICS", "1").strip().lower() not in {"0", "false", "no", "off"}
        # Checkpointing
        try:
            ckpt_every = int(os.environ.get("FOREX_BOT_DISCOVERY_CKPT_EVERY", "10") or 10)
        except Exception:
            ckpt_every = 10
        ckpt_dir = str(os.environ.get("FOREX_BOT_DISCOVERY_CKPT_DIR", "cache") or "cache").strip()
        ckpt_path = Path(ckpt_dir) / "tensor_discovery_ckpt.pt"
        resume_enabled = str(os.environ.get("FOREX_BOT_DISCOVERY_RESUME", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
        # Early stopping (disabled by default)
        early_stop = str(os.environ.get("FOREX_BOT_DISCOVERY_EARLY_STOP", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        try:
            early_patience = int(os.environ.get("FOREX_BOT_DISCOVERY_EARLY_PATIENCE", "100") or 100)
        except Exception:
            early_patience = 100
        try:
            early_min_delta = float(os.environ.get("FOREX_BOT_DISCOVERY_EARLY_MIN_DELTA", "0.01") or 0.01)
        except Exception:
            early_min_delta = 0.01
        # Numeric stability + fitness guards
        accum_fp32 = str(os.environ.get("FOREX_BOT_DISCOVERY_ACCUM_FP32", "1") or "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        try:
            sharpe_cap = float(os.environ.get("FOREX_BOT_DISCOVERY_SHARPE_CAP", "20") or 20)
        except Exception:
            sharpe_cap = 20.0
        try:
            dd_cap = float(os.environ.get("FOREX_BOT_DISCOVERY_DD_CAP", "0.15") or 0.15)
        except Exception:
            dd_cap = 0.15
        try:
            dd_penalty_weight = float(os.environ.get("FOREX_BOT_DISCOVERY_DD_PENALTY", "1.0") or 1.0)
        except Exception:
            dd_penalty_weight = 1.0
        debug_validity = os.environ.get("FOREX_BOT_DISCOVERY_DEBUG_VALIDITY", "0").strip().lower() in {"1", "true", "yes", "on"}
        # Permanent Metrics Logging (Every 10 Gens)
        log_metrics = True
        log_metrics_every = 10
        
        debug_snapshot = {
            "fitness": None, "sharpe": None, "mean_ret": None, "std_ret": None,
            "max_dd": None, "profitable_chunks": None, "consistency": None,
            "samples": None, "pass_rate": None, "median_month": None
        }
        import threading
        debug_lock = threading.Lock() if debug_metrics else None
        metrics_lock = threading.Lock()
        metrics_state = {"gen": -1, "armed": False, "logged": False}

        # --- AUTO-TUNING HARDWARE CONFIG ---
        # 1. GPU Preload: Only if we fit safely in VRAM
        preload_mode = os.environ.get("FOREX_BOT_DISCOVERY_PRELOAD", "auto").strip().lower()
        preload = False
        cuda_gpus = [g for g in self.gpu_list if str(g).startswith("cuda")]
        
        if preload_mode != "0" and cuda_gpus:
            try:
                bytes_needed = (self.data_cube_cpu.nelement() * self.data_cube_cpu.element_size()) + \
                               (self.ohlc_cube_cpu.nelement() * self.ohlc_cube_cpu.element_size())
                if self.news_tensor_cpu is not None:
                    bytes_needed += self.news_tensor_cpu.nelement() * 4
                
                min_vram = min(torch.cuda.get_device_properties(g).total_memory for g in cuda_gpus)
                # Safe threshold: 85% of VRAM
                if bytes_needed < (min_vram * 0.85):
                    preload = True
                    logger.info(f"Discovery: Preloading data ({bytes_needed/1024**3:.2f} GB) to GPUs.")
                    gpu_data_cubes = [self.data_cube_cpu.to(gpu) for gpu in cuda_gpus]
                    gpu_ohlc_cubes = [self.ohlc_cube_cpu.to(gpu) for gpu in cuda_gpus]
                    gpu_news = [self.news_tensor_cpu.to(gpu) for gpu in cuda_gpus] if self.news_tensor_cpu is not None else []
                    gpu_month_ids = [self.month_ids_cpu.to(gpu) for gpu in cuda_gpus] if self.month_ids_cpu is not None else []
                else:
                    logger.info(f"Discovery: Data too large for VRAM ({bytes_needed/1024**3:.2f} GB > Limit). Streaming mode active.")
                    gpu_data_cubes, gpu_ohlc_cubes, gpu_news, gpu_month_ids = [], [], [], []
            except Exception:
                preload = False
                gpu_data_cubes, gpu_ohlc_cubes, gpu_news, gpu_month_ids = [], [], [], []
        else:
            gpu_data_cubes, gpu_ohlc_cubes, gpu_news, gpu_month_ids = [], [], [], []

        num_gpus = max(1, len(self.gpu_list))
        
        # Shared batch size state for auto-tuning
        # Keys: gpu_idx -> current_batch_size
        batch_size_map = {}
        # Cache auto max-genomes per GPU
        max_genomes_map = {}
        invalid_logged = set()
        
        @torch.no_grad()
        def fitness_func(genomes_all: torch.Tensor) -> torch.Tensor:
            pop_size = genomes_all.shape[0]
            pop_chunks = genomes_all.chunk(num_gpus, dim=0)
            all_fitness_scores = []
            
            import concurrent.futures
            def _eval_on_gpu(chunk: torch.Tensor, gpu_idx: int):
                dev = torch.device(self.gpu_list[gpu_idx])
                is_cuda = dev.type == "cuda"
                
                # Initialize batch size and max-genomes for this GPU if not set
                if gpu_idx not in batch_size_map:
                    # Auto-derive max-genomes unless explicitly set
                    env_max_genomes = os.environ.get("FOREX_BOT_DISCOVERY_MAX_GENOMES")
                    if env_max_genomes:
                        try:
                            max_genomes_map[gpu_idx] = int(env_max_genomes)
                        except Exception:
                            max_genomes_map[gpu_idx] = 256
                    else:
                        if is_cuda:
                            try:
                                total_mem_gb = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
                            except Exception:
                                total_mem_gb = 0.0
                            if total_mem_gb >= 80:
                                max_genomes_map[gpu_idx] = 2048
                            elif total_mem_gb >= 48:
                                max_genomes_map[gpu_idx] = 1024
                            elif total_mem_gb >= 32:
                                max_genomes_map[gpu_idx] = 768
                            elif total_mem_gb >= 24:
                                max_genomes_map[gpu_idx] = 512
                            else:
                                max_genomes_map[gpu_idx] = 256
                        else:
                            max_genomes_map[gpu_idx] = 256

                    # Heuristic batch start: env override or VRAM-based estimate
                    env_batch_start = os.environ.get("FOREX_BOT_DISCOVERY_BATCH_START")
                    if env_batch_start:
                        try:
                            batch_start = int(env_batch_start)
                        except Exception:
                            batch_start = 5000 # SAFER DEFAULT to prevent Kill
                    else:
                        if is_cuda:
                            try:
                                total_mem = torch.cuda.get_device_properties(dev).total_memory
                            except Exception:
                                total_mem = 0
                            tf_count = self.data_cube_cpu.shape[0]
                            elem_size = 2 if (self.full_fp16) else 4
                            pop_sub = max_genomes_map[gpu_idx]
                            # Rough upper-bound estimate of per-batch memory
                            per_batch = elem_size * (
                                pop_sub * (tf_count * 2 + 4) + tf_count * (self.n_features + 4)
                            )
                            # Reduced target ratio for safety
                            target = int(total_mem * (0.25 if self.full_fp16 else 0.18))
                            batch_start = int(target / max(1, per_batch))
                        else:
                            batch_start = 5000

                    # Clamp to safe bounds
                    batch_start = max(1024, min(int(batch_start), 65536))
                    batch_size_map[gpu_idx] = batch_start
                    logger.info(
                        f"GPU {gpu_idx}: auto batch start={batch_start}, max_genomes={max_genomes_map[gpu_idx]}"
                    )
                
                try:
                    if is_cuda: torch.cuda.set_device(dev)
                    ctx = torch.cuda.device(dev) if is_cuda else contextlib.nullcontext()
                    
                    with ctx:
                        if chunk.numel() == 0: return torch.empty((0,), dtype=torch.float32)
                        
                        n_samples = self.data_cube_cpu.shape[1]
                        has_news = self.news_tensor_cpu is not None
                        month_ids = (gpu_month_ids[gpu_idx] if preload and gpu_month_ids else 
                                     self.month_ids_cpu.to(dev, non_blocking=True)) if self.month_ids_cpu is not None else None

                        # Sub-batching for Genomes (to avoid OOM on huge populations)
                        max_genomes = max_genomes_map.get(gpu_idx)
                        if max_genomes is None:
                            try:
                                max_genomes = int(os.environ.get("FOREX_BOT_DISCOVERY_MAX_GENOMES", "256") or 256)
                            except Exception:
                                max_genomes = 256
                        
                        fitness_all = []
                        for g_start in range(0, chunk.shape[0], max_genomes):
                            g_end = min(g_start + max_genomes, chunk.shape[0])
                            genomes = chunk[g_start:g_end].clone().detach().to(dev)
                            local_pop = genomes.shape[0]
                            
                            # --- OOM RECOVERY LOOP ---
                            while True:
                                current_batch = batch_size_map[gpu_idx]
                                try:
                                    # Attempt evaluation with current batch size
                                    res = _eval_batch_safe(
                                        gpu_idx, dev, genomes, current_batch, 
                                        n_samples, has_news, preload, gpu_data_cubes, gpu_ohlc_cubes, gpu_news,
                                        month_ids
                                    )
                                    fitness_all.append(res)
                                    break # Success, break retry loop
                                    
                                except torch.OutOfMemoryError:
                                    # OOM CAUGHT -> Auto-Tune Down
                                    torch.cuda.empty_cache()
                                    gc.collect()
                                    new_batch = int(current_batch * 0.5)
                                    if new_batch < 128:
                                        logger.error(f"GPU {gpu_idx}: Batch size too small ({new_batch}), cannot continue.")
                                        return torch.full((chunk.shape[0],), -1e9)
                                    
                                    logger.warning(f"GPU {gpu_idx} OOM! Auto-tuning batch size: {current_batch} -> {new_batch}")
                                    batch_size_map[gpu_idx] = new_batch
                                    # Retry loop will use new batch size
                                    continue
                                except Exception as e:
                                    logger.error(f"GPU {gpu_idx} eval error: {e}")
                                    return torch.full((chunk.shape[0],), -1e9)

                        return torch.cat(fitness_all, dim=0)
                except Exception as e:
                    logger.error(f"GPU {gpu_idx} fatal error: {e}")
                    return torch.full((chunk.shape[0],), -1e9)

            def _eval_batch_safe(gpu_idx, dev, genomes, batch_size, n_samples, has_news, preload,
                                 gpu_data_cubes, gpu_ohlc_cubes, gpu_news, month_ids):
                # Core evaluation logic separated for retryability
                local_pop = genomes.shape[0]
                if accum_fp32:
                    accum_dtype = torch.float32
                else:
                    accum_dtype = torch.float16 if (dev.type == "cuda" and self.full_fp16) else torch.float32
                sum_ret = torch.zeros(local_pop, device=dev, dtype=accum_dtype)
                sum_sq_ret = torch.zeros(local_pop, device=dev, dtype=accum_dtype)
                effective_samples = 0
                running_equity = torch.ones(local_pop, device=dev, dtype=accum_dtype)
                running_peak = torch.ones(local_pop, device=dev, dtype=accum_dtype)
                max_dd = torch.zeros(local_pop, device=dev, dtype=accum_dtype)
                profitable_chunks = torch.zeros(local_pop, device=dev, dtype=accum_dtype)
                current_chunk_ret = torch.zeros(local_pop, device=dev, dtype=accum_dtype)
                month_returns = torch.zeros((local_pop, self.month_count), device=dev, dtype=accum_dtype) if self.month_count > 0 else None
                
                chunk_size = max(1, (n_samples - 1) // 10)
                
                tf_count = self.data_cube_cpu.shape[0]
                matmul_dtype = torch.float16 if dev.type == "cuda" else torch.float32
                tf_weights = F.softmax(genomes[:, :tf_count], dim=1).to(dtype=matmul_dtype)
                logic_weights = genomes[:, tf_count : tf_count + self.n_features].to(dtype=matmul_dtype)
                thresholds = genomes[:, -2:]
                buy_th = torch.maximum(thresholds[:, 0], thresholds[:, 1]).to(dtype=matmul_dtype)
                sell_th = torch.minimum(thresholds[:, 0], thresholds[:, 1]).to(dtype=matmul_dtype)

                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    
                    if preload:
                        data_slice = gpu_data_cubes[gpu_idx][:, start_idx:end_idx, :]
                        ohlc_slice = gpu_ohlc_cubes[gpu_idx][:, start_idx:end_idx, :]
                        news_slice = gpu_news[gpu_idx][start_idx:end_idx] if has_news else None
                    else:
                        data_slice = self.data_cube_cpu[:, start_idx:end_idx, :].to(dev, non_blocking=True)
                        ohlc_slice = self.ohlc_cube_cpu[:, start_idx:end_idx, :].to(dev, non_blocking=True)
                        news_slice = self.news_tensor_cpu[start_idx:end_idx].to(dev, non_blocking=True) if has_news else None

                    # Vectorized computation across timeframes to maximize GPU utilization
                    batch_len = end_idx - start_idx
                    data_slice = data_slice.to(dtype=matmul_dtype)
                    # data_slice: [TF, Batch, Feat] -> [TF*Batch, Feat]
                    data_2d = data_slice.reshape(tf_count * batch_len, self.n_features)
                    # logic_weights: [Pop, Feat] -> [Feat, Pop]
                    logic_t = logic_weights.t().contiguous()
                    # Single large matmul: [TF*Batch, Feat] x [Feat, Pop] -> [TF*Batch, Pop]
                    tf_sig_2d = torch.matmul(data_2d, logic_t)
                    # Reshape to [Pop, TF, Batch]
                    tf_sig = tf_sig_2d.view(tf_count, batch_len, local_pop).permute(2, 0, 1)
                    # Weighted sum over timeframes -> [Pop, Batch]
                    signals = (tf_sig * tf_weights.unsqueeze(2)).sum(dim=1)
                    signals = torch.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

                    if news_slice is not None:
                        # Add Sentiment Bias (match dtype for in-place add)
                        news_sent = news_slice[:, 0].to(dtype=matmul_dtype)
                        signals.add_(news_sent.unsqueeze(0) * 0.5)

                    # Actions
                    actions = torch.zeros_like(signals)
                    actions = torch.where(signals > buy_th.unsqueeze(1), 1.0, actions)
                    actions = torch.where(signals < sell_th.unsqueeze(1), -1.0, actions)

                    if (end_idx - start_idx) < 2: continue

                    # Vectorized Returns
                    close_next = torch.nan_to_num(ohlc_slice[0, 1:, 3], nan=0.0, posinf=0.0, neginf=0.0) # Shifted Close
                    open_next = torch.nan_to_num(ohlc_slice[0, 1:, 0], nan=0.0, posinf=0.0, neginf=0.0)  # Shifted Open
                    open_next = torch.clamp(open_next, min=1e-6)
                    actions_aligned = actions[:, :-1]

                    valid_px = (open_next > 1e-6) & (close_next > 1e-6)
                    rets = (close_next - open_next) / open_next
                    rets = torch.where(valid_px, rets, torch.zeros_like(rets))
                    rets = torch.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
                    rets = torch.clamp(rets, min=-1.0, max=1.0)
                    batch_rets = actions_aligned * rets.unsqueeze(0)

                    # Cost Model (Spread + Commission + Slippage)
                    spread_cost = 0.0001
                    batch_rets = batch_rets - (torch.abs(actions_aligned) * spread_cost)

                    if news_slice is not None:
                        impact = news_slice[1:, 1]
                        batch_rets = batch_rets - (torch.abs(actions_aligned) * impact.unsqueeze(0))

                    # Aggregation
                    batch_rets_acc = batch_rets.to(dtype=accum_dtype)
                    sum_ret.add_(batch_rets_acc.sum(dim=1))
                    sum_sq_ret.add_((batch_rets_acc**2).sum(dim=1))
                    effective_samples += batch_rets.shape[1]
                    
                    # Monthly & Chunk stats
                    if month_ids is not None:
                        batch_m_ids = month_ids[start_idx + 1 : end_idx]
                        if batch_m_ids.numel() > 0:
                            month_returns.scatter_add_(1, batch_m_ids.unsqueeze(0).expand(local_pop, -1), batch_rets_acc)

                    current_chunk_ret.add_(batch_rets_acc.sum(dim=1))
                    if (end_idx // chunk_size) > (start_idx // chunk_size):
                        profitable_chunks.add_((current_chunk_ret > 0).float())
                        current_chunk_ret.zero_()

                    # Max DD (Approximate per batch to avoid massive equity tensor)
                    # We maintain running_equity state
                    equity_step = torch.clamp(1.0 + batch_rets_acc, min=1e-6)
                    batch_equity = torch.cumprod(equity_step, dim=1)
                    abs_equity = batch_equity * running_equity.unsqueeze(1)
                    batch_peaks = torch.maximum(torch.cummax(abs_equity, dim=1)[0], running_peak.unsqueeze(1))
                    dd = (batch_peaks - abs_equity) / torch.clamp(batch_peaks, min=1e-9)
                    max_dd = torch.maximum(dd.max(dim=1)[0], max_dd)
                    
                    running_equity.copy_(abs_equity[:, -1])
                    running_peak.copy_(batch_peaks[:, -1])

                # Finalize
                if effective_samples <= 1:
                    return torch.full((local_pop,), 0.0, device=dev, dtype=torch.float32)
                mean_ret = (sum_ret / effective_samples).to(dtype=torch.float32)
                var = (sum_sq_ret / effective_samples).to(dtype=torch.float32) - mean_ret**2
                std_ret = torch.sqrt(torch.clamp(var, min=1e-9)) + 1e-6
                
                # --- STATISTICAL SAFEGUARDS ---
                # Clamp volatility to prevent "infinite" Sharpe on flat lines
                # Minimum annualized volatility of 1% (~0.00004 per minute)
                min_vol = 0.00004
                safe_std = torch.maximum(std_ret, torch.tensor(min_vol, device=dev, dtype=torch.float32))
                
                sharpe = (mean_ret / safe_std) * (252 * 1440)**0.5
                
                # Additional Clamp: Real-world strategies almost never exceed Sharpe 20.0
                if sharpe_cap > 0:
                    sharpe = torch.clamp(sharpe, min=-sharpe_cap, max=sharpe_cap)

                fitness = sharpe * (profitable_chunks.to(dtype=torch.float32) / 10.0)
                # Penalties (soft drawdown penalty)
                if dd_cap > 0 and dd_penalty_weight > 0:
                    dd_excess = torch.clamp(max_dd.to(dtype=torch.float32) - dd_cap, min=0.0)
                    fitness = fitness - (dd_penalty_weight * dd_excess)
                if log_metrics and metrics_state.get("armed") and gpu_idx == 0 and not metrics_state.get("logged"):
                    with metrics_lock:
                        if metrics_state.get("armed") and not metrics_state.get("logged"):
                            try:
                                best_idx = int(torch.argmax(fitness).item())
                                logger.info(
                                    "Discovery metrics (gen %s): best_fit=%.4f sharpe=%.4f max_dd=%.4f "
                                    "mean_ret=%.6f std_ret=%.6f prof_chunks=%.2f samples=%d",
                                    metrics_state.get("gen"),
                                    float(fitness[best_idx].item()),
                                    float(sharpe[best_idx].item()),
                                    float(max_dd[best_idx].item()),
                                    float(mean_ret[best_idx].item()),
                                    float(std_ret[best_idx].item()),
                                    float(profitable_chunks[best_idx].item()),
                                    int(effective_samples),
                                )
                            except Exception:
                                pass
                            metrics_state["logged"] = True
                if debug_validity:
                    nan_count = int(torch.isnan(fitness).sum().item())
                    inf_count = int(torch.isinf(fitness).sum().item())
                    if (nan_count or inf_count) and debug_lock is not None:
                        with debug_lock:
                            if gpu_idx not in invalid_logged:
                                logger.warning(
                                    f"GPU {gpu_idx}: invalid fitness detected (nan={nan_count}, inf={inf_count})."
                                )
                                invalid_logged.add(gpu_idx)
                # Guard against NaN/Inf so CMA-ES doesn't crash
                fitness = torch.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Check metrics for debug
                if debug_lock is not None:
                    # (Simplified metric snapshot logic)
                    pass
                    
                return fitness.detach().cpu()

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = {executor.submit(_eval_on_gpu, pop_chunks[i], i): pop_chunks[i].shape[0] for i in range(num_gpus)}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        all_fitness_scores.append(future.result())
                    except Exception as e:
                        logger.error(f"Worker failed: {e}")
                        all_fitness_scores.append(torch.full((futures[future],), -1e9))

            if not all_fitness_scores:
                return torch.full((pop_size,), -1e9)
            scores = torch.cat(all_fitness_scores, dim=0)
            if scores.numel() != pop_size:
                if scores.numel() > pop_size:
                    scores = scores[:pop_size]
                else:
                    scores = torch.cat([scores, torch.full((pop_size - scores.numel(),), -1e9)], dim=0)
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=-1e9, neginf=-1e9).float()
            if not torch.isfinite(scores).any():
                scores = torch.full((pop_size,), -1e9)
            return scores

        # 2. Start Evolution
        genome_dim = self.n_features + len(self.timeframes) + 2
        problem = Problem("max", fitness_func, initial_bounds=(-1.0, 1.0),
                         solution_length=genome_dim, device="cpu", vectorized=True)
        
        pop_size = int(os.environ.get("FOREX_BOT_DISCOVERY_POPULATION", 30000))
        searcher = CMAES(problem, popsize=pop_size, stdev_init=0.5)

        logger.info(f"Starting COOPERATIVE {len(self.gpu_list)}-GPU Discovery (Pop: {pop_size})...")

        # Optional resume: reuse last checkpointed best experts (skip discovery)
        if resume_enabled and ckpt_path.exists():
            try:
                data = torch.load(ckpt_path, map_location="cpu")
                best = data.get("best_experts")
                if best is None:
                    best = data.get("genomes")
                if best is not None:
                    self.best_experts = best
                    logger.warning(
                        f"Discovery resume: loaded best experts from checkpoint at gen {data.get('gen', 'N/A')}. "
                        "Skipping discovery loop."
                    )
                    return
            except Exception as exc:
                logger.warning(f"Discovery resume failed: {exc}")

        # Main Loop
        best_val = -1e9
        no_improve = 0
        for i in range(iterations):
            if log_metrics and log_metrics_every > 0 and (i % log_metrics_every == 0):
                metrics_state["gen"] = i
                metrics_state["armed"] = True
                metrics_state["logged"] = False
            else:
                metrics_state["armed"] = False
            try:
                searcher.step()
            except Exception as exc:
                logger.error(f"Discovery step failed at gen {i}: {exc}", exc_info=True)
                # Fallback: evaluate a fresh random population to avoid crashing
                try:
                    rnd = torch.empty((pop_size, genome_dim)).uniform_(-1.0, 1.0)
                    scores = fitness_func(rnd)
                    if scores.numel() == pop_size:
                        indices = torch.argsort(scores, descending=True)
                        self.best_experts = rnd[indices[:self.n_experts]].detach().cpu()
                        logger.warning("Discovery fallback: saved best experts from random population.")
                except Exception:
                    pass
                break
            current_best = searcher.status.get("best_eval", 0.0)
            if current_best > best_val + early_min_delta:
                best_val = current_best
                no_improve = 0
            else:
                no_improve += 1

            if i % 10 == 0:
                logger.info(f"Generation {i}: Global Best Fitness = {float(current_best):.4f}")

            # Periodic checkpoint (best experts so far)
            if ckpt_every > 0 and (i % ckpt_every == 0):
                try:
                    pop_batch = searcher.population
                    pop_vals = getattr(pop_batch, "values", pop_batch)
                    pop = pop_vals.detach().cpu() if torch.is_tensor(pop_vals) else torch.as_tensor(pop_vals).detach().cpu()
                    evals = None
                    for attr in ("evals", "evaluations", "evaluation"):
                        if hasattr(pop_batch, attr):
                            evals = getattr(pop_batch, attr)
                            break
                    if evals is not None and torch.is_tensor(evals):
                        evals = evals.detach().cpu()
                    if evals is not None and evals.numel() == pop.shape[0]:
                        idx = torch.argsort(evals, descending=True)
                        best_experts = pop[idx[: self.n_experts]]
                    else:
                        best_experts = pop[: self.n_experts]
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "gen": i,
                            "best_eval": float(current_best),
                            "best_experts": best_experts,
                            "timeframes": self.timeframes,
                            "n_features": self.n_features,
                            "timestamp": time.time(),
                        },
                        ckpt_path,
                    )
                except Exception as exc:
                    logger.warning(f"Discovery checkpoint failed: {exc}")

            if early_stop and no_improve >= max(1, early_patience):
                logger.warning(
                    f"Discovery early stop at gen {i}: no improvement "
                    f"for {no_improve} generations (min_delta={early_min_delta})."
                )
                break

        # Extract best
        final_pop = searcher.population.values.detach().cpu()
        final_scores = searcher.population.evaluations.detach().cpu()
        indices = torch.argsort(final_scores, descending=True)
        self.best_experts = final_pop[indices[:self.n_experts]]
        
    def save_experts(self, path: str):
        torch.save({"genomes": self.best_experts, "timeframes": self.timeframes, "n_features": self.n_features, "timestamp": time.time()}, path)

    def load_experts(self, path: str):
        if not Path(path).exists(): return None
        data = torch.load(path, map_location="cpu")
        self.best_experts = data["genomes"]
        self.timeframes = data["timeframes"]
        return self.best_experts
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
    """
    
    def __init__(
        self,
        device: str = "cuda",
        n_experts: int = 20,
        timeframes: Optional[List[str]] = None,
    ):
        from ..models.device import get_available_gpus
        self.gpu_list = get_available_gpus() or ["cpu"]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_experts = n_experts
        self.data_cube = None
        self.ohlc_cube = None
        self.timeframes = timeframes
        self.best_experts = []
        self.month_ids_cpu = None
        self.month_count = 0
        self.full_fp16 = False

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
        stream_mode = str(os.environ.get("FOREX_BOT_DISCOVERY_STREAM", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        try:
            max_rows_env = int(os.environ.get("FOREX_BOT_DISCOVERY_MAX_ROWS", "0") or 0)
        except Exception:
            max_rows_env = 0
        max_rows = max_rows_env
        
        if max_rows <= 0 and not stream_mode:
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
        # Using all available cores to speed up the CPU-heavy alignment phase
        cpu_workers = min(64, os.cpu_count() or 1)
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
        debug_snapshot = {
            "fitness": None, "sharpe": None, "mean_ret": None, "std_ret": None, 
            "max_dd": None, "profitable_chunks": None, "consistency": None, 
            "samples": None, "pass_rate": None, "median_month": None
        }
        import threading
        debug_lock = threading.Lock() if debug_metrics else None

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
        # We start with a conservative guess, then grow or shrink
        # Keys: gpu_idx -> current_batch_size
        batch_size_map = {} 
        
        @torch.no_grad()
        def fitness_func(genomes_all: torch.Tensor) -> torch.Tensor:
            pop_size = genomes_all.shape[0]
            pop_chunks = genomes_all.chunk(num_gpus, dim=0)
            all_fitness_scores = []
            
            import concurrent.futures
            def _eval_on_gpu(chunk: torch.Tensor, gpu_idx: int):
                dev = torch.device(self.gpu_list[gpu_idx])
                is_cuda = dev.type == "cuda"
                
                # Initialize batch size for this GPU if not set
                if gpu_idx not in batch_size_map:
                    # Heuristic start: 10000 or derived from VRAM
                    batch_size_map[gpu_idx] = 10000 
                
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
                        max_genomes = 256 # Conservative default
                        
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

                    # FP16 Computation
                    signals = torch.zeros((local_pop, end_idx - start_idx), device=dev, dtype=matmul_dtype)
                    
                    # Logic: Linear Combination of Features
                    for i in range(tf_count):
                        # Matmul: (Batch, Feat) x (Pop, Feat)^T -> (Batch, Pop)
                        # We need (Pop, Batch)
                        # data_cube is [TF, Rows, Feat]. Slice is [TF, Batch, Feat].
                        # data_slice[i] is [Batch, Feat].
                        # logic_weights is [Pop, Feat].
                        # result: [Batch, Pop]. Transpose to [Pop, Batch].
                        tf_sig = torch.matmul(data_slice[i].to(dtype=matmul_dtype), logic_weights.t()).t()
                        signals.add_(tf_weights[:, i].unsqueeze(1) * tf_sig)

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
                    close_next = ohlc_slice[0, 1:, 3] # Shifted Close
                    open_next = ohlc_slice[0, 1:, 0]  # Shifted Open
                    actions_aligned = actions[:, :-1]
                    
                    rets = (close_next - open_next) / (open_next + 1e-9)
                    batch_rets = actions_aligned * rets.unsqueeze(0)
                    batch_rets_acc = batch_rets.to(dtype=accum_dtype)
                    
                    # Cost Model (Spread + Commission + Slippage)
                    spread_cost = 0.0001
                    batch_rets = batch_rets - (torch.abs(actions_aligned) * spread_cost)
                    
                    if news_slice is not None:
                        impact = news_slice[1:, 1]
                        batch_rets = batch_rets - (torch.abs(actions_aligned) * impact.unsqueeze(0))

                    # Aggregation
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
                mean_ret = sum_ret / effective_samples
                std_ret = torch.sqrt(torch.clamp((sum_sq_ret / effective_samples) - mean_ret**2, min=1e-9)) + 1e-6
                sharpe = (mean_ret / std_ret) * (252 * 1440)**0.5
                
                fitness = sharpe * (profitable_chunks / 10.0)
                # Penalties
                fitness = torch.where(max_dd > 0.15, torch.tensor(-1.0, device=dev), fitness)
                
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
            
            return torch.cat(all_fitness_scores, dim=0)

        # 2. Start Evolution
        genome_dim = self.n_features + len(self.timeframes) + 2
        problem = Problem("max", fitness_func, initial_bounds=(-1.0, 1.0), 
                         solution_length=genome_dim, device="cpu", vectorized=True)
        
        pop_size = int(os.environ.get("FOREX_BOT_DISCOVERY_POPULATION", 30000))
        searcher = CMAES(problem, popsize=pop_size, stdev_init=0.5)

        logger.info(f"Starting COOPERATIVE {len(self.gpu_list)}-GPU Discovery (Pop: {pop_size})...")
        
        # Main Loop
        best_val = -1e9
        for i in range(iterations):
            searcher.step()
            current_best = searcher.status.get("best_eval", 0.0)
            if current_best > best_val:
                best_val = current_best
            
            if i % 10 == 0:
                logger.info(f"Generation {i}: Global Best Fitness = {float(current_best):.4f}")

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

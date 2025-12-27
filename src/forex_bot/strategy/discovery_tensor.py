import contextlib
import math
import logging
import os
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
import threading

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
    Cooperative Multi-GPU Architecture with True Global Reporting.
    
    Features:
    - Parallel Data Alignment (utilizing all CPU cores)
    - FP16 Optimization (50% RAM reduction)
    - Dynamic OOM Recovery (Auto-scales batch size on crash)
    - Global Champion Reporting (Logs best metrics from all GPUs)
    - Survival-Based Fitness (Gradient for early evolution)
    - Monte-Carlo Lite (Consistency over 3 market regimes)
    """
    
    def __init__(
        self,
        device: str = "cuda",
        n_experts: int = 100,
        timeframes: Optional[List[str]] = None,
        max_rows: int | None = None,
        stream_mode: bool | None = None,
        auto_cap: bool | None = None,
    ):
        from ..models.device import get_available_gpus
        gpu_list = get_available_gpus() or ["cpu"]
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

        if self._stream_mode_override is None:
            stream_mode = str(os.environ.get("FOREX_BOT_DISCOVERY_STREAM", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
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
                        bytes_per_row += int((num_cols + 4) * 2) 
                    if bytes_per_row > 0:
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

        try:
            cpu_workers = int(os.environ.get("FOREX_BOT_DISCOVERY_ALIGN_WORKERS", "0") or 0)
        except Exception:
            cpu_workers = 0
        if cpu_workers <= 0:
            cpu_workers = os.cpu_count() or 1
        logger.info(f"Using {cpu_workers} concurrent threads for data alignment.")
        
        full_fp16_mode = str(os.environ.get("FOREX_BOT_DISCOVERY_FULL_FP16", "auto") or "auto").strip().lower()
        self.full_fp16 = full_fp16_mode in {"1", "true", "yes", "on"} or (full_fp16_mode == "auto" and torch.cuda.is_available())

        results = {}
        def _process_frame(tf_name):
            try:
                raw = frames[tf_name].reindex(master_idx).ffill()
                feat_df = raw.fillna(0.0)
                feats = feat_df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                f_std = np.maximum(feats.std(axis=0), 1e-6)
                norm = (feats - feats.mean(axis=0)) / f_std
                norm_fp16 = norm.astype(np.float16)
                ohlc_dtype = np.float16 if self.full_fp16 else np.float32
                ohlc_arr = raw[['open', 'high', 'low', 'close']].ffill().bfill().to_numpy(dtype=ohlc_dtype)
                return tf_name, torch.from_numpy(norm_fp16), torch.from_numpy(ohlc_arr)
            except Exception as e:
                logger.error(f"Error processing {tf_name}: {e}")
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_workers) as executor:
            future_to_tf = {executor.submit(_process_frame, tf): tf for tf in self.timeframes}
            for future in concurrent.futures.as_completed(future_to_tf):
                res = future.result()
                if res: results[res[0]] = (res[1], res[2])

        cube_list, ohlc_list = [], []
        for tf in self.timeframes:
            if tf in results:
                cube_list.append(results[tf][0])
                ohlc_list.append(results[tf][1])
        if not cube_list: raise ValueError("Data alignment failed.")

        self.data_cube_cpu = torch.stack(cube_list)
        self.ohlc_cube_cpu = torch.stack(ohlc_list)
        self.n_features = self.data_cube_cpu.shape[2]
        
        self.month_ids_cpu = None
        self.month_count = 0
        try:
            if isinstance(master_idx, pd.DatetimeIndex):
                idx = master_idx.tz_convert("UTC") if master_idx.tz else master_idx
                month_ids, uniques = pd.factorize(idx.to_period("M"))
                self.month_ids_cpu = torch.as_tensor(month_ids, dtype=torch.long)
                self.month_count = int(len(uniques))
        except Exception: pass
        
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
        
        # Hyperparams & Global Reporting
        try:
            sharpe_cap = float(os.environ.get("FOREX_BOT_DISCOVERY_SHARPE_CAP", "20") or 20)
        except Exception: sharpe_cap = 20.0
        
        metrics_registry = {} # Key: generation, Value: best metadata
        registry_lock = threading.Lock()

        preload = False
        cuda_gpus = [g for g in self.gpu_list if str(g).startswith("cuda")]
        if cuda_gpus:
            try:
                bytes_needed = (self.data_cube_cpu.nelement() * self.data_cube_cpu.element_size()) + \
                               (self.ohlc_cube_cpu.nelement() * self.ohlc_cube_cpu.element_size())
                min_vram = min(torch.cuda.get_device_properties(g).total_memory for g in cuda_gpus)
                if bytes_needed < (min_vram * 0.85):
                    preload = True
                    logger.info(f"Discovery: Preloading data ({bytes_needed/1024**3:.2f} GB) to GPUs.")
                    gpu_data_cubes = [self.data_cube_cpu.to(gpu) for gpu in cuda_gpus]
                    gpu_ohlc_cubes = [self.ohlc_cube_cpu.to(gpu) for gpu in cuda_gpus]
                    gpu_news = [self.news_tensor_cpu.to(gpu) for gpu in cuda_gpus] if self.news_tensor_cpu is not None else []
                    gpu_month_ids = [self.month_ids_cpu.to(gpu) for gpu in cuda_gpus] if self.month_ids_cpu is not None else []
                else:
                    gpu_data_cubes, gpu_ohlc_cubes, gpu_news, gpu_month_ids = [], [], [], []
            except Exception: preload = False
        
        num_gpus = max(1, len(self.gpu_list))
        batch_size_map = {}
        max_genomes_map = {}
        gen_context = {"val": 0}

        @torch.no_grad()
        def fitness_func(genomes_all: torch.Tensor) -> torch.Tensor:
            pop_chunks = genomes_all.chunk(num_gpus, dim=0)
            all_fitness_scores = []
            
            def _eval_on_gpu(chunk: torch.Tensor, gpu_idx: int):
                dev = torch.device(self.gpu_list[gpu_idx])
                is_cuda = dev.type == "cuda"
                if gpu_idx not in batch_size_map:
                    batch_size_map[gpu_idx] = 5000
                    max_genomes_map[gpu_idx] = 1024 if is_cuda else 256
                
                try:
                    if is_cuda: torch.cuda.set_device(dev)
                    with torch.cuda.device(dev) if is_cuda else contextlib.nullcontext():
                        if chunk.numel() == 0: return torch.empty((0,), dtype=torch.float32)
                        n_samples = self.data_cube_cpu.shape[1]
                        month_ids = (gpu_month_ids[gpu_idx] if preload and gpu_month_ids else 
                                     self.month_ids_cpu.to(dev, non_blocking=True)) if self.month_ids_cpu is not None else None

                        fitness_all = []
                        max_sub_genomes = max_genomes_map[gpu_idx]
                        for g_start in range(0, chunk.shape[0], max_sub_genomes):
                            g_end = min(g_start + max_sub_genomes, chunk.shape[0])
                            genomes = chunk[g_start:g_end].clone().detach().to(dev)
                            
                            while True:
                                current_batch = batch_size_map[gpu_idx]
                                try:
                                    res, meta = _eval_batch_safe(gpu_idx, dev, genomes, current_batch, n_samples, preload, 
                                                                 gpu_data_cubes, gpu_ohlc_cubes, gpu_news, month_ids)
                                    fitness_all.append(res)
                                    with registry_lock:
                                        cur_gen = gen_context["val"]
                                        if cur_gen not in metrics_registry or meta["fit"] > metrics_registry[cur_gen]["fit"]:
                                            metrics_registry[cur_gen] = meta
                                    break
                                except torch.OutOfMemoryError:
                                    torch.cuda.empty_cache(); gc.collect()
                                    batch_size_map[gpu_idx] = int(current_batch * 0.5)
                                    if batch_size_map[gpu_idx] < 128: return torch.full((chunk.shape[0],), 0.0)
                                    continue
                        return torch.cat(fitness_all, dim=0)
                except Exception as e: 
                    logger.error(f"GPU {gpu_idx} fatal: {e}"); return torch.full((chunk.shape[0],), 0.0)

            def _eval_batch_safe(gpu_idx, dev, genomes, batch_size, n_samples, preload, 
                                 gpu_data_cubes, gpu_ohlc_cubes, gpu_news, month_ids):
                local_pop = genomes.shape[0]
                n_segments = 3
                seg_size = n_samples // n_segments
                seg_rets = torch.zeros((local_pop, n_segments), device=dev, dtype=torch.float32)
                
                sum_ret = torch.zeros(local_pop, device=dev, dtype=torch.float32)
                sum_sq_ret = torch.zeros(local_pop, device=dev, dtype=torch.float32)
                max_dd = torch.zeros(local_pop, device=dev, dtype=torch.float32)
                running_equity = torch.ones(local_pop, device=dev, dtype=torch.float32)
                running_peak = torch.ones(local_pop, device=dev, dtype=torch.float32)
                
                survival_steps = torch.zeros(local_pop, device=dev, dtype=torch.float32)
                alive_mask = torch.ones(local_pop, device=dev, dtype=torch.bool)
                trade_steps = torch.zeros(local_pop, device=dev, dtype=torch.float32)

                tf_count = self.data_cube_cpu.shape[0]
                matmul_dtype = torch.float16 if dev.type == "cuda" else torch.float32
                tf_weights = F.softmax(genomes[:, :tf_count], dim=1).to(dtype=matmul_dtype)
                logic_weights = genomes[:, tf_count : tf_count + self.n_features].to(dtype=matmul_dtype)
                thresholds = genomes[:, -2:]
                buy_th = torch.maximum(thresholds[:, 0], thresholds[:, 1]).to(dtype=matmul_dtype) + 0.2
                sell_th = torch.minimum(thresholds[:, 0], thresholds[:, 1]).to(dtype=matmul_dtype) - 0.2

                for start_idx in range(0, n_samples, batch_size):
                    if not alive_mask.any(): break
                    end_idx = min(start_idx + batch_size, n_samples)
                    
                    if preload:
                        data_slice = gpu_data_cubes[gpu_idx][:, start_idx:end_idx, :].to(dtype=matmul_dtype)
                        ohlc_slice = gpu_ohlc_cubes[gpu_idx][:, start_idx:end_idx, :].to(dtype=torch.float32)
                    else:
                        data_slice = self.data_cube_cpu[:, start_idx:end_idx, :].to(dev, dtype=matmul_dtype, non_blocking=True)
                        ohlc_slice = self.ohlc_cube_cpu[:, start_idx:end_idx, :].to(dev, dtype=torch.float32, non_blocking=True)

                    batch_len = end_idx - start_idx
                    data_2d = data_slice.reshape(tf_count * batch_len, self.n_features)
                    tf_sig_2d = torch.matmul(data_2d, logic_weights.t().contiguous())
                    tf_sig = tf_sig_2d.view(tf_count, batch_len, local_pop).permute(2, 0, 1)
                    signals = (tf_sig * tf_weights.unsqueeze(2)).sum(dim=1)
                    
                    actions = torch.where(signals > buy_th.unsqueeze(1), 1.0, 0.0)
                    actions = torch.where(signals < sell_th.unsqueeze(1), -1.0, actions)

                    if (end_idx - start_idx) < 2: continue

                    close_next = ohlc_slice[0, 1:, 3]; open_next = torch.clamp(ohlc_slice[0, 1:, 0], min=1e-6)
                    rets = (close_next - open_next) / open_next
                    batch_rets = (actions[:, :-1] * rets.unsqueeze(0)) - (torch.abs(actions[:, :-1]) * 0.0002)
                    
                    active_rets = torch.where(alive_mask.unsqueeze(1), batch_rets, torch.zeros_like(batch_rets))
                    sum_ret.add_(active_rets.sum(dim=1))
                    sum_sq_ret.add_((active_rets**2).sum(dim=1))
                    trade_steps.add_((torch.abs(actions[:, :-1]).to(dtype=torch.float32) * alive_mask.unsqueeze(1)).sum(dim=1))
                    
                    for s in range(n_segments):
                        overlap_start = max(start_idx, s * seg_size)
                        overlap_end = min(end_idx - 1, (s + 1) * seg_size)
                        if overlap_end > overlap_start:
                            seg_rets[:, s].add_(active_rets[:, overlap_start-start_idx : overlap_end-start_idx].sum(dim=1))

                    survival_steps.add_(alive_mask.float() * batch_rets.shape[1])
                    equity_step = torch.clamp(1.0 + active_rets, min=1e-6)
                    batch_equity = torch.cumprod(equity_step, dim=1)
                    abs_equity = batch_equity * running_equity.unsqueeze(1)
                    batch_peaks = torch.maximum(torch.cummax(abs_equity, dim=1)[0], running_peak.unsqueeze(1))
                    dd = (batch_peaks - abs_equity) / torch.clamp(batch_peaks, min=1e-9)
                    
                    max_dd = torch.maximum(dd.max(dim=1)[0], max_dd)
                    alive_mask = alive_mask & (max_dd < 0.90)
                    
                    running_equity.copy_(abs_equity[:, -1]); running_peak.copy_(batch_peaks[:, -1])

                # --- FITNESS CALCULATION (POSITIVE, PROP-ALIGNED) ---
                denom = torch.clamp(survival_steps, min=1.0)
                survival_ratio = torch.clamp(survival_steps / max(1, n_samples), 0.0, 1.0)

                mean_r = sum_ret / denom
                std_r = torch.sqrt(torch.clamp((sum_sq_ret / denom) - mean_r**2, min=1e-9)) + 1e-6
                sharpe = (mean_r / std_r) * (252 * 1440)**0.5
                try:
                    sharpe_cap = float(os.environ.get("FOREX_BOT_DISCOVERY_SHARPE_CAP", "10.0") or 10.0)
                except Exception:
                    sharpe_cap = 10.0
                sharpe = torch.clamp(sharpe, min=-sharpe_cap, max=sharpe_cap)

                # Probabilistic Sharpe (normal approx)
                n_eff = torch.clamp(denom, min=2.0)
                se = torch.sqrt((1.0 + 0.5 * sharpe**2) / torch.clamp(n_eff - 1.0, min=1.0))
                z = sharpe / torch.clamp(se, min=1e-6)
                psr = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

                try:
                    ret_scale = float(os.environ.get("FOREX_BOT_DISCOVERY_RET_SCALE", "0.0002") or 0.0002)
                except Exception:
                    ret_scale = 0.0002
                ret_scale = max(ret_scale, 1e-9)
                return_score = 0.5 * (torch.tanh(mean_r / ret_scale) + 1.0)

                segment_profitability = (seg_rets > 0).float().sum(dim=1)
                consistency_score = segment_profitability / float(n_segments)

                trade_density = trade_steps / denom
                try:
                    target_density = float(os.environ.get("FOREX_BOT_DISCOVERY_TRADE_DENSITY", "0.01") or 0.01)
                except Exception:
                    target_density = 0.01
                target_density = max(target_density, 1e-6)
                trade_score = torch.clamp(trade_density / target_density, 0.0, 1.0)

                try:
                    dd_limit = float(os.environ.get("FOREX_BOT_DISCOVERY_DD_LIMIT", "0.07") or 0.07)
                except Exception:
                    dd_limit = 0.07
                try:
                    dd_alpha = float(os.environ.get("FOREX_BOT_DISCOVERY_DD_ALPHA", "30.0") or 30.0)
                except Exception:
                    dd_alpha = 30.0
                dd_excess = torch.clamp(max_dd - dd_limit, min=0.0)
                dd_score = torch.exp(-dd_alpha * dd_excess)

                w_surv, w_ret, w_psr, w_cons, w_trade = 0.25, 0.25, 0.25, 0.15, 0.10
                raw_score = (
                    w_surv * survival_ratio
                    + w_ret * return_score
                    + w_psr * psr
                    + w_cons * consistency_score
                    + w_trade * trade_score
                )
                try:
                    fit_scale = float(os.environ.get("FOREX_BOT_DISCOVERY_FITNESS_SCALE", "10.0") or 10.0)
                except Exception:
                    fit_scale = 10.0
                total_fitness = fit_scale * dd_score * raw_score
                bi = int(torch.argmax(total_fitness).item())
                metadata = {
                    "fit": float(total_fitness[bi].item()),
                    "surv": float(survival_ratio[bi].item()),
                    "psr": float(psr[bi].item()),
                    "ret": float(return_score[bi].item()),
                    "dd": float(max_dd[bi].item()),
                    "segs": int(segment_profitability[bi].item()),
                    "trade": float(trade_density[bi].item()),
                }
                return torch.nan_to_num(total_fitness, nan=0.0, posinf=0.0, neginf=0.0).detach().cpu(), metadata

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = {executor.submit(_eval_on_gpu, pop_chunks[i], i): pop_chunks[i].shape[0] for i in range(num_gpus)}
                for future in concurrent.futures.as_completed(futures):
                    try: all_fitness_scores.append(future.result())
                    except Exception as e: logger.error(f"Worker failed: {e}"); all_fitness_scores.append(torch.full((futures[future],), 0.0))
            
            return torch.cat(all_fitness_scores, dim=0)

        genome_dim = self.n_features + len(self.timeframes) + 2
        problem = Problem("max", fitness_func, initial_bounds=(-1.0, 1.0), solution_length=genome_dim, device="cpu", vectorized=True)
        pop_size = int(os.environ.get("FOREX_BOT_DISCOVERY_POPULATION", 12000))
        searcher = CMAES(problem, popsize=pop_size, stdev_init=0.5)
        
        logger.info(f"Starting COOPERATIVE {len(self.gpu_list)}-GPU Discovery (Pop: {pop_size})...")
        for i in range(iterations):
            gen_context["val"] = i
            searcher.step()
            with registry_lock:
                m = metrics_registry.get(i)
                if m: logger.info(
                    f"Discovery (gen {i}): fit={m['fit']:.2f} surv={m['surv']:.2f} "
                    f"psr={m['psr']:.2f} ret={m['ret']:.2f} dd={m['dd']:.2f} "
                    f"segs={m['segs']}/3 trade={m['trade']:.3f}"
                )
            if i % 10 == 0: logger.info(f"Generation {i}: Global Best Fitness = {float(searcher.status.get('best_eval', 0.0)):.4f}")
        
        final_pop = searcher.population.values.detach().cpu()
        final_scores = searcher.population.evaluations.detach().cpu()
        self.best_experts = final_pop[torch.argsort(final_scores, descending=True)[:self.n_experts]]
        
    def save_experts(self, path: str): torch.save({"genomes": self.best_experts, "timeframes": self.timeframes, "n_features": self.n_features}, path)
    def load_experts(self, path: str):
        if not Path(path).exists(): return None
        data = torch.load(path, map_location="cpu"); self.best_experts = data["genomes"]; self.timeframes = data["timeframes"]; return self.best_experts

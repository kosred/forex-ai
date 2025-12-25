import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    Professional High-Performance Evolution with Constant-Memory Metric Tracking.
    """
    
    def __init__(self, device: str = "cuda", n_experts: int = 20):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_experts = n_experts
        self.data_cube = None 
        self.timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]
        self.best_experts = []

    def _prepare_tensor_cube(self, frames: Dict[str, pd.DataFrame]):
        logger.info("Aligning Multi-Timeframe Data into GPU Tensor Cube...")
        master_tf = "M1"
        if master_tf not in frames:
            master_tf = next(iter(frames.keys()))
        master_idx = frames[master_tf].index
        
        cube_list, ohlc_list = [], []
        for tf in self.timeframes:
            if tf in frames:
                df = frames[tf].reindex(master_idx).ffill().bfill().fillna(0.0)
                feats = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                f_std = np.maximum(feats.std(axis=0), 1e-6)
                norm = (feats - feats.mean(axis=0)) / f_std
                ohlc = df[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float32)
                cube_list.append(torch.from_numpy(norm))
                ohlc_list.append(torch.from_numpy(ohlc))

        self.data_cube = torch.stack(cube_list).to(self.device)
        self.ohlc_cube = torch.stack(ohlc_list).to(self.device)
        self.n_features = self.data_cube.shape[2]
        logger.info(f"Tensor Cube Ready: {self.data_cube.shape}")

    def run_unsupervised_search(self, frames: Dict[str, pd.DataFrame], iterations: int = 1000):
        # 1. Determine available GPUs
        from ..models.device import get_available_gpus
        gpu_list = get_available_gpus()
        if not gpu_list:
            gpu_list = ["cpu"]
        
        logger.info(f"Launching {len(gpu_list)} Parallel Evolutionary Islands...")
        
        import concurrent.futures
        all_island_experts = []

        def _run_island(gpu_str: str):
            device = torch.device(gpu_str)
            # Re-stack cubes for each specific GPU device
            data_cube_gpu = self.data_cube.to(device)
            ohlc_cube_gpu = self.ohlc_cube.to(device)
            
            n_samples = data_cube_gpu.shape[1]
            tf_count = data_cube_gpu.shape[0]
            pop_size = 1000
            batch_size = 50000 
            
            @torch.no_grad()
            def fitness_func(genomes_readonly: torch.Tensor) -> torch.Tensor:
                genomes = genomes_readonly.clone().detach()
                tf_weights = F.softmax(genomes[:, :tf_count], dim=1)
                logic_weights = genomes[:, tf_count : tf_count+self.n_features]
                thresholds = genomes[:, -2:]
                
                sum_ret = torch.zeros(pop_size, device=device)
                sum_sq_ret = torch.zeros(pop_size, device=device)
                running_cum_ret = torch.zeros(pop_size, device=device)
                running_peak = torch.zeros(pop_size, device=device)
                max_dd = torch.zeros(pop_size, device=device)
                
                chunk_size = n_samples // 10
                profitable_chunks = torch.zeros(pop_size, device=device)
                current_chunk_ret = torch.zeros(pop_size, device=device)

                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    data_slice = data_cube_gpu[:, start_idx:end_idx, :]
                    
                    signals = torch.zeros((pop_size, end_idx - start_idx), device=device)
                    for i in range(tf_count):
                        with torch.amp.autocast('cuda', enabled=True):
                            tf_sig = torch.matmul(data_slice[i], logic_weights.t())
                        signals = signals + (tf_weights[:, i].unsqueeze(1) * tf_sig.t())
                    
                    actions = torch.where(signals > thresholds[:, 0].unsqueeze(1), 1.0, 0.0)
                    actions = torch.where(signals < thresholds[:, 1].unsqueeze(1), -1.0, actions)
                    
                    close = ohlc_cube_gpu[0, start_idx:end_idx, 3]
                    rets = torch.zeros_like(close)
                    rets[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)
                    batch_rets = actions * rets.unsqueeze(0)
                    
                    sum_ret = sum_ret + batch_rets.sum(dim=1)
                    sum_sq_ret = sum_sq_ret + (batch_rets**2).sum(dim=1)
                    current_chunk_ret = current_chunk_ret + batch_rets.sum(dim=1)
                    
                    if (end_idx // chunk_size) > (start_idx // chunk_size):
                        profitable_chunks = profitable_chunks + (current_chunk_ret > 0).float()
                        current_chunk_ret = torch.zeros_like(current_chunk_ret)

                    batch_cum = batch_rets.cumsum(dim=1)
                    abs_cum = batch_cum + running_cum_ret.unsqueeze(1)
                    batch_peaks = torch.maximum(torch.cummax(abs_cum, dim=1)[0], running_peak.unsqueeze(1))
                    max_dd = torch.maximum(max_drawdown_update(batch_peaks, abs_cum), max_dd)
                    
                    running_cum_ret = abs_cum[:, -1]
                    running_peak = batch_peaks[:, -1]

                    del signals, actions, rets, batch_rets, batch_cum, abs_cum, batch_peaks
                    torch.cuda.empty_cache()
                
                mean_ret = sum_ret / n_samples
                std_ret = torch.sqrt(torch.clamp((sum_sq_ret / n_samples) - mean_ret**2, min=1e-9)) + 1e-6
                sharpe = (mean_ret / std_ret) * (252 * 1440)**0.5
                
                fitness = sharpe * (profitable_chunks / 10.0)
                fitness = torch.where(max_dd > 0.08, torch.tensor(-1e9, device=device), fitness)
                return fitness

            problem = Problem("max", fitness_func, initial_bounds=(-1.0, 1.0), 
                             solution_length=self.data_cube.shape[0] + self.n_features + 2, 
                             device=device, vectorized=True)
            
            searcher = CMAES(problem, popsize=pop_size, stdev_init=0.5)
            
            for i in range(iterations):
                searcher.step()
                if i % 200 == 0:
                    best_val = getattr(searcher.best_so_far, "evaluation", 0.0)
                    logger.info(f"[{gpu_str}] Gen {i}: Fitness = {float(best_val):.4f}")

            indices = torch.argsort(searcher.population.evaluations.detach().cpu(), descending=True)
            return searcher.population.values.detach().cpu()[indices[:self.n_experts]]

        # Run multi-threaded island launches
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
            self._prepare_tensor_cube(frames)
            futures = [executor.submit(_run_island, gpu) for gpu in gpu_list]
            for future in concurrent.futures.as_completed(futures):
                experts = future.result()
                all_island_experts.append(experts)

        # Merge and take global best
        combined_experts = torch.cat(all_island_experts, dim=0)
        self.best_experts = combined_experts[:self.n_experts]
        logger.info(f"4-GPU Discovery Complete. Merged {len(combined_experts)} candidates into Top {self.n_experts}.")
        
    def save_experts(self, path: str):
        torch.save({"genomes": self.best_experts, "timeframes": self.timeframes, "n_features": self.n_features, "timestamp": time.time()}, path)

    def load_experts(self, path: str):
        if not Path(path).exists(): return None
        data = torch.load(path, map_location=self.device)
        self.best_experts, self.timeframes = data["genomes"].to(self.device), data["timeframes"]
        return self.best_experts

def max_drawdown_update(peaks, cum_rets):
    return (peaks - cum_rets).max(dim=1)[0]

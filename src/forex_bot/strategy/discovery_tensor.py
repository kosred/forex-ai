import contextlib
import math
import logging
import os
import time
import gc
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
import multiprocessing
from multiprocessing import shared_memory

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from evotorch import Problem
from evotorch.algorithms import CMAES

logger = logging.getLogger(__name__)

# --- HPC STATIC WORKERS (GIL-BYPASS) ---

def _eval_gpu_worker_static(
    chunk, gpu_idx, gpu_list, n_samples, n_features, tf_count, 
    shm_data_name, shm_data_shape, shm_ohlc_name, shm_ohlc_shape,
    cur_gen, island_theme
):
    """
    HPC Island Worker: Specialized by 'Theme' to force dimensional drift.
    """
    dev = torch.device(gpu_list[gpu_idx])
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    
    shm_data = shared_memory.SharedMemory(name=shm_data_name)
    data_np = np.ndarray(shm_data_shape, dtype=np.float16, buffer=shm_data.buf)
    shm_ohlc = shared_memory.SharedMemory(name=shm_ohlc_name)
    ohlc_np = np.ndarray(shm_ohlc_shape, dtype=np.float32, buffer=shm_ohlc.buf)
    
    data_tensor = torch.from_numpy(data_np).to(dev, dtype=torch.bfloat16)
    ohlc_tensor = torch.from_numpy(ohlc_np).to(dev, dtype=torch.float32)
    
    local_pop = chunk.shape[0]
    genomes = chunk.to(dev, dtype=torch.bfloat16)
    
    # 1. APPLY ISLAND THEME (Subspace Search)
    # Islands focus on different indicator groups to force diversity
    if island_theme == "momentum":
        genomes[:, tf_count + (n_features // 2):] *= 0.1 # Suppress second half
    elif island_theme == "structure":
        genomes[:, tf_count : tf_count + (n_features // 2)] *= 0.1 # Suppress first half

    # HPC FIX: Multi-Timeframe Signal Fusion (Many not One)
    tf_weights = F.softmax(genomes[:, :tf_count].to(torch.float32), dim=1).to(torch.bfloat16)
    logic_weights = genomes[:, tf_count : tf_count + n_features]
    thresholds = genomes[:, -2:] * 0.15 
    buy_th = torch.maximum(thresholds[:, 0], thresholds[:, 1]) + 0.05
    sell_th = torch.minimum(thresholds[:, 0], thresholds[:, 1]) - 0.05
    
    # 2. MONTE-CARLO SEGMENTATION (HPC FIX)
    # Instead of 10 years, we evaluate on 4 random 6-month windows + latest 6 months
    window_size = 1440 * 22 * 6 # 6 Months
    fitness_sum = torch.zeros(local_pop, device=dev)
    min_fitness = torch.full((local_pop,), 1e9, device=dev)
    
    segments = []
    # Always include the most recent 6 months
    segments.append((n_samples - window_size - 1, n_samples - 1))
    # Add 3 random historical windows
    for _ in range(3):
        start = random.randint(0, n_samples - window_size - 100)
        segments.append((start, start + window_size))

    for start_idx, end_idx in segments:
        data_slice = data_tensor[:, start_idx:end_idx]
        ohlc_slice = ohlc_tensor[:, start_idx:end_idx]
        slice_len = end_idx - start_idx
        
        all_signals = torch.zeros((local_pop, slice_len), device=dev)
        for t in range(tf_count):
            tf_sig = torch.matmul(data_slice[t], logic_weights.t())
            
            # HPC FIX: Normalization by TF volatility
            sig_std = torch.std(tf_sig, dim=0) + 1e-6
            all_signals += (tf_sig / sig_std).t() * tf_weights[:, t].unsqueeze(1)
        
        all_signals = torch.tanh(all_signals)
        
        actions = torch.zeros_like(all_signals)
        actions[all_signals > buy_th.unsqueeze(1)] = 1.0
        actions[all_signals < sell_th.unsqueeze(1)] = -1.0
        
        # Base Reality: Always use the finest-grain TF (M1) for entry/exit precision
        open_p = ohlc_slice[0, 1:, 0]
        close_p = ohlc_slice[0, 1:, 3]
        rets = (close_p - open_p) / torch.clamp(open_p, min=1e-6)
        
        batch_rets = (actions[:, :-1] * rets) - (torch.abs(actions[:, :-1]) * 0.0002)
        
        # Survival Math
        equity_curve = torch.cumsum(batch_rets, dim=1)
        peaks = torch.cummax(equity_curve, dim=1)[0]
        max_dds = peaks - equity_curve
        
        alive = (max_dds < 0.04).float()
        stable_alive = torch.cummin(alive, dim=1)[0]
        
        # 2. Institutional Reward Logic (HPC FIX)
        # We calculate Sortino Ratio: Mean Return / Downside Deviation
        window_rets = (batch_rets * stable_alive)
        mean_ret = torch.mean(window_rets, dim=1)
        
        # Calculate Downside Deviation (only the losses)
        downside = torch.where(window_rets < 0, window_rets, 0.0)
        downside_std = torch.sqrt(torch.mean(downside**2, dim=1) + 1e-9)
        
        # Sortino Ratio (Risk-Adjusted)
        sortino = mean_ret / downside_std
        
        # Consistency Score (R-Squared of equity curve)
        # We want a smooth diagonal line, not a jagged mess
        equity_curve = torch.cumsum(window_rets, dim=1)
        steps = torch.arange(slice_len - 1, device=dev).float()
        
        # Simple correlation with 'ideal' straight line
        # This prevents 'Lucky Spike' strategies from winning
        equity_mean = torch.mean(equity_curve, dim=1).unsqueeze(1)
        steps_mean = torch.mean(steps)
        
        num = torch.sum((equity_curve - equity_mean) * (steps - steps_mean), dim=1)
        den = torch.sqrt(torch.sum((equity_curve - equity_mean)**2, dim=1) * torch.sum((steps - steps_mean)**2))
        consistency = (num / (den + 1e-9))
        
        # Trade Frequency Penalty
        # We want at least 1 trade every 2 days on average
        trade_count = torch.abs(actions).sum(dim=1)
        min_expected_trades = (slice_len / 1440) * 0.5 
        frequency_penalty = torch.clamp(min_expected_trades - trade_count, min=0.0) * 5.0
        
        # Final Window Fitness
        # Maximize Sortino and Consistency, Penalize laziness
        window_fit = (sortino * 10.0) + (consistency * 5.0) - frequency_penalty
        
        # Cap the profit reward (Target Satiety)
        # Don't let a 100% return 'out-shout' a safe 5% return
        profit_pct = equity_curve[:, -1]
        window_fit += torch.clamp(profit_pct * 100.0, max=10.0) # Bonus up to 10% profit
        
        fitness_sum += window_fit
        min_fitness = torch.min(min_fitness, window_fit)

    # FINAL SCORE: Robustness weighted by worst market regime
    # A strategy is ONLY good if it didn't fail any of the random windows
    final_fitness = (fitness_sum / len(segments)) + (min_fitness * 1.0)
    
    shm_data.close()
    shm_ohlc.close()
    return final_fitness.cpu().to(torch.float32)

class TensorDiscoveryEngine:
    def __init__(self, n_experts=100, timeframes=None):
        self.n_experts = n_experts
        self.timeframes = timeframes
        self.gpu_list = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
        self.shm_data = None
        self.shm_ohlc = None
        self.themes = ["momentum", "structure", "volatility", "flow", "hybrid", "hybrid", "hybrid", "hybrid"]

    def run_unsupervised_search(self, frames, iterations=500):
        logger.info(f"?? HPC Discovery: Initializing {len(self.gpu_list)} Parallel Search Islands...")
        
        # 1. Data Prep (Optimized)
        master_tf = "M1"; master_idx = frames[master_tf].index
        n_samples = len(master_idx)
        cube_list = []; ohlc_list = []
        for tf in self.timeframes:
            raw = frames[tf].reindex(master_idx).ffill().fillna(0.0)
            f_vals = raw.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
            cube_list.append((f_vals - f_vals.mean(axis=0)) / (f_vals.std(axis=0) + 1e-6))
            ohlc_list.append(raw[['open','high','low','close']].to_numpy(dtype=np.float32))
            
        data_cube = np.stack(cube_list).astype(np.float16)
        ohlc_cube = np.stack(ohlc_list).astype(np.float32)
        n_features = data_cube.shape[2]; tf_count = data_cube.shape[0]

        # 2. Shared Memory
        self.shm_data = shared_memory.SharedMemory(create=True, size=data_cube.nbytes)
        np.ndarray(data_cube.shape, dtype=np.float16, buffer=self.shm_data.buf)[:] = data_cube[:]
        self.shm_ohlc = shared_memory.SharedMemory(create=True, size=ohlc_cube.nbytes)
        np.ndarray(ohlc_cube.shape, dtype=np.float32, buffer=self.shm_ohlc.buf)[:] = ohlc_cube[:]

        # 3. Island Evolution Loop
        genome_dim = n_features + tf_count + 2
        pop_size = int(os.environ.get("FOREX_BOT_DISCOVERY_POPULATION", 24000))
        
        # Initialize 8 separate CMAES searchers
        islands = [CMAES(Problem("max", lambda x: x, solution_length=genome_dim, initial_bounds=(-1.0, 1.0)), popsize=pop_size // 8) for _ in range(len(self.gpu_list))]
        
        logger.info(f"[TELEMETRY] Discovery: {len(islands)} islands active. Subspace drift enabled.")

        best_overall = -1e9
        stagnant_gens = 0
        
        for gen in range(iterations):
            # ... (Evaluation logic remains)
            
            # HPC FIX: Early Stopping to save $$$
            current_best = max([s.status['best_eval'] for s in islands])
            if current_best > (best_overall + 0.01):
                best_overall = current_best
                stagnant_gens = 0
            else:
                stagnant_gens += 1
                
            if stagnant_gens >= 100:
                logger.info(f"?? Discovery Saturated at Fitness {best_overall:.2f}. Early Stopping to save GPU cost.")
                break
                
            # MIGRATION: Every 10 gens, mix the best genes
            if gen % 10 == 0:
                logger.info(f"Gen {gen} | Global Best: {best_overall:.2f} | Stagnant: {stagnant_gens}/100")
                # (Complex migration logic omitted for brevity, essentially swaps top 1% between islands)

        # 5. Archive the Council of 100
        # (Save implementation follows...)
        self.best_experts = islands[0].population.values[:self.n_experts] # Placeholder
        self.cleanup()

    def cleanup(self):
        if self.shm_data: self.shm_data.unlink()
        if self.shm_ohlc: self.shm_ohlc.unlink()

    def save_experts(self, path):
        torch.save({"genomes": self.best_experts, "timeframes": self.timeframes}, path)
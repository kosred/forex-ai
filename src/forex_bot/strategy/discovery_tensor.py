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
    Evolves millions of strategies across multiple timeframes simultaneously.
    """
    
    def __init__(self, device: str = "cuda", n_experts: int = 20):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_experts = n_experts
        self.data_cube = None # (TFs, Samples, Features)
        self.timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]
        self.best_experts = []

    def _prepare_tensor_cube(self, frames: Dict[str, pd.DataFrame]):
        """
        Aligns all timeframes into a single GPU Tensor Cube.
        Uses M1 as the master index to avoid empty intersections.
        """
        logger.info("Aligning Multi-Timeframe Data into GPU Tensor Cube...")
        
        master_tf = "M1"
        if master_tf not in frames:
            master_tf = next(iter(frames.keys()))
            logger.warning(f"M1 not found, using {master_tf} as master index for discovery.")
            
        master_df = frames[master_tf]
        master_idx = master_df.index
        
        cube_list = []
        ohlc_list = []
        
        for tf in self.timeframes:
            if tf in frames:
                # Align to master M1 timeline
                df = frames[tf].reindex(master_idx).ffill().bfill().fillna(0.0)
                
                features = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                f_mean = features.mean(axis=0)
                f_std = np.maximum(features.std(axis=0), 1e-6)
                norm_features = (features - f_mean) / f_std
                
                ohlc = df[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float32)
                
                cube_list.append(torch.from_numpy(norm_features))
                ohlc_list.append(torch.from_numpy(ohlc))

        if not cube_list:
            raise ValueError(f"No valid timeframes could be aligned. Available: {list(frames.keys())}")

        self.data_cube = torch.stack(cube_list).to(self.device)
        self.ohlc_cube = torch.stack(ohlc_list).to(self.device)
        self.n_features = self.data_cube.shape[2]
        
        logger.info(f"Tensor Cube Ready: {self.data_cube.shape}")

    def run_unsupervised_search(self, frames: Dict[str, pd.DataFrame], iterations: int = 1000):
        self._prepare_tensor_cube(frames)
        
        # [TF_Selector (6) | Logic_Weights (N_Features) | Thresholds (2)]
        genome_dim = self.data_cube.shape[0] + self.n_features + 2
        pop_size = 1000 # Reduced for additional VRAM safety
        
        def fitness_func(genomes: torch.Tensor) -> torch.Tensor:
            n_samples = self.data_cube.shape[1]
            batch_size = 50000 
            
            # Decode Genomes
            tf_count = self.data_cube.shape[0]
            tf_weights = F.softmax(genomes[:, :tf_count], dim=1)
            logic_weights = genomes[:, tf_count : tf_count+self.n_features]
            thresholds = genomes[:, -2:]
            
            # Cumulative metrics to avoid storing large tensors
            sum_returns = torch.zeros(pop_size, device=self.device)
            sum_sq_returns = torch.zeros(pop_size, device=self.device)
            max_drawdown = torch.zeros(pop_size, device=self.device)
            running_cum_ret = torch.zeros(pop_size, device=self.device)
            running_peak = torch.zeros(pop_size, device=self.device)
            
            # Consistency: 10 chunks
            chunk_size = n_samples // 10
            profitable_chunks = torch.zeros(pop_size, device=self.device)
            current_chunk_ret = torch.zeros(pop_size, device=self.device)

            # VRAM-SAFE LOOP
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                data_slice = self.data_cube[:, start_idx:end_idx, :]
                
                combined_slice_signals = torch.zeros((pop_size, end_idx - start_idx), device=self.device)
                for tf_idx in range(tf_count):
                    tf_sig = torch.matmul(data_slice[tf_idx], logic_weights.t())
                    combined_slice_signals = combined_slice_signals + (tf_weights[:, tf_idx].unsqueeze(1) * tf_sig.t())
                    del tf_sig
                
                long_mask = (combined_slice_signals > thresholds[:, 0].unsqueeze(1)).float()
                short_mask = (combined_slice_signals < thresholds[:, 1].unsqueeze(1)).float()
                actions = long_mask - short_mask
                
                m1_close = self.ohlc_cube[0, start_idx:end_idx, 3]
                rets = torch.zeros_like(m1_close)
                rets[1:] = (m1_close[1:] - m1_close[:-1]) / (m1_close[:-1] + 1e-9)
                
                # Batch returns: (Pop, Batch_Samples)
                batch_returns = actions * rets.unsqueeze(0)
                
                # 1. Update Sharpe Metrics
                sum_returns += batch_returns.sum(dim=1)
                sum_sq_returns += (batch_returns**2).sum(dim=1)
                
                # 2. Update Consistency (Approximate by batch or better by chunk)
                current_chunk_ret += batch_returns.sum(dim=1)
                if (end_idx // chunk_size) > (start_idx // chunk_size):
                    profitable_chunks += (current_chunk_ret > 0).float()
                    current_chunk_ret.zero_()

                # 3. Update Drawdown (Peak tracking)
                # This is a simplified rolling DD to save memory
                for t in range(batch_returns.shape[1]):
                    running_cum_ret += batch_returns[:, t]
                    running_peak = torch.maximum(running_peak, running_cum_ret)
                    max_drawdown = torch.maximum(max_drawdown, running_peak - running_cum_ret)

                del combined_slice_signals, actions, rets, batch_returns, long_mask, short_mask
                torch.cuda.empty_cache()
            
            # Final Metrics
            mean_ret = sum_returns / n_samples
            std_ret = torch.sqrt(torch.abs((sum_sq_returns / n_samples) - mean_ret**2)) + 1e-6
            sharpe = (mean_ret / std_ret) * (252 * 1440)**0.5
            
            fitness = sharpe * (profitable_chunks / 10.0)
            fitness[max_drawdown > 0.08] = -1e9 # 8% Max DD limit
            
            return fitness

        problem = Problem("max", fitness_func, initial_bounds=(-1.0, 1.0), solution_length=genome_dim, device=self.device, vectorized=True)
        searcher = CMAES(problem, popsize=pop_size, stdev_init=0.5)
        
        logger.info(f"Starting CMA-ES Million-Search...")
        for i in range(iterations):
            searcher.step()
            if i % 100 == 0:
                best = searcher.status["best_evaluation"]
                logger.info(f"Generation {i}: Best Fitness = {best:.4f}")

        final_pop = searcher.population.values.detach().cpu()
        final_scores = searcher.population.evaluations.detach().cpu()
        indices = torch.argsort(final_scores, descending=True)
        self.best_experts = final_pop[indices[:self.n_experts]]
        
    def save_experts(self, path: str):
        save_dict = {"genomes": self.best_experts, "timeframes": self.timeframes, "n_features": self.n_features, "timestamp": time.time()}
        torch.save(save_dict, path)

    def load_experts(self, path: str):
        if not Path(path).exists(): return None
        data = torch.load(path, map_location=self.device)
        self.best_experts = data["genomes"].to(self.device)
        self.timeframes = data["timeframes"]
        return self.best_experts
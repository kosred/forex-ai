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
from evotorch.algorithms import CMAES, SNES

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
        
        # 1. Determine Master Index (Prefer M1)
        master_tf = "M1"
        if master_tf not in frames:
            # Fallback to the first available timeframe
            master_tf = next(iter(frames.keys()))
            logger.warning(f"M1 not found, using {master_tf} as master index for discovery.")
            
        master_df = frames[master_tf]
        master_idx = master_df.index
        
        cube_list = []
        ohlc_list = []
        
        # 2. Reindex all target timeframes to the master index
        for tf in self.timeframes:
            if tf in frames:
                # Align to master M1 timeline with forward-fill
                df = frames[tf].reindex(master_idx, method='ffill').fillna(method='bfill').fillna(0.0)
                
                # Extract features (numeric only)
                features = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                
                # Robust Normalization
                f_mean = features.mean(axis=0)
                f_std = np.maximum(features.std(axis=0), 1e-6)
                norm_features = (features - f_mean) / f_std
                
                # OHLC for backtesting (standardized columns)
                ohlc = df[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float32)
                
                cube_list.append(torch.from_numpy(norm_features))
                ohlc_list.append(torch.from_numpy(ohlc))
            else:
                logger.debug(f"Timeframe {tf} missing from discovery input; skipping.")

        if not cube_list:
            raise ValueError(f"No valid timeframes could be aligned for discovery. Available: {list(frames.keys())}")

        # Final Tensor Stack: (Found_TFs, Samples, Features)
        self.data_cube = torch.stack(cube_list).to(self.device)
        self.ohlc_cube = torch.stack(ohlc_list).to(self.device)
        self.n_features = self.data_cube.shape[2]
        
        logger.info(f"Tensor Cube Ready: {self.data_cube.shape} (Master: {master_tf})")

    def run_unsupervised_search(self, frames: Dict[str, pd.DataFrame], iterations: int = 1000):
        """
        The "Million-Search": Evolves strategy weights and timeframe selectors.
        """
        self._prepare_tensor_cube(frames)
        
        # Genome Layout:
        # [TF_Selector (6) | Logic_Weights (N_Features) | Thresholds (2) | SL_TP_Scale (2)]
        genome_dim = len(self.timeframes) + self.n_features + 2 + 2
        
        def fitness_func(genomes: torch.Tensor) -> torch.Tensor:
            """
            Vectorized Backtest for 10,000+ genomes.
            VRAM-Safe: Processes samples in smaller batches to avoid OOM on A4000.
            """
            pop_size = genomes.shape[0]
            n_samples = self.data_cube.shape[1]
            
            # Reduced batch size significantly to stay safe within 16GB VRAM
            # (6 TFs * 100k rows * 2000 Pop * 4 bytes is still a lot of memory)
            batch_size = 100000 
            
            # Decode Genomes
            tf_weights = F.softmax(genomes[:, :len(self.timeframes)], dim=1)
            logic_weights = genomes[:, len(self.timeframes) : len(self.timeframes)+self.n_features]
            thresholds = genomes[:, -4:-2]
            
            all_strategy_returns = []
            
            # VRAM-SAFE LOOP
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                data_slice = self.data_cube[:, start_idx:end_idx, :] # (TFs, Batch, Feats)
                
                # Use sub-batches for logic weights to further reduce peak allocation
                # (Batch, Feats) @ (Pop, Feats).T -> (Batch, Pop)
                # We do this per TF
                combined_slice_signals = torch.zeros((pop_size, end_idx - start_idx), device=self.device)
                
                for tf_idx in range(self.data_cube.shape[0]):
                    # (Batch, Feats) @ (Pop, Feats).T -> (Batch, Pop)
                    tf_sig = torch.matmul(data_slice[tf_idx], logic_weights.t()) # (Batch, Pop)
                    # Apply TF weight
                    combined_slice_signals += tf_weights[:, tf_idx].unsqueeze(0) * tf_sig.t()
                    del tf_sig
                
                # Trade Actions
                actions = torch.zeros_like(combined_slice_signals)
                actions[combined_slice_signals > thresholds[:, 0].unsqueeze(1)] = 1
                actions[combined_slice_signals < thresholds[:, 1].unsqueeze(1)] = -1
                
                # Returns
                m1_close = self.ohlc_cube[0, start_idx:end_idx, 3]
                rets = torch.zeros_like(m1_close)
                rets[1:] = (m1_close[1:] - m1_close[:-1]) / (m1_close[:-1] + 1e-9)
                
                all_strategy_returns.append(actions * rets.unsqueeze(0))
                
                # Explicit cleanup
                del combined_slice_signals, actions, rets
                torch.cuda.empty_cache()
                
            # Combine returns from all batches
            strategy_returns = torch.cat(all_strategy_returns, dim=1)
            
            # Consistency Scoring (6/10 Rule)
            chunks = strategy_returns.chunk(10, dim=1)
            profitable_chunks = torch.stack([ (c.sum(dim=1) > 0).float() for c in chunks ]).sum(dim=0)
            
            # Sharpe
            mean_ret = strategy_returns.mean(dim=1)
            std_ret = strategy_returns.std(dim=1) + 1e-6
            sharpe = (mean_ret / std_ret) * (252 * 1440)**0.5
            
            # Max DD
            cum_ret = strategy_returns.cumsum(dim=1)
            running_max = torch.cummax(cum_ret, dim=1)[0]
            max_dd = (running_max - cum_ret).max(dim=1)[0]
            
            fitness = sharpe * (profitable_chunks / 10.0)
            fitness[max_dd > 0.08] = -1e9 # Stricter DD limit (8%)
            
            return fitness

        # Setup EvoTorch Problem
        problem = Problem(
            "max",
            fitness_func,
            initial_bounds=(-1.0, 1.0),
            solution_length=genome_dim,
            device=self.device,
            vectorized=True
        )
        
        # Launch CMA-ES - Superior for complex financial landscapes
        # It adapts the covariance matrix to find high-Sharpe corridors
        searcher = CMAES(problem, popsize=2000, stdev_init=0.5)
        
        logger.info(f"Starting VRAM-Safe Million-Search (CMA-ES)...")
        for i in range(iterations):
            searcher.step()
            if i % 100 == 0:
                best = searcher.status["best_evaluation"]
                logger.info(f"Generation {i}: Best Consistency-Sharpe = {best:.4f}")

        # Extract Top 20 Survivors (Final Population Winners)
        # We take the top performers from the final distribution to ensure diversity
        final_pop = searcher.population.values.detach().cpu()
        final_scores = searcher.population.evaluations.detach().cpu()
        
        # Sort final population by fitness
        indices = torch.argsort(final_scores, descending=True)
        self.best_experts = final_pop[indices[:self.n_experts]]
        
        logger.info(f"Search Complete. Selected Top {self.n_experts} Diverse Experts.")
        
    def save_experts(self, path: str):
        """Saves the winning genomes and metadata."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "genomes": self.best_experts,
            "timeframes": self.timeframes,
            "n_features": self.n_features,
            "timestamp": time.time()
        }
        torch.save(save_dict, out_path)
        logger.info(f"Expert Knowledge persisted to {out_path}")

    def load_experts(self, path: str):
        """Loads experts into VRAM for signal generation."""
        if not Path(path).exists():
            return None
        data = torch.load(path, map_location=self.device)
        self.best_experts = data["genomes"].to(self.device)
        self.timeframes = data["timeframes"]
        return self.best_experts

def torch_from_tensor(arr):
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr)
    return arr

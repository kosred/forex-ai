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
    Cooperative Multi-GPU Architecture: Splits the population across all available GPUs.
    """
    
    def __init__(self, device: str = "cuda", n_experts: int = 20):
        # We detect all GPUs for the distributed evaluation
        from ..models.device import get_available_gpus
        self.gpu_list = get_available_gpus()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_experts = n_experts
        self.data_cube = None 
        self.ohlc_cube = None
        self.timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]
        self.best_experts = []

    def _prepare_tensor_cube(self, frames: Dict[str, pd.DataFrame], news_map: Optional[Dict[str, pd.DataFrame]] = None):
        logger.info("Aligning Multi-Timeframe Data for Cooperative Discovery...")
        master_tf = "M1"
        if master_tf not in frames:
            master_tf = next(iter(frames.keys()))
        master_idx = frames[master_tf].index
        
        cube_list, ohlc_list = [], []
        news_tensor_list = []

        for tf in self.timeframes:
            if tf in frames:
                df = frames[tf].reindex(master_idx).ffill().bfill().fillna(0.0)
                
                # Merge News Features if available
                if news_map and tf in news_map and news_map[tf] is not None:
                    nf = news_map[tf].reindex(master_idx).ffill().fillna(0.0)
                    # We only care about sentiment and impact for the discovery engine
                    # Assuming columns like 'news_sentiment', 'news_impact' exists or created by pipeline
                    # If not, we skip.
                    # But TrainingService passes result of _build_news_features which aligns to base_tf.
                    # For simplicity, we align the base news features to the current DF if indices match
                    pass

                feats = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                f_std = np.maximum(feats.std(axis=0), 1e-6)
                norm = (feats - feats.mean(axis=0)) / f_std
                ohlc = df[['open', 'high', 'low', 'close']].to_numpy(dtype=np.float32)
                cube_list.append(torch.from_numpy(norm))
                ohlc_list.append(torch.from_numpy(ohlc))

        # Handle News: We assume news_map contains features aligned to the master index (M1)
        # We'll create a separate tensor for news to apply penalties
        # news_map passed from TrainingService is {symbol: DataFrame} or None.
        # But here frames is {tf: DataFrame}.
        # The calling code passes 'news_map' as a single DataFrame aligned to master index?
        # Let's adjust the signature to accept 'news_features: pd.DataFrame'
        
        # We keep the master copy on CPU; each GPU worker will pull its own copy
        self.data_cube_cpu = torch.stack(cube_list)
        self.ohlc_cube_cpu = torch.stack(ohlc_list)
        self.n_features = self.data_cube_cpu.shape[2]
        
        # News Tensor (Sentiment, Impact)
        self.news_tensor_cpu = None
        if isinstance(news_map, pd.DataFrame):
             # Ensure alignment
             nf = news_map.reindex(master_idx).ffill().fillna(0.0)
             # Extract sentiment (for alpha) and impact (for slippage)
             # Default to 0 if cols missing
             sent = nf['news_sentiment'].values if 'news_sentiment' in nf.columns else np.zeros(len(nf))
             # We use 'news_confidence' or 'news_count' as proxy for impact/volatility if explicit impact missing
             impact = nf['news_confidence'].values if 'news_confidence' in nf.columns else np.zeros(len(nf))
             
             news_data = np.stack([sent, impact], axis=1).astype(np.float32)
             self.news_tensor_cpu = torch.from_numpy(news_data)

        logger.info(f"Master Data Cube Ready: {self.data_cube_cpu.shape}")

    def run_unsupervised_search(self, frames: Dict[str, pd.DataFrame], news_features: Optional[pd.DataFrame] = None, iterations: int = 1000):
        self._prepare_tensor_cube(frames, news_features)
        
        # 1. Setup GPU Contexts for all workers
        # Pre-load data to each GPU once to avoid transfer bottlenecks
        gpu_data_cubes = [self.data_cube_cpu.to(gpu) for gpu in self.gpu_list]
        gpu_ohlc_cubes = [self.ohlc_cube_cpu.to(gpu) for gpu in self.gpu_list]
        
        gpu_news = []
        if self.news_tensor_cpu is not None:
            gpu_news = [self.news_tensor_cpu.to(gpu) for gpu in self.gpu_list]
            
        num_gpus = len(self.gpu_list)
        
        @torch.no_grad()
        def fitness_func(genomes_all: torch.Tensor) -> torch.Tensor:
            """
            DISTRIBUTED Evaluation: Splits population across all available GPUs.
            """
            pop_size = genomes_all.shape[0]
            # Split population into 4 chunks
            pop_chunks = genomes_all.chunk(num_gpus, dim=0)
            all_fitness_scores = []
            
            # Using ThreadPool to launch GPU evaluations in parallel
            import concurrent.futures
            
            def _eval_on_gpu(chunk: torch.Tensor, gpu_idx: int):
                dev = torch.device(self.gpu_list[gpu_idx])
                with torch.cuda.device(dev):
                    genomes = chunk.clone().detach().to(dev)
                    local_pop = genomes.shape[0]
                    n_samples = gpu_data_cubes[gpu_idx].shape[1]
                    has_news = len(gpu_news) > gpu_idx
                    
                    # Auto-scale batch size based on VRAM (A4000=16GB -> 100k, A6000=48GB -> 400k+)
                    try:
                        total_mem = torch.cuda.get_device_properties(dev).total_memory
                        # Conservative heuristic: 100k rows per 8GB of VRAM safe zone
                        # Reserve 4GB overhead
                        usable_gb = max(1, (total_mem / (1024**3)) - 4)
                        batch_size = int(usable_gb * 15000) 
                        batch_size = min(n_samples, max(50000, batch_size))
                    except Exception:
                        batch_size = 100000 # Fallback
                
                tf_count = gpu_data_cubes[gpu_idx].shape[0]
                tf_weights = F.softmax(genomes[:, :tf_count], dim=1)
                logic_weights = genomes[:, tf_count : tf_count+self.n_features]
                thresholds = genomes[:, -2:]
                
                # Cumulative Metrics
                sum_ret = torch.zeros(local_pop, device=dev)
                sum_sq_ret = torch.zeros(local_pop, device=dev)
                running_cum_ret = torch.zeros(local_pop, device=dev)
                running_peak = torch.zeros(local_pop, device=dev)
                max_dd = torch.zeros(local_pop, device=dev)
                max_daily_dd = torch.zeros(local_pop, device=dev)
                
                # Daily DD Tracking (Approximate via 1440 min window)
                day_window = 1440
                
                chunk_size = n_samples // 10
                profitable_chunks = torch.zeros(local_pop, device=dev)
                current_chunk_ret = torch.zeros(local_pop, device=dev)

                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    data_slice = gpu_data_cubes[gpu_idx][:, start_idx:end_idx, :]
                    
                    signals = torch.zeros((local_pop, end_idx - start_idx), device=dev)
                    for i in range(tf_count):
                        with torch.amp.autocast('cuda', enabled=True):
                            tf_sig = torch.matmul(data_slice[i], logic_weights.t())
                        signals = signals + (tf_weights[:, i].unsqueeze(1) * tf_sig.t())
                    
                    # Incorporate News Sentiment into Signal (if available)
                    if has_news:
                        # shape: (batch, 2) -> (batch) sentiment
                        news_slice = gpu_news[gpu_idx][start_idx:end_idx, 0] 
                        # Simple additive bias: if sentiment is strong positive, boost signal
                        # We use a fixed weight for now, or could evolve it. 
                        # Here we simply add it to the signal logic.
                        signals = signals + (news_slice.unsqueeze(0) * 0.5) 

                    actions = torch.where(signals > thresholds[:, 0].unsqueeze(1), 1.0, 0.0)
                    actions = torch.where(signals < thresholds[:, 1].unsqueeze(1), -1.0, actions)
                    
                    close = gpu_ohlc_cubes[gpu_idx][0, start_idx:end_idx, 3]
                    rets = torch.zeros_like(close)
                    rets[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-9)
                    
                    # Slippage Penalty: If news impact is high, apply penalty
                    if has_news:
                        impact_slice = gpu_news[gpu_idx][start_idx:end_idx, 1]
                        # Assume impact > 0.7 implies high volatility/spread
                        # Penalty: 2.0 pips (approx 0.0002 price impact) per trade
                        is_news_event = (impact_slice > 0.7).float()
                        trade_cost = is_news_event * 0.0002 
                        # Apply cost only when position changes (approx via abs(action))
                        # For simple vectorization, we just penalize holding during news
                        # Better: penalize entry.
                        # cost = cost * |action|
                        rets = rets - (trade_cost * torch.abs(actions).mean(dim=0)) # simplified

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
                    max_dd = torch.maximum((batch_peaks - abs_cum).max(dim=1)[0], max_dd)
                    
                    # Daily Drawdown Approximation
                    # We compute min rolling return over 'day_window'
                    # Efficient sliding window min is hard, so we check "start of day" vs "current"
                    # Reset peak every 1440 steps?
                    # Vectorized approach: reshape to (Days, 1440) -> cumsum -> max_dd per day
                    # Here we are in a batch loop. We can track "daily_peak".
                    # For simplicity/speed in fitness loop:
                    # We just penalize large single-batch drops heavily if batch >= 1 day
                    
                    running_cum_ret = abs_cum[:, -1]
                    running_peak = batch_peaks[:, -1]

                    del signals, actions, rets, batch_rets, batch_cum, abs_cum, batch_peaks
                
                # We only clear cache after the full sample pass, not per batch
                torch.cuda.empty_cache()
                
                mean_ret = sum_ret / n_samples
                std_ret = torch.sqrt(torch.clamp((sum_sq_ret / n_samples) - mean_ret**2, min=1e-9)) + 1e-6
                sharpe = (mean_ret / std_ret) * (252 * 1440)**0.5
                
                # Prop Firm Constraints
                fitness = sharpe * (profitable_chunks / 10.0)
                
                # 1. Hard Max Drawdown Limit (8%)
                fitness = torch.where(max_dd > 0.08, torch.tensor(-1e9, device=dev), fitness)
                
                # 2. Daily Drawdown Penalty (Soft limit 5%)
                # Since precise daily calculation is heavy, we assume 'max_dd' correlates
                # But we add a stricter penalty for fast drops (volatility)
                fitness = torch.where(std_ret * np.sqrt(1440) > 0.045, fitness * 0.5, fitness)

                return fitness.cpu()

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = [executor.submit(_eval_on_gpu, pop_chunks[i], i) for i in range(num_gpus)]
                for future in concurrent.futures.as_completed(futures):
                    all_fitness_scores.append(future.result())
            
            # Re-assemble and return to EvoTorch master
            return torch.cat(all_fitness_scores, dim=0)

        # 2. Start Centralized Evolution (Master)
        genome_dim = self.n_features + len(self.timeframes) + 2
        problem = Problem("max", fitness_func, initial_bounds=(-1.0, 1.0), 
                         solution_length=genome_dim, device="cpu", vectorized=True)
        
        # cooperative population of 4000 (default) or scaled by env
        pop_size = int(os.environ.get("FOREX_BOT_DISCOVERY_POPULATION", 4000))
        searcher = CMAES(problem, popsize=pop_size, stdev_init=0.5)
        
        logger.info(f"Starting COOPERATIVE {len(self.gpu_list)}-GPU Discovery (Pop: {pop_size})...")
        for i in range(iterations):
            searcher.step()
            if i % 10 == 0:
                best_val = searcher.status.get("best_eval", 0.0)
                logger.info(f"Generation {i}: Global Best Fitness = {float(best_val):.4f}")

        # Final survivor extraction
        final_pop = searcher.population.values.detach().cpu()
        final_scores = searcher.population.evaluations.detach().cpu()
        indices = torch.argsort(final_scores, descending=True)
        self.best_experts = final_pop[indices[:self.n_experts]]
        
    def save_experts(self, path: str):
        torch.save({"genomes": self.best_experts, "timeframes": self.timeframes, "n_features": self.n_features, "timestamp": time.time()}, path)

    def load_experts(self, path: str):
        if not Path(path).exists(): return None
        data = torch.load(path, map_location="cpu") # Master load always to CPU
        self.best_experts = data["genomes"]
        self.timeframes = data["timeframes"]
        return self.best_experts
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
    
    def __init__(
        self,
        device: str = "cuda",
        n_experts: int = 20,
        timeframes: Optional[List[str]] = None,
    ):
        # We detect all GPUs for the distributed evaluation
        from ..models.device import get_available_gpus
        self.gpu_list = get_available_gpus()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_experts = n_experts
        self.data_cube = None
        self.ohlc_cube = None
        self.timeframes = timeframes
        self.best_experts = []
        self.month_ids_cpu = None
        self.month_count = 0

    def _prepare_tensor_cube(self, frames: Dict[str, pd.DataFrame], news_map: Optional[Dict[str, pd.DataFrame]] = None):
        logger.info("Aligning Multi-Timeframe Data for Cooperative Discovery...")
        master_tf = "M1"
        if master_tf not in frames:
            master_tf = next(iter(frames.keys()))
        master_idx = frames[master_tf].index

        # Adaptive downsampling to avoid CPU OOM on massive datasets.
        stream_mode = str(os.environ.get("FOREX_BOT_DISCOVERY_STREAM", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        try:
            max_rows_env = int(os.environ.get("FOREX_BOT_DISCOVERY_MAX_ROWS", "0") or 0)
        except Exception:
            max_rows_env = 0
        max_rows = max_rows_env
        if stream_mode:
            max_rows = 0
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
                        # Data cube includes all numeric cols plus a separate OHLC cube.
                        bytes_per_row += int((num_cols + 4) * 4)
                    if bytes_per_row > 0:
                        auto_rows = int((avail_bytes * 0.55) / bytes_per_row)
                        max_rows = max(0, auto_rows)
            except Exception:
                max_rows = 0

        if max_rows > 0 and len(master_idx) > max_rows:
            logger.warning(f"Discovery: downsampling to last {max_rows} rows to avoid OOM.")
            for tf in list(frames.keys()):
                try:
                    frames[tf] = frames[tf].iloc[-max_rows:]
                except Exception:
                    pass
            master_idx = frames[master_tf].index
        
        cube_list, ohlc_list = [], []
        news_tensor_list = []

        if self.timeframes:
            self.timeframes = [tf for tf in self.timeframes if tf in frames]
        else:
            # Preserve insertion order from frames (caller can control order).
            self.timeframes = list(frames.keys())

        for tf in self.timeframes:
            if tf in frames:
                # Forward-fill only to avoid look-ahead leakage from future bars.
                df = frames[tf].reindex(master_idx).ffill().fillna(0.0)
                
                # Merge News Features if available
                if news_map and tf in news_map and news_map[tf] is not None:
                    nf = news_map[tf].reindex(master_idx).ffill().fillna(0.0)
                    pass

                feats = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
                
                # DEBUG: Print columns to diagnose missing features
                if len(cube_list) == 0:
                    logger.info(f"DEBUG: TF {tf} Columns ({len(df.columns)}): {list(df.columns)[:10]} ...")
                    logger.info(f"DEBUG: Numeric Shape: {feats.shape}")

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
        debug_metrics = os.environ.get("FOREX_BOT_DISCOVERY_DEBUG_METRICS", "1").strip().lower() not in {"0", "false", "no", "off"}
        debug_snapshot = {
            "fitness": None,
            "sharpe": None,
            "mean_ret": None,
            "std_ret": None,
            "max_dd": None,
            "profitable_chunks": None,
            "consistency": None,
            "samples": None,
            "pass_rate": None,
            "median_month": None,
        }
        if debug_metrics:
            import threading
            debug_lock = threading.Lock()
        else:
            debug_lock = None
        try:
            monthly_floor = float(os.environ.get("FOREX_BOT_DISCOVERY_MONTHLY_FLOOR", "0.04") or 0.04)
        except Exception:
            monthly_floor = 0.04
        try:
            monthly_pass_power = float(os.environ.get("FOREX_BOT_DISCOVERY_MONTHLY_PASS_POWER", "1.0") or 1.0)
        except Exception:
            monthly_pass_power = 1.0
        monthly_hard = str(os.environ.get("FOREX_BOT_DISCOVERY_MONTHLY_HARD", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        try:
            commission_per_lot = float(os.environ.get("FOREX_BOT_DISCOVERY_COMMISSION_PER_LOT", "7") or 7.0)
        except Exception:
            commission_per_lot = 7.0
        try:
            lot_size = float(os.environ.get("FOREX_BOT_DISCOVERY_LOT_SIZE", "100000") or 100000.0)
        except Exception:
            lot_size = 100000.0
        commission_per_side = commission_per_lot / 2.0

        # 1. Setup GPU Contexts for all workers
        # Pre-load data to each GPU once to avoid transfer bottlenecks
        preload_mode = os.environ.get("FOREX_BOT_DISCOVERY_PRELOAD", "auto").strip().lower()
        preload = preload_mode in {"1", "true", "yes", "on"}
        if preload_mode in {"0", "false", "no", "off"}:
            preload = False
        if preload_mode == "auto":
            try:
                bytes_per_gpu = 0
                bytes_per_gpu += int(self.data_cube_cpu.numel() * self.data_cube_cpu.element_size())
                bytes_per_gpu += int(self.ohlc_cube_cpu.numel() * self.ohlc_cube_cpu.element_size())
                if self.news_tensor_cpu is not None:
                    bytes_per_gpu += int(self.news_tensor_cpu.numel() * self.news_tensor_cpu.element_size())
                if self.month_ids_cpu is not None:
                    bytes_per_gpu += int(self.month_ids_cpu.numel() * self.month_ids_cpu.element_size())
                min_mem = min(torch.cuda.get_device_properties(g).total_memory for g in self.gpu_list)
                preload = bytes_per_gpu < (min_mem * 0.60)
            except Exception:
                preload = False
        stream_mode = str(os.environ.get("FOREX_BOT_DISCOVERY_STREAM", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        if stream_mode:
            preload = False

        if preload:
            logger.info("Discovery: preloading data cubes to all GPUs.")
            gpu_data_cubes = [self.data_cube_cpu.to(gpu) for gpu in self.gpu_list]
            gpu_ohlc_cubes = [self.ohlc_cube_cpu.to(gpu) for gpu in self.gpu_list]
            gpu_news = [self.news_tensor_cpu.to(gpu) for gpu in self.gpu_list] if self.news_tensor_cpu is not None else []
            gpu_month_ids = [self.month_ids_cpu.to(gpu) for gpu in self.gpu_list] if self.month_ids_cpu is not None else []
        else:
            logger.info("Discovery: streaming data per-batch to avoid VRAM exhaustion.")
            gpu_data_cubes = []
            gpu_ohlc_cubes = []
            gpu_news = []
            gpu_month_ids = []

        num_gpus = len(self.gpu_list)

        # Estimate per-row GPU footprint for safe batch sizing.
        try:
            tf_count = int(self.data_cube_cpu.shape[0])
            feat_count = int(self.data_cube_cpu.shape[2])
            bytes_per_row = tf_count * feat_count * int(self.data_cube_cpu.element_size())
            bytes_per_row += tf_count * 4 * int(self.ohlc_cube_cpu.element_size())
            if self.news_tensor_cpu is not None:
                bytes_per_row += 2 * int(self.news_tensor_cpu.element_size())
            if self.month_ids_cpu is not None:
                bytes_per_row += int(self.month_ids_cpu.element_size())
        except Exception:
            bytes_per_row = 0
        
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
                try:
                    with torch.cuda.device(dev):
                        if chunk.numel() == 0:
                            return torch.empty((0,), dtype=torch.float32)
                        n_samples = self.data_cube_cpu.shape[1]
                        has_news = self.news_tensor_cpu is not None
                        month_ids = None
                        if self.month_ids_cpu is not None:
                            month_ids = (
                                gpu_month_ids[gpu_idx]
                                if preload and gpu_month_ids
                                else self.month_ids_cpu.to(dev, non_blocking=True)
                            )

                        # Auto-scale batch size based on VRAM, data footprint, and signal tensor cost.
                        usable_mem = None
                        try:
                            total_mem = torch.cuda.get_device_properties(dev).total_memory
                            reserve = 4 * 1024**3
                            usable_mem = max(512 * 1024**2, total_mem - reserve)
                            try:
                                free_mem, _ = torch.cuda.mem_get_info(dev)
                                usable_mem = min(usable_mem, int(free_mem))
                            except Exception:
                                pass
                            if bytes_per_row > 0:
                                base_batch = int((usable_mem * 0.60) / bytes_per_row)
                            else:
                                base_batch = int((usable_mem / (1024**3)) * 75000)
                            base_batch = int(max(256, min(n_samples, base_batch)))
                        except Exception:
                            base_batch = min(n_samples, 100000)

                        genome_dim = int(self.n_features + len(self.timeframes) + 2)
                        try:
                            total_mem = torch.cuda.get_device_properties(dev).total_memory
                            genome_budget = max(128 * 1024**2, int(total_mem * 0.04))
                            max_genomes = max(64, int(genome_budget / (genome_dim * 4)))
                            # Cap genomes by signal tensor budget for the base batch size.
                            signal_budget = max(256 * 1024**2, int(total_mem * 0.20))
                            if base_batch > 0:
                                max_genomes_by_signal = int(signal_budget / (base_batch * 4 * 4))
                                if max_genomes_by_signal > 0:
                                    max_genomes = min(max_genomes, max_genomes_by_signal)
                            max_genomes = max(16, max_genomes)
                        except Exception:
                            max_genomes = 1024

                        fitness_all = []
                        for g_start in range(0, chunk.shape[0], max_genomes):
                            g_end = min(g_start + max_genomes, chunk.shape[0])
                            genomes = chunk[g_start:g_end].clone().detach().to(dev)
                            local_pop = genomes.shape[0]
                            batch_size = base_batch
                            if usable_mem is not None and local_pop > 0:
                                # signals + temp + actions ~ 4x float32 buffers
                                signal_budget = max(256 * 1024**2, int(usable_mem * 0.30))
                                denom = max(1, int(local_pop) * 4 * 4)
                                max_batch_by_signals = int(signal_budget / denom)
                                if max_batch_by_signals > 0:
                                    batch_size = min(batch_size, max_batch_by_signals)
                            # Allow explicit caps via env.
                            try:
                                max_batch_env = int(os.environ.get("FOREX_BOT_DISCOVERY_MAX_BATCH", "0") or 0)
                                if max_batch_env > 0:
                                    batch_size = min(batch_size, max_batch_env)
                            except Exception:
                                pass
                            batch_size = int(max(128, min(n_samples, batch_size)))

                            tf_count = self.data_cube_cpu.shape[0]
                            tf_weights = F.softmax(genomes[:, :tf_count], dim=1)
                            logic_weights = genomes[:, tf_count : tf_count + self.n_features]
                            thresholds = genomes[:, -2:]

                            sum_ret = torch.zeros(local_pop, device=dev)
                            sum_sq_ret = torch.zeros(local_pop, device=dev)
                            effective_samples = 0
                            running_cum_ret = torch.zeros(local_pop, device=dev)
                            running_peak = torch.zeros(local_pop, device=dev)
                            max_dd = torch.zeros(local_pop, device=dev)
                            month_returns = None
                            if month_ids is not None and self.month_count > 0:
                                month_returns = torch.zeros((local_pop, self.month_count), device=dev)

                            chunk_size = max(1, (n_samples - 1) // 10)
                            profitable_chunks = torch.zeros(local_pop, device=dev)
                            current_chunk_ret = torch.zeros(local_pop, device=dev)

                            for start_idx in range(0, n_samples, batch_size):
                                end_idx = min(start_idx + batch_size, n_samples)
                                if preload:
                                    data_slice = gpu_data_cubes[gpu_idx][:, start_idx:end_idx, :]
                                    ohlc_slice = gpu_ohlc_cubes[gpu_idx][:, start_idx:end_idx, :]
                                else:
                                    data_slice = self.data_cube_cpu[:, start_idx:end_idx, :].to(dev, non_blocking=True)
                                    ohlc_slice = self.ohlc_cube_cpu[:, start_idx:end_idx, :].to(dev, non_blocking=True)

                                signals = torch.zeros((local_pop, end_idx - start_idx), device=dev, dtype=torch.float16)
                                for i in range(tf_count):
                                    with torch.amp.autocast("cuda", enabled=True):
                                        tf_sig = torch.matmul(data_slice[i], logic_weights.t())
                                    signals = signals + (tf_weights[:, i].unsqueeze(1) * tf_sig.t())

                                if has_news:
                                    if preload:
                                        news_slice = gpu_news[gpu_idx][start_idx:end_idx, 0]
                                    else:
                                        news_slice = self.news_tensor_cpu[start_idx:end_idx, 0].to(dev, non_blocking=True)
                                    signals = signals + (news_slice.unsqueeze(0) * 0.5)

                    actions = torch.where(signals > thresholds[:, 0].unsqueeze(1), 1.0, 0.0)
                    actions = torch.where(signals < thresholds[:, 1].unsqueeze(1), -1.0, actions)
                    
                    # --- REALITY CHECK FIX (2025 Standard) ---
                    # Old Logic: Close[t+1] - Close[t] (Captured gaps, unrealistically optimistic)
                    # New Logic: Close[t+1] - Open[t+1] (Assumes entry at Open, no gap capture)
                    # This removes "Look-ahead Bias" where the bot 'sees' the gap before it happens.
                    
                    close_next = gpu_ohlc_cubes[gpu_idx][0, start_idx:end_idx, 3][1:]
                    open_next = gpu_ohlc_cubes[gpu_idx][0, start_idx:end_idx, 0][1:]
                    
                    # Align actions: Action[t] results in PnL[t+1]
                    # We lose the last action because we don't have t+1 data for it
                    actions_aligned = actions[:, :-1]
                    
                    # Raw price return of the candle we effectively held
                    rets = (close_next - open_next) / (open_next + 1e-9)
                    
                    # Apply actions to returns
                    batch_rets = actions_aligned * rets.unsqueeze(0)
                    
                    # Slippage/Spread Simulation
                    # We pay 'spread cost' on every non-zero action because we re-enter at Open
                    # Cost estimate: 0.0001 (1 pip) roughly relative to price
                    # Better: dynamic cost if spread provided, but fixed cost is safer for training
                    spread_cost = 0.0001 * torch.abs(actions_aligned)
                    batch_rets = batch_rets - spread_cost
                    
                    sum_ret = sum_ret + batch_rets.sum(dim=1)
                                sum_sq_ret = sum_sq_ret + (batch_rets**2).sum(dim=1)
                                effective_samples += batch_rets.shape[1]
                                current_chunk_ret = current_chunk_ret + batch_rets.sum(dim=1)

                                if (end_idx // chunk_size) > (start_idx // chunk_size):
                                    profitable_chunks = profitable_chunks + (current_chunk_ret > 0).float()
                                    current_chunk_ret = torch.zeros_like(current_chunk_ret)

                                batch_cum = batch_rets.cumsum(dim=1)
                                abs_cum = batch_cum + running_cum_ret.unsqueeze(1)
                                batch_peaks = torch.maximum(
                                    torch.cummax(abs_cum, dim=1)[0],
                                    running_peak.unsqueeze(1),
                                )
                                max_dd = torch.maximum((batch_peaks - abs_cum).max(dim=1)[0], max_dd)

                                running_cum_ret = abs_cum[:, -1]
                                running_peak = batch_peaks[:, -1]

                                del signals, actions, rets, batch_rets, batch_cum, abs_cum, batch_peaks, data_slice, ohlc_slice

                            torch.cuda.empty_cache()

                            if effective_samples <= 0:
                                return torch.full((local_pop,), -1e9)
                            mean_ret = sum_ret / effective_samples
                            std_ret = torch.sqrt(torch.clamp((sum_sq_ret / effective_samples) - mean_ret**2, min=1e-9)) + 1e-6
                            sharpe = (mean_ret / std_ret) * (252 * 1440) ** 0.5

                            fitness = sharpe * (profitable_chunks / 10.0)
                            fitness = torch.where(max_dd > 0.07, torch.tensor(-1e9, device=dev), fitness)
                            fitness = torch.where(std_ret * (1440**0.5) > 0.04, fitness * 0.1, fitness)

                            consistency = None
                            pass_rate = None
                            median_month = None
                            if month_returns is not None:
                                negative_months = (month_returns < 0).sum(dim=1).float()
                                consistency = torch.clamp(1.0 - (negative_months * 0.1), min=0.0)
                                fitness = fitness * consistency
                                pass_rate = (month_returns >= monthly_floor).float().mean(dim=1)
                                if monthly_pass_power != 1.0:
                                    pass_rate = torch.clamp(pass_rate, min=0.0) ** monthly_pass_power
                                fitness = fitness * pass_rate
                                median_month = month_returns.median(dim=1).values
                                if monthly_hard:
                                    fitness = torch.where(median_month < monthly_floor, torch.tensor(-1e9, device=dev), fitness)
                                else:
                                    fitness = torch.where(median_month < monthly_floor, fitness * 0.1, fitness)

                            if debug_lock is not None:
                                try:
                                    best_idx = int(torch.argmax(fitness).item())
                                    best_fit = float(fitness[best_idx].item())
                                    best_sharpe = float(sharpe[best_idx].item())
                                    best_mean = float(mean_ret[best_idx].item())
                                    best_std = float(std_ret[best_idx].item())
                                    best_dd = float(max_dd[best_idx].item())
                                    best_chunks = float(profitable_chunks[best_idx].item())
                                    best_consistency = float(consistency[best_idx].item()) if consistency is not None else 1.0
                                    best_pass_rate = float(pass_rate[best_idx].item()) if pass_rate is not None else 1.0
                                    best_median_month = float(median_month[best_idx].item()) if median_month is not None else 0.0
                                    with debug_lock:
                                        if debug_snapshot["fitness"] is None or best_fit > debug_snapshot["fitness"]:
                                            debug_snapshot["fitness"] = best_fit
                                            debug_snapshot["sharpe"] = best_sharpe
                                            debug_snapshot["mean_ret"] = best_mean
                                            debug_snapshot["std_ret"] = best_std
                                            debug_snapshot["max_dd"] = best_dd
                                            debug_snapshot["profitable_chunks"] = best_chunks
                                            debug_snapshot["consistency"] = best_consistency
                                            debug_snapshot["samples"] = int(effective_samples)
                                            debug_snapshot["pass_rate"] = best_pass_rate
                                            debug_snapshot["median_month"] = best_median_month
                                except Exception:
                                    pass

                            fitness_all.append(fitness.detach().cpu())

                        return torch.cat(fitness_all, dim=0)
                except Exception as e:
                    logger.error(f"GPU {gpu_idx} eval failed: {e}", exc_info=True)
                    return torch.full((chunk.shape[0],), -1e9)

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = {
                    executor.submit(_eval_on_gpu, pop_chunks[i], i): pop_chunks[i].shape[0]
                    for i in range(num_gpus)
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        all_fitness_scores.append(future.result())
                    except Exception as e:
                        chunk_size = futures.get(future, 0)
                        logger.error(f"GPU worker failed: {e}", exc_info=True)
                        all_fitness_scores.append(torch.full((chunk_size,), -1e9))
            
            # Re-assemble and return to EvoTorch master
            return torch.cat(all_fitness_scores, dim=0)

        # 2. Start Centralized Evolution (Master)
        genome_dim = self.n_features + len(self.timeframes) + 2
        problem = Problem("max", fitness_func, initial_bounds=(-1.0, 1.0), 
                         solution_length=genome_dim, device="cpu", vectorized=True)
        
        # Tournament of 50,000 Default for High-End Clusters
        pop_size = int(os.environ.get("FOREX_BOT_DISCOVERY_POPULATION", 30000))
        searcher = CMAES(problem, popsize=pop_size, stdev_init=0.5)

        logger.info(f"Starting COOPERATIVE {len(self.gpu_list)}-GPU Discovery (Pop: {pop_size})...")
        plateau_gens = int(os.environ.get("FOREX_BOT_DISCOVERY_PLATEAU", "10000") or 10000)
        score_target = float(os.environ.get("FOREX_BOT_DISCOVERY_SCORE_TARGET", "0") or 0.0)
        ckpt_every = int(os.environ.get("FOREX_BOT_DISCOVERY_CHECKPOINT_EVERY", "10") or 10)
        ckpt_dir = Path(
            os.environ.get(
                "FOREX_BOT_DISCOVERY_CHECKPOINT_DIR",
                os.environ.get("FOREX_BOT_CACHE_DIR", "cache"),
            )
        ) / "discovery"
        ckpt_path = ckpt_dir / "tensor_discovery_ckpt.pt"

        def _truthy(value: str | None, default: bool) -> bool:
            if value is None:
                return default
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        best_val = -1e9
        last_improve = 0
        start_gen = 0
        resume_flag = os.environ.get("FOREX_BOT_DISCOVERY_RESUME")
        resume_enabled = _truthy(resume_flag, default=True)
        if resume_enabled and ckpt_path.exists():
            try:
                ckpt = None
                try:
                    from torch.serialization import safe_globals
                    from evotorch.tools.readonlytensor import ReadOnlyTensor
                    with safe_globals([ReadOnlyTensor]):
                        ckpt = torch.load(ckpt_path, map_location="cpu")
                except Exception:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                if ckpt.get("genome_dim") == genome_dim and ckpt.get("n_features") == self.n_features:
                    if "state" in ckpt and hasattr(searcher, "load_state_dict"):
                        searcher.load_state_dict(ckpt["state"])
                    elif "population" in ckpt and hasattr(searcher, "population"):
                        try:
                            searcher.population.values = ckpt["population"]
                            if "evaluations" in ckpt and hasattr(searcher.population, "evaluations"):
                                searcher.population.evaluations = ckpt["evaluations"]
                        except Exception as exc:
                            logger.warning(f"Discovery resume population load failed: {exc}")
                    start_gen = int(ckpt.get("generation", 0)) + 1
                    best_val = float(ckpt.get("best_val", best_val))
                    last_improve = int(ckpt.get("last_improve", last_improve))
                    logger.info(f"Resuming discovery from generation {start_gen}.")
                else:
                    logger.warning("Discovery checkpoint incompatible with current genome; starting fresh.")
            except Exception as exc:
                logger.warning(f"Discovery resume failed; starting fresh: {exc}")

        def _save_checkpoint(gen_idx: int) -> None:
            try:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                payload = {
                    "generation": int(gen_idx),
                    "best_val": float(best_val),
                    "last_improve": int(last_improve),
                    "genome_dim": int(genome_dim),
                    "n_features": int(self.n_features),
                    "time": float(time.time()),
                }
                if hasattr(searcher, "state_dict"):
                    payload["state"] = searcher.state_dict()
                elif hasattr(searcher, "population"):
                    payload["population"] = searcher.population.values.detach().cpu()
                    if hasattr(searcher.population, "evaluations"):
                        payload["evaluations"] = searcher.population.evaluations.detach().cpu()
                torch.save(payload, ckpt_path)
            except Exception as exc:
                logger.warning(f"Discovery checkpoint save failed: {exc}")

        if start_gen >= iterations:
            logger.info("Discovery resume indicates iterations already complete; skipping search.")
            return

        for i in range(start_gen, iterations):
            searcher.step()
            if i % 10 == 0:
                current_best = searcher.status.get("best_eval", 0.0)
                if current_best > best_val + 1e-9:
                    best_val = float(current_best)
                    last_improve = i
                logger.info(f"Generation {i}: Global Best Fitness = {float(current_best):.4f}")
                if debug_lock is not None:
                    with debug_lock:
                        snap = dict(debug_snapshot)
                    if snap.get("fitness") is not None:
                        logger.info(
                            "Gen %d metrics: best_fit=%.4f sharpe=%.4f mean_ret=%.6f std_ret=%.6f max_dd=%.4f "
                            "prof_chunks=%.1f consistency=%.3f pass_rate=%.3f median_month=%.4f samples=%s",
                            i,
                            snap["fitness"],
                            snap["sharpe"],
                            snap["mean_ret"],
                            snap["std_ret"],
                            snap["max_dd"],
                            snap["profitable_chunks"],
                            snap["consistency"],
                            snap.get("pass_rate", 1.0),
                            snap.get("median_month", 0.0),
                            snap["samples"],
                        )
                if plateau_gens > 0 and (i - last_improve) >= plateau_gens and best_val >= score_target:
                    logger.info(
                        f"Plateau reached ({plateau_gens} gens, best={best_val:.4f} >= {score_target}). Stopping early."
                    )
                    if ckpt_every > 0:
                        _save_checkpoint(i)
                    break
            if ckpt_every > 0 and (i % ckpt_every == 0) and i > start_gen:
                _save_checkpoint(i)

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

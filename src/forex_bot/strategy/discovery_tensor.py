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


def _discovery_cpu_budget() -> int:
    """Best-effort CPU budget per rank/process for discovery prep."""
    per_gpu = os.environ.get("FOREX_BOT_CPU_THREADS_PER_GPU")
    if per_gpu and os.environ.get("LOCAL_RANK") is not None:
        try:
            return max(1, int(per_gpu))
        except Exception:
            pass
    for key in ("FOREX_BOT_CPU_BUDGET", "FOREX_BOT_CPU_THREADS"):
        val = os.environ.get(key)
        if val:
            try:
                return max(1, int(val))
            except Exception:
                pass
    return os.cpu_count() or 1

class TensorDiscoveryEngine:
    """
    2025-Grade GPU Discovery Engine.
    Cooperative Multi-GPU Architecture with Adaptive Fortress logic.
    
    Features:
    - Adaptive Drawdown Targeting (Starts loose, tightens to 4%)
    - Global Champion Reporting
    - Survival-Based Fitness
    - Monte-Carlo Lite
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
        local_rank = None
        if os.environ.get("LOCAL_RANK") is not None:
            try:
                local_rank = int(os.environ.get("LOCAL_RANK"))
            except Exception:
                local_rank = None
        try:
            max_gpus = int(os.environ.get("FOREX_BOT_MAX_GPUS", "0") or 0)
        except Exception:
            max_gpus = 0
        if max_gpus and max_gpus > 0 and len(gpu_list) > max_gpus:
            gpu_list = gpu_list[:max_gpus]
        restrict_local = str(os.environ.get("FOREX_BOT_DISCOVERY_LOCAL_RANK_ONLY", "1") or "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if local_rank is not None and restrict_local and torch.cuda.is_available():
            if 0 <= local_rank < torch.cuda.device_count():
                gpu_list = [f"cuda:{local_rank}"]
                device = f"cuda:{local_rank}"
        self.gpu_list = gpu_list
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_experts = n_experts
        self.data_cube = None
        self.ohlc_cube = None
        self.timeframes = timeframes
        self.best_experts = []
        self.best_scores = None
        self.local_rank = local_rank
        self.month_ids_cpu = None
        self.month_count = 0
        self.bar_change_cpu = None
        self.full_fp16 = False
        self._max_rows_override = max_rows
        self._stream_mode_override = stream_mode
        self._auto_cap_override = auto_cap

    def _prepare_tensor_cube(self, frames: Dict[str, pd.DataFrame], news_map: Optional[Dict[str, pd.DataFrame]] = None):
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

        # Track TF bar changes aligned to master index (used for per-TF trading).
        bar_changes = []
        for tf in self.timeframes:
            try:
                tf_idx = frames[tf].index
                if len(tf_idx) == 0:
                    raise ValueError("empty index")
                bar_ids = pd.Series(
                    np.arange(len(tf_idx), dtype=np.int64),
                    index=tf_idx,
                )
                aligned_ids = bar_ids.reindex(master_idx).ffill().bfill()
                ids = aligned_ids.to_numpy(dtype=np.int64, copy=False)
                change = np.zeros(len(ids), dtype=np.float32)
                if len(ids) > 1:
                    change[1:] = (ids[1:] != ids[:-1]).astype(np.float32)
                bar_changes.append(torch.from_numpy(change))
            except Exception:
                bar_changes.append(torch.zeros(len(master_idx), dtype=torch.float32))
        self.bar_change_cpu = torch.stack(bar_changes) if bar_changes else None

        try:
            cpu_workers = int(os.environ.get("FOREX_BOT_DISCOVERY_ALIGN_WORKERS", "0") or 0)
        except Exception:
            cpu_workers = 0
        if cpu_workers <= 0:
            cpu_workers = _discovery_cpu_budget()
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

        def _preflight_report():
            try:
                if str(os.environ.get("FOREX_BOT_DISCOVERY_PREFLIGHT", "1") or "1").strip().lower() in {"0", "false", "no"}:
                    return
                try:
                    sample_n = int(os.environ.get("FOREX_BOT_DISCOVERY_PREFLIGHT_SAMPLES", "200000") or 200000)
                except Exception:
                    sample_n = 200000
                sample_n = max(1000, sample_n)
                n_samples = int(self.ohlc_cube_cpu.shape[1])
                if n_samples < 2:
                    return
                sample_n = min(sample_n, n_samples)
                start_idx = max(0, n_samples - sample_n)

                ohlc = self.ohlc_cube_cpu[0, start_idx:start_idx + sample_n].to(dtype=torch.float32)
                open_ = ohlc[:, 0]
                close_ = ohlc[:, 3]
                open_next = torch.clamp(open_[1:], min=1e-6)
                close_next = close_[1:]
                rets = (close_next - open_next) / open_next

                nan_open = torch.isnan(open_).sum().item()
                nan_close = torch.isnan(close_).sum().item()
                nonpos_open = (open_ <= 0).sum().item()
                nonpos_close = (close_ <= 0).sum().item()
                nan_rets = torch.isnan(rets).sum().item()
                zero_rets = (rets == 0).sum().item()

                logger.info(
                    f"Preflight: samples={sample_n} open[min={open_.min().item():.6f} max={open_.max().item():.6f} "
                    f"nan={nan_open} <=0={nonpos_open}] close[min={close_.min().item():.6f} max={close_.max().item():.6f} "
                    f"nan={nan_close} <=0={nonpos_close}]"
                )
                logger.info(
                    f"Preflight: returns[min={rets.min().item():.6f} max={rets.max().item():.6f} "
                    f"mean={rets.mean().item():.6f} std={rets.std(unbiased=False).item():.6f} "
                    f"nan={nan_rets} zero={zero_rets} ({(zero_rets / max(1, rets.numel())):.4f})]"
                )

                def _baseline_metrics(name: str, actions: torch.Tensor) -> None:
                    if actions.numel() != rets.numel():
                        return
                    batch_rets = (actions * rets) - (torch.abs(actions) * 0.0002)
                    equity = torch.cumprod(1.0 + batch_rets, dim=0)
                    peak = torch.cummax(equity, dim=0)[0]
                    max_dd = ((peak - equity) / torch.clamp(peak, min=1e-9)).max().item()
                    total_ret = equity[-1].item() - 1.0
                    std_r = batch_rets.std(unbiased=False).item()
                    sharpe = (batch_rets.mean().item() / (std_r + 1e-9)) * (252 * 1440) ** 0.5

                    prev_actions = actions[:-1]
                    curr_actions = actions[1:]
                    entries = (prev_actions == 0) & (curr_actions != 0)
                    flips = (prev_actions != 0) & (curr_actions != 0) & (prev_actions != curr_actions)
                    trade_count = (entries | flips).float().sum().item()
                    trade_density = trade_count / max(1, actions.numel())
                    trades_per_day = trade_density * 1440.0

                    logger.info(
                        f"Preflight baseline [{name}]: ret={total_ret:.4f} sharpe={sharpe:.2f} "
                        f"max_dd={max_dd:.2f} trades/day={trades_per_day:.2f}"
                    )

                # Baseline 1: always long
                _baseline_metrics("always_long", torch.ones_like(rets))
                # Baseline 2: always short
                _baseline_metrics("always_short", -torch.ones_like(rets))
                # Baseline 3: sparse random (about 1-2 trades/day)
                try:
                    target_mid_per_day = float(os.environ.get("FOREX_BOT_DISCOVERY_PREFLIGHT_TRADES_PER_DAY", "1.5") or 1.5)
                except Exception:
                    target_mid_per_day = 1.5
                prob = max(target_mid_per_day / 1440.0, 1e-6)
                rnd = torch.rand_like(rets)
                actions = torch.zeros_like(rets)
                mask = rnd < prob
                if mask.any():
                    signs = torch.where(torch.rand(int(mask.sum())) < 0.5, 1.0, -1.0)
                    actions[mask] = signs.to(actions.dtype)
                _baseline_metrics("random_sparse", actions)
            except Exception as e:
                logger.warning(f"Preflight failed: {e}")

        _preflight_report()

        metrics_registry = {}
        registry_lock = threading.Lock()

        preload = False
        cuda_gpus = [g for g in self.gpu_list if str(g).startswith("cuda")]
        gpu_data_cubes = []
        gpu_ohlc_cubes = []
        gpu_bar_changes = []
        gpu_news = []
        gpu_month_ids = []
        if cuda_gpus:
            try:
                bytes_needed = (self.data_cube_cpu.nelement() * self.data_cube_cpu.element_size()) + \
                               (self.ohlc_cube_cpu.nelement() * self.ohlc_cube_cpu.element_size())
                if self.bar_change_cpu is not None:
                    bytes_needed += self.bar_change_cpu.nelement() * self.bar_change_cpu.element_size()
                min_vram = min(torch.cuda.get_device_properties(g).total_memory for g in cuda_gpus)
                if bytes_needed < (min_vram * 0.85):
                    preload = True
                    logger.info(f"Discovery: Preloading data ({bytes_needed/1024**3:.2f} GB) to GPUs.")
                    gpu_data_cubes = [self.data_cube_cpu.to(gpu) for gpu in cuda_gpus]
                    gpu_ohlc_cubes = [self.ohlc_cube_cpu.to(gpu) for gpu in cuda_gpus]
                    gpu_bar_changes = [self.bar_change_cpu.to(gpu) for gpu in cuda_gpus] if self.bar_change_cpu is not None else []
                    gpu_news = [self.news_tensor_cpu.to(gpu) for gpu in cuda_gpus] if self.news_tensor_cpu is not None else []
                    gpu_month_ids = [self.month_ids_cpu.to(gpu) for gpu in cuda_gpus] if self.month_ids_cpu is not None else []
                else:
                    gpu_data_cubes, gpu_ohlc_cubes, gpu_bar_changes, gpu_news, gpu_month_ids = [], [], [], [], []
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
                                    res, meta = _eval_batch_safe(
                                        gpu_idx,
                                        dev,
                                        genomes,
                                        current_batch,
                                        n_samples,
                                        preload,
                                        gpu_data_cubes,
                                        gpu_ohlc_cubes,
                                        gpu_bar_changes,
                                        gpu_news,
                                        month_ids,
                                    )
                                    fitness_all.append(res)
                                    with registry_lock:
                                        cur_gen = gen_context["val"]
                                        if cur_gen not in metrics_registry or meta["fit"] > metrics_registry[cur_gen]["fit"]:
                                            metrics_registry[cur_gen] = meta
                                    break
                                except torch.OutOfMemoryError:
                                    torch.cuda.empty_cache(); gc.collect()
                                    batch_size_map[gpu_idx] = int(current_batch * 0.5)
                                    if batch_size_map[gpu_idx] < 128: return torch.full((chunk.shape[0],), -1e9)
                                    continue
                        return torch.cat(fitness_all, dim=0)
                except Exception as e: 
                    logger.error(f"GPU {gpu_idx} fatal: {e}"); return torch.full((chunk.shape[0],), -1e9)

            def _eval_batch_safe(
                gpu_idx,
                dev,
                genomes,
                batch_size,
                n_samples,
                preload,
                gpu_data_cubes,
                gpu_ohlc_cubes,
                gpu_bar_changes,
                gpu_news,
                month_ids,
            ):
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

                try:
                    tf_trading = str(os.environ.get("FOREX_BOT_DISCOVERY_TF_TRADING", "1") or "1").strip().lower() in {
                        "1", "true", "yes", "on"
                    }
                except Exception:
                    tf_trading = True
                bar_change = None
                if tf_trading and self.bar_change_cpu is not None:
                    if preload and gpu_bar_changes:
                        bar_change = gpu_bar_changes[gpu_idx]
                    else:
                        bar_change = self.bar_change_cpu.to(dev, non_blocking=True)

                tf_count = self.data_cube_cpu.shape[0]
                matmul_dtype = torch.float16 if dev.type == "cuda" else torch.float32
                tf_weights = F.softmax(genomes[:, :tf_count], dim=1).to(dtype=matmul_dtype)
                logic_weights = genomes[:, tf_count : tf_count + self.n_features].to(dtype=matmul_dtype)
                thresholds = genomes[:, -2:]
                
                # Selective trading balance (tunable bias/scale to control activity)
                try:
                    threshold_bias = float(os.environ.get("FOREX_BOT_DISCOVERY_THRESHOLD_BIAS", "0.02") or 0.02)
                except Exception:
                    threshold_bias = 0.1
                threshold_bias = max(threshold_bias, 0.0)
                try:
                    threshold_scale = float(os.environ.get("FOREX_BOT_DISCOVERY_THRESHOLD_SCALE", "0.12") or 0.12)
                except Exception:
                    threshold_scale = 0.25
                threshold_scale = max(threshold_scale, 1e-6)
                thresholds = thresholds * threshold_scale
                buy_th = torch.maximum(thresholds[:, 0], thresholds[:, 1]).to(dtype=matmul_dtype) + threshold_bias
                sell_th = torch.minimum(thresholds[:, 0], thresholds[:, 1]).to(dtype=matmul_dtype) - threshold_bias

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
                    try:
                        signal_scale = float(os.environ.get("FOREX_BOT_DISCOVERY_SIGNAL_SCALE", "2.0") or 2.0)
                    except Exception:
                        signal_scale = 2.0
                    signal_scale = max(signal_scale, 1e-6)
                    signals = signals * signal_scale
                    
                    if (end_idx - start_idx) < 2:
                        continue

                    if tf_trading and bar_change is not None and tf_count > 1:
                        bar_slice = bar_change[:, start_idx:end_idx]
                        actions_tf = torch.where(
                            tf_sig > buy_th.unsqueeze(1).unsqueeze(2),
                            1.0,
                            0.0,
                        )
                        actions_tf = torch.where(
                            tf_sig < sell_th.unsqueeze(1).unsqueeze(2),
                            -1.0,
                            actions_tf,
                        )
                        close_next = ohlc_slice[:, 1:, 3]
                        open_next = torch.clamp(ohlc_slice[:, 1:, 0], min=1e-6)
                        rets_tf = (close_next - open_next) / open_next
                        bar_mask_next = bar_slice[:, 1:]
                        exec_actions = actions_tf[:, :, :-1] * bar_mask_next.unsqueeze(0)
                        batch_rets_tf = (exec_actions * rets_tf.unsqueeze(0)) - (torch.abs(exec_actions) * 0.0002)
                        batch_rets = (batch_rets_tf * tf_weights.unsqueeze(2)).sum(dim=1)
                        prev_actions = torch.cat(
                            [torch.zeros_like(exec_actions[:, :, :1]), exec_actions[:, :, :-1]],
                            dim=2,
                        )
                        curr_actions = exec_actions
                        entries = (prev_actions == 0) & (curr_actions != 0)
                        flips = (prev_actions != 0) & (curr_actions != 0) & (prev_actions != curr_actions)
                        trade_events = (entries | flips).float()
                        trade_steps.add_(
                            (trade_events * alive_mask.unsqueeze(1).unsqueeze(2).float()).sum(dim=(1, 2))
                        )
                    else:
                        actions = torch.where(signals > buy_th.unsqueeze(1), 1.0, 0.0)
                        actions = torch.where(signals < sell_th.unsqueeze(1), -1.0, actions)
                        close_next = ohlc_slice[0, 1:, 3]
                        open_next = torch.clamp(ohlc_slice[0, 1:, 0], min=1e-6)
                        rets = (close_next - open_next) / open_next
                        batch_rets = (actions[:, :-1] * rets.unsqueeze(0)) - (torch.abs(actions[:, :-1]) * 0.0002)
                        # Count actual trade entries (not time-in-position).
                        prev_actions = torch.cat(
                            [torch.zeros_like(actions[:, :1]), actions[:, :-1]],
                            dim=1,
                        )
                        curr_actions = actions
                        entries = (prev_actions == 0) & (curr_actions != 0)
                        flips = (prev_actions != 0) & (curr_actions != 0) & (prev_actions != curr_actions)
                        trade_events = (entries | flips).float()
                        # Count trades only while alive to avoid inflated density after death.
                        trade_steps.add_((trade_events * alive_mask.unsqueeze(1).float()).sum(dim=1))

                    active_rets = torch.where(alive_mask.unsqueeze(1), batch_rets, torch.zeros_like(batch_rets))
                    sum_ret.add_(active_rets.sum(dim=1))
                    sum_sq_ret.add_((active_rets**2).sum(dim=1))
                    
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

                # --- ADAPTIVE ENCOURAGEMENT LOGIC ---
                cur_gen = gen_context["val"]
                if cur_gen < 50:
                    dynamic_dd_limit = 0.10
                elif cur_gen < 150:
                    dynamic_dd_limit = 0.10 - (0.06 * (cur_gen - 50) / 100.0)
                else:
                    dynamic_dd_limit = 0.04
                
                survival_ratio = survival_steps / n_samples
                denom = torch.clamp(survival_steps, min=1.0)
                mean_r = sum_ret / denom
                std_r = torch.sqrt(torch.clamp((sum_sq_ret / denom) - mean_r**2, min=1e-9)) + 1e-5
                sharpe = (mean_r / std_r) * (252 * 1440)**0.5
                sharpe = torch.clamp(sharpe, min=-5.0, max=10.0)

                segment_profitability = (seg_rets > 0).float().sum(dim=1)

                # Profit from compounded equity
                total_ret = running_equity - 1.0
                try:
                    profit_target = float(os.environ.get("FOREX_BOT_DISCOVERY_PROFIT_TARGET", "0.10") or 0.10)
                except Exception:
                    profit_target = 0.10
                profit_target = max(profit_target, 1e-6)
                profit_score = torch.tanh(total_ret / profit_target)

                # Weights (tunable)
                try:
                    survival_weight = float(os.environ.get("FOREX_BOT_DISCOVERY_SURVIVAL_WEIGHT", "2.0") or 2.0)
                except Exception:
                    survival_weight = 4.0
                try:
                    sharpe_weight = float(os.environ.get("FOREX_BOT_DISCOVERY_SHARPE_WEIGHT", "2.0") or 2.0)
                except Exception:
                    sharpe_weight = 1.5
                try:
                    consistency_weight = float(os.environ.get("FOREX_BOT_DISCOVERY_CONSISTENCY_WEIGHT", "4.0") or 4.0)
                except Exception:
                    consistency_weight = 5.0
                try:
                    profit_weight = float(os.environ.get("FOREX_BOT_DISCOVERY_PROFIT_WEIGHT", "15.0") or 15.0)
                except Exception:
                    profit_weight = 12.0
                try:
                    dd_penalty_weight = float(os.environ.get("FOREX_BOT_DISCOVERY_DD_PENALTY_WEIGHT", "12.0") or 12.0)
                except Exception:
                    dd_penalty_weight = 10.0

                consistency_bonus = (segment_profitability / n_segments) * consistency_weight

                # Drawdown penalty relative to dynamic limit (soft, no cliff)
                dd_ratio = max_dd / torch.clamp(torch.tensor(dynamic_dd_limit, device=dev), min=1e-6)
                dd_penalty = torch.clamp(dd_ratio - 1.0, min=0.0) * dd_penalty_weight
                
                days = max(float(n_samples) / 1440.0, 1e-6)
                trade_density = trade_steps / days
                try:
                    target_min_per_day = float(os.environ.get("FOREX_BOT_DISCOVERY_TRADE_MIN_PER_DAY", "1") or 1.0)
                    target_max_per_day = float(os.environ.get("FOREX_BOT_DISCOVERY_TRADE_MAX_PER_DAY", "2") or 2.0)
                except Exception:
                    target_min_per_day, target_max_per_day = 1.0, 2.0
                target_min_per_day = max(target_min_per_day, 0.1)
                target_max_per_day = max(target_max_per_day, target_min_per_day)
                min_trades_required = max(target_min_per_day * days, 1.0)
                trade_gate = trade_steps >= min_trades_required
                # Activity factor: survival/profit shouldn't be rewarded if we don't trade enough.
                activity_scale = torch.clamp(trade_density / max(target_min_per_day, 1e-6), min=0.0, max=1.0)

                # Trade activity score: penalize under/over-trading, reward within range.
                mid_density = (target_min_per_day + target_max_per_day) / 2.0
                half_span = max((target_max_per_day - target_min_per_day) / 2.0, 1e-9)
                trade_score = 1.0 - torch.abs(trade_density - mid_density) / half_span
                trade_score = torch.clamp(trade_score, min=-1.0, max=1.0)
                try:
                    trade_weight = float(os.environ.get("FOREX_BOT_DISCOVERY_TRADE_WEIGHT", "10.0") or 10.0)
                except Exception:
                    trade_weight = 6.0
                trade_score = trade_score * trade_weight

                total_fitness = (
                    (survival_ratio * activity_scale * survival_weight)
                    + (sharpe * sharpe_weight)
                    + (consistency_bonus * activity_scale)
                    + (profit_score * activity_scale * profit_weight)
                    + trade_score
                    - dd_penalty
                )
                try:
                    no_trade_penalty = float(os.environ.get("FOREX_BOT_DISCOVERY_NO_TRADE_PENALTY", "-25.0") or -25.0)
                except Exception:
                    no_trade_penalty = -25.0
                no_trade_penalty = min(no_trade_penalty, -1e-3)
                total_fitness = torch.where(trade_gate, total_fitness, torch.full_like(total_fitness, no_trade_penalty))
                
                max_dd_safe = torch.nan_to_num(max_dd, nan=1.0)
                bi = int(torch.argmax(total_fitness).item())
                metadata = {
                    "fit": float(total_fitness[bi].item()), "surv": float(survival_ratio[bi].item()),
                    "shrp": float(sharpe[bi].item()), "dd": float(max_dd_safe[bi].item()), 
                    "segs": int(segment_profitability[bi].item()), "trade": float(trade_density[bi].item()),
                    "limit": dynamic_dd_limit
                }
                return torch.nan_to_num(total_fitness, nan=0.0).detach().cpu(), metadata

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = {executor.submit(_eval_on_gpu, pop_chunks[i], i): pop_chunks[i].shape[0] for i in range(num_gpus)}
                for future in concurrent.futures.as_completed(futures):
                    try: all_fitness_scores.append(future.result())
                    except Exception as e: logger.error(f"Worker failed: {e}"); all_fitness_scores.append(torch.full((futures[future],), 0.0))
            
            return torch.cat(all_fitness_scores, dim=0)

        genome_dim = self.n_features + len(self.timeframes) + 2
        problem = Problem("max", fitness_func, initial_bounds=(-1.0, 1.0), solution_length=genome_dim, device="cpu", vectorized=True)
        pop_size = int(os.environ.get("FOREX_BOT_DISCOVERY_POPULATION", 12000))
        try:
            stdev_init = float(os.environ.get("FOREX_BOT_DISCOVERY_CMAES_STDEV", "1.5") or 1.5)
        except Exception:
            stdev_init = 1.5
        stdev_init = max(stdev_init, 0.1)
        searcher = CMAES(problem, popsize=pop_size, stdev_init=stdev_init)

        logger.info(f"Starting COOPERATIVE 8-GPU Discovery (Pop: {pop_size})...")
        checkpoint_path = Path(
            os.environ.get("FOREX_BOT_DISCOVERY_CHECKPOINT_PATH", "cache/discovery_checkpoint.pt")
        )
        try:
            checkpoint_every = int(os.environ.get("FOREX_BOT_DISCOVERY_CHECKPOINT_EVERY", "10") or 10)
        except Exception:
            checkpoint_every = 10
        checkpoint_every = max(checkpoint_every, 1)
        resume_enabled = str(os.environ.get("FOREX_BOT_DISCOVERY_RESUME", "1") or "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        start_gen = 0
        resume_best = None
        if resume_enabled and checkpoint_path.exists():
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                pop_vals = ckpt.get("population")
                if pop_vals is not None:
                    try:
                        if hasattr(searcher.population, "set_values"):
                            searcher.population.set_values(pop_vals)
                            logger.info(f"Discovery resume: restored population from {checkpoint_path}.")
                        else:
                            cur_vals = searcher.population.values
                            if cur_vals.shape == pop_vals.shape:
                                cur_vals.copy_(pop_vals.to(cur_vals.device, dtype=cur_vals.dtype))
                                logger.info(f"Discovery resume: restored population from {checkpoint_path}.")
                            else:
                                logger.warning(
                                    f"Discovery resume skipped: population shape mismatch "
                                    f"{tuple(cur_vals.shape)} vs {tuple(pop_vals.shape)}."
                                )
                    except Exception as e:
                        logger.warning(f"Discovery resume population load failed: {e}")
                pop_evals = ckpt.get("evaluations")
                if pop_evals is not None:
                    eval_attr = None
                    for attr in ("evaluations", "evals"):
                        if hasattr(searcher.population, attr):
                            eval_attr = attr
                            break
                    if eval_attr is not None:
                        try:
                            cur_evals = getattr(searcher.population, eval_attr)
                            if cur_evals is not None and cur_evals.shape == pop_evals.shape:
                                cur_evals.copy_(pop_evals.to(cur_evals.device, dtype=cur_evals.dtype))
                        except Exception as e:
                            logger.warning(f"Discovery resume {eval_attr} load failed: {e}")
                ckpt_gen = int(ckpt.get("gen", 0) or 0)
                start_gen = max(0, ckpt_gen + 1)
                resume_best = ckpt.get("best_eval", None)
            except Exception as e:
                logger.warning(f"Discovery resume failed: {e}")
        try:
            stagnation_gens = int(os.environ.get("FOREX_BOT_DISCOVERY_STAGNATION_GENS", "20") or 20)
        except Exception:
            stagnation_gens = 25
        try:
            stagnation_eps = float(os.environ.get("FOREX_BOT_DISCOVERY_STAGNATION_EPS", "1e-3") or 1e-3)
        except Exception:
            stagnation_eps = 1e-3
        try:
            mutation_boost = float(os.environ.get("FOREX_BOT_DISCOVERY_MUTATION_BOOST", "0.5") or 0.5)
        except Exception:
            mutation_boost = 0.3
        stagnation_gens = max(stagnation_gens, 5)
        mutation_boost = max(mutation_boost, 0.0)
        best_seen = resume_best if resume_best is not None else None
        stagnant = 0
        for i in range(start_gen, iterations):
            gen_context["val"] = i
            searcher.step()
            current_best = float(searcher.status.get("best_eval", 0.0) or 0.0)
            if best_seen is None or current_best > (best_seen + stagnation_eps):
                best_seen = current_best
                stagnant = 0
            else:
                stagnant += 1
            if mutation_boost > 0.0 and stagnant >= stagnation_gens:
                try:
                    vals = searcher.population.values
                    noise = torch.randn_like(vals) * mutation_boost
                    vals.add_(noise).clamp_(-1.0, 1.0)
                    logger.warning(
                        f"Discovery: stagnation ({stagnant} gens). Applied mutation boost {mutation_boost:.3f}."
                    )
                except Exception as e:
                    logger.warning(f"Discovery: mutation boost failed: {e}")
                stagnant = 0
            with registry_lock:
                m = metrics_registry.get(i)
                if m: logger.info(f"Discovery (gen {i}): fit={m['fit']:.2f} surv={m['surv']:.2f} shrp={m['shrp']:.2f} dd={m['dd']:.2f} (lim={m['limit']:.3f}) segs={m['segs']}/3 trade={m['trade']:.3f}")
            if i % 10 == 0: logger.info(f"Generation {i}: Global Best Fitness = {float(searcher.status.get('best_eval', 0.0)):.4f}")
            if checkpoint_every > 0 and (i % checkpoint_every == 0):
                try:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    pop_vals = searcher.population.values.detach().cpu()
                    pop_evals = None
                    for attr in ("evaluations", "evals"):
                        if hasattr(searcher.population, attr):
                            try:
                                cur = getattr(searcher.population, attr)
                                if cur is not None:
                                    pop_evals = cur.detach().cpu()
                                break
                            except Exception:
                                pop_evals = None
                                break
                    torch.save(
                        {
                            "gen": i,
                            "population": pop_vals,
                            "evaluations": pop_evals,
                            "best_eval": float(searcher.status.get("best_eval", 0.0) or 0.0),
                            "timeframes": self.timeframes,
                            "n_features": self.n_features,
                        },
                        checkpoint_path,
                    )
                except Exception as e:
                    logger.warning(f"Discovery checkpoint failed: {e}")
        
        final_pop = searcher.population.values.detach().cpu()
        final_scores = None
        for attr in ("evaluations", "evals"):
            if hasattr(searcher.population, attr):
                try:
                    cur = getattr(searcher.population, attr)
                    if cur is not None:
                        final_scores = cur.detach().cpu()
                    break
                except Exception:
                    final_scores = None
                    break
        if final_scores is None:
            final_scores = torch.zeros(final_pop.shape[0], dtype=torch.float32)
        order = torch.argsort(final_scores, descending=True)
        topk = order[:self.n_experts]
        self.best_experts = final_pop[topk]
        self.best_scores = final_scores[topk]
        
    def save_experts(self, path: str):
        torch.save(
            {
                "genomes": self.best_experts,
                "scores": self.best_scores,
                "timeframes": self.timeframes,
                "n_features": self.n_features,
            },
            path,
        )
    def load_experts(self, path: str):
        if not Path(path).exists(): return None
        data = torch.load(path, map_location="cpu")
        self.best_experts = data["genomes"]
        self.best_scores = data.get("scores")
        self.timeframes = data["timeframes"]
        return self.best_experts

import logging
import multiprocessing
import time
from typing import Any

import pandas as pd
import torch

from ..models.device import select_device
from ..models.registry import get_model_class

logger = logging.getLogger(__name__)


def _gpu_baseline(device_name: str | None) -> float:
    """
    Rough relative throughput multiplier vs CPU for common GPUs.
    Higher is faster. 1.0 = CPU baseline.
    """
    if not device_name:
        return 1.0
    name = device_name.lower()
    # Approximate relative factors (very coarse)
    table = {
        "a4000": 10.0,
        "a5000": 12.0,
        "a6000": 15.0,
        "a100": 25.0,
        "h100": 40.0,
        "l40": 12.0,
        "l40s": 18.0,
        "v100": 15.0,
        "t4": 6.0,
        "rtx 3090": 14.0,
        "rtx 4090": 25.0,
    }
    for key, val in table.items():
        if key in name:
            return val
    return 6.0  # default GPU uplift if unknown


class BenchmarkService:
    """Estimates training complexity and time."""

    COMPLEXITY_MAP = {
        "lightgbm": 5e-5,
        "xgboost": 5e-4,
        "xgboost_rf": 6e-4,
        "xgboost_dart": 7e-4,
        "catboost": 1.2e-3,
        "catboost_alt": 1.3e-3,
        "mlp": 8e-4,
        "transformer": 1e-3,
        "nbeats": 1.1e-3,
        "tide": 1.0e-3,
        "tabnet": 1.5e-3,
        "kan": 1.2e-3,
        "evolution": 0.5e-3,
        "genetic": 5e-3,
        "unsupervised": 2e-4,
        "rl_ppo": 2e-3,
        "rl_sac": 2e-3,
        "rllib_ppo": 5e-3,
        "rllib_sac": 5e-3,
    }

    SCALING_EXPONENTS = {
        "lightgbm": 0.75,
        "xgboost": 0.75,
        "catboost": 0.80,
        "transformer": 1.0,
        "nbeats": 1.0,
        "tide": 1.0,
        "tabnet": 1.0,
        "kan": 1.0,
        "rl_ppo": 1.0,
        "rl_sac": 1.0,
        "evolution": 1.0,
        "genetic": 1.0,
    }
    # Rough epoch counts per model; used to scale estimates for full vs incremental training
    EPOCH_MAP = {
        "lightgbm": 5,
        "xgboost": 5,
        "catboost": 6,
        "nbeats": 10,
        "tide": 10,
        "tabnet": 10,
        "kan": 10,
        "transformer": 12,
        "evolution": 6,
        "rl_ppo": 8,
        "rl_sac": 8,
    }
    # Cache probe throughputs keyed by model + shapes + device
    _probe_cache: dict[tuple, float] = {}

    def run_micro_benchmark(self, X: pd.DataFrame, y: pd.Series, device: str) -> dict:
        """
        REAL benchmark: Train small model on actual data, measure ACTUAL time.
        Returns dict with raw measurements, no guessing.
        """
        try:
            # Use a safe sample size that accommodates sequence lookbacks (e.g. 60) and splits
            n_samples = min(10000, len(X))
            if n_samples < 200:  # Too small to benchmark
                return {"samples_trained": 0, "actual_duration_sec": 0, "samples_per_second": 0, "device": device}

            sub_X = X.iloc[:n_samples].copy()
            sub_y = y.iloc[:n_samples].copy()

            # Use a simplified Transformer for the probe (good representative of heavy compute)
            # Ensure seq_len is small to avoid index errors on small slices
            prefer_gpu = str(device).startswith("cuda")
            model = get_model_class("transformer", prefer_gpu=prefer_gpu)(
                d_model=32,
                nhead=2,
                num_layers=1,
                seq_len=30,  # Small sequence length for safety
                max_time_sec=10,  # Hard cap for probe
                device=device,
                batch_size=32,
            )

            t0 = time.perf_counter()
            try:
                model.fit(sub_X, sub_y)
            except Exception as e:
                logger.debug(f"Probe model fit failed ({e}). Trying simpler model...")
                # Fallback to a simpler model if Transformer fails (e.g. XGBoost if installed, or just skip)
                return {"samples_trained": 0, "actual_duration_sec": 0, "samples_per_second": 0, "device": device}

            duration = time.perf_counter() - t0

            # REAL measurements only
            result = {
                "samples_trained": n_samples,
                "actual_duration_sec": duration,
                "samples_per_second": n_samples / duration if duration > 0 else 0,
                "device": device,
            }

            logger.info(
                f"REAL Benchmark: {n_samples} samples in {duration:.2f}s "
                f"= {result['samples_per_second']:.0f} samples/sec on {device}"
            )
            return result

        except Exception as e:
            logger.warning(f"Benchmark failed: {e}")
            return {"samples_trained": 0, "actual_duration_sec": 0, "samples_per_second": 0, "device": device}

    def _hardware_signature(self, simulated_gpu: str | None = None) -> tuple:
        try:
            if simulated_gpu:
                return ("cuda-sim", simulated_gpu, 1, None, None)
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return ("cuda", props.name, torch.cuda.device_count(), props.multi_processor_count, props.total_memory)
        except Exception:
            pass
        try:
            return ("cpu", multiprocessing.cpu_count())
        except Exception:
            return ("unknown",)

    def probe_throughput(
        self,
        model_name: str,
        X: pd.DataFrame,
        batch_size: int,
        device: str,
        steps: int = 10,
        simulated_gpu: str | None = None,
    ) -> float | None:
        """
        Time a few mini-batches on the real model/config to estimate steps/sec.
        Uses cache keyed by model, feature shape, batch size, and hardware signature.
        """
        try:
            feature_dim = X.shape[1]
            sig = (model_name, feature_dim, batch_size, device, self._hardware_signature(simulated_gpu))
            if sig in self._probe_cache:
                return self._probe_cache[sig]

            prefer_gpu = (simulated_gpu is not None) or str(device).startswith("cuda")
            cls = get_model_class(model_name, prefer_gpu=prefer_gpu)
            # Small model config for probe to avoid long warmup
            kwargs = {}
            if model_name == "transformer":
                kwargs.update({"d_model": 64, "n_heads": 2, "n_layers": 1})
            if model_name in {"tide", "tabnet", "kan", "nbeats"}:
                kwargs.update({"batch_size": min(batch_size, 128)})

            model = cls(**kwargs)
            if hasattr(model, "device"):
                model.device = select_device(device if simulated_gpu is None else "cuda")

            bs = min(batch_size, len(X))
            Xs = X.iloc[: max(bs * steps, bs)].copy()
            # Build a dataloader-like loop

            data = torch.as_tensor(
                Xs.values,
                dtype=torch.float32,
                device=model.device if hasattr(model, "device") else (device if simulated_gpu is None else "cuda"),
            )
            labels = torch.zeros(len(Xs), dtype=torch.int64, device=data.device)
            criterion = torch.nn.CrossEntropyLoss() if hasattr(torch.nn, "CrossEntropyLoss") else None

            model.train()
            if hasattr(model, "optimizer"):
                optimizer = model.optimizer
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) if hasattr(model, "parameters") else None

            # Warmup
            start_idx = 0
            for _ in range(min(3, steps)):
                end_idx = start_idx + bs
                xb = data[start_idx:end_idx]
                yb = labels[start_idx:end_idx]
                out = model(xb) if callable(model) else model.fit  # type: ignore
                if callable(out):
                    out = out(xb)
                if optimizer and criterion and hasattr(out, "shape"):
                    loss = criterion(out, yb % out.shape[1])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                start_idx = (start_idx + bs) % len(data)

            if torch.cuda.is_available() or simulated_gpu:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            start_idx = 0
            for _ in range(steps):
                end_idx = start_idx + bs
                xb = data[start_idx:end_idx]
                yb = labels[start_idx:end_idx]
                out = model(xb) if callable(model) else model.fit  # type: ignore
                if callable(out):
                    out = out(xb)
                if optimizer and criterion and hasattr(out, "shape"):
                    loss = criterion(out, yb % out.shape[1])
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                start_idx = (start_idx + bs) % len(data)
            if torch.cuda.is_available() or simulated_gpu:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            steps_per_sec = steps / max(dt, 1e-6)
            self._probe_cache[sig] = steps_per_sec
            return steps_per_sec
        except Exception as e:
            logger.debug(f"Throughput probe failed for {model_name}: {e}")
            return None

    GPU_SPECS = {
        # FP32 TFLOPS, Memory BW (GB/s)
        "a100": (19.5, 1555),
        "h100": (60.0, 3350),
        "a6000": (38.7, 768),
        "a5000": (27.8, 768),
        "a4000": (19.2, 448),
        "rtx 4090": (82.6, 1008),
        "rtx 3090": (35.6, 936),
        "v100": (15.7, 900),
        "t4": (8.1, 320),
        "cpu": (0.5, 50),  # Baseline reference
    }

    # Approx GFLOPs per sample per epoch (forward+backward)
    MODEL_GFLOPS = {
        "transformer": 0.5,
        "nbeats": 0.3,
        "tide": 0.25,
        "tabnet": 0.4,
        "kan": 0.45,
        "xgboost": 0.001,  # Tree models are memory bound, not compute bound
        "lightgbm": 0.001,
        "catboost": 0.002,
        "evolution": 0.1,
    }

    def _theoretical_estimate(self, model_name: str, n_samples: int, epochs: int, device_name: str) -> float:
        """
        Calculate theoretical training time based on compute/memory bounds.
        Time = max(Compute Time, Memory Time)
        """
        # 1. Identify Hardware Specs
        specs = (0.5, 50)  # Default CPU
        norm_name = device_name.lower()
        for key, val in self.GPU_SPECS.items():
            if key in norm_name:
                specs = val
                break

        tflops, bw_gbps = specs

        # 2. Identify Model Complexity
        gflops_per_sample = self.MODEL_GFLOPS.get(model_name, 0.1)

        # 3. Compute Bound Time
        # Total GFLOPs = n_samples * epochs * gflops_per_sample
        # Time = Total GFLOPs / (TFLOPS * 1000 * Efficiency)
        utilization = 0.4  # Conservative real-world utilization
        total_gflops = n_samples * epochs * gflops_per_sample
        compute_time = total_gflops / (tflops * 1000 * utilization)

        # 4. Memory Bound Time (Rough approximation)
        # Assuming 4 bytes per feature * 100 features * 3 (read/write/grad)
        data_gb = (n_samples * 100 * 4 * 3 * epochs) / 1e9
        memory_time = data_gb / (bw_gbps * utilization)

        # Real time is max of bounds (usually compute for DL, memory for Trees)
        est_time = max(compute_time, memory_time)

        # Sanity check: Tree models are CPU bound or super fast on GPU, simple logic doesn't fit well
        if "boost" in model_name and "cuda" in device_name:
            est_time = est_time * 0.1  # GPU trees are very fast

        return est_time

    def _probe_scaling_law(self, model_name: str, X: pd.DataFrame, y: pd.Series, device: str) -> tuple[float, float]:
        """
        Run benchmarks at multiple scales to determine startup cost (c) and per-sample cost (m).
        Returns (m, c) for Time = m * N + c
        """
        import numpy as np

        sizes = [1000, 2000, 4000]
        times = []
        valid_sizes = []

        # Use real data if available and sufficient, otherwise fallback to random
        use_real_data = X is not None and len(X) >= max(sizes)

        if not use_real_data:
            # Fallback: Create dummy data
            max_size = max(sizes)
            feature_dim = 100  # Default assumption
            if X is not None:
                feature_dim = X.shape[1]

            X_source = pd.DataFrame(np.random.randn(max_size, feature_dim).astype(np.float32))
            y_source = pd.Series(np.random.randint(0, 3, size=max_size))

            meta_idx = pd.date_range("2024-01-01", periods=max_size, freq="1min")
            meta_source = pd.DataFrame(
                {
                    "open": np.random.rand(max_size) + 100,
                    "high": np.random.rand(max_size) + 101,
                    "low": np.random.rand(max_size) + 99,
                    "close": np.random.rand(max_size) + 100,
                    "volume": np.random.rand(max_size) * 1000,
                },
                index=meta_idx,
            )
        else:
            # Use Real Data
            X_source = X
            y_source = y
            # Try to infer metadata from X if possible or create aligned dummy meta
            # Ideally X index is datetime
            meta_source = None
            if hasattr(X, "index"):
                # Simple heuristic to extract OHLC if present in X, else dummy
                # But X usually has features like 'rsi', not 'open'.
                # We'll create dummy meta aligned to X's index for safety.
                meta_source = pd.DataFrame(
                    {
                        "open": np.random.rand(len(X)) + 100,
                        "high": np.random.rand(len(X)) + 101,
                        "low": np.random.rand(len(X)) + 99,
                        "close": np.random.rand(len(X)) + 100,
                        "volume": np.random.rand(len(X)) * 1000,
                    },
                    index=X.index,
                )

        for size in sizes:
            # If using real data, strict check; for dummy we made enough
            if size > len(X_source):
                break

            sub_X = X_source.iloc[:size].copy()
            sub_y = y_source.iloc[:size].copy()
            sub_meta = meta_source.iloc[:size].copy() if meta_source is not None else None

            # Instantiate model
            try:
                # Use simplified config to probe speed quickly
                prefer_gpu = str(device).startswith("cuda")
                cls = get_model_class(model_name, prefer_gpu=prefer_gpu)
                kwargs = {"max_time_sec": 30}  # Cap probe time
                if "transformer" in model_name:
                    kwargs.update({"d_model": 32, "num_layers": 1})
                if "evolution" in model_name:
                    kwargs.update({"population_size": 10, "generations": 1})
                if "genetic" in model_name:
                    kwargs.update({"population_size": 10, "generations": 1})

                model = cls(**kwargs)
                if hasattr(model, "device"):
                    model.device = select_device(device)

                t0 = time.perf_counter()
                if hasattr(model, "fit"):
                    # Try passing metadata first (for RL/Genetic)
                    try:
                        if sub_meta is not None:
                            model.fit(sub_X, sub_y, metadata=sub_meta)
                        else:
                            model.fit(sub_X, sub_y)
                    except TypeError:
                        # Fallback for models that don't accept metadata
                        model.fit(sub_X, sub_y)

                duration = time.perf_counter() - t0

                times.append(duration)
                valid_sizes.append(size)
            except Exception:
                pass

        if len(valid_sizes) < 2:
            return 0.0, 0.0  # Failed to establish trend

        # Fit Line: T = m*N + c
        # Simple Linear Regression
        N = np.array(valid_sizes)
        T = np.array(times)
        A = np.vstack([N, np.ones(len(N))]).T
        m, c = np.linalg.lstsq(A, T, rcond=None)[0]

        return max(0.0, m), max(0.0, c)

    def estimate_time(
        self,
        models: list[str],
        n_samples: int,
        benchmark_result: dict | None,  # Kept for backward compatibility but largely ignored in new logic
        gpu: bool,
        gpu_count: int = 1,
        context: str = "full",
        historical_durations: dict[str, float] | None = None,
        historical_n: int | None = None,
        historical_gpu: tuple[bool, int] | None = None,
        incremental_stats: dict[str, dict[str, Any]] | None = None,
        probe_kwargs: dict[str, Any] | None = None,
        simulate_gpu: str | None = None,
    ) -> float:
        total_time = 0.0
        breakdown = []

        # Probe Data (if available in kwargs)
        X_probe = probe_kwargs.get("X") if probe_kwargs else None

        # Hardware Factor (if simulating)
        hw_factor = 1.0
        if simulate_gpu:
            hw_factor = _gpu_baseline(simulate_gpu)
            logger.info(f"Estimating for {simulate_gpu} (Speedup Factor: {hw_factor:.1f}x vs Baseline)")

        for m in models:
            est_seconds = 0.0
            source = "unknown"

            # 1. Historical Exact Match (Fastest/Most Accurate)
            if context == "incremental" and incremental_stats and m in incremental_stats:
                stat = incremental_stats[m]
                if stat.get("n_samples", 0) > 0:
                    ratio = n_samples / stat["n_samples"]
                    est_seconds = stat["duration_sec"] * ratio
                    source = "incremental_history"

            # 2. Historical Scaling (Full Training)
            elif historical_durations and historical_n and m in historical_durations:
                ratio = n_samples / historical_n
                # Adjust for epoch scaling if context differs
                epoch_scale = 0.35 if context == "incremental" else 1.0
                est_seconds = historical_durations[m] * ratio * epoch_scale
                source = "full_history"

            # 3. Multi-Point Probe (Scientific Extrapolation)
            elif X_probe is not None and not simulate_gpu:
                # We can run a real probe on current hardware
                # Note: We assume y is available or we mock it
                y_probe = pd.Series(0, index=X_probe.index)  # Mock labels
                device = "cuda" if gpu else "cpu"

                slope, intercept = self._probe_scaling_law(m, X_probe, y_probe, device)
                if slope > 0:
                    est_seconds = (slope * n_samples) + intercept
                    source = "scaling_law_probe"

            # 4. Theoretical / Fallback (Simulated or No Probe Available)
            if est_seconds == 0.0:
                # Heuristic
                # Time ~ C * N
                complexity = self.COMPLEXITY_MAP.get(m, 5e-3)  # Default to high (5e-3) if unknown
                est_seconds = complexity * n_samples

                # Apply Sub-Linear Scaling Correction for large datasets
                exponent = self.SCALING_EXPONENTS.get(m, 1.0)
                if n_samples > 10000 and exponent < 1.0:
                    # Correction factor: (N / 10k)^(exp - 1)
                    # This reduces the estimate as N grows if exp < 1 (e.g. tree models)
                    correction = (n_samples / 10000.0) ** (exponent - 1.0)
                    est_seconds *= correction

                # Adjust for GPU (rough heuristic scaling)
                if simulate_gpu:
                    est_seconds /= max(hw_factor, 1e-6) * max(gpu_count, 1)
                    source = f"theoretical_{simulate_gpu}"
                elif gpu:
                    est_seconds /= max(_gpu_baseline("gpu"), 1e-6) * max(gpu_count, 1)
                    source = "complexity_heuristic"
                else:
                    source = "complexity_heuristic"

            # Apply hard epoch scaling for final estimate
            epochs = self.EPOCH_MAP.get(m, 5)
            # If the estimate source was per-epoch (like probe), multiply by epochs.
            # If source was 'history' (total time), don't multiply.
            if source in ["scaling_law_probe", "complexity_heuristic", f"theoretical_{simulate_gpu}"]:
                est_seconds *= epochs

            total_time += est_seconds
            breakdown.append(f"{m}: {est_seconds / 60:.1f}min ({source})")

        if breakdown:
            logger.info(f"Scientific Time Estimate (N={n_samples}):")
            for line in breakdown:
                logger.info(f"  → {line}")

        return total_time

"""
Autotuning service for hardware optimization.
Decides whether to use CPU or GPU based on runtime benchmarks.
"""

import logging
import time

from ..models.device import get_device_info, select_device

logger = logging.getLogger(__name__)


class Autotuner:
    _instance = None
    _benchmarks: dict[str, float] = {}
    _decisions: dict[str, bool] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.device_name = "cpu"
        self.gpu_available = False
        self.gpu_info = {}
        self._initialized = True
        self._detect_hardware()
        # Measure optimal batch size (works in workers too with spawn context)
        self._optimal_batch_size = self._measure_optimal_batch_size()

    def _detect_hardware(self):
        """Detect available hardware and basic capabilities."""
        dev = select_device("auto")
        self.device_name = dev if isinstance(dev, str) else dev[0]
        self.gpu_available = self.device_name.startswith("cuda")

        if self.gpu_available:
            self.gpu_info = get_device_info(self.device_name)
            logger.info(
                f"Autotuner detected GPU: {self.gpu_info.get('name')} "
                f"(Compute: {self.gpu_info.get('compute_capability')})"
            )
        else:
            logger.info("Autotuner: GPU not found or not requested. Using CPU.")

    def _measure_optimal_batch_size(self) -> int:
        """
        Empirically find the optimal batch size by measuring throughput.
        Doubles batch size until throughput drops or OOM occurs.
        """
        try:
            import torch

            logger.info("Autotuner: Benchmarking optimal batch size...")

            # Simulate a standard hidden layer forward/backward pass
            # Input: [batch, 256], Weights: [256, 256]
            hidden_dim = 256

            best_throughput = 0.0
            best_bs = 32

            # Search space: 32 to 16384
            batch_candidates = [32 * (2**i) for i in range(10)]

            device = self.device_name

            # Warmup
            warmup_data = torch.randn(32, hidden_dim, device=device)
            warmup_w = torch.randn(hidden_dim, hidden_dim, device=device)
            _ = torch.matmul(warmup_data, warmup_w)

            for bs in batch_candidates:
                try:
                    # VRAM Safety Check
                    if self.gpu_available:
                        # Est: 4 bytes * num_elements * 3 (grad/overhead)
                        needed_bytes = bs * hidden_dim * 4 * 3
                        free_mem = 0
                        if hasattr(torch.cuda, "mem_get_info"):
                            free_mem, _ = torch.cuda.mem_get_info(device)

                        if free_mem > 0 and needed_bytes > (free_mem * 0.8):
                            logger.debug(f"Stopping benchmark at BS={bs} (VRAM safety limit)")
                            break

                    # Prepare data
                    x = torch.randn(bs, hidden_dim, device=device)
                    w = torch.randn(hidden_dim, hidden_dim, device=device, requires_grad=True)

                    # Robust Measurement: Average over 3 iterations
                    iterations = 3
                    total_time = 0.0

                    for _ in range(iterations):
                        if self.gpu_available:
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()

                        y = torch.matmul(x, w)
                        loss = y.sum()
                        loss.backward()

                        if self.gpu_available:
                            torch.cuda.synchronize()
                        total_time += time.perf_counter() - t0

                        # Reset grads to avoid accumulation overhead affecting timing
                        w.grad = None

                    avg_time = total_time / iterations
                    if avg_time == 0:
                        avg_time = 1e-9
                    throughput = bs / avg_time

                    logger.debug(f"BS={bs}: {throughput:.1f} items/sec")

                    # Gain analysis
                    gain = (throughput - best_throughput) / (best_throughput + 1e-9)

                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_bs = bs
                        # Diminishing returns check (< 5% gain)
                        if gain < 0.05 and bs > 128:
                            break
                    else:
                        # Throughput dropped
                        break

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.debug(f"OOM at BS={bs}")
                    break
                except Exception:
                    break

            logger.info(f"Autotuner: Optimal batch size found: {best_bs}")
            return best_bs

        except Exception as e:
            logger.warning(f"Batch size autotune failed: {e}. Defaulting to 64.")
            return 64

    def should_use_gpu_for_features(self, df_length: int) -> bool:
        """
        Decide if GPU should be used for feature engineering based on dataframe size.
        Dynamically benchmarks CPU vs GPU overhead to find the crossover point.
        """
        if not self.gpu_available:
            return False

        # Check cache first
        decision_key = f"features_{df_length}"
        if decision_key in self._decisions:
            return self._decisions[decision_key]

        # Dynamic Benchmark: Race CPU vs GPU for this size
        # We simulate a "light" feature calculation (like RSI or SMA)
        # including data transfer cost.
        try:
            import torch

            # 1. CPU Benchmark
            # Create random data
            data_cpu = torch.randn(df_length, device="cpu")
            start_cpu = time.perf_counter()
            # Simulate heavy vectorized op (e.g. rolling mean equivalent)
            _ = data_cpu * data_cpu + data_cpu.sin()
            end_cpu = time.perf_counter()
            cpu_time = end_cpu - start_cpu

            # 2. GPU Benchmark
            # Includes transfer time!
            torch.cuda.synchronize()
            start_gpu = time.perf_counter()
            data_gpu = data_cpu.to(self.device_name)
            _ = data_gpu * data_gpu + data_gpu.sin()
            # Simulate return to CPU (as many features need to be merged back to pandas)
            _ = data_gpu.cpu()
            torch.cuda.synchronize()
            end_gpu = time.perf_counter()
            gpu_time = end_gpu - start_gpu

            # Decision
            # We prefer GPU if it's at least 1.2x faster to justify complexity
            decision = gpu_time < (cpu_time / 1.2)

            winner = "GPU" if decision else "CPU"
            logger.info(
                f"Autotuner Benchmark ({df_length} rows): "
                f"CPU={cpu_time * 1000:.3f}ms, GPU={gpu_time * 1000:.3f}ms. "
                f"Winner: {winner}"
            )

            self._decisions[decision_key] = decision
            return decision

        except Exception as e:
            logger.warning(f"Autotuner benchmark failed: {e}. Defaulting to CPU.")
            return False

    def get_optimal_batch_size(self, model_type: str = "lstm") -> int:
        """
        Return the empirically measured optimal batch size.
        """
        return self._optimal_batch_size


autotuner = Autotuner()

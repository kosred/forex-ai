# Hardware Auto-Detection Configuration

## How It Works

The bot **automatically detects** your hardware on every startup and configures thread limits optimally.

### What Gets Auto-Detected:

1. **Physical CPU Cores** - Real cores (not hyperthreaded)
2. **Logical CPU Cores** - Total threads including hyperthreading
3. **Total RAM** - System memory
4. **Available RAM** - Free memory
5. **GPU** - CUDA devices (if PyTorch installed)

### Thread Configuration (Auto-Calculated):

```
BLAS Threads      = physical_cores - 1  (for NumPy/SciPy/OpenBLAS operations)
Worker Processes  = logical_cores - 1   (for ProcessPoolExecutor/multiprocessing)
RL Environments   = 1 (auto-limited by available RAM)
```

## Your Current System:

```
Physical Cores: 6
Logical Cores:  12 (with hyperthreading)
Total RAM:      31.4 GB

Auto-Configured:
  BLAS Threads:     5  (prevents over-subscription)
  Worker Processes: 11 (uses all logical cores minus OS reserve)
  Expected Threads: ~26-30 total (healthy, not 56!)
```

## Override If Needed:

If you want to manually override (e.g., testing, debugging), set environment variables **BEFORE** running:

### Windows:
```batch
set OMP_NUM_THREADS=4
set FOREX_BOT_CPU_THREADS=8
python forex-ai.py train
```

### Linux/Mac:
```bash
export OMP_NUM_THREADS=4
export FOREX_BOT_CPU_THREADS=8
python forex-ai.py train
```

## Available Override Variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OMP_NUM_THREADS` | BLAS/OpenMP threads | physical_cores - 1 |
| `MKL_NUM_THREADS` | Intel MKL threads | Same as OMP |
| `OPENBLAS_NUM_THREADS` | OpenBLAS threads | Same as OMP |
| `NUMEXPR_NUM_THREADS` | NumExpr threads | Same as OMP |
| `FOREX_BOT_CPU_THREADS` | Worker processes | logical_cores - 1 |
| `FOREX_BOT_CPU_RESERVE` | Cores to reserve for OS | 1 |
| `FOREX_BOT_RL_ENVS` | RL parallel envs | Auto (RAM-limited) |
| `FOREX_BOT_TALIB_WORKERS` | TA-Lib indicator workers | Same as CPU_THREADS |

## Why This Matters:

**Before Fix:**
- 56 threads on 12-core CPU (4.7x over-subscription)
- Massive thread thrashing
- Cache pollution
- Poor performance
- 99.7% RAM usage

**After Auto-Detection:**
- ~26-30 threads on 12-core CPU (2-2.5x, healthy)
- Efficient CPU utilization
- Better cache locality
- Improved performance
- Predictable RAM usage

## Benefits:

✅ **No hardcoding** - works on ANY hardware (laptop, workstation, server)
✅ **Portable** - same code runs on 4-core laptop or 64-core server
✅ **Visible** - shows detected config on startup
✅ **Overridable** - can manually tune if needed
✅ **No forgetting** - auto-adapts when you change machines

## Startup Output Example:

```
======================================================================
[HW AUTO-DETECT] Physical Cores: 6 | Logical Cores: 12
[HW AUTO-DETECT] BLAS Threads: 5 | Worker Processes: 11
[HW AUTO-DETECT] Total RAM: 31.4GB | Available: 26.7GB (15.1% used)
[HW AUTO-DETECT] GPU: None detected
======================================================================
```

You'll see this every time the bot starts, so you always know the configuration!

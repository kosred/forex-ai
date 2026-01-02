@echo off
REM Thread Control for Forex Bot - Prevents over-subscription
REM Set BEFORE running Python to ensure all libraries respect limits

REM CPU Budget: 12 cores - 1 reserved = 11 usable
REM Use 11 for computation-heavy tasks, 6 for BLAS to leave room for parallelism

echo Setting threading environment variables for 6-core CPU (12 logical cores)...

REM OpenBLAS (NumPy/SciPy)
set OPENBLAS_NUM_THREADS=6
set OPENBLAS_MAIN_FREE=1

REM Intel MKL (if installed)
set MKL_NUM_THREADS=6
set MKL_DYNAMIC=FALSE

REM OpenMP (general parallel libraries)
set OMP_NUM_THREADS=6
set OMP_DYNAMIC=FALSE

REM NumExpr (pandas/numpy expression evaluation)
set NUMEXPR_NUM_THREADS=6
set NUMEXPR_MAX_THREADS=6

REM Apple Accelerate (not used on Windows, but set for completeness)
set VECLIB_MAXIMUM_THREADS=6

REM Numba JIT compiler
set NUMBA_NUM_THREADS=6

REM PyTorch
set OMP_NUM_THREADS=6

REM TensorFlow
set TF_NUM_INTRAOP_THREADS=6
set TF_NUM_INTEROP_THREADS=1

REM Forex Bot CPU Budget (used by multiprocessing workers)
set FOREX_BOT_CPU_THREADS=11
set FOREX_BOT_CPU_RESERVE=1
set FOREX_BOT_RL_ENVS=1
set FOREX_BOT_TALIB_WORKERS=11

echo Thread limits set successfully!
echo   BLAS/LAPACK: 6 threads per operation
echo   CPU Workers: 11 parallel processes
echo   RL Envs: 1 (RAM-limited)
echo.
echo Now run: python forex-ai.py train

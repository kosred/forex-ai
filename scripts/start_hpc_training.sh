#!/bin/bash
# scripts/start_hpc_training.sh
# ---------------------------------------------------------
# Cooperative 4-GPU Training Launcher
# Automatically handles JIT/Library paths for A4000 clusters
# ---------------------------------------------------------

# 1. Activate Environment
source ~/forex-ai/venv/bin/activate
cd ~/forex-ai

# Force-update package link to ensure workers see latest source code
pip install -e .

# 2. Critical Stability Fixes
# Disable torch.compile to prevent nvvmAddNVVMContainerToProgram/JIT errors
export FOREX_BOT_DISABLE_COMPILE=1

# Ensure system CUDA libraries (Driver 570+) take precedence over pip-installed libs
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# 3. Tuning for A4000 16GB
export FOREX_BOT_GLOBAL_POOL_MEM_FRAC=0.20
export FOREX_BOT_PARALLEL_MODELS=auto

echo "[HPC] Starting Cooperative 4-GPU Training..."
echo "[HPC] Torch Compile: DISABLED"
echo "[HPC] LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 4. Launch with log redirection
python src/forex_bot/main.py --train > ~/forex-ai/training_stdout.log 2> ~/forex-ai/training_stderr.log

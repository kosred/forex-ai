#!/usr/bin/env bash
set -euo pipefail

# HPC MASTER GPU SETUP (2025 Standard)
# Target: 8x RTX-A6000 | CUDA 12.8 | Python 3.13
# Optimized for high-throughput discovery and training.

PYTHON_BIN="python3"
CUDA_INDEX="https://download.pytorch.org/whl/cu128"

echo "[*] Initializing GPU stack for high-performance Forex bot..."

sudo apt-get update -y
sudo apt-get install -y nvidia-cuda-toolkit libcudnn9-dev libcudnn9-cuda-12

echo "[*] Upgrading base environment..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel --user --break-system-packages

echo "[*] Installing verified HPC requirements (CUDA 12.8)..."
"$PYTHON_BIN" -m pip install -r requirements-hpc.txt --user --break-system-packages

# --- GPU VERIFICATION (HPC MODE) ---
echo "[*] Running multi-GPU integrity check..."
python3 -c "
import torch
print(f'Detected {torch.cuda.device_count()} GPUs')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} | VRAM: {props.total_memory/1024**3:.1f}GB | P2P: {torch.cuda.can_device_access_peer(0, i) if i > 0 else \"Primary\"}')
"

echo "[*] Optimizing NCCL for multi-GPU performance..."
# NCCL settings for A6000 P2P topology
export NCCL_P2P_LEVEL=5
export NCCL_IB_DISABLE=1 # Disable InfiniBand if not present to avoid socket timeouts

echo "[*] Done. Your 8-GPU cluster is ready for the blitz."
#!/usr/bin/env bash
set -euo pipefail

# One-shot GPU setup for Ubuntu 24.04 (NVIDIA, CUDA runtime present).
# Installs the latest available versions of GPU-capable ML/RL deps on Python 3.12/3.13.
#
# Usage:
#   bash scripts/setup_gpu_latest.sh [--python python3.12] [--cuda-index https://download.pytorch.org/whl/cu121] [--venv .venv]

PYTHON_BIN="python3"
CUDA_INDEX="https://download.pytorch.org/whl/cu121"  # adjust if using a different CUDA runtime
VENV_DIR=".venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --cuda-index) CUDA_INDEX="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "[*] Using python: $PYTHON_BIN"
echo "[*] CUDA index: $CUDA_INDEX"
echo "[*] Venv: $VENV_DIR"

sudo apt-get update -y
sudo apt-get install -y build-essential python3-venv python3-dev libopenblas-dev libomp-dev git

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

echo "[*] Installing latest CUDA PyTorch from $CUDA_INDEX ..."
pip install --index-url "$CUDA_INDEX" torch torchvision torchaudio

echo "[*] Installing core deps..."
pip install --upgrade numpy pandas scipy scikit-learn numba psutil pyarrow matplotlib tabulate tensorboard

echo "[*] Installing gradient boosting libs..."
pip install --upgrade lightgbm xgboost catboost

echo "[*] Installing Ray + RLlib (latest)..."
pip install --upgrade "ray[default,rllib]"

echo "[*] Installing Gymnasium and SB3..."
pip install --upgrade "gymnasium[box2d,atari]"
# Try stable SB3 first; if it fails on py3.13, fall back to git main
if ! pip install --upgrade stable-baselines3; then
  echo "[*] stable-baselines3 wheel not available; installing from git main..."
  pip install --upgrade "stable-baselines3 @ git+https://github.com/DLR-RM/stable-baselines3.git"
fi

echo "[*] Installing optional CuPy (CUDA 12.x build) for GPU TA calcs..."
pip install --upgrade cupy-cuda12x || echo "[!] CuPy install failed; continue without it."

echo "[*] Done. Activate with: source $VENV_DIR/bin/activate"

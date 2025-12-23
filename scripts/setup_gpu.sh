#!/usr/bin/env bash
set -euo pipefail

# GPU setup script for Ubuntu 24.04 (CUDA 12.1 example) with Python 3.12/3.13.
# - Installs Python venv, CUDA PyTorch, Ray 2.53.0 with RLlib, SB3 (git), and project deps.
# - Assumes NVIDIA driver/CUDA runtime already present on the VM (for torch/cuPy wheels).

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[*] Using python: $PYTHON_BIN"
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[*] Upgrading pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

echo "[*] Installing PyTorch (CUDA) from $PIP_EXTRA_INDEX_URL ..."
pip install --index-url "$PIP_EXTRA_INDEX_URL" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

echo "[*] Installing core/project requirements..."
pip install -r requirements-gpu.txt

echo "[*] Done. Activate with: source $VENV_DIR/bin/activate"

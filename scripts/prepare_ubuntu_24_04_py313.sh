#!/usr/bin/env bash
set -euo pipefail

# Prep script for Ubuntu 24.04: ensure Python 3.13 is installed + default,
# upgrade pip tooling, then run forex-ai.py.

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  SUDO="sudo"
else
  SUDO=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
    echo "[WARN] This script is intended for Ubuntu 24.04. Detected: ${ID:-unknown} ${VERSION_ID:-unknown}"
  fi
fi

echo "[*] Ensuring Python 3.13 is installed..."
if ! command -v python3.13 >/dev/null 2>&1; then
  ${SUDO} apt-get update -y
  ${SUDO} apt-get install -y software-properties-common
  if ! grep -Rqs "deadsnakes" /etc/apt/sources.list.d /etc/apt/sources.list; then
    ${SUDO} add-apt-repository -y ppa:deadsnakes/ppa
  fi
  ${SUDO} apt-get update -y
  ${SUDO} apt-get install -y python3.13 python3.13-venv python3.13-dev
fi

echo "[*] Setting python3 default to python3.13..."
${SUDO} update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 313 || true
${SUDO} update-alternatives --set python3 /usr/bin/python3.13 || true

echo "[*] Upgrading pip tooling for Python 3.13..."
python3.13 -m ensurepip --upgrade || true
python3.13 -m pip install --upgrade pip setuptools wheel

echo "[*] Launching forex-ai.py with Python 3.13..."
cd "${PROJECT_ROOT}"
exec python3.13 forex-ai.py

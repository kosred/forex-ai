#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/launch_training.sh single   # 1 GPU/CPU
#   ./scripts/launch_training.sh medium   # up to 8 GPUs, parallel single-GPU jobs
#   ./scripts/launch_training.sh giga     # >=10 GPUs, torchrun DDP across all visible GPUs
#
# Env overrides:
#   WORKDIR=/opt/forex-ai
#   REPO_URL=https://github.com/kosred/forex-ai.git
#   DATA_ARCHIVE=          # optional; if set, unzip into ./data (not needed if data is in repo)

PROFILE="${1:-single}"   # single | medium | giga
REPO_URL="${REPO_URL:-https://github.com/kosred/forex-ai.git}"
WORKDIR="${WORKDIR:-/opt/forex-ai}"
DATA_ARCHIVE="${DATA_ARCHIVE:-data.zip}"

# 1) Fetch code
mkdir -p "$WORKDIR"
if [ ! -d "$WORKDIR/.git" ]; then
  git clone "$REPO_URL" "$WORKDIR"
else
  git -C "$WORKDIR" pull --ff-only
fi

cd "$WORKDIR"

# 2) Optional: unpack data archive if explicitly provided
if [ -n "${DATA_ARCHIVE}" ] && [ -f "$DATA_ARCHIVE" ]; then
  unzip -o "$DATA_ARCHIVE" -d data
fi

# 3) Install deps (adjust to your environment)
python3 -m pip install --upgrade pip
python3 -m pip install -e .

# 4) Launch training based on profile
case "$PROFILE" in
  giga)
    # >=10 GPUs: use torchrun DDP across all visible GPUs
    NGPU=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
    if [ "$NGPU" -lt 2 ]; then echo "Need >=2 GPUs for giga"; exit 1; fi
    torchrun --nproc_per_node="$NGPU" forex-ai.py --config config.yaml --enable_ddp
    ;;
  medium)
    # Up to 8 GPUs: run parallel single-GPU jobs (edit model list as needed)
    # Auto-derive models from config (ml_models + common deep models); fallback to a short list
    read -r MODELS_STR <<'EOF'
$(python - <<'PY'
import os, sys, yaml
cfg = os.environ.get("CONFIG_FILE", "config.yaml")
try:
    data = yaml.safe_load(open(cfg)) or {}
    mcfg = data.get("models", {}) or {}
    ml = mcfg.get("ml_models", []) or []
    deep = [
        "transformer",
        "patchtst",
        "timesnet",
        "tide_nf",
        "nbeatsx_nf",
        "kan",
        "tabnet",
        "rl_ppo",
        "rl_sac",
    ]
    seen = set()
    out = []
    for m in ml + deep:
        if m not in seen:
            out.append(m)
            seen.add(m)
    print(" ".join(out))
except Exception:
    print("transformer kan")
PY
)
EOF
    read -a MODELS <<< "$MODELS_STR"
    GPU=0
    for M in "${MODELS[@]}"; do
      CUDA_VISIBLE_DEVICES="$GPU" forex-ai.py --config config.yaml --models "$M" &
      GPU=$(( (GPU + 1) % 8 ))
    done
    wait
    ;;
  single|*)
    forex-ai.py --config config.yaml
    ;;
esac

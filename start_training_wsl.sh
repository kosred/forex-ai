#!/usr/bin/env sh
set -eu

# WSL2-safe defaults (override by exporting before running)
: "${CONFIG_FILE:=config.wsl.yaml}"
: "${FOREX_BOT_FEATURE_WORKERS:=2}"
: "${FOREX_BOT_MAX_CONCURRENT_SYMBOLS:=1}"
: "${FOREX_BOT_MAX_CONCURRENT_TF_LOADS:=2}"
: "${FOREX_BOT_CACHE_TRAINING_FRAMES:=false}"
: "${FOREX_BOT_MAX_TRAINING_ROWS:=100000}"

export CONFIG_FILE
export FOREX_BOT_FEATURE_WORKERS
export FOREX_BOT_MAX_CONCURRENT_SYMBOLS
export FOREX_BOT_MAX_CONCURRENT_TF_LOADS
export FOREX_BOT_CACHE_TRAINING_FRAMES
export FOREX_BOT_MAX_TRAINING_ROWS

if [ -z "${PYTHON_BIN:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "python3 not found. Activate your venv or install Python 3." >&2
        exit 127
    fi
fi

export PYTHONPATH="${PYTHONPATH:-}:src"
exec "$PYTHON_BIN" -m forex_bot.main --train "$@"

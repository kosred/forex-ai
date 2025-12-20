#!/bin/bash
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
export OMP_NUM_THREADS=1

# Ensure venv is active
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Use module invocation for safety
python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 src/forex_bot/main.py "$@"
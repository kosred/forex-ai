#!/bin/bash
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
torchrun --nproc_per_node=8 --nnodes=1 src/forex_bot/main.py "$@"
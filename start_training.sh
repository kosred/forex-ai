#!/bin/bash
source ~/forex-ai/venv/bin/activate
cd ~/forex-ai
export FOREX_BOT_DISABLE_COMPILE=1
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
python src/forex_bot/main.py --train

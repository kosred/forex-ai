#!/usr/bin/env python3
"""
Multi-GPU Training Launcher

Spawns one training process per visible GPU, each pinned via CUDA_VISIBLE_DEVICES.

Examples:
  python scripts/train_multi_gpu.py
  python scripts/train_multi_gpu.py --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def _detect_gpus() -> list[int]:
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:
        pass

    try:
        res = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
        lines = [ln for ln in res.stdout.splitlines() if ln.strip().startswith("GPU")]
        return list(range(len(lines)))
    except Exception:
        return []


def _parse_gpu_ids(raw: str | None) -> list[int]:
    if not raw or raw.strip().lower() == "auto":
        return _detect_gpus()
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def _launch_gpu_worker(
    gpu_id: int,
    *,
    verbose: bool,
    stagger_seconds: float,
    tree_device: str | None,
    features_device: str | None,
) -> subprocess.Popen:
    """Launch a training worker pinned to a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    LOGS_DIR.mkdir(exist_ok=True)
    log_file = LOGS_DIR / f"training_gpu{gpu_id}.log"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src/forex_bot/main.py"),
        "--train",
        "--device",
        "gpu",
    ]
    if verbose:
        cmd.append("--verbose")
    if tree_device:
        cmd.extend(["--tree-device", tree_device])
    if features_device:
        cmd.extend(["--features-device", features_device])

    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)

    time.sleep(max(0.0, float(stagger_seconds)))
    return proc

def main():
    parser = argparse.ArgumentParser(description="Launch one training worker per GPU.")
    parser.add_argument("--gpus", default="auto", help="Comma-separated GPU ids (e.g. 0,1,2) or 'auto'.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose worker logging.")
    parser.add_argument("--stagger-seconds", type=float, default=2.0, help="Delay between worker launches.")
    parser.add_argument("--tree-device", choices=["auto", "cpu", "gpu"], default="cpu")
    parser.add_argument("--features-device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()

    gpu_ids = _parse_gpu_ids(args.gpus)
    if not gpu_ids:
        print("No GPUs detected. Aborting.")
        return 2

    print(f"Starting multi-GPU training with {len(gpu_ids)} GPUs: {gpu_ids}")
    print("=" * 70)

    # Clear old logs
    LOGS_DIR.mkdir(exist_ok=True)
    for f in LOGS_DIR.glob("training_gpu*.log"):
        try:
            f.unlink()
        except Exception:
            pass

    # Launch workers
    workers = []
    for gpu_id in gpu_ids:
        proc = _launch_gpu_worker(
            gpu_id,
            verbose=bool(args.verbose),
            stagger_seconds=float(args.stagger_seconds),
            tree_device=str(args.tree_device),
            features_device=str(args.features_device),
        )
        workers.append((proc, gpu_id))
        print(f"Launched worker on GPU {gpu_id} (PID: {proc.pid}) -> {LOGS_DIR / f'training_gpu{gpu_id}.log'}")

    print("\nAll workers launched. Monitoring...")
    print("Check logs: tail -f logs/training_gpu*.log")
    print("Check GPUs: nvidia-smi -l 2")

    # Wait for all workers
    try:
        while any(proc.poll() is None for proc, _ in workers):
            time.sleep(10)
            active = sum(1 for proc, _gid in workers if proc.poll() is None)
            print(f"Active workers: {active}/{len(workers)}")
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating workers...")
        for proc, _gid in workers:
            proc.terminate()

    print("\nAll workers finished.")
    for proc, gid in workers:
        rc = proc.returncode
        status = "OK" if rc == 0 else f"FAILED (code={rc})"
        print(f"GPU {gid}: {status}")

if __name__ == "__main__":
    raise SystemExit(main())

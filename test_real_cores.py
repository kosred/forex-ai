#!/usr/bin/env python3
"""
PROPER multi-core test - monitors ALL cores and uses multiprocessing
"""

import multiprocessing as mp
import time
import numpy as np
import pandas as pd
import psutil

print("="*80)
print("PROPER MULTI-CORE TEST")
print("="*80)

cpu_count = psutil.cpu_count(logical=True)
print(f"\nTotal CPU cores: {cpu_count}")

# Create sample data
n_samples = 10000
n_features = 50
print(f"Creating sample data ({n_samples} samples, {n_features} features)...")

X = pd.DataFrame(np.random.randn(n_samples, n_features))
y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

def train_one_model(model_id, n_jobs):
    """Train LightGBM with specified n_jobs"""
    import lightgbm as lgb
    import os

    pid = os.getpid()
    print(f"[PID {pid}] Model {model_id}: Training with n_jobs={n_jobs}")

    model = lgb.LGBMClassifier(
        n_estimators=100,
        n_jobs=n_jobs,
        verbose=-1
    )

    start = time.time()
    model.fit(X, y)
    duration = time.time() - start

    print(f"[PID {pid}] Model {model_id}: Completed in {duration:.2f}s")
    return duration

def monitor_cpu(duration_seconds):
    """Monitor CPU usage on ALL cores"""
    print(f"\n[MONITOR] Tracking {cpu_count} cores for {duration_seconds}s...")

    samples = []
    for i in range(duration_seconds * 2):  # Sample every 0.5s
        cpu_per_core = psutil.cpu_percent(interval=0.5, percpu=True)
        samples.append(cpu_per_core)

        # Print every 2 seconds
        if i % 4 == 0:
            avg_usage = sum(cpu_per_core) / len(cpu_per_core)
            print(f"[MONITOR] Sample {i//2}s: Avg={avg_usage:.1f}% | "
                  f"Min={min(cpu_per_core):.1f}% | Max={max(cpu_per_core):.1f}%")

    # Calculate average usage per core
    avg_per_core = [sum(s[i] for s in samples) / len(samples) for i in range(cpu_count)]

    print(f"\n[MONITOR] Average CPU usage per core over {duration_seconds}s:")
    for i, avg in enumerate(avg_per_core):
        print(f"  Core {i:2d}: {avg:5.1f}%")

    overall_avg = sum(avg_per_core) / len(avg_per_core)
    cores_above_50 = sum(1 for avg in avg_per_core if avg > 50)
    cores_above_80 = sum(1 for avg in avg_per_core if avg > 80)

    print(f"\n[MONITOR] Summary:")
    print(f"  Average across all cores: {overall_avg:.1f}%")
    print(f"  Cores above 50% usage: {cores_above_50}/{cpu_count}")
    print(f"  Cores above 80% usage: {cores_above_80}/{cpu_count}")

    return avg_per_core

if __name__ == '__main__':
    print("\n" + "="*80)
    print("TEST 1: Single model with n_jobs=-1 (should use ALL cores)")
    print("="*80)

    # Start CPU monitor in background
    monitor_process = mp.Process(target=monitor_cpu, args=(5,))
    monitor_process.start()
    time.sleep(0.5)

    # Train single model
    duration_single = train_one_model(1, n_jobs=-1)

    monitor_process.join()
    print(f"\n[RESULT] Single model with all cores: {duration_single:.2f}s")

    time.sleep(2)

    print("\n" + "="*80)
    print("TEST 2: 3 models in parallel, each with n_jobs=-1")
    print("      (This will cause CONTENTION - all fighting for cores)")
    print("="*80)

    # Start CPU monitor
    monitor_process = mp.Process(target=monitor_cpu, args=(8,))
    monitor_process.start()
    time.sleep(0.5)

    # Train 3 models in parallel using multiprocessing (NOT threading!)
    with mp.Pool(processes=3) as pool:
        results = pool.starmap(train_one_model, [(i, -1) for i in range(2, 5)])

    monitor_process.join()
    parallel_time = max(results)
    print(f"\n[RESULT] 3 models parallel (each n_jobs=-1): {parallel_time:.2f}s")

    time.sleep(2)

    print("\n" + "="*80)
    print("TEST 3: 3 models in parallel, each with n_jobs=4")
    print("      (12 total cores / 3 models = 4 cores each - NO contention)")
    print("="*80)

    cores_per_model = cpu_count // 3
    print(f"[INFO] Giving each model {cores_per_model} cores")

    # Start CPU monitor
    monitor_process = mp.Process(target=monitor_cpu, args=(8,))
    monitor_process.start()
    time.sleep(0.5)

    # Train 3 models in parallel, each with limited cores
    with mp.Pool(processes=3) as pool:
        results = pool.starmap(train_one_model, [(i, cores_per_model) for i in range(5, 8)])

    monitor_process.join()
    parallel_limited = max(results)
    print(f"\n[RESULT] 3 models parallel (each n_jobs={cores_per_model}): {parallel_limited:.2f}s")

    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"Single model (all cores):           {duration_single:.2f}s")
    print(f"3 models parallel (fight for cores): {parallel_time:.2f}s")
    print(f"3 models parallel (divided cores):   {parallel_limited:.2f}s")
    print(f"\nSpeedup (fighting): {duration_single / parallel_time:.2f}x")
    print(f"Speedup (divided):  {duration_single / parallel_limited:.2f}x")

    if parallel_limited < parallel_time:
        print(f"\n[INSIGHT] Dividing cores is FASTER by {parallel_time - parallel_limited:.2f}s!")
        print("[INSIGHT] This proves contention was limiting performance!")
    else:
        print(f"\n[INSIGHT] Fighting for cores was faster (less overhead)")

    print("\n[CONCLUSION] For VPS with 250 cores:")
    print("  - Train 250 models with n_jobs=1 each (no contention)")
    print("  - OR train 25 models with n_jobs=10 each")
    print("  - AVOID having all models fight for all cores!")

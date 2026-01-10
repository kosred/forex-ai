#!/usr/bin/env python3
"""
Simple test: Does LightGBM with n_jobs=-1 actually use all cores?
"""

import time
import psutil
import numpy as np
import pandas as pd
import lightgbm as lgb
import threading

print("="*80)
print("SIMPLE CORE USAGE TEST")
print("="*80)

cpu_count = psutil.cpu_count(logical=True)
print(f"\nTotal CPU cores: {cpu_count}")

# Create large dataset to ensure it takes time
n_samples = 50000
n_features = 100
print(f"Creating large dataset ({n_samples} samples, {n_features} features)...")

X = pd.DataFrame(np.random.randn(n_samples, n_features))
y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

# Monitor function
stop_monitoring = False
cpu_samples = []

def monitor():
    global stop_monitoring, cpu_samples
    print("\n[MONITOR] Starting... (Ctrl+C to stop early)\n")

    while not stop_monitoring:
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        cpu_samples.append(cpu_per_core)

        # Print current usage
        avg = sum(cpu_per_core) / len(cpu_per_core)
        active_cores = sum(1 for c in cpu_per_core if c > 50)
        print(f"[MONITOR] Avg: {avg:5.1f}% | Active cores (>50%): {active_cores}/{cpu_count} | "
              f"All cores: {' '.join([f'{c:3.0f}' for c in cpu_per_core])}%")

print("\n" + "="*80)
print(f"TEST: Training LightGBM with n_jobs=-1 (should use all {cpu_count} cores)")
print("="*80)

# Start monitoring
monitor_thread = threading.Thread(target=monitor, daemon=True)
monitor_thread.start()

time.sleep(2)  # Let monitor start

# Train model
print("\n[TRAIN] Starting LightGBM training...")
model = lgb.LGBMClassifier(
    n_estimators=500,  # Lots of trees to make it slow
    n_jobs=-1,  # Use ALL cores
    verbose=-1,
    max_depth=10,
)

start = time.time()
model.fit(X, y)
duration = time.time() - start

print(f"\n[TRAIN] Completed in {duration:.2f}s")

# Stop monitoring
stop_monitoring = True
time.sleep(1.5)

# Analyze results
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if cpu_samples:
    # Calculate average per core across all samples
    avg_per_core = []
    for core_idx in range(cpu_count):
        core_samples = [sample[core_idx] for sample in cpu_samples]
        avg_per_core.append(sum(core_samples) / len(core_samples))

    print("\nAverage CPU usage per core during training:")
    for i, avg in enumerate(avg_per_core):
        bar = "#" * int(avg / 5)  # Each # = 5%
        print(f"  Core {i:2d}: {avg:5.1f}% {bar}")

    overall_avg = sum(avg_per_core) / len(avg_per_core)
    cores_above_30 = sum(1 for avg in avg_per_core if avg > 30)
    cores_above_50 = sum(1 for avg in avg_per_core if avg > 50)
    cores_above_80 = sum(1 for avg in avg_per_core if avg > 80)

    print(f"\n[SUMMARY]")
    print(f"  Overall average: {overall_avg:.1f}%")
    print(f"  Cores >30%: {cores_above_30}/{cpu_count} ({cores_above_30/cpu_count*100:.0f}%)")
    print(f"  Cores >50%: {cores_above_50}/{cpu_count} ({cores_above_50/cpu_count*100:.0f}%)")
    print(f"  Cores >80%: {cores_above_80}/{cpu_count} ({cores_above_80/cpu_count*100:.0f}%)")

    if cores_above_50 >= cpu_count * 0.8:
        print(f"\n[SUCCESS] LightGBM used {cores_above_50}/{cpu_count} cores heavily!")
        print("[INFO] Multi-core training is working!")
    elif cores_above_30 >= cpu_count * 0.5:
        print(f"\n[PARTIAL] LightGBM used {cores_above_30}/{cpu_count} cores moderately")
        print("[WARN] Not all cores utilized fully")
    else:
        print(f"\n[FAIL] Only {cores_above_30}/{cpu_count} cores used!")
        print("[ERROR] Multi-core training NOT working properly")
else:
    print("[ERROR] No CPU samples collected")

print("\n" + "="*80)

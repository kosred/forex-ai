#!/usr/bin/env python3
"""
Test TRUE parallel training - proves multi-core usage
Uses psutil to monitor CPU usage per core
"""

import sys
import time
import threading
import multiprocessing
import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    print("[FAIL] psutil not installed: pip install psutil")
    sys.exit(1)

print("="*80)
print("TRUE PARALLEL TRAINING TEST")
print("="*80)

# Get CPU info
cpu_count_physical = psutil.cpu_count(logical=False)
cpu_count_logical = psutil.cpu_count(logical=True)
print(f"\n[INFO] Physical CPU cores: {cpu_count_physical}")
print(f"[INFO] Logical CPU cores: {cpu_count_logical}")

# Create sample data
n_samples = 10000
n_features = 50

print(f"\n[INFO] Creating sample data ({n_samples} samples, {n_features} features)...")
X = pd.DataFrame(np.random.randn(n_samples, n_features))
y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

def train_lightgbm(model_id):
    """Train LightGBM with n_jobs=-1 (all cores for THIS model)"""
    import lightgbm as lgb

    print(f"[START] Thread {threading.current_thread().name}: Training LightGBM model {model_id}")

    model = lgb.LGBMClassifier(
        n_estimators=100,
        n_jobs=-1,  # Use all cores for this model
        verbose=-1
    )
    model.fit(X, y)

    print(f"[DONE]  Thread {threading.current_thread().name}: Completed model {model_id}")
    return model

def monitor_cpu():
    """Monitor CPU usage across all cores"""
    print("\n[MONITOR] Starting CPU monitoring...")
    print(f"[MONITOR] Will show CPU % for first 8 cores every second\n")

    for i in range(15):  # Monitor for 15 seconds
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)

        # Show first 8 cores
        cores_display = " | ".join([f"Core{i}:{pct:5.1f}%" for i, pct in enumerate(cpu_per_core[:8])])
        print(f"[MONITOR] {cores_display}")

        time.sleep(0.1)

# Start CPU monitor in background thread
monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
monitor_thread.start()

time.sleep(2)  # Let monitor start

print("\n[TEST 1] SEQUENTIAL Training (1 model at a time)")
print("-" * 80)
start = time.time()

for i in range(3):
    train_lightgbm(i)

sequential_time = time.time() - start
print(f"\n[RESULT] Sequential training took: {sequential_time:.2f}s")

time.sleep(3)

print("\n[TEST 2] PARALLEL Training (3 models simultaneously)")
print("-" * 80)
start = time.time()

# Train 3 models in parallel using threading
threads = []
for i in range(3, 6):
    t = threading.Thread(target=train_lightgbm, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

parallel_time = time.time() - start
print(f"\n[RESULT] Parallel training took: {parallel_time:.2f}s")

# Results
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Sequential time: {sequential_time:.2f}s")
print(f"Parallel time:   {parallel_time:.2f}s")
print(f"Speedup:         {sequential_time/parallel_time:.2f}x")

if parallel_time < sequential_time * 0.5:
    print("\n[SUCCESS] TRUE parallel execution confirmed!")
    print("[INFO] Multiple cores were used simultaneously")
else:
    print("\n[WARN] Parallelism may be limited by GIL or CPU")
    print("[INFO] Check the CPU monitor output above")

time.sleep(3)  # Let monitor finish

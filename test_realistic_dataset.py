#!/usr/bin/env python3
"""
REALISTIC TEST: Use a LARGE dataset like real forex data
Previous tests used tiny 10K rows - too small to see real parallelism
"""

import time
import threading
import numpy as np
import pandas as pd
import lightgbm as lgb
import psutil

print("="*80)
print("REALISTIC DATASET TEST - Does parallelism work at scale?")
print("="*80)

cpu_count = psutil.cpu_count(logical=True)
print(f"\nCPU cores: {cpu_count}")

# Create REALISTIC forex-sized dataset
print("\n[DATA] Creating LARGE dataset (like real forex data)...")
n_samples = 1_000_000  # 1 million rows (realistic for forex)
n_features = 200       # 200 technical indicators

print(f"  Samples: {n_samples:,}")
print(f"  Features: {n_features}")
print(f"  Total data points: {n_samples * n_features:,}")
print(f"  Memory: ~{(n_samples * n_features * 8) / (1024**3):.1f} GB")

X = pd.DataFrame(np.random.randn(n_samples, n_features))
y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

print("  Data created!")

def train_one_model(model_id):
    """Train one LightGBM model"""
    model = lgb.LGBMClassifier(
        n_estimators=200,  # More trees
        n_jobs=-1,
        verbose=-1,
        max_depth=10,
    )

    start = time.time()
    model.fit(X, y)
    duration = time.time() - start

    print(f"  Model {model_id}: {duration:.2f}s")
    return duration

# TEST 1: Single model (baseline)
print("\n" + "="*80)
print("TEST 1: Single Model (Baseline)")
print("="*80)

start = time.time()
single_time = train_one_model(1)
print(f"\n[RESULT] Single model: {single_time:.2f}s")

# TEST 2: 3 models in parallel
print("\n" + "="*80)
print("TEST 2: 3 Models in Parallel (Threading)")
print("="*80)

results = []
def train_and_record(model_id):
    duration = train_one_model(model_id)
    results.append(duration)

start = time.time()
threads = [threading.Thread(target=train_and_record, args=(i,)) for i in range(2, 5)]
for t in threads:
    t.start()
for t in threads:
    t.join()
parallel_time = time.time() - start

print(f"\n[RESULT] 3 models parallel: {parallel_time:.2f}s")

# TEST 3: 6 models in parallel (saturate cores)
print("\n" + "="*80)
print("TEST 3: 6 Models in Parallel (Full saturation)")
print("="*80)

results = []
start = time.time()
threads = [threading.Thread(target=train_and_record, args=(i,)) for i in range(5, 11)]
for t in threads:
    t.start()
for t in threads:
    t.join()
saturated_time = time.time() - start

print(f"\n[RESULT] 6 models parallel: {saturated_time:.2f}s")

# ANALYSIS
print("\n" + "="*80)
print("ANALYSIS - REALISTIC EXPECTATIONS")
print("="*80)

speedup_3 = single_time / parallel_time
speedup_6 = single_time / saturated_time

efficiency_3 = (speedup_3 / 3) * 100
efficiency_6 = (speedup_6 / 6) * 100

print(f"\nSingle model:       {single_time:6.2f}s (baseline)")
print(f"3 models parallel:  {parallel_time:6.2f}s (speedup: {speedup_3:.2f}x, efficiency: {efficiency_3:.1f}%)")
print(f"6 models parallel:  {saturated_time:6.2f}s (speedup: {speedup_6:.2f}x, efficiency: {efficiency_6:.1f}%)")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if speedup_3 > 2.5:
    print("✓ EXCELLENT: 3 models gave >2.5x speedup")
    print("  Multi-core parallelism is working well!")
elif speedup_3 > 2.0:
    print("✓ GOOD: 3 models gave >2.0x speedup")
    print("  Decent parallelism, some overhead/contention")
elif speedup_3 > 1.5:
    print("⚠ MEDIOCRE: 3 models gave 1.5-2.0x speedup")
    print("  Limited parallelism, significant overhead")
else:
    print("✗ POOR: 3 models gave <1.5x speedup")
    print("  Parallelism barely working, mostly sequential")

print(f"\nFor VPS with 250 cores:")
if speedup_3 > 2.0:
    print(f"  Expected speedup with 10 models: ~{speedup_3 * 3:.1f}x")
    print(f"  Training time improvement: Significant!")
elif speedup_3 > 1.5:
    print(f"  Expected speedup with 10 models: ~{speedup_3 * 2:.1f}x")
    print(f"  Training time improvement: Moderate")
else:
    print(f"  Expected speedup with 10 models: ~{speedup_3 * 1.5:.1f}x")
    print(f"  Training time improvement: Limited")
    print("  ⚠ WARNING: May not benefit much from 250 cores")

print("\n" + "="*80)
print("THE TRUTH")
print("="*80)
print("""
If efficiency is <70%:
  - Yes, we're using multiple cores
  - But NO, we're not using them EFFICIENTLY
  - Memory bandwidth, GIL, contention are bottlenecks
  - 250 cores won't give 250x speedup (not even close)

Realistic expectation for VPS:
  - Tree models: 10-20x speedup (not 40x)
  - Limited by memory bandwidth and I/O
  - Still valuable, but not magical
""")

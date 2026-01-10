#!/usr/bin/env python3
"""
DEFINITIVE TEST: Show EACH core individually
This proves whether ALL cores are used or just ONE core
"""

import time
import psutil
import numpy as np
import pandas as pd
import lightgbm as lgb
import os

print("="*80)
print("PER-CORE USAGE TEST - See Each Core Individually")
print("="*80)

cpu_count = psutil.cpu_count(logical=True)
physical_count = psutil.cpu_count(logical=False)
print(f"\nPhysical cores: {physical_count}")
print(f"Logical cores (with HT): {cpu_count}")

print("\n" + "="*80)
print("TASK MANAGER SETUP:")
print("="*80)
print("1. Open Task Manager (Ctrl+Shift+Esc)")
print("2. Performance tab -> CPU")
print("3. RIGHT-CLICK on the CPU graph")
print("4. Select 'Change graph to' -> 'Logical processors'")
print("5. You should now see 12 SMALL GRAPHS (one per core)")
print("")
print("Press Enter when ready...")
input()

# Create dataset
print("\n[SETUP] Creating dataset...")
n_samples = 100000
n_features = 100
X = pd.DataFrame(np.random.randn(n_samples, n_features))
y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

def print_per_core_usage(label):
    """Print usage for EACH core individually"""
    cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
    avg = sum(cpu_per_core) / len(cpu_per_core)
    cores_above_80 = sum(1 for c in cpu_per_core if c > 80)

    print(f"\n{label}")
    print(f"  Overall average: {avg:5.1f}%")
    print(f"  Cores above 80%: {cores_above_80}/{cpu_count}")
    print(f"\n  Individual core usage:")

    # Print in grid format
    for i in range(0, cpu_count, 4):
        row = []
        for j in range(4):
            if i + j < cpu_count:
                core_num = i + j
                usage = cpu_per_core[core_num]
                bar = "#" * int(usage / 5)
                row.append(f"    Core {core_num:2d}: {usage:5.1f}% {bar}")
        print("\n".join(row))

# BASELINE
print("\n" + "="*80)
print("BASELINE (No work - should be LOW on ALL cores)")
print("="*80)
print_per_core_usage("[BASELINE]")

# TRAINING
print("\n" + "="*80)
print("TRAINING LightGBM with n_jobs=-1")
print("="*80)
print("\nWatch Task Manager NOW - All 12 small graphs should spike!")
print("Starting in 3 seconds...")
time.sleep(3)

print("\n[TRAINING...]\n")

# Monitor DURING training
import threading

monitoring = True
def monitor_during_training():
    while monitoring:
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        cores_above_80 = sum(1 for c in cpu_per_core if c > 80)
        avg = sum(cpu_per_core) / len(cpu_per_core)

        # Show compact view
        core_str = " ".join([f"{c:3.0f}" for c in cpu_per_core])
        print(f"[LIVE] Avg:{avg:5.1f}% | >80%: {cores_above_80:2d}/{cpu_count} | All: {core_str}%")

monitor_thread = threading.Thread(target=monitor_during_training, daemon=True)
monitor_thread.start()

time.sleep(1)

# Train
model = lgb.LGBMClassifier(
    n_estimators=500,
    n_jobs=-1,
    verbose=-1,
    max_depth=10,
)

start = time.time()
model.fit(X, y)
duration = time.time() - start

monitoring = False
time.sleep(1.5)

print(f"\n[COMPLETED] Training took {duration:.2f}s")

# POST-TRAINING
print("\n" + "="*80)
print("POST-TRAINING (Work done - should DROP on all cores)")
print("="*80)
print_per_core_usage("[POST-TRAINING]")

# ANALYSIS
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("""
If ONLY 1 core was used:
  - You'd see 1 core at ~100%
  - Other 11 cores at <20%
  - Task Manager: Only 1 small graph would spike

If ALL cores were used:
  - You'd see ALL 12 cores at 80-100%
  - Task Manager: All 12 small graphs spike together

What did YOU see in Task Manager during training?
""")

answer = input("How many cores spiked in Task Manager? (1, few, or all): ").strip().lower()

if "all" in answer or "12" in answer:
    print("\n✓ CONFIRMED: All cores were used!")
    print("✓ Multi-core training is REAL")
elif "1" in answer:
    print("\n✗ PROBLEM: Only 1 core used")
    print("✗ Need to investigate why parallelism isn't working")
else:
    print(f"\n? PARTIAL: Some cores used (not all)")
    print("? May have some parallelism but not optimal")

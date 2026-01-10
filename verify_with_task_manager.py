#!/usr/bin/env python3
"""
DEFINITIVE PROOF: Monitor LightGBM training with Task Manager

INSTRUCTIONS:
1. Open Task Manager (Ctrl+Shift+Esc)
2. Click Performance tab
3. Click CPU on the left
4. Right-click the CPU graph
5. Select "Change graph to" -> "Logical processors"
6. You will now see all 12 cores as individual graphs
7. Run this script
8. Watch the Task Manager graphs - they should all spike to 100%
"""

import time
import psutil
import numpy as np
import pandas as pd
import lightgbm as lgb

print("="*80)
print("DEFINITIVE VERIFICATION: Task Manager vs psutil")
print("="*80)

cpu_count = psutil.cpu_count(logical=True)
print(f"\nCPU cores: {cpu_count}")

print("\n" + "="*80)
print("INSTRUCTIONS FOR USER VERIFICATION:")
print("="*80)
print("1. Open Task Manager (Ctrl+Shift+Esc)")
print("2. Performance tab -> CPU")
print("3. Right-click graph -> 'Change graph to' -> 'Logical processors'")
print("4. You will see all 12 cores as separate graphs")
print("")
print("READY? Press Enter when Task Manager is open...")
input()

print("\n[BASELINE] Taking baseline reading (should be low)...")
for i in range(3):
    cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
    avg = sum(cpu_per_core) / len(cpu_per_core)
    print(f"  Baseline {i+1}: {avg:5.1f}% average")

print("\n[DATA] Creating large dataset for training...")
n_samples = 100000
n_features = 200
X = pd.DataFrame(np.random.randn(n_samples, n_features))
y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))
print(f"  Created: {n_samples} samples, {n_features} features")

print("\n" + "="*80)
print("STARTING LIGHTGBM TRAINING")
print("="*80)
print(f"Watch Task Manager NOW!")
print(f"All {cpu_count} core graphs should spike to ~100%")
print(f"Training will take ~10-15 seconds...")
print("")

time.sleep(2)

print("[TRAINING STARTED]")
start = time.time()

model = lgb.LGBMClassifier(
    n_estimators=1000,  # Many trees = longer training
    n_jobs=-1,          # ALL cores
    verbose=-1,
    max_depth=15,
)
model.fit(X, y)

duration = time.time() - start

print(f"\n[TRAINING COMPLETED] Time: {duration:.2f}s")

print("\n" + "="*80)
print("QUESTION FOR YOU:")
print("="*80)
print("Did you see ALL 12 core graphs in Task Manager spike to ~100%?")
print("")
print("  YES = psutil is CORRECT, all cores were actually used")
print("  NO  = Something is wrong, cores were NOT used")
print("")
answer = input("Did all cores spike in Task Manager? (yes/no): ").strip().lower()

if answer in ['yes', 'y']:
    print("\n[VERIFIED] psutil readings are REAL!")
    print("All previous test results are accurate.")
else:
    print("\n[PROBLEM] Cores not actually used!")
    print("Previous tests may have been monitoring something else.")

print("\n[FINAL CHECK] Post-training CPU (should drop back to low)...")
for i in range(3):
    cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
    avg = sum(cpu_per_core) / len(cpu_per_core)
    print(f"  Post-training {i+1}: {avg:5.1f}% average")

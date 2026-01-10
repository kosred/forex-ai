#!/usr/bin/env python3
"""
DEFINITIVE TEST: Are 3 models actually running at the SAME TIME?

If parallel:     3 models finish in ~55s (same as 1 model)
If sequential:   3 models finish in ~165s (3 × 55s)
Our result:      179s ≈ 165s → SEQUENTIAL!
"""

import time
import threading
import numpy as np
import pandas as pd
import lightgbm as lgb

print("="*80)
print("PROOF: Are Models Running at the SAME TIME?")
print("="*80)

# Large dataset
n_samples = 1_000_000
n_features = 200
X = pd.DataFrame(np.random.randn(n_samples, n_features))
y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

# Track when each model starts and ends
timestamps = {}

def train_with_timestamps(model_id):
    """Train and record exact start/end times"""
    timestamps[model_id] = {'start': time.time()}

    print(f"[{time.strftime('%H:%M:%S')}] Model {model_id} STARTED")

    model = lgb.LGBMClassifier(
        n_estimators=100,  # Fewer trees = faster, easier to see pattern
        n_jobs=-1,
        verbose=-1,
        max_depth=10,
    )
    model.fit(X, y)

    timestamps[model_id]['end'] = time.time()
    duration = timestamps[model_id]['end'] - timestamps[model_id]['start']

    print(f"[{time.strftime('%H:%M:%S')}] Model {model_id} FINISHED (took {duration:.1f}s)")

# Run 3 models with threading
print("\nStarting 3 models with threading...")
print("If they run in PARALLEL, all 3 should show STARTED at same time")
print("If they run SEQUENTIAL, they'll start one after another\n")

start_time = time.time()

threads = []
for i in range(1, 4):
    t = threading.Thread(target=train_with_timestamps, args=(i,))
    threads.append(t)

# Start all threads "simultaneously"
print(f"[{time.strftime('%H:%M:%S')}] Launching all 3 threads NOW...\n")
for t in threads:
    t.start()

for t in threads:
    t.join()

total_time = time.time() - start_time

# Analysis
print("\n" + "="*80)
print("TIMELINE ANALYSIS")
print("="*80)

# Calculate overlap
start_times = sorted([(id, data['start']) for id, data in timestamps.items()], key=lambda x: x[1])
end_times = sorted([(id, data['end']) for id, data in timestamps.items()], key=lambda x: x[1])

first_start = start_times[0][1]
last_start = start_times[-1][1]
first_end = end_times[0][1]
last_end = end_times[-1][1]

print("\nModel Start Times (relative to first):")
for model_id, start in start_times:
    relative = start - first_start
    print(f"  Model {model_id}: +{relative:.1f}s")

print("\nModel End Times (relative to first):")
for model_id, end in end_times:
    relative = end - first_start
    print(f"  Model {model_id}: +{relative:.1f}s")

# Check for overlap
print("\n" + "="*80)
print("OVERLAP ANALYSIS")
print("="*80)

start_spread = last_start - first_start
print(f"\nTime between first and last start: {start_spread:.1f}s")

if start_spread < 2:
    print("  → All models started within 2s of each other")
    print("  → Threading launched them together ✓")
else:
    print("  → Models started >2s apart")
    print("  → Models starting sequentially ✗")

# Check if they actually ran at the same time
model_1_running = (timestamps[1]['start'], timestamps[1]['end'])
model_2_running = (timestamps[2]['start'], timestamps[2]['end'])
model_3_running = (timestamps[3]['start'], timestamps[3]['end'])

# Calculate overlap between models 1 and 2
overlap_12 = min(model_1_running[1], model_2_running[1]) - max(model_1_running[0], model_2_running[0])
overlap_23 = min(model_2_running[1], model_3_running[1]) - max(model_2_running[0], model_3_running[0])
overlap_13 = min(model_1_running[1], model_3_running[1]) - max(model_1_running[0], model_3_running[0])

print(f"\nTime overlap between models:")
print(f"  Model 1 & 2: {max(0, overlap_12):.1f}s")
print(f"  Model 2 & 3: {max(0, overlap_23):.1f}s")
print(f"  Model 1 & 3: {max(0, overlap_13):.1f}s")

avg_duration = sum([timestamps[i]['end'] - timestamps[i]['start'] for i in range(1, 4)]) / 3

if overlap_12 > avg_duration * 0.8 and overlap_23 > avg_duration * 0.8:
    print("\n  → Models had >80% overlap")
    print("  → TRUE PARALLEL execution ✓")
elif overlap_12 > avg_duration * 0.3:
    print("\n  → Models had 30-80% overlap")
    print("  → PARTIAL parallel execution")
else:
    print("\n  → Models had <30% overlap")
    print("  → MOSTLY SEQUENTIAL execution ✗")

# Final verdict
print("\n" + "="*80)
print("VERDICT")
print("="*80)

individual_time = avg_duration
expected_sequential = individual_time * 3
expected_parallel = individual_time

print(f"\nAverage single model time: {individual_time:.1f}s")
print(f"Expected if PARALLEL:      {expected_parallel:.1f}s")
print(f"Expected if SEQUENTIAL:    {expected_sequential:.1f}s")
print(f"ACTUAL total time:         {total_time:.1f}s")

efficiency = individual_time / total_time * 3

print(f"\nEfficiency: {efficiency:.1%}")

if total_time < individual_time * 1.3:
    print("\n✓✓✓ TRUE PARALLEL: All 3 models ran at the same time!")
elif total_time < individual_time * 2.0:
    print("\n✓✓ PARTIAL PARALLEL: Some overlap, not fully parallel")
elif total_time < individual_time * 2.7:
    print("\n✓ WEAK PARALLEL: Mostly sequential, small overlap")
else:
    print("\n✗✗✗ SEQUENTIAL: Models ran one after another!")
    print("    Threading did NOT achieve parallelism")

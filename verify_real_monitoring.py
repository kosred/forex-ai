#!/usr/bin/env python3
"""
VERIFICATION: Does psutil actually monitor REAL CPU usage?
This test compares psutil readings with actual CPU work
"""

import time
import psutil
import threading

print("="*80)
print("VERIFICATION: Is psutil monitoring REAL CPU usage?")
print("="*80)

cpu_count = psutil.cpu_count(logical=True)
print(f"\nCPU cores: {cpu_count}")

def cpu_burner(duration):
    """Intentionally waste CPU cycles"""
    end_time = time.time() + duration
    count = 0
    while time.time() < end_time:
        count += 1  # Busy loop - wastes CPU
        _ = count ** 2

def monitor_and_print():
    """Monitor CPU usage in real-time"""
    print("\n[BASELINE] CPU usage when IDLE (no work):")
    for i in range(3):
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        avg = sum(cpu_per_core) / len(cpu_per_core)
        print(f"  Sample {i+1}: Avg={avg:5.1f}% | All cores: {' '.join([f'{c:3.0f}' for c in cpu_per_core])}%")

    print("\n[STARTING] Launching CPU burner threads...")

    # Launch CPU burners on 6 cores
    threads = []
    for i in range(6):
        t = threading.Thread(target=cpu_burner, args=(5,))
        t.start()
        threads.append(t)

    time.sleep(0.5)  # Let threads start

    print("\n[ACTIVE] CPU usage with 6 CPU burner threads running:")
    for i in range(5):
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        avg = sum(cpu_per_core) / len(cpu_per_core)
        active_cores = sum(1 for c in cpu_per_core if c > 50)
        print(f"  Sample {i+1}: Avg={avg:5.1f}% | Active (>50%): {active_cores}/{cpu_count} | All: {' '.join([f'{c:3.0f}' for c in cpu_per_core])}%")

    for t in threads:
        t.join()

    print("\n[STOPPED] CPU burners stopped, checking if usage drops...")
    for i in range(3):
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        avg = sum(cpu_per_core) / len(cpu_per_core)
        print(f"  Sample {i+1}: Avg={avg:5.1f}% | All cores: {' '.join([f'{c:3.0f}' for c in cpu_per_core])}%")

print("\n" + "="*80)
print("TEST: Can psutil detect REAL CPU load changes?")
print("="*80)

monitor_and_print()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
If psutil is REAL monitoring:
  - BASELINE should show LOW usage (< 20%)
  - ACTIVE should show HIGH usage (> 50%)
  - STOPPED should drop back to LOW usage

If psutil readings change with actual work, it's REAL monitoring!
""")

print("\n[VERIFY YOURSELF]")
print("1. Open Task Manager (Ctrl+Shift+Esc)")
print("2. Go to Performance tab → CPU")
print("3. Run this script again")
print("4. Watch if Task Manager CPU graph matches psutil readings")
print("5. You should see CPU spike when burner threads are active")

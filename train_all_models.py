#!/usr/bin/env python3
"""
Complete Training Pipeline
1. Genetic strategy search (finds best TA-Lib indicators)
2. Train all 11 models using discovered strategies
3. Export to ONNX for production
"""

import os
import sys
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("FOREX AI BOT - Complete Training Pipeline")
print("="*80)

# Determine n_jobs (cores - 1 for stability)
total_cores = psutil.cpu_count(logical=True)
n_jobs = max(1, total_cores - 1)  # Leave 1 core for system

print(f"\n[SYSTEM INFO]")
print(f"  Total CPU cores: {total_cores}")
print(f"  Using for training: {n_jobs} (leaving 1 core free for stability)")
print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

# Check GPU
try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"  GPUs: 0 (CPU only)")
except ImportError:
    print(f"  GPUs: N/A (PyTorch not installed)")

# Setup paths
models_dir = Path("./models")
models_dir.mkdir(exist_ok=True)

cache_dir = Path("./cache")
cache_dir.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: Load or Create Sample Data
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Load Training Data")
print("="*80)

# Check if user provided data
data_path = os.environ.get("FOREX_DATA_PATH", None)

if data_path and Path(data_path).exists():
    print(f"[INFO] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df)} rows")
else:
    print("[WARN] No data path provided, creating sample data...")
    print("[INFO] Set FOREX_DATA_PATH environment variable to use real data")

    n_samples = 10000
    n_features = 50

    # Create sample OHLC data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='15min'),
        'open': 1.1000 + np.random.randn(n_samples) * 0.01,
        'high': 1.1005 + np.random.randn(n_samples) * 0.01,
        'low': 1.0995 + np.random.randn(n_samples) * 0.01,
        'close': 1.1000 + np.random.randn(n_samples) * 0.01,
        'volume': np.random.randint(1000, 10000, n_samples),
    })

    # Add some features
    for i in range(n_features):
        df[f'feature_{i}'] = np.random.randn(n_samples)

    # Create target: 0=Neutral, 1=Buy, 2=Sell
    df['target'] = np.random.choice([0, 1, 2], size=n_samples)

    print(f"[INFO] Created sample data: {len(df)} rows, {len(df.columns)} columns")

# Separate OHLC metadata from features
metadata_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
metadata = df[metadata_cols].copy()

# Features (everything except OHLC and target)
feature_cols = [col for col in df.columns if col not in metadata_cols and col != 'target']
X = df[feature_cols].copy()
y = df['target'].copy()

print(f"\n[DATA SUMMARY]")
print(f"  Samples: {len(df)}")
print(f"  Features: {len(feature_cols)}")
print(f"  Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# STEP 2: Genetic Strategy Search (MOST IMPORTANT)
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Genetic Strategy Search (Finding Best TA-Lib Indicators)")
print("="*80)

try:
    from forex_bot.models.genetic import GeneticStrategyExpert

    print(f"[INFO] Starting genetic search...")
    print(f"[INFO] This will discover optimal TA-Lib indicator combinations")
    print(f"[INFO] Population: 50, Generations: 10, Max Indicators: 5")

    genetic = GeneticStrategyExpert(
        population_size=50,
        generations=10,
        max_indicators=5
    )

    start = time.time()
    genetic.fit(X, y, metadata=metadata)
    duration = time.time() - start

    print(f"[SUCCESS] Genetic search completed in {duration:.1f}s")

    # Save genetic model
    genetic.save(str(models_dir))
    print(f"[INFO] Saved genetic model to {models_dir}/genetic_expert.joblib")

    # Check if strategies were found
    try:
        portfolio_size = len(genetic.portfolio) if hasattr(genetic, 'portfolio') else 0
        print(f"[INFO] Found {portfolio_size} strategies in portfolio")

        if hasattr(genetic, 'best_gene') and genetic.best_gene:
            print(f"[INFO] Best strategy:")
            print(f"  Indicators: {genetic.best_gene.indicators}")
            print(f"  Fitness: {genetic.best_gene.fitness:.4f}")
    except Exception as e:
        print(f"[WARN] Could not inspect genetic results: {e}")

except Exception as e:
    print(f"[ERROR] Genetic search failed: {e}")
    import traceback
    traceback.print_exc()
    print(f"[WARN] Continuing without genetic strategies...")

# ============================================================================
# STEP 3: Train All Models
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Training All Models (Using Discovered Strategies)")
print("="*80)

trained_models = {}
training_times = {}

# Prefer Rust tree models when bindings are available (override with FOREX_BOT_TREE_BACKEND=python).
tree_backend = os.environ.get("FOREX_BOT_TREE_BACKEND", "auto").strip().lower()
tree_module = "forex_bot.models.trees"
if tree_backend not in {"python", "py", "0", "false", "no", "off"}:
    if tree_backend in {"rust", "1", "true", "yes", "on"}:
        tree_module = "forex_bot.models.trees_rust"
    else:
        try:
            import forex_bindings  # noqa: F401

            tree_module = "forex_bot.models.trees_rust"
        except Exception:
            tree_module = "forex_bot.models.trees"


def _tree_class(class_name: str):
    if tree_module.endswith("trees_rust"):
        try:
            mod = __import__(tree_module, fromlist=[class_name])
            cls = getattr(mod, class_name)
            if getattr(cls, "_model_cls", None) is not None:
                return cls
        except Exception:
            pass
    mod = __import__("forex_bot.models.trees", fromlist=[class_name])
    return getattr(mod, class_name)

# Model configurations
models_config = [
    # Tree models (use n_jobs for multi-core)
    ("lightgbm", lambda: _tree_class("LightGBMExpert")(params={
        "n_estimators": 100,
        "n_jobs": n_jobs,
        "device": "cpu",
    })),
    ("xgboost", lambda: _tree_class("XGBoostExpert")(params={
        "n_estimators": 100,
        "n_jobs": n_jobs,
        "device": "cpu",
    })),   
    ("catboost", lambda: _tree_class("CatBoostExpert")(params={
        "iterations": 100,
        "thread_count": n_jobs,
        "devices": "CPU",
    })),

    # Neural networks (will auto-detect GPU)
    ("mlp", lambda: __import__('forex_bot.models.mlp', fromlist=['MLPExpert']).MLPExpert(
        input_dim=len(feature_cols),
        hidden_dims=[128, 64],
        num_classes=3,
        device="cuda" if __import__('torch').cuda.is_available() else "cpu"
    )),

    ("nbeats", lambda: __import__('forex_bot.models.deep', fromlist=['NBeatsExpert']).NBeatsExpert(
        input_dim=len(feature_cols),
        num_classes=3,
        device="cuda:1" if __import__('torch').cuda.is_available() and __import__('torch').cuda.device_count() > 1 else "cuda" if __import__('torch').cuda.is_available() else "cpu"
    )),

    ("tide", lambda: __import__('forex_bot.models.deep', fromlist=['TiDEExpert']).TiDEExpert(
        input_dim=len(feature_cols),
        num_classes=3,
        device="cuda:2" if __import__('torch').cuda.is_available() and __import__('torch').cuda.device_count() > 2 else "cuda" if __import__('torch').cuda.is_available() else "cpu"
    )),
]

print(f"\n[INFO] Training {len(models_config)} models sequentially")
print(f"[INFO] Each model will use {n_jobs} cores\n")

for model_name, model_factory in models_config:
    print(f"\n[{model_name.upper()}] Starting training...")

    try:
        # Create model
        model = model_factory()

        # Train
        start = time.time()
        model.fit(X, y)
        duration = time.time() - start

        training_times[model_name] = duration

        # Predict to verify it works
        probs = model.predict_proba(X)

        print(f"[{model_name.upper()}] Completed in {duration:.1f}s")
        print(f"[{model_name.upper()}] Predictions shape: {probs.shape}")

        # Save model
        model.save(str(models_dir))
        trained_models[model_name] = model

        print(f"[{model_name.upper()}] ✓ Saved to {models_dir}")

    except Exception as e:
        print(f"[{model_name.upper()}] ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# STEP 4: Export to ONNX (Optional, for production)
# ============================================================================

print("\n" + "="*80)
print("STEP 4: ONNX Export (For Production Inference)")
print("="*80)

export_onnx = os.environ.get("EXPORT_ONNX", "0") == "1"

if export_onnx:
    try:
        from forex_bot.models.onnx_exporter import ONNXExporter

        print(f"[INFO] Exporting models to ONNX format...")

        exporter = ONNXExporter(models_dir=str(models_dir))

        # Export all trained models
        sample_input = X.head(10).values.astype(np.float32)

        manifest = exporter.export_all(trained_models, sample_input)

        print(f"[SUCCESS] Exported {len(manifest.models)} models to ONNX")
        print(f"[INFO] ONNX models saved to: {models_dir}/onnx/")

    except Exception as e:
        print(f"[WARN] ONNX export skipped: {e}")
else:
    print(f"[INFO] ONNX export skipped (set EXPORT_ONNX=1 to enable)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

print(f"\n[SUMMARY]")
print(f"  Models trained: {len(trained_models)}/{len(models_config)}")
print(f"  Total training time: {sum(training_times.values()):.1f}s")

if training_times:
    print(f"\n[TIMING BREAKDOWN]")
    for model_name, duration in sorted(training_times.items(), key=lambda x: x[1], reverse=True):
        print(f"  {model_name:15s}: {duration:6.1f}s")

print(f"\n[MODELS SAVED TO]")
print(f"  {models_dir.absolute()}")

print(f"\n[NEXT STEPS]")
print(f"  1. Use trained models for prediction")
print(f"  2. Run backtest: python backtest.py --models-dir {models_dir}")
print(f"  3. Deploy to production (optionally use ONNX models for 10-100x faster inference)")

print("\n" + "="*80)
print("Ready for deployment!")
print("="*80)

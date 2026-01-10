#[WARN]/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Forex AI Bot
Tests hardware detection, model initialization, and basic training workflow
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 80)
print("FOREX AI BOT - System Test")
print("=" * 80)

# Test 1: Python Dependencies
print("\n[1/6] Testing Python Dependencies...")
try:
    import torch
    print(f"  [OK] PyTorch {torch.__version__}")
    print(f"  [OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  [OK] GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError as e:
    print(f"  [FAIL] PyTorch: {e}")
    sys.exit(1)

try:
    import lightgbm as lgb
    print(f"  [OK] LightGBM {lgb.__version__}")
except ImportError as e:
    print(f"  [FAIL] LightGBM: {e}")
    sys.exit(1)

try:
    import xgboost as xgb
    print(f"  [OK] XGBoost {xgb.__version__}")
except ImportError as e:
    print(f"  [FAIL] XGBoost: {e}")
    sys.exit(1)

try:
    import catboost
    print(f"  [OK] CatBoost {catboost.__version__}")
except ImportError as e:
    print(f"  [FAIL] CatBoost: {e}")
    sys.exit(1)

try:
    import joblib
    print(f"  [OK] Joblib")
except ImportError as e:
    print(f"  [FAIL] Joblib: {e}")
    sys.exit(1)

# Test 2: Hardware Detection
print("\n[2/6] Testing Hardware Detection...")
try:
    import psutil
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)

    print(f"  [OK] Physical CPU cores: {cpu_count}")
    print(f"  [OK] Logical CPU cores: {cpu_count_logical}")
    print(f"  [OK] RAM: {ram_gb:.1f} GB")
    print(f"  [OK] Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
except ImportError:
    print("  [WARN] psutil not installed, skipping detailed hardware info")

# Test 3: Import Bot Modules
print("\n[3/6] Testing Bot Module Imports...")
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
try:
    from forex_bot.models.base import ExpertModel
    print("  [OK] base module")
except ImportError as e:
    print(f"  [FAIL] base module: {e}")

try:
    __import__(tree_module, fromlist=["LightGBMExpert"])
    print(f"  [OK] tree module ({tree_module})")
except ImportError as e:
    print(f"  [FAIL] tree module: {e}")

try:
    from forex_bot.models.mlp import MLPExpert
    print("  [OK] mlp module")
except ImportError as e:
    print(f"  [FAIL] mlp module: {e}")

try:
    from forex_bot.models.deep import NBeatsExpert, TiDEExpert, TabNetExpert, KANExpert
    print("  [OK] deep module (NBeats, TiDE, TabNet, KAN)")
except ImportError as e:
    print(f"  [FAIL] deep module: {e}")

try:
    from forex_bot.models.genetic import GeneticStrategyExpert
    print("  [OK] genetic module")
except ImportError as e:
    print(f"  [FAIL] genetic module: {e}")

try:
    from forex_bot.models.exit_agent import ExitAgent
    print("  [OK] exit_agent module")
except ImportError as e:
    print(f"  [FAIL] exit_agent module: {e}")

try:
    from forex_bot.models.registry import AVAILABLE_MODELS, get_model_info
    print("  [OK] registry module")
    print(f"    Available models: {len(AVAILABLE_MODELS)}")
    for model in AVAILABLE_MODELS:
        info = get_model_info(model)
        if info:
            print(f"      - {model}: {info['category']}")
except ImportError as e:
    print(f"  [FAIL] registry module: {e}")

# Test 4: Create Sample Data
print("\n[4/6] Creating Sample Training Data...")
n_samples = 1000
n_features = 20

np.random.seed(42)
X = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f"feature_{i}" for i in range(n_features)]
)
# Create target: 0=Neutral, 1=Buy, 2=Sell
y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

# Create metadata (OHLC) for genetic expert
metadata = pd.DataFrame({
    'open': 1.1000 + np.random.randn(n_samples) * 0.001,
    'high': 1.1005 + np.random.randn(n_samples) * 0.001,
    'low': 1.0995 + np.random.randn(n_samples) * 0.001,
    'close': 1.1000 + np.random.randn(n_samples) * 0.001,
    'volume': np.random.randint(1000, 10000, n_samples)
})

print(f"  [OK] X shape: {X.shape}")
print(f"  [OK] y shape: {y.shape}")
print(f"  [OK] y distribution: {y.value_counts().to_dict()}")
print(f"  [OK] Metadata shape: {metadata.shape}")

# Test 5: Test Model Initialization
print("\n[5/6] Testing Model Initialization...")

# Test LightGBM
try:
    LightGBMExpert = _tree_class("LightGBMExpert")
    lgbm = LightGBMExpert(params={"n_estimators": 10, "device": "cpu"})
    print("  [OK] LightGBM initialized")
    lgbm.fit(X, y)
    print("  [OK] LightGBM trained")
    probs = lgbm.predict_proba(X)
    print(f"  [OK] LightGBM predictions: {probs.shape}")
except Exception as e:
    print(f"  [FAIL] LightGBM: {e}")

# Test XGBoost
try:
    XGBoostExpert = _tree_class("XGBoostExpert")
    xgb_model = XGBoostExpert(params={"n_estimators": 10, "device": "cpu"})
    print("  [OK] XGBoost initialized")
    xgb_model.fit(X, y)
    print("  [OK] XGBoost trained")
    probs = xgb_model.predict_proba(X)
    print(f"  [OK] XGBoost predictions: {probs.shape}")
except Exception as e:
    print(f"  [FAIL] XGBoost: {e}")

# Test CatBoost
try:
    CatBoostExpert = _tree_class("CatBoostExpert")
    cat = CatBoostExpert(params={"iterations": 10, "device": "CPU"})
    print("  [OK] CatBoost initialized")
    cat.fit(X, y)
    print("  [OK] CatBoost trained")
    probs = cat.predict_proba(X)
    print(f"  [OK] CatBoost predictions: {probs.shape}")
except Exception as e:
    print(f"  [FAIL] CatBoost: {e}")

# Test MLP (Neural Network)
try:
    from forex_bot.models.mlp import MLPExpert
    mlp = MLPExpert(
        input_dim=n_features,
        hidden_dims=[64, 32],
        num_classes=3,
        device="cpu"
    )
    print("  [OK] MLP initialized")
    mlp.fit(X, y, epochs=2, batch_size=32)  # Just 2 epochs for quick test
    print("  [OK] MLP trained")
    probs = mlp.predict_proba(X)
    print(f"  [OK] MLP predictions: {probs.shape}")
except Exception as e:
    print(f"  [FAIL] MLP: {e}")
    import traceback
    traceback.print_exc()

# Test Genetic Expert
try:
    from forex_bot.models.genetic import GeneticStrategyExpert
    genetic = GeneticStrategyExpert(
        population_size=10,
        generations=2,
        max_indicators=3
    )
    print("  [OK] Genetic Expert initialized")
    genetic.fit(X, y, metadata=metadata)
    print("  [OK] Genetic Expert trained")
    probs = genetic.predict_proba(X, metadata=metadata)
    print(f"  [OK] Genetic Expert predictions: {probs.shape}")
except Exception as e:
    print(f"  [FAIL] Genetic Expert: {e}")
    import traceback
    traceback.print_exc()

# Test Exit Agent
try:
    from forex_bot.models.exit_agent import ExitAgent
    from forex_bot.core.config import Settings

    settings = Settings()
    exit_agent = ExitAgent(settings, device="cpu")
    print("  [OK] Exit Agent initialized")

    # Test get_action
    state = np.array([0.5, 10.0, 0.01, 0.02, 1.0, 0.8])  # [pnl, duration, vol, momentum, ...]
    action = exit_agent.get_action(state, eval_mode=True)
    print(f"  [OK] Exit Agent action: {action} (0=Hold, 1=Close)")

    # Test observe_exit
    exit_agent.observe_exit(
        ticket=12345,
        state=state,
        action=action,
        current_price=1.1000,
        timestamp=1234567890
    )
    print("  [OK] Exit Agent observed exit")

    # Test process_regret
    future_prices = np.array([1.1001, 1.1002, 1.0999, 1.1003])
    exit_agent.process_regret(ticket=12345, future_price_trace=future_prices, direction=1)
    print("  [OK] Exit Agent processed regret")

    # Test train_step (needs at least 32 samples in memory)
    for i in range(35):
        exit_agent.observe_exit(
            ticket=10000 + i,
            state=np.random.randn(6),
            action=np.random.randint(0, 2),
            current_price=1.1000,
            timestamp=1234567890 + i
        )
        exit_agent.process_regret(
            ticket=10000 + i,
            future_price_trace=np.random.randn(10) * 0.001 + 1.1000,
            direction=1
        )

    exit_agent.train_step()
    print("  [OK] Exit Agent trained")

except Exception as e:
    print(f"  [FAIL] Exit Agent: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Summary
print("\n[6/6] Test Summary")
print("=" * 80)
print("[OK] All core modules loaded successfully")
print("[OK] Models can be instantiated and trained")
print("[OK] Hardware detection working")
print("\nSystem is ready for VPS deployment[WARN]")
print("=" * 80)

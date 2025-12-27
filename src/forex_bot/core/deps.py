"""
Static Dependency Verifier.
Checks if critical packages are present and warns the user if setup is required.
Does NOT attempt runtime installation or process restarts.
"""

from __future__ import annotations
import importlib.metadata
import logging
import sys

logger = logging.getLogger(__name__)

CRITICAL_PACKAGES = [
    "numpy",
    "pandas",
    "torch",
    "cupy",
    "numba",
    "talib",
    "pydantic",
    "sklearn",
    "xgboost",
    "catboost",
]

def ensure_dependencies() -> None:
    """
    Verify that the environment is correctly set up.
    If packages are missing, it logs a warning but allows the bot to try and run.
    """
    missing = []
    for pkg in CRITICAL_PACKAGES:
        try:
            importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            # Check for alternative names used in some environments
            if pkg == "talib":
                try:
                    import talib
                except ImportError:
                    missing.append(pkg)
            elif pkg == "cupy":
                try:
                    import cupy
                except ImportError:
                    missing.append(pkg)
            else:
                missing.append(pkg)

    if missing:
        print("\n" + "!" * 60)
        print(f"[WARNING] Missing critical dependencies: {missing}")
        print("Please run the following command to set up your environment:")
        print("  python3 -m pip install -r requirements-hpc.txt --user --break-system-packages")
        print("!" * 60 + "\n")
    else:
        logger.info("✓ Environment dependencies verified.")
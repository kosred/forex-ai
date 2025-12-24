"""
Robust dependency bootstrapper with Dynamic Import Scanning.
Automatically detects hardware (GPU/CPU) and scans the codebase to install imported dependencies.
"""

from __future__ import annotations

import ast
import importlib.metadata as importlib_metadata
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
_DEPS_CHECKED = False

# Mapping of import names to PyPI package names
IMPORT_TO_PYPI = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "bs4": "beautifulsoup4",
    "metatrader5": "MetaTrader5",
    "ta": "ta-lib",  # ambiguous, usually ta-lib or ta
    "talib": "TA-Lib",
    "tensorboard": "tensorboard",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "gym": "gymnasium",  # modern preference
    "gymnasium": "gymnasium",
    "stable_baselines3": "stable-baselines3[extra]",
    "sb3_contrib": "sb3-contrib",
    "ray": "ray[default,rllib]",
    "optuna": "optuna",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "shap": "shap",
    "numba": "numba",
    "pandas": "pandas",
    "numpy": "numpy",
    "scipy": "scipy",
    "joblib": "joblib",
    "psutil": "psutil",
    "requests": "requests",
    "pydantic": "pydantic",
    "pydantic_settings": "pydantic-settings",
    "networkx": "networkx",
    "tqdm": "tqdm",
    "colorama": "colorama",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "sqlalchemy": "sqlalchemy",
    "alembic": "alembic",
    "click": "click",
    "typer": "typer",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "websockets": "websockets",
    "aiohttp": "aiohttp",
    "paramiko": "paramiko",
    "cryptography": "cryptography",
    "pynvml": "nvidia-ml-py",
    "cpuinfo": "py-cpuinfo",
    "feedparser": "feedparser",
    "numexpr": "numexpr",
    "pyarrow": "pyarrow",
    "efficient_kan": "efficient-kan",
    "neuralforecast": "neuralforecast",
    "tslib": "tslib",
    "pytorch_tabnet": "pytorch-tabnet",
}

# Special handling for git dependencies or non-standard sources
SPECIAL_SOURCES = {
    "efficient-kan": "git+https://github.com/Blealtan/efficient-kan.git",
    "stable-baselines3[extra]": "stable-baselines3[extra]>=2.4.0", # Allow pre-release
}

def scan_imports(root_dir: Path) -> set[str]:
    """
    Recursively scan python files in root_dir for import statements.
    Returns a set of top-level imported module names.
    """
    imports = set()
    for path in root_dir.rglob("*.py"):
        # Skip venvs and hidden dirs
        if any(p.startswith(".") or p in ("venv", "env", "build", "dist") for p in path.parts):
            continue
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(path))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
    return imports

def ensure_dependencies() -> None:
    """
    Main entry point: Scans code, checks installed packages, installs missing ones.
    """
    global _DEPS_CHECKED
    if _DEPS_CHECKED:
        return
    _DEPS_CHECKED = True

    # 1. Setup Environment info
    system = platform.system().lower()
    py_ver = sys.version_info
    is_py313 = py_ver >= (3, 13)
    
    logger.info(f"Scanning codebase for dependencies (Py {py_ver.major}.{py_ver.minor}, OS {system})...")

    # 2. Scan Imports
    src_root = Path(__file__).resolve().parent.parent.parent # src/forex_bot/core/.. -> src/forex_bot -> src
    imported_modules = scan_imports(src_root)
    
    # 3. Resolve to PyPI packages
    required_packages = set()
    for mod in imported_modules:
        if mod in sys.builtin_module_names:
            continue
        # Map module -> package
        pkg = IMPORT_TO_PYPI.get(mod, mod) # Default to module name if not in map
        
        # Filter exclusions
        if system != "windows" and pkg.lower() == "metatrader5":
            continue
        
        required_packages.add(pkg)

    # 4. Check what's missing
    missing = []
    for pkg in required_packages:
        # Strip extras for check: "ray[rllib]" -> "ray"
        base_pkg = pkg.split("[")[0].split(">")[0].split("=")[0]
        try:
            importlib_metadata.version(base_pkg)
        except importlib_metadata.PackageNotFoundError:
            missing.append(pkg)
            
    if not missing:
        logger.info("All dependencies appear to be satisfied.")
        return

    logger.info(f"Found {len(missing)} missing dependencies: {missing}")

    # 5. Install Logic
    # Separate PyTorch/GPU libs for special handling
    gpu_libs = []
    standard_libs = []
    
    for pkg in missing:
        if any(x in pkg for x in ["torch", "cupy"]):
            gpu_libs.append(pkg)
        else:
            standard_libs.append(pkg)

    # Install Standard Libs
    if standard_libs:
        logger.info("Installing standard dependencies...")
        _install(standard_libs, pre=is_py313)

    # Install GPU Libs (Special Index Handling)
    if gpu_libs:
        logger.info("Installing GPU dependencies...")
        # Simple heuristic for CUDA index
        index_url = "https://download.pytorch.org/whl/cpu"
        # Check nvidia-smi
        if shutil.which("nvidia-smi"):
             # For Py3.13 + GPU we might need nightly or specific index. 
             # Defaulting to nightly for 3.13 compatibility if needed.
             if is_py313:
                 index_url = "https://download.pytorch.org/whl/nightly/cu124"
             else:
                 index_url = "https://download.pytorch.org/whl/cu121"
        
        _install(gpu_libs, index_url=index_url, pre=True)

    # Restart
    if os.environ.get("FOREX_BOT_DEPS_REEXEC", "") != "1":
        os.environ["FOREX_BOT_DEPS_REEXEC"] = "1"
        logger.info("Restarting process to load new dependencies...")
        os.execv(sys.executable, [sys.executable, *sys.argv])

def _install(packages: list[str], index_url: str | None = None, pre: bool = False) -> None:
    if not packages: return
    
    cmd = [sys.executable, "-m", "pip", "install", "--timeout", "120"]
    cmd.append("--break-system-packages") # Modern Linux support
    
    if pre:
        cmd.append("--pre")
    if index_url:
        cmd.extend(["--index-url", index_url])
    
    # Process packages to swap in special sources (git urls)
    final_pkgs = []
    for p in packages:
        # Check EXACT match in special sources
        if p in SPECIAL_SOURCES:
            final_pkgs.append(SPECIAL_SOURCES[p])
        # Check base name match (e.g. efficient-kan in IMPORT_TO_PYPI maps to efficient-kan, which is in SPECIAL)
        elif p in SPECIAL_SOURCES:
             final_pkgs.append(SPECIAL_SOURCES[p])
        else:
             final_pkgs.append(p)
             
    cmd.extend(final_pkgs)
    
    logger.info(f"Running pip: {' '.join(cmd)}")
    subprocess.check_call(cmd)
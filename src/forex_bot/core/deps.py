"""
Robust dependency bootstrapper.
Automatically detects hardware (GPU/CPU) and OS to install the correct PyTorch and other dependencies.
"""

from __future__ import annotations

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
_APT_REFRESHED = False

# Target torch baseline
# For Python 3.13, we accept newer versions via --pre
TORCH_MIN_VERSION = "2.4.0"


def _get_cuda_version() -> str | None:
    """
    Detect CUDA version from nvidia-smi.
    Returns major.minor version string (e.g., '12.4') or None.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        # Parse output: "CUDA Version: 12.4"
        output = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
        for line in output.splitlines():
            if "CUDA Version" in line:
                # Extract version number
                parts = line.split("CUDA Version:")
                if len(parts) > 1:
                    version = parts[1].split()[0].strip()
                    return version
    except Exception:
        pass
    return None


def ensure_dependencies(requirements_path: str = "requirements.txt") -> None:
    """
    Check and auto-install missing dependencies based on Hardware/OS.
    Dynamic single-source-of-truth installation (ignores requirements.txt to prevent bloat).
    """
    global _DEPS_CHECKED
    if _DEPS_CHECKED:
        return
    _DEPS_CHECKED = True

    _cleanup_pip_leftovers()

    system = platform.system().lower()
    cuda_version = _get_cuda_version()
    has_gpu = cuda_version is not None
    
    # Python 3.13 Support: Enable pre-releases automatically
    py_ver = sys.version_info
    is_py313_plus = py_ver >= (3, 13)
    # Always allow pre-releases on 3.13 to catch alpha builds (like SB3 2.8.0a2)
    use_pre = is_py313_plus

    logger.info(f"Dependency Check: OS={system}, GPU={has_gpu} (CUDA {cuda_version or 'N/A'})")
    logger.info(f"Python Version: {platform.python_version()} (Pre-releases: {use_pre})")

    installed_any = False

    # 1. PyTorch (Hardware-Aware Dynamic Install)
    try:
        installed_ver = importlib_metadata.version("torch")
        # Simple check: if installed version is too old and we aren't on bleeding edge 3.13, complain
        if installed_ver < TORCH_MIN_VERSION and not is_py313_plus:
             raise importlib_metadata.PackageNotFoundError
    except importlib_metadata.PackageNotFoundError:
        logger.info("PyTorch not found or too old. Searching for compatible version...")
        
        torch_pkgs = ["torch", "torchvision", "torchaudio"]
        
        # Determine Index URL
        # Default to standard PyPI (CPU) or specific CUDA index
        index_url = None
        if has_gpu and cuda_version:
             # Try to find a specific stable CUDA index first
             index_url = _select_torch_index(cuda_version, is_nightly=False)
        
        logger.info(f"Attempting PyTorch install (Index: {index_url or 'Default'})...")
        try:
            # Try standard/stable install first (with --pre if on Py3.13)
            _install(torch_pkgs, index_url=index_url, pre=use_pre)
            installed_any = True
        except Exception as e:
            logger.warning(f"Standard PyTorch install failed ({e}).")
            
            # Fallback to Nightly (critical for early Python 3.13 support)
            if has_gpu and cuda_version:
                nightly_index = _select_torch_index(cuda_version, is_nightly=True)
            else:
                nightly_index = "https://download.pytorch.org/whl/nightly/cpu"
            
            logger.info(f"Attempting Nightly PyTorch install from {nightly_index}...")
            try:
                _install(torch_pkgs, index_url=nightly_index, pre=True)
                installed_any = True
            except Exception as e2:
                logger.error(f"Nightly PyTorch install also failed: {e2}")
                # Last resort: CPU only stable/pre
                if has_gpu:
                     logger.info("Falling back to CPU-only PyTorch...")
                     _install(torch_pkgs, index_url="https://download.pytorch.org/whl/cpu", pre=use_pre)
                     installed_any = True


    # 2. CuPy (Dynamic Match to Torch CUDA)
    if has_gpu:
        try:
            import cupy  # noqa: F401
        except ImportError:
            logger.info("Matching CuPy version to installed PyTorch...")
            try:
                import torch
                torch_cuda = torch.version.cuda
                if torch_cuda:
                    major_minor = torch_cuda.replace(".", "")[:3]
                    # Try specific cupy-cudaXX
                    pkg_name = f"cupy-cuda{major_minor}"
                    try:
                        _install([pkg_name], pre=use_pre)
                        installed_any = True
                    except Exception:
                        # Try generic cupy-cuda12x
                        generic = "cupy-cuda12x" if "12" in torch_cuda else "cupy-cuda11x"
                        logger.info(f"Wheel {pkg_name} not found. Trying {generic}...")
                        try:
                            _install([generic], pre=use_pre)
                            installed_any = True
                        except Exception:
                             logger.info("Falling back to source build for CuPy.")
                             _install(["cupy"], pre=use_pre)
                             installed_any = True
            except Exception as e:
                logger.warning(f"Failed to auto-install CuPy: {e}")

    # 3. TA-Lib (OS Specific)
    try:
        importlib_metadata.version("ta-lib")
    except importlib_metadata.PackageNotFoundError:
        # Try official TA-Lib wheel first (now exists for Py3.13)
        try:
            logger.info("Installing TA-Lib (Official)...")
            _install(["TA-Lib"], pre=use_pre)
            installed_any = True
        except Exception:
            if system == "windows":
                logger.info("Official TA-Lib failed. Trying ta-lib-binary...")
                try:
                    _install(["ta-lib-binary"], pre=use_pre)
                    installed_any = True
                except Exception:
                    logger.warning("Failed to install ta-lib-binary.")
            else:
                logger.warning("ta-lib wheel install failed. Trying apt-get libta-lib0-dev ...")
                if _apt_install(["libta-lib0", "libta-lib0-dev", "ta-lib"]):
                     try:
                         _install(["TA-Lib"], pre=use_pre)
                         installed_any = True
                     except:
                         pass

    # 4. Essential Packages (Dynamic List)
    # Updated constraints for 2025/Python 3.13
    essentials = [
        "numpy>=2.1.0",
        "pandas>=2.2.3",
        "scikit-learn>=1.5.0",
        "joblib",
        "psutil",
        "numba>=0.60.0",
        "llvmlite>=0.43.0",
        "matplotlib",
        "PyYAML",
        "tqdm",
        "colorama",
        "pydantic>=2.10",
        "pydantic-settings",
        "redis>=5.0",
        "duckdb>=1.0.0",
        "optuna>=4.0.0",
        "requests",
        "feedparser"
    ]
    
    if system == "windows":
        essentials.append("MetaTrader5>=5.0.45")

    logger.info("Verifying essential packages...")
    missing_essentials = []
    for pkg in essentials:
        pkg_name = pkg.split("<")[0].split(">")[0].split("=")[0]
        try:
            importlib_metadata.version(pkg_name)
        except importlib_metadata.PackageNotFoundError:
            missing_essentials.append(pkg)

    if missing_essentials:
        logger.info(f"Installing missing essentials: {missing_essentials}")
        try:
            _install(missing_essentials, pre=use_pre)
            installed_any = True
        except Exception as e:
            logger.warning(f"Batch install failed ({e}). Trying individually...")
            for pkg in missing_essentials:
                try:
                    _install([pkg], pre=use_pre)
                    installed_any = True
                except Exception as e2:
                    logger.error(f"Failed to install {pkg}: {e2}")

    # 5. Specialized Model Libraries
    # SB3 2.8.0a2+ supports Py3.13 natively now
    specialized_libs = [
        {"name": "pytorch-tabnet", "pip": "pytorch-tabnet"},
        {"name": "ray", "pip": "ray[rllib]", "skip_win": True},
        {"name": "gymnasium", "pip": "gymnasium[box2d,atari]"},
        {"name": "stable_baselines3", "pip": "stable-baselines3[extra]>=2.4.0"},
        {"name": "efficient_kan", "pip": "efficient-kan", "git": "git+https://github.com/Blealtan/efficient-kan.git"},
        {"name": "neuralforecast", "pip": "neuralforecast", "skip_win": True},
        {"name": "tslib", "pip": "tslib", "skip_win": True},
        {"name": "xgboost", "pip": "xgboost>=2.1.0"},
        {"name": "catboost", "pip": "catboost>=1.2.7"},
        {"name": "lightgbm", "pip": "lightgbm>=4.5.0"},
    ]

    for lib_def in specialized_libs:
        pkg_import_name = lib_def["name"]
        pip_name = lib_def["pip"]
        
        if lib_def.get("skip_win") and system == "windows":
            continue

        try:
            check_name = pkg_import_name.replace("-", "_")
            importlib_metadata.version(check_name)
        except importlib_metadata.PackageNotFoundError:
            logger.info(f"Installing {pip_name}...")
            try:
                _install([pip_name], pre=use_pre)
                installed_any = True
            except Exception:
                if "git" in lib_def:
                    git_url = lib_def["git"]
                    logger.info(f"Pip install failed. Trying source/git: {git_url}")
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", git_url])
                        installed_any = True
                    except Exception as e_git:
                        logger.warning(f"Git install failed for {pkg_import_name}: {e_git}")

    # 6. Transformer accelerators (GPU only)
    if has_gpu and not is_py313_plus: 
        # Skip on 3.13 for now unless we know for sure they exist to avoid noise
        for pkg in ["xformers", "flash-attn"]:
            try:
                importlib_metadata.version(pkg.replace("-", "_"))
            except importlib_metadata.PackageNotFoundError:
                try:
                    _install([pkg], pre=use_pre)
                    installed_any = True
                except:
                    pass

    _patch_third_party_warnings()

    logger.info("Dependency bootstrap complete.")

    if installed_any and os.environ.get("FOREX_BOT_DEPS_REEXEC", "") != "1":
        os.environ["FOREX_BOT_DEPS_REEXEC"] = "1"
        try:
            argv = [sys.executable, *sys.argv]
            print("[INIT] Dependencies were installed/updated. Restarting process once...", flush=True)
            os.execv(sys.executable, argv)
        except Exception as exc:
            logger.warning(f"Dependency bootstrap restart failed: {exc}")


def _patch_third_party_warnings() -> None:
    """
    Apply small, local compatibility patches to installed third-party libraries.
    """
    _patch_pytorch_tabnet_scipy_import()


def _patch_pytorch_tabnet_scipy_import() -> None:
    try:
        import importlib.util
        spec = importlib.util.find_spec("pytorch_tabnet")
        if spec is None or not spec.submodule_search_locations:
            return

        root = Path(next(iter(spec.submodule_search_locations)))
        target = root / "multiclass_utils.py"
        if not target.exists():
            return

        try:
            text = target.read_text(encoding="utf-8")
        except Exception:
            text = target.read_text()

        old = "from scipy.sparse.base import spmatrix"
        new = "from scipy.sparse import spmatrix"
        if old not in text:
            return

        target.write_text(text.replace(old, new), encoding="utf-8")
        logger.info("Patched pytorch-tabnet SciPy import to prevent DeprecationWarning spam.")
    except Exception as exc:
        logger.info(f"pytorch-tabnet SciPy import patch skipped: {exc!r}")


def _cleanup_pip_leftovers() -> None:
    try:
        import site
        user_site = site.getusersitepackages()
    except Exception:
        return
    try:
        root = Path(user_site)
    except Exception:
        return
    if not root.exists() or not root.is_dir():
        return
    targets = {"~~mpy", "~-mpy"}
    try:
        for item in root.iterdir():
            name = item.name
            if not name.startswith("~"):
                continue
            if name in targets or "umpy" in name.lower():
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)  # type: ignore[call-arg]
                except Exception:
                    continue
    except Exception:
        return


def _select_torch_index(cuda_version: str, is_nightly: bool = False) -> str:
    """
    Map CUDA version to the nearest supported PyTorch wheel index.
    """
    try:
        parts = cuda_version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return "https://download.pytorch.org/whl/cpu"

    # Nightly handling
    if is_nightly:
        if major == 12:
            return "https://download.pytorch.org/whl/nightly/cu124"
        return "https://download.pytorch.org/whl/nightly/cpu"

    # Stable handling
    if major >= 13 or (major == 12 and minor >= 4):
        return "https://download.pytorch.org/whl/cu124"
    if major == 12 and minor >= 1:
        return "https://download.pytorch.org/whl/cu121"
    
    return "https://download.pytorch.org/whl/cpu"


def _install(packages: list[str], index_url: str | None = None, upgrade: bool = False, pre: bool = False) -> None:
    """Install python packages via pip."""
    if not packages:
        return

    cmd = [sys.executable, "-m", "pip", "install", "--timeout", "60"]
    
    # Allow installing to system/global environment (PEP 668 override)
    cmd.append("--break-system-packages")
    
    if upgrade:
        cmd.append("--upgrade")
    if pre:
        cmd.append("--pre")
    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.extend(packages)

    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _apt_install(packages: list[str]) -> bool:
    """Best-effort apt-get install (non-interactive). Returns True on success."""
    if shutil.which("apt-get") is None:
        return False
    global _APT_REFRESHED
    if not _APT_REFRESHED:
        try:
            subprocess.check_call(
                ["sudo", "-n", "apt-get", "update"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            subprocess.check_call(
                ["sudo", "-n", "apt-get", "upgrade", "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _APT_REFRESHED = True
        except Exception:
            _APT_REFRESHED = True
    try:
        subprocess.check_call(["sudo", "-n", "apt-get", "install", "-y", *packages])
        return True
    except Exception:
        return False

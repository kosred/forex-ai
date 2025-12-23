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

# Target torch baseline (latest supported on CUDA runtime; override if needed)
TORCH_VERSION = "2.5.1"
VISION_VERSION = "0.20.1"
AUDIO_VERSION = "2.5.1"


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

    logger.info(f"Dependency Check: OS={system}, GPU={has_gpu} (CUDA {cuda_version or 'N/A'})")
    logger.info(f"Python Version: {platform.python_version()}")

    installed_any = False

    # 1. PyTorch (Hardware-Aware Dynamic Install - latest stable)
    try:
        importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        logger.info("PyTorch not found. Searching for compatible version...")
        torch_pkgs = [
            f"torch=={TORCH_VERSION}",
            f"torchvision=={VISION_VERSION}",
            f"torchaudio=={AUDIO_VERSION}",
        ]
        if has_gpu and cuda_version:
            cu_index = _select_torch_index(cuda_version)
            logger.info(f"Attempting PyTorch {TORCH_VERSION} for CUDA {cuda_version} via {cu_index} ...")
            try:
                _install(torch_pkgs, index_url=cu_index)
                installed_any = True
            except Exception as e:
                logger.warning(f"CUDA wheel install failed ({e}); falling back to CPU torch.")
                _install(torch_pkgs, index_url="https://download.pytorch.org/whl/cpu")
                installed_any = True
        else:
            _install(torch_pkgs, index_url="https://download.pytorch.org/whl/cpu")
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
                    pkg_name = f"cupy-cuda{major_minor}"
                    try:
                        _install([pkg_name])
                        installed_any = True
                    except Exception:
                        logger.info(f"Wheel {pkg_name} not found. Trying generic cupy-cuda12x...")
                        try:
                            _install(["cupy-cuda12x"])
                            installed_any = True
                        except Exception:
                            logger.info("Falling back to source build for CuPy (may be slow).")
                            _install(["cupy"])
                            installed_any = True
            except Exception as e:
                logger.warning(f"Failed to auto-install CuPy: {e}")

    # 2b. numba-cuda (best effort; wheels are scarce on py3.13, so default to numba CPU)
    if has_gpu:
        try:
            import numba.cuda  # type: ignore
        except Exception:
            logger.info("numba-cuda not available; continuing with CPU numba for GPU kernels fallback.")

    # 3. TA-Lib (OS Specific)
    try:
        importlib_metadata.version("ta-lib")
    except importlib_metadata.PackageNotFoundError:
        if system == "windows":
            logger.info("Installing TA-Lib binary for Windows...")
            try:
                _install(["ta-lib-binary"])
                installed_any = True
            except Exception:
                logger.warning("Failed to install ta-lib-binary.")
        else:
            try:
                _install(["ta-lib"])
                installed_any = True
            except Exception:
                logger.warning("ta-lib wheel install failed. Trying apt-get libta-lib0-dev ...")
                if not _apt_install(["libta-lib0", "libta-lib0-dev", "ta-lib"]):
                    logger.error("Failed to install ta-lib. On Linux, install libta-lib0-dev and retry.")

    # 4. Essential Packages (Dynamic List - No requirements.txt)
    essentials = [
        "numpy>=2.1,<3",
        "pandas>=2.2",
        "scikit-learn>=1.5",
        "joblib",
        "psutil",
        "numba>=0.60",
        "llvmlite>=0.43",
        "matplotlib",
        "PyYAML",
        "tqdm",
        "colorama",
        "pydantic>=2.6",
        "pydantic-settings",
        "redis>=5.0",  # for Optuna Redis storage
        "duckdb>=1.0.0",
        "duckdb-engine>=0.12.0",
        "optuna>=4.0.0",
    ]

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
        _install(missing_essentials)
        installed_any = True

    # 5. Specialized Model Libraries (Required for full feature set on Linux; best effort on Windows)
    optional_libs = [
        "efficient-kan",
        "pytorch-tabnet",
        "ray[rllib]",
        "gymnasium[box2d,atari]",
        # Prefer stable SB3; fall back to git main if wheel unavailable.
        "stable-baselines3[extra]",
        # Full TiDE and other SOTA models
        "neuralforecast",
    ]
    for lib in optional_libs:
        base_name = lib.split("[")[0]
        try:
            importlib_metadata.version(base_name)
        except importlib_metadata.PackageNotFoundError:
            # Skip ray[rllib] on Windows + Python 3.13+
            if "ray" in lib and system == "windows":
                logger.info("Skipping RLlib on Windows (unsupported). Install on Linux/WSL for RLlib.")
                continue
            if "stable-baselines3" in lib and system == "windows":
                logger.info("Skipping Stable-Baselines3 on Windows for RL usage; install on Linux/WSL.")
                continue
            if "neuralforecast" in lib and system == "windows":
                logger.info("Skipping neuralforecast on Windows; install on Linux/WSL for full TiDE.")
                continue

            logger.info(f"Installing {lib} (Best Effort)...")
            try:
                _install([lib])
                installed_any = True
            except Exception as e:
                if "stable-baselines3" in lib:
                    logger.info("stable-baselines3 wheel unavailable; trying bleeding-edge from GitHub...")
                    try:
                        subprocess.check_call(
                            [
                                sys.executable,
                                "-m",
                                "pip",
                                "install",
                                "git+https://github.com/DLR-RM/stable-baselines3.git#egg=stable-baselines3[extra]",
                            ]
                        )
                        installed_any = True
                        continue
                    except Exception:
                        logger.warning("Failed to install stable-baselines3 from git; RL (SB3) will be unavailable.")

                if "efficient-kan" in lib:
                    logger.info("Standard efficient-kan install failed. Trying bleeding-edge from GitHub...")
                    try:
                        subprocess.check_call(
                            [
                                sys.executable,
                                "-m",
                                "pip",
                                "install",
                                "git+https://github.com/Blealtan/efficient-kan.git",
                            ]
                        )
                        logger.info("Successfully installed efficient-kan from git.")
                        installed_any = True
                        continue
                    except Exception:
                        pass

                if "efficient-kan" in lib:
                    logger.warning(f"{lib} unavailable. KAN model will run in CPU-Optimized Lite mode.")
                elif "pytorch-tabnet" in lib:
                    logger.warning(f"{lib} unavailable. TabNet model will run in CPU mode if possible.")
                else:
                    logger.warning(
                        f"Library {lib} installation failed: {e}\n"
                        "  -> If this is RLlib, RL Agents will be disabled but others will work."
                    )

    # 6. Transformer accelerators (best-effort, GPU only, avoid Py3.13 build failures)
    if has_gpu:
        py_major, py_minor = sys.version_info[:2]
        if py_major == 3 and py_minor <= 12:
            for pkg in ["xformers", "flash-attn"]:
                try:
                    importlib_metadata.version(pkg.replace("-", "_"))
                except importlib_metadata.PackageNotFoundError:
                    logger.info(f"Installing {pkg} (best effort, GPU accel)...")
                    try:
                        _install([pkg])
                        installed_any = True
                    except Exception as e:
                        logger.warning(f"{pkg} install failed; continuing without it ({e})")
        else:
            logger.info("Skipping xformers/flash-attn on Python 3.13 (wheels not widely available yet).")

    # 7. MetaTrader5 (Windows Only)
    if system == "windows":
        try:
            importlib_metadata.version("MetaTrader5")
        except importlib_metadata.PackageNotFoundError:
            logger.info("Installing MetaTrader5 (Latest)...")
            _install(["MetaTrader5"], upgrade=True)
            installed_any = True

    _patch_third_party_warnings()

    logger.info("Dependency bootstrap complete.")

    # Installing packages into the running interpreter can leave it in a partially inconsistent state
    # (especially on Windows after upgrading compiled wheels like NumPy/Pandas). Restart once.
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

    Goal: keep runtime output clean and future-proof against upcoming deprecations.
    """
    _patch_pytorch_tabnet_scipy_import()


def _patch_pytorch_tabnet_scipy_import() -> None:
    """
    pytorch-tabnet 4.1.0 imports `spmatrix` from `scipy.sparse.base`, which is deprecated and emits a warning.

    SciPy's recommended import is `from scipy.sparse import spmatrix`.
    """
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
    """
    Remove pip's leftover `~...` folders from interrupted upgrades (common on Windows).

    These folders can trigger noisy pip warnings like:
      "Ignoring invalid distribution ~umpy"
    """
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


def _select_torch_index(cuda_version: str) -> str:
    """
    Map CUDA version to the nearest supported PyTorch wheel index.
    Prefers the highest compatible bucket supported by PyTorch 2.9.x.
    """
    try:
        parts = cuda_version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return "https://download.pytorch.org/whl/cpu"

    # PyTorch 2.9 provides cu128 and cu130 wheels; fall back to cu126 then cu124
    if major >= 13 or (major == 12 and minor >= 10):
        return "https://download.pytorch.org/whl/cu130"
    if major == 12 and minor >= 8:
        return "https://download.pytorch.org/whl/cu128"
    if major == 12 and minor >= 6:
        return "https://download.pytorch.org/whl/cu126"
    if major == 12 and minor >= 4:
        return "https://download.pytorch.org/whl/cu124"
    if major == 12 and minor >= 1:
        return "https://download.pytorch.org/whl/cu121"
    return "https://download.pytorch.org/whl/cpu"


def _install(packages: list[str], index_url: str | None = None, upgrade: bool = False, pre: bool = False) -> None:
    """Install python packages via pip."""
    if not packages:
        return

    cmd = [sys.executable, "-m", "pip", "install", "--timeout", "60"]
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
            # Light upgrade to pull security fixes on older base images
            subprocess.check_call(
                ["sudo", "-n", "apt-get", "upgrade", "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _APT_REFRESHED = True
        except Exception:
            # If sudo or update fails, continue without blocking
            _APT_REFRESHED = True
    try:
        subprocess.check_call(["sudo", "-n", "apt-get", "install", "-y", *packages])
        return True
    except Exception:
        return False

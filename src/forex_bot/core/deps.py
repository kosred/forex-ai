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
    "ta": "ta-lib",
    "talib": "TA-Lib",
    "tensorboard": "tensorboard",
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "gym": "gymnasium",
    "gymnasium": "gymnasium",
    "stable_baselines3": "stable-baselines3[extra]",
    "sb3_contrib": "sb3-contrib",
    "ray": "ray[default,rllib]",
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
    "redis": "redis",
    "duckdb": "duckdb",
    "duckdb_engine": "duckdb-engine",
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
    "evotorch": "evotorch",
    "ax": "ax-platform[mysql]",
    "botorch": "botorch",
    "cma": "cma",
    "cudf": "cudf-cu12",
    "cuml": "cuml-cu12",
    "cupy": "cupy-cuda12x",
    "flash_attn": "flash-attn",
    "onnx": "onnx",
    "onnxmltools": "onnxmltools",
    "onnxruntime": "onnxruntime-gpu",
    "openai": "openai",
    "polars": "polars",
    "pytorch_tabnet": "pytorch-tabnet2",
    "skl2onnx": "skl2onnx",
    "threadpoolctl": "threadpoolctl",
    "transformer_engine": "transformer-engine",
}

# Optional imports can be skipped unless explicitly enabled via env vars.
# Set FOREX_BOT_INSTALL_ALL=1 to force-install everything.
OPTIONAL_IMPORTS: dict[str, tuple[str, bool]] = {
    # Heavy GPU stacks (RAPIDS) - off by default
    "cudf": ("FOREX_BOT_INSTALL_RAPIDS", False),
    "cuml": ("FOREX_BOT_INSTALL_RAPIDS", False),
    # GPU array stack (optional; use only if you want GPU indicators)
    "cupy": ("FOREX_BOT_INSTALL_CUPY", False),
    # RL stack (very heavy) - off by default
    "ray": ("FOREX_BOT_INSTALL_RL", False),
    "stable_baselines3": ("FOREX_BOT_INSTALL_RL", False),
    "gymnasium": ("FOREX_BOT_INSTALL_RL", False),
    "sb3_contrib": ("FOREX_BOT_INSTALL_RL", False),
    # Bayesian optimization (Ax/Botorch) - off by default
    "ax": ("FOREX_BOT_INSTALL_BO", False),
    "botorch": ("FOREX_BOT_INSTALL_BO", False),
    # Forecasting stack - off by default (often constrains torch<=2.6)
    "neuralforecast": ("FOREX_BOT_INSTALL_NEURALFORECAST", False),
    # ONNX export/runtime - off by default
    "onnx": ("FOREX_BOT_INSTALL_ONNX", False),
    "onnxruntime": ("FOREX_BOT_INSTALL_ONNX", False),
    "onnxmltools": ("FOREX_BOT_INSTALL_ONNX", False),
    "skl2onnx": ("FOREX_BOT_INSTALL_ONNX", False),
    # Flash attention + transformer engine - optional
    "flash_attn": ("FOREX_BOT_INSTALL_FLASH_ATTN", False),
    "transformer_engine": ("FOREX_BOT_INSTALL_TRANSFORMER_ENGINE", False),
    # Optional data engines
    "polars": ("FOREX_BOT_INSTALL_POLARS", False),
    "duckdb": ("FOREX_BOT_INSTALL_DUCKDB", False),
}

# Known Standard Library modules to exclude (Python 3.10+)
STDLIB_MODULES = {
    "__future__", "_thread", "abc", "aifc", "argparse", "array", "ast", "asyncio",
    "base64", "bdb", "binascii", "bisect", "builtins", "bz2", "calendar", "cgi",
    "cgitb", "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections",
    "colorsys", "compileall", "concurrent", "configparser", "contextlib", "contextvars",
    "copy", "copyreg", "cProfile", "crypt", "csv", "ctypes", "curses", "dataclasses",
    "datetime", "dbm", "decimal", "difflib", "dis", "distutils", "doctest", "email",
    "encodings", "ensurepip", "enum", "errno", "faulthandler", "fcntl", "filecmp",
    "fileinput", "fnmatch", "formatter", "fractions", "ftplib", "functools", "gc",
    "getopt", "getpass", "gettext", "glob", "graphlib", "grp", "gzip", "hashlib",
    "heapq", "hmac", "html", "http", "imaplib", "imghdr", "imp", "importlib",
    "inspect", "io", "ipaddress", "itertools", "json", "keyword", "lib2to3",
    "linecache", "locale", "logging", "lzma", "mailbox", "mailcap", "marshal",
    "math", "mimetypes", "mmap", "modulefinder", "msvcrt", "multiprocessing",
    "netrc", "nntplib", "ntpath", "numbers", "operator", "optparse", "os",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile", "pstats",
    "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random",
    "re", "readline", "reprlib", "resource", "rlcompleter", "runpy", "sched",
    "secrets", "select", "selectors", "shelve", "shlex", "shutil", "signal",
    "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd",
    "sqlite3", "ssl", "stat", "statistics", "string", "stringprep", "struct",
    "subprocess", "sunau", "symbol", "symtable", "sys", "sysconfig", "syslog",
    "tabnanny", "tarfile", "telnetlib", "tempfile", "termios", "textwrap",
    "threading", "time", "timeit", "tkinter", "token", "tokenize", "trace",
    "traceback", "tracemalloc", "tty", "turtle", "turtledemo", "types", "typing",
    "unicodedata", "unittest", "urllib", "uu", "uuid", "venv", "warnings",
    "wave", "weakref", "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib",
    "xml", "xmlrpc", "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo"
}

# Special handling for git dependencies or non-standard sources
SPECIAL_SOURCES = {
    "efficient-kan": "git+https://github.com/Blealtan/efficient-kan.git",
    "stable-baselines3[extra]": "stable-baselines3[extra]>=2.4.0",
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
    Can be skipped by setting FOREX_BOT_SKIP_DEPS=1 (useful on locked-down systems).
    """
    global _DEPS_CHECKED
    if _DEPS_CHECKED:
        return
    if os.environ.get("FOREX_BOT_SKIP_DEPS") == "1":
        logger.info("Dependency bootstrap skipped via FOREX_BOT_SKIP_DEPS=1.")
        _DEPS_CHECKED = True
        return
    _DEPS_CHECKED = True

    # 1. Setup Environment info
    system = platform.system().lower()
    py_ver = sys.version_info
    is_py313 = py_ver >= (3, 13)
    
    logger.info(f"Scanning codebase for dependencies (Py {py_ver.major}.{py_ver.minor}, OS {system})...")

    # 2. Scan Imports
    src_root = Path(__file__).resolve().parent.parent.parent # src/forex_bot/core/.. -> src/forex_bot -> src
    
    # Identify local modules to exclude
    local_modules = set()
    if src_root.exists():
        for item in src_root.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                local_modules.add(item.name)
            elif item.is_file() and item.suffix == ".py":
                local_modules.add(item.stem)
                
    # Also exclude subpackages of forex_bot specifically if running from there
    forex_bot_dir = src_root / "forex_bot"
    if forex_bot_dir.exists():
         for item in forex_bot_dir.iterdir():
            if item.is_dir() and (item / "__init__.py").exists():
                local_modules.add(item.name)

    imported_modules = scan_imports(src_root)

    def _truthy(val: str | None) -> bool:
        if val is None:
            return False
        return str(val).strip().lower() in {"1", "true", "yes", "on"}

    install_all = _truthy(os.environ.get("FOREX_BOT_INSTALL_ALL"))
    # Enable cupy by default when a GPU is present, unless explicitly disabled.
    default_cupy = bool(shutil.which("nvidia-smi"))

    def _optional_enabled(mod: str) -> bool:
        if install_all:
            return True
        opt = OPTIONAL_IMPORTS.get(mod)
        if not opt:
            return True
        env_key, default_val = opt
        if env_key == "FOREX_BOT_INSTALL_CUPY":
            default_val = default_cupy
        env_val = os.environ.get(env_key)
        if env_val is None:
            return default_val
        return _truthy(env_val)
    
    # 3. Resolve to PyPI packages
    required_packages = set()
    for mod in imported_modules:
        if mod in STDLIB_MODULES:
            continue
        if mod in sys.builtin_module_names:
            continue
        if mod in local_modules:
            continue
        # Also exclude known internal module names that might be picked up
        if mod in {"forex_bot", "src", "tests"}:
            continue

        # Map only known modules to packages; skip unknowns to avoid false positives.
        pkg = IMPORT_TO_PYPI.get(mod)
        if not pkg:
            continue

        if not _optional_enabled(mod):
            logger.info(
                "Skipping optional dependency '%s'. Enable via %s=1 or FOREX_BOT_INSTALL_ALL=1.",
                pkg,
                OPTIONAL_IMPORTS[mod][0],
            )
            continue

        # Skip packages known to lack wheels for Py3.13 (e.g., transformer-engine).
        if is_py313 and pkg in {"transformer-engine"}:
            continue

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
    gpu_libs = []
    standard_libs = []

    torch_missing = any(x in missing for x in ("torch", "torchvision", "torchaudio"))
    if not torch_missing:
        try:
            import torch  # noqa: F401
        except Exception:
            torch_missing = True

    # Prefer a pinned torch bundle to avoid pip solving/downloading multiple versions.
    torch_bundle = None
    if torch_missing:
        torch_bundle = _torch_bundle_for_env(py_ver)
        # Remove bare torch entries from missing; we'll install the pinned bundle.
        missing = [p for p in missing if p not in {"torch", "torchvision", "torchaudio"}]

    for pkg in missing:
        # Only core torch libs are on the special index; evotorch is on standard PyPI
        if any(x == pkg for x in ["torch", "torchvision", "torchaudio"]):
            gpu_libs.append(pkg)
        else:
            standard_libs.append(pkg)

    post_torch_libs = []
    needs_torch = {"flash-attn", "transformer-engine"}
    # Defer packages that need torch if torch is missing right now.
    if torch_missing:
        for pkg in list(standard_libs):
            if pkg in needs_torch:
                standard_libs.remove(pkg)
                post_torch_libs.append(pkg)

    if standard_libs:
        logger.info("Installing standard dependencies...")
        _install(standard_libs, pre=is_py313)

    if gpu_libs:
        logger.info("Installing GPU dependencies...")
        index_url = "https://download.pytorch.org/whl/cpu"
        if shutil.which("nvidia-smi"):
            if is_py313:
                # Official 3.13 wheels for CUDA live under cu128 (stable) or nightly for newer.
                index_url = "https://download.pytorch.org/whl/cu128"
            else:
                index_url = "https://download.pytorch.org/whl/cu121"

        # Inject pinned torch bundle first (if any) to avoid resolver churn.
        if torch_bundle:
            gpu_libs = torch_bundle + gpu_libs
        _install(gpu_libs, index_url=index_url, pre=is_py313)

    if post_torch_libs:
        logger.info("Installing torch-dependent deps after torch is available...")
        _install(post_torch_libs, pre=is_py313)

    if os.environ.get("FOREX_BOT_DEPS_REEXEC", "") != "1":
        os.environ["FOREX_BOT_DEPS_REEXEC"] = "1"
        logger.info("Restarting process to load new dependencies...")
        os.execv(sys.executable, [sys.executable, *sys.argv])

def _install(packages: list[str], index_url: str | None = None, pre: bool = False) -> None:
    if not packages:
        return

    # 2025 GLOBAL BOOTSTRAP: Use --user to keep system clean, --upgrade to avoid stale junk,
    # and --break-system-packages to allow global pip usage on modern Linux (Debian/Ubuntu).
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--user", "--upgrade",
        "--timeout", "120",
        "--break-system-packages",
        "--prefer-binary",
        "--upgrade-strategy", "only-if-needed",
    ]
    if pre:
        cmd.append("--pre")
    if index_url:
        cmd.extend(["--index-url", index_url])

    final_pkgs: list[str] = []
    flash_build: list[str] = []
    def _truthy(val: str | None) -> bool:
        if val is None:
            return False
        return str(val).strip().lower() in {"1", "true", "yes", "on"}
    allow_flash_build = _truthy(os.environ.get("FOREX_BOT_FLASH_ATTN_BUILD"))
    for p in packages:
        if p == "flash-attn":
            wheel = _flash_attn_wheel_url()
            if wheel:
                final_pkgs.append(wheel)
                continue
            if allow_flash_build:
                flash_build.append(p)
                continue
            logger.warning(
                "Skipping flash-attn install: no compatible wheel detected. "
                "Set FOREX_BOT_FLASH_ATTN_BUILD=1 to attempt a source build."
            )
            continue
        if p in SPECIAL_SOURCES:
            final_pkgs.append(SPECIAL_SOURCES[p])
        else:
            final_pkgs.append(p)

    if final_pkgs:
        logger.info(f"Running pip: {' '.join(cmd + final_pkgs)}")
        env = os.environ.copy()
        # Disable hash enforcement if set globally; we don't use pinned hashes here.
        env.setdefault("PIP_REQUIRE_HASHES", "0")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        try:
            result = subprocess.run(
                cmd + final_pkgs,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "")
            if "--break-system-packages" in cmd and "no such option: --break-system-packages" in stderr:
                logger.warning("pip does not support --break-system-packages; retrying without it.")
                cmd_no_break = [c for c in cmd if c != "--break-system-packages"]
                subprocess.check_call(cmd_no_break + final_pkgs, env=env)
            else:
                raise
    if flash_build:
        build_cmd = list(cmd) + ["--no-build-isolation"]
        logger.info(f"Running pip (flash-attn source build): {' '.join(build_cmd + flash_build)}")
        env = os.environ.copy()
        env.setdefault("PIP_REQUIRE_HASHES", "0")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        subprocess.check_call(build_cmd + flash_build, env=env)


def _flash_attn_wheel_url() -> str | None:
    override = os.environ.get("FOREX_BOT_FLASH_ATTN_WHEEL_URL")
    if override:
        return override
    try:
        import platform as _platform
        if _platform.system().lower() != "linux":
            return None
        machine = _platform.machine().lower()
        if machine not in {"x86_64", "amd64"}:
            return None
        py_ver = sys.version_info[:2]
        try:
            import torch
        except Exception:
            return None
        torch_ver = str(getattr(torch, "__version__", "")).split("+")[0]
        torch_major_minor = ".".join(torch_ver.split(".")[:2])
        if torch_major_minor not in {"2.6", "2.7", "2.8", "2.9"}:
            return None
        cuda_ver = getattr(getattr(torch, "version", None), "cuda", None)
        if not cuda_ver or not str(cuda_ver).startswith("12"):
            return None
        abi_flag = "TRUE"
        try:
            abi_flag = "TRUE" if bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True)) else "FALSE"
        except Exception:
            pass
        cp_tag = f"cp{py_ver[0]}{py_ver[1]}"
        # Wheels exist for cp39-cp313 for torch2.6/2.7/2.8. Torch2.9 wheels are
        # currently published for cp312 only.
        if torch_major_minor == "2.9" and cp_tag != "cp312":
            return None
        return (
            "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
            f"flash_attn-2.8.3+cu12torch{torch_major_minor}cxx11abi{abi_flag}-{cp_tag}-{cp_tag}-linux_x86_64.whl"
        )
    except Exception:
        return None


def _torch_bundle_for_env(py_ver: tuple[int, int] | None = None) -> list[str]:
    """
    Return a pinned torch/torchvision/torchaudio bundle to avoid pip resolving
    multiple torch versions. Override via FOREX_BOT_TORCH_VERSION if desired.
    """
    if py_ver is None:
        py_ver = sys.version_info[:2]
    override = os.environ.get("FOREX_BOT_TORCH_VERSION")
    torch_ver = (override or "2.8.0").strip()
    # Best-known matching versions.
    vision_map = {
        "2.6.0": "0.21.0",
        "2.7.0": "0.22.0",
        "2.7.1": "0.22.1",
        "2.8.0": "0.23.0",
        "2.9.0": "0.24.0",
    }
    vision_ver = vision_map.get(torch_ver, "0.21.0")
    audio_ver = torch_ver
    return [
        f"torch=={torch_ver}",
        f"torchvision=={vision_ver}",
        f"torchaudio=={audio_ver}",
    ]

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

        # Skip packages known to lack wheels for Py3.13 (e.g., pytorch-tabnet).
        if is_py313 and pkg in {"pytorch-tabnet"}:
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
    
    for pkg in missing:
        if any(x in pkg for x in ["torch", "cupy"]):
            gpu_libs.append(pkg)
        else:
            standard_libs.append(pkg)

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

        _install(gpu_libs, index_url=index_url, pre=is_py313)

    if os.environ.get("FOREX_BOT_DEPS_REEXEC", "") != "1":
        os.environ["FOREX_BOT_DEPS_REEXEC"] = "1"
        logger.info("Restarting process to load new dependencies...")
        os.execv(sys.executable, [sys.executable, *sys.argv])

def _install(packages: list[str], index_url: str | None = None, pre: bool = False) -> None:
    if not packages:
        return

    cmd = [sys.executable, "-m", "pip", "install", "--timeout", "120", "--break-system-packages"]
    if pre:
        cmd.append("--pre")
    if index_url:
        cmd.extend(["--index-url", index_url])

    final_pkgs: list[str] = []
    for p in packages:
        if p in SPECIAL_SOURCES:
            final_pkgs.append(SPECIAL_SOURCES[p])
        else:
            final_pkgs.append(p)

    logger.info(f"Running pip: {' '.join(cmd + final_pkgs)}")
    subprocess.check_call(cmd + final_pkgs)

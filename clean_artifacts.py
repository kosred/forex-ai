import argparse
import shutil
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _is_within_project(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except Exception:
        return False
    root = _project_root()
    return resolved == root or root in resolved.parents


def _safe_rmtree(target: Path) -> bool:
    """Delete target only if it resides inside the project directory."""
    try:
        target = target.resolve()
        if not _is_within_project(target):
            print(f"[ABORT] Refusing to remove {target} (outside project).")
            return False
        if not target.exists():
            print(f"[SKIP] Not found: {target}")
            return False
        if not target.is_dir():
            print(f"[SKIP] Not a directory: {target}")
            return False
        print(f"[RM] {target}")
        shutil.rmtree(target)
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to remove {target}: {exc}")
        return False


def _safe_unlink(target: Path) -> bool:
    """Delete file only if it resides inside the project directory."""
    try:
        target = target.resolve()
        if not _is_within_project(target):
            print(f"[ABORT] Refusing to remove {target} (outside project).")
            return False
        if not target.exists():
            print(f"[SKIP] Not found: {target}")
            return False
        if not target.is_file():
            print(f"[SKIP] Not a file: {target}")
            return False
        print(f"[DEL] {target}")
        target.unlink()
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to delete {target}: {exc}")
        return False


def clean_artifacts(
    *,
    cache: bool = False,
    models: bool = False,
    ruff_cache: bool = False,
    pytest_cache: bool = False,
    catboost_info: bool = False,
    logs: bool = False,
    pycache: bool = False,
) -> int:
    """
    Remove generated artifacts (no source code).

    This deletes folders/files under the project root only.
    """
    root = _project_root()
    removed_any = False

    if cache:
        cache_dir = root / "cache"
        # Prefer granular cleanup so a locked sqlite file doesn't prevent clearing feature/optuna caches.
        removed_any |= _safe_rmtree(cache_dir / "features")
        removed_any |= _safe_rmtree(cache_dir / "optuna")
        for name in ("news.sqlite", "news.sqlite-shm", "news.sqlite-wal", "strategy_ledger.sqlite"):
            removed_any |= _safe_unlink(cache_dir / name)
        try:
            if cache_dir.exists() and cache_dir.is_dir() and not any(cache_dir.iterdir()):
                removed_any |= _safe_rmtree(cache_dir)
        except Exception:
            pass
    if models:
        removed_any |= _safe_rmtree(root / "models")
    if ruff_cache:
        removed_any |= _safe_rmtree(root / ".ruff_cache")
    if pytest_cache:
        removed_any |= _safe_rmtree(root / ".pytest_cache")
    if catboost_info:
        removed_any |= _safe_rmtree(root / "catboost_info")
    if logs:
        removed_any |= _safe_rmtree(root / "logs")

    if pycache:
        for p in root.rglob("__pycache__"):
            if p.is_dir():
                removed_any |= _safe_rmtree(p)

    if not removed_any:
        print("[INFO] Nothing to clean.")
        return 0

    print("[OK] Cleanup complete.")
    return 0


def deep_purge(*, mode: str = "cache") -> int:
    """
    Deep purge generated artifacts to avoid data leakage between runs.

    mode:
      - "cache": wipe cache/logs/tool caches + __pycache__ + sqlite/pkl in cache/root
      - "all": everything in "cache" plus models and all sqlite/pkl under project
    """
    mode = str(mode or "cache").strip().lower()
    if mode not in {"cache", "all"}:
        print(f"[SKIP] Unknown deep purge mode: {mode}")
        return 0

    # Base cleanup (safe directories only).
    clean_artifacts(
        cache=True,
        models=(mode == "all"),
        ruff_cache=True,
        pytest_cache=True,
        catboost_info=True,
        logs=True,
        pycache=True,
    )

    root = _project_root()

    def _purge_ext(root_dir: Path, exts: set[str]) -> None:
        if not root_dir.exists():
            return
        for p in root_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                _safe_unlink(p)

    # Always drop sqlite/pkl in cache dir if it exists.
    _purge_ext(root / "cache", {".pkl", ".sqlite", ".pyc"})

    # Remove top-level sqlite files (metrics, ledgers) to avoid stale state.
    for name in ("metrics.sqlite", "strategy_ledger.sqlite"):
        _safe_unlink(root / name)

    if mode == "all":
        _purge_ext(root, {".pkl", ".sqlite", ".pyc"})

    print("[OK] Deep purge complete.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely delete generated artifacts (cache/models/logs) in this project."
    )
    parser.add_argument("--all", action="store_true", help="Remove cache, models, tool caches, and __pycache__")
    parser.add_argument("--cache", action="store_true", help="Remove `cache/` (features, optuna journals, news cache)")
    parser.add_argument("--models", action="store_true", help="Remove `models/` (trained model artifacts)")
    parser.add_argument("--ruff-cache", action="store_true", help="Remove `.ruff_cache/`")
    parser.add_argument("--pytest-cache", action="store_true", help="Remove `.pytest_cache/`")
    parser.add_argument("--catboost-info", action="store_true", help="Remove `catboost_info/`")
    parser.add_argument("--logs", action="store_true", help="Remove `logs/`")
    parser.add_argument("--pycache", action="store_true", help="Remove all `__pycache__/` under the project")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.all:
        sys_exit = clean_artifacts(
            cache=True,
            models=True,
            ruff_cache=True,
            pytest_cache=True,
            catboost_info=True,
            logs=True,
            pycache=True,
        )
        raise SystemExit(sys_exit)

    sys_exit = clean_artifacts(
        cache=args.cache,
        models=args.models,
        ruff_cache=args.ruff_cache,
        pytest_cache=args.pytest_cache,
        catboost_info=args.catboost_info,
        logs=args.logs,
        pycache=args.pycache,
    )
    raise SystemExit(sys_exit)

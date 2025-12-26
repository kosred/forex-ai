import shutil
from pathlib import Path


def _safe_rmtree(target: Path) -> bool:
    """Delete target only if it resides inside the project directory."""
    try:
        target = target.resolve()
        project_root = Path(__file__).resolve().parent
        if project_root not in target.parents:
            print(f"[ABORT] Refusing to remove {target} (outside project).")
            return False
        if not target.exists():
            print("Cache directory not found.")
            return False
        print(f"Removing {target}...")
        shutil.rmtree(target)
        print("Cache cleared.")
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to remove {target}: {exc}")
        return False


def clean_cache():
    cache_dir = Path("cache")
    removed = False

    # Prefer granular cleanup so a locked sqlite file doesn't prevent clearing feature/HPO caches.
    removed |= _safe_rmtree(cache_dir / "features")
    removed |= _safe_rmtree(cache_dir / "hpo")
    removed |= _safe_rmtree(cache_dir / "ray_tune")
    for name in ("news.sqlite", "news.sqlite-shm", "news.sqlite-wal", "strategy_ledger.sqlite"):
        try:
            f = (cache_dir / name).resolve()
            project_root = Path(__file__).resolve().parent
            if project_root not in f.parents:
                print(f"[ABORT] Refusing to delete {f} (outside project).")
                continue
            if f.exists() and f.is_file():
                print(f"Deleting {f}...")
                f.unlink()
                removed = True
        except Exception as exc:
            print(f"[ERROR] Failed to delete {name}: {exc}")

    # Remove empty cache dir if possible.
    try:
        if cache_dir.exists() and cache_dir.is_dir() and not any(cache_dir.iterdir()):
            removed |= _safe_rmtree(cache_dir)
    except Exception:
        pass

    return removed


if __name__ == "__main__":
    clean_cache()

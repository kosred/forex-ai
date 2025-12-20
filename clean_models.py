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
            print("Models directory not found.")
            return False
        print(f"Removing {target}...")
        shutil.rmtree(target)
        print("Models cleared.")
        return True
    except Exception as exc:
        print(f"[ERROR] Failed to remove {target}: {exc}")
        return False


def clean_models():
    models_dir = Path("models")
    return _safe_rmtree(models_dir)


if __name__ == "__main__":
    clean_models()

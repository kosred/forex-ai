import json
import logging
import shutil
from pathlib import Path
from typing import Any

import joblib

from ..core.autotune import autotuner
from ..models.base import ExpertModel
from ..models.registry import get_model_class
from ..training.ensemble import MetaBlender

logger = logging.getLogger(__name__)

_JOBLIB_TMP_SUFFIX = ".tmp"


def _atomic_joblib_dump(obj: Any, path: Path) -> None:
    """
    Write joblib artifacts atomically to avoid partial/corrupt files on crash/interruption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + _JOBLIB_TMP_SUFFIX)
    joblib.dump(obj, tmp)
    tmp.replace(path)


class PersistenceService:
    """
    Handles saving/loading models, metadata, ONNX export, and log cleanup.
    """

    def __init__(self, models_dir: Path, logs_dir: Path):
        self.models_dir = models_dir
        self.logs_dir = logs_dir
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

    def save_run_summary(self, summary: dict[str, Any]) -> None:
        try:
            (self.models_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save run summary: {e}")

    def load_run_summary(self) -> dict[str, Any]:
        path = self.models_dir / "run_summary.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {}

    def load_incremental_stats(self) -> dict[str, Any]:
        path = self.models_dir / "incremental_stats.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception as e:
                logger.warning(f"Failed to load incremental stats: {e}")
        return {}

    def save_incremental_stats(self, stats: dict[str, Any]) -> None:
        try:
            (self.models_dir / "incremental_stats.json").write_text(json.dumps(stats, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save incremental stats: {e}")

    def save_active_models_list(self, models: dict[str, ExpertModel]) -> None:
        _atomic_joblib_dump(list(models.keys()), self.models_dir / "active_models.pkl")

    def save_models_bundle(self, models: dict[str, ExpertModel], scaler: Any = None) -> None:
        """
        Save a consolidated bundle for OnlineLearner and legacy loaders.

        Note: Many expert implementations hold non-pickleable runtime objects (e.g., torch ScriptModules),
        so we intentionally persist a thin bundle (names + scaler) and rely on per-model `save()/load()` for
        actual model state.
        """
        try:
            thin = {"models": {}, "model_names": list(models.keys()), "scaler": scaler, "schema_version": 2}
            _atomic_joblib_dump(thin, self.models_dir / "models.joblib")
            logger.info(f"Saved models bundle to {self.models_dir / 'models.joblib'}")
        except Exception as e:
            logger.warning(f"Failed to save models bundle: {e}")

    def load_models(self) -> tuple[dict[str, ExpertModel], MetaBlender | None]:
        models = {}
        blender = None
        try:
            active = joblib.load(self.models_dir / "active_models.pkl")
        except Exception:
            return {}, None

        for name in active:
            try:
                cls = get_model_class(name, prefer_gpu=autotuner.gpu_available)
                model = cls()
                model.load(str(self.models_dir))
                models[name] = model
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")

        try:
            blender = MetaBlender.load(self.models_dir / "meta_blender.joblib")
        except Exception:
            pass

        return models, blender

    def export_onnx(self, models: dict, blender: MetaBlender | None, sample_input: Any) -> list[str]:
        try:
            logger.info("Exporting to ONNX...")
            from ..models.onnx_exporter import ONNXExporter

            exporter = ONNXExporter(str(self.models_dir))
            exported = exporter.export_all(models, sample_input, blender)
            return list(exported.keys())
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return []

    def cleanup_logs(self, keep_last: int = 3) -> None:
        try:
            dirs = sorted(
                [d for d in self.logs_dir.iterdir() if d.is_dir() and d.name.startswith("training_")],
                key=lambda d: d.stat().st_mtime,
            )
            if len(dirs) > keep_last:
                for d in dirs[:-keep_last]:
                    shutil.rmtree(d, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Log cleanup failed: {e}")

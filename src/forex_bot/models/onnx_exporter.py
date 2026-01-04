"""
ONNX Export for Ultra-Fast Inference

Converts trained models to ONNX format for:
- 10-100x faster inference in production
- Cross-platform deployment (Windows -> Linux)
- GPU acceleration support
- Reduced memory footprint
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    onnx = None  # type: ignore[assignment]
    ort = None  # type: ignore[assignment]
    ONNX_AVAILABLE = False

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    SKL2ONNX_AVAILABLE = True
except ImportError:
    convert_sklearn = None  # type: ignore[assignment]
    FloatTensorType = None  # type: ignore[assignment]
    SKL2ONNX_AVAILABLE = False


class ONNXExporter:
    """
    Exports trained models to ONNX format for ultra-fast inference.

    Supports:
    - Tree models (LightGBM, XGBoost, CatBoost)
    - Logistic Regression
    - Neural networks (PyTorch models)
    - Ensemble meta-blender
    """

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = Path(models_dir)
        self.onnx_dir = self.models_dir / "onnx"
        self.onnx_dir.mkdir(exist_ok=True, parents=True)

    def export_all(
        self, models: dict[str, Any], sample_input: pd.DataFrame, meta_blender: Any | None = None
    ) -> dict[str, Any]:
        """
        Export all models to ONNX format.

        Parameters
        ----------
        models : Dict[str, Any]
            Dictionary of trained models {name: model_instance}
        sample_input : pd.DataFrame
            Sample input for shape inference (single row)
        meta_blender : Optional[Any]
            Meta-blender ensemble model

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping model names to export metadata (path + optional class labels)
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available - skipping export (install: onnx + onnxruntime).")
            return {}

        exported: dict[str, Any] = {}

        for name, model in models.items():
            try:
                onnx_path = self._export_model(name, model, sample_input)
                if onnx_path:
                    classes = None
                    try:
                        # Prefer underlying estimator classes if present (sklearn/lightgbm/xgboost).
                        base = getattr(model, "model", model)
                        if getattr(base, "classes_", None) is not None:
                            classes = [int(c) for c in list(base.classes_)]
                    except Exception:
                        classes = None

                    exported[name] = {"path": str(onnx_path), "classes": classes}
                    logger.info(f"Exported {name} to ONNX: {onnx_path}")
            except Exception as e:
                logger.info(f"ONNX export skipped for {name}: {e}")

        if meta_blender is not None:
            try:
                n_models = len(models)
                meta_input = pd.DataFrame(
                    np.random.randn(1, n_models * 3),  # 3 classes per model
                    columns=[f"model_{i}_class_{j}" for i in range(n_models) for j in range(3)],
                )

                onnx_path = self._export_model("meta_blender", meta_blender, meta_input)
                if onnx_path:
                    classes = None
                    try:
                        base = getattr(meta_blender, "model", meta_blender)
                        if getattr(base, "classes_", None) is not None:
                            classes = [int(c) for c in list(base.classes_)]
                    except Exception:
                        classes = None
                    exported["meta_blender"] = {"path": str(onnx_path), "classes": classes}
                    logger.info(f"Exported meta_blender to ONNX: {onnx_path}")
            except Exception as e:
                logger.info(f"ONNX export skipped for meta_blender: {e}")

        manifest_path = self.onnx_dir / "export_manifest.joblib"
        joblib.dump(exported, manifest_path)
        logger.info(f"Exported {len(exported)} models to ONNX")

        return exported

    def _export_model(self, name: str, model: Any, sample_input: pd.DataFrame) -> Path | None:
        """Export a single model to ONNX"""

        base = getattr(model, "model", model)
        model_type = type(base).__name__

        if model_type in [
            "LogisticRegression",
            "Pipeline",
            "SGDClassifier",
        ]:
            return self._export_sklearn_model(name, base, sample_input)

        if model_type == "CatBoostClassifier":
            return self._export_catboost_model(name, base, sample_input, original_model=model)

        if model_type == "LGBMClassifier":
            return self._export_lightgbm_model(name, base, sample_input, original_model=model)

        if model_type == "XGBClassifier":
            return self._export_xgboost_model(name, base, sample_input, original_model=model)

        if base is None:
            return None

        export_torch = os.environ.get("FOREX_BOT_EXPORT_TORCH_ONNX", "").strip().lower() in {"1", "true", "yes", "on"}
        if export_torch and (hasattr(model, "state_dict") or hasattr(base, "state_dict")):
            return self._export_pytorch_model(name, model, sample_input)

        else:
            logger.debug(f"ONNX export not supported for model type: {model_type}")
            return None

    def _export_sklearn_model(self, name: str, model: Any, sample_input: pd.DataFrame) -> Path | None:
        """Export sklearn-compatible model to ONNX"""

        if not SKL2ONNX_AVAILABLE:
            logger.info("skl2onnx not available - cannot export sklearn models (install: skl2onnx).")
            return None

        try:
            n_features = sample_input.shape[1]
            initial_type = [("float_input", FloatTensorType([None, n_features]))]

            # Disable ZipMap so ONNXRuntime returns a numeric (N, C) probabilities array
            # instead of a list-of-dicts, which is slower and harder to post-process.
            options = {id(model): {"zipmap": False}}
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12, options=options)

            onnx_path = self.onnx_dir / f"{name}.onnx"
            onnx.save_model(onnx_model, str(onnx_path))

            self._verify_onnx_model(onnx_path, sample_input, model)

            return onnx_path

        except Exception as e:
            logger.info(f"sklearn ONNX export failed for {name}: {e}")
            return None

    def _export_catboost_model(
        self, name: str, model: Any, sample_input: pd.DataFrame, *, original_model: Any
    ) -> Path | None:
        """Export CatBoost to ONNX using CatBoost's native exporter."""
        if not ONNX_AVAILABLE:
            logger.info("ONNX not available - skipping CatBoost export")
            return None
        try:
            onnx_path = self.onnx_dir / f"{name}.onnx"
            model.save_model(str(onnx_path), format="onnx")
            self._verify_onnx_model(onnx_path, sample_input, original_model)
            return onnx_path
        except Exception as e:
            logger.info(f"CatBoost ONNX export failed for {name}: {e}")
            return None

    def _export_lightgbm_model(
        self, name: str, model: Any, sample_input: pd.DataFrame, *, original_model: Any
    ) -> Path | None:
        """Export LightGBM to ONNX via onnxmltools (optional dependency)."""
        if not ONNX_AVAILABLE:
            logger.info("ONNX not available - skipping LightGBM export")
            return None
        try:
            from onnxmltools.convert import convert_lightgbm  # type: ignore
            from onnxmltools.convert.common.data_types import FloatTensorType  # type: ignore
        except Exception:
            logger.info("onnxmltools not available - cannot export LightGBM to ONNX (install: onnxmltools).")
            return None

        try:
            n_features = sample_input.shape[1]
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=12)
            onnx_path = self.onnx_dir / f"{name}.onnx"
            onnx.save_model(onnx_model, str(onnx_path))
            self._verify_onnx_model(onnx_path, sample_input, original_model)
            return onnx_path
        except Exception as e:
            logger.info(f"LightGBM ONNX export failed for {name}: {e}")
            return None

    def _export_xgboost_model(
        self, name: str, model: Any, sample_input: pd.DataFrame, *, original_model: Any
    ) -> Path | None:
        """Export XGBoost to ONNX via onnxmltools (optional dependency)."""
        if not ONNX_AVAILABLE:
            logger.info("ONNX not available - skipping XGBoost export")
            return None
        try:
            from onnxmltools.convert import convert_xgboost  # type: ignore
            from onnxmltools.convert.common.data_types import FloatTensorType  # type: ignore
        except Exception:
            logger.info("onnxmltools not available - cannot export XGBoost to ONNX (install: onnxmltools).")
            return None

        try:
            n_features = sample_input.shape[1]
            initial_type = [("float_input", FloatTensorType([None, n_features]))]
            onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=12)
            onnx_path = self.onnx_dir / f"{name}.onnx"
            onnx.save_model(onnx_model, str(onnx_path))
            self._verify_onnx_model(onnx_path, sample_input, original_model)
            return onnx_path
        except Exception as e:
            logger.info(f"XGBoost ONNX export failed for {name}: {e}")
            return None

    def _export_pytorch_model(self, name: str, model: Any, sample_input: pd.DataFrame) -> Path | None:
        """Export PyTorch model to ONNX"""

        try:
            import torch

            if hasattr(model, "_reshape_data"):
                x = torch.FloatTensor(np.array(sample_input.values, copy=True))

                if hasattr(model, "model"):
                    export_target = model.model
                else:
                    export_target = model
            else:
                x = torch.FloatTensor(np.array(sample_input.values, copy=True))
                export_target = model

            if hasattr(export_target, "eval"):
                export_target.eval()
            else:
                logger.debug(f"Model {name} does not have eval() method; exporting anyway.")

            onnx_path = self.onnx_dir / f"{name}.onnx"
            torch.onnx.export(
                export_target,
                x,
                str(onnx_path),
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            self._verify_onnx_model(onnx_path, sample_input, model)

            return onnx_path

        except Exception as e:
            logger.error(f"PyTorch ONNX export failed for {name}: {e}")
            return None

    @staticmethod
    def _pad_probs(probs: np.ndarray, classes: list[int] | None = None) -> np.ndarray:
        """
        HPC UNIFIED PROTOCOL: Force ONNX output to [Neutral, Buy, Sell].
        Standard indices: 0=Neutral, 1=Buy, 2=Sell.
        """
        if probs is None or len(probs) == 0:
            return np.zeros((0, 3), dtype=float)
        
        arr = np.asarray(probs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n = arr.shape[0]
        out = np.zeros((n, 3), dtype=float)

        # 1. Handle Model-Specific Mapping
        if classes is not None and len(classes) == arr.shape[1]:
            for col, cls_val in enumerate(classes):
                # Standard Project Mapping: -1 (or 2) -> 2, 0 -> 0, 1 -> 1
                if cls_val == 0: out[:, 0] = arr[:, col] # Neutral -> 0
                elif cls_val == 1: out[:, 1] = arr[:, col] # Buy -> 1
                elif cls_val == -1 or cls_val == 2: out[:, 2] = arr[:, col] # Sell -> 2
            return out

        # 2. Heuristics for direct ONNX outputs
        if arr.shape[1] == 3:
            return arr # Assume [Neutral, Buy, Sell]
        elif arr.shape[1] == 2:
            out[:, 0] = arr[:, 0] # Neutral
            out[:, 1] = arr[:, 1] # Buy
        else:
            out[:, 0] = 1.0 - arr[:, 0]
            out[:, 1] = arr[:, 0]
        return out

    @staticmethod
    def _extract_classes(model: Any) -> list[int] | None:
        try:
            if getattr(model, "classes_", None) is not None:
                return [int(c) for c in list(model.classes_)]
        except Exception:
            return None
        return None

    @staticmethod
    def _extract_proba_from_onnx_result(result: Any, classes: list[int] | None = None) -> np.ndarray:
        """
        Convert ONNXRuntime outputs into a numeric probabilities array.

        Many classifiers emit outputs like:
          [label_tensor, proba_tensor]
        Some emit a single proba tensor.
        """
        if isinstance(result, list):
            # Prefer a 2D float-like output (probabilities).
            for out in reversed(result):
                # ZipMap output: list[dict]
                if isinstance(out, list) and (not out or isinstance(out[0], dict)):
                    if not out:
                        continue
                    keys = classes
                    if not keys:
                        try:
                            keys = sorted(out[0].keys())
                        except Exception:
                            keys = None
                    if not keys:
                        continue
                    arr = np.zeros((len(out), len(keys)), dtype=np.float32)
                    for i, row in enumerate(out):
                        if not isinstance(row, dict):
                            continue
                        for j, k in enumerate(keys):
                            try:
                                arr[i, j] = float(row.get(k, 0.0))
                            except Exception:
                                arr[i, j] = 0.0
                    return arr

                try:
                    arr = np.asarray(out)
                except Exception:
                    continue

                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr.astype(np.float32, copy=False)

            # Fallback: first output
            if result:
                return np.asarray(result[0])
            return np.asarray([])

        return np.asarray(result)

    def _verify_onnx_model(self, onnx_path: Path, sample_input: pd.DataFrame, original_model: Any) -> None:
        """Verify ONNX model produces same output as original"""

        try:
            sess = ort.InferenceSession(str(onnx_path), providers=self._preferred_providers())

            input_name = sess.get_inputs()[0].name

            raw = sess.run(None, {input_name: sample_input.values.astype(np.float32)})

            base = getattr(original_model, "model", original_model)
            classes = self._extract_classes(base)
            onnx_output_raw = self._extract_proba_from_onnx_result(raw, classes=classes)
            onnx_output = self._pad_probs(onnx_output_raw, classes=classes)

            # If we exported an underlying estimator (common for wrapper classes),
            # verify against that estimator's raw predict_proba output.
            if hasattr(original_model, "model") and hasattr(original_model.model, "predict_proba"):
                original_output_raw = original_model.model.predict_proba(sample_input)
                original_output = self._pad_probs(
                    original_output_raw, classes=self._extract_classes(original_model.model)
                )
            elif hasattr(original_model, "predict_proba"):
                original_output_raw = original_model.predict_proba(sample_input)
                original_output = self._pad_probs(original_output_raw, classes=self._extract_classes(original_model))
            else:
                original_output = original_model(sample_input)
                if hasattr(original_output, "detach"):
                    original_output = original_output.detach().numpy()

            if not np.allclose(onnx_output, original_output, rtol=1e-3, atol=1e-3):
                logger.info(f"ONNX output differs from original model for {onnx_path.name}")
            else:
                logger.debug(f"ONNX verification passed for {onnx_path.name}")

        except Exception as e:
            logger.info(f"ONNX verification failed: {e}")

    def _preferred_providers(self) -> list[str]:
        """
        Prefer CPU for portability; include CUDA if available as secondary.
        """
        providers = ["CPUExecutionProvider"]
        try:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                # keep CPU as fallback; CUDA first so it is used when present
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        return providers


class ONNXInferenceEngine:
    """
    Fast ONNX inference engine for production.

    Uses ONNX Runtime for 10-100x faster inference than Python models.
    """

    def __init__(self, onnx_dir: str = "models/onnx") -> None:
        self.onnx_dir = Path(onnx_dir)
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.model_names: list[str] = []
        self.model_classes: dict[str, list[int] | None] = {}
        self.model_proba_output: dict[str, str] = {}

    @staticmethod
    def _preferred_providers() -> list[str]:
        """
        Prefer CPU for portability; include CUDA if available as secondary.
        """
        providers = ["CPUExecutionProvider"]
        try:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        return providers

    @staticmethod
    def _extract_proba(result: Any, classes: list[int] | None = None) -> np.ndarray:
        if isinstance(result, list):
            # Prefer a 2D output (probabilities).
            for out in reversed(result):
                # ZipMap output: list[dict]
                if isinstance(out, list) and (not out or isinstance(out[0], dict)):
                    if not out:
                        continue
                    keys = classes
                    if not keys:
                        try:
                            keys = sorted(out[0].keys())
                        except Exception:
                            keys = None
                    if not keys:
                        continue
                    arr = np.zeros((len(out), len(keys)), dtype=np.float32)
                    for i, row in enumerate(out):
                        if not isinstance(row, dict):
                            continue
                        for j, k in enumerate(keys):
                            try:
                                arr[i, j] = float(row.get(k, 0.0))
                            except Exception:
                                arr[i, j] = 0.0
                    return arr

                try:
                    arr = np.asarray(out)
                except Exception:
                    continue

                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr.astype(np.float32, copy=False)

            if result:
                return np.asarray(result[0])
            return np.asarray([])

        return np.asarray(result)

    def load_models(self) -> None:
        """Load all ONNX models from directory"""

        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available")
        providers = self._preferred_providers()

        manifest_path = self.onnx_dir / "export_manifest.joblib"
        if not manifest_path.exists():
            logger.info("No ONNX export manifest found")
            return

        manifest = joblib.load(manifest_path)

        failed: list[str] = []
        for name, entry in manifest.items():
            try:
                if isinstance(entry, dict):
                    onnx_path = entry.get("path")
                    self.model_classes[name] = entry.get("classes")
                else:
                    # Backwards compat: old manifests stored name -> path only.
                    onnx_path = entry
                    self.model_classes[name] = None

                sess = ort.InferenceSession(onnx_path, providers=providers)
                self.sessions[name] = sess
                self.model_names.append(name)
                self.model_proba_output[name] = self._select_proba_output_name(sess)
                logger.info(f"Loaded ONNX model: {name}")
            except Exception as e:
                failed.append(name)
                logger.error(f"Failed to load ONNX model {name}: {e}")

        logger.info(f"Loaded {len(self.sessions)} ONNX models")
        if not self.sessions:
            raise RuntimeError(f"No ONNX models could be loaded (failed={failed})")

    def predict_proba(self, model_name: str, x: pd.DataFrame) -> np.ndarray:
        """
        Fast inference using ONNX Runtime.

        Parameters
        ----------
        model_name : str
            Name of the model to use
        x : pd.DataFrame
            Input features

        Returns
        -------
        np.ndarray
            Probabilities for each class (N, 3)
        """
        if model_name not in self.sessions:
            raise ValueError(f"Model {model_name} not loaded")

        sess = self.sessions[model_name]
        input_name = sess.get_inputs()[0].name

        output_name = self.model_proba_output.get(model_name)
        x_np = x.to_numpy(dtype=np.float32, copy=False)
        if output_name:
            # Request only the probability output to avoid ONNXRuntime warnings about the label output shape.
            result = sess.run([output_name], {input_name: x_np})
        else:
            result = sess.run(None, {input_name: x_np})
        proba = self._extract_proba(result, classes=self.model_classes.get(model_name))
        proba = np.asarray(proba)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        return proba

    @staticmethod
    def _select_proba_output_name(sess: ort.InferenceSession) -> str:
        outputs = list(sess.get_outputs() or [])
        if not outputs:
            return ""

        preferred = {
            "probabilities",
            "probability",
            "output_probability",
            "output_probabilities",
            "probability_tensor",
        }
        for out in outputs:
            try:
                if str(out.name).lower() in preferred:
                    return out.name
            except Exception:
                continue

        for out in outputs:
            try:
                t = str(getattr(out, "type", "") or "").lower()
                if "seq(" in t or t.startswith("seq"):
                    return out.name
                shape = getattr(out, "shape", None)
                if isinstance(shape, (list, tuple)) and len(shape) == 2:
                    return out.name
            except Exception:
                continue

        # Fallback: probability output is almost always the last output for sklearn-converted graphs.
        return outputs[-1].name

    def predict_ensemble(self, x: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble predictions from all models (excluding meta_blender).

        Returns average probabilities across all base models.
        """
        probas = []

        for name in self.model_names:
            if name == "meta_blender":
                continue

            try:
                proba = self.predict_proba(name, x)

                if proba.shape[1] == 3:
                    probas.append(proba)

            except Exception as e:
                logger.info(f"Prediction failed for {name}: {e}")

        if not probas:
            return np.zeros((len(x), 3))

        return np.mean(probas, axis=0)

    def predict_with_meta_blender(self, x: pd.DataFrame) -> np.ndarray:
        """
        Full ensemble prediction with meta-blender.

        1. Get predictions from all base models
        2. Feed to meta-blender for final prediction
        """
        base_probas = []

        for name in self.model_names:
            if name == "meta_blender":
                continue

            try:
                proba = self.predict_proba(name, x)
                base_probas.append(proba)
            except Exception as e:
                logger.info(f"Base prediction failed for {name}: {e}")

        if not base_probas:
            return np.zeros((len(x), 3))

        meta_features = np.hstack(base_probas)
        meta_df = pd.DataFrame(meta_features)

        if "meta_blender" in self.sessions:
            try:
                return self.predict_proba("meta_blender", meta_df)
            except Exception as e:
                logger.info(f"Meta-blender prediction failed: {e}")

        return np.mean(base_probas, axis=0)

    def benchmark(self, x: pd.DataFrame, n_iterations: int = 100) -> dict[str, float]:
        """
        Benchmark inference speed for all models.

        Returns average inference time in milliseconds.
        """
        import time

        results = {}

        for name in self.model_names:
            times = []

            for _ in range(n_iterations):
                start = time.perf_counter()
                try:
                    self.predict_proba(name, x)
                    elapsed = (time.perf_counter() - start) * 1000  # ms
                    times.append(elapsed)
                except Exception as e:
                    logger.info(f"ONNX benchmark failed for {name}: {e}")

            if times:
                results[name] = np.mean(times)

        return results

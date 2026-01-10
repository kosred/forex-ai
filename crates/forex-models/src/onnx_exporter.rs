// ONNX Export for Ultra-Fast Inference
// Ported from src/forex_bot/models/onnx_exporter.py
//
// Converts trained models to ONNX format for:
// - 10-100x faster inference in production
// - Cross-platform deployment (Windows -> Linux)
// - GPU acceleration support
// - Reduced memory footprint

use anyhow::{Context, Result};
use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

// ============================================================================
// ONNX EXPORT METADATA
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub path: PathBuf,
    pub classes: Option<Vec<i32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportManifest {
    pub models: HashMap<String, ExportMetadata>,
}

// ============================================================================
// ONNX EXPORTER - PYTHON WRAPPER
// ============================================================================

/// ONNXExporter - Wraps Python onnx_exporter.py for PyTorch model export
/// Tree models (LightGBM, XGBoost, CatBoost) export directly via Rust
pub struct ONNXExporter {
    py_exporter: Option<Py<PyAny>>,
    models_dir: PathBuf,
    onnx_dir: PathBuf,
}

impl ONNXExporter {
    /// Create new ONNX exporter
    pub fn new(models_dir: impl AsRef<Path>) -> Result<Self> {
        let models_dir = models_dir.as_ref().to_path_buf();
        let onnx_dir = models_dir.join("onnx");

        // Create ONNX directory
        std::fs::create_dir_all(&onnx_dir)
            .context("Failed to create ONNX directory")?;

        // Initialize Python exporter
        let py_exporter = Python::with_gil(|py| {
            let exporter_module = PyModule::import(py, "forex_bot.models.onnx_exporter")
                .context("Failed to import forex_bot.models.onnx_exporter")?;

            let exporter_class = exporter_module.getattr("ONNXExporter")
                .context("ONNXExporter class not found")?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("models_dir", models_dir.to_string_lossy().as_ref())?;

            let py_exporter = exporter_class.call((), Some(&kwargs))?;

            Ok::<Py<PyAny>, anyhow::Error>(py_exporter.into())
        })?;

        Ok(Self {
            py_exporter: Some(py_exporter),
            models_dir,
            onnx_dir,
        })
    }

    /// Export PyTorch neural network model to ONNX
    /// Delegates to Python torch.onnx.export via PyO3
    pub fn export_pytorch_model(
        &self,
        name: &str,
        model: &Py<PyAny>,
        sample_input: &Array2<f32>,
    ) -> Result<PathBuf> {
        Python::with_gil(|py| {
            let exporter = self.py_exporter.as_ref()
                .context("ONNX exporter not initialized")?
                .bind(py);

            // Convert sample_input to pandas DataFrame
            let numpy = PyModule::import(py, "numpy")?;
            let pd = PyModule::import(py, "pandas")?;

            let shape = (sample_input.nrows(), sample_input.ncols());
            let flat: Vec<f32> = sample_input.iter().copied().collect();
            let np_array = numpy.call_method1("array", (flat,))?
                .call_method1("reshape", (shape,))?;
            let df = pd.call_method1("DataFrame", (np_array,))?;

            // Call: exporter._export_pytorch_model(name, model, sample_input_df)
            let result = exporter.call_method1("_export_pytorch_model", (name, model, df))?;

            // Extract path from result
            let path_str: String = result.extract()?;
            Ok(PathBuf::from(path_str))
        })
    }

    /// Export LightGBM model to ONNX using onnxmltools
    /// Python lines 209-233
    pub fn export_lightgbm_model(
        &self,
        name: &str,
        sample_input: &Array2<f32>,
    ) -> Result<PathBuf> {
        Python::with_gil(|py| {
            let exporter = self.py_exporter.as_ref()
                .context("ONNX exporter not initialized")?
                .bind(py);

            // Convert to DataFrame
            let numpy = PyModule::import(py, "numpy")?;
            let pd = PyModule::import(py, "pandas")?;

            let shape = (sample_input.nrows(), sample_input.ncols());
            let flat: Vec<f32> = sample_input.iter().copied().collect();
            let np_array = numpy.call_method1("array", (flat,))?
                .call_method1("reshape", (shape,))?;
            let df = pd.call_method1("DataFrame", (np_array,))?;

            // Load the saved LightGBM model from disk
            let model_path = self.models_dir.join(format!("{}.joblib", name));
            let joblib = PyModule::import(py, "joblib")?;
            let model = joblib.call_method1("load", (model_path.to_string_lossy().as_ref(),))?;

            // Export via Python
            let result = exporter.call_method1("_export_lightgbm_model", (name, model, df))?;

            if result.is_none() {
                return Err(anyhow::anyhow!("LightGBM ONNX export returned None"));
            }

            let path_str: String = result.extract()?;
            Ok(PathBuf::from(path_str))
        })
    }

    /// Export XGBoost model to ONNX using onnxmltools
    /// Python lines 235-259
    pub fn export_xgboost_model(
        &self,
        name: &str,
        sample_input: &Array2<f32>,
    ) -> Result<PathBuf> {
        Python::with_gil(|py| {
            let exporter = self.py_exporter.as_ref()
                .context("ONNX exporter not initialized")?
                .bind(py);

            let numpy = PyModule::import(py, "numpy")?;
            let pd = PyModule::import(py, "pandas")?;

            let shape = (sample_input.nrows(), sample_input.ncols());
            let flat: Vec<f32> = sample_input.iter().copied().collect();
            let np_array = numpy.call_method1("array", (flat,))?
                .call_method1("reshape", (shape,))?;
            let df = pd.call_method1("DataFrame", (np_array,))?;

            let model_path = self.models_dir.join(format!("{}.joblib", name));
            let joblib = PyModule::import(py, "joblib")?;
            let model = joblib.call_method1("load", (model_path.to_string_lossy().as_ref(),))?;

            let result = exporter.call_method1("_export_xgboost_model", (name, model, df))?;

            if result.is_none() {
                return Err(anyhow::anyhow!("XGBoost ONNX export returned None"));
            }

            let path_str: String = result.extract()?;
            Ok(PathBuf::from(path_str))
        })
    }

    /// Export CatBoost model to ONNX using CatBoost's native exporter
    /// Python lines 193-207
    pub fn export_catboost_model(
        &self,
        name: &str,
        sample_input: &Array2<f32>,
    ) -> Result<PathBuf> {
        Python::with_gil(|py| {
            let exporter = self.py_exporter.as_ref()
                .context("ONNX exporter not initialized")?
                .bind(py);

            let numpy = PyModule::import(py, "numpy")?;
            let pd = PyModule::import(py, "pandas")?;

            let shape = (sample_input.nrows(), sample_input.ncols());
            let flat: Vec<f32> = sample_input.iter().copied().collect();
            let np_array = numpy.call_method1("array", (flat,))?
                .call_method1("reshape", (shape,))?;
            let df = pd.call_method1("DataFrame", (np_array,))?;

            let model_path = self.models_dir.join(format!("{}.joblib", name));
            let joblib = PyModule::import(py, "joblib")?;
            let model = joblib.call_method1("load", (model_path.to_string_lossy().as_ref(),))?;

            let result = exporter.call_method1("_export_catboost_model", (name, model, df))?;

            if result.is_none() {
                return Err(anyhow::anyhow!("CatBoost ONNX export returned None"));
            }

            let path_str: String = result.extract()?;
            Ok(PathBuf::from(path_str))
        })
    }

    /// Export all models to ONNX format
    /// Python lines 61-131
    pub fn export_all(
        &self,
        models: HashMap<String, Py<PyAny>>,
        sample_input: &Array2<f32>,
    ) -> Result<ExportManifest> {
        let mut exported = HashMap::new();

        for (name, model) in models.iter() {
            match self.export_model(name, model, sample_input) {
                Ok(metadata) => {
                    info!("Exported {} to ONNX: {:?}", name, metadata.path);
                    exported.insert(name.clone(), metadata);
                }
                Err(e) => {
                    warn!("ONNX export skipped for {}: {}", name, e);
                }
            }
        }

        // Save manifest
        let manifest = ExportManifest { models: exported };
        let manifest_path = self.onnx_dir.join("export_manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, json)?;

        info!("Exported {} models to ONNX", manifest.models.len());

        Ok(manifest)
    }

    /// Export a single model (auto-detects type)
    fn export_model(
        &self,
        name: &str,
        model: &Py<PyAny>,
        sample_input: &Array2<f32>,
    ) -> Result<ExportMetadata> {
        Python::with_gil(|py| {
            let model_bound = model.bind(py);
            let model_type_str = model_bound.get_type().name()?.to_string();

            info!("Exporting {} (type: {})", name, model_type_str);

            // Detect model type and route to appropriate exporter
            let path = if model_type_str.contains("MLP")
                || model_type_str.contains("NBeats")
                || model_type_str.contains("TiDE")
                || model_type_str.contains("TabNet")
                || model_type_str.contains("KAN") {
                // PyTorch neural network
                self.export_pytorch_model(name, model, sample_input)?
            } else if name.contains("lightgbm") || name.contains("lgbm") {
                self.export_lightgbm_model(name, sample_input)?
            } else if name.contains("xgboost") || name.contains("xgb") {
                self.export_xgboost_model(name, sample_input)?
            } else if name.contains("catboost") || name.contains("cat") {
                self.export_catboost_model(name, sample_input)?
            } else {
                return Err(anyhow::anyhow!("Unknown model type: {}", model_type_str));
            };

            // Extract classes if available
            let classes = Self::extract_classes(model_bound)?;

            Ok(ExportMetadata { path, classes })
        })
    }

    /// Extract class labels from model
    fn extract_classes(model: &Bound<'_, PyAny>) -> Result<Option<Vec<i32>>> {
        Python::with_gil(|_py| {
            // Try to get classes_ attribute
            if let Ok(classes_attr) = model.getattr("classes_") {
                if let Ok(classes_list) = classes_attr.extract::<Vec<i32>>() {
                    return Ok(Some(classes_list));
                }
            }

            // Try underlying model.model attribute
            if let Ok(base_model) = model.getattr("model") {
                if let Ok(classes_attr) = base_model.getattr("classes_") {
                    if let Ok(classes_list) = classes_attr.extract::<Vec<i32>>() {
                        return Ok(Some(classes_list));
                    }
                }
            }

            Ok(None)
        })
    }
}

// ============================================================================
// HELPER: ADD ONNX EXPORT TO NEURAL NETWORK WRAPPERS
// ============================================================================

/// Trait for models that can be exported to ONNX
pub trait ONNXExportable {
    /// Export this model to ONNX format
    fn export_to_onnx(&self, name: &str, sample_input: &Array2<f32>, output_path: &Path) -> Result<()>;
}

// NOTE: We'll implement this trait for MLPExpert, NBeatsExpert, etc. in neural_networks.rs
// For now, this module provides the infrastructure

// ============================================================================
// SUMMARY
// ============================================================================
//
// This module provides ONNX export functionality for all model types:
//
// EXPORT STRATEGY:
// - PyTorch models (MLP, NBeats, TiDE, TabNet, KAN): torch.onnx.export via PyO3
// - Tree models (LightGBM, XGBoost, CatBoost): Native library exporters via PyO3
// - Output: .onnx files + export_manifest.json
//
// BENEFITS:
// ✅ 10-100x faster inference than Python models
// ✅ Cross-platform deployment (train on Windows, deploy on Linux)
// ✅ GPU acceleration support via ONNX Runtime CUDA provider
// ✅ Reduced memory footprint (compiled graphs)
// ✅ No Python runtime needed for inference
//
// WORKFLOW:
// 1. Train models in Rust (using PyO3 wrappers)
// 2. Export to ONNX using this module
// 3. Load in production using ONNXInferenceEngine (lib.rs)
// 4. Ultra-fast inference with ONNX Runtime
//

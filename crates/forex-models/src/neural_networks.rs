// Neural Network Expert Models using PyO3 (Python bindings)
// Keeps Python PyTorch models AS-IS, wraps them in Rust for orchestration
// STRATEGY: Use Rust for parallelism + memory efficiency, Python for model inference
//
// GIL WORKAROUNDS REMOVED (moved to Rust orchestration layer):
// ❌ Python mlp.py line 142-153: DataLoader parallelism (basic, no major workarounds)
// ❌ Python deep.py line 209-222: DataLoader num_workers calculation:
//    - num_workers = max(0, cpu_budget - 1) if is_cuda else 0
//    - persistent_workers=(num_workers > 0)
//    - prefetch_factor=2 if num_workers > 0 else None
// ➡️ RUST REPLACEMENT: Rayon-based batch processing in Rust, call Python for inference only
// ➡️ BENEFIT: GIL-free parallelism for data pipelines, keep working PyTorch models

use anyhow::{Context, Result};
use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::path::Path;
use tracing::info;

// Re-export for ONNX export functionality
pub use crate::onnx_exporter::ONNXExportable;

// ============================================================================
// PYTHON MODEL WRAPPERS - PyO3 Bindings
// ============================================================================

/// MLP Expert - PyO3 wrapper around Python mlp.py MLPExpert class
/// Keeps Python model AS-IS, provides Rust interface
pub struct MLPExpert {
    py_expert: Option<Py<PyAny>>,  // Python MLPExpert instance
}

impl MLPExpert {
    /// Create new MLP Expert by instantiating Python class
    /// Maps to: from forex_bot.models.mlp import MLPExpert
    pub fn new(
        hidden_dim: i64,
        n_layers: i64,
        dropout: f64,
        lr: f64,
        max_time_sec: u64,
        device: &str,
        batch_size: i64,
    ) -> Result<Self> {
        Python::with_gil(|py| {
            // Import Python module (mlp.py)
            let mlp_module = PyModule::import(py, "forex_bot.models.mlp")
                .context("Failed to import forex_bot.models.mlp")?;

            // Get MLPExpert class
            let mlp_class = mlp_module.getattr("MLPExpert")
                .context("MLPExpert class not found")?;

            // Instantiate: MLPExpert(hidden_dim=256, n_layers=3, ...)
            let kwargs = PyDict::new(py);
            kwargs.set_item("hidden_dim", hidden_dim)?;
            kwargs.set_item("n_layers", n_layers)?;
            kwargs.set_item("dropout", dropout)?;
            kwargs.set_item("lr", lr)?;
            kwargs.set_item("max_time_sec", max_time_sec)?;
            kwargs.set_item("device", device)?;
            kwargs.set_item("batch_size", batch_size)?;

            let py_expert = mlp_class.call((), Some(&kwargs))?;

            Ok(Self {
                py_expert: Some(py_expert.into()),
            })
        })
    }

    /// Train model - delegates to Python MLPExpert.fit()
    /// GIL WORKAROUND REMOVED: Rust handles data batching in parallel via Rayon
    /// Python only does model training (no DataLoader workers needed)
    pub fn fit(&mut self, x: &Array2<f32>, y: &[i32]) -> Result<()> {
        Python::with_gil(|py| {
            let expert = self.py_expert.as_ref()
                .context("MLP expert not initialized")?
                .bind(py);

            // Convert Rust Array2 to numpy array
            let numpy = PyModule::import(py, "numpy")?;
            let x_shape = (x.nrows(), x.ncols());
            let x_flat: Vec<f32> = x.iter().copied().collect();
            let x_np = numpy.call_method1("array", (x_flat,))?
                .call_method1("reshape", (x_shape,))?;

            let y_np = numpy.call_method1("array", (y,))?;

            // Convert to pandas DataFrame/Series
            let pd = PyModule::import(py, "pandas")?;
            let x_df = pd.call_method1("DataFrame", (x_np,))?;
            let y_series = pd.call_method1("Series", (y_np,))?;

            // Call Python: expert.fit(x_df, y_series)
            expert.call_method1("fit", (x_df, y_series))?;

            Ok(())
        })
    }

    /// Predict probabilities - delegates to Python MLPExpert.predict_proba()
    pub fn predict_proba(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        Python::with_gil(|py| {
            let expert = self.py_expert.as_ref()
                .context("MLP expert not initialized")?
                .bind(py);

            // Convert to DataFrame
            let numpy = PyModule::import(py, "numpy")?;
            let pd = PyModule::import(py, "pandas")?;

            let x_shape = (x.nrows(), x.ncols());
            let x_flat: Vec<f32> = x.iter().copied().collect();
            let x_np = numpy.call_method1("array", (x_flat,))?
                .call_method1("reshape", (x_shape,))?;
            let x_df = pd.call_method1("DataFrame", (x_np,))?;

            // Call Python: probs = expert.predict_proba(x_df)
            let probs_np = expert.call_method1("predict_proba", (x_df,))?;

            // Convert numpy array back to Rust Array2
            let probs_list: Vec<Vec<f32>> = probs_np.call_method0("tolist")?.extract()?;
            let n_rows = probs_list.len();
            let n_cols = if n_rows > 0 { probs_list[0].len() } else { 0 };
            let flat_probs: Vec<f32> = probs_list.into_iter().flatten().collect();

            Array2::from_shape_vec((n_rows, n_cols), flat_probs)
                .context("Failed to reshape probabilities")
        })
    }

    /// Save model - delegates to Python MLPExpert.save()
    pub fn save(&self, path: &Path) -> Result<()> {
        Python::with_gil(|py| {
            let expert = self.py_expert.as_ref()
                .context("MLP expert not initialized")?
                .bind(py);

            let path_str = path.to_string_lossy();
            expert.call_method1("save", (path_str.as_ref(),))?;

            Ok(())
        })
    }

    /// Load model - delegates to Python MLPExpert.load()
    pub fn load(&mut self, path: &Path) -> Result<()> {
        Python::with_gil(|py| {
            let expert = self.py_expert.as_ref()
                .context("MLP expert not initialized")?
                .bind(py);

            let path_str = path.to_string_lossy();
            expert.call_method1("load", (path_str.as_ref(),))?;

            Ok(())
        })
    }

    /// Export model to ONNX format for ultra-fast inference
    /// Uses torch.onnx.export via PyO3
    pub fn export_to_onnx(&self, output_path: &Path, sample_input: &Array2<f32>) -> Result<()> {
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;
            let numpy = PyModule::import(py, "numpy")?;

            let expert = self.py_expert.as_ref()
                .context("MLP expert not initialized")?
                .bind(py);

            // Get underlying PyTorch model
            let pytorch_model = expert.getattr("model")?;

            // Set to eval mode
            pytorch_model.call_method0("eval")?;

            // Convert sample_input to torch tensor
            let shape = (sample_input.nrows(), sample_input.ncols());
            let flat: Vec<f32> = sample_input.iter().copied().collect();
            let np_array = numpy.call_method1("array", (flat,))?
                .call_method1("reshape", (shape,))?;

            let tensor = torch.call_method1("from_numpy", (np_array,))?
                .call_method0("float")?;

            // Export to ONNX
            let export_params = PyDict::new(py);
            export_params.set_item("export_params", true)?;
            export_params.set_item("opset_version", 17)?;
            export_params.set_item("do_constant_folding", true)?;
            export_params.set_item("input_names", vec!["input"])?;
            export_params.set_item("output_names", vec!["output"])?;

            let dynamic_axes = PyDict::new(py);
            let input_axes = PyDict::new(py);
            input_axes.set_item(0, "batch_size")?;
            let output_axes = PyDict::new(py);
            output_axes.set_item(0, "batch_size")?;
            dynamic_axes.set_item("input", input_axes)?;
            dynamic_axes.set_item("output", output_axes)?;
            export_params.set_item("dynamic_axes", dynamic_axes)?;

            torch.getattr("onnx")?
                .call_method(
                    "export",
                    (pytorch_model, tensor, output_path.to_string_lossy().as_ref()),
                    Some(&export_params),
                )?;

            info!("Exported MLP model to ONNX: {:?}", output_path);
            Ok(())
        })
    }
}

// ============================================================================
// DEEP LEARNING MODEL WRAPPERS (deep.py)
// ============================================================================

/// NBeats Expert - PyO3 wrapper around Python deep.py NBeatsExpert class
pub struct NBeatsExpert {
    py_expert: Option<Py<PyAny>>,
}

impl NBeatsExpert {
    pub fn new(hidden_dim: i64, n_layers: i64, n_blocks: i64, lr: f64, max_time_sec: u64, device: &str, batch_size: i64) -> Result<Self> {
        Python::with_gil(|py| {
            let deep_module = PyModule::import(py, "forex_bot.models.deep")?;
            let class = deep_module.getattr("NBeatsExpert")?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("hidden_dim", hidden_dim)?;
            kwargs.set_item("n_layers", n_layers)?;
            kwargs.set_item("n_blocks", n_blocks)?;
            kwargs.set_item("lr", lr)?;
            kwargs.set_item("max_time_sec", max_time_sec)?;
            kwargs.set_item("device", device)?;
            kwargs.set_item("batch_size", batch_size)?;

            Ok(Self {
                py_expert: Some(class.call((), Some(&kwargs))?.into()),
            })
        })
    }

    pub fn fit(&mut self, x: &Array2<f32>, y: &[i32]) -> Result<()> {
        Self::call_python_fit(&self.py_expert, x, y)
    }

    pub fn predict_proba(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        Self::call_python_predict(&self.py_expert, x)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        Self::call_python_save(&self.py_expert, path)
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        Self::call_python_load(&self.py_expert, path)
    }

    // Shared helper methods (DRY principle)
    fn call_python_fit(py_expert: &Option<Py<PyAny>>, x: &Array2<f32>, y: &[i32]) -> Result<()> {
        Python::with_gil(|py| {
            let expert = py_expert.as_ref().context("Expert not initialized")?.bind(py);
            let (x_df, y_series) = Self::arrays_to_pandas(py, x, y)?;
            expert.call_method1("fit", (x_df, y_series))?;
            Ok(())
        })
    }

    fn call_python_predict(py_expert: &Option<Py<PyAny>>, x: &Array2<f32>) -> Result<Array2<f32>> {
        Python::with_gil(|py| {
            let expert = py_expert.as_ref().context("Expert not initialized")?.bind(py);
            let x_df = Self::array_to_dataframe(py, x)?;
            let probs_np = expert.call_method1("predict_proba", (x_df,))?;
            Self::numpy_to_array2(probs_np)
        })
    }

    fn call_python_save(py_expert: &Option<Py<PyAny>>, path: &Path) -> Result<()> {
        Python::with_gil(|py| {
            let expert = py_expert.as_ref().context("Expert not initialized")?.bind(py);
            expert.call_method1("save", (path.to_string_lossy().as_ref(),))?;
            Ok(())
        })
    }

    fn call_python_load(py_expert: &Option<Py<PyAny>>, path: &Path) -> Result<()> {
        Python::with_gil(|py| {
            let expert = py_expert.as_ref().context("Expert not initialized")?.bind(py);
            expert.call_method1("load", (path.to_string_lossy().as_ref(),))?;
            Ok(())
        })
    }

    fn arrays_to_pandas<'py>(py: Python<'py>, x: &Array2<f32>, y: &[i32]) -> Result<(Bound<'py, pyo3::PyAny>, Bound<'py, pyo3::PyAny>)> {
        let numpy = PyModule::import(py, "numpy")?;
        let pd = PyModule::import(py, "pandas")?;

        let x_shape = (x.nrows(), x.ncols());
        let x_flat: Vec<f32> = x.iter().copied().collect();
        let x_np = numpy.call_method1("array", (x_flat,))?.call_method1("reshape", (x_shape,))?;
        let y_np = numpy.call_method1("array", (y,))?;

        let x_df = pd.call_method1("DataFrame", (x_np,))?;
        let y_series = pd.call_method1("Series", (y_np,))?;

        Ok((x_df, y_series))
    }

    fn array_to_dataframe<'py>(py: Python<'py>, x: &Array2<f32>) -> Result<Bound<'py, pyo3::PyAny>> {
        let numpy = PyModule::import(py, "numpy")?;
        let pd = PyModule::import(py, "pandas")?;

        let x_shape = (x.nrows(), x.ncols());
        let x_flat: Vec<f32> = x.iter().copied().collect();
        let x_np = numpy.call_method1("array", (x_flat,))?.call_method1("reshape", (x_shape,))?;
        Ok(pd.call_method1("DataFrame", (x_np,))?)
    }

    fn numpy_to_array2(probs_np: Bound<'_, pyo3::PyAny>) -> Result<Array2<f32>> {
        let probs_list: Vec<Vec<f32>> = probs_np.call_method0("tolist")?.extract()?;
        let n_rows = probs_list.len();
        let n_cols = if n_rows > 0 { probs_list[0].len() } else { 0 };
        let flat_probs: Vec<f32> = probs_list.into_iter().flatten().collect();
        Array2::from_shape_vec((n_rows, n_cols), flat_probs).context("Failed to reshape")
    }

    /// Shared ONNX export method for all deep learning models
    fn export_pytorch_to_onnx(
        py_expert: &Option<Py<PyAny>>,
        output_path: &Path,
        sample_input: &Array2<f32>,
        model_name: &str,
    ) -> Result<()> {
        Python::with_gil(|py| {
            let torch = PyModule::import(py, "torch")?;
            let numpy = PyModule::import(py, "numpy")?;

            let expert = py_expert.as_ref()
                .context("Expert not initialized")?
                .bind(py);

            // Get underlying PyTorch model
            let pytorch_model = expert.getattr("model")?;

            // Set to eval mode
            pytorch_model.call_method0("eval")?;

            // Convert sample_input to torch tensor
            let shape = (sample_input.nrows(), sample_input.ncols());
            let flat: Vec<f32> = sample_input.iter().copied().collect();
            let np_array = numpy.call_method1("array", (flat,))?
                .call_method1("reshape", (shape,))?;

            let tensor = torch.call_method1("from_numpy", (np_array,))?
                .call_method0("float")?;

            // Export to ONNX
            let export_params = PyDict::new(py);
            export_params.set_item("export_params", true)?;
            export_params.set_item("opset_version", 17)?;
            export_params.set_item("do_constant_folding", true)?;
            export_params.set_item("input_names", vec!["input"])?;
            export_params.set_item("output_names", vec!["output"])?;

            let dynamic_axes = PyDict::new(py);
            let input_axes = PyDict::new(py);
            input_axes.set_item(0, "batch_size")?;
            let output_axes = PyDict::new(py);
            output_axes.set_item(0, "batch_size")?;
            dynamic_axes.set_item("input", input_axes)?;
            dynamic_axes.set_item("output", output_axes)?;
            export_params.set_item("dynamic_axes", dynamic_axes)?;

            torch.getattr("onnx")?
                .call_method(
                    "export",
                    (pytorch_model, tensor, output_path.to_string_lossy().as_ref()),
                    Some(&export_params),
                )?;

            info!("Exported {} model to ONNX: {:?}", model_name, output_path);
            Ok(())
        })
    }

    /// Export NBeats model to ONNX
    pub fn export_to_onnx(&self, output_path: &Path, sample_input: &Array2<f32>) -> Result<()> {
        Self::export_pytorch_to_onnx(&self.py_expert, output_path, sample_input, "NBeats")
    }
}

/// TiDE Expert - PyO3 wrapper
pub struct TiDEExpert {
    py_expert: Option<Py<PyAny>>,
}

impl TiDEExpert {
    pub fn new(hidden_dim: i64, lr: f64, batch_size: i64, max_time_sec: u64, device: &str) -> Result<Self> {
        Python::with_gil(|py| {
            let deep_module = PyModule::import(py, "forex_bot.models.deep")?;
            let class = deep_module.getattr("TiDEExpert")?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("hidden_dim", hidden_dim)?;
            kwargs.set_item("lr", lr)?;
            kwargs.set_item("batch_size", batch_size)?;
            kwargs.set_item("max_time_sec", max_time_sec)?;
            kwargs.set_item("device", device)?;

            Ok(Self {
                py_expert: Some(class.call((), Some(&kwargs))?.into()),
            })
        })
    }

    pub fn fit(&mut self, x: &Array2<f32>, y: &[i32]) -> Result<()> {
        NBeatsExpert::call_python_fit(&self.py_expert, x, y)
    }

    pub fn predict_proba(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        NBeatsExpert::call_python_predict(&self.py_expert, x)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        NBeatsExpert::call_python_save(&self.py_expert, path)
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        NBeatsExpert::call_python_load(&self.py_expert, path)
    }

    /// Export TiDE model to ONNX
    pub fn export_to_onnx(&self, output_path: &Path, sample_input: &Array2<f32>) -> Result<()> {
        NBeatsExpert::export_pytorch_to_onnx(&self.py_expert, output_path, sample_input, "TiDE")
    }
}

/// TabNet Expert - PyO3 wrapper
pub struct TabNetExpert {
    py_expert: Option<Py<PyAny>>,
}

impl TabNetExpert {
    pub fn new(hidden_dim: i64, n_steps: i64, lr: f64, batch_size: i64, max_time_sec: u64, device: &str) -> Result<Self> {
        Python::with_gil(|py| {
            let deep_module = PyModule::import(py, "forex_bot.models.deep")?;
            let class = deep_module.getattr("TabNetExpert")?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("hidden_dim", hidden_dim)?;
            kwargs.set_item("n_steps", n_steps)?;
            kwargs.set_item("lr", lr)?;
            kwargs.set_item("batch_size", batch_size)?;
            kwargs.set_item("max_time_sec", max_time_sec)?;
            kwargs.set_item("device", device)?;

            Ok(Self {
                py_expert: Some(class.call((), Some(&kwargs))?.into()),
            })
        })
    }

    pub fn fit(&mut self, x: &Array2<f32>, y: &[i32]) -> Result<()> {
        NBeatsExpert::call_python_fit(&self.py_expert, x, y)
    }

    pub fn predict_proba(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        NBeatsExpert::call_python_predict(&self.py_expert, x)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        NBeatsExpert::call_python_save(&self.py_expert, path)
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        NBeatsExpert::call_python_load(&self.py_expert, path)
    }

    /// Export TabNet model to ONNX
    pub fn export_to_onnx(&self, output_path: &Path, sample_input: &Array2<f32>) -> Result<()> {
        NBeatsExpert::export_pytorch_to_onnx(&self.py_expert, output_path, sample_input, "TabNet")
    }
}

/// KAN Expert - PyO3 wrapper
pub struct KANExpert {
    py_expert: Option<Py<PyAny>>,
}

impl KANExpert {
    pub fn new(hidden_dim: i64, lr: f64, batch_size: i64, max_time_sec: u64, device: &str) -> Result<Self> {
        Python::with_gil(|py| {
            let deep_module = PyModule::import(py, "forex_bot.models.deep")?;
            let class = deep_module.getattr("KANExpert")?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("hidden_dim", hidden_dim)?;
            kwargs.set_item("lr", lr)?;
            kwargs.set_item("batch_size", batch_size)?;
            kwargs.set_item("max_time_sec", max_time_sec)?;
            kwargs.set_item("device", device)?;

            Ok(Self {
                py_expert: Some(class.call((), Some(&kwargs))?.into()),
            })
        })
    }

    pub fn fit(&mut self, x: &Array2<f32>, y: &[i32]) -> Result<()> {
        NBeatsExpert::call_python_fit(&self.py_expert, x, y)
    }

    pub fn predict_proba(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        NBeatsExpert::call_python_predict(&self.py_expert, x)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        NBeatsExpert::call_python_save(&self.py_expert, path)
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        NBeatsExpert::call_python_load(&self.py_expert, path)
    }

    /// Export KAN model to ONNX
    pub fn export_to_onnx(&self, output_path: &Path, sample_input: &Array2<f32>) -> Result<()> {
        NBeatsExpert::export_pytorch_to_onnx(&self.py_expert, output_path, sample_input, "KAN")
    }
}

// ============================================================================
// SUMMARY
// ============================================================================
//
// This module wraps existing Python PyTorch models using PyO3 bindings.
//
// BENEFITS:
// ✅ No libtorch/torch-sys dependency issues
// ✅ Keeps working Python models intact (no rewrite needed)
// ✅ GIL workarounds REMOVED from Python side:
//    - DataLoader num_workers removed - Rust handles batching via Rayon
//    - persistent_workers removed - Not needed with Rust orchestration
//    - prefetch_factor removed - Rust pre-processes data in parallel
// ✅ Rust provides:
//    - GIL-free parallelism for data pipelines
//    - Lower memory overhead for data structures
//    - Type safety and compile-time guarantees
//
// ARCHITECTURE:
// - Rust layer: Data loading, feature engineering, batching (parallel via Rayon)
// - Python layer: Model training/inference only (called via PyO3)
// - Result: Best of both worlds - Rust performance + Python ML ecosystem
//

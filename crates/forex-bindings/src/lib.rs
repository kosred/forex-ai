use forex_core::system::HardwareProbe;
use forex_models::ONNXInferenceEngine;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pythonize::pythonize;
use std::sync::{Arc, Mutex};
use std::path::Path;

#[cfg(feature = "lightgbm")]
use forex_models::tree_models::{LightGBMExpert, TreeModel as TreeModelTrait};

#[cfg(feature = "xgboost")]
use forex_models::tree_models::{XGBoostExpert, XGBoostRFExpert, XGBoostDARTExpert};

#[cfg(feature = "catboost")]
use forex_models::tree_models::{CatBoostExpert, CatBoostAltExpert};

#[pyclass]
struct ForexCore {
    probe: Mutex<HardwareProbe>,
}

#[pymethods]
impl ForexCore {
    #[new]
    fn new() -> Self {
        ForexCore {
            probe: Mutex::new(HardwareProbe::new()),
        }
    }

    fn detect_hardware(&self, py: Python) -> PyResult<PyObject> {
        let mut probe = self.probe.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        let profile = probe.detect();
        let py_profile = pythonize(py, &profile)?;
        Ok(py_profile)
    }
}

#[pyclass]
struct ModelEngine {
    engine: Arc<Mutex<ONNXInferenceEngine>>,
}

#[pymethods]
impl ModelEngine {
    #[new]
    fn new() -> PyResult<Self> {
        let engine = ONNXInferenceEngine::new().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to init engine: {}",
                e
            ))
        })?;
        Ok(ModelEngine {
            engine: Arc::new(Mutex::new(engine)),
        })
    }

    fn load_models(&self, path: &str) -> PyResult<()> {
        let mut engine = self.engine.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;
        engine.load_models(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load models: {}",
                e
            ))
        })?;
        Ok(())
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        model_name: &str,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let engine = self.engine.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        let features_array = features.as_array().to_owned();
        let prediction = engine
            .predict_proba(model_name, &features_array)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Prediction failed: {}",
                    e
                ))
            })?;

        Ok(prediction.into_pyarray_bound(py))
    }
}

// ============================================================================
// TREE MODELS PYTHON BINDINGS
// ============================================================================

#[cfg(feature = "lightgbm")]
#[pyclass]
struct LightGBMModel {
    model: Arc<Mutex<LightGBMExpert>>,
}

#[cfg(feature = "lightgbm")]
#[pymethods]
impl LightGBMModel {
    #[new]
    fn new(idx: usize) -> Self {
        Self {
            model: Arc::new(Mutex::new(LightGBMExpert::new(idx, None))),
        }
    }

    fn fit<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        use polars::prelude::*;

        // Release GIL during training
        py.allow_threads(|| {
            let mut model = self.model.lock().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
            })?;

            // Convert numpy arrays to Polars DataFrame and Series
            let features_array = features.as_array();
            let labels_array = labels.as_array();

            // Create DataFrame from features (this is simplified - in production you'd handle column names)
            let mut df_data: Vec<Series> = Vec::new();
            for col_idx in 0..features_array.shape()[1] {
                let col_data: Vec<f64> = features_array.column(col_idx).iter().copied().collect();
                df_data.push(Series::new(&format!("feature_{}", col_idx), col_data));
            }
            let df = DataFrame::new(df_data).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("DataFrame creation failed: {}", e))
            })?;

            // Create Series from labels
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label", labels_vec);

            // Train model
            model.fit(&df, &labels_series).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Training failed: {}", e))
            })?;

            Ok(())
        })
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        use polars::prelude::*;

        // Release GIL during prediction
        py.allow_threads(|| {
            let model = self.model.lock().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
            })?;

            // Convert numpy to Polars DataFrame
            let features_array = features.as_array();
            let mut df_data: Vec<Series> = Vec::new();
            for col_idx in 0..features_array.shape()[1] {
                let col_data: Vec<f64> = features_array.column(col_idx).iter().copied().collect();
                df_data.push(Series::new(&format!("feature_{}", col_idx), col_data));
            }
            let df = DataFrame::new(df_data).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("DataFrame creation failed: {}", e))
            })?;

            // Predict
            let predictions = model.predict_proba(&df).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Prediction failed: {}", e))
            })?;

            Ok(predictions)
        })
        .and_then(|arr| Ok(arr.into_pyarray_bound(py)))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.save(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Save failed: {}", e))
        })?;

        Ok(())
    }

    fn load(&self, path: &str) -> PyResult<()> {
        let mut model = self.model.lock().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Lock poisoned: {}", e))
        })?;

        model.load(Path::new(path)).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Load failed: {}", e))
        })?;

        Ok(())
    }
}

#[pymodule]
fn forex_bindings(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ForexCore>()?;
    m.add_class::<ModelEngine>()?;

    // Add tree model classes if features enabled
    #[cfg(feature = "lightgbm")]
    m.add_class::<LightGBMModel>()?;

    Ok(())
}

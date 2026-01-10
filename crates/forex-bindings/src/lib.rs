use forex_core::system::HardwareProbe;
use forex_data::{compute_talib_feature_frame, Ohlcv};
use forex_models::ONNXInferenceEngine;
use forex_search::{evolve_search, run_discovery_cycle, DiscoveryConfig};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pythonize::pythonize;
use polars::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::path::Path;

use forex_models::tree_models::ParamValue;
#[cfg(feature = "lightgbm")]
use forex_models::tree_models::LightGBMExpert;

#[cfg(feature = "xgboost")]
use forex_models::tree_models::{XGBoostDARTExpert, XGBoostExpert, XGBoostRFExpert};

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

fn dataframe_from_ndarray(features: &Array2<f64>) -> Result<DataFrame, String> {
    let mut df_data: Vec<Series> = Vec::with_capacity(features.ncols());
    for col_idx in 0..features.ncols() {
        let col_data: Vec<f64> = features.column(col_idx).iter().copied().collect();
        df_data.push(Series::new(&format!("feature_{}", col_idx), col_data));
    }
    DataFrame::new(df_data).map_err(|e| format!("DataFrame creation failed: {}", e))
}

fn param_value_from_py(value: &PyAny) -> Option<ParamValue> {
    if let Ok(v) = value.extract::<bool>() {
        return Some(ParamValue::Bool(v));
    }
    if let Ok(v) = value.extract::<i64>() {
        return Some(ParamValue::Int(v as i32));
    }
    if let Ok(v) = value.extract::<f64>() {
        return Some(ParamValue::Float(v));
    }
    if let Ok(v) = value.extract::<String>() {
        return Some(ParamValue::String(v));
    }
    None
}

fn params_from_py(params: Option<&PyAny>) -> Result<Option<HashMap<String, ParamValue>>, String> {
    let Some(obj) = params else {
        return Ok(None);
    };
    let dict: &pyo3::types::PyDict = obj
        .downcast()
        .map_err(|_| "params must be a dict".to_string())?;
    let mut map = HashMap::new();
    for (k, v) in dict.iter() {
        let key: String = k
            .extract()
            .map_err(|_| "params keys must be strings".to_string())?;
        if let Some(val) = param_value_from_py(v) {
            map.insert(key, val);
        }
    }
    Ok(Some(map))
}

fn vec_from_py_f64(arr: &PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn vec_from_py_i64(arr: &PyReadonlyArray1<i64>) -> Vec<i64> {
    arr.as_array().iter().copied().collect()
}

fn build_ohlcv(
    open: &PyReadonlyArray1<f64>,
    high: &PyReadonlyArray1<f64>,
    low: &PyReadonlyArray1<f64>,
    close: &PyReadonlyArray1<f64>,
    timestamps: Option<&PyReadonlyArray1<i64>>,
    volume: Option<&PyReadonlyArray1<f64>>,
) -> Result<Ohlcv, String> {
    let open_vec = vec_from_py_f64(open);
    let high_vec = vec_from_py_f64(high);
    let low_vec = vec_from_py_f64(low);
    let close_vec = vec_from_py_f64(close);

    let n = close_vec.len();
    if open_vec.len() != n || high_vec.len() != n || low_vec.len() != n {
        return Err("OHLC arrays must have equal length".to_string());
    }

    let timestamp_vec = match timestamps {
        Some(ts) => {
            let ts_vec = vec_from_py_i64(ts);
            if ts_vec.len() != n {
                return Err("timestamps length does not match close length".to_string());
            }
            Some(ts_vec)
        }
        None => Some((0..n as i64).collect()),
    };

    let volume_vec = match volume {
        Some(v) => {
            let vol = vec_from_py_f64(v);
            if vol.len() != n {
                return Err("volume length does not match close length".to_string());
            }
            Some(vol)
        }
        None => None,
    };

    Ok(Ohlcv {
        timestamp: timestamp_vec,
        open: open_vec,
        high: high_vec,
        low: low_vec,
        close: close_vec,
        volume: volume_vec,
    })
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, timestamps=None, volume=None, population=64, generations=20, max_indicators=12, include_raw=true))]
fn search_evolve_ohlcv(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    timestamps: Option<PyReadonlyArray1<i64>>,
    volume: Option<PyReadonlyArray1<f64>>,
    population: usize,
    generations: usize,
    max_indicators: usize,
    include_raw: bool,
) -> PyResult<PyObject> {
    let ohlcv = build_ohlcv(
        &open,
        &high,
        &low,
        &close,
        timestamps.as_ref(),
        volume.as_ref(),
    )
    .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;

    let features = compute_talib_feature_frame(&ohlcv, include_raw).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Feature computation failed: {}",
            e
        ))
    })?;

    let result = evolve_search(&features, &ohlcv, population, generations, max_indicators)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Search failed: {}",
                e
            ))
        })?;

    let metrics: Vec<Vec<f64>> = result.metrics.iter().map(|m| m.to_vec()).collect();
    let genes_py: Vec<PyObject> = result
        .genes
        .iter()
        .map(|g| pythonize(py, g).unwrap_or_else(|_| py.None()))
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("genes", genes_py)?;
    dict.set_item("metrics", metrics)?;
    dict.set_item("feature_names", features.names)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (open, high, low, close, timestamps=None, volume=None, population=100, generations=5, max_indicators=12, candidate_count=200, portfolio_size=100, corr_threshold=0.7, min_trades_per_day=1.0, include_raw=true))]
fn search_discovery_ohlcv(
    py: Python,
    open: PyReadonlyArray1<f64>,
    high: PyReadonlyArray1<f64>,
    low: PyReadonlyArray1<f64>,
    close: PyReadonlyArray1<f64>,
    timestamps: Option<PyReadonlyArray1<i64>>,
    volume: Option<PyReadonlyArray1<f64>>,
    population: usize,
    generations: usize,
    max_indicators: usize,
    candidate_count: usize,
    portfolio_size: usize,
    corr_threshold: f64,
    min_trades_per_day: f64,
    include_raw: bool,
) -> PyResult<PyObject> {
    let ohlcv = build_ohlcv(
        &open,
        &high,
        &low,
        &close,
        timestamps.as_ref(),
        volume.as_ref(),
    )
    .map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;

    let features = compute_talib_feature_frame(&ohlcv, include_raw).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Feature computation failed: {}",
            e
        ))
    })?;

    let config = DiscoveryConfig {
        population,
        generations,
        max_indicators,
        candidate_count,
        portfolio_size,
        corr_threshold,
        min_trades_per_day,
    };

    let result = run_discovery_cycle(&features, &ohlcv, &config).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Discovery failed: {}",
            e
        ))
    })?;

    let portfolio_py: Vec<PyObject> = result
        .portfolio
        .iter()
        .map(|g| pythonize(py, g).unwrap_or_else(|_| py.None()))
        .collect();
    let candidates_py: Vec<PyObject> = result
        .candidates
        .iter()
        .map(|g| pythonize(py, g).unwrap_or_else(|_| py.None()))
        .collect();

    let dict = PyDict::new(py);
    dict.set_item("portfolio", portfolio_py)?;
    dict.set_item("candidates", candidates_py)?;
    dict.set_item("feature_names", features.names)?;
    Ok(dict.into())
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
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&PyAny>) -> PyResult<Self> {
        let params = params_from_py(params).map_err(|msg| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
        })?;
        Self {
            model: Arc::new(Mutex::new(LightGBMExpert::new(idx, params))),
        }
    }

    fn fit<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = py.allow_threads(|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label", labels_vec);

            model
                .fit(&df, &labels_series)
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        });

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = py.allow_threads(|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            model
                .predict_proba(&df)
                .map_err(|e| format!("Prediction failed: {}", e))
        });

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr| arr.into_pyarray_bound(py))
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

#[cfg(feature = "xgboost")]
#[pyclass]
struct XGBoostModel {
    model: Arc<Mutex<XGBoostExpert>>,
}

#[cfg(feature = "xgboost")]
#[pymethods]
impl XGBoostModel {
    #[new]
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&PyAny>) -> PyResult<Self> {
        let params = params_from_py(params).map_err(|msg| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
        })?;
        Self {
            model: Arc::new(Mutex::new(XGBoostExpert::new(idx, params))),
        }
    }

    fn fit<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = py.allow_threads(|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label", labels_vec);

            model
                .fit(&df, &labels_series)
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        });

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = py.allow_threads(|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            model
                .predict_proba(&df)
                .map_err(|e| format!("Prediction failed: {}", e))
        });

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr| arr.into_pyarray_bound(py))
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

#[cfg(feature = "xgboost")]
#[pyclass]
struct XGBoostRFModel {
    model: Arc<Mutex<XGBoostRFExpert>>,
}

#[cfg(feature = "xgboost")]
#[pymethods]
impl XGBoostRFModel {
    #[new]
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&PyAny>) -> PyResult<Self> {
        let params = params_from_py(params).map_err(|msg| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
        })?;
        Self {
            model: Arc::new(Mutex::new(XGBoostRFExpert::new(idx, params))),
        }
    }

    fn fit<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = py.allow_threads(|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label", labels_vec);

            model
                .fit(&df, &labels_series)
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        });

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = py.allow_threads(|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            model
                .predict_proba(&df)
                .map_err(|e| format!("Prediction failed: {}", e))
        });

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr| arr.into_pyarray_bound(py))
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

#[cfg(feature = "xgboost")]
#[pyclass]
struct XGBoostDARTModel {
    model: Arc<Mutex<XGBoostDARTExpert>>,
}

#[cfg(feature = "xgboost")]
#[pymethods]
impl XGBoostDARTModel {
    #[new]
    #[pyo3(signature = (idx=1, params=None))]
    fn new(idx: usize, params: Option<&PyAny>) -> PyResult<Self> {
        let params = params_from_py(params).map_err(|msg| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)
        })?;
        Self {
            model: Arc::new(Mutex::new(XGBoostDARTExpert::new(idx, params))),
        }
    }

    fn fit<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
        labels: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<()> {
        let features_array = features.as_array().to_owned();
        let labels_array = labels.as_array().to_owned();

        let result: Result<(), String> = py.allow_threads(|| {
            let mut model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            let labels_vec: Vec<i64> = labels_array.iter().copied().collect();
            let labels_series = Series::new("label", labels_vec);

            model
                .fit(&df, &labels_series)
                .map_err(|e| format!("Training failed: {}", e))?;

            Ok(())
        });

        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let features_array = features.as_array().to_owned();

        let result: Result<Array2<f32>, String> = py.allow_threads(|| {
            let model = self
                .model
                .lock()
                .map_err(|e| format!("Lock poisoned: {}", e))?;

            let df = dataframe_from_ndarray(&features_array)?;
            model
                .predict_proba(&df)
                .map_err(|e| format!("Prediction failed: {}", e))
        });

        result
            .map_err(|msg| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(msg))
            .map(|arr| arr.into_pyarray_bound(py))
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
    m.add_function(wrap_pyfunction!(search_evolve_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(search_discovery_ohlcv, m)?)?;

    // Add tree model classes if features enabled
    #[cfg(feature = "lightgbm")]
    m.add_class::<LightGBMModel>()?;
    #[cfg(feature = "xgboost")]
    {
        m.add_class::<XGBoostModel>()?;
        m.add_class::<XGBoostRFModel>()?;
        m.add_class::<XGBoostDARTModel>()?;
    }

    Ok(())
}

// Genetic Strategy Expert
// Ported from src/forex_bot/models/genetic.py
//
// Genetic Algorithm for evolving TA-Lib indicator combinations.
// Uses OHLC data (metadata) to generate signals, not just pre-computed features.

use anyhow::{Context, Result};
use ndarray::Array2;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::path::Path;
use tracing::info;

/// GeneticStrategyExpert - Wrapper for Python genetic.py module
///
/// Evolutionary algorithm that discovers optimal TA-Lib indicator combinations.
/// Supports both internal evolution and loading pre-discovered strategies from JSON.
pub struct GeneticStrategyExpert {
    py_expert: Option<Py<PyAny>>,
    population_size: usize,
    generations: usize,
    max_indicators: usize,
}

impl GeneticStrategyExpert {
    /// Create new Genetic Strategy Expert
    /// Python lines 15-30
    pub fn new(
        population_size: usize,
        generations: usize,
        max_indicators: usize,
    ) -> Result<Self> {
        let py_expert = Python::with_gil(|py| {
            // Import the Python module
            let genetic_module = PyModule::import(py, "forex_bot.models.genetic")
                .context("Failed to import forex_bot.models.genetic")?;

            // Get the GeneticStrategyExpert class
            let genetic_class = genetic_module.getattr("GeneticStrategyExpert")
                .context("GeneticStrategyExpert class not found")?;

            // Create instance with kwargs
            let kwargs = PyDict::new(py);
            kwargs.set_item("population_size", population_size)?;
            kwargs.set_item("generations", generations)?;
            kwargs.set_item("max_indicators", max_indicators)?;

            // Instantiate
            let py_expert = genetic_class.call((), Some(&kwargs))?;

            Ok::<Py<PyAny>, anyhow::Error>(py_expert.into())
        })?;

        Ok(Self {
            py_expert: Some(py_expert),
            population_size,
            generations,
            max_indicators,
        })
    }

    /// Fit the genetic model
    /// Python lines 54-203
    ///
    /// This will:
    /// 1. Try to load pre-discovered strategies from Discovery Engine JSON
    /// 2. Fall back to internal evolution if no cached strategies found
    pub fn fit(&mut self, x: &DataFrame, y: &Series, metadata: Option<&DataFrame>) -> Result<()> {
        Python::with_gil(|py| {
            let expert = self.py_expert.as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);

            // Convert DataFrames to pandas
            let pd = PyModule::import(py, "pandas")?;

            // Convert x to pandas DataFrame
            let x_csv = {
                let mut buf = Vec::new();
                let mut x_clone = x.clone();
                CsvWriter::new(&mut buf).finish(&mut x_clone)?;
                String::from_utf8(buf)?
            };
            let x_str_io = PyModule::import(py, "io")?.getattr("StringIO")?.call1((x_csv,))?;
            let x_pd = pd.call_method1("read_csv", (x_str_io,))?;

            // Convert y to pandas Series
            let y_values: Vec<f64> = y.f64()?.into_iter().map(|v| v.unwrap_or(0.0)).collect();
            let y_pd = pd.call_method1("Series", (y_values,))?;

            // Convert metadata if provided
            let metadata_pd = if let Some(meta) = metadata {
                let meta_csv = {
                    let mut buf = Vec::new();
                    let mut meta_clone = meta.clone();
                    CsvWriter::new(&mut buf).finish(&mut meta_clone)?;
                    String::from_utf8(buf)?
                };
                let meta_str_io = PyModule::import(py, "io")?.getattr("StringIO")?.call1((meta_csv,))?;
                Some(pd.call_method1("read_csv", (meta_str_io,))?)
            } else {
                None
            };

            // Call fit method
            let kwargs = PyDict::new(py);
            kwargs.set_item("metadata", metadata_pd)?;
            expert.call_method("fit", (x_pd, y_pd), Some(&kwargs))?;

            info!(
                "Genetic expert fitted ({} generations, population {})",
                self.generations, self.population_size
            );

            Ok(())
        })
    }

    /// Predict probabilities
    /// Python lines 204-236
    ///
    /// Returns 3-class probabilities: [Neutral, Buy, Sell]
    /// Uses portfolio voting across multiple evolved strategies
    pub fn predict_proba(&self, x: &DataFrame, metadata: Option<&DataFrame>) -> Result<Array2<f32>> {
        Python::with_gil(|py| {
            let expert = self.py_expert.as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);

            let pd = PyModule::import(py, "pandas")?;

            // Convert x to pandas
            let x_csv = {
                let mut buf = Vec::new();
                let mut x_clone = x.clone();
                CsvWriter::new(&mut buf).finish(&mut x_clone)?;
                String::from_utf8(buf)?
            };
            let x_str_io = PyModule::import(py, "io")?.getattr("StringIO")?.call1((x_csv,))?;
            let x_pd = pd.call_method1("read_csv", (x_str_io,))?;

            // Convert metadata if provided
            let metadata_pd = if let Some(meta) = metadata {
                let meta_csv = {
                    let mut buf = Vec::new();
                    let mut meta_clone = meta.clone();
                    CsvWriter::new(&mut buf).finish(&mut meta_clone)?;
                    String::from_utf8(buf)?
                };
                let meta_str_io = PyModule::import(py, "io")?.getattr("StringIO")?.call1((meta_csv,))?;
                Some(pd.call_method1("read_csv", (meta_str_io,))?)
            } else {
                None
            };

            // Call predict_proba
            let kwargs = PyDict::new(py);
            kwargs.set_item("metadata", metadata_pd)?;
            let result = expert.call_method("predict_proba", (x_pd,), Some(&kwargs))?;

            // Convert numpy array to ndarray
            let _numpy = PyModule::import(py, "numpy")?;
            let result_list: Vec<Vec<f32>> = result.call_method0("tolist")?.extract()?;

            let nrows = result_list.len();
            let ncols = if nrows > 0 { result_list[0].len() } else { 3 };

            let mut flat = Vec::new();
            for row in result_list {
                flat.extend(row);
            }

            let array = Array2::from_shape_vec((nrows, ncols), flat)?;

            Ok(array)
        })
    }

    /// Save the model
    /// Python lines 238-244
    pub fn save(&self, path: &Path) -> Result<()> {
        Python::with_gil(|py| {
            let expert = self.py_expert.as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);

            expert.call_method1("save", (path.to_string_lossy().as_ref(),))?;

            info!("Saved genetic expert to: {:?}", path);

            Ok(())
        })
    }

    /// Load the model
    /// Python lines 246-256
    pub fn load(&mut self, path: &Path) -> Result<()> {
        Python::with_gil(|py| {
            let expert = self.py_expert.as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);

            expert.call_method1("load", (path.to_string_lossy().as_ref(),))?;

            info!("Loaded genetic expert from: {:?}", path);

            Ok(())
        })
    }

    /// Get the number of strategies in the portfolio
    pub fn portfolio_size(&self) -> Result<usize> {
        Python::with_gil(|py| {
            let expert = self.py_expert.as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);

            let portfolio = expert.getattr("portfolio")?;
            let size: usize = portfolio.len()?;

            Ok(size)
        })
    }

    /// Get the best gene's fitness score
    pub fn best_fitness(&self) -> Result<f64> {
        Python::with_gil(|py| {
            let expert = self.py_expert.as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);

            let best_gene = expert.getattr("best_gene")?;

            if best_gene.is_none() {
                return Ok(0.0);
            }

            let fitness: f64 = best_gene.getattr("fitness")?.extract()?;

            Ok(fitness)
        })
    }
}

// ============================================================================
// SUMMARY
// ============================================================================
//
// Genetic Strategy Expert - PyO3 Wrapper
//
// FEATURES:
// ✅ Wraps Python GeneticStrategyExpert via PyO3
// ✅ Loads pre-discovered strategies from Discovery Engine JSON
// ✅ Falls back to internal evolution (50 population, 10 generations)
// ✅ Portfolio voting across multiple evolved strategies
// ✅ Smooth 3-class probability output: [Neutral, Buy, Sell]
// ✅ Save/Load support via joblib
//
// BRIDGE TO DISCOVERY ENGINE:
// - Loads from cache/talib_knowledge.json
// - Supports symbol-specific caches (cache/talib_knowledge_EURUSD.json)
// - Supports opportunistic strategies (cache/talib_knowledge_opportunistic.json)
// - Validates all loaded genes before use
//
// INTERNAL EVOLUTION:
// - Generates random TA-Lib indicator combinations
// - Evaluates fitness via Sharpe ratio on historical data
// - Evolves over N generations via genetic algorithm
// - Clamps thresholds to avoid degenerate strategies
//
// This module bridges the gap between:
// 1. Python's rich TA-Lib ecosystem (50+ indicators)
// 2. Rust's performance and type safety
// 3. Pre-discovered optimal strategies from external analysis
//

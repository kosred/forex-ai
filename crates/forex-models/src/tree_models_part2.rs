// ============================================================================
// XGBOOST VARIANT MODELS
// ============================================================================

/// XGBoost Random Forest Expert (num_parallel_tree=8)
pub struct XGBoostRFExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    gpu_only_disabled: bool,
    #[cfg(feature = "xgboost")]
    model: Option<xgb::Booster>,
    #[cfg(not(feature = "xgboost"))]
    model: Option<()>,
}

impl XGBoostRFExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let default_params = Self::default_params();
        let params = params.unwrap_or(default_params);

        let config = TreeModelConfig {
            idx,
            params,
            device_pref: tree_device_preference(),
            gpu_only: gpu_only_mode(),
            cpu_threads: if cpu_threads_hint() > 0 {
                Some(cpu_threads_hint())
            } else {
                None
            },
        };

        Self {
            idx,
            config,
            gpu_only_disabled: false,
            model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();
        // RF variant: different from base XGBoost
        params.insert("n_estimators".to_string(), ParamValue::Int(400)); // not 800
        params.insert("max_depth".to_string(), ParamValue::Int(6)); // not 8
        params.insert("learning_rate".to_string(), ParamValue::Float(0.3)); // not 0.05
        params.insert("subsample".to_string(), ParamValue::Float(0.8)); // not 0.9
        params.insert("colsample_bynode".to_string(), ParamValue::Float(0.8)); // NEW
        params.insert("colsample_bytree".to_string(), ParamValue::Float(0.8)); // not 0.9
        params.insert("num_parallel_tree".to_string(), ParamValue::Int(8)); // NEW - RF mode
        params.insert("objective".to_string(), ParamValue::String("multi:softprob".to_string()));
        params.insert("num_class".to_string(), ParamValue::Int(3));
        params.insert("random_state".to_string(), ParamValue::Int(42));
        params.insert("verbosity".to_string(), ParamValue::Int(0));
        params
    }
}

/// XGBoost DART Expert (booster=dart)
pub struct XGBoostDARTExpert {
    pub idx: usize,
    pub config: TreeModelConfig,
    gpu_only_disabled: bool,
    #[cfg(feature = "xgboost")]
    model: Option<xgb::Booster>,
    #[cfg(not(feature = "xgboost"))]
    model: Option<()>,
}

impl XGBoostDARTExpert {
    pub fn new(idx: usize, params: Option<HashMap<String, ParamValue>>) -> Self {
        let default_params = Self::default_params();
        let params = params.unwrap_or(default_params);

        let config = TreeModelConfig {
            idx,
            params,
            device_pref: tree_device_preference(),
            gpu_only: gpu_only_mode(),
            cpu_threads: if cpu_threads_hint() > 0 {
                Some(cpu_threads_hint())
            } else {
                None
            },
        };

        Self {
            idx,
            config,
            gpu_only_disabled: false,
            model: None,
        }
    }

    fn default_params() -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();
        // DART variant
        params.insert("n_estimators".to_string(), ParamValue::Int(600)); // not 800
        params.insert("booster".to_string(), ParamValue::String("dart".to_string())); // NEW
        params.insert("rate_drop".to_string(), ParamValue::Float(0.10)); // NEW
        params.insert("skip_drop".to_string(), ParamValue::Float(0.50)); // NEW
        params.insert("sample_type".to_string(), ParamValue::String("uniform".to_string())); // NEW
        params.insert("normalize_type".to_string(), ParamValue::String("tree".to_string())); // NEW
        params.insert("max_depth".to_string(), ParamValue::Int(8));
        params.insert("learning_rate".to_string(), ParamValue::Float(0.05));
        params.insert("objective".to_string(), ParamValue::String("multi:softprob".to_string()));
        params.insert("num_class".to_string(), ParamValue::Int(3));
        params.insert("random_state".to_string(), ParamValue::Int(42));
        params.insert("verbosity".to_string(), ParamValue::Int(0));
        params
    }
}

// ============================================================================
// XGBOOST VARIANT IMPLEMENTATIONS
// ============================================================================

impl ExpertModel for XGBoostRFExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let mut base = XGBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: None,
            };
            base.fit(x, y)?;
            self.model = base.model;
            self.gpu_only_disabled = base.gpu_only_disabled;
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        #[cfg(feature = "xgboost")]
        {
            let base = XGBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: self.model.clone(),
            };
            base.predict_proba(x)
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            if let Some(model) = &self.model {
                model.save(path.to_str().context("Invalid path")?)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            Ok(())
        }
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let booster = xgb::Booster::load(path.to_str().context("Invalid path")?)?;
            self.model = Some(booster);
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }
}

impl ExpertModel for XGBoostDARTExpert {
    fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let mut base = XGBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: None,
            };
            base.fit(x, y)?;
            self.model = base.model;
            self.gpu_only_disabled = base.gpu_only_disabled;
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        #[cfg(feature = "xgboost")]
        {
            let base = XGBoostExpert {
                idx: self.idx,
                config: self.config.clone(),
                gpu_only_disabled: self.gpu_only_disabled,
                model: self.model.clone(),
            };
            base.predict_proba(x)
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            if let Some(model) = &self.model {
                model.save(path.to_str().context("Invalid path")?)?;
            }
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            Ok(())
        }
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        #[cfg(feature = "xgboost")]
        {
            let booster = xgb::Booster::load(path.to_str().context("Invalid path")?)?;
            self.model = Some(booster);
            Ok(())
        }
        #[cfg(not(feature = "xgboost"))]
        {
            anyhow::bail!("XGBoost feature not enabled");
        }
    }
}

// ============================================================================
// CATBOOST IMPLEMENTATION (INFERENCE ONLY - HYBRID APPROACH)
// ============================================================================

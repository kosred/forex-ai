use chrono::{DateTime, Utc};
use ndarray::Array1; // Minimal ndarray usage for now
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalResult {
    pub signal: i32,
    pub confidence: f64,
    pub model_votes: HashMap<String, f64>,
    pub regime: String,
    pub meta_features: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    // Store probabilities as simple Vec for serialization simplicity, or use ndarray with serde feature if enabled
    pub probs: Option<Vec<f64>>,
    // Pandas Series are typically just arrays/Vecs in this context.
    pub trade_probability: Option<f64>, // Porting Series as scalar for single result or Vec if history?
    // Python code implies these are Series, possibly single row or full history?
    // "SignalResult" implies a single point in time, likely scalars.
    // If it's a history, it should be Vec<f64>.
    // Looking at usage: "signals: pd.Series | None = None # Full series for backtest"
    // So some are scalars, some are series.
    pub stacking_confidence: Option<f64>,
    pub recommended_rr: Option<f64>,
    pub recommended_sl: Option<f64>,
    pub win_probability: Option<f64>,
    // For backtesting, we might store fuller data, but let's keep it simple for core struct.
    // In Rust, for high performance, we might separate "LiveSignal" from "BacktestResult".
}

impl Default for SignalResult {
    fn default() -> Self {
        Self {
            signal: 0,
            confidence: 0.0,
            model_votes: HashMap::new(),
            regime: "neutral".to_string(),
            meta_features: HashMap::new(),
            timestamp: Utc::now(),
            probs: None,
            trade_probability: None,
            stacking_confidence: None,
            recommended_rr: None,
            recommended_sl: None,
            win_probability: None,
        }
    }
}

// PreparedDataset corresponds to python's dataclass with X/y
// In Rust, we might just pass references to ndarrays or Polars DataFrames.
// For a DTO/event, we might store simplified paths or small batches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedDatasetLite {
    pub feature_names: Vec<String>,
    // Actual data might be too heavy for a simple event struct unless we use shared pointers or paths.
    // Keeping it minimal for now.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEvent {
    pub symbol: String,
    pub side: String, // "buy" / "sell"
    pub volume: f64,
    pub open_price: f64,
    pub open_time: DateTime<Utc>,
    pub close_price: Option<f64>,
    pub close_time: Option<DateTime<Utc>>,
    pub pnl: Option<f64>,
    pub commission: f64,
    pub swap: f64,
    pub comment: String,
    pub magic: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub category: String,
    pub message: String,
    pub severity: String,
    pub context: serde_json::Value,
}

impl RiskEvent {
    pub fn new(
        category: impl Into<String>,
        message: impl Into<String>,
        severity: impl Into<String>,
        context: Option<serde_json::Value>,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4().simple().to_string()[..16].to_string(),
            timestamp: Utc::now(),
            category: category.into(),
            message: message.into(),
            severity: severity.into(),
            context: context.unwrap_or(serde_json::json!({})),
        }
    }
}

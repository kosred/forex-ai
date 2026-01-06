use crate::config::Settings;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};

// use rusqlite::{params, Connection}; // will use fully qualified or add imports as needed

/// Represents a stored strategy gene (DTO for TALibStrategyGene)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedStrategy {
    pub strategy_id: String,
    pub indicators: JsonValue,
    pub params: JsonValue,
    pub weights: JsonValue,
    pub thresholds: JsonValue,
    pub smc_flags: JsonValue,
    pub metrics: JsonValue,
    pub last_used: Option<DateTime<Utc>>,
}

/// Main storage handler for strategies, trades, and intents
pub struct StrategyLedger {
    db_path: PathBuf,
}

impl StrategyLedger {
    pub fn new(db_path: impl AsRef<Path>, symbol: Option<&str>) -> Result<Self> {
        let mut path = db_path.as_ref().to_path_buf();

        // HPC FIX: Symbol-Isolated Databases
        if let Some(sym) = symbol {
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy())
                .unwrap_or_default();
            path.set_file_name(format!("{}_{}.sqlite", stem, sym));
        }

        // Ensure absolute path
        if !path.is_absolute() {
            // In Rust, we usually assume CWD or relative to binary.
            // But if we want to mimic Python's repo_root anchoring, we might need logic.
            // For now, let's assume it references CWD or absolute path provided via config.
            // If relative, it remains relative to CWD.
        }

        let ledger = Self { db_path: path };
        ledger.init_db()?;
        Ok(ledger)
    }

    fn init_db(&self) -> Result<()> {
        if let Some(parent) = self.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = match rusqlite::Connection::open(&self.db_path) {
            Ok(c) => c,
            Err(e) => {
                // Handle disk I/O errors by attempting to clean up WAL files
                let msg = e.to_string().to_lowercase();
                if msg.contains("disk i/o") || msg.contains("disk io") {
                    warn!("Disk I/O error detected. cleaning up WAL/SHM files and retrying...");
                    // Try to remove -wal and -shm files
                    let _ = std::fs::remove_file(self.db_path.with_extension("sqlite-wal"));
                    let _ = std::fs::remove_file(self.db_path.with_extension("sqlite-shm"));
                    // Retry open
                    rusqlite::Connection::open(&self.db_path)?
                } else {
                    return Err(e.into());
                }
            }
        };

        // Enable Write-Ahead Logging and tuning
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA wal_autocheckpoint=1000;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;",
        )?;

        // Create tables
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alpha_strategies (
                strategy_id TEXT PRIMARY KEY,
                indicators TEXT,
                params TEXT,
                weights TEXT,
                thresholds TEXT,
                smc_flags TEXT,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                closed_at TIMESTAMP,
                regime TEXT,
                pnl REAL,
                risk_pct REAL,
                confidence REAL,
                context TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS pending_intents (
                ticket INTEGER PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                volume REAL,
                sl REAL,
                tp REAL,
                meta_risk_mult REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS live_metrics (
                symbol TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                returns_json TEXT,
                sharpe REAL,
                volatility REAL,
                drawdown REAL,
                position_count INTEGER,
                equity REAL
            )",
            [],
        )?;

        Ok(())
    }

    pub fn log_live_metrics(
        &self,
        symbol: &str,
        returns: &[f64],
        sharpe: f64,
        volatility: f64,
        drawdown: f64,
        position_count: i64,
        equity: f64,
    ) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let returns_json = serde_json::to_string(returns)?;

        conn.execute(
            "INSERT OR REPLACE INTO live_metrics
            (symbol, timestamp, returns_json, sharpe, volatility, drawdown, position_count, equity)
            VALUES (?1, CURRENT_TIMESTAMP, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                symbol,
                returns_json,
                sharpe,
                volatility,
                drawdown,
                position_count,
                equity
            ],
        )
        .context("Failed to log live metrics")?;

        Ok(())
    }

    pub fn get_all_live_metrics(&self) -> Result<impl Iterator<Item = JsonValue>> {
        // Implementation note: In Rust we might return a dedicated Struct or HashMap.
        // Python returns dict[str, dict]. Let's keep it simple for now or return a Vec.
        // For efficiency, we just return a stub or Result<Vec<...>>.
        // Given complexity of returning iterator with open connection ref, let's fetch all into Vec.
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let mut stmt = conn.prepare("SELECT * FROM live_metrics")?;

        // Mapping rows to JsonValue for flexible return, similar to Python dict
        let rows = stmt.query_map([], |row| {
            let symbol: String = row.get("symbol")?;
            let returns_json: String = row.get("returns_json")?;
            let returns: Vec<f64> = serde_json::from_str(&returns_json).unwrap_or_default();

            let metrics = serde_json::json!({
                "symbol": symbol,
                "returns": returns,
                "sharpe": row.get::<_, f64>("sharpe")?,
                "volatility": row.get::<_, f64>("volatility")?,
                "drawdown": row.get::<_, f64>("drawdown")?,
                "position_count": row.get::<_, i64>("position_count")?,
                "equity": row.get::<_, f64>("equity")?,
                "timestamp": row.get::<_, String>("timestamp")?,
            });
            Ok(metrics)
        })?;

        let mut results = Vec::new();
        for r in rows {
            results.push(r?);
        }
        Ok(results.into_iter())
    }

    pub fn save_alpha(&self, gene: &SavedStrategy) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;

        // In Rust, we assume `gene` is already configured.
        // Logic to construct JSONs happens before call or `SavedStrategy` holds them as JsonValue.

        conn.execute(
            "INSERT OR REPLACE INTO alpha_strategies
            (strategy_id, indicators, params, weights, thresholds, smc_flags, metrics, last_used)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, CURRENT_TIMESTAMP)",
            rusqlite::params![
                gene.strategy_id,
                gene.indicators.to_string(),
                gene.params.to_string(),
                gene.weights.to_string(),
                gene.thresholds.to_string(),
                gene.smc_flags.to_string(),
                gene.metrics.to_string()
            ],
        )?;

        info!("Saved Alpha Strategy: {}", gene.strategy_id);
        Ok(())
    }

    // Additional methods like log_intent, remove_intent, log_trade, etc. would follow similar patterns.
    // Implementing a few key ones for completeness.

    pub fn save_setting(&self, key: &str, value: &impl Serialize) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let val_str = serde_json::to_string(value)?;
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)",
            rusqlite::params![key, val_str],
        )?;
        Ok(())
    }

    pub fn load_setting<T: for<'a> Deserialize<'a>>(&self, key: &str) -> Result<Option<T>> {
        let conn = rusqlite::Connection::open(&self.db_path)?;
        let mut stmt = conn.prepare("SELECT value FROM settings WHERE key = ?1")?;
        let mut rows = stmt.query(rusqlite::params![key])?;

        if let Some(row) = rows.next()? {
            let s: String = row.get(0)?;
            let val = serde_json::from_str(&s)?;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }
}

/// Metrics Recorder for cycle metrics
pub struct MetricsRecorder {
    db_path: PathBuf,
}

impl MetricsRecorder {
    pub fn new(settings: &Settings) -> Result<Self> {
        let path = PathBuf::from(&settings.system.metrics_db_path); // Assuming this field exists and is string/path
        let recorder = Self { db_path: path };
        recorder.init_db()?;
        Ok(recorder)
    }

    fn init_db(&self) -> Result<()> {
        if let Some(parent) = self.db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = rusqlite::Connection::open(&self.db_path)?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS cycle_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT,
                equity REAL,
                daily_pnl REAL,
                session_trades INTEGER,
                signal INTEGER,
                confidence REAL,
                position_size REAL,
                data_json TEXT
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cycle_timestamp ON cycle_metrics(timestamp)",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cycle_symbol ON cycle_metrics(symbol)",
            [],
        )?;

        debug!("Metrics database initialized at {:?}", self.db_path);
        Ok(())
    }

    pub fn log_cycle(&self, data: &serde_json::Map<String, JsonValue>) -> Result<()> {
        let conn = rusqlite::Connection::open(&self.db_path)?;

        let timestamp = data
            .get("timestamp")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(); // Should use chrono::Utc::now() if missing
        let symbol = data
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN");
        let equity = data.get("equity").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let daily_pnl = data
            .get("daily_pnl")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let session_trades = data
            .get("session_trades")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let signal = data.get("signal").and_then(|v| v.as_i64()).unwrap_or(0);
        let confidence = data
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let position_size = data
            .get("position_size")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let data_json = serde_json::to_string(data)?;

        // If timestamp is empty, generate it
        let timestamp = if timestamp.is_empty() {
            Utc::now().to_rfc3339()
        } else {
            timestamp
        };

        conn.execute(
            "INSERT INTO cycle_metrics (
                timestamp, symbol, equity, daily_pnl, session_trades, signal, confidence, position_size, data_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                timestamp, symbol, equity, daily_pnl, session_trades, signal, confidence, position_size, data_json
            ],
        )?;

        Ok(())
    }
}

/// Risk Event Ledger
pub struct RiskLedger {
    events: Arc<Mutex<VecDeque<RiskEvent>>>,
    max_events: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEvent {
    pub timestamp: String, // ISO8601
    pub type_: String,
    pub message: String,
    pub severity: String,
    pub context: JsonValue,
}

impl RiskLedger {
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(Mutex::new(VecDeque::with_capacity(max_events))),
            max_events,
        }
    }

    pub fn record(
        &self,
        event_type: &str,
        message: &str,
        severity: &str,
        context: Option<JsonValue>,
    ) {
        let event = RiskEvent {
            timestamp: Utc::now().to_rfc3339(),
            type_: event_type.to_string(),
            message: message.to_string(),
            severity: severity.to_string(),
            context: context.unwrap_or(serde_json::json!({})),
        };

        if let Ok(mut lock) = self.events.lock() {
            if lock.len() >= self.max_events {
                lock.pop_front();
            }
            lock.push_back(event);
        }

        // Log to system logger
        match severity {
            "warning" | "WARN" => warn!("Risk Event [{}]: {}", event_type, message),
            "error" | "ERROR" => error!("Risk Event [{}]: {}", event_type, message),
            _ => info!("Risk Event [{}]: {}", event_type, message),
        }
    }
}

from __future__ import annotations

import json
import logging
import sqlite3
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..features.talib_mixer import TALibStrategyGene
    from .config import Settings

logger = logging.getLogger(__name__)


class StrategyLedger:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        if not path.is_absolute():
            # Anchor relative paths to repo root (avoid CWD-dependent I/O errors).
            repo_root = Path(__file__).resolve().parents[3]
            path = repo_root / path
        self.db_path = str(path)
        self._init_db()

    def _init_db(self) -> None:
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            if "disk i/o" in msg or "disk io" in msg:
                # Attempt a clean rebuild of the ledger database.
                for suffix in ("", "-wal", "-shm"):
                    try:
                        (db_path.with_suffix(db_path.suffix + suffix)).unlink(missing_ok=True)
                    except Exception:
                        pass
                conn = sqlite3.connect(self.db_path)
            else:
                raise
        with conn:
            # Enable Write-Ahead Logging for concurrency
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alpha_strategies (
                    strategy_id TEXT PRIMARY KEY,
                    indicators TEXT,
                    params TEXT,
                    weights TEXT,
                    thresholds TEXT,
                    smc_flags TEXT,
                    metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    closed_at TIMESTAMP,
                    regime TEXT,
                    pnl REAL,
                    risk_pct REAL,
                    confidence REAL,
                    context TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_intents (
                    ticket INTEGER PRIMARY KEY,
                    symbol TEXT,
                    direction TEXT,
                    volume REAL,
                    sl REAL,
                    tp REAL,
                    meta_risk_mult REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS live_metrics (
                    symbol TEXT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    returns_json TEXT,
                    sharpe REAL,
                    volatility REAL,
                    drawdown REAL,
                    position_count INTEGER,
                    equity REAL
                )
            """)

    def log_live_metrics(
        self,
        symbol: str,
        returns: list[float],
        sharpe: float,
        volatility: float,
        drawdown: float,
        position_count: int,
        equity: float,
    ) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                returns_json = json.dumps(returns)
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO live_metrics
                    (symbol, timestamp, returns_json, sharpe, volatility, drawdown, position_count, equity)
                    VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
                """,
                    (symbol, returns_json, sharpe, volatility, drawdown, position_count, equity),
                )
        except Exception as e:
            logger.error(f"Failed to log live metrics for {symbol}: {e}")

    def get_all_live_metrics(self) -> dict[str, dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM live_metrics")
                rows = cursor.fetchall()

                all_metrics = {}
                for row in rows:
                    metrics = dict(row)
                    metrics["returns"] = json.loads(metrics["returns_json"])
                    del metrics["returns_json"]
                    all_metrics[metrics["symbol"]] = metrics
                return all_metrics
        except Exception:
            return {}

    def log_intent(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        volume: float,
        sl: float,
        tp: float,
        meta_risk_mult: float,
    ) -> None:
        """Log an order intent immediately after placement for recovery."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO pending_intents (ticket, symbol, direction, volume, sl, tp, meta_risk_mult)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (ticket, symbol, direction, volume, sl, tp, meta_risk_mult),
                )
        except Exception as e:
            logger.error(f"Failed to log intent: {e}")

    def get_pending_intents(self) -> list[dict[str, Any]]:
        """Retrieve pending intents to reconcile with MT5 state."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM pending_intents")
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception:
            return []

    def remove_intent(self, ticket: int) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM pending_intents WHERE ticket = ?", (ticket,))
        except Exception as e:
            logger.error(f"Data persistence failed: {e}", exc_info=True)

    def save_alpha(self, gene: TALibStrategyGene) -> None:
        """Save a high-performing strategy gene to the ledger."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                indicators_json = json.dumps(gene.indicators)
                params_json = json.dumps(gene.params)
                weights_json = json.dumps(gene.weights)
                thresholds_json = json.dumps(
                    {"long": gene.long_threshold, "short": gene.short_threshold, "method": gene.combination_method}
                )
                smc_flags_json = json.dumps(
                    {
                        "use_ob": gene.use_ob,
                        "use_fvg": gene.use_fvg,
                        "use_liq_sweep": gene.use_liq_sweep,
                        "mtf_confirmation": gene.mtf_confirmation,
                        "use_premium_discount": getattr(gene, "use_premium_discount", False),
                        "use_inducement": getattr(gene, "use_inducement", False),
                    }
                )
                metrics_json = json.dumps(
                    {"fitness": gene.fitness, "sharpe": gene.sharpe_ratio, "win_rate": gene.win_rate}
                )

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO alpha_strategies
                    (strategy_id, indicators, params, weights, thresholds, smc_flags, metrics, last_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                    (
                        gene.strategy_id,
                        indicators_json,
                        params_json,
                        weights_json,
                        thresholds_json,
                        smc_flags_json,
                        metrics_json,
                    ),
                )

                logger.info(f"Saved Alpha Strategy: {gene.strategy_id}")
        except Exception as e:
            logger.error(f"Failed to save alpha strategy: {e}")

    def load_top_alphas(self, limit: int = 10) -> list[TALibStrategyGene]:
        """Load top strategies sorted by fitness."""
        from ..features.talib_mixer import TALibStrategyGene  # Runtime import

        genes = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM alpha_strategies")
                rows = cursor.fetchall()

                for row in rows:
                    try:
                        metrics = json.loads(row["metrics"])
                        fitness = metrics.get("fitness", 0.0)

                        thresholds = json.loads(row["thresholds"])
                        smc = json.loads(row["smc_flags"])

                        gene = TALibStrategyGene(
                            indicators=json.loads(row["indicators"]),
                            params=json.loads(row["params"]),
                            combination_method=thresholds.get("method", "weighted_vote"),
                            long_threshold=thresholds.get("long", 0.6),
                            short_threshold=thresholds.get("short", -0.6),
                            weights=json.loads(row["weights"]),
                            strategy_id=row["strategy_id"],
                            fitness=fitness,
                            sharpe_ratio=metrics.get("sharpe", 0.0),
                            win_rate=metrics.get("win_rate", 0.0),
                            use_ob=smc.get("use_ob", False),
                            use_fvg=smc.get("use_fvg", False),
                            use_liq_sweep=smc.get("use_liq_sweep", False),
                            mtf_confirmation=smc.get("mtf_confirmation", False),
                            use_premium_discount=smc.get("use_premium_discount", False),
                            use_inducement=smc.get("use_inducement", False),
                        )
                        genes.append(gene)
                    except Exception:
                        continue

                genes.sort(key=lambda x: x.fitness, reverse=True)
                return genes[:limit]

        except Exception as e:
            logger.error(f"Failed to load alphas: {e}")
            return []

    def log_trade(
        self,
        closed_at: datetime,
        regime: str,
        feature_vector: list[object] | None,
        model_votes: dict[str, object],
        policy: str,
        pnl: float,
        pnl_pct: float,
        risk_pct: float,
        slippage: float,
        news_context: dict[str, object],
        confidence: float,
        meta_risk_multiplier: float | None = None,
    ) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                def default_serializer(obj: object) -> object:
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if hasattr(obj, "to_dict"):  # Pandas Series/DF
                        return obj.to_dict()
                    if hasattr(obj, "tolist"):  # Torch tensor
                        return obj.tolist()
                    return str(obj)

                context_json = json.dumps(
                    {
                        "features": feature_vector[:10] if feature_vector else [],  # Store only top 10 for space
                        "votes": model_votes,
                        "news": news_context,
                        "slippage": slippage,
                        "meta_risk_multiplier": meta_risk_multiplier,
                    },
                    default=default_serializer,
                )

                cursor.execute(
                    """
                    INSERT INTO trade_log (closed_at, regime, pnl, risk_pct, confidence, context)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (closed_at, regime, pnl, risk_pct, confidence, context_json),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log trade: {e}", exc_info=True)
            raise

    def save_setting(self, key: str, value: object) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                val_str = json.dumps(value)
                cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, val_str))
        except Exception as e:
            logger.error(f"Data persistence failed: {e}", exc_info=True)

    def load_setting(self, key: str, default: object = None) -> object:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return default
        except Exception:
            return default


class MetricsRecorder:
    def __init__(self, settings: Settings) -> None:
        self.db_path = settings.system.metrics_db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize metrics database with proper schema."""
        try:
            import sqlite3

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cycle_metrics (
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
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cycle_timestamp ON cycle_metrics(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cycle_symbol ON cycle_metrics(symbol)
            """)
            conn.commit()
            conn.close()
            logger.debug(f"Metrics database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}", exc_info=True)

    def log_cycle(self, data: dict[str, Any]) -> None:
        """Log trading cycle metrics to database."""
        try:
            import json
            import sqlite3
            from datetime import datetime

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = data.get("timestamp", datetime.now(UTC).isoformat())
            symbol = data.get("symbol", "UNKNOWN")
            equity = data.get("equity", 0.0)
            daily_pnl = data.get("daily_pnl", 0.0)
            session_trades = data.get("session_trades", 0)
            signal = data.get("signal", 0)
            confidence = data.get("confidence", 0.0)
            position_size = data.get("position_size", 0.0)
            data_json = json.dumps(data, default=str)

            cursor.execute(
                """
                INSERT INTO cycle_metrics (
                    timestamp,
                    symbol,
                    equity,
                    daily_pnl,
                    session_trades,
                    signal,
                    confidence,
                    position_size,
                    data_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    symbol,
                    equity,
                    daily_pnl,
                    session_trades,
                    signal,
                    confidence,
                    position_size,
                    data_json,
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log cycle metrics: {e}", exc_info=True)


class RiskLedger:
    def __init__(self, max_events: int = 1000) -> None:
        self.events = deque(maxlen=max_events)

    def record(self, event_type: str, message: str, severity: str = "info", context: dict | None = None) -> None:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "type": event_type,
            "message": message,
            "severity": severity,
            "context": context or {},
        }
        self.events.append(event)

        if severity in ("warning", "error", "critical"):
            log_method = getattr(logger, severity, logger.info)
            log_method(f"Risk Event [{event_type}]: {message}")

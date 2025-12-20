import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd


@dataclass(slots=True)
class SignalResult:
    signal: int  # -1, 0, 1
    confidence: float
    model_votes: dict[str, float]
    regime: str
    meta_features: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    probs: list[float] | None = None  # [neutral, buy, sell] for the latest bar

    trade_probability: pd.Series | None = None
    stacking_confidence: pd.Series | None = None
    regimes: pd.Series | None = None
    uncertainty: pd.Series | None = None
    recommended_rr: pd.Series | None = None
    recommended_sl: pd.Series | None = None
    win_probability: pd.Series | None = None
    signals: pd.Series | None = None  # Full series for backtest


@dataclass(slots=True)
class PreparedDataset:
    X: Any  # DataFrame or numpy array
    y: Any  # Series or numpy array
    index: Any  # DatetimeIndex
    feature_names: list[str]
    metadata: pd.DataFrame | None = None  # OHLCV for backtesting
    labels: pd.Series | None = None  # For training convenience


@dataclass(slots=True)
class TradeEvent:
    symbol: str
    side: str  # 'buy' or 'sell'
    volume: float
    open_price: float
    open_time: datetime
    close_price: float | None = None
    close_time: datetime | None = None
    pnl: float | None = None
    commission: float = 0.0
    swap: float = 0.0
    comment: str = ""
    magic: int = 0


@dataclass(slots=True)
class RiskEvent:
    """Risk event for ledger tracking."""

    category: str
    message: str
    severity: str = "info"
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_id: str = field(init=False)

    def __post_init__(self):
        # UUIDv4 avoids collisions and is fast; truncate for compactness.
        self.event_id = uuid.uuid4().hex[:12]

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "message": self.message,
            "severity": self.severity,
            "context": self.context,
        }

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class NewsEvent:
    title: str
    currency: str
    impact: str  # "High", "Medium", "Low"
    published_at: datetime
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    source: str = "generic"
    sentiment: float = 0.0
    confidence: float = 0.0
    currencies: list[str] = None
    summary: str = ""

    def __post_init__(self):
        if self.currencies is None:
            self.currencies = [self.currency] if self.currency else []


class BaseNewsSource(ABC):
    @abstractmethod
    def fetch_events(self, start_date: datetime, end_date: datetime) -> list[NewsEvent]:
        pass

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ..core.config import Settings
from ..data.news.scorers import _load_key_from_file, _safe_json_object_loads
from ..data.news.searchers import PerplexitySearcher

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, OpenAI
except Exception:  # pragma: no cover
    OpenAI = None
    AsyncOpenAI = None


@dataclass(slots=True)
class StrategistAdvice:
    stance: str
    risk_adjustments: dict[str, Any]
    trade_filters: dict[str, Any]
    commentary: str
    raw: dict[str, Any]


class LLMStrategist:
    """
    Lightweight strategic advisor using GPT-4o-mini for policy hints and Perplexity for breaking news lookups.
    Designed to be called infrequently (e.g., every N minutes) to avoid latency/cost.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_run_ts = 0.0
        self.interval_min = max(5, int(getattr(settings.news, "strategist_interval_minutes", 30)))

        self.async_client = None
        self.available = False
        if AsyncOpenAI is not None:
            key_env = getattr(settings.news, "openai_api_key_env", "OPENAI_API_KEY")
            api_key = os.environ.get(key_env) or _load_key_from_file(
                getattr(settings, "secrets_file", "keys.txt"), key_env
            )
            if api_key:
                try:
                    self.async_client = AsyncOpenAI(api_key=api_key)
                    self.available = True
                except Exception as exc:
                    logger.warning(f"LLMStrategist OpenAI init failed: {exc}")

        self.pplx = PerplexitySearcher(settings)

    def due(self) -> bool:
        return (time.time() - self.last_run_ts) >= self.interval_min * 60

    def fetch_breaking_news(self, query: str, hours: int = 6) -> list[dict[str, Any]]:
        """Fetch recent headlines via Perplexity (best-effort)."""
        if not self.pplx.available:
            return []
        try:
            results = self.pplx.search(query, num_results=self.settings.news.perplexity_num_results)
            cutoff = datetime.now(UTC) - timedelta(hours=hours)
            filtered = []
            for r in results:
                ts = r.get("published_at")
                if ts is not None and ts < cutoff:
                    continue
                filtered.append(
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", ""),
                        "url": r.get("url", ""),
                        "published_at": ts.isoformat() if ts else None,
                    }
                )
            return filtered
        except Exception as exc:
            logger.warning(f"Perplexity breaking-news fetch failed: {exc}")
            return []

    async def advise(
        self, news_events: list[dict[str, Any]], risk_state: dict[str, Any], kpis: dict[str, Any]
    ) -> StrategistAdvice | None:
        """
        Create a strategic policy hint using LLM (Async).
        """
        if not self.available or not self.async_client:
            return None
        self.last_run_ts = time.time()

        latest_news = []
        for ev in news_events[-8:]:  # limit to last few items
            latest_news.append(
                {
                    "title": ev.get("title", ""),
                    "sentiment": ev.get("sentiment", 0.0),
                    "confidence": ev.get("confidence", 0.0),
                    "currencies": ev.get("currencies", []),
                }
            )

        system_prompt = (
            "You are the Chief Risk Officer (CRO) at an FX Prop Firm. "
            "Your goal is account survival and consistency (Targeting 4% Monthly). "
            "Analyze the news and state for 'Signal Congestion' and 'Liquidity Traps'. "
            "1. Sentiment Check: If news sentiment is > 0.8 or < -0.8, flag as 'Crowded' and reduce risk. "
            "2. Risk State: If PnL is near daily drawdown limits, enforce 'Maintenance Mode'. "
            "3. Rationale: Explain the Liquidity/Yield reasoning in 10 words. "
            "OUTPUT JSON: "
            "{ 'stance': 'conservative'|'balanced'|'aggressive', "
            "  'risk_adjustments': {'max_risk_per_trade': float, 'min_confidence_threshold': float}, "
            "  'trade_filters': {'avoid_pairs': [], 'focus_pairs': [], 'suspend_trading': bool}, "
            "  'commentary': str }"
        )
        user_payload = {
            "news": latest_news,
            "risk": risk_state,
            "kpis": kpis,
            "format": {
                "stance": "conservative|balanced|aggressive",
                "risk_adjustments": {"max_risk_per_trade": "float 0-0.05", "min_confidence_threshold": "0-1"},
                "trade_filters": {"avoid_pairs": ["list"], "focus_pairs": ["list"], "suspend_trading": "bool"},
                "commentary": "short rationale",
            },
        }
        try:
            resp = await self.async_client.chat.completions.create(
                model=getattr(self.settings.news, "openai_model", "gpt-5-nano-2025-08-07"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload)},
                ],
                temperature=0.2,
                max_completion_tokens=getattr(self.settings.news, "openai_max_tokens", 256),
            )
            content = resp.choices[0].message.content if resp.choices else ""
            data = _safe_json_object_loads(content)
            if not data:
                logger.warning("Strategist returned non-JSON; skipping advice.")
                return None
            stance = str(data.get("stance", "balanced"))
            risk_adj = data.get("risk_adjustments", {}) or {}
            trade_filters = data.get("trade_filters", {}) or {}
            commentary = data.get("commentary", "")
            return StrategistAdvice(
                stance=stance,
                risk_adjustments=risk_adj,
                trade_filters=trade_filters,
                commentary=commentary,
                raw=data,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(f"LLM strategist error: {exc}")
            return None

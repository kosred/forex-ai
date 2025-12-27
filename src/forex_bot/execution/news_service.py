import asyncio
import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from ..core.config import Settings
from ..core.storage import RiskLedger
from ..data.news.client import get_sentiment_analyzer
from ..llm.strategist import LLMStrategist

logger = logging.getLogger(__name__)


class NewsService:
    """
    Handles news fetching, sentiment analysis, and LLM Strategist interactions.
    Manages the 'News Policy' (risk adjustments based on events).
    """

    def __init__(self, settings: Settings, risk_ledger: RiskLedger | None = None):
        self.settings = settings
        self.risk_ledger = risk_ledger
        self.analyzer = None
        # Note: get_sentiment_analyzer is now async, but init can't be async
        # We'll lazy-init in get_news_policy if needed

        self.strategist = LLMStrategist(settings) if getattr(settings.news, "strategist_enabled", False) else None

        self._last_news_fetch: datetime | None = None
        lookahead = int(getattr(self.settings.news, "news_lookahead_minutes", 60))
        self._news_fetch_min_seconds = max(900, int(lookahead * 60 / 2))
        self._news_pre_event_seconds = 300
        self._news_post_event_seconds = 120

        self.latest_events: list = []
        self.breaking_news: list = []
        self.last_advice: dict | None = None

    async def get_news_policy(self, symbol: str) -> dict[str, Any]:
        """
        Fetch news, update internal state, and return the current News Policy
        (flags for tier1 events, confidence scores, etc).
        """
        # Lazy-init analyzer
        if not self.analyzer and getattr(self.settings.news, "enable_news", True):
            self.analyzer = await get_sentiment_analyzer(self.settings)

        if not self.analyzer:
            return {}

        now_utc = datetime.now(UTC)
        should_fetch = self._check_fetch_needed(now_utc)

        # Window calculation
        lookahead = int(getattr(self.settings.news, "news_lookahead_minutes", 60))
        back_min = max(60, lookahead)
        start = now_utc - timedelta(minutes=back_min)
        end_future = now_utc + timedelta(minutes=lookahead)

        # If we have cached events, check if a calendar event is imminent to force fetch
        if not should_fetch and self.latest_events:
            for ev in self.latest_events:
                if hasattr(ev, "source") and ev.source == "MT5Calendar" and ev.published_at:
                    delta = (ev.published_at - now_utc).total_seconds()
                    if -self._news_post_event_seconds <= delta <= self._news_pre_event_seconds:
                        should_fetch = True
                        logger.info(f"News fetch forced by event window (dt={delta:.1f}s).")
                        break

        if should_fetch:
            try:
                max_openai = int(getattr(self.settings.news, "openai_max_events_per_fetch", 50) or 50)

                def _fetch_score_persist() -> list:
                    # Blocking web fetch + LLM scoring (run off the event loop).
                    events_local = self.analyzer.fetch_news_bundle(symbol, start, end_future, use_live_llm=True) or []
                    if (
                        getattr(self.analyzer, "openai_scorer", None) is not None
                        and self.analyzer.openai_scorer.available
                    ):
                        events_local = self.analyzer.score_events(events_local, max_openai=max_openai)
                    try:
                        # Persist so the bot's historical/news-aware features improve over time.
                        self.analyzer.db.insert_events(events_local, update_existing=True)
                    except Exception as exc:
                        logger.debug(f"News persist failed: {exc}")
                    return events_local

                events = await asyncio.to_thread(_fetch_score_persist)
                self.latest_events = events or []
                self._last_news_fetch = now_utc
                logger.info(f"Fetched {len(self.latest_events)} news/calendar events.")
            except Exception as e:
                logger.warning(f"News fetch failed: {e}")

        # Analyze Events for Policy
        tier1_nearby = False
        for ev in self.latest_events:
            if hasattr(ev, "source") and ev.source == "MT5Calendar" and ev.published_at:
                if start <= ev.published_at <= end_future:
                    imp = ev.confidence if ev.confidence is not None else 0.0
                    if imp >= 0.66:
                        tier1_nearby = True
                        break

        # HPC FIX: Exponential Sentiment/Confidence Decay
        decay_half_life = float(getattr(self.settings.news, "news_decay_minutes", 120))
        decay_factor = math.log(2) / decay_half_life
        
        impactful_events = []
        for ev in self.latest_events:
            if ev.published_at:
                age_mins = (now_utc - ev.published_at).total_seconds() / 60.0
                # Weight by time: Weight = exp(-decay * time)
                time_weight = math.exp(-decay_factor * max(0, age_mins))
                
                impactful_events.append({
                    "conf": (ev.confidence or 0.0) * time_weight,
                    "sent": (ev.sentiment or 0.0) * time_weight
                })

        max_conf = max((e["conf"] for e in impactful_events), default=0.0)
        max_sent = max((e["sent"] for e in impactful_events), default=0.0)
        
        # Risk cap reduces as confidence/surprise increases
        news_cap = self.settings.risk.max_risk_per_trade * max(0.2, 1.0 - max_conf)

        policy = {
            "tier1_nearby": tier1_nearby,
            "news_confidence": max_conf,
            "news_surprise": max_sent,
            "suggested_risk_cap": news_cap,
        }

        if tier1_nearby and self.risk_ledger:
            self.risk_ledger.record(
                "NEWS_TIER1", "Tier-1 news nearby; risk reduced", severity="warning", context={"max_conf": max_conf}
            )

        # Breaking News Fetch (Perplexity)
        if self.strategist:
            if tier1_nearby or len(self.latest_events) < max(3, self.settings.news.perplexity_num_results):
                # Run blocking fetch off the event loop to avoid stalls
                await asyncio.to_thread(self._fetch_breaking_news, symbol, tier1_nearby)

        return policy

    def _check_fetch_needed(self, now: datetime) -> bool:
        """HPC FIX: Dynamic Event-Aware Polling."""
        if self._last_news_fetch is None: return True
        
        # Default: 15 minutes
        interval = self._news_fetch_min_seconds
        
        # Check for imminent high-impact events
        for ev in self.latest_events:
            if hasattr(ev, "published_at") and ev.published_at:
                delta = (ev.published_at - now).total_seconds()
                # FLASH WINDOW: 5 mins before/after event
                if -300 <= delta <= 300:
                    # Increase frequency to every 10 seconds
                    interval = 10 
                    logger.debug(f"SmartNews: Flash Window detected for {ev.title}. Polling every 10s.")
                    break
        
        elapsed = (now - self._last_news_fetch).total_seconds()
        return elapsed >= interval

    def _fetch_breaking_news(self, symbol: str, tier1: bool):
        try:
            query = f"{symbol} forex news breaking"
            self.breaking_news = self.strategist.fetch_breaking_news(query, hours=6)
            if tier1:
                speech_query = "fed chair live speech OR president live remarks forex"
                self.breaking_news.extend(self.strategist.fetch_breaking_news(speech_query, hours=6))
        except Exception as e:
            logger.warning(f"Breaking news fetch failed: {e}")

    async def get_strategic_advice(self, risk_state: dict, kpis: dict) -> dict | None:
        """Ask the LLM Strategist for advice based on current news and risk state."""
        if not self.strategist or not self.strategist.due():
            return None

        try:
            news_payload = [
                {
                    "title": ev.title,
                    "summary": ev.summary,
                    "sentiment": ev.sentiment,
                    "confidence": ev.confidence,
                    "currencies": ev.currencies,
                    "source": ev.source,
                }
                for ev in (self.latest_events or [])[-8:]
            ]
            # Add breaking news
            for b in self.breaking_news[-5:]:
                news_payload.append(
                    {
                        "title": b.get("title", ""),
                        "summary": b.get("snippet", ""),
                        "source": "PerplexityBreaking",
                        "sentiment": 0.0,
                        "confidence": 0.0,
                        "currencies": [],
                    }
                )

            advice = await self.strategist.advise(news_payload, risk_state, kpis)
            if advice:
                self.last_advice = advice.raw
                logger.info(f"Strategist stance={advice.stance} filters={advice.trade_filters}")
                if self.risk_ledger:
                    self.risk_ledger.record(
                        "STRATEGIST", advice.commentary, severity="info", context={"stance": advice.stance}
                    )
                return self.last_advice
        except Exception as e:
            logger.warning(f"Strategist overlay failed: {e}")
        return None

    def get_news_features(self, base_idx: pd.Index) -> pd.DataFrame | None:
        """Build feature dataframe from current news events."""
        if not self.analyzer or not self.latest_events:
            return None
        try:
            return self.analyzer.build_features(self.latest_events, base_idx)
        except Exception as e:
            logger.warning(f"News feature build failed: {e}")
            return None

#!/usr/bin/env python3
"""
Rescore (or initially score) local/backfilled news using OpenAI, then persist the
sentiment/confidence back into `cache/news.sqlite`.

This avoids online headline search: it only reads existing events from the DB,
scores them (LLM), and updates stored scores.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime

import pandas as pd

from forex_bot.core.config import Settings
from forex_bot.data.news.client import get_sentiment_analyzer


def _parse_dt(s: str) -> datetime:
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty datetime")
    # Accept YYYY-MM-DD or full ISO strings.
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return datetime.fromisoformat(s).replace(tzinfo=UTC)
    dt = pd.to_datetime(s, utc=True, errors="raise").to_pydatetime()
    return dt.astimezone(UTC)


async def main_async() -> int:
    p = argparse.ArgumentParser(description="Rescore local news in cache/news.sqlite using OpenAI.")
    p.add_argument("--start", required=True, help="Start datetime (YYYY-MM-DD or ISO).")
    p.add_argument("--end", required=True, help="End datetime (YYYY-MM-DD or ISO).")
    p.add_argument("--symbol", default="", help="FX symbol like EURUSD (used to infer currencies).")
    p.add_argument(
        "--currencies", default="", help="Comma-separated currency filter (e.g. USD,EUR). Overrides --symbol."
    )
    p.add_argument("--max-openai", type=int, default=200, help="Max events to score via OpenAI this run.")
    p.add_argument("--openai-model", default="", help="Override settings.news.openai_model for this run.")
    args = p.parse_args()

    settings = Settings()
    settings.system.mt5_required = False
    settings.news.enable_news = True
    # Avoid online fetch/search; only rescore local DB events.
    settings.news.perplexity_enabled = False
    settings.news.openai_news_enabled = False
    settings.news.enable_llm_helper = False
    settings.news.llm_helper_enabled = False
    settings.news.strategist_enabled = False

    if args.openai_model.strip():
        settings.news.openai_model = args.openai_model.strip()

    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")

    currencies: list[str] | None = None
    if args.currencies.strip():
        currencies = [c.strip().upper() for c in args.currencies.split(",") if c.strip()]
    elif args.symbol.strip():
        sym = args.symbol.strip().upper()
        if len(sym) == 6:
            currencies = [sym[:3], sym[3:]]

    analyzer = await get_sentiment_analyzer(settings)
    if not analyzer.openai_scorer.available:
        raise SystemExit("OpenAI scorer is not available. Set OPENAI_API_KEY (or keys.txt) and try again.")

    events = analyzer.db.fetch_events(start, end, currencies=currencies)
    print(f"Loaded {len(events)} events from DB for scoring.")
    if not events:
        return 0

    scored = analyzer.score_events(events, max_openai=int(args.max_openai))
    # Update existing rows with new scores.
    analyzer.db.insert_events(scored, update_existing=True)
    print(f"Done. Scored (or reused cached scores for) {len(scored)} events.")
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

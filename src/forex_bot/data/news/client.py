import asyncio
import glob
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...core.config import Settings
from .scorers import OpenAIScorer, ScoreResult
from .searchers import PerplexitySearcher
from .store import NewsDatabase, NewsEvent

logger = logging.getLogger(__name__)


# String utilities to stay future-proof with pandas nullable dtypes
def _to_string(series: pd.Series) -> pd.Series:
    """Convert a Series to pandas StringDtype, strip, and fill missing with empty."""
    return series.astype("string", copy=False).fillna("").str.strip()


def _normalize_df_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Normalize listed columns to string dtype with stripping."""
    for col in cols:
        if col in df.columns:
            df[col] = _to_string(df[col])
    return df


class SentimentAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        db_path = Path(settings.system.cache_dir) / "news.sqlite"
        self.db = NewsDatabase(db_path)
        self.openai_scorer = OpenAIScorer(settings)
        self.pplx_searcher = PerplexitySearcher(settings)
        self._local_seeded = False
        self.degraded = False
        self.last_error: str | None = None
        self.cross_influence = {
            "AUD": ["NZD", "CNY", "XAU", "XAG"],
            "NZD": ["AUD", "CNY"],
            "CAD": ["USD", "OIL"],
            "CHF": ["EUR"],
            "EUR": ["CHF"],
            "GBP": ["EUR"],
            "USD": ["CAD", "JPY"],
            "JPY": ["USD"],
        }

        self.currency_synonyms = {
            "USD": ["USD", "US DOLLAR", "GREENBACK"],
            "EUR": ["EUR", "EURO", "EUROZONE"],
        }
        self._last_daily_fetch: date | None = None
        self._seed_local_archives()

    def ensure_news_history(
        self, symbol: str, start: datetime, end: datetime, force: bool = False, use_live_llm: bool = True
    ) -> int:
        """
        Fetch, score, and persist news between start/end.
        Returns the count of newly inserted events (best effort).
        """
        if not force:
            latest = self.db.latest_timestamp()
            if latest and latest >= end:
                return 0
        events = self.fetch_news_bundle(symbol, start, end, use_live_llm=use_live_llm)
        if not use_live_llm:
            scored = events  # Keep existing sentiment/confidence (e.g., from archives/DB)
        else:
            scored = self.score_events(events)
        try:
            # When LLM scoring is enabled, update existing rows so backfilled archives can be rescored.
            return self.db.insert_events(scored, update_existing=bool(use_live_llm))
        except Exception as exc:
            logger.warning(f"Failed to persist news events: {exc}")
            return 0

    def update_sentiments(self, force: bool = False) -> int:
        """
        Fetch new articles and update sentiment scores.
        By default this runs at most once per UTC day to avoid constant polling; set force=True to override.
        """
        count = 0
        now = datetime.now(UTC)
        start = datetime(now.year, now.month, now.day, tzinfo=UTC)
        end = now
        try:
            count = self.ensure_news_history(symbol="ALL", start=start, end=end, force=force)
        except Exception as exc:
            logger.warning(f"Sentiment update skipped: {exc}")
        return count

    def fetch_news_bundle(
        self, symbol: str, start: datetime, end: datetime, use_live_llm: bool = True
    ) -> list[NewsEvent]:
        """
        End-to-end fetch using Perplexity (primary) and OpenAI (augment/fallback).
        Returns list of NewsEvent (unscored).
        """
        self.degraded = False
        self.last_error = None
        all_events: list[NewsEvent] = []
        target_currencies = [symbol[:3], symbol[3:]]

        existing_db_events = self.db.fetch_events(start, end, currencies=target_currencies)
        all_events.extend(existing_db_events)

        known_event_ids = set()
        for ev in existing_db_events:
            if ev.published_at and ev.title:
                # Use date+hour for deduplication to avoid microsecond differences
                pub_key = ev.published_at.strftime("%Y-%m-%d %H")
                known_event_ids.add((ev.title.strip().lower(), pub_key))

        if use_live_llm and self.settings.news.perplexity_enabled:
            try:
                if (end - start).days < 2:
                    query = f"Breaking forex news for {symbol} today and potential market impact"
                else:
                    query = f"Major economic events and news for {symbol} from {start.date()} to {end.date()}"
                pplx_events = self._fetch_perplexity_headlines(query, target_currencies)
                new_pplx_events = [
                    ev
                    for ev in pplx_events
                    if (ev.title.strip().lower(), ev.published_at.isoformat()) not in known_event_ids
                ]
                if new_pplx_events:
                    logger.info(f"Perplexity found {len(new_pplx_events)} new relevant news items.")
                    all_events.extend(new_pplx_events)
                    for ev in new_pplx_events:
                        known_event_ids.add((ev.title.strip().lower(), ev.published_at.isoformat()))
            except Exception as e:
                self.degraded = True
                self.last_error = f"Perplexity search failed: {e}"
                logger.warning(self.last_error)

        if use_live_llm and getattr(self.settings.news, "openai_news_enabled", True):
            try:
                oa_query = (
                    f"Latest high-impact FX and macro headlines affecting {symbol} "
                    f"between {start.isoformat()} and {end.isoformat()}"
                )
                oa_events = self._fetch_openai_headlines(oa_query, target_currencies)
                new_oa_events = [
                    ev
                    for ev in oa_events
                    if (ev.title.strip().lower(), ev.published_at.isoformat()) not in known_event_ids
                ]
                if new_oa_events:
                    logger.info(f"OpenAI provided {len(new_oa_events)} headline candidates.")
                    all_events.extend(new_oa_events)
                    for ev in new_oa_events:
                        known_event_ids.add((ev.title.strip().lower(), ev.published_at.isoformat()))
            except Exception as exc:
                logger.warning(f"OpenAI headline fetch failed: {exc}")

        dedup_final: dict[str, NewsEvent] = {}
        for ev in all_events:
            key = (ev.url or ev.title).lower().strip()
            if key in dedup_final:
                existing = dedup_final[key]
                if ev.source in ["Perplexity", "OpenAI"] and existing.source not in ["Perplexity", "OpenAI"]:
                    dedup_final[key] = ev
            else:
                dedup_final[key] = ev
        all_events = list(dedup_final.values())

        for ev in all_events:
            cur_list = {c for c in (ev.currencies or []) if c}
            for c in list(cur_list):
                for xc in self.cross_influence.get(c, []):
                    cur_list.add(xc)
            ev.currencies = list(cur_list)

        return all_events

    def _parse_timestamp(self, raw: Any) -> datetime:
        if isinstance(raw, datetime):
            return raw.astimezone(UTC)
        if isinstance(raw, str) and raw:
            # Clean string
            clean = raw.strip().replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(clean).astimezone(UTC)
            except Exception:
                try:
                    # Fallback to pandas flexible parser
                    dt = pd.to_datetime(clean, utc=True).to_pydatetime()
                    if pd.isna(dt):
                        raise ValueError("NaT result")
                    return dt
                except Exception as e:
                    logger.debug(f"Timestamp parse failed for '{raw}': {e}")

        # Fallback to epoch if parsing fails - DO NOT use now() to avoid triggering kill switches
        return datetime(1970, 1, 1, tzinfo=UTC)

    def _seed_local_archives(self) -> None:
        """
        Ingest local news CSV archives (best-effort) into the news DB.
        Supports improved currency heuristics and column normalization.
        """
        if self._local_seeded:
            return

        # Default directory + configured glob
        search_paths = [Path("data/news"), Path("data")]
        glob_pattern = getattr(self.settings.news, "news_local_glob", "")

        files = []
        if glob_pattern:
            files.extend(glob.glob(glob_pattern))

        for base in search_paths:
            if base.exists():
                files.extend(list(base.glob("*.csv")))
                files.extend(list(base.glob("*.json")))

        if not files:
            self._local_seeded = True
            return

        total_inserted = 0
        # Deduplicate file paths
        unique_files = {str(p) for p in files}

        for path_str in unique_files:
            path = Path(path_str)
            try:
                if path.suffix == ".json":
                    df = pd.read_json(path)
                else:
                    df = pd.read_csv(path)

                # Normalize columns for flexible matching (covers ForexFactory archive: DateTime/Currency/Event/Impact)
                df.columns = [str(c).lower().strip() for c in df.columns]
                df = df.convert_dtypes()
            except Exception as exc:
                logger.warning(f"Failed to read news archive {path}: {exc}")
                continue

            if df.empty:
                continue

            # Flexible column resolution
            title_col = next((c for c in ["title", "event", "headline"] if c in df.columns), None)
            ts_col = next(
                (c for c in ["published_at", "timestamp", "date", "time", "datetime"] if c in df.columns), None
            )
            summary_col = next((c for c in ["summary", "detail", "description", "snippet"] if c in df.columns), None)
            currency_col = next((c for c in ["currencies", "currency", "ccy"] if c in df.columns), None)
            impact_col = "impact" if "impact" in df.columns else None

            if not title_col or not ts_col:
                continue

            # Vectorized preprocessing
            df = _normalize_df_strings(df, [title_col, summary_col or "", "url"])
            df["title_clean"] = df[title_col]
            df = df[df["title_clean"] != ""]
            if df.empty:
                continue

            # Parse timestamps vectorized (UTC)
            df["ts_parsed"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
            df = df[df["ts_parsed"].notna()]
            df = df[df["ts_parsed"].dt.year >= 1990]
            if df.empty:
                continue

            # Vectorized string columns
            df["url_clean"] = df.get("url", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
            df["summary_clean"] = _to_string(df[summary_col]) if summary_col else ""
            df["sentiment_val"] = pd.to_numeric(df.get("sentiment", pd.Series(dtype=float)), errors="coerce").fillna(
                0.0
            )

            # Impact -> confidence (forexfactory style). Defaults preserved for other archives.
            impact_map = {
                "HIGH": 0.9,
                "MEDIUM": 0.6,
                "MED": 0.6,
                "LOW": 0.35,
                "NON-ECONOMIC": 0.2,
                "NONE": 0.2,
            }
            if impact_col:
                df["confidence_val"] = (
                    df[impact_col]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .map(impact_map)
                    .fillna(pd.to_numeric(df.get("confidence", pd.Series(dtype=float)), errors="coerce"))
                    .fillna(0.0)
                )
            else:
                df["confidence_val"] = pd.to_numeric(
                    df.get("confidence", pd.Series(dtype=float)), errors="coerce"
                ).fillna(0.0)

            # Currencies
            if currency_col:
                cur_col = _to_string(df[currency_col])
            else:
                cur_col = _to_string(df.get("currencies", pd.Series(dtype=str)))
            df["currencies_list"] = (
                cur_col.str.replace(";", ",")
                .str.split(",")
                .apply(lambda x: [c.strip().upper() for c in x if len(c.strip()) >= 3][:3])
            )

            # Fallback currency detection in title
            def extract_currencies_from_title(row):
                if row["currencies_list"]:
                    return row["currencies_list"]
                return [
                    c
                    for c in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "XAU", "XAG"]
                    if c in row["title_clean"].upper()
                ]

            df["currencies_final"] = df.apply(extract_currencies_from_title, axis=1)

            # Build events using itertuples (8x faster than iterrows)
            events: list[NewsEvent] = []
            for i, row in enumerate(df.itertuples(index=False)):
                url = str(row.url_clean) if hasattr(row, "url_clean") else ""
                if not url:
                    url = f"localcsv://{path.name}-{i}"

                # Enrich summary for ForexFactory-like rows with Actual/Forecast/Previous if present
                summary = str(getattr(row, "summary_clean", "") or "")
                if hasattr(row, "actual") or hasattr(row, "forecast") or hasattr(row, "previous"):
                    extras = []
                    if hasattr(row, "actual"):
                        extras.append(f"Actual={row.actual}")
                    if hasattr(row, "forecast"):
                        extras.append(f"Forecast={row.forecast}")
                    if hasattr(row, "previous"):
                        extras.append(f"Previous={row.previous}")
                    if extras:
                        summary = (summary + " | " + ", ".join(extras)).strip(" |,")

                try:
                    events.append(
                        NewsEvent(
                            title=str(getattr(row, "title_clean", "") or "").strip(),
                            summary=summary,
                            url=str(url),
                            source="LocalArchive",
                            published_at=row.ts_parsed.to_pydatetime(),
                            sentiment=float(row.sentiment_val),
                            confidence=float(row.confidence_val),
                            currencies=row.currencies_final,
                        )
                    )
                except Exception as exc:
                    logger.debug(f"Skipping malformed news row {i} in {path.name}: {exc}")

            if events:
                try:
                    inserted = self.db.insert_events(events)
                    total_inserted += inserted
                    logger.info(f"Seeded {inserted} events from {path.name}")
                except Exception as exc:
                    logger.warning(f"Failed to seed archive {path}: {exc}")

        if total_inserted > 0:
            logger.info(f"Local news archive ingest complete. Inserted {total_inserted} events.")
        self._local_seeded = True

    def _fetch_perplexity_headlines(self, query: str, currencies: list[str]) -> list[NewsEvent]:
        events: list[NewsEvent] = []
        if not self.pplx_searcher.available:
            return events
        raw = self.pplx_searcher.search(
            query,
            num_results=int(getattr(self.settings.news, "perplexity_num_results", 10)),
        )
        for item in raw:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            ts = self._parse_timestamp(item.get("published_at") or item.get("date"))
            url = str(item.get("url") or f"perplexity://{hash(title + ts.isoformat())}")
            summary = str(item.get("snippet", "") or "")
            events.append(
                NewsEvent(
                    title=title,
                    summary=summary,
                    url=url,
                    source="Perplexity",
                    published_at=ts,
                    sentiment=0.0,
                    confidence=0.0,
                    currencies=currencies,
                )
            )
        return events

    def _fetch_openai_headlines(self, query: str, currencies: list[str]) -> list[NewsEvent]:
        events: list[NewsEvent] = []
        if not self.openai_scorer.available or self.openai_scorer.client is None:
            return events

        system_prompt = (
            "You are a real-time FX news scout. "
            "Return high-impact macro/FX headlines as JSON: "
            '{ "headlines": [ {"title": str, "summary": str, "url": str, "published_at": iso8601} ] } '
            "Only include items from reputable sources and avoid speculation."
        )
        try:
            resp = self.openai_scorer.client.chat.completions.create(
                model=self.openai_scorer.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_completion_tokens=self.openai_scorer.max_tokens,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            headlines = data.get("headlines", [])
            if isinstance(headlines, dict):
                headlines = headlines.get("items", [])
            for item in headlines:
                title = str(item.get("title", "")).strip()
                if not title:
                    continue
                ts = self._parse_timestamp(item.get("published_at") or item.get("date"))
                url = str(item.get("url") or f"openai://{hash(title + ts.isoformat())}")
                summary = str(item.get("summary") or item.get("snippet") or "")
                events.append(
                    NewsEvent(
                        title=title,
                        summary=summary,
                        url=url,
                        source="OpenAI",
                        published_at=ts,
                        sentiment=0.0,
                        confidence=0.0,
                        currencies=currencies,
                    )
                )
        except Exception as exc:
            logger.warning(f"OpenAI headline parsing failed: {exc}")
        return events

    def score_events(
        self,
        events: list[NewsEvent],
        max_openai: int | None = None,
        *,
        force_openai: bool = False,
    ) -> list[NewsEvent]:
        """
        Score events with OpenAI (LLM-first).
        """
        if max_openai is None:
            try:
                max_openai = int(getattr(self.settings.news, "openai_max_events_per_fetch", 50) or 50)
            except Exception:
                max_openai = 50
        max_openai = int(max(0, max_openai))

        scored = []
        openai_available = self.openai_scorer.available
        used_openai = 0

        sorted_events = sorted(events, key=lambda e: e.confidence or 0.0, reverse=True)

        for ev in sorted_events:
            cached = None
            try:
                cached = self.db.get_cached_score(ev.url, ev.title)
            except Exception:
                cached = None
            if cached:
                ev.sentiment = cached["sentiment"]
                ev.confidence = cached["confidence"]
                scored.append(ev)
                continue

            result: ScoreResult | None = None

            use_openai = False
            if openai_available:
                if force_openai:
                    use_openai = used_openai < max_openai
                else:
                    is_high_impact = (ev.confidence or 0.0) >= 0.8
                    if is_high_impact and used_openai < max_openai:
                        use_openai = True
                    elif used_openai < (max_openai * 0.5):
                        use_openai = True

            if use_openai:
                try:
                    result = self.openai_scorer.score(ev.title, ev.currencies or [])
                    used_openai += 1
                except Exception as exc:
                    logger.warning(f"OpenAI scoring failed: {exc}")
                    use_openai = False
                    openai_available = False  # Disable further OpenAI attempts in this batch

            if result:
                ev.sentiment = result.sentiment
                ev.confidence = result.confidence

                try:
                    self.db.set_cached_score(ev.url, ev.title, result.sentiment, result.confidence, result.direction)
                except Exception as e:
                    logger.warning(f"News client request failed: {e}", exc_info=True)
            else:
                ev.sentiment = ev.sentiment or 0.0
                ev.confidence = ev.confidence or 0.0
            scored.append(ev)

        return scored

    def rescore_existing_events(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        max_events: int | None = None,
        only_missing: bool | None = None,
    ) -> int:
        """
        Rescore existing DB/archive events (no web search) and persist updated sentiment/confidence.

        Intended for backfilled archives that have missing or heuristic scores.
        """
        if not self.openai_scorer.available:
            return 0

        if max_events is None:
            try:
                max_events = int(getattr(self.settings.news, "auto_rescore_max_events", 200) or 200)
            except Exception:
                max_events = 200
        max_events = int(max(0, max_events))
        if max_events <= 0:
            return 0

        if only_missing is None:
            only_missing = bool(getattr(self.settings.news, "auto_rescore_only_missing", True))

        target_currencies = [symbol[:3], symbol[3:]] if isinstance(symbol, str) and len(symbol) == 6 else []
        try:
            events = self.db.fetch_events(start, end, currencies=target_currencies or None)
        except Exception as exc:
            logger.warning(f"Rescore fetch failed: {exc}")
            return 0

        if not events:
            return 0

        candidates: list[NewsEvent] = []
        for ev in events:
            if not only_missing:
                candidates.append(ev)
                continue
            try:
                sent = float(ev.sentiment or 0.0)
                conf = float(ev.confidence or 0.0)
                if sent == 0.0 and conf == 0.0:
                    candidates.append(ev)
            except Exception:
                continue

        if not candidates:
            return 0

        # Prefer most recent rows first to improve current training/live relevance under a tight budget.
        candidates.sort(key=lambda e: e.published_at or datetime(1970, 1, 1, tzinfo=UTC), reverse=True)
        candidates = candidates[:max_events]

        try:
            scored = self.score_events(candidates, max_openai=max_events, force_openai=True)
        except Exception as exc:
            logger.warning(f"Rescore scoring failed: {exc}")
            return 0

        try:
            self.db.insert_events(scored, update_existing=True)
        except Exception as exc:
            logger.warning(f"Rescore persist failed: {exc}")

        return len(scored)

    def build_features(
        self,
        events: list[NewsEvent],
        base_index: pd.DatetimeIndex,
        *,
        include_activation: bool = False,
    ) -> pd.DataFrame:
        """
        Aggregate news events into time-aligned features for models.

        Outputs columns:
          - news_sentiment: last known sentiment (ffill)
          - news_confidence: last known confidence (ffill)
          - news_count: number of events in the prior 6 hours (inclusive)
          - news_recency_minutes: minutes since the most recent event (9999 if none)
        """
        if events is None:
            return pd.DataFrame(index=base_index) if base_index is not None else pd.DataFrame()
        if isinstance(events, pd.DataFrame):
            try:
                events = events.to_dict("records")
            except Exception:
                events = list(events)
        if len(events) == 0 or base_index is None or len(base_index) == 0:
            return pd.DataFrame(index=base_index) if base_index is not None else pd.DataFrame()

        try:
            base_index = pd.DatetimeIndex(base_index).sort_values()
            if base_index.has_duplicates:
                base_index = base_index[~base_index.duplicated(keep="first")]
        except Exception:
            return pd.DataFrame(index=base_index)

        rows = []
        for ev in events:
            try:
                ts_raw = getattr(ev, "published_at", None)
                if ts_raw is None and isinstance(ev, dict):
                    ts_raw = ev.get("published_at")
                ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
                if ts is pd.NaT or pd.isna(ts):
                    continue

                sent_raw = getattr(ev, "sentiment", None)
                conf_raw = getattr(ev, "confidence", None)
                if isinstance(ev, dict):
                    sent_raw = ev.get("sentiment", sent_raw)
                    conf_raw = ev.get("confidence", conf_raw)

                rows.append(
                    {
                        "published_at": ts,
                        "sentiment": float(sent_raw or 0.0),
                        "confidence": float(conf_raw or 0.0),
                    }
                )
            except Exception:
                continue
        if not rows:
            return pd.DataFrame(index=base_index)
        df = pd.DataFrame(rows).set_index("published_at")
        df = df.sort_index()
        if df.index.has_duplicates:
            df = df.groupby(df.index).agg({"sentiment": "mean", "confidence": "mean"}).sort_index()
        feat = pd.DataFrame(index=base_index)
        feat["news_sentiment"] = df["sentiment"].reindex(base_index, method="ffill").fillna(0.0).astype(np.float32)
        feat["news_confidence"] = df["confidence"].reindex(base_index, method="ffill").fillna(0.0).astype(np.float32)

        # Vectorized window stats (avoid per-bar slicing loops).
        ev_times = df.index.values.astype("datetime64[ns]")
        base_times = base_index.values.astype("datetime64[ns]")

        right = np.searchsorted(ev_times, base_times, side="right")
        left = np.searchsorted(ev_times, base_times - np.timedelta64(6, "h"), side="left")
        feat["news_count"] = (right - left).astype(np.float32)

        recency = np.full(base_times.shape, 9999.0, dtype=np.float32)
        has_prev = right > 0
        if np.any(has_prev):
            prev_idx = right[has_prev] - 1
            prev_times = ev_times[prev_idx]
            recency[has_prev] = ((base_times[has_prev] - prev_times) / np.timedelta64(1, "m")).astype(np.float32)
        feat["news_recency_minutes"] = recency

        if include_activation:
            act = self._build_activation(events, base_index, back_minutes=60, fwd_minutes=15)
            for col in act.columns:
                feat[col] = act[col]

        return feat

    def _build_activation(
        self, events: list[NewsEvent], base_index: pd.DatetimeIndex, back_minutes: int = 60, fwd_minutes: int = 15
    ) -> pd.DataFrame:
        """Flag whether news is near each timestamp and the max sentiment/confidence in window."""
        df = pd.DataFrame(index=base_index)
        if not events or base_index is None or len(base_index) == 0:
            df["news_nearby"] = 0
            df["news_conf_max"] = 0.0
            df["news_sent_max"] = 0.0
            return df
        ev_df = []
        for ev in events:
            ts = ev.published_at if isinstance(ev.published_at, datetime) else None
            if ts is None:
                continue
            ev_df.append(
                {
                    "ts": pd.to_datetime(ts, utc=True),
                    "sent": float(ev.sentiment),
                    "conf": float(ev.confidence),
                }
            )
        if not ev_df:
            df["news_nearby"] = 0
            df["news_conf_max"] = 0.0
            df["news_sent_max"] = 0.0
            return df
        ev = pd.DataFrame(ev_df)
        df["news_nearby"] = 0
        df["news_conf_max"] = 0.0
        df["news_sent_max"] = 0.0
        back = pd.Timedelta(minutes=back_minutes)
        fwd = pd.Timedelta(minutes=fwd_minutes)
        for i, ts in enumerate(base_index):
            window = ev.loc[(ev["ts"] >= ts - back) & (ev["ts"] <= ts + fwd)]
            if not window.empty:
                df.iat[i, df.columns.get_loc("news_nearby")] = 1
                df.iat[i, df.columns.get_loc("news_conf_max")] = float(window["conf"].max())
                df.iat[i, df.columns.get_loc("news_sent_max")] = float(window["sent"].max())
        return df


_ANALYZER: SentimentAnalyzer | None = None
_ANALYZER_LOCK = asyncio.Lock()


async def get_sentiment_analyzer(settings: Settings) -> SentimentAnalyzer:
    """Thread-safe singleton for SentimentAnalyzer with async lock to prevent race conditions."""
    global _ANALYZER
    async with _ANALYZER_LOCK:
        if _ANALYZER is None:
            _ANALYZER = SentimentAnalyzer(settings)
        return _ANALYZER

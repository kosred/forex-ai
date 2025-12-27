import hashlib
import logging
import sqlite3
import threading
import sys
import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NewsEvent:
    """Normalized representation of a news article."""

    title: str
    summary: str
    url: str
    source: str
    published_at: datetime
    sentiment: float
    confidence: float
    currencies: Sequence[str]


class NewsDatabase:
    """Manages persistent storage of news events with cross-process locking."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # HPC FIX: Cross-process lock file
        self.lock_file = self.path.with_suffix(".lock")
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    @contextlib.contextmanager
    def _get_lock(self):
        """High-performance cross-process file lock."""
        import fcntl
        with open(self.lock_file, 'w') as f:
            if sys.platform != "win32":
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if sys.platform != "win32":
                    fcntl.flock(f, fcntl.LOCK_UN)

    def _init_schema(self) -> None:
        with self._get_lock():
            cur = self._conn.cursor()
            # ... (Existing schema logic)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS news_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    summary TEXT,
                    url TEXT UNIQUE,
                    source TEXT,
                    published_at INTEGER NOT NULL,
                    sentiment REAL NOT NULL,
                    confidence REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS news_event_currency (
                    event_id INTEGER NOT NULL,
                    currency TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES news_events(id),
                    UNIQUE(event_id, currency)
                )
                """
            )
            # Indexes for faster range + currency filtering.
            cur.execute("CREATE INDEX IF NOT EXISTS idx_news_events_published_at ON news_events(published_at)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_news_event_currency_currency ON news_event_currency(currency)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_news_event_currency_event_id ON news_event_currency(event_id)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS news_scores_cache (
                    key TEXT PRIMARY KEY,
                    url TEXT,
                    title TEXT,
                    sentiment REAL,
                    confidence REAL,
                    direction TEXT,
                    created_at INTEGER
                )
                """
            )
            self._conn.commit()

    def insert_events(self, events: Iterable[NewsEvent], *, update_existing: bool = False) -> int:
        """
        Insert events, skipping duplicates by URL.

        If `update_existing` is True and an event with the same URL already exists,
        update its sentiment/confidence (and currency mapping) instead of ignoring it.
        """
        with self._get_lock():
            inserted = 0
            cur = self._conn.cursor()
            for event in events:
                try:
                    url = event.url.strip() if event.url else None
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO news_events
                        (title, summary, url, source, published_at, sentiment, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.title.strip()[:512],
                            (event.summary or "").strip()[:1024],
                            url,
                            (event.source or "").strip()[:128],
                            int(event.published_at.replace(tzinfo=UTC).timestamp()),
                            float(event.sentiment),
                            float(event.confidence),
                        ),
                    )
                    inserted_now = bool(getattr(cur, "rowcount", 0) == 1)
                    event_id: int | None = cur.lastrowid if inserted_now else None

                    if inserted_now and event_id:
                        inserted += 1
                    elif (not inserted_now) and update_existing and url:
                        # Update scores for existing records so rescoring/backfills can replace defaults.
                        try:
                            cur.execute(
                                "UPDATE news_events SET sentiment = ?, confidence = ? WHERE url = ?",
                                (float(event.sentiment), float(event.confidence), url),
                            )
                            row = cur.execute("SELECT id FROM news_events WHERE url = ?", (url,)).fetchone()
                            if row and row[0] is not None:
                                event_id = int(row[0])
                        except sqlite3.Error:
                            event_id = None

                    if event_id:
                        for currency in {str(c).upper() for c in (event.currencies or []) if c}:
                            cur.execute(
                                """
                                INSERT OR IGNORE INTO news_event_currency (event_id, currency)
                                VALUES (?, ?)
                                """,
                                (event_id, currency),
                            )
                except sqlite3.Error:
                    continue
            self._conn.commit()
            return inserted

    def fetch_events(
        self,
        start: datetime,
        end: datetime,
        currencies: Sequence[str] | None = None,
    ) -> list[NewsEvent]:
        """Fetch events between start/end optionally filtered by currency."""
        with self._get_lock():
            start_ts = int(start.replace(tzinfo=UTC).timestamp())
            end_ts = int(end.replace(tzinfo=UTC).timestamp())
            cur = self._conn.cursor()
            if currencies:
                placeholders = ",".join("?" for _ in currencies)
                query = f"""
                    SELECT e.id, e.title, e.summary, e.url, e.source,
                           e.published_at, e.sentiment, e.confidence,
                           GROUP_CONCAT(DISTINCT c_all.currency) AS currencies
                    FROM news_events e
                    JOIN news_event_currency c_filter
                      ON c_filter.event_id = e.id
                     AND c_filter.currency IN ({placeholders})
                    LEFT JOIN news_event_currency c_all
                      ON c_all.event_id = e.id
                    WHERE e.published_at BETWEEN ? AND ?
                    GROUP BY e.id
                    ORDER BY e.published_at ASC
                """
                params: tuple = (*[c.upper() for c in currencies], start_ts, end_ts)
            else:
                query = """
                    SELECT e.id, e.title, e.summary, e.url, e.source,
                           e.published_at, e.sentiment, e.confidence,
                           GROUP_CONCAT(DISTINCT c.currency) AS currencies
                    FROM news_events e
                    LEFT JOIN news_event_currency c
                      ON c.event_id = e.id
                    WHERE e.published_at BETWEEN ? AND ?
                    GROUP BY e.id
                    ORDER BY e.published_at ASC
                """
                params = (start_ts, end_ts)

            rows = cur.execute(query, params).fetchall()
            events: list[NewsEvent] = []
            for _event_id, title, summary, url, source, published_ts, sentiment, confidence, currencies_concat in rows:
                currency_list = []
                if currencies_concat:
                    try:
                        currency_list = [c for c in str(currencies_concat).split(",") if c]
                    except Exception:
                        currency_list = []
                published = datetime.fromtimestamp(int(published_ts), tz=UTC)
                events.append(
                    NewsEvent(
                        title=title or "",
                        summary=summary or "",
                        url=url or "",
                        source=source or "",
                        published_at=published,
                        sentiment=float(sentiment),
                        confidence=float(confidence),
                        currencies=currency_list,
                    )
                )
            return events

    def latest_timestamp(self) -> datetime | None:
        with self._get_lock():
            cur = self._conn.cursor()
            row = cur.execute("SELECT MAX(published_at) FROM news_events").fetchone()
            if not row or row[0] is None:
                return None
            return datetime.fromtimestamp(int(row[0]), tz=UTC)

    def _make_key(self, url: str, title: str) -> str:
        raw = (url or "") + "|" + (title or "")
        return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()

    def get_cached_score(self, url: str, title: str) -> dict | None:
        with self._get_lock():
            key = self._make_key(url, title)
            cur = self._conn.cursor()
            row = cur.execute(
                "SELECT sentiment, confidence, direction FROM news_scores_cache WHERE key = ?", (key,)
            ).fetchone()
            if not row:
                return None
            return {"sentiment": float(row[0]), "confidence": float(row[1]), "direction": row[2] or "neutral"}

    def set_cached_score(self, url: str, title: str, sentiment: float, confidence: float, direction: str) -> None:
        with self._get_lock():
            key = self._make_key(url, title)
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO news_scores_cache (key, url, title, sentiment, confidence, direction, created_at)
                VALUES (?, ?, ?, ?, ?, ?, strftime('%s','now'))
                """,
                (key, url or "", title or "", float(sentiment), float(confidence), direction or "neutral"),
            )
            self._conn.commit()

    def close(self) -> None:
        try:
            with self._get_lock():
                self._conn.close()
        except Exception as e:
            logger.warning(f"News store operation failed: {e}", exc_info=True)

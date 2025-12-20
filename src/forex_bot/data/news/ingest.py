"""
Local News Ingestion for Backfilling.

Scans data/news directory for CSV/JSON files and populates the SQLite database.
This allows the bot to have historical context without expensive API calls.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path for imports to work
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forex_bot.core.config import settings  # noqa: E402
from forex_bot.data.news.store import NewsDatabase, NewsEvent  # noqa: E402

logger = logging.getLogger(__name__)


def ingest_local_news(data_dir: Path = Path("data/news")):
    """
    Scan data_dir for news files and ingest them into the database.
    Supported formats: .csv, .json
    Expected columns/fields: title, date/published_at, summary/snippet (optional), sentiment (optional)
    """
    if not data_dir.exists():
        logger.warning(f"News data directory {data_dir} does not exist.")
        return

    db_path = Path(settings.system.cache_dir) / "news.sqlite"
    db = NewsDatabase(db_path)

    files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))
    if not files:
        logger.info(f"No news files found in {data_dir}")
        return

    total_inserted = 0

    for file_path in files:
        logger.info(f"Processing {file_path}...")
        try:
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_json(file_path)

            # Normalize columns
            df.columns = [c.lower() for c in df.columns]

            if df.empty:
                continue

            # Vectorized preprocessing
            df["title_clean"] = df.get("title", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
            df = df[df["title_clean"] != ""]

            if df.empty:
                continue

            # Parse timestamps vectorized
            date_col = df.get("published_at", df.get("date", df.get("time")))
            if date_col is None:
                continue
            df["dt_parsed"] = pd.to_datetime(date_col, utc=True, errors="coerce")
            df = df[df["dt_parsed"].notna()]

            if df.empty:
                continue

            # Vectorized columns
            df["summary_clean"] = df.get("summary", df.get("snippet", pd.Series(dtype=str))).fillna("").astype(str)
            df["url_clean"] = df.get("url", pd.Series(dtype=str)).fillna("").astype(str)
            df["source_clean"] = df.get("source", pd.Series(dtype=str)).fillna("LocalArchive").astype(str)
            df["sentiment_val"] = pd.to_numeric(df.get("sentiment", 0.0), errors="coerce").fillna(0.0)
            df["confidence_val"] = pd.to_numeric(df.get("confidence", 0.0), errors="coerce").fillna(0.0)

            # Parse currencies vectorized
            curr_col = df.get("currencies", df.get("currency", pd.Series(dtype=str))).fillna("").astype(str)
            df["currencies_list"] = (
                curr_col.str.replace(";", ",")
                .str.split(",")
                .apply(lambda x: [c.strip().upper() for c in x if len(c.strip()) == 3])
            )

            # Fallback currency detection
            def extract_currencies(row):
                if row["currencies_list"]:
                    return row["currencies_list"]
                return [
                    c
                    for c in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
                    if c in row["title_clean"].upper()
                ]

            df["currencies_final"] = df.apply(extract_currencies, axis=1)

            # Build events using itertuples (8x faster than iterrows)
            events = []
            for i, row in enumerate(df.itertuples(index=False)):
                url = row.url_clean or f"file://{file_path.name}/{i}"
                events.append(
                    NewsEvent(
                        title=row.title_clean,
                        summary=row.summary_clean,
                        url=url,
                        source=row.source_clean,
                        published_at=row.dt_parsed.to_pydatetime(),
                        sentiment=float(row.sentiment_val),
                        confidence=float(row.confidence_val),
                        currencies=row.currencies_final,
                    )
                )

            if events:
                count = db.insert_events(events)
                total_inserted += count
                logger.info(f"Inserted {count} events from {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")

    logger.info(f"News ingestion complete. Total new events: {total_inserted}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_local_news()

import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging with JSON output capability."""
    level = logging.DEBUG if verbose else logging.INFO

    def _get_int_env(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    class _SilenceThirdPartyFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
            name = getattr(record, "name", "")
            if name.startswith("numba"):
                return False
            return True

    class JSONFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
            log_obj = {
                "timestamp": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                log_obj["exception"] = self.formatException(record.exc_info)
            if hasattr(record, "event_id"):
                log_obj["event_id"] = record.event_id
            if hasattr(record, "context"):
                log_obj["context"] = record.context
            return json.dumps(log_obj)

    formatter: logging.Formatter
    if os.environ.get("LOG_FORMAT") == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    silence_filter = _SilenceThirdPartyFilter()

    handlers = []

    stream_handler = logging.StreamHandler()
    stream_handler.addFilter(silence_filter)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / os.environ.get("LOG_FILE", "forex_bot.log")
    max_bytes = 10 * 1024 * 1024  # 10MB max size (smaller for manageability)
    backup_count = 1  # Keep last 2 files (current + 1 backup)
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.addFilter(silence_filter)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    except Exception as exc:
        logging.getLogger(__name__).warning("File logging disabled: %s", exc)

    logging.root.handlers = []
    for handler in handlers:
        logging.root.addHandler(handler)
    logging.root.setLevel(level)
    try:
        for name in (
            "httpcore",
            "httpx",
            "numba",
            "numba.core",
            "numba.core.byteflow",
            "openai",
            "urllib3",
            "matplotlib",
        ):
            lg = logging.getLogger(name)
            lg.setLevel(logging.WARNING)
            # Keep propagation on for HTTP clients so WARNING/ERROR still surface in our handlers,
            # but silence chatty libraries that add little value in a trading loop.
            if name not in {"httpcore", "httpx"}:
                lg.propagate = False
    except Exception as e:
        logging.getLogger(__name__).warning(f"Log setup failed: {e}", exc_info=True)

    logging.captureWarnings(True)

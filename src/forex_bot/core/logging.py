import json
import logging
import os
import queue
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """HPC Optimized: Non-blocking structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Define a robust formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handlers = []
    
    # 1. Base Stream Handler (Standard Out)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    # 2. Asynchronous File Handler
    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / os.environ.get("LOG_FILE", "forex_bot.log")
    
    try:
        raw_file_handler = RotatingFileHandler(
            log_file, maxBytes=50*1024*1024, backupCount=3, encoding="utf-8"
        )
        raw_file_handler.setFormatter(formatter)
        
        # HPC FIX: Offload I/O to background thread
        log_queue = queue.Queue(-1) # Infinite buffer
        queue_handler = QueueHandler(log_queue)
        listener = QueueListener(log_queue, raw_file_handler)
        listener.start()
        
        handlers.append(queue_handler)
    except Exception as exc:
        logging.warning(f"Async logging setup failed: {exc}")

    logging.root.handlers = []
    for h in handlers:
        logging.root.addHandler(h)
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

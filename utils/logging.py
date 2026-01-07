"""
Structured logging for Senter.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logs."""

    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'message': record.getMessage(),
        }

        if hasattr(record, 'extra'):
            log_data.update(record.extra)

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files (default: data/logs)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        console: Enable console output
        file: Enable file output

    Returns:
        Root logger
    """
    if log_dir is None:
        log_dir = Path("data/logs")

    log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level))

    # Clear existing handlers
    root.handlers = []

    # Console handler (human readable)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ConsoleFormatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        ))
        root.addHandler(console_handler)

    # File handler (JSON, rotating)
    if file:
        file_handler = RotatingFileHandler(
            log_dir / "senter.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

        # Separate error log
        error_handler = RotatingFileHandler(
            log_dir / "errors.log",
            maxBytes=5_000_000,
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root.addHandler(error_handler)

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)

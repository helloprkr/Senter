#!/usr/bin/env python3
"""
Structured JSON Logging with Rotation (DI-005)

Provides JSON-formatted logs with automatic rotation.
"""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Any


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON (DI-005).

    Output format:
    {
        "timestamp": "2026-01-07T12:00:00.000000",
        "level": "INFO",
        "component": "senter.daemon",
        "message": "Log message",
        "context": {...}
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra context if present
        context = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'message', 'thread',
                'threadName', 'taskName'
            }:
                try:
                    # Ensure value is JSON serializable
                    json.dumps(value)
                    context[key] = value
                except (TypeError, ValueError):
                    context[key] = str(value)

        if context:
            log_entry["context"] = context

        return json.dumps(log_entry)


class StructuredLogger:
    """
    Structured JSON logger with rotation support (DI-005).

    Features:
    - JSON formatted log entries
    - Automatic file rotation (10MB, 5 backups)
    - Per-component log levels
    - Context injection
    """

    # Default configuration
    DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
    DEFAULT_BACKUP_COUNT = 5

    def __init__(
        self,
        log_dir: Path = None,
        max_bytes: int = None,
        backup_count: int = None,
        default_level: int = logging.INFO
    ):
        self.log_dir = Path(log_dir) if log_dir else Path("data/logs")
        self.max_bytes = max_bytes or self.DEFAULT_MAX_BYTES
        self.backup_count = backup_count or self.DEFAULT_BACKUP_COUNT
        self.default_level = default_level

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Component-specific levels
        self.component_levels: dict[str, int] = {}

        # Configured handlers
        self._handlers: dict[str, logging.Handler] = {}

    def configure_logging(self, log_file: str = "senter.log"):
        """Configure root logger with JSON formatting and rotation."""
        log_path = self.log_dir / log_file

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count
        )
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(self.default_level)

        # Create console handler for errors
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(JSONFormatter())
        console_handler.setLevel(logging.ERROR)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Add new handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        self._handlers["file"] = file_handler
        self._handlers["console"] = console_handler

        logging.info("Structured JSON logging configured", extra={
            "log_file": str(log_path),
            "max_bytes": self.max_bytes,
            "backup_count": self.backup_count
        })

    def set_component_level(self, component: str, level: int):
        """Set log level for a specific component."""
        self.component_levels[component] = level
        logger = logging.getLogger(component)
        logger.setLevel(level)

    def get_logger(self, component: str) -> logging.Logger:
        """Get a logger for a component."""
        logger = logging.getLogger(component)

        if component in self.component_levels:
            logger.setLevel(self.component_levels[component])

        return logger


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that adds context to all log messages.

    Usage:
        logger = ContextLogger(
            logging.getLogger("mycomponent"),
            {"session_id": "abc123"}
        )
        logger.info("Processing", extra={"item_id": 456})
    """

    def process(self, msg, kwargs):
        # Merge context with extra
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def setup_json_logging(
    log_dir: Path = None,
    log_file: str = "senter.log",
    level: int = logging.INFO
) -> StructuredLogger:
    """
    Set up JSON logging for the application.

    Args:
        log_dir: Directory for log files
        log_file: Name of main log file
        level: Default log level

    Returns:
        Configured StructuredLogger instance
    """
    logger = StructuredLogger(log_dir=log_dir, default_level=level)
    logger.configure_logging(log_file)
    return logger


# CLI for testing
if __name__ == "__main__":
    import tempfile

    # Test with temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)

        # Configure
        struct_logger = setup_json_logging(log_dir, level=logging.DEBUG)

        # Get component logger
        logger = struct_logger.get_logger("senter.test")

        # Test logs
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning", extra={"user_id": 123})
        logger.error("Test error", extra={"error_code": "E001"})

        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.exception("Caught exception")

        # Read log file
        log_file = log_dir / "senter.log"
        print(f"\nLog file contents ({log_file}):")
        print("-" * 40)
        for line in log_file.read_text().strip().split("\n"):
            entry = json.loads(line)
            print(json.dumps(entry, indent=2))
            print()

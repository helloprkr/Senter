#!/usr/bin/env python3
"""
Tests for JSON Logger (DI-005)
Tests structured JSON logging with rotation.
"""

import sys
import json
import logging
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_json_formatter():
    """Test JSONFormatter output format"""
    from daemon.json_logger import JSONFormatter

    formatter = JSONFormatter()

    # Create a log record
    record = logging.LogRecord(
        name="test.component",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test message",
        args=(),
        exc_info=None
    )

    output = formatter.format(record)
    parsed = json.loads(output)

    assert "timestamp" in parsed
    assert parsed["level"] == "INFO"
    assert parsed["component"] == "test.component"
    assert parsed["message"] == "Test message"

    return True


def test_log_json_format():
    """Test that logs are in JSON format"""
    from daemon.json_logger import setup_json_logging

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        struct_logger = setup_json_logging(log_dir, level=logging.DEBUG)

        logger = struct_logger.get_logger("test.format")
        logger.info("Test message")

        # Read log file
        log_file = log_dir / "senter.log"
        assert log_file.exists()

        # Parse last line (each line is a separate JSON object)
        lines = log_file.read_text().strip().split("\n")
        parsed = json.loads(lines[-1])

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["component"] == "test.format"

        return True


def test_log_with_context():
    """Test logging with extra context"""
    from daemon.json_logger import setup_json_logging

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        struct_logger = setup_json_logging(log_dir, level=logging.DEBUG)

        logger = struct_logger.get_logger("test.context")
        logger.info("With context", extra={"user_id": 123, "action": "test"})

        log_file = log_dir / "senter.log"
        lines = log_file.read_text().strip().split("\n")
        parsed = json.loads(lines[-1])

        assert "context" in parsed
        assert parsed["context"]["user_id"] == 123
        assert parsed["context"]["action"] == "test"

        return True


def test_log_rotation_config():
    """Test log rotation configuration"""
    from daemon.json_logger import StructuredLogger

    struct_logger = StructuredLogger(
        max_bytes=5 * 1024 * 1024,  # 5MB
        backup_count=3
    )

    assert struct_logger.max_bytes == 5 * 1024 * 1024
    assert struct_logger.backup_count == 3

    return True


def test_log_rotation():
    """Test that log rotation is configured"""
    from daemon.json_logger import setup_json_logging
    from logging.handlers import RotatingFileHandler

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        struct_logger = setup_json_logging(log_dir, level=logging.DEBUG)

        # Check that rotating handler is configured
        assert "file" in struct_logger._handlers
        handler = struct_logger._handlers["file"]
        assert isinstance(handler, RotatingFileHandler)
        assert handler.maxBytes == struct_logger.max_bytes
        assert handler.backupCount == struct_logger.backup_count

        return True


def test_component_levels():
    """Test per-component log levels"""
    from daemon.json_logger import StructuredLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        struct_logger = StructuredLogger(log_dir=Path(tmpdir))
        struct_logger.configure_logging()

        # Set component level
        struct_logger.set_component_level("senter.verbose", logging.DEBUG)
        struct_logger.set_component_level("senter.quiet", logging.ERROR)

        assert struct_logger.component_levels["senter.verbose"] == logging.DEBUG
        assert struct_logger.component_levels["senter.quiet"] == logging.ERROR

        return True


def test_context_logger():
    """Test ContextLogger adapter"""
    from daemon.json_logger import ContextLogger, setup_json_logging

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        struct_logger = setup_json_logging(log_dir, level=logging.DEBUG)

        base_logger = struct_logger.get_logger("test.context")
        ctx_logger = ContextLogger(base_logger, {"session_id": "abc123"})

        ctx_logger.info("Test with context")

        log_file = log_dir / "senter.log"
        lines = log_file.read_text().strip().split("\n")
        parsed = json.loads(lines[-1])

        assert "context" in parsed
        assert parsed["context"]["session_id"] == "abc123"

        return True


def test_exception_logging():
    """Test exception logging includes traceback"""
    from daemon.json_logger import setup_json_logging

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        struct_logger = setup_json_logging(log_dir, level=logging.DEBUG)

        logger = struct_logger.get_logger("test.exception")

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.exception("Caught error")

        log_file = log_dir / "senter.log"
        lines = log_file.read_text().strip().split("\n")
        parsed = json.loads(lines[-1])

        assert parsed["level"] == "ERROR"
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]

        return True


def test_default_settings():
    """Test default configuration values"""
    from daemon.json_logger import StructuredLogger

    struct_logger = StructuredLogger()

    assert struct_logger.max_bytes == 10 * 1024 * 1024  # 10MB
    assert struct_logger.backup_count == 5

    return True


def test_log_directory_creation():
    """Test that log directory is created"""
    from daemon.json_logger import StructuredLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "subdir" / "logs"
        struct_logger = StructuredLogger(log_dir=log_dir)

        assert log_dir.exists()

        return True


if __name__ == "__main__":
    tests = [
        test_json_formatter,
        test_log_json_format,
        test_log_with_context,
        test_log_rotation_config,
        test_log_rotation,
        test_component_levels,
        test_context_logger,
        test_exception_logging,
        test_default_settings,
        test_log_directory_creation,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")

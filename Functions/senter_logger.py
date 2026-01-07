#!/usr/bin/env python3
"""
Senter Logging System
Quiet console, verbose logs to file
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Global state
_verbose_mode = False
_logger = None


def setup_logging(verbose: bool = False, log_file: str = None):
    """
    Configure Senter logging.

    Args:
        verbose: If True, show all messages on console
        log_file: Path to log file (default: data/senter.log)
    """
    global _verbose_mode, _logger
    _verbose_mode = verbose

    # Ensure data directory exists
    if log_file is None:
        log_path = Path("data/senter.log")
    else:
        log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create logger
    _logger = logging.getLogger("senter")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers = []  # Clear existing handlers

    # File handler - always logs everything
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    _logger.addHandler(file_handler)

    # Console handler - only errors unless verbose
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        _logger.addHandler(console_handler)


def get_logger(name: str = "senter") -> logging.Logger:
    """Get a logger instance"""
    if _logger is None:
        setup_logging()
    return logging.getLogger(name)


def optional_warning(message: str, show_anyway: bool = False):
    """
    Log warning for optional/non-critical features.
    Only shows on console if verbose mode is on.

    Args:
        message: Warning message
        show_anyway: If True, show on console even in quiet mode
    """
    logger = get_logger()
    logger.warning(f"[OPTIONAL] {message}")

    if _verbose_mode or show_anyway:
        print(f"  ⚠️  {message}")


def required_error(message: str):
    """
    Log error for required components.
    Always shows on console.
    """
    logger = get_logger()
    logger.error(f"[REQUIRED] {message}")
    print(f"❌ {message}")


def info(message: str, console: bool = True):
    """
    Log info message.

    Args:
        message: Info message
        console: Whether to show on console (default True)
    """
    logger = get_logger()
    logger.info(message)

    if console:
        print(message)


def debug(message: str):
    """Debug message - file only unless verbose"""
    logger = get_logger()
    logger.debug(message)

    if _verbose_mode:
        print(f"  [DEBUG] {message}")


def success(message: str):
    """Success message - always show"""
    logger = get_logger()
    logger.info(f"[SUCCESS] {message}")
    print(f"✓ {message}")


def is_verbose() -> bool:
    """Check if verbose mode is enabled"""
    return _verbose_mode

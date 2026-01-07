#!/usr/bin/env python3
"""
Circuit Breaker Pattern

Prevents cascade failures by temporarily blocking calls to failing services.
"""

import time
import logging
from typing import Callable, Any
from functools import wraps

from daemon.errors import CircuitOpenError

logger = logging.getLogger('senter.circuit')


class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    States:
    - closed: Normal operation, calls pass through
    - open: Failing, calls blocked
    - half-open: Testing if service recovered
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: int = 60,
        success_threshold: int = 2
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "closed"

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        if self.state == "open":
            if self._should_try_reset():
                self.state = "half-open"
                logger.info(f"Circuit {self.name}: half-open, testing...")
            else:
                raise CircuitOpenError(f"Circuit {self.name} is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_try_reset(self) -> bool:
        """Check if enough time has passed to try resetting"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.reset_timeout

    def _on_success(self):
        """Handle successful call"""
        if self.state == "half-open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit {self.name}: closed (recovered)")
        else:
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = time.time()

        if self.state == "half-open":
            self.state = "open"
            logger.warning(f"Circuit {self.name}: open (still failing)")
        elif self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit {self.name}: open after {self.failure_count} failures")

    def reset(self):
        """Manually reset the circuit"""
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info(f"Circuit {self.name}: manually reset")

    @property
    def is_open(self) -> bool:
        return self.state == "open"

    def get_status(self) -> dict:
        """Get circuit status"""
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time
        }


def with_circuit_breaker(breaker: CircuitBreaker):
    """Decorator to wrap function with circuit breaker"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        exponential: Use exponential backoff
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e

                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt if exponential else 1)
                        logger.warning(
                            f"{func.__name__} attempt {attempt + 1}/{max_retries + 1} "
                            f"failed: {e}, retrying in {delay:.1f}s"
                        )
                        time.sleep(delay)

            logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            raise last_error

        return wrapper
    return decorator

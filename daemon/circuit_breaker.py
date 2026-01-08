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


# DI-003: Component Circuit Breaker Registry
class CircuitBreakerRegistry:
    """
    Registry for managing circuit breakers per component (DI-003).

    Integrates with message bus to prevent routing to failed components.
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 30):
        self.breakers: dict[str, CircuitBreaker] = {}
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

    def get_or_create(self, component: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a component."""
        if component not in self.breakers:
            self.breakers[component] = CircuitBreaker(
                name=component,
                failure_threshold=self.failure_threshold,
                reset_timeout=self.reset_timeout,
                success_threshold=2
            )
        return self.breakers[component]

    def record_failure(self, component: str):
        """Record a failure for a component."""
        breaker = self.get_or_create(component)
        breaker._on_failure()

    def record_success(self, component: str):
        """Record a success for a component."""
        breaker = self.get_or_create(component)
        breaker._on_success()

    def is_component_available(self, component: str) -> bool:
        """Check if a component is available (circuit not open)."""
        if component not in self.breakers:
            return True

        breaker = self.breakers[component]

        # Check for half-open transition
        if breaker.state == "open" and breaker._should_try_reset():
            breaker.state = "half-open"
            logger.info(f"Circuit {component}: half-open, testing...")
            return True

        return breaker.state != "open"

    def get_available_components(self, components: list[str]) -> list[str]:
        """Filter list to only available components."""
        return [c for c in components if self.is_component_available(c)]

    def get_all_status(self) -> dict[str, dict]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self.breakers.items()}

    def get_open_circuits(self) -> list[str]:
        """Get list of components with open circuits."""
        return [name for name, breaker in self.breakers.items() if breaker.is_open]

    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")

    def reset_component(self, component: str):
        """Reset a specific component's circuit breaker."""
        if component in self.breakers:
            self.breakers[component].reset()


# Global registry instance
_registry: CircuitBreakerRegistry = None


def get_circuit_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry


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

#!/usr/bin/env python3
"""
Tests for Circuit Breaker (DI-003)
Tests circuit breaker pattern for component isolation.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_circuit_breaker_init():
    """Test CircuitBreaker initialization"""
    from daemon.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test", failure_threshold=5, reset_timeout=60)

    assert cb.name == "test"
    assert cb.state == "closed"
    assert cb.failure_count == 0
    assert cb.failure_threshold == 5

    return True


def test_circuit_opens_on_failures():
    """Test that circuit opens after threshold failures"""
    from daemon.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test", failure_threshold=3, reset_timeout=1)

    # Record failures
    for i in range(3):
        cb._on_failure()

    # Circuit should be open after 3 failures
    assert cb.state == "open"
    assert cb.is_open

    return True


def test_circuit_stays_closed_on_success():
    """Test that success resets failure count"""
    from daemon.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test", failure_threshold=5)

    # Some failures
    cb._on_failure()
    cb._on_failure()
    assert cb.failure_count == 2

    # Success resets
    cb._on_success()
    assert cb.failure_count == 0
    assert cb.state == "closed"

    return True


def test_circuit_half_open_recovery():
    """Test circuit transitions to half-open and recovers"""
    from daemon.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test", failure_threshold=2, reset_timeout=0.1, success_threshold=2)

    # Open the circuit
    cb._on_failure()
    cb._on_failure()
    assert cb.state == "open"

    # Wait for reset timeout
    time.sleep(0.2)

    # Should transition to half-open on next check
    assert cb._should_try_reset()

    # Simulate successful test calls
    cb.state = "half-open"
    cb._on_success()
    assert cb.state == "half-open"  # Need 2 successes

    cb._on_success()
    assert cb.state == "closed"  # Now closed

    return True


def test_circuit_half_open_failure():
    """Test that failure in half-open reopens circuit"""
    from daemon.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test", failure_threshold=2, reset_timeout=0.1)

    # Open circuit
    cb._on_failure()
    cb._on_failure()
    assert cb.state == "open"

    # Simulate half-open state
    cb.state = "half-open"

    # Failure should reopen
    cb._on_failure()
    assert cb.state == "open"

    return True


def test_circuit_breaker_registry():
    """Test CircuitBreakerRegistry"""
    from daemon.circuit_breaker import CircuitBreakerRegistry

    registry = CircuitBreakerRegistry(failure_threshold=3, reset_timeout=1)

    # Get creates new breaker
    cb = registry.get_or_create("component_a")
    assert cb is not None
    assert cb.name == "component_a"

    # Get returns existing
    cb2 = registry.get_or_create("component_a")
    assert cb is cb2

    return True


def test_registry_record_failure():
    """Test registry failure recording"""
    from daemon.circuit_breaker import CircuitBreakerRegistry

    registry = CircuitBreakerRegistry(failure_threshold=2)

    # Record failures
    registry.record_failure("component_x")
    registry.record_failure("component_x")

    # Circuit should be open
    assert not registry.is_component_available("component_x")

    return True


def test_registry_get_available_components():
    """Test filtering available components"""
    from daemon.circuit_breaker import CircuitBreakerRegistry

    registry = CircuitBreakerRegistry(failure_threshold=2)

    components = ["comp_a", "comp_b", "comp_c"]

    # All available initially
    available = registry.get_available_components(components)
    assert len(available) == 3

    # Fail one component
    registry.record_failure("comp_b")
    registry.record_failure("comp_b")

    # comp_b should be filtered out
    available = registry.get_available_components(components)
    assert len(available) == 2
    assert "comp_b" not in available

    return True


def test_registry_get_open_circuits():
    """Test getting list of open circuits"""
    from daemon.circuit_breaker import CircuitBreakerRegistry

    registry = CircuitBreakerRegistry(failure_threshold=2)

    # No open circuits initially
    assert registry.get_open_circuits() == []

    # Open some circuits
    registry.record_failure("comp_a")
    registry.record_failure("comp_a")
    registry.record_failure("comp_b")
    registry.record_failure("comp_b")

    open_circuits = registry.get_open_circuits()
    assert len(open_circuits) == 2
    assert "comp_a" in open_circuits
    assert "comp_b" in open_circuits

    return True


def test_registry_reset_component():
    """Test resetting a specific component"""
    from daemon.circuit_breaker import CircuitBreakerRegistry

    registry = CircuitBreakerRegistry(failure_threshold=2)

    # Open circuit
    registry.record_failure("comp_x")
    registry.record_failure("comp_x")
    assert not registry.is_component_available("comp_x")

    # Reset
    registry.reset_component("comp_x")
    assert registry.is_component_available("comp_x")

    return True


def test_circuit_call_success():
    """Test calling function through circuit breaker"""
    from daemon.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test")

    def success_func():
        return "success"

    result = cb.call(success_func)
    assert result == "success"
    assert cb.state == "closed"

    return True


def test_circuit_call_failure():
    """Test circuit breaker handles exceptions"""
    from daemon.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker("test", failure_threshold=2)

    def fail_func():
        raise ValueError("test error")

    # First failure
    try:
        cb.call(fail_func)
    except ValueError:
        pass
    assert cb.failure_count == 1

    # Second failure opens circuit
    try:
        cb.call(fail_func)
    except ValueError:
        pass
    assert cb.state == "open"

    return True


if __name__ == "__main__":
    tests = [
        test_circuit_breaker_init,
        test_circuit_opens_on_failures,
        test_circuit_stays_closed_on_success,
        test_circuit_half_open_recovery,
        test_circuit_half_open_failure,
        test_circuit_breaker_registry,
        test_registry_record_failure,
        test_registry_get_available_components,
        test_registry_get_open_circuits,
        test_registry_reset_component,
        test_circuit_call_success,
        test_circuit_call_failure,
    ]

    for test in tests:
        try:
            result = test()
            status = "PASS" if result else "FAIL"
        except Exception as e:
            status = f"FAIL ({e})"
        print(f"{test.__name__}: {status}")

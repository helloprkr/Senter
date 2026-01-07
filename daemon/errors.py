#!/usr/bin/env python3
"""
Senter Error Types

Custom exceptions for better error handling and debugging.
"""


class SenterError(Exception):
    """Base Senter error"""
    pass


class ConfigurationError(SenterError):
    """Configuration is invalid or missing"""
    pass


class ModelError(SenterError):
    """Model loading or inference error"""
    pass


class CommunicationError(SenterError):
    """IPC or message bus error"""
    pass


class ComponentError(SenterError):
    """Component failed to start or crashed"""
    def __init__(self, component: str, message: str):
        self.component = component
        super().__init__(f"{component}: {message}")


class CircuitOpenError(SenterError):
    """Circuit breaker is open"""
    pass


class TimeoutError(SenterError):
    """Operation timed out"""
    pass


class RetryExhaustedError(SenterError):
    """All retry attempts failed"""
    def __init__(self, operation: str, attempts: int, last_error: Exception):
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"{operation} failed after {attempts} attempts: {last_error}")

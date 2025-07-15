"""Stub distributed tracing module."""

from collections.abc import Callable
from functools import wraps
from typing import Any


def trace_operation(
    operation_name: str = "operation", **kwargs: Any
) -> Callable[..., Any]:
    """Stub decorator for tracing operations.

    Args:
        operation_name: Name of the operation to trace
        **kwargs: Additional tracing metadata

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # No-op implementation - just call the original function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def start_span(name: str, **kwargs: Any) -> None:
    """Stub function to start a tracing span.

    Args:
        name: Span name
        **kwargs: Additional span attributes

    Returns:
        None (no-op)
    """
    pass


def end_span() -> None:
    """Stub function to end a tracing span."""
    pass


def add_span_attribute(key: str, value: Any) -> None:
    """Stub function to add attributes to current span.

    Args:
        key: Attribute key
        value: Attribute value
    """
    pass


def set_span_error(error: Exception) -> None:
    """Stub function to mark span as having an error.

    Args:
        error: Exception that occurred
    """
    pass

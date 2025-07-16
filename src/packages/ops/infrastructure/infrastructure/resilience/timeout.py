"""Timeout handling utilities for async and sync operations."""

from __future__ import annotations

import asyncio
import builtins
import signal
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import TypeVar

T = TypeVar("T")


class TimeoutError(Exception):
    """Raised when an operation times out."""

    pass


def timeout_handler(timeout_seconds: float):
    """Decorator for adding timeout to functions.

    Args:
        timeout_seconds: Timeout in seconds

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs), timeout=timeout_seconds
                    )
                except builtins.TimeoutError:
                    raise TimeoutError(
                        f"Function {func.__name__} timed out after {timeout_seconds} seconds"
                    )

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                return _run_with_timeout(func, timeout_seconds, *args, **kwargs)

            return sync_wrapper

    return decorator


def _run_with_timeout(func: Callable[..., T], timeout: float, *args, **kwargs) -> T:
    """Run synchronous function with timeout using signal alarm (Unix only)."""
    import platform

    if platform.system() == "Windows":
        # On Windows, we can't use signal.alarm, so we just run the function
        # This is a limitation - for Windows timeout support, consider using
        # threading or multiprocessing
        return func(*args, **kwargs)

    def timeout_handler_signal(signum, frame):
        raise TimeoutError(
            f"Function {func.__name__} timed out after {timeout} seconds"
        )

    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler_signal)
    signal.alarm(int(timeout))

    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        raise
    finally:
        # Restore old signal handler
        signal.signal(signal.SIGALRM, old_handler)


@asynccontextmanager
async def async_timeout(timeout_seconds: float):
    """Async context manager for timeout handling.

    Args:
        timeout_seconds: Timeout in seconds

    Usage:
        async with async_timeout(5.0):
            await some_async_operation()
    """
    try:
        async with asyncio.timeout(timeout_seconds):
            yield
    except builtins.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")


@contextmanager
def sync_timeout(timeout_seconds: float):
    """Sync context manager for timeout handling.

    Args:
        timeout_seconds: Timeout in seconds

    Usage:
        with sync_timeout(5.0):
            some_sync_operation()
    """
    import platform

    if platform.system() == "Windows":
        # Windows doesn't support signal.alarm, so just yield without timeout
        yield
        return

    def timeout_handler_signal(signum, frame):
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler_signal)
    signal.alarm(int(timeout_seconds))

    try:
        yield
        signal.alarm(0)  # Cancel the alarm
    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        raise
    finally:
        # Restore old signal handler
        signal.signal(signal.SIGALRM, old_handler)


# Convenience timeout decorators for common scenarios
def database_timeout(timeout_seconds: float = 30.0):
    """Timeout decorator optimized for database operations."""
    return timeout_handler(timeout_seconds)


def api_timeout(timeout_seconds: float = 60.0):
    """Timeout decorator optimized for API calls."""
    return timeout_handler(timeout_seconds)


def file_timeout(timeout_seconds: float = 10.0):
    """Timeout decorator optimized for file operations."""
    return timeout_handler(timeout_seconds)


def cache_timeout(timeout_seconds: float = 5.0):
    """Timeout decorator optimized for cache operations."""
    return timeout_handler(timeout_seconds)


class TimeoutManager:
    """Manager for handling multiple timeout scenarios."""

    def __init__(self):
        self.timeouts = {}

    def set_timeout(self, operation: str, timeout_seconds: float):
        """Set timeout for a specific operation type.

        Args:
            operation: Operation name/type
            timeout_seconds: Timeout in seconds
        """
        self.timeouts[operation] = timeout_seconds

    def get_timeout(self, operation: str, default: float = 30.0) -> float:
        """Get timeout for a specific operation type.

        Args:
            operation: Operation name/type
            default: Default timeout if not set

        Returns:
            Timeout in seconds
        """
        return self.timeouts.get(operation, default)

    def timeout_for(self, operation: str, default: float = 30.0):
        """Get timeout decorator for a specific operation type.

        Args:
            operation: Operation name/type
            default: Default timeout if not set

        Returns:
            Timeout decorator
        """
        timeout_seconds = self.get_timeout(operation, default)
        return timeout_handler(timeout_seconds)


# Global timeout manager instance
timeout_manager = TimeoutManager()

# Set default timeouts for common operations
timeout_manager.set_timeout("database", 30.0)
timeout_manager.set_timeout("api", 60.0)
timeout_manager.set_timeout("cache", 5.0)
timeout_manager.set_timeout("file", 10.0)
timeout_manager.set_timeout("ml_training", 300.0)  # 5 minutes
timeout_manager.set_timeout("ml_prediction", 30.0)
timeout_manager.set_timeout("data_loading", 120.0)  # 2 minutes

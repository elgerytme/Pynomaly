"""Retry mechanisms with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    exceptions: type | tuple = Exception

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.base_delay * (self.exponential_base**attempt)

        # Cap at max_delay
        delay = min(delay, self.max_delay)

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay


class RetryExhausted(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempts. "
            f"Last exception: {last_exception}"
        )


def retry_with_backoff(
    policy: RetryPolicy | None = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: type | tuple = Exception,
):
    """Decorator for retrying functions with exponential backoff.

    Args:
        policy: RetryPolicy instance (overrides individual parameters)
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        exceptions: Exception types to retry on

    Returns:
        Decorator function
    """
    if policy is None:
        policy = RetryPolicy(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            exceptions=exceptions,
        )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                return await _retry_async_call(func, policy, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                return _retry_sync_call(func, policy, *args, **kwargs)

            return sync_wrapper

    return decorator


def _retry_sync_call(func: Callable[..., T], policy: RetryPolicy, *args, **kwargs) -> T:
    """Execute synchronous function with retry logic."""
    last_exception = None

    for attempt in range(policy.max_attempts):
        try:
            result = func(*args, **kwargs)

            # Log successful retry
            if attempt > 0:
                logger.info(
                    f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                )

            return result

        except policy.exceptions as e:
            last_exception = e

            # Don't wait after the last attempt
            if attempt < policy.max_attempts - 1:
                delay = policy.calculate_delay(attempt)
                logger.warning(
                    f"Function {func.__name__} failed on attempt {attempt + 1}, "
                    f"retrying in {delay:.2f} seconds: {e}"
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Function {func.__name__} failed on final attempt {attempt + 1}: {e}"
                )

    # All attempts exhausted
    raise RetryExhausted(policy.max_attempts, last_exception)


async def _retry_async_call(
    func: Callable[..., T], policy: RetryPolicy, *args, **kwargs
) -> T:
    """Execute async function with retry logic."""
    last_exception = None

    for attempt in range(policy.max_attempts):
        try:
            result = await func(*args, **kwargs)

            # Log successful retry
            if attempt > 0:
                logger.info(
                    f"Function {func.__name__} succeeded on attempt {attempt + 1}"
                )

            return result

        except policy.exceptions as e:
            last_exception = e

            # Don't wait after the last attempt
            if attempt < policy.max_attempts - 1:
                delay = policy.calculate_delay(attempt)
                logger.warning(
                    f"Function {func.__name__} failed on attempt {attempt + 1}, "
                    f"retrying in {delay:.2f} seconds: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"Function {func.__name__} failed on final attempt {attempt + 1}: {e}"
                )

    # All attempts exhausted
    raise RetryExhausted(policy.max_attempts, last_exception)


# Predefined retry policies for common scenarios
DATABASE_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=0.5,
    max_delay=10.0,
    exponential_base=2.0,
    jitter=True,
)

API_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
)

CACHE_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    base_delay=0.1,
    max_delay=1.0,
    exponential_base=2.0,
    jitter=True,
)

FILE_IO_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=0.2,
    max_delay=5.0,
    exponential_base=2.0,
    jitter=True,
)


# Convenience decorators for common scenarios
def retry_database_operation(
    max_attempts: int = 3, base_delay: float = 0.5, max_delay: float = 10.0
):
    """Decorator for database operations with appropriate retry policy."""
    import sqlalchemy.exc

    try:
        db_exceptions = (
            sqlalchemy.exc.SQLAlchemyError,
            ConnectionError,
            TimeoutError,
        )
    except ImportError:
        db_exceptions = (ConnectionError, TimeoutError)

    return retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exceptions=db_exceptions,
    )


def retry_api_call(
    max_attempts: int = 5, base_delay: float = 1.0, max_delay: float = 30.0
):
    """Decorator for external API calls with appropriate retry policy."""
    try:
        import httpx
        import requests

        api_exceptions = (
            httpx.RequestError,
            httpx.HTTPStatusError,
            requests.RequestException,
            ConnectionError,
            TimeoutError,
        )
    except ImportError:
        api_exceptions = (ConnectionError, TimeoutError)

    return retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exceptions=api_exceptions,
    )


def retry_redis_operation(
    max_attempts: int = 2, base_delay: float = 0.1, max_delay: float = 1.0
):
    """Decorator for Redis operations with appropriate retry policy."""
    try:
        import redis

        redis_exceptions = (
            redis.RedisError,
            redis.ConnectionError,
            redis.TimeoutError,
            ConnectionError,
        )
    except ImportError:
        redis_exceptions = (ConnectionError, TimeoutError)

    return retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exceptions=redis_exceptions,
    )


def retry_file_operation(
    max_attempts: int = 3, base_delay: float = 0.2, max_delay: float = 5.0
):
    """Decorator for file I/O operations with appropriate retry policy."""
    file_exceptions = (
        OSError,
        IOError,
        FileNotFoundError,
        PermissionError,
    )

    return retry_with_backoff(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exceptions=file_exceptions,
    )


# Advanced retry with circuit breaker integration
def retry_with_circuit_breaker(
    circuit_breaker_name: str,
    retry_policy: RetryPolicy | None = None,
    **retry_kwargs,
):
    """Decorator combining retry logic with circuit breaker pattern.

    Args:
        circuit_breaker_name: Name of circuit breaker to use
        retry_policy: RetryPolicy instance
        **retry_kwargs: Retry policy parameters if policy not provided

    Returns:
        Decorator function
    """
    from .circuit_breaker import circuit_breaker_registry

    if retry_policy is None:
        retry_policy = RetryPolicy(**retry_kwargs)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Apply retry decorator first
        retried_func = retry_with_backoff(retry_policy)(func)

        # Get or create circuit breaker
        breaker = circuit_breaker_registry.get(circuit_breaker_name)
        if breaker is None:
            logger.warning(
                f"Circuit breaker '{circuit_breaker_name}' not found. "
                f"Function will use retry logic only."
            )
            return retried_func

        # Apply circuit breaker
        return breaker(retried_func)

    return decorator

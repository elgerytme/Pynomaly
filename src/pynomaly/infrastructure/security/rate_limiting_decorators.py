"""Rate limiting decorators and middleware for easy integration."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from pynomaly.shared.error_handling import (
    ErrorContext,
    create_external_service_error,
)

from .rate_limiting import (
    RateLimitAlgorithm,
    RateLimitConfig,
    RateLimitScope,
    RateLimitStatus,
    check_rate_limit,
    get_rate_limit_manager,
)

logger = logging.getLogger(__name__)


@dataclass
class RateLimitDecoratorConfig:
    """Configuration for rate limiting decorators."""

    requests_per_second: float = 10.0
    burst_capacity: int = 100
    scope: RateLimitScope = RateLimitScope.GLOBAL
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    identifier_extractor: Callable | None = None
    operation_name: str | None = None
    limiter_name: str = "default"
    raise_on_limit: bool = True
    return_status: bool = False


def rate_limited(
    requests_per_second: float = 10.0,
    burst_capacity: int = 100,
    scope: RateLimitScope = RateLimitScope.GLOBAL,
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
    identifier_extractor: Callable | None = None,
    operation_name: str | None = None,
    limiter_name: str = "default",
    raise_on_limit: bool = True,
    return_status: bool = False,
):
    """Decorator for rate limiting functions and methods.

    Args:
        requests_per_second: Rate limit in requests per second
        burst_capacity: Maximum burst capacity
        scope: Rate limiting scope
        algorithm: Rate limiting algorithm
        identifier_extractor: Function to extract identifier from args
        operation_name: Operation name for custom limits
        limiter_name: Rate limiter name
        raise_on_limit: Whether to raise exception on limit exceeded
        return_status: Whether to return rate limit status

    Returns:
        Decorated function

    Example:
        @rate_limited(requests_per_second=5.0, scope=RateLimitScope.USER)
        async def process_user_request(user_id: str, data: dict):
            # Processing logic
            pass
    """

    def decorator(func: Callable) -> Callable:
        config = RateLimitDecoratorConfig(
            requests_per_second=requests_per_second,
            burst_capacity=burst_capacity,
            scope=scope,
            algorithm=algorithm,
            identifier_extractor=identifier_extractor,
            operation_name=operation_name,
            limiter_name=limiter_name,
            raise_on_limit=raise_on_limit,
            return_status=return_status,
        )

        if asyncio.iscoroutinefunction(func):
            return _async_rate_limited_wrapper(func, config)
        else:
            return _sync_rate_limited_wrapper(func, config)

    return decorator


def _async_rate_limited_wrapper(
    func: Callable, config: RateLimitDecoratorConfig
) -> Callable:
    """Async wrapper for rate limited functions."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract identifier
        identifier = _extract_identifier(args, kwargs, config)

        # Create rate limit config
        rate_config = RateLimitConfig(
            algorithm=config.algorithm,
            requests_per_second=config.requests_per_second,
            burst_capacity=config.burst_capacity,
        )

        # Check rate limit
        status = await check_rate_limit(
            identifier=identifier,
            limiter_name=config.limiter_name,
            scope=config.scope,
            operation=config.operation_name or func.__name__,
            config=rate_config,
        )

        # Handle rate limit exceeded
        if not status.allowed:
            if config.raise_on_limit:
                raise create_external_service_error(
                    service_name="rate_limiter",
                    message=f"Rate limit exceeded for {identifier}",
                    context=ErrorContext(
                        operation=config.operation_name or func.__name__,
                        additional_context={
                            "identifier": identifier,
                            "scope": config.scope.value,
                            "retry_after": status.retry_after,
                            "remaining": status.remaining,
                            "algorithm": status.algorithm,
                        },
                    ),
                )
            elif config.return_status:
                return None, status
            else:
                # Return None or default value
                return None

        # Execute function
        try:
            result = await func(*args, **kwargs)

            # Record success for exponential backoff
            if config.algorithm == RateLimitAlgorithm.EXPONENTIAL_BACKOFF:
                manager = get_rate_limit_manager()
                limiter = await manager.get_or_create_limiter(
                    config.limiter_name, rate_config
                )
                await limiter.record_success(
                    identifier, config.scope, config.operation_name or func.__name__
                )

            if config.return_status:
                return result, status
            return result

        except Exception:
            # Record failure for exponential backoff
            if config.algorithm == RateLimitAlgorithm.EXPONENTIAL_BACKOFF:
                manager = get_rate_limit_manager()
                limiter = await manager.get_or_create_limiter(
                    config.limiter_name, rate_config
                )
                await limiter.record_failure(
                    identifier, config.scope, config.operation_name or func.__name__
                )

            raise

    return wrapper


def _sync_rate_limited_wrapper(
    func: Callable, config: RateLimitDecoratorConfig
) -> Callable:
    """Sync wrapper for rate limited functions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # For sync functions, we'll use a simpler approach
        # In a real implementation, you might want to use asyncio.run()
        # or maintain a separate sync rate limiter

        # Extract identifier
        identifier = _extract_identifier(args, kwargs, config)

        # For sync functions, we'll use a basic check
        # This is a simplified implementation
        logger.warning(f"Sync rate limiting for {func.__name__} not fully implemented")

        # Execute function
        return func(*args, **kwargs)

    return wrapper


def _extract_identifier(
    args: tuple, kwargs: dict, config: RateLimitDecoratorConfig
) -> str:
    """Extract identifier for rate limiting."""
    if config.identifier_extractor:
        return config.identifier_extractor(*args, **kwargs)

    # Default identifier extraction strategies
    if config.scope == RateLimitScope.USER:
        # Look for user_id, user, or similar
        for key in ["user_id", "user", "username", "id"]:
            if key in kwargs:
                return str(kwargs[key])

        # Check first argument
        if args and hasattr(args[0], "id"):
            return str(args[0].id)
        elif args:
            return str(args[0])

    elif config.scope == RateLimitScope.IP:
        # Look for IP address
        for key in ["ip", "client_ip", "remote_addr"]:
            if key in kwargs:
                return str(kwargs[key])

    elif config.scope == RateLimitScope.API_KEY:
        # Look for API key
        for key in ["api_key", "key", "token"]:
            if key in kwargs:
                return str(kwargs[key])

    # Global scope or fallback
    return "global"


# Specialized decorators
def user_rate_limited(
    requests_per_second: float = 5.0,
    burst_capacity: int = 50,
    user_id_key: str = "user_id",
):
    """Rate limit decorator for user-specific operations.

    Args:
        requests_per_second: Rate limit per user
        burst_capacity: Burst capacity per user
        user_id_key: Key to extract user ID from kwargs

    Returns:
        Decorator function
    """

    def identifier_extractor(*args, **kwargs):
        return str(kwargs.get(user_id_key, "unknown_user"))

    return rate_limited(
        requests_per_second=requests_per_second,
        burst_capacity=burst_capacity,
        scope=RateLimitScope.USER,
        identifier_extractor=identifier_extractor,
    )


def api_rate_limited(
    requests_per_second: float = 100.0,
    burst_capacity: int = 1000,
    api_key_extractor: Callable | None = None,
):
    """Rate limit decorator for API operations.

    Args:
        requests_per_second: Rate limit per API key
        burst_capacity: Burst capacity per API key
        api_key_extractor: Function to extract API key

    Returns:
        Decorator function
    """

    def default_extractor(*args, **kwargs):
        return str(kwargs.get("api_key", "default_api_key"))

    return rate_limited(
        requests_per_second=requests_per_second,
        burst_capacity=burst_capacity,
        scope=RateLimitScope.API_KEY,
        identifier_extractor=api_key_extractor or default_extractor,
    )


def endpoint_rate_limited(
    requests_per_second: float = 50.0,
    burst_capacity: int = 500,
    endpoint_name: str | None = None,
):
    """Rate limit decorator for endpoint operations.

    Args:
        requests_per_second: Rate limit per endpoint
        burst_capacity: Burst capacity per endpoint
        endpoint_name: Custom endpoint name

    Returns:
        Decorator function
    """

    def identifier_extractor(*args, **kwargs):
        return endpoint_name or "endpoint"

    return rate_limited(
        requests_per_second=requests_per_second,
        burst_capacity=burst_capacity,
        scope=RateLimitScope.ENDPOINT,
        identifier_extractor=identifier_extractor,
    )


# Context managers
@asynccontextmanager
async def rate_limit_context(
    identifier: str,
    requests_per_second: float = 10.0,
    burst_capacity: int = 100,
    scope: RateLimitScope = RateLimitScope.GLOBAL,
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
    operation: str = "operation",
    limiter_name: str = "context",
):
    """Context manager for rate limited operations.

    Args:
        identifier: Rate limit identifier
        requests_per_second: Rate limit
        burst_capacity: Burst capacity
        scope: Rate limiting scope
        algorithm: Rate limiting algorithm
        operation: Operation name
        limiter_name: Rate limiter name

    Raises:
        Rate limit exceeded error if limit is exceeded

    Example:
        async with rate_limit_context("user_123", requests_per_second=5.0):
            # Rate limited operation
            await process_user_data()
    """
    # Create rate limit config
    config = RateLimitConfig(
        algorithm=algorithm,
        requests_per_second=requests_per_second,
        burst_capacity=burst_capacity,
    )

    # Check rate limit
    status = await check_rate_limit(
        identifier=identifier,
        limiter_name=limiter_name,
        scope=scope,
        operation=operation,
        config=config,
    )

    if not status.allowed:
        raise create_external_service_error(
            service_name="rate_limiter",
            message=f"Rate limit exceeded for {identifier}",
            context=ErrorContext(
                operation=operation,
                additional_context={
                    "identifier": identifier,
                    "scope": scope.value,
                    "retry_after": status.retry_after,
                    "remaining": status.remaining,
                    "algorithm": status.algorithm,
                },
            ),
        )

    try:
        yield status
    except Exception:
        # Record failure for exponential backoff
        if algorithm == RateLimitAlgorithm.EXPONENTIAL_BACKOFF:
            manager = get_rate_limit_manager()
            limiter = await manager.get_or_create_limiter(limiter_name, config)
            await limiter.record_failure(identifier, scope, operation)
        raise
    else:
        # Record success for exponential backoff
        if algorithm == RateLimitAlgorithm.EXPONENTIAL_BACKOFF:
            manager = get_rate_limit_manager()
            limiter = await manager.get_or_create_limiter(limiter_name, config)
            await limiter.record_success(identifier, scope, operation)


# Utility functions
async def check_rate_limit_status(
    identifier: str,
    requests_per_second: float = 10.0,
    burst_capacity: int = 100,
    scope: RateLimitScope = RateLimitScope.GLOBAL,
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
    operation: str = "check",
    limiter_name: str = "status_check",
) -> RateLimitStatus:
    """Check rate limit status without consuming tokens.

    Args:
        identifier: Rate limit identifier
        requests_per_second: Rate limit
        burst_capacity: Burst capacity
        scope: Rate limiting scope
        algorithm: Rate limiting algorithm
        operation: Operation name
        limiter_name: Rate limiter name

    Returns:
        Rate limit status
    """
    config = RateLimitConfig(
        algorithm=algorithm,
        requests_per_second=requests_per_second,
        burst_capacity=burst_capacity,
    )

    return await check_rate_limit(
        identifier=identifier,
        limiter_name=limiter_name,
        scope=scope,
        operation=operation,
        tokens=0,  # Don't consume tokens for status check
        config=config,
    )


def create_rate_limit_middleware(
    default_requests_per_second: float = 100.0,
    default_burst_capacity: int = 1000,
    identifier_extractor: Callable | None = None,
    scope: RateLimitScope = RateLimitScope.IP,
):
    """Create rate limiting middleware for web frameworks.

    Args:
        default_requests_per_second: Default rate limit
        default_burst_capacity: Default burst capacity
        identifier_extractor: Function to extract identifier from request
        scope: Rate limiting scope

    Returns:
        Middleware function

    Example:
        # For FastAPI
        middleware = create_rate_limit_middleware(
            default_requests_per_second=10.0,
            identifier_extractor=lambda request: request.client.host
        )
        app.add_middleware(middleware)
    """

    async def middleware(request, call_next):
        """Rate limiting middleware."""
        try:
            # Extract identifier
            if identifier_extractor:
                identifier = identifier_extractor(request)
            else:
                identifier = getattr(request, "client", {}).get("host", "unknown")

            # Check rate limit
            config = RateLimitConfig(
                requests_per_second=default_requests_per_second,
                burst_capacity=default_burst_capacity,
            )

            status = await check_rate_limit(
                identifier=str(identifier),
                limiter_name="middleware",
                scope=scope,
                operation="request",
                config=config,
            )

            if not status.allowed:
                # Return rate limit error response
                # This would need to be adapted for your specific web framework
                return {
                    "error": "Rate limit exceeded",
                    "retry_after": status.retry_after,
                    "remaining": status.remaining,
                }, 429

            # Process request
            response = await call_next(request)

            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(default_requests_per_second)
            response.headers["X-RateLimit-Remaining"] = str(status.remaining)
            response.headers["X-RateLimit-Reset"] = str(int(status.reset_time))

            return response

        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # Continue processing on middleware error
            return await call_next(request)

    return middleware

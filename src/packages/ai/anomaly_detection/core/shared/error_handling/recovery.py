"""Error recovery system with automatic fallback strategies."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

from ..exceptions import APIError, CacheError, DatabaseError, FileError, RecoveryError
from .unified_exceptions import (
    ErrorCodes,
    ErrorContext,
    InfrastructureError,
    PynamolyError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RecoveryStrategy(Enum):
    """Recovery strategy types."""

    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    CACHE = "cache"
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL = "manual"


@dataclass
class RecoveryConfig:
    """Configuration for recovery operations."""

    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    fallback_enabled: bool = True
    circuit_breaker_enabled: bool = False
    recovery_strategies: list[RecoveryStrategy] = field(
        default_factory=lambda: [RecoveryStrategy.RETRY]
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay <= 0:
            raise ValueError("retry_delay must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "fallback_enabled": self.fallback_enabled,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "recovery_strategies": [s.value for s in self.recovery_strategies],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecoveryConfig:
        """Create from dictionary."""
        strategies = data.get("recovery_strategies", ["retry"])
        if (
            isinstance(strategies, list)
            and strategies
            and isinstance(strategies[0], str)
        ):
            strategies = [RecoveryStrategy(s) for s in strategies]

        return cls(
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
            timeout=data.get("timeout", 30.0),
            fallback_enabled=data.get("fallback_enabled", True),
            circuit_breaker_enabled=data.get("circuit_breaker_enabled", False),
            recovery_strategies=strategies,
        )


@dataclass
class RecoveryContext:
    """Context for recovery operations."""

    operation: str
    error: Exception
    attempt: int = 1
    total_attempts: int = 3
    start_time: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def elapsed_time(self) -> timedelta:
        """Get elapsed time since start."""
        return datetime.utcnow() - self.start_time

    def is_timeout(self, timeout_seconds: float) -> bool:
        """Check if operation has timed out."""
        return self.elapsed_time().total_seconds() > timeout_seconds

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata."""
        self.metadata[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "error": str(self.error),
            "attempt": self.attempt,
            "total_attempts": self.total_attempts,
            "start_time": self.start_time.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RecoveryResult:
    """Result of recovery operation."""

    success: bool
    result: Any
    strategy_used: RecoveryStrategy
    attempts: int
    recovery_time: timedelta
    error: Exception | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "strategy_used": self.strategy_used.value,
            "attempts": self.attempts,
            "recovery_time": self.recovery_time.total_seconds(),
            "error": str(self.error) if self.error else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecoveryResult:
        """Create from dictionary."""
        strategy = RecoveryStrategy(data["strategy_used"])
        recovery_time = timedelta(seconds=data["recovery_time"])
        error = Exception(data["error"]) if data.get("error") else None

        return cls(
            success=data["success"],
            result=data["result"],
            strategy_used=strategy,
            attempts=data["attempts"],
            recovery_time=recovery_time,
            error=error,
        )


@dataclass
class RecoveryMetrics:
    """Metrics for recovery operations."""

    total_recoveries: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    total_attempts: int = 0
    average_recovery_time: float = 0.0
    strategy_usage: dict[RecoveryStrategy, int] = field(default_factory=dict)

    def record_recovery(self, result: RecoveryResult) -> None:
        """Record recovery result."""
        self.total_recoveries += 1
        self.total_attempts += result.attempts

        if result.success:
            self.successful_recoveries += 1
        else:
            self.failed_recoveries += 1

        # Update average recovery time
        total_time = self.average_recovery_time * (self.total_recoveries - 1)
        total_time += result.recovery_time.total_seconds()
        self.average_recovery_time = total_time / self.total_recoveries

        # Update strategy usage
        if result.strategy_used not in self.strategy_usage:
            self.strategy_usage[result.strategy_used] = 0
        self.strategy_usage[result.strategy_used] += 1

    def get_success_rate(self) -> float:
        """Get success rate."""
        if self.total_recoveries == 0:
            return 0.0
        return self.successful_recoveries / self.total_recoveries

    def get_average_attempts(self) -> float:
        """Get average attempts per recovery."""
        if self.total_recoveries == 0:
            return 0.0
        return self.total_attempts / self.total_recoveries

    def reset(self) -> None:
        """Reset metrics."""
        self.total_recoveries = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.total_attempts = 0
        self.average_recovery_time = 0.0
        self.strategy_usage.clear()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_recoveries": self.total_recoveries,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "total_attempts": self.total_attempts,
            "average_recovery_time": self.average_recovery_time,
            "success_rate": self.get_success_rate(),
            "average_attempts": self.get_average_attempts(),
            "strategy_usage": {k.value: v for k, v in self.strategy_usage.items()},
        }


class RecoveryStatus(Enum):
    """Recovery attempt status."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class RecoveryAttempt:
    """Recovery attempt information."""

    strategy: RecoveryStrategy
    timestamp: float
    duration: float
    status: RecoveryStatus
    error: Exception | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdvancedRecoveryConfig:
    """Advanced recovery configuration."""

    max_attempts: int = 3
    enabled_strategies: list[RecoveryStrategy] = field(
        default_factory=lambda: [
            RecoveryStrategy.CACHE,
            RecoveryStrategy.FALLBACK,
            RecoveryStrategy.DEGRADE,
        ]
    )
    strategy_timeout: float = 30.0
    cooldown_seconds: float = 60.0
    auto_recovery: bool = True


class RecoveryHandler(ABC):
    """Abstract base class for recovery handlers."""

    def __init__(self, config: RecoveryConfig):
        """Initialize recovery handler."""
        self.config = config
        self.metrics = RecoveryMetrics()

    @abstractmethod
    async def can_handle(self, context: RecoveryContext) -> bool:
        """Check if this handler can handle the error."""
        pass

    @abstractmethod
    async def handle_recovery(self, context: RecoveryContext) -> Any:
        """Attempt to recover from the error."""
        pass

    async def execute_recovery(self, context: RecoveryContext) -> RecoveryResult:
        """Execute recovery with timeout and retry logic."""
        start_time = datetime.utcnow()

        # Check if we can handle this error
        if not await self.can_handle(context):
            error = RecoveryError(f"Cannot handle error: {context.error}")
            return RecoveryResult(
                success=False,
                result=None,
                strategy_used=RecoveryStrategy.RETRY,
                attempts=0,
                recovery_time=datetime.utcnow() - start_time,
                error=error,
            )

        attempts = 0
        last_error = None

        for attempt in range(self.config.max_retries):
            attempts += 1
            context.attempt = attempt + 1

            try:
                # Check timeout
                if context.is_timeout(self.config.timeout):
                    raise TimeoutError(f"Recovery timeout after {self.config.timeout}s")

                # Attempt recovery
                result = await self.handle_recovery(context)

                # Success
                recovery_result = RecoveryResult(
                    success=True,
                    result=result,
                    strategy_used=RecoveryStrategy.RETRY,
                    attempts=attempts,
                    recovery_time=datetime.utcnow() - start_time,
                )

                self.metrics.record_recovery(recovery_result)
                return recovery_result

            except Exception as e:
                last_error = e

                # Wait before retry (except on last attempt)
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)

        # All attempts failed
        recovery_result = RecoveryResult(
            success=False,
            result=None,
            strategy_used=RecoveryStrategy.RETRY,
            attempts=attempts,
            recovery_time=datetime.utcnow() - start_time,
            error=last_error,
        )

        self.metrics.record_recovery(recovery_result)
        return recovery_result

    @abstractmethod
    def get_strategy(self) -> RecoveryStrategy:
        """Get the recovery strategy type."""
        pass


class CacheRecoveryHandler(RecoveryHandler):
    """Recovery handler that uses cached data."""

    def __init__(self, config: RecoveryConfig, fallback_cache: Any | None = None):
        """Initialize cache recovery handler.

        Args:
            config: Recovery configuration
            fallback_cache: Fallback cache instance
        """
        super().__init__(config)
        self.fallback_cache = fallback_cache

    async def can_handle(self, context: RecoveryContext) -> bool:
        """Check if we can recover using cache."""
        return isinstance(context.error, CacheError)

    async def handle_recovery(self, context: RecoveryContext) -> Any:
        """Attempt to recover using cached data."""
        if not self.fallback_cache:
            raise RecoveryError("No fallback cache available")

        operation = context.operation

        if operation == "cache_get":
            key = context.metadata.get("key")
            if not key:
                raise RecoveryError("No cache key provided")
            return await self.fallback_cache.get(key)

        elif operation == "cache_set":
            key = context.metadata.get("key")
            value = context.metadata.get("value")
            ttl = context.metadata.get("ttl")
            if not key or value is None:
                raise RecoveryError("Missing cache key or value")
            return await self.fallback_cache.set(key, value, ttl=ttl)

        elif operation == "cache_delete":
            key = context.metadata.get("key")
            if not key:
                raise RecoveryError("No cache key provided")
            return await self.fallback_cache.delete(key)

        else:
            raise RecoveryError(f"Unsupported cache operation: {operation}")

    def get_strategy(self) -> RecoveryStrategy:
        """Get the recovery strategy type."""
        return RecoveryStrategy.CACHE


class DatabaseRecoveryHandler(RecoveryHandler):
    """Recovery handler for database operations."""

    def __init__(self, config: RecoveryConfig, fallback_db: Any | None = None):
        """Initialize database recovery handler."""
        super().__init__(config)
        self.fallback_db = fallback_db

    async def can_handle(self, context: RecoveryContext) -> bool:
        """Check if we can handle this error."""
        return isinstance(context.error, DatabaseError)

    async def handle_recovery(self, context: RecoveryContext) -> Any:
        """Attempt to recover using fallback database."""
        if not self.fallback_db:
            raise RecoveryError("No fallback database available")

        operation = context.operation

        if operation == "db_query":
            query = context.metadata.get("query")
            params = context.metadata.get("params", {})
            if not query:
                raise RecoveryError("No query provided")
            return await self.fallback_db.execute(query, params)

        elif operation == "db_connect":
            return await self.fallback_db.reconnect()

        else:
            raise RecoveryError(f"Unsupported database operation: {operation}")

    def get_strategy(self) -> RecoveryStrategy:
        """Get the recovery strategy type."""
        return RecoveryStrategy.FALLBACK


class APIRecoveryHandler(RecoveryHandler):
    """Recovery handler for API operations."""

    def __init__(self, config: RecoveryConfig, fallback_endpoints: list[str] = None):
        """Initialize API recovery handler."""
        super().__init__(config)
        self.fallback_endpoints = fallback_endpoints or []

    async def can_handle(self, context: RecoveryContext) -> bool:
        """Check if we can handle this error."""
        return isinstance(context.error, APIError)

    async def handle_recovery(self, context: RecoveryContext) -> Any:
        """Attempt to recover using fallback endpoints."""
        if not self.fallback_endpoints:
            raise RecoveryError("No fallback endpoints available")

        import aiohttp

        url = context.metadata.get("url", "")
        method = context.metadata.get("method", "GET")
        headers = context.metadata.get("headers", {})

        # Extract endpoint path from original URL
        endpoint_path = url.split("/")[-1] if "/" in url else ""

        # Try each fallback endpoint
        for fallback_url in self.fallback_endpoints:
            try:
                full_url = f"{fallback_url.rstrip('/')}/{endpoint_path}"

                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method, full_url, headers=headers
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            continue
            except Exception:
                continue

        raise RecoveryError("All fallback endpoints failed")

    def get_strategy(self) -> RecoveryStrategy:
        """Get the recovery strategy type."""
        return RecoveryStrategy.FALLBACK


class FileRecoveryHandler(RecoveryHandler):
    """Recovery handler for file operations."""

    def __init__(self, config: RecoveryConfig, backup_locations: list[str] = None):
        """Initialize file recovery handler."""
        super().__init__(config)
        self.backup_locations = backup_locations or []

    async def can_handle(self, context: RecoveryContext) -> bool:
        """Check if we can handle this error."""
        return isinstance(context.error, FileError)

    async def handle_recovery(self, context: RecoveryContext) -> Any:
        """Attempt to recover using backup locations."""
        if not self.backup_locations:
            raise RecoveryError("No backup locations available")

        operation = context.operation
        file_path = context.metadata.get("file_path")

        if not file_path:
            raise RecoveryError("No file path provided")

        # Extract filename from original path
        import os

        filename = os.path.basename(file_path)

        if operation == "file_read":
            # Try each backup location
            for backup_location in self.backup_locations:
                backup_path = os.path.join(backup_location, filename)
                try:
                    with open(backup_path) as f:
                        return f.read()
                except Exception:
                    continue
            raise RecoveryError("File not found in any backup location")

        elif operation == "file_write":
            content = context.metadata.get("content")
            if content is None:
                raise RecoveryError("No content provided for write operation")

            # Try to write to first available backup location
            for backup_location in self.backup_locations:
                backup_path = os.path.join(backup_location, filename)
                try:
                    with open(backup_path, "w") as f:
                        f.write(content)
                    return True
                except Exception:
                    continue
            raise RecoveryError("Could not write to any backup location")

        else:
            raise RecoveryError(f"Unsupported file operation: {operation}")

    def get_strategy(self) -> RecoveryStrategy:
        """Get the recovery strategy type."""
        return RecoveryStrategy.FALLBACK


class FallbackRecoveryHandler(RecoveryHandler):
    """Recovery handler that uses fallback functions."""

    def __init__(self, config: RecoveryConfig, fallback_func: Callable):
        """Initialize fallback recovery handler.

        Args:
            config: Recovery configuration
            fallback_func: Fallback function to call
        """
        super().__init__(config)
        self.fallback_func = fallback_func

    async def can_handle(self, context: RecoveryContext) -> bool:
        """Check if we can handle any error (fallback is universal)."""
        return True

    async def handle_recovery(self, context: RecoveryContext) -> Any:
        """Attempt to recover using fallback function."""
        import inspect

        # Check if fallback function accepts context parameter
        sig = inspect.signature(self.fallback_func)

        if asyncio.iscoroutinefunction(self.fallback_func):
            if "context" in sig.parameters:
                return await self.fallback_func(context)
            else:
                return await self.fallback_func()
        else:
            if "context" in sig.parameters:
                return self.fallback_func(context)
            else:
                return self.fallback_func()

    def get_strategy(self) -> RecoveryStrategy:
        """Get the recovery strategy type."""
        return RecoveryStrategy.FALLBACK


class DegradeRecoveryHandler(RecoveryHandler):
    """Recovery handler that provides degraded functionality."""

    def __init__(self, degraded_implementations: dict[str, Callable] = None):
        """Initialize degrade recovery handler.

        Args:
            degraded_implementations: Dict of degraded implementations by operation
        """
        self.degraded_implementations = degraded_implementations or {}

    async def can_handle(self, error: PynamolyError) -> bool:
        """Check if we have a degraded implementation."""
        operation = error.details.context.operation
        return operation in self.degraded_implementations

    async def recover(self, error: PynamolyError, context: dict[str, Any]) -> Any:
        """Attempt to recover with degraded functionality."""
        operation = error.details.context.operation
        degraded_func = self.degraded_implementations.get(operation)

        if not degraded_func:
            raise InfrastructureError(
                error_code=ErrorCodes.INF_CACHE_UNAVAILABLE,
                message=f"No degraded implementation for operation: {operation}",
                context=error.details.context,
            )

        try:
            logger.warning(f"Using degraded implementation for operation: {operation}")
            return await degraded_func(**context.get("args", {}))
        except Exception as e:
            raise InfrastructureError(
                error_code=ErrorCodes.INF_CACHE_UNAVAILABLE,
                message=f"Degraded implementation failed: {str(e)}",
                cause=e,
                context=error.details.context,
            )

    def get_strategy(self) -> RecoveryStrategy:
        """Get the recovery strategy type."""
        return RecoveryStrategy.DEGRADE


class DefaultRecoveryHandler(RecoveryHandler):
    """Default recovery handler that returns safe default values."""

    def __init__(self, default_values: dict[str, Any] = None):
        """Initialize default recovery handler.

        Args:
            default_values: Dict of default values by operation
        """
        self.default_values = default_values or {}

    async def can_handle(self, error: PynamolyError) -> bool:
        """Check if we have a default value."""
        operation = error.details.context.operation
        return operation in self.default_values

    async def recover(self, error: PynamolyError, context: dict[str, Any]) -> Any:
        """Return default value."""
        operation = error.details.context.operation
        default_value = self.default_values.get(operation)

        logger.warning(f"Using default value for operation: {operation}")
        return default_value

    def get_strategy(self) -> RecoveryStrategy:
        """Get the recovery strategy type."""
        return RecoveryStrategy.FALLBACK


class RecoveryManager:
    """Manager for error recovery strategies."""

    def __init__(self, config: RecoveryConfig = None):
        """Initialize recovery manager.

        Args:
            config: Recovery configuration
        """
        self.config = config or RecoveryConfig()
        self.handlers: list[RecoveryHandler] = []
        self.metrics = RecoveryMetrics()

    def add_handler(self, handler: RecoveryHandler) -> None:
        """Add recovery handler."""
        self.handlers.append(handler)
        logger.info(f"Added recovery handler: {handler.get_strategy().value}")

    def remove_handler(self, handler: RecoveryHandler) -> bool:
        """Remove recovery handler."""
        try:
            self.handlers.remove(handler)
            logger.info(f"Removed recovery handler: {handler.get_strategy().value}")
            return True
        except ValueError:
            return False

    async def recover(self, context: RecoveryContext) -> RecoveryResult:
        """Attempt to recover from error using available strategies."""
        # Try each handler
        for handler in self.handlers:
            try:
                # Check if handler can handle this error
                if await handler.can_handle(context):
                    result = await handler.execute_recovery(context)

                    # Record metrics
                    self.metrics.record_recovery(result)
                    return result

            except Exception as e:
                logger.warning(f"Handler {handler.__class__.__name__} failed: {e}")
                continue

        # No suitable handler found
        error = RecoveryError("No suitable recovery handler found")
        result = RecoveryResult(
            success=False,
            result=None,
            strategy_used=RecoveryStrategy.MANUAL,
            attempts=0,
            recovery_time=timedelta(seconds=0),
            error=error,
        )

        self.metrics.record_recovery(result)
        return result

    def get_metrics(self) -> RecoveryMetrics:
        """Get recovery metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset recovery metrics."""
        self.metrics.reset()


# Global recovery manager
_recovery_manager: RecoveryManager | None = None


# Recovery decorators
def recovery_handler(config: RecoveryConfig):
    """Decorator for automatic recovery with retry logic."""

    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            manager = RecoveryManager(config)

            for attempt in range(config.max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    if attempt == config.max_retries - 1:
                        raise
                    await asyncio.sleep(config.retry_delay)

        return wrapper

    return decorator


def fallback_on_error(fallback_func: Callable):
    """Decorator for fallback on error."""

    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception:
                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func()
                else:
                    return fallback_func()

        return wrapper

    return decorator


def circuit_breaker_recovery(
    failure_threshold: int = 5, recovery_timeout: float = 60.0
):
    """Decorator for circuit breaker pattern."""

    def decorator(func):
        import functools

        state = {"failures": 0, "last_failure": None, "state": "closed"}

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()

            # Check if circuit should be half-open
            if (
                state["state"] == "open"
                and state["last_failure"]
                and current_time - state["last_failure"] > recovery_timeout
            ):
                state["state"] = "half-open"

            # If circuit is open, fail fast
            if state["state"] == "open":
                raise Exception("Circuit breaker is open")

            try:
                result = await func(*args, **kwargs)

                # Success - reset failures if half-open or closed
                if state["state"] in ["half-open", "closed"]:
                    state["failures"] = 0
                    state["state"] = "closed"

                return result

            except Exception:
                state["failures"] += 1
                state["last_failure"] = current_time

                # Open circuit if threshold reached
                if state["failures"] >= failure_threshold:
                    state["state"] = "open"

                raise

        return wrapper

    return decorator


def get_recovery_manager(config: RecoveryConfig = None) -> RecoveryManager:
    """Get or create global recovery manager."""
    global _recovery_manager

    if _recovery_manager is None:
        _recovery_manager = RecoveryManager(config)

    return _recovery_manager


async def attempt_recovery(
    error: PynamolyError,
    context: dict[str, Any] = None,
) -> Any:
    """Attempt to recover from error using global recovery manager."""
    manager = get_recovery_manager()
    recovery_context = RecoveryContext(
        operation=context.get("operation", "unknown") if context else "unknown",
        error=error,
    )
    return await manager.recover(recovery_context)


@asynccontextmanager
async def recovery_context(
    operation: str,
    context: dict[str, Any] = None,
    recovery_config: RecoveryConfig = None,
):
    """Context manager for automatic error recovery."""
    context = context or {}
    context["operation"] = operation

    try:
        yield context
    except PynamolyError as e:
        # Set operation in error context
        e.details.context.operation = operation

        # Attempt recovery
        try:
            result = await attempt_recovery(e, context)
            yield result
        except Exception:
            # Recovery failed, re-raise original error
            raise e
    except Exception as e:
        # Convert to PynamolyError and attempt recovery
        pynomaly_error = InfrastructureError(
            error_code=ErrorCodes.INF_CACHE_UNAVAILABLE,
            message=f"Unexpected error in operation '{operation}': {str(e)}",
            cause=e,
            context=ErrorContext(operation=operation),
        )

        try:
            result = await attempt_recovery(pynomaly_error, context)
            yield result
        except Exception:
            # Recovery failed, re-raise original error
            raise e


def recovery_decorator(
    operation: str,
    context: dict[str, Any] = None,
    recovery_config: RecoveryConfig = None,
):
    """Decorator for automatic error recovery."""

    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with recovery_context(operation, context, recovery_config) as ctx:
                return await func(*args, **kwargs)

        return wrapper

    return decorator

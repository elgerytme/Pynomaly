"""Error recovery system with automatic fallback strategies."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from .monitoring import track_error
from .unified_exceptions import (
    ErrorCategory,
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
class RecoveryConfig:
    """Recovery configuration."""

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

    @abstractmethod
    async def can_handle(self, error: PynamolyError) -> bool:
        """Check if this handler can handle the error."""
        pass

    @abstractmethod
    async def recover(self, error: PynamolyError, context: dict[str, Any]) -> Any:
        """Attempt to recover from the error."""
        pass

    @abstractmethod
    def get_strategy(self) -> RecoveryStrategy:
        """Get the recovery strategy type."""
        pass


class CacheRecoveryHandler(RecoveryHandler):
    """Recovery handler that uses cached data."""

    def __init__(self, cache_manager: Any | None = None):
        """Initialize cache recovery handler.

        Args:
            cache_manager: Cache manager instance
        """
        self.cache_manager = cache_manager

    async def can_handle(self, error: PynamolyError) -> bool:
        """Check if we can recover using cache."""
        return self.cache_manager is not None and error.details.category in [
            ErrorCategory.EXTERNAL_SERVICE,
            ErrorCategory.NETWORK,
            ErrorCategory.TIMEOUT,
        ]

    async def recover(self, error: PynamolyError, context: dict[str, Any]) -> Any:
        """Attempt to recover using cached data."""
        cache_key = context.get("cache_key")
        if not cache_key:
            raise InfrastructureError(
                error_code=ErrorCodes.INF_CACHE_UNAVAILABLE,
                message="No cache key provided for recovery",
                context=error.details.context,
            )

        try:
            # Try to get data from cache
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data is not None:
                logger.info(f"Recovered from cache with key: {cache_key}")
                return cached_data
            else:
                raise InfrastructureError(
                    error_code=ErrorCodes.INF_CACHE_UNAVAILABLE,
                    message=f"No cached data found for key: {cache_key}",
                    context=error.details.context,
                )
        except Exception as e:
            raise InfrastructureError(
                error_code=ErrorCodes.INF_CACHE_UNAVAILABLE,
                message=f"Cache recovery failed: {str(e)}",
                cause=e,
                context=error.details.context,
            )

    def get_strategy(self) -> RecoveryStrategy:
        """Get the recovery strategy type."""
        return RecoveryStrategy.CACHE


class FallbackRecoveryHandler(RecoveryHandler):
    """Recovery handler that uses fallback implementations."""

    def __init__(self, fallback_implementations: dict[str, Callable] = None):
        """Initialize fallback recovery handler.

        Args:
            fallback_implementations: Dict of fallback implementations by operation
        """
        self.fallback_implementations = fallback_implementations or {}

    async def can_handle(self, error: PynamolyError) -> bool:
        """Check if we have a fallback implementation."""
        operation = error.details.context.operation
        return operation in self.fallback_implementations

    async def recover(self, error: PynamolyError, context: dict[str, Any]) -> Any:
        """Attempt to recover using fallback implementation."""
        operation = error.details.context.operation
        fallback_func = self.fallback_implementations.get(operation)

        if not fallback_func:
            raise InfrastructureError(
                error_code=ErrorCodes.INF_CACHE_UNAVAILABLE,
                message=f"No fallback implementation for operation: {operation}",
                context=error.details.context,
            )

        try:
            logger.info(f"Using fallback implementation for operation: {operation}")
            return await fallback_func(**context.get("args", {}))
        except Exception as e:
            raise InfrastructureError(
                error_code=ErrorCodes.INF_CACHE_UNAVAILABLE,
                message=f"Fallback implementation failed: {str(e)}",
                cause=e,
                context=error.details.context,
            )

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
        self.recovery_attempts: dict[str, list[RecoveryAttempt]] = {}
        self.recovery_stats = {
            "total_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "recovery_by_strategy": {},
        }

    def add_handler(self, handler: RecoveryHandler) -> None:
        """Add recovery handler."""
        self.handlers.append(handler)
        logger.info(f"Added recovery handler: {handler.get_strategy().value}")

    def remove_handler(self, strategy: RecoveryStrategy) -> bool:
        """Remove recovery handler by strategy."""
        for i, handler in enumerate(self.handlers):
            if handler.get_strategy() == strategy:
                self.handlers.pop(i)
                logger.info(f"Removed recovery handler: {strategy.value}")
                return True
        return False

    async def attempt_recovery(
        self,
        error: PynamolyError,
        context: dict[str, Any] = None,
    ) -> Any:
        """Attempt to recover from error using available strategies.

        Args:
            error: Error to recover from
            context: Recovery context

        Returns:
            Recovered result or raises error if all strategies fail
        """
        context = context or {}
        error_id = error.details.context.error_id

        # Track error
        track_error(error)

        # Check if recovery is enabled
        if not self.config.auto_recovery:
            logger.info("Auto-recovery disabled, not attempting recovery")
            raise error

        # Check cooldown
        if not self._check_cooldown(error_id):
            logger.info("Recovery cooldown active, not attempting recovery")
            raise error

        # Initialize recovery attempts for this error
        if error_id not in self.recovery_attempts:
            self.recovery_attempts[error_id] = []

        # Check max attempts
        if len(self.recovery_attempts[error_id]) >= self.config.max_attempts:
            logger.error(f"Maximum recovery attempts reached for error: {error_id}")
            raise error

        # Try each recovery strategy
        for handler in self.handlers:
            if handler.get_strategy() not in self.config.enabled_strategies:
                continue

            try:
                # Check if handler can handle this error
                if not await handler.can_handle(error):
                    continue

                # Attempt recovery
                start_time = time.time()

                recovery_result = await asyncio.wait_for(
                    handler.recover(error, context),
                    timeout=self.config.strategy_timeout,
                )

                # Record successful recovery
                attempt = RecoveryAttempt(
                    strategy=handler.get_strategy(),
                    timestamp=start_time,
                    duration=time.time() - start_time,
                    status=RecoveryStatus.SUCCESS,
                    details={"handler": handler.__class__.__name__},
                )

                self.recovery_attempts[error_id].append(attempt)
                self._update_stats(handler.get_strategy(), True)

                logger.info(
                    f"Recovery successful using {handler.get_strategy().value}",
                    extra={
                        "error_id": error_id,
                        "strategy": handler.get_strategy().value,
                        "duration": attempt.duration,
                    },
                )

                return recovery_result

            except TimeoutError:
                # Recovery timed out
                attempt = RecoveryAttempt(
                    strategy=handler.get_strategy(),
                    timestamp=start_time,
                    duration=self.config.strategy_timeout,
                    status=RecoveryStatus.FAILED,
                    error=TimeoutError("Recovery timeout"),
                    details={"handler": handler.__class__.__name__},
                )

                self.recovery_attempts[error_id].append(attempt)
                self._update_stats(handler.get_strategy(), False)

                logger.warning(
                    f"Recovery timed out using {handler.get_strategy().value}",
                    extra={
                        "error_id": error_id,
                        "strategy": handler.get_strategy().value,
                        "timeout": self.config.strategy_timeout,
                    },
                )

            except Exception as recovery_error:
                # Recovery failed
                attempt = RecoveryAttempt(
                    strategy=handler.get_strategy(),
                    timestamp=start_time,
                    duration=time.time() - start_time,
                    status=RecoveryStatus.FAILED,
                    error=recovery_error,
                    details={"handler": handler.__class__.__name__},
                )

                self.recovery_attempts[error_id].append(attempt)
                self._update_stats(handler.get_strategy(), False)

                logger.warning(
                    f"Recovery failed using {handler.get_strategy().value}: {str(recovery_error)}",
                    extra={
                        "error_id": error_id,
                        "strategy": handler.get_strategy().value,
                        "error": str(recovery_error),
                    },
                )

        # All recovery strategies failed
        logger.error(f"All recovery strategies failed for error: {error_id}")
        raise error

    def _check_cooldown(self, error_id: str) -> bool:
        """Check if recovery cooldown has passed."""
        if error_id not in self.recovery_attempts:
            return True

        attempts = self.recovery_attempts[error_id]
        if not attempts:
            return True

        last_attempt = attempts[-1]
        current_time = time.time()

        return (current_time - last_attempt.timestamp) >= self.config.cooldown_seconds

    def _update_stats(self, strategy: RecoveryStrategy, success: bool) -> None:
        """Update recovery statistics."""
        self.recovery_stats["total_attempts"] += 1

        if success:
            self.recovery_stats["successful_recoveries"] += 1
        else:
            self.recovery_stats["failed_recoveries"] += 1

        strategy_key = strategy.value
        if strategy_key not in self.recovery_stats["recovery_by_strategy"]:
            self.recovery_stats["recovery_by_strategy"][strategy_key] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
            }

        self.recovery_stats["recovery_by_strategy"][strategy_key]["attempts"] += 1

        if success:
            self.recovery_stats["recovery_by_strategy"][strategy_key]["successes"] += 1
        else:
            self.recovery_stats["recovery_by_strategy"][strategy_key]["failures"] += 1

    def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery statistics."""
        success_rate = 0.0
        if self.recovery_stats["total_attempts"] > 0:
            success_rate = (
                self.recovery_stats["successful_recoveries"]
                / self.recovery_stats["total_attempts"]
            )

        return {
            "total_attempts": self.recovery_stats["total_attempts"],
            "successful_recoveries": self.recovery_stats["successful_recoveries"],
            "failed_recoveries": self.recovery_stats["failed_recoveries"],
            "success_rate": success_rate,
            "recovery_by_strategy": self.recovery_stats["recovery_by_strategy"],
            "active_handlers": len(self.handlers),
            "enabled_strategies": [s.value for s in self.config.enabled_strategies],
            "config": {
                "max_attempts": self.config.max_attempts,
                "strategy_timeout": self.config.strategy_timeout,
                "cooldown_seconds": self.config.cooldown_seconds,
                "auto_recovery": self.config.auto_recovery,
            },
        }

    def get_error_recovery_history(self, error_id: str) -> list[RecoveryAttempt]:
        """Get recovery history for specific error."""
        return self.recovery_attempts.get(error_id, [])

    def clear_recovery_history(self, error_id: str | None = None) -> None:
        """Clear recovery history."""
        if error_id:
            self.recovery_attempts.pop(error_id, None)
        else:
            self.recovery_attempts.clear()
        logger.info("Recovery history cleared")


# Global recovery manager
_recovery_manager: RecoveryManager | None = None


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
    return await manager.attempt_recovery(error, context)


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

"""Comprehensive resilience service combining all patterns."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    database_circuit_breaker,
    external_api_circuit_breaker,
    redis_circuit_breaker,
)
from .retry import (
    RetryPolicy,
    retry_api_call,
    retry_database_operation,
    retry_file_operation,
    retry_redis_operation,
    retry_with_backoff,
)
from .timeout import (
    TimeoutManager,
    timeout_handler,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ResilienceService:
    """Service providing comprehensive resilience patterns."""

    def __init__(
        self,
        circuit_breaker_registry: CircuitBreakerRegistry | None = None,
        timeout_manager: TimeoutManager | None = None,
    ):
        """Initialize resilience service.

        Args:
            circuit_breaker_registry: Circuit breaker registry instance
            timeout_manager: Timeout manager instance
        """
        self.circuit_breaker_registry = (
            circuit_breaker_registry or CircuitBreakerRegistry()
        )
        self.timeout_manager = timeout_manager or TimeoutManager()

        # Initialize common circuit breakers
        self._initialize_common_circuit_breakers()

    def _initialize_common_circuit_breakers(self):
        """Initialize commonly used circuit breakers."""
        # Only create if they don't already exist
        if not self.circuit_breaker_registry.get("database"):
            database_circuit_breaker("database")

        if not self.circuit_breaker_registry.get("external_api"):
            external_api_circuit_breaker("external_api")

        if not self.circuit_breaker_registry.get("redis"):
            redis_circuit_breaker("redis")

    def create_resilient_wrapper(
        self,
        operation_name: str,
        circuit_breaker_config: dict | None = None,
        retry_config: dict | None = None,
        timeout_seconds: float | None = None,
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True,
        enable_timeout: bool = True,
    ):
        """Create a resilient wrapper for functions with all patterns.

        Args:
            operation_name: Name for the operation (used for circuit breaker)
            circuit_breaker_config: Circuit breaker configuration
            retry_config: Retry configuration
            timeout_seconds: Timeout in seconds
            enable_circuit_breaker: Whether to enable circuit breaker
            enable_retry: Whether to enable retry
            enable_timeout: Whether to enable timeout

        Returns:
            Decorator function
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            wrapped_func = func

            # Apply timeout (innermost layer)
            if enable_timeout:
                if timeout_seconds is not None:
                    wrapped_func = timeout_handler(timeout_seconds)(wrapped_func)
                else:
                    # Use default timeout for operation type
                    default_timeout = self.timeout_manager.get_timeout(operation_name)
                    wrapped_func = timeout_handler(default_timeout)(wrapped_func)

            # Apply retry (middle layer)
            if enable_retry:
                if retry_config:
                    retry_policy = RetryPolicy(**retry_config)
                    wrapped_func = retry_with_backoff(retry_policy)(wrapped_func)
                else:
                    # Use default retry based on operation type
                    wrapped_func = self._get_default_retry_decorator(operation_name)(
                        wrapped_func
                    )

            # Apply circuit breaker (outermost layer)
            if enable_circuit_breaker:
                breaker = self._get_or_create_circuit_breaker(
                    operation_name, circuit_breaker_config
                )
                wrapped_func = breaker(wrapped_func)

            return wrapped_func

        return decorator

    def _get_or_create_circuit_breaker(
        self, operation_name: str, config: dict | None = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        breaker = self.circuit_breaker_registry.get(operation_name)

        if breaker is None:
            # Create with provided config or defaults
            config = config or {}
            breaker = self.circuit_breaker_registry.create_and_register(
                name=operation_name,
                failure_threshold=config.get("failure_threshold", 5),
                recovery_timeout=config.get("recovery_timeout", 60.0),
                expected_exception=config.get("expected_exception", Exception),
            )

        return breaker

    def _get_default_retry_decorator(self, operation_name: str):
        """Get default retry decorator based on operation type."""
        if "database" in operation_name.lower():
            return retry_database_operation()
        elif "api" in operation_name.lower() or "http" in operation_name.lower():
            return retry_api_call()
        elif "redis" in operation_name.lower() or "cache" in operation_name.lower():
            return retry_redis_operation()
        elif "file" in operation_name.lower() or "io" in operation_name.lower():
            return retry_file_operation()
        else:
            # Default retry policy
            return retry_with_backoff()

    def database_operation(
        self,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ):
        """Decorator for database operations with optimized resilience patterns.

        Args:
            timeout_seconds: Operation timeout
            max_attempts: Maximum retry attempts
            failure_threshold: Circuit breaker failure threshold
            recovery_timeout: Circuit breaker recovery timeout

        Returns:
            Decorator function
        """
        return self.create_resilient_wrapper(
            operation_name="database",
            circuit_breaker_config={
                "failure_threshold": failure_threshold,
                "recovery_timeout": recovery_timeout,
            },
            retry_config={
                "max_attempts": max_attempts,
                "base_delay": 0.5,
                "max_delay": 10.0,
            },
            timeout_seconds=timeout_seconds,
        )

    def api_operation(
        self,
        timeout_seconds: float = 60.0,
        max_attempts: int = 5,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        """Decorator for API operations with optimized resilience patterns.

        Args:
            timeout_seconds: Operation timeout
            max_attempts: Maximum retry attempts
            failure_threshold: Circuit breaker failure threshold
            recovery_timeout: Circuit breaker recovery timeout

        Returns:
            Decorator function
        """
        return self.create_resilient_wrapper(
            operation_name="external_api",
            circuit_breaker_config={
                "failure_threshold": failure_threshold,
                "recovery_timeout": recovery_timeout,
            },
            retry_config={
                "max_attempts": max_attempts,
                "base_delay": 1.0,
                "max_delay": 30.0,
            },
            timeout_seconds=timeout_seconds,
        )

    def cache_operation(
        self,
        timeout_seconds: float = 5.0,
        max_attempts: int = 2,
        failure_threshold: int = 3,
        recovery_timeout: float = 15.0,
    ):
        """Decorator for cache operations with optimized resilience patterns.

        Args:
            timeout_seconds: Operation timeout
            max_attempts: Maximum retry attempts
            failure_threshold: Circuit breaker failure threshold
            recovery_timeout: Circuit breaker recovery timeout

        Returns:
            Decorator function
        """
        return self.create_resilient_wrapper(
            operation_name="redis",
            circuit_breaker_config={
                "failure_threshold": failure_threshold,
                "recovery_timeout": recovery_timeout,
            },
            retry_config={
                "max_attempts": max_attempts,
                "base_delay": 0.1,
                "max_delay": 1.0,
            },
            timeout_seconds=timeout_seconds,
        )

    def ml_operation(
        self,
        timeout_seconds: float = 300.0,
        max_attempts: int = 2,
        failure_threshold: int = 2,
        recovery_timeout: float = 120.0,
    ):
        """Decorator for ML operations with optimized resilience patterns.

        Args:
            timeout_seconds: Operation timeout
            max_attempts: Maximum retry attempts
            failure_threshold: Circuit breaker failure threshold
            recovery_timeout: Circuit breaker recovery timeout

        Returns:
            Decorator function
        """
        return self.create_resilient_wrapper(
            operation_name="ml_operation",
            circuit_breaker_config={
                "failure_threshold": failure_threshold,
                "recovery_timeout": recovery_timeout,
            },
            retry_config={
                "max_attempts": max_attempts,
                "base_delay": 5.0,
                "max_delay": 30.0,
            },
            timeout_seconds=timeout_seconds,
        )

    def get_health_status(self) -> dict:
        """Get health status of all resilience components.

        Returns:
            Dictionary with health status information
        """
        circuit_breaker_stats = self.circuit_breaker_registry.get_stats()

        health_status = {"circuit_breakers": {}, "overall_health": "healthy"}

        # Analyze circuit breaker health
        open_breakers = []
        for name, stats in circuit_breaker_stats.items():
            health_status["circuit_breakers"][name] = {
                "state": stats["state"],
                "failure_rate": stats["failure_rate"],
                "total_calls": stats["total_calls"],
                "blocked_calls": stats["blocked_calls"],
            }

            if stats["state"] == "open":
                open_breakers.append(name)

        # Determine overall health
        if open_breakers:
            health_status["overall_health"] = "degraded"
            health_status["open_circuit_breakers"] = open_breakers

        # Calculate overall failure rate
        total_calls = sum(
            stats["total_calls"] for stats in circuit_breaker_stats.values()
        )
        total_failures = sum(
            stats["failed_calls"] for stats in circuit_breaker_stats.values()
        )

        if total_calls > 0:
            health_status["overall_failure_rate"] = total_failures / total_calls
        else:
            health_status["overall_failure_rate"] = 0.0

        return health_status

    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers to closed state."""
        self.circuit_breaker_registry.reset_all()
        logger.info("All circuit breakers reset")

    def configure_timeout(self, operation: str, timeout_seconds: float):
        """Configure timeout for specific operation type.

        Args:
            operation: Operation name/type
            timeout_seconds: Timeout in seconds
        """
        self.timeout_manager.set_timeout(operation, timeout_seconds)
        logger.info(f"Set timeout for '{operation}' to {timeout_seconds} seconds")


# Global resilience service instance
resilience_service = ResilienceService()

# Export convenience decorators
database_resilient = resilience_service.database_operation
api_resilient = resilience_service.api_operation
cache_resilient = resilience_service.cache_operation
ml_resilient = resilience_service.ml_operation

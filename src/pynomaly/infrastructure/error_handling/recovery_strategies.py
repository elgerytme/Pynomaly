"""Error recovery strategies and automatic retry mechanisms."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

from pynomaly.domain.exceptions import InfrastructureError, PynamolyError

T = TypeVar("T")


class RecoveryStrategy(ABC):
    """Abstract base class for error recovery strategies."""

    @abstractmethod
    async def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if this strategy can recover from the given error.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            
        Returns:
            True if recovery is possible, False otherwise
        """
        pass

    @abstractmethod
    async def recover(
        self,
        operation: Callable[[], T],
        error: Exception,
        context: dict[str, Any],
    ) -> T:
        """Attempt to recover from the error by retrying the operation.
        
        Args:
            operation: The operation to retry
            error: The original exception
            context: Additional context information
            
        Returns:
            Result of the operation if recovery succeeds
            
        Raises:
            Exception: If recovery fails
        """
        pass


class RetryStrategy(RecoveryStrategy):
    """Strategy that retries operations with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize retry strategy.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.logger = logging.getLogger(__name__)

    async def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if error is recoverable through retries."""
        # Generally recoverable errors
        recoverable_errors = (
            InfrastructureError,
            ConnectionError,
            TimeoutError,
        )
        
        # Check error type
        if isinstance(error, recoverable_errors):
            return True
        
        # Check error message for specific patterns
        error_message = str(error).lower()
        recoverable_patterns = [
            "connection",
            "timeout",
            "temporary",
            "unavailable",
            "rate limit",
            "service unavailable",
        ]
        
        return any(pattern in error_message for pattern in recoverable_patterns)

    async def recover(
        self,
        operation: Callable[[], T],
        error: Exception,
        context: dict[str, Any],
    ) -> T:
        """Retry operation with exponential backoff."""
        last_error = error
        
        for attempt in range(self.max_retries):
            # Calculate delay
            delay = min(
                self.base_delay * (self.exponential_base ** attempt),
                self.max_delay
            )
            
            # Add jitter
            if self.jitter:
                import random
                delay *= (0.5 + random.random() * 0.5)
            
            self.logger.info(
                f"Retrying operation (attempt {attempt + 1}/{self.max_retries}) "
                f"after {delay:.2f}s delay. Previous error: {type(last_error).__name__}"
            )
            
            # Wait before retry
            await asyncio.sleep(delay)
            
            try:
                # Retry the operation
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()
                    
            except Exception as e:
                last_error = e
                self.logger.warning(
                    f"Retry attempt {attempt + 1} failed: {type(e).__name__}: {e}"
                )
        
        # All retries failed
        raise PynamolyError(
            f"Operation failed after {self.max_retries} retry attempts",
            details={
                "max_retries": self.max_retries,
                "final_error": str(last_error),
                "error_type": type(last_error).__name__,
            },
            cause=last_error,
        )


class CircuitBreakerStrategy(RecoveryStrategy):
    """Strategy that implements circuit breaker pattern."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        """Initialize circuit breaker strategy.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.logger = logging.getLogger(__name__)

    async def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if circuit breaker can handle this error."""
        return isinstance(error, self.expected_exception)

    async def recover(
        self,
        operation: Callable[[], T],
        error: Exception,
        context: dict[str, Any],
    ) -> T:
        """Apply circuit breaker logic."""
        current_time = time.time()
        
        # Update failure count
        self.failure_count += 1
        self.last_failure_time = current_time
        
        # Check if we should open the circuit
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )
        
        # If circuit is open, check if we can try recovery
        if self.state == "open":
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self.state = "half-open"
                self.logger.info("Circuit breaker entering half-open state")
            else:
                raise PynamolyError(
                    "Circuit breaker is open - operation not attempted",
                    details={
                        "state": self.state,
                        "failure_count": self.failure_count,
                        "time_until_retry": self.recovery_timeout - (current_time - self.last_failure_time),
                    },
                )
        
        # Try the operation in half-open state
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()
            
            # Success - reset circuit breaker
            self.failure_count = 0
            self.state = "closed"
            self.logger.info("Circuit breaker reset after successful operation")
            
            return result
            
        except Exception as e:
            # Failure - back to open state
            self.state = "open"
            self.last_failure_time = current_time
            raise e


class FallbackStrategy(RecoveryStrategy):
    """Strategy that provides fallback operations when primary fails."""

    def __init__(self, fallback_operation: Callable[[], T] | None = None):
        """Initialize fallback strategy.
        
        Args:
            fallback_operation: Fallback operation to execute
        """
        self.fallback_operation = fallback_operation
        self.logger = logging.getLogger(__name__)

    async def can_recover(self, error: Exception, context: dict[str, Any]) -> bool:
        """Check if fallback is available."""
        return self.fallback_operation is not None

    async def recover(
        self,
        operation: Callable[[], T],
        error: Exception,
        context: dict[str, Any],
    ) -> T:
        """Execute fallback operation."""
        if not self.fallback_operation:
            raise PynamolyError(
                "No fallback operation available",
                cause=error,
            )
        
        self.logger.info(
            f"Executing fallback operation due to: {type(error).__name__}"
        )
        
        try:
            if asyncio.iscoroutinefunction(self.fallback_operation):
                return await self.fallback_operation()
            else:
                return self.fallback_operation()
                
        except Exception as fallback_error:
            raise PynamolyError(
                "Both primary and fallback operations failed",
                details={
                    "primary_error": str(error),
                    "fallback_error": str(fallback_error),
                },
                cause=fallback_error,
            )


class RecoveryStrategyRegistry:
    """Registry for managing and applying recovery strategies."""

    def __init__(self):
        """Initialize recovery strategy registry."""
        self.strategies: list[RecoveryStrategy] = []
        self.logger = logging.getLogger(__name__)

    def register_strategy(self, strategy: RecoveryStrategy) -> None:
        """Register a recovery strategy."""
        self.strategies.append(strategy)

    async def attempt_recovery(
        self,
        operation: Callable[[], T],
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> T:
        """Attempt recovery using registered strategies.
        
        Args:
            operation: Operation to retry
            error: Original exception
            context: Additional context
            
        Returns:
            Result of successful recovery
            
        Raises:
            Exception: If no recovery strategy succeeds
        """
        context = context or {}
        
        for strategy in self.strategies:
            try:
                if await strategy.can_recover(error, context):
                    self.logger.info(
                        f"Attempting recovery with {type(strategy).__name__}"
                    )
                    return await strategy.recover(operation, error, context)
            except Exception as recovery_error:
                self.logger.warning(
                    f"Recovery strategy {type(strategy).__name__} failed: {recovery_error}"
                )
                continue
        
        # No recovery strategy succeeded
        raise PynamolyError(
            "All recovery strategies failed",
            details={
                "original_error": str(error),
                "strategies_attempted": len(self.strategies),
            },
            cause=error,
        )


def create_default_recovery_registry() -> RecoveryStrategyRegistry:
    """Create default recovery strategy registry."""
    registry = RecoveryStrategyRegistry()
    
    # Add retry strategy for temporary failures
    retry_strategy = RetryStrategy(max_retries=3, base_delay=1.0)
    registry.register_strategy(retry_strategy)
    
    # Add circuit breaker for infrastructure failures
    circuit_breaker = CircuitBreakerStrategy(
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=InfrastructureError,
    )
    registry.register_strategy(circuit_breaker)
    
    return registry
"""Circuit breaker implementation for fault tolerance."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Union
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit is open, calls fail immediately
    HALF_OPEN = "half_open" # Testing if service is back


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and call is blocked."""
    pass


class CircuitBreaker:
    """Circuit breaker implementation with automatic failure detection and recovery."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Union[type, tuple] = Exception,
        name: Optional[str] = None
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery (seconds)
            expected_exception: Exception types that count as failures
            name: Optional name for logging and monitoring
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name or "unnamed"
        
        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._next_attempt_time: Optional[float] = None
        
        # Statistics
        self._total_calls = 0
        self._failed_calls = 0
        self._successful_calls = 0
        self._blocked_calls = 0
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count
    
    @property
    def stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "failed_calls": self._failed_calls,
            "successful_calls": self._successful_calls,
            "blocked_calls": self._blocked_calls,
            "failure_rate": self._failed_calls / max(self._total_calls, 1),
            "last_failure_time": self._last_failure_time,
            "next_attempt_time": self._next_attempt_time,
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        logger.info(f"Resetting circuit breaker '{self.name}'")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._next_attempt_time = None
    
    def _can_attempt_call(self) -> bool:
        """Check if a call can be attempted based on current state."""
        current_time = time.time()
        
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self._next_attempt_time and 
                current_time >= self._next_attempt_time):
                logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        elif self._state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _record_success(self) -> None:
        """Record a successful call."""
        self._total_calls += 1
        self._successful_calls += 1
        
        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit breaker '{self.name}' recovered, transitioning to CLOSED")
            self.reset()
        else:
            # In CLOSED state, reset failure count on success
            self._failure_count = 0
    
    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        self._total_calls += 1
        self._failed_calls += 1
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        logger.warning(
            f"Circuit breaker '{self.name}' recorded failure {self._failure_count}: {exception}"
        )
        
        # Check if we should open the circuit
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._next_attempt_time = time.time() + self.recovery_timeout
            logger.error(
                f"Circuit breaker '{self.name}' opened after {self._failure_count} failures. "
                f"Next attempt at {self._next_attempt_time}"
            )
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection (synchronous).
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception if function fails
        """
        if not self._can_attempt_call():
            self._blocked_calls += 1
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Next attempt in {self._next_attempt_time - time.time():.1f} seconds"
            )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure(e)
            raise
    
    async def acall(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception if function fails
        """
        if not self._can_attempt_call():
            self._blocked_calls += 1
            raise CircuitBreakerError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Next attempt in {self._next_attempt_time - time.time():.1f} seconds"
            )
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure(e)
            raise
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for applying circuit breaker to functions."""
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.acall(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
            return sync_wrapper


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
    
    def register(self, name: str, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker.
        
        Args:
            name: Unique name for the circuit breaker
            breaker: Circuit breaker instance
        """
        self._breakers[name] = breaker
        breaker.name = name
        logger.info(f"Registered circuit breaker '{name}'")
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name.
        
        Args:
            name: Circuit breaker name
            
        Returns:
            Circuit breaker instance or None if not found
        """
        return self._breakers.get(name)
    
    def create_and_register(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Union[type, tuple] = Exception
    ) -> CircuitBreaker:
        """Create and register a new circuit breaker.
        
        Args:
            name: Unique name for the circuit breaker
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception types that count as failures
            
        Returns:
            Created circuit breaker instance
        """
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=name
        )
        self.register(name, breaker)
        return breaker
    
    def get_stats(self) -> dict[str, dict]:
        """Get statistics for all registered circuit breakers.
        
        Returns:
            Dictionary mapping breaker names to their statistics
        """
        return {name: breaker.stats for name, breaker in self._breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all registered circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()
        logger.info("Reset all circuit breakers")
    
    def list_breakers(self) -> list[str]:
        """List all registered circuit breaker names.
        
        Returns:
            List of circuit breaker names
        """
        return list(self._breakers.keys())


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Union[type, tuple] = Exception
):
    """Decorator factory for applying circuit breaker pattern.
    
    Args:
        name: Unique name for the circuit breaker
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception types that count as failures
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Get or create circuit breaker
        breaker = circuit_breaker_registry.get(name)
        if breaker is None:
            breaker = circuit_breaker_registry.create_and_register(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception
            )
        
        return breaker(func)
    
    return decorator


# Convenience function for common use cases
def database_circuit_breaker(name: str = "database") -> CircuitBreaker:
    """Create a circuit breaker optimized for database operations.
    
    Args:
        name: Circuit breaker name
        
    Returns:
        Configured circuit breaker
    """
    import sqlalchemy.exc
    import psycopg2
    
    try:
        database_exceptions = (
            sqlalchemy.exc.SQLAlchemyError,
            psycopg2.OperationalError,
            psycopg2.InterfaceError,
            ConnectionError,
            TimeoutError,
        )
    except ImportError:
        database_exceptions = (ConnectionError, TimeoutError)
    
    return circuit_breaker_registry.create_and_register(
        name=name,
        failure_threshold=3,
        recovery_timeout=30.0,
        expected_exception=database_exceptions
    )


def external_api_circuit_breaker(name: str = "external_api") -> CircuitBreaker:
    """Create a circuit breaker optimized for external API calls.
    
    Args:
        name: Circuit breaker name
        
    Returns:
        Configured circuit breaker
    """
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
    
    return circuit_breaker_registry.create_and_register(
        name=name,
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=api_exceptions
    )


def redis_circuit_breaker(name: str = "redis") -> CircuitBreaker:
    """Create a circuit breaker optimized for Redis operations.
    
    Args:
        name: Circuit breaker name
        
    Returns:
        Configured circuit breaker
    """
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
    
    return circuit_breaker_registry.create_and_register(
        name=name,
        failure_threshold=3,
        recovery_timeout=15.0,
        expected_exception=redis_exceptions
    )
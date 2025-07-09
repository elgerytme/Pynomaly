"""Resilience patterns for error handling: circuit breakers, retries, and fallbacks."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union, List
from dataclasses import dataclass, field
from enum import Enum
import random
import functools
import logging
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from .unified_exceptions import (
    PynamolyError,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    TimeoutError,
    ResourceExhaustionError,
    ExternalServiceError,
    ErrorCodes,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5           # Number of failures to open circuit
    recovery_timeout: float = 60.0       # Seconds to wait before trying recovery
    success_threshold: int = 3           # Successes needed to close circuit
    timeout: float = 30.0                # Operation timeout in seconds
    expected_exception: type = Exception  # Expected exception type
    monitor_calls: bool = True           # Whether to monitor all calls


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: List[type] = field(default_factory=lambda: [Exception])
    stop_on: List[type] = field(default_factory=list)


@dataclass
class BulkheadConfig:
    """Bulkhead configuration for resource isolation."""
    max_concurrent_calls: int = 10
    max_queue_size: int = 100
    timeout: float = 30.0


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascading failures."""

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        error_context: Optional[ErrorContext] = None,
    ):
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Configuration options
            error_context: Error context for logging
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.error_context = error_context or ErrorContext()
        
        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        
        # Threading
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry."""
        await self._pre_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            await self._on_success()
        else:
            await self._on_failure(exc_val)
        return False

    async def _pre_call(self) -> None:
        """Check if call should be allowed."""
        async with self._lock:
            self.total_calls += 1
            
            current_time = time.time()
            
            if self.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                if (current_time - self.last_failure_time) >= self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN state")
                else:
                    # Still in open state, reject call
                    self.rejected_calls += 1
                    raise ExternalServiceError(
                        error_code=ErrorCodes.EXT_SERVICE_UNAVAILABLE,
                        message=f"Circuit breaker '{self.name}' is OPEN",
                        service_name=self.name,
                        context=self.error_context,
                        recovery_suggestions=[
                            f"Wait {self.config.recovery_timeout} seconds for recovery",
                            "Check service health",
                            "Implement fallback mechanism"
                        ]
                    )

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self.successful_calls += 1
            self.last_success_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' moved to CLOSED state")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0

    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed call."""
        async with self._lock:
            self.failed_calls += 1
            self.last_failure_time = time.time()
            
            # Check if this is an expected exception type
            if isinstance(exception, self.config.expected_exception):
                self.failure_count += 1
                
                if self.state == CircuitBreakerState.CLOSED:
                    if self.failure_count >= self.config.failure_threshold:
                        self.state = CircuitBreakerState.OPEN
                        logger.error(f"Circuit breaker '{self.name}' moved to OPEN state")
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.OPEN
                    logger.error(f"Circuit breaker '{self.name}' moved back to OPEN state")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            }
        }

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self:
            try:
                # Add timeout
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    error_code=ErrorCodes.PERF_TIMEOUT,
                    message=f"Operation timed out after {self.config.timeout} seconds",
                    timeout_seconds=self.config.timeout,
                    context=self.error_context
                )


class RetryHandler:
    """Advanced retry handler with exponential backoff and jitter."""

    def __init__(
        self,
        config: RetryConfig = None,
        error_context: Optional[ErrorContext] = None,
    ):
        """Initialize retry handler.
        
        Args:
            config: Retry configuration
            error_context: Error context for logging
        """
        self.config = config or RetryConfig()
        self.error_context = error_context or ErrorContext()

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should stop retrying
                if any(isinstance(e, exc_type) for exc_type in self.config.stop_on):
                    logger.info(f"Stopping retry due to non-retryable exception: {type(e).__name__}")
                    break
                
                # Check if we should retry
                if not any(isinstance(e, exc_type) for exc_type in self.config.retry_on):
                    logger.info(f"Not retrying due to exception type: {type(e).__name__}")
                    break
                
                # Calculate delay for next attempt
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay:.2f} seconds: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed")
        
        # All attempts failed, raise the last exception
        if isinstance(last_exception, PynamolyError):
            raise last_exception
        else:
            raise ExternalServiceError(
                error_code=ErrorCodes.EXT_SERVICE_UNAVAILABLE,
                message=f"All retry attempts failed: {str(last_exception)}",
                service_name="retry_handler",
                cause=last_exception,
                context=self.error_context
            )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


class Bulkhead:
    """Bulkhead pattern for resource isolation."""

    def __init__(
        self,
        name: str,
        config: BulkheadConfig = None,
        error_context: Optional[ErrorContext] = None,
    ):
        """Initialize bulkhead.
        
        Args:
            name: Bulkhead name
            config: Configuration options
            error_context: Error context for logging
        """
        self.name = name
        self.config = config or BulkheadConfig()
        self.error_context = error_context or ErrorContext()
        
        # Semaphore for concurrent calls
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)
        
        # Queue for pending calls
        self.queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        
        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        self.current_calls = 0

    @asynccontextmanager
    async def acquire(self):
        """Acquire bulkhead resource."""
        # Check if we can accept more calls
        if self.semaphore.locked() and self.queue.full():
            self.rejected_calls += 1
            raise ResourceExhaustionError(
                error_code=ErrorCodes.PERF_RESOURCE_EXHAUSTED,
                message=f"Bulkhead '{self.name}' is full",
                resource_type="bulkhead",
                context=self.error_context
            )
        
        # Acquire semaphore
        async with self.semaphore:
            self.total_calls += 1
            self.current_calls += 1
            
            try:
                yield
                self.successful_calls += 1
            except Exception:
                self.failed_calls += 1
                raise
            finally:
                self.current_calls -= 1

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with bulkhead protection."""
        async with self.acquire():
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    error_code=ErrorCodes.PERF_TIMEOUT,
                    message=f"Bulkhead operation timed out after {self.config.timeout} seconds",
                    timeout_seconds=self.config.timeout,
                    context=self.error_context
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self.name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "current_calls": self.current_calls,
            "available_slots": self.config.max_concurrent_calls - self.current_calls,
            "queue_size": self.queue.qsize(),
            "queue_capacity": self.config.max_queue_size,
            "config": {
                "max_concurrent_calls": self.config.max_concurrent_calls,
                "max_queue_size": self.config.max_queue_size,
                "timeout": self.config.timeout,
            }
        }


class ResilienceManager:
    """Central manager for resilience patterns."""

    def __init__(self):
        """Initialize resilience manager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.global_stats = {
            "circuit_breakers": {},
            "bulkheads": {},
            "retry_attempts": 0,
            "retry_successes": 0,
            "retry_failures": 0,
        }

    def get_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig = None,
        error_context: Optional[ErrorContext] = None,
    ) -> CircuitBreaker:
        """Get or create circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name=name,
                config=config,
                error_context=error_context,
            )
        return self.circuit_breakers[name]

    def get_bulkhead(
        self,
        name: str,
        config: BulkheadConfig = None,
        error_context: Optional[ErrorContext] = None,
    ) -> Bulkhead:
        """Get or create bulkhead."""
        if name not in self.bulkheads:
            self.bulkheads[name] = Bulkhead(
                name=name,
                config=config,
                error_context=error_context,
            )
        return self.bulkheads[name]

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all resilience patterns."""
        stats = {
            "circuit_breakers": {},
            "bulkheads": {},
            "summary": {
                "total_circuit_breakers": len(self.circuit_breakers),
                "total_bulkheads": len(self.bulkheads),
                "open_circuit_breakers": 0,
                "half_open_circuit_breakers": 0,
                "full_bulkheads": 0,
            }
        }
        
        # Circuit breaker stats
        for name, cb in self.circuit_breakers.items():
            cb_stats = cb.get_stats()
            stats["circuit_breakers"][name] = cb_stats
            
            if cb_stats["state"] == "open":
                stats["summary"]["open_circuit_breakers"] += 1
            elif cb_stats["state"] == "half_open":
                stats["summary"]["half_open_circuit_breakers"] += 1
        
        # Bulkhead stats
        for name, bulkhead in self.bulkheads.items():
            bulkhead_stats = bulkhead.get_stats()
            stats["bulkheads"][name] = bulkhead_stats
            
            if bulkhead_stats["available_slots"] == 0:
                stats["summary"]["full_bulkheads"] += 1
        
        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all resilience patterns."""
        health_status = {
            "overall_health": "healthy",
            "circuit_breakers": {},
            "bulkheads": {},
            "issues": [],
        }
        
        # Check circuit breakers
        for name, cb in self.circuit_breakers.items():
            cb_stats = cb.get_stats()
            cb_health = "healthy"
            
            if cb_stats["state"] == "open":
                cb_health = "unhealthy"
                health_status["issues"].append(f"Circuit breaker '{name}' is open")
            elif cb_stats["state"] == "half_open":
                cb_health = "recovering"
                health_status["issues"].append(f"Circuit breaker '{name}' is half-open")
            
            health_status["circuit_breakers"][name] = cb_health
        
        # Check bulkheads
        for name, bulkhead in self.bulkheads.items():
            bulkhead_stats = bulkhead.get_stats()
            bulkhead_health = "healthy"
            
            utilization = bulkhead_stats["current_calls"] / bulkhead_stats["config"]["max_concurrent_calls"]
            
            if utilization > 0.9:
                bulkhead_health = "stressed"
                health_status["issues"].append(f"Bulkhead '{name}' is highly utilized ({utilization:.1%})")
            elif utilization > 0.7:
                bulkhead_health = "warning"
            
            health_status["bulkheads"][name] = bulkhead_health
        
        # Overall health
        if health_status["issues"]:
            health_status["overall_health"] = "degraded"
        
        return health_status


# Global resilience manager
_resilience_manager = ResilienceManager()


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager."""
    return _resilience_manager


# Decorator functions for easy use
def circuit_breaker(
    name: str,
    config: CircuitBreakerConfig = None,
    error_context: Optional[ErrorContext] = None,
):
    """Decorator for circuit breaker protection."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            cb = get_resilience_manager().get_circuit_breaker(name, config, error_context)
            return await cb.call(func, *args, **kwargs)
        return wrapper
    return decorator


def retry(
    config: RetryConfig = None,
    error_context: Optional[ErrorContext] = None,
):
    """Decorator for retry logic."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            retry_handler = RetryHandler(config, error_context)
            return await retry_handler.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def bulkhead(
    name: str,
    config: BulkheadConfig = None,
    error_context: Optional[ErrorContext] = None,
):
    """Decorator for bulkhead protection."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            bulkhead_instance = get_resilience_manager().get_bulkhead(name, config, error_context)
            return await bulkhead_instance.execute(func, *args, **kwargs)
        return wrapper
    return decorator
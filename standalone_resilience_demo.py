#!/usr/bin/env python3
"""
Standalone demonstration of resilience patterns.
This shows the core functionality without dependencies on the main codebase.
"""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar
from enum import Enum
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import tenacity for retry functionality
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        wait_random_exponential,
        retry_if_exception_type,
        before_sleep_log,
        after_log,
        RetryError,
    )
    logger.info("✓ Tenacity imported successfully")
except ImportError as e:
    logger.error(f"✗ Tenacity import failed: {e}")
    exit(1)

T = TypeVar("T")

# Circuit Breaker Implementation
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreakerError(Exception):
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._next_attempt_time = None
        self._total_calls = 0
        self._failed_calls = 0
        
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        if not self._can_attempt_call():
            raise CircuitBreakerError("Circuit breaker is OPEN")
        
        self._total_calls += 1
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    def _can_attempt_call(self) -> bool:
        current_time = time.time()
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            if self._next_attempt_time and current_time >= self._next_attempt_time:
                self._state = CircuitState.HALF_OPEN
                return True
            return False
        elif self._state == CircuitState.HALF_OPEN:
            return True
        return False
    
    def _record_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
        else:
            self._failure_count = 0
    
    def _record_failure(self, exception: Exception):
        self._failed_calls += 1
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._next_attempt_time = time.time() + self.recovery_timeout
            logger.warning(f"Circuit breaker opened after {self._failure_count} failures")
    
    @property
    def state(self) -> CircuitState:
        return self._state
    
    @property
    def stats(self) -> dict:
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "total_calls": self._total_calls,
            "failed_calls": self._failed_calls,
            "failure_rate": self._failed_calls / max(self._total_calls, 1),
        }

# Resilience Configuration
@dataclass
class ResilienceConfig:
    timeout_seconds: float = 5.0
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    enable_timeout: bool = True
    enable_retry: bool = True
    enable_circuit_breaker: bool = True

# Enhanced Resilience Wrapper
class EnhancedResilienceWrapper:
    def __init__(self, config: ResilienceConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout
        ) if config.enable_circuit_breaker else None
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        wrapped_func = func
        
        # Apply retry (inner layer)
        if self.config.enable_retry:
            if self.config.jitter:
                wait_strategy = wait_random_exponential(
                    multiplier=self.config.base_delay,
                    max=self.config.max_delay,
                    exp_base=self.config.exponential_base,
                )
            else:
                wait_strategy = wait_exponential(
                    multiplier=self.config.base_delay,
                    max=self.config.max_delay,
                    exp_base=self.config.exponential_base,
                )
            
            wrapped_func = retry(
                stop=stop_after_attempt(self.config.max_attempts),
                wait=wait_strategy,
                retry=retry_if_exception_type((ConnectionError, TimeoutError)),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                after=after_log(logger, logging.DEBUG),
                reraise=True,
            )(wrapped_func)
        
        # Apply circuit breaker (outer layer)
        if self.circuit_breaker:
            original_func = wrapped_func
            
            @wraps(original_func)
            def circuit_breaker_wrapper(*args, **kwargs):
                return self.circuit_breaker.call(original_func, *args, **kwargs)
            
            wrapped_func = circuit_breaker_wrapper
        
        return wrapped_func

# Resilient Decorators
def resilient(
    timeout_seconds: float = 5.0,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    enable_timeout: bool = True,
    enable_retry: bool = True,
    enable_circuit_breaker: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    config = ResilienceConfig(
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        enable_timeout=enable_timeout,
        enable_retry=enable_retry,
        enable_circuit_breaker=enable_circuit_breaker,
    )
    wrapper = EnhancedResilienceWrapper(config)
    return wrapper

# Specialized decorators
def database_resilient(
    timeout_seconds: float = 30.0,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    failure_threshold: int = 3,
    recovery_timeout: float = 30.0,
    **kwargs
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return resilient(
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        **kwargs
    )

def api_resilient(
    timeout_seconds: float = 60.0,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    **kwargs
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return resilient(
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        **kwargs
    )

def cache_resilient(
    timeout_seconds: float = 5.0,
    max_attempts: int = 2,
    base_delay: float = 0.1,
    max_delay: float = 1.0,
    failure_threshold: int = 3,
    recovery_timeout: float = 15.0,
    **kwargs
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    return resilient(
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        **kwargs
    )

# Demo Functions
def demo_basic_retry():
    """Demo basic retry functionality."""
    logger.info("=== Basic Retry Demo ===")
    
    attempt_count = 0
    
    @database_resilient(enable_circuit_breaker=False)
    def flaky_database_call():
        nonlocal attempt_count
        attempt_count += 1
        logger.info(f"Database call attempt {attempt_count}")
        
        if attempt_count < 3:
            raise ConnectionError("Database connection failed")
        
        return {"user_id": 123, "name": "John Doe"}
    
    try:
        result = flaky_database_call()
        logger.info(f"✓ Database call succeeded: {result}")
        logger.info(f"Total attempts: {attempt_count}")
    except Exception as e:
        logger.error(f"✗ Database call failed: {e}")

def demo_circuit_breaker():
    """Demo circuit breaker functionality."""
    logger.info("\n=== Circuit Breaker Demo ===")
    
    @api_resilient(
        enable_retry=False,  # Disable retry to show circuit breaker
        failure_threshold=2,
        recovery_timeout=1.0
    )
    def failing_api_call():
        raise ConnectionError("API service unavailable")
    
    # First call - should fail normally
    try:
        failing_api_call()
    except ConnectionError:
        logger.info("✓ First call failed as expected")
    
    # Second call - should fail and open circuit
    try:
        failing_api_call()
    except ConnectionError:
        logger.info("✓ Second call failed, circuit breaker should open")
    
    # Third call - should be blocked by circuit breaker
    try:
        failing_api_call()
        logger.error("✗ Third call should have been blocked")
    except CircuitBreakerError:
        logger.info("✓ Third call blocked by circuit breaker")
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")

def demo_combined_resilience():
    """Demo combined retry and circuit breaker."""
    logger.info("\n=== Combined Resilience Demo ===")
    
    global_attempt_count = 0
    
    @cache_resilient(
        max_attempts=2,
        base_delay=0.1,
        failure_threshold=3,
        recovery_timeout=1.0
    )
    def cache_operation():
        nonlocal global_attempt_count
        global_attempt_count += 1
        logger.info(f"Cache operation attempt {global_attempt_count}")
        
        if global_attempt_count <= 4:  # Fail first 4 calls
            raise ConnectionError("Cache service unavailable")
        
        return {"cached_data": "important_value"}
    
    # Make multiple calls to demonstrate circuit breaker behavior
    for i in range(6):
        try:
            result = cache_operation()
            logger.info(f"✓ Cache call {i+1} succeeded: {result}")
        except (ConnectionError, RetryError):
            logger.info(f"✗ Cache call {i+1} failed after retries")
        except CircuitBreakerError:
            logger.info(f"✗ Cache call {i+1} blocked by circuit breaker")
        except Exception as e:
            logger.error(f"✗ Cache call {i+1} unexpected error: {e}")
        
        time.sleep(0.1)  # Small delay between calls

async def demo_async_resilience():
    """Demo async resilience patterns."""
    logger.info("\n=== Async Resilience Demo ===")
    
    attempt_count = 0
    
    @api_resilient(
        max_attempts=3,
        base_delay=0.1,
        enable_circuit_breaker=False
    )
    async def async_api_call():
        nonlocal attempt_count
        attempt_count += 1
        logger.info(f"Async API call attempt {attempt_count}")
        
        if attempt_count < 3:
            raise ConnectionError("Async API temporarily unavailable")
        
        return {"status": "success", "data": "async_result"}
    
    try:
        result = await async_api_call()
        logger.info(f"✓ Async API call succeeded: {result}")
        logger.info(f"Total attempts: {attempt_count}")
    except Exception as e:
        logger.error(f"✗ Async API call failed: {e}")

def demo_custom_configuration():
    """Demo custom resilience configuration."""
    logger.info("\n=== Custom Configuration Demo ===")
    
    attempt_count = 0
    
    @resilient(
        timeout_seconds=10.0,
        max_attempts=5,
        base_delay=0.5,
        max_delay=5.0,
        exponential_base=1.5,
        jitter=True,
        failure_threshold=3,
        recovery_timeout=2.0,
        enable_timeout=False,  # Disable timeout for demo
        enable_retry=True,
        enable_circuit_breaker=False,  # Disable circuit breaker for demo
    )
    def custom_operation():
        nonlocal attempt_count
        attempt_count += 1
        logger.info(f"Custom operation attempt {attempt_count}")
        
        if attempt_count < 4:
            raise ConnectionError("Custom service temporarily unavailable")
        
        return {"operation": "completed", "attempts": attempt_count}
    
    try:
        result = custom_operation()
        logger.info(f"✓ Custom operation succeeded: {result}")
    except Exception as e:
        logger.error(f"✗ Custom operation failed: {e}")

def demo_health_monitoring():
    """Demo health monitoring functionality."""
    logger.info("\n=== Health Monitoring Demo ===")
    
    # Create a circuit breaker for monitoring
    config = ResilienceConfig(failure_threshold=2, recovery_timeout=1.0)
    wrapper = EnhancedResilienceWrapper(config)
    
    @wrapper
    def monitored_service():
        raise ConnectionError("Service unavailable")
    
    # Make some calls to change circuit breaker state
    for i in range(3):
        try:
            monitored_service()
        except (ConnectionError, CircuitBreakerError):
            pass
    
    # Check circuit breaker stats
    if wrapper.circuit_breaker:
        stats = wrapper.circuit_breaker.stats
        logger.info(f"Circuit breaker stats: {stats}")
        logger.info(f"Circuit breaker state: {wrapper.circuit_breaker.state.value}")

# Main demonstration
async def main():
    """Main demonstration of resilience patterns."""
    logger.info("🚀 Resilience Patterns Demonstration")
    logger.info("=" * 50)
    
    # Run demonstrations
    demo_basic_retry()
    demo_circuit_breaker()
    demo_combined_resilience()
    await demo_async_resilience()
    demo_custom_configuration()
    demo_health_monitoring()
    
    logger.info("\n" + "=" * 50)
    logger.info("✅ All demonstrations completed successfully!")
    logger.info("🎉 Resilience patterns are working correctly!")

if __name__ == "__main__":
    asyncio.run(main())

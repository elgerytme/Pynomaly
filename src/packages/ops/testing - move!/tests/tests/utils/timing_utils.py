"""Timing utilities for improved test reliability and performance."""

import asyncio
import time
from typing import Callable, Any, Optional, Union
import logging
import os

logger = logging.getLogger(__name__)


class AdaptiveTimeout:
    """Dynamic timeout calculation based on system performance and CI environment."""
    
    def __init__(self, base_timeout: float = 1.0, ci_multiplier: float = 3.0, debug_multiplier: float = 5.0):
        self.base_timeout = base_timeout
        self.ci_multiplier = ci_multiplier
        self.debug_multiplier = debug_multiplier
    
    def get_timeout(self, operation_type: str = "default") -> float:
        """Get adaptive timeout based on environment and operation type."""
        timeout = self.base_timeout
        
        # Adjust for CI environment
        if self._is_ci_environment():
            timeout *= self.ci_multiplier
        
        # Adjust for debug mode
        if self._is_debug_mode():
            timeout *= self.debug_multiplier
        
        # Adjust for specific operation types
        multipliers = {
            "network": 2.0,
            "database": 1.5,
            "file_io": 1.2,
            "computation": 0.8,
            "cache": 0.5,
        }
        
        if operation_type in multipliers:
            timeout *= multipliers[operation_type]
        
        return max(timeout, 0.1)  # Minimum 100ms
    
    def _is_ci_environment(self) -> bool:
        """Check if running in CI environment."""
        ci_indicators = [
            'CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 
            'GITLAB_CI', 'JENKINS_URL', 'BUILDKITE'
        ]
        return any(os.getenv(var) for var in ci_indicators)
    
    def _is_debug_mode(self) -> bool:
        """Check if running in debug mode."""
        return os.getenv('PYTEST_DEBUG') == '1' or logger.isEnabledFor(logging.DEBUG)


class ConditionWaiter:
    """Condition-based waiting utilities to replace hardcoded sleeps."""
    
    def __init__(self, timeout_calculator: Optional[AdaptiveTimeout] = None):
        self.timeout_calculator = timeout_calculator or AdaptiveTimeout()
    
    async def wait_for_condition(
        self,
        condition_func: Callable[[], Union[bool, Any]],
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        operation_type: str = "default",
        error_message: str = "Condition not met within timeout"
    ) -> bool:
        """Wait for a condition to be true with adaptive polling."""
        if timeout is None:
            timeout = self.timeout_calculator.get_timeout(operation_type)
        
        if poll_interval is None:
            # Adaptive polling interval based on timeout
            poll_interval = min(timeout / 20, 0.1)  # Max 20 polls, max 100ms interval
        
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            try:
                if asyncio.iscoroutinefunction(condition_func):
                    result = await condition_func()
                else:
                    result = condition_func()
                
                if result:
                    logger.debug(f"Condition met after {attempt} attempts in {time.time() - start_time:.3f}s")
                    return True
                    
            except Exception as e:
                logger.debug(f"Condition check failed (attempt {attempt}): {e}")
            
            attempt += 1
            # Exponential backoff with jitter
            actual_interval = poll_interval * (1.1 ** attempt) * (0.8 + 0.4 * time.time() % 1)
            await asyncio.sleep(min(actual_interval, poll_interval * 5))  # Cap at 5x base interval
        
        logger.warning(f"Condition not met after {attempt} attempts in {timeout:.3f}s")
        raise TimeoutError(error_message)
    
    def wait_for_condition_sync(
        self,
        condition_func: Callable[[], Union[bool, Any]],
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
        operation_type: str = "default",
        error_message: str = "Condition not met within timeout"
    ) -> bool:
        """Synchronous version of condition waiting."""
        if timeout is None:
            timeout = self.timeout_calculator.get_timeout(operation_type)
        
        if poll_interval is None:
            poll_interval = min(timeout / 20, 0.1)
        
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            try:
                result = condition_func()
                if result:
                    logger.debug(f"Condition met after {attempt} attempts in {time.time() - start_time:.3f}s")
                    return True
            except Exception as e:
                logger.debug(f"Condition check failed (attempt {attempt}): {e}")
            
            attempt += 1
            actual_interval = poll_interval * (1.1 ** attempt) * (0.8 + 0.4 * time.time() % 1)
            time.sleep(min(actual_interval, poll_interval * 5))
        
        logger.warning(f"Condition not met after {attempt} attempts in {timeout:.3f}s")
        raise TimeoutError(error_message)


class RetryMechanism:
    """Retry mechanism with exponential backoff and jitter."""
    
    def __init__(self, timeout_calculator: Optional[AdaptiveTimeout] = None):
        self.timeout_calculator = timeout_calculator or AdaptiveTimeout()
    
    async def retry_async(
        self,
        func: Callable,
        max_attempts: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 5.0,
        exponential_base: float = 2.0,
        jitter_range: float = 0.1,
        retryable_exceptions: tuple = (Exception,),
        operation_type: str = "default"
    ) -> Any:
        """Retry an async function with exponential backoff and jitter."""
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt == max_attempts - 1:
                    logger.error(f"Function failed after {max_attempts} attempts: {e}")
                    raise
                
                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                jitter = (time.time() % 1 - 0.5) * 2 * jitter_range * delay
                actual_delay = max(delay + jitter, 0.01)
                
                logger.debug(f"Attempt {attempt + 1} failed: {e}. Retrying in {actual_delay:.3f}s")
                await asyncio.sleep(actual_delay)
        
        raise last_exception
    
    def retry_sync(
        self,
        func: Callable,
        max_attempts: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 5.0,
        exponential_base: float = 2.0,
        jitter_range: float = 0.1,
        retryable_exceptions: tuple = (Exception,),
        operation_type: str = "default"
    ) -> Any:
        """Retry a sync function with exponential backoff and jitter."""
        last_exception = None
        
        for attempt in range(max_attempts):
            try:
                return func()
            except retryable_exceptions as e:
                last_exception = e
                
                if attempt == max_attempts - 1:
                    logger.error(f"Function failed after {max_attempts} attempts: {e}")
                    raise
                
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                jitter = (time.time() % 1 - 0.5) * 2 * jitter_range * delay
                actual_delay = max(delay + jitter, 0.01)
                
                logger.debug(f"Attempt {attempt + 1} failed: {e}. Retrying in {actual_delay:.3f}s")
                time.sleep(actual_delay)
        
        raise last_exception


# Global instances for convenience
adaptive_timeout = AdaptiveTimeout()
condition_waiter = ConditionWaiter(adaptive_timeout)
retry_mechanism = RetryMechanism(adaptive_timeout)


# Convenience functions
async def wait_for(condition_func: Callable, timeout: Optional[float] = None, 
                  operation_type: str = "default", error_message: str = "Condition not met") -> bool:
    """Convenience function for async condition waiting."""
    return await condition_waiter.wait_for_condition(
        condition_func, timeout, operation_type=operation_type, error_message=error_message
    )


def wait_for_sync(condition_func: Callable, timeout: Optional[float] = None,
                  operation_type: str = "default", error_message: str = "Condition not met") -> bool:
    """Convenience function for sync condition waiting."""
    return condition_waiter.wait_for_condition_sync(
        condition_func, timeout, operation_type=operation_type, error_message=error_message
    )


async def retry_async(func: Callable, max_attempts: int = 3, **kwargs) -> Any:
    """Convenience function for async retry."""
    return await retry_mechanism.retry_async(func, max_attempts, **kwargs)


def retry_sync(func: Callable, max_attempts: int = 3, **kwargs) -> Any:
    """Convenience function for sync retry."""
    return retry_mechanism.retry_sync(func, max_attempts, **kwargs)


def get_timeout(operation_type: str = "default") -> float:
    """Convenience function for getting adaptive timeout."""
    return adaptive_timeout.get_timeout(operation_type)
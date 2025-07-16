"""
Shared error handling utilities for all packages.
Provides common error handling decorators and utilities.
"""

import functools
import logging
from typing import Any, Callable, Optional, Type, Union


logger = logging.getLogger(__name__)


def handle_exceptions(
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False
):
    """
    Decorator to handle exceptions in functions and methods.
    
    Args:
        exceptions: Exception type(s) to catch
        default_return: Default return value when exception is caught
        log_errors: Whether to log caught exceptions
        reraise: Whether to reraise the exception after handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_errors:
                    logger.error(
                        f"Exception in {func.__name__}: {str(e)}",
                        exc_info=True,
                        extra={
                            "function": func.__name__,
                            "module": func.__module__,
                            "args": str(args)[:100],  # Truncate for safety
                            "kwargs": str(kwargs)[:100]
                        }
                    )
                
                if reraise:
                    raise
                
                return default_return
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except exceptions as e:
                if log_errors:
                    logger.error(
                        f"Exception in {func.__name__}: {str(e)}",
                        exc_info=True,
                        extra={
                            "function": func.__name__,
                            "module": func.__module__,
                            "args": str(args)[:100],
                            "kwargs": str(kwargs)[:100]
                        }
                    )
                
                if reraise:
                    raise
                
                return default_return
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Default return value on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
    
    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(
                f"Exception in safe_execute for {func.__name__}: {str(e)}",
                exc_info=True
            )
        return default_return


async def safe_execute_async(
    func: Callable,
    *args,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute an async function with error handling.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        default_return: Default return value on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
    
    Returns:
        Function result or default_return on error
    """
    try:
        return await func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(
                f"Exception in safe_execute_async for {func.__name__}: {str(e)}",
                exc_info=True
            )
        return default_return


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception
):
    """
    Decorator to retry function execution on specified exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor by which to multiply delay after each retry
        exceptions: Exception type(s) that should trigger a retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {current_delay}s..."
                        )
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}. "
                            f"Final error: {str(e)}"
                        )
            
            # Reraise the last exception if all retries failed
            if last_exception:
                raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {current_delay}s..."
                        )
                        import asyncio
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}. "
                            f"Final error: {str(e)}"
                        )
            
            # Reraise the last exception if all retries failed
            if last_exception:
                raise last_exception
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


class ErrorContext:
    """Context manager for capturing and handling errors."""
    
    def __init__(
        self,
        error_message: str = "An error occurred",
        log_errors: bool = True,
        reraise: bool = True,
        default_return: Any = None
    ):
        self.error_message = error_message
        self.log_errors = log_errors
        self.reraise = reraise
        self.default_return = default_return
        self.exception: Optional[Exception] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.exception = exc_val
            
            if self.log_errors:
                logger.error(
                    f"{self.error_message}: {str(exc_val)}",
                    exc_info=True
                )
            
            if not self.reraise:
                return True  # Suppress the exception
        
        return False  # Let the exception propagate
    
    def get_result(self, success_value: Any = None) -> Any:
        """Get the result based on whether an exception occurred."""
        if self.exception is not None:
            return self.default_return
        return success_value
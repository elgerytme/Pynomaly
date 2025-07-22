"""Decorators for automatic logging and error handling."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union
import inspect

from .structured_logger import get_logger
from .error_handler import ErrorHandler, AnomalyDetectionError

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


def log_decorator(
    operation: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    log_duration: bool = True,
    handle_errors: bool = True,
    logger_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator for automatic operation logging and error handling.
    
    Args:
        operation: Custom operation name (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_duration: Whether to log execution duration
        handle_errors: Whether to automatically handle and convert errors
        logger_name: Custom logger name (defaults to module name)
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        # Get logger name from module if not provided
        if logger_name is None:
            module_name = func.__module__
        else:
            module_name = logger_name
        
        logger = get_logger(module_name)
        error_handler = ErrorHandler(logger._logger) if handle_errors else None
        
        # Determine operation name
        op_name = operation or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            
            # Prepare log context
            context: Dict[str, Any] = {
                "function": func.__name__,
                "module": func.__module__
            }
            
            if log_args:
                # Get parameter names and values
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Filter out sensitive parameters
                safe_args = {}
                for param_name, param_value in bound_args.arguments.items():
                    if any(sensitive in param_name.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                        safe_args[param_name] = '[REDACTED]'
                    else:
                        # Convert non-serializable objects to string representation
                        try:
                            # Test if it's JSON serializable
                            import json
                            json.dumps(param_value)
                            safe_args[param_name] = param_value
                        except (TypeError, ValueError):
                            safe_args[param_name] = str(type(param_value))
                
                context["arguments"] = safe_args
            
            logger.log_operation_start(op_name, **context)
            
            try:
                result = func(*args, **kwargs)
                
                # Log result if requested
                if log_result and result is not None:
                    try:
                        import json
                        json.dumps(result)
                        context["result"] = result
                    except (TypeError, ValueError):
                        context["result"] = str(type(result))
                
                # Log success
                if log_duration:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    logger.log_operation_end(op_name, duration_ms, success=True, **context)
                else:
                    logger.info(f"Operation completed: {op_name}", **context)
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000 if log_duration else None
                
                if error_handler:
                    # Use error handler for structured error handling
                    context["error_occurred"] = True
                    if duration_ms is not None:
                        logger.log_operation_end(op_name, duration_ms, success=False, **context)
                    
                    return error_handler.handle_error(
                        error=e,
                        context=context,
                        operation=op_name,
                        reraise=True
                    )
                else:
                    # Simple logging without error conversion
                    context.update({
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    })
                    
                    if duration_ms is not None:
                        logger.log_operation_end(op_name, duration_ms, success=False, **context)
                    else:
                        logger.error(f"Operation failed: {op_name}", **context)
                    
                    raise
        
        return wrapper  # type: ignore
    
    return decorator


def timing_decorator(
    operation: Optional[str] = None,
    log_slow_threshold_ms: float = 1000.0,
    logger_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator for timing function execution.
    
    Args:
        operation: Custom operation name (defaults to function name)
        log_slow_threshold_ms: Threshold in milliseconds to log slow operations
        logger_name: Custom logger name (defaults to module name)
    
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        # Get logger name from module if not provided
        if logger_name is None:
            module_name = func.__module__
        else:
            module_name = logger_name
        
        logger = get_logger(module_name)
        op_name = operation or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Log timing information
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "duration_ms": round(duration_ms, 2)
                }
                
                if duration_ms > log_slow_threshold_ms:
                    logger.warning(f"Slow operation detected: {op_name}", **context)
                else:
                    logger.debug(f"Operation timing: {op_name}", **context)
                
                # Log as metric
                logger.log_metric(
                    name=f"{op_name}_duration",
                    value=duration_ms,
                    unit="milliseconds",
                    **context
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                logger.error(
                    f"Operation failed after {duration_ms:.2f}ms: {op_name}",
                    function=func.__name__,
                    module=func.__module__,
                    duration_ms=round(duration_ms, 2),
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                
                raise
        
        return wrapper  # type: ignore
    
    return decorator


def async_log_decorator(
    operation: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    log_duration: bool = True,
    handle_errors: bool = True,
    logger_name: Optional[str] = None
) -> Callable[[F], F]:
    """Async version of log_decorator.
    
    Args:
        operation: Custom operation name (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result  
        log_duration: Whether to log execution duration
        handle_errors: Whether to automatically handle and convert errors
        logger_name: Custom logger name (defaults to module name)
    
    Returns:
        Decorated async function
    """
    def decorator(func: F) -> F:
        # Get logger name from module if not provided
        if logger_name is None:
            module_name = func.__module__
        else:
            module_name = logger_name
        
        logger = get_logger(module_name)
        error_handler = ErrorHandler(logger._logger) if handle_errors else None
        
        # Determine operation name
        op_name = operation or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            
            # Prepare log context (same as sync version)
            context: Dict[str, Any] = {
                "function": func.__name__,
                "module": func.__module__
            }
            
            if log_args:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                safe_args = {}
                for param_name, param_value in bound_args.arguments.items():
                    if any(sensitive in param_name.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                        safe_args[param_name] = '[REDACTED]'
                    else:
                        try:
                            import json
                            json.dumps(param_value)
                            safe_args[param_name] = param_value
                        except (TypeError, ValueError):
                            safe_args[param_name] = str(type(param_value))
                
                context["arguments"] = safe_args
            
            logger.log_operation_start(op_name, **context)
            
            try:
                result = await func(*args, **kwargs)
                
                if log_result and result is not None:
                    try:
                        import json
                        json.dumps(result)
                        context["result"] = result
                    except (TypeError, ValueError):
                        context["result"] = str(type(result))
                
                if log_duration:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    logger.log_operation_end(op_name, duration_ms, success=True, **context)
                else:
                    logger.info(f"Async operation completed: {op_name}", **context)
                
                return result
                
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000 if log_duration else None
                
                if error_handler:
                    context["error_occurred"] = True
                    if duration_ms is not None:
                        logger.log_operation_end(op_name, duration_ms, success=False, **context)
                    
                    return error_handler.handle_error(
                        error=e,
                        context=context,
                        operation=op_name,
                        reraise=True
                    )
                else:
                    context.update({
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    })
                    
                    if duration_ms is not None:
                        logger.log_operation_end(op_name, duration_ms, success=False, **context)
                    else:
                        logger.error(f"Async operation failed: {op_name}", **context)
                    
                    raise
        
        return async_wrapper  # type: ignore
    
    return decorator
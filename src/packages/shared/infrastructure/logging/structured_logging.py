"""Shared structured logging framework using structlog.

This module provides standardized structured logging across all packages
in the monorepo, implementing the infrastructure standardization recommendations.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import logging.handlers
from typing import Any, Dict, Optional, Union, List, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from contextlib import contextmanager

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    # Fallback to standard logging
    import logging as structlog


class LogLevel:
    """Standard log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat:
    """Standard log formats."""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"
    CONSOLE = "console"


class BaseLoggerConfig:
    """Base configuration for loggers."""
    
    def __init__(
        self,
        package_name: str,
        level: str = LogLevel.INFO,
        format_type: str = LogFormat.JSON,
        enable_console: bool = True,
        enable_file: bool = False,
        file_path: Optional[str] = None,
        max_file_size: str = "10MB",
        backup_count: int = 5,
        enable_structured: bool = True,
        enable_request_tracking: bool = True,
        enable_performance_logging: bool = True,
        sanitize_sensitive_data: bool = True,
        slow_operation_threshold_ms: float = 1000.0
    ):
        self.package_name = package_name
        self.level = level
        self.format_type = format_type
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.file_path = file_path or f"{package_name}.log"
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_structured = enable_structured
        self.enable_request_tracking = enable_request_tracking
        self.enable_performance_logging = enable_performance_logging
        self.sanitize_sensitive_data = sanitize_sensitive_data
        self.slow_operation_threshold_ms = slow_operation_threshold_ms


def _parse_file_size(size_str: str) -> int:
    """Parse file size string to bytes."""
    size_str = size_str.upper()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


def _sanitize_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize sensitive data from log entries."""
    sensitive_keys = {
        'password', 'secret', 'key', 'token', 'auth', 'credential',
        'private', 'api_key', 'access_token', 'refresh_token', 'session'
    }
    
    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_sensitive_data(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_sensitive_data(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


class StructuredLogger:
    """Enhanced structured logger with context management."""
    
    def __init__(
        self, 
        config: BaseLoggerConfig,
        logger: Optional[Union[structlog.BoundLogger, logging.Logger]] = None
    ):
        self.config = config
        self._context: Dict[str, Any] = {}
        self._request_context: Dict[str, Any] = {}
        
        if STRUCTLOG_AVAILABLE:
            self._logger = logger or structlog.get_logger(config.package_name)
        else:
            self._logger = self._setup_fallback_logger()
    
    def _setup_fallback_logger(self) -> logging.Logger:
        """Setup fallback logger when structlog is not available."""
        logger = logging.getLogger(self.config.package_name)
        logger.setLevel(getattr(logging, self.config.level.upper()))
        
        if not logger.handlers:
            # Console handler
            if self.config.enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)
            
            # File handler
            if self.config.enable_file:
                try:
                    file_handler = logging.handlers.RotatingFileHandler(
                        self.config.file_path,
                        maxBytes=_parse_file_size(self.config.max_file_size),
                        backupCount=self.config.backup_count
                    )
                    file_formatter = logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                    file_handler.setFormatter(file_formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    print(f"Warning: Could not setup file logging: {e}")
            
            logger.propagate = False
        
        return logger
    
    def set_context(self, **kwargs) -> None:
        """Set persistent context for all log messages."""
        if self.config.sanitize_sensitive_data:
            kwargs = _sanitize_sensitive_data(kwargs)
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear persistent context."""
        self._context.clear()
    
    def set_request_context(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        method: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs
    ) -> None:
        """Set request-specific context."""
        request_context = {
            "request_id": request_id,
            "user_id": user_id,
            "method": method,
            "path": path,
            **kwargs
        }
        
        # Remove None values
        request_context = {k: v for k, v in request_context.items() if v is not None}
        
        if self.config.sanitize_sensitive_data:
            request_context = _sanitize_sensitive_data(request_context)
        
        self._request_context.update(request_context)
    
    def clear_request_context(self) -> None:
        """Clear request-specific context."""
        self._request_context.clear()
    
    def _prepare_log_data(self, **kwargs) -> Dict[str, Any]:
        """Prepare log data by merging contexts."""
        log_data = {
            "package": self.config.package_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self._context,
            **self._request_context,
            **kwargs
        }
        
        if self.config.sanitize_sensitive_data:
            log_data = _sanitize_sensitive_data(log_data)
        
        return log_data
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        log_data = self._prepare_log_data(**kwargs)
        
        if STRUCTLOG_AVAILABLE:
            self._logger.debug(message, **log_data)
        else:
            self._logger.debug(f"{message} | {log_data}")
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        log_data = self._prepare_log_data(**kwargs)
        
        if STRUCTLOG_AVAILABLE:
            self._logger.info(message, **log_data)
        else:
            self._logger.info(f"{message} | {log_data}")
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        log_data = self._prepare_log_data(**kwargs)
        
        if STRUCTLOG_AVAILABLE:
            self._logger.warning(message, **log_data)
        else:
            self._logger.warning(f"{message} | {log_data}")
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        log_data = self._prepare_log_data(**kwargs)
        
        if STRUCTLOG_AVAILABLE:
            self._logger.error(message, **log_data)
        else:
            self._logger.error(f"{message} | {log_data}")
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        log_data = self._prepare_log_data(**kwargs)
        
        if STRUCTLOG_AVAILABLE:
            self._logger.critical(message, **log_data)
        else:
            self._logger.critical(f"{message} | {log_data}")
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **kwargs
    ) -> None:
        """Log performance metrics."""
        if not self.config.enable_performance_logging:
            return
        
        log_data = self._prepare_log_data(
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            performance_log=True,
            **kwargs
        )
        
        # Log as warning if operation is slow
        is_slow = duration_ms > self.config.slow_operation_threshold_ms
        
        if is_slow:
            self.warning(f"Slow operation detected: {operation}", **log_data)
        else:
            self.info(f"Operation completed: {operation}", **log_data)
    
    def log_data_quality(
        self,
        dataset_name: str,
        check_name: str,
        result: str,
        score: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log data quality metrics."""
        log_data = self._prepare_log_data(
            dataset_name=dataset_name,
            check_name=check_name,
            result=result,
            score=score,
            data_quality_log=True,
            **kwargs
        )
        
        self.info(f"Data quality check: {check_name}", **log_data)
    
    def log_model_performance(
        self,
        model_name: str,
        metric_name: str,
        metric_value: float,
        dataset: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log model performance metrics."""
        log_data = self._prepare_log_data(
            model_name=model_name,
            metric_name=metric_name,
            metric_value=metric_value,
            dataset=dataset,
            model_performance_log=True,
            **kwargs
        )
        
        self.info(f"Model performance: {model_name}", **log_data)
    
    def log_business_metric(
        self,
        metric_name: str,
        metric_value: Union[int, float, str],
        metric_type: str = "counter",
        **kwargs
    ) -> None:
        """Log business metrics."""
        log_data = self._prepare_log_data(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_type=metric_type,
            business_metric_log=True,
            **kwargs
        )
        
        self.info(f"Business metric: {metric_name}", **log_data)


class PerformanceTimer:
    """Context manager for performance timing."""
    
    def __init__(
        self,
        logger: StructuredLogger,
        operation: str,
        **context
    ):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
        self.end_time = None
    
    def __enter__(self) -> 'PerformanceTimer':
        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting operation: {self.operation}", **self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        success = exc_type is None
        
        if not success:
            self.context.update({
                "error_type": exc_type.__name__ if exc_type else None,
                "error_message": str(exc_val) if exc_val else None
            })
        
        self.logger.log_performance(
            operation=self.operation,
            duration_ms=duration_ms,
            success=success,
            **self.context
        )


def setup_structlog(
    package_name: str,
    level: str = LogLevel.INFO,
    format_type: str = LogFormat.JSON,
    enable_console: bool = True,
    enable_file: bool = False,
    file_path: Optional[str] = None
) -> None:
    """Setup structlog with standardized configuration."""
    if not STRUCTLOG_AVAILABLE:
        return
    
    processors = [
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Add log level
        structlog.processors.add_log_level,
        # Add logger name
        structlog.processors.add_logger_name,
        # Add package name
        lambda logger, method_name, event_dict: {
            **event_dict,
            "package": package_name
        },
    ]
    
    # Add processors based on format
    if format_type == LogFormat.JSON:
        processors.extend([
            structlog.processors.JSONRenderer()
        ])
    elif format_type == LogFormat.CONSOLE:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    else:
        processors.extend([
            structlog.processors.KeyValueRenderer(key_order=['timestamp', 'level', 'logger', 'package'])
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Setup standard library integration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=[]
    )
    
    # Add console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        logging.getLogger().addHandler(console_handler)
    
    # Add file handler
    if enable_file and file_path:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            logging.getLogger().addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")


def create_logger(
    package_name: str,
    level: str = LogLevel.INFO,
    format_type: str = LogFormat.JSON,
    **kwargs
) -> StructuredLogger:
    """Create a structured logger for a package."""
    config = BaseLoggerConfig(
        package_name=package_name,
        level=level,
        format_type=format_type,
        **kwargs
    )
    
    return StructuredLogger(config)


def create_logger_from_env(package_name: str, env_prefix: str = "") -> StructuredLogger:
    """Create a logger from environment variables."""
    prefix = env_prefix or f"{package_name.upper()}_"
    
    config = BaseLoggerConfig(
        package_name=package_name,
        level=os.getenv(f"{prefix}LOG_LEVEL", LogLevel.INFO),
        format_type=os.getenv(f"{prefix}LOG_FORMAT", LogFormat.JSON),
        enable_console=os.getenv(f"{prefix}LOG_CONSOLE", "true").lower() == "true",
        enable_file=os.getenv(f"{prefix}LOG_FILE_ENABLED", "false").lower() == "true",
        file_path=os.getenv(f"{prefix}LOG_FILE_PATH"),
        max_file_size=os.getenv(f"{prefix}LOG_MAX_FILE_SIZE", "10MB"),
        backup_count=int(os.getenv(f"{prefix}LOG_BACKUP_COUNT", "5")),
        enable_structured=os.getenv(f"{prefix}LOG_STRUCTURED", "true").lower() == "true",
        enable_request_tracking=os.getenv(f"{prefix}LOG_REQUEST_TRACKING", "true").lower() == "true",
        enable_performance_logging=os.getenv(f"{prefix}LOG_PERFORMANCE", "true").lower() == "true",
        sanitize_sensitive_data=os.getenv(f"{prefix}LOG_SANITIZE_DATA", "true").lower() == "true",
        slow_operation_threshold_ms=float(os.getenv(f"{prefix}LOG_SLOW_THRESHOLD_MS", "1000"))
    )
    
    return StructuredLogger(config)


def log_decorator(
    operation: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    log_duration: bool = True,
    level: str = LogLevel.INFO
):
    """Decorator for automatic logging of function calls."""
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get logger from first argument if it's a StructuredLogger
            logger = None
            if args and isinstance(args[0], StructuredLogger):
                logger = args[0]
            else:
                # Create a default logger
                logger = create_logger(func.__module__ or "unknown")
            
            op_name = operation or f"{func.__module__}.{func.__name__}"
            
            log_data = {
                "function": func.__name__,
                "module": func.__module__
            }
            
            if log_args:
                log_data.update({
                    "args": args[1:] if isinstance(args[0], StructuredLogger) else args,
                    "kwargs": kwargs
                })
            
            with PerformanceTimer(logger, op_name, **log_data) as timer:
                result = func(*args, **kwargs)
                
                if log_result:
                    log_data["result"] = result
                
                return result
        
        return wrapper
    return decorator


@contextmanager
def log_context(logger: StructuredLogger, **context):
    """Context manager for temporary logging context."""
    original_context = logger._context.copy()
    
    try:
        logger.set_context(**context)
        yield logger
    finally:
        logger._context = original_context


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(package_name: str, **config_overrides) -> StructuredLogger:
    """Get or create a logger for a package."""
    if package_name not in _loggers:
        _loggers[package_name] = create_logger_from_env(package_name, **config_overrides)
    
    return _loggers[package_name]


def setup_package_logging(
    package_name: str,
    config: Optional[BaseLoggerConfig] = None,
    **config_kwargs
) -> StructuredLogger:
    """Setup logging for a package with optional configuration."""
    if config is None:
        config = BaseLoggerConfig(package_name=package_name, **config_kwargs)
    
    # Setup structlog if available
    if STRUCTLOG_AVAILABLE:
        setup_structlog(
            package_name=package_name,
            level=config.level,
            format_type=config.format_type,
            enable_console=config.enable_console,
            enable_file=config.enable_file,
            file_path=config.file_path
        )
    
    # Create and register logger
    logger = StructuredLogger(config)
    _loggers[package_name] = logger
    
    return logger


__all__ = [
    # Classes
    "StructuredLogger",
    "BaseLoggerConfig", 
    "PerformanceTimer",
    
    # Constants
    "LogLevel",
    "LogFormat",
    
    # Factory functions
    "create_logger",
    "create_logger_from_env",
    "get_logger",
    "setup_package_logging",
    "setup_structlog",
    
    # Decorators and context managers
    "log_decorator",
    "log_context",
    
    # Feature detection
    "STRUCTLOG_AVAILABLE"
]
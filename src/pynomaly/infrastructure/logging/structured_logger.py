"""Structured logging system with comprehensive context management."""

from __future__ import annotations

import logging
import sys
import time
import traceback
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog


class LogLevel(Enum):
    """Log levels with numeric values."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogContext:
    """Comprehensive logging context with automatic field management."""

    # Core identification
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    user_id: str | None = None
    request_id: str | None = None

    # Service context
    service_name: str = "pynomaly"
    service_version: str = "1.0.0"
    environment: str = "development"
    instance_id: str = field(default_factory=lambda: str(uuid4())[:8])

    # Request context
    endpoint: str | None = None
    method: str | None = None
    user_agent: str | None = None
    ip_address: str | None = None

    # Business context
    detector_id: str | None = None
    dataset_id: str | None = None
    experiment_id: str | None = None
    workflow_id: str | None = None

    # Performance context
    operation: str | None = None
    duration_ms: float | None = None
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None

    # Error context
    error_type: str | None = None
    error_code: str | None = None
    stack_trace: str | None = None

    # Custom fields
    custom_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if key == "custom_fields" and isinstance(value, dict):
                    result.update(value)
                else:
                    result[key] = value
        return result

    def add_custom(self, key: str, value: Any) -> LogContext:
        """Add custom field and return new context."""
        new_context = LogContext(**asdict(self))
        new_context.custom_fields[key] = value
        return new_context

    def with_operation(self, operation: str) -> LogContext:
        """Create new context with operation set."""
        new_context = LogContext(**asdict(self))
        new_context.operation = operation
        return new_context

    def with_performance(
        self,
        duration_ms: float,
        memory_usage_mb: float | None = None,
        cpu_usage_percent: float | None = None,
    ) -> LogContext:
        """Create new context with performance metrics."""
        new_context = LogContext(**asdict(self))
        new_context.duration_ms = duration_ms
        new_context.memory_usage_mb = memory_usage_mb
        new_context.cpu_usage_percent = cpu_usage_percent
        return new_context

    def with_error(
        self, error: Exception, error_code: str | None = None
    ) -> LogContext:
        """Create new context with error information."""
        new_context = LogContext(**asdict(self))
        new_context.error_type = type(error).__name__
        new_context.error_code = error_code
        new_context.stack_trace = traceback.format_exc()
        return new_context


# Context variable for storing current log context
_log_context: ContextVar[LogContext | None] = ContextVar("log_context", default=None)


class StructuredLogger:
    """Production-ready structured logger with comprehensive features."""

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        output_path: Path | None = None,
        enable_console: bool = True,
        enable_json: bool = True,
        max_file_size_mb: int = 100,
        backup_count: int = 5,
        include_caller_info: bool = True,
        sanitize_sensitive_data: bool = True,
        correlation_id_header: str = "X-Correlation-ID",
    ):
        """Initialize structured logger with comprehensive configuration.

        Args:
            name: Logger name (typically module or service name)
            level: Minimum log level to output
            output_path: Optional file path for log output
            enable_console: Whether to output to console
            enable_json: Whether to use JSON formatting
            max_file_size_mb: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            include_caller_info: Whether to include caller information
            sanitize_sensitive_data: Whether to sanitize sensitive data
            correlation_id_header: Header name for correlation ID
        """
        self.name = name
        self.level = level
        self.output_path = output_path
        self.enable_console = enable_console
        self.enable_json = enable_json
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        self.include_caller_info = include_caller_info
        self.sanitize_sensitive_data = sanitize_sensitive_data
        self.correlation_id_header = correlation_id_header

        # Sensitive field patterns for sanitization
        self.sensitive_patterns = {
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "key",
            "auth",
            "credential",
            "pin",
            "ssn",
            "credit_card",
            "api_key",
            "private_key",
        }

        # Performance metrics
        self.metrics = {
            "logs_written": 0,
            "errors_logged": 0,
            "warnings_logged": 0,
            "performance_logs": 0,
            "sanitized_fields": 0,
        }

        # Initialize structlog
        self._setup_structlog()
        self._logger = structlog.get_logger(name)

    def _setup_structlog(self):
        """Configure structlog with comprehensive processors."""
        processors = [
            # Add timestamp
            structlog.processors.TimeStamper(fmt="ISO"),
            # Add log level
            structlog.processors.add_log_level,
            # Add logger name
            structlog.processors.add_logger_name,
            # Add caller info if enabled
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            )
            if self.include_caller_info
            else None,
            # Process context variables
            self._context_processor,
            # Sanitize sensitive data
            self._sanitizer_processor if self.sanitize_sensitive_data else None,
            # Format for output
            structlog.processors.JSONRenderer()
            if self.enable_json
            else structlog.dev.ConsoleRenderer(),
        ]

        # Remove None processors
        processors = [p for p in processors if p is not None]

        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=self._create_logger_factory(),
            cache_logger_on_first_use=True,
        )

    def _create_logger_factory(self):
        """Create logger factory with file and console handlers."""

        def factory(name):
            logger = logging.getLogger(name)
            logger.setLevel(self.level.value)

            # Clear existing handlers
            logger.handlers.clear()

            # Console handler
            if self.enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(self.level.value)
                logger.addHandler(console_handler)

            # File handler with rotation
            if self.output_path:
                from logging.handlers import RotatingFileHandler

                # Ensure directory exists
                self.output_path.parent.mkdir(parents=True, exist_ok=True)

                file_handler = RotatingFileHandler(
                    self.output_path,
                    maxBytes=self.max_file_size_mb * 1024 * 1024,
                    backupCount=self.backup_count,
                )
                file_handler.setLevel(self.level.value)
                logger.addHandler(file_handler)

            return logger

        return factory

    def _context_processor(self, logger, method_name, event_dict):
        """Add current log context to event dict."""
        context = self.get_current_context()
        if context:
            event_dict.update(context.to_dict())
        return event_dict

    def _sanitizer_processor(self, logger, method_name, event_dict):
        """Sanitize sensitive data from log entries."""
        sanitized_dict = {}

        for key, value in event_dict.items():
            if any(pattern in key.lower() for pattern in self.sensitive_patterns):
                sanitized_dict[key] = "***REDACTED***"
                self.metrics["sanitized_fields"] += 1
            elif isinstance(value, dict):
                sanitized_dict[key] = self._sanitize_dict(value)
            elif isinstance(value, str) and len(value) > 100:
                # Truncate very long strings to prevent log bloat
                sanitized_dict[key] = value[:100] + "..." if len(value) > 100 else value
            else:
                sanitized_dict[key] = value

        return sanitized_dict

    def _sanitize_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Recursively sanitize dictionary."""
        sanitized = {}
        for key, value in data.items():
            if any(pattern in key.lower() for pattern in self.sensitive_patterns):
                sanitized[key] = "***REDACTED***"
                self.metrics["sanitized_fields"] += 1
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            else:
                sanitized[key] = value
        return sanitized

    @classmethod
    def set_context(cls, context: LogContext):
        """Set the current log context."""
        _log_context.set(context)

    @classmethod
    def get_current_context(cls) -> LogContext | None:
        """Get the current log context."""
        return _log_context.get()

    @classmethod
    def clear_context(cls):
        """Clear the current log context."""
        _log_context.set(None)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)
        self.metrics["warnings_logged"] += 1

    def error(self, message: str, error: Exception | None = None, **kwargs):
        """Log error message with optional exception."""
        if error:
            kwargs.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "stack_trace": traceback.format_exc(),
                }
            )

        self._log(LogLevel.ERROR, message, **kwargs)
        self.metrics["errors_logged"] += 1

    def critical(self, message: str, error: Exception | None = None, **kwargs):
        """Log critical message with optional exception."""
        if error:
            kwargs.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "stack_trace": traceback.format_exc(),
                }
            )

        self._log(LogLevel.CRITICAL, message, **kwargs)
        self.metrics["errors_logged"] += 1

    def performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics."""
        kwargs.update(
            {
                "operation": operation,
                "duration_ms": duration_ms,
                "performance_log": True,
            }
        )

        self._log(LogLevel.INFO, f"Performance: {operation}", **kwargs)
        self.metrics["performance_logs"] += 1

    def audit(
        self,
        action: str,
        user_id: str | None = None,
        resource: str | None = None,
        **kwargs,
    ):
        """Log audit trail."""
        kwargs.update(
            {
                "audit_log": True,
                "action": action,
                "user_id": user_id,
                "resource": resource,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        self._log(LogLevel.INFO, f"Audit: {action}", **kwargs)

    def security(self, event: str, severity: str = "medium", **kwargs):
        """Log security events."""
        kwargs.update(
            {
                "security_log": True,
                "security_event": event,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        level = LogLevel.WARNING if severity in ["medium", "high"] else LogLevel.INFO
        self._log(level, f"Security: {event}", **kwargs)

    def business(self, event: str, metric_value: float | None = None, **kwargs):
        """Log business metrics and events."""
        kwargs.update(
            {
                "business_log": True,
                "business_event": event,
                "metric_value": metric_value,
            }
        )

        self._log(LogLevel.INFO, f"Business: {event}", **kwargs)

    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal logging method."""
        if level.value < self.level.value:
            return

        # Add metadata
        kwargs.update(
            {"logger_name": self.name, "log_level": level.name, "message": message}
        )

        # Get logger method
        method_name = level.name.lower()
        log_method = getattr(self._logger, method_name)

        # Log the message
        log_method(message, **kwargs)
        self.metrics["logs_written"] += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get logger performance metrics."""
        return {
            "logger_name": self.name,
            "logs_written": self.metrics["logs_written"],
            "errors_logged": self.metrics["errors_logged"],
            "warnings_logged": self.metrics["warnings_logged"],
            "performance_logs": self.metrics["performance_logs"],
            "sanitized_fields": self.metrics["sanitized_fields"],
            "level": self.level.name,
            "output_path": str(self.output_path) if self.output_path else None,
        }


class PerformanceLogger:
    """Context manager for automatic performance logging."""

    def __init__(
        self,
        logger: StructuredLogger,
        operation: str,
        context: LogContext | None = None,
        log_args: bool = False,
        log_result: bool = False,
        min_duration_ms: float = 0,
    ):
        """Initialize performance logger.

        Args:
            logger: Structured logger instance
            operation: Operation name
            context: Optional log context
            log_args: Whether to log function arguments
            log_result: Whether to log function result
            min_duration_ms: Minimum duration to log (filter out fast operations)
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.log_args = log_args
        self.log_result = log_result
        self.min_duration_ms = min_duration_ms
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start performance measurement."""
        if self.context:
            StructuredLogger.set_context(self.context)

        self.start_time = time.perf_counter()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End performance measurement and log results."""
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000

        # Only log if duration exceeds minimum threshold
        if duration_ms >= self.min_duration_ms:
            extra_data = {}

            if exc_type:
                extra_data["error"] = str(exc_val)
                extra_data["error_type"] = exc_type.__name__
                self.logger.error(
                    f"Operation failed: {self.operation}",
                    duration_ms=duration_ms,
                    **extra_data,
                )
            else:
                self.logger.performance(self.operation, duration_ms, **extra_data)

        # Clear context if we set it
        if self.context:
            StructuredLogger.clear_context()


def performance_logger(
    operation: str,
    logger: StructuredLogger | None = None,
    log_args: bool = False,
    log_result: bool = False,
    min_duration_ms: float = 0,
):
    """Decorator for automatic performance logging."""

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Get logger (use provided or create default)
            nonlocal logger
            if logger is None:
                logger = StructuredLogger(func.__module__)

            # Create performance logger
            perf_logger = PerformanceLogger(
                logger=logger,
                operation=f"{func.__module__}.{func.__name__}",
                log_args=log_args,
                log_result=log_result,
                min_duration_ms=min_duration_ms,
            )

            with perf_logger:
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global logger registry
_loggers: dict[str, StructuredLogger] = {}


def get_logger(
    name: str, level: LogLevel = LogLevel.INFO, **kwargs
) -> StructuredLogger:
    """Get or create a structured logger instance."""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, level, **kwargs)
    return _loggers[name]


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    output_dir: Path | None = None,
    enable_console: bool = True,
    enable_json: bool = True,
    service_name: str = "pynomaly",
    service_version: str = "1.0.0",
    environment: str = "development",
):
    """Configure global logging settings."""
    # Set default context
    default_context = LogContext(
        service_name=service_name,
        service_version=service_version,
        environment=environment,
    )
    StructuredLogger.set_context(default_context)

    # Configure root logger
    root_logger = get_logger(
        "pynomaly",
        level=level,
        output_path=output_dir / "pynomaly.log" if output_dir else None,
        enable_console=enable_console,
        enable_json=enable_json,
    )

    return root_logger

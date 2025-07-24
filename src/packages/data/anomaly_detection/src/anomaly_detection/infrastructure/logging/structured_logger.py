"""Structured logging configuration and utilities."""

from __future__ import annotations

import logging
import structlog
from typing import Any, Dict, Optional, List
from datetime import datetime
from pathlib import Path
import json

from ..config.settings import get_settings


class StructuredLogger:
    """Enhanced structured logger with context management."""
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        """Initialize structured logger with optional context.
        
        Args:
            name: Logger name (usually module or class name)
            context: Default context to include in all log entries
        """
        self.name = name
        self.base_context = context or {}
        self._logger = structlog.get_logger(name)
        self._request_context = {}
    
    def with_context(self, **kwargs: Any) -> StructuredLogger:
        """Create a new logger instance with additional context.
        
        Args:
            **kwargs: Additional context key-value pairs
            
        Returns:
            New StructuredLogger instance with merged context
        """
        new_context = {**self.base_context, **kwargs}
        return StructuredLogger(self.name, new_context)
    
    def set_request_context(self, request_id: str, user_id: Optional[str] = None, **kwargs: Any) -> None:
        """Set request-specific context for tracking operations.
        
        Args:
            request_id: Unique request identifier
            user_id: Optional user identifier
            **kwargs: Additional request context
        """
        self._request_context = {
            "request_id": request_id,
            "user_id": user_id,
            **kwargs
        }
    
    def clear_request_context(self) -> None:
        """Clear request-specific context."""
        self._request_context = {}
    
    def _merge_context(self, **kwargs: Any) -> Dict[str, Any]:
        """Merge all context sources."""
        merged = {
            **self.base_context,
            **self._request_context,
            **kwargs,
            "timestamp": datetime.utcnow().isoformat(),
            "logger_name": self.name
        }
        return merged
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._logger.debug(message, **self._merge_context(**kwargs))
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._logger.info(message, **self._merge_context(**kwargs))
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._logger.warning(message, **self._merge_context(**kwargs))
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._logger.error(message, **self._merge_context(**kwargs))
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._logger.critical(message, **self._merge_context(**kwargs))
    
    def log_operation_start(self, operation: str, **kwargs: Any) -> None:
        """Log the start of an operation."""
        self.info(f"Starting operation: {operation}", 
                 operation=operation, 
                 phase="start",
                 **kwargs)
    
    def log_operation_end(self, operation: str, duration_ms: float, success: bool = True, **kwargs: Any) -> None:
        """Log the end of an operation."""
        level_method = self.info if success else self.error
        level_method(f"{'Completed' if success else 'Failed'} operation: {operation}",
                    operation=operation,
                    phase="end",
                    duration_ms=duration_ms,
                    success=success,
                    **kwargs)
    
    def log_metric(self, name: str, value: float, unit: Optional[str] = None, **kwargs: Any) -> None:
        """Log a metric value."""
        self.info(f"Metric recorded: {name}",
                 metric_name=name,
                 metric_value=value,
                 metric_unit=unit,
                 metric=True,
                 **kwargs)
    
    def log_data_quality(self, dataset_name: str, quality_metrics: Dict[str, Any], **kwargs: Any) -> None:
        """Log data quality metrics."""
        self.info(f"Data quality metrics for {dataset_name}",
                 dataset_name=dataset_name,
                 data_quality=True,
                 **quality_metrics,
                 **kwargs)
    
    def log_model_performance(self, model_id: str, metrics: Dict[str, float], **kwargs: Any) -> None:
        """Log model performance metrics."""
        self.info(f"Model performance recorded for {model_id}",
                 model_id=model_id,
                 model_performance=True,
                 **{f"perf_{k}": v for k, v in metrics.items()},
                 **kwargs)


class LoggerFactory:
    """Factory for creating configured loggers."""
    
    _configured = False
    
    @classmethod
    def configure_logging(cls, settings: Optional[Any] = None) -> None:
        """Configure global logging settings."""
        if cls._configured:
            return
        
        if settings is None:
            settings = get_settings()
        
        # Configure processors
        processors: List[Any] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]
        
        # Add JSON formatter for production, pretty formatter for development
        if settings.environment == "production":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=None,
            level=getattr(logging, settings.logging.level.upper())
        )
        
        # Configure file logging if enabled and file_path is provided
        if settings.logging.file_enabled and settings.logging.file_path:
            file_handler = logging.FileHandler(settings.logging.file_path)
            file_handler.setLevel(getattr(logging, settings.logging.level.upper()))
            
            if settings.environment == "production":
                file_formatter = logging.Formatter('%(message)s')  # JSON already formatted
            else:
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            file_handler.setFormatter(file_formatter)
            logging.getLogger().addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str, context: Optional[Dict[str, Any]] = None) -> StructuredLogger:
        """Get a configured structured logger.
        
        Args:
            name: Logger name
            context: Optional default context
            
        Returns:
            Configured StructuredLogger instance
        """
        cls.configure_logging()
        return StructuredLogger(name, context)


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """Convenience function to get a structured logger.
    
    Args:
        name: Logger name
        context: Optional default context
        
    Returns:
        Configured StructuredLogger instance
    """
    return LoggerFactory.get_logger(name, context)


class RequestLoggingMiddleware:
    """Middleware for request-level logging context."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def __call__(self, request_id: str, user_id: Optional[str] = None):
        """Set request context and return a context manager."""
        return RequestContext(self.logger, request_id, user_id)


class RequestContext:
    """Context manager for request-scoped logging."""
    
    def __init__(self, logger: StructuredLogger, request_id: str, user_id: Optional[str] = None):
        self.logger = logger
        self.request_id = request_id
        self.user_id = user_id
    
    def __enter__(self) -> StructuredLogger:
        self.logger.set_request_context(self.request_id, self.user_id)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.logger.clear_request_context()


def configure_logging(settings: Optional[Any] = None) -> None:
    """Configure global logging settings."""
    LoggerFactory.configure_logging(settings)


# Export common functions
__all__ = [
    "configure_logging",
    "get_logger",
    "StructuredLogger",
    "LoggerFactory", 
    "RequestLoggingMiddleware",
    "RequestContext"
]
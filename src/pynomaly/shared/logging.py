"""Structured logging configuration for PyNomaly."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    *,
    level: str = "INFO",
    json_logs: bool = True,
    development: bool = False,
    service_name: str = "pynomaly",
    service_version: str = "unknown",
) -> None:
    """Configure structured logging for PyNomaly.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Whether to output JSON logs
        development: Whether in development mode (pretty prints)
        service_name: Name of the service
        service_version: Version of the service
    """
    # Set up stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.dict_tracebacks,
        # Add service information
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]
    
    if development:
        # Pretty console output for development
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add service context
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        service_name=service_name,
        service_version=service_version,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables to all future log messages.
    
    Args:
        **kwargs: Context variables to bind
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()


def with_context(**kwargs: Any) -> Any:
    """Decorator to bind context for the duration of a function.
    
    Args:
        **kwargs: Context variables to bind
        
    Returns:
        Decorator function
    """
    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **func_kwargs: Any) -> Any:
            # Store current context
            old_context = structlog.contextvars.get_contextvars()
            
            try:
                # Bind new context
                structlog.contextvars.bind_contextvars(**kwargs)
                return func(*args, **func_kwargs)
            finally:
                # Restore old context
                structlog.contextvars.clear_contextvars()
                structlog.contextvars.bind_contextvars(**old_context)
        
        return wrapper
    return decorator


class LoggingMixin:
    """Mixin class to add structured logging to domain entities."""
    
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Set up logger for subclasses."""
        super().__init_subclass__(**kwargs)
        cls._logger = get_logger(f"{cls.__module__}.{cls.__name__}")
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get the structured logger for this instance."""
        return self._logger.bind(
            entity_id=getattr(self, "id", None),
            entity_type=self.__class__.__name__,
        )

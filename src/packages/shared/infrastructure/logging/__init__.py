"""Logging configuration and utilities."""

from .structured_logger import StructuredLogger
from .log_decorator import log_decorator, timing_decorator, async_log_decorator
from .error_handler import ErrorHandler, AnomalyDetectionError

# Create convenience function for getting logger
def get_logger(name: str = __name__) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)

# Setup logging configuration
def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    import structlog
    import logging
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=getattr(logging, level.upper())
    )

__all__ = [
    "StructuredLogger",
    "log_decorator", 
    "timing_decorator",
    "async_log_decorator",
    "ErrorHandler",
    "AnomalyDetectionError",
    "get_logger",
    "setup_logging"
]
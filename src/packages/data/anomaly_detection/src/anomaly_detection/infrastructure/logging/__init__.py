"""Logging configuration and utilities."""

from .structured_logger import StructuredLogger
from .log_decorator import log_decorator, timing_decorator
from .error_handler import ErrorHandler, AnomalyDetectionError

# Create convenience function for getting logger
def get_logger(name: str = __name__) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)

__all__ = [
    "StructuredLogger",
    "log_decorator", 
    "timing_decorator",
    "ErrorHandler",
    "AnomalyDetectionError",
    "get_logger"
]
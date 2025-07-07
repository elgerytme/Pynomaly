"""Comprehensive logging configuration for Pynomaly."""

import json
import logging
import logging.config
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from pythonjsonlogger import jsonlogger

# Context variables for correlation tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class CorrelationFilter(logging.Filter):
    """Add correlation IDs to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation context to log record."""
        record.correlation_id = correlation_id.get()
        record.user_id = user_id.get()
        record.request_id = request_id.get()
        return True


class JSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['timestamp'] = self.formatTime(record, self.datefmt)
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add correlation fields if available
        if hasattr(record, 'correlation_id') and record.correlation_id:
            log_record['correlation_id'] = record.correlation_id
        if hasattr(record, 'user_id') and record.user_id:
            log_record['user_id'] = record.user_id
        if hasattr(record, 'request_id') and record.request_id:
            log_record['request_id'] = record.request_id
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


class StructlogProcessor:
    """Structlog processor for consistent formatting."""
    
    @staticmethod
    def add_correlation_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add correlation context to structlog events."""
        if correlation_id.get():
            event_dict['correlation_id'] = correlation_id.get()
        if user_id.get():
            event_dict['user_id'] = user_id.get()
        if request_id.get():
            event_dict['request_id'] = request_id.get()
        return event_dict


def configure_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    enable_correlation: bool = True,
    enable_structlog: bool = True,
) -> None:
    """Configure comprehensive logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ('json' or 'text')
        log_file: Optional log file path
        enable_correlation: Enable correlation ID tracking
        enable_structlog: Enable structured logging with structlog
    """
    # Create logs directory if using file logging
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define formatters
    if log_format == "json":
        formatter = JSONFormatter(
            fmt='%(timestamp)s %(level)s %(logger)s %(module)s %(function)s %(line)d %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Define handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    if enable_correlation:
        console_handler.addFilter(CorrelationFilter())
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        if enable_correlation:
            file_handler.addFilter(CorrelationFilter())
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True
    )
    
    # Configure third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Configure structlog if enabled
    if enable_structlog:
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        if enable_correlation:
            processors.insert(0, StructlogProcessor.add_correlation_context)
        
        if log_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def get_structured_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger instance
    """
    return structlog.get_logger(name)


def set_correlation_context(
    correlation_id_value: Optional[str] = None,
    user_id_value: Optional[str] = None,
    request_id_value: Optional[str] = None,
) -> None:
    """Set correlation context for request tracking.
    
    Args:
        correlation_id_value: Correlation ID for the request
        user_id_value: User ID associated with the request
        request_id_value: Request ID for the specific request
    """
    if correlation_id_value:
        correlation_id.set(correlation_id_value)
    if user_id_value:
        user_id.set(user_id_value)
    if request_id_value:
        request_id.set(request_id_value)


def generate_correlation_id() -> str:
    """Generate a new correlation ID.
    
    Returns:
        New UUID-based correlation ID
    """
    return str(uuid.uuid4())


def clear_correlation_context() -> None:
    """Clear all correlation context variables."""
    correlation_id.set(None)
    user_id.set(None)
    request_id.set(None)


# Logging configuration presets
LOGGING_CONFIGS = {
    "development": {
        "level": "DEBUG",
        "log_format": "text",
        "enable_correlation": True,
        "enable_structlog": True,
    },
    "staging": {
        "level": "INFO",
        "log_format": "json",
        "log_file": "logs/pynomaly-staging.log",
        "enable_correlation": True,
        "enable_structlog": True,
    },
    "production": {
        "level": "INFO",
        "log_format": "json",
        "log_file": "logs/pynomaly-production.log",
        "enable_correlation": True,
        "enable_structlog": True,
    },
    "testing": {
        "level": "WARNING",
        "log_format": "text",
        "enable_correlation": False,
        "enable_structlog": False,
    },
}


def configure_logging_for_environment(environment: str = "development") -> None:
    """Configure logging for a specific environment.
    
    Args:
        environment: Environment name (development, staging, production, testing)
    """
    config = LOGGING_CONFIGS.get(environment, LOGGING_CONFIGS["development"])
    configure_logging(**config)
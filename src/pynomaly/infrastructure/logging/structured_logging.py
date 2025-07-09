"""
Structured logging configuration with trace IDs using structlog.

This module provides JSON-formatted structured logging with trace correlation
and comprehensive observability features.
"""

import json
import logging
import logging.config
import os
import sys
import uuid
from typing import Any, Dict, Optional

try:
    import structlog
    from opentelemetry import trace
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None
    trace = None


class TraceIDProcessor:
    """Processor that adds trace and span IDs to log entries."""
    
    def __call__(self, logger, method_name, event_dict):
        """Add trace and span IDs to log entry."""
        if not STRUCTLOG_AVAILABLE:
            return event_dict
            
        try:
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                span_context = span.get_span_context()
                event_dict["trace_id"] = f"{span_context.trace_id:032x}"
                event_dict["span_id"] = f"{span_context.span_id:016x}"
        except Exception:
            pass  # Ignore tracing errors
        
        return event_dict


class ServiceContextProcessor:
    """Processor that adds service context information."""
    
    def __init__(self, service_name: str = "pynomaly", service_version: str = "1.0.0"):
        self.service_name = service_name
        self.service_version = service_version
    
    def __call__(self, logger, method_name, event_dict):
        """Add service context to log entry."""
        event_dict["service"] = {
            "name": self.service_name,
            "version": self.service_version,
            "instance_id": os.environ.get("INSTANCE_ID", "unknown"),
            "environment": os.environ.get("ENVIRONMENT", "development"),
        }
        return event_dict


def initialize_logging(
    service_name: str = "pynomaly",
    service_version: str = "1.0.0",
    log_level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    log_file: str = "logs/pynomaly.log",
) -> None:
    """Setup structured logging for the application."""
    
    if not STRUCTLOG_AVAILABLE:
        # Fallback to standard logging if structlog is not available
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file) if enable_file else logging.NullHandler()
            ]
        )
        return
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            TraceIDProcessor(),
            ServiceContextProcessor(service_name, service_version),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    if enable_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": True,
            },
        },
    }
    
    if enable_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
        }
        config["loggers"][""]["handlers"].append("file")
    
    logging.config.dictConfig(config)


def get_logger(name: str):
    """Get a logger instance."""
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


# Global logger instance
_logger_instance = None


def get_logger_instance():
    """Get the global logger instance."""
    return _logger_instance

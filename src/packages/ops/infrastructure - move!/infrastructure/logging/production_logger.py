#!/usr/bin/env python3
"""
Production Logging Infrastructure for Pynomaly

This module provides comprehensive logging capabilities for production
environments including structured logging, log aggregation, and monitoring.
"""

import json
import logging
import logging.handlers
import os
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog


class LogLevel(Enum):
    """Log level enumeration."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Log format enumeration."""

    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"


@dataclass
class LogContext:
    """Log context information."""

    service_name: str
    version: str
    environment: str
    hostname: str
    process_id: int
    thread_id: int
    correlation_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None


class StructuredFormatter(logging.Formatter):
    """Custom structured log formatter."""

    def __init__(self, context: LogContext):
        super().__init__()
        self.context = context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "@timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.context.service_name,
            "version": self.context.version,
            "environment": self.context.environment,
            "hostname": self.context.hostname,
            "process_id": self.context.process_id,
            "thread_id": record.thread,
            "thread_name": record.threadName,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "pathname": record.pathname,
        }

        # Add context information
        if self.context.correlation_id:
            log_entry["correlation_id"] = self.context.correlation_id
        if self.context.user_id:
            log_entry["user_id"] = self.context.user_id
        if self.context.session_id:
            log_entry["session_id"] = self.context.session_id
        if self.context.request_id:
            log_entry["request_id"] = self.context.request_id

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from log record
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
            }:
                if not key.startswith("_"):
                    log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class MetricsHandler(logging.Handler):
    """Handler to collect logging metrics."""

    def __init__(self):
        super().__init__()
        self.metrics = {
            "total_logs": 0,
            "logs_by_level": {level.value: 0 for level in LogLevel},
            "logs_by_logger": {},
            "error_count": 0,
            "warning_count": 0,
            "last_error": None,
            "last_warning": None,
        }
        self.lock = threading.Lock()

    def emit(self, record: logging.LogRecord):
        """Emit log record and update metrics."""
        with self.lock:
            self.metrics["total_logs"] += 1

            # Count by level
            level_name = record.levelname
            if level_name in self.metrics["logs_by_level"]:
                self.metrics["logs_by_level"][level_name] += 1

            # Count by logger
            logger_name = record.name
            if logger_name not in self.metrics["logs_by_logger"]:
                self.metrics["logs_by_logger"][logger_name] = 0
            self.metrics["logs_by_logger"][logger_name] += 1

            # Track errors and warnings
            if record.levelno >= logging.ERROR:
                self.metrics["error_count"] += 1
                self.metrics["last_error"] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": record.getMessage(),
                    "logger": record.name,
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }
            elif record.levelno >= logging.WARNING:
                self.metrics["warning_count"] += 1
                self.metrics["last_warning"] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": record.getMessage(),
                    "logger": record.name,
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }

    def get_metrics(self) -> dict:
        """Get current logging metrics."""
        with self.lock:
            return self.metrics.copy()

    def reset_metrics(self):
        """Reset logging metrics."""
        with self.lock:
            self.metrics = {
                "total_logs": 0,
                "logs_by_level": {level.value: 0 for level in LogLevel},
                "logs_by_logger": {},
                "error_count": 0,
                "warning_count": 0,
                "last_error": None,
                "last_warning": None,
            }


class SyslogHandler(logging.handlers.SysLogHandler):
    """Enhanced syslog handler with JSON support."""

    def __init__(
        self,
        address=("localhost", 514),
        facility=logging.handlers.SysLogHandler.LOG_USER,
    ):
        super().__init__(address, facility)
        self.formatter = None

    def emit(self, record):
        """Emit log record to syslog."""
        try:
            msg = self.format(record)
            # Remove newlines as syslog doesn't handle them well
            msg = msg.replace("\n", " ").replace("\r", " ")
            self.socket.send(msg.encode("utf-8"))
        except Exception:
            self.handleError(record)


class CloudWatchHandler(logging.Handler):
    """Handler for AWS CloudWatch Logs."""

    def __init__(self, log_group: str, log_stream: str, region: str = "us-east-1"):
        super().__init__()
        self.log_group = log_group
        self.log_stream = log_stream
        self.region = region
        self.sequence_token = None
        self.buffer = []
        self.buffer_size = 100
        self.flush_interval = 5.0  # seconds
        self.lock = threading.Lock()

        # Start flush timer
        self.timer = threading.Timer(self.flush_interval, self._flush_buffer)
        self.timer.daemon = True
        self.timer.start()

    def emit(self, record: logging.LogRecord):
        """Buffer log record for CloudWatch."""
        with self.lock:
            log_event = {
                "timestamp": int(
                    record.created * 1000
                ),  # CloudWatch expects milliseconds
                "message": self.format(record),
            }
            self.buffer.append(log_event)

            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffered logs to CloudWatch."""
        with self.lock:
            if not self.buffer:
                return

            try:
                # This would integrate with boto3 CloudWatch Logs client
                # For now, we'll simulate the flush
                print(
                    f"Flushing {len(self.buffer)} logs to CloudWatch {self.log_group}/{self.log_stream}"
                )
                self.buffer.clear()
            except Exception as e:
                print(f"Error flushing to CloudWatch: {e}")

        # Restart timer
        self.timer = threading.Timer(self.flush_interval, self._flush_buffer)
        self.timer.daemon = True
        self.timer.start()

    def close(self):
        """Close handler and flush remaining logs."""
        self._flush_buffer()
        super().close()


class ElasticsearchHandler(logging.Handler):
    """Handler for Elasticsearch/OpenSearch."""

    def __init__(self, hosts: list[str], index_prefix: str = "pynomaly-logs"):
        super().__init__()
        self.hosts = hosts
        self.index_prefix = index_prefix
        self.buffer = []
        self.buffer_size = 50
        self.flush_interval = 10.0  # seconds
        self.lock = threading.Lock()

        # Start flush timer
        self.timer = threading.Timer(self.flush_interval, self._flush_buffer)
        self.timer.daemon = True
        self.timer.start()

    def emit(self, record: logging.LogRecord):
        """Buffer log record for Elasticsearch."""
        with self.lock:
            # Create index name with date rotation
            index_name = f"{self.index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}"

            doc = {"_index": index_name, "_source": json.loads(self.format(record))}
            self.buffer.append(doc)

            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffered logs to Elasticsearch."""
        with self.lock:
            if not self.buffer:
                return

            try:
                # This would integrate with elasticsearch client
                print(f"Bulk indexing {len(self.buffer)} logs to Elasticsearch")
                self.buffer.clear()
            except Exception as e:
                print(f"Error indexing to Elasticsearch: {e}")

        # Restart timer
        self.timer = threading.Timer(self.flush_interval, self._flush_buffer)
        self.timer.daemon = True
        self.timer.start()

    def close(self):
        """Close handler and flush remaining logs."""
        self._flush_buffer()
        super().close()


class ProductionLogger:
    """Production logging manager."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.context = LogContext(
            service_name=config.get("service_name", "monorepo"),
            version=config.get("version", "1.0.0"),
            environment=config.get("environment", "production"),
            hostname=socket.gethostname(),
            process_id=os.getpid(),
            thread_id=threading.get_ident(),
        )

        self.metrics_handler = MetricsHandler()
        self.handlers = []
        self.loggers = {}

        # Configure structured logging
        self._configure_structlog()

        # Set up handlers
        self._setup_handlers()

        # Configure root logger
        self._configure_root_logger()

    def _configure_structlog(self):
        """Configure structured logging with structlog."""

        def add_timestamp(logger, method_name, event_dict):
            event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
            return event_dict

        def add_context(logger, method_name, event_dict):
            event_dict.update(asdict(self.context))
            return event_dict

        def add_level(logger, method_name, event_dict):
            event_dict["level"] = method_name.upper()
            return event_dict

        structlog.configure(
            processors=[
                add_timestamp,
                add_context,
                add_level,
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    ]
                ),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _setup_handlers(self):
        """Set up log handlers based on configuration."""
        # Console handler
        if self.config.get("console_logging", True):
            console_handler = logging.StreamHandler(sys.stdout)

            if self.config.get("log_format") == "json":
                console_handler.setFormatter(StructuredFormatter(self.context))
            else:
                console_format = (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "[%(process)d:%(thread)d] - %(message)s"
                )
                console_handler.setFormatter(logging.Formatter(console_format))

            self.handlers.append(console_handler)

        # File handler with rotation
        if self.config.get("file_logging", True):
            log_dir = Path(self.config.get("log_directory", "/var/log/pynomaly"))
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"{self.context.service_name}.log"

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get("max_log_size", 100 * 1024 * 1024),  # 100MB
                backupCount=self.config.get("backup_count", 10),
            )
            file_handler.setFormatter(StructuredFormatter(self.context))
            self.handlers.append(file_handler)

        # Syslog handler
        if self.config.get("syslog_logging", False):
            syslog_config = self.config.get("syslog", {})
            syslog_handler = SyslogHandler(
                address=(
                    syslog_config.get("host", "localhost"),
                    syslog_config.get("port", 514),
                )
            )
            syslog_handler.setFormatter(StructuredFormatter(self.context))
            self.handlers.append(syslog_handler)

        # CloudWatch handler
        if self.config.get("cloudwatch_logging", False):
            cloudwatch_config = self.config.get("cloudwatch", {})
            cloudwatch_handler = CloudWatchHandler(
                log_group=cloudwatch_config.get("log_group", "pynomaly-logs"),
                log_stream=cloudwatch_config.get(
                    "log_stream", f"{self.context.hostname}-{self.context.process_id}"
                ),
                region=cloudwatch_config.get("region", "us-east-1"),
            )
            cloudwatch_handler.setFormatter(StructuredFormatter(self.context))
            self.handlers.append(cloudwatch_handler)

        # Elasticsearch handler
        if self.config.get("elasticsearch_logging", False):
            elasticsearch_config = self.config.get("elasticsearch", {})
            elasticsearch_handler = ElasticsearchHandler(
                hosts=elasticsearch_config.get("hosts", ["localhost:9200"]),
                index_prefix=elasticsearch_config.get("index_prefix", "pynomaly-logs"),
            )
            elasticsearch_handler.setFormatter(StructuredFormatter(self.context))
            self.handlers.append(elasticsearch_handler)

        # Always add metrics handler
        self.handlers.append(self.metrics_handler)

    def _configure_root_logger(self):
        """Configure root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add configured handlers
        for handler in self.handlers:
            root_logger.addHandler(handler)

    def get_logger(self, name: str) -> logging.Logger:
        """Get logger instance."""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, self.config.get("log_level", "INFO")))
            self.loggers[name] = logger

        return self.loggers[name]

    def get_structured_logger(self, name: str):
        """Get structured logger instance."""
        return structlog.get_logger(name)

    def update_context(self, **kwargs):
        """Update logging context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)

        # Update formatter context
        for handler in self.handlers:
            if isinstance(handler.formatter, StructuredFormatter):
                handler.formatter.context = self.context

    def get_metrics(self) -> dict:
        """Get logging metrics."""
        return self.metrics_handler.get_metrics()

    def reset_metrics(self):
        """Reset logging metrics."""
        self.metrics_handler.reset_metrics()

    def health_check(self) -> dict:
        """Perform logging system health check."""
        health = {
            "status": "healthy",
            "handlers": len(self.handlers),
            "loggers": len(self.loggers),
            "context": asdict(self.context),
            "metrics": self.get_metrics(),
            "issues": [],
        }

        # Check handler health
        for i, handler in enumerate(self.handlers):
            try:
                # Try to emit a test log
                test_record = logging.LogRecord(
                    name="health_check",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg="Health check test",
                    args=(),
                    exc_info=None,
                )
                handler.handle(test_record)
            except Exception as e:
                health["issues"].append(
                    f"Handler {i} ({type(handler).__name__}): {str(e)}"
                )
                health["status"] = "degraded"

        return health

    def close(self):
        """Close all handlers."""
        for handler in self.handlers:
            handler.close()


# Global logger instance
_production_logger = None


def initialize_production_logging(config: dict[str, Any]) -> ProductionLogger:
    """Initialize production logging system."""
    global _production_logger
    _production_logger = ProductionLogger(config)
    return _production_logger


def get_production_logger(name: str = None) -> logging.Logger:
    """Get production logger instance."""
    if _production_logger is None:
        raise RuntimeError(
            "Production logging not initialized. Call initialize_production_logging() first."
        )

    if name is None:
        name = __name__

    return _production_logger.get_logger(name)


def get_structured_logger(name: str = None):
    """Get structured logger instance."""
    if _production_logger is None:
        raise RuntimeError(
            "Production logging not initialized. Call initialize_production_logging() first."
        )

    if name is None:
        name = __name__

    return _production_logger.get_structured_logger(name)


def log_performance(func):
    """Decorator to log function performance."""

    def wrapper(*args, **kwargs):
        logger = get_production_logger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            logger.info(
                f"Function {func.__name__} completed",
                extra={
                    "function": func.__name__,
                    "module": func.__module__,
                    "execution_time": execution_time,
                    "success": True,
                },
            )

            return result
        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                f"Function {func.__name__} failed",
                extra={
                    "function": func.__name__,
                    "module": func.__module__,
                    "execution_time": execution_time,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "success": False,
                },
                exc_info=True,
            )

            raise

    return wrapper


def main():
    """Main function for testing."""
    # Test configuration
    config = {
        "service_name": "monorepo",
        "version": "1.0.0",
        "environment": "production",
        "log_level": "INFO",
        "log_format": "json",
        "console_logging": True,
        "file_logging": True,
        "log_directory": "/tmp/pynomaly-logs",
        "syslog_logging": False,
        "cloudwatch_logging": False,
        "elasticsearch_logging": False,
        "max_log_size": 10 * 1024 * 1024,  # 10MB for testing
        "backup_count": 5,
    }

    # Initialize production logging
    prod_logger = initialize_production_logging(config)

    # Get loggers
    app_logger = get_production_logger("monorepo.app")
    struct_logger = get_structured_logger("monorepo.structured")

    # Test logging
    app_logger.info("Production logging system initialized")
    app_logger.warning("This is a warning message", extra={"component": "test"})

    # Test structured logging
    struct_logger.info("Structured log entry", user_id="12345", action="login")

    # Test performance logging
    @log_performance
    def test_function():
        time.sleep(0.1)
        return "test result"

    result = test_function()
    app_logger.info(f"Test function result: {result}")

    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception:
        app_logger.error("Caught test exception", exc_info=True)

    # Update context
    prod_logger.update_context(correlation_id="test-123", user_id="user-456")
    app_logger.info("Message with updated context")

    # Get metrics
    metrics = prod_logger.get_metrics()
    print("\nLogging Metrics:")
    print(f"Total logs: {metrics['total_logs']}")
    print(f"Errors: {metrics['error_count']}")
    print(f"Warnings: {metrics['warning_count']}")
    print(f"Logs by level: {metrics['logs_by_level']}")

    # Health check
    health = prod_logger.health_check()
    print(f"\nHealth Check: {health['status']}")
    print(f"Handlers: {health['handlers']}")
    print(f"Issues: {health['issues']}")

    # Close logging system
    prod_logger.close()


if __name__ == "__main__":
    main()

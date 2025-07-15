import logging
import logging.config
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
import traceback
from pathlib import Path


class ProfilingLogger:
    """Centralized logging configuration for data profiling operations."""
    
    def __init__(self, name: str = "data_profiling", level: str = "INFO"):
        self.name = name
        self.level = level
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up the logger with appropriate configuration."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, self.level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def add_file_handler(self, log_file: str, level: str = "DEBUG") -> None:
        """Add file handler for logging to file."""
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def add_json_handler(self, log_file: str, level: str = "INFO") -> None:
        """Add JSON file handler for structured logging."""
        json_handler = logging.FileHandler(log_file)
        json_handler.setLevel(getattr(logging, level.upper()))
        json_formatter = JSONFormatter()
        json_handler.setFormatter(json_formatter)
        self.logger.addHandler(json_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry['extra'] = log_entry.get('extra', {})
                log_entry['extra'][key] = value
        
        return json.dumps(log_entry)


class ProfilingMetrics:
    """Metrics collection for profiling operations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.metrics[operation] = {
            'start_time': datetime.now(),
            'status': 'running'
        }
        self.logger.info(f"Started operation: {operation}")
    
    def end_timer(self, operation: str, success: bool = True, **kwargs) -> None:
        """End timing an operation."""
        if operation not in self.metrics:
            self.logger.warning(f"No start time found for operation: {operation}")
            return
        
        end_time = datetime.now()
        start_time = self.metrics[operation]['start_time']
        duration = (end_time - start_time).total_seconds()
        
        self.metrics[operation].update({
            'end_time': end_time,
            'duration_seconds': duration,
            'status': 'success' if success else 'failed',
            **kwargs
        })
        
        status = "completed successfully" if success else "failed"
        self.logger.info(f"Operation {operation} {status} in {duration:.2f} seconds")
    
    def log_metric(self, metric_name: str, value: Any, **kwargs) -> None:
        """Log a custom metric."""
        self.logger.info(f"Metric {metric_name}: {value}", extra={
            'metric_name': metric_name,
            'metric_value': value,
            **kwargs
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self.metrics.copy()


class ErrorHandler:
    """Centralized error handling for profiling operations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def handle_error(self, error: Exception, context: str, **kwargs) -> None:
        """Handle and log an error with context."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            **kwargs
        }
        
        self.logger.error(
            f"Error in {context}: {error_info['error_type']} - {error_info['error_message']}",
            exc_info=True,
            extra=error_info
        )
    
    def handle_warning(self, message: str, context: str, **kwargs) -> None:
        """Handle and log a warning with context."""
        warning_info = {
            'context': context,
            **kwargs
        }
        
        self.logger.warning(f"Warning in {context}: {message}", extra=warning_info)
    
    def handle_validation_error(self, validation_errors: list, context: str) -> None:
        """Handle validation errors."""
        for error in validation_errors:
            self.logger.error(
                f"Validation error in {context}: {error}",
                extra={'validation_error': error, 'context': context}
            )


class ProfilingContext:
    """Context manager for profiling operations with automatic logging and metrics."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or ProfilingLogger().get_logger()
        self.metrics = ProfilingMetrics(self.logger)
        self.error_handler = ErrorHandler(self.logger)
        self.success = False
    
    def __enter__(self):
        self.metrics.start_timer(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.success = True
            self.metrics.end_timer(self.operation_name, success=True)
        else:
            self.error_handler.handle_error(exc_val, self.operation_name)
            self.metrics.end_timer(self.operation_name, success=False, 
                                 error_type=exc_type.__name__, 
                                 error_message=str(exc_val))
        return False  # Don't suppress exceptions
    
    def log_progress(self, message: str, **kwargs) -> None:
        """Log progress during the operation."""
        self.logger.info(f"{self.operation_name}: {message}", extra=kwargs)
    
    def log_metric(self, metric_name: str, value: Any, **kwargs) -> None:
        """Log a metric during the operation."""
        self.metrics.log_metric(metric_name, value, operation=self.operation_name, **kwargs)


def setup_profiling_logging(config: Optional[Dict[str, Any]] = None) -> ProfilingLogger:
    """Set up profiling logging with configuration."""
    if config is None:
        config = {
            'level': 'INFO',
            'name': 'data_profiling'
        }
    
    profiling_logger = ProfilingLogger(
        name=config.get('name', 'data_profiling'),
        level=config.get('level', 'INFO')
    )
    
    # Add file handler if configured
    if 'log_file' in config:
        profiling_logger.add_file_handler(
            config['log_file'],
            level=config.get('file_level', 'DEBUG')
        )
    
    # Add JSON handler if configured
    if 'json_log_file' in config:
        profiling_logger.add_json_handler(
            config['json_log_file'],
            level=config.get('json_level', 'INFO')
        )
    
    return profiling_logger


def create_default_logging_config() -> Dict[str, Any]:
    """Create default logging configuration."""
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            },
            'json': {
                '()': JSONFormatter
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'data_profiling.log',
                'mode': 'a'
            },
            'json_file': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': 'data_profiling.json',
                'mode': 'a'
            }
        },
        'loggers': {
            'data_profiling': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'json_file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }


# Global logger instance
_global_logger = None


def get_logger(name: str = "data_profiling") -> logging.Logger:
    """Get a logger instance with the specified name."""
    global _global_logger
    if _global_logger is None:
        _global_logger = ProfilingLogger(name)
    return _global_logger.get_logger()


def configure_logging_from_file(config_file: str) -> None:
    """Configure logging from a configuration file."""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Logging configuration file not found: {config_file}")
    
    if config_path.suffix.lower() == '.json':
        with open(config_file, 'r') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.config.fileConfig(config_file)
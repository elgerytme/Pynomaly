"""Log formatters for different output formats."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""

    def __init__(
        self,
        include_timestamp: bool = True,
        timestamp_format: str = "ISO",
        include_level: bool = True,
        include_logger_name: bool = True,
        include_thread_info: bool = False,
        include_process_info: bool = False,
        extra_fields: dict[str, Any] | None = None,
    ):
        """Initialize JSON formatter.

        Args:
            include_timestamp: Whether to include timestamp
            timestamp_format: Format for timestamp ('ISO', 'epoch', or custom format)
            include_level: Whether to include log level
            include_logger_name: Whether to include logger name
            include_thread_info: Whether to include thread information
            include_process_info: Whether to include process information
            extra_fields: Additional fields to include in every log record
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.timestamp_format = timestamp_format
        self.include_level = include_level
        self.include_logger_name = include_logger_name
        self.include_thread_info = include_thread_info
        self.include_process_info = include_process_info
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {}

        # Add timestamp
        if self.include_timestamp:
            if self.timestamp_format == "ISO":
                log_entry["timestamp"] = datetime.fromtimestamp(record.created).isoformat()
            elif self.timestamp_format == "epoch":
                log_entry["timestamp"] = record.created
            else:
                log_entry["timestamp"] = datetime.fromtimestamp(record.created).strftime(
                    self.timestamp_format
                )

        # Add log level
        if self.include_level:
            log_entry["level"] = record.levelname

        # Add logger name
        if self.include_logger_name:
            log_entry["logger"] = record.name

        # Add message
        log_entry["message"] = record.getMessage()

        # Add thread info
        if self.include_thread_info:
            log_entry["thread_id"] = record.thread
            log_entry["thread_name"] = record.threadName

        # Add process info
        if self.include_process_info:
            log_entry["process_id"] = record.process
            log_entry["process_name"] = record.processName

        # Add extra fields
        log_entry.update(self.extra_fields)

        # Add record attributes (excluding standard ones)
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "getMessage",
            "exc_info", "exc_text", "stack_info"
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_entry[key] = value

        # Handle exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }

        # Add stack info if available
        if record.stack_info:
            log_entry["stack_info"] = record.stack_info

        # Add caller info
        log_entry["caller"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
            "module": record.module,
            "pathname": record.pathname
        }

        return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'))


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with color support."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def __init__(
        self,
        use_colors: bool = True,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger_name: bool = True,
        include_caller_info: bool = False,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
        max_logger_name_length: int = 20,
    ):
        """Initialize console formatter.

        Args:
            use_colors: Whether to use ANSI colors
            include_timestamp: Whether to include timestamp
            include_level: Whether to include log level
            include_logger_name: Whether to include logger name
            include_caller_info: Whether to include caller information
            timestamp_format: Format for timestamp
            max_logger_name_length: Maximum length for logger name display
        """
        super().__init__()
        self.use_colors = use_colors
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger_name = include_logger_name
        self.include_caller_info = include_caller_info
        self.timestamp_format = timestamp_format
        self.max_logger_name_length = max_logger_name_length

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        parts = []

        # Add timestamp
        if self.include_timestamp:
            timestamp = datetime.fromtimestamp(record.created).strftime(self.timestamp_format)
            parts.append(f"[{timestamp}]")

        # Add log level with color
        if self.include_level:
            level = record.levelname
            if self.use_colors:
                color = self.COLORS.get(level, self.COLORS['RESET'])
                reset = self.COLORS['RESET']
                level_str = f"{color}{level:8}{reset}"
            else:
                level_str = f"{level:8}"
            parts.append(level_str)

        # Add logger name
        if self.include_logger_name:
            logger_name = record.name
            if len(logger_name) > self.max_logger_name_length:
                logger_name = "..." + logger_name[-(self.max_logger_name_length-3):]
            parts.append(f"[{logger_name:{self.max_logger_name_length}}]")

        # Add caller info
        if self.include_caller_info:
            caller = f"{record.filename}:{record.lineno}:{record.funcName}"
            parts.append(f"({caller})")

        # Add message
        message = record.getMessage()
        parts.append(message)

        # Join all parts
        formatted_message = " ".join(parts)

        # Add exception info if present
        if record.exc_info:
            formatted_message += "\n" + self.formatException(record.exc_info)

        # Add stack info if present
        if record.stack_info:
            formatted_message += "\n" + record.stack_info

        return formatted_message


class StructuredConsoleFormatter(logging.Formatter):
    """Console formatter that shows structured data in a readable format."""

    def __init__(
        self,
        use_colors: bool = True,
        indent_size: int = 2,
        max_line_length: int = 120,
        show_metadata: bool = True,
    ):
        """Initialize structured console formatter.

        Args:
            use_colors: Whether to use ANSI colors
            indent_size: Number of spaces for indentation
            max_line_length: Maximum line length before wrapping
            show_metadata: Whether to show metadata fields
        """
        super().__init__()
        self.use_colors = use_colors
        self.indent_size = indent_size
        self.max_line_length = max_line_length
        self.show_metadata = show_metadata

        # Colors for different data types
        self.colors = {
            'timestamp': '\033[90m',    # Dark gray
            'level': '\033[1m',         # Bold
            'logger': '\033[94m',       # Blue
            'message': '\033[97m',      # White
            'key': '\033[96m',          # Cyan
            'value': '\033[93m',        # Yellow
            'error': '\033[91m',        # Light red
            'reset': '\033[0m'          # Reset
        } if use_colors else dict.fromkeys(['timestamp', 'level', 'logger', 'message', 'key', 'value', 'error', 'reset'], '')

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data display."""
        lines = []

        # Header line with timestamp, level, logger
        header_parts = []

        if self.show_metadata:
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
            header_parts.append(f"{self.colors['timestamp']}{timestamp}{self.colors['reset']}")

            level_color = {
                'DEBUG': self.colors['reset'],
                'INFO': '\033[32m',
                'WARNING': '\033[33m',
                'ERROR': '\033[31m',
                'CRITICAL': '\033[35m'
            }.get(record.levelname, self.colors['level']) if self.use_colors else ''

            header_parts.append(f"{level_color}{record.levelname:8}{self.colors['reset']}")
            header_parts.append(f"{self.colors['logger']}{record.name}{self.colors['reset']}")

        # Main message
        message = record.getMessage()
        if header_parts:
            lines.append(" ".join(header_parts) + f" {self.colors['message']}{message}{self.colors['reset']}")
        else:
            lines.append(f"{self.colors['message']}{message}{self.colors['reset']}")

        # Add structured data from record
        structured_data = {}
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "getMessage",
            "exc_info", "exc_text", "stack_info"
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                structured_data[key] = value

        # Format structured data
        if structured_data:
            lines.append(self._format_dict(structured_data, indent=self.indent_size))

        # Add exception info
        if record.exc_info:
            lines.append(f"{self.colors['error']}Exception:{self.colors['reset']}")
            exception_lines = self.formatException(record.exc_info).split('\n')
            for line in exception_lines:
                if line.strip():
                    lines.append(f"{' ' * self.indent_size}{self.colors['error']}{line}{self.colors['reset']}")

        return "\n".join(lines)

    def _format_dict(self, data: dict, indent: int = 0) -> str:
        """Format dictionary data with proper indentation."""
        lines = []
        indent_str = " " * indent

        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{self.colors['key']}{key}:{self.colors['reset']}")
                lines.append(self._format_dict(value, indent + self.indent_size))
            elif isinstance(value, list | tuple):
                lines.append(f"{indent_str}{self.colors['key']}{key}:{self.colors['reset']}")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(f"{indent_str}  [{i}]:")
                        lines.append(self._format_dict(item, indent + self.indent_size + 2))
                    else:
                        lines.append(f"{indent_str}  [{i}] {self.colors['value']}{item}{self.colors['reset']}")
            else:
                # Truncate very long values
                str_value = str(value)
                if len(str_value) > 100:
                    str_value = str_value[:97] + "..."

                lines.append(f"{indent_str}{self.colors['key']}{key}:{self.colors['reset']} {self.colors['value']}{str_value}{self.colors['reset']}")

        return "\n".join(lines)


class MetricsFormatter(logging.Formatter):
    """Specialized formatter for metrics logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format metrics log record."""
        # Check if this is a metrics log
        if not hasattr(record, 'metric_name'):
            return super().format(record)

        datetime.fromtimestamp(record.created).isoformat()

        # Format as Prometheus-style metrics
        metric_line = f"{record.metric_name}"

        # Add labels if present
        if hasattr(record, 'labels') and record.labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in record.labels.items())
            metric_line += f"{{{label_str}}}"

        # Add value
        metric_line += f" {record.metric_value}"

        # Add timestamp
        metric_line += f" {int(record.created * 1000)}"

        return metric_line


def create_formatter(
    format_type: str = "json",
    **kwargs
) -> logging.Formatter:
    """Factory function to create formatters.

    Args:
        format_type: Type of formatter ('json', 'console', 'structured', 'metrics')
        **kwargs: Additional formatter-specific arguments

    Returns:
        Configured formatter instance
    """
    formatters = {
        "json": JSONFormatter,
        "console": ConsoleFormatter,
        "structured": StructuredConsoleFormatter,
        "metrics": MetricsFormatter,
    }

    formatter_class = formatters.get(format_type)
    if not formatter_class:
        raise ValueError(f"Unknown formatter type: {format_type}")

    return formatter_class(**kwargs)

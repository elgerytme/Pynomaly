"""Error reporting to external monitoring systems."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class ErrorReporter(ABC):
    """Abstract base class for error reporters."""

    @abstractmethod
    def report(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: str = "error",
    ) -> bool:
        """Report an error to external system.

        Args:
            error: The exception to report
            context: Additional context information
            severity: Error severity level

        Returns:
            True if reported successfully, False otherwise
        """
        pass


class LoggingErrorReporter(ErrorReporter):
    """Error reporter that logs to Python logging system."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize logging reporter.

        Args:
            logger: Logger instance to use
        """
        self.logger = logger or logging.getLogger(__name__)

    def report(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: str = "error",
    ) -> bool:
        """Report error via logging."""
        try:
            log_level = getattr(logging, severity.upper(), logging.ERROR)

            log_data = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": datetime.utcnow().isoformat(),
                "severity": severity,
            }

            if context:
                log_data["context"] = context

            self.logger.log(
                log_level,
                f"Error reported: {type(error).__name__}: {str(error)}",
                extra={"error_data": log_data},
                exc_info=True,
            )

            return True

        except Exception as e:
            # Never let error reporting break the application
            self.logger.error(f"Failed to report error: {e}")
            return False


class FileErrorReporter(ErrorReporter):
    """Error reporter that writes to a file."""

    def __init__(self, file_path: str = "error_reports.jsonl"):
        """Initialize file reporter.

        Args:
            file_path: Path to error report file
        """
        self.file_path = file_path

    def report(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: str = "error",
    ) -> bool:
        """Report error to file."""
        try:
            error_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "severity": severity,
                "context": context or {},
            }

            # Append to JSONL file
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_data) + "\n")

            return True

        except Exception:
            # Never let error reporting break the application
            return False


class SentryErrorReporter(ErrorReporter):
    """Error reporter for Sentry (requires sentry-sdk)."""

    def __init__(self, dsn: str | None = None):
        """Initialize Sentry reporter.

        Args:
            dsn: Sentry DSN (optional if configured elsewhere)
        """
        self.dsn = dsn
        self._sentry_available = False

        try:
            import sentry_sdk

            self.sentry = sentry_sdk
            self._sentry_available = True

            if dsn:
                sentry_sdk.init(dsn=dsn)

        except ImportError:
            self.sentry = None

    def report(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: str = "error",
    ) -> bool:
        """Report error to Sentry."""
        if not self._sentry_available:
            return False

        try:
            # Set context
            if context:
                with self.sentry.configure_scope() as scope:
                    for key, value in context.items():
                        scope.set_tag(key, str(value))

            # Set severity level
            level_mapping = {
                "debug": "debug",
                "info": "info",
                "warning": "warning",
                "error": "error",
                "critical": "fatal",
            }
            sentry_level = level_mapping.get(severity, "error")

            # Capture exception
            self.sentry.capture_exception(error, level=sentry_level)

            return True

        except Exception:
            return False


class CompositeErrorReporter(ErrorReporter):
    """Error reporter that delegates to multiple reporters."""

    def __init__(self, reporters: list[ErrorReporter] | None = None):
        """Initialize composite reporter.

        Args:
            reporters: List of error reporters to use
        """
        self.reporters = reporters or []

    def add_reporter(self, reporter: ErrorReporter) -> None:
        """Add an error reporter."""
        self.reporters.append(reporter)

    def report(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        severity: str = "error",
    ) -> bool:
        """Report error to all configured reporters."""
        success_count = 0

        for reporter in self.reporters:
            try:
                if reporter.report(error, context, severity):
                    success_count += 1
            except Exception:
                # Continue with other reporters
                pass

        # Return True if at least one reporter succeeded
        return success_count > 0


def create_default_error_reporter() -> ErrorReporter:
    """Create default error reporter with logging and file output."""
    composite = CompositeErrorReporter()

    # Add logging reporter
    composite.add_reporter(LoggingErrorReporter())

    # Add file reporter
    composite.add_reporter(FileErrorReporter("logs/error_reports.jsonl"))

    # Add Sentry if available and configured
    try:
        import os

        sentry_dsn = os.getenv("SENTRY_DSN")
        if sentry_dsn:
            composite.add_reporter(SentryErrorReporter(sentry_dsn))
    except Exception:
        pass

    return composite

"""Error response formatting utilities."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pynomaly.domain.exceptions import PynamolyError


class ErrorResponseFormatter:
    """Formats error responses for different contexts and audiences."""

    def __init__(self, include_stack_traces: bool = False):
        """Initialize formatter.

        Args:
            include_stack_traces: Whether to include stack traces in responses
        """
        self.include_stack_traces = include_stack_traces

    def format_for_api(
        self,
        error: Exception,
        error_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Format error for API response."""
        response = {
            "success": False,
            "error": {
                "id": error_id or "unknown",
                "type": type(error).__name__,
                "message": str(error),
                "timestamp": datetime.utcnow().isoformat(),
            },
        }

        # Add domain error details
        if isinstance(error, PynamolyError) and error.details:
            response["error"]["details"] = error.details

        # Add context if provided
        if context:
            response["error"]["context"] = context

        # Add stack trace if enabled
        if self.include_stack_traces:
            import traceback

            response["error"]["stack_trace"] = traceback.format_exc()

        return response

    def format_for_cli(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Format error for command-line interface."""
        lines = [
            f"❌ Error: {type(error).__name__}",
            f"Message: {str(error)}",
        ]

        # Add domain error details
        if isinstance(error, PynamolyError) and error.details:
            lines.append("Details:")
            for key, value in error.details.items():
                lines.append(f"  {key}: {value}")

        # Add context
        if context:
            lines.append("Context:")
            for key, value in context.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def format_for_logging(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Format error for structured logging."""
        log_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add domain error details
        if isinstance(error, PynamolyError):
            if error.details:
                log_data["error_details"] = error.details
            if error.cause:
                log_data["caused_by"] = {
                    "type": type(error.cause).__name__,
                    "message": str(error.cause),
                }

        # Add context
        if context:
            log_data["context"] = context

        return log_data

    def format_validation_errors(
        self,
        validation_errors: list[dict[str, Any]],
        error_id: str | None = None,
    ) -> dict[str, Any]:
        """Format validation errors for API response."""
        return {
            "success": False,
            "error": {
                "id": error_id or "validation_error",
                "type": "ValidationError",
                "message": "Input validation failed",
                "timestamp": datetime.utcnow().isoformat(),
                "validation_errors": validation_errors,
            },
        }

    def format_for_user_display(
        self,
        error: Exception,
        hide_technical_details: bool = True,
    ) -> dict[str, Any]:
        """Format error for end-user display in UI."""
        # Friendly error messages for common errors
        friendly_messages = {
            "ValidationError": "Please check your input and try again.",
            "AuthenticationError": "Please log in and try again.",
            "AuthorizationError": "You don't have permission to perform this action.",
            "DatasetError": "There's an issue with your data. Please check the format and try again.",
            "DetectorError": "There's an issue with the anomaly detector. Please try again or contact support.",
            "InfrastructureError": "We're experiencing technical difficulties. Please try again in a few moments.",
        }

        error_type = type(error).__name__
        user_message = friendly_messages.get(
            error_type, "An unexpected error occurred."
        )

        response = {
            "title": "Error",
            "message": user_message,
            "type": "error",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add technical details if not hidden
        if not hide_technical_details:
            response["technical"] = {
                "error_type": error_type,
                "error_message": str(error),
            }

            if isinstance(error, PynamolyError) and error.details:
                response["technical"]["details"] = error.details

        return response

    def create_error_summary(
        self,
        errors: list[Exception],
        title: str = "Multiple Errors Occurred",
    ) -> dict[str, Any]:
        """Create summary for multiple errors."""
        error_counts = {}
        for error in errors:
            error_type = type(error).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return {
            "title": title,
            "total_errors": len(errors),
            "error_counts": error_counts,
            "timestamp": datetime.utcnow().isoformat(),
            "errors": [
                {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for error in errors[:10]  # Limit to first 10 errors
            ],
        }

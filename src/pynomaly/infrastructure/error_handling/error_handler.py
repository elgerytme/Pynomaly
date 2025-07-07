"""Central error handler for the application."""

from __future__ import annotations

import logging
import traceback
from datetime import datetime
from typing import Any
from uuid import uuid4

from pynomaly.domain.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    DatasetError,
    DetectorError,
    InfrastructureError,
    PynamolyError,
    ValidationError,
)


class ErrorHandler:
    """Central error handler with logging, reporting, and recovery capabilities."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        enable_recovery: bool = True,
        enable_reporting: bool = True,
    ) -> None:
        """Initialize error handler.
        
        Args:
            logger: Logger instance for error logging
            enable_recovery: Whether to attempt error recovery
            enable_reporting: Whether to report errors to external systems
        """
        self.logger = logger or logging.getLogger(__name__)
        self.enable_recovery = enable_recovery
        self.enable_reporting = enable_reporting

    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        user_id: str | None = None,
        operation: str | None = None,
    ) -> dict[str, Any]:
        """Handle an error with logging, reporting, and response formatting.
        
        Args:
            error: The exception to handle
            context: Additional context information
            user_id: ID of the user who encountered the error
            operation: Operation that was being performed
            
        Returns:
            Formatted error response
        """
        # Generate error ID for tracking
        error_id = str(uuid4())
        
        # Create error context
        error_context = {
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "user_id": user_id,
            **(context or {}),
        }
        
        # Log the error
        self._log_error(error, error_context)
        
        # Report the error if enabled
        if self.enable_reporting:
            self._report_error(error, error_context)
        
        # Format response
        return self._format_error_response(error, error_context)

    def _log_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Log the error with appropriate level and context."""
        error_level = self._get_error_log_level(error)
        
        log_message = f"Error {context['error_id']}: {type(error).__name__}: {str(error)}"
        
        # Add context to log message
        if context.get("operation"):
            log_message += f" | Operation: {context['operation']}"
        if context.get("user_id"):
            log_message += f" | User: {context['user_id']}"
        
        # Log with stack trace for severe errors
        if error_level >= logging.ERROR:
            self.logger.log(
                error_level,
                log_message,
                extra={"error_context": context},
                exc_info=True,
            )
        else:
            self.logger.log(
                error_level,
                log_message,
                extra={"error_context": context},
            )

    def _get_error_log_level(self, error: Exception) -> int:
        """Determine appropriate logging level for the error."""
        if isinstance(error, ValidationError):
            return logging.WARNING
        elif isinstance(error, (AuthenticationError, AuthorizationError)):
            return logging.WARNING
        elif isinstance(error, ConfigurationError):
            return logging.ERROR
        elif isinstance(error, (DatasetError, DetectorError)):
            return logging.WARNING
        elif isinstance(error, InfrastructureError):
            return logging.ERROR
        elif isinstance(error, PynamolyError):
            return logging.WARNING
        else:
            # Unknown exceptions are logged as errors
            return logging.ERROR

    def _report_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Report error to external monitoring systems."""
        try:
            # This would integrate with external error reporting services
            # For now, we'll just log at debug level
            self.logger.debug(
                f"Error reported: {context['error_id']}",
                extra={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "context": context,
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception as reporting_error:
            # Never let error reporting break the application
            self.logger.warning(
                f"Failed to report error {context['error_id']}: {reporting_error}"
            )

    def _format_error_response(
        self, error: Exception, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Format error for API response."""
        # Base response structure
        response = {
            "error": True,
            "error_id": context["error_id"],
            "timestamp": context["timestamp"],
            "type": type(error).__name__,
            "message": str(error),
        }
        
        # Add details for domain errors
        if isinstance(error, PynamolyError):
            if error.details:
                response["details"] = error.details
        
        # Add specific handling for different error types
        if isinstance(error, ValidationError):
            response["error_code"] = "VALIDATION_ERROR"
            response["category"] = "client_error"
        elif isinstance(error, AuthenticationError):
            response["error_code"] = "AUTHENTICATION_ERROR"
            response["category"] = "auth_error"
        elif isinstance(error, AuthorizationError):
            response["error_code"] = "AUTHORIZATION_ERROR"
            response["category"] = "auth_error"
        elif isinstance(error, (DatasetError, DetectorError)):
            response["error_code"] = "DOMAIN_ERROR"
            response["category"] = "business_error"
        elif isinstance(error, InfrastructureError):
            response["error_code"] = "INFRASTRUCTURE_ERROR"
            response["category"] = "server_error"
        else:
            response["error_code"] = "UNKNOWN_ERROR"
            response["category"] = "server_error"
        
        # Add recovery suggestions
        response["recovery_suggestions"] = self._get_recovery_suggestions(error)
        
        return response

    def _get_recovery_suggestions(self, error: Exception) -> list[str]:
        """Get recovery suggestions for the error."""
        suggestions = []
        
        if isinstance(error, ValidationError):
            suggestions.append("Check input parameters and try again")
            suggestions.append("Refer to API documentation for valid parameters")
        elif isinstance(error, AuthenticationError):
            suggestions.append("Check your authentication credentials")
            suggestions.append("Try logging in again")
        elif isinstance(error, AuthorizationError):
            suggestions.append("Contact administrator for required permissions")
        elif isinstance(error, DatasetError):
            suggestions.append("Verify dataset format and contents")
            suggestions.append("Check for missing or invalid data")
        elif isinstance(error, DetectorError):
            suggestions.append("Check detector configuration")
            suggestions.append("Ensure detector is properly trained")
        elif isinstance(error, InfrastructureError):
            suggestions.append("Try again in a few moments")
            suggestions.append("Contact support if problem persists")
        else:
            suggestions.append("Try again or contact support")
        
        return suggestions

    def handle_validation_error(
        self,
        field: str,
        value: Any,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle validation error with specific field information."""
        error = ValidationError(message, field=field, value=value)
        return self.handle_error(error, context)

    def handle_not_found_error(
        self,
        resource_type: str,
        resource_id: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle resource not found error."""
        from pynomaly.domain.exceptions import EntityNotFoundError
        
        error = EntityNotFoundError(
            f"{resource_type} with ID '{resource_id}' not found",
            entity_type=resource_type,
            entity_id=resource_id,
        )
        return self.handle_error(error, context)

    def handle_unexpected_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle unexpected/unknown errors."""
        # Wrap unknown errors in PynamolyError
        wrapped_error = PynamolyError(
            f"Unexpected error: {str(error)}",
            details={"original_error_type": type(error).__name__},
            cause=error,
        )
        return self.handle_error(wrapped_error, context)


def create_default_error_handler() -> ErrorHandler:
    """Create default error handler with standard configuration."""
    logger = logging.getLogger("pynomaly.errors")
    return ErrorHandler(
        logger=logger,
        enable_recovery=True,
        enable_reporting=True,
    )
"""Shared exception hierarchy and error handling framework.

This module provides standardized error handling across all packages
in the monorepo, implementing the infrastructure standardization recommendations.
"""

from __future__ import annotations

import traceback
import structlog
from typing import Any, Dict, Optional, Type, Union, List
from enum import Enum
from abc import ABC, abstractmethod


logger = structlog.get_logger()


class ErrorCategory(Enum):
    """Standardized error categories across all packages."""
    INPUT_VALIDATION = "input_validation"
    DATA_PROCESSING = "data_processing"
    MODEL_OPERATION = "model_operation"
    ALGORITHM_ERROR = "algorithm_error"
    PERSISTENCE = "persistence"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    EXTERNAL = "external"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    BUSINESS_LOGIC = "business_logic"
    NETWORK = "network"
    RESOURCE = "resource"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BaseApplicationError(Exception, ABC):
    """Base application error class for all domain exceptions.
    
    This class provides a standardized interface for all application-specific
    errors across the monorepo, with support for structured error information,
    context preservation, and integration with logging systems.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        recoverable: bool = False,
        context: Optional[Dict[str, Any]] = None,
        operation: Optional[str] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.original_error = original_error
        self.recoverable = recoverable
        self.context = context or {}
        self.operation = operation
        self.user_message = user_message or message
        
        # Capture traceback information
        self.traceback_str = traceback.format_exc() if original_error else None
        
        # Add error metadata
        self.timestamp = None  # Will be set by error handler
        self.request_id = None  # Will be set by error handler
        
    def _generate_error_code(self) -> str:
        """Generate a default error code based on class name."""
        return f"{self.__class__.__name__.upper()}_ERROR"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "details": self.details,
            "context": self.context,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "original_error": {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error)
            } if self.original_error else None
        }
    
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"{self.error_code}: {self.message}"]
        
        if self.operation:
            parts.append(f"Operation: {self.operation}")
            
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            parts.append(f"Context: {context_str}")
            
        return " | ".join(parts)


class ValidationError(BaseApplicationError):
    """Error for invalid input data or parameters."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Any = None,
        validation_rule: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if field_name:
            details['field_name'] = field_name
        if field_value is not None:
            details['field_value'] = field_value
        if validation_rule:
            details['validation_rule'] = validation_rule
            
        super().__init__(
            message=message,
            category=ErrorCategory.INPUT_VALIDATION,
            severity=ErrorSeverity.LOW,
            details=details,
            recoverable=True,
            **kwargs
        )


class DataProcessingError(BaseApplicationError):
    """Error during data preprocessing or transformation."""
    
    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        processing_step: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if data_source:
            details['data_source'] = data_source
        if processing_step:
            details['processing_step'] = processing_step
            
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_PROCESSING,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            recoverable=True,
            **kwargs
        )


class ModelOperationError(BaseApplicationError):
    """Error during model training, loading, or inference."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        model_operation: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if model_name:
            details['model_name'] = model_name
        if model_version:
            details['model_version'] = model_version
        if model_operation:
            details['model_operation'] = model_operation
            
        super().__init__(
            message=message,
            category=ErrorCategory.MODEL_OPERATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            recoverable=False,
            **kwargs
        )


class AlgorithmError(BaseApplicationError):
    """Error in algorithm execution or configuration."""
    
    def __init__(
        self,
        message: str,
        algorithm_name: Optional[str] = None,
        algorithm_parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if algorithm_name:
            details['algorithm_name'] = algorithm_name
        if algorithm_parameters:
            details['algorithm_parameters'] = algorithm_parameters
            
        super().__init__(
            message=message,
            category=ErrorCategory.ALGORITHM_ERROR,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            recoverable=True,
            **kwargs
        )


class PersistenceError(BaseApplicationError):
    """Error during file I/O or database operations."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_path: Optional[str] = None,
        operation_type: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if resource_type:
            details['resource_type'] = resource_type
        if resource_path:
            details['resource_path'] = resource_path
        if operation_type:
            details['operation_type'] = operation_type
            
        super().__init__(
            message=message,
            category=ErrorCategory.PERSISTENCE,
            severity=ErrorSeverity.HIGH,
            details=details,
            recoverable=True,
            **kwargs
        )


class ConfigurationError(BaseApplicationError):
    """Error in configuration or settings."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Any = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            details['config_value'] = config_value
        if config_file:
            details['config_file'] = config_file
            
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            recoverable=False,
            **kwargs
        )


class SystemError(BaseApplicationError):
    """System-level error."""
    
    def __init__(
        self,
        message: str,
        system_component: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if system_component:
            details['system_component'] = system_component
            
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            details=details,
            recoverable=False,
            **kwargs
        )


class ExternalServiceError(BaseApplicationError):
    """Error communicating with external services."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if service_name:
            details['service_name'] = service_name
        if endpoint:
            details['endpoint'] = endpoint
        if status_code:
            details['status_code'] = status_code
            
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            recoverable=True,
            **kwargs
        )


class AuthenticationError(BaseApplicationError):
    """Authentication-related errors."""
    
    def __init__(
        self,
        message: str,
        auth_method: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if auth_method:
            details['auth_method'] = auth_method
            
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            recoverable=True,
            user_message="Authentication failed. Please check your credentials.",
            **kwargs
        )


class AuthorizationError(BaseApplicationError):
    """Authorization-related errors."""
    
    def __init__(
        self,
        message: str,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if resource:
            details['resource'] = resource
        if action:
            details['action'] = action
        if user_id:
            details['user_id'] = user_id
            
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            details=details,
            recoverable=True,
            user_message="You don't have permission to perform this action.",
            **kwargs
        )


class BusinessLogicError(BaseApplicationError):
    """Business logic violation errors."""
    
    def __init__(
        self,
        message: str,
        business_rule: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if business_rule:
            details['business_rule'] = business_rule
            
        super().__init__(
            message=message,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            recoverable=True,
            **kwargs
        )


class ResourceError(BaseApplicationError):
    """Resource-related errors (memory, disk, etc.)."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[str] = None,
        limit: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if resource_type:
            details['resource_type'] = resource_type
        if current_usage:
            details['current_usage'] = current_usage
        if limit:
            details['limit'] = limit
            
        super().__init__(
            message=message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            details=details,
            recoverable=True,
            **kwargs
        )


class NetworkError(BaseApplicationError):
    """Network-related errors."""
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        details = kwargs.pop('details', {})
        
        if endpoint:
            details['endpoint'] = endpoint
        if timeout:
            details['timeout'] = timeout
            
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            details=details,
            recoverable=True,
            **kwargs
        )


class StandardErrorHandler:
    """Centralized error handling with structured logging.
    
    This class provides standardized error handling across all packages,
    with automatic error classification, structured logging, and context preservation.
    """
    
    def __init__(self, logger: Optional[structlog.BoundLogger] = None, package_name: str = "unknown"):
        self.logger = logger or structlog.get_logger()
        self.package_name = package_name
        
        # Error classification rules
        self.classification_rules = {
            ValueError: (ValidationError, "Invalid value provided"),
            TypeError: (ValidationError, "Invalid type provided"),
            KeyError: (DataProcessingError, "Missing required key"),
            AttributeError: (DataProcessingError, "Missing required attribute"),
            IndexError: (DataProcessingError, "Index out of range"),
            FileNotFoundError: (PersistenceError, "File not found"),
            PermissionError: (PersistenceError, "Permission denied"),
            OSError: (PersistenceError, "OS operation failed"),
            ImportError: (ConfigurationError, "Import failed"),
            ModuleNotFoundError: (ConfigurationError, "Module not found"),
            ConnectionError: (NetworkError, "Connection failed"),
            TimeoutError: (NetworkError, "Operation timed out"),
        }
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        operation: str = "unknown",
        reraise: bool = True,
        request_id: Optional[str] = None
    ) -> Optional[BaseApplicationError]:
        """Handle and log errors with context.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            operation: The operation being performed when the error occurred
            reraise: Whether to re-raise the error after handling
            request_id: Optional request ID for tracking
            
        Returns:
            The processed application error (if not re-raising)
        """
        context = context or {}
        
        # Convert to our custom error type if needed
        if isinstance(error, BaseApplicationError):
            app_error = error
        else:
            app_error = self._classify_error(error, context, operation)
        
        # Enrich error with additional context
        app_error.context.update(context)
        app_error.operation = app_error.operation or operation
        app_error.request_id = request_id
        app_error.timestamp = app_error.timestamp or self._current_timestamp()
        
        # Log the error
        self._log_error(app_error)
        
        if reraise:
            raise app_error
        
        return app_error
    
    def _classify_error(
        self, 
        error: Exception, 
        context: Dict[str, Any], 
        operation: str
    ) -> BaseApplicationError:
        """Classify generic exceptions into our error types."""
        error_type = type(error)
        error_msg = str(error)
        
        # Check for exact type match
        if error_type in self.classification_rules:
            error_class, default_message = self.classification_rules[error_type]
            return error_class(
                message=f"{default_message}: {error_msg}",
                operation=operation,
                context=context,
                original_error=error
            )
        
        # Check for message patterns for more specific classification
        error_msg_lower = error_msg.lower()
        
        # Input validation patterns
        if any(keyword in error_msg_lower for keyword in [
            "invalid", "validation", "constraint", "required", "missing"
        ]):
            return ValidationError(
                message=f"Validation failed in {operation}: {error_msg}",
                operation=operation,
                context=context,
                original_error=error
            )
        
        # Data processing patterns  
        if any(keyword in error_msg_lower for keyword in [
            "data", "format", "parse", "decode", "transform"
        ]):
            return DataProcessingError(
                message=f"Data processing failed in {operation}: {error_msg}",
                operation=operation,
                context=context,
                original_error=error
            )
        
        # Model/algorithm patterns
        if any(keyword in error_msg_lower for keyword in [
            "model", "algorithm", "fit", "predict", "train", "sklearn", "tensorflow", "pytorch"
        ]):
            return ModelOperationError(
                message=f"Model operation failed in {operation}: {error_msg}",
                operation=operation,
                context=context,
                original_error=error
            )
        
        # Configuration patterns
        if any(keyword in error_msg_lower for keyword in [
            "config", "setting", "environment", "variable"
        ]):
            return ConfigurationError(
                message=f"Configuration error in {operation}: {error_msg}",
                operation=operation,
                context=context,
                original_error=error
            )
        
        # Network patterns
        if any(keyword in error_msg_lower for keyword in [
            "connection", "timeout", "network", "http", "request", "response"
        ]):
            return NetworkError(
                message=f"Network error in {operation}: {error_msg}",
                operation=operation,  
                context=context,
                original_error=error
            )
        
        # Default to system error
        return SystemError(
            message=f"System error in {operation}: {error_msg}",
            operation=operation,
            context=context,
            original_error=error
        )
    
    def _log_error(self, error: BaseApplicationError) -> None:
        """Log error with appropriate level and structured data."""
        log_data = {
            "package": self.package_name,
            "error_code": error.error_code,
            "error_category": error.category.value,
            "error_severity": error.severity.value,
            "operation": error.operation,
            "recoverable": error.recoverable,
            "request_id": error.request_id,
            **error.context,
            **error.details
        }
        
        # Add original error information if present
        if error.original_error:
            log_data.update({
                "original_error_type": type(error.original_error).__name__,
                "original_error_message": str(error.original_error)
            })
        
        # Log based on severity
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            self.logger.error(error.message, **log_data)
            
            # Log stack trace for critical/high severity errors
            if error.traceback_str:
                self.logger.debug("Error traceback", traceback=error.traceback_str, **log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error.message, **log_data)
        else:
            self.logger.info(error.message, **log_data)
    
    def create_error_response(self, error: BaseApplicationError) -> Dict[str, Any]:
        """Create a standardized error response for APIs."""
        return {
            "success": False,
            "error": {
                "code": error.error_code,
                "message": error.user_message,
                "category": error.category.value,
                "severity": error.severity.value,
                "recoverable": error.recoverable,
                "details": error.details,
                "timestamp": error.timestamp,
                "request_id": error.request_id
            }
        }
    
    def log_success(
        self, 
        operation: str, 
        context: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> None:
        """Log successful operations."""
        log_data = {
            "package": self.package_name,
            "operation": operation,
            "status": "success",
            "request_id": request_id
        }
        
        if context:
            log_data.update(context)
        
        self.logger.info(f"Operation completed successfully: {operation}", **log_data)
    
    def _current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


def create_error_handler(package_name: str, logger: Optional[structlog.BoundLogger] = None) -> StandardErrorHandler:
    """Factory function to create package-specific error handlers."""
    return StandardErrorHandler(logger=logger, package_name=package_name)


# Convenience functions for common error handling patterns
def handle_error(
    error: Exception,
    package_name: str = "unknown",
    context: Optional[Dict[str, Any]] = None,
    operation: str = "unknown",
    reraise: bool = True,
    request_id: Optional[str] = None
) -> Optional[BaseApplicationError]:
    """Global convenience function for error handling."""
    handler = StandardErrorHandler(package_name=package_name)
    return handler.handle_error(error, context, operation, reraise, request_id)


__all__ = [
    # Error categories and severity
    "ErrorCategory",
    "ErrorSeverity",
    
    # Base classes
    "BaseApplicationError",
    
    # Specific error types
    "ValidationError",
    "DataProcessingError", 
    "ModelOperationError",
    "AlgorithmError",
    "PersistenceError",
    "ConfigurationError",
    "SystemError",
    "ExternalServiceError",
    "AuthenticationError",
    "AuthorizationError",
    "BusinessLogicError",
    "ResourceError",
    "NetworkError",
    
    # Error handling
    "StandardErrorHandler",
    "create_error_handler",
    "handle_error"
]
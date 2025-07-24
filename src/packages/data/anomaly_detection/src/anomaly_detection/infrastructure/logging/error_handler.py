"""Custom error handling for anomaly detection."""

from __future__ import annotations

from typing import Optional, Dict, Any, Type
from enum import Enum
import traceback
import structlog

logger = structlog.get_logger()


class ErrorCategory(Enum):
    """Categories for different types of errors."""
    INPUT_VALIDATION = "input_validation"
    DATA_PROCESSING = "data_processing"
    MODEL_OPERATION = "model_operation"
    ALGORITHM_ERROR = "algorithm_error"
    PERSISTENCE = "persistence"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    EXTERNAL = "external"


class BaseApplicationError(Exception):
    """Base application error class."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class AnomalyDetectionError(BaseApplicationError):
    """Base exception for anomaly detection errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        details: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        recoverable: bool = False,
        error_code: Optional[str] = None
    ):
        super().__init__(message, error_code)
        self.category = category
        self.details = details or {}
        self.original_error = original_error
        self.recoverable = recoverable
        self.traceback_str = traceback.format_exc() if original_error else None


class InputValidationError(AnomalyDetectionError):
    """Error for invalid input data or parameters."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.INPUT_VALIDATION,
            details=details,
            original_error=original_error,
            recoverable=True
        )


class DataProcessingError(AnomalyDetectionError):
    """Error during data preprocessing or transformation."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_PROCESSING,
            details=details,
            original_error=original_error,
            recoverable=True
        )


class ModelOperationError(AnomalyDetectionError):
    """Error during model training, loading, or saving."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.MODEL_OPERATION,
            details=details,
            original_error=original_error,
            recoverable=False
        )


class AlgorithmError(AnomalyDetectionError):
    """Error in algorithm execution or configuration."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.ALGORITHM_ERROR,
            details=details,
            original_error=original_error,
            recoverable=True
        )


class PersistenceError(AnomalyDetectionError):
    """Error during file I/O or database operations."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.PERSISTENCE,
            details=details,
            original_error=original_error,
            recoverable=True
        )


class ConfigurationError(AnomalyDetectionError):
    """Error in configuration or settings."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            details=details,
            original_error=original_error,
            recoverable=False
        )


class ErrorHandler:
    """Centralized error handling with structured logging."""
    
    def __init__(self, logger: Optional[structlog.BoundLogger] = None):
        self.logger = logger or structlog.get_logger()
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        operation: str = "unknown",
        reraise: bool = True
    ) -> Optional[AnomalyDetectionError]:
        """Handle and log errors with context."""
        context = context or {}
        
        # Convert to our custom error type if needed
        if isinstance(error, AnomalyDetectionError):
            ad_error = error
        else:
            ad_error = self._classify_error(error, context, operation)
        
        # Prepare log context
        log_context = {
            "operation": operation,
            "error_category": ad_error.category.value,
            "error_type": type(ad_error).__name__,
            "error_message": str(ad_error),
            "recoverable": ad_error.recoverable,
            **context,
            **ad_error.details
        }
        
        # Add original error info if present
        if ad_error.original_error:
            log_context["original_error_type"] = type(ad_error.original_error).__name__
            log_context["original_error_message"] = str(ad_error.original_error)
        
        # Log based on severity
        if ad_error.recoverable:
            self.logger.warning("Recoverable error occurred", **log_context)
        else:
            self.logger.error("Critical error occurred", **log_context)
            if ad_error.traceback_str:
                self.logger.debug("Error traceback", traceback=ad_error.traceback_str)
        
        if reraise:
            raise ad_error
        
        return ad_error
    
    def _classify_error(
        self, 
        error: Exception, 
        context: Dict[str, Any], 
        operation: str
    ) -> AnomalyDetectionError:
        """Classify generic exceptions into our error types."""
        error_msg = str(error)
        error_type = type(error).__name__
        
        # Classification based on error type and message patterns
        if isinstance(error, (ValueError, TypeError)) and any(
            keyword in error_msg.lower() 
            for keyword in ["input", "parameter", "dimension", "shape", "invalid"]
        ):
            return InputValidationError(
                message=f"Input validation failed in {operation}: {error_msg}",
                details={"error_type": error_type, "operation": operation},
                original_error=error
            )
        
        elif isinstance(error, (KeyError, AttributeError, IndexError)):
            return DataProcessingError(
                message=f"Data processing error in {operation}: {error_msg}",
                details={"error_type": error_type, "operation": operation},
                original_error=error
            )
        
        elif isinstance(error, (FileNotFoundError, PermissionError, OSError)):
            return PersistenceError(
                message=f"File operation failed in {operation}: {error_msg}",
                details={"error_type": error_type, "operation": operation},
                original_error=error
            )
        
        elif "algorithm" in operation.lower() or any(
            keyword in error_msg.lower() 
            for keyword in ["fit", "predict", "transform", "sklearn", "model"]
        ):
            return AlgorithmError(
                message=f"Algorithm error in {operation}: {error_msg}",
                details={"error_type": error_type, "operation": operation},
                original_error=error
            )
        
        elif "config" in operation.lower() or isinstance(error, (ImportError, ModuleNotFoundError)):
            return ConfigurationError(
                message=f"Configuration error in {operation}: {error_msg}",
                details={"error_type": error_type, "operation": operation},
                original_error=error
            )
        
        else:
            # Generic system error
            return AnomalyDetectionError(
                message=f"System error in {operation}: {error_msg}",
                category=ErrorCategory.SYSTEM,
                details={"error_type": error_type, "operation": operation},
                original_error=error,
                recoverable=False
            )
    
    def create_error_response(self, error: AnomalyDetectionError) -> Dict[str, Any]:
        """Create a standardized error response for APIs."""
        return {
            "success": False,
            "error": {
                "type": type(error).__name__,
                "category": error.category.value,
                "message": error.message,
                "recoverable": error.recoverable,
                "details": error.details
            }
        }
    
    def log_success(self, operation: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log successful operations."""
        log_context = {"operation": operation, "status": "success"}
        if context:
            log_context.update(context)
        
        self.logger.info("Operation completed successfully", **log_context)


# Global error handler instance
default_error_handler = ErrorHandler()


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    operation: str = "unknown",
    reraise: bool = True
) -> Optional[AnomalyDetectionError]:
    """Global convenience function for error handling."""
    return default_error_handler.handle_error(error, context, operation, reraise)


# Alias for backward compatibility
ValidationError = InputValidationError

# Export common functions
__all__ = [
    "handle_error",
    "ErrorHandler", 
    "BaseApplicationError",
    "AnomalyDetectionError",
    "InputValidationError",
    "ValidationError",  # Alias for InputValidationError
    "DataProcessingError", 
    "ModelOperationError",
    "AlgorithmError",
    "PersistenceError",
    "ConfigurationError",
    "ErrorCategory",
    "default_error_handler"
]
"""Validation error strategies for domain validation."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ValidationSeverity(Enum):
    """Severity levels for validation errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        severity: ValidationSeverity = ValidationSeverity.ERROR
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.severity = severity
    
    def add_error(self, error: str, severity: ValidationSeverity = ValidationSeverity.ERROR) -> None:
        """Add an error to the validation result."""
        self.errors.append(error)
        self.is_valid = False
        if severity.value > self.severity.value:
            self.severity = severity
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the validation result."""
        self.warnings.append(warning)


class ValidationErrorStrategy(ABC):
    """Abstract base class for validation error strategies."""
    
    @abstractmethod
    def handle_error(self, error: str, context: Dict[str, Any]) -> ValidationResult:
        """Handle a validation error."""
        pass


class StrictValidationStrategy(ValidationErrorStrategy):
    """Strict validation strategy that treats all errors as failures."""
    
    def handle_error(self, error: str, context: Dict[str, Any]) -> ValidationResult:
        """Handle error with strict validation."""
        return ValidationResult(
            is_valid=False,
            errors=[error],
            severity=ValidationSeverity.ERROR
        )


class LenientValidationStrategy(ValidationErrorStrategy):
    """Lenient validation strategy that allows some errors as warnings."""
    
    def handle_error(self, error: str, context: Dict[str, Any]) -> ValidationResult:
        """Handle error with lenient validation."""
        # In lenient mode, some errors become warnings
        if "warning" in error.lower() or "minor" in error.lower():
            return ValidationResult(
                is_valid=True,
                warnings=[error],
                severity=ValidationSeverity.WARNING
            )
        return ValidationResult(
            is_valid=False,
            errors=[error],
            severity=ValidationSeverity.ERROR
        )


class CustomValidationStrategy(ValidationErrorStrategy):
    """Custom validation strategy with configurable behavior."""
    
    def __init__(self, error_threshold: int = 5, warning_threshold: int = 10):
        self.error_threshold = error_threshold
        self.warning_threshold = warning_threshold
        self.error_count = 0
        self.warning_count = 0
    
    def handle_error(self, error: str, context: Dict[str, Any]) -> ValidationResult:
        """Handle error with custom validation logic."""
        self.error_count += 1
        
        if self.error_count > self.error_threshold:
            return ValidationResult(
                is_valid=False,
                errors=[f"Too many errors ({self.error_count}): {error}"],
                severity=ValidationSeverity.CRITICAL
            )
        
        return ValidationResult(
            is_valid=True,
            errors=[error],
            severity=ValidationSeverity.ERROR
        )


class ValidationErrorHandler:
    """Handler for validation errors using different strategies."""
    
    def __init__(self, strategy: ValidationErrorStrategy):
        self.strategy = strategy
    
    def handle(self, error: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Handle a validation error using the configured strategy."""
        context = context or {}
        return self.strategy.handle_error(error, context)
    
    def set_strategy(self, strategy: ValidationErrorStrategy) -> None:
        """Set a new validation strategy."""
        self.strategy = strategy

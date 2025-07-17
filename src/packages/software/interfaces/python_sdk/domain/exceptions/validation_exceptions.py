"""
Domain Validation Exceptions

Defines exceptions specific to domain validation logic.
"""

from typing import List, Optional


class DomainException(Exception):
    """Base exception for all domain-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ValidationError(DomainException):
    """
    Exception raised when domain validation fails.
    
    This exception is raised when entities, value objects, or
    domain services detect invalid data or configurations.
    """
    
    def __init__(self, message: str, validation_errors: Optional[List[str]] = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.validation_errors = validation_errors or []


class BusinessRuleViolation(DomainException):
    """
    Exception raised when business rules are violated.
    
    This exception is raised when operations violate
    established business rules within the domain.
    """
    
    def __init__(self, message: str, rule_name: Optional[str] = None):
        super().__init__(message, "BUSINESS_RULE_VIOLATION")
        self.rule_name = rule_name


class EntityNotFoundError(DomainException):
    """
    Exception raised when a required entity is not found.
    
    This exception is raised when domain operations require
    an entity that cannot be located.
    """
    
    def __init__(self, entity_type: str, entity_id: str):
        message = f"{entity_type} with ID '{entity_id}' not found"
        super().__init__(message, "ENTITY_NOT_FOUND")
        self.entity_type = entity_type
        self.entity_id = entity_id


class InvalidAlgorithmConfigError(ValidationError):
    """
    Exception raised when algorithm configuration is invalid.
    
    This exception is raised when algorithm configurations
    contain invalid parameters or incompatible settings.
    """
    
    def __init__(self, algorithm_type: str, message: str):
        full_message = f"Invalid configuration for {algorithm_type}: {message}"
        super().__init__(full_message)
        self.algorithm_type = algorithm_type


class DataValidationError(ValidationError):
    """
    Exception raised when input data validation fails.
    
    This exception is raised when input data does not meet
    the requirements for pattern analysis processing.
    """
    
    def __init__(self, message: str, data_issues: Optional[List[str]] = None):
        super().__init__(message, data_issues)
        self.data_issues = data_issues or []


class IncompatibleDataAlgorithmError(DomainException):
    """
    Exception raised when data is incompatible with the selected algorithm.
    
    This exception is raised when the input data characteristics
    are not suitable for the chosen pattern analysis algorithm.
    """
    
    def __init__(self, algorithm_type: str, data_size: int, reason: str):
        message = f"Data (size: {data_size}) is incompatible with {algorithm_type}: {reason}"
        super().__init__(message, "INCOMPATIBLE_DATA_ALGORITHM")
        self.algorithm_type = algorithm_type
        self.data_size = data_size
        self.reason = reason


class PatternAnalysisRequestError(DomainException):
    """
    Exception raised when pattern analysis request processing fails.
    
    This exception is raised when pattern analysis requests cannot
    be processed due to various domain-level issues.
    """
    
    def __init__(self, request_id: str, message: str):
        full_message = f"Pattern analysis request {request_id} failed: {message}"
        super().__init__(full_message, "PATTERN_ANALYSIS_REQUEST_ERROR")
        self.request_id = request_id


class ConcurrencyError(DomainException):
    """
    Exception raised when concurrent operations conflict.
    
    This exception is raised when multiple operations attempt
    to modify the same entity simultaneously.
    """
    
    def __init__(self, entity_type: str, entity_id: str):
        message = f"Concurrent modification detected for {entity_type} '{entity_id}'"
        super().__init__(message, "CONCURRENCY_ERROR")
        self.entity_type = entity_type
        self.entity_id = entity_id
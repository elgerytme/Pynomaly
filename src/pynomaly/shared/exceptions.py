"""
Shared exception classes for the Pynomaly application.
"""

from typing import Any


class PynomaryError(Exception):
    """Base exception class for all Pynomaly errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


# Domain exceptions
class DomainError(PynomaryError):
    """Base class for domain-related errors."""

    pass


class ValidationError(DomainError):
    """Raised when validation fails."""

    pass


class BusinessRuleViolationError(DomainError):
    """Raised when a business rule is violated."""

    pass


# User management exceptions
class UserError(DomainError):
    """Base class for user-related errors."""

    pass


class UserNotFoundError(UserError):
    """Raised when a user is not found."""

    pass


class UserAlreadyExistsError(UserError):
    """Raised when attempting to create a user that already exists."""

    pass


class AuthenticationError(UserError):
    """Raised when authentication fails."""

    pass


class AuthorizationError(UserError):
    """Raised when authorization fails."""

    pass


class TenantError(DomainError):
    """Base class for tenant-related errors."""

    pass


class TenantNotFoundError(TenantError):
    """Raised when a tenant is not found."""

    pass


class TenantAlreadyExistsError(TenantError):
    """Raised when attempting to create a tenant that already exists."""

    pass


class ResourceLimitError(TenantError):
    """Raised when tenant resource limits are exceeded."""

    pass


# Data-related exceptions
class DataError(PynomaryError):
    """Base class for data-related errors."""

    pass


class DatasetNotFoundError(DataError):
    """Raised when a dataset is not found."""

    pass


class DatasetValidationError(DataError):
    """Raised when dataset validation fails."""

    pass


class DataFormatError(DataError):
    """Raised when data format is invalid."""

    pass


# Model-related exceptions
class ModelError(PynomaryError):
    """Base class for model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a model is not found."""

    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""

    pass


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""

    pass


class UnsupportedAlgorithmError(ModelError):
    """Raised when an unsupported algorithm is requested."""

    pass


# Detection-related exceptions
class DetectionError(PynomaryError):
    """Base class for detection-related errors."""

    pass


class DetectorNotFoundError(DetectionError):
    """Raised when a detector is not found."""

    pass


class DetectionConfigurationError(DetectionError):
    """Raised when detection configuration is invalid."""

    pass


# Infrastructure exceptions
class InfrastructureError(PynomaryError):
    """Base class for infrastructure-related errors."""

    pass


class DatabaseError(InfrastructureError):
    """Raised when database operations fail."""

    pass


class CacheError(InfrastructureError):
    """Raised when cache operations fail."""

    pass


class StorageError(InfrastructureError):
    """Raised when storage operations fail."""

    pass


class ConfigurationError(InfrastructureError):
    """Raised when configuration is invalid."""

    pass


class ExternalServiceError(InfrastructureError):
    """Raised when external service calls fail."""

    pass


# Performance exceptions
class PerformanceError(PynomaryError):
    """Base class for performance-related errors."""

    pass


class MemoryError(PerformanceError):
    """Raised when memory limits are exceeded."""

    pass


class TimeoutError(PerformanceError):
    """Raised when operations timeout."""

    pass


class ResourceExhaustionError(PerformanceError):
    """Raised when system resources are exhausted."""

    pass


# API exceptions
class APIError(PynomaryError):
    """Base class for API-related errors."""

    pass


class InvalidRequestError(APIError):
    """Raised when API request is invalid."""

    pass


class RateLimitExceededError(APIError):
    """Raised when API rate limits are exceeded."""

    pass


class ServiceUnavailableError(APIError):
    """Raised when service is temporarily unavailable."""

    pass


# Integration exceptions
class IntegrationError(PynomaryError):
    """Base class for integration-related errors."""

    pass


class WebhookError(IntegrationError):
    """Raised when webhook operations fail."""

    pass


class NotificationError(IntegrationError):
    """Raised when notification delivery fails."""

    pass


class ExportError(IntegrationError):
    """Raised when data export fails."""

    pass


class ImportError(IntegrationError):
    """Raised when data import fails."""

    pass


# Reporting exceptions
class ReportingError(PynomaryError):
    """Base class for reporting-related errors."""

    pass


class ReportNotFoundError(ReportingError):
    """Raised when a report is not found."""

    pass


class DashboardNotFoundError(ReportingError):
    """Raised when a dashboard is not found."""

    pass


class MetricNotFoundError(ReportingError):
    """Raised when a metric is not found."""

    pass


class ReportGenerationError(ReportingError):
    """Raised when report generation fails."""

    pass

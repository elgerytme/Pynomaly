"""Tests for shared exception classes."""

import pytest

from pynomaly.shared.exceptions import (
    AlertingError,
    AnomalyDetectionError,
    # API exceptions
    APIError,
    AuthenticationError,
    AuthorizationError,
    BusinessRuleViolationError,
    CacheError,
    ConfigurationError,
    DashboardNotFoundError,
    DatabaseError,
    # Data-related exceptions
    DataError,
    DataFormatError,
    # Core Architecture exceptions
    DataIngestionError,
    DatasetNotFoundError,
    DatasetValidationError,
    DetectionConfigurationError,
    # Detection-related exceptions
    DetectionError,
    DetectorNotFoundError,
    DomainError,
    ExportError,
    ExternalServiceError,
    ImportError,
    # Infrastructure exceptions
    InfrastructureError,
    # Integration exceptions
    IntegrationError,
    InvalidRequestError,
    MemoryError,
    MetricNotFoundError,
    # Model-related exceptions
    ModelError,
    ModelNotFoundError,
    ModelPredictionError,
    ModelTrainingError,
    NotificationError,
    # Performance exceptions
    PerformanceError,
    # Base exceptions
    PynomaryError,
    RateLimitExceededError,
    ReportGenerationError,
    # Reporting exceptions
    ReportingError,
    ReportNotFoundError,
    ResourceExhaustionError,
    ResourceLimitError,
    ServiceUnavailableError,
    StorageError,
    TenantAlreadyExistsError,
    # Tenant exceptions
    TenantError,
    TenantNotFoundError,
    TimeoutError,
    UnsupportedAlgorithmError,
    UserAlreadyExistsError,
    # User management exceptions
    UserError,
    UserNotFoundError,
    ValidationError,
    WebhookError,
)


class TestPynomaryError:
    """Test suite for PynomaryError base class."""

    def test_basic_error_creation(self):
        """Test basic error creation."""
        error = PynomaryError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}

    def test_error_with_details(self):
        """Test error creation with details."""
        details = {"key": "value", "number": 42}
        error = PynomaryError("Test error", details=details)
        assert error.message == "Test error"
        assert error.details == details

    def test_error_with_none_details(self):
        """Test error creation with None details."""
        error = PynomaryError("Test error", details=None)
        assert error.message == "Test error"
        assert error.details == {}

    def test_error_inheritance(self):
        """Test error inheritance."""
        error = PynomaryError("Base error")
        assert isinstance(error, Exception)
        assert isinstance(error, PynomaryError)

    def test_error_string_representation(self):
        """Test error string representation."""
        error = PynomaryError("Test message")
        assert str(error) == "Test message"

    def test_error_with_complex_details(self):
        """Test error with complex details."""
        details = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "bool": True,
            "none": None,
        }
        error = PynomaryError("Complex error", details=details)
        assert error.details == details


class TestDomainExceptions:
    """Test suite for domain-related exceptions."""

    def test_domain_error_inheritance(self):
        """Test DomainError inheritance."""
        error = DomainError("Domain error")
        assert isinstance(error, PynomaryError)
        assert isinstance(error, DomainError)

    def test_validation_error_inheritance(self):
        """Test ValidationError inheritance."""
        error = ValidationError("Validation failed")
        assert isinstance(error, DomainError)
        assert isinstance(error, PynomaryError)

    def test_business_rule_violation_error_inheritance(self):
        """Test BusinessRuleViolationError inheritance."""
        error = BusinessRuleViolationError("Business rule violated")
        assert isinstance(error, DomainError)
        assert isinstance(error, PynomaryError)

    def test_validation_error_with_details(self):
        """Test ValidationError with details."""
        details = {"field": "email", "value": "invalid-email"}
        error = ValidationError("Email validation failed", details=details)
        assert error.message == "Email validation failed"
        assert error.details == details

    def test_business_rule_violation_with_details(self):
        """Test BusinessRuleViolationError with details."""
        details = {"rule": "max_attempts", "current": 5, "max": 3}
        error = BusinessRuleViolationError("Max attempts exceeded", details=details)
        assert error.message == "Max attempts exceeded"
        assert error.details == details


class TestUserExceptions:
    """Test suite for user-related exceptions."""

    def test_user_error_inheritance(self):
        """Test UserError inheritance."""
        error = UserError("User error")
        assert isinstance(error, DomainError)
        assert isinstance(error, PynomaryError)

    def test_user_not_found_error_inheritance(self):
        """Test UserNotFoundError inheritance."""
        error = UserNotFoundError("User not found")
        assert isinstance(error, UserError)
        assert isinstance(error, DomainError)

    def test_user_already_exists_error_inheritance(self):
        """Test UserAlreadyExistsError inheritance."""
        error = UserAlreadyExistsError("User already exists")
        assert isinstance(error, UserError)
        assert isinstance(error, DomainError)

    def test_authentication_error_inheritance(self):
        """Test AuthenticationError inheritance."""
        error = AuthenticationError("Authentication failed")
        assert isinstance(error, UserError)
        assert isinstance(error, DomainError)

    def test_authorization_error_inheritance(self):
        """Test AuthorizationError inheritance."""
        error = AuthorizationError("Authorization failed")
        assert isinstance(error, UserError)
        assert isinstance(error, DomainError)

    def test_user_not_found_with_details(self):
        """Test UserNotFoundError with details."""
        details = {"user_id": "123", "search_field": "email"}
        error = UserNotFoundError("User not found", details=details)
        assert error.message == "User not found"
        assert error.details == details

    def test_authentication_error_with_details(self):
        """Test AuthenticationError with details."""
        details = {"username": "test_user", "reason": "invalid_password"}
        error = AuthenticationError("Authentication failed", details=details)
        assert error.message == "Authentication failed"
        assert error.details == details


class TestTenantExceptions:
    """Test suite for tenant-related exceptions."""

    def test_tenant_error_inheritance(self):
        """Test TenantError inheritance."""
        error = TenantError("Tenant error")
        assert isinstance(error, DomainError)
        assert isinstance(error, PynomaryError)

    def test_tenant_not_found_error_inheritance(self):
        """Test TenantNotFoundError inheritance."""
        error = TenantNotFoundError("Tenant not found")
        assert isinstance(error, TenantError)
        assert isinstance(error, DomainError)

    def test_tenant_already_exists_error_inheritance(self):
        """Test TenantAlreadyExistsError inheritance."""
        error = TenantAlreadyExistsError("Tenant already exists")
        assert isinstance(error, TenantError)
        assert isinstance(error, DomainError)

    def test_resource_limit_error_inheritance(self):
        """Test ResourceLimitError inheritance."""
        error = ResourceLimitError("Resource limit exceeded")
        assert isinstance(error, TenantError)
        assert isinstance(error, DomainError)

    def test_resource_limit_error_with_details(self):
        """Test ResourceLimitError with details."""
        details = {"resource": "storage", "current": 1000, "limit": 500}
        error = ResourceLimitError("Storage limit exceeded", details=details)
        assert error.message == "Storage limit exceeded"
        assert error.details == details


class TestDataExceptions:
    """Test suite for data-related exceptions."""

    def test_data_error_inheritance(self):
        """Test DataError inheritance."""
        error = DataError("Data error")
        assert isinstance(error, PynomaryError)

    def test_dataset_not_found_error_inheritance(self):
        """Test DatasetNotFoundError inheritance."""
        error = DatasetNotFoundError("Dataset not found")
        assert isinstance(error, DataError)
        assert isinstance(error, PynomaryError)

    def test_dataset_validation_error_inheritance(self):
        """Test DatasetValidationError inheritance."""
        error = DatasetValidationError("Dataset validation failed")
        assert isinstance(error, DataError)
        assert isinstance(error, PynomaryError)

    def test_data_format_error_inheritance(self):
        """Test DataFormatError inheritance."""
        error = DataFormatError("Invalid data format")
        assert isinstance(error, DataError)
        assert isinstance(error, PynomaryError)

    def test_dataset_validation_error_with_details(self):
        """Test DatasetValidationError with details."""
        details = {"column": "age", "issue": "negative_values", "count": 5}
        error = DatasetValidationError("Invalid age values", details=details)
        assert error.message == "Invalid age values"
        assert error.details == details


class TestModelExceptions:
    """Test suite for model-related exceptions."""

    def test_model_error_inheritance(self):
        """Test ModelError inheritance."""
        error = ModelError("Model error")
        assert isinstance(error, PynomaryError)

    def test_model_not_found_error_inheritance(self):
        """Test ModelNotFoundError inheritance."""
        error = ModelNotFoundError("Model not found")
        assert isinstance(error, ModelError)
        assert isinstance(error, PynomaryError)

    def test_model_training_error_inheritance(self):
        """Test ModelTrainingError inheritance."""
        error = ModelTrainingError("Model training failed")
        assert isinstance(error, ModelError)
        assert isinstance(error, PynomaryError)

    def test_model_prediction_error_inheritance(self):
        """Test ModelPredictionError inheritance."""
        error = ModelPredictionError("Model prediction failed")
        assert isinstance(error, ModelError)
        assert isinstance(error, PynomaryError)

    def test_unsupported_algorithm_error_inheritance(self):
        """Test UnsupportedAlgorithmError inheritance."""
        error = UnsupportedAlgorithmError("Unsupported algorithm")
        assert isinstance(error, ModelError)
        assert isinstance(error, PynomaryError)

    def test_model_training_error_with_details(self):
        """Test ModelTrainingError with details."""
        details = {"algorithm": "IsolationForest", "error": "insufficient_data"}
        error = ModelTrainingError("Training failed", details=details)
        assert error.message == "Training failed"
        assert error.details == details


class TestDetectionExceptions:
    """Test suite for detection-related exceptions."""

    def test_detection_error_inheritance(self):
        """Test DetectionError inheritance."""
        error = DetectionError("Detection error")
        assert isinstance(error, PynomaryError)

    def test_detector_not_found_error_inheritance(self):
        """Test DetectorNotFoundError inheritance."""
        error = DetectorNotFoundError("Detector not found")
        assert isinstance(error, DetectionError)
        assert isinstance(error, PynomaryError)

    def test_detection_configuration_error_inheritance(self):
        """Test DetectionConfigurationError inheritance."""
        error = DetectionConfigurationError("Invalid configuration")
        assert isinstance(error, DetectionError)
        assert isinstance(error, PynomaryError)

    def test_detector_not_found_with_details(self):
        """Test DetectorNotFoundError with details."""
        details = {"detector_id": "123", "name": "fraud_detector"}
        error = DetectorNotFoundError("Detector not found", details=details)
        assert error.message == "Detector not found"
        assert error.details == details


class TestInfrastructureExceptions:
    """Test suite for infrastructure-related exceptions."""

    def test_infrastructure_error_inheritance(self):
        """Test InfrastructureError inheritance."""
        error = InfrastructureError("Infrastructure error")
        assert isinstance(error, PynomaryError)

    def test_database_error_inheritance(self):
        """Test DatabaseError inheritance."""
        error = DatabaseError("Database connection failed")
        assert isinstance(error, InfrastructureError)
        assert isinstance(error, PynomaryError)

    def test_cache_error_inheritance(self):
        """Test CacheError inheritance."""
        error = CacheError("Cache operation failed")
        assert isinstance(error, InfrastructureError)
        assert isinstance(error, PynomaryError)

    def test_storage_error_inheritance(self):
        """Test StorageError inheritance."""
        error = StorageError("Storage operation failed")
        assert isinstance(error, InfrastructureError)
        assert isinstance(error, PynomaryError)

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inheritance."""
        error = ConfigurationError("Invalid configuration")
        assert isinstance(error, InfrastructureError)
        assert isinstance(error, PynomaryError)

    def test_external_service_error_inheritance(self):
        """Test ExternalServiceError inheritance."""
        error = ExternalServiceError("External service failed")
        assert isinstance(error, InfrastructureError)
        assert isinstance(error, PynomaryError)

    def test_database_error_with_details(self):
        """Test DatabaseError with details."""
        details = {"operation": "insert", "table": "users", "error_code": 1062}
        error = DatabaseError("Duplicate entry", details=details)
        assert error.message == "Duplicate entry"
        assert error.details == details


class TestPerformanceExceptions:
    """Test suite for performance-related exceptions."""

    def test_performance_error_inheritance(self):
        """Test PerformanceError inheritance."""
        error = PerformanceError("Performance error")
        assert isinstance(error, PynomaryError)

    def test_memory_error_inheritance(self):
        """Test MemoryError inheritance."""
        error = MemoryError("Memory limit exceeded")
        assert isinstance(error, PerformanceError)
        assert isinstance(error, PynomaryError)

    def test_timeout_error_inheritance(self):
        """Test TimeoutError inheritance."""
        error = TimeoutError("Operation timed out")
        assert isinstance(error, PerformanceError)
        assert isinstance(error, PynomaryError)

    def test_resource_exhaustion_error_inheritance(self):
        """Test ResourceExhaustionError inheritance."""
        error = ResourceExhaustionError("Resources exhausted")
        assert isinstance(error, PerformanceError)
        assert isinstance(error, PynomaryError)

    def test_memory_error_with_details(self):
        """Test MemoryError with details."""
        details = {
            "requested": "8GB",
            "available": "4GB",
            "operation": "model_training",
        }
        error = MemoryError("Insufficient memory", details=details)
        assert error.message == "Insufficient memory"
        assert error.details == details


class TestAPIExceptions:
    """Test suite for API-related exceptions."""

    def test_api_error_inheritance(self):
        """Test APIError inheritance."""
        error = APIError("API error")
        assert isinstance(error, PynomaryError)

    def test_invalid_request_error_inheritance(self):
        """Test InvalidRequestError inheritance."""
        error = InvalidRequestError("Invalid request")
        assert isinstance(error, APIError)
        assert isinstance(error, PynomaryError)

    def test_rate_limit_exceeded_error_inheritance(self):
        """Test RateLimitExceededError inheritance."""
        error = RateLimitExceededError("Rate limit exceeded")
        assert isinstance(error, APIError)
        assert isinstance(error, PynomaryError)

    def test_service_unavailable_error_inheritance(self):
        """Test ServiceUnavailableError inheritance."""
        error = ServiceUnavailableError("Service unavailable")
        assert isinstance(error, APIError)
        assert isinstance(error, PynomaryError)

    def test_rate_limit_error_with_details(self):
        """Test RateLimitExceededError with details."""
        details = {"limit": 1000, "window": "1h", "reset_time": "2023-01-01T12:00:00Z"}
        error = RateLimitExceededError("Rate limit exceeded", details=details)
        assert error.message == "Rate limit exceeded"
        assert error.details == details


class TestIntegrationExceptions:
    """Test suite for integration-related exceptions."""

    def test_integration_error_inheritance(self):
        """Test IntegrationError inheritance."""
        error = IntegrationError("Integration error")
        assert isinstance(error, PynomaryError)

    def test_webhook_error_inheritance(self):
        """Test WebhookError inheritance."""
        error = WebhookError("Webhook failed")
        assert isinstance(error, IntegrationError)
        assert isinstance(error, PynomaryError)

    def test_notification_error_inheritance(self):
        """Test NotificationError inheritance."""
        error = NotificationError("Notification failed")
        assert isinstance(error, IntegrationError)
        assert isinstance(error, PynomaryError)

    def test_export_error_inheritance(self):
        """Test ExportError inheritance."""
        error = ExportError("Export failed")
        assert isinstance(error, IntegrationError)
        assert isinstance(error, PynomaryError)

    def test_import_error_inheritance(self):
        """Test ImportError inheritance."""
        error = ImportError("Import failed")
        assert isinstance(error, IntegrationError)
        assert isinstance(error, PynomaryError)

    def test_webhook_error_with_details(self):
        """Test WebhookError with details."""
        details = {"url": "https://example.com/webhook", "status_code": 500}
        error = WebhookError("Webhook delivery failed", details=details)
        assert error.message == "Webhook delivery failed"
        assert error.details == details


class TestReportingExceptions:
    """Test suite for reporting-related exceptions."""

    def test_reporting_error_inheritance(self):
        """Test ReportingError inheritance."""
        error = ReportingError("Reporting error")
        assert isinstance(error, PynomaryError)

    def test_report_not_found_error_inheritance(self):
        """Test ReportNotFoundError inheritance."""
        error = ReportNotFoundError("Report not found")
        assert isinstance(error, ReportingError)
        assert isinstance(error, PynomaryError)

    def test_dashboard_not_found_error_inheritance(self):
        """Test DashboardNotFoundError inheritance."""
        error = DashboardNotFoundError("Dashboard not found")
        assert isinstance(error, ReportingError)
        assert isinstance(error, PynomaryError)

    def test_metric_not_found_error_inheritance(self):
        """Test MetricNotFoundError inheritance."""
        error = MetricNotFoundError("Metric not found")
        assert isinstance(error, ReportingError)
        assert isinstance(error, PynomaryError)

    def test_report_generation_error_inheritance(self):
        """Test ReportGenerationError inheritance."""
        error = ReportGenerationError("Report generation failed")
        assert isinstance(error, ReportingError)
        assert isinstance(error, PynomaryError)

    def test_report_generation_error_with_details(self):
        """Test ReportGenerationError with details."""
        details = {"report_type": "anomaly_summary", "template": "monthly"}
        error = ReportGenerationError("Template not found", details=details)
        assert error.message == "Template not found"
        assert error.details == details


class TestCoreArchitectureExceptions:
    """Test suite for core architecture exceptions."""

    def test_data_ingestion_error_inheritance(self):
        """Test DataIngestionError inheritance."""
        error = DataIngestionError("Data ingestion failed")
        assert isinstance(error, InfrastructureError)
        assert isinstance(error, PynomaryError)

    def test_anomaly_detection_error_inheritance(self):
        """Test AnomalyDetectionError inheritance."""
        error = AnomalyDetectionError("Anomaly detection failed")
        assert isinstance(error, DetectionError)
        assert isinstance(error, PynomaryError)

    def test_alerting_error_inheritance(self):
        """Test AlertingError inheritance."""
        error = AlertingError("Alerting failed")
        assert isinstance(error, IntegrationError)
        assert isinstance(error, PynomaryError)

    def test_data_ingestion_error_with_details(self):
        """Test DataIngestionError with details."""
        details = {"source": "kafka", "topic": "events", "error": "connection_timeout"}
        error = DataIngestionError("Ingestion failed", details=details)
        assert error.message == "Ingestion failed"
        assert error.details == details


class TestExceptionUsagePatterns:
    """Test exception usage patterns and edge cases."""

    def test_exception_chaining(self):
        """Test exception chaining."""
        original_error = ValueError("Original error")

        try:
            raise ValidationError("Validation failed") from original_error
        except ValidationError as e:
            assert e.__cause__ == original_error
            assert isinstance(e, ValidationError)

    def test_exception_raising_and_catching(self):
        """Test raising and catching exceptions."""

        def raise_domain_error():
            raise DomainError("Domain error occurred")

        with pytest.raises(DomainError) as exc_info:
            raise_domain_error()

        assert exc_info.value.message == "Domain error occurred"
        assert isinstance(exc_info.value, PynomaryError)

    def test_exception_inheritance_chain(self):
        """Test exception inheritance chain."""
        error = UserNotFoundError("User not found")

        assert isinstance(error, UserNotFoundError)
        assert isinstance(error, UserError)
        assert isinstance(error, DomainError)
        assert isinstance(error, PynomaryError)
        assert isinstance(error, Exception)

    def test_multiple_inheritance_compatibility(self):
        """Test that exceptions work with multiple inheritance patterns."""
        # Test that all exception types can be caught by their base classes
        exceptions_and_bases = [
            (ValidationError("test"), DomainError),
            (UserNotFoundError("test"), UserError),
            (DatabaseError("test"), InfrastructureError),
            (ModelTrainingError("test"), ModelError),
            (DetectorNotFoundError("test"), DetectionError),
            (RateLimitExceededError("test"), APIError),
            (WebhookError("test"), IntegrationError),
            (ReportNotFoundError("test"), ReportingError),
            (MemoryError("test"), PerformanceError),
        ]

        for exception, base_class in exceptions_and_bases:
            assert isinstance(exception, base_class)
            assert isinstance(exception, PynomaryError)

    def test_exception_with_empty_message(self):
        """Test exception with empty message."""
        error = PynomaryError("")
        assert error.message == ""
        assert str(error) == ""

    def test_exception_with_special_characters(self):
        """Test exception with special characters."""
        special_message = "Error with special chars: àáâãäåæçèéêë 中文 🚀"
        error = PynomaryError(special_message)
        assert error.message == special_message
        assert str(error) == special_message

    def test_exception_details_modification(self):
        """Test exception details can be modified."""
        error = PynomaryError("Test error")
        error.details["new_key"] = "new_value"
        assert error.details["new_key"] == "new_value"

        # Test with initial details
        error2 = PynomaryError("Test error", details={"initial": "value"})
        error2.details["added"] = "later"
        assert error2.details["initial"] == "value"
        assert error2.details["added"] == "later"

    def test_exception_repr(self):
        """Test exception repr method."""
        error = PynomaryError("Test error")
        repr_str = repr(error)
        assert "PynomaryError" in repr_str
        assert "Test error" in repr_str

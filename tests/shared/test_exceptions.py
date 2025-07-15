"""
Tests for shared exceptions module.

This module tests the shared exception classes to ensure proper inheritance,
error handling, and consistent behavior across the system.
"""

import pytest

from pynomaly.shared.exceptions import (
    ConcurrencyError,
    ConfigurationError,
    DataValidationError,
    DetectionError,
    IntegrationError,
    ModelError,
    PerformanceError,
    PynomAlyBaseException,
    ResourceError,
    SecurityError,
)


class TestPynomAlyBaseException:
    """Test suite for the base exception class."""

    def test_base_exception_creation(self):
        """Test basic exception creation."""
        message = "Test base exception"
        exc = PynomAlyBaseException(message)

        assert str(exc) == message
        assert exc.args[0] == message
        assert isinstance(exc, Exception)

    def test_base_exception_with_details(self):
        """Test exception creation with additional details."""
        message = "Test exception with details"
        details = {"component": "test", "error_code": "TEST_001"}

        exc = PynomAlyBaseException(message, details=details)

        assert str(exc) == message
        assert hasattr(exc, "details")
        assert exc.details == details

    def test_base_exception_inheritance(self):
        """Test that base exception inherits from Exception."""
        exc = PynomAlyBaseException("test")

        assert isinstance(exc, Exception)
        assert isinstance(exc, PynomAlyBaseException)

    def test_base_exception_without_message(self):
        """Test exception creation without message."""
        exc = PynomAlyBaseException()

        # Should have some default representation
        assert str(exc) is not None

    def test_base_exception_repr(self):
        """Test exception representation."""
        message = "Test repr"
        exc = PynomAlyBaseException(message)

        repr_str = repr(exc)
        assert "PynomAlyBaseException" in repr_str
        assert message in repr_str


class TestConfigurationError:
    """Test suite for configuration-related exceptions."""

    def test_configuration_error_creation(self):
        """Test configuration error creation."""
        message = "Invalid configuration"
        exc = ConfigurationError(message)

        assert str(exc) == message
        assert isinstance(exc, PynomAlyBaseException)
        assert isinstance(exc, ConfigurationError)

    def test_configuration_error_with_config_path(self):
        """Test configuration error with config path."""
        message = "Missing required field"
        config_path = "model.training.batch_size"

        exc = ConfigurationError(message, config_path=config_path)

        assert str(exc) == message
        assert hasattr(exc, "config_path")
        assert exc.config_path == config_path

    def test_configuration_error_with_details(self):
        """Test configuration error with additional details."""
        message = "Configuration validation failed"
        details = {
            "field": "learning_rate",
            "value": -0.1,
            "expected": "positive float",
        }

        exc = ConfigurationError(message, details=details)

        assert exc.details == details

    def test_configuration_error_inheritance_chain(self):
        """Test inheritance chain for configuration error."""
        exc = ConfigurationError("test")

        assert isinstance(exc, Exception)
        assert isinstance(exc, PynomAlyBaseException)
        assert isinstance(exc, ConfigurationError)


class TestDataValidationError:
    """Test suite for data validation exceptions."""

    def test_data_validation_error_creation(self):
        """Test data validation error creation."""
        message = "Invalid data format"
        exc = DataValidationError(message)

        assert str(exc) == message
        assert isinstance(exc, PynomAlyBaseException)
        assert isinstance(exc, DataValidationError)

    def test_data_validation_error_with_field_info(self):
        """Test data validation error with field information."""
        message = "Invalid field value"
        field_name = "feature_1"
        field_value = "invalid_value"

        exc = DataValidationError(
            message, field_name=field_name, field_value=field_value
        )

        assert hasattr(exc, "field_name")
        assert hasattr(exc, "field_value")
        assert exc.field_name == field_name
        assert exc.field_value == field_value

    def test_data_validation_error_with_validation_rules(self):
        """Test data validation error with validation rules."""
        message = "Data validation failed"
        validation_rules = {"type": "numeric", "range": [0, 100], "required": True}

        exc = DataValidationError(message, validation_rules=validation_rules)

        assert hasattr(exc, "validation_rules")
        assert exc.validation_rules == validation_rules

    def test_data_validation_error_with_dataset_info(self):
        """Test data validation error with dataset information."""
        message = "Dataset validation failed"
        dataset_info = {
            "name": "test_dataset",
            "rows": 1000,
            "columns": 5,
            "missing_values": 25,
        }

        exc = DataValidationError(message, dataset_info=dataset_info)

        assert hasattr(exc, "dataset_info")
        assert exc.dataset_info == dataset_info


class TestDetectionError:
    """Test suite for detection-related exceptions."""

    def test_detection_error_creation(self):
        """Test detection error creation."""
        message = "Detection algorithm failed"
        exc = DetectionError(message)

        assert str(exc) == message
        assert isinstance(exc, PynomAlyBaseException)
        assert isinstance(exc, DetectionError)

    def test_detection_error_with_algorithm_info(self):
        """Test detection error with algorithm information."""
        message = "Algorithm execution failed"
        algorithm = "isolation_forest"

        exc = DetectionError(message, algorithm=algorithm)

        assert hasattr(exc, "algorithm")
        assert exc.algorithm == algorithm

    def test_detection_error_with_detector_state(self):
        """Test detection error with detector state."""
        message = "Detector not fitted"
        detector_state = {
            "fitted": False,
            "algorithm": "one_class_svm",
            "last_training": None,
        }

        exc = DetectionError(message, detector_state=detector_state)

        assert hasattr(exc, "detector_state")
        assert exc.detector_state == detector_state

    def test_detection_error_with_performance_metrics(self):
        """Test detection error with performance context."""
        message = "Detection performance degraded"
        performance_metrics = {
            "execution_time": 150.5,
            "memory_usage": 1024,
            "accuracy": 0.65,
        }

        exc = DetectionError(message, performance_metrics=performance_metrics)

        assert hasattr(exc, "performance_metrics")
        assert exc.performance_metrics == performance_metrics


class TestModelError:
    """Test suite for model-related exceptions."""

    def test_model_error_creation(self):
        """Test model error creation."""
        message = "Model loading failed"
        exc = ModelError(message)

        assert str(exc) == message
        assert isinstance(exc, PynomAlyBaseException)
        assert isinstance(exc, ModelError)

    def test_model_error_with_model_info(self):
        """Test model error with model information."""
        message = "Model validation failed"
        model_info = {
            "name": "anomaly_detector_v1",
            "version": "1.2.3",
            "algorithm": "isolation_forest",
            "size_mb": 15.7,
        }

        exc = ModelError(message, model_info=model_info)

        assert hasattr(exc, "model_info")
        assert exc.model_info == model_info

    def test_model_error_with_checkpoint_info(self):
        """Test model error with checkpoint information."""
        message = "Checkpoint corruption detected"
        checkpoint_path = "/models/checkpoint_epoch_100.pkl"

        exc = ModelError(message, checkpoint_path=checkpoint_path)

        assert hasattr(exc, "checkpoint_path")
        assert exc.checkpoint_path == checkpoint_path

    def test_model_error_with_training_context(self):
        """Test model error with training context."""
        message = "Training convergence failed"
        training_context = {
            "epoch": 95,
            "loss": 0.856,
            "learning_rate": 0.001,
            "batch_size": 32,
        }

        exc = ModelError(message, training_context=training_context)

        assert hasattr(exc, "training_context")
        assert exc.training_context == training_context


class TestIntegrationError:
    """Test suite for integration-related exceptions."""

    def test_integration_error_creation(self):
        """Test integration error creation."""
        message = "External service unavailable"
        exc = IntegrationError(message)

        assert str(exc) == message
        assert isinstance(exc, PynomAlyBaseException)
        assert isinstance(exc, IntegrationError)

    def test_integration_error_with_service_info(self):
        """Test integration error with service information."""
        message = "Database connection failed"
        service_name = "postgresql"
        endpoint = "localhost:5432"

        exc = IntegrationError(message, service_name=service_name, endpoint=endpoint)

        assert hasattr(exc, "service_name")
        assert hasattr(exc, "endpoint")
        assert exc.service_name == service_name
        assert exc.endpoint == endpoint

    def test_integration_error_with_response_info(self):
        """Test integration error with response information."""
        message = "API request failed"
        response_info = {
            "status_code": 503,
            "response_body": {"error": "Service temporarily unavailable"},
            "headers": {"retry-after": "300"},
        }

        exc = IntegrationError(message, response_info=response_info)

        assert hasattr(exc, "response_info")
        assert exc.response_info == response_info

    def test_integration_error_with_retry_info(self):
        """Test integration error with retry information."""
        message = "Max retries exceeded"
        retry_info = {
            "attempt": 5,
            "max_retries": 5,
            "backoff_strategy": "exponential",
            "total_duration": 127.3,
        }

        exc = IntegrationError(message, retry_info=retry_info)

        assert hasattr(exc, "retry_info")
        assert exc.retry_info == retry_info


class TestPerformanceError:
    """Test suite for performance-related exceptions."""

    def test_performance_error_creation(self):
        """Test performance error creation."""
        message = "Operation timeout"
        exc = PerformanceError(message)

        assert str(exc) == message
        assert isinstance(exc, PynomAlyBaseException)
        assert isinstance(exc, PerformanceError)

    def test_performance_error_with_timing_info(self):
        """Test performance error with timing information."""
        message = "Execution time exceeded threshold"
        timing_info = {
            "execution_time": 120.5,
            "threshold": 60.0,
            "operation": "model_training",
        }

        exc = PerformanceError(message, timing_info=timing_info)

        assert hasattr(exc, "timing_info")
        assert exc.timing_info == timing_info

    def test_performance_error_with_resource_usage(self):
        """Test performance error with resource usage."""
        message = "Memory usage exceeded limit"
        resource_usage = {
            "memory_mb": 2048,
            "memory_limit_mb": 1024,
            "cpu_percent": 95.2,
            "disk_io_mb": 150.7,
        }

        exc = PerformanceError(message, resource_usage=resource_usage)

        assert hasattr(exc, "resource_usage")
        assert exc.resource_usage == resource_usage

    def test_performance_error_with_bottleneck_info(self):
        """Test performance error with bottleneck information."""
        message = "Performance bottleneck detected"
        bottleneck_info = {
            "component": "data_loader",
            "bottleneck_type": "io_bound",
            "suggested_fix": "increase batch size",
            "impact_factor": 0.75,
        }

        exc = PerformanceError(message, bottleneck_info=bottleneck_info)

        assert hasattr(exc, "bottleneck_info")
        assert exc.bottleneck_info == bottleneck_info


class TestSecurityError:
    """Test suite for security-related exceptions."""

    def test_security_error_creation(self):
        """Test security error creation."""
        message = "Authentication failed"
        exc = SecurityError(message)

        assert str(exc) == message
        assert isinstance(exc, PynomAlyBaseException)
        assert isinstance(exc, SecurityError)

    def test_security_error_with_security_context(self):
        """Test security error with security context."""
        message = "Unauthorized access attempt"
        security_context = {
            "user_id": "user_123",
            "requested_resource": "/api/models/sensitive",
            "permission_required": "model:read:sensitive",
            "client_ip": "192.168.1.100",
        }

        exc = SecurityError(message, security_context=security_context)

        assert hasattr(exc, "security_context")
        assert exc.security_context == security_context

    def test_security_error_with_threat_info(self):
        """Test security error with threat information."""
        message = "Potential security threat detected"
        threat_info = {
            "threat_type": "injection_attempt",
            "severity": "high",
            "source": "external",
            "blocked": True,
        }

        exc = SecurityError(message, threat_info=threat_info)

        assert hasattr(exc, "threat_info")
        assert exc.threat_info == threat_info


class TestResourceError:
    """Test suite for resource-related exceptions."""

    def test_resource_error_creation(self):
        """Test resource error creation."""
        message = "Insufficient resources"
        exc = ResourceError(message)

        assert str(exc) == message
        assert isinstance(exc, PynomAlyBaseException)
        assert isinstance(exc, ResourceError)

    def test_resource_error_with_resource_info(self):
        """Test resource error with resource information."""
        message = "Disk space exhausted"
        resource_info = {
            "resource_type": "disk_space",
            "available": 512,  # MB
            "required": 2048,  # MB
            "usage_percent": 97.5,
        }

        exc = ResourceError(message, resource_info=resource_info)

        assert hasattr(exc, "resource_info")
        assert exc.resource_info == resource_info

    def test_resource_error_with_allocation_info(self):
        """Test resource error with allocation information."""
        message = "Resource allocation failed"
        allocation_info = {
            "requested_memory": 4096,
            "available_memory": 2048,
            "allocation_strategy": "best_fit",
            "allocation_id": "alloc_789",
        }

        exc = ResourceError(message, allocation_info=allocation_info)

        assert hasattr(exc, "allocation_info")
        assert exc.allocation_info == allocation_info


class TestConcurrencyError:
    """Test suite for concurrency-related exceptions."""

    def test_concurrency_error_creation(self):
        """Test concurrency error creation."""
        message = "Deadlock detected"
        exc = ConcurrencyError(message)

        assert str(exc) == message
        assert isinstance(exc, PynomAlyBaseException)
        assert isinstance(exc, ConcurrencyError)

    def test_concurrency_error_with_lock_info(self):
        """Test concurrency error with lock information."""
        message = "Lock acquisition timeout"
        lock_info = {
            "lock_name": "model_training_lock",
            "timeout_seconds": 30.0,
            "current_holder": "worker_thread_2",
            "queue_position": 3,
        }

        exc = ConcurrencyError(message, lock_info=lock_info)

        assert hasattr(exc, "lock_info")
        assert exc.lock_info == lock_info

    def test_concurrency_error_with_thread_info(self):
        """Test concurrency error with thread information."""
        message = "Thread synchronization failed"
        thread_info = {
            "thread_id": "thread_123",
            "thread_name": "data_processor",
            "state": "blocked",
            "waiting_for": "resource_semaphore",
        }

        exc = ConcurrencyError(message, thread_info=thread_info)

        assert hasattr(exc, "thread_info")
        assert exc.thread_info == thread_info

    def test_concurrency_error_with_race_condition_info(self):
        """Test concurrency error with race condition information."""
        message = "Race condition detected"
        race_condition_info = {
            "competing_operations": ["read", "write"],
            "shared_resource": "model_state",
            "detection_method": "mutex_violation",
            "resolution": "retry_with_backoff",
        }

        exc = ConcurrencyError(message, race_condition_info=race_condition_info)

        assert hasattr(exc, "race_condition_info")
        assert exc.race_condition_info == race_condition_info


class TestExceptionHierarchy:
    """Test the complete exception hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from base exception."""
        exception_classes = [
            ConfigurationError,
            DataValidationError,
            DetectionError,
            ModelError,
            IntegrationError,
            PerformanceError,
            SecurityError,
            ResourceError,
            ConcurrencyError,
        ]

        for exc_class in exception_classes:
            exc = exc_class("test message")
            assert isinstance(exc, PynomAlyBaseException)
            assert isinstance(exc, Exception)

    def test_exception_type_discrimination(self):
        """Test that different exception types can be distinguished."""
        exceptions = [
            ConfigurationError("config error"),
            DataValidationError("validation error"),
            DetectionError("detection error"),
            ModelError("model error"),
            IntegrationError("integration error"),
            PerformanceError("performance error"),
            SecurityError("security error"),
            ResourceError("resource error"),
            ConcurrencyError("concurrency error"),
        ]

        # Each exception should be its own type
        for i, exc1 in enumerate(exceptions):
            for j, exc2 in enumerate(exceptions):
                if i != j:
                    assert type(exc1) != type(exc2)

    def test_exception_catching_specificity(self):
        """Test that exceptions can be caught at different levels of specificity."""

        def raise_config_error():
            raise ConfigurationError("test config error")

        # Should be catchable as specific type
        with pytest.raises(ConfigurationError):
            raise_config_error()

        # Should be catchable as base type
        with pytest.raises(PynomAlyBaseException):
            raise_config_error()

        # Should be catchable as general Exception
        with pytest.raises(Exception):
            raise_config_error()

    def test_exception_details_preservation(self):
        """Test that exception details are preserved through the hierarchy."""
        details = {"component": "test", "error_code": "ERR_001"}

        exc = ConfigurationError("test error", details=details)

        # Details should be preserved when caught as base type
        try:
            raise exc
        except PynomAlyBaseException as caught_exc:
            assert hasattr(caught_exc, "details")
            assert caught_exc.details == details

        # Details should be preserved when caught as Exception
        try:
            raise exc
        except Exception as caught_exc:
            assert hasattr(caught_exc, "details")
            assert caught_exc.details == details


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

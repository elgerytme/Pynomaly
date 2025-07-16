"""Tests for storage exceptions."""

import pytest

from monorepo.domain.exceptions.base import InfrastructureError, PynamolyError
from monorepo.domain.exceptions.storage_exceptions import (
    SerializationError,
    StorageConfigurationError,
    StorageConnectionError,
    StorageError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageQuotaError,
    StorageTimeoutError,
)


class TestStorageError:
    """Test suite for StorageError base class."""

    def test_basic_creation(self):
        """Test basic StorageError creation."""
        error = StorageError()

        assert str(error) == "Storage operation failed"
        assert isinstance(error, InfrastructureError)
        assert isinstance(error, PynamolyError)

    def test_creation_with_message(self):
        """Test StorageError creation with custom message."""
        error = StorageError("Custom storage error")

        assert "Custom storage error" in str(error)

    def test_creation_with_storage_type(self):
        """Test StorageError creation with storage type."""
        error = StorageError("Storage failed", storage_type="S3")

        assert "Storage failed" in str(error)
        assert error.details.get("storage_type") == "S3"

    def test_creation_with_operation(self):
        """Test StorageError creation with operation."""
        error = StorageError("Upload failed", operation="upload")

        assert "Upload failed" in str(error)
        assert error.details.get("operation") == "upload"

    def test_creation_with_all_parameters(self):
        """Test StorageError creation with all parameters."""
        error = StorageError(
            "Complex storage error",
            storage_type="Azure",
            operation="download",
            file_size=1024,
            error_code="NETWORK_ERROR",
        )

        assert "Complex storage error" in str(error)
        assert error.details.get("storage_type") == "Azure"
        assert error.details.get("operation") == "download"
        assert error.details.get("file_size") == 1024
        assert error.details.get("error_code") == "NETWORK_ERROR"

    def test_inheritance_chain(self):
        """Test StorageError inheritance chain."""
        error = StorageError()

        assert isinstance(error, StorageError)
        assert isinstance(error, InfrastructureError)
        assert isinstance(error, PynamolyError)
        assert isinstance(error, Exception)


class TestStorageConnectionError:
    """Test suite for StorageConnectionError."""

    def test_basic_creation(self):
        """Test basic StorageConnectionError creation."""
        error = StorageConnectionError()

        assert "Failed to connect to storage" in str(error)
        assert isinstance(error, StorageError)

    def test_creation_with_storage_type(self):
        """Test StorageConnectionError creation with storage type."""
        error = StorageConnectionError("Connection timeout", storage_type="GCS")

        assert "Connection timeout" in str(error)
        assert error.details.get("storage_type") == "GCS"
        assert error.details.get("operation") == "connect"

    def test_creation_with_endpoint(self):
        """Test StorageConnectionError creation with endpoint."""
        error = StorageConnectionError(
            "Cannot reach endpoint",
            storage_type="MinIO",
            endpoint="https://minio.example.com:9000",
        )

        assert "Cannot reach endpoint" in str(error)
        assert error.details.get("endpoint") == "https://minio.example.com:9000"

    def test_creation_with_all_parameters(self):
        """Test StorageConnectionError creation with all parameters."""
        error = StorageConnectionError(
            "Connection failed",
            storage_type="S3",
            endpoint="s3.amazonaws.com",
            region="us-west-2",
            timeout=30,
        )

        assert "Connection failed" in str(error)
        assert error.details.get("storage_type") == "S3"
        assert error.details.get("endpoint") == "s3.amazonaws.com"
        assert error.details.get("region") == "us-west-2"
        assert error.details.get("timeout") == 30


class TestStorageNotFoundError:
    """Test suite for StorageNotFoundError."""

    def test_basic_creation(self):
        """Test basic StorageNotFoundError creation."""
        error = StorageNotFoundError()

        assert "Storage resource not found" in str(error)
        assert isinstance(error, StorageError)

    def test_creation_with_resource_path(self):
        """Test StorageNotFoundError creation with resource path."""
        error = StorageNotFoundError(
            "File not found", storage_type="S3", resource_path="bucket/path/to/file.txt"
        )

        assert "File not found" in str(error)
        assert error.details.get("storage_type") == "S3"
        assert error.details.get("resource_path") == "bucket/path/to/file.txt"
        assert error.details.get("operation") == "find"

    def test_creation_with_all_parameters(self):
        """Test StorageNotFoundError creation with all parameters."""
        error = StorageNotFoundError(
            "Bucket does not exist",
            storage_type="Azure",
            resource_path="container/blob.csv",
            object_type="blob",
            last_modified=None,
        )

        assert "Bucket does not exist" in str(error)
        assert error.details.get("storage_type") == "Azure"
        assert error.details.get("resource_path") == "container/blob.csv"
        assert error.details.get("object_type") == "blob"
        assert "last_modified" in error.details


class TestStoragePermissionError:
    """Test suite for StoragePermissionError."""

    def test_basic_creation(self):
        """Test basic StoragePermissionError creation."""
        error = StoragePermissionError()

        assert "Storage permission denied" in str(error)
        assert isinstance(error, StorageError)

    def test_creation_with_operation(self):
        """Test StoragePermissionError creation with operation."""
        error = StoragePermissionError(
            "Write access denied", storage_type="S3", operation="write"
        )

        assert "Write access denied" in str(error)
        assert error.details.get("storage_type") == "S3"
        assert error.details.get("operation") == "write"

    def test_creation_with_resource_path(self):
        """Test StoragePermissionError creation with resource path."""
        error = StoragePermissionError(
            "Access forbidden",
            storage_type="GCS",
            operation="read",
            resource_path="bucket/secure/file.txt",
        )

        assert "Access forbidden" in str(error)
        assert error.details.get("resource_path") == "bucket/secure/file.txt"

    def test_creation_with_all_parameters(self):
        """Test StoragePermissionError creation with all parameters."""
        error = StoragePermissionError(
            "Insufficient permissions",
            storage_type="Azure",
            operation="delete",
            resource_path="container/protected.dat",
            user_id="user123",
            required_permission="storage.objects.delete",
        )

        assert "Insufficient permissions" in str(error)
        assert error.details.get("user_id") == "user123"
        assert error.details.get("required_permission") == "storage.objects.delete"


class TestStorageTimeoutError:
    """Test suite for StorageTimeoutError."""

    def test_basic_creation(self):
        """Test basic StorageTimeoutError creation."""
        error = StorageTimeoutError()

        assert "Storage operation timed out" in str(error)
        assert isinstance(error, StorageError)

    def test_creation_with_timeout_seconds(self):
        """Test StorageTimeoutError creation with timeout duration."""
        error = StorageTimeoutError(
            "Upload timeout",
            storage_type="S3",
            operation="upload",
            timeout_seconds=30.5,
        )

        assert "Upload timeout" in str(error)
        assert error.details.get("storage_type") == "S3"
        assert error.details.get("operation") == "upload"
        assert error.details.get("timeout_seconds") == 30.5

    def test_creation_with_all_parameters(self):
        """Test StorageTimeoutError creation with all parameters."""
        error = StorageTimeoutError(
            "Large file upload timeout",
            storage_type="GCS",
            operation="multipart_upload",
            timeout_seconds=300.0,
            file_size_mb=1024,
            network_speed="slow",
        )

        assert "Large file upload timeout" in str(error)
        assert error.details.get("timeout_seconds") == 300.0
        assert error.details.get("file_size_mb") == 1024
        assert error.details.get("network_speed") == "slow"


class TestStorageConfigurationError:
    """Test suite for StorageConfigurationError."""

    def test_basic_creation(self):
        """Test basic StorageConfigurationError creation."""
        error = StorageConfigurationError()

        assert "Storage configuration error" in str(error)
        assert isinstance(error, StorageError)

    def test_creation_with_parameter(self):
        """Test StorageConfigurationError creation with parameter."""
        error = StorageConfigurationError(
            "Invalid region", storage_type="S3", parameter="region"
        )

        assert "Invalid region" in str(error)
        assert error.details.get("storage_type") == "S3"
        assert error.details.get("parameter") == "region"
        assert error.details.get("operation") == "configure"

    def test_creation_with_all_parameters(self):
        """Test StorageConfigurationError creation with all parameters."""
        error = StorageConfigurationError(
            "Authentication credentials invalid",
            storage_type="Azure",
            parameter="access_key",
            value="***redacted***",
            expected_format="alphanumeric",
        )

        assert "Authentication credentials invalid" in str(error)
        assert error.details.get("parameter") == "access_key"
        assert error.details.get("value") == "***redacted***"
        assert error.details.get("expected_format") == "alphanumeric"


class TestStorageQuotaError:
    """Test suite for StorageQuotaError."""

    def test_basic_creation(self):
        """Test basic StorageQuotaError creation."""
        error = StorageQuotaError()

        assert "Storage quota exceeded" in str(error)
        assert isinstance(error, StorageError)

    def test_creation_with_usage_info(self):
        """Test StorageQuotaError creation with usage information."""
        error = StorageQuotaError(
            "Upload would exceed quota",
            storage_type="S3",
            current_usage=950,
            quota_limit=1000,
        )

        assert "Upload would exceed quota" in str(error)
        assert error.details.get("storage_type") == "S3"
        assert error.details.get("current_usage") == 950
        assert error.details.get("quota_limit") == 1000
        assert error.details.get("operation") == "quota_check"

    def test_creation_with_all_parameters(self):
        """Test StorageQuotaError creation with all parameters."""
        error = StorageQuotaError(
            "Monthly bandwidth limit reached",
            storage_type="GCS",
            current_usage=2048,
            quota_limit=2000,
            unit="GB",
            quota_type="bandwidth",
            reset_date="2024-02-01",
        )

        assert "Monthly bandwidth limit reached" in str(error)
        assert error.details.get("current_usage") == 2048
        assert error.details.get("quota_limit") == 2000
        assert error.details.get("unit") == "GB"
        assert error.details.get("quota_type") == "bandwidth"
        assert error.details.get("reset_date") == "2024-02-01"


class TestSerializationError:
    """Test suite for SerializationError."""

    def test_basic_creation(self):
        """Test basic SerializationError creation."""
        error = SerializationError()

        assert "Serialization/deserialization failed" in str(error)
        assert isinstance(error, StorageError)

    def test_creation_with_format_type(self):
        """Test SerializationError creation with format type."""
        error = SerializationError(
            "JSON encoding failed", format_type="json", operation="serialize"
        )

        assert "JSON encoding failed" in str(error)
        assert error.details.get("format_type") == "json"
        assert error.details.get("operation") == "serialize"

    def test_creation_with_deserialization(self):
        """Test SerializationError creation for deserialization."""
        error = SerializationError(
            "Pickle loading failed", format_type="pickle", operation="deserialize"
        )

        assert "Pickle loading failed" in str(error)
        assert error.details.get("format_type") == "pickle"
        assert error.details.get("operation") == "deserialize"

    def test_creation_with_all_parameters(self):
        """Test SerializationError creation with all parameters."""
        error = SerializationError(
            "YAML parsing error",
            format_type="yaml",
            operation="deserialize",
            line_number=15,
            character_position=23,
            invalid_token="@invalid",
        )

        assert "YAML parsing error" in str(error)
        assert error.details.get("format_type") == "yaml"
        assert error.details.get("operation") == "deserialize"
        assert error.details.get("line_number") == 15
        assert error.details.get("character_position") == 23
        assert error.details.get("invalid_token") == "@invalid"


class TestStorageExceptionsInheritance:
    """Test inheritance relationships between storage exceptions."""

    def test_all_inherit_from_storage_error(self):
        """Test that all storage exceptions inherit from StorageError."""
        storage_exceptions = [
            StorageConnectionError,
            StorageNotFoundError,
            StoragePermissionError,
            StorageTimeoutError,
            StorageConfigurationError,
            StorageQuotaError,
            SerializationError,
        ]

        for exception_class in storage_exceptions:
            error = exception_class()
            assert isinstance(error, StorageError)
            assert isinstance(error, InfrastructureError)
            assert isinstance(error, PynamolyError)

    def test_inheritance_chain_consistency(self):
        """Test inheritance chain consistency."""
        error = StorageConnectionError("Test error")

        # Should be instance of all parent classes
        assert isinstance(error, StorageConnectionError)
        assert isinstance(error, StorageError)
        assert isinstance(error, InfrastructureError)
        assert isinstance(error, PynamolyError)
        assert isinstance(error, Exception)

    def test_method_resolution_order(self):
        """Test method resolution order."""
        # All storage exceptions should have StorageError in their MRO
        storage_exceptions = [
            StorageConnectionError,
            StorageNotFoundError,
            StoragePermissionError,
            StorageTimeoutError,
            StorageConfigurationError,
            StorageQuotaError,
            SerializationError,
        ]

        for exception_class in storage_exceptions:
            mro = exception_class.__mro__
            assert StorageError in mro
            assert InfrastructureError in mro
            assert PynamolyError in mro


class TestStorageExceptionsUsagePatterns:
    """Test common usage patterns for storage exceptions."""

    def test_exception_chaining(self):
        """Test exception chaining with storage exceptions."""
        original_error = ConnectionError("Network unreachable")

        storage_error = StorageConnectionError(
            "Failed to connect to S3", storage_type="S3", endpoint="s3.amazonaws.com"
        )

        # Exception chaining should work
        try:
            raise storage_error from original_error
        except StorageConnectionError as e:
            assert e.__cause__ == original_error
            assert isinstance(e, StorageConnectionError)

    def test_catching_by_base_class(self):
        """Test catching storage exceptions by base class."""
        errors = [
            StorageConnectionError("Connection failed"),
            StorageNotFoundError("File not found"),
            StoragePermissionError("Access denied"),
            StorageTimeoutError("Operation timed out"),
        ]

        for error in errors:
            with pytest.raises(StorageError):
                raise error

            with pytest.raises(InfrastructureError):
                raise error

            with pytest.raises(PynamolyError):
                raise error

    def test_error_details_consistency(self):
        """Test that error details are consistently available."""
        errors = [
            StorageConnectionError("Test", storage_type="S3", endpoint="test.com"),
            StorageNotFoundError("Test", storage_type="Azure", resource_path="/path"),
            StoragePermissionError("Test", storage_type="GCS", operation="read"),
            StorageTimeoutError("Test", storage_type="MinIO", timeout_seconds=30.0),
            StorageConfigurationError("Test", storage_type="S3", parameter="region"),
            StorageQuotaError("Test", storage_type="Azure", current_usage=100),
            SerializationError("Test", format_type="json", operation="serialize"),
        ]

        for error in errors:
            assert hasattr(error, "details")
            assert isinstance(error.details, dict)
            assert len(error.details) > 0

    def test_string_representation_quality(self):
        """Test string representation quality."""
        error = StorageConnectionError(
            "Connection failed",
            storage_type="S3",
            endpoint="s3.amazonaws.com",
            region="us-west-2",
        )

        error_str = str(error)

        # Should contain the main message
        assert "Connection failed" in error_str

        # Should be informative but not too verbose
        assert len(error_str) > 10
        assert len(error_str) < 500

    def test_custom_details_preservation(self):
        """Test that custom details are preserved."""
        custom_details = {
            "request_id": "req_123456",
            "retry_count": 3,
            "user_agent": "pynomaly/1.0",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        error = StorageError(
            "Custom error with details", storage_type="Custom", **custom_details
        )

        for key, value in custom_details.items():
            assert error.details.get(key) == value

    def test_none_values_handling(self):
        """Test handling of None values in parameters."""
        # Should not crash with None values
        error = StorageConnectionError("Test error", storage_type=None, endpoint=None)

        assert isinstance(error, StorageConnectionError)
        # None values should not be added to details
        assert (
            "storage_type" not in error.details or error.details["storage_type"] is None
        )
        assert "endpoint" not in error.details or error.details["endpoint"] is None

    def test_error_context_addition(self):
        """Test adding context to storage errors."""
        error = StorageError("Original error")

        # Add context using the with_context method from base class
        error_with_context = error.with_context(
            retry_attempt=2, total_retries=3, backoff_seconds=5.0
        )

        assert error_with_context is error  # Should return same instance
        assert error.details["retry_attempt"] == 2
        assert error.details["total_retries"] == 3
        assert error.details["backoff_seconds"] == 5.0

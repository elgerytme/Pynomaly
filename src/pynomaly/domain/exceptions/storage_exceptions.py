"""Storage-related exceptions."""

from __future__ import annotations

from typing import Any

from .base import InfrastructureError


class StorageError(InfrastructureError):
    """Base exception for storage-related errors."""

    def __init__(
        self,
        message: str = "Storage operation failed",
        storage_type: str | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize storage error.

        Args:
            message: Error message
            storage_type: Type of storage (e.g., 'S3', 'Azure', 'GCS')
            operation: Operation that failed
            **kwargs: Additional details
        """
        details = kwargs
        if storage_type:
            details["storage_type"] = storage_type
        if operation:
            details["operation"] = operation

        super().__init__(message, "storage", operation, **details)


class StorageConnectionError(StorageError):
    """Exception raised when storage connection fails."""

    def __init__(
        self,
        message: str = "Failed to connect to storage",
        storage_type: str | None = None,
        endpoint: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize storage connection error.

        Args:
            message: Error message
            storage_type: Type of storage
            endpoint: Storage endpoint
            **kwargs: Additional details
        """
        details = kwargs
        if endpoint:
            details["endpoint"] = endpoint

        super().__init__(message, storage_type, "connect", **details)


class StorageNotFoundError(StorageError):
    """Exception raised when storage resource is not found."""

    def __init__(
        self,
        message: str = "Storage resource not found",
        storage_type: str | None = None,
        resource_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize storage not found error.

        Args:
            message: Error message
            storage_type: Type of storage
            resource_path: Path to the resource
            **kwargs: Additional details
        """
        details = kwargs
        if resource_path:
            details["resource_path"] = resource_path

        super().__init__(message, storage_type, "find", **details)


class StoragePermissionError(StorageError):
    """Exception raised when storage permission is denied."""

    def __init__(
        self,
        message: str = "Storage permission denied",
        storage_type: str | None = None,
        operation: str | None = None,
        resource_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize storage permission error.

        Args:
            message: Error message
            storage_type: Type of storage
            operation: Operation that was denied
            resource_path: Path to the resource
            **kwargs: Additional details
        """
        details = kwargs
        if resource_path:
            details["resource_path"] = resource_path

        super().__init__(message, storage_type, operation, **details)


class StorageTimeoutError(StorageError):
    """Exception raised when storage operation times out."""

    def __init__(
        self,
        message: str = "Storage operation timed out",
        storage_type: str | None = None,
        operation: str | None = None,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize storage timeout error.

        Args:
            message: Error message
            storage_type: Type of storage
            operation: Operation that timed out
            timeout_seconds: Timeout duration in seconds
            **kwargs: Additional details
        """
        details = kwargs
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds

        super().__init__(message, storage_type, operation, **details)


class StorageConfigurationError(StorageError):
    """Exception raised when storage configuration is invalid."""

    def __init__(
        self,
        message: str = "Storage configuration error",
        storage_type: str | None = None,
        parameter: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize storage configuration error.

        Args:
            message: Error message
            storage_type: Type of storage
            parameter: Configuration parameter that is invalid
            **kwargs: Additional details
        """
        details = kwargs
        if parameter:
            details["parameter"] = parameter

        super().__init__(message, storage_type, "configure", **details)


class StorageQuotaError(StorageError):
    """Exception raised when storage quota is exceeded."""

    def __init__(
        self,
        message: str = "Storage quota exceeded",
        storage_type: str | None = None,
        current_usage: int | None = None,
        quota_limit: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize storage quota error.

        Args:
            message: Error message
            storage_type: Type of storage
            current_usage: Current storage usage
            quota_limit: Storage quota limit
            **kwargs: Additional details
        """
        details = kwargs
        if current_usage:
            details["current_usage"] = current_usage
        if quota_limit:
            details["quota_limit"] = quota_limit

        super().__init__(message, storage_type, "quota_check", **details)


class SerializationError(StorageError):
    """Exception raised when serialization/deserialization fails."""

    def __init__(
        self,
        message: str = "Serialization/deserialization failed",
        format_type: str | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize serialization error.

        Args:
            message: Error message
            format_type: Serialization format (e.g., 'json', 'pickle')
            operation: Operation that failed ('serialize' or 'deserialize')
            **kwargs: Additional details
        """
        details = kwargs
        if format_type:
            details["format_type"] = format_type

        super().__init__(message, "serialization", operation, **details)
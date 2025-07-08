"""Storage-related exceptions for cloud storage operations."""

from __future__ import annotations

from .base import DomainException


class StorageError(DomainException):
    """Base exception for storage-related errors."""
    pass


class StorageConnectionError(StorageError):
    """Exception raised when connection to storage fails."""
    pass


class StorageAuthenticationError(StorageError):
    """Exception raised when authentication to storage fails."""
    pass


class StoragePermissionError(StorageError):
    """Exception raised when permission is denied for storage operation."""
    pass


class StorageNotFoundError(StorageError):
    """Exception raised when storage resource is not found."""
    pass


class StorageQuotaExceededError(StorageError):
    """Exception raised when storage quota is exceeded."""
    pass


class StorageTimeoutError(StorageError):
    """Exception raised when storage operation times out."""
    pass


class StorageValidationError(StorageError):
    """Exception raised when storage validation fails."""
    pass

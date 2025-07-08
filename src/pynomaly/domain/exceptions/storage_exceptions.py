"""Storage-specific domain exceptions."""

from __future__ import annotations

from .base import DomainError


class StorageException(DomainError):
    """Base exception for storage-related errors."""
    pass


class StorageError(StorageException):
    """General storage operation error."""
    pass


class StorageAuthenticationError(StorageException):
    """Authentication-related storage error."""
    pass


class StoragePermissionError(StorageException):
    """Permission-related storage error."""
    pass


class StorageNotFoundError(StorageException):
    """Object not found error."""
    pass


class StorageConnectionError(StorageException):
    """Storage connection error."""
    pass


class EncryptionError(StorageException):
    """Encryption/decryption error."""
    pass


class QuotaExceededError(StorageException):
    """Storage quota exceeded error."""
    pass


class InvalidOperationError(StorageException):
    """Invalid operation error."""
    pass


class CorruptedDataError(StorageException):
    """Data corruption error."""
    pass

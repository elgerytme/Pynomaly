"""Factory for creating storage adapters based on StorageBackend enum."""

from __future__ import annotations

from typing import Optional

from pynomaly.domain.value_objects.model_storage_info import StorageBackend
from pynomaly.domain.value_objects.storage_credentials import StorageCredentials
from pynomaly.domain.services.cloud_storage_adapter import AbstractCloudStorageAdapter


class StorageAdapterFactory:
    """Factory for creating appropriate storage adapters based on StorageBackend enum."""
    
    def __init__(self):
        """Initialize the storage adapter factory."""
        self._adapters = {}
        self._register_adapters()
    
    def _register_adapters(self) -> None:
        """Register available storage adapters."""
        # Try to import and register AWS S3 adapter
        try:
            from .aws_s3_adapter import AWSS3Adapter
            self._adapters[StorageBackend.AWS_S3] = AWSS3Adapter
        except (ImportError, ModuleNotFoundError):
            # AWS S3 adapter dependencies not available
            pass
    
    def create_adapter(
        self,
        storage_backend: StorageBackend,
        credentials: StorageCredentials
    ) -> AbstractCloudStorageAdapter:
        """Create a storage adapter for the specified backend.
        
        Args:
            storage_backend: The storage backend type
            credentials: Storage credentials for authentication
            
        Returns:
            Configured storage adapter instance
            
        Raises:
            ValueError: If the storage backend is not supported
            ImportError: If the required dependencies are not installed
        """
        if storage_backend not in self._adapters:
            available_backends = list(self._adapters.keys())
            raise ValueError(
                f"Storage backend '{storage_backend.value}' is not supported. "
                f"Available backends: {[b.value for b in available_backends]}"
            )
        
        adapter_class = self._adapters[storage_backend]
        return adapter_class(credentials)
    
    def get_supported_backends(self) -> list[StorageBackend]:
        """Get list of supported storage backends.
        
        Returns:
            List of supported StorageBackend enum values
        """
        return list(self._adapters.keys())
    
    def is_backend_supported(self, storage_backend: StorageBackend) -> bool:
        """Check if a storage backend is supported.
        
        Args:
            storage_backend: The storage backend to check
            
        Returns:
            True if the backend is supported, False otherwise
        """
        return storage_backend in self._adapters
    
    def get_adapter_info(self) -> dict[str, str]:
        """Get information about registered adapters.
        
        Returns:
            Dictionary mapping backend names to adapter class names
        """
        return {
            backend.value: adapter_class.__name__
            for backend, adapter_class in self._adapters.items()
        }


# Global factory instance
_storage_adapter_factory: Optional[StorageAdapterFactory] = None


def get_storage_adapter_factory() -> StorageAdapterFactory:
    """Get the global storage adapter factory instance.
    
    Returns:
        Global StorageAdapterFactory instance
    """
    global _storage_adapter_factory
    if _storage_adapter_factory is None:
        _storage_adapter_factory = StorageAdapterFactory()
    return _storage_adapter_factory


def create_storage_adapter(
    storage_backend: StorageBackend,
    credentials: StorageCredentials
) -> AbstractCloudStorageAdapter:
    """Create a storage adapter for the specified backend.
    
    Convenience function that uses the global factory.
    
    Args:
        storage_backend: The storage backend type
        credentials: Storage credentials for authentication
        
    Returns:
        Configured storage adapter instance
        
    Raises:
        ValueError: If the storage backend is not supported
        ImportError: If the required dependencies are not installed
    """
    factory = get_storage_adapter_factory()
    return factory.create_adapter(storage_backend, credentials)


def get_supported_storage_backends() -> list[StorageBackend]:
    """Get list of supported storage backends.
    
    Returns:
        List of supported StorageBackend enum values
    """
    factory = get_storage_adapter_factory()
    return factory.get_supported_backends()


def is_storage_backend_supported(storage_backend: StorageBackend) -> bool:
    """Check if a storage backend is supported.
    
    Args:
        storage_backend: The storage backend to check
        
    Returns:
        True if the backend is supported, False otherwise
    """
    factory = get_storage_adapter_factory()
    return factory.is_backend_supported(storage_backend)

"""Abstract CloudStorageAdapter interface for unified cloud storage operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, IO, Optional, Union

from ..value_objects.storage_credentials import StorageCredentials


class EncryptionType(Enum):
    """Supported encryption types for cloud storage."""
    
    NONE = "none"
    SERVER_SIDE_AES256 = "server_side_aes256"
    SERVER_SIDE_KMS = "server_side_kms"
    CLIENT_SIDE_AES256 = "client_side_aes256"


class ContentType(Enum):
    """Common content types for cloud storage objects."""
    
    BINARY = "application/octet-stream"
    TEXT = "text/plain"
    JSON = "application/json"
    XML = "application/xml"
    CSV = "text/csv"
    PARQUET = "application/parquet"
    PICKLE = "application/pickle"
    JOBLIB = "application/joblib"
    ONNX = "application/onnx"
    TENSORFLOW = "application/tensorflow"
    PYTORCH = "application/pytorch"
    HUGGINGFACE = "application/huggingface"
    ZIP = "application/zip"
    GZIP = "application/gzip"


@dataclass(frozen=True)
class StorageMetadata:
    """Metadata about a storage object.
    
    Attributes:
        size_bytes: Size in bytes
        content_type: MIME content type
        last_modified: Last modification timestamp
        etag: Entity tag for versioning
        encryption_type: Encryption type used
        encryption_key_id: Encryption key identifier
        custom_metadata: Custom metadata dictionary
    """
    
    size_bytes: int
    content_type: str
    last_modified: Optional[str] = None
    etag: Optional[str] = None
    encryption_type: EncryptionType = EncryptionType.NONE
    encryption_key_id: Optional[str] = None
    custom_metadata: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class UploadOptions:
    """Options for upload operations.
    
    Attributes:
        content_type: MIME content type
        encryption_type: Encryption type to use
        encryption_key_id: Encryption key identifier
        custom_metadata: Custom metadata dictionary
        cache_control: Cache control header
        expires: Expiration date
        storage_class: Storage class/tier
        enable_multipart: Enable multipart upload for large files
        multipart_chunk_size: Chunk size for multipart uploads
    """
    
    content_type: ContentType = ContentType.BINARY
    encryption_type: EncryptionType = EncryptionType.NONE
    encryption_key_id: Optional[str] = None
    custom_metadata: Optional[Dict[str, str]] = None
    cache_control: Optional[str] = None
    expires: Optional[str] = None
    storage_class: Optional[str] = None
    enable_multipart: bool = True
    multipart_chunk_size: int = 8 * 1024 * 1024  # 8MB default


@dataclass(frozen=True)
class DownloadOptions:
    """Options for download operations.
    
    Attributes:
        byte_range: Optional byte range (start, end)
        version_id: Specific version to download
        encryption_key_id: Encryption key identifier for decryption
        enable_streaming: Enable streaming download
        stream_chunk_size: Chunk size for streaming
    """
    
    byte_range: Optional[tuple[int, int]] = None
    version_id: Optional[str] = None
    encryption_key_id: Optional[str] = None
    enable_streaming: bool = False
    stream_chunk_size: int = 64 * 1024  # 64KB default


@dataclass(frozen=True)
class ProgressInfo:
    """Information about upload/download progress.
    
    Attributes:
        bytes_transferred: Number of bytes transferred
        total_bytes: Total number of bytes
        percentage: Completion percentage
        transfer_rate: Transfer rate in bytes/second
        eta_seconds: Estimated time to completion
    """
    
    bytes_transferred: int
    total_bytes: int
    percentage: float
    transfer_rate: float
    eta_seconds: Optional[float] = None


class AbstractCloudStorageAdapter(ABC):
    """Abstract adapter for unified cloud storage operations.
    
    This abstract class defines the interface for cloud storage operations
    while remaining provider-agnostic and maintaining domain boundary integrity.
    Concrete implementations should handle provider-specific details.
    """
    
    def __init__(self, credentials: StorageCredentials):
        """Initialize the storage adapter.
        
        Args:
            credentials: Storage credentials for authentication
        """
        self._credentials = credentials
        self._is_connected = False
    
    @property
    def credentials(self) -> StorageCredentials:
        """Get the storage credentials."""
        return self._credentials
    
    @property
    def is_connected(self) -> bool:
        """Check if adapter is connected to storage."""
        return self._is_connected
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the storage provider.
        
        Raises:
            StorageConnectionError: If connection fails
            StorageAuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the storage provider."""
        pass
    
    @abstractmethod
    async def upload(
        self,
        container: str,
        key: str,
        data: Union[bytes, IO],
        options: Optional[UploadOptions] = None,
    ) -> str:
        """Upload data to storage.
        
        Args:
            container: Container/bucket name
            key: Object key/path
            data: Data to upload (bytes or file-like object)
            options: Upload options
            
        Returns:
            ETag or version identifier of uploaded object
            
        Raises:
            StorageError: If upload fails
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def upload_progressive(
        self,
        container: str,
        key: str,
        data: Union[bytes, IO],
        options: Optional[UploadOptions] = None,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Upload data with progress tracking.
        
        Args:
            container: Container/bucket name
            key: Object key/path
            data: Data to upload (bytes or file-like object)
            options: Upload options
            progress_callback: Callback function for progress updates
            
        Returns:
            ETag or version identifier of uploaded object
            
        Raises:
            StorageError: If upload fails
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def download(
        self,
        container: str,
        key: str,
        options: Optional[DownloadOptions] = None,
    ) -> bytes:
        """Download data from storage.
        
        Args:
            container: Container/bucket name
            key: Object key/path
            options: Download options
            
        Returns:
            Downloaded data as bytes
            
        Raises:
            StorageError: If download fails
            StorageNotFoundError: If object doesn't exist
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def download_stream(
        self,
        container: str,
        key: str,
        options: Optional[DownloadOptions] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Download data as a stream.
        
        Args:
            container: Container/bucket name
            key: Object key/path
            options: Download options
            
        Yields:
            Data chunks as bytes
            
        Raises:
            StorageError: If download fails
            StorageNotFoundError: If object doesn't exist
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def download_progressive(
        self,
        container: str,
        key: str,
        options: Optional[DownloadOptions] = None,
        progress_callback: Optional[callable] = None,
    ) -> bytes:
        """Download data with progress tracking.
        
        Args:
            container: Container/bucket name
            key: Object key/path
            options: Download options
            progress_callback: Callback function for progress updates
            
        Returns:
            Downloaded data as bytes
            
        Raises:
            StorageError: If download fails
            StorageNotFoundError: If object doesn't exist
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        container: str,
        key: str,
        version_id: Optional[str] = None,
    ) -> bool:
        """Delete an object from storage.
        
        Args:
            container: Container/bucket name
            key: Object key/path
            version_id: Specific version to delete
            
        Returns:
            True if object was deleted, False if it didn't exist
            
        Raises:
            StorageError: If deletion fails
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def exists(
        self,
        container: str,
        key: str,
        version_id: Optional[str] = None,
    ) -> bool:
        """Check if an object exists in storage.
        
        Args:
            container: Container/bucket name
            key: Object key/path
            version_id: Specific version to check
            
        Returns:
            True if object exists, False otherwise
            
        Raises:
            StorageError: If check fails
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def get_metadata(
        self,
        container: str,
        key: str,
        version_id: Optional[str] = None,
    ) -> StorageMetadata:
        """Get metadata for an object.
        
        Args:
            container: Container/bucket name
            key: Object key/path
            version_id: Specific version to get metadata for
            
        Returns:
            Storage metadata
            
        Raises:
            StorageError: If metadata retrieval fails
            StorageNotFoundError: If object doesn't exist
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def list_objects(
        self,
        container: str,
        prefix: Optional[str] = None,
        max_keys: Optional[int] = None,
        continuation_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List objects in a container.
        
        Args:
            container: Container/bucket name
            prefix: Object key prefix to filter by
            max_keys: Maximum number of objects to return
            continuation_token: Token for paginated results
            
        Returns:
            Dictionary containing object list and pagination info
            
        Raises:
            StorageError: If listing fails
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def copy_object(
        self,
        source_container: str,
        source_key: str,
        dest_container: str,
        dest_key: str,
        options: Optional[UploadOptions] = None,
    ) -> str:
        """Copy an object within or between containers.
        
        Args:
            source_container: Source container/bucket name
            source_key: Source object key/path
            dest_container: Destination container/bucket name
            dest_key: Destination object key/path
            options: Options for the copied object
            
        Returns:
            ETag or version identifier of copied object
            
        Raises:
            StorageError: If copy fails
            StorageNotFoundError: If source object doesn't exist
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def generate_presigned_url(
        self,
        container: str,
        key: str,
        operation: str,
        expires_in: int = 3600,
    ) -> str:
        """Generate a presigned URL for an object.
        
        Args:
            container: Container/bucket name
            key: Object key/path
            operation: Operation type ('get', 'put', 'delete')
            expires_in: URL expiration time in seconds
            
        Returns:
            Presigned URL
            
        Raises:
            StorageError: If URL generation fails
            StorageAuthenticationError: If authentication fails
            StoragePermissionError: If insufficient permissions
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """Test the connection to storage provider.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return f"{self.__class__.__name__}(connected={self.is_connected})"

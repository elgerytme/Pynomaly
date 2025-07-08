"""Base cloud storage adapter and models."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, BinaryIO, Union
from pathlib import Path
import io

from pydantic import BaseModel, Field


class StorageMetadata(BaseModel):
    """Cloud storage metadata model."""
    
    size: int = Field(description="Size of the object in bytes")
    content_type: str = Field(description="MIME type of the object")
    etag: str = Field(description="ETag of the object")
    last_modified: datetime = Field(description="Last modified timestamp")
    custom_metadata: Dict[str, str] = Field(default_factory=dict, description="Custom metadata")
    encryption_info: Optional[Dict[str, Any]] = Field(default=None, description="Encryption information")
    storage_class: Optional[str] = Field(default=None, description="Storage class")


class CloudStorageConfig(BaseModel):
    """Cloud storage configuration model."""
    
    provider: str = Field(description="Cloud provider (aws, azure, gcp)")
    bucket_name: str = Field(description="Bucket/container name")
    region: Optional[str] = Field(default=None, description="Cloud region")
    endpoint_url: Optional[str] = Field(default=None, description="Custom endpoint URL")
    access_key_id: Optional[str] = Field(default=None, description="Access key ID")
    secret_access_key: Optional[str] = Field(default=None, description="Secret access key")
    connection_string: Optional[str] = Field(default=None, description="Connection string (Azure)")
    service_account_path: Optional[str] = Field(default=None, description="Service account path (GCP)")
    enable_encryption: bool = Field(default=False, description="Enable encryption at rest")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key")
    multipart_threshold: int = Field(default=5 * 1024 * 1024 * 1024, description="Multipart upload threshold (5GB)")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay: float = Field(default=1.0, description="Retry delay in seconds")


class CloudStorageAdapter(ABC):
    """Abstract base class for cloud storage adapters."""
    
    def __init__(self, config: CloudStorageConfig):
        self.config = config
        self._client = None
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the cloud storage service."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the cloud storage service."""
        pass
    
    @abstractmethod
    async def upload_file(
        self,
        file_path: Union[str, Path],
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encrypt: bool = False,
    ) -> StorageMetadata:
        """Upload a file to cloud storage."""
        pass
    
    @abstractmethod
    async def upload_stream(
        self,
        stream: BinaryIO,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encrypt: bool = False,
    ) -> StorageMetadata:
        """Upload a stream to cloud storage."""
        pass
    
    @abstractmethod
    async def download_file(
        self,
        key: str,
        file_path: Union[str, Path],
        decrypt: bool = False,
    ) -> StorageMetadata:
        """Download a file from cloud storage."""
        pass
    
    @abstractmethod
    async def download_stream(
        self,
        key: str,
        decrypt: bool = False,
    ) -> io.BytesIO:
        """Download a stream from cloud storage."""
        pass
    
    @abstractmethod
    async def get_metadata(self, key: str) -> StorageMetadata:
        """Get metadata for an object."""
        pass
    
    @abstractmethod
    async def delete_object(self, key: str) -> bool:
        """Delete an object from cloud storage."""
        pass
    
    @abstractmethod
    async def list_objects(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List objects in cloud storage."""
        pass
    
    @abstractmethod
    async def object_exists(self, key: str) -> bool:
        """Check if an object exists in cloud storage."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

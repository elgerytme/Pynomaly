"""Azure Blob Storage adapter implementation."""

import asyncio
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, BinaryIO, Union
import aiofiles
from azure.storage.blob.aio import BlobServiceClient

from .base import CloudStorageAdapter, CloudStorageConfig, StorageMetadata
from ...shared.exceptions import CloudStorageError


class AzureAdapter(CloudStorageAdapter):
    """Azure Blob Storage adapter."""

    def __init__(self, config: CloudStorageConfig):
        super().__init__(config)
        self._blob_service_client = None
        self._container_client = None

    async def connect(self) -> None:
        """Connect to Azure Blob Storage."""
        try:
            self._blob_service_client = BlobServiceClient.from_connection_string(self.config.connection_string)
            self._container_client = self._blob_service_client.get_container_client(self.config.bucket_name)
        except Exception as e:
            raise CloudStorageError(f"Failed to connect to Azure: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Azure Blob Storage."""
        self._blob_service_client = None
        self._container_client = None

    async def upload_file(
        self,
        file_path: Union[str, Path],
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encrypt: bool = False,
    ) -> StorageMetadata:
        """Upload a file to Azure Blob Storage."""
        if not self._container_client:
            raise CloudStorageError("Azure client not connected")

        file_path = Path(file_path)
        if not file_path.exists():
            raise CloudStorageError(f"File not found: {file_path}")

        try:
            blob_client = self._container_client.get_blob_client(key)
            async with aiofiles.open(file_path, 'rb') as data:
                await blob_client.upload_blob(data, metadata=metadata, content_settings=content_type and {'content_type': content_type})

            return await self.get_metadata(key)

        except Exception as e:
            raise CloudStorageError(f"Failed to upload file to Azure: {e}")

    async def upload_stream(
        self,
        stream: BinaryIO,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encrypt: bool = False,
    ) -> StorageMetadata:
        """Upload a stream to Azure Blob Storage."""
        if not self._container_client:
            raise CloudStorageError("Azure client not connected")

        try:
            blob_client = self._container_client.get_blob_client(key)
            await blob_client.upload_blob(stream, metadata=metadata, content_settings=content_type and {'content_type': content_type})
            return await self.get_metadata(key)

        except Exception as e:
            raise CloudStorageError(f"Failed to upload stream to Azure: {e}")

    async def download_file(
        self,
        key: str,
        file_path: Union[str, Path],
        decrypt: bool = False,
    ) -> StorageMetadata:
        """Download a file from Azure Blob Storage."""
        if not self._container_client:
            raise CloudStorageError("Azure client not connected")

        file_path = Path(file_path)

        try:
            blob_client = self._container_client.get_blob_client(key)
            stream = await blob_client.download_blob()
            async with aiofiles.open(file_path, 'wb') as f:
                async for chunk in stream.chunks():
                    await f.write(chunk)

            return await self.get_metadata(key)

        except Exception as e:
            raise CloudStorageError(f"Failed to download file from Azure: {e}")

    async def download_stream(
        self,
        key: str,
        decrypt: bool = False,
    ) -> io.BytesIO:
        """Download a stream from Azure Blob Storage."""
        if not self._container_client:
            raise CloudStorageError("Azure client not connected")

        try:
            blob_client = self._container_client.get_blob_client(key)
            stream = await blob_client.download_blob()
            data = io.BytesIO()
            async for chunk in stream.chunks():
                data.write(chunk)

            data.seek(0)
            return data

        except Exception as e:
            raise CloudStorageError(f"Failed to download stream from Azure: {e}")

    async def get_metadata(self, key: str) -> StorageMetadata:
        """Get metadata for an Azure Blob Storage object."""
        if not self._container_client:
            raise CloudStorageError("Azure client not connected")

        try:
            blob_client = self._container_client.get_blob_client(key)
            properties = await blob_client.get_blob_properties()

            return StorageMetadata(
                size=properties.size,
                content_type=properties.content_settings.content_type,
                etag=properties.etag.strip('"'),
                last_modified=properties.last_modified,
                custom_metadata=properties.metadata,
                storage_class=properties.blob_tier
            )
        except Exception as e:
            raise CloudStorageError(f"Failed to get metadata from Azure: {e}")

    async def delete_object(self, key: str) -> bool:
        """Delete an object from Azure Blob Storage."""
        if not self._container_client:
            raise CloudStorageError("Azure client not connected")

        try:
            blob_client = self._container_client.get_blob_client(key)
            await blob_client.delete_blob()
            return True

        except Exception as e:
            raise CloudStorageError(f"Failed to delete object from Azure: {e}")

    async def list_objects(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List objects in Azure Blob Storage."""
        if not self._container_client:
            raise CloudStorageError("Azure client not connected")

        try:
            blob_list = self._container_client.list_blobs(name_starts_with=prefix)
            objects = []

            async for blob in blob_list:
                objects.append(blob.name)
                if limit and len(objects) >= limit:
                    break

            return objects

        except Exception as e:
            raise CloudStorageError(f"Failed to list objects in Azure: {e}")

    async def object_exists(self, key: str) -> bool:
        """Check if an object exists in Azure Blob Storage."""
        if not self._container_client:
            raise CloudStorageError("Azure client not connected")

        try:
            blob_client = self._container_client.get_blob_client(key)
            await blob_client.get_blob_properties()
            return True

        except Exception:
            return False


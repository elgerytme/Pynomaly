"""Google Cloud Storage adapter implementation."""

import asyncio
import io
from pathlib import Path
from typing import BinaryIO

import aiofiles
from google.cloud.storage import Client
from google.oauth2 import service_account

from ...shared.exceptions import CloudStorageError
from .base import CloudStorageAdapter, CloudStorageConfig, StorageMetadata


class GCPAdapter(CloudStorageAdapter):
    """Google Cloud Storage adapter."""

    def __init__(self, config: CloudStorageConfig):
        super().__init__(config)
        self._client = None
        self._bucket = None

    async def connect(self) -> None:
        """Connect to Google Cloud Storage."""
        try:
            if self.config.service_account_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.service_account_path
                )
                self._client = Client(credentials=credentials)
            else:
                self._client = Client()

            self._bucket = self._client.bucket(self.config.bucket_name)
        except Exception as e:
            raise CloudStorageError(f"Failed to connect to GCP: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Google Cloud Storage."""
        self._client = None
        self._bucket = None

    async def upload_file(
        self,
        file_path: str | Path,
        key: str,
        metadata: dict[str, str] | None = None,
        content_type: str | None = None,
        encrypt: bool = False,
    ) -> StorageMetadata:
        """Upload a file to Google Cloud Storage."""
        if not self._bucket:
            raise CloudStorageError("GCP client not connected")

        file_path = Path(file_path)
        if not file_path.exists():
            raise CloudStorageError(f"File not found: {file_path}")

        try:
            blob = self._bucket.blob(key)
            if metadata:
                blob.metadata = metadata
            if content_type:
                blob.content_type = content_type

            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()
                await asyncio.get_event_loop().run_in_executor(
                    None, blob.upload_from_string, data
                )

            return await self.get_metadata(key)

        except Exception as e:
            raise CloudStorageError(f"Failed to upload file to GCP: {e}")

    async def upload_stream(
        self,
        stream: BinaryIO,
        key: str,
        metadata: dict[str, str] | None = None,
        content_type: str | None = None,
        encrypt: bool = False,
    ) -> StorageMetadata:
        """Upload a stream to Google Cloud Storage."""
        if not self._bucket:
            raise CloudStorageError("GCP client not connected")

        try:
            blob = self._bucket.blob(key)
            if metadata:
                blob.metadata = metadata
            if content_type:
                blob.content_type = content_type

            data = stream.read()
            await asyncio.get_event_loop().run_in_executor(
                None, blob.upload_from_string, data
            )

            return await self.get_metadata(key)

        except Exception as e:
            raise CloudStorageError(f"Failed to upload stream to GCP: {e}")

    async def download_file(
        self,
        key: str,
        file_path: str | Path,
        decrypt: bool = False,
    ) -> StorageMetadata:
        """Download a file from Google Cloud Storage."""
        if not self._bucket:
            raise CloudStorageError("GCP client not connected")

        file_path = Path(file_path)

        try:
            blob = self._bucket.blob(key)
            await asyncio.get_event_loop().run_in_executor(
                None, blob.download_to_filename, str(file_path)
            )

            return await self.get_metadata(key)

        except Exception as e:
            raise CloudStorageError(f"Failed to download file from GCP: {e}")

    async def download_stream(
        self,
        key: str,
        decrypt: bool = False,
    ) -> io.BytesIO:
        """Download a stream from Google Cloud Storage."""
        if not self._bucket:
            raise CloudStorageError("GCP client not connected")

        try:
            blob = self._bucket.blob(key)
            data = await asyncio.get_event_loop().run_in_executor(
                None, blob.download_as_bytes
            )
            return io.BytesIO(data)

        except Exception as e:
            raise CloudStorageError(f"Failed to download stream from GCP: {e}")

    async def get_metadata(self, key: str) -> StorageMetadata:
        """Get metadata for a Google Cloud Storage object."""
        if not self._bucket:
            raise CloudStorageError("GCP client not connected")

        try:
            blob = self._bucket.blob(key)
            await asyncio.get_event_loop().run_in_executor(None, blob.reload)

            return StorageMetadata(
                size=blob.size,
                content_type=blob.content_type or "application/octet-stream",
                etag=blob.etag,
                last_modified=blob.updated,
                custom_metadata=blob.metadata or {},
                storage_class=blob.storage_class,
            )
        except Exception as e:
            raise CloudStorageError(f"Failed to get metadata from GCP: {e}")

    async def delete_object(self, key: str) -> bool:
        """Delete an object from Google Cloud Storage."""
        if not self._bucket:
            raise CloudStorageError("GCP client not connected")

        try:
            blob = self._bucket.blob(key)
            await asyncio.get_event_loop().run_in_executor(None, blob.delete)
            return True

        except Exception as e:
            raise CloudStorageError(f"Failed to delete object from GCP: {e}")

    async def list_objects(
        self,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """List objects in Google Cloud Storage."""
        if not self._bucket:
            raise CloudStorageError("GCP client not connected")

        try:
            blobs = await asyncio.get_event_loop().run_in_executor(
                None, self._bucket.list_blobs, prefix, None, limit
            )
            return [blob.name for blob in blobs]

        except Exception as e:
            raise CloudStorageError(f"Failed to list objects in GCP: {e}")

    async def object_exists(self, key: str) -> bool:
        """Check if an object exists in Google Cloud Storage."""
        if not self._bucket:
            raise CloudStorageError("GCP client not connected")

        try:
            blob = self._bucket.blob(key)
            return await asyncio.get_event_loop().run_in_executor(None, blob.exists)

        except Exception:
            return False

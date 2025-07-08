from azure.storage.blob.aio import BlobServiceClient
from azure.storage.blob import ContentSettings, BlobSasPermissions
from azure.identity.aio import DefaultAzureCredential
from .cloud_storage_adapter import AbstractCloudStorageAdapter, UploadOptions, EncryptionType
from ..domain.value_objects.storage_credentials import StorageCredentials
from ..domain.exceptions.storage_exceptions import StorageError, StorageAuthenticationError, StoragePermissionError, StorageConnectionError, StorageNotFoundError
from azure.storage.blob import generate_blob_sas
import asyncio

class AzureBlobAdapter(AbstractCloudStorageAdapter):
    """Azure Blob Storage implementation of CloudStorageAdapter."""

    def __init__(self, credentials: StorageCredentials):
        super().__init__(credentials)
        self._blob_service_client = None
        self._credential = None

    async def connect(self) -> None:
        """Connect to Azure Blob Storage."""
        try:
            if self.credentials.auth_type == 'sas_token':
                conn_str = self.credentials.credentials.get('connection_string')
                self._blob_service_client = BlobServiceClient.from_connection_string(conn_str)
            else:
                self._credential = DefaultAzureCredential()
                self._blob_service_client = BlobServiceClient(
                    account_url=self.credentials.credentials.get('account_url'),
                    credential=self._credential
                )
            self._is_connected = True
        except Exception as e:
            raise StorageConnectionError(f"Failed to connect to Azure Blob Storage: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Azure Blob Storage."""
        try:
            if self._credential:
                await self._credential.close()
            self._is_connected = False
        except Exception as e:
            raise StorageError(f"Failed to disconnect: {e}")

    async def upload(
        self,
        container: str,
        key: str,
        data: bytes,
        options: UploadOptions = None
    ) -> str:
        """Upload data to Azure Blob Storage."""
        try:
            blob_client = self._blob_service_client.get_blob_client(container=container, blob=key)

            content_settings = ContentSettings(content_type=options.content_type.value if options else 'application/octet-stream')

            if options and options.encryption_type != EncryptionType.NONE:
                raise NotImplementedError("Server-side encryption not yet implemented for Azure Blob Storage.")

            await blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)
            return blob_client.get_blob_properties().etag

        except Exception as e:
            raise StorageError(f"Failed to upload data: {e}")

    async def upload_progressive(
        self,
        container: str,
        key: str,
        data: bytes,
        options: UploadOptions = None,
        progress_callback: callable = None
    ) -> str:
        """Upload data to Azure Blob Storage with progress tracking."""
        chunk_size = options.multipart_chunk_size if options else 8 * 1024 * 1024
        total_size = len(data)
        num_chunks = (total_size + chunk_size - 1) // chunk_size

        try:
            blob_client = self._blob_service_client.get_blob_client(container=container, blob=key)
            uploaded_size = 0

            for i in range(num_chunks):
                chunk_start = i * chunk_size
                chunk_end = min(chunk_start + chunk_size, total_size)
                chunk_data = data[chunk_start:chunk_end]

                await blob_client.upload_blob(chunk_data, blob_type="BlockBlob", overwrite=(i == 0))
                uploaded_size += len(chunk_data)

                if progress_callback:
                    progress_callback(uploaded_size, total_size)

            return blob_client.get_blob_properties().etag

        except Exception as e:
            raise StorageError(f"Failed to upload data: {e}")

    # Implement other methods based on requirements...
    # async def download(...): pass
    # async def delete(...): pass
    # async def exists(...): pass
    # async def generate_presigned_url(...): pass

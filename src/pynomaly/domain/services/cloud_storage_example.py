"""Example usage of the CloudStorageAdapter interface.

This file demonstrates how to use the unified CloudStorageAdapter interface
for cloud storage operations while remaining provider-agnostic.
"""

from typing import Optional

from ..exceptions.storage_exceptions import StorageError
from ..value_objects.storage_credentials import (
    AuthenticationType,
    StorageCredentials,
)
from .cloud_storage_adapter import (
    AbstractCloudStorageAdapter,
    ContentType,
    DownloadOptions,
    EncryptionType,
    UploadOptions,
)


async def example_cloud_storage_usage(
    adapter: AbstractCloudStorageAdapter,
    container: str = "my-bucket",
) -> None:
    """Example demonstrating cloud storage operations.
    
    Args:
        adapter: Cloud storage adapter implementation
        container: Container/bucket name
    """
    
    # Example 1: Simple upload and download
    try:
        async with adapter:  # Uses context manager for connection management
            # Upload a simple text file
            upload_options = UploadOptions(
                content_type=ContentType.TEXT,
                encryption_type=EncryptionType.SERVER_SIDE_AES256,
                custom_metadata={"purpose": "example", "version": "1.0"},
            )
            
            data = b"Hello, cloud storage!"
            etag = await adapter.upload(
                container=container,
                key="examples/hello.txt",
                data=data,
                options=upload_options,
            )
            print(f"Uploaded file with ETag: {etag}")
            
            # Check if object exists
            exists = await adapter.exists(container, "examples/hello.txt")
            print(f"File exists: {exists}")
            
            # Get metadata
            metadata = await adapter.get_metadata(container, "examples/hello.txt")
            print(f"File size: {metadata.size_bytes} bytes")
            print(f"Content type: {metadata.content_type}")
            print(f"Encrypted: {metadata.encryption_type != EncryptionType.NONE}")
            
            # Download the file
            downloaded_data = await adapter.download(container, "examples/hello.txt")
            print(f"Downloaded: {downloaded_data.decode()}")
            
    except StorageError as e:
        print(f"Storage operation failed: {e}")


async def example_progressive_upload(
    adapter: AbstractCloudStorageAdapter,
    container: str,
    large_data: bytes,
) -> None:
    """Example demonstrating progressive upload with progress tracking.
    
    Args:
        adapter: Cloud storage adapter implementation
        container: Container/bucket name
        large_data: Large data to upload
    """
    
    def progress_callback(bytes_transferred: int, total_bytes: int) -> None:
        """Progress callback for tracking upload progress."""
        percentage = (bytes_transferred / total_bytes) * 100
        print(f"Upload progress: {percentage:.1f}% ({bytes_transferred}/{total_bytes} bytes)")
    
    upload_options = UploadOptions(
        content_type=ContentType.BINARY,
        enable_multipart=True,
        multipart_chunk_size=8 * 1024 * 1024,  # 8MB chunks
        custom_metadata={"source": "progressive_upload_example"},
    )
    
    try:
        async with adapter:
            etag = await adapter.upload_progressive(
                container=container,
                key="examples/large_file.bin",
                data=large_data,
                options=upload_options,
                progress_callback=progress_callback,
            )
            print(f"Large file uploaded with ETag: {etag}")
            
    except StorageError as e:
        print(f"Progressive upload failed: {e}")


async def example_streaming_download(
    adapter: AbstractCloudStorageAdapter,
    container: str,
    key: str,
) -> None:
    """Example demonstrating streaming download.
    
    Args:
        adapter: Cloud storage adapter implementation
        container: Container/bucket name
        key: Object key to download
    """
    
    download_options = DownloadOptions(
        enable_streaming=True,
        stream_chunk_size=64 * 1024,  # 64KB chunks
    )
    
    try:
        async with adapter:
            total_bytes = 0
            async for chunk in adapter.download_stream(container, key, download_options):
                total_bytes += len(chunk)
                # Process chunk here (e.g., write to file, process data)
                print(f"Received chunk of {len(chunk)} bytes")
            
            print(f"Total downloaded: {total_bytes} bytes")
            
    except StorageError as e:
        print(f"Streaming download failed: {e}")


def create_storage_credentials_examples() -> list[StorageCredentials]:
    """Create examples of different credential types.
    
    Returns:
        List of StorageCredentials examples
    """
    credentials = []
    
    # AWS-style access key credentials
    aws_creds = StorageCredentials.create_access_key_credentials(
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
    )
    credentials.append(aws_creds)
    
    # Token-based credentials
    token_creds = StorageCredentials.create_token_credentials(
        token="example-bearer-token",
        region="us-west-2",
    )
    credentials.append(token_creds)
    
    # Service account credentials (GCP-style)
    service_account_creds = StorageCredentials.create_service_account_credentials(
        service_account_key="/path/to/service-account.json",
        region="us-central1",
    )
    credentials.append(service_account_creds)
    
    # Role-based credentials
    role_creds = StorageCredentials.create_role_based_credentials(
        role_arn="arn:aws:iam::123456789012:role/MyRole",
        region="eu-west-1",
        session_name="example-session",
    )
    credentials.append(role_creds)
    
    # Anonymous credentials (for public buckets)
    anon_creds = StorageCredentials.create_anonymous_credentials()
    credentials.append(anon_creds)
    
    return credentials


async def example_object_management(
    adapter: AbstractCloudStorageAdapter,
    container: str,
) -> None:
    """Example demonstrating object management operations.
    
    Args:
        adapter: Cloud storage adapter implementation
        container: Container/bucket name
    """
    
    try:
        async with adapter:
            # List objects with prefix
            result = await adapter.list_objects(
                container=container,
                prefix="examples/",
                max_keys=10,
            )
            
            print(f"Found {len(result.get('objects', []))} objects:")
            for obj in result.get("objects", []):
                print(f"  - {obj['key']} ({obj['size']} bytes)")
            
            # Copy an object
            if result.get("objects"):
                source_key = result["objects"][0]["key"]
                dest_key = f"backup/{source_key}"
                
                copy_etag = await adapter.copy_object(
                    source_container=container,
                    source_key=source_key,
                    dest_container=container,
                    dest_key=dest_key,
                )
                print(f"Copied {source_key} to {dest_key} with ETag: {copy_etag}")
            
            # Generate presigned URL
            presigned_url = await adapter.generate_presigned_url(
                container=container,
                key="examples/hello.txt",
                operation="get",
                expires_in=3600,  # 1 hour
            )
            print(f"Presigned URL: {presigned_url}")
            
            # Clean up - delete objects
            for obj in result.get("objects", []):
                deleted = await adapter.delete(container, obj["key"])
                print(f"Deleted {obj['key']}: {deleted}")
                
    except StorageError as e:
        print(f"Object management failed: {e}")


# Example of how a concrete implementation might start
# (This would be implemented in the infrastructure layer)
"""
class S3CloudStorageAdapter(AbstractCloudStorageAdapter):
    '''AWS S3 implementation of CloudStorageAdapter.'''
    
    def __init__(self, credentials: StorageCredentials):
        super().__init__(credentials)
        self._s3_client = None
    
    async def connect(self) -> None:
        '''Connect to AWS S3.'''
        # Implementation would create S3 client using credentials
        # Handle different credential types appropriately
        self._is_connected = True
    
    async def upload(
        self,
        container: str,
        key: str,
        data: Union[bytes, IO],
        options: Optional[UploadOptions] = None,
    ) -> str:
        '''Upload to S3.'''
        # Implementation would use boto3 or similar
        # Convert domain objects to S3-specific parameters
        pass
    
    # ... other method implementations
"""

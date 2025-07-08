"""Integration tests for cloud storage adapters."""

import aiofiles
import pytest
from moto import mock_s3

from pynomaly.infrastructure.cloud_storage import (
    AzureAdapter,
    CloudStorageConfig,
    GCPAdapter,
    S3Adapter,
)

# Use appropriate Azure mock or emulator when needed


@pytest.fixture
async def aws_mock_s3_client():
    """Mock S3 client setup."""
    with mock_s3():
        import boto3

        conn = boto3.client(
            "s3",
            aws_access_key_id="AKIAEXAMPLE",
            aws_secret_access_key="EXAMPLEKEY",
            region_name="us-east-1",
        )
        conn.create_bucket(Bucket="mock-bucket")

        yield conn


@pytest.mark.asyncio
async def test_s3_adapter_upload_download(aws_mock_s3_client):
    """Test S3Adapter upload and download."""
    config = CloudStorageConfig(
        provider="aws",
        bucket_name="mock-bucket",
        region="us-east-1",
        access_key_id="AKIAEXAMPLE",
        secret_access_key="EXAMPLEKEY",
    )

    adapter = S3Adapter(config)

    await adapter.connect()

    # Upload Test
    file_path = "test_upload.txt"
    async with aiofiles.open(file_path, "w") as f:
        await f.write("Hello World")

    await adapter.upload_file(file_path, "test_key.txt")

    # Download Test
    download_path = "test_download.txt"
    await adapter.download_file("test_key.txt", download_path)

    async with aiofiles.open(download_path) as f:
        content = await f.read()

    assert content == "Hello World"

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_azure_adapter_upload_download(azurite_container_client):
    """Test AzureAdapter upload and download."""
    config = CloudStorageConfig(
        provider="azure",
        bucket_name="mock-container",
        connection_string="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02x=W3Pv1+\nRT1Y6AO6qC3ENzzrMP/-kvody9OuJ9i/mJwzfjYXjf5Z4Zso6fys9EI==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;",
    )

    adapter = AzureAdapter(config)

    await adapter.connect()

    # Upload Test
    file_path = "test_upload_azure.txt"
    async with aiofiles.open(file_path, "w") as f:
        await f.write("Hello Azure")

    await adapter.upload_file(file_path, "test_key.txt")

    # Download Test
    download_path = "test_download_azure.txt"
    await adapter.download_file("test_key.txt", download_path)

    async with aiofiles.open(download_path) as f:
        content = await f.read()

    assert content == "Hello Azure"

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_gcp_adapter_upload_download(gcp_storage_client):
    """Test GCPAdapter upload and download."""
    config = CloudStorageConfig(provider="gcp", bucket_name="mock-bucket")

    adapter = GCPAdapter(config)

    await adapter.connect()

    # Upload Test
    file_path = "test_upload_gcp.txt"
    async with aiofiles.open(file_path, "w") as f:
        await f.write("Hello GCP")

    await adapter.upload_file(file_path, "test_key.txt")

    # Download Test
    download_path = "test_download_gcp.txt"
    await adapter.download_file("test_key.txt", download_path)

    async with aiofiles.open(download_path) as f:
        content = await f.read()

    assert content == "Hello GCP"

    await adapter.disconnect()

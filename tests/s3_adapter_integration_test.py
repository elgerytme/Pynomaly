"""Integration Test for S3 Adapter using Moto mock."""

import io

import boto3
import pytest
from moto import mock_s3

from pynomaly.infrastructure.cloud_storage import CloudStorageConfig, S3Adapter


@pytest.fixture
async def s3_config():
    return CloudStorageConfig(
        provider="aws",
        bucket_name="test-bucket",
        access_key_id="testid",
        secret_access_key="testkey",
    )


@pytest.mark.asyncio
async def test_s3_adapter(s3_config):
    with mock_s3():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="test-bucket")

        adapter = S3Adapter(s3_config)
        await adapter.connect()

        # Test uploading a stream
        data = io.BytesIO(b"Hello World")
        metadata = await adapter.upload_stream(data, "test.txt")
        assert metadata.size == 11

        # Test downloading as stream
        stream = await adapter.download_stream("test.txt")
        assert stream.read() == b"Hello World"

        # Test getting metadata
        metadata = await adapter.get_metadata("test.txt")
        assert metadata.size == 11

        await adapter.disconnect()

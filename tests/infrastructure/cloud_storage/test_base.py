"""Unit tests for cloud storage base components."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from pynomaly.infrastructure.cloud_storage.base import (
    CloudStorageAdapter,
    CloudStorageConfig,
    StorageMetadata,
)
from pynomaly.shared.exceptions import CloudStorageError


class TestStorageMetadata:
    """Test StorageMetadata model."""
    
    def test_storage_metadata_creation(self):
        """Test StorageMetadata model creation."""
        metadata = StorageMetadata(
            size=1024,
            content_type="application/json",
            etag="abc123",
            last_modified=datetime.now(),
            custom_metadata={"key": "value"},
            encryption_info={"algorithm": "AES256"},
            storage_class="STANDARD"
        )
        
        assert metadata.size == 1024
        assert metadata.content_type == "application/json"
        assert metadata.etag == "abc123"
        assert metadata.custom_metadata == {"key": "value"}
        assert metadata.encryption_info == {"algorithm": "AES256"}
        assert metadata.storage_class == "STANDARD"
    
    def test_storage_metadata_defaults(self):
        """Test StorageMetadata with default values."""
        metadata = StorageMetadata(
            size=1024,
            content_type="application/json",
            etag="abc123",
            last_modified=datetime.now(),
        )
        
        assert metadata.custom_metadata == {}
        assert metadata.encryption_info is None
        assert metadata.storage_class is None


class TestCloudStorageConfig:
    """Test CloudStorageConfig model."""
    
    def test_cloud_storage_config_creation(self):
        """Test CloudStorageConfig model creation."""
        config = CloudStorageConfig(
            provider="aws",
            bucket_name="test-bucket",
            region="us-east-1",
            access_key_id="AKIAIOSFODNN7EXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            enable_encryption=True,
            encryption_key="kms-key-id",
            max_retries=5,
            retry_delay=2.0,
        )
        
        assert config.provider == "aws"
        assert config.bucket_name == "test-bucket"
        assert config.region == "us-east-1"
        assert config.access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert config.secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert config.enable_encryption is True
        assert config.encryption_key == "kms-key-id"
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
    
    def test_cloud_storage_config_defaults(self):
        """Test CloudStorageConfig with default values."""
        config = CloudStorageConfig(
            provider="aws",
            bucket_name="test-bucket",
        )
        
        assert config.region is None
        assert config.endpoint_url is None
        assert config.enable_encryption is False
        assert config.encryption_key is None
        assert config.multipart_threshold == 5 * 1024 * 1024 * 1024  # 5GB
        assert config.max_retries == 3
        assert config.retry_delay == 1.0


class TestCloudStorageAdapter:
    """Test CloudStorageAdapter abstract base class."""
    
    def test_cloud_storage_adapter_instantiation(self):
        """Test CloudStorageAdapter cannot be instantiated directly."""
        config = CloudStorageConfig(provider="aws", bucket_name="test-bucket")
        
        with pytest.raises(TypeError):
            CloudStorageAdapter(config)
    
    def test_cloud_storage_adapter_subclass_methods(self):
        """Test CloudStorageAdapter subclass must implement abstract methods."""
        config = CloudStorageConfig(provider="aws", bucket_name="test-bucket")
        
        class IncompleteAdapter(CloudStorageAdapter):
            pass
        
        with pytest.raises(TypeError):
            IncompleteAdapter(config)
    
    def test_cloud_storage_adapter_context_manager(self):
        """Test CloudStorageAdapter context manager protocol."""
        config = CloudStorageConfig(provider="aws", bucket_name="test-bucket")
        
        class MockAdapter(CloudStorageAdapter):
            async def connect(self):
                self.connected = True
            
            async def disconnect(self):
                self.connected = False
            
            async def upload_file(self, *args, **kwargs):
                pass
            
            async def upload_stream(self, *args, **kwargs):
                pass
            
            async def download_file(self, *args, **kwargs):
                pass
            
            async def download_stream(self, *args, **kwargs):
                pass
            
            async def get_metadata(self, *args, **kwargs):
                pass
            
            async def delete_object(self, *args, **kwargs):
                pass
            
            async def list_objects(self, *args, **kwargs):
                pass
            
            async def object_exists(self, *args, **kwargs):
                pass
        
        adapter = MockAdapter(config)
        
        # Test context manager protocol exists
        assert hasattr(adapter, '__aenter__')
        assert hasattr(adapter, '__aexit__')


class TestCloudStorageAdapterImplementation:
    """Test CloudStorageAdapter implementation with mock."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock CloudStorageAdapter implementation."""
        config = CloudStorageConfig(provider="aws", bucket_name="test-bucket")
        
        class MockAdapter(CloudStorageAdapter):
            def __init__(self, config):
                super().__init__(config)
                self.connected = False
            
            async def connect(self):
                self.connected = True
            
            async def disconnect(self):
                self.connected = False
            
            async def upload_file(self, file_path, key, metadata=None, content_type=None, encrypt=False):
                if not self.connected:
                    raise CloudStorageError("Not connected")
                return StorageMetadata(
                    size=1024,
                    content_type=content_type or "application/octet-stream",
                    etag="abc123",
                    last_modified=datetime.now(),
                    custom_metadata=metadata or {},
                )
            
            async def upload_stream(self, stream, key, metadata=None, content_type=None, encrypt=False):
                if not self.connected:
                    raise CloudStorageError("Not connected")
                return StorageMetadata(
                    size=1024,
                    content_type=content_type or "application/octet-stream",
                    etag="abc123",
                    last_modified=datetime.now(),
                    custom_metadata=metadata or {},
                )
            
            async def download_file(self, key, file_path, decrypt=False):
                if not self.connected:
                    raise CloudStorageError("Not connected")
                return StorageMetadata(
                    size=1024,
                    content_type="application/octet-stream",
                    etag="abc123",
                    last_modified=datetime.now(),
                )
            
            async def download_stream(self, key, decrypt=False):
                if not self.connected:
                    raise CloudStorageError("Not connected")
                import io
                return io.BytesIO(b"test data")
            
            async def get_metadata(self, key):
                if not self.connected:
                    raise CloudStorageError("Not connected")
                return StorageMetadata(
                    size=1024,
                    content_type="application/octet-stream",
                    etag="abc123",
                    last_modified=datetime.now(),
                )
            
            async def delete_object(self, key):
                if not self.connected:
                    raise CloudStorageError("Not connected")
                return True
            
            async def list_objects(self, prefix=None, limit=None):
                if not self.connected:
                    raise CloudStorageError("Not connected")
                return ["file1.txt", "file2.txt"]
            
            async def object_exists(self, key):
                if not self.connected:
                    raise CloudStorageError("Not connected")
                return True
        
        return MockAdapter(config)
    
    @pytest.mark.asyncio
    async def test_context_manager_connect_disconnect(self, mock_adapter):
        """Test context manager properly connects and disconnects."""
        assert not mock_adapter.connected
        
        async with mock_adapter as adapter:
            assert adapter.connected
            assert adapter is mock_adapter
        
        assert not mock_adapter.connected
    
    @pytest.mark.asyncio
    async def test_upload_file_when_connected(self, mock_adapter):
        """Test upload_file when connected."""
        await mock_adapter.connect()
        
        metadata = await mock_adapter.upload_file(
            "test.txt",
            "test-key",
            metadata={"project": "test"},
            content_type="text/plain",
            encrypt=True
        )
        
        assert metadata.size == 1024
        assert metadata.content_type == "text/plain"
        assert metadata.custom_metadata == {"project": "test"}
    
    @pytest.mark.asyncio
    async def test_upload_file_when_not_connected(self, mock_adapter):
        """Test upload_file when not connected raises error."""
        with pytest.raises(CloudStorageError, match="Not connected"):
            await mock_adapter.upload_file("test.txt", "test-key")
    
    @pytest.mark.asyncio
    async def test_download_file_when_connected(self, mock_adapter):
        """Test download_file when connected."""
        await mock_adapter.connect()
        
        metadata = await mock_adapter.download_file("test-key", "local-file.txt")
        
        assert metadata.size == 1024
        assert metadata.content_type == "application/octet-stream"
    
    @pytest.mark.asyncio
    async def test_list_objects_when_connected(self, mock_adapter):
        """Test list_objects when connected."""
        await mock_adapter.connect()
        
        objects = await mock_adapter.list_objects(prefix="test/", limit=10)
        
        assert objects == ["file1.txt", "file2.txt"]
    
    @pytest.mark.asyncio
    async def test_object_exists_when_connected(self, mock_adapter):
        """Test object_exists when connected."""
        await mock_adapter.connect()
        
        exists = await mock_adapter.object_exists("test-key")
        
        assert exists is True
    
    @pytest.mark.asyncio
    async def test_delete_object_when_connected(self, mock_adapter):
        """Test delete_object when connected."""
        await mock_adapter.connect()
        
        result = await mock_adapter.delete_object("test-key")
        
        assert result is True

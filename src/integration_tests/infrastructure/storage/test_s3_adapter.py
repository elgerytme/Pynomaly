"""
Tests for S3Adapter implementation.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from botocore.exceptions import ClientError

from pynomaly.domain.exceptions.storage_exceptions import (
    StorageConnectionError,
    StorageError,
    StorageNotFoundError,
)
from pynomaly.domain.value_objects.storage_credentials import StorageCredentials
from pynomaly.infrastructure.storage.s3_adapter import S3Adapter


class TestS3Adapter:
    @pytest.fixture
    def mock_credentials(self):
        """Create mock credentials for testing."""
        return StorageCredentials.create_access_key_credentials(
            access_key_id="test_key",
            secret_access_key="test_secret",
            region="us-east-1",
        )

    def test_init(self, mock_credentials):
        """Test S3Adapter initialization."""
        adapter = S3Adapter(mock_credentials)
        
        assert adapter.credentials == mock_credentials
        assert not adapter.is_connected
        assert adapter._s3_client is None
        assert adapter._s3_resource is None

    @patch('boto3.Session')
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_session, mock_credentials):
        """Test successful connection to S3."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        mock_client = Mock()
        mock_resource = Mock()
        mock_session_instance.client.return_value = mock_client
        mock_session_instance.resource.return_value = mock_resource
        
        adapter = S3Adapter(mock_credentials)
        await adapter.connect()
        
        assert adapter.is_connected
        assert adapter._s3_client == mock_client
        assert adapter._s3_resource == mock_resource

    @patch('boto3.Session')
    @pytest.mark.asyncio
    async def test_connect_failure(self, mock_session, mock_credentials):
        """Test connection failure."""
        mock_session.side_effect = ClientError(
            error_response={'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}},
            operation_name='CreateSession'
        )
        
        adapter = S3Adapter(mock_credentials)
        
        with pytest.raises(StorageConnectionError):
            await adapter.connect()
        
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_credentials):
        """Test disconnection from S3."""
        adapter = S3Adapter(mock_credentials)
        adapter._s3_client = Mock()
        adapter._s3_resource = Mock()
        adapter._is_connected = True
        
        await adapter.disconnect()
        
        assert not adapter.is_connected
        assert adapter._s3_client is None
        assert adapter._s3_resource is None

    @patch('boto3.Session')
    @pytest.mark.asyncio
    async def test_test_connection_success(self, mock_session, mock_credentials):
        """Test successful connection test."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        mock_client = Mock()
        mock_client.list_buckets.return_value = {'Buckets': []}
        mock_session_instance.client.return_value = mock_client
        
        adapter = S3Adapter(mock_credentials)
        await adapter.connect()
        
        result = await adapter.test_connection()
        
        assert result is True
        mock_client.list_buckets.assert_called_once()

    @patch('boto3.Session')
    @pytest.mark.asyncio
    async def test_test_connection_failure(self, mock_session, mock_credentials):
        """Test connection test failure."""
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        mock_client = Mock()
        mock_client.list_buckets.side_effect = ClientError(
            error_response={'Error': {'Code': 'AccessDenied', 'Message': 'Access denied'}},
            operation_name='ListBuckets'
        )
        mock_session_instance.client.return_value = mock_client
        
        adapter = S3Adapter(mock_credentials)
        await adapter.connect()
        
        result = await adapter.test_connection()
        
        assert result is False

    def test_from_environment_with_env_vars(self):
        """Test creating adapter from environment variables."""
        with patch.dict('os.environ', {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret',
            'AWS_DEFAULT_REGION': 'us-west-2',
        }):
            adapter = S3Adapter.from_environment()
            
            assert adapter.credentials.auth_type.value == 'access_key'
            assert adapter.credentials.region == 'us-west-2'
            assert adapter.credentials.get_credential_value('access_key_id') == 'test_key'
            assert adapter.credentials.get_credential_value('secret_access_key') == 'test_secret'

    def test_from_environment_without_env_vars(self):
        """Test creating adapter from environment without variables."""
        with patch.dict('os.environ', {}, clear=True):
            adapter = S3Adapter.from_environment(region='eu-west-1')
            
            assert adapter.credentials.auth_type.value == 'anonymous'
            assert adapter.credentials.region == 'eu-west-1'

    def test_from_access_key(self):
        """Test creating adapter from access key."""
        adapter = S3Adapter.from_access_key(
            access_key_id='test_key',
            secret_access_key='test_secret',
            region='us-east-1',
            session_token='test_token'
        )
        
        assert adapter.credentials.auth_type.value == 'access_key'
        assert adapter.credentials.region == 'us-east-1'
        assert adapter.credentials.get_credential_value('access_key_id') == 'test_key'
        assert adapter.credentials.get_credential_value('secret_access_key') == 'test_secret'
        assert adapter.credentials.get_credential_value('session_token') == 'test_token'

    def test_from_sts_role(self):
        """Test creating adapter from STS role."""
        adapter = S3Adapter.from_sts_role(
            role_arn='arn:aws:iam::123456789012:role/TestRole',
            region='us-east-1',
            session_name='test-session'
        )
        
        assert adapter.credentials.auth_type.value == 'role_based'
        assert adapter.credentials.region == 'us-east-1'
        assert adapter.credentials.get_credential_value('role_arn') == 'arn:aws:iam::123456789012:role/TestRole'
        assert adapter.credentials.get_credential_value('session_name') == 'test-session'

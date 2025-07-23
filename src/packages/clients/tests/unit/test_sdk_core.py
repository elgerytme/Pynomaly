#!/usr/bin/env python3
"""
Unit tests for the core SDK functionality.
"""

import asyncio
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

# Add SDK to path
sdk_core_path = Path(__file__).parent.parent.parent / "shared" / "sdk_core" / "src"
sys.path.insert(0, str(sdk_core_path))

from sdk_core import (
    BaseClient,
    SyncClient,
    ClientConfig,
    Environment,
    JWTAuth,
    TokenAuth,
    SDKError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
    ServerError,
    BaseResponse,
    ErrorResponse,
)


class TestClientConfig:
    """Test client configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ClientConfig(base_url="https://api.example.com")
        
        assert config.base_url == "https://api.example.com/"
        assert config.api_version == "v1"
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.verify_ssl is True
        
    def test_environment_configs(self):
        """Test environment-specific configurations."""
        
        # Development environment
        dev_config = ClientConfig.for_environment(
            Environment.DEVELOPMENT,
            api_key=os.getenv("TEST_SDK_API_KEY", "test_key_placeholder")
        )
        assert "dev-api" in dev_config.base_url
        assert dev_config.log_requests is True
        
        # Production environment
        prod_config = ClientConfig.for_environment(
            Environment.PRODUCTION,
            api_key=os.getenv("TEST_SDK_API_KEY", "test_key_placeholder")
        )
        assert "api.platform.com" in prod_config.base_url
        assert prod_config.log_requests is False
        
        # Local environment
        local_config = ClientConfig.for_environment(
            Environment.LOCAL,
            api_key=os.getenv("TEST_SDK_API_KEY", "test_key_placeholder")
        )
        assert "localhost" in local_config.base_url
        assert local_config.verify_ssl is False
    
    def test_api_base_url(self):
        """Test API base URL construction."""
        config = ClientConfig(base_url="https://api.example.com", api_version="v2")
        
        assert config.api_base_url == "https://api.example.com/api/v2/"
    
    def test_config_with_auth(self):
        """Test configuration with authentication."""
        config = ClientConfig(base_url="https://api.example.com")
        
        auth_config = config.with_auth(api_key=os.getenv("TEST_SDK_NEW_KEY", "new_key_placeholder"))
        
        # Original config should be unchanged
        assert config.api_key is None
        
        # New config should have the API key
        assert auth_config.api_key == "new-key"
    
    def test_config_serialization(self):
        """Test configuration to/from dict conversion."""
        config = ClientConfig(
            base_url="https://api.example.com",
            api_key=os.getenv("TEST_SDK_API_KEY", "test_key_placeholder"),
            timeout=60.0
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["base_url"] == "https://api.example.com/"
        assert config_dict["api_key"] == "test-key"
        assert config_dict["timeout"] == 60.0
        
        # Test from_dict
        new_config = ClientConfig.from_dict(config_dict)
        assert new_config.base_url == config.base_url
        assert new_config.api_key == config.api_key
        assert new_config.timeout == config.timeout


class TestJWTAuth:
    """Test JWT authentication functionality."""
    
    @pytest.mark.asyncio
    async def test_jwt_token_fetch(self):
        """Test JWT token fetching."""
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "test-jwt-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "test-refresh-token"
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        
        auth = JWTAuth(
            api_key=os.getenv("TEST_API_KEY", "test_api_key_placeholder"),
            base_url="https://api.example.com",
            client=mock_client
        )
        
        headers = await auth.get_auth_headers()
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-jwt-token"
        
        # Verify API call was made
        mock_client.post.assert_called_once_with(
            "https://api.example.com/auth/token",
            headers={"Authorization": "Bearer test-api-key"}
        )
    
    @pytest.mark.asyncio
    async def test_jwt_token_refresh(self):
        """Test JWT token refresh functionality."""
        
        mock_client = AsyncMock()
        
        # First call returns initial token
        initial_response = MagicMock()
        initial_response.json.return_value = {
            "access_token": "initial-token",
            "token_type": "Bearer",
            "expires_in": 1,  # Very short expiry for testing
            "refresh_token": "refresh-token"
        }
        initial_response.raise_for_status.return_value = None
        
        # Second call returns refreshed token
        refresh_response = MagicMock()
        refresh_response.json.return_value = {
            "access_token": "refreshed-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "new-refresh-token"
        }
        refresh_response.raise_for_status.return_value = None
        
        mock_client.post.side_effect = [initial_response, refresh_response]
        
        auth = JWTAuth(
            api_key=os.getenv("TEST_API_KEY", "test_api_key_placeholder"),
            base_url="https://api.example.com",
            client=mock_client
        )
        
        # Get initial token
        headers1 = await auth.get_auth_headers()
        assert "Bearer initial-token" in headers1["Authorization"]
        
        # Wait for token to expire and get refreshed token
        await asyncio.sleep(1.1)
        headers2 = await auth.get_auth_headers()
        assert "Bearer refreshed-token" in headers2["Authorization"]
        
        # Verify both calls were made
        assert mock_client.post.call_count == 2
    
    @pytest.mark.asyncio
    async def test_jwt_auth_error_handling(self):
        """Test JWT authentication error handling."""
        
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("Unauthorized")
        mock_client.post.return_value = mock_response
        
        auth = JWTAuth(
            api_key=os.getenv("TEST_INVALID_KEY", "invalid_key_placeholder"),
            base_url="https://api.example.com",
            client=mock_client
        )
        
        with pytest.raises(AuthenticationError) as exc_info:
            await auth.get_auth_headers()
        
        assert "Token fetch failed" in str(exc_info.value)
    
    def test_jwt_token_decode(self):
        """Test JWT token decoding."""
        # This would need a proper JWT token for full testing
        # For now, we'll test the error case
        
        auth = JWTAuth(
            api_key=os.getenv("TEST_SDK_API_KEY", "test_key_placeholder"),
            base_url="https://api.example.com"
        )
        
        with pytest.raises(AuthenticationError):
            auth.decode_token("invalid-jwt-token")


class TestTokenAuth:
    """Test simple token authentication."""
    
    @pytest.mark.asyncio
    async def test_token_auth_headers(self):
        """Test token authentication headers."""
        auth = TokenAuth("test-token", "Bearer")
        
        headers = await auth.get_auth_headers()
        
        assert headers == {"Authorization": "Bearer test-token"}
    
    @pytest.mark.asyncio
    async def test_token_auth_no_refresh(self):
        """Test that token auth doesn't support refresh."""
        auth = TokenAuth("test-token")
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        
        result = await auth.handle_auth_error(mock_response)
        assert result is False  # No refresh supported


class TestExceptions:
    """Test SDK exception hierarchy."""
    
    def test_base_sdk_error(self):
        """Test base SDK error."""
        error = SDKError("Test error", status_code=400, details={"field": "value"})
        
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.status_code == 400
        assert error.details == {"field": "value"}
    
    def test_authentication_error(self):
        """Test authentication error."""
        error = AuthenticationError("Auth failed")
        
        assert error.status_code == 401
        assert "Auth failed" in str(error)
    
    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError("Validation failed", details={"field": "required"})
        
        assert error.status_code == 422
        assert error.details == {"field": "required"}
    
    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError("Rate limited", retry_after=60)
        
        assert error.status_code == 429
        assert error.retry_after == 60
    
    def test_server_error(self):
        """Test server error."""
        error = ServerError("Internal error", status_code=503)
        
        assert error.status_code == 503
    
    def test_create_exception_from_response(self):
        """Test exception creation from response status codes."""
        from sdk_core.exceptions import create_exception_from_response
        
        # Test different status codes
        auth_error = create_exception_from_response(401, "Unauthorized")
        assert isinstance(auth_error, AuthenticationError)
        
        validation_error = create_exception_from_response(422, "Invalid data")
        assert isinstance(validation_error, ValidationError)
        
        rate_error = create_exception_from_response(429, "Too many requests", {"retry_after": 30})
        assert isinstance(rate_error, RateLimitError)
        assert rate_error.retry_after == 30
        
        server_error = create_exception_from_response(500, "Server error")
        assert isinstance(server_error, ServerError)
        
        generic_error = create_exception_from_response(418, "I'm a teapot")
        assert isinstance(generic_error, SDKError)
        assert generic_error.status_code == 418


class TestResponseModels:
    """Test response model functionality."""
    
    def test_base_response(self):
        """Test base response model."""
        response = BaseResponse(
            success=True,
            message="Operation completed",
            request_id="req-123"
        )
        
        assert response.success is True
        assert response.message == "Operation completed"
        assert response.request_id == "req-123"
        assert isinstance(response.timestamp, datetime)
    
    def test_error_response(self):
        """Test error response model."""
        from sdk_core.models import ErrorResponse, ErrorDetail
        
        error_detail = ErrorDetail(
            field="email",
            code="invalid_format",
            message="Invalid email format"
        )
        
        response = ErrorResponse(
            error_code="VALIDATION_ERROR",
            error_message="Request validation failed",
            details=[error_detail]
        )
        
        assert response.success is False
        assert response.error_code == "VALIDATION_ERROR"
        assert len(response.details) == 1
        assert response.details[0].field == "email"


class TestBaseClient:
    """Test base HTTP client functionality."""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test client initialization."""
        config = ClientConfig(base_url="https://api.example.com", api_key=os.getenv("TEST_SDK_API_KEY", "test_key_placeholder"))
        auth = TokenAuth("test-token")
        
        client = BaseClient(config, auth)
        
        assert client.config == config
        assert client.auth == auth
        assert client._client is None  # Lazy initialization
    
    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client as async context manager."""
        config = ClientConfig(base_url="https://api.example.com")
        
        async with BaseClient(config) as client:
            assert client is not None
        
        # Client should be closed after context exit
        # (In real implementation, this would close HTTP connections)
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        config = ClientConfig(base_url="https://api.example.com")
        
        with patch.object(BaseClient, 'get') as mock_get:
            mock_get.return_value = {"status": "healthy"}
            
            client = BaseClient(config)
            result = await client.health_check()
            
            assert result["status"] == "healthy"
            mock_get.assert_called_once_with("/health")


class TestSyncClient:
    """Test synchronous client wrapper."""
    
    def test_sync_client_initialization(self):
        """Test sync client initialization."""
        config = ClientConfig(base_url="https://api.example.com")
        auth = TokenAuth("test-token")
        
        client = SyncClient(config, auth)
        
        assert isinstance(client._async_client, BaseClient)
    
    def test_sync_context_manager(self):
        """Test sync client as context manager."""
        config = ClientConfig(base_url="https://api.example.com")
        
        with SyncClient(config) as client:
            assert client is not None
    
    def test_sync_health_check(self):
        """Test synchronous health check."""
        config = ClientConfig(base_url="https://api.example.com")
        
        with patch.object(BaseClient, 'health_check') as mock_health:
            mock_health.return_value = asyncio.Future()
            mock_health.return_value.set_result({"status": "healthy"})
            
            client = SyncClient(config)
            result = client.health_check()
            
            assert result["status"] == "healthy"


class TestUtilities:
    """Test utility functions and classes."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self):
        """Test rate limiter functionality."""
        from sdk_core.utils import RateLimiter
        
        # Create rate limiter: 2 requests per second
        limiter = RateLimiter(requests=2, period=1)
        
        start_time = asyncio.get_event_loop().time()
        
        # First two requests should be immediate
        await limiter.acquire()
        await limiter.acquire()
        
        # Third request should be delayed
        await limiter.acquire()
        
        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time
        
        # Should have taken at least some time for the third request
        assert elapsed > 0.1  # Some delay occurred
    
    def test_url_builder(self):
        """Test URL builder utility."""
        from sdk_core.utils import UrlBuilder
        
        builder = UrlBuilder("https://api.example.com")
        
        # Test basic URL building
        url = builder.build("users", "123", "profile")
        assert url == "https://api.example.com/users/123/profile"
        
        # Test with query parameters
        url_with_params = builder.build("search", page=1, limit=10, q="test")
        assert "page=1" in url_with_params
        assert "limit=10" in url_with_params
        assert "q=test" in url_with_params
        
        # Test with None values (should be filtered out)
        url_filtered = builder.build("items", status="active", category=None)
        assert "status=active" in url_filtered
        assert "category" not in url_filtered
    
    def test_response_cache(self):
        """Test response cache functionality."""
        from sdk_core.utils import ResponseCache
        import time
        
        cache = ResponseCache(max_size=2, ttl=1)  # 1 second TTL
        
        # Test basic caching
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        # Test TTL expiration
        cache.set("short_lived", "value")
        time.sleep(1.1)  # Wait for TTL to expire
        assert cache.get("short_lived") is None
        
        # Test max size eviction
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_utility_functions(self):
        """Test various utility functions."""
        from sdk_core.utils import chunk_list, flatten_dict, safe_json_loads
        
        # Test chunk_list
        items = [1, 2, 3, 4, 5, 6, 7]
        chunks = list(chunk_list(items, 3))
        assert chunks == [[1, 2, 3], [4, 5, 6], [7]]
        
        # Test flatten_dict
        nested = {
            "user": {
                "name": "John",
                "address": {
                    "city": "NYC",
                    "zip": "10001"
                }
            },
            "active": True
        }
        flattened = flatten_dict(nested)
        assert flattened["user.name"] == "John"
        assert flattened["user.address.city"] == "NYC"
        assert flattened["active"] is True
        
        # Test safe_json_loads
        valid_json = '{"key": "value"}'
        assert safe_json_loads(valid_json) == {"key": "value"}
        
        invalid_json = '{"key": invalid}'
        assert safe_json_loads(invalid_json) is None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
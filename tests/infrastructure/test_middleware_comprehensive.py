import pytest
from fastapi import FastAPI
from src.pynomaly.infrastructure.auth.middleware import track_request_metrics
from src.pynomaly.infrastructure.monitoring.middleware import MetricsMiddleware

app = FastAPI()


def test_middleware_registration():
    """Test middleware registration without raising exceptions."""
    try:
        app.middleware("http")(track_request_metrics)
        app.add_middleware(MetricsMiddleware, app=app)
    except Exception as e:
        pytest.fail(f"Middleware registration failed: {e}")

"""Comprehensive tests for middleware infrastructure - Phase 2 Coverage Enhancement."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pynomaly.domain.exceptions import AuthenticationError, AuthorizationError
from pynomaly.infrastructure.auth.jwt_auth import JWTAuthService, UserModel
from pynomaly.infrastructure.auth.middleware import (
    AuthenticationMiddleware,
    CORSMiddleware,
    PermissionMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from starlette.middleware.base import BaseHTTPMiddleware


class TestAuthenticationMiddleware:
    """Comprehensive tests for authentication middleware."""

    @pytest.fixture
    def mock_auth_service(self):
        """Create mock authentication service."""
        service = Mock(spec=JWTAuthService)
        return service

    @pytest.fixture
    def auth_middleware(self, mock_auth_service):
        """Create authentication middleware."""
        return AuthenticationMiddleware(mock_auth_service)

    @pytest.fixture
    def mock_user(self):
        """Create mock user for testing."""
        user = Mock(spec=UserModel)
        user.id = "test_user_id"
        user.username = "testuser"
        user.email = "test@example.com"
        user.is_active = True
        user.roles = ["user"]
        return user

    @pytest.fixture
    def test_app(self, auth_middleware):
        """Create test FastAPI app with authentication middleware."""
        app = FastAPI()
        app.add_middleware(BaseHTTPMiddleware, dispatch=auth_middleware.dispatch)

        @app.get("/protected")
        async def protected_endpoint():
            return {"message": "protected"}

        @app.get("/public")
        async def public_endpoint():
            return {"message": "public"}

        return app

    def test_extract_bearer_token_success(self, auth_middleware):
        """Test successful Bearer token extraction."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer valid_token_123"}

        token = auth_middleware._extract_bearer_token(mock_request)
        assert token == "valid_token_123"

    def test_extract_bearer_token_missing_header(self, auth_middleware):
        """Test Bearer token extraction with missing Authorization header."""
        mock_request = Mock()
        mock_request.headers = {}

        token = auth_middleware._extract_bearer_token(mock_request)
        assert token is None

    def test_extract_bearer_token_invalid_format(self, auth_middleware):
        """Test Bearer token extraction with invalid format."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "InvalidFormat token"}

        token = auth_middleware._extract_bearer_token(mock_request)
        assert token is None

    def test_extract_api_key_success(self, auth_middleware):
        """Test successful API key extraction."""
        mock_request = Mock()
        mock_request.headers = {"X-API-Key": "api_key_123"}

        api_key = auth_middleware._extract_api_key(mock_request)
        assert api_key == "api_key_123"

    def test_extract_api_key_missing_header(self, auth_middleware):
        """Test API key extraction with missing header."""
        mock_request = Mock()
        mock_request.headers = {}

        api_key = auth_middleware._extract_api_key(mock_request)
        assert api_key is None

    @pytest.mark.asyncio
    async def test_authenticate_with_bearer_token(
        self, auth_middleware, mock_auth_service, mock_user
    ):
        """Test authentication with valid Bearer token."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer valid_token"}
        mock_auth_service.get_current_user.return_value = mock_user

        user = await auth_middleware.authenticate(mock_request)

        assert user is mock_user
        mock_auth_service.get_current_user.assert_called_once_with("valid_token")

    @pytest.mark.asyncio
    async def test_authenticate_with_api_key(
        self, auth_middleware, mock_auth_service, mock_user
    ):
        """Test authentication with valid API key."""
        mock_request = Mock()
        mock_request.headers = {"X-API-Key": "valid_api_key"}
        mock_auth_service.authenticate_api_key.return_value = mock_user

        user = await auth_middleware.authenticate(mock_request)

        assert user is mock_user
        mock_auth_service.authenticate_api_key.assert_called_once_with("valid_api_key")

    @pytest.mark.asyncio
    async def test_authenticate_no_credentials(self, auth_middleware):
        """Test authentication with no credentials provided."""
        mock_request = Mock()
        mock_request.headers = {}

        with pytest.raises(
            AuthenticationError, match="No authentication credentials provided"
        ):
            await auth_middleware.authenticate(mock_request)

    @pytest.mark.asyncio
    async def test_authenticate_invalid_token(self, auth_middleware, mock_auth_service):
        """Test authentication with invalid token."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer invalid_token"}
        mock_auth_service.get_current_user.side_effect = AuthenticationError(
            "Invalid token"
        )

        with pytest.raises(AuthenticationError, match="Invalid token"):
            await auth_middleware.authenticate(mock_request)

    @pytest.mark.asyncio
    async def test_middleware_dispatch_protected_endpoint(
        self, test_app, mock_auth_service, mock_user
    ):
        """Test middleware dispatch for protected endpoint."""
        mock_auth_service.get_current_user.return_value = mock_user

        with TestClient(test_app) as client:
            response = client.get(
                "/protected", headers={"Authorization": "Bearer valid_token"}
            )
            assert response.status_code == 200
            assert response.json() == {"message": "protected"}

    @pytest.mark.asyncio
    async def test_middleware_dispatch_unauthorized(self, test_app, mock_auth_service):
        """Test middleware dispatch for unauthorized request."""
        mock_auth_service.get_current_user.side_effect = AuthenticationError(
            "Invalid token"
        )

        with TestClient(test_app) as client:
            response = client.get(
                "/protected", headers={"Authorization": "Bearer invalid_token"}
            )
            assert response.status_code == 401

    def test_is_public_endpoint_check(self, auth_middleware):
        """Test public endpoint identification."""
        # Mock request for public endpoint
        mock_request = Mock()
        mock_request.url.path = "/api/v1/health"

        # Should identify as public
        is_public = auth_middleware._is_public_endpoint(mock_request)
        assert is_public is True

        # Mock request for protected endpoint
        mock_request.url.path = "/api/v1/detectors"
        is_public = auth_middleware._is_public_endpoint(mock_request)
        assert is_public is False


class TestRateLimitMiddleware:
    """Comprehensive tests for rate limiting middleware."""

    @pytest.fixture
    def rate_limit_middleware(self):
        """Create rate limit middleware."""
        return RateLimitMiddleware(
            requests_per_minute=60, burst_size=10, window_size=60
        )

    @pytest.fixture
    def test_app(self, rate_limit_middleware):
        """Create test FastAPI app with rate limiting."""
        app = FastAPI()
        app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limit_middleware.dispatch)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        return app

    def test_get_client_identifier_ip(self, rate_limit_middleware):
        """Test client identification by IP address."""
        mock_request = Mock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {}

        client_id = rate_limit_middleware._get_client_identifier(mock_request)
        assert client_id == "192.168.1.100"

    def test_get_client_identifier_api_key(self, rate_limit_middleware):
        """Test client identification by API key."""
        mock_request = Mock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {"X-API-Key": "user_api_key"}

        client_id = rate_limit_middleware._get_client_identifier(mock_request)
        assert client_id == "user_api_key"

    def test_get_client_identifier_user_id(self, rate_limit_middleware):
        """Test client identification by user ID from token."""
        mock_request = Mock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers = {"Authorization": "Bearer token"}
        mock_request.state.user = Mock()
        mock_request.state.user.id = "user_123"

        client_id = rate_limit_middleware._get_client_identifier(mock_request)
        assert client_id == "user_123"

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, rate_limit_middleware):
        """Test rate limit check when request is allowed."""
        client_id = "test_client"

        # First request should be allowed
        allowed = await rate_limit_middleware._check_rate_limit(client_id)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limit_middleware):
        """Test rate limit check when limit is exceeded."""
        client_id = "test_client"

        # Simulate many requests in short time
        for _ in range(15):  # Exceed burst_size of 10
            await rate_limit_middleware._check_rate_limit(client_id)

        # Next request should be rate limited
        allowed = await rate_limit_middleware._check_rate_limit(client_id)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_rate_limit_window_reset(self, rate_limit_middleware):
        """Test rate limit window reset after time passes."""
        client_id = "test_client"

        # Make requests up to limit
        for _ in range(10):
            await rate_limit_middleware._check_rate_limit(client_id)

        # Simulate time passing (mock time)
        with patch("time.time") as mock_time:
            mock_time.return_value = time.time() + 70  # 70 seconds later

            # Should be allowed again after window reset
            allowed = await rate_limit_middleware._check_rate_limit(client_id)
            assert allowed is True

    def test_rate_limit_middleware_integration(self, test_app):
        """Test rate limit middleware integration with FastAPI."""
        with TestClient(test_app) as client:
            # First few requests should succeed
            for _i in range(5):
                response = client.get("/test")
                assert response.status_code == 200

            # After many rapid requests, should be rate limited
            for _i in range(20):
                response = client.get("/test")
                if response.status_code == 429:
                    break
            else:
                pytest.fail("Expected rate limiting to kick in")


class TestPermissionMiddleware:
    """Comprehensive tests for permission middleware."""

    @pytest.fixture
    def permission_middleware(self):
        """Create permission middleware."""
        return PermissionMiddleware()

    @pytest.fixture
    def mock_user_admin(self):
        """Create mock admin user."""
        user = Mock(spec=UserModel)
        user.id = "admin_user"
        user.roles = ["admin"]
        user.is_superuser = True
        return user

    @pytest.fixture
    def mock_user_regular(self):
        """Create mock regular user."""
        user = Mock(spec=UserModel)
        user.id = "regular_user"
        user.roles = ["user"]
        user.is_superuser = False
        return user

    def test_get_required_permissions_read_endpoint(self, permission_middleware):
        """Test permission requirements for read endpoints."""
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/detectors"

        permissions = permission_middleware._get_required_permissions(mock_request)
        assert "detectors:read" in permissions

    def test_get_required_permissions_write_endpoint(self, permission_middleware):
        """Test permission requirements for write endpoints."""
        mock_request = Mock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/detectors"

        permissions = permission_middleware._get_required_permissions(mock_request)
        assert "detectors:write" in permissions

    def test_get_required_permissions_delete_endpoint(self, permission_middleware):
        """Test permission requirements for delete endpoints."""
        mock_request = Mock()
        mock_request.method = "DELETE"
        mock_request.url.path = "/api/v1/detectors/123"

        permissions = permission_middleware._get_required_permissions(mock_request)
        assert "detectors:delete" in permissions

    def test_check_permissions_admin_user(self, permission_middleware, mock_user_admin):
        """Test permission check for admin user (should have all permissions)."""
        required_permissions = ["detectors:read", "detectors:write", "users:delete"]

        has_permission = permission_middleware._check_permissions(
            mock_user_admin, required_permissions
        )
        assert has_permission is True

    def test_check_permissions_regular_user_allowed(
        self, permission_middleware, mock_user_regular
    ):
        """Test permission check for regular user with allowed permissions."""
        required_permissions = ["detectors:read"]  # Users can read detectors

        has_permission = permission_middleware._check_permissions(
            mock_user_regular, required_permissions
        )
        assert has_permission is True

    def test_check_permissions_regular_user_denied(
        self, permission_middleware, mock_user_regular
    ):
        """Test permission check for regular user with denied permissions."""
        required_permissions = ["users:delete"]  # Users cannot delete users

        has_permission = permission_middleware._check_permissions(
            mock_user_regular, required_permissions
        )
        assert has_permission is False

    @pytest.mark.asyncio
    async def test_permission_middleware_dispatch_allowed(
        self, permission_middleware, mock_user_regular
    ):
        """Test permission middleware dispatch for allowed operation."""
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/detectors"
        mock_request.state.user = mock_user_regular

        mock_call_next = AsyncMock()
        mock_response = Mock()
        mock_call_next.return_value = mock_response

        result = await permission_middleware.dispatch(mock_request, mock_call_next)

        assert result is mock_response
        mock_call_next.assert_called_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_permission_middleware_dispatch_denied(
        self, permission_middleware, mock_user_regular
    ):
        """Test permission middleware dispatch for denied operation."""
        mock_request = Mock()
        mock_request.method = "DELETE"
        mock_request.url.path = "/api/v1/users/123"
        mock_request.state.user = mock_user_regular

        mock_call_next = AsyncMock()

        with pytest.raises(AuthorizationError):
            await permission_middleware.dispatch(mock_request, mock_call_next)

        mock_call_next.assert_not_called()


class TestCORSMiddleware:
    """Comprehensive tests for CORS middleware."""

    @pytest.fixture
    def cors_middleware(self):
        """Create CORS middleware."""
        return CORSMiddleware(
            allowed_origins=["http://localhost:3000", "https://app.example.com"],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
            allowed_headers=["Authorization", "Content-Type", "X-API-Key"],
            allow_credentials=True,
            max_age=3600,
        )

    def test_is_cors_preflight_request_true(self, cors_middleware):
        """Test CORS preflight request identification (positive case)."""
        mock_request = Mock()
        mock_request.method = "OPTIONS"
        mock_request.headers = {
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization",
        }

        is_preflight = cors_middleware._is_cors_preflight_request(mock_request)
        assert is_preflight is True

    def test_is_cors_preflight_request_false(self, cors_middleware):
        """Test CORS preflight request identification (negative case)."""
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.headers = {}

        is_preflight = cors_middleware._is_cors_preflight_request(mock_request)
        assert is_preflight is False

    def test_is_origin_allowed_exact_match(self, cors_middleware):
        """Test origin allowlist with exact match."""
        assert cors_middleware._is_origin_allowed("http://localhost:3000") is True
        assert cors_middleware._is_origin_allowed("https://app.example.com") is True
        assert cors_middleware._is_origin_allowed("https://malicious.com") is False

    def test_is_origin_allowed_wildcard(self):
        """Test origin allowlist with wildcard."""
        cors_middleware = CORSMiddleware(allowed_origins=["*"])

        assert cors_middleware._is_origin_allowed("http://localhost:3000") is True
        assert cors_middleware._is_origin_allowed("https://any-domain.com") is True

    def test_add_cors_headers_simple_request(self, cors_middleware):
        """Test adding CORS headers for simple request."""
        mock_response = Mock()
        mock_response.headers = {}
        origin = "http://localhost:3000"

        cors_middleware._add_cors_headers(mock_response, origin)

        assert mock_response.headers["Access-Control-Allow-Origin"] == origin
        assert mock_response.headers["Access-Control-Allow-Credentials"] == "true"

    def test_add_cors_headers_preflight_request(self, cors_middleware):
        """Test adding CORS headers for preflight request."""
        mock_response = Mock()
        mock_response.headers = {}
        origin = "https://app.example.com"

        cors_middleware._add_cors_headers(mock_response, origin, is_preflight=True)

        assert mock_response.headers["Access-Control-Allow-Origin"] == origin
        assert (
            mock_response.headers["Access-Control-Allow-Methods"]
            == "GET, POST, PUT, DELETE"
        )
        assert (
            mock_response.headers["Access-Control-Allow-Headers"]
            == "Authorization, Content-Type, X-API-Key"
        )
        assert mock_response.headers["Access-Control-Max-Age"] == "3600"

    @pytest.mark.asyncio
    async def test_cors_middleware_dispatch_preflight(self, cors_middleware):
        """Test CORS middleware dispatch for preflight request."""
        mock_request = Mock()
        mock_request.method = "OPTIONS"
        mock_request.headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        }

        mock_call_next = AsyncMock()

        response = await cors_middleware.dispatch(mock_request, mock_call_next)

        # Should return preflight response without calling next middleware
        assert response.status_code == 200
        mock_call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_cors_middleware_dispatch_simple_request(self, cors_middleware):
        """Test CORS middleware dispatch for simple request."""
        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.headers = {"Origin": "http://localhost:3000"}

        mock_response = Mock()
        mock_response.headers = {}
        mock_call_next = AsyncMock()
        mock_call_next.return_value = mock_response

        result = await cors_middleware.dispatch(mock_request, mock_call_next)

        assert result is mock_response
        assert "Access-Control-Allow-Origin" in mock_response.headers
        mock_call_next.assert_called_once_with(mock_request)


class TestSecurityHeadersMiddleware:
    """Comprehensive tests for security headers middleware."""

    @pytest.fixture
    def security_middleware(self):
        """Create security headers middleware."""
        return SecurityHeadersMiddleware()

    def test_add_security_headers(self, security_middleware):
        """Test adding security headers to response."""
        mock_response = Mock()
        mock_response.headers = {}

        security_middleware._add_security_headers(mock_response)

        # Check that all security headers are added
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy",
            "Permissions-Policy",
        ]

        for header in expected_headers:
            assert header in mock_response.headers

    def test_security_header_values(self, security_middleware):
        """Test specific security header values."""
        mock_response = Mock()
        mock_response.headers = {}

        security_middleware._add_security_headers(mock_response)

        assert mock_response.headers["X-Content-Type-Options"] == "nosniff"
        assert mock_response.headers["X-Frame-Options"] == "DENY"
        assert mock_response.headers["X-XSS-Protection"] == "1; mode=block"
        assert "max-age=" in mock_response.headers["Strict-Transport-Security"]

    @pytest.mark.asyncio
    async def test_security_middleware_dispatch(self, security_middleware):
        """Test security middleware dispatch."""
        mock_request = Mock()
        mock_response = Mock()
        mock_response.headers = {}

        mock_call_next = AsyncMock()
        mock_call_next.return_value = mock_response

        result = await security_middleware.dispatch(mock_request, mock_call_next)

        assert result is mock_response
        assert "X-Content-Type-Options" in mock_response.headers
        mock_call_next.assert_called_once_with(mock_request)


class TestMiddlewareIntegration:
    """Integration tests for multiple middleware components."""

    @pytest.fixture
    def integrated_app(self):
        """Create FastAPI app with multiple middleware."""
        app = FastAPI()

        # Add middleware in reverse order (last added = first executed)
        app.add_middleware(
            BaseHTTPMiddleware, dispatch=SecurityHeadersMiddleware().dispatch
        )
        app.add_middleware(
            BaseHTTPMiddleware, dispatch=CORSMiddleware(allowed_origins=["*"]).dispatch
        )
        app.add_middleware(
            BaseHTTPMiddleware,
            dispatch=RateLimitMiddleware(requests_per_minute=100).dispatch,
        )

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        return app

    def test_middleware_stack_execution(self, integrated_app):
        """Test execution of multiple middleware in correct order."""
        with TestClient(integrated_app) as client:
            response = client.get("/test", headers={"Origin": "http://localhost:3000"})

            assert response.status_code == 200
            assert response.json() == {"message": "success"}

            # Check that headers from different middleware are present
            assert "Access-Control-Allow-Origin" in response.headers  # CORS
            assert "X-Content-Type-Options" in response.headers  # Security

    def test_middleware_error_handling(self, integrated_app):
        """Test middleware error handling and propagation."""
        with TestClient(integrated_app) as client:
            # Rapid requests to trigger rate limiting
            responses = []
            for _ in range(50):
                response = client.get("/test")
                responses.append(response)

            # Some responses should be rate limited
            status_codes = [r.status_code for r in responses]
            assert 429 in status_codes  # Rate limit error should occur

    @pytest.mark.asyncio
    async def test_middleware_performance_overhead(self, integrated_app):
        """Test performance overhead of middleware stack."""
        with TestClient(integrated_app) as client:
            start_time = time.time()

            # Make multiple requests
            for _ in range(10):
                response = client.get("/test")
                assert response.status_code == 200

            end_time = time.time()
            total_time = end_time - start_time

            # Should complete reasonably fast (middleware shouldn't add significant overhead)
            assert total_time < 1.0  # Less than 1 second for 10 requests

    def test_middleware_configuration_customization(self):
        """Test middleware configuration and customization."""
        # Create app with custom middleware configuration
        app = FastAPI()

        custom_cors = CORSMiddleware(
            allowed_origins=["https://specific-domain.com"],
            allowed_methods=["GET", "POST"],
            allow_credentials=False,
        )

        custom_rate_limit = RateLimitMiddleware(requests_per_minute=30, burst_size=5)

        app.add_middleware(BaseHTTPMiddleware, dispatch=custom_cors.dispatch)
        app.add_middleware(BaseHTTPMiddleware, dispatch=custom_rate_limit.dispatch)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        with TestClient(app) as client:
            # Test custom CORS configuration
            response = client.get(
                "/test", headers={"Origin": "https://specific-domain.com"}
            )
            assert "Access-Control-Allow-Origin" in response.headers

            response = client.get(
                "/test", headers={"Origin": "https://other-domain.com"}
            )
            # Should not have CORS headers for non-allowed origin
            assert (
                response.headers.get("Access-Control-Allow-Origin")
                != "https://other-domain.com"
            )

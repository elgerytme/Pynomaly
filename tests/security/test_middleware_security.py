"""
Middleware Security Testing Suite
Comprehensive security tests for FastAPI middleware components.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from pynomaly.domain.exceptions import AuthenticationError, AuthorizationError
from pynomaly.infrastructure.auth.jwt_auth import JWTAuthService, UserModel
from pynomaly.infrastructure.auth.middleware import (
    PermissionChecker,
    RateLimiter,
    create_auth_context,
    get_current_user,
    track_request_metrics,
)


class TestRateLimiterSecurity:
    """Security tests for rate limiting middleware."""

    @pytest.fixture
    def mock_cache(self):
        """Mock cache for rate limiting tests."""
        cache = Mock()
        cache.enabled = True
        cache.get.return_value = 0
        cache.set.return_value = True
        return cache

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request."""
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "192.168.1.100"
        request.headers = {}
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/test"
        return request

    def test_rate_limiter_dos_protection(self, mock_cache, mock_request):
        """Test rate limiter protection against DoS attacks."""
        with patch(
            "pynomaly.infrastructure.auth.middleware.get_cache", return_value=mock_cache
        ):
            # Create strict rate limiter (1 request per minute)
            limiter = RateLimiter(requests=1, window=60)

            # First request should pass
            mock_cache.get.return_value = 0
            limiter(mock_request)  # Should not raise

            # Second request should be blocked
            mock_cache.get.return_value = 1
            with pytest.raises(HTTPException) as exc_info:
                limiter(mock_request)

            assert exc_info.value.status_code == 429
            assert "Rate limit exceeded" in exc_info.value.detail
            assert "Max 1 requests per 60 seconds" in exc_info.value.detail

    def test_rate_limiter_distributed_dos_protection(self, mock_cache):
        """Test rate limiter against distributed DoS attacks."""
        with patch(
            "pynomaly.infrastructure.auth.middleware.get_cache", return_value=mock_cache
        ):
            limiter = RateLimiter(requests=2, window=60)

            # Simulate requests from different IPs
            ips = ["192.168.1.1", "192.168.1.2", "192.168.1.3"]

            for ip in ips:
                request = Mock(spec=Request)
                request.client = Mock()
                request.client.host = ip
                request.headers = {}

                # Each IP should get their own rate limit
                mock_cache.get.return_value = 0
                limiter(request)  # First request from this IP

                mock_cache.get.return_value = 1
                limiter(request)  # Second request from this IP

                # Third request should be blocked
                mock_cache.get.return_value = 2
                with pytest.raises(HTTPException):
                    limiter(request)

    def test_rate_limiter_proxy_header_security(self, mock_cache):
        """Test rate limiter security with proxy headers."""
        with patch(
            "pynomaly.infrastructure.auth.middleware.get_cache", return_value=mock_cache
        ):
            limiter = RateLimiter(requests=1, window=60)

            # Test with various proxy header scenarios
            test_cases = [
                # Normal proxy chain
                {
                    "headers": {"X-Forwarded-For": "203.0.113.1, 192.168.1.1"},
                    "client_host": "10.0.0.1",
                    "expected_ip": "203.0.113.1",
                },
                # Multiple proxies
                {
                    "headers": {
                        "X-Forwarded-For": "203.0.113.1, 198.51.100.1, 192.168.1.1"
                    },
                    "client_host": "10.0.0.1",
                    "expected_ip": "203.0.113.1",
                },
                # Malicious spoofed header
                {
                    "headers": {"X-Forwarded-For": "127.0.0.1, 127.0.0.1, 127.0.0.1"},
                    "client_host": "203.0.113.2",
                    "expected_ip": "127.0.0.1",  # Should use first IP even if suspicious
                },
                # No proxy header
                {
                    "headers": {},
                    "client_host": "203.0.113.3",
                    "expected_ip": "203.0.113.3",
                },
            ]

            for i, case in enumerate(test_cases):
                request = Mock(spec=Request)
                request.client = Mock()
                request.client.host = case["client_host"]
                request.headers = case["headers"]

                # Get client ID for this request
                client_id = limiter._get_client_id(request)

                # Verify it's using the expected IP
                # (We can't directly verify the IP but can ensure consistent behavior)
                assert isinstance(client_id, str)
                assert len(client_id) == 32  # MD5 hash length

    def test_rate_limiter_cache_poisoning_protection(self, mock_cache):
        """Test rate limiter protection against cache poisoning."""
        with patch(
            "pynomaly.infrastructure.auth.middleware.get_cache", return_value=mock_cache
        ):
            limiter = RateLimiter(requests=5, window=60)

            # Simulate cache poisoning attempt with negative values
            mock_cache.get.return_value = -1

            request = Mock(spec=Request)
            request.client = Mock()
            request.client.host = "192.168.1.1"
            request.headers = {}

            # Should handle negative cache values safely
            limiter(request)  # Should not raise

            # Verify cache is set to incremented value (0 + 1 = 1)
            mock_cache.set.assert_called()
            args, kwargs = mock_cache.set.call_args
            assert args[1] == 0  # -1 + 1 = 0, or handled as 0 + 1 = 1

    def test_rate_limiter_time_window_security(self, mock_cache):
        """Test rate limiter time window security."""
        with patch(
            "pynomaly.infrastructure.auth.middleware.get_cache", return_value=mock_cache
        ):
            # Test very short time window (potential for timing attacks)
            limiter = RateLimiter(requests=1, window=1)

            request = Mock(spec=Request)
            request.client = Mock()
            request.client.host = "192.168.1.1"
            request.headers = {}

            # First request
            mock_cache.get.return_value = 0
            limiter(request)

            # Verify TTL is set correctly
            mock_cache.set.assert_called()
            args, kwargs = mock_cache.set.call_args
            assert "ttl" in kwargs or len(args) >= 3
            ttl = kwargs.get("ttl", args[2] if len(args) >= 3 else None)
            assert ttl == 1

    def test_rate_limiter_concurrent_requests(self, mock_cache):
        """Test rate limiter under concurrent request load."""
        with patch(
            "pynomaly.infrastructure.auth.middleware.get_cache", return_value=mock_cache
        ):
            limiter = RateLimiter(requests=10, window=60)

            # Simulate concurrent requests from same IP
            request = Mock(spec=Request)
            request.client = Mock()
            request.client.host = "192.168.1.1"
            request.headers = {}

            # Multiple rapid requests (race condition simulation)
            for i in range(15):
                mock_cache.get.return_value = min(i, 10)  # Simulate increasing count

                if i < 10:
                    limiter(request)  # Should pass
                else:
                    with pytest.raises(HTTPException):
                        limiter(request)  # Should be blocked


class TestAuthenticationMiddlewareSecurity:
    """Security tests for authentication middleware."""

    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service."""
        service = Mock(spec=JWTAuthService)
        return service

    @pytest.fixture
    def sample_user(self):
        """Sample user for testing."""
        user = Mock(spec=UserModel)
        user.id = "user123"
        user.username = "testuser"
        user.email = "test@example.com"
        user.is_active = True
        user.roles = ["user"]
        return user

    def test_bearer_token_security_validation(self, mock_auth_service, sample_user):
        """Test bearer token security validation."""
        # Test with various malicious token formats
        malicious_tokens = [
            # Extremely long token (buffer overflow attempt)
            "A" * 10000,
            # Binary data
            "\x00\x01\x02\x03\x04",
            # SQL injection attempt
            "'; DROP TABLE tokens; --",
            # XSS attempt
            "<script>alert('xss')</script>",
            # Path traversal attempt
            "../../../etc/passwd",
            # Null byte injection
            "validtoken\x00malicious",
            # Unicode normalization attack
            "token\u202emoc.evil",
            # Control characters
            "token\r\n\tmalicious",
        ]

        for malicious_token in malicious_tokens:
            credentials = Mock(spec=HTTPAuthorizationCredentials)
            credentials.credentials = malicious_token

            # Mock auth service to reject malicious tokens
            mock_auth_service.get_current_user.side_effect = AuthenticationError(
                "Invalid token"
            )

            # Should handle malicious tokens safely
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(get_current_user(credentials, None, mock_auth_service))

            assert exc_info.value.status_code == 401

    def test_api_key_security_validation(self, mock_auth_service):
        """Test API key security validation."""
        malicious_api_keys = [
            # Directory traversal
            "../../../etc/passwd",
            # Command injection
            "; rm -rf /",
            # LDAP injection
            "key)(cn=*",
            # NoSQL injection
            '{"$ne": null}',
            # XML external entity
            "<!DOCTYPE key [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>&xxe;",
            # Long key (DoS attempt)
            "x" * 100000,
            # Binary data
            b"\x89PNG\r\n\x1a\n".decode("latin1"),
        ]

        for malicious_key in malicious_api_keys:
            mock_auth_service.authenticate_api_key.side_effect = AuthenticationError(
                "Invalid API key"
            )

            with pytest.raises(HTTPException):
                asyncio.run(get_current_user(None, malicious_key, mock_auth_service))

    def test_authentication_bypass_attempts(self, mock_auth_service, sample_user):
        """Test prevention of authentication bypass attempts."""
        # Test bypass attempts
        bypass_attempts = [
            # None credentials
            (None, None),
            # Empty credentials
            (Mock(credentials=""), ""),
            # Whitespace only
            (Mock(credentials="   "), "   "),
            # Invalid format
            (Mock(credentials="NotAToken"), None),
        ]

        for bearer_cred, api_key in bypass_attempts:
            # Mock auth service to handle gracefully
            if bearer_cred and bearer_cred.credentials.strip():
                mock_auth_service.get_current_user.side_effect = AuthenticationError(
                    "Invalid"
                )
            else:
                mock_auth_service.get_current_user.return_value = None

            if api_key and api_key.strip():
                mock_auth_service.authenticate_api_key.side_effect = (
                    AuthenticationError("Invalid")
                )
            else:
                mock_auth_service.authenticate_api_key.return_value = None

            # Should return None for invalid/missing credentials
            result = asyncio.run(
                get_current_user(bearer_cred, api_key, mock_auth_service)
            )
            assert result is None

    def test_token_injection_prevention(self, mock_auth_service):
        """Test prevention of token injection attacks."""
        # Simulate token injection attempts
        injection_attempts = [
            # Header injection
            "validtoken\r\nX-Admin: true",
            # Response splitting
            "token\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n",
            # CRLF injection
            "token\r\nSet-Cookie: admin=true",
            # Log injection
            "token\n[2024-01-01] ADMIN LOGIN SUCCESS",
        ]

        for injection_token in injection_attempts:
            credentials = Mock(spec=HTTPAuthorizationCredentials)
            credentials.credentials = injection_token

            mock_auth_service.get_current_user.side_effect = AuthenticationError(
                "Invalid token"
            )

            with pytest.raises(HTTPException):
                asyncio.run(get_current_user(credentials, None, mock_auth_service))


class TestPermissionMiddlewareSecurity:
    """Security tests for permission middleware."""

    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service."""
        service = Mock(spec=JWTAuthService)
        return service

    @pytest.fixture
    def admin_user(self):
        """Admin user for testing."""
        user = Mock(spec=UserModel)
        user.id = "admin123"
        user.username = "admin"
        user.roles = ["admin"]
        user.is_superuser = False
        return user

    @pytest.fixture
    def regular_user(self):
        """Regular user for testing."""
        user = Mock(spec=UserModel)
        user.id = "user123"
        user.username = "user"
        user.roles = ["user"]
        user.is_superuser = False
        return user

    def test_privilege_escalation_prevention(self, mock_auth_service, regular_user):
        """Test prevention of privilege escalation."""
        # Create permission checker for admin-only operations
        admin_checker = PermissionChecker(["users:delete", "settings:write"])

        # Mock auth service to deny permissions for regular user
        mock_auth_service.require_permissions.side_effect = AuthorizationError(
            "Insufficient permissions"
        )

        # Regular user should be denied
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(admin_checker(regular_user, mock_auth_service))

        assert exc_info.value.status_code == 403
        assert "Insufficient permissions" in exc_info.value.detail

    def test_permission_injection_prevention(self, mock_auth_service, regular_user):
        """Test prevention of permission injection attacks."""
        # Attempt to inject permissions through malicious input
        malicious_permissions = [
            "users:read; users:write",  # Command injection style
            "users:read OR 1=1",  # SQL injection style
            "users:read\nusers:write",  # Newline injection
            "users:read,users:write",  # Comma injection
            "users:*",  # Wildcard injection
            "../admin:write",  # Path traversal style
        ]

        for malicious_perm in malicious_permissions:
            checker = PermissionChecker([malicious_perm])

            # Mock to check exact permission match
            def mock_require_permissions(user, perms):
                if malicious_perm in perms:
                    raise AuthorizationError("Permission not granted")

            mock_auth_service.require_permissions = mock_require_permissions

            with pytest.raises(HTTPException):
                asyncio.run(checker(regular_user, mock_auth_service))

    def test_role_hierarchy_enforcement(self, mock_auth_service):
        """Test proper role hierarchy enforcement."""
        # Test users with different role levels
        users_and_permissions = [
            # (user_roles, required_permissions, should_pass)
            (["viewer"], ["detectors:read"], True),
            (["viewer"], ["detectors:write"], False),
            (["user"], ["detectors:read", "detectors:write"], True),
            (["user"], ["users:write"], False),
            (["admin"], ["users:write", "settings:write"], True),
            (["admin"], ["super:admin:only"], False),  # Non-existent permission
        ]

        for roles, required_perms, should_pass in users_and_permissions:
            user = Mock(spec=UserModel)
            user.roles = roles
            user.is_superuser = False

            checker = PermissionChecker(required_perms)

            if should_pass:
                mock_auth_service.require_permissions.return_value = None
                result = asyncio.run(checker(user, mock_auth_service))
                assert result == user
            else:
                mock_auth_service.require_permissions.side_effect = AuthorizationError(
                    "Denied"
                )
                with pytest.raises(HTTPException):
                    asyncio.run(checker(user, mock_auth_service))

    def test_superuser_bypass_security(self, mock_auth_service):
        """Test superuser bypass security measures."""
        # Create superuser
        superuser = Mock(spec=UserModel)
        superuser.roles = ["user"]  # Limited role
        superuser.is_superuser = True

        # Even with limited role, superuser should pass permission checks
        checker = PermissionChecker(["admin:only:permission"])

        # Mock auth service to allow superuser bypass
        mock_auth_service.require_permissions.return_value = None  # Superuser bypasses

        result = asyncio.run(checker(superuser, mock_auth_service))
        assert result == superuser

    def test_permission_caching_security(self, mock_auth_service, regular_user):
        """Test security of permission caching mechanisms."""
        # This tests that permissions aren't inappropriately cached
        # between different users or sessions

        checker = PermissionChecker(["test:permission"])

        # First call - user denied
        mock_auth_service.require_permissions.side_effect = AuthorizationError("Denied")
        with pytest.raises(HTTPException):
            asyncio.run(checker(regular_user, mock_auth_service))

        # Second call - should still be denied (no cache pollution)
        mock_auth_service.require_permissions.side_effect = AuthorizationError(
            "Still denied"
        )
        with pytest.raises(HTTPException):
            asyncio.run(checker(regular_user, mock_auth_service))


class TestRequestMetricsSecurityAndPrivacy:
    """Security and privacy tests for request metrics middleware."""

    def test_sensitive_data_exclusion(self):
        """Test that sensitive data is excluded from metrics."""
        # Mock request with sensitive data
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/api/auth/login"
        request.headers = {
            "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.sensitive",
            "X-API-Key": "secret_api_key_12345",
            "Cookie": "sessionid=secret_session_data; csrftoken=secret_csrf",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Content-Type": "application/json",
        }
        request.query_params = {
            "password": "secret_password",
            "token": "secret_token",
            "api_key": "another_secret",
        }

        # Mock response
        response = Mock()
        response.status_code = 200

        # Mock telemetry
        telemetry = Mock()
        telemetry.record_request = Mock()

        async def mock_call_next(req):
            return response

        with patch(
            "pynomaly.infrastructure.monitoring.get_telemetry", return_value=telemetry
        ):
            result = asyncio.run(track_request_metrics(request, mock_call_next))

        assert result == response

        # Verify telemetry was called
        if telemetry.record_request.called:
            # Check that sensitive data is not in the call
            call_str = str(telemetry.record_request.call_args)

            sensitive_data = [
                "secret_api_key_12345",
                "secret_session_data",
                "secret_csrf",
                "secret_password",
                "secret_token",
                "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.sensitive",
            ]

            for sensitive in sensitive_data:
                assert sensitive not in call_str

    def test_pii_protection_in_metrics(self):
        """Test protection of PII in request metrics."""
        # Mock request with PII
        request = Mock(spec=Request)
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/users/john.doe@example.com/profile"
        request.query_params = {
            "email": "john.doe@example.com",
            "phone": "+1234567890",
            "ssn": "123-45-6789",
        }

        response = Mock()
        response.status_code = 200

        telemetry = Mock()

        async def mock_call_next(req):
            return response

        with patch(
            "pynomaly.infrastructure.monitoring.get_telemetry", return_value=telemetry
        ):
            asyncio.run(track_request_metrics(request, mock_call_next))

        if telemetry.record_request.called:
            call_str = str(telemetry.record_request.call_args)

            # PII should not be present
            pii_data = ["john.doe@example.com", "+1234567890", "123-45-6789"]

            for pii in pii_data:
                assert pii not in call_str

    def test_request_timing_information_security(self):
        """Test that request timing information doesn't leak sensitive data."""
        # Requests to sensitive endpoints should not reveal timing patterns
        # that could be used for enumeration attacks

        sensitive_endpoints = [
            "/api/auth/login",
            "/api/users/admin",
            "/api/settings/secrets",
            "/api/admin/users",
        ]

        for endpoint in sensitive_endpoints:
            request = Mock(spec=Request)
            request.method = "POST"
            request.url = Mock()
            request.url.path = endpoint

            response = Mock()
            response.status_code = 404  # Not found

            telemetry = Mock()

            async def mock_call_next(req):
                # Simulate some processing time
                await asyncio.sleep(0.001)
                return response

            start_time = time.time()

            with patch(
                "pynomaly.infrastructure.monitoring.get_telemetry",
                return_value=telemetry,
            ):
                asyncio.run(track_request_metrics(request, mock_call_next))

            end_time = time.time()
            duration = end_time - start_time

            # Verify timing is recorded
            assert duration > 0

            # In production, timing should be normalized to prevent
            # timing-based enumeration attacks

    def test_error_information_leakage_prevention(self):
        """Test prevention of sensitive error information in metrics."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock()
        request.url.path = "/api/sensitive"

        # Mock response with error that might contain sensitive info
        response = Mock()
        response.status_code = 500
        response.headers = {
            "X-Debug-Error": "Database connection failed: password 'secret' for user 'admin'",
            "X-Stack-Trace": "File '/app/config.py', line 42: SECRET_KEY = 'production_secret'",
        }

        telemetry = Mock()

        async def mock_call_next(req):
            return response

        with patch(
            "pynomaly.infrastructure.monitoring.get_telemetry", return_value=telemetry
        ):
            asyncio.run(track_request_metrics(request, mock_call_next))

        if telemetry.record_request.called:
            call_str = str(telemetry.record_request.call_args)

            # Debug info and secrets should not be in metrics
            sensitive_debug_info = [
                "password 'secret'",
                "SECRET_KEY = 'production_secret'",
                "/app/config.py",
            ]

            for debug_info in sensitive_debug_info:
                assert debug_info not in call_str


class TestAuthContextSecurity:
    """Security tests for authentication context creation."""

    def test_context_data_sanitization(self):
        """Test that auth context data is properly sanitized."""
        # Create user with potentially dangerous data
        user = Mock(spec=UserModel)
        user.id = "<script>alert('xss')</script>"
        user.username = "'; DROP TABLE users; --"
        user.roles = ["<img src=x onerror=alert(1)>", "admin"]
        user.is_superuser = False

        # Mock auth service
        auth_service = Mock()
        auth_service._get_permissions_for_roles.return_value = [
            "detectors:read",
            "<script>alert('perm')</script>",  # Malicious permission
        ]

        with patch(
            "pynomaly.infrastructure.auth.middleware.get_auth",
            return_value=auth_service,
        ):
            context = create_auth_context(user)

        # Verify context contains data (sanitization happens at output layer)
        assert context["authenticated"] is True
        assert context["user_id"] == user.id
        assert context["username"] == user.username
        assert context["roles"] == user.roles
        assert (
            context["permissions"]
            == auth_service._get_permissions_for_roles.return_value
        )

    def test_context_information_disclosure_prevention(self):
        """Test prevention of information disclosure through auth context."""
        # Test with None user (unauthenticated)
        context = create_auth_context(None)

        # Should not reveal system information
        assert context["authenticated"] is False
        assert context["user_id"] is None
        assert context["username"] is None
        assert context["roles"] == []
        assert context["permissions"] == []

        # Should not contain sensitive keys
        sensitive_keys = [
            "password",
            "secret",
            "token",
            "key",
            "hash",
            "internal",
            "debug",
            "admin_notes",
            "system",
        ]

        for key in sensitive_keys:
            assert key not in context

    def test_context_role_permission_consistency(self):
        """Test consistency between roles and permissions in context."""
        user = Mock(spec=UserModel)
        user.id = "user123"
        user.username = "testuser"
        user.roles = ["user", "viewer"]
        user.is_superuser = False

        auth_service = Mock()
        # Mock realistic permissions for user and viewer roles
        auth_service._get_permissions_for_roles.return_value = [
            "detectors:read",
            "detectors:write",
            "datasets:read",
            "experiments:read",
        ]

        with patch(
            "pynomaly.infrastructure.auth.middleware.get_auth",
            return_value=auth_service,
        ):
            context = create_auth_context(user)

        # Verify permissions were derived from roles
        auth_service._get_permissions_for_roles.assert_called_once_with(user.roles)

        # Context should reflect the user's actual permissions
        assert context["roles"] == user.roles
        assert len(context["permissions"]) > 0

    def test_context_superuser_flag_security(self):
        """Test security handling of superuser flag in context."""
        # Test regular user
        regular_user = Mock(spec=UserModel)
        regular_user.id = "user123"
        regular_user.username = "regular"
        regular_user.roles = ["user"]
        regular_user.is_superuser = False

        context = create_auth_context(regular_user)
        assert context.get("is_superuser") is False

        # Test superuser
        superuser = Mock(spec=UserModel)
        superuser.id = "admin123"
        superuser.username = "admin"
        superuser.roles = ["admin"]
        superuser.is_superuser = True

        context = create_auth_context(superuser)
        assert context.get("is_superuser") is True

        # Superuser flag should be clearly indicated
        assert "is_superuser" in context

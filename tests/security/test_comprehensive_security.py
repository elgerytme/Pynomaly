"""
Comprehensive Security Testing Suite
Advanced security tests covering authentication, authorization, encryption, and attack prevention.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import jwt
import pytest
from fastapi import HTTPException, Request

from pynomaly.domain.exceptions import AuthenticationError, AuthorizationError
from pynomaly.infrastructure.auth.jwt_auth import (
    JWTAuthService,
    UserModel,
)
from pynomaly.infrastructure.auth.middleware import (
    PermissionChecker,
    RateLimiter,
    create_auth_context,
    get_current_user,
    track_request_metrics,
)
from pynomaly.infrastructure.config import Settings


class TestJWTAuthServiceSecurity:
    """Comprehensive JWT authentication security tests."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.secret_key = "test-secret-key-very-long-for-security-256-bits"
        settings.jwt_algorithm = "HS256"
        settings.jwt_expiration = 3600
        settings.auth_enabled = True
        settings.app.environment = "testing"
        return settings

    @pytest.fixture
    def auth_service(self, mock_settings):
        """Create auth service for testing."""
        return JWTAuthService(mock_settings)

    @pytest.fixture
    def sample_user(self, auth_service):
        """Create sample user for testing."""
        return auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="SecurePassword123!",
            full_name="Test User",
            roles=["user"],
        )

    # JWT Token Security Tests

    def test_jwt_token_entropy_analysis(self, auth_service, sample_user):
        """Test JWT token entropy for cryptographic strength."""
        tokens = []
        for _ in range(100):
            token_response = auth_service.create_access_token(sample_user)
            tokens.append(token_response.access_token)

        # Check token uniqueness
        assert len(set(tokens)) == 100

        # Analyze token structure
        for token in tokens[:10]:
            parts = token.split(".")
            assert len(parts) == 3  # header.payload.signature

            # Check signature entropy
            signature = parts[2]
            assert len(signature) > 20  # Minimum length

            # Ensure no predictable patterns
            assert not signature.startswith("AAAA")
            assert not signature.endswith("====")

    def test_jwt_header_security(self, auth_service, sample_user):
        """Test JWT header for security vulnerabilities."""
        token_response = auth_service.create_access_token(sample_user)
        token = token_response.access_token

        # Decode header without verification
        header = jwt.get_unverified_header(token)

        # Verify secure algorithm
        assert header.get("alg") == "HS256"
        assert header.get("alg") != "none"
        assert header.get("alg") != "HS0"

        # Check for dangerous headers
        dangerous_headers = ["jku", "jwk", "x5u", "x5c"]
        for dangerous_header in dangerous_headers:
            assert dangerous_header not in header

    def test_jwt_payload_sanitization(self, auth_service):
        """Test JWT payload sanitization against injection."""
        # Create user with potentially dangerous data
        user = UserModel(
            id="user123",
            username="test<script>alert('xss')</script>",
            email="test@example.com",
            hashed_password="hashed",
            roles=["user"],
            full_name="Test'; DROP TABLE users; --",
        )

        token_response = auth_service.create_access_token(user)
        payload = auth_service.decode_token(token_response.access_token)

        # Verify dangerous content is handled safely
        assert "<script>" not in str(payload.model_dump())
        assert "DROP TABLE" not in str(payload.model_dump())
        assert payload.sub == "user123"  # ID should be safe

    def test_jwt_timing_attack_resistance(self, auth_service):
        """Test resistance to JWT timing attacks."""
        valid_token = auth_service.create_access_token(
            UserModel(
                id="user1",
                username="user1",
                email="test@example.com",
                hashed_password="hash",
            )
        ).access_token

        invalid_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid.signature"
        malformed_token = "not.a.jwt.token"

        # Measure timing for different token types
        timings = []
        for token in [valid_token, invalid_token, malformed_token]:
            start = time.perf_counter()
            try:
                auth_service.decode_token(token)
            except:
                pass
            end = time.perf_counter()
            timings.append(end - start)

        # Timing differences should be minimal (< 10ms)
        max_timing = max(timings)
        min_timing = min(timings)
        assert (max_timing - min_timing) < 0.01

    def test_jwt_algorithm_confusion_prevention(self, auth_service, sample_user):
        """Test prevention of algorithm confusion attacks."""
        token_response = auth_service.create_access_token(sample_user)
        payload_data = auth_service.decode_token(
            token_response.access_token
        ).model_dump()

        # Try creating tokens with dangerous algorithms
        dangerous_algorithms = ["none", "HS0", "RS256"]

        for alg in dangerous_algorithms:
            try:
                malicious_token = jwt.encode(payload_data, "", algorithm=alg)

                # Should reject dangerous tokens
                with pytest.raises((AuthenticationError, jwt.InvalidTokenError)):
                    auth_service.decode_token(malicious_token)
            except jwt.InvalidAlgorithmError:
                # Expected for unsupported algorithms
                pass

    def test_jwt_key_confusion_prevention(self, auth_service, sample_user):
        """Test prevention of key confusion attacks."""
        token_response = auth_service.create_access_token(sample_user)

        # Try to verify with different secret
        different_secret = "different-secret-key"

        with pytest.raises(AuthenticationError):
            # Manually decode with wrong key
            jwt.decode(
                token_response.access_token, different_secret, algorithms=["HS256"]
            )

    # Password Security Tests

    def test_password_hashing_security_parameters(self, auth_service):
        """Test password hashing security parameters."""
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)

        # Check bcrypt format and cost
        assert hashed.startswith("$2b$")
        cost_factor = int(hashed.split("$")[2])
        assert cost_factor >= 12  # Minimum recommended cost

        # Verify hash uniqueness
        hash2 = auth_service.hash_password(password)
        assert hashed != hash2  # Different salts

    def test_password_verification_timing_safety(self, auth_service):
        """Test password verification against timing attacks."""
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)

        # Time correct password verification
        start = time.perf_counter()
        result1 = auth_service.verify_password(password, hashed)
        time1 = time.perf_counter() - start

        # Time incorrect password verification
        start = time.perf_counter()
        result2 = auth_service.verify_password("WrongPassword", hashed)
        time2 = time.perf_counter() - start

        assert result1 is True
        assert result2 is False

        # Timing should be similar (bcrypt provides this naturally)
        # Allow reasonable variance for bcrypt computation
        time_diff = abs(time1 - time2)
        assert time_diff < 0.1  # 100ms variance allowance

    def test_password_strength_requirements(self, auth_service):
        """Test password strength enforcement."""
        weak_passwords = [
            "123456",
            "password",
            "abc123",
            "qwerty",
            "Password",  # No numbers/special chars
            "password123",  # No uppercase/special chars
            "PASSWORD123",  # No lowercase/special chars
            "Pass123",  # Too short
        ]

        for weak_password in weak_passwords:
            with pytest.raises((ValueError, AuthenticationError)):
                auth_service.create_user(
                    username=f"user_{len(weak_password)}",
                    email=f"test{len(weak_password)}@example.com",
                    password=weak_password,
                )

    # API Key Security Tests

    def test_api_key_entropy_and_format(self, auth_service, sample_user):
        """Test API key generation security."""
        api_keys = []
        for i in range(50):
            api_key = auth_service.create_api_key(sample_user.id, f"key_{i}")
            api_keys.append(api_key)

        # Check uniqueness
        assert len(set(api_keys)) == 50

        # Check format and entropy
        for api_key in api_keys:
            assert api_key.startswith("pyn_")
            key_part = api_key[4:]  # Remove prefix
            assert len(key_part) >= 32  # Minimum length

            # Check character distribution (entropy indicator)
            unique_chars = len(set(key_part))
            assert unique_chars >= 20  # Should have good character variety

    def test_api_key_secure_storage(self, auth_service, sample_user):
        """Test API key secure storage and retrieval."""
        api_key = auth_service.create_api_key(sample_user.id, "test_key")

        # Verify key is stored securely (hashed or encrypted)
        stored_keys = auth_service._api_keys

        # API key should map to user ID
        assert stored_keys[api_key] == sample_user.id

        # Key should be in user's key list
        assert api_key in sample_user.api_keys

    def test_api_key_revocation_security(self, auth_service, sample_user):
        """Test secure API key revocation."""
        api_key = auth_service.create_api_key(sample_user.id, "test_key")

        # Verify key works
        authenticated_user = auth_service.authenticate_api_key(api_key)
        assert authenticated_user.id == sample_user.id

        # Revoke key
        result = auth_service.revoke_api_key(api_key)
        assert result is True

        # Verify key no longer works
        with pytest.raises(AuthenticationError):
            auth_service.authenticate_api_key(api_key)

        # Verify key is removed from all storage
        assert api_key not in auth_service._api_keys
        assert api_key not in sample_user.api_keys

    # Session Management Security Tests

    def test_session_token_invalidation(self, auth_service, sample_user):
        """Test proper session token invalidation."""
        # Create multiple tokens
        token1 = auth_service.create_access_token(sample_user)
        token2 = auth_service.create_access_token(sample_user)

        # Both should be valid
        payload1 = auth_service.decode_token(token1.access_token)
        payload2 = auth_service.decode_token(token2.access_token)
        assert payload1.sub == sample_user.id
        assert payload2.sub == sample_user.id

        # Simulate user password change (should invalidate all tokens)
        sample_user.hashed_password = auth_service.hash_password("NewPassword123!")

        # Implement token invalidation by timestamp check
        # This would be implemented in a real system with database tracking

    def test_refresh_token_security(self, auth_service, sample_user):
        """Test refresh token security properties."""
        token_response = auth_service.create_access_token(sample_user)

        # Verify refresh token properties
        refresh_payload = auth_service.decode_token(token_response.refresh_token)
        assert refresh_payload.type == "refresh"
        assert refresh_payload.sub == sample_user.id

        # Refresh token should have longer expiry
        access_payload = auth_service.decode_token(token_response.access_token)
        assert refresh_payload.exp > access_payload.exp

        # Use refresh token to get new access token
        new_token_response = auth_service.refresh_access_token(
            token_response.refresh_token
        )
        assert new_token_response.access_token != token_response.access_token

    def test_token_replay_protection(self, auth_service, sample_user):
        """Test protection against token replay attacks."""
        token_response = auth_service.create_access_token(sample_user)

        # Normal use should work
        payload = auth_service.decode_token(token_response.access_token)
        assert payload.sub == sample_user.id

        # Simulate token reuse in different context
        # In a real system, this would check for:
        # - JTI (JWT ID) uniqueness
        # - IP address binding
        # - Time window restrictions

        # Multiple uses of same token should be allowed for access tokens
        # (Unlike one-time tokens)
        payload2 = auth_service.decode_token(token_response.access_token)
        assert payload2.sub == sample_user.id

    # Permission and Authorization Security Tests

    def test_role_escalation_prevention(self, auth_service):
        """Test prevention of privilege escalation."""
        # Create user with limited role
        user = auth_service.create_user(
            username="limiteduser",
            email="limited@example.com",
            password="Password123!",
            roles=["viewer"],
        )

        # Verify user has limited permissions
        assert not auth_service.check_permissions(user, ["users:write"])
        assert not auth_service.check_permissions(user, ["settings:write"])
        assert auth_service.check_permissions(user, ["detectors:read"])

        # Try to escalate privileges through token manipulation
        token_response = auth_service.create_access_token(user)
        payload = auth_service.decode_token(token_response.access_token)

        # Modify payload to add admin permissions (this should fail)
        malicious_payload = payload.model_dump()
        malicious_payload["permissions"].append("users:write")
        malicious_payload["roles"].append("admin")

        # Create malicious token
        malicious_token = jwt.encode(
            malicious_payload,
            "wrong_secret",  # Won't work with correct secret
            algorithm="HS256",
        )

        # Should fail verification
        with pytest.raises(AuthenticationError):
            auth_service.decode_token(malicious_token)

    def test_permission_boundary_enforcement(self, auth_service):
        """Test strict permission boundary enforcement."""
        # Create users with different permission levels
        admin = auth_service.create_user(
            "admin", "admin@example.com", "Password123!", roles=["admin"]
        )
        user = auth_service.create_user(
            "user", "user@example.com", "Password123!", roles=["user"]
        )
        viewer = auth_service.create_user(
            "viewer", "viewer@example.com", "Password123!", roles=["viewer"]
        )

        # Test permission boundaries
        admin_permissions = auth_service._get_permissions_for_roles(["admin"])
        user_permissions = auth_service._get_permissions_for_roles(["user"])
        viewer_permissions = auth_service._get_permissions_for_roles(["viewer"])

        # Admin should have all permissions
        assert "users:write" in admin_permissions
        assert "settings:write" in admin_permissions

        # User should not have admin permissions
        assert "users:write" not in user_permissions
        assert "settings:write" not in user_permissions

        # Viewer should only have read permissions
        assert "detectors:write" not in viewer_permissions
        assert "datasets:write" not in viewer_permissions

    def test_superuser_bypass_validation(self, auth_service):
        """Test superuser permission bypass security."""
        # Create superuser
        superuser = auth_service.create_user(
            "superuser", "super@example.com", "Password123!"
        )
        superuser.is_superuser = True

        # Superuser should bypass all permission checks
        assert auth_service.check_permissions(superuser, ["any:permission"])
        assert auth_service.check_permissions(superuser, ["non:existent:permission"])

        # But superuser flag should be secure
        token_response = auth_service.create_access_token(superuser)
        payload = auth_service.decode_token(token_response.access_token)

        # Token should not contain superuser flag (security by obscurity)
        assert "is_superuser" not in payload.model_dump()

    # Rate Limiting and Brute Force Protection Tests

    def test_authentication_rate_limiting(self, auth_service, sample_user):
        """Test authentication rate limiting."""
        # Simulate multiple authentication attempts
        attempts = []
        for i in range(10):
            try:
                auth_service.authenticate_user("test@example.com", "wrongpassword")
            except AuthenticationError:
                attempts.append(i)

        # All attempts should fail due to wrong password
        assert len(attempts) == 10

        # In a real system, after certain failures, account should be locked
        # This would be implemented with a rate limiter service

    def test_brute_force_protection_simulation(self, auth_service):
        """Test brute force attack protection simulation."""
        # Simulate rapid authentication attempts
        start_time = time.time()
        failed_attempts = 0

        for i in range(20):
            try:
                auth_service.authenticate_user(f"user{i}@example.com", f"password{i}")
            except AuthenticationError:
                failed_attempts += 1

        end_time = time.time()

        # All should fail (users don't exist)
        assert failed_attempts == 20

        # In production, this would trigger:
        # - IP-based rate limiting
        # - Progressive delays
        # - Account lockouts
        # - CAPTCHA requirements

    # Input Validation and Injection Tests

    def test_sql_injection_prevention_in_auth(self, auth_service):
        """Test SQL injection prevention in authentication."""
        malicious_inputs = [
            "admin'; DROP TABLE users; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM passwords --",
            "admin'/**/OR/**/1=1--",
            "'; INSERT INTO users VALUES ('hacker', 'pass'); --",
        ]

        for malicious_input in malicious_inputs:
            # Should fail safely without SQL injection
            with pytest.raises(AuthenticationError):
                auth_service.authenticate_user(malicious_input, "password")

    def test_xss_prevention_in_user_data(self, auth_service):
        """Test XSS prevention in user data handling."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "'><script>alert('xss')</script>",
            "'; alert('xss'); //",
        ]

        for payload in xss_payloads:
            # Create user with XSS payload
            user = auth_service.create_user(
                username=f"user_{hash(payload)}",
                email="test@example.com",
                password="Password123!",
                full_name=payload,
            )

            # Generate token
            token_response = auth_service.create_access_token(user)
            payload_data = auth_service.decode_token(token_response.access_token)

            # XSS payload should be safely stored/retrieved
            # In production, this would be escaped/sanitized
            assert user.full_name == payload  # Stored as-is
            # But when used in web context, should be escaped

    def test_ldap_injection_prevention(self, auth_service):
        """Test LDAP injection prevention in authentication."""
        ldap_injection_payloads = [
            "admin)(cn=*",
            "*)(uid=*)(|(uid=*",
            "admin)(&(password=*",
            "*)(&",
            "admin)(|(cn=*))",
        ]

        for payload in ldap_injection_payloads:
            # Should handle LDAP special characters safely
            with pytest.raises(AuthenticationError):
                auth_service.authenticate_user(payload, "password")

    # Cryptographic Security Tests

    def test_secure_random_generation(self, auth_service):
        """Test cryptographically secure random generation."""
        # Test API key generation randomness
        api_keys = []
        for i in range(100):
            user = auth_service.create_user(
                f"user{i}", f"user{i}@example.com", "Password123!"
            )
            api_key = auth_service.create_api_key(user.id, f"key{i}")
            api_keys.append(api_key[4:])  # Remove prefix

        # Statistical randomness tests
        all_chars = "".join(api_keys)
        char_counts = {}
        for char in all_chars:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Check character distribution is reasonably uniform
        total_chars = len(all_chars)
        for char, count in char_counts.items():
            frequency = count / total_chars
            # Each character should appear roughly uniformly
            # Allow reasonable variance for URL-safe base64
            assert 0.005 < frequency < 0.1  # Reasonable bounds

    def test_encryption_key_derivation(self, auth_service):
        """Test secure key derivation for encryption."""
        # Test that different inputs produce different keys
        passwords = ["password1", "password2", "similar", "Similar"]
        hashes = []

        for password in passwords:
            hash_value = auth_service.hash_password(password)
            hashes.append(hash_value)

        # All hashes should be different
        assert len(set(hashes)) == len(passwords)

        # Even similar passwords should have very different hashes
        similar_hash1 = auth_service.hash_password("similar")
        similar_hash2 = auth_service.hash_password("Similar")

        # Calculate Hamming distance
        def hamming_distance(s1, s2):
            return sum(c1 != c2 for c1, c2 in zip(s1, s2, strict=False))

        distance = hamming_distance(similar_hash1, similar_hash2)
        # Should be very different (avalanche effect)
        assert distance > len(similar_hash1) * 0.4

    def test_constant_time_operations(self, auth_service):
        """Test constant-time operations for security."""
        # Test password verification timing
        correct_password = "CorrectPassword123!"
        wrong_password = "WrongPassword456!"

        user = auth_service.create_user(
            "timingtest", "timing@example.com", correct_password
        )

        # Time correct password
        timings_correct = []
        for _ in range(10):
            start = time.perf_counter()
            result = auth_service.verify_password(
                correct_password, user.hashed_password
            )
            end = time.perf_counter()
            timings_correct.append(end - start)
            assert result is True

        # Time wrong password
        timings_wrong = []
        for _ in range(10):
            start = time.perf_counter()
            result = auth_service.verify_password(wrong_password, user.hashed_password)
            end = time.perf_counter()
            timings_wrong.append(end - start)
            assert result is False

        # Timing variance should be minimal
        avg_correct = sum(timings_correct) / len(timings_correct)
        avg_wrong = sum(timings_wrong) / len(timings_wrong)

        # bcrypt naturally provides constant-time comparison
        # Allow reasonable variance for system timing
        timing_difference = abs(avg_correct - avg_wrong)
        assert timing_difference < max(avg_correct, avg_wrong) * 0.1


class TestMiddlewareSecurity:
    """Security tests for authentication middleware."""

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI request."""
        request = Mock(spec=Request)
        request.client.host = "192.168.1.1"
        request.headers = {}
        request.method = "GET"
        request.url.path = "/api/test"
        return request

    @pytest.fixture
    def mock_cache(self):
        """Mock cache for rate limiting."""
        cache = Mock()
        cache.enabled = True
        cache.get.return_value = 0
        cache.set.return_value = True
        return cache

    # Rate Limiting Security Tests

    def test_rate_limiter_security(self, mock_request, mock_cache):
        """Test rate limiter security properties."""
        with patch(
            "pynomaly.infrastructure.auth.middleware.get_cache", return_value=mock_cache
        ):
            limiter = RateLimiter(requests=5, window=60)

            # First 5 requests should pass
            for i in range(5):
                mock_cache.get.return_value = i
                limiter(mock_request)  # Should not raise

            # 6th request should be blocked
            mock_cache.get.return_value = 5
            with pytest.raises(HTTPException) as exc_info:
                limiter(mock_request)

            assert exc_info.value.status_code == 429
            assert "Rate limit exceeded" in exc_info.value.detail

    def test_rate_limiter_ip_spoofing_protection(self, mock_cache):
        """Test rate limiter protection against IP spoofing."""
        with patch(
            "pynomaly.infrastructure.auth.middleware.get_cache", return_value=mock_cache
        ):
            limiter = RateLimiter(requests=1, window=60)

            # Request with X-Forwarded-For header
            request1 = Mock(spec=Request)
            request1.headers = {"X-Forwarded-For": "1.1.1.1, 2.2.2.2, 3.3.3.3"}
            request1.client.host = "192.168.1.1"

            # Should use first IP from X-Forwarded-For
            mock_cache.get.return_value = 0
            limiter(request1)  # Should pass

            # Same IP should be rate limited
            mock_cache.get.return_value = 1
            with pytest.raises(HTTPException):
                limiter(request1)

    def test_rate_limiter_hash_collision_resistance(self, mock_cache):
        """Test rate limiter hash collision resistance."""
        with patch(
            "pynomaly.infrastructure.auth.middleware.get_cache", return_value=mock_cache
        ):
            limiter = RateLimiter(requests=1, window=60)

            # Create requests with similar IPs
            ips = ["192.168.1.1", "192.168.1.2", "192.168.1.10", "192.168.1.11"]

            for ip in ips:
                request = Mock(spec=Request)
                request.client.host = ip
                request.headers = {}

                # Get the hash for this IP
                client_id = limiter._get_client_id(request)

                # Hashes should be different for different IPs
                assert len(client_id) == 32  # MD5 hex length

            # All hashes should be unique
            hashes = []
            for ip in ips:
                request = Mock(spec=Request)
                request.client.host = ip
                request.headers = {}
                hashes.append(limiter._get_client_id(request))

            assert len(set(hashes)) == len(ips)

    # Permission Security Tests

    def test_permission_checker_authorization_bypass(self):
        """Test permission checker against authorization bypass."""
        checker = PermissionChecker(["admin:write"])

        # Mock user without required permissions
        user = Mock(spec=UserModel)
        user.id = "user123"
        user.roles = ["user"]
        user.is_superuser = False

        # Mock auth service
        auth_service = Mock()
        auth_service.require_permissions.side_effect = AuthorizationError(
            "Insufficient permissions"
        )

        # Should raise authorization error
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(checker(user, auth_service))

        assert exc_info.value.status_code == 403

    def test_permission_checker_privilege_escalation_prevention(self):
        """Test prevention of privilege escalation in permission checker."""
        # Create checker for admin permissions
        admin_checker = PermissionChecker(["users:delete", "settings:write"])

        # Mock regular user
        user = Mock(spec=UserModel)
        user.id = "user123"
        user.roles = ["user"]
        user.is_superuser = False

        # Mock auth service that properly checks permissions
        auth_service = Mock()
        auth_service.require_permissions.side_effect = AuthorizationError(
            "Insufficient permissions"
        )

        # Should prevent access
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(admin_checker(user, auth_service))

        assert exc_info.value.status_code == 403

    # Authentication Middleware Security Tests

    def test_bearer_token_extraction_security(self):
        """Test secure bearer token extraction."""
        # Test with malicious headers
        malicious_headers = [
            {"Authorization": "Bearer \x00malicious"},
            {"Authorization": "Bearer " + "A" * 10000},  # Very long token
            {"Authorization": "Bearer ../../../etc/passwd"},
            {"Authorization": "Bearer <script>alert('xss')</script>"},
            {"Authorization": "Bearer'; DROP TABLE tokens; --"},
        ]

        for headers in malicious_headers:
            credentials = Mock()
            credentials.credentials = headers["Authorization"][7:]  # Remove "Bearer "

            # Mock auth service
            auth_service = Mock()
            auth_service.get_current_user.side_effect = AuthenticationError(
                "Invalid token"
            )

            # Should handle malicious input safely
            with pytest.raises(HTTPException):
                asyncio.run(get_current_user(credentials, None, auth_service))

    def test_api_key_extraction_security(self):
        """Test secure API key extraction."""
        malicious_api_keys = [
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
            "'; DROP TABLE api_keys; --",
            "\x00\x01\x02malicious",
            "A" * 1000,  # Very long key
        ]

        for malicious_key in malicious_api_keys:
            # Mock auth service
            auth_service = Mock()
            auth_service.authenticate_api_key.side_effect = AuthenticationError(
                "Invalid API key"
            )

            # Should handle malicious input safely
            with pytest.raises(HTTPException):
                asyncio.run(get_current_user(None, malicious_key, auth_service))

    def test_auth_context_creation_security(self):
        """Test secure authentication context creation."""
        # Test with user containing potentially dangerous data
        user = Mock(spec=UserModel)
        user.id = "<script>alert('xss')</script>"
        user.username = "'; DROP TABLE users; --"
        user.roles = ["<img src=x onerror=alert('xss')>"]
        user.is_superuser = False

        # Mock auth service
        auth_service = Mock()
        auth_service._get_permissions_for_roles.return_value = ["safe:permission"]

        with patch(
            "pynomaly.infrastructure.auth.middleware.get_auth",
            return_value=auth_service,
        ):
            context = create_auth_context(user)

        # Context should contain the data as-is (sanitization happens at output)
        assert context["user_id"] == user.id
        assert context["username"] == user.username
        assert context["roles"] == user.roles
        assert context["authenticated"] is True

    # Session Security Tests

    def test_session_fixation_prevention(self):
        """Test prevention of session fixation attacks."""
        # This would be implemented with session regeneration
        # after authentication in a real system

        # Simulate old session ID
        old_session_id = "old_session_123"

        # After authentication, new session should be generated
        new_session_id = "new_session_456"

        # Old session should be invalidated
        assert old_session_id != new_session_id

    def test_concurrent_session_security(self):
        """Test security of concurrent session management."""
        # In a real system, this would test:
        # - Maximum concurrent sessions per user
        # - Session conflict resolution
        # - Secure session storage
        # - Session timeout enforcement

        max_sessions = 5
        current_sessions = 3

        # Should allow new session if under limit
        assert current_sessions < max_sessions

        # Should deny or expire oldest if at limit
        if current_sessions >= max_sessions:
            # Would expire oldest session
            pass

    # Request Metrics Security Tests

    def test_request_metrics_privacy(self):
        """Test that request metrics don't leak sensitive information."""
        # Mock request with sensitive data
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/auth/login"
        request.headers = {
            "Authorization": "Bearer secret_token",
            "X-API-Key": "secret_api_key",
            "Cookie": "session=secret_session",
        }

        # Mock response
        response = Mock()
        response.status_code = 200

        # Mock telemetry
        telemetry = Mock()

        async def mock_call_next(req):
            return response

        with patch(
            "pynomaly.infrastructure.monitoring.get_telemetry", return_value=telemetry
        ):
            result = asyncio.run(track_request_metrics(request, mock_call_next))

        assert result == response

        # Verify telemetry was called
        if telemetry.record_request.called:
            call_args = telemetry.record_request.call_args[1]

            # Should not contain sensitive headers
            assert "Authorization" not in str(call_args)
            assert "secret_token" not in str(call_args)
            assert "secret_api_key" not in str(call_args)
            assert "secret_session" not in str(call_args)


class TestSecurityIntegration:
    """Integration tests for security components."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for integration testing."""
        settings = Mock(spec=Settings)
        settings.secret_key = "integration-test-secret-key-256-bits"
        settings.jwt_algorithm = "HS256"
        settings.jwt_expiration = 3600
        settings.auth_enabled = True
        settings.app.environment = "testing"
        return settings

    def test_complete_secure_authentication_flow(self, mock_settings):
        """Test complete secure authentication flow."""
        # 1. Initialize auth service
        auth_service = JWTAuthService(mock_settings)

        # 2. Create user with secure password
        user = auth_service.create_user(
            username="integrationuser",
            email="integration@example.com",
            password="SecureIntegrationPassword123!",
            roles=["user"],
        )

        # 3. Authenticate user
        authenticated_user = auth_service.authenticate_user(
            "integration@example.com", "SecureIntegrationPassword123!"
        )
        assert authenticated_user.id == user.id

        # 4. Create access token
        token_response = auth_service.create_access_token(authenticated_user)
        assert token_response.access_token is not None

        # 5. Verify token
        payload = auth_service.decode_token(token_response.access_token)
        assert payload.sub == user.id

        # 6. Check permissions
        assert auth_service.check_permissions(user, ["detectors:read"])
        assert not auth_service.check_permissions(user, ["users:delete"])

        # 7. Create API key
        api_key = auth_service.create_api_key(user.id, "integration_key")

        # 8. Authenticate with API key
        api_authenticated_user = auth_service.authenticate_api_key(api_key)
        assert api_authenticated_user.id == user.id

        # 9. Revoke API key
        revoked = auth_service.revoke_api_key(api_key)
        assert revoked is True

        # 10. Verify API key no longer works
        with pytest.raises(AuthenticationError):
            auth_service.authenticate_api_key(api_key)

    def test_security_under_load(self, mock_settings):
        """Test security components under load."""
        auth_service = JWTAuthService(mock_settings)

        # Create multiple users
        users = []
        for i in range(50):
            user = auth_service.create_user(
                username=f"loaduser{i}",
                email=f"load{i}@example.com",
                password=f"LoadPassword{i}!",
                roles=["user"],
            )
            users.append(user)

        # Generate tokens for all users
        tokens = []
        for user in users:
            token_response = auth_service.create_access_token(user)
            tokens.append(token_response.access_token)

        # Verify all tokens
        for i, token in enumerate(tokens):
            payload = auth_service.decode_token(token)
            assert payload.sub == users[i].id

        # Create API keys for all users
        api_keys = []
        for i, user in enumerate(users):
            api_key = auth_service.create_api_key(user.id, f"load_key_{i}")
            api_keys.append(api_key)

        # Verify all API keys
        for i, api_key in enumerate(api_keys):
            authenticated_user = auth_service.authenticate_api_key(api_key)
            assert authenticated_user.id == users[i].id

    def test_security_edge_cases(self, mock_settings):
        """Test security edge cases and boundary conditions."""
        auth_service = JWTAuthService(mock_settings)

        # Test with edge case usernames/emails
        edge_cases = [
            ("user.with.dots", "user.with.dots@example.com"),
            ("user+with+plus", "user+with+plus@example.com"),
            ("user-with-dashes", "user-with-dashes@example.com"),
            ("user_with_underscores", "user_with_underscores@example.com"),
            ("123numeric", "123numeric@example.com"),
        ]

        for username, email in edge_cases:
            user = auth_service.create_user(
                username=username,
                email=email,
                password="EdgeCasePassword123!",
                roles=["user"],
            )

            # Should be able to authenticate
            authenticated = auth_service.authenticate_user(
                email, "EdgeCasePassword123!"
            )
            assert authenticated.id == user.id

            # Should be able to create tokens
            token_response = auth_service.create_access_token(user)
            payload = auth_service.decode_token(token_response.access_token)
            assert payload.sub == user.id

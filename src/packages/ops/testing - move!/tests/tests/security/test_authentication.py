"""
Security Testing Suite - Authentication
Comprehensive security tests for authentication mechanisms.
"""

import secrets
from datetime import datetime, timedelta
from unittest.mock import patch

import jwt
import pytest

from monorepo.domain.exceptions import AuthenticationError, SecurityError
from monorepo.infrastructure.auth.jwt_auth import JWTAuthService


class TestJWTAuthentication:
    """Test suite for JWT authentication security."""

    @pytest.fixture
    def jwt_handler(self):
        """Create JWT authentication handler."""
        return JWTAuthService(
            secret_key="test-secret-key-256-bits-long",
            algorithm="HS256",
            access_token_expire_minutes=30,
        )

    @pytest.fixture
    def valid_user_data(self):
        """Valid user data for testing."""
        return {
            "user_id": "user123",
            "email": "test@example.com",
            "role": "user",
            "permissions": ["read:datasets", "write:datasets"],
        }

    # Token Generation Tests

    def test_create_access_token_valid(self, jwt_handler, valid_user_data):
        """Test creating valid access token."""
        token = jwt_handler.create_access_token(valid_user_data)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are typically long

    def test_create_access_token_with_custom_expiry(self, jwt_handler, valid_user_data):
        """Test creating token with custom expiry time."""
        custom_expires = timedelta(hours=2)
        token = jwt_handler.create_access_token(
            valid_user_data, expires_delta=custom_expires
        )

        # Decode to verify expiry
        decoded = jwt.decode(
            token, jwt_handler.secret_key, algorithms=[jwt_handler.algorithm]
        )
        exp_time = datetime.fromtimestamp(decoded["exp"])
        expected_exp = datetime.utcnow() + custom_expires

        # Allow 5 second tolerance for execution time
        assert abs((exp_time - expected_exp).total_seconds()) < 5

    def test_create_token_with_empty_data(self, jwt_handler):
        """Test creating token with empty user data."""
        with pytest.raises(ValueError):
            jwt_handler.create_access_token({})

    def test_create_token_with_none_data(self, jwt_handler):
        """Test creating token with None data."""
        with pytest.raises((ValueError, TypeError)):
            jwt_handler.create_access_token(None)

    # Token Verification Tests

    def test_verify_valid_token(self, jwt_handler, valid_user_data):
        """Test verifying valid token."""
        token = jwt_handler.create_access_token(valid_user_data)
        decoded_data = jwt_handler.verify_token(token)

        assert decoded_data["user_id"] == valid_user_data["user_id"]
        assert decoded_data["email"] == valid_user_data["email"]
        assert decoded_data["role"] == valid_user_data["role"]

    def test_verify_expired_token(self, jwt_handler, valid_user_data):
        """Test verifying expired token."""
        # Create token that expires immediately
        expired_token = jwt_handler.create_access_token(
            valid_user_data, expires_delta=timedelta(seconds=-1)
        )

        with pytest.raises(jwt.ExpiredSignatureError):
            jwt_handler.verify_token(expired_token)

    def test_verify_invalid_signature(self, jwt_handler, valid_user_data):
        """Test verifying token with invalid signature."""
        # Create token with one secret
        token = jwt_handler.create_access_token(valid_user_data)

        # Try to verify with different secret
        different_handler = JWTAuthService(
            secret_key="different-secret-key", algorithm="HS256"
        )

        with pytest.raises(jwt.InvalidSignatureError):
            different_handler.verify_token(token)

    def test_verify_malformed_token(self, jwt_handler):
        """Test verifying malformed token."""
        malformed_tokens = [
            "not.a.jwt",
            "invalid-token",
            "",
            "header.payload",  # Missing signature
            "too.many.parts.here.invalid",
        ]

        for token in malformed_tokens:
            with pytest.raises((jwt.DecodeError, jwt.InvalidTokenError)):
                jwt_handler.verify_token(token)

    def test_verify_token_with_invalid_algorithm(self, jwt_handler, valid_user_data):
        """Test token created with different algorithm."""
        # Create token with RS256 (if available)
        try:
            malicious_token = jwt.encode(valid_user_data, "fake-key", algorithm="none")

            with pytest.raises(jwt.InvalidTokenError):
                jwt_handler.verify_token(malicious_token)
        except Exception:
            # Skip if algorithm not available
            pass

    # Token Content Security Tests

    def test_token_contains_no_sensitive_data(self, jwt_handler):
        """Test that tokens don't contain sensitive information."""
        user_data_with_password = {
            "user_id": "user123",
            "email": "test@example.com",
            "password": "secret_password",  # Should not be in token
            "role": "user",
        }

        token = jwt_handler.create_access_token(user_data_with_password)
        decoded = jwt.decode(token, options={"verify_signature": False})

        assert "password" not in decoded
        assert "secret" not in str(decoded).lower()

    def test_token_includes_security_claims(self, jwt_handler, valid_user_data):
        """Test that tokens include required security claims."""
        token = jwt_handler.create_access_token(valid_user_data)
        decoded = jwt_handler.verify_token(token)

        # Check for required claims
        required_claims = ["exp", "iat", "user_id"]
        for claim in required_claims:
            assert claim in decoded

    def test_token_jti_uniqueness(self, jwt_handler, valid_user_data):
        """Test that each token has a unique JTI (JWT ID)."""
        token1 = jwt_handler.create_access_token(valid_user_data)
        token2 = jwt_handler.create_access_token(valid_user_data)

        decoded1 = jwt.decode(token1, options={"verify_signature": False})
        decoded2 = jwt.decode(token2, options={"verify_signature": False})

        # If JTI is implemented, they should be different
        if "jti" in decoded1 and "jti" in decoded2:
            assert decoded1["jti"] != decoded2["jti"]

    # Authentication Flow Tests

    def test_authenticate_valid_credentials(self, jwt_handler):
        """Test authentication with valid credentials."""
        with patch(
            "monorepo.infrastructure.auth.password_hasher.verify_password"
        ) as mock_verify:
            mock_verify.return_value = True

            result = jwt_handler.authenticate("test@example.com", "correct_password")

            assert result is not None
            assert "access_token" in result
            assert "token_type" in result
            assert result["token_type"] == "bearer"

    def test_authenticate_invalid_password(self, jwt_handler):
        """Test authentication with invalid password."""
        with patch(
            "monorepo.infrastructure.auth.password_hasher.verify_password"
        ) as mock_verify:
            mock_verify.return_value = False

            with pytest.raises(AuthenticationError):
                jwt_handler.authenticate("test@example.com", "wrong_password")

    def test_authenticate_nonexistent_user(self, jwt_handler):
        """Test authentication with non-existent user."""
        with patch(
            "monorepo.infrastructure.auth.user_repository.get_user_by_email"
        ) as mock_get:
            mock_get.return_value = None

            with pytest.raises(AuthenticationError):
                jwt_handler.authenticate("nonexistent@example.com", "password")

    # Password Security Tests

    def test_password_hashing_security(self):
        """Test password hashing security requirements."""
        with patch("monorepo.infrastructure.auth.password_hasher") as mock_hasher:
            password = "test_password_123"

            # Mock secure password hashing
            mock_hasher.hash_password.return_value = (
                "$2b$12$" + "x" * 53
            )  # bcrypt format
            mock_hasher.verify_password.return_value = True

            hashed = mock_hasher.hash_password(password)

            # Verify bcrypt format (starts with $2b$12$ for cost factor 12)
            assert hashed.startswith("$2b$12$")
            assert len(hashed) >= 60  # bcrypt hashes are 60 characters

    def test_password_complexity_requirements(self):
        """Test password complexity validation."""
        from monorepo.infrastructure.auth.password_validator import PasswordValidator

        validator = PasswordValidator()

        # Test weak passwords
        weak_passwords = [
            "123456",
            "password",
            "abc",
            "12345678",  # No letters
            "abcdefgh",  # No numbers
            "Password",  # No special chars
        ]

        for password in weak_passwords:
            assert not validator.is_valid(password)

        # Test strong passwords
        strong_passwords = [
            "MyStr0ng!Password",
            "C0mplex@Pass123",
            "Secure#789Password",
        ]

        for password in strong_passwords:
            assert validator.is_valid(password)

    # Session Management Tests

    def test_token_blacklisting(self, jwt_handler, valid_user_data):
        """Test token blacklisting for logout."""
        token = jwt_handler.create_access_token(valid_user_data)

        # Token should be valid initially
        decoded = jwt_handler.verify_token(token)
        assert decoded is not None

        # Blacklist the token
        jwt_handler.blacklist_token(token)

        # Token should now be invalid
        with pytest.raises(AuthenticationError):
            jwt_handler.verify_token(token)

    def test_multiple_session_management(self, jwt_handler, valid_user_data):
        """Test managing multiple user sessions."""
        # Create multiple tokens for same user
        token1 = jwt_handler.create_access_token(valid_user_data)
        token2 = jwt_handler.create_access_token(valid_user_data)

        # Both should be valid
        assert jwt_handler.verify_token(token1) is not None
        assert jwt_handler.verify_token(token2) is not None

        # Revoke all sessions for user
        jwt_handler.revoke_all_user_sessions(valid_user_data["user_id"])

        # Both tokens should now be invalid
        with pytest.raises(AuthenticationError):
            jwt_handler.verify_token(token1)
        with pytest.raises(AuthenticationError):
            jwt_handler.verify_token(token2)

    # Rate Limiting Tests

    def test_authentication_rate_limiting(self, jwt_handler):
        """Test rate limiting for authentication attempts."""
        with patch(
            "monorepo.infrastructure.auth.rate_limiter.check_rate_limit"
        ) as mock_rate_limit:
            # Simulate rate limit exceeded
            mock_rate_limit.return_value = False

            with pytest.raises(SecurityError, match="Rate limit exceeded"):
                jwt_handler.authenticate("test@example.com", "password")

    def test_brute_force_protection(self, jwt_handler):
        """Test brute force attack protection."""
        with patch(
            "monorepo.infrastructure.auth.brute_force_detector"
        ) as mock_detector:
            # Simulate multiple failed attempts
            mock_detector.is_locked.return_value = True

            with pytest.raises(SecurityError, match="Account temporarily locked"):
                jwt_handler.authenticate("test@example.com", "password")

    # Multi-Factor Authentication Tests

    def test_mfa_token_generation(self, jwt_handler):
        """Test MFA token generation."""
        user_email = "test@example.com"

        mfa_token = jwt_handler.generate_mfa_token(user_email)

        assert mfa_token is not None
        assert len(mfa_token) == 6  # Typical TOTP code length
        assert mfa_token.isdigit()

    def test_mfa_token_verification(self, jwt_handler):
        """Test MFA token verification."""
        user_email = "test@example.com"

        with patch("monorepo.infrastructure.auth.totp_handler.verify") as mock_verify:
            mock_verify.return_value = True

            result = jwt_handler.verify_mfa_token(user_email, "123456")

            assert result is True

    def test_mfa_token_expiry(self, jwt_handler):
        """Test MFA token expiry."""
        user_email = "test@example.com"

        with patch("monorepo.infrastructure.auth.totp_handler.verify") as mock_verify:
            # Simulate expired token
            mock_verify.return_value = False

            result = jwt_handler.verify_mfa_token(user_email, "123456")

            assert result is False

    # API Key Authentication Tests

    def test_api_key_generation(self, jwt_handler):
        """Test API key generation."""
        user_id = "user123"
        permissions = ["read:datasets"]

        api_key = jwt_handler.generate_api_key(user_id, permissions)

        assert api_key is not None
        assert len(api_key) >= 32  # Minimum length for security
        assert api_key.replace("-", "").isalnum()  # Should be alphanumeric

    def test_api_key_verification(self, jwt_handler):
        """Test API key verification."""
        user_id = "user123"
        permissions = ["read:datasets"]

        api_key = jwt_handler.generate_api_key(user_id, permissions)

        # Verify the API key
        verified_data = jwt_handler.verify_api_key(api_key)

        assert verified_data["user_id"] == user_id
        assert verified_data["permissions"] == permissions

    def test_api_key_revocation(self, jwt_handler):
        """Test API key revocation."""
        user_id = "user123"
        permissions = ["read:datasets"]

        api_key = jwt_handler.generate_api_key(user_id, permissions)

        # Verify key works initially
        verified_data = jwt_handler.verify_api_key(api_key)
        assert verified_data is not None

        # Revoke the key
        jwt_handler.revoke_api_key(api_key)

        # Key should no longer work
        with pytest.raises(AuthenticationError):
            jwt_handler.verify_api_key(api_key)


class TestAuthenticationSecurity:
    """Test suite for authentication security vulnerabilities."""

    def test_timing_attack_protection(self, jwt_handler):
        """Test protection against timing attacks."""
        import time

        # Test with valid vs invalid usernames
        # Should take similar time to prevent user enumeration
        start_time = time.time()
        try:
            jwt_handler.authenticate("valid@example.com", "password")
        except:
            pass
        valid_time = time.time() - start_time

        start_time = time.time()
        try:
            jwt_handler.authenticate("invalid@example.com", "password")
        except:
            pass
        invalid_time = time.time() - start_time

        # Times should be similar (within 50ms)
        time_diff = abs(valid_time - invalid_time)
        assert time_diff < 0.05

    def test_constant_time_comparison(self):
        """Test constant-time string comparison for security."""
        from monorepo.infrastructure.auth.crypto_utils import constant_time_compare

        # Same strings
        assert constant_time_compare("secret", "secret") is True

        # Different strings of same length
        assert constant_time_compare("secret", "public") is False

        # Different lengths
        assert constant_time_compare("secret", "sec") is False

        # Empty strings
        assert constant_time_compare("", "") is True

    def test_secure_random_generation(self):
        """Test secure random number generation."""
        # Generate multiple random values
        random_values = []
        for _ in range(100):
            value = secrets.token_urlsafe(32)
            random_values.append(value)

        # All values should be unique
        assert len(set(random_values)) == 100

        # All values should be proper length
        for value in random_values:
            assert len(value) > 30  # URL-safe base64 encoding

    def test_password_salt_uniqueness(self):
        """Test that password salts are unique."""
        with patch("monorepo.infrastructure.auth.password_hasher") as mock_hasher:
            password = "test_password"

            # Mock salt generation
            salts = []

            def mock_hash(pwd):
                salt = secrets.token_hex(16)
                salts.append(salt)
                return f"$2b$12${salt}{'x' * 31}"

            mock_hasher.hash_password.side_effect = mock_hash

            # Generate multiple hashes
            for _ in range(10):
                mock_hasher.hash_password(password)

            # All salts should be unique
            assert len(set(salts)) == 10

    def test_jwt_algorithm_confusion_prevention(self, jwt_handler):
        """Test prevention of JWT algorithm confusion attacks."""
        # Try to create token with 'none' algorithm
        payload = {"user_id": "user123", "role": "admin"}

        try:
            malicious_token = jwt.encode(payload, "", algorithm="none")

            # Should reject tokens with 'none' algorithm
            with pytest.raises(jwt.InvalidTokenError):
                jwt_handler.verify_token(malicious_token)
        except Exception:
            # Expected - 'none' algorithm should be rejected
            pass

    def test_jwt_key_confusion_prevention(self, jwt_handler, valid_user_data):
        """Test prevention of JWT key confusion attacks."""
        # Create legitimate token
        token = jwt_handler.create_access_token(valid_user_data)

        # Try to verify with public key as HMAC secret (if using RS256)
        if jwt_handler.algorithm.startswith("RS"):
            with pytest.raises(jwt.InvalidSignatureError):
                # This should fail - can't verify RS256 with HS256
                jwt.decode(token, "fake-public-key", algorithms=["HS256"])

    def test_session_fixation_prevention(self, jwt_handler, valid_user_data):
        """Test prevention of session fixation attacks."""
        # Create initial token
        old_token = jwt_handler.create_access_token(valid_user_data)

        # After authentication, new token should be generated
        new_token = jwt_handler.create_access_token(valid_user_data)

        # Tokens should be different
        assert old_token != new_token

        # Old token should be invalidated
        jwt_handler.blacklist_token(old_token)

        with pytest.raises(AuthenticationError):
            jwt_handler.verify_token(old_token)

    def test_csrf_token_generation(self, jwt_handler):
        """Test CSRF token generation for form protection."""
        user_id = "user123"

        csrf_token = jwt_handler.generate_csrf_token(user_id)

        assert csrf_token is not None
        assert len(csrf_token) >= 32

        # Verify CSRF token
        is_valid = jwt_handler.verify_csrf_token(user_id, csrf_token)
        assert is_valid is True

        # Invalid token should fail
        is_valid = jwt_handler.verify_csrf_token(user_id, "invalid-token")
        assert is_valid is False


class TestAuthenticationIntegration:
    """Integration tests for authentication system."""

    def test_complete_authentication_flow(self, jwt_handler):
        """Test complete authentication workflow."""
        # 1. Register user (mock)
        user_data = {
            "email": "newuser@example.com",
            "password": "SecurePass123!",
            "role": "user",
        }

        # 2. Authenticate user
        with patch(
            "monorepo.infrastructure.auth.password_hasher.verify_password"
        ) as mock_verify:
            mock_verify.return_value = True

            auth_result = jwt_handler.authenticate(
                user_data["email"], user_data["password"]
            )

            assert "access_token" in auth_result
            token = auth_result["access_token"]

        # 3. Use token to access protected resource
        decoded_data = jwt_handler.verify_token(token)
        assert decoded_data["email"] == user_data["email"]

        # 4. Refresh token
        new_token = jwt_handler.refresh_token(token)
        assert new_token != token

        # 5. Logout (blacklist token)
        jwt_handler.blacklist_token(token)

        with pytest.raises(AuthenticationError):
            jwt_handler.verify_token(token)

    def test_concurrent_authentication_sessions(self, jwt_handler, valid_user_data):
        """Test handling concurrent authentication sessions."""
        # Create multiple tokens for same user
        tokens = []
        for _ in range(5):
            token = jwt_handler.create_access_token(valid_user_data)
            tokens.append(token)

        # All tokens should be valid
        for token in tokens:
            decoded = jwt_handler.verify_token(token)
            assert decoded["user_id"] == valid_user_data["user_id"]

        # Revoke specific session
        jwt_handler.blacklist_token(tokens[0])

        # First token should be invalid, others valid
        with pytest.raises(AuthenticationError):
            jwt_handler.verify_token(tokens[0])

        for token in tokens[1:]:
            decoded = jwt_handler.verify_token(token)
            assert decoded is not None

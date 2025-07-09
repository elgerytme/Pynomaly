"""Unit tests for enhanced JWT authentication service."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from pynomaly.domain.exceptions import AuthenticationError, AuthorizationError
from pynomaly.infrastructure.auth.jwt_auth_enhanced import (
    EnhancedJWTAuthService,
    JWKSResponse,
    TokenResponse,
    UserModel,
    init_auth,
    get_auth,
)
from pynomaly.infrastructure.config import Settings


class TestEnhancedJWTAuthService:
    """Test cases for EnhancedJWTAuthService."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            secret_key="test-secret-key",
            jwt_algorithm="HS256",
            jwt_expiration=3600,
            app=Mock(environment="testing")
        )

    @pytest.fixture
    def auth_service(self, settings):
        """Create test auth service."""
        return EnhancedJWTAuthService(settings)

    def test_init_auth_service(self, auth_service, settings):
        """Test authentication service initialization."""
        assert auth_service.settings == settings
        assert auth_service.algorithm == "RS256"
        assert auth_service.access_token_expire.total_seconds() == 3600
        assert auth_service.refresh_token_expire.days == 7
        assert len(auth_service.jwks_keys) == 1
        assert auth_service.current_key_id == "1"

    def test_key_initialization(self, auth_service):
        """Test RSA key initialization."""
        assert auth_service.current_key_id in auth_service.private_keys
        assert auth_service.current_key_id in auth_service.public_keys
        assert len(auth_service.jwks_keys) == 1
        assert auth_service.jwks_keys[0].kid == "1"
        assert auth_service.jwks_keys[0].kty == "RSA"
        assert auth_service.jwks_keys[0].use == "sig"

    def test_get_jwks(self, auth_service):
        """Test JWKS retrieval."""
        jwks = auth_service.get_jwks()
        assert isinstance(jwks, JWKSResponse)
        assert len(jwks.keys) == 1
        assert jwks.keys[0].kid == "1"
        assert jwks.keys[0].kty == "RSA"
        assert jwks.keys[0].use == "sig"

    def test_rotate_keys(self, auth_service):
        """Test key rotation."""
        initial_key_id = auth_service.current_key_id
        initial_key_count = len(auth_service.jwks_keys)
        
        new_key_id = auth_service.rotate_keys()
        
        assert new_key_id != initial_key_id
        assert auth_service.current_key_id == new_key_id
        assert len(auth_service.jwks_keys) == initial_key_count + 1
        assert new_key_id in auth_service.private_keys
        assert new_key_id in auth_service.public_keys

    def test_cleanup_old_keys(self, auth_service):
        """Test cleaning up old keys."""
        # Generate multiple keys
        for _ in range(3):
            auth_service.rotate_keys()
        
        assert len(auth_service.jwks_keys) == 4
        
        # Clean up, keeping only 2 keys
        auth_service.cleanup_old_keys(keep_count=2)
        
        assert len(auth_service.jwks_keys) == 2
        assert len(auth_service.private_keys) == 2
        assert len(auth_service.public_keys) == 2

    def test_hash_password(self, auth_service):
        """Test password hashing."""
        password = "test-password"
        hashed = auth_service.hash_password(password)
        
        assert hashed != password
        assert auth_service.verify_password(password, hashed)
        assert not auth_service.verify_password("wrong-password", hashed)

    def test_create_user(self, auth_service):
        """Test user creation."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            full_name="Test User",
            roles=["user"]
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.roles == ["user"]
        assert user.is_active is True
        assert user.is_superuser is False
        assert auth_service.verify_password("password123", user.hashed_password)

    def test_create_user_duplicate(self, auth_service):
        """Test creating duplicate user raises error."""
        auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        # Try to create user with same username
        with pytest.raises(ValueError, match="User already exists"):
            auth_service.create_user(
                username="testuser",
                email="different@example.com",
                password="password123"
            )
        
        # Try to create user with same email
        with pytest.raises(ValueError, match="User already exists"):
            auth_service.create_user(
                username="different",
                email="test@example.com",
                password="password123"
            )

    def test_authenticate_user_success(self, auth_service):
        """Test successful user authentication."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        # Authenticate with username
        authenticated_user = auth_service.authenticate_user("testuser", "password123")
        assert authenticated_user.id == user.id
        assert authenticated_user.last_login is not None
        
        # Authenticate with email
        authenticated_user = auth_service.authenticate_user("test@example.com", "password123")
        assert authenticated_user.id == user.id

    def test_authenticate_user_invalid_credentials(self, auth_service):
        """Test authentication with invalid credentials."""
        auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        # Wrong password
        with pytest.raises(AuthenticationError, match="Invalid username or password"):
            auth_service.authenticate_user("testuser", "wrong-password")
        
        # Wrong username
        with pytest.raises(AuthenticationError, match="Invalid username or password"):
            auth_service.authenticate_user("wronguser", "password123")

    def test_authenticate_user_inactive(self, auth_service):
        """Test authentication with inactive user."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        user.is_active = False
        
        with pytest.raises(AuthenticationError, match="User account is inactive"):
            auth_service.authenticate_user("testuser", "password123")

    def test_account_lockout(self, auth_service):
        """Test account lockout after failed attempts."""
        auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        # Make 5 failed attempts
        for _ in range(5):
            with pytest.raises(AuthenticationError):
                auth_service.authenticate_user("testuser", "wrong-password")
        
        # Account should be locked now
        with pytest.raises(AuthenticationError, match="Account temporarily locked"):
            auth_service.authenticate_user("testuser", "password123")

    def test_create_access_token(self, auth_service):
        """Test access token creation."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"]
        )
        
        token_response = auth_service.create_access_token(user)
        
        assert isinstance(token_response, TokenResponse)
        assert token_response.access_token is not None
        assert token_response.refresh_token is not None
        assert token_response.token_type == "bearer"
        assert token_response.expires_in == 3600

    def test_decode_token(self, auth_service):
        """Test token decoding."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"]
        )
        
        token_response = auth_service.create_access_token(user)
        payload = auth_service.decode_token(token_response.access_token)
        
        assert payload.sub == user.id
        assert payload.type == "access"
        assert payload.roles == ["user"]
        assert payload.kid == auth_service.current_key_id

    def test_decode_token_invalid(self, auth_service):
        """Test decoding invalid token."""
        with pytest.raises(AuthenticationError, match="Invalid token"):
            auth_service.decode_token("invalid-token")

    def test_decode_token_blacklisted(self, auth_service):
        """Test decoding blacklisted token."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        token_response = auth_service.create_access_token(user)
        token = token_response.access_token
        
        # Blacklist the token
        auth_service.blacklist_token(token)
        
        with pytest.raises(AuthenticationError, match="Token has been revoked"):
            auth_service.decode_token(token)

    def test_get_current_user(self, auth_service):
        """Test getting current user from token."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        token_response = auth_service.create_access_token(user)
        current_user = auth_service.get_current_user(token_response.access_token)
        
        assert current_user.id == user.id
        assert current_user.username == user.username

    def test_refresh_access_token(self, auth_service):
        """Test refreshing access token."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        token_response = auth_service.create_access_token(user)
        new_token_response = auth_service.refresh_access_token(token_response.refresh_token)
        
        assert isinstance(new_token_response, TokenResponse)
        assert new_token_response.access_token != token_response.access_token
        assert new_token_response.refresh_token != token_response.refresh_token

    def test_refresh_access_token_invalid_type(self, auth_service):
        """Test refreshing with access token instead of refresh token."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        token_response = auth_service.create_access_token(user)
        
        with pytest.raises(AuthenticationError, match="Invalid token type"):
            auth_service.refresh_access_token(token_response.access_token)

    def test_check_permissions(self, auth_service):
        """Test permission checking."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"]
        )
        
        # User should have user permissions
        assert auth_service.check_permissions(user, ["detectors:read"])
        assert auth_service.check_permissions(user, ["datasets:read"])
        
        # User should not have admin permissions
        assert not auth_service.check_permissions(user, ["users:delete"])
        assert not auth_service.check_permissions(user, ["settings:write"])

    def test_check_permissions_superuser(self, auth_service):
        """Test permission checking for superuser."""
        user = auth_service.create_user(
            username="superuser",
            email="super@example.com",
            password="password123"
        )
        user.is_superuser = True
        
        # Superuser should have all permissions
        assert auth_service.check_permissions(user, ["users:delete"])
        assert auth_service.check_permissions(user, ["settings:write"])
        assert auth_service.check_permissions(user, ["any:permission"])

    def test_require_permissions(self, auth_service):
        """Test requiring permissions."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"]
        )
        
        # Should not raise for allowed permissions
        auth_service.require_permissions(user, ["detectors:read"])
        
        # Should raise for disallowed permissions
        with pytest.raises(AuthorizationError, match="User lacks required permissions"):
            auth_service.require_permissions(user, ["users:delete"])

    def test_create_api_key(self, auth_service):
        """Test API key creation."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        api_key = auth_service.create_api_key(user.id, "test-key")
        
        assert api_key.startswith("pyn_")
        assert api_key in user.api_keys
        assert auth_service._api_keys[api_key] == user.id

    def test_authenticate_api_key(self, auth_service):
        """Test API key authentication."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        api_key = auth_service.create_api_key(user.id, "test-key")
        authenticated_user = auth_service.authenticate_api_key(api_key)
        
        assert authenticated_user.id == user.id

    def test_authenticate_api_key_invalid(self, auth_service):
        """Test authentication with invalid API key."""
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            auth_service.authenticate_api_key("invalid-key")

    def test_revoke_api_key(self, auth_service):
        """Test API key revocation."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        api_key = auth_service.create_api_key(user.id, "test-key")
        
        # Revoke the key
        result = auth_service.revoke_api_key(api_key)
        
        assert result is True
        assert api_key not in user.api_keys
        assert api_key not in auth_service._api_keys
        
        # Try to authenticate with revoked key
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            auth_service.authenticate_api_key(api_key)

    def test_get_permissions_for_roles(self, auth_service):
        """Test getting permissions for roles."""
        admin_permissions = auth_service._get_permissions_for_roles(["admin"])
        user_permissions = auth_service._get_permissions_for_roles(["user"])
        viewer_permissions = auth_service._get_permissions_for_roles(["viewer"])
        
        assert "users:delete" in admin_permissions
        assert "settings:write" in admin_permissions
        assert "detectors:read" in admin_permissions
        
        assert "detectors:read" in user_permissions
        assert "users:delete" not in user_permissions
        
        assert "detectors:read" in viewer_permissions
        assert "detectors:write" not in viewer_permissions

    def test_global_auth_service(self, settings):
        """Test global auth service initialization."""
        auth_service = init_auth(settings)
        
        assert get_auth() == auth_service
        assert isinstance(auth_service, EnhancedJWTAuthService)


class TestJWTAuthIntegration:
    """Integration tests for JWT authentication."""

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            secret_key="test-secret-key",
            jwt_algorithm="HS256",
            jwt_expiration=3600,
            app=Mock(environment="testing")
        )

    @pytest.fixture
    def auth_service(self, settings):
        """Create test auth service."""
        return EnhancedJWTAuthService(settings)

    def test_complete_auth_flow(self, auth_service):
        """Test complete authentication flow."""
        # Create user
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"]
        )
        
        # Authenticate user
        authenticated_user = auth_service.authenticate_user("testuser", "password123")
        assert authenticated_user.id == user.id
        
        # Create token
        token_response = auth_service.create_access_token(authenticated_user)
        assert token_response.access_token is not None
        
        # Verify token
        current_user = auth_service.get_current_user(token_response.access_token)
        assert current_user.id == user.id
        
        # Check permissions
        assert auth_service.check_permissions(current_user, ["detectors:read"])
        assert not auth_service.check_permissions(current_user, ["users:delete"])
        
        # Refresh token
        new_token_response = auth_service.refresh_access_token(token_response.refresh_token)
        assert new_token_response.access_token != token_response.access_token

    def test_api_key_flow(self, auth_service):
        """Test API key authentication flow."""
        # Create user
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"]
        )
        
        # Create API key
        api_key = auth_service.create_api_key(user.id, "test-key")
        
        # Authenticate with API key
        authenticated_user = auth_service.authenticate_api_key(api_key)
        assert authenticated_user.id == user.id
        
        # Check permissions
        assert auth_service.check_permissions(authenticated_user, ["detectors:read"])
        
        # Revoke API key
        auth_service.revoke_api_key(api_key)
        
        # Should fail after revocation
        with pytest.raises(AuthenticationError):
            auth_service.authenticate_api_key(api_key)

    def test_key_rotation_flow(self, auth_service):
        """Test key rotation flow."""
        # Create user and token
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        token_response = auth_service.create_access_token(user)
        old_token = token_response.access_token
        
        # Verify token works
        auth_service.get_current_user(old_token)
        
        # Rotate keys
        new_key_id = auth_service.rotate_keys()
        
        # Old token should still work (until cleanup)
        auth_service.get_current_user(old_token)
        
        # New tokens should use new key
        new_token_response = auth_service.create_access_token(user)
        payload = auth_service.decode_token(new_token_response.access_token)
        assert payload.kid == new_key_id
        
        # Clean up old keys
        auth_service.cleanup_old_keys(keep_count=1)
        
        # Old token should fail now
        with pytest.raises(AuthenticationError):
            auth_service.get_current_user(old_token)
        
        # New token should still work
        auth_service.get_current_user(new_token_response.access_token)

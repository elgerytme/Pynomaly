"""Comprehensive tests for authentication infrastructure - Phase 2 Coverage Enhancement."""

from __future__ import annotations

import pytest
import jwt
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional

from pynomaly.infrastructure.auth.jwt_auth import (
    JWTAuthService,
    UserModel,
    TokenPayload,
    TokenResponse,
    init_auth,
    get_auth
)
from pynomaly.infrastructure.auth.middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    PermissionMiddleware
)
from pynomaly.infrastructure.config.settings import Settings
from pynomaly.domain.exceptions import AuthenticationError, AuthorizationError


class TestJWTAuthService:
    """Comprehensive tests for JWT authentication service."""
    
    @pytest.fixture
    def auth_settings(self):
        """Create test settings for authentication."""
        settings = Mock(spec=Settings)
        settings.secret_key = "test-secret-key-very-long-for-security"
        settings.jwt_algorithm = "HS256"
        settings.jwt_expiration = 3600  # 1 hour
        settings.app.environment = "testing"
        return settings
    
    @pytest.fixture
    def auth_service(self, auth_settings):
        """Create JWT auth service for testing."""
        return JWTAuthService(auth_settings)
    
    def test_jwt_auth_service_initialization(self, auth_service, auth_settings):
        """Test JWT auth service initialization."""
        assert auth_service.secret_key == auth_settings.secret_key
        assert auth_service.algorithm == auth_settings.jwt_algorithm
        assert auth_service.access_token_expire.total_seconds() == 3600
        assert auth_service.refresh_token_expire.days == 7
        assert len(auth_service._users) == 0  # No default users in testing
    
    def test_password_hashing_and_verification(self, auth_service):
        """Test password hashing and verification."""
        password = "test_password_123"
        
        # Test hashing
        hashed = auth_service.hash_password(password)
        assert hashed != password
        assert hashed.startswith("$2b$")  # bcrypt hash
        
        # Test verification
        assert auth_service.verify_password(password, hashed) is True
        assert auth_service.verify_password("wrong_password", hashed) is False
    
    def test_create_user_success(self, auth_service):
        """Test successful user creation."""
        user = auth_service.create_user(
            username="testuser",
            email="test@example.com",
            password="secure_password",
            full_name="Test User",
            roles=["user", "viewer"]
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.roles == ["user", "viewer"]
        assert user.is_active is True
        assert user.is_superuser is False
        assert auth_service.verify_password("secure_password", user.hashed_password)
        
        # Verify user is stored
        assert user.id in auth_service._users
    
    def test_create_user_duplicate_username(self, auth_service):
        """Test user creation with duplicate username."""
        auth_service.create_user("testuser", "test1@example.com", "password")
        
        with pytest.raises(ValueError, match="User already exists"):
            auth_service.create_user("testuser", "test2@example.com", "password")
    
    def test_create_user_duplicate_email(self, auth_service):
        """Test user creation with duplicate email."""
        auth_service.create_user("testuser1", "test@example.com", "password")
        
        with pytest.raises(ValueError, match="User already exists"):
            auth_service.create_user("testuser2", "test@example.com", "password")
    
    def test_create_access_token_success(self, auth_service):
        """Test successful access token creation."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        
        token_response = auth_service.create_access_token(user)
        
        assert isinstance(token_response, TokenResponse)
        assert token_response.token_type == "bearer"
        assert token_response.expires_in == 3600
        assert token_response.access_token is not None
        assert token_response.refresh_token is not None
        
        # Verify token can be decoded
        payload = auth_service.decode_token(token_response.access_token)
        assert payload.sub == user.id
        assert payload.type == "access"
        assert payload.roles == user.roles
    
    def test_decode_token_success(self, auth_service):
        """Test successful token decoding."""
        user = auth_service.create_user("testuser", "test@example.com", "password", roles=["admin"])
        token_response = auth_service.create_access_token(user)
        
        payload = auth_service.decode_token(token_response.access_token)
        
        assert payload.sub == user.id
        assert payload.type == "access"
        assert payload.roles == ["admin"]
        assert len(payload.permissions) > 0  # Admin should have permissions
        assert payload.exp > datetime.now(timezone.utc)
    
    def test_decode_token_expired(self, auth_service):
        """Test decoding expired token."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        
        # Create expired token manually
        now = datetime.now(timezone.utc)
        expired_payload = {
            "sub": user.id,
            "exp": now - timedelta(hours=1),  # 1 hour ago
            "iat": now - timedelta(hours=2),
            "type": "access",
            "roles": user.roles,
            "permissions": []
        }
        
        expired_token = jwt.encode(expired_payload, auth_service.secret_key, algorithm=auth_service.algorithm)
        
        with pytest.raises(AuthenticationError, match="Token has expired"):
            auth_service.decode_token(expired_token)
    
    def test_decode_token_invalid(self, auth_service):
        """Test decoding invalid token."""
        with pytest.raises(AuthenticationError, match="Invalid token"):
            auth_service.decode_token("invalid.token.here")
    
    def test_authenticate_user_success(self, auth_service):
        """Test successful user authentication."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        
        # Authenticate with username
        authenticated = auth_service.authenticate_user("testuser", "password")
        assert authenticated.id == user.id
        assert authenticated.last_login is not None
        
        # Authenticate with email
        authenticated = auth_service.authenticate_user("test@example.com", "password")
        assert authenticated.id == user.id
    
    def test_authenticate_user_wrong_password(self, auth_service):
        """Test authentication with wrong password."""
        auth_service.create_user("testuser", "test@example.com", "password")
        
        with pytest.raises(AuthenticationError, match="Invalid username or password"):
            auth_service.authenticate_user("testuser", "wrong_password")
    
    def test_authenticate_user_not_found(self, auth_service):
        """Test authentication with non-existent user."""
        with pytest.raises(AuthenticationError, match="Invalid username or password"):
            auth_service.authenticate_user("nonexistent", "password")
    
    def test_authenticate_user_inactive(self, auth_service):
        """Test authentication with inactive user."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        user.is_active = False
        
        with pytest.raises(AuthenticationError, match="User account is inactive"):
            auth_service.authenticate_user("testuser", "password")
    
    def test_create_api_key_success(self, auth_service):
        """Test successful API key creation."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        
        api_key = auth_service.create_api_key(user.id, "Test API Key")
        
        assert api_key.startswith("pyn_")
        assert len(api_key) > 20
        assert api_key in user.api_keys
        assert auth_service._api_keys[api_key] == user.id
    
    def test_create_api_key_user_not_found(self, auth_service):
        """Test API key creation for non-existent user."""
        with pytest.raises(ValueError, match="User not found"):
            auth_service.create_api_key("nonexistent_user", "Test Key")
    
    def test_authenticate_api_key_success(self, auth_service):
        """Test successful API key authentication."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        api_key = auth_service.create_api_key(user.id, "Test Key")
        
        authenticated = auth_service.authenticate_api_key(api_key)
        assert authenticated.id == user.id
    
    def test_authenticate_api_key_invalid(self, auth_service):
        """Test authentication with invalid API key."""
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            auth_service.authenticate_api_key("invalid_api_key")
    
    def test_authenticate_api_key_inactive_user(self, auth_service):
        """Test API key authentication with inactive user."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        api_key = auth_service.create_api_key(user.id, "Test Key")
        user.is_active = False
        
        with pytest.raises(AuthenticationError, match="User account is inactive"):
            auth_service.authenticate_api_key(api_key)
    
    def test_revoke_api_key_success(self, auth_service):
        """Test successful API key revocation."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        api_key = auth_service.create_api_key(user.id, "Test Key")
        
        # Verify key exists
        assert api_key in auth_service._api_keys
        assert api_key in user.api_keys
        
        # Revoke key
        result = auth_service.revoke_api_key(api_key)
        assert result is True
        
        # Verify key is removed
        assert api_key not in auth_service._api_keys
        assert api_key not in user.api_keys
    
    def test_revoke_api_key_not_found(self, auth_service):
        """Test revoking non-existent API key."""
        result = auth_service.revoke_api_key("nonexistent_key")
        assert result is False
    
    def test_get_current_user_success(self, auth_service):
        """Test getting current user from token."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        token_response = auth_service.create_access_token(user)
        
        current_user = auth_service.get_current_user(token_response.access_token)
        assert current_user.id == user.id
    
    def test_get_current_user_invalid_token(self, auth_service):
        """Test getting current user with invalid token."""
        with pytest.raises(AuthenticationError):
            auth_service.get_current_user("invalid.token")
    
    def test_get_current_user_not_found(self, auth_service):
        """Test getting current user when user no longer exists."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        token_response = auth_service.create_access_token(user)
        
        # Remove user
        del auth_service._users[user.id]
        
        with pytest.raises(AuthenticationError, match="User not found"):
            auth_service.get_current_user(token_response.access_token)
    
    def test_refresh_access_token_success(self, auth_service):
        """Test successful token refresh."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        token_response = auth_service.create_access_token(user)
        
        new_token_response = auth_service.refresh_access_token(token_response.refresh_token)
        
        assert isinstance(new_token_response, TokenResponse)
        assert new_token_response.access_token != token_response.access_token
        assert new_token_response.expires_in == 3600
    
    def test_refresh_access_token_invalid_type(self, auth_service):
        """Test token refresh with access token instead of refresh token."""
        user = auth_service.create_user("testuser", "test@example.com", "password")
        token_response = auth_service.create_access_token(user)
        
        with pytest.raises(AuthenticationError, match="Invalid token type"):
            auth_service.refresh_access_token(token_response.access_token)
    
    def test_check_permissions_superuser(self, auth_service):
        """Test permission checking for superuser."""
        user = auth_service.create_user("admin", "admin@example.com", "password")
        user.is_superuser = True
        
        # Superuser should have all permissions
        assert auth_service.check_permissions(user, ["any:permission"]) is True
        assert auth_service.check_permissions(user, ["multiple", "permissions"]) is True
    
    def test_check_permissions_admin_role(self, auth_service):
        """Test permission checking for admin role."""
        user = auth_service.create_user("admin", "admin@example.com", "password", roles=["admin"])
        
        # Admin should have all admin permissions
        assert auth_service.check_permissions(user, ["detectors:read"]) is True
        assert auth_service.check_permissions(user, ["users:write"]) is True
        assert auth_service.check_permissions(user, ["settings:read"]) is True
    
    def test_check_permissions_user_role(self, auth_service):
        """Test permission checking for user role."""
        user = auth_service.create_user("user", "user@example.com", "password", roles=["user"])
        
        # User should have limited permissions
        assert auth_service.check_permissions(user, ["detectors:read"]) is True
        assert auth_service.check_permissions(user, ["detectors:write"]) is True
        assert auth_service.check_permissions(user, ["users:delete"]) is False
        assert auth_service.check_permissions(user, ["settings:write"]) is False
    
    def test_check_permissions_viewer_role(self, auth_service):
        """Test permission checking for viewer role."""
        user = auth_service.create_user("viewer", "viewer@example.com", "password", roles=["viewer"])
        
        # Viewer should have read-only permissions
        assert auth_service.check_permissions(user, ["detectors:read"]) is True
        assert auth_service.check_permissions(user, ["detectors:write"]) is False
        assert auth_service.check_permissions(user, ["datasets:read"]) is True
        assert auth_service.check_permissions(user, ["datasets:write"]) is False
    
    def test_require_permissions_success(self, auth_service):
        """Test successful permission requirement."""
        user = auth_service.create_user("admin", "admin@example.com", "password", roles=["admin"])
        
        # Should not raise exception
        auth_service.require_permissions(user, ["detectors:read"])
        auth_service.require_permissions(user, ["detectors:read", "detectors:write"])
    
    def test_require_permissions_failure(self, auth_service):
        """Test failed permission requirement."""
        user = auth_service.create_user("viewer", "viewer@example.com", "password", roles=["viewer"])
        
        with pytest.raises(AuthorizationError, match="User lacks required permissions"):
            auth_service.require_permissions(user, ["users:delete"])
    
    def test_get_permissions_for_roles_multiple(self, auth_service):
        """Test getting permissions for multiple roles."""
        permissions = auth_service._get_permissions_for_roles(["admin", "user"])
        
        # Should contain admin permissions (which includes user permissions)
        assert "detectors:read" in permissions
        assert "detectors:write" in permissions
        assert "users:write" in permissions  # Admin-only
        assert "settings:write" in permissions  # Admin-only
    
    def test_get_permissions_for_roles_unknown(self, auth_service):
        """Test getting permissions for unknown role."""
        permissions = auth_service._get_permissions_for_roles(["unknown_role"])
        assert permissions == []


class TestAuthGlobalFunctions:
    """Test global authentication functions."""
    
    def test_init_and_get_auth(self):
        """Test initialization and retrieval of global auth service."""
        settings = Mock(spec=Settings)
        settings.secret_key = "test-key"
        settings.jwt_algorithm = "HS256"
        settings.jwt_expiration = 3600
        settings.app.environment = "testing"
        
        # Initialize global auth
        auth_service = init_auth(settings)
        assert isinstance(auth_service, JWTAuthService)
        
        # Retrieve global auth
        retrieved_auth = get_auth()
        assert retrieved_auth is auth_service
        assert retrieved_auth.secret_key == "test-key"


class TestUserModel:
    """Test UserModel validation and behavior."""
    
    def test_user_model_creation_minimal(self):
        """Test creating user model with minimal fields."""
        user = UserModel(
            id="test_id",
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password"
        )
        
        assert user.id == "test_id"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.is_superuser is False
        assert user.roles == []
        assert user.api_keys == []
        assert user.full_name is None
        assert user.last_login is None
        assert isinstance(user.created_at, datetime)
    
    def test_user_model_creation_complete(self):
        """Test creating user model with all fields."""
        now = datetime.now(timezone.utc)
        user = UserModel(
            id="test_id",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password",
            is_active=False,
            is_superuser=True,
            roles=["admin", "user"],
            api_keys=["key1", "key2"],
            created_at=now,
            last_login=now
        )
        
        assert user.full_name == "Test User"
        assert user.is_active is False
        assert user.is_superuser is True
        assert user.roles == ["admin", "user"]
        assert user.api_keys == ["key1", "key2"]
        assert user.created_at == now
        assert user.last_login == now


class TestTokenModels:
    """Test token model validation and behavior."""
    
    def test_token_payload_creation(self):
        """Test creating token payload."""
        now = datetime.now(timezone.utc)
        expire = now + timedelta(hours=1)
        
        payload = TokenPayload(
            sub="user_id",
            exp=expire,
            iat=now,
            type="access",
            roles=["user"],
            permissions=["read"]
        )
        
        assert payload.sub == "user_id"
        assert payload.exp == expire
        assert payload.iat == now
        assert payload.type == "access"
        assert payload.roles == ["user"]
        assert payload.permissions == ["read"]
    
    def test_token_response_creation(self):
        """Test creating token response."""
        response = TokenResponse(
            access_token="access_token_value",
            expires_in=3600,
            refresh_token="refresh_token_value"
        )
        
        assert response.access_token == "access_token_value"
        assert response.token_type == "bearer"  # Default value
        assert response.expires_in == 3600
        assert response.refresh_token == "refresh_token_value"
    
    def test_token_response_defaults(self):
        """Test token response with default values."""
        response = TokenResponse(
            access_token="access_token_value",
            expires_in=3600
        )
        
        assert response.token_type == "bearer"
        assert response.refresh_token is None


class TestAuthenticationMiddleware:
    """Test authentication middleware functionality."""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Create mock authentication service."""
        service = Mock(spec=JWTAuthService)
        return service
    
    @pytest.fixture
    def auth_middleware(self, mock_auth_service):
        """Create authentication middleware."""
        return AuthenticationMiddleware(mock_auth_service)
    
    def test_middleware_initialization(self, auth_middleware, mock_auth_service):
        """Test middleware initialization."""
        assert auth_middleware.auth_service is mock_auth_service
    
    @pytest.mark.asyncio
    async def test_authenticate_with_bearer_token(self, auth_middleware, mock_auth_service):
        """Test authentication with Bearer token."""
        # Mock user
        mock_user = Mock(spec=UserModel)
        mock_user.id = "user_id"
        mock_auth_service.get_current_user.return_value = mock_user
        
        # Mock request
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer valid_token"}
        
        user = await auth_middleware.authenticate(mock_request)
        
        assert user is mock_user
        mock_auth_service.get_current_user.assert_called_once_with("valid_token")
    
    @pytest.mark.asyncio
    async def test_authenticate_with_api_key(self, auth_middleware, mock_auth_service):
        """Test authentication with API key."""
        # Mock user
        mock_user = Mock(spec=UserModel)
        mock_user.id = "user_id"
        mock_auth_service.authenticate_api_key.return_value = mock_user
        
        # Mock request
        mock_request = Mock()
        mock_request.headers = {"X-API-Key": "valid_api_key"}
        
        user = await auth_middleware.authenticate(mock_request)
        
        assert user is mock_user
        mock_auth_service.authenticate_api_key.assert_called_once_with("valid_api_key")
    
    @pytest.mark.asyncio
    async def test_authenticate_no_credentials(self, auth_middleware):
        """Test authentication with no credentials."""
        mock_request = Mock()
        mock_request.headers = {}
        
        with pytest.raises(AuthenticationError, match="No authentication credentials provided"):
            await auth_middleware.authenticate(mock_request)
    
    @pytest.mark.asyncio
    async def test_authenticate_invalid_bearer_format(self, auth_middleware):
        """Test authentication with invalid Bearer token format."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Invalid format"}
        
        with pytest.raises(AuthenticationError, match="Invalid authentication header format"):
            await auth_middleware.authenticate(mock_request)
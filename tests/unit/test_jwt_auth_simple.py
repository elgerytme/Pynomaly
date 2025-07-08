"""Simple unit tests for JWT authentication without complex dependencies."""

import pytest
from datetime import datetime, UTC
from unittest.mock import MagicMock

from pynomaly.infrastructure.auth.jwt_auth import (
    JWTAuthService,
    UserModel,
    PasswordRotationStrategy,
)
from pynomaly.infrastructure.config.settings import Settings


@pytest.fixture
def settings():
    """Create test settings for JWT service."""
    return Settings(
        secret_key="test-secret-key",
        jwt_algorithm="HS256",
        jwt_expiration=3600,
        auth_enabled=True,
    )


@pytest.fixture
def auth_service(settings):
    """Create JWT auth service for testing."""
    return JWTAuthService(settings)


def test_password_hashing(auth_service):
    """Test password hashing and verification."""
    password = "test-password-123"
    hashed = auth_service.hash_password(password)

    assert hashed != password  # Should be hashed
    assert auth_service.verify_password(password, hashed)  # Should verify correctly
    assert not auth_service.verify_password(
        "wrong-password", hashed
    )  # Should fail wrong password


def test_user_creation(auth_service):
    """Test creating new users."""
    user = auth_service.create_user(
        username="testuser",
        email="test@example.com",
        password="password123",
        full_name="Test User",
        roles=["user"],
    )

    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.full_name == "Test User"
    assert "user" in user.roles
    assert user.is_active
    assert auth_service.verify_password("password123", user.hashed_password)


def test_user_authentication(auth_service):
    """Test user authentication."""
    # Create a user
    user = auth_service.create_user(
        username="testuser", email="test@example.com", password="password123"
    )

    # Test authentication with username
    auth_user = auth_service.authenticate_user("testuser", "password123")
    assert auth_user.id == user.id

    # Test authentication with email
    auth_user = auth_service.authenticate_user("test@example.com", "password123")
    assert auth_user.id == user.id

    # Test failed authentication
    with pytest.raises(Exception):  # AuthenticationError
        auth_service.authenticate_user("testuser", "wrongpassword")


def test_jwt_token_creation_and_validation(auth_service):
    """Test JWT token creation and validation."""
    user = auth_service.create_user(
        username="testuser",
        email="test@example.com",
        password="password123",
        roles=["user", "admin"],
    )

    # Create token
    token_response = auth_service.create_access_token(user)

    assert token_response.access_token
    assert token_response.token_type == "bearer"
    assert token_response.expires_in == 3600
    assert token_response.refresh_token

    # Validate token
    payload = auth_service.decode_token(token_response.access_token)

    assert payload.sub == user.id
    assert payload.type == "access"
    assert "user" in payload.roles
    assert "admin" in payload.roles


def test_api_key_management(auth_service):
    """Test API key creation and management."""
    user = auth_service.create_user(
        username="testuser", email="test@example.com", password="password123"
    )

    # Create API key
    api_key = auth_service.create_api_key(user.id, "test-key")

    assert api_key.startswith("pyn_")
    assert api_key in user.api_keys

    # Authenticate with API key
    auth_user = auth_service.authenticate_api_key(api_key)
    assert auth_user.id == user.id

    # Revoke API key
    success = auth_service.revoke_api_key(api_key)
    assert success
    assert api_key not in user.api_keys

    # Should fail after revocation
    with pytest.raises(Exception):  # AuthenticationError
        auth_service.authenticate_api_key(api_key)


def test_password_rotation(auth_service):
    """Test password rotation functionality."""
    user = auth_service.create_user(
        username="testuser", email="test@example.com", password="oldpassword123"
    )

    # Change password
    success = auth_service.change_password(user.id, "oldpassword123", "newpassword456")
    assert success

    # Should authenticate with new password
    auth_user = auth_service.authenticate_user("testuser", "newpassword456")
    assert auth_user.id == user.id

    # Should fail with old password
    with pytest.raises(Exception):
        auth_service.authenticate_user("testuser", "oldpassword123")


def test_account_lockout(auth_service):
    """Test account lockout after failed login attempts."""
    user = auth_service.create_user(
        username="testuser", email="test@example.com", password="password123"
    )

    # Try failed logins
    for _ in range(5):
        try:
            auth_service.authenticate_user("testuser", "wrongpassword")
        except:
            pass

    # Should be locked out now
    with pytest.raises(Exception) as exc_info:
        auth_service.authenticate_user("testuser", "password123")

    assert "locked" in str(exc_info.value).lower()


def test_token_blacklisting(auth_service):
    """Test token blacklisting functionality."""
    user = auth_service.create_user(
        username="testuser", email="test@example.com", password="password123"
    )

    # Create token
    token_response = auth_service.create_access_token(user)
    token = token_response.access_token

    # Should decode successfully initially
    payload = auth_service.decode_token(token)
    assert payload.sub == user.id

    # Blacklist token
    auth_service.blacklist_token(token)

    # Should fail after blacklisting
    with pytest.raises(Exception):  # AuthenticationError
        auth_service.decode_token(token)


def test_password_rotation_strategy():
    """Test password rotation strategy configuration."""
    strategy = PasswordRotationStrategy(
        enabled=True,
        max_age_days=90,
        force_change_on_first_login=True,
        notify_before_expiry_days=7,
        password_history_count=12,
    )

    assert strategy.enabled
    assert strategy.max_age_days == 90
    assert strategy.force_change_on_first_login
    assert strategy.notify_before_expiry_days == 7
    assert strategy.password_history_count == 12


if __name__ == "__main__":
    pytest.main([__file__])

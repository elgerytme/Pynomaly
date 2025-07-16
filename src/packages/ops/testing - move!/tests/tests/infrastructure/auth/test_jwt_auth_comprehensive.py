"""Comprehensive tests for JWT authentication infrastructure."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from passlib.context import CryptContext

from monorepo.domain.exceptions import AuthenticationError, AuthorizationError
from monorepo.infrastructure.auth.jwt_auth_enhanced import (
    EnhancedJWTAuthService,
    JWKSKey,
    JWKSResponse,
    PasswordRotationStrategy,
    SessionInfo,
    TokenPayload,
    TokenResponse,
    UserModel,
    get_auth,
    init_auth,
)
from monorepo.infrastructure.config import Settings


class TestUserModel:
    """Test user model functionality."""

    def test_user_model_creation(self):
        """Test basic user model creation."""
        user = UserModel(
            id="user123",
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password",
            is_active=True,
            is_superuser=False,
            roles=["user"],
            api_keys=["key1", "key2"],
        )

        assert user.id == "user123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.hashed_password == "hashed_password"
        assert user.is_active is True
        assert user.is_superuser is False
        assert user.roles == ["user"]
        assert user.api_keys == ["key1", "key2"]
        assert user.created_at is not None
        assert user.last_login is None

    def test_user_model_defaults(self):
        """Test user model with default values."""
        user = UserModel(
            id="user123",
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password",
        )

        assert user.full_name is None
        assert user.is_active is True
        assert user.is_superuser is False
        assert user.roles == []
        assert user.api_keys == []
        assert user.created_at is not None
        assert user.last_login is None


class TestTokenPayload:
    """Test token payload functionality."""

    def test_token_payload_creation(self):
        """Test token payload creation."""
        now = datetime.now(UTC)
        expire = now + timedelta(hours=1)

        payload = TokenPayload(
            sub="user123",
            exp=expire,
            iat=now,
            type="access",
            roles=["user", "admin"],
            permissions=["read", "write"],
            kid="1",
        )

        assert payload.sub == "user123"
        assert payload.exp == expire
        assert payload.iat == now
        assert payload.type == "access"
        assert payload.roles == ["user", "admin"]
        assert payload.permissions == ["read", "write"]
        assert payload.kid == "1"

    def test_token_payload_defaults(self):
        """Test token payload with default values."""
        now = datetime.now(UTC)
        expire = now + timedelta(hours=1)

        payload = TokenPayload(
            sub="user123",
            exp=expire,
            iat=now,
        )

        assert payload.type == "access"
        assert payload.roles == []
        assert payload.permissions == []
        assert payload.kid == "1"


class TestTokenResponse:
    """Test token response functionality."""

    def test_token_response_creation(self):
        """Test token response creation."""
        response = TokenResponse(
            access_token="access_token_123",
            token_type="bearer",
            expires_in=3600,
            refresh_token="refresh_token_456",
        )

        assert response.access_token == "access_token_123"
        assert response.token_type == "bearer"
        assert response.expires_in == 3600
        assert response.refresh_token == "refresh_token_456"

    def test_token_response_defaults(self):
        """Test token response with default values."""
        response = TokenResponse(
            access_token="access_token_123",
            expires_in=3600,
        )

        assert response.token_type == "bearer"
        assert response.refresh_token is None


class TestJWKSKey:
    """Test JWKS key functionality."""

    def test_jwks_key_creation(self):
        """Test JWKS key creation."""
        key = JWKSKey(
            kty="RSA",
            use="sig",
            kid="test-key-id",
            n="test_modulus",
            e="test_exponent",
            alg="RS256",
        )

        assert key.kty == "RSA"
        assert key.use == "sig"
        assert key.kid == "test-key-id"
        assert key.n == "test_modulus"
        assert key.e == "test_exponent"
        assert key.alg == "RS256"

    def test_jwks_key_defaults(self):
        """Test JWKS key with default values."""
        key = JWKSKey(
            kid="test-key-id",
            n="test_modulus",
            e="test_exponent",
        )

        assert key.kty == "RSA"
        assert key.use == "sig"
        assert key.alg == "RS256"


class TestPasswordRotationStrategy:
    """Test password rotation strategy."""

    def test_password_rotation_strategy_defaults(self):
        """Test password rotation strategy with defaults."""
        strategy = PasswordRotationStrategy()

        assert strategy.enabled is True
        assert strategy.max_age_days == 90
        assert strategy.force_change_on_first_login is True
        assert strategy.notify_before_expiry_days == 7
        assert strategy.password_history_count == 12

    def test_password_rotation_strategy_custom(self):
        """Test password rotation strategy with custom values."""
        strategy = PasswordRotationStrategy(
            enabled=False,
            max_age_days=60,
            force_change_on_first_login=False,
            notify_before_expiry_days=14,
            password_history_count=5,
        )

        assert strategy.enabled is False
        assert strategy.max_age_days == 60
        assert strategy.force_change_on_first_login is False
        assert strategy.notify_before_expiry_days == 14
        assert strategy.password_history_count == 5


class TestSessionInfo:
    """Test session information functionality."""

    def test_session_info_creation(self):
        """Test session info creation."""
        now = datetime.now(UTC)
        expire = now + timedelta(hours=1)

        session = SessionInfo(
            session_id="session123",
            user_id="user456",
            token_id="token789",
            created_at=now,
            last_activity=now,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            device_info="Desktop",
            is_active=True,
            expires_at=expire,
        )

        assert session.session_id == "session123"
        assert session.user_id == "user456"
        assert session.token_id == "token789"
        assert session.created_at == now
        assert session.last_activity == now
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Mozilla/5.0"
        assert session.device_info == "Desktop"
        assert session.is_active is True
        assert session.expires_at == expire


class TestEnhancedJWTAuthService:
    """Test enhanced JWT authentication service."""

    def test_initialization(self):
        """Test auth service initialization."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        assert service.settings is settings
        assert service.algorithm == "RS256"
        assert service.access_token_expire.total_seconds() == settings.jwt_expiration
        assert service.refresh_token_expire.days == 7
        assert service.current_key_id == "1"
        assert len(service.private_keys) == 1
        assert len(service.public_keys) == 1
        assert len(service.jwks_keys) == 1
        assert isinstance(service.pwd_context, CryptContext)
        assert isinstance(service.password_rotation, PasswordRotationStrategy)

    def test_initialization_with_user_repository(self):
        """Test auth service initialization with user repository."""
        settings = Settings()
        mock_repo = Mock()

        service = EnhancedJWTAuthService(settings, user_repository=mock_repo)

        assert service.user_repository is mock_repo

    def test_key_initialization(self):
        """Test RSA key initialization."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Check that keys were generated
        assert "1" in service.private_keys
        assert "1" in service.public_keys
        assert len(service.jwks_keys) == 1

        # Check key properties
        private_key = service.private_keys["1"]
        public_key = service.public_keys["1"]
        jwks_key = service.jwks_keys[0]

        assert isinstance(private_key, rsa.RSAPrivateKey)
        assert isinstance(public_key, rsa.RSAPublicKey)
        assert jwks_key.kid == "1"
        assert jwks_key.kty == "RSA"

    def test_get_jwks(self):
        """Test JWKS endpoint."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        jwks = service.get_jwks()

        assert isinstance(jwks, JWKSResponse)
        assert len(jwks.keys) == 1
        assert jwks.keys[0].kid == "1"
        assert jwks.keys[0].kty == "RSA"

    def test_key_rotation(self):
        """Test key rotation functionality."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Initial state
        assert service.current_key_id == "1"
        assert len(service.private_keys) == 1

        # Rotate keys
        new_key_id = service.rotate_keys()

        # Check new state
        assert new_key_id == "2"
        assert service.current_key_id == "2"
        assert len(service.private_keys) == 2
        assert len(service.public_keys) == 2
        assert len(service.jwks_keys) == 2

    def test_cleanup_old_keys(self):
        """Test cleanup of old keys."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Generate multiple keys
        service.rotate_keys()  # Key 2
        service.rotate_keys()  # Key 3
        service.rotate_keys()  # Key 4

        assert len(service.private_keys) == 4

        # Clean up old keys, keep only 2
        service.cleanup_old_keys(keep_count=2)

        assert len(service.private_keys) == 2
        assert len(service.public_keys) == 2
        assert len(service.jwks_keys) == 2
        assert "3" in service.private_keys  # Should keep latest 2
        assert "4" in service.private_keys
        assert "1" not in service.private_keys  # Should remove oldest
        assert "2" not in service.private_keys

    def test_password_hashing_and_verification(self):
        """Test password hashing and verification."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        password = "test_password_123"
        hashed = service.hash_password(password)

        # Check hash is different from password
        assert hashed != password
        assert len(hashed) > 20  # Bcrypt hashes are long

        # Check verification
        assert service.verify_password(password, hashed) is True
        assert service.verify_password("wrong_password", hashed) is False

    def test_create_user(self):
        """Test user creation."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            full_name="Test User",
            roles=["user", "admin"],
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.roles == ["user", "admin"]
        assert service.verify_password("password123", user.hashed_password)
        assert user.id in service._users

    def test_create_user_duplicate(self):
        """Test user creation with duplicate username/email."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Create first user
        service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Try to create duplicate username
        with pytest.raises(ValueError, match="User already exists"):
            service.create_user(
                username="testuser",
                email="other@example.com",
                password="password123",
            )

        # Try to create duplicate email
        with pytest.raises(ValueError, match="User already exists"):
            service.create_user(
                username="otheruser",
                email="test@example.com",
                password="password123",
            )

    def test_create_access_token(self):
        """Test access token creation."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"],
        )

        token_response = service.create_access_token(
            user,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        assert isinstance(token_response, TokenResponse)
        assert token_response.access_token is not None
        assert token_response.refresh_token is not None
        assert token_response.token_type == "bearer"
        assert token_response.expires_in == int(
            service.access_token_expire.total_seconds()
        )

    def test_decode_token(self):
        """Test token decoding."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"],
        )

        token_response = service.create_access_token(user)
        payload = service.decode_token(token_response.access_token)

        assert isinstance(payload, TokenPayload)
        assert payload.sub == user.id
        assert payload.type == "access"
        assert payload.roles == ["user"]

    def test_decode_token_expired(self):
        """Test decoding expired token."""
        settings = Settings()
        settings.jwt_expiration = 1  # 1 second
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        token_response = service.create_access_token(user)

        # Wait for token to expire
        time.sleep(2)

        with pytest.raises(AuthenticationError, match="Token has expired"):
            service.decode_token(token_response.access_token)

    def test_decode_token_invalid(self):
        """Test decoding invalid token."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        with pytest.raises(AuthenticationError, match="Invalid token"):
            service.decode_token("invalid_token")

    def test_decode_token_blacklisted(self):
        """Test decoding blacklisted token."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        token_response = service.create_access_token(user)

        # Blacklist the token
        service.blacklist_token(token_response.access_token)

        with pytest.raises(AuthenticationError, match="Token has been revoked"):
            service.decode_token(token_response.access_token)

    def test_authenticate_user_success(self):
        """Test successful user authentication."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Test authentication with username
        authenticated_user = asyncio.run(
            service.authenticate_user("testuser", "password123")
        )
        assert authenticated_user.id == user.id

        # Test authentication with email
        authenticated_user = asyncio.run(
            service.authenticate_user("test@example.com", "password123")
        )
        assert authenticated_user.id == user.id

    def test_authenticate_user_invalid_credentials(self):
        """Test authentication with invalid credentials."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Test invalid username
        with pytest.raises(AuthenticationError, match="Invalid username or password"):
            asyncio.run(service.authenticate_user("wronguser", "password123"))

        # Test invalid password
        with pytest.raises(AuthenticationError, match="Invalid username or password"):
            asyncio.run(service.authenticate_user("testuser", "wrongpassword"))

    def test_authenticate_user_inactive(self):
        """Test authentication with inactive user."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Deactivate user
        user.is_active = False
        service._users[user.id] = user

        with pytest.raises(AuthenticationError, match="User account is inactive"):
            asyncio.run(service.authenticate_user("testuser", "password123"))

    def test_authenticate_user_account_lockout(self):
        """Test account lockout after failed attempts."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Make 5 failed attempts
        for _ in range(5):
            try:
                asyncio.run(service.authenticate_user("testuser", "wrongpassword"))
            except AuthenticationError:
                pass

        # Account should be locked
        with pytest.raises(AuthenticationError, match="Account temporarily locked"):
            asyncio.run(service.authenticate_user("testuser", "password123"))

    def test_authenticate_api_key_success(self):
        """Test successful API key authentication."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        api_key = service.create_api_key(user.id, "test-key")
        authenticated_user = service.authenticate_api_key(api_key)

        assert authenticated_user.id == user.id

    def test_authenticate_api_key_invalid(self):
        """Test authentication with invalid API key."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        with pytest.raises(AuthenticationError, match="Invalid API key"):
            service.authenticate_api_key("invalid_key")

    def test_get_current_user(self):
        """Test getting current user from token."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        token_response = service.create_access_token(user)
        current_user = service.get_current_user(token_response.access_token)

        assert current_user.id == user.id

    def test_refresh_access_token(self):
        """Test refreshing access token."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        token_response = service.create_access_token(user)
        new_token_response = service.refresh_access_token(token_response.refresh_token)

        assert isinstance(new_token_response, TokenResponse)
        assert new_token_response.access_token != token_response.access_token
        assert new_token_response.refresh_token is not None

    def test_refresh_access_token_invalid_type(self):
        """Test refreshing with invalid token type."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        token_response = service.create_access_token(user)

        with pytest.raises(AuthenticationError, match="Invalid token type"):
            service.refresh_access_token(
                token_response.access_token
            )  # Use access token instead of refresh

    def test_check_permissions_superuser(self):
        """Test permissions check for superuser."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="admin",
            email="admin@example.com",
            password="password123",
        )
        user.is_superuser = True
        service._users[user.id] = user

        # Superuser should have all permissions
        assert service.check_permissions(user, ["any", "permission"]) is True

    def test_check_permissions_with_roles(self):
        """Test permissions check with user roles."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"],
        )

        # User should have basic permissions
        assert service.check_permissions(user, ["detectors:read"]) is True
        assert service.check_permissions(user, ["users:delete"]) is False

    def test_require_permissions_success(self):
        """Test requiring permissions successfully."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"],
        )

        # Should not raise exception
        service.require_permissions(user, ["detectors:read"])

    def test_require_permissions_failure(self):
        """Test requiring permissions with insufficient permissions."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["viewer"],
        )

        with pytest.raises(AuthorizationError, match="User lacks required permissions"):
            service.require_permissions(user, ["users:delete"])

    def test_api_key_management(self):
        """Test API key creation and revocation."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Create API key
        api_key = service.create_api_key(user.id, "test-key")

        assert api_key.startswith("pyn_")
        assert api_key in service._api_keys
        assert api_key in user.api_keys

        # Revoke API key
        success = service.revoke_api_key(api_key)

        assert success is True
        assert api_key not in service._api_keys
        assert api_key not in user.api_keys

    def test_password_reset_token_creation(self):
        """Test password reset token creation."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        reset_token = service.create_password_reset_token(user.email)

        assert reset_token is not None
        assert len(reset_token) > 20  # JWT tokens are long

    def test_password_reset_token_invalid_email(self):
        """Test password reset token with invalid email."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        with pytest.raises(ValueError, match="User not found"):
            service.create_password_reset_token("nonexistent@example.com")

    def test_password_reset_with_token(self):
        """Test password reset using token."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        reset_token = service.create_password_reset_token(user.email)

        # Reset password
        service.reset_password_with_token(reset_token, "newpassword456")

        # Check password was changed
        updated_user = service._users[user.id]
        assert service.verify_password("newpassword456", updated_user.hashed_password)
        assert not service.verify_password("password123", updated_user.hashed_password)

    def test_password_reset_with_expired_token(self):
        """Test password reset with expired token."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Create token that expires immediately
        with patch(
            "monorepo.infrastructure.auth.jwt_auth_enhanced.timedelta"
        ) as mock_timedelta:
            mock_timedelta.return_value = timedelta(minutes=-1)  # Already expired
            reset_token = service.create_password_reset_token(user.email)

        with pytest.raises(ValueError, match="Password reset token has expired"):
            service.reset_password_with_token(reset_token, "newpassword456")

    def test_session_management(self):
        """Test session management functionality."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Create token (creates session)
        token_response = service.create_access_token(
            user,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        # Check session was created
        sessions = service.get_user_sessions(user.id)
        assert len(sessions) == 1
        assert sessions[0]["user_id"] == user.id
        assert sessions[0]["ip_address"] == "192.168.1.1"
        assert sessions[0]["is_active"] is True

    def test_session_limits_enforcement(self):
        """Test session limits enforcement."""
        settings = Settings()
        settings.security.max_concurrent_sessions = 2
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Create 3 sessions (should remove oldest)
        for i in range(3):
            service.create_access_token(user)
            time.sleep(0.01)  # Ensure different timestamps

        # Should only have 2 active sessions
        sessions = service.get_user_sessions(user.id)
        assert len(sessions) == 2

    def test_session_invalidation(self):
        """Test session invalidation."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Create multiple sessions
        for _ in range(3):
            service.create_access_token(user)

        # Invalidate all sessions
        count = service.invalidate_user_sessions(user.id)

        assert count == 3
        sessions = service.get_user_sessions(user.id)
        assert len(sessions) == 0

    def test_device_info_extraction(self):
        """Test device information extraction."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Test mobile device
        mobile_ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
        device_info = service._extract_device_info(mobile_ua)
        assert device_info == "Mobile"

        # Test desktop device
        desktop_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        device_info = service._extract_device_info(desktop_ua)
        assert device_info == "Desktop"

        # Test tablet device
        tablet_ua = "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
        device_info = service._extract_device_info(tablet_ua)
        assert device_info == "Tablet"

        # Test unknown device
        unknown_ua = "Unknown User Agent"
        device_info = service._extract_device_info(unknown_ua)
        assert device_info == "Unknown"

        # Test None user agent
        device_info = service._extract_device_info(None)
        assert device_info is None

    def test_concurrent_access_thread_safety(self):
        """Test thread safety of auth service."""
        import threading

        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Create test user
        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        results = []
        errors = []

        def auth_worker():
            try:
                for i in range(10):
                    # Authenticate user
                    auth_user = asyncio.run(
                        service.authenticate_user("testuser", "password123")
                    )

                    # Create token
                    token_response = service.create_access_token(auth_user)

                    # Decode token
                    payload = service.decode_token(token_response.access_token)

                    results.append(payload.sub)

                    time.sleep(0.001)  # Small delay to encourage concurrency
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=auth_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 iterations
        assert all(result == user.id for result in results)


class TestGlobalAuthManagement:
    """Test global auth management functions."""

    def test_init_auth(self):
        """Test initializing global auth service."""
        settings = Settings()
        service = init_auth(settings)

        assert isinstance(service, EnhancedJWTAuthService)
        assert service.settings is settings

    def test_get_auth(self):
        """Test getting global auth service."""
        settings = Settings()

        # Initially should be None
        assert get_auth() is None

        # Initialize auth
        service = init_auth(settings)

        # Should return initialized service
        assert get_auth() is service

    def test_auth_service_persistence(self):
        """Test that auth service persists across calls."""
        settings = Settings()
        service1 = init_auth(settings)
        service2 = get_auth()

        assert service1 is service2


class TestAuthServiceIntegration:
    """Test auth service integration scenarios."""

    def test_end_to_end_authentication_flow(self):
        """Test complete authentication flow."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Create user
        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"],
        )

        # Authenticate user
        auth_user = asyncio.run(service.authenticate_user("testuser", "password123"))
        assert auth_user.id == user.id

        # Create access token
        token_response = service.create_access_token(
            auth_user,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        # Verify token
        payload = service.decode_token(token_response.access_token)
        assert payload.sub == user.id

        # Get current user from token
        current_user = service.get_current_user(token_response.access_token)
        assert current_user.id == user.id

        # Check permissions
        assert service.check_permissions(current_user, ["detectors:read"]) is True

        # Refresh token
        new_token_response = service.refresh_access_token(token_response.refresh_token)
        assert new_token_response.access_token != token_response.access_token

        # Blacklist original token
        service.blacklist_token(token_response.access_token)

        # Verify token is blacklisted
        with pytest.raises(AuthenticationError, match="Token has been revoked"):
            service.decode_token(token_response.access_token)

    def test_api_key_authentication_flow(self):
        """Test API key authentication flow."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Create user
        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=["user"],
        )

        # Create API key
        api_key = service.create_api_key(user.id, "test-api-key")

        # Authenticate with API key
        auth_user = service.authenticate_api_key(api_key)
        assert auth_user.id == user.id

        # Check permissions
        assert service.check_permissions(auth_user, ["detectors:read"]) is True

        # Revoke API key
        success = service.revoke_api_key(api_key)
        assert success is True

        # Verify API key is revoked
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            service.authenticate_api_key(api_key)

    def test_password_reset_flow(self):
        """Test password reset flow."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Create user
        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Create password reset token
        reset_token = service.create_password_reset_token(user.email)

        # Reset password
        service.reset_password_with_token(reset_token, "newpassword456")

        # Verify old password doesn't work
        with pytest.raises(AuthenticationError, match="Invalid username or password"):
            asyncio.run(service.authenticate_user("testuser", "password123"))

        # Verify new password works
        auth_user = asyncio.run(service.authenticate_user("testuser", "newpassword456"))
        assert auth_user.id == user.id

    def test_session_management_flow(self):
        """Test session management flow."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Create user
        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        # Create multiple sessions
        sessions = []
        for i in range(3):
            token_response = service.create_access_token(
                user,
                ip_address=f"192.168.1.{i+1}",
                user_agent=f"Browser {i+1}",
            )
            sessions.append(token_response)

        # Check sessions
        active_sessions = service.get_user_sessions(user.id)
        assert len(active_sessions) == 3

        # Update session activity
        service.update_session_activity(sessions[0].access_token)

        # Invalidate all sessions
        count = service.invalidate_user_sessions(user.id)
        assert count == 3

        # Verify sessions are invalidated
        active_sessions = service.get_user_sessions(user.id)
        assert len(active_sessions) == 0

    def test_key_rotation_flow(self):
        """Test key rotation flow."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Create user and token with key 1
        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        token1 = service.create_access_token(user)

        # Rotate keys
        new_key_id = service.rotate_keys()
        assert new_key_id == "2"

        # Create token with key 2
        token2 = service.create_access_token(user)

        # Both tokens should be valid
        payload1 = service.decode_token(token1.access_token)
        payload2 = service.decode_token(token2.access_token)

        assert payload1.kid == "1"
        assert payload2.kid == "2"

        # JWKS should contain both keys
        jwks = service.get_jwks()
        assert len(jwks.keys) == 2
        assert any(key.kid == "1" for key in jwks.keys)
        assert any(key.kid == "2" for key in jwks.keys)

        # Clean up old keys
        service.cleanup_old_keys(keep_count=1)

        # Key 1 should be removed
        jwks = service.get_jwks()
        assert len(jwks.keys) == 1
        assert jwks.keys[0].kid == "2"

        # Token with key 1 should now be invalid
        with pytest.raises(AuthenticationError, match="Invalid key ID"):
            service.decode_token(token1.access_token)

        # Token with key 2 should still be valid
        payload2 = service.decode_token(token2.access_token)
        assert payload2.kid == "2"

    def test_error_handling_robustness(self):
        """Test auth service robustness under various error conditions."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Test with None values
        with pytest.raises(AuthenticationError):
            service.get_current_user(None)

        # Test with malformed tokens
        with pytest.raises(AuthenticationError):
            service.decode_token("malformed.token")

        # Test with empty strings
        with pytest.raises(AuthenticationError):
            service.decode_token("")

        # Test password reset with empty token
        with pytest.raises(ValueError):
            service.reset_password_with_token("", "newpassword")

        # Test API key authentication with empty key
        with pytest.raises(AuthenticationError):
            service.authenticate_api_key("")

        # Test user creation with empty values
        with pytest.raises(ValueError):
            service.create_user("", "", "")

        # Service should remain functional after errors
        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )
        assert user.username == "testuser"

    def test_performance_under_load(self):
        """Test auth service performance under load."""
        settings = Settings()
        service = EnhancedJWTAuthService(settings)

        # Create user
        user = service.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
        )

        start_time = time.time()

        # Perform many authentication operations
        for i in range(100):
            # Authenticate user
            auth_user = asyncio.run(
                service.authenticate_user("testuser", "password123")
            )

            # Create token
            token_response = service.create_access_token(auth_user)

            # Decode token
            payload = service.decode_token(token_response.access_token)

            # Verify user
            current_user = service.get_current_user(token_response.access_token)
            assert current_user.id == user.id

        end_time = time.time()
        duration = end_time - start_time

        # Should complete within reasonable time (less than 5 seconds)
        assert duration < 5.0

        # Check that sessions were managed properly
        sessions = service.get_user_sessions(user.id)
        assert len(sessions) <= 100  # Should have session limits

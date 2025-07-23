"""
Unit tests for AuthService.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, Mock, patch

from enterprise_auth.application.services.auth_service import (
    AuthService, AuthConfig, AuthenticationError
)
from enterprise_auth.application.dto.auth_dto import (
    LoginRequest, RegisterRequest, RefreshTokenRequest,
    SAMLAuthRequest, OAuth2AuthRequest, MFASetupRequest, MFAVerifyRequest
)
from enterprise_auth.domain.entities.user import User, UserSession, UserStatus, UserRole, AuthProvider
from enterprise_auth.domain.entities.tenant import Tenant, TenantStatus, TenantPlan, TenantFeature


class TestAuthConfig:
    """Test cases for AuthConfig."""
    
    def test_auth_config_defaults(self):
        """Test AuthConfig with default values."""
        config = AuthConfig(jwt_secret_key="test-secret")
        
        assert config.jwt_secret_key == "test-secret"
        assert config.jwt_algorithm == "HS256"
        assert config.access_token_expire_minutes == 60
        assert config.refresh_token_expire_days == 30
        assert config.min_password_length == 8
        assert config.require_password_uppercase is True
        assert config.require_password_lowercase is True
        assert config.require_password_numbers is True
        assert config.require_password_special is True
        assert config.max_failed_attempts == 5
        assert config.lockout_duration_minutes == 30
        assert config.max_concurrent_sessions == 5
        assert config.session_timeout_minutes == 480
        assert config.mfa_issuer_name == "anomaly_detection Enterprise"
        assert config.mfa_token_lifetime_seconds == 300
        
    def test_auth_config_custom_values(self):
        """Test AuthConfig with custom values."""
        config = AuthConfig(
            jwt_secret_key="custom-secret",
            jwt_algorithm="RS256",
            access_token_expire_minutes=120,
            refresh_token_expire_days=60,
            min_password_length=12,
            require_password_uppercase=False,
            max_failed_attempts=3,
            max_concurrent_sessions=10,
            mfa_issuer_name="Custom Corp"
        )
        
        assert config.jwt_secret_key == "custom-secret"
        assert config.jwt_algorithm == "RS256"
        assert config.access_token_expire_minutes == 120
        assert config.refresh_token_expire_days == 60
        assert config.min_password_length == 12
        assert config.require_password_uppercase is False
        assert config.max_failed_attempts == 3
        assert config.max_concurrent_sessions == 10
        assert config.mfa_issuer_name == "Custom Corp"


class TestAuthService:
    """Test cases for AuthService."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AuthConfig(jwt_secret_key="test-secret-key")
    
    @pytest.fixture
    def mock_repositories(self):
        """Create mock repositories."""
        return {
            'user_repo': AsyncMock(),
            'tenant_repo': AsyncMock(),
            'session_repo': AsyncMock()
        }
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        return {
            'password_service': Mock(),
            'mfa_service': Mock()
        }
    
    @pytest.fixture
    def auth_service(self, config, mock_repositories, mock_services):
        """Create AuthService instance with mocks."""
        return AuthService(
            user_repository=mock_repositories['user_repo'],
            tenant_repository=mock_repositories['tenant_repo'],
            session_repository=mock_repositories['session_repo'],
            password_service=mock_services['password_service'],
            mfa_service=mock_services['mfa_service'],
            config=config
        )
    
    @pytest.fixture
    def sample_user(self):
        """Create sample user for testing."""
        return User(
            id=uuid4(),
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            password_hash="hashed_password",
            status=UserStatus.ACTIVE,
            auth_provider=AuthProvider.LOCAL,
            roles={UserRole.VIEWER}
        )
    
    @pytest.fixture
    def sample_tenant(self):
        """Create sample tenant for testing."""
        return Tenant(
            id=uuid4(),
            name="Test Company",
            slug="test-company",
            admin_email="admin@test.com",
            status=TenantStatus.ACTIVE,
            plan=TenantPlan.PROFESSIONAL,
            max_users=100,
            current_user_count=10
        )
    
    # Test authenticate_local method
    
    @pytest.mark.asyncio
    async def test_authenticate_local_success(self, auth_service, mock_repositories, mock_services, sample_user, sample_tenant):
        """Test successful local authentication."""
        # Setup mocks
        mock_repositories['user_repo'].get_by_email.return_value = sample_user
        mock_repositories['tenant_repo'].get_by_id.return_value = sample_tenant
        mock_services['password_service'].verify_password.return_value = True
        mock_repositories['session_repo'].get_active_sessions.return_value = []
        mock_repositories['session_repo'].create.return_value = None
        
        request = LoginRequest(
            email="test@example.com",
            password="password123"
        )
        
        with patch.object(auth_service, '_generate_access_token', return_value="access_token_123"):
            with patch.object(auth_service, '_generate_refresh_token', return_value="refresh_token_123"):
                response = await auth_service.authenticate_local(request, "192.168.1.1")
        
        assert response.success is True
        assert response.user_id == sample_user.id
        assert response.tenant_id == sample_user.tenant_id
        assert response.access_token == "access_token_123"
        assert response.refresh_token == "refresh_token_123"
        assert response.user_info["email"] == "test@example.com"
        
        # Verify method calls
        mock_repositories['user_repo'].get_by_email.assert_called_once_with("test@example.com")
        mock_repositories['tenant_repo'].get_by_id.assert_called_once_with(sample_user.tenant_id)
        mock_services['password_service'].verify_password.assert_called_once_with("password123", "hashed_password")
    
    @pytest.mark.asyncio
    async def test_authenticate_local_user_not_found(self, auth_service, mock_repositories):
        """Test authentication with non-existent user."""
        mock_repositories['user_repo'].get_by_email.return_value = None
        
        request = LoginRequest(
            email="nonexistent@example.com",
            password="password123"
        )
        
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            await auth_service.authenticate_local(request)
    
    @pytest.mark.asyncio
    async def test_authenticate_local_inactive_tenant(self, auth_service, mock_repositories, sample_user):
        """Test authentication with inactive tenant."""
        inactive_tenant = Tenant(
            id=uuid4(),
            name="Inactive Company",
            slug="inactive-company",
            admin_email="admin@inactive.com",
            status=TenantStatus.SUSPENDED
        )
        
        mock_repositories['user_repo'].get_by_email.return_value = sample_user
        mock_repositories['tenant_repo'].get_by_id.return_value = inactive_tenant
        
        request = LoginRequest(
            email="test@example.com",
            password="password123"
        )
        
        with pytest.raises(AuthenticationError, match="Account suspended"):
            await auth_service.authenticate_local(request)
    
    @pytest.mark.asyncio
    async def test_authenticate_local_inactive_user(self, auth_service, mock_repositories, sample_tenant):
        """Test authentication with inactive user."""
        inactive_user = User(
            id=uuid4(),
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            status=UserStatus.SUSPENDED
        )
        
        mock_repositories['user_repo'].get_by_email.return_value = inactive_user
        mock_repositories['tenant_repo'].get_by_id.return_value = sample_tenant
        
        request = LoginRequest(
            email="test@example.com",
            password="password123"
        )
        
        with pytest.raises(AuthenticationError, match="Account suspended"):
            await auth_service.authenticate_local(request)
    
    @pytest.mark.asyncio
    async def test_authenticate_local_locked_user(self, auth_service, mock_repositories, sample_tenant):
        """Test authentication with locked user."""
        locked_user = User(
            id=uuid4(),
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            status=UserStatus.LOCKED
        )
        
        mock_repositories['user_repo'].get_by_email.return_value = locked_user
        mock_repositories['tenant_repo'].get_by_id.return_value = sample_tenant
        
        request = LoginRequest(
            email="test@example.com",
            password="password123"
        )
        
        with pytest.raises(AuthenticationError, match="Account locked"):
            await auth_service.authenticate_local(request)
    
    @pytest.mark.asyncio
    async def test_authenticate_local_invalid_password(self, auth_service, mock_repositories, mock_services, sample_user, sample_tenant):
        """Test authentication with invalid password."""
        mock_repositories['user_repo'].get_by_email.return_value = sample_user
        mock_repositories['tenant_repo'].get_by_id.return_value = sample_tenant
        mock_services['password_service'].verify_password.return_value = False
        mock_repositories['user_repo'].update.return_value = None
        
        request = LoginRequest(
            email="test@example.com",
            password="wrongpassword"
        )
        
        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            await auth_service.authenticate_local(request)
        
        # Verify failed login was handled
        mock_repositories['user_repo'].update.assert_called_once_with(sample_user)
        assert sample_user.failed_login_attempts == 1
    
    @pytest.mark.asyncio
    async def test_authenticate_local_mfa_required(self, auth_service, mock_repositories, mock_services, sample_tenant):
        """Test authentication requiring MFA."""
        mfa_user = User(
            id=uuid4(),
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            password_hash="hashed_password",
            status=UserStatus.ACTIVE,
            mfa_enabled=True
        )
        
        mock_repositories['user_repo'].get_by_email.return_value = mfa_user
        mock_repositories['tenant_repo'].get_by_id.return_value = sample_tenant
        mock_services['password_service'].verify_password.return_value = True
        
        request = LoginRequest(
            email="test@example.com",
            password="password123"
        )
        
        response = await auth_service.authenticate_local(request)
        
        assert response.success is False
        assert response.requires_mfa is True
        assert "MFA verification required" in response.message
    
    @pytest.mark.asyncio
    async def test_authenticate_local_mfa_success(self, auth_service, mock_repositories, mock_services, sample_tenant):
        """Test authentication with valid MFA code."""
        mfa_user = User(
            id=uuid4(),
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            password_hash="hashed_password",
            status=UserStatus.ACTIVE,
            mfa_enabled=True,
            mfa_secret="mfa_secret"
        )
        
        mock_repositories['user_repo'].get_by_email.return_value = mfa_user
        mock_repositories['tenant_repo'].get_by_id.return_value = sample_tenant
        mock_services['password_service'].verify_password.return_value = True
        mock_services['mfa_service'].verify_code.return_value = True
        mock_repositories['session_repo'].get_active_sessions.return_value = []
        mock_repositories['session_repo'].create.return_value = None
        
        request = LoginRequest(
            email="test@example.com",
            password="password123",
            mfa_code="123456"
        )
        
        with patch.object(auth_service, '_generate_access_token', return_value="access_token_123"):
            with patch.object(auth_service, '_generate_refresh_token', return_value="refresh_token_123"):
                response = await auth_service.authenticate_local(request)
        
        assert response.success is True
        assert response.access_token == "access_token_123"
        mock_services['mfa_service'].verify_code.assert_called_once_with("mfa_secret", "123456")
    
    @pytest.mark.asyncio
    async def test_authenticate_local_invalid_mfa(self, auth_service, mock_repositories, mock_services, sample_tenant):
        """Test authentication with invalid MFA code."""
        mfa_user = User(
            id=uuid4(),
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            password_hash="hashed_password",
            status=UserStatus.ACTIVE,
            mfa_enabled=True,
            mfa_secret="mfa_secret"
        )
        
        mock_repositories['user_repo'].get_by_email.return_value = mfa_user
        mock_repositories['tenant_repo'].get_by_id.return_value = sample_tenant
        mock_services['password_service'].verify_password.return_value = True
        mock_services['mfa_service'].verify_code.return_value = False
        
        request = LoginRequest(
            email="test@example.com",
            password="password123",
            mfa_code="invalid"
        )
        
        with pytest.raises(AuthenticationError, match="Invalid MFA code"):
            await auth_service.authenticate_local(request)
    
    # Test register_user method
    
    @pytest.mark.asyncio
    async def test_register_user_success(self, auth_service, mock_repositories, mock_services, sample_tenant):
        """Test successful user registration."""
        mock_repositories['user_repo'].get_by_email.return_value = None
        mock_repositories['tenant_repo'].get_by_id.return_value = sample_tenant
        mock_services['password_service'].hash_password.return_value = "hashed_password"
        
        new_user = User(
            id=uuid4(),
            tenant_id=sample_tenant.id,
            email="newuser@example.com",
            username="newuser",
            first_name="New",
            last_name="User",
            password_hash="hashed_password"
        )
        mock_repositories['user_repo'].create.return_value = new_user
        mock_repositories['tenant_repo'].update.return_value = None
        
        request = RegisterRequest(
            tenant_id=sample_tenant.id,
            email="newuser@example.com",
            password="Password123!",
            first_name="New",
            last_name="User"
        )
        
        with patch.object(auth_service, '_send_verification_email') as mock_send_email:
            response = await auth_service.register_user(request)
        
        assert response.success is True
        assert response.user_id == new_user.id
        assert "verification instructions" in response.message
        
        # Verify method calls
        mock_repositories['user_repo'].get_by_email.assert_called_once_with("newuser@example.com")
        mock_repositories['tenant_repo'].get_by_id.assert_called_once_with(sample_tenant.id)
        mock_services['password_service'].hash_password.assert_called_once_with("Password123!")
        mock_repositories['user_repo'].create.assert_called_once()
        mock_send_email.assert_called_once_with(new_user)
    
    @pytest.mark.asyncio
    async def test_register_user_already_exists(self, auth_service, mock_repositories, sample_user):
        """Test registration with existing email."""
        mock_repositories['user_repo'].get_by_email.return_value = sample_user
        
        request = RegisterRequest(
            tenant_id=uuid4(),
            email="test@example.com",
            password="Password123!",
            first_name="Test",
            last_name="User"
        )
        
        with pytest.raises(AuthenticationError, match="User with this email already exists"):
            await auth_service.register_user(request)
    
    @pytest.mark.asyncio
    async def test_register_user_invalid_tenant(self, auth_service, mock_repositories):
        """Test registration with invalid tenant."""
        mock_repositories['user_repo'].get_by_email.return_value = None
        mock_repositories['tenant_repo'].get_by_id.return_value = None
        
        request = RegisterRequest(
            tenant_id=uuid4(),
            email="newuser@example.com",
            password="Password123!",
            first_name="New",
            last_name="User"
        )
        
        with pytest.raises(AuthenticationError, match="Invalid tenant"):
            await auth_service.register_user(request)
    
    @pytest.mark.asyncio
    async def test_register_user_tenant_limit_exceeded(self, auth_service, mock_repositories):
        """Test registration when tenant user limit is exceeded."""
        full_tenant = Tenant(
            id=uuid4(),
            name="Full Company",
            slug="full-company",
            admin_email="admin@full.com",
            status=TenantStatus.ACTIVE,
            max_users=10,
            current_user_count=10  # At limit
        )
        
        mock_repositories['user_repo'].get_by_email.return_value = None
        mock_repositories['tenant_repo'].get_by_id.return_value = full_tenant
        
        request = RegisterRequest(
            tenant_id=full_tenant.id,
            email="newuser@example.com",
            password="Password123!",
            first_name="New",
            last_name="User"
        )
        
        with pytest.raises(AuthenticationError, match="Tenant user limit exceeded"):
            await auth_service.register_user(request)
    
    @pytest.mark.asyncio
    async def test_register_user_weak_password(self, auth_service, mock_repositories, sample_tenant):
        """Test registration with weak password."""
        mock_repositories['user_repo'].get_by_email.return_value = None
        mock_repositories['tenant_repo'].get_by_id.return_value = sample_tenant
        
        request = RegisterRequest(
            tenant_id=sample_tenant.id,
            email="newuser@example.com",
            password="weak",  # Doesn't meet policy
            first_name="New",
            last_name="User"
        )
        
        with pytest.raises(AuthenticationError, match="Password does not meet policy requirements"):
            await auth_service.register_user(request)
    
    # Test refresh_token method
    
    @pytest.mark.asyncio
    async def test_refresh_token_success(self, auth_service, mock_repositories, sample_user):
        """Test successful token refresh."""
        session = UserSession(
            id=uuid4(),
            user_id=sample_user.id,
            tenant_id=sample_user.tenant_id,
            session_token="old_access_token",
            refresh_token="refresh_token_123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
        mock_repositories['session_repo'].get_by_refresh_token.return_value = session
        mock_repositories['user_repo'].get_by_id.return_value = sample_user
        mock_repositories['session_repo'].update.return_value = None
        
        request = RefreshTokenRequest(refresh_token="refresh_token_123")
        
        with patch.object(auth_service, '_generate_access_token', return_value="new_access_token"):
            response = await auth_service.refresh_token(request)
        
        assert response.success is True
        assert response.user_id == sample_user.id
        assert response.access_token == "new_access_token"
        assert response.refresh_token == "refresh_token_123"  # Same refresh token
        
        mock_repositories['session_repo'].get_by_refresh_token.assert_called_once_with("refresh_token_123")
        mock_repositories['user_repo'].get_by_id.assert_called_once_with(sample_user.id)
    
    @pytest.mark.asyncio
    async def test_refresh_token_invalid_token(self, auth_service, mock_repositories):
        """Test token refresh with invalid refresh token."""
        mock_repositories['session_repo'].get_by_refresh_token.return_value = None
        
        request = RefreshTokenRequest(refresh_token="invalid_token")
        
        with pytest.raises(AuthenticationError, match="Invalid refresh token"):
            await auth_service.refresh_token(request)
    
    @pytest.mark.asyncio
    async def test_refresh_token_expired_session(self, auth_service, mock_repositories, sample_user):
        """Test token refresh with expired session."""
        expired_session = UserSession(
            id=uuid4(),
            user_id=sample_user.id,
            tenant_id=sample_user.tenant_id,
            session_token="old_access_token",
            refresh_token="refresh_token_123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() - timedelta(days=1)  # Expired
        )
        
        mock_repositories['session_repo'].get_by_refresh_token.return_value = expired_session
        
        request = RefreshTokenRequest(refresh_token="refresh_token_123")
        
        with pytest.raises(AuthenticationError, match="Invalid refresh token"):
            await auth_service.refresh_token(request)
    
    # Test logout method
    
    @pytest.mark.asyncio
    async def test_logout_success(self, auth_service, mock_repositories):
        """Test successful logout."""
        session = UserSession(
            id=uuid4(),
            user_id=uuid4(),
            tenant_id=uuid4(),
            session_token="session_token_123",
            refresh_token="refresh_token_123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
        mock_repositories['session_repo'].get_by_token.return_value = session
        mock_repositories['session_repo'].update.return_value = None
        
        result = await auth_service.logout("session_token_123")
        
        assert result is True
        assert session.is_active is False
        assert session.revoked_at is not None
        mock_repositories['session_repo'].update.assert_called_once_with(session)
    
    @pytest.mark.asyncio
    async def test_logout_session_not_found(self, auth_service, mock_repositories):
        """Test logout with non-existent session."""
        mock_repositories['session_repo'].get_by_token.return_value = None
        
        result = await auth_service.logout("nonexistent_token")
        
        assert result is True  # Should still return True
    
    # Test setup_mfa method
    
    @pytest.mark.asyncio
    async def test_setup_mfa_success(self, auth_service, mock_repositories, mock_services, sample_user):
        """Test successful MFA setup."""
        mock_repositories['user_repo'].get_by_id.return_value = sample_user
        mock_services['mfa_service'].generate_secret.return_value = "mfa_secret_123"
        mock_services['mfa_service'].get_qr_code_url.return_value = "https://example.com/qr"
        mock_services['mfa_service'].generate_backup_codes.return_value = ["code1", "code2"]
        mock_repositories['user_repo'].update.return_value = None
        
        request = MFASetupRequest()
        
        response = await auth_service.setup_mfa(sample_user.id, request)
        
        assert response["secret"] == "mfa_secret_123"
        assert response["qr_code_url"] == "https://example.com/qr"
        assert response["backup_codes"] == ["code1", "code2"]
        assert "instructions" in response
        
        # Verify user was updated
        assert sample_user.mfa_secret == "mfa_secret_123"
        assert sample_user.backup_codes == ["code1", "code2"]
        mock_repositories['user_repo'].update.assert_called_once_with(sample_user)
    
    @pytest.mark.asyncio
    async def test_setup_mfa_user_not_found(self, auth_service, mock_repositories):
        """Test MFA setup with non-existent user."""
        mock_repositories['user_repo'].get_by_id.return_value = None
        
        request = MFASetupRequest()
        
        with pytest.raises(AuthenticationError, match="User not found"):
            await auth_service.setup_mfa(uuid4(), request)
    
    # Test verify_mfa_setup method
    
    @pytest.mark.asyncio
    async def test_verify_mfa_setup_success(self, auth_service, mock_repositories, mock_services):
        """Test successful MFA verification and enablement."""
        user_with_mfa_secret = User(
            id=uuid4(),
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            mfa_secret="mfa_secret_123",
            mfa_enabled=False
        )
        
        mock_repositories['user_repo'].get_by_id.return_value = user_with_mfa_secret
        mock_services['mfa_service'].verify_code.return_value = True
        mock_repositories['user_repo'].update.return_value = None
        
        request = MFAVerifyRequest(code="123456")
        
        result = await auth_service.verify_mfa_setup(user_with_mfa_secret.id, request)
        
        assert result is True
        assert user_with_mfa_secret.mfa_enabled is True
        mock_services['mfa_service'].verify_code.assert_called_once_with("mfa_secret_123", "123456")
        mock_repositories['user_repo'].update.assert_called_once_with(user_with_mfa_secret)
    
    @pytest.mark.asyncio
    async def test_verify_mfa_setup_invalid_code(self, auth_service, mock_repositories, mock_services):
        """Test MFA verification with invalid code."""
        user_with_mfa_secret = User(
            id=uuid4(),
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            mfa_secret="mfa_secret_123",
            mfa_enabled=False
        )
        
        mock_repositories['user_repo'].get_by_id.return_value = user_with_mfa_secret
        mock_services['mfa_service'].verify_code.return_value = False
        
        request = MFAVerifyRequest(code="invalid")
        
        with pytest.raises(AuthenticationError, match="Invalid verification code"):
            await auth_service.verify_mfa_setup(user_with_mfa_secret.id, request)
    
    @pytest.mark.asyncio
    async def test_verify_mfa_setup_no_secret(self, auth_service, mock_repositories, sample_user):
        """Test MFA verification without setup."""
        mock_repositories['user_repo'].get_by_id.return_value = sample_user
        
        request = MFAVerifyRequest(code="123456")
        
        with pytest.raises(AuthenticationError, match="MFA not set up"):
            await auth_service.verify_mfa_setup(sample_user.id, request)
    
    # Test disable_mfa method
    
    @pytest.mark.asyncio
    async def test_disable_mfa_success(self, auth_service, mock_repositories, mock_services):
        """Test successful MFA disable."""
        user_with_mfa = User(
            id=uuid4(),
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            password_hash="hashed_password",
            mfa_enabled=True,
            mfa_secret="mfa_secret_123",
            backup_codes=["code1", "code2"]
        )
        
        mock_repositories['user_repo'].get_by_id.return_value = user_with_mfa
        mock_services['password_service'].verify_password.return_value = True
        mock_repositories['user_repo'].update.return_value = None
        
        result = await auth_service.disable_mfa(user_with_mfa.id, "password123")
        
        assert result is True
        assert user_with_mfa.mfa_enabled is False
        assert user_with_mfa.mfa_secret is None
        assert user_with_mfa.backup_codes == []
        mock_repositories['user_repo'].update.assert_called_once_with(user_with_mfa)
    
    @pytest.mark.asyncio
    async def test_disable_mfa_invalid_password(self, auth_service, mock_repositories, mock_services):
        """Test MFA disable with invalid password."""
        user_with_mfa = User(
            id=uuid4(),
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            password_hash="hashed_password",
            mfa_enabled=True
        )
        
        mock_repositories['user_repo'].get_by_id.return_value = user_with_mfa
        mock_services['password_service'].verify_password.return_value = False
        
        with pytest.raises(AuthenticationError, match="Invalid password"):
            await auth_service.disable_mfa(user_with_mfa.id, "wrongpassword")
    
    # Test private helper methods
    
    def test_validate_password_policy_valid(self, auth_service):
        """Test password policy validation with valid password."""
        assert auth_service._validate_password_policy("Password123!") is True
    
    def test_validate_password_policy_too_short(self, auth_service):
        """Test password policy validation with short password."""
        assert auth_service._validate_password_policy("Pass1!") is False
    
    def test_validate_password_policy_no_uppercase(self, auth_service):
        """Test password policy validation without uppercase."""
        assert auth_service._validate_password_policy("password123!") is False
    
    def test_validate_password_policy_no_lowercase(self, auth_service):
        """Test password policy validation without lowercase."""
        assert auth_service._validate_password_policy("PASSWORD123!") is False
    
    def test_validate_password_policy_no_numbers(self, auth_service):
        """Test password policy validation without numbers."""
        assert auth_service._validate_password_policy("Password!") is False
    
    def test_validate_password_policy_no_special(self, auth_service):
        """Test password policy validation without special characters."""
        assert auth_service._validate_password_policy("Password123") is False
    
    def test_validate_password_policy_custom_config(self, mock_repositories, mock_services):
        """Test password policy validation with custom config."""
        config = AuthConfig(
            jwt_secret_key="test-secret",
            min_password_length=6,
            require_password_uppercase=False,
            require_password_special=False
        )
        
        auth_service = AuthService(
            user_repository=mock_repositories['user_repo'],
            tenant_repository=mock_repositories['tenant_repo'],
            session_repository=mock_repositories['session_repo'],
            password_service=mock_services['password_service'],
            mfa_service=mock_services['mfa_service'],
            config=config
        )
        
        assert auth_service._validate_password_policy("pass123") is True
        assert auth_service._validate_password_policy("pass") is False  # Too short
    
    @pytest.mark.asyncio
    async def test_create_user_session_session_limit(self, auth_service, mock_repositories, sample_user, config):
        """Test session creation with concurrent session limit."""
        # Create max number of active sessions
        active_sessions = []
        for i in range(config.max_concurrent_sessions):
            session = UserSession(
                id=uuid4(),
                user_id=sample_user.id,
                tenant_id=sample_user.tenant_id,
                session_token=f"token_{i}",
                refresh_token=f"refresh_{i}",
                ip_address="192.168.1.1",
                expires_at=datetime.utcnow() + timedelta(days=30),
                last_accessed_at=datetime.utcnow() - timedelta(minutes=i)  # Different access times
            )
            active_sessions.append(session)
        
        mock_repositories['session_repo'].get_active_sessions.return_value = active_sessions
        mock_repositories['session_repo'].update.return_value = None
        mock_repositories['session_repo'].create.return_value = None
        
        with patch.object(auth_service, '_generate_access_token', return_value="new_access_token"):
            with patch.object(auth_service, '_generate_refresh_token', return_value="new_refresh_token"):
                tokens = await auth_service._create_user_session(sample_user, "192.168.1.1")
        
        # Verify oldest session was revoked
        oldest_session = min(active_sessions, key=lambda s: s.last_accessed_at)
        assert oldest_session.is_active is False
        assert oldest_session.revoked_at is not None
        
        # Verify new session was created
        mock_repositories['session_repo'].create.assert_called_once()
        
        assert tokens["access_token"] == "new_access_token"
        assert tokens["refresh_token"] == "new_refresh_token"
    
    def test_generate_access_token(self, auth_service, sample_user):
        """Test JWT access token generation."""
        with patch('jwt.encode', return_value="jwt_token_123") as mock_encode:
            token = auth_service._generate_access_token(sample_user)
        
        assert token == "jwt_token_123"
        
        # Verify JWT payload
        call_args = mock_encode.call_args
        payload = call_args[0][0]
        
        assert payload["sub"] == str(sample_user.id)
        assert payload["tenant_id"] == str(sample_user.tenant_id)
        assert payload["email"] == sample_user.email
        assert payload["roles"] == list(sample_user.roles)
        assert payload["permissions"] == list(sample_user.permissions)
        assert payload["type"] == "access"
    
    def test_generate_refresh_token(self, auth_service):
        """Test refresh token generation."""
        with patch('secrets.token_urlsafe', return_value="secure_token_123") as mock_token:
            token = auth_service._generate_refresh_token()
        
        assert token == "secure_token_123"
        mock_token.assert_called_once_with(32)


class TestAuthenticationError:
    """Test cases for AuthenticationError."""
    
    def test_authentication_error_creation(self):
        """Test AuthenticationError creation."""
        error = AuthenticationError("Authentication failed")
        assert str(error) == "Authentication failed"
        assert isinstance(error, Exception)
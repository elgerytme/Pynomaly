"""
Unit tests for User domain entity.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from unittest.mock import patch

from enterprise_auth.domain.entities.user import (
    User, UserSession, UserStatus, UserRole, AuthProvider
)


class TestUser:
    """Test cases for User entity."""
    
    def test_user_creation_with_defaults(self):
        """Test user creation with default values."""
        tenant_id = uuid4()
        user = User(
            tenant_id=tenant_id,
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User"
        )
        
        assert isinstance(user.id, UUID)
        assert user.tenant_id == tenant_id
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.first_name == "Test"
        assert user.last_name == "User"
        assert user.display_name == "Test User"
        assert user.status == UserStatus.PENDING_VERIFICATION
        assert user.auth_provider == AuthProvider.LOCAL
        assert user.roles == set()
        assert user.permissions == set()
        assert user.mfa_enabled is False
        assert user.failed_login_attempts == 0
        assert user.timezone == "UTC"
        assert user.language == "en"
        
    def test_user_creation_with_all_fields(self):
        """Test user creation with all fields specified."""
        tenant_id = uuid4()
        created_by = uuid4()
        roles = {UserRole.ANALYST, UserRole.VIEWER}
        permissions = {"read:data", "write:reports"}
        
        user = User(
            tenant_id=tenant_id,
            email="ADMIN@EXAMPLE.COM",
            username="ADMIN_USER",
            first_name="Admin",
            last_name="User",
            display_name="Administrator",
            avatar_url="https://example.com/avatar.jpg",
            phone="+1234567890",
            password_hash="hashed_password",
            auth_provider=AuthProvider.SAML,
            external_id="saml_123",
            status=UserStatus.ACTIVE,
            roles=roles,
            permissions=permissions,
            mfa_enabled=True,
            mfa_secret="secret123",
            backup_codes=["code1", "code2"],
            timezone="America/New_York",
            language="es",
            preferences={"theme": "dark"},
            created_by=created_by
        )
        
        assert user.email == "admin@example.com"  # Should be lowercase
        assert user.username == "admin_user"  # Should be lowercase
        assert user.display_name == "Administrator"
        assert user.avatar_url == "https://example.com/avatar.jpg"
        assert user.phone == "+1234567890"
        assert user.password_hash == "hashed_password"
        assert user.auth_provider == AuthProvider.SAML
        assert user.external_id == "saml_123"
        assert user.status == UserStatus.ACTIVE
        assert user.roles == roles
        assert user.permissions == permissions
        assert user.mfa_enabled is True
        assert user.mfa_secret == "secret123"
        assert user.backup_codes == ["code1", "code2"]
        assert user.timezone == "America/New_York"
        assert user.language == "es"
        assert user.preferences == {"theme": "dark"}
        assert user.created_by == created_by
        
    def test_email_validation_lowercase(self):
        """Test email is converted to lowercase."""
        user = User(
            tenant_id=uuid4(),
            email="TEST@EXAMPLE.COM",
            username="testuser",
            first_name="Test",
            last_name="User"
        )
        assert user.email == "test@example.com"
        
    def test_username_validation_lowercase(self):
        """Test username is converted to lowercase."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="TEST_USER-123",
            first_name="Test",
            last_name="User"
        )
        assert user.username == "test_user-123"
        
    def test_username_validation_invalid_characters(self):
        """Test username validation with invalid characters."""
        with pytest.raises(ValueError, match="Username must contain only alphanumeric characters"):
            User(
                tenant_id=uuid4(),
                email="test@example.com",
                username="test@user",
                first_name="Test",
                last_name="User"
            )
            
    def test_display_name_auto_generation(self):
        """Test display name is auto-generated from first and last name."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="John",
            last_name="Doe"
        )
        assert user.display_name == "John Doe"
        
    def test_display_name_custom(self):
        """Test custom display name overrides auto-generation."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="John",
            last_name="Doe",
            display_name="Johnny D"
        )
        assert user.display_name == "Johnny D"
        
    def test_roles_list_conversion(self):
        """Test roles list is converted to set."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            roles=[UserRole.ANALYST, UserRole.VIEWER, UserRole.ANALYST]  # Duplicate
        )
        assert user.roles == {UserRole.ANALYST, UserRole.VIEWER}
        
    def test_permissions_list_conversion(self):
        """Test permissions list is converted to set."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            permissions=["read", "write", "read"]  # Duplicate
        )
        assert user.permissions == {"read", "write"}
        
    def test_has_role(self):
        """Test has_role method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            roles={UserRole.ANALYST}
        )
        
        assert user.has_role(UserRole.ANALYST) is True
        assert user.has_role(UserRole.VIEWER) is False
        
    def test_has_permission(self):
        """Test has_permission method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            permissions={"read:data"}
        )
        
        assert user.has_permission("read:data") is True
        assert user.has_permission("write:data") is False
        
    def test_add_role(self):
        """Test add_role method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User"
        )
        
        original_updated_at = user.updated_at
        user.add_role(UserRole.ANALYST)
        
        assert UserRole.ANALYST in user.roles
        assert user.updated_at > original_updated_at
        
    def test_remove_role(self):
        """Test remove_role method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            roles={UserRole.ANALYST, UserRole.VIEWER}
        )
        
        original_updated_at = user.updated_at
        user.remove_role(UserRole.ANALYST)
        
        assert UserRole.ANALYST not in user.roles
        assert UserRole.VIEWER in user.roles
        assert user.updated_at > original_updated_at
        
    def test_add_permission(self):
        """Test add_permission method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User"
        )
        
        original_updated_at = user.updated_at
        user.add_permission("read:data")
        
        assert "read:data" in user.permissions
        assert user.updated_at > original_updated_at
        
    def test_remove_permission(self):
        """Test remove_permission method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            permissions={"read:data", "write:data"}
        )
        
        original_updated_at = user.updated_at
        user.remove_permission("read:data")
        
        assert "read:data" not in user.permissions
        assert "write:data" in user.permissions
        assert user.updated_at > original_updated_at
        
    def test_is_active_true(self):
        """Test is_active returns True for active user."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            status=UserStatus.ACTIVE
        )
        assert user.is_active() is True
        
    def test_is_active_false_status(self):
        """Test is_active returns False for inactive status."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            status=UserStatus.SUSPENDED
        )
        assert user.is_active() is False
        
    def test_is_active_false_deleted(self):
        """Test is_active returns False for deleted user."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            status=UserStatus.ACTIVE,
            deleted_at=datetime.utcnow()
        )
        assert user.is_active() is False
        
    def test_is_admin_super_admin(self):
        """Test is_admin returns True for super admin."""
        user = User(
            tenant_id=uuid4(),
            email="admin@example.com",
            username="admin",
            first_name="Admin",
            last_name="User",
            roles={UserRole.SUPER_ADMIN}
        )
        assert user.is_admin() is True
        
    def test_is_admin_tenant_admin(self):
        """Test is_admin returns True for tenant admin."""
        user = User(
            tenant_id=uuid4(),
            email="admin@example.com",
            username="admin",
            first_name="Admin",
            last_name="User",
            roles={UserRole.TENANT_ADMIN}
        )
        assert user.is_admin() is True
        
    def test_is_admin_user_admin(self):
        """Test is_admin returns True for user admin."""
        user = User(
            tenant_id=uuid4(),
            email="admin@example.com",
            username="admin",
            first_name="Admin",
            last_name="User",
            roles={UserRole.USER_ADMIN}
        )
        assert user.is_admin() is True
        
    def test_is_admin_false(self):
        """Test is_admin returns False for non-admin user."""
        user = User(
            tenant_id=uuid4(),
            email="user@example.com",
            username="user",
            first_name="Regular",
            last_name="User",
            roles={UserRole.VIEWER}
        )
        assert user.is_admin() is False
        
    def test_can_access_tenant_super_admin(self):
        """Test super admin can access any tenant."""
        tenant_id = uuid4()
        other_tenant_id = uuid4()
        
        user = User(
            tenant_id=tenant_id,
            email="admin@example.com",
            username="admin",
            first_name="Admin",
            last_name="User",
            roles={UserRole.SUPER_ADMIN}
        )
        
        assert user.can_access_tenant(tenant_id) is True
        assert user.can_access_tenant(other_tenant_id) is True
        
    def test_can_access_tenant_regular_user(self):
        """Test regular user can only access their own tenant."""
        tenant_id = uuid4()
        other_tenant_id = uuid4()
        
        user = User(
            tenant_id=tenant_id,
            email="user@example.com",
            username="user",
            first_name="Regular",
            last_name="User",
            roles={UserRole.VIEWER}
        )
        
        assert user.can_access_tenant(tenant_id) is True
        assert user.can_access_tenant(other_tenant_id) is False
        
    def test_increment_failed_login(self):
        """Test increment_failed_login method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User"
        )
        
        original_updated_at = user.updated_at
        user.increment_failed_login()
        
        assert user.failed_login_attempts == 1
        assert user.updated_at > original_updated_at
        
    def test_increment_failed_login_auto_lock(self):
        """Test account is auto-locked after 5 failed attempts."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            failed_login_attempts=4
        )
        
        user.increment_failed_login()
        
        assert user.failed_login_attempts == 5
        assert user.status == UserStatus.LOCKED
        
    def test_reset_failed_login(self):
        """Test reset_failed_login method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            failed_login_attempts=3
        )
        
        original_updated_at = user.updated_at
        user.reset_failed_login()
        
        assert user.failed_login_attempts == 0
        assert user.last_login_at is not None
        assert user.updated_at > original_updated_at
        
    def test_update_last_login(self):
        """Test update_last_login method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User"
        )
        
        ip_address = "192.168.1.1"
        original_updated_at = user.updated_at
        
        user.update_last_login(ip_address)
        
        assert user.last_login_at is not None
        assert user.last_login_ip == ip_address
        assert user.updated_at > original_updated_at
        
    def test_update_last_login_no_ip(self):
        """Test update_last_login method without IP address."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User"
        )
        
        user.update_last_login()
        
        assert user.last_login_at is not None
        assert user.last_login_ip is None
        
    def test_soft_delete(self):
        """Test soft_delete method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            status=UserStatus.ACTIVE
        )
        
        deleted_by = uuid4()
        user.soft_delete(deleted_by)
        
        assert user.deleted_at is not None
        assert user.deleted_by == deleted_by
        assert user.status == UserStatus.DELETED
        
    def test_restore(self):
        """Test restore method."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            status=UserStatus.DELETED,
            deleted_at=datetime.utcnow(),
            deleted_by=uuid4()
        )
        
        user.restore()
        
        assert user.deleted_at is None
        assert user.deleted_by is None
        assert user.status == UserStatus.INACTIVE
        
    def test_full_name_property(self):
        """Test full_name property."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="John",
            last_name="Doe"
        )
        assert user.full_name == "John Doe"
        
    def test_full_name_property_with_spaces(self):
        """Test full_name property handles extra spaces."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="  John  ",
            last_name="  Doe  "
        )
        # Note: Pydantic validation should handle trimming
        assert user.full_name.strip() == "John   Doe"
        
    def test_is_deleted_property(self):
        """Test is_deleted property."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User"
        )
        
        assert user.is_deleted is False
        
        user.deleted_at = datetime.utcnow()
        assert user.is_deleted is True
        
    def test_is_locked_property(self):
        """Test is_locked property."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User"
        )
        
        assert user.is_locked is False
        
        user.status = UserStatus.LOCKED
        assert user.is_locked is True
        
    def test_to_dict_without_sensitive(self):
        """Test to_dict method without sensitive fields."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            password_hash="secret",
            mfa_secret="mfa_secret",
            backup_codes=["code1", "code2"],
            failed_login_attempts=2,
            last_login_ip="192.168.1.1"
        )
        
        data = user.to_dict(include_sensitive=False)
        
        assert "password_hash" not in data
        assert "mfa_secret" not in data
        assert "backup_codes" not in data
        assert "failed_login_attempts" not in data
        assert "last_login_ip" not in data
        assert "email" in data
        assert "username" in data
        
    def test_to_dict_with_sensitive(self):
        """Test to_dict method with sensitive fields."""
        user = User(
            tenant_id=uuid4(),
            email="test@example.com",
            username="testuser",
            first_name="Test",
            last_name="User",
            password_hash="secret",
            mfa_secret="mfa_secret"
        )
        
        data = user.to_dict(include_sensitive=True)
        
        assert "password_hash" in data
        assert "mfa_secret" in data
        assert data["password_hash"] == "secret"
        assert data["mfa_secret"] == "mfa_secret"
        

class TestUserSession:
    """Test cases for UserSession entity."""
    
    def test_user_session_creation(self):
        """Test user session creation."""
        user_id = uuid4()
        tenant_id = uuid4()
        expires_at = datetime.utcnow() + timedelta(hours=8)
        
        session = UserSession(
            user_id=user_id,
            tenant_id=tenant_id,
            session_token="token123",
            ip_address="192.168.1.1",
            expires_at=expires_at
        )
        
        assert isinstance(session.id, UUID)
        assert session.user_id == user_id
        assert session.tenant_id == tenant_id
        assert session.session_token == "token123"
        assert session.ip_address == "192.168.1.1"
        assert session.expires_at == expires_at
        assert session.is_active is True
        assert session.revoked_at is None
        
    def test_user_session_with_optional_fields(self):
        """Test user session creation with optional fields."""
        user_id = uuid4()
        tenant_id = uuid4()
        expires_at = datetime.utcnow() + timedelta(hours=8)
        
        session = UserSession(
            user_id=user_id,
            tenant_id=tenant_id,
            session_token="token123",
            refresh_token="refresh123",
            device_id="device123",
            device_name="iPhone 12",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0...",
            location={"country": "US", "city": "New York"},
            expires_at=expires_at
        )
        
        assert session.refresh_token == "refresh123"
        assert session.device_id == "device123"
        assert session.device_name == "iPhone 12"
        assert session.user_agent == "Mozilla/5.0..."
        assert session.location == {"country": "US", "city": "New York"}
        
    def test_is_expired_false(self):
        """Test is_expired returns False for valid session."""
        session = UserSession(
            user_id=uuid4(),
            tenant_id=uuid4(),
            session_token="token123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        assert session.is_expired() is False
        
    def test_is_expired_true(self):
        """Test is_expired returns True for expired session."""
        session = UserSession(
            user_id=uuid4(),
            tenant_id=uuid4(),
            session_token="token123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        assert session.is_expired() is True
        
    def test_is_valid_true(self):
        """Test is_valid returns True for valid session."""
        session = UserSession(
            user_id=uuid4(),
            tenant_id=uuid4(),
            session_token="token123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        assert session.is_valid() is True
        
    def test_is_valid_false_inactive(self):
        """Test is_valid returns False for inactive session."""
        session = UserSession(
            user_id=uuid4(),
            tenant_id=uuid4(),
            session_token="token123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            is_active=False
        )
        assert session.is_valid() is False
        
    def test_is_valid_false_expired(self):
        """Test is_valid returns False for expired session."""
        session = UserSession(
            user_id=uuid4(),
            tenant_id=uuid4(),
            session_token="token123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        assert session.is_valid() is False
        
    def test_is_valid_false_revoked(self):
        """Test is_valid returns False for revoked session."""
        session = UserSession(
            user_id=uuid4(),
            tenant_id=uuid4(),
            session_token="token123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            revoked_at=datetime.utcnow()
        )
        assert session.is_valid() is False
        
    def test_revoke(self):
        """Test revoke method."""
        session = UserSession(
            user_id=uuid4(),
            tenant_id=uuid4(),
            session_token="token123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        revoked_by = uuid4()
        reason = "Security breach"
        
        session.revoke(revoked_by, reason)
        
        assert session.is_active is False
        assert session.revoked_at is not None
        assert session.revoked_by == revoked_by
        assert session.revoke_reason == reason
        
    def test_extend(self):
        """Test extend method."""
        original_expires = datetime.utcnow() + timedelta(hours=1)
        session = UserSession(
            user_id=uuid4(),
            tenant_id=uuid4(),
            session_token="token123",
            ip_address="192.168.1.1",
            expires_at=original_expires
        )
        
        original_last_accessed = session.last_accessed_at
        session.extend(120)  # 2 hours
        
        assert session.expires_at > original_expires
        assert session.last_accessed_at > original_last_accessed
        
    def test_update_access(self):
        """Test update_access method."""
        session = UserSession(
            user_id=uuid4(),
            tenant_id=uuid4(),
            session_token="token123",
            ip_address="192.168.1.1",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        original_last_accessed = session.last_accessed_at
        session.update_access()
        
        assert session.last_accessed_at > original_last_accessed
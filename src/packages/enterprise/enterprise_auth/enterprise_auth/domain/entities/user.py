"""
User domain entities for enterprise authentication.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field, validator


class UserStatus(str, Enum):
    """User account status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    LOCKED = "locked"
    DELETED = "deleted"


class UserRole(str, Enum):
    """User role enumeration."""
    SUPER_ADMIN = "super_admin"
    TENANT_ADMIN = "tenant_admin"
    USER_ADMIN = "user_admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


class AuthProvider(str, Enum):
    """Authentication provider enumeration."""
    LOCAL = "local"
    SAML = "saml" 
    OAUTH2 = "oauth2"
    LDAP = "ldap"
    OIDC = "oidc"


class User(BaseModel):
    """
    User domain entity representing a user in the system.
    
    This is the core user entity that handles user identity,
    authentication, and basic profile information.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique user identifier")
    tenant_id: UUID = Field(..., description="Tenant this user belongs to")
    email: EmailStr = Field(..., description="User email address (unique)")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    
    # Profile Information
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    display_name: Optional[str] = Field(None, max_length=200)
    avatar_url: Optional[str] = Field(None)
    phone: Optional[str] = Field(None)
    
    # Authentication
    password_hash: Optional[str] = Field(None, description="Hashed password (for local auth)")
    auth_provider: AuthProvider = Field(default=AuthProvider.LOCAL)
    external_id: Optional[str] = Field(None, description="External provider user ID")
    
    # Status and Metadata
    status: UserStatus = Field(default=UserStatus.PENDING_VERIFICATION)
    roles: Set[UserRole] = Field(default_factory=set)
    permissions: Set[str] = Field(default_factory=set, description="Direct permissions")
    
    # Multi-Factor Authentication
    mfa_enabled: bool = Field(default=False)
    mfa_secret: Optional[str] = Field(None, description="MFA secret key")
    backup_codes: List[str] = Field(default_factory=list)
    
    # Session Management
    failed_login_attempts: int = Field(default=0)
    last_login_at: Optional[datetime] = Field(None)
    last_login_ip: Optional[str] = Field(None)
    password_changed_at: Optional[datetime] = Field(None)
    
    # Preferences
    timezone: str = Field(default="UTC")
    language: str = Field(default="en")
    preferences: Dict[str, any] = Field(default_factory=dict)
    
    # Audit Fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[UUID] = Field(None)
    updated_by: Optional[UUID] = Field(None)
    
    # Soft Delete
    deleted_at: Optional[datetime] = Field(None)
    deleted_by: Optional[UUID] = Field(None)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('display_name', pre=True, always=True)
    def set_display_name(cls, v, values):
        """Set display name from first and last name if not provided."""
        if v is None and 'first_name' in values and 'last_name' in values:
            return f"{values['first_name']} {values['last_name']}"
        return v
    
    @validator('roles', pre=True)
    def validate_roles(cls, v):
        """Validate user roles."""
        if isinstance(v, list):
            return set(v)
        return v
    
    @validator('permissions', pre=True) 
    def validate_permissions(cls, v):
        """Validate user permissions."""
        if isinstance(v, list):
            return set(v)
        return v
    
    @validator('email')
    def email_must_be_lowercase(cls, v):
        """Ensure email is lowercase."""
        return v.lower()
    
    @validator('username')
    def username_must_be_alphanumeric(cls, v):
        """Validate username format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must contain only alphanumeric characters, hyphens, and underscores')
        return v.lower()
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions
    
    def add_role(self, role: UserRole) -> None:
        """Add a role to the user."""
        self.roles.add(role)
        self.updated_at = datetime.utcnow()
    
    def remove_role(self, role: UserRole) -> None:
        """Remove a role from the user."""
        self.roles.discard(role)
        self.updated_at = datetime.utcnow()
    
    def add_permission(self, permission: str) -> None:
        """Add a permission to the user."""
        self.permissions.add(permission)
        self.updated_at = datetime.utcnow()
    
    def remove_permission(self, permission: str) -> None:
        """Remove a permission from the user."""
        self.permissions.discard(permission)
        self.updated_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE and self.deleted_at is None
    
    def is_admin(self) -> bool:
        """Check if user has admin privileges."""
        return (
            UserRole.SUPER_ADMIN in self.roles or
            UserRole.TENANT_ADMIN in self.roles or
            UserRole.USER_ADMIN in self.roles
        )
    
    def can_access_tenant(self, tenant_id: UUID) -> bool:
        """Check if user can access a specific tenant."""
        # Super admins can access any tenant
        if UserRole.SUPER_ADMIN in self.roles:
            return True
        
        # Regular users can only access their own tenant
        return self.tenant_id == tenant_id
    
    def increment_failed_login(self) -> None:
        """Increment failed login attempts."""
        self.failed_login_attempts += 1
        self.updated_at = datetime.utcnow()
        
        # Auto-lock account after 5 failed attempts
        if self.failed_login_attempts >= 5:
            self.status = UserStatus.LOCKED
    
    def reset_failed_login(self) -> None:
        """Reset failed login attempts after successful login."""
        self.failed_login_attempts = 0
        self.last_login_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_last_login(self, ip_address: Optional[str] = None) -> None:
        """Update last login information."""
        self.last_login_at = datetime.utcnow()
        if ip_address:
            self.last_login_ip = ip_address
        self.updated_at = datetime.utcnow()
    
    def soft_delete(self, deleted_by: Optional[UUID] = None) -> None:
        """Soft delete the user."""
        self.deleted_at = datetime.utcnow()
        self.deleted_by = deleted_by
        self.status = UserStatus.DELETED
        self.updated_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore a soft-deleted user."""
        self.deleted_at = None
        self.deleted_by = None
        self.status = UserStatus.INACTIVE  # Require re-activation
        self.updated_at = datetime.utcnow()
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}".strip()
    
    @property
    def is_deleted(self) -> bool:
        """Check if user is deleted."""
        return self.deleted_at is not None
    
    @property
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        return self.status == UserStatus.LOCKED
    
    def to_dict(self, include_sensitive: bool = False) -> Dict[str, any]:
        """
        Convert user to dictionary.
        
        Args:
            include_sensitive: Whether to include sensitive fields like password_hash
        """
        data = self.dict()
        
        if not include_sensitive:
            # Remove sensitive fields
            sensitive_fields = [
                'password_hash', 'mfa_secret', 'backup_codes',
                'failed_login_attempts', 'last_login_ip'
            ]
            for field in sensitive_fields:
                data.pop(field, None)
        
        return data


class UserSession(BaseModel):
    """User session entity for tracking active sessions."""
    
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(...)
    tenant_id: UUID = Field(...)
    
    # Session Information
    session_token: str = Field(...)
    refresh_token: Optional[str] = Field(None)
    device_id: Optional[str] = Field(None)
    device_name: Optional[str] = Field(None)
    
    # Network Information
    ip_address: str = Field(...)
    user_agent: Optional[str] = Field(None)
    location: Optional[Dict[str, str]] = Field(None)
    
    # Session Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(...)
    
    # Security
    is_active: bool = Field(default=True)
    revoked_at: Optional[datetime] = Field(None)
    revoked_by: Optional[UUID] = Field(None)
    revoke_reason: Optional[str] = Field(None)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    def is_valid(self) -> bool:
        """Check if session is valid (active and not expired)."""
        return self.is_active and not self.is_expired() and self.revoked_at is None
    
    def revoke(self, revoked_by: Optional[UUID] = None, reason: Optional[str] = None) -> None:
        """Revoke the session."""
        self.is_active = False
        self.revoked_at = datetime.utcnow()
        self.revoked_by = revoked_by
        self.revoke_reason = reason
    
    def extend(self, duration_minutes: int = 60) -> None:
        """Extend session expiration."""
        from datetime import timedelta
        self.expires_at = datetime.utcnow() + timedelta(minutes=duration_minutes)
        self.last_accessed_at = datetime.utcnow()
    
    def update_access(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed_at = datetime.utcnow()
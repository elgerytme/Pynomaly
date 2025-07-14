"""User entity for authentication and authorization."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Set

from pydantic import EmailStr, Field, validator

from .base import Entity


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class User(Entity):
    """User entity with authentication and authorization."""
    
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    full_name: str = Field(..., min_length=1, max_length=100, description="Full name")
    hashed_password: str = Field(..., description="Hashed password")
    roles: Set[UserRole] = Field(default_factory=set, description="User roles")
    status: UserStatus = Field(default=UserStatus.ACTIVE, description="Account status")
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")
    email_verified: bool = Field(default=False, description="Email verification status")
    failed_login_attempts: int = Field(default=0, description="Failed login attempts")
    locked_until: Optional[datetime] = Field(default=None, description="Account lock expiration")
    
    @validator('email')
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        return v.lower()
    
    @validator('username')
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must contain only alphanumeric characters, hyphens, and underscores')
        return v.lower()
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def add_role(self, role: UserRole) -> None:
        """Add a role to the user."""
        self.roles.add(role)
        self.increment_version()
    
    def remove_role(self, role: UserRole) -> None:
        """Remove a role from the user."""
        self.roles.discard(role)
        self.increment_version()
    
    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE and not self.is_locked()
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def lock_account(self, duration_minutes: int = 30) -> None:
        """Lock user account for specified duration."""
        self.locked_until = datetime.now(timezone.utc).replace(
            minute=datetime.now(timezone.utc).minute + duration_minutes
        )
        self.increment_version()
    
    def unlock_account(self) -> None:
        """Unlock user account."""
        self.locked_until = None
        self.failed_login_attempts = 0
        self.increment_version()
    
    def record_login(self) -> None:
        """Record successful login."""
        self.last_login = datetime.now(timezone.utc)
        self.failed_login_attempts = 0
        self.increment_version()
    
    def record_failed_login(self) -> None:
        """Record failed login attempt."""
        self.failed_login_attempts += 1
        self.increment_version()
        
        # Lock account after 5 failed attempts
        if self.failed_login_attempts >= 5:
            self.lock_account()
    
    def verify_email(self) -> None:
        """Mark email as verified."""
        self.email_verified = True
        if self.status == UserStatus.PENDING_VERIFICATION:
            self.status = UserStatus.ACTIVE
        self.increment_version()
    
    def deactivate(self) -> None:
        """Deactivate user account."""
        self.status = UserStatus.INACTIVE
        self.increment_version()
    
    def suspend(self) -> None:
        """Suspend user account."""
        self.status = UserStatus.SUSPENDED
        self.increment_version()
    
    def reactivate(self) -> None:
        """Reactivate user account."""
        self.status = UserStatus.ACTIVE
        self.increment_version()
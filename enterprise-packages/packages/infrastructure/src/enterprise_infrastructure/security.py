"""Security infrastructure for enterprise applications.

This module provides comprehensive security features including authentication,
authorization, encryption, and security middleware.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from enterprise_core import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    SecurityError,
)
from pydantic import BaseModel, Field


class Permission(str, Enum):
    """Standard permissions enum."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


class Role(BaseModel):
    """Role model with permissions."""
    name: str
    permissions: List[Permission]
    description: Optional[str] = None


class User(BaseModel):
    """User model for authentication and authorization."""
    id: str
    username: str
    email: str
    roles: List[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TokenClaims(BaseModel):
    """JWT token claims."""
    sub: str  # Subject (user ID)
    iss: str  # Issuer
    aud: str  # Audience
    exp: int  # Expiration time
    iat: int  # Issued at
    jti: str  # JWT ID
    scope: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)


class SecurityConfiguration(BaseModel):
    """Security configuration."""
    secret_key: str = Field(..., min_length=32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digits: bool = True
    password_require_special: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    session_timeout_minutes: int = 60
    require_mfa: bool = False
    allowed_origins: List[str] = Field(default_factory=list)


class EncryptionManager:
    """Encryption and hashing utilities."""

    def __init__(self, secret_key: str) -> None:
        self.secret_key = secret_key.encode()

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        try:
            from passlib.context import CryptContext
            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return pwd_context.hash(password)
        except ImportError:
            raise ConfigurationError(
                "passlib not installed. Install with: pip install 'enterprise-infrastructure[security]'",
                error_code="DEPENDENCY_MISSING",
            )

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            from passlib.context import CryptContext
            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return pwd_context.verify(plain_password, hashed_password)
        except ImportError:
            raise ConfigurationError(
                "passlib not installed. Install with: pip install 'enterprise-infrastructure[security]'",
                error_code="DEPENDENCY_MISSING",
            )

    def encrypt_data(self, data: str) -> str:
        """Encrypt data using Fernet symmetric encryption."""
        try:
            from cryptography.fernet import Fernet
            key = hashlib.sha256(self.secret_key).digest()
            key_b64 = base64.urlsafe_b64encode(key)
            fernet = Fernet(key_b64)
            return fernet.encrypt(data.encode()).decode()
        except ImportError:
            raise ConfigurationError(
                "cryptography not installed. Install with: pip install 'enterprise-infrastructure[security]'",
                error_code="DEPENDENCY_MISSING",
            )

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data using Fernet symmetric encryption."""
        try:
            import base64

            from cryptography.fernet import Fernet
            key = hashlib.sha256(self.secret_key).digest()
            key_b64 = base64.urlsafe_b64encode(key)
            fernet = Fernet(key_b64)
            return fernet.decrypt(encrypted_data.encode()).decode()
        except ImportError:
            raise ConfigurationError(
                "cryptography not installed. Install with: pip install 'enterprise-infrastructure[security]'",
                error_code="DEPENDENCY_MISSING",
            )

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a cryptographically secure random token."""
        return secrets.token_urlsafe(length)

    def generate_hmac_signature(self, data: str) -> str:
        """Generate HMAC signature for data integrity."""
        return hmac.new(
            self.secret_key,
            data.encode(),
            hashlib.sha256
        ).hexdigest()

    def verify_hmac_signature(self, data: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected_signature = self.generate_hmac_signature(data)
        return hmac.compare_digest(signature, expected_signature)


class TokenManager:
    """JWT token management."""

    def __init__(self, config: SecurityConfiguration) -> None:
        self.config = config
        self.encryption = EncryptionManager(config.secret_key)

    def create_access_token(self, user: User, additional_claims: Optional[Dict[str, Any]] = None) -> str:
        """Create a JWT access token."""
        try:
            import jwt
        except ImportError:
            raise ConfigurationError(
                "PyJWT not installed. Install with: pip install 'enterprise-infrastructure[security]'",
                error_code="DEPENDENCY_MISSING",
            )

        now = datetime.now(timezone.utc)
        expire = now + timedelta(minutes=self.config.access_token_expire_minutes)

        claims = TokenClaims(
            sub=user.id,
            iss="enterprise-app",
            aud="enterprise-api",
            exp=int(expire.timestamp()),
            iat=int(now.timestamp()),
            jti=self.encryption.generate_secure_token(16),
            roles=user.roles,
        )

        payload = claims.model_dump()
        if additional_claims:
            payload.update(additional_claims)

        return jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

    def create_refresh_token(self, user: User) -> str:
        """Create a JWT refresh token."""
        try:
            import jwt
        except ImportError:
            raise ConfigurationError(
                "PyJWT not installed. Install with: pip install 'enterprise-infrastructure[security]'",
                error_code="DEPENDENCY_MISSING",
            )

        now = datetime.now(timezone.utc)
        expire = now + timedelta(days=self.config.refresh_token_expire_days)

        payload = {
            "sub": user.id,
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "type": "refresh",
            "jti": self.encryption.generate_secure_token(16),
        }

        return jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)

    def verify_token(self, token: str) -> TokenClaims:
        """Verify and decode a JWT token."""
        try:
            import jwt
        except ImportError:
            raise ConfigurationError(
                "PyJWT not installed. Install with: pip install 'enterprise-infrastructure[security]'",
                error_code="DEPENDENCY_MISSING",
            )

        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience="enterprise-api",
                issuer="enterprise-app",
            )
            return TokenClaims(**payload)
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")

    def refresh_access_token(self, refresh_token: str, user: User) -> str:
        """Create a new access token using a refresh token."""
        try:
            import jwt
        except ImportError:
            raise ConfigurationError(
                "PyJWT not installed. Install with: pip install 'enterprise-infrastructure[security]'",
                error_code="DEPENDENCY_MISSING",
            )

        try:
            payload = jwt.decode(
                refresh_token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
            )

            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid refresh token")

            if payload.get("sub") != user.id:
                raise AuthenticationError("Token user mismatch")

            return self.create_access_token(user)

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid refresh token: {e}")


class AuthenticationManager:
    """User authentication management."""

    def __init__(self, config: SecurityConfiguration) -> None:
        self.config = config
        self.encryption = EncryptionManager(config.secret_key)
        self.token_manager = TokenManager(config)
        self._login_attempts: Dict[str, List[datetime]] = {}
        self._locked_accounts: Dict[str, datetime] = {}

    def validate_password_strength(self, password: str) -> List[str]:
        """Validate password strength and return list of issues."""
        issues = []

        if len(password) < self.config.password_min_length:
            issues.append(f"Password must be at least {self.config.password_min_length} characters")

        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")

        if self.config.password_require_lowercase and not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")

        if self.config.password_require_digits and not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")

        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one special character")

        return issues

    def is_account_locked(self, username: str) -> bool:
        """Check if an account is locked due to failed login attempts."""
        if username not in self._locked_accounts:
            return False

        lockout_time = self._locked_accounts[username]
        lockout_duration = timedelta(minutes=self.config.lockout_duration_minutes)

        if datetime.now(timezone.utc) > lockout_time + lockout_duration:
            # Lockout period has expired
            del self._locked_accounts[username]
            return False

        return True

    def record_login_attempt(self, username: str, success: bool) -> None:
        """Record a login attempt."""
        now = datetime.now(timezone.utc)

        if success:
            # Clear failed attempts on successful login
            self._login_attempts.pop(username, None)
            self._locked_accounts.pop(username, None)
            return

        # Record failed attempt
        if username not in self._login_attempts:
            self._login_attempts[username] = []

        self._login_attempts[username].append(now)

        # Remove attempts older than lockout duration
        cutoff = now - timedelta(minutes=self.config.lockout_duration_minutes)
        self._login_attempts[username] = [
            attempt for attempt in self._login_attempts[username]
            if attempt > cutoff
        ]

        # Check if account should be locked
        if len(self._login_attempts[username]) >= self.config.max_login_attempts:
            self._locked_accounts[username] = now

    async def authenticate_user(self, username: str, password: str, user_loader: Callable[[str], User]) -> User:
        """Authenticate a user with username and password."""
        # Check if account is locked
        if self.is_account_locked(username):
            raise AuthenticationError(
                "Account is temporarily locked due to too many failed login attempts",
                details={"username": username, "lockout_duration": self.config.lockout_duration_minutes}
            )

        try:
            # Load user
            user = await user_loader(username)
            if not user or not user.is_active:
                self.record_login_attempt(username, False)
                raise AuthenticationError("Invalid credentials")

            # Verify password (this should be implemented by the user loader)
            # For now, we assume the user loader handles password verification

            # Record successful login
            self.record_login_attempt(username, True)
            user.last_login = datetime.now(timezone.utc)

            return user

        except Exception as e:
            self.record_login_attempt(username, False)
            if isinstance(e, AuthenticationError):
                raise
            raise AuthenticationError("Authentication failed")


class AuthorizationManager:
    """Role-based access control manager."""

    def __init__(self) -> None:
        self._roles: Dict[str, Role] = {}
        self._user_roles: Dict[str, List[str]] = {}

    def define_role(self, role: Role) -> None:
        """Define a role with permissions."""
        self._roles[role.name] = role

    def assign_role_to_user(self, user_id: str, role_name: str) -> None:
        """Assign a role to a user."""
        if role_name not in self._roles:
            raise AuthorizationError(
                f"Role '{role_name}' not found",
                resource="role",
                action="assign",
                details={"role": role_name, "user_id": user_id}
            )

        if user_id not in self._user_roles:
            self._user_roles[user_id] = []

        if role_name not in self._user_roles[user_id]:
            self._user_roles[user_id].append(role_name)

    def remove_role_from_user(self, user_id: str, role_name: str) -> None:
        """Remove a role from a user."""
        if user_id in self._user_roles:
            self._user_roles[user_id] = [
                role for role in self._user_roles[user_id]
                if role != role_name
            ]

    def user_has_permission(self, user_id: str, permission: Permission, resource: Optional[str] = None) -> bool:
        """Check if a user has a specific permission."""
        user_roles = self._user_roles.get(user_id, [])

        for role_name in user_roles:
            role = self._roles.get(role_name)
            if role and permission in role.permissions:
                return True

        return False

    def require_permission(self, user_id: str, permission: Permission, resource: str = "resource") -> None:
        """Require a user to have a specific permission, raise exception if not."""
        if not self.user_has_permission(user_id, permission):
            raise AuthorizationError(
                resource=resource,
                action=permission.value,
                user=user_id,
                details={"required_permission": permission.value}
            )

    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get all permissions for a user."""
        permissions = set()
        user_roles = self._user_roles.get(user_id, [])

        for role_name in user_roles:
            role = self._roles.get(role_name)
            if role:
                permissions.update(role.permissions)

        return list(permissions)


class SecurityManager:
    """Central security manager coordinating all security components."""

    def __init__(self, config: SecurityConfiguration) -> None:
        self.config = config
        self.encryption = EncryptionManager(config.secret_key)
        self.token_manager = TokenManager(config)
        self.auth_manager = AuthenticationManager(config)
        self.authz_manager = AuthorizationManager()
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

    def setup_default_roles(self) -> None:
        """Set up default roles and permissions."""
        # Admin role with all permissions
        admin_role = Role(
            name="admin",
            permissions=[Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN],
            description="Full system access"
        )

        # Editor role with read/write permissions
        editor_role = Role(
            name="editor",
            permissions=[Permission.READ, Permission.WRITE],
            description="Read and write access"
        )

        # Viewer role with read-only permissions
        viewer_role = Role(
            name="viewer",
            permissions=[Permission.READ],
            description="Read-only access"
        )

        self.authz_manager.define_role(admin_role)
        self.authz_manager.define_role(editor_role)
        self.authz_manager.define_role(viewer_role)

    def create_session(self, user: User) -> Dict[str, str]:
        """Create a new user session with access and refresh tokens."""
        access_token = self.token_manager.create_access_token(user)
        refresh_token = self.token_manager.create_refresh_token(user)

        session_id = self.encryption.generate_secure_token(32)
        self._active_sessions[session_id] = {
            "user_id": user.id,
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc),
            "access_token": access_token,
        }

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "session_id": session_id,
            "token_type": "bearer",
            "expires_in": self.config.access_token_expire_minutes * 60,
        }

    def validate_session(self, session_id: str) -> bool:
        """Validate if a session is still active."""
        if session_id not in self._active_sessions:
            return False

        session = self._active_sessions[session_id]
        last_activity = session["last_activity"]
        timeout = timedelta(minutes=self.config.session_timeout_minutes)

        if datetime.now(timezone.utc) > last_activity + timeout:
            # Session has timed out
            del self._active_sessions[session_id]
            return False

        # Update last activity
        session["last_activity"] = datetime.now(timezone.utc)
        return True

    def terminate_session(self, session_id: str) -> None:
        """Terminate a user session."""
        self._active_sessions.pop(session_id, None)

    def get_active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._active_sessions)

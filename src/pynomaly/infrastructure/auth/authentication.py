#!/usr/bin/env python3
"""
User Authentication System for Pynomaly

This module provides comprehensive authentication and authorization
capabilities including JWT tokens, session management, and RBAC.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import bcrypt
import jwt
import redis
from cryptography.fernet import Fernet
from passlib.context import CryptContext


class UserRole(Enum):
    """User roles for RBAC."""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_USER = "api_user"
    SERVICE_ACCOUNT = "service_account"


class Permission(Enum):
    """System permissions."""
    # User management
    CREATE_USER = "create_user"
    READ_USER = "read_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    RESET_PASSWORD = "reset_password"
    
    # Data access
    READ_DATA = "read_data"
    WRITE_DATA = "write_data"
    DELETE_DATA = "delete_data"
    EXPORT_DATA = "export_data"
    
    # System administration
    MANAGE_SYSTEM = "manage_system"
    VIEW_LOGS = "view_logs"
    MANAGE_BACKUPS = "manage_backups"
    
    # API access
    API_ACCESS = "api_access"
    API_ADMIN = "api_admin"
    
    # Monitoring
    VIEW_METRICS = "view_metrics"
    MANAGE_ALERTS = "manage_alerts"


class AuthenticationMethod(Enum):
    """Authentication methods."""
    PASSWORD = "password"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    SESSION = "session"
    OAUTH = "oauth"
    LDAP = "ldap"


@dataclass
class User:
    """User data structure."""
    id: str
    username: str
    email: str
    password_hash: str
    roles: set[UserRole] = field(default_factory=set)
    permissions: set[Permission] = field(default_factory=set)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_sensitive: bool = False) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "roles": [role.value for role in self.roles],
            "permissions": [perm.value for perm in self.permissions],
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "failed_login_attempts": self.failed_login_attempts,
            "locked_until": self.locked_until.isoformat() if self.locked_until else None,
            "metadata": self.metadata
        }
        
        if include_sensitive:
            data["password_hash"] = self.password_hash
        
        return data
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role."""
        return role in self.roles
    
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until:
            return datetime.now() < self.locked_until
        return False


@dataclass
class APIKey:
    """API key data structure."""
    id: str
    key_hash: str
    user_id: str
    name: str
    permissions: set[Permission] = field(default_factory=set)
    is_active: bool = True
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit: Optional[int] = None  # requests per hour
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "permissions": [perm.value for perm in self.permissions],
            "is_active": self.is_active,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "rate_limit": self.rate_limit,
            "metadata": self.metadata
        }


@dataclass
class Session:
    """User session data structure."""
    id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.now() > self.expires_at
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "is_active": self.is_active,
            "metadata": self.metadata
        }


class UserStore(ABC):
    """Abstract base class for user storage."""
    
    @abstractmethod
    async def create_user(self, user: User) -> bool:
        """Create new user."""
        pass
    
    @abstractmethod
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        pass
    
    @abstractmethod
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        pass
    
    @abstractmethod
    async def update_user(self, user: User) -> bool:
        """Update user."""
        pass
    
    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        pass
    
    @abstractmethod
    async def list_users(self, limit: int = 100, offset: int = 0) -> list[User]:
        """List users."""
        pass


class MemoryUserStore(UserStore):
    """In-memory user storage for development/testing."""
    
    def __init__(self):
        self.users: dict[str, User] = {}
        self.username_index: dict[str, str] = {}
        self.email_index: dict[str, str] = {}
    
    async def create_user(self, user: User) -> bool:
        """Create new user."""
        if user.id in self.users:
            return False
        
        if user.username in self.username_index:
            return False
        
        if user.email in self.email_index:
            return False
        
        self.users[user.id] = user
        self.username_index[user.username] = user.id
        self.email_index[user.email] = user.id
        
        return True
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        user_id = self.username_index.get(username)
        return self.users.get(user_id) if user_id else None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        user_id = self.email_index.get(email)
        return self.users.get(user_id) if user_id else None
    
    async def update_user(self, user: User) -> bool:
        """Update user."""
        if user.id not in self.users:
            return False
        
        # Update indexes if username or email changed
        old_user = self.users[user.id]
        
        if old_user.username != user.username:
            del self.username_index[old_user.username]
            self.username_index[user.username] = user.id
        
        if old_user.email != user.email:
            del self.email_index[old_user.email]
            self.email_index[user.email] = user.id
        
        user.updated_at = datetime.now()
        self.users[user.id] = user
        
        return True
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        user = self.users.get(user_id)
        if not user:
            return False
        
        del self.users[user_id]
        del self.username_index[user.username]
        del self.email_index[user.email]
        
        return True
    
    async def list_users(self, limit: int = 100, offset: int = 0) -> list[User]:
        """List users."""
        users = list(self.users.values())
        return users[offset:offset + limit]


class PasswordManager:
    """Manage password hashing and validation."""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.min_length = 8
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special = True
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(password, password_hash)
    
    def validate_password_strength(self, password: str) -> list[str]:
        """Validate password strength and return list of issues."""
        issues = []
        
        if len(password) < self.min_length:
            issues.append(f"Password must be at least {self.min_length} characters long")
        
        if self.require_uppercase and not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        
        if self.require_lowercase and not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")
        
        if self.require_digits and not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one digit")
        
        if self.require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one special character")
        
        return issues


class JWTManager:
    """Manage JWT token creation and validation."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=30)
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        expire = datetime.utcnow() + (expires_delta or self.access_token_expire)
        
        payload = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + (expires_delta or self.refresh_token_expire)
        
        payload = {
            "sub": user.id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Optional[dict[str, Any]]:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_access_token(self, refresh_token: str, user_store: UserStore) -> Optional[str]:
        """Refresh access token using refresh token."""
        payload = self.decode_token(refresh_token)
        
        if not payload or payload.get("type") != "refresh":
            return None
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        # Get current user data
        user = asyncio.run(user_store.get_user_by_id(user_id))
        if not user or not user.is_active:
            return None
        
        return self.create_access_token(user)


class SessionManager:
    """Manage user sessions."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.sessions: dict[str, Session] = {} if not redis_client else None
        self.session_timeout = timedelta(hours=24)
    
    async def create_session(self, user_id: str, ip_address: str, user_agent: str,
                           expires_delta: Optional[timedelta] = None) -> Session:
        """Create new session."""
        session_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + (expires_delta or self.session_timeout)
        
        session = Session(
            id=session_id,
            user_id=user_id,
            created_at=now,
            expires_at=expires_at,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        await self._store_session(session)
        return session
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        if self.redis_client:
            data = self.redis_client.get(f"session:{session_id}")
            if data:
                session_data = json.loads(data)
                return self._dict_to_session(session_data)
        else:
            return self.sessions.get(session_id)
        
        return None
    
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity."""
        session = await self.get_session(session_id)
        if not session or session.is_expired():
            return False
        
        session.update_activity()
        await self._store_session(session)
        return True
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate session."""
        if self.redis_client:
            result = self.redis_client.delete(f"session:{session_id}")
            return result > 0
        else:
            return self.sessions.pop(session_id, None) is not None
    
    async def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for user."""
        count = 0
        
        if self.redis_client:
            # This would require additional indexing in Redis
            # For now, implement basic version
            pass
        else:
            to_remove = []
            for session_id, session in self.sessions.items():
                if session.user_id == user_id:
                    to_remove.append(session_id)
            
            for session_id in to_remove:
                del self.sessions[session_id]
                count += 1
        
        return count
    
    async def _store_session(self, session: Session):
        """Store session."""
        if self.redis_client:
            session_data = session.to_dict()
            ttl = int((session.expires_at - datetime.now()).total_seconds())
            self.redis_client.setex(
                f"session:{session.id}",
                ttl,
                json.dumps(session_data, default=str)
            )
        else:
            self.sessions[session.id] = session
    
    def _dict_to_session(self, data: dict[str, Any]) -> Session:
        """Convert dictionary to Session object."""
        return Session(
            id=data["id"],
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            ip_address=data["ip_address"],
            user_agent=data["user_agent"],
            is_active=data["is_active"],
            metadata=data["metadata"]
        )


class APIKeyManager:
    """Manage API keys."""
    
    def __init__(self):
        self.api_keys: dict[str, APIKey] = {}
        self.key_index: dict[str, str] = {}  # key_hash -> api_key_id
    
    def generate_api_key(self) -> str:
        """Generate new API key."""
        return secrets.token_urlsafe(32)
    
    def hash_key(self, key: str) -> str:
        """Hash API key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def create_api_key(self, user_id: str, name: str,
                           permissions: set[Permission],
                           expires_delta: Optional[timedelta] = None) -> tuple[str, APIKey]:
        """Create new API key."""
        key = self.generate_api_key()
        key_hash = self.hash_key(key)
        
        api_key = APIKey(
            id=str(uuid.uuid4()),
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions,
            expires_at=datetime.now() + expires_delta if expires_delta else None
        )
        
        self.api_keys[api_key.id] = api_key
        self.key_index[key_hash] = api_key.id
        
        return key, api_key
    
    async def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate API key and return associated API key object."""
        key_hash = self.hash_key(key)
        api_key_id = self.key_index.get(key_hash)
        
        if not api_key_id:
            return None
        
        api_key = self.api_keys.get(api_key_id)
        
        if not api_key or not api_key.is_active or api_key.is_expired():
            return None
        
        # Update usage
        api_key.last_used = datetime.now()
        api_key.usage_count += 1
        
        return api_key
    
    async def revoke_api_key(self, api_key_id: str) -> bool:
        """Revoke API key."""
        api_key = self.api_keys.get(api_key_id)
        if not api_key:
            return False
        
        api_key.is_active = False
        return True
    
    async def list_user_api_keys(self, user_id: str) -> list[APIKey]:
        """List API keys for user."""
        return [
            api_key for api_key in self.api_keys.values()
            if api_key.user_id == user_id
        ]


class RoleBasedAccessControl:
    """Role-based access control system."""
    
    def __init__(self):
        # Define role permissions
        self.role_permissions = {
            UserRole.SUPER_ADMIN: set(Permission),  # All permissions
            UserRole.ADMIN: {
                Permission.CREATE_USER, Permission.READ_USER, Permission.UPDATE_USER,
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.DELETE_DATA,
                Permission.VIEW_LOGS, Permission.MANAGE_BACKUPS, Permission.VIEW_METRICS,
                Permission.MANAGE_ALERTS, Permission.API_ACCESS
            },
            UserRole.USER: {
                Permission.LOGIN, Permission.LOGOUT, Permission.RESET_PASSWORD,
                Permission.READ_DATA, Permission.WRITE_DATA, Permission.EXPORT_DATA,
                Permission.API_ACCESS
            },
            UserRole.VIEWER: {
                Permission.LOGIN, Permission.LOGOUT, Permission.READ_DATA,
                Permission.VIEW_METRICS
            },
            UserRole.API_USER: {
                Permission.API_ACCESS, Permission.READ_DATA, Permission.WRITE_DATA
            },
            UserRole.SERVICE_ACCOUNT: {
                Permission.API_ACCESS, Permission.READ_DATA, Permission.WRITE_DATA,
                Permission.API_ADMIN
            }
        }
    
    def get_role_permissions(self, role: UserRole) -> set[Permission]:
        """Get permissions for role."""
        return self.role_permissions.get(role, set())
    
    def assign_role_permissions(self, user: User):
        """Assign permissions based on user roles."""
        user.permissions.clear()
        
        for role in user.roles:
            user.permissions.update(self.get_role_permissions(role))
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has permission."""
        return user.has_permission(permission)
    
    def require_permission(self, permission: Permission):
        """Decorator to require permission for function access."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # This would need to extract user from context
                # Implementation depends on web framework
                pass
            return wrapper
        return decorator


class AuthenticationService:
    """Main authentication service."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.user_store = MemoryUserStore()  # Replace with database store
        self.password_manager = PasswordManager()
        self.jwt_manager = JWTManager(config["jwt_secret"])
        self.session_manager = SessionManager()
        self.api_key_manager = APIKeyManager()
        self.rbac = RoleBasedAccessControl()
        
        # Security settings
        self.max_login_attempts = config.get("max_login_attempts", 5)
        self.lockout_duration = timedelta(minutes=config.get("lockout_minutes", 30))
        
        # Create default admin user
        asyncio.create_task(self._create_default_admin())
    
    async def _create_default_admin(self):
        """Create default admin user if none exists."""
        admin_username = self.config.get("default_admin_username", "admin")
        admin_password = self.config.get("default_admin_password", "admin123")
        admin_email = self.config.get("default_admin_email", "admin@pynomaly.com")
        
        # Check if admin already exists
        existing_admin = await self.user_store.get_user_by_username(admin_username)
        if existing_admin:
            return
        
        # Create admin user
        admin_user = User(
            id=str(uuid.uuid4()),
            username=admin_username,
            email=admin_email,
            password_hash=self.password_manager.hash_password(admin_password),
            roles={UserRole.SUPER_ADMIN},
            is_verified=True
        )
        
        self.rbac.assign_role_permissions(admin_user)
        
        success = await self.user_store.create_user(admin_user)
        if success:
            self.logger.info(f"Created default admin user: {admin_username}")
        else:
            self.logger.error("Failed to create default admin user")
    
    async def register_user(self, username: str, email: str, password: str,
                          roles: Optional[set[UserRole]] = None) -> tuple[bool, str]:
        """Register new user."""
        # Validate password strength
        password_issues = self.password_manager.validate_password_strength(password)
        if password_issues:
            return False, "; ".join(password_issues)
        
        # Check if user already exists
        existing_user = await self.user_store.get_user_by_username(username)
        if existing_user:
            return False, "Username already exists"
        
        existing_email = await self.user_store.get_user_by_email(email)
        if existing_email:
            return False, "Email already exists"
        
        # Create user
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=self.password_manager.hash_password(password),
            roles=roles or {UserRole.USER}
        )
        
        self.rbac.assign_role_permissions(user)
        
        success = await self.user_store.create_user(user)
        if success:
            self.logger.info(f"Registered new user: {username}")
            return True, "User registered successfully"
        else:
            return False, "Failed to create user"
    
    async def authenticate_user(self, username: str, password: str,
                              ip_address: str = "", user_agent: str = "") -> tuple[Optional[User], Optional[Session], str]:
        """Authenticate user with username/password."""
        user = await self.user_store.get_user_by_username(username)
        
        if not user:
            return None, None, "Invalid username or password"
        
        # Check if account is locked
        if user.is_locked():
            return None, None, f"Account is locked until {user.locked_until}"
        
        # Check if account is active
        if not user.is_active:
            return None, None, "Account is deactivated"
        
        # Verify password
        if not self.password_manager.verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_login_attempts:
                user.locked_until = datetime.now() + self.lockout_duration
                self.logger.warning(f"Account locked due to failed login attempts: {username}")
            
            await self.user_store.update_user(user)
            return None, None, "Invalid username or password"
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        await self.user_store.update_user(user)
        
        # Create session
        session = await self.session_manager.create_session(
            user.id, ip_address, user_agent
        )
        
        self.logger.info(f"User authenticated: {username}")
        return user, session, "Authentication successful"
    
    async def authenticate_api_key(self, api_key: str) -> tuple[Optional[User], Optional[APIKey], str]:
        """Authenticate using API key."""
        api_key_obj = await self.api_key_manager.validate_api_key(api_key)
        
        if not api_key_obj:
            return None, None, "Invalid API key"
        
        user = await self.user_store.get_user_by_id(api_key_obj.user_id)
        
        if not user or not user.is_active:
            return None, None, "User account is deactivated"
        
        self.logger.info(f"API key authenticated for user: {user.username}")
        return user, api_key_obj, "API key authentication successful"
    
    async def authenticate_jwt(self, token: str) -> tuple[Optional[User], str]:
        """Authenticate using JWT token."""
        payload = self.jwt_manager.decode_token(token)
        
        if not payload:
            return None, "Invalid or expired token"
        
        if payload.get("type") != "access":
            return None, "Invalid token type"
        
        user_id = payload.get("sub")
        user = await self.user_store.get_user_by_id(user_id)
        
        if not user or not user.is_active:
            return None, "User account is deactivated"
        
        return user, "JWT authentication successful"
    
    async def logout_user(self, session_id: str) -> bool:
        """Logout user by invalidating session."""
        return await self.session_manager.invalidate_session(session_id)
    
    async def create_user_api_key(self, user_id: str, name: str,
                                permissions: set[Permission],
                                expires_days: Optional[int] = None) -> tuple[str, APIKey]:
        """Create API key for user."""
        expires_delta = timedelta(days=expires_days) if expires_days else None
        return await self.api_key_manager.create_api_key(
            user_id, name, permissions, expires_delta
        )
    
    async def get_user_profile(self, user_id: str) -> Optional[dict[str, Any]]:
        """Get user profile."""
        user = await self.user_store.get_user_by_id(user_id)
        return user.to_dict() if user else None
    
    async def update_user_password(self, user_id: str, current_password: str,
                                 new_password: str) -> tuple[bool, str]:
        """Update user password."""
        user = await self.user_store.get_user_by_id(user_id)
        
        if not user:
            return False, "User not found"
        
        # Verify current password
        if not self.password_manager.verify_password(current_password, user.password_hash):
            return False, "Current password is incorrect"
        
        # Validate new password
        password_issues = self.password_manager.validate_password_strength(new_password)
        if password_issues:
            return False, "; ".join(password_issues)
        
        # Update password
        user.password_hash = self.password_manager.hash_password(new_password)
        user.updated_at = datetime.now()
        
        success = await self.user_store.update_user(user)
        return success, "Password updated successfully" if success else "Failed to update password"
    
    def get_auth_stats(self) -> dict[str, Any]:
        """Get authentication statistics."""
        return {
            "total_users": len(self.user_store.users),
            "active_sessions": len(self.session_manager.sessions) if self.session_manager.sessions else 0,
            "total_api_keys": len(self.api_key_manager.api_keys),
            "active_api_keys": len([k for k in self.api_key_manager.api_keys.values() if k.is_active]),
            "user_roles": {
                role.value: len([u for u in self.user_store.users.values() if role in u.roles])
                for role in UserRole
            }
        }


async def main():
    """Main function for testing."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Testing authentication system")
    
    # Configuration
    config = {
        "jwt_secret": "your-secret-key-here",
        "default_admin_username": "admin",
        "default_admin_password": "SecureAdminPass123!",
        "default_admin_email": "admin@pynomaly.com",
        "max_login_attempts": 3,
        "lockout_minutes": 15
    }
    
    # Create authentication service
    auth_service = AuthenticationService(config)
    
    # Wait for default admin to be created
    await asyncio.sleep(0.1)
    
    # Test user registration
    logger.info("Testing user registration...")
    success, message = await auth_service.register_user(
        "testuser", "test@example.com", "TestPass123!"
    )
    logger.info(f"Registration result: {success} - {message}")
    
    # Test authentication
    logger.info("Testing authentication...")
    user, session, message = await auth_service.authenticate_user(
        "testuser", "TestPass123!", "127.0.0.1", "Test-Agent"
    )
    logger.info(f"Authentication result: {message}")
    
    if user:
        logger.info(f"Authenticated user: {user.username}")
        logger.info(f"User roles: {[role.value for role in user.roles]}")
        logger.info(f"User permissions: {[perm.value for perm in user.permissions]}")
        
        # Test JWT token creation
        jwt_token = auth_service.jwt_manager.create_access_token(user)
        logger.info(f"Created JWT token: {jwt_token[:50]}...")
        
        # Test JWT authentication
        auth_user, message = await auth_service.authenticate_jwt(jwt_token)
        logger.info(f"JWT authentication: {message}")
        
        # Test API key creation
        api_key, api_key_obj = await auth_service.create_user_api_key(
            user.id, "Test API Key", {Permission.READ_DATA, Permission.API_ACCESS}
        )
        logger.info(f"Created API key: {api_key[:20]}...")
        
        # Test API key authentication
        auth_user, api_key_obj, message = await auth_service.authenticate_api_key(api_key)
        logger.info(f"API key authentication: {message}")
    
    # Print statistics
    stats = auth_service.get_auth_stats()
    logger.info(f"Authentication statistics: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())

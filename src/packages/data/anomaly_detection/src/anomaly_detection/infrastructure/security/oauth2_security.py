"""OAuth2/JWT security implementation for anomaly detection API."""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac

try:
    import jwt
    from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
    JWT_AVAILABLE = True
except ImportError:
    jwt = None
    JWT_AVAILABLE = False

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    from fastapi import HTTPException, Depends, status
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    FASTAPI_AVAILABLE = True
except ImportError:
    HTTPException = None
    Depends = None
    status = None
    HTTPBearer = None
    HTTPAuthorizationCredentials = None
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_CLIENT = "api_client"


class Permission(Enum):
    """System permissions."""
    # Model operations
    MODEL_CREATE = "model:create"
    MODEL_READ = "model:read"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    MODEL_TRAIN = "model:train"
    
    # Detection operations
    DETECTION_RUN = "detection:run"
    DETECTION_READ = "detection:read"
    DETECTION_BATCH = "detection:batch"
    
    # Data operations
    DATA_UPLOAD = "data:upload"
    DATA_READ = "data:read"
    DATA_DELETE = "data:delete"
    
    # System operations
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"
    USER_MANAGE = "user:manage"
    
    # Streaming operations
    STREAM_READ = "stream:read"
    STREAM_WRITE = "stream:write"


@dataclass
class User:
    """User information."""
    user_id: str
    username: str
    email: str
    roles: List[UserRole]
    permissions: List[Permission]
    is_active: bool = True
    created_at: datetime = None
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TokenData:
    """JWT token data."""
    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]
    issued_at: datetime
    expires_at: datetime
    token_type: str = "access"
    client_id: Optional[str] = None


class SecurityConfig:
    """Security configuration."""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        password_min_length: int = 8,
        max_login_attempts: int = 5,
        lockout_duration_minutes: int = 15,
        require_https: bool = True,
        enable_rate_limiting: bool = True
    ):
        """Initialize security configuration.
        
        Args:
            secret_key: Secret key for JWT signing
            algorithm: JWT signing algorithm
            access_token_expire_minutes: Access token expiration
            refresh_token_expire_days: Refresh token expiration
            password_min_length: Minimum password length
            max_login_attempts: Maximum failed login attempts
            lockout_duration_minutes: Account lockout duration
            require_https: Whether to require HTTPS
            enable_rate_limiting: Whether to enable rate limiting
        """
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT is required for JWT security")
        
        self.secret_key = secret_key or self._generate_secret_key()
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.password_min_length = password_min_length
        self.max_login_attempts = max_login_attempts
        self.lockout_duration_minutes = lockout_duration_minutes
        self.require_https = require_https
        self.enable_rate_limiting = enable_rate_limiting
        
        # RSA keys for asymmetric signing (if needed)
        self.private_key = None
        self.public_key = None
        
        if algorithm.startswith('RS') or algorithm.startswith('ES'):
            self._generate_rsa_keys()
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_urlsafe(32)
    
    def _generate_rsa_keys(self) -> None:
        """Generate RSA key pair for asymmetric algorithms."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography is required for RSA algorithms")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        self.private_key = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        self.public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )


class RoleBasedAccessControl:
    """Role-based access control system."""
    
    def __init__(self):
        """Initialize RBAC system."""
        self.role_permissions = self._define_role_permissions()
    
    def _define_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Define permissions for each role."""
        return {
            UserRole.ADMIN: [
                # All permissions
                Permission.MODEL_CREATE, Permission.MODEL_READ, Permission.MODEL_UPDATE, 
                Permission.MODEL_DELETE, Permission.MODEL_TRAIN,
                Permission.DETECTION_RUN, Permission.DETECTION_READ, Permission.DETECTION_BATCH,
                Permission.DATA_UPLOAD, Permission.DATA_READ, Permission.DATA_DELETE,
                Permission.SYSTEM_MONITOR, Permission.SYSTEM_CONFIG, Permission.USER_MANAGE,
                Permission.STREAM_READ, Permission.STREAM_WRITE
            ],
            UserRole.ANALYST: [
                # Model and detection operations
                Permission.MODEL_CREATE, Permission.MODEL_READ, Permission.MODEL_UPDATE, Permission.MODEL_TRAIN,
                Permission.DETECTION_RUN, Permission.DETECTION_READ, Permission.DETECTION_BATCH,
                Permission.DATA_UPLOAD, Permission.DATA_READ,
                Permission.SYSTEM_MONITOR, Permission.STREAM_READ, Permission.STREAM_WRITE
            ],
            UserRole.VIEWER: [
                # Read-only access
                Permission.MODEL_READ, Permission.DETECTION_READ,
                Permission.DATA_READ, Permission.SYSTEM_MONITOR, Permission.STREAM_READ
            ],
            UserRole.API_CLIENT: [
                # API-specific permissions
                Permission.MODEL_READ, Permission.DETECTION_RUN, Permission.DETECTION_BATCH,
                Permission.DATA_UPLOAD, Permission.STREAM_WRITE
            ]
        }
    
    def get_user_permissions(self, user: User) -> List[Permission]:
        """Get all permissions for a user based on their roles."""
        permissions = set()
        
        for role in user.roles:
            role_perms = self.role_permissions.get(role, [])
            permissions.update(role_perms)
        
        # Add explicit permissions
        permissions.update(user.permissions)
        
        return list(permissions)
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions
    
    def require_permission(self, permission: Permission):
        """Decorator to require specific permission."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # This would be implemented with actual user context
                # For now, it's a placeholder
                return func(*args, **kwargs)
            return wrapper
        return decorator


class OAuth2Security:
    """OAuth2/JWT security implementation."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize OAuth2 security.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.rbac = RoleBasedAccessControl()
        
        # User storage (in production, this would be a database)
        self.users: Dict[str, User] = {}
        self.login_attempts: Dict[str, List[datetime]] = {}
        self.refresh_tokens: Dict[str, str] = {}  # token -> user_id
        
        # Security bearer for FastAPI
        if FASTAPI_AVAILABLE:
            self.security = HTTPBearer(auto_error=False)
        
        logger.info("OAuth2 security initialized")
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[UserRole],
        permissions: Optional[List[Permission]] = None
    ) -> User:
        """Create new user.
        
        Args:
            username: Username
            email: Email address
            password: Password
            roles: User roles
            permissions: Additional permissions
            
        Returns:
            Created user
        """
        if len(password) < self.config.password_min_length:
            raise ValueError(f"Password must be at least {self.config.password_min_length} characters")
        
        if username in [u.username for u in self.users.values()]:
            raise ValueError("Username already exists")
        
        user_id = secrets.token_urlsafe(16)
        password_hash = self._hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            permissions=permissions or [],
            metadata={"password_hash": password_hash}
        )
        
        self.users[user_id] = user
        logger.info(f"Created user: {username} with roles: {[r.value for r in roles]}")
        
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Authenticated user or None
        """
        # Check login attempts
        if self._is_account_locked(username):
            logger.warning(f"Account locked: {username}")
            return None
        
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user or not user.is_active:
            self._record_failed_login(username)
            return None
        
        # Verify password
        password_hash = user.metadata.get("password_hash")
        if not password_hash or not self._verify_password(password, password_hash):
            self._record_failed_login(username)
            return None
        
        # Clear login attempts on successful login
        self.login_attempts.pop(username, None)
        user.last_login = datetime.now(timezone.utc)
        
        logger.info(f"User authenticated: {username}")
        return user
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token.
        
        Args:
            user: User to create token for
            
        Returns:
            JWT access token
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=self.config.access_token_expire_minutes)
        
        permissions = self.rbac.get_user_permissions(user)
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in permissions],
            "iat": now.timestamp(),
            "exp": expires_at.timestamp(),
            "type": "access"
        }
        
        if self.config.algorithm.startswith('HS'):
            token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
        else:
            token = jwt.encode(payload, self.config.private_key, algorithm=self.config.algorithm)
        
        return token
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token.
        
        Args:
            user: User to create token for
            
        Returns:
            JWT refresh token
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self.config.refresh_token_expire_days)
        
        payload = {
            "sub": user.user_id,
            "iat": now.timestamp(),
            "exp": expires_at.timestamp(),
            "type": "refresh"
        }
        
        if self.config.algorithm.startswith('HS'):
            token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
        else:
            token = jwt.encode(payload, self.config.private_key, algorithm=self.config.algorithm)
        
        # Store refresh token
        self.refresh_tokens[token] = user.user_id
        
        return token
    
    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token data if valid, None otherwise
        """
        try:
            if self.config.algorithm.startswith('HS'):
                payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])
            else:
                payload = jwt.decode(token, self.config.public_key, algorithms=[self.config.algorithm])
            
            user_id = payload.get("sub")
            if not user_id:
                return None
            
            # Check if user exists and is active
            user = self.users.get(user_id)
            if not user or not user.is_active:
                return None
            
            return TokenData(
                user_id=user_id,
                username=payload.get("username", ""),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                issued_at=datetime.fromtimestamp(payload.get("iat", 0), timezone.utc),
                expires_at=datetime.fromtimestamp(payload.get("exp", 0), timezone.utc),
                token_type=payload.get("type", "access")
            )
            
        except ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token
            
        Returns:
            New access token if valid, None otherwise
        """
        token_data = self.verify_token(refresh_token)
        if not token_data or token_data.token_type != "refresh":
            return None
        
        if refresh_token not in self.refresh_tokens:
            return None
        
        user_id = self.refresh_tokens[refresh_token]
        user = self.users.get(user_id)
        
        if not user or not user.is_active:
            return None
        
        # Create new access token
        return self.create_access_token(user)
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token.
        
        Args:
            token: Token to revoke
            
        Returns:
            True if revoked successfully
        """
        # Remove from refresh tokens if it's a refresh token
        if token in self.refresh_tokens:
            del self.refresh_tokens[token]
            return True
        
        # For access tokens, we'd need a token blacklist in production
        return False
    
    def get_current_user(self, credentials: Optional[HTTPAuthorizationCredentials] = None) -> Optional[User]:
        """Get current user from authorization header.
        
        Args:
            credentials: HTTP authorization credentials
            
        Returns:
            Current user if authenticated, None otherwise
        """
        if not credentials:
            return None
        
        token_data = self.verify_token(credentials.credentials)
        if not token_data:
            return None
        
        return self.users.get(token_data.user_id)
    
    def require_auth(self):
        """FastAPI dependency for requiring authentication."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for authentication dependencies")
        
        def _require_auth(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            if not credentials:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            user = self.get_current_user(credentials)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            return user
        
        return _require_auth
    
    def require_permission(self, permission: Permission):
        """FastAPI dependency for requiring specific permission."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for permission dependencies")
        
        def _require_permission(user: User = Depends(self.require_auth())):
            if not self.rbac.check_permission(user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission.value}"
                )
            return user
        
        return _require_permission
    
    def _hash_password(self, password: str) -> str:
        """Hash password using PBKDF2."""
        salt = secrets.token_bytes(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt.hex() + pwdhash.hex()
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt_hex = stored_hash[:64]
            pwdhash_hex = stored_hash[64:]
            
            salt = bytes.fromhex(salt_hex)
            stored_pwdhash = bytes.fromhex(pwdhash_hex)
            
            pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
            
            return hmac.compare_digest(pwdhash, stored_pwdhash)
        except Exception:
            return False
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed login attempts."""
        if username not in self.login_attempts:
            return False
        
        attempts = self.login_attempts[username]
        
        # Remove old attempts
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=self.config.lockout_duration_minutes)
        attempts = [attempt for attempt in attempts if attempt > cutoff_time]
        self.login_attempts[username] = attempts
        
        return len(attempts) >= self.config.max_login_attempts
    
    def _record_failed_login(self, username: str) -> None:
        """Record failed login attempt."""
        if username not in self.login_attempts:
            self.login_attempts[username] = []
        
        self.login_attempts[username].append(datetime.now(timezone.utc))
        logger.warning(f"Failed login attempt for: {username}")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics."""
        now = datetime.now(timezone.utc)
        
        # Count active users
        active_users = sum(1 for user in self.users.values() if user.is_active)
        
        # Count recent logins (last 24 hours)
        recent_logins = sum(
            1 for user in self.users.values() 
            if user.last_login and (now - user.last_login).days < 1
        )
        
        # Count locked accounts
        locked_accounts = sum(1 for username in self.login_attempts if self._is_account_locked(username))
        
        # Count active refresh tokens
        active_refresh_tokens = len(self.refresh_tokens)
        
        return {
            "total_users": len(self.users),
            "active_users": active_users,
            "recent_logins_24h": recent_logins,
            "locked_accounts": locked_accounts,
            "active_refresh_tokens": active_refresh_tokens,
            "failed_login_attempts": len(self.login_attempts)
        }
    
    def create_api_key(self, user: User, name: str, permissions: Optional[List[Permission]] = None) -> str:
        """Create API key for user.
        
        Args:
            user: User to create API key for
            name: API key name
            permissions: Optional specific permissions
            
        Returns:
            API key token
        """
        # Create long-lived token for API access
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=365)  # 1 year expiration
        
        user_permissions = permissions or self.rbac.get_user_permissions(user)
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "api_key_name": name,
            "permissions": [perm.value for perm in user_permissions],
            "iat": now.timestamp(),
            "exp": expires_at.timestamp(),
            "type": "api_key"
        }
        
        if self.config.algorithm.startswith('HS'):
            token = jwt.encode(payload, self.config.secret_key, algorithm=self.config.algorithm)
        else:
            token = jwt.encode(payload, self.config.private_key, algorithm=self.config.algorithm)
        
        logger.info(f"Created API key '{name}' for user: {user.username}")
        return token
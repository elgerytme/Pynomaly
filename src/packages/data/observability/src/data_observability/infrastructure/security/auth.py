"""Authentication and authorization middleware."""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from ..config.settings import settings


class AuthenticationError(Exception):
    """Authentication error."""
    pass


class AuthorizationError(Exception):
    """Authorization error.""" 
    pass


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# JWT Bearer token security
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.security.access_token_expire_minutes
        )
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.security.secret_key, 
        algorithm=settings.security.algorithm
    )
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(
            token,
            settings.security.secret_key,
            algorithms=[settings.security.algorithm]
        )
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {e}")


def get_current_user_from_token(credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
    """Extract user information from JWT token."""
    try:
        payload = verify_token(credentials.credentials)
        username: str = payload.get("sub")
        if username is None:
            raise AuthenticationError("Token missing subject")
        
        return {
            "username": username,
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", []),
            "expires": payload.get("exp")
        }
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


class RoleChecker:
    """Role-based access control checker."""
    
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: Dict[str, Any]) -> bool:
        """Check if user has required roles."""
        user_roles = user.get("roles", [])
        return any(role in user_roles for role in self.allowed_roles)


class PermissionChecker:
    """Permission-based access control checker."""
    
    def __init__(self, required_permissions: list[str]):
        self.required_permissions = required_permissions
    
    def __call__(self, user: Dict[str, Any]) -> bool:
        """Check if user has required permissions."""
        user_permissions = user.get("permissions", [])
        return all(perm in user_permissions for perm in self.required_permissions)


def require_roles(*roles: str):
    """Decorator to require specific roles."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get user from request context (implementation depends on framework setup)
            user = kwargs.get("current_user")
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            checker = RoleChecker(list(roles))
            if not checker(user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient roles. Required: {roles}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_permissions(*permissions: str):
    """Decorator to require specific permissions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            user = kwargs.get("current_user")
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            checker = PermissionChecker(list(permissions))
            if not checker(user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permissions}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# User model for authentication
class User:
    """Simple user model for authentication."""
    
    def __init__(
        self,
        username: str,
        hashed_password: str,
        roles: Optional[list[str]] = None,
        permissions: Optional[list[str]] = None,
        is_active: bool = True
    ):
        self.username = username
        self.hashed_password = hashed_password
        self.roles = roles or []
        self.permissions = permissions or []
        self.is_active = is_active
    
    def check_password(self, password: str) -> bool:
        """Check if provided password is correct."""
        return verify_password(password, self.hashed_password)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary for token payload."""
        return {
            "sub": self.username,
            "roles": self.roles,
            "permissions": self.permissions,
            "active": self.is_active
        }


# Simple in-memory user store (replace with database in production)
class UserStore:
    """Simple user storage."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for testing."""
        # Admin user
        admin_user = User(
            username="admin",
            hashed_password=get_password_hash("admin123"),
            roles=["admin", "user"],
            permissions=[
                "data:read", "data:write", "data:delete",
                "catalog:read", "catalog:write", "catalog:delete",
                "pipeline:read", "pipeline:write", "pipeline:manage",
                "quality:read", "quality:write", "quality:manage"
            ]
        )
        
        # Regular user
        regular_user = User(
            username="user",
            hashed_password=get_password_hash("user123"),
            roles=["user"],
            permissions=[
                "data:read",
                "catalog:read", 
                "pipeline:read",
                "quality:read"
            ]
        )
        
        # Data engineer
        data_engineer = User(
            username="data_engineer",
            hashed_password=get_password_hash("engineer123"),
            roles=["data_engineer", "user"],
            permissions=[
                "data:read", "data:write",
                "catalog:read", "catalog:write",
                "pipeline:read", "pipeline:write",
                "quality:read", "quality:write"
            ]
        )
        
        self.users[admin_user.username] = admin_user
        self.users[regular_user.username] = regular_user
        self.users[data_engineer.username] = data_engineer
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.users.get(username)
    
    def add_user(self, user: User) -> None:
        """Add new user."""
        self.users[user.username] = user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        user = self.get_user(username)
        if not user or not user.is_active:
            return None
        
        if not user.check_password(password):
            return None
        
        return user


# Global user store instance
user_store = UserStore()
"""Authentication and authorization utilities for hexagonal architecture."""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    ANALYST = "analyst"
    VIEWER = "viewer"

@dataclass
class User:
    """User entity."""
    user_id: str
    username: str
    email: str
    roles: List[UserRole]
    is_active: bool = True
    created_at: Optional[str] = None
    last_login: Optional[str] = None

@dataclass
class AuthToken:
    """Authentication token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    scope: Optional[str] = None

class AuthenticationPort(ABC):
    """Port for authentication operations."""
    
    @abstractmethod
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        pass
    
    @abstractmethod
    async def create_access_token(self, user: User) -> AuthToken:
        """Create access token for authenticated user."""
        pass
    
    @abstractmethod
    async def verify_token(self, token: str) -> Optional[User]:
        """Verify and decode access token."""
        pass
    
    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> Optional[AuthToken]:
        """Refresh access token using refresh token."""
        pass

class AuthorizationPort(ABC):
    """Port for authorization operations."""
    
    @abstractmethod
    async def has_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for action on resource."""
        pass
    
    @abstractmethod
    async def get_user_permissions(self, user: User) -> Dict[str, List[str]]:
        """Get all permissions for user."""
        pass

class UserManagementPort(ABC):
    """Port for user management operations."""
    
    @abstractmethod
    async def create_user(self, username: str, password: str, email: str, roles: List[UserRole]) -> User:
        """Create new user."""
        pass
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user information."""
        pass
    
    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        pass
    
    @abstractmethod
    async def list_users(self) -> List[User]:
        """List all users."""
        pass

class JWTAuthenticationService(AuthenticationPort):
    """JWT-based authentication service."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256", access_token_expire_minutes: int = 60):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        # In production, use a proper user store
        self.users = self._initialize_demo_users()
    
    def _initialize_demo_users(self) -> Dict[str, Dict[str, Any]]:
        """Initialize demo users for testing."""
        demo_password = bcrypt.hashpw("demo123".encode('utf-8'), bcrypt.gensalt())
        
        return {
            "admin": {
                "user_id": "admin_001",
                "username": "admin",
                "email": "admin@hexagonal-arch.com",
                "password_hash": demo_password,
                "roles": [UserRole.ADMIN],
                "is_active": True
            },
            "data_scientist": {
                "user_id": "ds_001", 
                "username": "data_scientist",
                "email": "ds@hexagonal-arch.com",
                "password_hash": demo_password,
                "roles": [UserRole.DATA_SCIENTIST, UserRole.ANALYST],
                "is_active": True
            },
            "ml_engineer": {
                "user_id": "mle_001",
                "username": "ml_engineer", 
                "email": "mle@hexagonal-arch.com",
                "password_hash": demo_password,
                "roles": [UserRole.ML_ENGINEER],
                "is_active": True
            },
            "viewer": {
                "user_id": "viewer_001",
                "username": "viewer",
                "email": "viewer@hexagonal-arch.com", 
                "password_hash": demo_password,
                "roles": [UserRole.VIEWER],
                "is_active": True
            }
        }
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        user_data = self.users.get(username)
        if not user_data or not user_data["is_active"]:
            return None
        
        if bcrypt.checkpw(password.encode('utf-8'), user_data["password_hash"]):
            return User(
                user_id=user_data["user_id"],
                username=user_data["username"],
                email=user_data["email"],
                roles=user_data["roles"],
                is_active=user_data["is_active"],
                last_login=datetime.utcnow().isoformat()
            )
        
        return None
    
    async def create_access_token(self, user: User) -> AuthToken:
        """Create JWT access token."""
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "roles": [role.value for role in user.roles],
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "hexagonal-architecture"
        }
        
        access_token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Create refresh token (longer expiry)
        refresh_expire = datetime.utcnow() + timedelta(days=7)
        refresh_payload = {
            "user_id": user.user_id,
            "type": "refresh",
            "exp": refresh_expire,
            "iat": datetime.utcnow()
        }
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        return AuthToken(
            access_token=access_token,
            expires_in=self.access_token_expire_minutes * 60,
            refresh_token=refresh_token
        )
    
    async def verify_token(self, token: str) -> Optional[User]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            if datetime.utcnow() > datetime.fromtimestamp(payload["exp"]):
                return None
            
            roles = [UserRole(role) for role in payload.get("roles", [])]
            
            return User(
                user_id=payload["user_id"],
                username=payload["username"],
                email=payload["email"],
                roles=roles,
                is_active=True
            )
        
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    async def refresh_token(self, refresh_token: str) -> Optional[AuthToken]:
        """Refresh access token using refresh token."""
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "refresh":
                return None
            
            # Get user and create new access token
            user_data = None
            for user in self.users.values():
                if user["user_id"] == payload["user_id"]:
                    user_data = user
                    break
            
            if not user_data:
                return None
            
            user = User(
                user_id=user_data["user_id"],
                username=user_data["username"],
                email=user_data["email"],
                roles=user_data["roles"],
                is_active=user_data["is_active"]
            )
            
            return await self.create_access_token(user)
        
        except jwt.InvalidTokenError:
            return None

class RoleBasedAuthorizationService(AuthorizationPort):
    """Role-based authorization service."""
    
    def __init__(self):
        self.permissions = self._initialize_permissions()
    
    def _initialize_permissions(self) -> Dict[UserRole, Dict[str, List[str]]]:
        """Initialize role-based permissions."""
        return {
            UserRole.ADMIN: {
                "users": ["create", "read", "update", "delete"],
                "data-quality": ["create", "read", "update", "delete", "execute"],
                "machine-learning": ["create", "read", "update", "delete", "train", "predict"],
                "mlops": ["create", "read", "update", "delete", "execute", "deploy"],
                "anomaly-detection": ["create", "read", "update", "delete", "detect"],
                "monitoring": ["read", "configure"],
                "system": ["configure", "manage"]
            },
            UserRole.DATA_SCIENTIST: {
                "data-quality": ["create", "read", "execute"],
                "machine-learning": ["create", "read", "update", "train", "predict"],
                "mlops": ["read", "execute"],
                "anomaly-detection": ["create", "read", "detect"],
                "monitoring": ["read"]
            },
            UserRole.ML_ENGINEER: {
                "machine-learning": ["create", "read", "update", "delete", "train", "predict"],
                "mlops": ["create", "read", "update", "delete", "execute", "deploy"],
                "anomaly-detection": ["read", "detect"],
                "monitoring": ["read"]
            },
            UserRole.ANALYST: {
                "data-quality": ["read", "execute"],
                "machine-learning": ["read", "predict"],
                "anomaly-detection": ["read", "detect"],
                "monitoring": ["read"]
            },
            UserRole.VIEWER: {
                "data-quality": ["read"],
                "machine-learning": ["read"],
                "mlops": ["read"],
                "anomaly-detection": ["read"],
                "monitoring": ["read"]
            }
        }
    
    async def has_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for action on resource."""
        for role in user.roles:
            resource_permissions = self.permissions.get(role, {}).get(resource, [])
            if action in resource_permissions:
                return True
        return False
    
    async def get_user_permissions(self, user: User) -> Dict[str, List[str]]:
        """Get all permissions for user."""
        all_permissions = {}
        
        for role in user.roles:
            role_permissions = self.permissions.get(role, {})
            for resource, actions in role_permissions.items():
                if resource not in all_permissions:
                    all_permissions[resource] = []
                all_permissions[resource].extend(actions)
        
        # Remove duplicates
        for resource in all_permissions:
            all_permissions[resource] = list(set(all_permissions[resource]))
        
        return all_permissions

class InMemoryUserManagementService(UserManagementPort):
    """In-memory user management service (for demo/testing)."""
    
    def __init__(self):
        self.users = {}
        self._initialize_demo_users()
    
    def _initialize_demo_users(self):
        """Initialize demo users."""
        demo_users = [
            User("admin_001", "admin", "admin@hexagonal-arch.com", [UserRole.ADMIN]),
            User("ds_001", "data_scientist", "ds@hexagonal-arch.com", [UserRole.DATA_SCIENTIST, UserRole.ANALYST]),
            User("mle_001", "ml_engineer", "mle@hexagonal-arch.com", [UserRole.ML_ENGINEER]),
            User("viewer_001", "viewer", "viewer@hexagonal-arch.com", [UserRole.VIEWER])
        ]
        
        for user in demo_users:
            self.users[user.user_id] = user
    
    async def create_user(self, username: str, password: str, email: str, roles: List[UserRole]) -> User:
        """Create new user."""
        user_id = f"user_{len(self.users) + 1:03d}"
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles,
            created_at=datetime.utcnow().isoformat()
        )
        self.users[user_id] = user
        return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user information."""
        user = self.users.get(user_id)
        if not user:
            return False
        
        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        return True
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        if user_id in self.users:
            del self.users[user_id]
            return True
        return False
    
    async def list_users(self) -> List[User]:
        """List all users."""
        return list(self.users.values())

# Global service instances (in production, use dependency injection)
auth_service = JWTAuthenticationService("your-secret-key-change-in-production")
authz_service = RoleBasedAuthorizationService()
user_service = InMemoryUserManagementService()

async def authenticate_user(username: str, password: str) -> Optional[User]:
    """Convenience function for user authentication."""
    return await auth_service.authenticate_user(username, password)

async def create_access_token(user: User) -> AuthToken:
    """Convenience function for token creation."""
    return await auth_service.create_access_token(user)

async def verify_token(token: str) -> Optional[User]:
    """Convenience function for token verification."""
    return await auth_service.verify_token(token)

async def has_permission(user: User, resource: str, action: str) -> bool:
    """Convenience function for permission checking."""
    return await authz_service.has_permission(user, resource, action)
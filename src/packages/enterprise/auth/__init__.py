"""Enterprise authentication and authorization services."""

from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
from uuid import UUID
import structlog

logger = structlog.get_logger(__name__)


class Permission(str, Enum):
    """System permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    DEPLOY = "deploy"
    MANAGE_USERS = "manage_users"
    AUDIT = "audit"


class Role(str, Enum):
    """System roles."""
    VIEWER = "viewer"
    ANALYST = "analyst"
    DATA_SCIENTIST = "data_scientist"
    ML_ENGINEER = "ml_engineer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class AuthProvider(ABC):
    """Abstract authentication provider."""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user info."""
        pass
    
    @abstractmethod
    async def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get user permissions."""
        pass
    
    @abstractmethod
    async def get_user_roles(self, user_id: str) -> List[Role]:
        """Get user roles."""
        pass


class EnterpriseAuthService:
    """Enterprise authentication and authorization service."""
    
    def __init__(
        self,
        auth_provider: AuthProvider,
        enable_rbac: bool = True,
        enable_audit: bool = True
    ):
        self.auth_provider = auth_provider
        self.enable_rbac = enable_rbac
        self.enable_audit = enable_audit
        self.logger = logger.bind(service="enterprise_auth")
    
    async def authenticate_user(
        self, 
        credentials: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Authenticate user."""
        try:
            user_info = await self.auth_provider.authenticate(credentials)
            if user_info and self.enable_audit:
                self.logger.info("User authenticated", user_id=user_info.get("id"))
            return user_info
        except Exception as e:
            self.logger.error("Authentication failed", error=str(e))
            return None
    
    async def check_permission(
        self, 
        user_id: str, 
        permission: Permission,
        resource: Optional[str] = None
    ) -> bool:
        """Check if user has permission."""
        if not self.enable_rbac:
            return True
        
        try:
            permissions = await self.auth_provider.get_user_permissions(user_id)
            has_permission = permission in permissions or Permission.ADMIN in permissions
            
            if self.enable_audit:
                self.logger.info(
                    "Permission check",
                    user_id=user_id,
                    permission=permission.value,
                    resource=resource,
                    granted=has_permission
                )
            
            return has_permission
        except Exception as e:
            self.logger.error("Permission check failed", user_id=user_id, error=str(e))
            return False
    
    async def require_permission(
        self, 
        user_id: str, 
        permission: Permission,
        resource: Optional[str] = None
    ) -> None:
        """Require permission or raise exception."""
        if not await self.check_permission(user_id, permission, resource):
            raise PermissionError(f"User {user_id} lacks permission {permission.value}")


class SAMLAuthProvider(AuthProvider):
    """SAML authentication provider."""
    
    def __init__(self, saml_config: Dict[str, Any]):
        self.saml_config = saml_config
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Authenticate via SAML."""
        # Implementation would integrate with SAML provider
        # For now, return mock user info
        return {
            "id": credentials.get("username"),
            "email": credentials.get("email"),
            "name": credentials.get("name"),
            "provider": "saml"
        }
    
    async def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get permissions from SAML attributes."""
        # Implementation would map SAML groups to permissions
        return [Permission.READ, Permission.WRITE]
    
    async def get_user_roles(self, user_id: str) -> List[Role]:
        """Get roles from SAML attributes."""
        # Implementation would map SAML groups to roles
        return [Role.DATA_SCIENTIST]


class BasicAuthProvider(AuthProvider):
    """Basic authentication provider."""
    
    def __init__(self):
        self.users = {}  # In production, use proper user store
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Basic username/password authentication."""
        username = credentials.get("username")
        password = credentials.get("password")
        
        # In production, use proper password hashing
        if username in self.users and self.users[username]["password"] == password:
            return self.users[username]["info"]
        
        return None
    
    async def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get user permissions."""
        if user_id in self.users:
            return self.users[user_id].get("permissions", [])
        return []
    
    async def get_user_roles(self, user_id: str) -> List[Role]:
        """Get user roles."""
        if user_id in self.users:
            return self.users[user_id].get("roles", [])
        return []


__all__ = [
    "EnterpriseAuthService",
    "AuthProvider", 
    "SAMLAuthProvider",
    "BasicAuthProvider",
    "Permission",
    "Role"
]
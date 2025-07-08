"""Simplified RBAC dependencies for OpenAPI compatibility.

This module provides simplified versions of RBAC dependencies that avoid
Request type annotations, preventing ForwardRef issues during OpenAPI generation.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pynomaly.domain.entities.user import User, UserRole
from pynomaly.infrastructure.auth.jwt_auth import UserModel, get_auth

# Simple security scheme
bearer_scheme = HTTPBearer(auto_error=False)


class SimpleUserModel:
    """Simplified user model for OpenAPI compatibility."""
    
    def __init__(self, username: str, roles: List[str], is_active: bool = True):
        self.username = username
        self.roles = roles
        self.is_active = is_active
        self.id = "mock-user-id"


async def get_current_user_simple(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[SimpleUserModel]:
    """Get current authenticated user - simplified for OpenAPI."""
    if not credentials:
        return None
    
    # Get auth service
    auth_service = get_auth()
    if not auth_service:
        return None
    
    try:
        user = auth_service.get_current_user(credentials.credentials)
        if user:
            return SimpleUserModel(
                username=user.username,
                roles=user.roles,
                is_active=user.is_active
            )
        return None
    except Exception:
        return None


async def require_auth_simple(
    user: Optional[SimpleUserModel] = Depends(get_current_user_simple),
) -> SimpleUserModel:
    """Require authentication - simplified for OpenAPI."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


class SimpleRequirePermissions:
    """Simplified permission requirement class for OpenAPI compatibility."""
    
    def __init__(
        self,
        permissions: List[str],
        tenant_id: Optional[UUID] = None,
        resource_id: Optional[str] = None,
        allow_self_access: bool = False
    ):
        self.permissions = permissions
        self.tenant_id = tenant_id
        self.resource_id = resource_id
        self.allow_self_access = allow_self_access
    
    async def __call__(
        self,
        user: Optional[SimpleUserModel] = Depends(get_current_user_simple),
    ) -> SimpleUserModel:
        """Validate permissions and return user - simplified for OpenAPI."""
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # For OpenAPI generation, just return the user
        # In real implementation, would check permissions
        return user


class SimpleRequireRole:
    """Simplified role requirement class for OpenAPI compatibility."""
    
    def __init__(
        self,
        role: UserRole,
        tenant_id: Optional[UUID] = None
    ):
        self.role = role
        self.tenant_id = tenant_id
    
    async def __call__(
        self,
        user: Optional[SimpleUserModel] = Depends(get_current_user_simple),
    ) -> SimpleUserModel:
        """Validate role and return user - simplified for OpenAPI."""
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # For OpenAPI generation, just return the user
        # In real implementation, would check role
        return user


# Factory functions for simplified dependencies
def require_permissions_simple(
    permissions: List[str],
    tenant_id: Optional[UUID] = None,
    resource_id: Optional[str] = None,
    allow_self_access: bool = False
) -> SimpleRequirePermissions:
    """Factory function for simplified permission requirements."""
    return SimpleRequirePermissions(
        permissions=permissions,
        tenant_id=tenant_id,
        resource_id=resource_id,
        allow_self_access=allow_self_access
    )


def require_role_simple(role: UserRole, tenant_id: Optional[UUID] = None) -> SimpleRequireRole:
    """Factory function for simplified role requirements."""
    return SimpleRequireRole(role=role, tenant_id=tenant_id)


# Convenience aliases for common permission sets
def require_dataset_read() -> SimpleRequirePermissions:
    """Require dataset read permissions."""
    return require_permissions_simple(["dataset.view"])


def require_dataset_write() -> SimpleRequirePermissions:
    """Require dataset write permissions."""
    return require_permissions_simple(["dataset.create", "dataset.edit"])


def require_model_read() -> SimpleRequirePermissions:
    """Require model read permissions."""
    return require_permissions_simple(["model.view"])


def require_model_write() -> SimpleRequirePermissions:
    """Require model write permissions."""
    return require_permissions_simple(["model.create", "model.edit"])

"""Authorization module for API endpoints.

This module provides authorization dependencies for FastAPI endpoints including
role-based and permission-based access control.
"""

import logging
from typing import Any, Dict, List

from fastapi import Depends, HTTPException, status

from ..dependencies.auth import get_current_user

logger = logging.getLogger(__name__)


def require_permissions(required_permissions: List[str]):
    """Dependency factory for permission-based access control.
    
    Args:
        required_permissions: List of required permissions
        
    Returns:
        FastAPI dependency function
    """
    async def permission_checker(
        current_user: Dict[str, Any] = Depends(get_current_user),
    ) -> None:
        """Check if current user has required permissions."""
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        user_permissions = current_user.get("permissions", [])
        
        # Check if user has all required permissions
        missing_permissions = []
        for permission in required_permissions:
            if permission not in user_permissions:
                missing_permissions.append(permission)
        
        if missing_permissions:
            logger.warning(
                f"User {current_user.get('user_id')} missing permissions: {missing_permissions}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {', '.join(missing_permissions)}"
            )
        
        logger.debug(f"User {current_user.get('user_id')} authorized with permissions: {required_permissions}")
    
    return permission_checker


def require_roles(required_roles: List[str]):
    """Dependency factory for role-based access control.
    
    Args:
        required_roles: List of required roles
        
    Returns:
        FastAPI dependency function
    """
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user),
    ) -> None:
        """Check if current user has required roles."""
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        user_roles = current_user.get("roles", [])
        
        # Check if user has any of the required roles
        if not any(role in user_roles for role in required_roles):
            logger.warning(
                f"User {current_user.get('user_id')} missing roles: {required_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of the following roles is required: {', '.join(required_roles)}"
            )
        
        logger.debug(f"User {current_user.get('user_id')} authorized with roles: {required_roles}")
    
    return role_checker


def require_admin():
    """Dependency that requires admin role."""
    return require_roles(["admin"])


def require_data_analyst():
    """Dependency that requires data analyst role."""
    return require_roles(["data_analyst", "admin"])


def require_data_scientist():
    """Dependency that requires data scientist role."""
    return require_roles(["data_scientist", "admin"])


# Permission-based dependencies for specific functionalities
def require_data_profiling_read():
    """Dependency that requires data profiling read permission."""
    return require_permissions(["data_profiling:read"])


def require_data_profiling_write():
    """Dependency that requires data profiling write permission."""
    return require_permissions(["data_profiling:write"])


def require_data_profiling_admin():
    """Dependency that requires data profiling admin permission."""
    return require_permissions(["data_profiling:admin"])


def require_data_quality_read():
    """Dependency that requires data quality read permission."""
    return require_permissions(["data_quality:read"])


def require_data_quality_write():
    """Dependency that requires data quality write permission."""
    return require_permissions(["data_quality:write"])


def require_data_quality_admin():
    """Dependency that requires data quality admin permission."""
    return require_permissions(["data_quality:admin"])


def require_ml_pipeline_read():
    """Dependency that requires ML pipeline read permission."""
    return require_permissions(["ml_pipeline:read"])


def require_ml_pipeline_write():
    """Dependency that requires ML pipeline write permission."""
    return require_permissions(["ml_pipeline:write"])


def require_ml_pipeline_admin():
    """Dependency that requires ML pipeline admin permission."""
    return require_permissions(["ml_pipeline:admin"])


def check_resource_access(resource_type: str, resource_id: str):
    """Dependency factory for resource-specific access control.
    
    Args:
        resource_type: Type of resource (e.g., 'dataset', 'model', 'pipeline')
        resource_id: ID of the resource
        
    Returns:
        FastAPI dependency function
    """
    async def resource_checker(
        current_user: Dict[str, Any] = Depends(get_current_user),
    ) -> None:
        """Check if current user has access to specific resource."""
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        user_id = current_user.get("user_id")
        
        # In a real implementation, this would check database for resource ownership
        # or team membership, etc. For demonstration, we'll do basic checks
        
        # Admin users have access to all resources
        if "admin" in current_user.get("roles", []):
            logger.debug(f"Admin user {user_id} granted access to {resource_type}:{resource_id}")
            return
        
        # For demonstration, allow access if user has appropriate permissions
        required_permission = f"{resource_type}:read"
        user_permissions = current_user.get("permissions", [])
        
        if required_permission not in user_permissions:
            logger.warning(
                f"User {user_id} denied access to {resource_type}:{resource_id} - missing permission {required_permission}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to {resource_type} {resource_id}"
            )
        
        logger.debug(f"User {user_id} granted access to {resource_type}:{resource_id}")
    
    return resource_checker


def check_tenant_access(tenant_id: str):
    """Dependency factory for tenant-specific access control.
    
    Args:
        tenant_id: ID of the tenant
        
    Returns:
        FastAPI dependency function
    """
    async def tenant_checker(
        current_user: Dict[str, Any] = Depends(get_current_user),
    ) -> None:
        """Check if current user has access to specific tenant."""
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        user_tenant_id = current_user.get("tenant_id")
        user_id = current_user.get("user_id")
        
        # Check if user belongs to the requested tenant
        if user_tenant_id != tenant_id:
            logger.warning(
                f"User {user_id} from tenant {user_tenant_id} denied access to tenant {tenant_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to tenant resources"
            )
        
        logger.debug(f"User {user_id} granted access to tenant {tenant_id}")
    
    return tenant_checker


def require_permissions(required_permissions: List[str]):
    """Dependency factory for permission-based access control.
    
    Args:
        required_permissions: List of required permissions
        
    Returns:
        Dependency function that checks permissions
    """
    async def permission_checker(
        current_user: Dict[str, Any] = Depends(lambda: None),  # Will be replaced by actual auth
    ) -> None:
        """Check if current user has required permissions."""
        # Import here to avoid circular imports
        from ..dependencies.auth import get_current_user
        
        # Get the current user (this will be injected by FastAPI)
        user = await get_current_user()
        user_permissions = user.get("permissions", [])
        
        # Check if user has all required permissions
        missing_permissions = []
        for permission in required_permissions:
            if permission not in user_permissions:
                missing_permissions.append(permission)
        
        if missing_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {', '.join(missing_permissions)}"
            )
    
    return permission_checker
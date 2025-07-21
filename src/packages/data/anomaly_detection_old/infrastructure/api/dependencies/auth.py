"""Simplified authentication dependencies for OpenAPI compatibility."""

from typing import Any, Dict
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from interfaces.infrastructure.auth.jwt_auth import UserModel, get_auth

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user_safe(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserModel | None:
    """Get current authenticated user.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        UserModel instance or None if not authenticated
    """
    if not credentials:
        return None

    # Get auth service
    auth_service = get_auth()
    if not auth_service:
        return None

    try:
        return auth_service.get_current_user(credentials.credentials)
    except Exception:
        return None


def require_auth_safe(user: UserModel = Depends(get_current_user_safe)) -> UserModel:
    """Require authenticated user.

    Returns:
        UserModel if authenticated

    Raises:
        HTTPException: If not authenticated
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """Get current user from JWT token (returns dict format for compatibility)."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get auth service
    auth_service = get_auth()
    if not auth_service:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication service unavailable",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user = auth_service.get_current_user(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Return user information as dict for compatibility
        return {
            "user_id": user.username,
            "username": user.username,
            "roles": user.roles,
            "permissions": getattr(user, 'permissions', []),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


class SimpleAuthContext:
    """Simple context for user authentication and permissions."""

    def __init__(self, user: UserModel, permissions: list[str]) -> None:
        self.user = user
        self.permissions = permissions

    # Define properties for easy attribute access
    @property
    def username(self) -> str:
        return self.user.username

    @property
    def roles(self) -> list[str]:
        return self.user.roles if self.user else []

    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions


# Create additional role-based requirements as simplified functions
async def require_role_safe(
    roles: list[str], user: UserModel = Depends(get_current_user_safe)
) -> UserModel:
    """Require user to have specific roles.

    Args:
        roles: List of roles

    Returns:
        UserModel if user has required roles

    Raises:
        HTTPException: If user does not have required roles
    """
    if user is None or not any(role in user.roles for role in roles):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    return user


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """Get current user for endpoint compatibility."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # For demonstration, return mock user data
    # In production, this would validate the token and return actual user data
    return {
        "user_id": "demo_user",
        "username": "demo_user", 
        "roles": ["user", "analyst"],
        "permissions": [
            "data_profiling:read", "data_profiling:write", "data_profiling:delete", "data_profiling:admin",
            "data_quality:read", "data_quality:write", "data_quality:delete", "data_quality:admin"
        ]
    }

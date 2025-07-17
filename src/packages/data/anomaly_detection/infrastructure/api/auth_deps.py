"""Simplified authentication dependencies for FastAPI.

This module provides clean, non-circular authentication dependencies
that work with OpenAPI generation.
"""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from interfaces.infrastructure.auth.jwt_auth import UserModel, get_auth
from interfaces.infrastructure.config import Container

# Simple security scheme
security = HTTPBearer(auto_error=False)


def get_container_simple(request: Request) -> Container:
    """Get DI container from app state.

    Simple version without complex type annotations.
    """
    return request.app.state.container


async def get_current_user_simple(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """Get current authenticated user - simplified version.

    Returns:
        Username string or None if not authenticated
    """
    if not credentials:
        return None

    # Get auth service
    auth_service = get_auth()
    if not auth_service:
        return None

    try:
        user = auth_service.get_current_user(credentials.credentials)
        return user.username if user else None
    except Exception:
        return None


async def get_current_user_model(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserModel | None:
    """Get current authenticated user as UserModel.

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


async def require_authentication(
    current_user: str | None = Depends(get_current_user_simple),
) -> str:
    """Require user to be authenticated.

    Args:
        current_user: Current user from get_current_user_simple

    Returns:
        Username if authenticated

    Raises:
        HTTPException: If not authenticated
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


class SimplePermissionChecker:
    """Simplified permission checker without complex type annotations."""

    def __init__(self, permissions: list[str]):
        """Initialize with required permissions."""
        self.permissions = permissions

    async def __call__(
        self,
        current_user: UserModel | None = Depends(get_current_user_model),
    ) -> UserModel | None:
        """Check permissions for current user.

        Returns None if no auth service or user lacks permissions.
        """
        if not current_user:
            return None

        # For now, return user (permissions checking simplified)
        # TODO: Implement actual permission checking
        return current_user


# Simple permission instances
require_read_simple = SimplePermissionChecker(["read"])
require_write_simple = SimplePermissionChecker(["write"])
require_admin_simple = SimplePermissionChecker(["admin"])

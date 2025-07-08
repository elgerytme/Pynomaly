"""API dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pynomaly.infrastructure.auth import (
    UserModel,
    get_current_user as auth_get_current_user,
    require_analyst,
    require_data_scientist,
    require_super_admin,
    require_tenant_admin,
    require_viewer,
)
from pynomaly.infrastructure.config import Container

# Security scheme
security = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    """Get DI container from app state."""
    return request.app.state.container


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    container: Container = Depends(get_container),
    request: Request = Depends(),
) -> str | None:
    """Get current authenticated user.

    Returns None if auth is disabled, otherwise validates token.
    Supports both Bearer token and cookie-based authentication.
    """
    settings = container.config()

    if not settings.auth_enabled:
        return None

    # Get auth service
    from pynomaly.domain.exceptions import AuthenticationError
    from pynomaly.infrastructure.auth import get_auth
    from fastapi import Request

    auth_service = get_auth()
    if not auth_service:
        # If auth is enabled but service unavailable, raise error
        # Return None (handled gracefully for web endpoints)
        raise HTTPException(
            status_code=503, detail="Authentication service not available"
        )

    # Try to get token from Authorization header first
    request = Request()
    token = credentials.credentials if credentials else request.cookies.get("access_token")[7:] if request.cookies.get("access_token").startswith("Bearer ") else None

    if not token:
        # Require authentication
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Validate JWT token and get user
        user = auth_service.get_current_user(token)
        return user.username

    except AuthenticationError as e:
        # Raise HTTP exception if authentication fails
        raise HTTPException(
            status_code=401, detail=str(e), headers={"WWW-Authenticate": "Bearer"}
        )


async def require_auth(
    user: Annotated[str | None, Depends(get_current_user)],
) -> str:
    """Require authenticated user."""
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# RBAC dependencies - these now use proper role-based access control
# Import the actual RBAC dependencies from the auth module

# For backward compatibility, provide simplified role mapping
def require_read() -> UserModel:
    """Require user with read permissions (viewer role or higher)."""
    return require_viewer

def require_write() -> UserModel:
    """Require user with write permissions (analyst role or higher)."""
    return require_analyst

def require_admin() -> UserModel:
    """Require user with admin permissions (tenant admin or higher)."""
    return require_tenant_admin

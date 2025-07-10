"""API dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pynomaly.infrastructure.auth import (
    UserModel,
    require_analyst,
    require_tenant_admin,
    require_viewer,
)
from pynomaly.infrastructure.config import Container

# Security scheme
security = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    """Get DI container from app state."""
    try:
        return request.app.state.container
    except AttributeError:
        # Create default container for testing/fallback scenarios
        from pynomaly.infrastructure.config import create_container

        container = create_container()
        request.app.state.container = container
        return container


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    container: Container = Depends(get_container),
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

    auth_service = get_auth()
    if not auth_service:
        # If auth is enabled but service unavailable, raise error for API endpoints
        # For web endpoints, we'll handle this gracefully
        if request.url.path.startswith("/api/"):
            raise HTTPException(
                status_code=503, detail="Authentication service not available"
            )
        return None

    # Try to get token from Authorization header first
    token = None
    if credentials:
        token = credentials.credentials
    else:
        # Try to get token from cookie (for Web UI)
        cookie_token = request.cookies.get("access_token")
        if cookie_token and cookie_token.startswith("Bearer "):
            token = cookie_token[7:]  # Remove "Bearer " prefix

    if not token:
        # For API endpoints, require authentication
        if request.url.path.startswith("/api/"):
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # For web endpoints, return None (will redirect to login)
        return None

    try:
        # Validate JWT token and get user
        user = auth_service.get_current_user(token)
        return user.username

    except AuthenticationError as e:
        # For API endpoints, raise HTTP exception
        if request.url.path.startswith("/api/"):
            raise HTTPException(
                status_code=401, detail=str(e), headers={"WWW-Authenticate": "Bearer"}
            )
        # For web endpoints, return None (will redirect to login)
        return None


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

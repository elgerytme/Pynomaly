"""API dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pynomaly.infrastructure.config import Container

# Security scheme
security = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    """Get DI container from app state."""
    return request.app.state.container


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
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


async def require_read(
    user: Annotated[str | None, Depends(get_current_user)],
) -> str:
    """Require user with read permissions."""
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    # For now, any authenticated user has read permissions
    # In a full RBAC system, this would check specific permissions
    return user


async def require_write(
    user: Annotated[str | None, Depends(get_current_user)],
) -> str:
    """Require user with write permissions."""
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    # For now, any authenticated user has write permissions
    # In a full RBAC system, this would check specific permissions
    return user


async def require_admin(
    user: Annotated[str | None, Depends(get_current_user)],
) -> str:
    """Require user with admin permissions."""
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    # For now, any authenticated user has admin permissions
    # In a full RBAC system, this would check admin roles
    return user

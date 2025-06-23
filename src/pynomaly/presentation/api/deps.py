"""API dependencies."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from pynomaly.infrastructure.config import Container


# Security scheme
security = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    """Get DI container from app state."""
    return request.app.state.container


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    container: Container = Depends(get_container)
) -> Optional[str]:
    """Get current authenticated user.
    
    Returns None if auth is disabled, otherwise validates token.
    """
    settings = container.config()
    
    if not settings.auth_enabled:
        return None
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # TODO: Implement actual JWT validation
    # For now, just return a mock user
    return "user@example.com"


async def require_auth(
    user: Annotated[Optional[str], Depends(get_current_user)]
) -> str:
    """Require authenticated user."""
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    return user
"""Authentication middleware and dependencies for FastAPI.

This module provides FastAPI dependencies for:
- JWT token validation
- API key validation
- Permission checking
- Rate limiting
"""

from __future__ import annotations

import hashlib
import time
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from monorepo.domain.exceptions import AuthenticationError, AuthorizationError
from monorepo.infrastructure.auth.jwt_auth import JWTAuthService, get_auth
from monorepo.infrastructure.cache import get_cache
from monorepo.infrastructure.config import Settings

# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """Simple rate limiter using cache."""

    def __init__(self, requests: int = 100, window: int = 60):
        """Initialize rate limiter.

        Args:
            requests: Number of requests allowed
            window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.cache = get_cache()

    def __call__(self, request: Request) -> None:
        """Check rate limit for request.

        Args:
            request: FastAPI request

        Raises:
            HTTPException: If rate limit exceeded
        """
        if not self.cache or not self.cache.enabled:
            return

        # Get client identifier
        client_id = self._get_client_id(request)
        key = f"rate_limit:{client_id}"

        # Get current count
        current = self.cache.get(key, 0)

        if current >= self.requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.requests} requests per {self.window} seconds.",
            )

        # Increment counter
        self.cache.set(key, current + 1, ttl=self.window)

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request.

        Args:
            request: FastAPI request

        Returns:
            Client ID
        """
        # Use IP address or user ID if authenticated
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0]
        else:
            ip = request.client.host if request.client else "unknown"

        return hashlib.md5(ip.encode()).hexdigest()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    api_key: Annotated[str | None, Depends(api_key_scheme)],
    auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
):
    """Get current authenticated user.

    Args:
        credentials: Bearer token credentials
        api_key: API key
        auth_service: Auth service

    Returns:
        Current user or None

    Raises:
        HTTPException: If authentication fails
    """
    if not auth_service:
        return None

    # Try bearer token first
    if credentials and credentials.credentials:
        try:
            return auth_service.get_current_user(credentials.credentials)
        except AuthenticationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Try API key
    if api_key:
        try:
            return auth_service.authenticate_api_key(api_key)
        except AuthenticationError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    return None


async def get_current_active_user(
    current_user: Annotated[Any, Depends(get_current_user)],
):
    """Get current active user, enforcing auth if enabled.

    Args:
        current_user: Current user from auth

    Returns:
        Active user or None

    Raises:
        HTTPException: If auth required but not provided
    """
    # Get settings
    settings = Settings()

    # Check if auth is enabled and user is required
    if settings.security.auth_enabled and not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Return the user as-is (already a UserModel or None)
    return current_user


class PermissionChecker:
    """Permission checker dependency."""

    def __init__(self, permissions: list[str]):
        """Initialize with required permissions.

        Args:
            permissions: Required permissions
        """
        self.permissions = permissions

    async def __call__(
        self,
        current_user: Annotated[Any, Depends(get_current_user)],
        auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
    ) -> Any:
        """Check permissions for current user.

        Args:
            current_user: Current user
            auth_service: Auth service

        Returns:
            User if authorized

        Raises:
            HTTPException: If not authorized
        """
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )

        if not auth_service:
            return current_user

        try:
            auth_service.require_permissions(current_user, self.permissions)
            return current_user
        except AuthorizationError as e:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


# Common permission dependencies
require_read = PermissionChecker(["detectors:read", "datasets:read"])
require_write = PermissionChecker(["detectors:write", "datasets:write"])
require_admin = PermissionChecker(["users:write", "settings:write"])


def require_permission(permission: str) -> PermissionChecker:
    """Create a permission checker for a specific permission.

    Args:
        permission: Required permission string

    Returns:
        PermissionChecker instance
    """
    return PermissionChecker([permission])


# Role-based access control dependencies
class RoleChecker:
    """Role checker dependency that validates user has specific roles."""

    def __init__(self, roles: list[str]):
        """Initialize with required roles.

        Args:
            roles: Required roles
        """
        self.roles = roles

    async def __call__(
        self,
        current_user: Annotated[Any, Depends(get_current_user)],
        auth_service: Annotated[JWTAuthService | None, Depends(get_auth)],
    ) -> Any:
        """Check roles for current user.

        Args:
            current_user: Current user
            auth_service: Auth service

        Returns:
            User if authorized

        Raises:
            HTTPException: If not authenticated or authorized
        """
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not auth_service:
            # If no auth service, allow access (backward compatibility)
            return current_user

        # Check if user has any of the required roles
        user_roles = set(current_user.roles or [])
        required_roles = set(self.roles)

        # Super admin has access to everything
        if "super_admin" in user_roles:
            return current_user

        if not user_roles.intersection(required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(self.roles)}",
            )

        return current_user


def require_role(*roles: str) -> RoleChecker:
    """Create a role checker for specific roles.

    Args:
        *roles: Required roles

    Returns:
        RoleChecker instance
    """
    return RoleChecker(list(roles))


# Predefined role dependencies
require_super_admin = RoleChecker(["super_admin"])
require_tenant_admin = RoleChecker(["tenant_admin", "super_admin"])
require_data_scientist = RoleChecker(["data_scientist", "tenant_admin", "super_admin"])
require_analyst = RoleChecker(
    ["analyst", "data_scientist", "tenant_admin", "super_admin"]
)
require_viewer = RoleChecker(
    ["viewer", "analyst", "data_scientist", "tenant_admin", "super_admin"]
)

# Rate limiter instances
default_limiter = RateLimiter(requests=100, window=60)
strict_limiter = RateLimiter(requests=10, window=60)


async def track_request_metrics(request: Request, call_next):
    """Middleware to track request metrics.

    Args:
        request: FastAPI request
        call_next: Next middleware

    Returns:
        Response
    """
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Track metrics
    time.time() - start_time

    # Record metrics if telemetry available (temporarily disabled)
    # from monorepo.infrastructure.monitoring import get_telemetry
    # telemetry = get_telemetry()
    # if telemetry:
    #     telemetry.record_request(
    #         method=request.method,
    #         endpoint=request.url.path,
    #         status_code=response.status_code,
    #         duration=duration
    #     )

    return response


def create_auth_context(user: Any) -> dict[str, any]:
    """Create authentication context for request.

    Args:
        user: Current user

    Returns:
        Auth context dict
    """
    if not user:
        return {
            "authenticated": False,
            "user_id": None,
            "username": None,
            "roles": [],
            "permissions": [],
        }

    auth_service = get_auth()
    permissions = []
    if auth_service:
        permissions = auth_service._get_permissions_for_roles(user.roles)

    return {
        "authenticated": True,
        "user_id": user.id,
        "username": user.username,
        "roles": user.roles,
        "permissions": permissions,
        "is_superuser": user.is_superuser,
    }


class AuthenticationMiddleware:
    """Authentication middleware for processing requests."""

    def __init__(self, auth_service: JWTAuthService):
        """Initialize authentication middleware.

        Args:
            auth_service: JWT authentication service
        """
        self.auth_service = auth_service

    async def authenticate(self, request: Request):
        """Authenticate request and return user.

        Args:
            request: HTTP request

        Returns:
            Authenticated user

        Raises:
            AuthenticationError: If authentication fails
        """
        # Check for Authorization header
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            return self.auth_service.get_current_user(token)

        # Check for API key header
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return self.auth_service.authenticate_api_key(api_key)

        # No authentication credentials provided
        raise AuthenticationError("No authentication credentials provided")

    def _parse_bearer_token(self, authorization: str) -> str | None:
        """Parse bearer token from Authorization header.

        Args:
            authorization: Authorization header value

        Returns:
            Bearer token or None
        """
        if not authorization:
            return None

        parts = authorization.split(" ")
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthenticationError("Invalid authentication header format")

        return parts[1]

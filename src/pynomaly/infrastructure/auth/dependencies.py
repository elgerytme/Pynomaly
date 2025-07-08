"""Authentication dependencies for FastAPI endpoints."""

from collections.abc import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt_auth import JWTAuthService, get_auth

security = HTTPBearer()


def require_role(*roles: str) -> Callable:
    """Create a dependency that requires specific roles.

    Args:
        *roles: Required roles (e.g., 'admin', 'developer', 'business')

    Returns:
        FastAPI dependency function

    Example:
        @app.get("/admin-only")
        async def admin_endpoint(user = Depends(require_role('admin'))):
            return {"message": "Admin access granted"}
    """

    def dependency(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        auth_service: JWTAuthService = Depends(get_auth),
    ):
        """Dependency function that validates user has required roles."""
        try:
            # Decode and validate token
            token_payload = auth_service.decode_token(credentials.credentials)
            user = auth_service.get_user(token_payload.sub)

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
                )

            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is disabled",
                )

            # Check if user has any of the required roles
            user_roles = set(user.roles or [])
            required_roles = set(roles)

            if not user_roles.intersection(required_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required roles: {', '.join(roles)}",
                )

            return user

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            ) from e

    return dependency


def require_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: JWTAuthService = Depends(get_auth),
):
    """Dependency for API key authentication.

    Args:
        credentials: HTTP Bearer credentials
        auth_service: JWT auth service

    Returns:
        Authenticated user

    Raises:
        HTTPException: If API key is invalid
    """
    try:
        # Check if this is an API key (starts with 'pyn_')
        if credentials.credentials.startswith("pyn_"):
            user = auth_service.authenticate_api_key(credentials.credentials)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
                )
            return user
        else:
            # Fall back to JWT token validation
            token_payload = auth_service.decode_token(credentials.credentials)
            user = auth_service.get_user(token_payload.sub)

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
                )

            return user

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        ) from e


def require_role_or_api_key(*roles: str) -> Callable:
    """Create a dependency that accepts either role-based JWT auth or API key auth.

    Args:
        *roles: Required roles for JWT authentication

    Returns:
        FastAPI dependency function that accepts either auth method
    """

    def dependency(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        auth_service: JWTAuthService = Depends(get_auth),
    ):
        """Dependency function that validates either API key or JWT with roles."""
        try:
            # Check if this is an API key
            if credentials.credentials.startswith("pyn_"):
                user = auth_service.authenticate_api_key(credentials.credentials)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid API key",
                    )
                return user
            else:
                # JWT token - check roles
                token_payload = auth_service.decode_token(credentials.credentials)
                user = auth_service.get_user(token_payload.sub)

                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User not found",
                    )

                if not user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account is disabled",
                    )

                # Check if user has any of the required roles
                user_roles = set(user.roles or [])
                required_roles = set(roles)

                if not user_roles.intersection(required_roles):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions. Required roles: {', '.join(roles)}",
                    )

                return user

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            ) from e

    return dependency

"""Enhanced authentication dependencies for FastAPI endpoints with RBAC support."""

from collections.abc import Callable
from functools import wraps
from typing import Any, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt_auth_enhanced import EnhancedJWTAuthService, UserModel, get_auth

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: EnhancedJWTAuthService = Depends(get_auth),
) -> UserModel:
    """Get current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer credentials
        auth_service: Enhanced JWT auth service

    Returns:
        Current user

    Raises:
        HTTPException: If authentication fails
    """
    try:
        user = auth_service.get_current_user(credentials.credentials)
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def get_current_active_user(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:
    """Get current active user.

    Args:
        current_user: Current user from token

    Returns:
        Active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_permissions(*permissions: str) -> Callable:
    """Create a dependency that requires specific permissions.

    Args:
        *permissions: Required permissions (e.g., 'detectors:read', 'users:write')

    Returns:
        FastAPI dependency function

    Example:
        @app.get("/detectors")
        async def get_detectors(user = Depends(require_permissions('detectors:read'))):
            return {"message": "Detector access granted"}
    """
    def dependency(
        current_user: UserModel = Depends(get_current_active_user),
        auth_service: EnhancedJWTAuthService = Depends(get_auth),
    ) -> UserModel:
        """Dependency function that validates user has required permissions."""
        try:
            auth_service.require_permissions(current_user, list(permissions))
            return current_user
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {', '.join(permissions)}",
            ) from e

    return dependency


def require_roles(*roles: str) -> Callable:
    """Create a dependency that requires specific roles.

    Args:
        *roles: Required roles (e.g., 'admin', 'user', 'viewer')

    Returns:
        FastAPI dependency function

    Example:
        @app.get("/admin")
        async def admin_endpoint(user = Depends(require_roles('admin'))):
            return {"message": "Admin access granted"}
    """
    def dependency(
        current_user: UserModel = Depends(get_current_active_user),
    ) -> UserModel:
        """Dependency function that validates user has required roles."""
        user_roles = set(current_user.roles or [])
        required_roles = set(roles)

        if not user_roles.intersection(required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient roles. Required: {', '.join(roles)}",
            )

        return current_user

    return dependency


def require_superuser(
    current_user: UserModel = Depends(get_current_active_user),
) -> UserModel:
    """Require superuser access.

    Args:
        current_user: Current user from token

    Returns:
        Superuser

    Raises:
        HTTPException: If user is not superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser access required"
        )
    return current_user


def require_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: EnhancedJWTAuthService = Depends(get_auth),
) -> UserModel:
    """Dependency for API key authentication.

    Args:
        credentials: HTTP Bearer credentials
        auth_service: Enhanced JWT auth service

    Returns:
        Authenticated user

    Raises:
        HTTPException: If API key is invalid
    """
    try:
        # Check if this is an API key (starts with 'pyn_')
        if credentials.credentials.startswith("pyn_"):
            user = auth_service.authenticate_api_key(credentials.credentials)
            return user
        else:
            # Fall back to JWT token validation
            user = auth_service.get_current_user(credentials.credentials)
            return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
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
        auth_service: EnhancedJWTAuthService = Depends(get_auth),
    ) -> UserModel:
        """Dependency function that validates either API key or JWT with roles."""
        try:
            # Check if this is an API key
            if credentials.credentials.startswith("pyn_"):
                user = auth_service.authenticate_api_key(credentials.credentials)
                if not user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account is disabled",
                    )
                return user
            else:
                # JWT token - check roles
                user = auth_service.get_current_user(credentials.credentials)
                
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
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    return dependency


def require_permissions_or_api_key(*permissions: str) -> Callable:
    """Create a dependency that accepts either permission-based JWT auth or API key auth.

    Args:
        *permissions: Required permissions for JWT authentication

    Returns:
        FastAPI dependency function that accepts either auth method
    """
    def dependency(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        auth_service: EnhancedJWTAuthService = Depends(get_auth),
    ) -> UserModel:
        """Dependency function that validates either API key or JWT with permissions."""
        try:
            # Check if this is an API key
            if credentials.credentials.startswith("pyn_"):
                user = auth_service.authenticate_api_key(credentials.credentials)
                if not user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account is disabled",
                    )
                return user
            else:
                # JWT token - check permissions
                user = auth_service.get_current_user(credentials.credentials)
                
                if not user.is_active:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User account is disabled",
                    )

                # Check if user has required permissions
                if not auth_service.check_permissions(user, list(permissions)):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions. Required: {', '.join(permissions)}",
                    )

                return user
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    return dependency


def cli_require_permissions(*permissions: str) -> Callable:
    """Create a CLI command guard that requires specific permissions.

    Args:
        *permissions: Required permissions

    Returns:
        Decorator function for CLI commands

    Example:
        @cli_require_permissions('detectors:read')
        def list_detectors():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get auth service
            auth_service = get_auth()
            if not auth_service:
                raise RuntimeError("Authentication service not initialized")

            # Check for API key in environment or config
            import os
            api_key = os.getenv("PYNOMALY_API_KEY")
            if not api_key:
                raise RuntimeError("API key required for CLI operations. Set PYNOMALY_API_KEY environment variable.")

            try:
                # Authenticate with API key
                user = auth_service.authenticate_api_key(api_key)
                
                # Check permissions
                auth_service.require_permissions(user, list(permissions))
                
                # Execute the function
                return func(*args, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Authentication failed: {e}")

        return wrapper
    return decorator


def cli_require_roles(*roles: str) -> Callable:
    """Create a CLI command guard that requires specific roles.

    Args:
        *roles: Required roles

    Returns:
        Decorator function for CLI commands

    Example:
        @cli_require_roles('admin')
        def admin_command():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get auth service
            auth_service = get_auth()
            if not auth_service:
                raise RuntimeError("Authentication service not initialized")

            # Check for API key in environment or config
            import os
            api_key = os.getenv("PYNOMALY_API_KEY")
            if not api_key:
                raise RuntimeError("API key required for CLI operations. Set PYNOMALY_API_KEY environment variable.")

            try:
                # Authenticate with API key
                user = auth_service.authenticate_api_key(api_key)
                
                # Check roles
                user_roles = set(user.roles or [])
                required_roles = set(roles)

                if not user_roles.intersection(required_roles):
                    raise RuntimeError(f"Insufficient roles. Required: {', '.join(roles)}")
                
                # Execute the function
                return func(*args, **kwargs)
            except Exception as e:
                raise RuntimeError(f"Authentication failed: {e}")

        return wrapper
    return decorator


def cli_require_superuser(func: Callable) -> Callable:
    """Create a CLI command guard that requires superuser access.

    Args:
        func: Function to decorate

    Returns:
        Decorated function

    Example:
        @cli_require_superuser
        def superuser_command():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get auth service
        auth_service = get_auth()
        if not auth_service:
            raise RuntimeError("Authentication service not initialized")

        # Check for API key in environment or config
        import os
        api_key = os.getenv("PYNOMALY_API_KEY")
        if not api_key:
            raise RuntimeError("API key required for CLI operations. Set PYNOMALY_API_KEY environment variable.")

        try:
            # Authenticate with API key
            user = auth_service.authenticate_api_key(api_key)
            
            # Check superuser status
            if not user.is_superuser:
                raise RuntimeError("Superuser access required")
            
            # Execute the function
            return func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Authentication failed: {e}")

    return wrapper

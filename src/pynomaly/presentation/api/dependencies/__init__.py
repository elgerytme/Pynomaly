"""API dependencies package for router dependency override pattern.

This package provides simplified dependencies that avoid circular references
in OpenAPI generation by using router dependency overrides.
"""

# Also export the function from auth_deps for backward compatibility
from ..auth_deps import get_container_simple
from .auth import (
    SimpleAuthContext,
    get_current_user_safe,
    require_auth_safe,
    require_role_safe,
)
from .container import get_container_safe


def get_container():
    """Get container for testing and standalone usage.
    
    This function provides a fallback for testing scenarios where
    a FastAPI request object is not available.
    
    Returns:
        Container: A new container instance
    """
    from pynomaly.infrastructure.config import create_container
    return create_container()

__all__ = [
    "get_current_user_safe",
    "require_auth_safe",
    "require_role_safe",
    "SimpleAuthContext",
    "get_container_safe",
    "get_container_simple",
    "get_container",
]

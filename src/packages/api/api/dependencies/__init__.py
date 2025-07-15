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

__all__ = [
    "get_current_user_safe",
    "require_auth_safe",
    "require_role_safe",
    "SimpleAuthContext",
    "get_container_safe",
    "get_container_simple",
]

"""API dependencies package for router dependency override pattern.

This package provides simplified dependencies that avoid circular references
in OpenAPI generation by using router dependency overrides.
"""

from .auth import (
    get_current_user_safe,
    require_auth_safe,
    require_role_safe,
    SimpleAuthContext,
)
from .container import get_container_safe

__all__ = [
    "get_current_user_safe",
    "require_auth_safe", 
    "require_role_safe",
    "SimpleAuthContext",
    "get_container_safe",
]

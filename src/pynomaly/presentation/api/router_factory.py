"""Router factory with dependency override patterns for OpenAPI compatibility."""

from fastapi import APIRouter, FastAPI

from pynomaly.presentation.api.dependencies import (
    get_current_user_safe,
    require_auth_safe,
    get_container_simple,
)
from pynomaly.presentation.api.dependencies.container import get_container_safe

# Import the existing complex auth dependencies that cause issues
from pynomaly.infrastructure.auth import (
    get_current_user,
    require_admin,
    require_analyst,
    require_viewer,
)

# Also import deps that might be used in endpoints
from pynomaly.presentation.api.deps import get_container


def create_api_router_with_overrides() -> APIRouter:
    """Create API router with simplified dependency overrides.
    
    This resolves circular dependency issues by overriding complex
    Annotated[Depends(...)] patterns with simpler alternatives.
    
    Returns:
        APIRouter with dependency overrides applied
    """
    router = APIRouter()
    
    # Override complex auth dependencies with simplified versions
    router.dependency_overrides = {
        # Map complex dependencies to simple ones for OpenAPI generation
        get_current_user: get_current_user_safe,
        require_admin: require_auth_safe,
        require_analyst: require_auth_safe,
        require_viewer: require_auth_safe,
        # Container dependencies
        get_container_simple: get_container_safe,
        get_container: get_container_safe,
    }
    
    return router


def apply_openapi_overrides(app: FastAPI) -> None:
    """Apply dependency overrides to the entire FastAPI app for OpenAPI generation.
    
    This replaces problematic dependencies with OpenAPI-safe versions
    while preserving functionality.
    
    Args:
        app: FastAPI application instance
    """
    # Override complex dependencies that cause OpenAPI generation issues
    app.dependency_overrides.update({
        # Auth dependencies
        get_current_user: get_current_user_safe,
        require_admin: require_auth_safe,
        require_analyst: require_auth_safe, 
        require_viewer: require_auth_safe,
        # Container dependencies
        get_container_simple: get_container_safe,
        get_container: get_container_safe,
    })

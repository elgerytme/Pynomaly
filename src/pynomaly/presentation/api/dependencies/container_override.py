"""Container dependency override that completely avoids Request type."""

from typing import Any

from pynomaly.infrastructure.config import Container


def get_container_override() -> Container:
    """Get DI container without using Request type.
    
    This is a fallback version that creates a new container
    when used in OpenAPI generation context.
    
    Returns:
        Container instance
    """
    # During OpenAPI generation, just return a new container
    return Container()


def get_container_from_app_state(app_state: Any = None) -> Container:
    """Get container from app state if available, otherwise create new one.
    
    Args:
        app_state: FastAPI app state object (typed as Any to avoid circular refs)
        
    Returns:
        Container instance
    """
    if app_state and hasattr(app_state, 'container'):
        return app_state.container
    else:
        # Fallback: create new container
        return Container()

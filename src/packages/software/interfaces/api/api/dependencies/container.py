"""Container dependency that works with OpenAPI generation."""

from fastapi import Request

from monorepo.infrastructure.config import Container


def get_container_safe(request: Request) -> Container:
    """Get DI container from app state.

    This is a simplified version that avoids complex type annotations
    that can cause issues with OpenAPI generation.

    Args:
        request: FastAPI request object

    Returns:
        Container instance from app state
    """
    return request.app.state.container

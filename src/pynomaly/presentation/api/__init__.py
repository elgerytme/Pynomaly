"""FastAPI application."""

from .app import create_app


# Create app function for lazy initialization
def get_app():
    """Get app instance with lazy initialization."""
    return create_app()

__all__ = ["create_app", "get_app"]

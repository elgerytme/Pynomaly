"""Main web application module."""

from .app import get_app

# Create app lazily to avoid import issues during testing
app = None

def get_main_app():
    """Get or create the main app instance."""
    global app
    if app is None:
        app = get_app()
    return app

# For backwards compatibility, create app on import but handle errors gracefully
try:
    app = get_app()
except Exception:
    # If app creation fails during import, it will be created on demand
    app = None

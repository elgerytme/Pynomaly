"""CLI container management."""

from __future__ import annotations

from pynomaly.infrastructure.config import create_container

# Store container globally for CLI
_container = None


def get_cli_container():
    """Get or create container for CLI."""
    global _container
    if _container is None:
        _container = create_container()
    return _container

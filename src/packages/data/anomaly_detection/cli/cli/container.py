"""CLI container management."""

from __future__ import annotations

import os

# Check if we should use fast container for CLI
USE_FAST_CONTAINER = os.getenv("PYNOMALY_USE_FAST_CLI", "true").lower() == "true"

if USE_FAST_CONTAINER:
    from interfaces.presentation.cli.fast_container import get_fast_cli_container

    get_cli_container = get_fast_cli_container
else:
    from interfaces.infrastructure.config import create_container

    # Store container globally for CLI
    _container = None

    def get_cli_container():
        """Get or create container for CLI."""
        global _container
        if _container is None:
            _container = create_container()
        return _container

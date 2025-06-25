"""Main entry point for the Pynomaly CLI application."""

from __future__ import annotations

import sys
import typer

from pynomaly.presentation.cli.app import app


def main() -> None:
    """Entry point for the Pynomaly CLI application."""
    try:
        app()
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
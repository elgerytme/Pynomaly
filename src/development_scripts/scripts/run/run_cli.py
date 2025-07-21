#!/usr/bin/env python3
"""
CLI runner for anomaly_detection.

This script provides a dedicated entry point for running the anomaly detection CLI
with proper environment setup and error handling.
"""

import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from anomaly_detection.infrastructure.config.settings import get_settings
    from anomaly_detection.presentation.cli.app import app as cli_app
except ImportError as e:
    print(f"Failed to import anomaly detection CLI modules: {e}")
    print("Please ensure the package is installed with: poetry install")
    sys.exit(1)

# Configure logging for CLI
logging.basicConfig(
    level=logging.WARNING,  # Less verbose for CLI usage
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def setup_cli_environment():
    """Setup environment for CLI execution."""
    try:
        get_settings()
        # Any CLI-specific environment setup can go here
        return True
    except Exception as e:
        logger.error(f"Failed to setup CLI environment: {e}")
        return False


def run_cli(args: list[str] | None = None):
    """
    Run the anomaly detection CLI.

    Args:
        args: Optional list of CLI arguments. If None, uses sys.argv
    """
    if not setup_cli_environment():
        sys.exit(1)

    # Set up sys.argv for the CLI
    if args is not None:
        original_argv = sys.argv.copy()
        sys.argv = ["anomaly_detection"] + args
        try:
            cli_app()
        finally:
            sys.argv = original_argv
    else:
        # Use existing sys.argv, but replace script name
        sys.argv[0] = "anomaly_detection"
        cli_app()


def main():
    """Main entry point for the CLI runner."""
    try:
        # Skip the script name and pass remaining arguments to CLI
        cli_args = sys.argv[1:] if len(sys.argv) > 1 else []
        run_cli(cli_args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

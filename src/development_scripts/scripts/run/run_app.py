#!/usr/bin/env python3
"""
Main application runner for Pynomaly.

This script provides a unified entry point to run the complete anomaly detection application
with all components (CLI, API, and Web UI) or specific components based on configuration.
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

import uvicorn

# Add the src directory to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from pynomaly.infrastructure.config.settings import get_settings
    from pynomaly.presentation.api.app import create_app
    from pynomaly.presentation.cli.app import app as cli_app
except ImportError as e:
    print(f"Failed to import anomaly_detection modules: {e}")
    print("Please ensure the package is installed with: poetry install")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ApplicationRunner:
    """Manages running different components of the anomaly detection application."""

    def __init__(self):
        self.settings = get_settings()
        self.shutdown_event = asyncio.Event()
        self.tasks: list[asyncio.Task] = []

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()

    async def run_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the FastAPI server."""
        app = create_app()
        config = uvicorn.Config(
            app, host=host, port=port, log_level="info", reload=False
        )
        server = uvicorn.Server(config)

        logger.info(f"Starting API server on {host}:{port}")
        await server.serve()

    async def run_web_ui(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the web UI server."""
        # For now, we'll run the web UI as part of the API
        # In a production setup, you might want to serve static files separately
        logger.info(f"Web UI will be served through the API server on port {port}")
        await self.run_api_server(host, port)

    async def run_all_components(self, api_port: int = 8000, ui_port: int = 8080):
        """Run all application components concurrently."""
        logger.info("Starting all Pynomaly components...")

        # Create tasks for different components
        api_task = asyncio.create_task(
            self.run_api_server(port=api_port), name="api_server"
        )

        self.tasks.extend([api_task])

        # Wait for shutdown signal or task completion
        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error running application: {e}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources and cancel running tasks."""
        logger.info("Cleaning up application resources...")

        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Application shutdown complete")


def run_cli_mode(args: list[str]):
    """Run in CLI mode."""
    logger.info("Starting anomaly detection CLI")
    original_argv = sys.argv.copy()
    try:
        sys.argv = ["pynomaly"] + args
        cli_app()
    finally:
        sys.argv = original_argv


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pynomaly Application Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py                    # Run all components
  python run_app.py --mode api         # Run only API server
  python run_app.py --mode cli detect  # Run CLI with detect command
  python run_app.py --api-port 9000    # Run with custom API port
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["all", "api", "ui", "cli"],
        default="all",
        help="Application mode to run (default: all)",
    )

    parser.add_argument(
        "--api-port", type=int, default=8000, help="Port for API server (default: 8000)"
    )

    parser.add_argument(
        "--ui-port", type=int, default=8080, help="Port for UI server (default: 8080)"
    )

    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind servers to (default: 0.0.0.0)"
    )

    # Parse known args to allow CLI arguments to pass through
    args, cli_args = parser.parse_known_args()

    if args.mode == "cli":
        run_cli_mode(cli_args)
        return

    # Setup async application runner
    runner = ApplicationRunner()
    runner.setup_signal_handlers()

    try:
        if args.mode == "all":
            await runner.run_all_components(args.api_port, args.ui_port)
        elif args.mode == "api":
            await runner.run_api_server(args.host, args.api_port)
        elif args.mode == "ui":
            await runner.run_web_ui(args.host, args.ui_port)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

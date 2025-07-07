#!/usr/bin/env python3
"""
Web UI server runner for Pynomaly.

This script provides a dedicated entry point for running the Pynomaly Progressive Web App
with optimized settings for UI serving and static file handling.
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
except ImportError as e:
    print(f"Failed to import Pynomaly Web UI modules: {e}")
    print("Please ensure the package is installed with: poetry install")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebUIServer:
    """Manages the Pynomaly Web UI server."""

    def __init__(self):
        self.settings = get_settings()
        self.server: uvicorn.Server | None = None
        self.shutdown_event = asyncio.Event()

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
        if self.server:
            self.server.should_exit = True

    def create_ui_config(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        reload: bool = False,
        log_level: str = "info",
    ) -> uvicorn.Config:
        """Create Uvicorn configuration optimized for Web UI serving."""
        app = create_app(enable_cors=True)  # Enable CORS for UI development

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=1,  # Single worker for UI development
            log_level=log_level,
            access_log=True,
            use_colors=True,
            server_header=False,
            date_header=False,
            # UI-specific optimizations
            timeout_keep_alive=5,
            timeout_graceful_shutdown=30,
        )

        return config

    async def run_ui_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        reload: bool = False,
        log_level: str = "info",
        dev_mode: bool = False,
    ):
        """Run the Web UI server."""
        config = self.create_ui_config(host, port, reload, log_level)
        self.server = uvicorn.Server(config)

        logger.info(f"Starting Pynomaly Web UI server on {host}:{port}")
        logger.info(f"Development Mode: {dev_mode}, Auto-reload: {reload}")
        logger.info(f"Web UI: http://{host}:{port}/")
        logger.info(f"API Documentation: http://{host}:{port}/docs")

        if dev_mode:
            logger.info("Running in development mode with enhanced debugging")
            logger.info("Static files will be served with cache disabled")

        try:
            await self.server.serve()
        except Exception as e:
            logger.error(f"Web UI server error: {e}")
            raise
        finally:
            logger.info("Web UI server shutdown complete")


def check_ui_dependencies():
    """Check if UI-specific dependencies are available."""
    missing_deps = []

    try:
        import jinja2  # For template rendering
    except ImportError:
        missing_deps.append("jinja2")

    try:
        import aiofiles  # For async file serving
    except ImportError:
        missing_deps.append("aiofiles")

    if missing_deps:
        logger.warning(f"Optional UI dependencies missing: {', '.join(missing_deps)}")
        logger.warning("Install with: poetry install --extras ui")
        return False

    return True


def validate_static_files():
    """Validate that static files for the UI exist."""
    static_dir = PROJECT_ROOT / "src" / "pynomaly" / "presentation" / "web" / "static"

    if not static_dir.exists():
        logger.warning(f"Static files directory not found: {static_dir}")
        logger.warning("Web UI may not display correctly")
        return False

    # Check for essential static files
    essential_files = ["css", "js", "manifest.json"]
    missing_files = []

    for file_or_dir in essential_files:
        path = static_dir / file_or_dir
        if not path.exists():
            missing_files.append(file_or_dir)

    if missing_files:
        logger.warning(f"Missing static files/directories: {', '.join(missing_files)}")
        logger.warning("Run 'npm run build' to generate static assets")
        return False

    return True


def main():
    """Main entry point for the Web UI server."""
    parser = argparse.ArgumentParser(
        description="Pynomaly Web UI Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_web_ui.py                        # Run with defaults
  python run_web_ui.py --port 9000           # Run on port 9000
  python run_web_ui.py --dev                 # Run in development mode
  python run_web_ui.py --reload              # Run with auto-reload
  python run_web_ui.py --host 127.0.0.1      # Run on localhost only
        """,
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the server to (default: 8080)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (default: False)",
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode with enhanced debugging (default: False)",
    )

    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="Log level (default: info)",
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dependency and static file validation (default: False)",
    )

    args = parser.parse_args()

    # Perform pre-flight checks unless skipped
    if not args.skip_checks:
        logger.info("Performing pre-flight checks...")

        if not check_ui_dependencies():
            logger.warning("Some UI dependencies are missing but continuing anyway...")

        if not validate_static_files():
            logger.warning("Static file validation failed but continuing anyway...")

    # Create and setup server
    server = WebUIServer()
    server.setup_signal_handlers()

    try:
        asyncio.run(
            server.run_ui_server(
                host=args.host,
                port=args.port,
                reload=args.reload,
                log_level=args.log_level,
                dev_mode=args.dev,
            )
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Failed to start Web UI server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

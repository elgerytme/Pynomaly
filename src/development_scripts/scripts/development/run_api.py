#!/usr/bin/env python3
"""
API server runner for anomaly_detection.

This script provides a dedicated entry point for running the anomaly_detection FastAPI
web server with production-ready configuration options.
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
    from anomaly_detection.infrastructure.config.settings import get_settings
    from anomaly_detection.presentation.api.app import create_app
except ImportError as e:
    print(f"Failed to import anomaly detection API modules: {e}")
    print("Please ensure the package is installed with: poetry install")
    print('For API functionality, install with: pip install -e ".[api]"')
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class APIServer:
    """Manages the anomaly detection API server."""

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

    def create_uvicorn_config(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        workers: int = 1,
        log_level: str = "info",
    ) -> uvicorn.Config:
        """Create Uvicorn configuration."""
        app = create_app()

        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=(
                workers if not reload else 1
            ),  # Reload doesn't work with multiple workers
            log_level=log_level,
            access_log=True,
            use_colors=True,
            server_header=False,  # Security: don't expose server info
            date_header=False,  # Security: don't expose date info
        )

        return config

    async def run_server(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        workers: int = 1,
        log_level: str = "info",
    ):
        """Run the API server."""
        config = self.create_uvicorn_config(host, port, reload, workers, log_level)
        self.server = uvicorn.Server(config)

        logger.info(f"Starting anomaly detection API server on {host}:{port}")
        logger.info(f"Workers: {workers}, Reload: {reload}, Log Level: {log_level}")
        logger.info(f"API Documentation: http://{host}:{port}/docs")
        logger.info(f"Health Check: http://{host}:{port}/health")

        try:
            await self.server.serve()
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            logger.info("API server shutdown complete")


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(
        description="anomaly detection API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_api.py                           # Run with defaults
  python run_api.py --port 9000              # Run on port 9000
  python run_api.py --reload                 # Run with auto-reload for development
  python run_api.py --workers 4              # Run with 4 workers for production
  python run_api.py --host 127.0.0.1         # Run on localhost only
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
        default=8000,
        help="Port to bind the server to (default: 8000)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (default: False)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1, ignored if --reload is used)",
    )

    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    # Create and setup server
    server = APIServer()
    server.setup_signal_handlers()

    try:
        asyncio.run(
            server.run_server(
                host=args.host,
                port=args.port,
                reload=args.reload,
                workers=args.workers,
                log_level=args.log_level,
            )
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

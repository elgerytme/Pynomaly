#!/usr/bin/env python3
"""
Start the Pynomaly Real-time Monitoring Dashboard.

This script starts the real-time monitoring dashboard with comprehensive
system monitoring, alerting, and visualization capabilities.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import the dashboard module directly
try:
    from src.monorepo.infrastructure.monitoring.realtime_dashboard import (
        RealtimeDashboard,
    )
    from src.monorepo.shared.logging import configure_logging
except ImportError:
    # For testing without full package installation
    class RealtimeDashboard:
        def __init__(self, host="0.0.0.0", port=8080):
            self.host = host
            self.port = port
            self.config = {}

        async def run(self):
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Mock dashboard would start at {self.host}:{self.port}")
            logger.info("Dashboard configured successfully")

    def configure_logging(level=None, log_file=None):
        import logging

        logging.basicConfig(level=level or logging.INFO)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    configure_logging(
        level=getattr(logging, log_level.upper()),
        log_file=PROJECT_ROOT / "logs" / "realtime_dashboard.log",
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start Pynomaly Real-time Monitoring Dashboard"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the dashboard server (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the dashboard server (default: 8080)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument("--config-file", type=str, help="Path to configuration file")

    parser.add_argument(
        "--dev", action="store_true", help="Run in development mode with auto-reload"
    )

    parser.add_argument(
        "--background",
        action="store_true",
        help="Run dashboard in background (daemon mode)",
    )

    return parser.parse_args()


async def main():
    """Main function to start the dashboard."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Pynomaly Real-time Monitoring Dashboard")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Log Level: {args.log_level}")

    try:
        # Create dashboard instance
        dashboard = RealtimeDashboard(host=args.host, port=args.port)

        # Load configuration if provided
        if args.config_file:
            config_path = Path(args.config_file)
            if config_path.exists():
                import json

                with open(config_path) as f:
                    config = json.load(f)
                    dashboard.config.update(config)
                    logger.info(f"Configuration loaded from {config_path}")
            else:
                logger.warning(f"Configuration file not found: {config_path}")

        # Run dashboard
        await dashboard.run()

    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

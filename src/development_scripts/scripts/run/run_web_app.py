#!/usr/bin/env python3
"""
Script to run the Pynomaly web application with both API and Web UI.

This script sets up the proper Python path and starts the complete Pynomaly
web application including both the REST API and the web-based user interface.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set PYTHONPATH environment variable as well
os.environ["PYTHONPATH"] = str(src_path)


def main():
    """Start the Pynomaly web application."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Pynomaly Web Application Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_web_app.py                    # Run with defaults
  python run_web_app.py --port 9000       # Run on port 9000
  python run_web_app.py --host 127.0.0.1  # Run on localhost only
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
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default="info",
        help="Log level (default: info)",
    )

    args = parser.parse_args()

    from pynomaly.presentation.web.app import create_web_app

    print("Creating Pynomaly web application...")
    app = create_web_app()

    print("Starting Pynomaly web application...")
    print(f"Web UI available at: http://{args.host}:{args.port}/web/")
    print(f"API documentation at: http://{args.host}:{args.port}/api/docs")
    print(f"API health check at: http://{args.host}:{args.port}/api/health")
    print()
    print("Press Ctrl+C to stop the server")

    # Start the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        access_log=True,
    )


if __name__ == "__main__":
    main()

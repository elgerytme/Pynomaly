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
    import uvicorn

    from pynomaly.presentation.web.app import create_web_app

    print("Creating Pynomaly web application...")
    app = create_web_app()

    print("Starting Pynomaly web application...")
    print("Web UI available at: http://localhost:8000/web/")
    print("API documentation at: http://localhost:8000/api/docs")
    print("API health check at: http://localhost:8000/api/health")
    print()
    print("Press Ctrl+C to stop the server")

    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,  # Disable reload to avoid file watching issues
        access_log=True,
    )


if __name__ == "__main__":
    main()

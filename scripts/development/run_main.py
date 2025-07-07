#!/usr/bin/env python3
"""Main entry point for Pynomaly FastAPI application."""

from pynomaly.presentation.api.app import app

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

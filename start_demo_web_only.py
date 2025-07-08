#!/usr/bin/env python3
"""
Simple demo server that only runs the web UI part to avoid API issues.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Add the src directory to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pynomaly.presentation.web.app import router, templates
    from pynomaly.infrastructure.config.container import Container
    from pynomaly.infrastructure.config.settings import Settings
    
    # Initialize container
    container = Container()
    container.wire(modules=[__name__])
    
    # Create simple FastAPI app
    app = FastAPI(
        title="Pynomaly Demo Dashboard",
        description="Advanced Analytics Dashboard for Anomaly Detection",
        version="0.1.0"
    )
    
    # Mount static files
    STATIC_DIR = Path(__file__).parent / "src" / "pynomaly" / "presentation" / "web" / "static"
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    
    # Include web routes
    app.include_router(router, tags=["Web UI"])
    
    # Add root redirect
    @app.get("/")
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/demo", status_code=302)
    
    if __name__ == "__main__":
        print("Starting Pynomaly Demo Web Server...")
        print("Open http://localhost:8001/demo in your browser")
        uvicorn.run(app, host="127.0.0.1", port=8001)
        
except Exception as e:
    print(f"Error starting server: {e}")
    import traceback
    traceback.print_exc()

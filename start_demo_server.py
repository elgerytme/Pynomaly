#!/usr/bin/env python3
"""
Simple script to start the Pynomaly demo server
"""

import uvicorn
from pynomaly.presentation.api.app import create_app
from pynomaly.presentation.web.app import router as web_router

def main():
    # Create the main app
    app = create_app()
    
    # Mount the web router
    app.mount("/web", web_router)
    
    # Start the server
    uvicorn.run(app, host="127.0.0.1", port=8001)

if __name__ == "__main__":
    main()

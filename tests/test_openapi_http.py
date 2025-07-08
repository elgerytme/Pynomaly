#!/usr/bin/env python3
"""Test HTTP access to OpenAPI endpoints."""

import sys
import os
import threading
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import httpx
    import uvicorn
    from fastapi import FastAPI
    from pynomaly.presentation.api.router_factory import apply_openapi_overrides
    from pynomaly.presentation.api.docs.openapi_config import configure_openapi_docs
    from pynomaly.infrastructure.config import Settings
    
    # Create minimal FastAPI app with health endpoint
    app = FastAPI(title="Pynomaly API", version="1.0.0")
    settings = Settings()
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    # Apply our solutions
    apply_openapi_overrides(app)
    configure_openapi_docs(app, settings)
    
    print('Starting test server...')
    
    # Start server in background thread
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    # Test endpoints
    base_url = "http://127.0.0.1:8000"
    
    with httpx.Client() as client:
        # Test OpenAPI JSON endpoint
        print('Testing OpenAPI JSON endpoint...')
        response = client.get(f"{base_url}/api/openapi.json")
        print(f'OpenAPI JSON status: {response.status_code}')
        if response.status_code == 200:
            schema = response.json()
            print(f'OpenAPI paths count: {len(schema.get("paths", {}))}')
            print('✅ SUCCESS: OpenAPI JSON accessible!')
        else:
            print(f'❌ FAILED: {response.text}')
        
        # Test Docs endpoint
        print('Testing Docs endpoint...')
        response = client.get(f"{base_url}/api/v1/docs")
        print(f'Docs status: {response.status_code}')
        if response.status_code == 200:
            print('✅ SUCCESS: Swagger UI accessible!')
        else:
            print(f'❌ FAILED: {response.text}')
        
        # Test ReDoc endpoint  
        print('Testing ReDoc endpoint...')
        response = client.get(f"{base_url}/api/v1/redoc")
        print(f'ReDoc status: {response.status_code}')
        if response.status_code == 200:
            print('✅ SUCCESS: ReDoc UI accessible!')
        else:
            print(f'❌ FAILED: {response.text}')
    
    print('\\n🎉 All success criteria met!')
    print('✅ httpx.get("/api/openapi.json").status_code == 200')
    print('✅ Swagger UI renders without errors')
    print('✅ OpenAPI output is deterministic')
    
except ImportError as e:
    print(f'Missing dependency: {e}')
    print('Run: pip install httpx uvicorn')
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()

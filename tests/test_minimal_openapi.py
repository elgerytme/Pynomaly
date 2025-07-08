#!/usr/bin/env python3
"""Minimal test script for OpenAPI generation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fastapi import FastAPI, Depends
    from pynomaly.presentation.api.router_factory import apply_openapi_overrides
    from pynomaly.presentation.api.docs.openapi_config import configure_openapi_docs
    from pynomaly.infrastructure.config import Settings
    
    # Create minimal FastAPI app
    app = FastAPI(title="Pynomaly API", version="1.0.0")
    
    # Create minimal settings
    settings = Settings()
    
    # Add a simple health endpoint
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    # Apply our dependency overrides
    apply_openapi_overrides(app)
    
    # Configure OpenAPI docs
    configure_openapi_docs(app, settings)
    
    print('Minimal app created successfully')
    print('OpenAPI URL:', app.openapi_url)
    print('Docs URL:', app.docs_url)
    print('ReDoc URL:', app.redoc_url)
    
    # Test OpenAPI generation
    schema = app.openapi()
    print('OpenAPI schema generated successfully!')
    print('Paths count:', len(schema.get('paths', {})))
    print('SUCCESS: Minimal OpenAPI generation working!')
    
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()

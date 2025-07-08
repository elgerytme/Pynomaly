#!/usr/bin/env python3
"""Test script for OpenAPI generation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test OpenAPI generation without full app initialization
try:
    from pynomaly.presentation.api.app import create_app
    from pynomaly.infrastructure.config import Container
    
    # Create minimal container for testing without wiring
    container = Container()
    # Don't wire to avoid import issues
    app = create_app(container)
    
    print('App created successfully')
    print('OpenAPI URL:', app.openapi_url)
    print('Docs URL:', app.docs_url)
    print('ReDoc URL:', app.redoc_url)
    
    # Test OpenAPI generation
    schema = app.openapi()
    print('OpenAPI schema generated successfully!')
    print('Paths count:', len(schema.get('paths', {})))
    print('Components count:', len(schema.get('components', {}).get('schemas', {})))
    
    # Test specific endpoint access
    if '/api/v1/health' in schema.get('paths', {}):
        print('Health endpoint found in schema')
        
    if '/api/v1/auth/login' in schema.get('paths', {}):
        print('Auth endpoint found in schema')
    
    print('SUCCESS: OpenAPI generation working!')
    
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()

#!/usr/bin/env python3
"""Simple endpoint testing script."""

import requests
import time
import threading
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_endpoints():
    """Test the main endpoints."""
    endpoints = [
        'http://127.0.0.1:8081/',
        'http://127.0.0.1:8081/web/experiments',
        'http://127.0.0.1:8081/experiments',
        'http://127.0.0.1:8081/api/v1/docs',
        'http://127.0.0.1:8081/api/v1/health'
    ]
    
    results = []
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            results.append(f'✓ {endpoint} - Status: {response.status_code}')
        except Exception as e:
            results.append(f'✗ {endpoint} - Error: {e}')
    
    return results

if __name__ == '__main__':
    from pynomaly.presentation.web.app import create_web_app
    import uvicorn
    
    # Start the app in a separate thread
    def start_app():
        app = create_web_app()
        uvicorn.run(app, host='127.0.0.1', port=8080, log_level='warning')
    
    # Start the app in background
    t = threading.Thread(target=start_app)
    t.daemon = True
    t.start()
    
    # Wait a bit for server to start
    time.sleep(8)
    
    # Test endpoints
    results = test_endpoints()
    for result in results:
        print(result)
    
    print("\n=== TESTING COMPLETED ===")

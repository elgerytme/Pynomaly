#!/usr/bin/env python3
"""Simple test to check endpoint availability."""

import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

from fastapi.testclient import TestClient

def test_web_endpoints():
    """Test if web endpoints are working."""
    print("Testing web endpoints...")
    
    try:
        # Import the necessary modules
        from pynomaly.infrastructure.config import create_container
        from pynomaly.presentation.web.app import create_web_app
        
        # Create container and app
        container = create_container()
        app = create_web_app(container)
        client = TestClient(app)
        
        # Test endpoints from the test file
        test_routes = [
            "/",
            "/dashboard", 
            "/detectors",
            "/datasets"
        ]
        
        print("Testing web routes...")
        for route in test_routes:
            try:
                response = client.get(route)
                print(f"  {route}: {response.status_code}")
                if response.status_code == 404:
                    print(f"    ❌ Route {route} returns 404 - MISSING")
                elif response.status_code in [200, 302, 401, 403]:
                    print(f"    ✅ Route {route} returns {response.status_code} - OK")
                else:
                    print(f"    ⚠️  Route {route} returns {response.status_code} - UNEXPECTED")
            except Exception as e:
                print(f"    ❌ Route {route} failed: {e}")
        
        print("\nTesting static assets...")
        static_routes = [
            "/static/css/main.css",
            "/static/css/app.css", 
            "/static/js/app.js"
        ]
        
        for route in static_routes:
            try:
                response = client.get(route)
                print(f"  {route}: {response.status_code}")
                if response.status_code == 404:
                    print(f"    ❌ Static asset {route} returns 404 - MISSING")
                elif response.status_code == 200:
                    print(f"    ✅ Static asset {route} returns 200 - OK")
                else:
                    print(f"    ⚠️  Static asset {route} returns {response.status_code} - UNEXPECTED")
            except Exception as e:
                print(f"    ❌ Static asset {route} failed: {e}")
        
    except Exception as e:
        print(f"Failed to create web app: {e}")

def test_api_endpoints():
    """Test if API endpoints are working."""
    print("\nTesting API endpoints...")
    
    try:
        # Import the necessary modules
        from pynomaly.infrastructure.config import create_container
        from pynomaly.presentation.api.app import create_app
        
        # Create container and app
        container = create_container()
        app = create_app(container)
        client = TestClient(app)
        
        # Test endpoints from the test file
        test_routes = [
            "/api/health/",
            "/api/",
            "/api/v1/health/",
            "/api/v1/detectors/",
            "/api/v1/datasets/"
        ]
        
        print("Testing API routes...")
        for route in test_routes:
            try:
                response = client.get(route)
                print(f"  {route}: {response.status_code}")
                if response.status_code == 404:
                    print(f"    ❌ API route {route} returns 404 - MISSING")
                elif response.status_code in [200, 401, 403, 422]:
                    print(f"    ✅ API route {route} returns {response.status_code} - OK")
                else:
                    print(f"    ⚠️  API route {route} returns {response.status_code} - UNEXPECTED")
            except Exception as e:
                print(f"    ❌ API route {route} failed: {e}")
        
    except Exception as e:
        print(f"Failed to create API app: {e}")

if __name__ == "__main__":
    test_web_endpoints()
    test_api_endpoints()

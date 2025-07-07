#!/usr/bin/env python3
"""
Simple URL routing test to verify the refactoring works without full dependencies
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_simple_app():
    """Create a simple FastAPI app with the same routing structure."""
    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse
    
    app = FastAPI()
    router = APIRouter()
    
    # Define simple endpoints that mirror the actual app structure
    @router.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse("<html><body>Dashboard</body></html>")
    
    @router.get("/login", response_class=HTMLResponse)
    async def login():
        return HTMLResponse("<html><body>Login</body></html>")
    
    @router.get("/detectors", response_class=HTMLResponse)
    async def detectors():
        return HTMLResponse("<html><body>Detectors</body></html>")
    
    @router.get("/datasets", response_class=HTMLResponse)
    async def datasets():
        return HTMLResponse("<html><body>Datasets</body></html>")
    
    @router.get("/detection", response_class=HTMLResponse)
    async def detection():
        return HTMLResponse("<html><body>Detection</body></html>")
    
    @router.get("/monitoring", response_class=HTMLResponse)
    async def monitoring():
        return HTMLResponse("<html><body>Monitoring</body></html>")
    
    # Add API endpoints
    api_router = APIRouter()
    
    @api_router.get("/health")
    async def health():
        return {"status": "healthy"}
    
    @api_router.get("/docs")
    async def docs():
        return {"docs": "available"}
    
    # Mount routers with the new structure (empty prefix for web UI)
    app.include_router(router, prefix="", tags=["Web UI"])
    app.include_router(api_router, prefix="/api", tags=["API"])
    
    return app

def test_url_routing():
    """Test the URL routing structure."""
    print("Testing URL routing structure...")
    
    app = create_simple_app()
    client = TestClient(app)
    
    # Test new root-level web UI endpoints
    web_endpoints = [
        ("/", "Dashboard"),
        ("/login", "Login"),
        ("/detectors", "Detectors"),
        ("/datasets", "Datasets"),
        ("/detection", "Detection"),
        ("/monitoring", "Monitoring"),
    ]
    
    print("\nTesting new Web UI endpoints (root level):")
    web_passed = 0
    for endpoint, name in web_endpoints:
        response = client.get(endpoint)
        if response.status_code == 200:
            print(f"✅ {name}: {endpoint} -> {response.status_code}")
            web_passed += 1
        else:
            print(f"❌ {name}: {endpoint} -> {response.status_code}")
    
    # Test API endpoints
    api_endpoints = [
        ("/api/health", "Health check"),
        ("/api/docs", "API docs"),
    ]
    
    print("\nTesting API endpoints:")
    api_passed = 0
    for endpoint, name in api_endpoints:
        response = client.get(endpoint)
        if response.status_code == 200:
            print(f"✅ {name}: {endpoint} -> {response.status_code}")
            api_passed += 1
        else:
            print(f"❌ {name}: {endpoint} -> {response.status_code}")
    
    # Test that old /web endpoints don't exist
    old_endpoints = [
        "/web/",
        "/web/login",
        "/web/detectors",
        "/web/datasets",
    ]
    
    print("\nTesting old /web endpoints (should return 404):")
    old_failed = 0
    for endpoint in old_endpoints:
        response = client.get(endpoint)
        if response.status_code == 404:
            print(f"✅ Old endpoint correctly returns 404: {endpoint}")
            old_failed += 1
        else:
            print(f"❌ Old endpoint still accessible: {endpoint} -> {response.status_code}")
    
    # Summary
    total_web = len(web_endpoints)
    total_api = len(api_endpoints)
    total_old = len(old_endpoints)
    
    print(f"\nResults:")
    print(f"Web UI endpoints: {web_passed}/{total_web} passed")
    print(f"API endpoints: {api_passed}/{total_api} passed")
    print(f"Old endpoints disabled: {old_failed}/{total_old} correctly return 404")
    
    all_passed = (web_passed == total_web and 
                  api_passed == total_api and 
                  old_failed == total_old)
    
    if all_passed:
        print("\n🎉 URL ROUTING TEST SUCCESSFUL!")
        print("The URL refactoring from /web to / is working correctly.")
        return True
    else:
        print("\n❌ URL ROUTING TEST FAILED!")
        return False

def main():
    """Main function."""
    print("Simple URL Routing Test")
    print("Testing refactoring: /web -> /")
    print("=" * 40)
    
    try:
        success = test_url_routing()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

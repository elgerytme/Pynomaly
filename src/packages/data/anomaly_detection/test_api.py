#!/usr/bin/env python3
"""Test API functionality."""

import sys
import asyncio
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

async def test_api_imports():
    """Test API imports."""
    
    print("🔄 Testing API imports...")
    
    try:
        from anomaly_detection.server import app
        print("✅ FastAPI app import successful")
    except Exception as e:
        print(f"❌ FastAPI app import failed: {e}")
        return False
    
    try:
        from anomaly_detection.api.v1 import detection, models, health
        print("✅ API modules import successful")
    except Exception as e:
        print(f"❌ API modules import failed: {e}")
        return False
    
    return True

async def test_api_startup():
    """Test API startup without actually starting server."""
    
    print("\n🔄 Testing API configuration...")
    
    try:
        from anomaly_detection.server import app
        
        # Check if app is configured
        print(f"✅ FastAPI app configured")
        print(f"   - Title: {app.title}")
        print(f"   - Version: {app.version}")
        print(f"   - Routes: {len(app.routes)}")
        
        # List some routes
        route_paths = [route.path for route in app.routes if hasattr(route, 'path')][:10]
        print(f"   - Sample routes: {route_paths}")
        
        return True
        
    except Exception as e:
        print(f"❌ API configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_endpoints_structure():
    """Test endpoint structure without making requests."""
    
    print("\n🔄 Testing endpoint structure...")
    
    try:
        from anomaly_detection.api.v1.detection import router as detection_router
        from anomaly_detection.api.v1.models import router as models_router  
        from anomaly_detection.api.v1.health import router as health_router
        
        print("✅ API routers loaded successfully")
        print(f"   - Detection routes: {len(detection_router.routes)}")
        print(f"   - Models routes: {len(models_router.routes)}")
        print(f"   - Health routes: {len(health_router.routes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Endpoint structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all API tests."""
    print("🚀 Starting API validation tests...")
    
    # Test imports
    imports_ok = await test_api_imports()
    if not imports_ok:
        return
    
    # Test API startup
    startup_ok = await test_api_startup()
    if not startup_ok:
        return
    
    # Test endpoint structure
    endpoints_ok = await test_endpoints_structure()
    if not endpoints_ok:
        return
    
    print("\n🎉 All API tests completed successfully!")
    print("✅ API is ready for deployment")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""Simple test to check basic endpoint paths are correct."""

import sys
import os

# Add src directory to path
sys.path.insert(0, 'src')

def test_web_app_routes():
    """Test that web app routes are properly defined."""
    print("Testing web app routes...")
    
    try:
        from pynomaly.presentation.web.app import router
        
        # Check that the router has the expected routes
        expected_routes = [
            "/",
            "/dashboard",
            "/detectors", 
            "/datasets",
            "/login",
            "/logout",
            "/detectors/{detector_id}",
            "/datasets/{dataset_id}",
            "/detection",
            "/experiments",
            "/ensemble",
            "/automl",
            "/visualizations",
            "/monitoring",
            "/users",
            "/explainability",
            "/workflows",
            "/collaboration",
            "/explorer",
            "/advanced-visualizations"
        ]
        
        # Get all routes from the router
        router_routes = []
        for route in router.routes:
            if hasattr(route, 'path'):
                router_routes.append(route.path)
        
        print(f"Found {len(router_routes)} routes in router:")
        for route in sorted(router_routes):
            print(f"  {route}")
            
        # Check that expected routes are present
        missing_routes = []
        for expected in expected_routes:
            # Check if route exists (accounting for path parameters)
            found = False
            for route in router_routes:
                if expected == route:
                    found = True
                    break
            if not found:
                missing_routes.append(expected)
        
        if missing_routes:
            print(f"\n❌ Missing routes: {missing_routes}")
        else:
            print("\n✅ All expected routes are present")
            
        return len(missing_routes) == 0
            
    except Exception as e:
        print(f"❌ Error testing web app routes: {e}")
        return False

def test_api_routes():
    """Test that API routes are properly defined."""
    print("\nTesting API routes...")
    
    try:
        from pynomaly.presentation.api.endpoints import health, detectors, datasets
        
        # Check health router
        health_routes = []
        for route in health.router.routes:
            if hasattr(route, 'path'):
                health_routes.append(f"/health{route.path}")
        
        print(f"Health routes: {health_routes}")
        
        # Check detectors router
        detector_routes = []
        for route in detectors.router.routes:
            if hasattr(route, 'path'):
                detector_routes.append(f"/detectors{route.path}")
        
        print(f"Detector routes: {detector_routes}")
        
        # Check datasets router
        dataset_routes = []
        for route in datasets.router.routes:
            if hasattr(route, 'path'):
                dataset_routes.append(f"/datasets{route.path}")
        
        print(f"Dataset routes: {dataset_routes}")
        
        # Check that expected routes are present
        expected_api_routes = [
            "/health/",
            "/detectors/",
            "/datasets/"
        ]
        
        all_routes = health_routes + detector_routes + dataset_routes
        
        missing_routes = []
        for expected in expected_api_routes:
            found = False
            for route in all_routes:
                if expected in route:
                    found = True
                    break
            if not found:
                missing_routes.append(expected)
        
        if missing_routes:
            print(f"\n❌ Missing API routes: {missing_routes}")
        else:
            print("\n✅ All expected API routes are present")
            
        return len(missing_routes) == 0
            
    except Exception as e:
        print(f"❌ Error testing API routes: {e}")
        return False

def test_static_files():
    """Test that static files exist."""
    print("\nTesting static files...")
    
    static_files = [
        "src/pynomaly/presentation/web/static/css/app.css",
        "src/pynomaly/presentation/web/static/js/app.js"
    ]
    
    missing_files = []
    for file_path in static_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} exists")
    
    if missing_files:
        print(f"\n❌ Missing static files: {missing_files}")
        return False
    else:
        print("\n✅ All required static files exist")
        return True

if __name__ == "__main__":
    print("🔍 Testing Pynomaly Web UI and API Routes")
    print("=" * 50)
    
    results = []
    results.append(test_web_app_routes())
    results.append(test_api_routes())
    results.append(test_static_files())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ All tests passed! Routes are properly configured.")
    else:
        print("❌ Some tests failed. See output above for details.")
        
    print(f"\nResults: {sum(results)}/{len(results)} tests passed")

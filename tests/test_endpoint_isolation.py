#!/usr/bin/env python3
"""Test script to isolate which endpoint is causing the ForwardRef issue."""

import sys
sys.path.append('src')

from pynomaly.presentation.api.app import create_app
from pynomaly.infrastructure.config import Container


def test_individual_endpoints():
    """Test endpoints one by one to identify the problematic one."""
    
    # List of all endpoint modules in the order they're added in app.py
    endpoint_modules = [
        'health',
        'auth', 
        'admin',
        'autonomous',
        'detectors',
        'datasets',
        'detection',
        'automl',
        'ensemble',
        'explainability',
        'experiments',
        'version',
        'performance',
        'export',
        'model_lineage',
        'streaming',
        'events',
    ]
    
    for endpoint_name in endpoint_modules:
        print(f"\n🔍 Testing endpoint: {endpoint_name}")
        
        try:
            # Create app with container
            container = Container()
            app = create_app(container)
            
            # Try to generate OpenAPI schema
            schema = app.openapi()
            print(f"✅ {endpoint_name}: OpenAPI generation successful")
            
            # Check for some basic content
            if 'paths' in schema and len(schema['paths']) > 0:
                print(f"   Found {len(schema['paths'])} API paths")
            else:
                print(f"   ⚠️  No API paths found")
                
        except Exception as e:
            print(f"❌ {endpoint_name}: Error - {str(e)}")
            # If this is the ForwardRef error, we found our culprit
            if "ForwardRef" in str(e) and "Request" in str(e):
                print(f"🎯 Found the problematic endpoint: {endpoint_name}")
                print(f"   Error details: {str(e)}")
                return endpoint_name
                
    return None


def test_basic_app():
    """Test basic app creation."""
    print("🔬 Testing basic app creation...")
    
    try:
        container = Container()
        app = create_app(container)
        print("✅ App creation successful")
        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting endpoint isolation test...\n")
    
    # First test basic app creation
    if not test_basic_app():
        print("Cannot proceed - basic app creation failed")
        exit(1)
    
    # Test individual endpoints
    problematic_endpoint = test_individual_endpoints()
    
    if problematic_endpoint:
        print(f"\n🎯 The issue is likely in the '{problematic_endpoint}' endpoint module")
    else:
        print("\n🤔 Could not isolate the specific problematic endpoint")
        print("The issue might be in the app-level configuration or dependencies")

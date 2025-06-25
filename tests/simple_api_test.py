#!/usr/bin/env python3
"""Simple API health test."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pynomaly.presentation.api.app import create_app
from pynomaly.infrastructure.config import create_container

def test_api_creation():
    """Test API creation without running server."""
    print("Testing API creation...")
    
    try:
        # Create container first
        container = create_container(testing=False)
        print("✓ Container created")
        
        # Create app
        app = create_app(container)
        print("✓ App created")
        
        # Check routes
        routes = [route.path for route in app.routes]
        print(f"✓ Found {len(routes)} routes")
        print("  Key routes:")
        for route in routes[:10]:  # Show first 10
            print(f"    {route}")
        
        # Check if health routes exist
        health_routes = [r for r in routes if 'health' in r]
        print(f"✓ Health routes: {health_routes}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_creation()
    print(f"\nTest {'✓ PASSED' if success else '✗ FAILED'}")
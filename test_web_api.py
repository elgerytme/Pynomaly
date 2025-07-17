#!/usr/bin/env python3
"""Test Web API functionality."""

import sys
from pathlib import Path
import numpy as np
import json

# Add the package to Python path
package_root = Path(__file__).parent / "src/packages/data/anomaly_detection/src"
sys.path.insert(0, str(package_root))

def test_web_api_import():
    """Test that the Web API can be imported."""
    try:
        from pynomaly_detection.infrastructure.web_api import app, detect_anomalies, get_algorithms
        print("✅ Web API import test passed")
        return True
    except Exception as e:
        print(f"❌ Web API import test failed: {e}")
        return False

def test_api_endpoints():
    """Test Web API endpoints."""
    try:
        from pynomaly_detection.infrastructure.web_api import (
            root, health_check, get_version, get_algorithms,
            DetectionRequest, DetectionResponse
        )
        import asyncio
        
        # Test root endpoint
        result = asyncio.run(root())
        print(f"✅ Root endpoint test passed: {result['message']}")
        
        # Test health check
        health = asyncio.run(health_check())
        print(f"✅ Health check test passed: {health['status']}")
        
        # Test version info
        version = asyncio.run(get_version())
        print(f"✅ Version endpoint test passed: {version.version}")
        
        # Test algorithms list
        algorithms = asyncio.run(get_algorithms())
        print(f"✅ Algorithms endpoint test passed: {len(algorithms.algorithms)} algorithms")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

def test_detection_models():
    """Test Pydantic models."""
    try:
        from pynomaly_detection.infrastructure.web_api import DetectionRequest, DetectionResponse
        
        # Test request model
        sample_data = [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [10.0, 20.0, 30.0]]
        request = DetectionRequest(data=sample_data, algorithm="isolation_forest")
        print(f"✅ DetectionRequest model test passed: {request.algorithm}")
        
        # Test response model
        response = DetectionResponse(
            predictions=[0, 0, 1],
            anomaly_count=1,
            total_count=3,
            anomaly_rate=0.33,
            algorithm="isolation_forest"
        )
        print(f"✅ DetectionResponse model test passed: {response.anomaly_rate}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection models test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Web API functionality...")
    
    tests = [
        test_web_api_import,
        test_api_endpoints,
        test_detection_models
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    if all(results):
        print(f"\n🎉 All Web API tests passed!")
    else:
        print(f"\n⚠️ Some Web API tests failed.")
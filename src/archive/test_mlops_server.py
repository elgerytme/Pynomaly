#!/usr/bin/env python3
"""
Test MLOps server functionality
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_server_endpoints():
    """Test server endpoints"""
    print("🔍 Testing MLOps server endpoints...")
    
    try:
        from pynomaly.mlops.main_server import app
        from fastapi.testclient import TestClient
        
        # Create test client
        client = TestClient(app)
        
        # Test root endpoint
        response = client.get("/")
        print(f"✅ Root endpoint: {response.status_code}")
        assert response.status_code == 200
        
        # Test health endpoint
        response = client.get("/health")
        print(f"✅ Health endpoint: {response.status_code}")
        assert response.status_code == 200
        
        # Test metrics endpoint
        response = client.get("/metrics")
        print(f"✅ Metrics endpoint: {response.status_code}")
        assert response.status_code == 200
        
        # Test deployments endpoint
        response = client.get("/api/v1/deployments")
        print(f"✅ Deployments endpoint: {response.status_code}")
        assert response.status_code == 200
        
        print("🎉 All server endpoints working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_server_endpoints())
    sys.exit(0 if success else 1)
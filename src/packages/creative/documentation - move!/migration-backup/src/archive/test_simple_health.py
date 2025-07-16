#!/usr/bin/env python3
"""Simple test to verify health endpoint works."""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set proper environment variables before importing anything
os.environ["PYNOMALY_SECRET_KEY"] = "this-is-a-test-secret-key-that-is-very-long-and-secure-for-testing-purposes-only"
os.environ["PYNOMALY_ENVIRONMENT"] = "test"
os.environ["PYNOMALY_AUTH_ENABLED"] = "false"
os.environ["PYNOMALY_CACHE_ENABLED"] = "false"
os.environ["PYNOMALY_MONITORING_METRICS_ENABLED"] = "false"

# Now import after setting environment
from fastapi.testclient import TestClient
from monorepo.presentation.api.app import create_app

def test_health_endpoint():
    """Test that the health endpoint works."""
    app = create_app()
    client = TestClient(app)
    
    response = client.get("/api/v1/health/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    
    # Basic checks
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data
    assert data["overall_status"] in ["healthy", "degraded", "unhealthy"]
    
    print("✅ Health endpoint test passed!")
    return True

if __name__ == "__main__":
    test_health_endpoint()
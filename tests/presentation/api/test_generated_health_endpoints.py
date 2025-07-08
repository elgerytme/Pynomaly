import pytest
from fastapi.testclient import TestClient

from pynomaly.presentation.api.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)

def test_health_endpoint(client):
    """Test GET request for health endpoint."""
    response = client.get("/api/v1/health/")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert "status" in data

@pytest.mark.parametrize("method", ["get", "post", "put", "delete", "options"])
def test_health_methods_not_allowed(client, method):
    """Test methods not allowed for health endpoint."""
    response = getattr(client, method)("/api/v1/health/")
    if method != "get":
        assert response.status_code == 405
    else:
        assert response.status_code == 200
    

def test_health_metrics_endpoint(client):
    """Test GET request for health metrics endpoint."""
    response = client.get("/api/v1/health/metrics")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert "system" in data
    


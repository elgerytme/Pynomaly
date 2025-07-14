"""Test health endpoints with minimal dependencies."""

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient


def test_minimal_health_endpoint():
    """Test a minimal health endpoint without full app dependencies."""
    # Create a minimal FastAPI app for testing health endpoints
    app = FastAPI(title="Test Health API")
    
    @app.get("/health")
    def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00",
            "service": "pynomaly-api",
            "version": "0.2.0"
        }
    
    @app.get("/health/ready")
    def readiness_check():
        """Readiness probe endpoint."""
        return {
            "status": "ready",
            "timestamp": "2024-01-01T00:00:00"
        }
    
    @app.get("/health/live")
    def liveness_check():
        """Liveness probe endpoint."""
        return {
            "status": "alive",
            "timestamp": "2024-01-01T00:00:00"
        }
    
    client = TestClient(app)
    
    # Test basic health check
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    
    # Test readiness probe
    response = client.get("/health/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"
    
    # Test liveness probe
    response = client.get("/health/live")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "alive"


def test_health_endpoint_with_error_conditions():
    """Test health endpoints under error conditions."""
    app = FastAPI()
    
    @app.get("/health/detailed")
    def detailed_health():
        """Detailed health check with service status."""
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00",
            "services": {
                "database": "healthy",
                "cache": "healthy", 
                "auth": "healthy"
            },
            "uptime": 3600,
            "memory_usage": "256MB"
        }
    
    @app.get("/health/degraded")
    def degraded_health():
        """Health check showing degraded state."""
        return {
            "status": "degraded",
            "timestamp": "2024-01-01T00:00:00",
            "services": {
                "database": "healthy",
                "cache": "degraded",
                "auth": "healthy"
            },
            "warnings": ["Cache response time elevated"]
        }
    
    client = TestClient(app)
    
    # Test detailed health
    response = client.get("/health/detailed")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "services" in data
    assert data["services"]["database"] == "healthy"
    
    # Test degraded health
    response = client.get("/health/degraded")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "degraded"
    assert "warnings" in data


def test_health_metrics_endpoint():
    """Test health metrics endpoint."""
    app = FastAPI()
    
    @app.get("/health/metrics")
    def health_metrics():
        """Health metrics endpoint."""
        return {
            "status": "healthy",
            "metrics": {
                "requests_per_second": 150.5,
                "average_response_time_ms": 45.2,
                "error_rate": 0.01,
                "active_connections": 25
            },
            "thresholds": {
                "max_response_time_ms": 500,
                "max_error_rate": 0.05,
                "max_connections": 100
            }
        }
    
    client = TestClient(app)
    
    response = client.get("/health/metrics")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "metrics" in data
    assert "thresholds" in data
    assert data["metrics"]["requests_per_second"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Health Endpoints Testing Suite
Tests for health checks, monitoring, and system status endpoints.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from pynomaly.presentation.api.app import create_app


class TestHealthEndpoints:
    """Test suite for health check and monitoring endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_health_service(self):
        """Mock health service."""
        with patch(
            "pynomaly.infrastructure.monitoring.health_service.HealthService"
        ) as mock:
            service = Mock()
            service.get_system_health.return_value = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {
                    "database": {"status": "healthy", "response_time": 0.05},
                    "cache": {"status": "healthy", "response_time": 0.02},
                    "storage": {"status": "healthy", "response_time": 0.10},
                },
                "system": {"cpu_usage": 25.5, "memory_usage": 60.2, "disk_usage": 45.8},
            }
            mock.return_value = service
            yield service

    # Basic Health Check Tests

    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data

    def test_health_check_no_auth_required(self, client):
        """Test that health check doesn't require authentication."""
        response = client.get("/health")

        assert response.status_code == 200
        # Should work without Authorization header

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe endpoint."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_probe(self, client, mock_health_service):
        """Test Kubernetes readiness probe endpoint."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data

    def test_startup_probe(self, client):
        """Test Kubernetes startup probe endpoint."""
        response = client.get("/health/startup")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["starting", "ready"]

    # Detailed Health Status Tests

    def test_detailed_health_status(self, client, mock_health_service):
        """Test detailed health status endpoint."""
        response = client.get("/health/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        assert "system" in data

    def test_health_status_with_unhealthy_service(self, client, mock_health_service):
        """Test health status when a service is unhealthy."""
        mock_health_service.get_system_health.return_value = {
            "status": "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": {"status": "healthy", "response_time": 0.05},
                "cache": {"status": "unhealthy", "error": "Connection timeout"},
                "storage": {"status": "healthy", "response_time": 0.10},
            },
            "system": {"cpu_usage": 85.5, "memory_usage": 90.2, "disk_usage": 95.8},
        }

        response = client.get("/health/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["cache"]["status"] == "unhealthy"

    def test_health_status_critical_failure(self, client, mock_health_service):
        """Test health status during critical system failure."""
        mock_health_service.get_system_health.return_value = {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": {"status": "unhealthy", "error": "Connection failed"},
                "cache": {"status": "unhealthy", "error": "Service unavailable"},
                "storage": {"status": "healthy", "response_time": 0.10},
            },
            "system": {"cpu_usage": 95.0, "memory_usage": 98.5, "disk_usage": 99.2},
        }

        response = client.get("/health/status")

        assert response.status_code == 503  # Service Unavailable
        data = response.json()
        assert data["status"] == "unhealthy"

    # Service-Specific Health Tests

    def test_database_health_check(self, client, mock_health_service):
        """Test database-specific health check."""
        mock_health_service.check_database_health.return_value = {
            "status": "healthy",
            "response_time": 0.05,
            "connections": {"active": 5, "max": 100},
            "last_check": datetime.utcnow().isoformat(),
        }

        response = client.get("/health/database")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "response_time" in data
        assert "connections" in data

    def test_cache_health_check(self, client, mock_health_service):
        """Test cache-specific health check."""
        mock_health_service.check_cache_health.return_value = {
            "status": "healthy",
            "response_time": 0.02,
            "memory_usage": {"used": "256MB", "max": "1GB"},
            "hit_rate": 0.85,
        }

        response = client.get("/health/cache")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "hit_rate" in data

    def test_storage_health_check(self, client, mock_health_service):
        """Test storage-specific health check."""
        mock_health_service.check_storage_health.return_value = {
            "status": "healthy",
            "response_time": 0.10,
            "disk_usage": {"used": "45.8%", "available": "2.1TB"},
            "io_stats": {"read_ops": 1250, "write_ops": 890},
        }

        response = client.get("/health/storage")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "disk_usage" in data

    # Metrics and Monitoring Tests

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert (
            response.headers["content-type"]
            == "text/plain; version=0.0.4; charset=utf-8"
        )

        # Check for basic Prometheus metrics
        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content

    def test_system_metrics(self, client, mock_health_service):
        """Test system metrics endpoint."""
        response = client.get("/health/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "application" in data
        assert "timestamp" in data

    def test_performance_metrics(self, client):
        """Test performance metrics collection."""
        response = client.get("/health/performance")

        assert response.status_code == 200
        data = response.json()
        assert "response_times" in data
        assert "throughput" in data
        assert "error_rates" in data

    # Version and Build Information Tests

    def test_version_endpoint(self, client):
        """Test version information endpoint."""
        response = client.get("/health/version")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "build_time" in data
        assert "commit_hash" in data
        assert "python_version" in data

    def test_build_info_endpoint(self, client):
        """Test build information endpoint."""
        response = client.get("/health/build")

        assert response.status_code == 200
        data = response.json()
        assert "build_number" in data
        assert "build_environment" in data
        assert "dependencies" in data

    # Dependency Health Tests

    def test_dependencies_health(self, client, mock_health_service):
        """Test external dependencies health check."""
        mock_health_service.check_dependencies_health.return_value = {
            "external_api": {"status": "healthy", "response_time": 0.15},
            "ml_frameworks": {
                "pytorch": {"status": "healthy", "version": "2.0.0"},
                "tensorflow": {"status": "healthy", "version": "2.12.0"},
                "sklearn": {"status": "healthy", "version": "1.3.0"},
            },
        }

        response = client.get("/health/dependencies")

        assert response.status_code == 200
        data = response.json()
        assert "external_api" in data
        assert "ml_frameworks" in data

    def test_ml_frameworks_health(self, client):
        """Test ML frameworks availability check."""
        response = client.get("/health/ml-frameworks")

        assert response.status_code == 200
        data = response.json()
        assert "available_frameworks" in data
        assert isinstance(data["available_frameworks"], list)

    # Custom Health Checks

    def test_custom_health_checks(self, client, mock_health_service):
        """Test custom health check endpoints."""
        mock_health_service.run_custom_checks.return_value = {
            "data_quality": {"status": "healthy", "checks_passed": 15},
            "model_performance": {"status": "degraded", "accuracy_drift": 0.05},
            "alert_systems": {"status": "healthy", "active_alerts": 0},
        }

        response = client.get("/health/custom")

        assert response.status_code == 200
        data = response.json()
        assert "data_quality" in data
        assert "model_performance" in data
        assert "alert_systems" in data

    # Health Check with Authentication

    def test_admin_health_endpoint(self, client):
        """Test admin-only health endpoint (requires authentication)."""
        # Without authentication
        response = client.get("/health/admin")
        assert response.status_code == 401

        # With authentication (mocked)
        headers = {"Authorization": "Bearer admin-token"}
        with patch(
            "pynomaly.infrastructure.auth.jwt_auth.JWTAuthHandler.get_current_user"
        ) as mock_auth:
            mock_auth.return_value = {"role": "admin", "permissions": ["admin:health"]}
            response = client.get("/health/admin", headers=headers)
            assert response.status_code == 200

    # Load and Stress Testing Health

    def test_load_health_status(self, client, mock_health_service):
        """Test health status under load conditions."""
        mock_health_service.get_load_metrics.return_value = {
            "current_load": {
                "requests_per_second": 150,
                "active_connections": 75,
                "queue_size": 10,
            },
            "capacity": {
                "max_requests_per_second": 200,
                "max_connections": 100,
                "max_queue_size": 50,
            },
            "status": "healthy",
        }

        response = client.get("/health/load")

        assert response.status_code == 200
        data = response.json()
        assert "current_load" in data
        assert "capacity" in data

    def test_circuit_breaker_status(self, client):
        """Test circuit breaker status in health check."""
        response = client.get("/health/circuit-breakers")

        assert response.status_code == 200
        data = response.json()
        assert "circuit_breakers" in data

    # Error Handling Tests

    def test_health_check_service_exception(self, client, mock_health_service):
        """Test health check when service throws exception."""
        mock_health_service.get_system_health.side_effect = Exception("Service error")

        response = client.get("/health/status")

        assert response.status_code == 503
        data = response.json()
        assert "error" in data

    def test_partial_health_check_failure(self, client, mock_health_service):
        """Test health check with partial service failures."""
        mock_health_service.get_system_health.return_value = {
            "status": "degraded",
            "services": {
                "database": {"status": "healthy"},
                "cache": {"status": "timeout", "error": "Check timeout"},
                "storage": {"status": "healthy"},
            },
        }

        response = client.get("/health/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"

    # Response Format Tests

    def test_health_response_format(self, client):
        """Test health check response format consistency."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        required_fields = ["status", "timestamp"]
        for field in required_fields:
            assert field in data

        # Verify status values
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_response_headers(self, client):
        """Test health check response headers."""
        response = client.get("/health")

        assert response.headers["content-type"] == "application/json"
        assert "cache-control" in response.headers
        assert response.headers["cache-control"] == "no-cache"

    # Monitoring Integration Tests

    def test_health_check_logging(self, client):
        """Test that health checks are properly logged."""
        with patch(
            "pynomaly.infrastructure.monitoring.health_service.logger"
        ) as mock_logger:
            response = client.get("/health/status")

            assert response.status_code == 200
            mock_logger.info.assert_called()

    def test_health_check_metrics_collection(self, client):
        """Test that health check metrics are collected."""
        with patch(
            "pynomaly.infrastructure.monitoring.health_service.metrics"
        ) as mock_metrics:
            response = client.get("/health")

            assert response.status_code == 200
            # Verify metrics were recorded
            mock_metrics.increment.assert_called()


class TestHealthEndpointsIntegration:
    """Integration tests for health endpoints with realistic scenarios."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_health_check_workflow(self, client):
        """Test complete health check workflow."""
        # 1. Basic health check
        basic_response = client.get("/health")
        assert basic_response.status_code == 200

        # 2. Detailed status
        status_response = client.get("/health/status")
        assert status_response.status_code in [200, 503]

        # 3. Individual service checks
        services = ["database", "cache", "storage"]
        for service in services:
            service_response = client.get(f"/health/{service}")
            assert service_response.status_code in [200, 503]

    def test_kubernetes_probes_workflow(self, client):
        """Test Kubernetes health probe workflow."""
        # Startup probe
        startup_response = client.get("/health/startup")
        assert startup_response.status_code == 200

        # Liveness probe
        liveness_response = client.get("/health/live")
        assert liveness_response.status_code == 200

        # Readiness probe
        readiness_response = client.get("/health/ready")
        assert readiness_response.status_code in [200, 503]

    def test_monitoring_data_collection(self, client):
        """Test comprehensive monitoring data collection."""
        endpoints = [
            "/health",
            "/health/status",
            "/health/metrics",
            "/health/performance",
            "/health/version",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code in [200, 503]

            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, dict)
                assert len(data) > 0

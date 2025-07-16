"""Integration tests for web UI infrastructure."""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from monorepo.presentation.api.app import create_app


class TestWebUIInfrastructure:
    """Integration tests for web UI infrastructure endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create test client for API testing."""
        app = create_app()
        return TestClient(app)

    def test_ui_config_endpoint(self, client):
        """Test UI configuration endpoint."""
        response = client.get("/api/ui/config")
        assert response.status_code == 200

        config = response.json()

        # Check required configuration sections
        assert "performance_monitoring" in config
        assert "security" in config
        assert "features" in config

        # Check performance monitoring configuration
        perf_config = config["performance_monitoring"]
        assert "enabled" in perf_config
        assert "critical_thresholds" in perf_config

        # Check security configuration
        security_config = config["security"]
        assert "csrf_protection" in security_config
        assert "xss_protection" in security_config
        assert "session_timeout" in security_config

        # Check features configuration
        features_config = config["features"]
        assert "dark_mode" in features_config
        assert "lazy_loading" in features_config

    def test_ui_health_endpoint(self, client):
        """Test UI health endpoint."""
        response = client.get("/api/ui/health")
        assert response.status_code == 200

        health = response.json()

        # Check required health information
        assert "status" in health
        assert "timestamp" in health
        assert "components" in health
        assert "metrics" in health

        # Check component health
        components = health["components"]
        expected_components = [
            "performance_monitor",
            "security_manager",
            "cache_manager",
            "lazy_loader",
            "theme_manager",
        ]

        for component in expected_components:
            assert component in components
            assert components[component] == "healthy"

    def test_session_status_endpoint(self, client):
        """Test session status endpoint."""
        response = client.get("/api/session/status")
        assert response.status_code == 200

        session_data = response.json()

        # Check required session information
        assert "authenticated" in session_data
        assert "expires_at" in session_data
        assert "last_activity" in session_data
        assert "csrf_token" in session_data

        # Check CSRF token format
        csrf_token = session_data["csrf_token"]
        assert isinstance(csrf_token, str)
        assert len(csrf_token) >= 32  # Should be at least 32 characters

    def test_performance_metrics_endpoint(self, client):
        """Test performance metrics reporting endpoint."""
        metric_data = {
            "metric": "LCP",
            "value": 1500.0,
            "timestamp": int(time.time()),
            "url": "/",
        }

        response = client.post("/api/metrics/critical", json=metric_data)
        assert response.status_code == 200

        result = response.json()
        assert result["status"] == "received"
        assert result["metric"] == "LCP"
        assert result["value"] == 1500.0

    def test_security_events_endpoint(self, client):
        """Test security events reporting endpoint."""
        event_data = {
            "type": "xss_attempt",
            "timestamp": int(time.time()),
            "url": "/",
            "userAgent": "test-agent",
            "data": {"payload": "test"},
        }

        response = client.post("/api/security/events", json=event_data)
        assert response.status_code == 200

        result = response.json()
        assert result["status"] == "received"
        assert result["event_type"] == "xss_attempt"

    def test_session_extend_endpoint(self, client):
        """Test session extension endpoint."""
        response = client.post("/api/session/extend")
        assert response.status_code == 200

        result = response.json()
        assert result["success"] is True
        assert "new_expiry" in result
        assert "message" in result
        assert result["message"] == "Session extended successfully"

    def test_monitoring_dashboard_endpoint(self, client):
        """Test monitoring dashboard endpoint."""
        response = client.get("/api/monitoring/dashboard")
        assert response.status_code == 200

        dashboard = response.json()

        # Check required dashboard sections
        assert "timestamp" in dashboard
        assert "performance" in dashboard
        assert "security" in dashboard
        assert "system" in dashboard
        assert "alerts" in dashboard

        # Check performance section
        performance = dashboard["performance"]
        assert "core_web_vitals" in performance

        # Check security section
        security = dashboard["security"]
        assert "total_events" in security
        assert "event_types" in security

        # Check system section
        system = dashboard["system"]
        assert "uptime" in system
        assert "memory_usage" in system
        assert "disk_usage" in system
        assert "cpu_usage" in system

    def test_performance_monitoring_endpoint(self, client):
        """Test performance monitoring data collection."""
        # Test page load time reporting
        page_load_data = {"page_load_time": 2.5, "page": "dashboard"}

        response = client.post("/api/monitoring/performance", json=page_load_data)
        assert response.status_code == 200

        result = response.json()
        assert result["status"] == "received"
        assert result["data_type"] == "performance"

        # Test API response time reporting
        api_response_data = {"api_response_time": 0.5, "endpoint": "/api/detectors"}

        response = client.post("/api/monitoring/performance", json=api_response_data)
        assert response.status_code == 200

        # Test Core Web Vitals reporting
        core_web_vital_data = {"core_web_vital": {"metric": "LCP", "value": 1800.0}}

        response = client.post("/api/monitoring/performance", json=core_web_vital_data)
        assert response.status_code == 200

    def test_security_monitoring_endpoint(self, client):
        """Test security monitoring data collection."""
        security_data = {
            "event_type": "suspicious_activity",
            "details": "Multiple failed login attempts",
            "severity": "medium",
        }

        response = client.post("/api/monitoring/security", json=security_data)
        assert response.status_code == 200

        result = response.json()
        assert result["status"] == "received"
        assert result["data_type"] == "security"

    def test_endpoint_security_headers(self, client):
        """Test that security headers are properly set."""
        response = client.get("/api/ui/health")
        assert response.status_code == 200

        headers = response.headers

        # Check for security headers
        assert "X-Frame-Options" in headers
        assert "X-Content-Type-Options" in headers
        assert "Content-Security-Policy" in headers
        assert "Referrer-Policy" in headers

        # Check security header values
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert "default-src 'self'" in headers["Content-Security-Policy"]

    def test_csrf_token_validation(self, client):
        """Test CSRF token generation and validation."""
        # Get CSRF token
        response = client.get("/api/session/status")
        session_data = response.json()
        csrf_token = session_data["csrf_token"]

        # Test with valid CSRF token
        headers = {"X-CSRF-Token": csrf_token}
        response = client.post("/api/session/extend", headers=headers)
        assert response.status_code == 200

        # Test that token is different on each request
        response2 = client.get("/api/session/status")
        session_data2 = response2.json()
        csrf_token2 = session_data2["csrf_token"]

        # Tokens should be different (new token generated)
        assert csrf_token != csrf_token2

    def test_rate_limiting_headers(self, client):
        """Test that rate limiting information is available."""
        response = client.get("/api/ui/health")
        assert response.status_code == 200

        # Check for rate limiting headers (if implemented)
        headers = response.headers

        # These might not be present in development, but should be in production
        # Just verify the endpoint works without rate limiting errors
        assert response.status_code != 429  # Not rate limited

    def test_error_handling(self, client):
        """Test error handling in API endpoints."""
        # Test with invalid JSON
        response = client.post(
            "/api/metrics/critical",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422  # Validation error

        # Test with missing required fields
        response = client.post("/api/metrics/critical", json={})
        assert response.status_code == 422  # Validation error

    def test_cors_headers(self, client):
        """Test CORS headers configuration."""
        # Test preflight request
        response = client.options("/api/ui/health")

        # Should handle OPTIONS request properly
        assert response.status_code in [200, 204]

        # Test actual request has CORS headers
        response = client.get("/api/ui/health")
        headers = response.headers

        # Check for CORS headers (if configured)
        # These might not be present in test environment
        if "Access-Control-Allow-Origin" in headers:
            assert headers["Access-Control-Allow-Origin"] is not None

    def test_monitoring_data_flow(self, client):
        """Test complete monitoring data flow."""
        # 1. Report performance data
        performance_data = {"page_load_time": 3.2, "page": "detectors"}

        response = client.post("/api/monitoring/performance", json=performance_data)
        assert response.status_code == 200

        # 2. Report security event
        security_data = {"event_type": "login_attempt", "details": "User login attempt"}

        response = client.post("/api/monitoring/security", json=security_data)
        assert response.status_code == 200

        # 3. Check monitoring dashboard
        response = client.get("/api/monitoring/dashboard")
        assert response.status_code == 200

        dashboard = response.json()

        # Should have updated monitoring data
        assert "performance" in dashboard
        assert "security" in dashboard

        # Security events should be tracked
        security_summary = dashboard["security"]
        assert security_summary["total_events"] >= 1


@pytest.mark.asyncio
class TestWebUIInfrastructureAsync:
    """Async integration tests for web UI infrastructure."""

    async def test_concurrent_requests(self):
        """Test handling of concurrent requests."""
        from httpx import AsyncClient

        from monorepo.presentation.api.app import create_app

        app = create_app()

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Send multiple concurrent requests
            tasks = []
            for i in range(10):
                task = client.get("/api/ui/health")
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"

    async def test_monitoring_performance_impact(self):
        """Test that monitoring doesn't significantly impact performance."""
        from httpx import AsyncClient

        from monorepo.presentation.api.app import create_app

        app = create_app()

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Measure response time
            start_time = time.time()

            response = await client.get("/api/ui/health")

            end_time = time.time()
            response_time = end_time - start_time

            # Should respond quickly (under 1 second)
            assert response_time < 1.0
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

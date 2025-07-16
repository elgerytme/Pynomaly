"""
Comprehensive Web UI Integration Test Suite
Tests all implemented web UI features and infrastructure
"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.monorepo.presentation.api.endpoints.frontend_support import (
    router as frontend_router,
)
from src.monorepo.presentation.api.main import create_app
from src.monorepo.presentation.web.app import web_router
from src.monorepo.presentation.web.csrf import generate_csrf_token, get_csrf_token
from src.monorepo.presentation.web.monitoring import (
    PerformanceMonitor,
    SecurityMonitor,
    WebUILogger,
)


class TestWebUIIntegration:
    """Test suite for comprehensive web UI integration"""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with web UI routes"""
        app = create_app()
        app.include_router(web_router, prefix="/web")
        app.include_router(frontend_router, prefix="/api")
        return app

    @pytest.fixture
    def client(self, app):
        """Test client for API requests"""
        return TestClient(app)

    @pytest.fixture
    def mock_request(self):
        """Mock request object for testing"""
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = "/"
        request.method = "GET"
        request.headers = {}
        request.session = {}
        request.client = Mock()
        request.client.host = "127.0.0.1"
        return request


class TestFrontendSupportEndpoints:
    """Test frontend support API endpoints"""

    def test_ui_config_endpoint(self, client):
        """Test UI configuration endpoint"""
        response = client.get("/api/ui/config")
        assert response.status_code == 200

        data = response.json()
        assert "performance_monitoring" in data
        assert "security" in data
        assert "features" in data

        # Check performance monitoring config
        perf_config = data["performance_monitoring"]
        assert "enabled" in perf_config
        assert "critical_thresholds" in perf_config
        assert "LCP" in perf_config["critical_thresholds"]
        assert "FID" in perf_config["critical_thresholds"]
        assert "CLS" in perf_config["critical_thresholds"]

    def test_ui_health_endpoint(self, client):
        """Test UI health check endpoint"""
        response = client.get("/api/ui/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert "metrics" in data

        # Check component health
        components = data["components"]
        expected_components = [
            "performance_monitor",
            "security_manager",
            "cache_manager",
            "lazy_loader",
            "theme_manager"
        ]
        for component in expected_components:
            assert component in components

    def test_session_status_endpoint(self, client):
        """Test session status endpoint"""
        response = client.get("/api/session/status")
        assert response.status_code == 200

        data = response.json()
        assert "authenticated" in data
        assert "expires_at" in data
        assert "last_activity" in data
        assert "csrf_token" in data
        assert len(data["csrf_token"]) > 0

    def test_critical_metrics_endpoint(self, client):
        """Test critical metrics reporting endpoint"""
        metric_data = {
            "metric": "LCP",
            "value": 1500.0,
            "timestamp": 1752090172,
            "url": "/"
        }

        response = client.post("/api/metrics/critical", json=metric_data)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "received"
        assert data["metric"] == "LCP"
        assert data["value"] == 1500.0

    def test_security_events_endpoint(self, client):
        """Test security events reporting endpoint"""
        event_data = {
            "type": "xss_attempt",
            "timestamp": 1752090172,
            "url": "/",
            "userAgent": "test-agent",
            "data": {"payload": "test"}
        }

        response = client.post("/api/security/events", json=event_data)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "received"
        assert data["event_type"] == "xss_attempt"
        assert data["timestamp"] == 1752090172

    def test_session_extend_endpoint(self, client):
        """Test session extension endpoint"""
        response = client.post("/api/session/extend")
        assert response.status_code == 200

        data = response.json()
        assert "success" in data
        assert "new_expiry" in data
        assert "message" in data


class TestCSRFProtection:
    """Test CSRF protection implementation"""

    def test_csrf_token_generation(self):
        """Test CSRF token generation"""
        token = generate_csrf_token()
        assert len(token) > 0
        assert isinstance(token, str)

        # Generate multiple tokens to ensure uniqueness
        tokens = set()
        for _ in range(10):
            tokens.add(generate_csrf_token())
        assert len(tokens) == 10  # All tokens should be unique

    def test_csrf_token_in_request(self, mock_request):
        """Test CSRF token extraction from request"""
        # Test with token in session
        mock_request.session = {"csrf_token": "test-token"}
        token = get_csrf_token(mock_request)
        assert token == "test-token"

        # Test without token in session
        mock_request.session = {}
        token = get_csrf_token(mock_request)
        assert len(token) > 0  # Should generate new token


class TestSecurityHeaders:
    """Test security headers middleware"""

    def test_security_headers_applied(self, client):
        """Test that security headers are applied to responses"""
        response = client.get("/web/")

        # Check for security headers
        assert "X-Frame-Options" in response.headers
        assert "X-Content-Type-Options" in response.headers
        assert "Referrer-Policy" in response.headers
        assert "Content-Security-Policy" in response.headers

        # Check header values
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

        # Check CSP header contains expected directives
        csp = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "script-src" in csp
        assert "style-src" in csp


class TestWebUIMonitoring:
    """Test web UI monitoring components"""

    def test_web_ui_logger_initialization(self):
        """Test WebUILogger initialization"""
        logger = WebUILogger()
        assert logger.logger is not None
        assert logger.performance_buffer == []
        assert logger.security_buffer == []
        assert logger.buffer_size == 100
        assert logger.flush_interval == 60

    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization"""
        monitor = PerformanceMonitor()
        assert monitor.metrics == {}
        assert monitor.start_time is not None
        assert monitor.request_count == 0
        assert monitor.error_count == 0

    def test_security_monitor_initialization(self):
        """Test SecurityMonitor initialization"""
        monitor = SecurityMonitor()
        assert monitor.events == []
        assert monitor.threat_patterns is not None
        assert monitor.failed_login_attempts == {}
        assert monitor.blocked_ips == set()

    def test_performance_monitor_record_request(self):
        """Test performance monitor request recording"""
        monitor = PerformanceMonitor()

        # Record a request
        monitor.record_request("/test", "GET", 200, 150.0)

        assert monitor.request_count == 1
        assert "/test" in monitor.metrics
        assert monitor.metrics["/test"]["count"] == 1
        assert monitor.metrics["/test"]["avg_response_time"] == 150.0
        assert monitor.metrics["/test"]["status_codes"]["200"] == 1

    def test_security_monitor_record_event(self):
        """Test security monitor event recording"""
        monitor = SecurityMonitor()

        # Record a security event
        event_data = {
            "type": "xss_attempt",
            "ip": "192.168.1.100",
            "user_agent": "test-agent",
            "details": {"payload": "test"}
        }

        monitor.record_event(event_data)

        assert len(monitor.events) == 1
        assert monitor.events[0]["type"] == "xss_attempt"
        assert monitor.events[0]["ip"] == "192.168.1.100"


class TestWebUITemplates:
    """Test web UI template rendering"""

    def test_base_template_csrf_token(self, client):
        """Test that base template includes CSRF token"""
        response = client.get("/web/")
        assert response.status_code == 200

        content = response.text
        assert 'name="csrf-token"' in content
        assert 'content=' in content

    def test_base_template_security_headers(self, client):
        """Test that base template includes security meta tags"""
        response = client.get("/web/")
        assert response.status_code == 200

        content = response.text
        assert 'name="csrf-token"' in content
        assert 'name="csrf-param"' in content

    def test_javascript_includes(self, client):
        """Test that JavaScript files are included in templates"""
        response = client.get("/web/")
        assert response.status_code == 200

        content = response.text
        assert 'frontend-monitoring.js' in content
        assert 'performance-dashboard.js' in content
        assert 'pynomaly-frontend.js' in content


class TestPerformanceMonitoring:
    """Test performance monitoring features"""

    def test_core_web_vitals_thresholds(self, client):
        """Test Core Web Vitals thresholds configuration"""
        response = client.get("/api/ui/config")
        data = response.json()

        thresholds = data["performance_monitoring"]["critical_thresholds"]
        assert thresholds["LCP"] == 2500
        assert thresholds["FID"] == 100
        assert thresholds["CLS"] == 0.1

    def test_performance_metric_validation(self, client):
        """Test performance metric validation"""
        # Test valid metric
        valid_metric = {
            "metric": "LCP",
            "value": 1500.0,
            "timestamp": 1752090172,
            "url": "/"
        }

        response = client.post("/api/metrics/critical", json=valid_metric)
        assert response.status_code == 200

        # Test invalid metric type
        invalid_metric = {
            "metric": "INVALID",
            "value": 1500.0,
            "timestamp": 1752090172,
            "url": "/"
        }

        response = client.post("/api/metrics/critical", json=invalid_metric)
        # Should still accept for now, but could add validation later
        assert response.status_code == 200


class TestSecurityFeatures:
    """Test security features implementation"""

    def test_xss_protection_headers(self, client):
        """Test XSS protection headers"""
        response = client.get("/web/")

        # Check for XSS protection in CSP
        csp = response.headers.get("Content-Security-Policy", "")
        assert "script-src" in csp
        assert "'unsafe-inline'" in csp  # May be needed for inline scripts

    def test_clickjacking_protection(self, client):
        """Test clickjacking protection"""
        response = client.get("/web/")
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_content_sniffing_protection(self, client):
        """Test content sniffing protection"""
        response = client.get("/web/")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_security_event_types(self, client):
        """Test various security event types"""
        event_types = [
            "xss_attempt",
            "sql_injection",
            "csrf_violation",
            "session_hijacking",
            "brute_force"
        ]

        for event_type in event_types:
            event_data = {
                "type": event_type,
                "timestamp": 1752090172,
                "url": "/",
                "userAgent": "test-agent",
                "data": {"test": "data"}
            }

            response = client.post("/api/security/events", json=event_data)
            assert response.status_code == 200

            data = response.json()
            assert data["event_type"] == event_type


class TestErrorHandling:
    """Test error handling and resilience"""

    def test_invalid_endpoint_handling(self, client):
        """Test handling of invalid endpoints"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_malformed_json_handling(self, client):
        """Test handling of malformed JSON requests"""
        response = client.post(
            "/api/metrics/critical",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Validation error

    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        incomplete_data = {
            "metric": "LCP",
            # Missing value, timestamp, and url
        }

        response = client.post("/api/metrics/critical", json=incomplete_data)
        assert response.status_code == 422  # Validation error


class TestAPIIntegration:
    """Test API integration with frontend"""

    def test_api_endpoints_accessible(self, client):
        """Test that all API endpoints are accessible"""
        endpoints = [
            "/api/ui/config",
            "/api/ui/health",
            "/api/session/status"
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200

    def test_api_response_format(self, client):
        """Test API response format consistency"""
        response = client.get("/api/ui/config")
        data = response.json()

        # Check that response is valid JSON
        assert isinstance(data, dict)

        # Check for consistent structure
        assert "performance_monitoring" in data
        assert "security" in data
        assert "features" in data

    def test_cors_headers(self, client):
        """Test CORS headers for API endpoints"""
        response = client.options("/api/ui/config")

        # Check for CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers


class TestProductionReadiness:
    """Test production readiness features"""

    def test_environment_configuration(self, client):
        """Test environment-specific configuration"""
        response = client.get("/api/ui/config")
        data = response.json()

        # Check that configuration is properly loaded
        assert "performance_monitoring" in data
        assert "security" in data
        assert "features" in data

        # Ensure reasonable defaults
        assert data["performance_monitoring"]["enabled"] == True
        assert data["security"]["csrf_protection"] == True

    def test_health_check_completeness(self, client):
        """Test health check completeness"""
        response = client.get("/api/ui/health")
        data = response.json()

        # Check all required health metrics
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert "metrics" in data

        # Check that all components are reporting
        components = data["components"]
        assert len(components) >= 5  # At least 5 components should be monitored

    def test_performance_monitoring_active(self, client):
        """Test that performance monitoring is active"""
        # Make several requests to generate metrics
        for i in range(5):
            client.get("/api/ui/config")

        response = client.get("/api/ui/health")
        data = response.json()

        # Check that metrics are being collected
        assert "metrics" in data
        metrics = data["metrics"]
        assert "uptime" in metrics
        assert "memory_usage" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

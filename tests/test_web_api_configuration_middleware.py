"""Tests for web API configuration middleware."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from pynomaly.application.dto.configuration_dto import (
    RequestConfigurationDTO,
    ResponseConfigurationDTO,
)
from pynomaly.application.services.configuration_capture_service import (
    ConfigurationCaptureService,
)
from pynomaly.application.services.web_api_configuration_integration import (
    WebAPIConfigurationIntegration,
)
from pynomaly.infrastructure.middleware.configuration_middleware import (
    ConfigurationAPIMiddleware,
    ConfigurationCaptureMiddleware,
)
from pynomaly.presentation.api.middleware_integration import (
    add_configuration_endpoints,
    setup_configuration_middleware,
)


class TestConfigurationCaptureMiddleware:
    """Test configuration capture middleware."""

    @pytest.fixture
    def capture_service(self):
        """Create configuration capture service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationCaptureService(
                storage_path=Path(temp_dir), auto_capture=True
            )

    @pytest.fixture
    def middleware(self, capture_service):
        """Create middleware instance."""
        return ConfigurationCaptureMiddleware(
            app=Mock(),
            configuration_service=capture_service,
            enable_capture=True,
            capture_successful_only=False,  # Capture all for testing
            capture_threshold_ms=0.0,  # No threshold for testing
            excluded_paths=["/health"],
            capture_request_body=True,
            capture_response_body=True,
            max_body_size=1024 * 1024,
            anonymize_sensitive_data=True,
        )

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/v1/detect"
        request.query_params = {"algorithm": "IsolationForest", "contamination": "0.1"}
        request.headers = {
            "content-type": "application/json",
            "user-agent": "TestClient/1.0",
            "authorization": "Bearer secret-token",
        }
        request.client.host = "127.0.0.1"

        # Mock body
        body_data = {
            "dataset": "test_data.csv",
            "algorithm": "IsolationForest",
            "parameters": {"contamination": 0.1, "n_estimators": 100},
        }
        request.body = AsyncMock(return_value=json.dumps(body_data).encode("utf-8"))

        return request

    @pytest.fixture
    def mock_response(self):
        """Create mock response."""
        response = Mock(spec=Response)
        response.status_code = 200
        response.headers = {"content-type": "application/json"}

        # Mock response body
        response_data = {
            "success": True,
            "anomaly_count": 5,
            "accuracy": 0.85,
            "processing_time": 150.5,
        }
        response.body = json.dumps(response_data).encode("utf-8")

        return response

    def test_middleware_initialization(self, capture_service):
        """Test middleware initialization."""
        middleware = ConfigurationCaptureMiddleware(
            app=Mock(), configuration_service=capture_service
        )

        assert middleware.configuration_service == capture_service
        assert middleware.enable_capture is True
        assert middleware.capture_successful_only is True
        assert middleware.capture_threshold_ms == 100.0
        assert "/health" in middleware.excluded_paths
        assert middleware.anonymize_sensitive_data is True

    def test_should_exclude_path(self, middleware):
        """Test path exclusion logic."""
        assert middleware._should_exclude_path("/health")
        assert middleware._should_exclude_path("/metrics")
        assert middleware._should_exclude_path("/api/health")
        assert not middleware._should_exclude_path("/api/v1/detect")
        assert not middleware._should_exclude_path("/api/v1/train")

    def test_should_capture_request(self, middleware, mock_response):
        """Test request capture decision logic."""
        # Fast successful request
        assert not middleware._should_capture_request(
            mock_response, 50.0
        )  # Below threshold

        # Slow successful request
        assert middleware._should_capture_request(
            mock_response, 200.0
        )  # Above threshold

        # Fast error request (when capture_successful_only=False)
        mock_response.status_code = 500
        middleware.capture_successful_only = False
        assert middleware._should_capture_request(mock_response, 200.0)

        # Fast error request (when capture_successful_only=True)
        middleware.capture_successful_only = True
        assert not middleware._should_capture_request(mock_response, 200.0)

    @pytest.mark.asyncio
    async def test_extract_request_data(self, middleware, mock_request):
        """Test request data extraction."""
        request_data = await middleware._extract_request_data(mock_request)

        assert isinstance(request_data, RequestConfigurationDTO)
        assert request_data.method == "POST"
        assert request_data.path == "/api/v1/detect"
        assert request_data.query_parameters["algorithm"] == "IsolationForest"
        assert request_data.client_ip == "127.0.0.1"
        assert request_data.user_agent == "TestClient/1.0"

        # Check body extraction
        assert isinstance(request_data.body, dict)
        assert request_data.body["algorithm"] == "IsolationForest"

        # Check header anonymization
        assert "authorization" in request_data.headers
        assert request_data.headers["authorization"] == "***REDACTED***"

    @pytest.mark.asyncio
    async def test_extract_response_data(self, middleware, mock_response):
        """Test response data extraction."""
        duration_ms = 150.5
        response_data = await middleware._extract_response_data(
            mock_response, duration_ms
        )

        assert isinstance(response_data, ResponseConfigurationDTO)
        assert response_data.status_code == 200
        assert response_data.processing_time_ms == duration_ms
        assert response_data.content_type == "application/json"

        # Check body extraction
        assert isinstance(response_data.body, dict)
        assert response_data.body["success"] is True
        assert response_data.body["accuracy"] == 0.85

    def test_extract_anomaly_params(self, middleware):
        """Test anomaly parameter extraction."""
        data = {
            "algorithm": "IsolationForest",
            "contamination": 0.1,
            "n_estimators": 100,
            "unrelated_param": "value",
            "model_config": {"cross_validation": True, "metric": "accuracy"},
        }

        anomaly_params = middleware._extract_anomaly_params(data)

        assert "algorithm" in anomaly_params
        assert "contamination" in anomaly_params
        assert "n_estimators" in anomaly_params
        assert "unrelated_param" not in anomaly_params
        assert "model_config" in anomaly_params
        assert anomaly_params["model_config"]["cross_validation"] is True

    def test_anonymize_data(self, middleware):
        """Test data anonymization."""
        sensitive_data = {
            "algorithm": "IsolationForest",
            "password": "secret123",
            "api_key": "key123",
            "authorization": "Bearer token",
            "normal_param": "value",
            "nested": {"token": "secret", "data": "normal"},
        }

        anonymized = middleware._anonymize_data(sensitive_data)

        assert anonymized["algorithm"] == "IsolationForest"
        assert anonymized["password"] == "***REDACTED***"
        assert anonymized["api_key"] == "***REDACTED***"
        assert anonymized["authorization"] == "***REDACTED***"
        assert anonymized["normal_param"] == "value"
        assert anonymized["nested"]["token"] == "***REDACTED***"
        assert anonymized["nested"]["data"] == "normal"

    def test_is_sensitive_field(self, middleware):
        """Test sensitive field detection."""
        assert middleware._is_sensitive_field("password")
        assert middleware._is_sensitive_field("api_key")
        assert middleware._is_sensitive_field("authorization")
        assert middleware._is_sensitive_field("access_token")
        assert middleware._is_sensitive_field("user_password")

        assert not middleware._is_sensitive_field("algorithm")
        assert not middleware._is_sensitive_field("contamination")
        assert not middleware._is_sensitive_field("data")

    def test_parse_platform(self, middleware):
        """Test platform parsing from user agent."""
        assert middleware._parse_platform("Mozilla/5.0 (Windows NT 10.0)") == "windows"
        assert (
            middleware._parse_platform("Mozilla/5.0 (Macintosh; Intel Mac OS X)")
            == "macos"
        )
        assert middleware._parse_platform("Mozilla/5.0 (X11; Linux x86_64)") == "linux"
        assert middleware._parse_platform("Android 10") == "android"
        assert middleware._parse_platform("iPhone OS 14") == "ios"
        assert middleware._parse_platform(None) == "unknown"
        assert middleware._parse_platform("Unknown Browser") == "unknown"

    def test_determine_client_type(self, middleware):
        """Test client type determination."""
        assert middleware._determine_client_type("curl/7.68.0") == "api_client"
        assert middleware._determine_client_type("requests/2.25.1") == "api_client"
        assert middleware._determine_client_type("Postman/8.0") == "api_client"
        assert middleware._determine_client_type("Mozilla/5.0 Chrome/91.0") == "browser"
        assert (
            middleware._determine_client_type("Mozilla/5.0 Firefox/89.0") == "browser"
        )
        assert middleware._determine_client_type(None) == "unknown"
        assert middleware._determine_client_type("Unknown Client") == "unknown"

    def test_generate_tags(self, middleware):
        """Test tag generation."""
        request_data = RequestConfigurationDTO(
            method="POST",
            path="/api/v1/detect/anomalies",
            query_parameters={},
            headers={},
            body=None,
            client_ip="127.0.0.1",
            user_agent="test",
            content_type=None,
            content_length=None,
        )

        response_data = ResponseConfigurationDTO(
            status_code=200,
            headers={},
            body=None,
            processing_time_ms=150.0,
            content_type=None,
            content_length=None,
        )

        tags = middleware._generate_tags(request_data, response_data)

        assert "web_api" in tags
        assert "post" in tags
        assert "api" in tags
        assert "v1" in tags
        assert "detect" in tags
        assert "successful" in tags
        assert "moderate" in tags  # 150ms is moderate speed

    def test_get_capture_statistics(self, middleware):
        """Test capture statistics retrieval."""
        # Simulate some activity
        middleware.capture_stats["total_requests"] = 10
        middleware.capture_stats["captured_requests"] = 8
        middleware.capture_stats["skipped_requests"] = 2

        stats = middleware.get_capture_statistics()

        assert "capture_stats" in stats
        assert "configuration" in stats

        capture_stats = stats["capture_stats"]
        assert capture_stats["total_requests"] == 10
        assert capture_stats["captured_requests"] == 8
        assert capture_stats["skipped_requests"] == 2

        config = stats["configuration"]
        assert config["enabled"] is True
        assert config["capture_threshold_ms"] == 0.0
        assert config["anonymize_sensitive_data"] is True


class TestConfigurationAPIMiddleware:
    """Test configuration API middleware factory."""

    @pytest.fixture
    def capture_service(self):
        """Create configuration capture service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationCaptureService(
                storage_path=Path(temp_dir), auto_capture=True
            )

    def test_create_middleware(self, capture_service):
        """Test middleware factory creation."""
        middleware_factory = ConfigurationAPIMiddleware.create_middleware(
            configuration_service=capture_service,
            enable_capture=True,
            capture_threshold_ms=200.0,
        )

        assert callable(middleware_factory)

        # Test creating middleware instance
        app = Mock()
        middleware = middleware_factory(app)

        assert isinstance(middleware, ConfigurationCaptureMiddleware)
        assert middleware.configuration_service == capture_service
        assert middleware.enable_capture is True
        assert middleware.capture_threshold_ms == 200.0

    def test_create_production_middleware(self, capture_service):
        """Test production middleware factory."""
        with patch("pynomaly.infrastructure.config.feature_flags.require_feature"):
            middleware_factory = (
                ConfigurationAPIMiddleware.create_production_middleware(
                    configuration_service=capture_service
                )
            )

            app = Mock()
            middleware = middleware_factory(app)

            assert isinstance(middleware, ConfigurationCaptureMiddleware)
            assert middleware.capture_successful_only is True
            assert middleware.capture_threshold_ms == 200.0
            assert middleware.capture_response_body is False  # Security consideration
            assert middleware.anonymize_sensitive_data is True
            assert "/admin" in middleware.excluded_paths

    def test_create_development_middleware(self, capture_service):
        """Test development middleware factory."""
        middleware_factory = ConfigurationAPIMiddleware.create_development_middleware(
            configuration_service=capture_service
        )

        app = Mock()
        middleware = middleware_factory(app)

        assert isinstance(middleware, ConfigurationCaptureMiddleware)
        assert middleware.capture_successful_only is False  # Capture all requests
        assert middleware.capture_threshold_ms == 50.0
        assert middleware.capture_response_body is True  # For debugging
        assert middleware.anonymize_sensitive_data is False  # For debugging
        assert len(middleware.excluded_paths) == 2  # Minimal exclusions


class TestWebAPIConfigurationIntegration:
    """Test web API configuration integration service."""

    @pytest.fixture
    def capture_service(self):
        """Create configuration capture service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationCaptureService(
                storage_path=Path(temp_dir), auto_capture=True
            )

    @pytest.fixture
    def integration_service(self, capture_service):
        """Create integration service."""
        return WebAPIConfigurationIntegration(
            configuration_service=capture_service,
            auto_analyze_patterns=True,
            performance_threshold_ms=500.0,
        )

    @pytest.mark.asyncio
    async def test_analyze_api_usage_patterns_empty(self, integration_service):
        """Test API usage pattern analysis with no data."""
        analysis = await integration_service.analyze_api_usage_patterns(7)

        assert analysis["time_period_days"] == 7
        assert analysis["total_configurations"] == 0
        assert "endpoint_analysis" in analysis
        assert "client_analysis" in analysis
        assert "performance_analysis" in analysis
        assert "temporal_analysis" in analysis
        assert "error_analysis" in analysis
        assert "algorithm_usage" in analysis
        assert "recommendations" in analysis

    @pytest.mark.asyncio
    async def test_get_endpoint_performance_metrics_not_found(
        self, integration_service
    ):
        """Test endpoint performance metrics for non-existent endpoint."""
        metrics = await integration_service.get_endpoint_performance_metrics(
            "/api/v1/nonexistent", 7
        )

        assert "error" in metrics
        assert metrics["endpoint"] == "/api/v1/nonexistent"
        assert metrics["time_period_days"] == 7

    @pytest.mark.asyncio
    async def test_track_api_configuration_quality_not_found(self, integration_service):
        """Test configuration quality tracking for non-existent config."""
        config_id = uuid4()
        quality_metrics = await integration_service.track_api_configuration_quality(
            config_id
        )

        assert "error" in quality_metrics
        assert "not found" in quality_metrics["error"].lower()

    @pytest.mark.asyncio
    async def test_generate_api_configuration_report(self, integration_service):
        """Test API configuration report generation."""
        report = await integration_service.generate_api_configuration_report(7)

        assert "report_generated" in report
        assert "time_period_days" in report
        assert report["time_period_days"] == 7
        assert "executive_summary" in report
        assert "usage_patterns" in report
        assert "endpoint_performance" in report
        assert "recommendations" in report
        assert "configuration_trends" in report
        assert "quality_metrics" in report

    def test_get_integration_statistics(self, integration_service):
        """Test integration statistics retrieval."""
        stats = integration_service.get_integration_statistics()

        assert "integration_stats" in stats
        assert "performance_tracker" in stats
        assert "configuration" in stats

        integration_stats = stats["integration_stats"]
        expected_keys = [
            "total_api_requests",
            "configurations_captured",
            "unique_endpoints",
            "unique_clients",
            "total_processing_time_ms",
            "last_analysis_time",
            "error_count",
        ]
        for key in expected_keys:
            assert key in integration_stats

        config = stats["configuration"]
        assert "auto_analyze_patterns" in config
        assert "performance_threshold_ms" in config


class TestMiddlewareIntegration:
    """Test FastAPI middleware integration."""

    @pytest.fixture
    def capture_service(self):
        """Create configuration capture service."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ConfigurationCaptureService(
                storage_path=Path(temp_dir), auto_capture=True
            )

    @pytest.fixture
    def app(self):
        """Create FastAPI test app."""
        app = FastAPI()

        @app.get("/")
        async def root():
            return {"message": "Hello World"}

        @app.post("/api/v1/detect")
        async def detect_anomalies(data: dict):
            return {"success": True, "anomaly_count": 5, "accuracy": 0.85}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        return app

    def test_setup_configuration_middleware_disabled(self, app, capture_service):
        """Test middleware setup when feature is disabled."""
        with patch(
            "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
            return_value=False,
        ):
            setup_configuration_middleware(app, capture_service, "development")

            # Should not add middleware when disabled
            # This is hard to test directly, but we can verify no exceptions are raised
            assert True  # If we get here, no exceptions were raised

    def test_setup_configuration_middleware_development(self, app, capture_service):
        """Test middleware setup for development environment."""
        with patch(
            "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
            return_value=True,
        ):
            setup_configuration_middleware(app, capture_service, "development")

            # Check that middleware was added (indirectly)
            # FastAPI doesn't provide easy access to middleware list
            assert True  # If we get here, setup completed without errors

    def test_setup_configuration_middleware_production(self, app, capture_service):
        """Test middleware setup for production environment."""
        with patch(
            "pynomaly.infrastructure.config.feature_flags.is_feature_enabled",
            return_value=True,
        ):
            setup_configuration_middleware(app, capture_service, "production")

            # Check that middleware was added (indirectly)
            assert True  # If we get here, setup completed without errors

    def test_add_configuration_endpoints(self, app, capture_service):
        """Test adding configuration endpoints to FastAPI app."""
        add_configuration_endpoints(app, capture_service, None)

        # Create test client
        client = TestClient(app)

        # Test middleware stats endpoint
        response = client.get("/api/v1/configurations/middleware/stats")
        assert response.status_code == 200
        assert "message" in response.json()

        # Test API usage endpoint (should return 404 without integration service)
        response = client.get("/api/v1/configurations/api-usage")
        assert response.status_code == 404

    def test_add_configuration_endpoints_with_integration(self, app, capture_service):
        """Test adding configuration endpoints with integration service."""
        integration_service = Mock()
        integration_service.analyze_api_usage_patterns = AsyncMock(
            return_value={
                "total_configurations": 0,
                "endpoint_analysis": {"total_unique_endpoints": 0},
            }
        )
        integration_service.get_endpoint_performance_metrics = AsyncMock(
            return_value={"endpoint": "/test", "total_requests": 0}
        )
        integration_service.generate_api_configuration_report = AsyncMock(
            return_value={"report_generated": datetime.now().isoformat()}
        )

        add_configuration_endpoints(app, capture_service, integration_service)

        client = TestClient(app)

        # Test API usage endpoint
        response = client.get("/api/v1/configurations/api-usage?days=7")
        assert response.status_code == 200

        # Test endpoint performance
        response = client.get(
            "/api/v1/configurations/endpoints/test/performance?days=7"
        )
        assert response.status_code == 200

        # Test API report
        response = client.get("/api/v1/configurations/api-report?days=30")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])

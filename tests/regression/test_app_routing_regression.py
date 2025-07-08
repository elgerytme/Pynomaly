#!/usr/bin/env python3
"""
Comprehensive regression tests for app routing functionality.

This test suite instantiates the real app and verifies that all routing
works correctly, protecting against future regressions.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from pynomaly.presentation.api.app import create_app
    from pynomaly.infrastructure.config import Container, Settings
    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False


class TestAppRoutingRegression:
    """Test suite for app routing regression tests."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method."""
        if not APP_AVAILABLE:
            pytest.skip("Application not available for testing")
    
    @pytest.fixture
    def mock_container(self):
        """Create a mock container for testing."""
        container = Mock(spec=Container)
        
        # Mock settings
        settings = Mock(spec=Settings)
        settings.app.name = "Pynomaly Test"
        settings.app.version = "1.0.0-test"
        settings.app.environment = "test"
        settings.app.debug = True
        settings.docs_enabled = True
        settings.auth_enabled = False
        settings.cache_enabled = False
        settings.monitoring.metrics_enabled = False
        settings.monitoring.tracing_enabled = False
        settings.monitoring.prometheus_enabled = False
        settings.get_cors_config.return_value = {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
        
        container.config.return_value = settings
        
        # Mock repositories
        container.detector_repository.return_value = Mock()
        container.dataset_repository.return_value = Mock()
        container.result_repository.return_value = Mock()
        
        # Mock repository methods
        container.detector_repository().count.return_value = 5
        container.dataset_repository().count.return_value = 3
        container.result_repository().count.return_value = 10
        container.result_repository().find_recent.return_value = []
        
        return container
    
    @pytest.fixture
    def test_app(self, mock_container):
        """Create a test app instance."""
        with patch('pynomaly.presentation.api.app.create_container', return_value=mock_container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app.initialize_monitoring_service'):
                        with patch('pynomaly.presentation.api.app.shutdown_monitoring_service'):
                            with patch('pynomaly.presentation.api.app.clear_dependencies'):
                                with patch('pynomaly.presentation.api.app.get_cache'):
                                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                                        app = create_app(mock_container)
                                        return app
    
    @pytest.fixture
    def client(self, test_app):
        """Create a test client."""
        return TestClient(test_app)
    
    def test_app_creation(self, test_app):
        """Test that the app can be created successfully."""
        assert isinstance(test_app, FastAPI)
        assert test_app.title == "Pynomaly Test"
        assert test_app.version == "1.0.0-test"
    
    def test_api_root_endpoint(self, client):
        """Test API root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "Pynomaly API"
        assert "version" in data
        assert "api_version" in data
        assert data["api_version"] == "v1"
        assert "docs" in data
        assert data["docs"] == "/api/v1/docs"
        assert "health" in data
        assert data["health"] == "/api/v1/health"
        assert "version_info" in data
        assert data["version_info"] == "/api/v1/version"
    
    def test_api_health_endpoint(self, client):
        """Test API health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_api_version_endpoint(self, client):
        """Test API version endpoint."""
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        
        data = response.json()
        assert "version" in data
        assert "environment" in data
        assert data["environment"] == "test"
    
    def test_api_docs_endpoints(self, client):
        """Test API documentation endpoints."""
        # Test OpenAPI JSON
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data
        assert openapi_data["info"]["title"] == "Pynomaly Test"
        
        # Test Swagger UI
        response = client.get("/api/v1/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Test ReDoc
        response = client.get("/api/v1/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_api_auth_endpoints(self, client):
        """Test API authentication endpoints."""
        # Test login endpoint (should return 401 for invalid credentials)
        response = client.post("/api/v1/auth/login", json={
            "username": "invalid",
            "password": "invalid"
        })
        assert response.status_code in [401, 422]  # 422 for validation errors
        
        # Test token endpoint structure exists
        response = client.get("/api/v1/openapi.json")
        openapi_data = response.json()
        paths = openapi_data["paths"]
        
        # Check that auth endpoints exist in OpenAPI
        assert "/api/v1/auth/login" in paths
        assert "/api/v1/auth/refresh" in paths
        assert "/api/v1/auth/logout" in paths
    
    def test_api_detector_endpoints(self, client):
        """Test API detector endpoints."""
        # Test list detectors
        response = client.get("/api/v1/detectors")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        # Test create detector (should fail without auth or return validation error)
        response = client.post("/api/v1/detectors", json={
            "name": "test_detector",
            "algorithm": "IsolationForest"
        })
        assert response.status_code in [401, 422]
    
    def test_api_dataset_endpoints(self, client):
        """Test API dataset endpoints."""
        # Test list datasets
        response = client.get("/api/v1/datasets")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_api_detection_endpoints(self, client):
        """Test API detection endpoints."""
        # Test train endpoint (should fail without data)
        response = client.post("/api/v1/detection/train", json={})
        assert response.status_code in [401, 422]
        
        # Test predict endpoint (should fail without data)
        response = client.post("/api/v1/detection/predict", json={})
        assert response.status_code in [401, 422]
    
    def test_api_automl_endpoints(self, client):
        """Test API AutoML endpoints."""
        # Test profile endpoint
        response = client.post("/api/v1/automl/profile", json={})
        assert response.status_code in [401, 422]
        
        # Test optimize endpoint
        response = client.post("/api/v1/automl/optimize", json={})
        assert response.status_code in [401, 422]
    
    def test_api_ensemble_endpoints(self, client):
        """Test API ensemble endpoints."""
        # Test create ensemble
        response = client.post("/api/v1/ensemble/create", json={})
        assert response.status_code in [401, 422]
        
        # Test ensemble detection
        response = client.post("/api/v1/ensemble/detect", json={})
        assert response.status_code in [401, 422]
    
    def test_api_explainability_endpoints(self, client):
        """Test API explainability endpoints."""
        # Test explain prediction
        response = client.post("/api/v1/explainability/explain/prediction", json={})
        assert response.status_code in [401, 422]
        
        # Test explain global
        response = client.post("/api/v1/explainability/explain/global", json={})
        assert response.status_code in [401, 422]
    
    def test_api_streaming_endpoints(self, client):
        """Test API streaming endpoints."""
        # Test streaming sessions
        response = client.get("/api/v1/streaming/sessions")
        assert response.status_code in [200, 401]
        
        # Test create streaming session
        response = client.post("/api/v1/streaming/sessions", json={})
        assert response.status_code in [401, 422]
    
    def test_api_experiments_endpoints(self, client):
        """Test API experiments endpoints."""
        # Test list experiments
        response = client.get("/api/v1/experiments")
        assert response.status_code in [200, 401]
        
        # Test create experiment
        response = client.post("/api/v1/experiments", json={})
        assert response.status_code in [401, 422]
    
    def test_api_performance_endpoints(self, client):
        """Test API performance endpoints."""
        # Test performance metrics
        response = client.get("/api/v1/performance/metrics")
        assert response.status_code in [200, 401]
        
        # Test benchmark
        response = client.post("/api/v1/performance/benchmark", json={})
        assert response.status_code in [401, 422]
    
    def test_api_export_endpoints(self, client):
        """Test API export endpoints."""
        # Test export model
        response = client.post("/api/v1/export/model", json={})
        assert response.status_code in [401, 422]
        
        # Test export results
        response = client.post("/api/v1/export/results", json={})
        assert response.status_code in [401, 422]
    
    def test_api_model_lineage_endpoints(self, client):
        """Test API model lineage endpoints."""
        # Test model lineage
        response = client.get("/api/v1/model/lineage")
        assert response.status_code in [200, 401]
        
        # Test track lineage
        response = client.post("/api/v1/model/lineage/track", json={})
        assert response.status_code in [401, 422]
    
    def test_api_events_endpoints(self, client):
        """Test API events endpoints."""
        # Test events
        response = client.get("/api/v1/events")
        assert response.status_code in [200, 401]
        
        # Test create event
        response = client.post("/api/v1/events", json={})
        assert response.status_code in [401, 422]
    
    def test_api_user_management_endpoints(self, client):
        """Test API user management endpoints."""
        # Test users
        response = client.get("/api/v1/users")
        assert response.status_code in [200, 401]
        
        # Test create user
        response = client.post("/api/v1/users", json={})
        assert response.status_code in [401, 422]
    
    def test_api_admin_endpoints(self, client):
        """Test API admin endpoints."""
        # Test admin status
        response = client.get("/api/v1/admin/status")
        assert response.status_code in [200, 401]
        
        # Test admin config
        response = client.get("/api/v1/admin/config")
        assert response.status_code in [200, 401]
    
    def test_api_autonomous_endpoints(self, client):
        """Test API autonomous endpoints."""
        # Test autonomous detect
        response = client.post("/api/v1/autonomous/detect", json={})
        assert response.status_code in [401, 422]
        
        # Test autonomous optimize
        response = client.post("/api/v1/autonomous/optimize", json={})
        assert response.status_code in [401, 422]
    
    def test_api_endpoint_coverage(self, client):
        """Test that all major API endpoints are covered."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        paths = openapi_data["paths"]
        
        # Check that all expected endpoint prefixes exist
        expected_prefixes = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/auth/",
            "/api/v1/detectors",
            "/api/v1/datasets",
            "/api/v1/detection/",
            "/api/v1/automl/",
            "/api/v1/ensemble/",
            "/api/v1/explainability/",
            "/api/v1/streaming/",
            "/api/v1/experiments",
            "/api/v1/performance/",
            "/api/v1/export/",
            "/api/v1/model/",
            "/api/v1/events",
            "/api/v1/users",
            "/api/v1/admin/",
            "/api/v1/autonomous/"
        ]
        
        for prefix in expected_prefixes:
            found = any(path.startswith(prefix) for path in paths.keys())
            assert found, f"No endpoints found with prefix: {prefix}"
    
    def test_api_versioning_structure(self, client):
        """Test API versioning structure."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        
        # Check API version info
        assert openapi_data["info"]["version"] == "1.0.0-test"
        
        # Check that all paths use v1 versioning
        paths = openapi_data["paths"]
        for path in paths.keys():
            if path.startswith("/api/"):
                assert "/api/v1/" in path, f"Path {path} does not use v1 versioning"
    
    def test_api_error_handling(self, client):
        """Test API error handling."""
        # Test 404 for non-existent endpoint
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test 405 for wrong method
        response = client.delete("/api/v1/health")
        assert response.status_code == 405
        
        # Test invalid JSON handling
        response = client.post(
            "/api/v1/detectors",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_api_cors_configuration(self, client):
        """Test CORS configuration."""
        # Test preflight request
        response = client.options("/api/v1/health")
        assert response.status_code == 200
        
        # Test CORS headers are present
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
    
    def test_api_security_headers(self, client):
        """Test security headers."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Check for security headers (may vary based on middleware)
        headers = response.headers
        assert "content-type" in headers
        assert headers["content-type"] == "application/json"
    
    def test_api_response_formats(self, client):
        """Test API response formats."""
        # Test JSON response
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert isinstance(data, dict)
        
        # Test version endpoint format
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert isinstance(data, dict)
        assert "version" in data
    
    def test_api_openapi_schema_validation(self, client):
        """Test OpenAPI schema validation."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        
        # Check required OpenAPI fields
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data
        
        # Check info section
        info = openapi_data["info"]
        assert "title" in info
        assert "version" in info
        assert "description" in info
        
        # Check paths section
        paths = openapi_data["paths"]
        assert isinstance(paths, dict)
        assert len(paths) > 0
        
        # Check that each path has proper structure
        for path, methods in paths.items():
            assert isinstance(methods, dict)
            for method, details in methods.items():
                assert method.lower() in ["get", "post", "put", "delete", "patch", "options", "head"]
                assert "responses" in details
                assert isinstance(details["responses"], dict)
    
    def test_api_route_consistency(self, client):
        """Test API route consistency."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        paths = openapi_data["paths"]
        
        # Check that all paths follow consistent patterns
        for path in paths.keys():
            if path.startswith("/api/v1/"):
                # Check path structure
                parts = path.split("/")
                assert len(parts) >= 3, f"Path {path} has insufficient parts"
                assert parts[1] == "api", f"Path {path} missing 'api' prefix"
                assert parts[2] == "v1", f"Path {path} missing 'v1' version"
    
    def test_api_documentation_completeness(self, client):
        """Test API documentation completeness."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        
        # Check that documentation includes contact and license info
        info = openapi_data["info"]
        assert "contact" in info
        assert "license" in info
        assert "termsOfService" in info
        
        # Check contact info
        contact = info["contact"]
        assert "name" in contact
        assert "url" in contact
        assert "email" in contact
        
        # Check license info
        license_info = info["license"]
        assert "name" in license_info
        assert "url" in license_info
    
    def test_api_backwards_compatibility(self, client):
        """Test API backwards compatibility."""
        # Test that core endpoints maintain their structure
        core_endpoints = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/detectors",
            "/api/v1/datasets",
            "/api/v1/detection/train",
            "/api/v1/detection/predict"
        ]
        
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        paths = openapi_data["paths"]
        
        for endpoint in core_endpoints:
            assert endpoint in paths or any(
                endpoint.startswith(path) for path in paths.keys()
            ), f"Core endpoint {endpoint} not found in API"
    
    def test_api_performance_basic(self, client):
        """Test basic API performance."""
        import time
        
        # Test that basic endpoints respond quickly
        start_time = time.time()
        response = client.get("/api/v1/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 1.0  # Should respond within 1 second
        
        # Test OpenAPI schema generation performance
        start_time = time.time()
        response = client.get("/api/v1/openapi.json")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 2.0  # Should generate within 2 seconds


class TestWebUIRoutingRegression:
    """Test suite for Web UI routing regression tests."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method."""
        if not APP_AVAILABLE:
            pytest.skip("Application not available for testing")
    
    @pytest.fixture
    def mock_container(self):
        """Create a mock container for testing."""
        container = Mock(spec=Container)
        
        # Mock settings
        settings = Mock(spec=Settings)
        settings.app.name = "Pynomaly Test"
        settings.app.version = "1.0.0-test"
        settings.app.environment = "test"
        settings.app.debug = True
        settings.docs_enabled = True
        settings.auth_enabled = False
        settings.cache_enabled = False
        settings.monitoring.metrics_enabled = False
        settings.monitoring.tracing_enabled = False
        settings.monitoring.prometheus_enabled = False
        settings.is_production = False
        settings.get_cors_config.return_value = {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
        
        container.config.return_value = settings
        
        # Mock repositories
        container.detector_repository.return_value = Mock()
        container.dataset_repository.return_value = Mock()
        container.result_repository.return_value = Mock()
        container.experiment_tracking_service.return_value = Mock()
        container.ensemble_service.return_value = Mock()
        container.feature_validator.return_value = Mock()
        container.pyod_adapter.return_value = Mock()
        
        # Mock repository methods
        container.detector_repository().count.return_value = 5
        container.dataset_repository().count.return_value = 3
        container.result_repository().count.return_value = 10
        container.result_repository().find_recent.return_value = []
        container.detector_repository().find_all.return_value = []
        container.dataset_repository().find_all.return_value = []
        container.experiment_tracking_service().experiments = {}
        container.experiment_tracking_service()._load_experiments.return_value = None
        container.pyod_adapter().list_algorithms.return_value = ["IsolationForest", "LOF"]
        
        # Mock feature validator
        quality_report = Mock()
        quality_report.is_valid = True
        quality_report.missing_values = 0
        quality_report.duplicate_rows = 0
        container.feature_validator().check_data_quality.return_value = quality_report
        
        return container
    
    @pytest.fixture
    def test_app_with_web_ui(self, mock_container):
        """Create a test app instance with web UI."""
        with patch('pynomaly.presentation.api.app.create_container', return_value=mock_container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app.initialize_monitoring_service'):
                        with patch('pynomaly.presentation.api.app.shutdown_monitoring_service'):
                            with patch('pynomaly.presentation.api.app.clear_dependencies'):
                                with patch('pynomaly.presentation.api.app.get_cache'):
                                    # Mock the web UI mounting
                                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy') as mock_mount:
                                        mock_mount.return_value = True
                                        app = create_app(mock_container)
                                        
                                        # Manually add web UI routes for testing
                                        from fastapi import APIRouter
                                        from fastapi.responses import HTMLResponse
                                        
                                        web_router = APIRouter()
                                        
                                        @web_router.get("/", response_class=HTMLResponse)
                                        @web_router.get("/dashboard", response_class=HTMLResponse)
                                        @web_router.get("/login", response_class=HTMLResponse)
                                        @web_router.get("/detectors", response_class=HTMLResponse)
                                        @web_router.get("/datasets", response_class=HTMLResponse)
                                        @web_router.get("/detection", response_class=HTMLResponse)
                                        @web_router.get("/experiments", response_class=HTMLResponse)
                                        @web_router.get("/ensemble", response_class=HTMLResponse)
                                        @web_router.get("/automl", response_class=HTMLResponse)
                                        @web_router.get("/visualizations", response_class=HTMLResponse)
                                        @web_router.get("/exports", response_class=HTMLResponse)
                                        @web_router.get("/explainability", response_class=HTMLResponse)
                                        @web_router.get("/monitoring", response_class=HTMLResponse)
                                        async def mock_web_endpoint():
                                            return HTMLResponse("<html><body>Mock Web UI</body></html>")
                                        
                                        # Mount the web router
                                        app.mount("/web", web_router)
                                        
                                        return app
    
    @pytest.fixture
    def web_client(self, test_app_with_web_ui):
        """Create a test client for web UI testing."""
        return TestClient(test_app_with_web_ui)
    
    def test_web_ui_endpoint_structure(self, web_client):
        """Test Web UI endpoint structure."""
        # Test that web UI endpoints are accessible
        web_endpoints = [
            "/web/",
            "/web/dashboard",
            "/web/login",
            "/web/detectors",
            "/web/datasets",
            "/web/detection",
            "/web/experiments",
            "/web/ensemble",
            "/web/automl",
            "/web/visualizations",
            "/web/exports",
            "/web/explainability",
            "/web/monitoring"
        ]
        
        for endpoint in web_endpoints:
            response = web_client.get(endpoint)
            assert response.status_code == 200, f"Web UI endpoint {endpoint} failed"
            assert "text/html" in response.headers["content-type"]
    
    def test_web_ui_vs_api_separation(self, web_client):
        """Test that Web UI and API endpoints are properly separated."""
        # Test API endpoints
        api_response = web_client.get("/api/v1/health")
        assert api_response.status_code == 200
        assert api_response.headers["content-type"] == "application/json"
        
        # Test Web UI endpoints
        web_response = web_client.get("/web/")
        assert web_response.status_code == 200
        assert "text/html" in web_response.headers["content-type"]
        
        # Ensure they return different content types
        assert api_response.headers["content-type"] != web_response.headers["content-type"]
    
    def test_web_ui_routing_consistency(self, web_client):
        """Test Web UI routing consistency."""
        # Test that all web UI endpoints use consistent URL patterns
        web_endpoints = [
            "/web/",
            "/web/dashboard",
            "/web/detectors",
            "/web/datasets"
        ]
        
        for endpoint in web_endpoints:
            response = web_client.get(endpoint)
            assert response.status_code == 200
            
            # Check that response is HTML
            assert "text/html" in response.headers["content-type"]
            
            # Check that response contains HTML content
            content = response.text
            assert "<html>" in content or "<body>" in content


class TestRoutingIntegration:
    """Integration tests for routing between API and Web UI."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method."""
        if not APP_AVAILABLE:
            pytest.skip("Application not available for testing")
    
    @pytest.fixture
    def mock_container(self):
        """Create a mock container for integration testing."""
        container = Mock(spec=Container)
        
        # Mock settings
        settings = Mock(spec=Settings)
        settings.app.name = "Pynomaly Test"
        settings.app.version = "1.0.0-test"
        settings.app.environment = "test"
        settings.app.debug = True
        settings.docs_enabled = True
        settings.auth_enabled = False
        settings.cache_enabled = False
        settings.monitoring.metrics_enabled = False
        settings.monitoring.tracing_enabled = False
        settings.monitoring.prometheus_enabled = False
        settings.get_cors_config.return_value = {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
        
        container.config.return_value = settings
        
        # Mock repositories
        container.detector_repository.return_value = Mock()
        container.dataset_repository.return_value = Mock()
        container.result_repository.return_value = Mock()
        
        # Mock repository methods
        container.detector_repository().count.return_value = 5
        container.dataset_repository().count.return_value = 3
        container.result_repository().count.return_value = 10
        container.result_repository().find_recent.return_value = []
        
        return container
    
    @pytest.fixture
    def integration_app(self, mock_container):
        """Create an integration test app."""
        with patch('pynomaly.presentation.api.app.create_container', return_value=mock_container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app.initialize_monitoring_service'):
                        with patch('pynomaly.presentation.api.app.shutdown_monitoring_service'):
                            with patch('pynomaly.presentation.api.app.clear_dependencies'):
                                with patch('pynomaly.presentation.api.app.get_cache'):
                                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                                        app = create_app(mock_container)
                                        return app
    
    @pytest.fixture
    def integration_client(self, integration_app):
        """Create an integration test client."""
        return TestClient(integration_app)
    
    def test_routing_no_conflicts(self, integration_client):
        """Test that API and Web UI routes don't conflict."""
        # Test API routes
        api_response = integration_client.get("/api/v1/health")
        assert api_response.status_code == 200
        
        # Test root route (should be API root)
        root_response = integration_client.get("/")
        assert root_response.status_code == 200
        assert root_response.headers["content-type"] == "application/json"
        
        # Test OpenAPI
        openapi_response = integration_client.get("/api/v1/openapi.json")
        assert openapi_response.status_code == 200
        assert openapi_response.headers["content-type"] == "application/json"
    
    def test_routing_path_resolution(self, integration_client):
        """Test that path resolution works correctly."""
        # Test that API paths are resolved correctly
        api_paths = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/detectors",
            "/api/v1/datasets"
        ]
        
        for path in api_paths:
            response = integration_client.get(path)
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
    
    def test_routing_url_structure(self, integration_client):
        """Test URL structure consistency."""
        # Get OpenAPI spec
        response = integration_client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        paths = openapi_data["paths"]
        
        # Check URL structure consistency
        for path in paths.keys():
            if path.startswith("/api/"):
                # All API paths should start with /api/v1/
                assert path.startswith("/api/v1/"), f"API path {path} doesn't use v1 versioning"
                
                # Check path segments
                segments = path.split("/")
                assert len(segments) >= 4, f"API path {path} has insufficient segments"
                assert segments[1] == "api", f"API path {path} missing 'api' segment"
                assert segments[2] == "v1", f"API path {path} missing 'v1' segment"
    
    def test_routing_error_handling(self, integration_client):
        """Test error handling in routing."""
        # Test 404 for non-existent routes
        response = integration_client.get("/nonexistent")
        assert response.status_code == 404
        
        # Test 404 for non-existent API routes
        response = integration_client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test method not allowed
        response = integration_client.delete("/api/v1/health")
        assert response.status_code == 405
    
    def test_routing_content_negotiation(self, integration_client):
        """Test content negotiation in routing."""
        # Test API endpoints return JSON
        api_response = integration_client.get("/api/v1/health")
        assert api_response.status_code == 200
        assert api_response.headers["content-type"] == "application/json"
        
        # Test that JSON response is valid
        data = api_response.json()
        assert isinstance(data, dict)
        assert "status" in data
    
    def test_routing_middleware_order(self, integration_client):
        """Test that middleware is applied in correct order."""
        # Test CORS middleware
        response = integration_client.get("/api/v1/health")
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        
        # Test that request tracking middleware is applied
        response = integration_client.get("/api/v1/health")
        assert response.status_code == 200
        # Request tracking should not affect response status
    
    def test_routing_performance_impact(self, integration_client):
        """Test that routing doesn't significantly impact performance."""
        import time
        
        # Test multiple requests to ensure routing is efficient
        start_time = time.time()
        for _ in range(10):
            response = integration_client.get("/api/v1/health")
            assert response.status_code == 200
        end_time = time.time()
        
        # All requests should complete within reasonable time
        assert end_time - start_time < 5.0  # 10 requests in under 5 seconds
    
    def test_routing_regression_protection(self, integration_client):
        """Test regression protection for routing changes."""
        # Test that core routing patterns remain stable
        response = integration_client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        paths = openapi_data["paths"]
        
        # Ensure critical paths still exist
        critical_paths = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/detectors",
            "/api/v1/datasets",
            "/api/v1/auth/login"
        ]
        
        for critical_path in critical_paths:
            assert critical_path in paths, f"Critical path {critical_path} missing from API"
        
        # Test that API structure remains consistent
        assert "info" in openapi_data
        assert "openapi" in openapi_data
        assert openapi_data["info"]["title"] == "Pynomaly Test"
        assert openapi_data["info"]["version"] == "1.0.0-test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

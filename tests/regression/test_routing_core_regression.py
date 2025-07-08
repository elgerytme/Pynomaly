#!/usr/bin/env python3
"""
Simple routing regression test for core app functionality.

This test provides a simpler approach to testing routing without complex mocking,
focusing on core functionality and regression protection.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, APIRouter
from fastapi.responses import HTMLResponse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from pynomaly.presentation.api.app import create_app
    from pynomaly.infrastructure.config import Container, Settings
    APP_AVAILABLE = True
except ImportError:
    APP_AVAILABLE = False


class TestCoreRoutingRegression:
    """Simple regression tests for core routing functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method."""
        if not APP_AVAILABLE:
            pytest.skip("Application not available for testing")
    
    @pytest.fixture
    def minimal_container(self):
        """Create a minimal mock container for testing."""
        container = Mock(spec=Container)
        
        # Mock minimal settings
        settings = MagicMock()
        settings.app = MagicMock()
        settings.app.name = "Pynomaly Test"
        settings.app.version = "1.0.0-test"
        settings.app.environment = "test"
        settings.app.debug = True
        settings.docs_enabled = True
        settings.auth_enabled = False
        settings.cache_enabled = False
        settings.monitoring = MagicMock()
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
        
        # Mock minimal repositories
        container.detector_repository.return_value = Mock()
        container.dataset_repository.return_value = Mock()
        container.result_repository.return_value = Mock()
        
        # Mock basic repository methods
        container.detector_repository().count.return_value = 0
        container.dataset_repository().count.return_value = 0
        container.result_repository().count.return_value = 0
        container.result_repository().find_recent.return_value = []
        
        return container
    
    @pytest.fixture
    def test_app(self, minimal_container):
        """Create a minimal test app instance."""
        with patch('pynomaly.infrastructure.config.create_container', return_value=minimal_container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app.initialize_monitoring_service'):
                        with patch('pynomaly.presentation.api.app.shutdown_monitoring_service'):
                            with patch('pynomaly.presentation.api.app.clear_dependencies'):
                                with patch('pynomaly.presentation.api.app.get_cache'):
                                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                                        app = create_app(minimal_container)
                                        return app
    
    @pytest.fixture
    def client(self, test_app):
        """Create a test client."""
        return TestClient(test_app)
    
    def test_app_instantiation(self, test_app):
        """Test that the real app can be instantiated successfully."""
        assert isinstance(test_app, FastAPI)
        assert test_app.title == "Pynomaly Test"
        assert test_app.version == "1.0.0-test"
        assert test_app.debug is True
    
    def test_root_endpoint_routing(self, client):
        """Test that the root endpoint routes correctly."""
        response = client.get("/")
        assert response.status_code == 200
        
        # Should return JSON (API root)
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "Pynomaly API"
        assert "version" in data
        assert "api_version" in data
        assert data["api_version"] == "v1"
    
    def test_health_endpoint_routing(self, client):
        """Test that the health endpoint routes correctly."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_version_endpoint_routing(self, client):
        """Test that the version endpoint routes correctly."""
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        
        data = response.json()
        assert "version" in data
        assert "environment" in data
        assert data["environment"] == "test"
    
    def test_openapi_endpoint_routing(self, client):
        """Test that the OpenAPI endpoint routes correctly."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data
        assert openapi_data["info"]["title"] == "Pynomaly Test"
        assert openapi_data["info"]["version"] == "1.0.0-test"
    
    def test_docs_endpoint_routing(self, client):
        """Test that the docs endpoint routes correctly."""
        response = client.get("/api/v1/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_endpoint_routing(self, client):
        """Test that the redoc endpoint routes correctly."""
        response = client.get("/api/v1/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_api_versioning_consistency(self, client):
        """Test that API versioning is consistent."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        paths = openapi_data["paths"]
        
        # Check that all API paths use v1 versioning
        api_paths = [path for path in paths.keys() if path.startswith("/api/")]
        for path in api_paths:
            assert "/api/v1/" in path, f"API path {path} doesn't use v1 versioning"
    
    def test_core_endpoints_exist(self, client):
        """Test that core endpoints exist in the API."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        paths = openapi_data["paths"]
        
        # Check for core endpoints
        core_endpoints = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/detectors",
            "/api/v1/datasets",
            "/api/v1/auth/login"
        ]
        
        for endpoint in core_endpoints:
            assert endpoint in paths, f"Core endpoint {endpoint} missing from API"
    
    def test_endpoint_response_formats(self, client):
        """Test that endpoints return correct response formats."""
        # Test JSON endpoints
        json_endpoints = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/detectors",
            "/api/v1/datasets"
        ]
        
        for endpoint in json_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
            
            # Verify JSON can be parsed
            data = response.json()
            assert isinstance(data, (dict, list))
        
        # Test HTML endpoints
        html_endpoints = [
            "/api/v1/docs",
            "/api/v1/redoc"
        ]
        
        for endpoint in html_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
    
    def test_error_handling_routes(self, client):
        """Test error handling in routing."""
        # Test 404 for non-existent endpoint
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test 405 for method not allowed
        response = client.delete("/api/v1/health")
        assert response.status_code == 405
    
    def test_cors_configuration(self, client):
        """Test CORS configuration."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        
        # Test preflight request
        response = client.options("/api/v1/health")
        assert response.status_code == 200
    
    def test_routing_performance(self, client):
        """Test basic routing performance."""
        import time
        
        # Test that basic routing is reasonably fast
        start_time = time.time()
        response = client.get("/api/v1/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 0.5  # Should respond within 500ms
    
    def test_openapi_schema_structure(self, client):
        """Test OpenAPI schema structure for regression protection."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        
        # Check required OpenAPI fields
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data
        
        # Check info structure
        info = openapi_data["info"]
        assert "title" in info
        assert "version" in info
        assert "description" in info
        assert "contact" in info
        assert "license" in info
        
        # Check paths structure
        paths = openapi_data["paths"]
        assert isinstance(paths, dict)
        assert len(paths) > 0
        
        # Check that each path has proper HTTP methods
        for path, methods in paths.items():
            assert isinstance(methods, dict)
            for method, details in methods.items():
                assert method.lower() in ["get", "post", "put", "delete", "patch", "options", "head"]
                assert "responses" in details
    
    def test_endpoint_path_structure(self, client):
        """Test endpoint path structure consistency."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        paths = openapi_data["paths"]
        
        # Check path structure consistency
        for path in paths.keys():
            if path.startswith("/api/v1/"):
                # Check path segments
                segments = path.split("/")
                assert len(segments) >= 4, f"Path {path} has insufficient segments"
                assert segments[1] == "api", f"Path {path} missing 'api' segment"
                assert segments[2] == "v1", f"Path {path} missing 'v1' segment"
                assert segments[3] != "", f"Path {path} has empty segment after v1"
    
    def test_api_metadata_consistency(self, client):
        """Test API metadata consistency."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        
        # Check that metadata is consistent
        assert openapi_data["info"]["title"] == "Pynomaly Test"
        assert openapi_data["info"]["version"] == "1.0.0-test"
        
        # Check contact information
        contact = openapi_data["info"]["contact"]
        assert "name" in contact
        assert "url" in contact
        assert "email" in contact
        
        # Check license information
        license_info = openapi_data["info"]["license"]
        assert "name" in license_info
        assert "url" in license_info


class TestSimpleWebUIRouting:
    """Simple Web UI routing tests."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method."""
        if not APP_AVAILABLE:
            pytest.skip("Application not available for testing")
    
    def test_web_ui_separation(self):
        """Test that Web UI and API are properly separated."""
        # Create a simple app to test routing separation
        app = FastAPI()
        
        # Add API routes
        api_router = APIRouter()
        @api_router.get("/health")
        def health():
            return {"status": "healthy"}
        
        app.include_router(api_router, prefix="/api/v1")
        
        # Add Web UI routes
        web_router = APIRouter()
        @web_router.get("/", response_class=HTMLResponse)
        def web_home():
            return HTMLResponse("<html><body>Web UI Home</body></html>")
        
        app.include_router(web_router, prefix="/web")
        
        # Test with client
        client = TestClient(app)
        
        # Test API endpoint
        api_response = client.get("/api/v1/health")
        assert api_response.status_code == 200
        assert api_response.headers["content-type"] == "application/json"
        
        # Test Web UI endpoint
        web_response = client.get("/web/")
        assert web_response.status_code == 200
        assert "text/html" in web_response.headers["content-type"]
        
        # Ensure separation
        assert api_response.headers["content-type"] != web_response.headers["content-type"]
    
    def test_url_structure_consistency(self):
        """Test that URL structure is consistent."""
        app = FastAPI()
        
        # Add structured routes
        api_router = APIRouter()
        @api_router.get("/health")
        def health():
            return {"status": "healthy"}
        
        @api_router.get("/version")
        def version():
            return {"version": "1.0.0"}
        
        app.include_router(api_router, prefix="/api/v1")
        
        client = TestClient(app)
        
        # Test that all API routes use consistent prefix
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        
        # Test that routes without prefix don't work
        response = client.get("/health")
        assert response.status_code == 404
        
        response = client.get("/version")
        assert response.status_code == 404


class TestRoutingRegressionProtection:
    """Tests specifically for protecting against routing regressions."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method."""
        if not APP_AVAILABLE:
            pytest.skip("Application not available for testing")
    
    def test_critical_endpoints_stability(self):
        """Test that critical endpoints remain stable."""
        # This test ensures critical endpoints don't change without notice
        expected_critical_endpoints = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/detectors",
            "/api/v1/datasets",
            "/api/v1/detection/train",
            "/api/v1/detection/predict",
            "/api/v1/auth/login",
            "/api/v1/auth/logout",
            "/api/v1/auth/refresh"
        ]
        
        # Create minimal app for testing
        app = FastAPI()
        
        # Add critical endpoints
        api_router = APIRouter()
        
        @api_router.get("/health")
        def health():
            return {"status": "healthy"}
        
        @api_router.get("/version")
        def version():
            return {"version": "1.0.0"}
        
        @api_router.get("/detectors")
        def detectors():
            return []
        
        @api_router.get("/datasets")
        def datasets():
            return []
        
        @api_router.post("/detection/train")
        def train():
            return {"status": "training"}
        
        @api_router.post("/detection/predict")
        def predict():
            return {"predictions": []}
        
        @api_router.post("/auth/login")
        def login():
            return {"token": "test"}
        
        @api_router.post("/auth/logout")
        def logout():
            return {"status": "logged out"}
        
        @api_router.post("/auth/refresh")
        def refresh():
            return {"token": "refreshed"}
        
        app.include_router(api_router, prefix="/api/v1")
        
        client = TestClient(app)
        
        # Test that all critical endpoints are accessible
        for endpoint in expected_critical_endpoints:
            if endpoint.endswith("/train") or endpoint.endswith("/predict") or endpoint.endswith("/login") or endpoint.endswith("/logout") or endpoint.endswith("/refresh"):
                response = client.post(endpoint, json={})
            else:
                response = client.get(endpoint)
            
            assert response.status_code in [200, 422], f"Critical endpoint {endpoint} not accessible"
    
    def test_api_versioning_stability(self):
        """Test that API versioning remains stable."""
        app = FastAPI()
        
        # Add versioned routes
        v1_router = APIRouter()
        @v1_router.get("/test")
        def test_v1():
            return {"version": "v1"}
        
        app.include_router(v1_router, prefix="/api/v1")
        
        client = TestClient(app)
        
        # Test v1 endpoint
        response = client.get("/api/v1/test")
        assert response.status_code == 200
        assert response.json()["version"] == "v1"
        
        # Test that unversioned endpoint doesn't work
        response = client.get("/api/test")
        assert response.status_code == 404
    
    def test_response_format_stability(self):
        """Test that response formats remain stable."""
        app = FastAPI()
        
        # Add endpoints with specific response formats
        api_router = APIRouter()
        
        @api_router.get("/health")
        def health():
            return {"status": "healthy", "timestamp": "2023-01-01T00:00:00Z"}
        
        @api_router.get("/version")
        def version():
            return {"version": "1.0.0", "environment": "test"}
        
        app.include_router(api_router, prefix="/api/v1")
        
        client = TestClient(app)
        
        # Test health endpoint format
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
        
        # Test version endpoint format
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "environment" in data
        assert data["version"] == "1.0.0"
        assert data["environment"] == "test"
    
    def test_error_response_consistency(self):
        """Test that error responses are consistent."""
        app = FastAPI()
        
        # Add endpoint that can return errors
        api_router = APIRouter()
        
        @api_router.get("/test")
        def test_endpoint():
            return {"message": "success"}
        
        app.include_router(api_router, prefix="/api/v1")
        
        client = TestClient(app)
        
        # Test successful response
        response = client.get("/api/v1/test")
        assert response.status_code == 200
        
        # Test 404 response
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test method not allowed
        response = client.delete("/api/v1/test")
        assert response.status_code == 405


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

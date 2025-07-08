#!/usr/bin/env python3
"""
Integration test for real app routing functionality.

This test instantiates the actual Pynomaly app and tests routing behavior
to ensure everything works correctly end-to-end.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
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


class TestAppRoutingIntegration:
    """Integration tests for app routing with real app instance."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method."""
        if not APP_AVAILABLE:
            pytest.skip("Application not available for testing")
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies to allow app creation."""
        with patch('pynomaly.presentation.api.app.init_cache') as mock_init_cache:
            with patch('pynomaly.presentation.api.app.init_auth') as mock_init_auth:
                with patch('pynomaly.presentation.api.app.initialize_monitoring_service') as mock_init_monitoring:
                    mock_init_monitoring.return_value = AsyncMock()
                    with patch('pynomaly.presentation.api.app.shutdown_monitoring_service') as mock_shutdown_monitoring:
                        mock_shutdown_monitoring.return_value = AsyncMock()
                        with patch('pynomaly.presentation.api.app.clear_dependencies') as mock_clear_deps:
                            with patch('pynomaly.presentation.api.app.get_cache') as mock_get_cache:
                                mock_get_cache.return_value = None
                                with patch('pynomaly.presentation.api.app._mount_web_ui_lazy') as mock_mount_web:
                                    mock_mount_web.return_value = True
                                    yield {
                                        'init_cache': mock_init_cache,
                                        'init_auth': mock_init_auth,
                                        'init_monitoring': mock_init_monitoring,
                                        'shutdown_monitoring': mock_shutdown_monitoring,
                                        'clear_deps': mock_clear_deps,
                                        'get_cache': mock_get_cache,
                                        'mount_web': mock_mount_web,
                                    }
    
    @pytest.fixture
    def test_container(self):
        """Create a test container with minimal configuration."""
        container = Mock(spec=Container)
        
        # Mock settings
        settings = Mock(spec=Settings)
        settings.app.name = "Pynomaly Integration Test"
        settings.app.version = "1.0.0-integration"
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
        
        # Mock repositories with basic functionality
        detector_repo = Mock()
        detector_repo.count.return_value = 0
        detector_repo.find_all.return_value = []
        container.detector_repository.return_value = detector_repo
        
        dataset_repo = Mock()
        dataset_repo.count.return_value = 0
        dataset_repo.find_all.return_value = []
        container.dataset_repository.return_value = dataset_repo
        
        result_repo = Mock()
        result_repo.count.return_value = 0
        result_repo.find_recent.return_value = []
        container.result_repository.return_value = result_repo
        
        return container
    
    @pytest.fixture
    def app(self, test_container, mock_dependencies):
        """Create the actual app instance."""
        return create_app(test_container)
    
    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return TestClient(app)
    
    def test_app_creation_success(self, app):
        """Test that the app can be created successfully."""
        assert isinstance(app, FastAPI)
        assert app.title == "Pynomaly Integration Test"
        assert app.version == "1.0.0-integration"
    
    def test_root_route_accessible(self, client):
        """Test that the root route is accessible."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "message" in data
        assert data["message"] == "Pynomaly API"
    
    def test_health_endpoint_functional(self, client):
        """Test that the health endpoint is functional."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_version_endpoint_functional(self, client):
        """Test that the version endpoint is functional."""
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert "version" in data
        assert "environment" in data
        assert data["environment"] == "test"
    
    def test_openapi_schema_generation(self, client):
        """Test that OpenAPI schema is generated correctly."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check basic schema structure
        assert schema["info"]["title"] == "Pynomaly Integration Test"
        assert schema["info"]["version"] == "1.0.0-integration"
        
        # Check that paths are present
        paths = schema["paths"]
        assert len(paths) > 0
        
        # Check that critical paths exist
        critical_paths = ["/api/v1/health", "/api/v1/version"]
        for path in critical_paths:
            assert path in paths, f"Critical path {path} missing from schema"
    
    def test_docs_interface_accessible(self, client):
        """Test that the docs interface is accessible."""
        response = client.get("/api/v1/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Basic check for Swagger UI content
        content = response.text
        assert "swagger" in content.lower() or "openapi" in content.lower()
    
    def test_redoc_interface_accessible(self, client):
        """Test that the ReDoc interface is accessible."""
        response = client.get("/api/v1/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Basic check for ReDoc content
        content = response.text
        assert "redoc" in content.lower() or "api" in content.lower()
    
    def test_api_route_structure(self, client):
        """Test that API routes follow the correct structure."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        paths = schema["paths"]
        
        # Check that all API paths start with /api/v1/
        for path in paths:
            if path.startswith("/api/"):
                assert path.startswith("/api/v1/"), f"Path {path} doesn't follow /api/v1/ structure"
    
    def test_detector_endpoints_available(self, client):
        """Test that detector endpoints are available."""
        # Test list detectors
        response = client.get("/api/v1/detectors")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_dataset_endpoints_available(self, client):
        """Test that dataset endpoints are available."""
        # Test list datasets
        response = client.get("/api/v1/datasets")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_auth_endpoints_structure(self, client):
        """Test that auth endpoints have correct structure."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        paths = schema["paths"]
        
        # Check for auth endpoints
        auth_endpoints = [
            "/api/v1/auth/login",
            "/api/v1/auth/logout",
            "/api/v1/auth/refresh"
        ]
        
        for endpoint in auth_endpoints:
            assert endpoint in paths, f"Auth endpoint {endpoint} missing from API"
    
    def test_error_handling_routes(self, client):
        """Test error handling in routes."""
        # Test 404 for non-existent endpoint
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        # Test 405 for method not allowed
        response = client.delete("/api/v1/health")
        assert response.status_code == 405
    
    def test_cors_middleware_active(self, client):
        """Test that CORS middleware is active."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        
        # Test OPTIONS request
        response = client.options("/api/v1/health")
        assert response.status_code == 200
    
    def test_request_tracking_middleware(self, client):
        """Test that request tracking middleware is working."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        # Request tracking should not interfere with normal responses
        assert response.headers["content-type"] == "application/json"
    
    def test_multiple_requests_stability(self, client):
        """Test that multiple requests work stably."""
        # Make multiple requests to ensure stability
        for i in range(5):
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
    
    def test_concurrent_requests_handling(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            try:
                response = client.get("/api/v1/health")
                results.append(response.status_code)
            except Exception as e:
                results.append(f"Error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all requests succeeded
        assert len(results) == 5
        for result in results:
            assert result == 200, f"Request failed with result: {result}"
    
    def test_app_metadata_consistency(self, client):
        """Test that app metadata is consistent across endpoints."""
        # Check root endpoint
        response = client.get("/")
        assert response.status_code == 200
        root_data = response.json()
        
        # Check version endpoint
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        version_data = response.json()
        
        # Check OpenAPI schema
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        
        # Verify consistency
        assert root_data["api_version"] == "v1"
        assert version_data["environment"] == "test"
        assert schema["info"]["title"] == "Pynomaly Integration Test"
    
    def test_routing_performance_acceptable(self, client):
        """Test that routing performance is acceptable."""
        import time
        
        # Test response time for health check
        start_time = time.time()
        response = client.get("/api/v1/health")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 1.0  # Should respond within 1 second
        
        # Test response time for OpenAPI schema
        start_time = time.time()
        response = client.get("/api/v1/openapi.json")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 2.0  # Should respond within 2 seconds
    
    def test_endpoint_method_handling(self, client):
        """Test that endpoints handle HTTP methods correctly."""
        # Test GET methods
        get_endpoints = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/detectors",
            "/api/v1/datasets"
        ]
        
        for endpoint in get_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
        
        # Test that POST to GET-only endpoints returns 405
        for endpoint in ["/api/v1/health", "/api/v1/version"]:
            response = client.post(endpoint)
            assert response.status_code == 405
    
    def test_route_registration_completeness(self, client):
        """Test that all expected routes are registered."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        paths = schema["paths"]
        
        # Check for major route groups
        route_groups = [
            "health",
            "version",
            "detectors",
            "datasets",
            "auth",
            "detection",
            "automl",
            "ensemble",
            "explainability"
        ]
        
        for group in route_groups:
            found = any(group in path for path in paths.keys())
            assert found, f"No routes found for group: {group}"
    
    def test_middleware_chain_integrity(self, client):
        """Test that middleware chain is properly configured."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        
        # Check content type
        assert response.headers["content-type"] == "application/json"
        
        # Check response body
        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data
    
    def test_app_lifecycle_handling(self, app):
        """Test that app handles lifecycle events correctly."""
        # Test that app can be created and has proper state
        assert hasattr(app, 'state')
        assert hasattr(app.state, 'container')
        
        # Test that app has proper router configuration
        assert len(app.routes) > 0
        
        # Test that app has proper middleware configuration
        assert len(app.middleware_stack) > 0


class TestRoutingRegressionIntegration:
    """Integration tests specifically for regression protection."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup for each test method."""
        if not APP_AVAILABLE:
            pytest.skip("Application not available for testing")
    
    def test_api_structure_stability(self):
        """Test that API structure remains stable."""
        # This test ensures the API structure doesn't change unexpectedly
        
        # Mock dependencies for app creation
        with patch('pynomaly.presentation.api.app.init_cache'):
            with patch('pynomaly.presentation.api.app.init_auth'):
                with patch('pynomaly.presentation.api.app.initialize_monitoring_service'):
                    with patch('pynomaly.presentation.api.app.shutdown_monitoring_service'):
                        with patch('pynomaly.presentation.api.app.clear_dependencies'):
                            with patch('pynomaly.presentation.api.app.get_cache'):
                                with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                                    
                                    # Create minimal container
                                    container = Mock(spec=Container)
                                    settings = Mock(spec=Settings)
                                    settings.app.name = "Pynomaly Regression Test"
                                    settings.app.version = "1.0.0-regression"
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
                                    container.detector_repository().count.return_value = 0
                                    container.dataset_repository().count.return_value = 0
                                    container.result_repository().count.return_value = 0
                                    container.result_repository().find_recent.return_value = []
                                    
                                    # Create app
                                    app = create_app(container)
                                    client = TestClient(app)
                                    
                                    # Test API structure
                                    response = client.get("/api/v1/openapi.json")
                                    assert response.status_code == 200
                                    
                                    schema = response.json()
                                    paths = schema["paths"]
                                    
                                    # Check that critical paths exist
                                    critical_paths = [
                                        "/api/v1/health",
                                        "/api/v1/version",
                                        "/api/v1/detectors",
                                        "/api/v1/datasets"
                                    ]
                                    
                                    for path in critical_paths:
                                        assert path in paths, f"Critical path {path} missing from API"
    
    def test_response_format_stability(self):
        """Test that response formats remain stable."""
        # This test ensures response formats don't change unexpectedly
        
        # Mock dependencies for app creation
        with patch('pynomaly.presentation.api.app.init_cache'):
            with patch('pynomaly.presentation.api.app.init_auth'):
                with patch('pynomaly.presentation.api.app.initialize_monitoring_service'):
                    with patch('pynomaly.presentation.api.app.shutdown_monitoring_service'):
                        with patch('pynomaly.presentation.api.app.clear_dependencies'):
                            with patch('pynomaly.presentation.api.app.get_cache'):
                                with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                                    
                                    # Create minimal container
                                    container = Mock(spec=Container)
                                    settings = Mock(spec=Settings)
                                    settings.app.name = "Pynomaly Format Test"
                                    settings.app.version = "1.0.0-format"
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
                                    container.detector_repository().count.return_value = 0
                                    container.dataset_repository().count.return_value = 0
                                    container.result_repository().count.return_value = 0
                                    container.result_repository().find_recent.return_value = []
                                    
                                    # Create app
                                    app = create_app(container)
                                    client = TestClient(app)
                                    
                                    # Test health endpoint format
                                    response = client.get("/api/v1/health")
                                    assert response.status_code == 200
                                    assert response.headers["content-type"] == "application/json"
                                    
                                    data = response.json()
                                    assert "status" in data
                                    assert data["status"] == "healthy"
                                    
                                    # Test version endpoint format
                                    response = client.get("/api/v1/version")
                                    assert response.status_code == 200
                                    assert response.headers["content-type"] == "application/json"
                                    
                                    data = response.json()
                                    assert "version" in data
                                    assert "environment" in data
                                    assert data["environment"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

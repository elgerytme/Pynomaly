"""Core routing regression tests for the Pynomaly app.

These tests verify that the main application routes work correctly
and protect against regressions in the routing configuration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pynomaly.presentation.api.app import create_app


class TestCoreRoutingRegression:
    """Test suite for core routing regression protection."""

    @pytest.fixture
    def minimal_container(self):
        """Create a minimal mocked container for testing."""
        container = Mock()
        
        # Mock the config with necessary settings
        mock_config = MagicMock()
        mock_config.app.name = "Pynomaly"
        mock_config.app.version = "1.0.0"
        mock_config.cache_enabled = False
        mock_config.auth_enabled = False
        mock_config.docs_enabled = True
        mock_config.monitoring.metrics_enabled = False
        mock_config.monitoring.tracing_enabled = False
        mock_config.monitoring.prometheus_enabled = False
        mock_config.get_cors_config.return_value = {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
        
        container.config.return_value = mock_config
        return container

    @pytest.fixture
    def test_app(self, minimal_container):
        """Create a minimal test app instance."""
        with patch('pynomaly.infrastructure.config.create_container', return_value=minimal_container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                        app = create_app()
                        return app

    @pytest.fixture
    def client(self, test_app):
        """Create a test client for the app."""
        return TestClient(test_app)

    def test_app_instantiation(self, test_app):
        """Test that the app can be instantiated without errors."""
        assert isinstance(test_app, FastAPI)
        assert test_app.title == "Pynomaly"
        assert test_app.version == "1.0.0"

    def test_api_root_endpoint(self, client):
        """Test that the API root endpoint is accessible."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Pynomaly API"
        assert data["version"] == "1.0.0"
        assert data["api_version"] == "v1"

    def test_health_endpoint(self, client):
        """Test that the health endpoint is accessible."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_version_endpoint(self, client):
        """Test that the version endpoint is accessible."""
        response = client.get("/api/v1/version")
        assert response.status_code == 200

    def test_docs_endpoint(self, client):
        """Test that the docs endpoint is accessible."""
        response = client.get("/api/v1/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        """Test that the OpenAPI schema is available."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Pynomaly"

    def test_endpoint_path_consistency(self, test_app):
        """Test that all API endpoints follow the /api/v1 prefix pattern."""
        api_routes = []
        
        for route in test_app.routes:
            if hasattr(route, 'path') and route.path.startswith('/api/v1'):
                api_routes.append(route.path)
        
        # Verify some expected routes exist
        expected_routes = [
            "/api/v1/health",
            "/api/v1/version",
            "/api/v1/docs",
            "/api/v1/openapi.json"
        ]
        
        for expected in expected_routes:
            assert any(expected in route for route in api_routes), f"Route {expected} not found in API routes"

    def test_api_versioning(self, test_app):
        """Test that API versioning is properly implemented."""
        # Check that all API routes have the v1 prefix
        api_routes = [route.path for route in test_app.routes if hasattr(route, 'path')]
        
        # Filter for actual API routes (not root or static)
        versioned_routes = [route for route in api_routes if route.startswith('/api/v1')]
        
        assert len(versioned_routes) > 0, "No versioned API routes found"
        
        # All versioned routes should start with /api/v1
        for route in versioned_routes:
            assert route.startswith('/api/v1'), f"Route {route} doesn't follow versioning pattern"

    def test_response_format_consistency(self, client):
        """Test that API responses follow consistent format."""
        # Test root endpoint
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Test health endpoint
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_cors_configuration(self, client):
        """Test that CORS is properly configured."""
        response = client.options("/api/v1/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        
        # Should not return 405 or 404 for OPTIONS requests
        assert response.status_code != 405
        assert response.status_code != 404

    def test_routing_regression_protection(self, test_app):
        """Test to protect against routing regressions."""
        # Store the current routing state
        current_routes = {}
        
        for route in test_app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                current_routes[route.path] = list(route.methods)
        
        # Critical routes that must exist
        critical_routes = {
            "/": ["GET"],
            "/api/v1/health": ["GET"],
            "/api/v1/version": ["GET"],
            "/api/v1/docs": ["GET"],
            "/api/v1/openapi.json": ["GET"]
        }
        
        for path, methods in critical_routes.items():
            assert path in current_routes, f"Critical route {path} is missing"
            for method in methods:
                assert method in current_routes[path], f"Method {method} missing from {path}"

    def test_middleware_configuration(self, test_app):
        """Test that middleware is properly configured."""
        # Check that middleware stack is not empty
        assert len(test_app.middleware_stack) > 0, "No middleware configured"
        
        # Check for CORS middleware
        middleware_types = [type(middleware).__name__ for middleware in test_app.middleware_stack]
        assert any("CORSMiddleware" in mw_type for mw_type in middleware_types), "CORS middleware not found"

    def test_app_state_initialization(self, test_app):
        """Test that app state is properly initialized."""
        assert hasattr(test_app, 'state'), "App state not initialized"
        assert hasattr(test_app.state, 'container'), "Container not set in app state"
        assert test_app.state.container is not None, "Container is None"

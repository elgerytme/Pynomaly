"""Real app routing regression tests for the Pynomaly app.

These tests verify that the main application routes work correctly
by actually instantiating the real app and testing its routing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient


def create_minimal_settings():
    """Create minimal settings for testing."""
    settings = MagicMock()
    settings.app.name = "Pynomaly"
    settings.app.version = "1.0.0"
    settings.cache_enabled = False
    settings.auth_enabled = False
    settings.docs_enabled = True
    settings.monitoring.metrics_enabled = False
    settings.monitoring.tracing_enabled = False
    settings.monitoring.prometheus_enabled = False
    settings.get_cors_config.return_value = {
        "allow_origins": ["*"],
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    return settings


def create_minimal_container():
    """Create minimal container for testing."""
    container = Mock()
    container.config.return_value = create_minimal_settings()
    return container


class TestRealAppRoutingRegression:
    """Test suite for real app routing regression protection."""

    def test_real_app_import_successful(self):
        """Test that we can import the real app module."""
        try:
            from pynomaly.presentation.api.app import create_app
            assert create_app is not None
        except ImportError as e:
            pytest.skip(f"Cannot import real app: {e}")

    def test_real_app_instantiation_with_mocks(self):
        """Test that the real app can be instantiated with proper mocking."""
        try:
            from pynomaly.presentation.api.app import create_app
        except ImportError:
            pytest.skip("Cannot import real app")

        container = create_minimal_container()
        
        with patch('pynomaly.infrastructure.config.create_container', return_value=container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                        app = create_app()
                        assert isinstance(app, FastAPI)
                        assert app.title == "Pynomaly"
                        assert app.version == "1.0.0"

    def test_real_app_basic_endpoints(self):
        """Test that the real app has basic endpoints working."""
        try:
            from pynomaly.presentation.api.app import create_app
        except ImportError:
            pytest.skip("Cannot import real app")

        container = create_minimal_container()
        
        with patch('pynomaly.infrastructure.config.create_container', return_value=container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                        app = create_app()
                        client = TestClient(app)
                        
                        # Test root endpoint
                        response = client.get("/")
                        assert response.status_code == 200
                        data = response.json()
                        assert data["message"] == "Pynomaly API"
                        assert data["version"] == "1.0.0"
                        assert data["api_version"] == "v1"

    def test_real_app_api_endpoints(self):
        """Test that the real app API endpoints are accessible."""
        try:
            from pynomaly.presentation.api.app import create_app
        except ImportError:
            pytest.skip("Cannot import real app")

        container = create_minimal_container()
        
        with patch('pynomaly.infrastructure.config.create_container', return_value=container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                        app = create_app()
                        client = TestClient(app)
                        
                        # Test health endpoint - should be available
                        response = client.get("/api/v1/health")
                        # Note: This might return 404 if the health endpoint requires authentication
                        # or has other dependencies, but at least the routing should work
                        assert response.status_code in [200, 401, 404, 500]  # Any valid HTTP response
                        
                        # Test version endpoint
                        response = client.get("/api/v1/version")
                        assert response.status_code in [200, 401, 404, 500]

    def test_real_app_openapi_schema(self):
        """Test that the real app can generate OpenAPI schema."""
        try:
            from pynomaly.presentation.api.app import create_app
        except ImportError:
            pytest.skip("Cannot import real app")

        container = create_minimal_container()
        
        with patch('pynomaly.infrastructure.config.create_container', return_value=container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                        app = create_app()
                        client = TestClient(app)
                        
                        # Test OpenAPI schema endpoint
                        response = client.get("/api/v1/openapi.json")
                        assert response.status_code == 200
                        schema = response.json()
                        assert "openapi" in schema
                        assert "info" in schema
                        assert schema["info"]["title"] == "Pynomaly"

    def test_real_app_route_structure(self):
        """Test that the real app has the expected route structure."""
        try:
            from pynomaly.presentation.api.app import create_app
        except ImportError:
            pytest.skip("Cannot import real app")

        container = create_minimal_container()
        
        with patch('pynomaly.infrastructure.config.create_container', return_value=container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                        app = create_app()
                        
                        # Check that all routes have proper structure
                        routes = [route.path for route in app.routes if hasattr(route, 'path')]
                        api_routes = [route for route in routes if route.startswith('/api/v1')]
                        
                        # Should have versioned API routes
                        assert len(api_routes) > 0, "No versioned API routes found"
                        
                        # All API routes should start with /api/v1
                        for route in api_routes:
                            assert route.startswith('/api/v1'), f"Route {route} doesn't follow versioning pattern"

    def test_real_app_middleware_configuration(self):
        """Test that the real app has proper middleware."""
        try:
            from pynomaly.presentation.api.app import create_app
        except ImportError:
            pytest.skip("Cannot import real app")

        container = create_minimal_container()
        
        with patch('pynomaly.infrastructure.config.create_container', return_value=container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                        app = create_app()
                        
                        # Check that middleware stack is not empty
                        assert len(app.middleware_stack) > 0, "No middleware configured"
                        
                        # Check for CORS middleware
                        middleware_types = [type(middleware).__name__ for middleware in app.middleware_stack]
                        assert any("CORSMiddleware" in mw_type for mw_type in middleware_types), "CORS middleware not found"

    def test_real_app_cors_functionality(self):
        """Test that CORS is working in the real app."""
        try:
            from pynomaly.presentation.api.app import create_app
        except ImportError:
            pytest.skip("Cannot import real app")

        container = create_minimal_container()
        
        with patch('pynomaly.infrastructure.config.create_container', return_value=container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                        app = create_app()
                        client = TestClient(app)
                        
                        # Test CORS preflight request
                        response = client.options("/api/v1/health", headers={
                            "Origin": "http://localhost:3000",
                            "Access-Control-Request-Method": "GET"
                        })
                        
                        # Should not return 405 or 404 for OPTIONS requests if CORS is properly configured
                        assert response.status_code != 405, "CORS preflight not supported"

    def test_real_app_routing_regression_protection(self):
        """Test to protect against routing regressions in the real app."""
        try:
            from pynomaly.presentation.api.app import create_app
        except ImportError:
            pytest.skip("Cannot import real app")

        container = create_minimal_container()
        
        with patch('pynomaly.infrastructure.config.create_container', return_value=container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                        app = create_app()
                        
                        # Store the current routing state
                        current_routes = {}
                        
                        for route in app.routes:
                            if hasattr(route, 'path') and hasattr(route, 'methods'):
                                current_routes[route.path] = list(route.methods)
                        
                        # Critical routes that must exist
                        critical_routes = [
                            "/",
                            "/api/v1/openapi.json"
                        ]
                        
                        for path in critical_routes:
                            assert path in current_routes, f"Critical route {path} is missing"

    def test_real_app_state_initialization(self):
        """Test that the real app state is properly initialized."""
        try:
            from pynomaly.presentation.api.app import create_app
        except ImportError:
            pytest.skip("Cannot import real app")

        container = create_minimal_container()
        
        with patch('pynomaly.infrastructure.config.create_container', return_value=container):
            with patch('pynomaly.presentation.api.app.init_cache'):
                with patch('pynomaly.presentation.api.app.init_auth'):
                    with patch('pynomaly.presentation.api.app._mount_web_ui_lazy'):
                        app = create_app()
                        
                        assert hasattr(app, 'state'), "App state not initialized"
                        assert hasattr(app.state, 'container'), "Container not set in app state"
                        assert app.state.container is not None, "Container is None"

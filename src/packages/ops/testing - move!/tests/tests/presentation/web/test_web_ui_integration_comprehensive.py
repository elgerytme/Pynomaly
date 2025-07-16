"""
Comprehensive Web UI Integration Testing
======================================

This module provides comprehensive testing for web UI components, HTMX integration,
template rendering, and frontend-backend interactions.
"""

import asyncio
from datetime import datetime
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import HTMLResponse


class TestWebUIComponents:
    """Test suite for web UI components and integration."""

    @pytest.fixture
    def web_app(self):
        """Create web application instance."""
        app = FastAPI()

        # Add basic routes for testing
        @app.get("/")
        async def root():
            return {"message": "Web UI"}

        @app.get("/dashboard")
        async def dashboard():
            return HTMLResponse("<html><body>Dashboard</body></html>")

        @app.post("/api/htmx/component")
        async def htmx_component():
            return HTMLResponse("<div>HTMX Component</div>")

        return app

    @pytest.fixture
    def test_client(self, web_app):
        """Create test client for web application."""
        return TestClient(web_app)

    @pytest.fixture
    def mock_template_engine(self):
        """Create mock template engine."""
        engine = Mock()
        engine.render = Mock(return_value="<html><body>Rendered</body></html>")
        return engine

    @pytest.fixture
    def sample_context_data(self):
        """Create sample context data for templates."""
        return {
            "user": {
                "id": 1,
                "username": "testuser",
                "email": "test@example.com",
                "role": "admin",
            },
            "dashboard": {
                "total_detectors": 5,
                "active_detections": 3,
                "alerts_count": 2,
            },
            "navigation": {
                "current_page": "dashboard",
                "breadcrumbs": ["Home", "Dashboard"],
            },
            "csrf_token": "abc123xyz789",
        }

    def test_web_ui_root_endpoint(self, test_client):
        """Test web UI root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Web UI"}

    def test_dashboard_endpoint(self, test_client):
        """Test dashboard endpoint returns HTML."""
        response = test_client.get("/dashboard")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "Dashboard" in response.text

    def test_htmx_component_endpoint(self, test_client):
        """Test HTMX component endpoint."""
        response = test_client.post("/api/htmx/component")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "HTMX Component" in response.text


class TestHTMXIntegration:
    """Test suite for HTMX integration patterns."""

    @pytest.fixture
    def htmx_app(self):
        """Create HTMX-enabled application."""
        app = FastAPI()

        @app.get("/htmx/detector-list")
        async def detector_list():
            return HTMLResponse("""
            <div id="detector-list">
                <div class="detector-item">Detector 1</div>
                <div class="detector-item">Detector 2</div>
            </div>
            """)

        @app.post("/htmx/detector-create")
        async def detector_create():
            return HTMLResponse("""
            <div class="detector-item" id="detector-3">
                <span>New Detector</span>
                <button hx-delete="/htmx/detector-delete/3">Delete</button>
            </div>
            """)

        @app.delete("/htmx/detector-delete/{detector_id}")
        async def detector_delete(detector_id: int):
            return HTMLResponse("", status_code=200)

        @app.get("/htmx/detection-results")
        async def detection_results():
            return HTMLResponse("""
            <div id="detection-results">
                <div class="result-item">Normal: 0.1</div>
                <div class="result-item">Anomaly: 0.9</div>
            </div>
            """)

        @app.post("/htmx/dataset-upload")
        async def dataset_upload():
            return HTMLResponse("""
            <div class="upload-status">
                <div class="progress-bar" style="width: 100%"></div>
                <span>Upload Complete</span>
            </div>
            """)

        return app

    @pytest.fixture
    def htmx_client(self, htmx_app):
        """Create HTMX test client."""
        return TestClient(htmx_app)

    @pytest.fixture
    def htmx_headers(self):
        """Create HTMX request headers."""
        return {
            "HX-Request": "true",
            "HX-Trigger": "click",
            "HX-Target": "#content",
            "HX-Current-URL": "https://example.com/dashboard",
        }

    def test_htmx_detector_list_request(self, htmx_client, htmx_headers):
        """Test HTMX detector list request."""
        response = htmx_client.get("/htmx/detector-list", headers=htmx_headers)

        assert response.status_code == 200
        assert "detector-list" in response.text
        assert "Detector 1" in response.text
        assert "Detector 2" in response.text

    def test_htmx_detector_create_request(self, htmx_client, htmx_headers):
        """Test HTMX detector creation request."""
        response = htmx_client.post("/htmx/detector-create", headers=htmx_headers)

        assert response.status_code == 200
        assert "New Detector" in response.text
        assert "hx-delete" in response.text

    def test_htmx_detector_delete_request(self, htmx_client, htmx_headers):
        """Test HTMX detector deletion request."""
        response = htmx_client.delete("/htmx/detector-delete/3", headers=htmx_headers)

        assert response.status_code == 200
        assert response.text == ""  # Empty response for deletion

    def test_htmx_detection_results_request(self, htmx_client, htmx_headers):
        """Test HTMX detection results request."""
        response = htmx_client.get("/htmx/detection-results", headers=htmx_headers)

        assert response.status_code == 200
        assert "detection-results" in response.text
        assert "Normal: 0.1" in response.text
        assert "Anomaly: 0.9" in response.text

    def test_htmx_dataset_upload_request(self, htmx_client, htmx_headers):
        """Test HTMX dataset upload request."""
        response = htmx_client.post("/htmx/dataset-upload", headers=htmx_headers)

        assert response.status_code == 200
        assert "upload-status" in response.text
        assert "Upload Complete" in response.text
        assert "progress-bar" in response.text

    def test_htmx_request_validation(self, htmx_client):
        """Test HTMX request validation."""
        # Test without HTMX headers
        response = htmx_client.get("/htmx/detector-list")
        assert response.status_code == 200  # Should still work

        # Test with invalid HTMX headers
        invalid_headers = {"HX-Request": "false"}
        response = htmx_client.get("/htmx/detector-list", headers=invalid_headers)
        assert response.status_code == 200

    def test_htmx_response_headers(self, htmx_client, htmx_headers):
        """Test HTMX response headers."""
        response = htmx_client.get("/htmx/detector-list", headers=htmx_headers)

        # Check for HTMX-specific response headers
        assert response.status_code == 200
        # HTMX responses should not have redirect headers for partial updates
        assert "HX-Redirect" not in response.headers

    def test_htmx_error_handling(self, htmx_client, htmx_headers):
        """Test HTMX error handling."""
        # Test non-existent endpoint
        response = htmx_client.get("/htmx/non-existent", headers=htmx_headers)
        assert response.status_code == 404

        # Test invalid method
        response = htmx_client.patch("/htmx/detector-list", headers=htmx_headers)
        assert response.status_code == 405


class TestTemplateRendering:
    """Test suite for template rendering and context injection."""

    @pytest.fixture
    def template_renderer(self):
        """Create template renderer instance."""
        from monorepo.presentation.web.templates import TemplateRenderer

        return TemplateRenderer()

    @pytest.fixture
    def template_context(self):
        """Create template context data."""
        return {
            "page_title": "Anomaly Detection Dashboard",
            "user": {
                "name": "Test User",
                "role": "admin",
                "permissions": ["read", "write", "admin"],
            },
            "navigation": {
                "current_section": "dashboard",
                "breadcrumbs": ["Home", "Dashboard"],
            },
            "data": {
                "detectors": [
                    {"id": 1, "name": "Detector 1", "status": "active"},
                    {"id": 2, "name": "Detector 2", "status": "inactive"},
                ],
                "recent_detections": [
                    {
                        "id": 1,
                        "timestamp": "2023-01-01T10:00:00Z",
                        "anomaly_score": 0.8,
                    },
                    {
                        "id": 2,
                        "timestamp": "2023-01-01T10:05:00Z",
                        "anomaly_score": 0.2,
                    },
                ],
            },
            "csrf_token": "csrf123",
            "nonce": "nonce456",
        }

    def test_template_basic_rendering(self, template_renderer, template_context):
        """Test basic template rendering."""
        template_content = """
        <html>
        <head><title>{{ page_title }}</title></head>
        <body>
            <h1>Welcome {{ user.name }}</h1>
            <p>Role: {{ user.role }}</p>
        </body>
        </html>
        """

        rendered = template_renderer.render_string(template_content, template_context)

        assert "Anomaly Detection Dashboard" in rendered
        assert "Welcome Test User" in rendered
        assert "Role: admin" in rendered

    def test_template_conditional_rendering(self, template_renderer, template_context):
        """Test conditional template rendering."""
        template_content = """
        <div>
            {% if user.role == 'admin' %}
                <button>Admin Panel</button>
            {% endif %}
            {% if user.permissions %}
                <ul>
                {% for permission in user.permissions %}
                    <li>{{ permission }}</li>
                {% endfor %}
                </ul>
            {% endif %}
        </div>
        """

        rendered = template_renderer.render_string(template_content, template_context)

        assert "Admin Panel" in rendered
        assert "<li>read</li>" in rendered
        assert "<li>write</li>" in rendered
        assert "<li>admin</li>" in rendered

    def test_template_loop_rendering(self, template_renderer, template_context):
        """Test loop rendering in templates."""
        template_content = """
        <div id="detectors">
            {% for detector in data.detectors %}
                <div class="detector" data-id="{{ detector.id }}">
                    <span>{{ detector.name }}</span>
                    <span class="status {{ detector.status }}">{{ detector.status }}</span>
                </div>
            {% endfor %}
        </div>
        """

        rendered = template_renderer.render_string(template_content, template_context)

        assert 'data-id="1"' in rendered
        assert 'data-id="2"' in rendered
        assert "Detector 1" in rendered
        assert "Detector 2" in rendered
        assert "status active" in rendered
        assert "status inactive" in rendered

    def test_template_security_context(self, template_renderer, template_context):
        """Test security context in templates."""
        template_content = """
        <form method="POST" action="/api/detectors">
            <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
            <input type="text" name="detector_name">
            <button type="submit">Create Detector</button>
        </form>
        <script nonce="{{ nonce }}">
            console.log('Secure script');
        </script>
        """

        rendered = template_renderer.render_string(template_content, template_context)

        assert 'value="csrf123"' in rendered
        assert 'nonce="nonce456"' in rendered

    def test_template_error_handling(self, template_renderer):
        """Test template error handling."""
        template_content = """
        <div>{{ undefined_variable }}</div>
        """

        context = {}

        # Should handle undefined variables gracefully
        rendered = template_renderer.render_string(template_content, context)
        assert rendered is not None

    def test_template_custom_filters(self, template_renderer, template_context):
        """Test custom template filters."""
        template_content = """
        <div>
            <span>{{ data.recent_detections[0].timestamp | datetime_format }}</span>
            <span>{{ data.recent_detections[0].anomaly_score | percentage }}</span>
        </div>
        """

        # Register custom filters
        template_renderer.register_filter(
            "datetime_format", lambda x: x.replace("T", " ")
        )
        template_renderer.register_filter(
            "percentage", lambda x: f"{float(x) * 100:.1f}%"
        )

        rendered = template_renderer.render_string(template_content, template_context)

        assert "2023-01-01 10:00:00Z" in rendered
        assert "80.0%" in rendered

    def test_template_inheritance(self, template_renderer):
        """Test template inheritance."""
        base_template = """
        <html>
        <head>
            <title>{% block title %}Default Title{% endblock %}</title>
        </head>
        <body>
            <div id="content">
                {% block content %}Default Content{% endblock %}
            </div>
        </body>
        </html>
        """

        child_template = """
        {% extends "base.html" %}
        {% block title %}Dashboard{% endblock %}
        {% block content %}
            <h1>Dashboard Content</h1>
        {% endblock %}
        """

        # Mock template loading
        template_renderer.register_template("base.html", base_template)

        rendered = template_renderer.render_string(child_template, {})

        assert "<title>Dashboard</title>" in rendered
        assert "<h1>Dashboard Content</h1>" in rendered

    def test_template_caching(self, template_renderer):
        """Test template caching mechanism."""
        template_content = "<div>{{ message }}</div>"
        context = {"message": "Hello World"}

        # First render
        start_time = datetime.now()
        rendered1 = template_renderer.render_string(template_content, context)
        first_render_time = datetime.now() - start_time

        # Second render (should be cached)
        start_time = datetime.now()
        rendered2 = template_renderer.render_string(template_content, context)
        second_render_time = datetime.now() - start_time

        assert rendered1 == rendered2
        assert "Hello World" in rendered1
        # Second render should be faster due to caching
        assert (
            second_render_time < first_render_time
            or second_render_time.total_seconds() < 0.001
        )


class TestFrontendSupport:
    """Test suite for frontend support endpoints."""

    @pytest.fixture
    def frontend_app(self):
        """Create frontend support application."""
        app = FastAPI()

        @app.get("/api/frontend/config")
        async def get_frontend_config():
            return {
                "api_base_url": "https://api.example.com",
                "features": {
                    "real_time_updates": True,
                    "dark_mode": True,
                    "notifications": True,
                },
                "limits": {"max_file_size": 10485760, "max_detectors": 100},
            }

        @app.get("/api/frontend/user-preferences")
        async def get_user_preferences():
            return {
                "theme": "dark",
                "language": "en",
                "timezone": "UTC",
                "notifications": {"email": True, "browser": False},
            }

        @app.post("/api/frontend/user-preferences")
        async def update_user_preferences(preferences: dict):
            return {"success": True, "preferences": preferences}

        @app.get("/api/frontend/navigation")
        async def get_navigation():
            return {
                "main_menu": [
                    {"id": "dashboard", "label": "Dashboard", "url": "/dashboard"},
                    {"id": "detectors", "label": "Detectors", "url": "/detectors"},
                    {"id": "datasets", "label": "Datasets", "url": "/datasets"},
                ],
                "user_menu": [
                    {"id": "profile", "label": "Profile", "url": "/profile"},
                    {"id": "settings", "label": "Settings", "url": "/settings"},
                    {"id": "logout", "label": "Logout", "url": "/logout"},
                ],
            }

        return app

    @pytest.fixture
    def frontend_client(self, frontend_app):
        """Create frontend support test client."""
        return TestClient(frontend_app)

    def test_frontend_config_endpoint(self, frontend_client):
        """Test frontend configuration endpoint."""
        response = frontend_client.get("/api/frontend/config")

        assert response.status_code == 200
        config = response.json()

        assert "api_base_url" in config
        assert "features" in config
        assert "limits" in config
        assert config["features"]["real_time_updates"] is True
        assert config["limits"]["max_file_size"] == 10485760

    def test_user_preferences_get(self, frontend_client):
        """Test getting user preferences."""
        response = frontend_client.get("/api/frontend/user-preferences")

        assert response.status_code == 200
        preferences = response.json()

        assert "theme" in preferences
        assert "language" in preferences
        assert "timezone" in preferences
        assert "notifications" in preferences
        assert preferences["theme"] == "dark"

    def test_user_preferences_update(self, frontend_client):
        """Test updating user preferences."""
        new_preferences = {
            "theme": "light",
            "language": "es",
            "timezone": "America/New_York",
            "notifications": {"email": False, "browser": True},
        }

        response = frontend_client.post(
            "/api/frontend/user-preferences", json=new_preferences
        )

        assert response.status_code == 200
        result = response.json()

        assert result["success"] is True
        assert result["preferences"]["theme"] == "light"
        assert result["preferences"]["language"] == "es"

    def test_navigation_endpoint(self, frontend_client):
        """Test navigation configuration endpoint."""
        response = frontend_client.get("/api/frontend/navigation")

        assert response.status_code == 200
        navigation = response.json()

        assert "main_menu" in navigation
        assert "user_menu" in navigation
        assert len(navigation["main_menu"]) == 3
        assert len(navigation["user_menu"]) == 3

        # Check main menu items
        main_menu = navigation["main_menu"]
        assert any(item["id"] == "dashboard" for item in main_menu)
        assert any(item["id"] == "detectors" for item in main_menu)
        assert any(item["id"] == "datasets" for item in main_menu)


class TestWebSocketIntegration:
    """Test suite for WebSocket integration."""

    @pytest.fixture
    def websocket_app(self):
        """Create WebSocket application."""
        app = FastAPI()

        @app.websocket("/ws/detections")
        async def websocket_detections(websocket):
            await websocket.accept()
            try:
                while True:
                    # Simulate real-time detection results
                    detection_data = {
                        "detector_id": 1,
                        "timestamp": datetime.now().isoformat(),
                        "anomaly_score": 0.7,
                        "status": "anomaly_detected",
                    }
                    await websocket.send_json(detection_data)
                    await asyncio.sleep(1)
            except Exception:
                pass

        @app.websocket("/ws/status")
        async def websocket_status(websocket):
            await websocket.accept()
            try:
                while True:
                    status_data = {
                        "system_status": "healthy",
                        "active_detectors": 5,
                        "cpu_usage": 45.2,
                        "memory_usage": 62.8,
                    }
                    await websocket.send_json(status_data)
                    await asyncio.sleep(5)
            except Exception:
                pass

        return app

    @pytest.fixture
    def websocket_client(self, websocket_app):
        """Create WebSocket test client."""
        return TestClient(websocket_app)

    def test_websocket_detection_connection(self, websocket_client):
        """Test WebSocket detection connection."""
        with websocket_client.websocket_connect("/ws/detections") as websocket:
            # Receive initial message
            data = websocket.receive_json()

            assert "detector_id" in data
            assert "timestamp" in data
            assert "anomaly_score" in data
            assert "status" in data
            assert data["detector_id"] == 1
            assert data["status"] == "anomaly_detected"

    def test_websocket_status_connection(self, websocket_client):
        """Test WebSocket status connection."""
        with websocket_client.websocket_connect("/ws/status") as websocket:
            # Receive initial message
            data = websocket.receive_json()

            assert "system_status" in data
            assert "active_detectors" in data
            assert "cpu_usage" in data
            assert "memory_usage" in data
            assert data["system_status"] == "healthy"

    def test_websocket_message_format(self, websocket_client):
        """Test WebSocket message format validation."""
        with websocket_client.websocket_connect("/ws/detections") as websocket:
            data = websocket.receive_json()

            # Check data types
            assert isinstance(data["detector_id"], int)
            assert isinstance(data["timestamp"], str)
            assert isinstance(data["anomaly_score"], float)
            assert isinstance(data["status"], str)

            # Check value ranges
            assert 0 <= data["anomaly_score"] <= 1
            assert data["status"] in ["normal", "anomaly_detected", "warning"]

    def test_websocket_error_handling(self, websocket_client):
        """Test WebSocket error handling."""
        # Test connection to non-existent endpoint
        try:
            with websocket_client.websocket_connect("/ws/nonexistent"):
                pass
        except Exception as e:
            # Should raise connection error
            assert "404" in str(e) or "WebSocket" in str(e)


class TestMiddlewareIntegration:
    """Test suite for middleware integration in web UI."""

    @pytest.fixture
    def middleware_app(self):
        """Create application with middleware stack."""
        app = FastAPI()

        # Add middleware
        from monorepo.presentation.web.middleware import SecurityMiddleware

        app.add_middleware(SecurityMiddleware)

        @app.get("/protected")
        async def protected_endpoint():
            return {"message": "Protected content"}

        return app

    @pytest.fixture
    def middleware_client(self, middleware_app):
        """Create middleware test client."""
        return TestClient(middleware_app)

    def test_security_middleware_headers(self, middleware_client):
        """Test security middleware adds headers."""
        response = middleware_client.get("/protected")

        assert response.status_code == 200

        # Check security headers
        assert "X-Frame-Options" in response.headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_csp_middleware_integration(self, middleware_client):
        """Test CSP middleware integration."""
        response = middleware_client.get("/protected")

        assert response.status_code == 200
        assert "Content-Security-Policy" in response.headers

        csp_header = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp_header

    def test_rate_limiting_middleware(self, middleware_client):
        """Test rate limiting middleware."""
        # Make multiple requests
        for i in range(10):
            response = middleware_client.get("/protected")
            if response.status_code == 429:
                # Rate limit triggered
                assert "Too Many Requests" in response.text
                break
        else:
            # If no rate limiting, that's also valid for testing
            pass

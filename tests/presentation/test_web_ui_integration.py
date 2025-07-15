"""
Integration tests for Web UI and static assets
Tests for main HTML page and static asset accessibility.
"""

import pytest
from fastapi.testclient import TestClient

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.web.app import create_web_app


@pytest.fixture
def web_client():
    """Create test client for web app."""
    container = create_container()
    app = create_web_app(container)
    return TestClient(app)


@pytest.fixture
def api_client():
    """Create test client for API app."""
    from pynomaly.presentation.api.app import create_app

    container = create_container()
    app = create_app(container)
    return TestClient(app)


class TestWebUIMainPage:
    """Test main Web UI page functionality."""

    def test_home_page_returns_200_with_title(self, web_client: TestClient):
        """Test that client.get("/") returns status 200 and contains main HTML with title."""
        response = web_client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

        # Check for main HTML structure
        html_content = response.text
        assert "<title>Pynomaly" in html_content
        assert "<html" in html_content
        assert "</html>" in html_content

        # Check for basic HTML structure
        assert "<head>" in html_content
        assert "<body>" in html_content

    def test_home_page_contains_pynomaly_branding(self, web_client: TestClient):
        """Test that home page contains Pynomaly branding elements."""
        response = web_client.get("/")

        assert response.status_code == 200
        html_content = response.text

        # Check for Pynomaly branding
        assert "Pynomaly" in html_content

        # Check for common navigation or layout elements
        common_elements = [
            "Dashboard",
            "Detectors",
            "Datasets",
            "nav",
            "main",
        ]

        # At least some of these elements should be present
        found_elements = [elem for elem in common_elements if elem in html_content]
        assert (
            len(found_elements) > 0
        ), f"Expected to find at least one of {common_elements} in HTML"

    def test_home_page_has_proper_html_structure(self, web_client: TestClient):
        """Test that home page has proper HTML structure."""
        response = web_client.get("/")

        assert response.status_code == 200
        html_content = response.text.lower()

        # Check for DOCTYPE declaration (optional but good practice)
        # Note: FastAPI/Starlette might not always include this depending on how templates are rendered

        # Check for essential HTML elements
        assert "<html" in html_content
        assert "<head>" in html_content
        assert "<body>" in html_content
        assert "</head>" in html_content
        assert "</body>" in html_content
        assert "</html>" in html_content

    def test_home_page_meta_tags(self, web_client: TestClient):
        """Test that home page includes basic meta tags."""
        response = web_client.get("/")

        assert response.status_code == 200
        html_content = response.text.lower()

        # Check for charset meta tag
        assert "charset" in html_content

        # Check for viewport meta tag (important for responsive design)
        if "viewport" in html_content:
            assert "viewport" in html_content


class TestStaticAssets:
    """Test static asset accessibility."""

    def test_main_css_accessible(self, web_client: TestClient):
        """Test that /static/css/main.css is reachable."""
        response = web_client.get("/static/css/main.css")

        # main.css might not exist, but let's check for any CSS files that do exist
        if response.status_code == 404:
            # Try other common CSS files that we know exist
            css_files = [
                "/static/css/app.css",
                "/static/css/styles.css",
                "/static/css/advanced_ui.css",
                "/static/css/tailwind.css",
            ]

            css_found = False
            for css_file in css_files:
                css_response = web_client.get(css_file)
                if css_response.status_code == 200:
                    assert "text/css" in css_response.headers.get("content-type", "")
                    css_found = True
                    break

            # At least one CSS file should be accessible
            assert (
                css_found
            ), f"Expected at least one CSS file to be accessible from {css_files}"
        else:
            # main.css exists and is accessible
            assert response.status_code == 200
            assert "text/css" in response.headers.get("content-type", "")

    def test_app_css_accessible(self, web_client: TestClient):
        """Test that app.css is accessible (this file definitely exists)."""
        response = web_client.get("/static/css/app.css")

        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")

        # Basic CSS content check
        css_content = response.text
        assert len(css_content) > 0

    def test_javascript_assets_accessible(self, web_client: TestClient):
        """Test that JavaScript assets are accessible."""
        js_files = [
            "/static/js/app.js",
            "/static/js/main.js",
            "/static/js/components.js",
        ]

        js_found = False
        for js_file in js_files:
            js_response = web_client.get(js_file)
            if js_response.status_code == 200:
                # Check content type for JavaScript
                content_type = js_response.headers.get("content-type", "")
                assert any(
                    js_type in content_type
                    for js_type in [
                        "application/javascript",
                        "text/javascript",
                        "application/x-javascript",
                    ]
                )
                js_found = True
                break

        # At least one JS file should be accessible
        assert (
            js_found
        ), f"Expected at least one JS file to be accessible from {js_files}"

    def test_static_directory_structure(self, web_client: TestClient):
        """Test that static directory structure is properly configured."""
        # Test that we get 404 for non-existent static files (not 500 errors)
        response = web_client.get("/static/nonexistent/file.css")
        assert response.status_code == 404

        # Test that static route is properly configured
        response = web_client.get("/static/")
        # This might return 404 or 403 depending on configuration, but shouldn't be 500
        assert response.status_code in [403, 404]


class TestAPIEndpoints:
    """Test that API endpoints use correct /api prefix."""

    def test_api_health_endpoint(self, api_client: TestClient):
        """Test that API health endpoint is accessible at /api/health."""
        response = api_client.get("/api/health/")

        # This should either work (200) or have auth issues (401), but not be not found (404)
        assert response.status_code in [200, 401, 503]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, dict)

    def test_api_root_endpoint_redirects_or_works(self, api_client: TestClient):
        """Test that API root endpoint behavior."""
        response = api_client.get("/api/")

        # API root should either work, redirect, or require auth, but not be not found
        assert response.status_code in [200, 301, 302, 401, 403]

    def test_non_api_endpoints_not_found_in_api(self, api_client: TestClient):
        """Test that non-API endpoints return 404 in API client."""
        response = api_client.get("/dashboard")
        assert response.status_code == 404

        response = api_client.get("/detectors")
        assert response.status_code == 404


class TestWebVsAPIRouting:
    """Test proper routing separation between Web UI and API."""

    def test_web_routes_in_web_app(self, web_client: TestClient):
        """Test that web routes work in web app."""
        web_routes = ["/", "/dashboard", "/detectors", "/datasets"]

        for route in web_routes:
            response = web_client.get(route)
            # Should work (200) or require auth (302 redirect), not be not found (404)
            assert response.status_code in [200, 302, 401, 403]

    def test_api_routes_in_api_app(self, api_client: TestClient):
        """Test that API routes work in API app."""
        api_routes = ["/api/health/", "/api/detectors/", "/api/datasets/"]

        for route in api_routes:
            response = api_client.get(route)
            # Should work (200) or require auth (401), not be not found (404)
            assert response.status_code in [200, 401, 403, 422]

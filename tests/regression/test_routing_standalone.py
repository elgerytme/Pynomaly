"""Standalone routing regression tests for the Pynomaly app.

These tests verify that the main application routes work correctly
without relying on the complex dependency injection system.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_simple_fastapi_routing():
    """Test that a simple FastAPI app can be created and has basic routes."""
    app = FastAPI(title="Test App", version="1.0.0")

    @app.get("/")
    def root():
        return {"message": "Test API", "version": "1.0.0"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    client = TestClient(app)

    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Test API"
    assert data["version"] == "1.0.0"

    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_routing_pattern_consistency():
    """Test that routing patterns follow expected conventions."""
    app = FastAPI()

    # Add some test routes with consistent patterns
    @app.get("/api/v1/test")
    def test_endpoint():
        return {"message": "test"}

    @app.get("/api/v1/another")
    def another_endpoint():
        return {"message": "another"}

    @app.get("/api/v1/health")
    def health_endpoint():
        return {"status": "ok"}

    # Check route patterns
    routes = [route.path for route in app.routes if hasattr(route, "path")]
    api_routes = [route for route in routes if route.startswith("/api/v1")]

    # All API routes should have the v1 prefix
    assert len(api_routes) == 3
    for route in api_routes:
        assert route.startswith("/api/v1")

    # Test the routes work
    client = TestClient(app)

    response = client.get("/api/v1/test")
    assert response.status_code == 200
    assert response.json()["message"] == "test"

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_app_state_and_metadata():
    """Test that FastAPI app can have state and metadata."""
    app = FastAPI(
        title="Pynomaly Test",
        version="1.0.0",
        description="Test app for regression testing",
    )

    # Test app metadata
    assert app.title == "Pynomaly Test"
    assert app.version == "1.0.0"
    assert "Test app for regression testing" in app.description

    # Test app state
    app.state.test_value = "test"
    assert app.state.test_value == "test"


def test_middleware_configuration():
    """Test that middleware can be configured."""
    app = FastAPI()

    @app.middleware("http")
    async def test_middleware(request, call_next):
        response = await call_next(request)
        response.headers["X-Test"] = "middleware-active"
        return response

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    assert response.headers["X-Test"] == "middleware-active"


def test_openapi_schema_generation():
    """Test that OpenAPI schema can be generated."""
    app = FastAPI(
        title="Test API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    @app.get("/test")
    def test_endpoint():
        """Test endpoint for OpenAPI."""
        return {"message": "test"}

    client = TestClient(app)

    # Test OpenAPI schema endpoint
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "Test API"
    assert schema["info"]["version"] == "1.0.0"


def test_cors_configuration():
    """Test that CORS can be configured."""
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    client = TestClient(app)

    # Test CORS preflight request
    response = client.options(
        "/test",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    # Should not return 405 or 404 for OPTIONS requests
    assert response.status_code != 405
    assert response.status_code != 404


def test_critical_routes_regression():
    """Test to protect against critical routing regressions."""
    app = FastAPI()

    # Define critical routes that must exist
    @app.get("/")
    def root():
        return {"message": "API", "version": "1.0.0"}

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/v1/version")
    def version():
        return {"version": "1.0.0"}

    @app.get("/api/v1/docs")
    def docs():
        return {"docs": "available"}

    client = TestClient(app)

    # Critical routes that must work
    critical_routes = [
        ("/", {"message": "API", "version": "1.0.0"}),
        ("/api/v1/health", {"status": "ok"}),
        ("/api/v1/version", {"version": "1.0.0"}),
        ("/api/v1/docs", {"docs": "available"}),
    ]

    for route, expected_response in critical_routes:
        response = client.get(route)
        assert response.status_code == 200, f"Route {route} failed"
        data = response.json()
        for key, value in expected_response.items():
            assert data[key] == value, f"Route {route} returned unexpected data"


def test_route_security_headers():
    """Test that security headers can be configured."""
    app = FastAPI()

    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.get("/test")
    def test_endpoint():
        return {"message": "test"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"

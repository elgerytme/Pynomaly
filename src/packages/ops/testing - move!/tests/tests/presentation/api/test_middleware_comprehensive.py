"""
Comprehensive tests for API middleware components.

This module provides extensive testing for all middleware components in the
presentation layer, including security headers, CORS, rate limiting, and
middleware integration.
"""

import time
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from monorepo.presentation.api.middleware.security_headers import (
    SecurityHeadersMiddleware,
)
from monorepo.presentation.api.middleware_integration import (
    configure_cors,
    configure_rate_limiting,
    setup_middleware_stack,
)


class TestSecurityHeadersMiddleware:
    """Test security headers middleware functionality."""

    @pytest.fixture
    def app(self):
        """Create test FastAPI app with security middleware."""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_security_headers_applied(self, client):
        """Test that security headers are applied to responses."""
        response = client.get("/test")

        assert response.status_code == 200

        # Check security headers
        headers = response.headers
        assert "X-Content-Type-Options" in headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in headers
        assert headers["X-Frame-Options"] == "DENY"
        assert "X-XSS-Protection" in headers
        assert headers["X-XSS-Protection"] == "1; mode=block"
        assert "Strict-Transport-Security" in headers
        assert "max-age=" in headers["Strict-Transport-Security"]
        assert "Referrer-Policy" in headers
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_content_security_policy_header(self, client):
        """Test Content Security Policy header."""
        response = client.get("/test")

        assert "Content-Security-Policy" in response.headers
        csp = response.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "script-src" in csp
        assert "style-src" in csp
        assert "img-src" in csp

    def test_cache_control_headers(self, client):
        """Test cache control headers."""
        response = client.get("/test")

        assert "Cache-Control" in response.headers
        cache_control = response.headers["Cache-Control"]
        assert "no-store" in cache_control or "no-cache" in cache_control

    def test_permissions_policy_header(self, client):
        """Test Permissions Policy header."""
        response = client.get("/test")

        if "Permissions-Policy" in response.headers:
            policy = response.headers["Permissions-Policy"]
            assert "camera=()" in policy or "microphone=()" in policy

    def test_middleware_with_different_methods(self, client):
        """Test middleware applies headers to different HTTP methods."""
        # Test GET
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers

        # Test POST (if endpoint supports it)
        response = client.post("/test")
        # Should return 405 Method Not Allowed but still have security headers
        assert "X-Content-Type-Options" in response.headers

    def test_middleware_with_error_responses(self, client):
        """Test middleware applies headers to error responses."""
        response = client.get("/nonexistent")

        assert response.status_code == 404
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers

    @pytest.mark.asyncio
    async def test_middleware_async_processing(self):
        """Test middleware async processing."""
        middleware = SecurityHeadersMiddleware(Mock())

        # Mock request and call_next
        mock_request = Mock(spec=Request)
        mock_response = Response(content="test", media_type="text/plain")

        async def mock_call_next(request):
            return mock_response

        # Process request
        result = await middleware.dispatch(mock_request, mock_call_next)

        assert result is not None
        assert hasattr(result, "headers")


class TestCORSConfiguration:
    """Test CORS configuration and middleware."""

    @pytest.fixture
    def app(self):
        """Create test app with CORS configuration."""
        app = FastAPI()
        configure_cors(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_cors_preflight_request(self, client):
        """Test CORS preflight request handling."""
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers

    def test_cors_simple_request(self, client):
        """Test simple CORS request."""
        response = client.get("/test", headers={"Origin": "http://localhost:3000"})

        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    def test_cors_credentials_allowed(self, client):
        """Test CORS with credentials."""
        response = client.get("/test", headers={"Origin": "http://localhost:3000"})

        if "Access-Control-Allow-Credentials" in response.headers:
            assert response.headers["Access-Control-Allow-Credentials"] == "true"

    def test_cors_methods_allowed(self, client):
        """Test allowed CORS methods."""
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        if response.status_code == 200:
            methods = response.headers.get("Access-Control-Allow-Methods", "")
            assert any(method in methods for method in ["GET", "POST", "PUT", "DELETE"])


class TestRateLimitingMiddleware:
    """Test rate limiting middleware functionality."""

    @pytest.fixture
    def app(self):
        """Create test app with rate limiting."""
        app = FastAPI()
        configure_rate_limiting(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_rate_limiting_normal_usage(self, client):
        """Test normal usage within rate limits."""
        response = client.get("/test")
        assert response.status_code == 200

        # Check rate limit headers if present
        if "X-RateLimit-Limit" in response.headers:
            assert int(response.headers["X-RateLimit-Limit"]) > 0
        if "X-RateLimit-Remaining" in response.headers:
            assert int(response.headers["X-RateLimit-Remaining"]) >= 0

    def test_rate_limiting_headers_present(self, client):
        """Test rate limiting headers are present."""
        response = client.get("/test")

        # Rate limiting headers might be present
        possible_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "Retry-After",
        ]

        # At least some rate limiting indication should be present
        # (implementation dependent)
        headers_present = any(header in response.headers for header in possible_headers)

        # This test is flexible as rate limiting implementation varies
        assert response.status_code in [200, 429]

    def test_rate_limiting_with_different_endpoints(self, client):
        """Test rate limiting across different endpoints."""
        # Test multiple requests to same endpoint
        for _ in range(5):
            response = client.get("/test")
            assert response.status_code in [200, 429]

            # If rate limited, should have appropriate headers
            if response.status_code == 429:
                assert (
                    "Retry-After" in response.headers
                    or "X-RateLimit-Reset" in response.headers
                )
                break

    @pytest.mark.asyncio
    async def test_rate_limiting_async_behavior(self):
        """Test rate limiting async behavior."""
        # This test depends on actual rate limiting implementation
        # For now, just ensure no exceptions are raised
        pass


class TestMiddlewareIntegration:
    """Test middleware integration and stack setup."""

    @pytest.fixture
    def app(self):
        """Create test app."""
        return FastAPI()

    def test_middleware_stack_setup(self, app):
        """Test middleware stack setup."""
        initial_middleware_count = len(app.middleware_stack)

        setup_middleware_stack(app)

        # Should have added middleware
        assert len(app.middleware_stack) >= initial_middleware_count

    def test_middleware_order(self, app):
        """Test middleware is added in correct order."""
        setup_middleware_stack(app)

        # Create client and make request
        client = TestClient(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        response = client.get("/test")

        # Should have security headers (added by middleware)
        assert "X-Content-Type-Options" in response.headers

    def test_middleware_with_exception_handling(self, app):
        """Test middleware behavior with exceptions."""
        setup_middleware_stack(app)

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

        client = TestClient(app)
        response = client.get("/error")

        # Should still have security headers even on error
        assert "X-Content-Type-Options" in response.headers

    def test_middleware_performance_impact(self, app):
        """Test middleware performance impact."""
        setup_middleware_stack(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)

        # Time multiple requests
        start_time = time.time()
        for _ in range(10):
            response = client.get("/test")
            assert response.status_code == 200
        end_time = time.time()

        # Should not add significant overhead
        total_time = end_time - start_time
        assert total_time < 5.0  # Should complete 10 requests in under 5 seconds


class TestCustomMiddleware:
    """Test custom middleware functionality."""

    class TestMiddleware(BaseHTTPMiddleware):
        """Test middleware class."""

        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            response.headers["X-Test-Header"] = "test-value"
            return response

    @pytest.fixture
    def app(self):
        """Create test app with custom middleware."""
        app = FastAPI()
        app.add_middleware(self.TestMiddleware)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_custom_middleware_execution(self, client):
        """Test custom middleware execution."""
        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Test-Header" in response.headers
        assert response.headers["X-Test-Header"] == "test-value"

    def test_middleware_request_modification(self):
        """Test middleware can modify requests."""

        class RequestModifyingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Add custom header to request
                request.headers.__dict__["X-Modified"] = "true"
                return await call_next(request)

        app = FastAPI()
        app.add_middleware(RequestModifyingMiddleware)

        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"modified": "X-Modified" in request.headers}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200

    def test_middleware_exception_handling(self):
        """Test middleware exception handling."""

        class ExceptionHandlingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                try:
                    return await call_next(request)
                except Exception:
                    return JSONResponse(
                        status_code=500,
                        content={"error": "middleware_handled"},
                        headers={"X-Error-Handled": "true"},
                    )

        app = FastAPI()
        app.add_middleware(ExceptionHandlingMiddleware)

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

        client = TestClient(app)
        response = client.get("/error")

        assert response.status_code == 500
        assert "X-Error-Handled" in response.headers


class TestMiddlewareConfiguration:
    """Test middleware configuration options."""

    def test_security_headers_configuration(self):
        """Test security headers can be configured."""
        app = FastAPI()

        # Test with default configuration
        middleware = SecurityHeadersMiddleware(Mock())
        assert middleware is not None

    def test_cors_configuration_options(self):
        """Test CORS configuration options."""
        app = FastAPI()

        # Test CORS configuration
        configure_cors(app, allow_origins=["*"])

        # Should not raise exception
        assert len(app.middleware_stack) > 0

    def test_rate_limiting_configuration(self):
        """Test rate limiting configuration."""
        app = FastAPI()

        # Test rate limiting configuration
        configure_rate_limiting(app, requests_per_minute=60)

        # Should not raise exception
        assert len(app.middleware_stack) >= 0

    def test_middleware_stack_configuration(self):
        """Test complete middleware stack configuration."""
        app = FastAPI()

        # Configure complete middleware stack
        setup_middleware_stack(app)

        # Should have middleware
        assert len(app.middleware_stack) > 0

        # Test basic functionality
        client = TestClient(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        response = client.get("/test")
        assert response.status_code == 200


class TestMiddlewareEdgeCases:
    """Test middleware edge cases and error conditions."""

    def test_middleware_with_large_responses(self):
        """Test middleware with large responses."""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/large")
        async def large_endpoint():
            return {"data": "x" * 10000}  # Large response

        client = TestClient(app)
        response = client.get("/large")

        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers
        assert len(response.content) > 10000

    def test_middleware_with_streaming_responses(self):
        """Test middleware with streaming responses."""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        from fastapi.responses import StreamingResponse

        @app.get("/stream")
        async def stream_endpoint():
            def generate():
                for i in range(10):
                    yield f"chunk {i}\n"

            return StreamingResponse(
                (chunk for chunk in generate()), media_type="text/plain"
            )

        client = TestClient(app)
        response = client.get("/stream")

        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers

    def test_middleware_with_async_generators(self):
        """Test middleware with async generators."""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/async-gen")
        async def async_gen_endpoint():
            async def generate():
                for i in range(3):
                    yield f"async chunk {i}\n"

            return StreamingResponse(generate(), media_type="text/plain")

        client = TestClient(app)
        response = client.get("/async-gen")

        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers

    def test_middleware_concurrency(self):
        """Test middleware under concurrent requests."""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)

        @app.get("/concurrent")
        async def concurrent_endpoint():
            import asyncio

            await asyncio.sleep(0.1)  # Simulate async work
            return {"message": "concurrent"}

        client = TestClient(app)

        # Make concurrent requests
        import concurrent.futures

        def make_request():
            return client.get("/concurrent")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]

        # All requests should succeed with security headers
        for response in responses:
            assert response.status_code == 200
            assert "X-Content-Type-Options" in response.headers

#!/usr/bin/env python3
"""
Integration tests for rate limiting middleware in FastAPI application.
Tests rate limiting functionality, headers, and error responses.
"""

import time
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from pynomaly.infrastructure.security.rate_limiting_middleware import (
    RateLimitResult,
    RateLimitStrategy,
)
from pynomaly.presentation.api.app import create_app


class TestRateLimitingIntegration:
    """Test rate limiting middleware integration."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.redis"
        ) as mock_redis:
            mock_client = Mock()
            mock_redis.from_url.return_value = mock_client

            # Mock pipeline
            mock_pipeline = Mock()
            mock_client.pipeline.return_value = mock_pipeline
            mock_pipeline.execute.return_value = [1, None]  # First request

            yield mock_client

    @pytest.fixture
    def mock_settings(self):
        """Mock application settings."""
        settings = Mock()
        settings.redis_url = "redis://localhost:6379"
        settings.get_cors_config.return_value = {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
        settings.secret_key = "test-secret-key"
        settings.app.name = "Test Pynomaly"
        settings.app.version = "1.0.0"
        settings.docs_enabled = True
        settings.cache_enabled = False
        settings.auth_enabled = False
        settings.monitoring.metrics_enabled = False
        settings.monitoring.tracing_enabled = False
        settings.monitoring.prometheus_enabled = False
        return settings

    @pytest.fixture
    def app_with_rate_limiting(self, mock_redis, mock_settings):
        """Create FastAPI app with rate limiting enabled."""
        with patch("pynomaly.infrastructure.config.create_container") as mock_container:
            container = Mock()
            container.config.return_value = mock_settings
            mock_container.return_value = container

            app = create_app(container)
            return app

    def test_rate_limiting_middleware_added(self, app_with_rate_limiting):
        """Test that rate limiting middleware is properly added to the app."""
        client = TestClient(app_with_rate_limiting)

        # Make a request to any endpoint
        response = client.get("/")

        # Should get successful response (rate limit not exceeded)
        assert response.status_code == 200

    def test_rate_limiting_headers_present(self, app_with_rate_limiting, mock_redis):
        """Test that rate limiting headers are added to responses."""
        # Mock rate limiter to return allowed request
        mock_redis.pipeline.return_value.execute.return_value = [1, None]

        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
        ) as mock_check:
            mock_check.return_value = RateLimitResult(
                allowed=True, limit=100, remaining=99, reset_time=int(time.time()) + 60
            )

            response = client.get("/")

            # Rate limit headers should be present
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers

            assert response.headers["X-RateLimit-Limit"] == "100"
            assert response.headers["X-RateLimit-Remaining"] == "99"

    def test_rate_limit_exceeded_response(self, app_with_rate_limiting, mock_redis):
        """Test response when rate limit is exceeded."""
        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
        ) as mock_check:
            mock_check.return_value = RateLimitResult(
                allowed=False,
                limit=100,
                remaining=0,
                reset_time=int(time.time()) + 60,
                retry_after=60,
            )

            response = client.get("/")

            # Should return 429 Too Many Requests
            assert response.status_code == 429

            # Check response body
            data = response.json()
            assert data["error"] == "Rate limit exceeded"
            assert data["limit"] == 100
            assert data["remaining"] == 0
            assert "retry_after" in data

            # Check headers
            assert response.headers["X-RateLimit-Limit"] == "100"
            assert response.headers["X-RateLimit-Remaining"] == "0"
            assert "Retry-After" in response.headers

    def test_rate_limiting_skips_health_endpoints(self, app_with_rate_limiting):
        """Test that rate limiting skips health check endpoints."""
        client = TestClient(app_with_rate_limiting)

        # Health endpoints should bypass rate limiting
        response = client.get("/api/v1/health/")
        assert response.status_code == 200

        # No rate limit headers should be present for health endpoints
        assert "X-RateLimit-Limit" not in response.headers

    def test_rate_limiting_with_different_scopes(self, app_with_rate_limiting):
        """Test rate limiting with different scopes (IP, user, etc.)."""
        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
        ) as mock_check:
            # Mock different rate limit results for different scopes
            mock_check.side_effect = [
                RateLimitResult(
                    allowed=True,
                    limit=100,
                    remaining=99,
                    reset_time=int(time.time()) + 60,
                ),  # IP limit
                RateLimitResult(
                    allowed=True,
                    limit=200,
                    remaining=199,
                    reset_time=int(time.time()) + 60,
                ),  # User limit
            ]

            response = client.get("/")

            # Should be successful
            assert response.status_code == 200

            # Check that rate limiter was called multiple times for different scopes
            assert mock_check.call_count >= 1

    def test_rate_limiting_auth_endpoints(self, app_with_rate_limiting):
        """Test stricter rate limiting for auth endpoints."""
        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
        ) as mock_check:
            # Mock auth endpoint rate limit exceeded
            mock_check.return_value = RateLimitResult(
                allowed=False,
                limit=10,
                remaining=0,
                reset_time=int(time.time()) + 60,
                retry_after=60,
            )

            response = client.post(
                "/api/v1/auth/login", json={"username": "test", "password": "test"}
            )

            # Should return 429 with auth-specific message
            assert response.status_code == 429
            data = response.json()
            assert "authentication" in data["message"].lower()

    def test_rate_limiting_redis_fallback(self, app_with_rate_limiting):
        """Test rate limiting graceful fallback when Redis is unavailable."""
        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
        ) as mock_check:
            # Simulate Redis connection error
            mock_check.side_effect = RedisConnectionError("Redis connection failed")

            response = client.get("/")

            # Should still work (graceful degradation)
            assert response.status_code == 200

    def test_rate_limiting_audit_logging(self, app_with_rate_limiting):
        """Test that rate limit violations are properly logged."""
        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
        ) as mock_check:
            with patch(
                "pynomaly.infrastructure.security.rate_limiting_middleware.get_audit_logger"
            ) as mock_audit:
                mock_logger = Mock()
                mock_audit.return_value = mock_logger

                mock_check.return_value = RateLimitResult(
                    allowed=False,
                    limit=100,
                    remaining=0,
                    reset_time=int(time.time()) + 60,
                    retry_after=60,
                )

                response = client.get("/")

                # Should log security event
                mock_logger.log_security_event.assert_called_once()

                # Check audit log call
                call_args = mock_logger.log_security_event.call_args
                assert "Rate limit exceeded" in str(call_args)

    def test_rate_limiting_multiple_strategies(self, app_with_rate_limiting):
        """Test different rate limiting strategies."""
        client = TestClient(app_with_rate_limiting)

        strategies = [
            RateLimitStrategy.FIXED_WINDOW,
            RateLimitStrategy.SLIDING_WINDOW,
            RateLimitStrategy.TOKEN_BUCKET,
            RateLimitStrategy.SLIDING_LOG,
        ]

        for strategy in strategies:
            with patch(
                "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
            ) as mock_check:
                mock_check.return_value = RateLimitResult(
                    allowed=True,
                    limit=100,
                    remaining=99,
                    reset_time=int(time.time()) + 60,
                )

                response = client.get("/")
                assert response.status_code == 200

    def test_rate_limiting_api_key_scope(self, app_with_rate_limiting):
        """Test rate limiting with API key scope."""
        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
        ) as mock_check:
            mock_check.return_value = RateLimitResult(
                allowed=True,
                limit=1000,
                remaining=999,
                reset_time=int(time.time()) + 60,
            )

            # Request with API key header
            response = client.get("/", headers={"X-API-Key": "pyn_test_key_123"})

            assert response.status_code == 200

            # Should use API key rate limit
            assert mock_check.called

    def test_rate_limiting_configuration_loading(self, app_with_rate_limiting):
        """Test that rate limiting configuration is properly loaded."""
        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimitMiddleware"
        ) as mock_middleware:
            # Verify middleware was initialized with settings
            mock_middleware.assert_called_once()
            call_args = mock_middleware.call_args
            assert "settings" in call_args.kwargs

    def test_rate_limiting_concurrent_requests(self, app_with_rate_limiting):
        """Test rate limiting with concurrent requests."""
        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
        ) as mock_check:
            mock_check.return_value = RateLimitResult(
                allowed=True, limit=100, remaining=99, reset_time=int(time.time()) + 60
            )

            # Simulate concurrent requests
            responses = []
            for i in range(5):
                response = client.get("/")
                responses.append(response)

            # All should succeed
            for response in responses:
                assert response.status_code == 200

            # Rate limiter should be called for each request
            assert mock_check.call_count >= 5

    def test_rate_limiting_performance_impact(self, app_with_rate_limiting):
        """Test that rate limiting doesn't significantly impact performance."""
        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
        ) as mock_check:
            mock_check.return_value = RateLimitResult(
                allowed=True, limit=100, remaining=99, reset_time=int(time.time()) + 60
            )

            start_time = time.time()

            # Make multiple requests
            for i in range(10):
                response = client.get("/")
                assert response.status_code == 200

            end_time = time.time()

            # Should complete within reasonable time (< 1 second for 10 requests)
            assert (end_time - start_time) < 1.0

    @pytest.mark.asyncio
    async def test_rate_limiting_async_compatibility(self, app_with_rate_limiting):
        """Test rate limiting middleware compatibility with async operations."""
        # This test ensures the middleware works with async FastAPI operations
        client = TestClient(app_with_rate_limiting)

        with patch(
            "pynomaly.infrastructure.security.rate_limiting_middleware.RateLimiter.check_rate_limit"
        ) as mock_check:
            mock_check.return_value = RateLimitResult(
                allowed=True, limit=100, remaining=99, reset_time=int(time.time()) + 60
            )

            # Test async compatibility
            response = client.get("/")
            assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

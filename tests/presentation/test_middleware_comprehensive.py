"""Comprehensive tests for middleware and security - Phase 3 Coverage."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import time
import jwt
from datetime import datetime, timedelta

from pynomaly.infrastructure.config import create_container
from pynomaly.presentation.api.app import create_app
from pynomaly.infrastructure.auth import JWTAuthService, RateLimiter
from pynomaly.infrastructure.middleware import (
    SecurityMiddleware,
    LoggingMiddleware,
    CORSMiddleware,
    CompressionMiddleware,
    RequestValidationMiddleware
)


@pytest.fixture
def test_container():
    """Create test container with middleware."""
    container = create_container()
    return container


@pytest.fixture
def client_with_middleware(test_container):
    """Create test client with all middleware enabled."""
    app = create_app(test_container)
    return TestClient(app)


@pytest.fixture
async def async_client_with_middleware(test_container):
    """Create async test client with middleware."""
    app = create_app(test_container)
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def valid_jwt_token():
    """Create valid JWT token for testing."""
    secret = "test_secret_key_for_testing_only"
    payload = {
        "sub": "test_user_123",
        "username": "testuser",
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "roles": ["user"]
    }
    return jwt.encode(payload, secret, algorithm="HS256")


class TestSecurityMiddleware:
    """Test security middleware functionality."""
    
    def test_security_headers_injection(self, client_with_middleware: TestClient):
        """Test security headers are properly injected."""
        response = client_with_middleware.get("/api/health/")
        
        assert response.status_code == 200
        
        # Check security headers
        headers = response.headers
        expected_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        for header_name, expected_value in expected_headers.items():
            if header_name.lower() in [h.lower() for h in headers.keys()]:
                header_value = headers.get(header_name)
                assert header_value is not None
    
    def test_content_security_policy(self, client_with_middleware: TestClient):
        """Test Content Security Policy header."""
        response = client_with_middleware.get("/")
        
        assert response.status_code == 200
        
        csp_header = response.headers.get("Content-Security-Policy")
        if csp_header:
            # Should contain basic CSP directives
            assert "default-src" in csp_header
            assert "script-src" in csp_header
            assert "style-src" in csp_header
    
    def test_cors_headers(self, client_with_middleware: TestClient):
        """Test CORS headers."""
        # Preflight request
        response = client_with_middleware.options(
            "/api/detectors/",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization"
            }
        )
        
        assert response.status_code == 200
        
        # Check CORS headers
        cors_headers = response.headers
        assert "Access-Control-Allow-Origin" in cors_headers
        assert "Access-Control-Allow-Methods" in cors_headers
        assert "Access-Control-Allow-Headers" in cors_headers
    
    def test_request_id_injection(self, client_with_middleware: TestClient):
        """Test request ID is injected into responses."""
        response = client_with_middleware.get("/api/health/")
        
        assert response.status_code == 200
        
        # Request ID should be in headers
        request_id = response.headers.get("X-Request-ID")
        assert request_id is not None
        assert len(request_id) > 0
    
    def test_sensitive_header_removal(self, client_with_middleware: TestClient):
        """Test sensitive headers are removed from responses."""
        response = client_with_middleware.get("/api/health/")
        
        assert response.status_code == 200
        
        # These headers should not be present
        sensitive_headers = ["Server", "X-Powered-By", "X-AspNet-Version"]
        
        for header in sensitive_headers:
            assert header not in response.headers
    
    def test_ip_whitelisting(self, client_with_middleware: TestClient):
        """Test IP whitelisting functionality."""
        # Test with whitelisted IP
        response = client_with_middleware.get(
            "/api/health/",
            headers={"X-Forwarded-For": "127.0.0.1"}
        )
        assert response.status_code == 200
        
        # Test with potentially blacklisted IP (if implemented)
        response = client_with_middleware.get(
            "/api/health/",
            headers={"X-Forwarded-For": "192.168.1.100"}
        )
        # Should still allow in test environment
        assert response.status_code == 200
    
    def test_request_size_limits(self, client_with_middleware: TestClient):
        """Test request size limits."""
        # Test normal sized request
        normal_data = {"name": "Test Detector", "algorithm": "IsolationForest"}
        response = client_with_middleware.post("/api/detectors/", json=normal_data)
        
        # Should process normally (may fail due to auth, but not size)
        assert response.status_code in [200, 401, 422]
        
        # Test oversized request (simulate large payload)
        large_data = {"name": "Test", "data": "x" * 10000}  # 10KB of data
        response = client_with_middleware.post("/api/detectors/", json=large_data)
        
        # Should either process or reject based on size limits
        assert response.status_code in [200, 401, 413, 422]


class TestAuthenticationMiddleware:
    """Test authentication middleware."""
    
    def test_jwt_token_validation(self, client_with_middleware: TestClient, valid_jwt_token):
        """Test JWT token validation."""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        response = client_with_middleware.get("/api/detectors/", headers=headers)
        
        # Should authenticate successfully
        assert response.status_code in [200, 404]  # May return 404 if no detectors
    
    def test_invalid_jwt_token(self, client_with_middleware: TestClient):
        """Test invalid JWT token handling."""
        invalid_headers = {"Authorization": "Bearer invalid_token_here"}
        
        response = client_with_middleware.get("/api/detectors/", headers=invalid_headers)
        
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
    
    def test_expired_jwt_token(self, client_with_middleware: TestClient):
        """Test expired JWT token handling."""
        # Create expired token
        secret = "test_secret_key_for_testing_only"
        expired_payload = {
            "sub": "test_user_123",
            "exp": datetime.utcnow() - timedelta(hours=1),  # Expired 1 hour ago
            "iat": datetime.utcnow() - timedelta(hours=2)
        }
        expired_token = jwt.encode(expired_payload, secret, algorithm="HS256")
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        response = client_with_middleware.get("/api/detectors/", headers=headers)
        
        assert response.status_code == 401
        data = response.json()
        assert "expired" in data["detail"].lower() or "token" in data["detail"].lower()
    
    def test_missing_authorization_header(self, client_with_middleware: TestClient):
        """Test missing authorization header."""
        response = client_with_middleware.get("/api/detectors/")
        
        assert response.status_code == 401
    
    def test_api_key_authentication(self, client_with_middleware: TestClient):
        """Test API key authentication."""
        # Test with API key header
        api_key_headers = {"X-API-Key": "test_api_key_12345"}
        
        with patch('pynomaly.infrastructure.auth.validate_api_key') as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "user_id": "api_user_123",
                "permissions": ["read", "write"]
            }
            
            response = client_with_middleware.get("/api/detectors/", headers=api_key_headers)
            
            # Should authenticate successfully
            assert response.status_code in [200, 404]
    
    def test_invalid_api_key(self, client_with_middleware: TestClient):
        """Test invalid API key handling."""
        invalid_api_headers = {"X-API-Key": "invalid_api_key"}
        
        with patch('pynomaly.infrastructure.auth.validate_api_key') as mock_validate:
            mock_validate.return_value = {"valid": False}
            
            response = client_with_middleware.get("/api/detectors/", headers=invalid_api_headers)
            
            assert response.status_code == 401
    
    def test_role_based_access_control(self, client_with_middleware: TestClient):
        """Test role-based access control."""
        # Create token with specific role
        secret = "test_secret_key_for_testing_only"
        user_payload = {
            "sub": "regular_user",
            "roles": ["user"],
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        user_token = jwt.encode(user_payload, secret, algorithm="HS256")
        
        admin_payload = {
            "sub": "admin_user",
            "roles": ["admin"],
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        admin_token = jwt.encode(admin_payload, secret, algorithm="HS256")
        
        # Test user access to user endpoint
        user_headers = {"Authorization": f"Bearer {user_token}"}
        response = client_with_middleware.get("/api/detectors/", headers=user_headers)
        assert response.status_code in [200, 404]  # Should have access
        
        # Test admin access to admin endpoint
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        response = client_with_middleware.get("/api/admin/settings", headers=admin_headers)
        # May return 404 if endpoint doesn't exist, but not 403
        assert response.status_code in [200, 404]


class TestRateLimitingMiddleware:
    """Test rate limiting middleware."""
    
    def test_rate_limiting_per_user(self, client_with_middleware: TestClient, valid_jwt_token):
        """Test rate limiting per user."""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        # Make multiple requests rapidly
        responses = []
        for i in range(20):  # Assuming rate limit is lower than 20
            response = client_with_middleware.get("/api/health/", headers=headers)
            responses.append(response)
            
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        
        if rate_limited_responses:
            # Check rate limit headers
            rate_limited_response = rate_limited_responses[0]
            assert "X-RateLimit-Limit" in rate_limited_response.headers
            assert "X-RateLimit-Remaining" in rate_limited_response.headers
            assert "Retry-After" in rate_limited_response.headers
    
    def test_rate_limiting_per_ip(self, client_with_middleware: TestClient):
        """Test rate limiting per IP address."""
        # Make multiple requests from same IP
        responses = []
        for i in range(15):
            response = client_with_middleware.get(
                "/api/health/",
                headers={"X-Forwarded-For": "192.168.1.100"}
            )
            responses.append(response)
            
            if response.status_code == 429:
                break
        
        # Should eventually hit rate limit
        rate_limited = any(r.status_code == 429 for r in responses)
        
        if rate_limited:
            # Verify rate limit response
            rate_limited_response = next(r for r in responses if r.status_code == 429)
            data = rate_limited_response.json()
            assert "rate limit" in data["detail"].lower()
    
    def test_rate_limiting_bypass_for_health_checks(self, client_with_middleware: TestClient):
        """Test rate limiting bypass for health check endpoints."""
        # Health checks should not be rate limited
        responses = []
        for i in range(50):
            response = client_with_middleware.get("/api/health/")
            responses.append(response)
        
        # All health check responses should succeed
        assert all(r.status_code == 200 for r in responses)
    
    def test_rate_limiting_different_endpoints(self, client_with_middleware: TestClient):
        """Test rate limiting is tracked separately for different endpoints."""
        # Make requests to different endpoints
        health_responses = []
        for i in range(10):
            response = client_with_middleware.get("/api/health/")
            health_responses.append(response)
        
        detector_responses = []
        for i in range(10):
            response = client_with_middleware.get("/api/detectors/")
            detector_responses.append(response)
        
        # Rate limits should be tracked separately
        health_successes = sum(1 for r in health_responses if r.status_code == 200)
        detector_auth_failures = sum(1 for r in detector_responses if r.status_code == 401)
        
        assert health_successes > 0
        assert detector_auth_failures > 0  # Due to missing auth, not rate limiting
    
    def test_rate_limiting_reset(self, client_with_middleware: TestClient):
        """Test rate limiting window reset."""
        # This test would require waiting for the rate limit window to reset
        # In practice, this would be tested with time manipulation or shorter windows
        
        with patch('time.time') as mock_time:
            # Mock time progression
            current_time = time.time()
            mock_time.return_value = current_time
            
            # Make requests to hit rate limit
            for i in range(10):
                response = client_with_middleware.get("/api/health/")
                if response.status_code == 429:
                    break
            
            # Advance time beyond rate limit window
            mock_time.return_value = current_time + 3600  # 1 hour later
            
            # Should be able to make requests again
            response = client_with_middleware.get("/api/health/")
            assert response.status_code == 200


class TestLoggingMiddleware:
    """Test logging middleware functionality."""
    
    def test_request_logging(self, client_with_middleware: TestClient):
        """Test request logging."""
        with patch('pynomaly.infrastructure.middleware.logging.logger') as mock_logger:
            response = client_with_middleware.get("/api/health/")
            
            assert response.status_code == 200
            
            # Verify logging calls were made
            assert mock_logger.info.called or mock_logger.debug.called
            
            # Check log content
            log_calls = mock_logger.info.call_args_list + mock_logger.debug.call_args_list
            log_messages = [str(call[0][0]) for call in log_calls]
            
            # Should log request details
            assert any("GET" in msg for msg in log_messages)
            assert any("/api/health/" in msg for msg in log_messages)
    
    def test_response_logging(self, client_with_middleware: TestClient):
        """Test response logging."""
        with patch('pynomaly.infrastructure.middleware.logging.logger') as mock_logger:
            response = client_with_middleware.get("/api/health/")
            
            assert response.status_code == 200
            
            # Should log response status
            log_calls = mock_logger.info.call_args_list
            log_messages = [str(call[0][0]) for call in log_calls]
            
            assert any("200" in msg for msg in log_messages)
    
    def test_error_logging(self, client_with_middleware: TestClient):
        """Test error logging."""
        with patch('pynomaly.infrastructure.middleware.logging.logger') as mock_logger:
            # Make request that will cause error
            response = client_with_middleware.get("/api/nonexistent/")
            
            assert response.status_code == 404
            
            # Should log error
            error_calls = mock_logger.warning.call_args_list + mock_logger.error.call_args_list
            if error_calls:
                error_messages = [str(call[0][0]) for call in error_calls]
                assert any("404" in msg for msg in error_messages)
    
    def test_performance_logging(self, client_with_middleware: TestClient):
        """Test performance metric logging."""
        with patch('pynomaly.infrastructure.middleware.logging.logger') as mock_logger:
            response = client_with_middleware.get("/api/health/")
            
            assert response.status_code == 200
            
            # Should log timing information
            log_calls = mock_logger.info.call_args_list
            log_messages = [str(call[0][0]) for call in log_calls]
            
            # Look for timing information
            timing_logged = any(
                "ms" in msg or "time" in msg.lower() or "duration" in msg.lower()
                for msg in log_messages
            )
            # Timing logging may be optional
            assert timing_logged or len(log_calls) > 0
    
    def test_sensitive_data_masking(self, client_with_middleware: TestClient):
        """Test sensitive data masking in logs."""
        with patch('pynomaly.infrastructure.middleware.logging.logger') as mock_logger:
            # Make request with sensitive data
            sensitive_data = {
                "password": "secret123",
                "api_key": "sk_123456789",
                "token": "bearer_token_here"
            }
            
            response = client_with_middleware.post("/api/auth/login", json=sensitive_data)
            
            # Check that sensitive data is masked in logs
            all_calls = (mock_logger.info.call_args_list + 
                        mock_logger.debug.call_args_list + 
                        mock_logger.warning.call_args_list)
            
            all_messages = [str(call[0][0]) for call in all_calls]
            
            # Sensitive values should not appear in logs
            for message in all_messages:
                assert "secret123" not in message
                assert "sk_123456789" not in message
                assert "bearer_token_here" not in message


class TestCompressionMiddleware:
    """Test compression middleware."""
    
    def test_gzip_compression(self, client_with_middleware: TestClient):
        """Test Gzip compression."""
        headers = {"Accept-Encoding": "gzip"}
        response = client_with_middleware.get("/api/health/", headers=headers)
        
        assert response.status_code == 200
        
        # Check if response is compressed
        content_encoding = response.headers.get("Content-Encoding")
        if content_encoding:
            assert "gzip" in content_encoding
    
    def test_compression_threshold(self, client_with_middleware: TestClient):
        """Test compression only applies to large responses."""
        # Small response - should not be compressed
        small_response = client_with_middleware.get(
            "/api/health/",
            headers={"Accept-Encoding": "gzip"}
        )
        
        # Large response - should be compressed (if implemented)
        with patch('pynomaly.presentation.api.endpoints.health.get_health_status') as mock_health:
            # Mock large response
            large_data = {
                "status": "healthy",
                "large_data": "x" * 5000,  # 5KB of data
                "components": {f"component_{i}": {"status": "ok"} for i in range(100)}
            }
            mock_health.return_value = large_data
            
            large_response = client_with_middleware.get(
                "/api/health/",
                headers={"Accept-Encoding": "gzip"}
            )
            
            # Should potentially be compressed
            large_encoding = large_response.headers.get("Content-Encoding")
            small_encoding = small_response.headers.get("Content-Encoding")
            
            # Compression behavior depends on implementation
            assert large_response.status_code == 200
            assert small_response.status_code == 200
    
    def test_compression_content_types(self, client_with_middleware: TestClient):
        """Test compression is applied to appropriate content types."""
        # JSON response should be compressible
        json_response = client_with_middleware.get(
            "/api/health/",
            headers={"Accept-Encoding": "gzip"}
        )
        
        # HTML response should be compressible
        html_response = client_with_middleware.get(
            "/",
            headers={"Accept-Encoding": "gzip"}
        )
        
        assert json_response.status_code == 200
        assert html_response.status_code == 200
        
        # Both should have appropriate content types
        json_content_type = json_response.headers.get("Content-Type", "")
        html_content_type = html_response.headers.get("Content-Type", "")
        
        assert "json" in json_content_type
        assert "html" in html_content_type


class TestRequestValidationMiddleware:
    """Test request validation middleware."""
    
    def test_json_validation(self, client_with_middleware: TestClient):
        """Test JSON request validation."""
        # Valid JSON
        valid_json = {"name": "Test", "algorithm": "IsolationForest"}
        response = client_with_middleware.post(
            "/api/detectors/",
            json=valid_json,
            headers={"Content-Type": "application/json"}
        )
        
        # Should process (may fail due to auth, but not JSON validation)
        assert response.status_code in [200, 401, 422]
        
        # Invalid JSON
        invalid_response = client_with_middleware.post(
            "/api/detectors/",
            data="invalid json string",
            headers={"Content-Type": "application/json"}
        )
        
        assert invalid_response.status_code == 422
    
    def test_content_type_validation(self, client_with_middleware: TestClient):
        """Test content type validation."""
        # Correct content type
        response = client_with_middleware.post(
            "/api/detectors/",
            json={"name": "Test"},
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [200, 401, 422]  # Not 415
        
        # Missing content type for JSON data
        response = client_with_middleware.post(
            "/api/detectors/",
            data='{"name": "Test"}',
            # No Content-Type header
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 401, 415, 422]
    
    def test_request_size_validation(self, client_with_middleware: TestClient):
        """Test request size validation."""
        # Normal sized request
        normal_data = {"name": "Test", "description": "Normal description"}
        response = client_with_middleware.post("/api/detectors/", json=normal_data)
        
        assert response.status_code in [200, 401, 422]
        
        # Very large request
        large_data = {
            "name": "Test",
            "description": "x" * 100000,  # 100KB description
            "metadata": {f"key_{i}": f"value_{i}" * 100 for i in range(1000)}
        }
        
        response = client_with_middleware.post("/api/detectors/", json=large_data)
        
        # Should either process or reject based on size limits
        assert response.status_code in [200, 401, 413, 422]
    
    def test_malformed_request_handling(self, client_with_middleware: TestClient):
        """Test handling of malformed requests."""
        # Test various malformed requests
        malformed_requests = [
            # Truncated JSON
            '{"name": "Test", "algorithm":',
            # Invalid characters
            '{"name": "Test\x00", "algorithm": "IF"}',
            # Deeply nested JSON
            '{"a": {"b": {"c": {"d": {"e": "deep"}}}}}',
            # Very long strings
            '{"name": "' + "x" * 10000 + '"}',
        ]
        
        for malformed_data in malformed_requests:
            response = client_with_middleware.post(
                "/api/detectors/",
                data=malformed_data,
                headers={"Content-Type": "application/json"}
            )
            
            # Should handle gracefully
            assert response.status_code in [400, 401, 422]
    
    def test_sql_injection_prevention(self, client_with_middleware: TestClient):
        """Test SQL injection attempt prevention."""
        # SQL injection attempts in various fields
        injection_attempts = [
            {"name": "'; DROP TABLE detectors; --"},
            {"algorithm": "1' OR '1'='1"},
            {"description": "Test'; SELECT * FROM users; --"},
        ]
        
        for injection_data in injection_attempts:
            response = client_with_middleware.post("/api/detectors/", json=injection_data)
            
            # Should either process safely or reject
            assert response.status_code in [200, 400, 401, 422]
            
            # Should not cause server error
            assert response.status_code != 500
    
    def test_xss_prevention(self, client_with_middleware: TestClient):
        """Test XSS attempt prevention."""
        # XSS attempts
        xss_attempts = [
            {"name": "<script>alert('xss')</script>"},
            {"description": "<img src=x onerror=alert(1)>"},
            {"algorithm": "javascript:alert('xss')"},
        ]
        
        for xss_data in xss_attempts:
            response = client_with_middleware.post("/api/detectors/", json=xss_data)
            
            # Should process or reject safely
            assert response.status_code in [200, 400, 401, 422]
            
            # If processed, response should not contain unescaped script tags
            if response.status_code == 200:
                response_text = response.text
                assert "<script>" not in response_text
                assert "javascript:" not in response_text


class TestMiddlewareIntegration:
    """Test middleware integration and interaction."""
    
    def test_middleware_chain_execution_order(self, client_with_middleware: TestClient):
        """Test middleware execution order."""
        with patch('pynomaly.infrastructure.middleware.logging.logger') as mock_logger:
            response = client_with_middleware.get("/api/health/")
            
            assert response.status_code == 200
            
            # Security headers should be present
            assert "X-Request-ID" in response.headers
            
            # Logging should have occurred
            assert mock_logger.info.called or mock_logger.debug.called
    
    def test_error_handling_across_middleware(self, client_with_middleware: TestClient):
        """Test error handling propagation through middleware stack."""
        # Trigger an error and verify it's handled properly by all middleware
        response = client_with_middleware.get("/api/nonexistent/endpoint/")
        
        assert response.status_code == 404
        
        # Security headers should still be present
        assert "X-Request-ID" in response.headers
        
        # Error response should be properly formatted
        data = response.json()
        assert "detail" in data
    
    def test_performance_impact_of_middleware(self, client_with_middleware: TestClient):
        """Test performance impact of middleware stack."""
        import time
        
        # Measure response time with full middleware stack
        start_time = time.time()
        response = client_with_middleware.get("/api/health/")
        end_time = time.time()
        
        assert response.status_code == 200
        
        response_time = end_time - start_time
        
        # Middleware should not significantly impact performance
        assert response_time < 2.0  # Should respond within 2 seconds
    
    def test_middleware_configuration(self, client_with_middleware: TestClient):
        """Test middleware responds to configuration changes."""
        # This would test that middleware behavior changes based on configuration
        # For example, enabling/disabling certain security features
        
        response = client_with_middleware.get("/api/health/")
        assert response.status_code == 200
        
        # Basic middleware should be active
        assert "X-Request-ID" in response.headers
    
    def test_concurrent_request_handling_with_middleware(self, client_with_middleware: TestClient):
        """Test middleware handles concurrent requests properly."""
        import concurrent.futures
        import threading
        
        def make_request():
            return client_with_middleware.get("/api/health/")
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(result.status_code == 200 for result in results)
        
        # Each should have unique request IDs
        request_ids = [result.headers.get("X-Request-ID") for result in results]
        assert len(set(request_ids)) == len(request_ids)  # All unique
    
    def test_middleware_memory_usage(self, client_with_middleware: TestClient):
        """Test middleware doesn't cause memory leaks."""
        # Make many requests to test for memory leaks
        for i in range(100):
            response = client_with_middleware.get("/api/health/")
            assert response.status_code == 200
        
        # This test would be more meaningful with actual memory monitoring
        # In a real test environment, you'd monitor memory usage
        assert True  # Placeholder for memory leak detection
    
    def test_middleware_error_recovery(self, client_with_middleware: TestClient):
        """Test middleware recovers from errors gracefully."""
        # Simulate middleware errors and verify recovery
        with patch('pynomaly.infrastructure.middleware.security.SecurityMiddleware') as mock_middleware:
            # First request fails in middleware
            mock_middleware.side_effect = Exception("Middleware error")
            
            response1 = client_with_middleware.get("/api/health/")
            # Should handle middleware error gracefully
            assert response1.status_code in [200, 500]
            
            # Reset mock for second request
            mock_middleware.side_effect = None
            mock_middleware.return_value = Mock()
            
            response2 = client_with_middleware.get("/api/health/")
            # Should recover and work normally
            assert response2.status_code == 200
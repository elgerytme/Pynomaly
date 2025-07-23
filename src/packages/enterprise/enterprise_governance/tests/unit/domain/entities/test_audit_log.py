"""
Unit tests for AuditLog domain entity.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID

from enterprise_governance.domain.entities.audit_log import (
    AuditLog, AuditQuery, AuditStatistics, AuditRetentionPolicy,
    AuditEventType, AuditSeverity, AuditStatus
)


class TestAuditLog:
    """Test cases for AuditLog entity."""
    
    def test_audit_log_creation_basic(self):
        """Test basic audit log creation."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login successful",
            source_system="auth_service",
            environment="production"
        )
        
        assert isinstance(audit_log.id, UUID)
        assert audit_log.event_type == AuditEventType.USER_LOGIN
        assert audit_log.category == "authentication"
        assert audit_log.severity == AuditSeverity.INFO
        assert audit_log.message == "User login successful"
        assert audit_log.source_system == "auth_service"
        assert audit_log.environment == "production"
        assert audit_log.status == AuditStatus.PENDING
        assert audit_log.details == {}
        assert audit_log.compliance_tags == []
        assert audit_log.encrypted is False
        assert audit_log.checksum is None
        
    def test_audit_log_creation_comprehensive(self):
        """Test comprehensive audit log creation."""
        tenant_id = uuid4()
        user_id = uuid4()
        session_id = uuid4()
        resource_id = uuid4()
        details = {"action": "update", "field": "email"}
        old_values = {"email": "old@example.com"}
        new_values = {"email": "new@example.com"}
        compliance_tags = ["GDPR", "SOC2"]
        
        audit_log = AuditLog(
            event_type=AuditEventType.USER_UPDATED,
            category="user_management",
            severity=AuditSeverity.MEDIUM,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            message="User email updated",
            details=details,
            resource_type="user",
            resource_id=resource_id,
            resource_name="John Doe",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0...",
            request_id="req_123456",
            operation="UPDATE",
            status_code=200,
            response_time_ms=150.5,
            old_values=old_values,
            new_values=new_values,
            compliance_tags=compliance_tags,
            retention_policy="7_years",
            source_system="user_service",
            source_component="user_controller",
            environment="production",
            encrypted=True
        )
        
        assert audit_log.tenant_id == tenant_id
        assert audit_log.user_id == user_id
        assert audit_log.session_id == session_id
        assert audit_log.resource_id == resource_id
        assert audit_log.resource_name == "John Doe"
        assert audit_log.ip_address == "192.168.1.100"
        assert audit_log.user_agent == "Mozilla/5.0..."
        assert audit_log.request_id == "req_123456"
        assert audit_log.operation == "UPDATE"
        assert audit_log.status_code == 200
        assert audit_log.response_time_ms == 150.5
        assert audit_log.old_values == old_values
        assert audit_log.new_values == new_values
        assert audit_log.compliance_tags == compliance_tags
        assert audit_log.retention_policy == "7_years"
        assert audit_log.source_component == "user_controller"
        assert audit_log.encrypted is True
        
    def test_compliance_tags_string_conversion(self):
        """Test compliance tags string is converted to list."""
        audit_log = AuditLog(
            event_type=AuditEventType.DATA_ACCESSED,
            category="data_access",
            severity=AuditSeverity.INFO,
            message="Data accessed",
            source_system="data_service",
            environment="production",
            compliance_tags="GDPR"
        )
        assert audit_log.compliance_tags == ["GDPR"]
        
    def test_compliance_tags_none_conversion(self):
        """Test compliance tags None is converted to empty list."""
        audit_log = AuditLog(
            event_type=AuditEventType.DATA_ACCESSED,
            category="data_access",
            severity=AuditSeverity.INFO,
            message="Data accessed",
            source_system="data_service",
            environment="production",
            compliance_tags=None
        )
        assert audit_log.compliance_tags == []
        
    def test_ip_address_validation_valid(self):
        """Test valid IP address validation."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login",
            source_system="auth_service",
            environment="production",
            ip_address="192.168.1.1"
        )
        assert audit_log.ip_address == "192.168.1.1"
        
    def test_ip_address_validation_unknown(self):
        """Test 'unknown' IP address is valid."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login",
            source_system="auth_service",
            environment="production",
            ip_address="unknown"
        )
        assert audit_log.ip_address == "unknown"
        
    def test_ip_address_validation_ipv6(self):
        """Test IPv6 address validation."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login",
            source_system="auth_service",
            environment="production",
            ip_address="2001:0db8:85a3:0000:0000:8a2e:0370:7334"
        )
        assert audit_log.ip_address == "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
        
    def test_ip_address_validation_invalid(self):
        """Test invalid IP address validation."""
        with pytest.raises(ValueError, match="Invalid IP address format"):
            AuditLog(
                event_type=AuditEventType.USER_LOGIN,
                category="authentication",
                severity=AuditSeverity.INFO,
                message="User login",
                source_system="auth_service",
                environment="production",
                ip_address="invalid_ip"
            )
            
    def test_add_compliance_tag(self):
        """Test adding compliance tag."""
        audit_log = AuditLog(
            event_type=AuditEventType.DATA_ACCESSED,
            category="data_access",
            severity=AuditSeverity.INFO,
            message="Data accessed",
            source_system="data_service",
            environment="production"
        )
        
        audit_log.add_compliance_tag("GDPR")
        assert "GDPR" in audit_log.compliance_tags
        
        # Adding duplicate should not add again
        audit_log.add_compliance_tag("GDPR")
        assert audit_log.compliance_tags.count("GDPR") == 1
        
    def test_remove_compliance_tag(self):
        """Test removing compliance tag."""
        audit_log = AuditLog(
            event_type=AuditEventType.DATA_ACCESSED,
            category="data_access",
            severity=AuditSeverity.INFO,
            message="Data accessed",
            source_system="data_service",
            environment="production",
            compliance_tags=["GDPR", "SOC2"]
        )
        
        audit_log.remove_compliance_tag("GDPR")
        assert "GDPR" not in audit_log.compliance_tags
        assert "SOC2" in audit_log.compliance_tags
        
        # Removing non-existent tag should not error
        audit_log.remove_compliance_tag("HIPAA")
        
    def test_is_security_event(self):
        """Test security event identification."""
        security_log = AuditLog(
            event_type=AuditEventType.SECURITY_BREACH_DETECTED,
            category="security",
            severity=AuditSeverity.CRITICAL,
            message="Security breach detected",
            source_system="security_service",
            environment="production"
        )
        assert security_log.is_security_event() is True
        
        non_security_log = AuditLog(
            event_type=AuditEventType.USER_CREATED,
            category="user_management",
            severity=AuditSeverity.INFO,
            message="User created",
            source_system="user_service",
            environment="production"
        )
        assert non_security_log.is_security_event() is False
        
    def test_is_compliance_event(self):
        """Test compliance event identification."""
        compliance_log = AuditLog(
            event_type=AuditEventType.COMPLIANCE_AUDIT_STARTED,
            category="compliance",
            severity=AuditSeverity.INFO,
            message="Compliance audit started",
            source_system="compliance_service",
            environment="production"
        )
        assert compliance_log.is_compliance_event() is True
        
        non_compliance_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login",
            source_system="auth_service",
            environment="production"
        )
        assert non_compliance_log.is_compliance_event() is False
        
    def test_is_critical(self):
        """Test critical severity check."""
        critical_log = AuditLog(
            event_type=AuditEventType.SECURITY_BREACH_DETECTED,
            category="security",
            severity=AuditSeverity.CRITICAL,
            message="Critical security event",
            source_system="security_service",
            environment="production"
        )
        assert critical_log.is_critical() is True
        
        non_critical_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login",
            source_system="auth_service",
            environment="production"
        )
        assert non_critical_log.is_critical() is False
        
    def test_requires_immediate_attention(self):
        """Test immediate attention requirement check."""
        # Critical severity should require attention
        critical_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.CRITICAL,
            message="Critical event",
            source_system="auth_service",
            environment="production"
        )
        assert critical_log.requires_immediate_attention() is True
        
        # High severity should require attention
        high_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.HIGH,
            message="High severity event",
            source_system="auth_service",
            environment="production"
        )
        assert high_log.requires_immediate_attention() is True
        
        # Security events should require attention
        security_log = AuditLog(
            event_type=AuditEventType.SECURITY_BREACH_DETECTED,
            category="security",
            severity=AuditSeverity.MEDIUM,
            message="Security breach",
            source_system="security_service",
            environment="production"
        )
        assert security_log.requires_immediate_attention() is True
        
        # Low severity non-security events should not require attention
        low_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.LOW,
            message="Low severity event",
            source_system="auth_service",
            environment="production"
        )
        assert low_log.requires_immediate_attention() is False
        
    def test_sanitize_for_export(self):
        """Test audit log sanitization for export."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_UPDATED,
            category="user_management",
            severity=AuditSeverity.INFO,
            message="User updated",
            source_system="user_service",
            environment="production",
            details={"sensitive": "data"},
            old_values={"password": "old_hash"},
            new_values={"password": "new_hash"},
            ip_address="192.168.1.100"
        )
        
        sanitized = audit_log.sanitize_for_export()
        
        assert sanitized["details"] == "***REDACTED***"
        assert sanitized["old_values"] == "***REDACTED***"
        assert sanitized["new_values"] == "***REDACTED***"
        assert sanitized["ip_address"] == "192.168.xxx.xxx"
        assert sanitized["message"] == "User updated"  # Non-sensitive field preserved
        
    def test_sanitize_for_export_no_ip_redaction(self):
        """Test export sanitization without IP redaction for non-IPv4."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login",
            source_system="auth_service",
            environment="production",
            ip_address="unknown"
        )
        
        sanitized = audit_log.sanitize_for_export()
        assert sanitized["ip_address"] == "unknown"  # Should not be redacted
        
    def test_calculate_checksum(self):
        """Test checksum calculation."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login successful",
            source_system="auth_service",
            environment="production",
            tenant_id=uuid4(),
            user_id=uuid4()
        )
        
        checksum = audit_log.calculate_checksum()
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest length
        
        # Same log should produce same checksum
        checksum2 = audit_log.calculate_checksum()
        assert checksum == checksum2
        
    def test_verify_integrity_valid(self):
        """Test integrity verification with valid checksum."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login successful",
            source_system="auth_service",
            environment="production"
        )
        
        # Calculate and store checksum
        audit_log.checksum = audit_log.calculate_checksum()
        
        assert audit_log.verify_integrity() is True
        
    def test_verify_integrity_invalid(self):
        """Test integrity verification with invalid checksum."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login successful",
            source_system="auth_service",
            environment="production"
        )
        
        # Set invalid checksum
        audit_log.checksum = "invalid_checksum"
        
        assert audit_log.verify_integrity() is False
        
    def test_verify_integrity_no_checksum(self):
        """Test integrity verification without checksum."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login successful",
            source_system="auth_service",
            environment="production"
        )
        
        assert audit_log.verify_integrity() is False
        
    def test_mark_processed(self):
        """Test marking audit log as processed."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login successful",
            source_system="auth_service",
            environment="production"
        )
        
        assert audit_log.status == AuditStatus.PENDING
        assert audit_log.processed_at is None
        assert audit_log.checksum is None
        
        audit_log.mark_processed()
        
        assert audit_log.status == AuditStatus.PROCESSED
        assert audit_log.processed_at is not None
        assert audit_log.checksum is not None
        
    def test_archive(self):
        """Test archiving audit log."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_LOGIN,
            category="authentication",
            severity=AuditSeverity.INFO,
            message="User login successful",
            source_system="auth_service",
            environment="production"
        )
        
        audit_log.archive()
        assert audit_log.status == AuditStatus.ARCHIVED
        
    def test_to_dict_with_sensitive(self):
        """Test to_dict with sensitive information."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_UPDATED,
            category="user_management",
            severity=AuditSeverity.INFO,
            message="User updated",
            source_system="user_service",
            environment="production",
            details={"sensitive": "data"}
        )
        
        data = audit_log.to_dict(include_sensitive=True)
        assert data["details"] == {"sensitive": "data"}
        
    def test_to_dict_without_sensitive(self):
        """Test to_dict without sensitive information."""
        audit_log = AuditLog(
            event_type=AuditEventType.USER_UPDATED,
            category="user_management",
            severity=AuditSeverity.INFO,
            message="User updated",
            source_system="user_service",
            environment="production",
            details={"sensitive": "data"}
        )
        
        data = audit_log.to_dict(include_sensitive=False)
        assert data["details"] == "***REDACTED***"


class TestAuditQuery:
    """Test cases for AuditQuery entity."""
    
    def test_audit_query_creation_defaults(self):
        """Test audit query creation with defaults."""
        query = AuditQuery()
        
        assert query.start_time is None
        assert query.end_time is None
        assert query.tenant_id is None
        assert query.user_id is None
        assert query.event_types is None
        assert query.severities is None
        assert query.page == 1
        assert query.page_size == 100
        assert query.sort_by == "timestamp"
        assert query.sort_order == "desc"
        assert query.include_sensitive is False
        assert query.export_format is None
        
    def test_audit_query_creation_comprehensive(self):
        """Test comprehensive audit query creation."""
        tenant_id = uuid4()
        user_id = uuid4()
        resource_id = uuid4()
        session_id = uuid4()
        start_time = datetime.utcnow() - timedelta(days=7)
        end_time = datetime.utcnow()
        
        query = AuditQuery(
            start_time=start_time,
            end_time=end_time,
            tenant_id=tenant_id,
            user_id=user_id,
            resource_type="user",
            resource_id=resource_id,
            event_types=[AuditEventType.USER_LOGIN, AuditEventType.USER_LOGOUT],
            severities=[AuditSeverity.HIGH, AuditSeverity.CRITICAL],
            categories=["authentication", "security"],
            compliance_tags=["GDPR", "SOC2"],
            ip_address="192.168.1.100",
            session_id=session_id,
            request_id="req_123",
            search_term="login failed",
            page=2,
            page_size=50,
            sort_by="severity",
            sort_order="asc",
            include_sensitive=True,
            export_format="csv"
        )
        
        assert query.start_time == start_time
        assert query.end_time == end_time
        assert query.tenant_id == tenant_id
        assert query.user_id == user_id
        assert query.resource_type == "user"
        assert query.resource_id == resource_id
        assert query.event_types == [AuditEventType.USER_LOGIN, AuditEventType.USER_LOGOUT]
        assert query.severities == [AuditSeverity.HIGH, AuditSeverity.CRITICAL]
        assert query.categories == ["authentication", "security"]
        assert query.compliance_tags == ["GDPR", "SOC2"]
        assert query.ip_address == "192.168.1.100"
        assert query.session_id == session_id
        assert query.request_id == "req_123"
        assert query.search_term == "login failed"
        assert query.page == 2
        assert query.page_size == 50
        assert query.sort_by == "severity"
        assert query.sort_order == "asc"
        assert query.include_sensitive is True
        assert query.export_format == "csv"
        
    def test_sort_order_validation_valid(self):
        """Test valid sort order validation."""
        query = AuditQuery(sort_order="asc")
        assert query.sort_order == "asc"
        
        query = AuditQuery(sort_order="desc")
        assert query.sort_order == "desc"
        
    def test_sort_order_validation_invalid(self):
        """Test invalid sort order validation."""
        with pytest.raises(ValueError, match='Sort order must be "asc" or "desc"'):
            AuditQuery(sort_order="invalid")
            
    def test_export_format_validation_valid(self):
        """Test valid export format validation."""
        for format_type in ["json", "csv", "xlsx", "pdf"]:
            query = AuditQuery(export_format=format_type)
            assert query.export_format == format_type
            
    def test_export_format_validation_invalid(self):
        """Test invalid export format validation."""
        with pytest.raises(ValueError, match="Export format must be one of"):
            AuditQuery(export_format="invalid")
            
    def test_page_validation(self):
        """Test page number validation."""
        query = AuditQuery(page=1)
        assert query.page == 1
        
        with pytest.raises(ValueError):
            AuditQuery(page=0)
            
    def test_page_size_validation(self):
        """Test page size validation."""
        query = AuditQuery(page_size=100)
        assert query.page_size == 100
        
        with pytest.raises(ValueError):
            AuditQuery(page_size=0)
            
        with pytest.raises(ValueError):
            AuditQuery(page_size=1001)


class TestAuditStatistics:
    """Test cases for AuditStatistics entity."""
    
    def test_audit_statistics_creation(self):
        """Test audit statistics creation."""
        start_time = datetime.utcnow() - timedelta(days=30)
        end_time = datetime.utcnow()
        
        stats = AuditStatistics(
            start_time=start_time,
            end_time=end_time
        )
        
        assert stats.start_time == start_time
        assert stats.end_time == end_time
        assert stats.total_events == 0
        assert stats.unique_users == 0
        assert stats.unique_tenants == 0
        assert stats.events_by_type == {}
        assert stats.events_by_severity == {}
        assert stats.events_by_category == {}
        assert stats.security_events == 0
        assert stats.failed_logins == 0
        assert stats.locked_accounts == 0
        assert stats.compliance_events == 0
        assert stats.policy_violations == 0
        assert stats.api_calls == 0
        assert stats.data_access_events == 0
        assert stats.data_export_events == 0
        assert stats.average_response_time is None
        assert stats.error_rate is None
        assert stats.top_users_by_activity == []
        assert stats.top_resources_accessed == []
        assert stats.top_ip_addresses == []
        
    def test_audit_statistics_with_data(self):
        """Test audit statistics with populated data."""
        start_time = datetime.utcnow() - timedelta(days=30)
        end_time = datetime.utcnow()
        
        stats = AuditStatistics(
            start_time=start_time,
            end_time=end_time,
            total_events=1000,
            unique_users=50,
            unique_tenants=5,
            events_by_type={"user.login": 300, "user.logout": 250},
            events_by_severity={"info": 800, "high": 200},
            events_by_category={"authentication": 550, "user_management": 450},
            security_events=25,
            failed_logins=15,
            locked_accounts=3,
            compliance_events=10,
            policy_violations=2,
            api_calls=5000,
            data_access_events=200,
            data_export_events=50,
            average_response_time=125.5,
            error_rate=2.5,
            top_users_by_activity=[{"user_id": "user1", "count": 100}],
            top_resources_accessed=[{"resource": "dataset1", "count": 50}],
            top_ip_addresses=[{"ip": "192.168.1.1", "count": 200}]
        )
        
        assert stats.total_events == 1000
        assert stats.unique_users == 50
        assert stats.unique_tenants == 5
        assert stats.events_by_type == {"user.login": 300, "user.logout": 250}
        assert stats.events_by_severity == {"info": 800, "high": 200}
        assert stats.events_by_category == {"authentication": 550, "user_management": 450}
        assert stats.security_events == 25
        assert stats.failed_logins == 15
        assert stats.locked_accounts == 3
        assert stats.compliance_events == 10
        assert stats.policy_violations == 2
        assert stats.api_calls == 5000
        assert stats.data_access_events == 200
        assert stats.data_export_events == 50
        assert stats.average_response_time == 125.5
        assert stats.error_rate == 2.5
        assert len(stats.top_users_by_activity) == 1
        assert len(stats.top_resources_accessed) == 1
        assert len(stats.top_ip_addresses) == 1


class TestAuditRetentionPolicy:
    """Test cases for AuditRetentionPolicy entity."""
    
    def test_retention_policy_creation(self):
        """Test retention policy creation."""
        policy = AuditRetentionPolicy(
            name="Standard Retention Policy",
            description="Standard audit log retention policy",
            default_retention_days=2555,  # 7 years
            archive_after_days=365
        )
        
        assert isinstance(policy.id, UUID)
        assert policy.name == "Standard Retention Policy"
        assert policy.description == "Standard audit log retention policy"
        assert policy.default_retention_days == 2555
        assert policy.retention_rules == {}
        assert policy.compliance_frameworks == []
        assert policy.legal_hold_enabled is False
        assert policy.archive_enabled is True
        assert policy.archive_after_days == 365
        assert policy.archive_location is None
        assert policy.permanent_delete_after_days is None
        assert policy.require_approval_for_deletion is True
        assert policy.is_active is True
        
    def test_retention_policy_comprehensive(self):
        """Test comprehensive retention policy creation."""
        retention_rules = {
            "user.login": 90,
            "security.breach_detected": 3650  # 10 years
        }
        compliance_frameworks = ["GDPR", "SOC2", "HIPAA"]
        
        policy = AuditRetentionPolicy(
            name="GDPR Compliance Policy",
            description="GDPR compliant retention policy",
            default_retention_days=2555,
            retention_rules=retention_rules,
            compliance_frameworks=compliance_frameworks,
            legal_hold_enabled=True,
            archive_enabled=True,
            archive_after_days=730,
            archive_location="s3://audit-archive/",
            permanent_delete_after_days=3650,
            require_approval_for_deletion=True
        )
        
        assert policy.retention_rules == retention_rules
        assert policy.compliance_frameworks == compliance_frameworks
        assert policy.legal_hold_enabled is True
        assert policy.archive_location == "s3://audit-archive/"
        assert policy.permanent_delete_after_days == 3650
        
    def test_get_retention_days_default(self):
        """Test getting retention days with default value."""
        policy = AuditRetentionPolicy(
            name="Test Policy",
            description="Test policy",
            default_retention_days=365,
            archive_after_days=30
        )
        
        # Should return default for non-configured event type
        retention_days = policy.get_retention_days(AuditEventType.USER_LOGIN)
        assert retention_days == 365
        
    def test_get_retention_days_specific(self):
        """Test getting retention days with specific rule."""
        retention_rules = {
            AuditEventType.SECURITY_BREACH_DETECTED: 3650
        }
        
        policy = AuditRetentionPolicy(
            name="Test Policy",
            description="Test policy",
            default_retention_days=365,
            retention_rules=retention_rules,
            archive_after_days=30
        )
        
        # Should return specific rule for configured event type
        retention_days = policy.get_retention_days(AuditEventType.SECURITY_BREACH_DETECTED)
        assert retention_days == 3650
        
        # Should return default for non-configured event type
        retention_days = policy.get_retention_days(AuditEventType.USER_LOGIN)
        assert retention_days == 365
        
    def test_should_archive_true(self):
        """Test should_archive returns True when conditions met."""
        policy = AuditRetentionPolicy(
            name="Test Policy",
            description="Test policy",
            default_retention_days=365,
            archive_enabled=True,
            archive_after_days=30
        )
        
        assert policy.should_archive(45) is True
        
    def test_should_archive_false_disabled(self):
        """Test should_archive returns False when archiving disabled."""
        policy = AuditRetentionPolicy(
            name="Test Policy",
            description="Test policy",
            default_retention_days=365,
            archive_enabled=False,
            archive_after_days=30
        )
        
        assert policy.should_archive(45) is False
        
    def test_should_archive_false_too_recent(self):
        """Test should_archive returns False when log too recent."""
        policy = AuditRetentionPolicy(
            name="Test Policy",
            description="Test policy",
            default_retention_days=365,
            archive_enabled=True,
            archive_after_days=30
        )
        
        assert policy.should_archive(15) is False
        
    def test_should_delete_true(self):
        """Test should_delete returns True when conditions met."""
        policy = AuditRetentionPolicy(
            name="Test Policy",
            description="Test policy",
            default_retention_days=365,
            archive_after_days=30,
            permanent_delete_after_days=2555,
            legal_hold_enabled=False
        )
        
        assert policy.should_delete(3000) is True
        
    def test_should_delete_false_legal_hold(self):
        """Test should_delete returns False when legal hold enabled."""
        policy = AuditRetentionPolicy(
            name="Test Policy",
            description="Test policy",
            default_retention_days=365,
            archive_after_days=30,
            permanent_delete_after_days=2555,
            legal_hold_enabled=True
        )
        
        assert policy.should_delete(3000) is False
        
    def test_should_delete_false_no_delete_period(self):
        """Test should_delete returns False when no delete period set."""
        policy = AuditRetentionPolicy(
            name="Test Policy",
            description="Test policy",
            default_retention_days=365,
            archive_after_days=30,
            permanent_delete_after_days=None,
            legal_hold_enabled=False
        )
        
        assert policy.should_delete(3000) is False
        
    def test_should_delete_false_too_recent(self):
        """Test should_delete returns False when log too recent."""
        policy = AuditRetentionPolicy(
            name="Test Policy",
            description="Test policy",
            default_retention_days=365,
            archive_after_days=30,
            permanent_delete_after_days=2555,
            legal_hold_enabled=False
        )
        
        assert policy.should_delete(1000) is False
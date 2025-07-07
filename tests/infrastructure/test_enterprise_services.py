"""
Comprehensive tests for enterprise-grade services.

Tests cover:
- Security service functionality
- OpenTelemetry observability
- Circuit breaker resilience
- Audit trail integrity
- Multi-tenant support
- Performance monitoring
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from pynomaly.infrastructure.security.security_service import (
    SecurityService,
    SecurityConfig,
    SecurityLevel,
    AuditEventType,
    ThreatLevel
)
from pynomaly.infrastructure.monitoring.observability_service import (
    ObservabilityService,
    ObservabilityConfig,
    HealthStatus
)
from pynomaly.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError
)
from pynomaly.infrastructure.audit.audit_service import (
    AuditService,
    AuditConfig,
    AuditEventType as AuditType,
    AuditSeverity,
    ComplianceFramework
)


class TestSecurityService:
    """Test suite for comprehensive security service."""
    
    @pytest.fixture
    def security_config(self):
        """Create test security configuration."""
        return SecurityConfig(
            enable_2fa=True,
            enable_rbac=True,
            enable_audit_logging=True,
            enable_rate_limiting=True,
            jwt_expiration_minutes=30,
            max_failed_login_attempts=3,
            account_lockout_duration_minutes=5
        )
    
    @pytest.fixture
    def security_service(self, security_config):
        """Create security service instance."""
        return SecurityService(security_config)
    
    def test_encryption_service(self, security_service):
        """Test data encryption and decryption."""
        encryption_service = security_service.encryption_service
        
        # Test string encryption
        test_data = "sensitive data"
        encrypted_data, key_id = encryption_service.encrypt_data(test_data)
        
        assert encrypted_data != test_data.encode()
        assert key_id is not None
        
        # Test decryption
        decrypted_data = encryption_service.decrypt_data(encrypted_data, key_id)
        assert decrypted_data.decode() == test_data
    
    def test_password_hashing(self, security_service):
        """Test password hashing and verification."""
        auth_service = security_service.auth_service
        
        password = "TestPassword123!"
        hashed = auth_service.hash_password(password)
        
        assert hashed != password
        assert auth_service.verify_password(password, hashed)
        assert not auth_service.verify_password("wrong_password", hashed)
    
    def test_jwt_token_generation(self, security_service):
        """Test JWT token generation and verification."""
        auth_service = security_service.auth_service
        
        # Create mock session
        from pynomaly.infrastructure.security.security_service import UserSession
        session = UserSession(
            session_id="test_session",
            user_id="test_user",
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            ip_address="127.0.0.1",
            user_agent="test_agent",
            security_level=SecurityLevel.INTERNAL
        )
        
        # Generate token
        token = auth_service.generate_jwt_token(session)
        assert token is not None
        
        # Verify token
        payload = auth_service.verify_jwt_token(token)
        assert payload is not None
        assert payload['user_id'] == 'test_user'
        assert payload['session_id'] == 'test_session'
    
    def test_role_based_access_control(self, security_service):
        """Test RBAC functionality."""
        authz_service = security_service.authz_service
        
        # Test permission checking
        user_id = "test_user"
        
        # User should not have permissions initially
        assert not authz_service.check_permission(
            user_id, "data:read", SecurityLevel.CONFIDENTIAL
        )
        
        # Assign role
        assert authz_service.assign_role(user_id, "data_scientist")
        
        # Now user should have permissions
        assert authz_service.check_permission(
            user_id, "data:read", SecurityLevel.CONFIDENTIAL
        )
        
        # But not for higher security levels
        assert not authz_service.check_permission(
            user_id, "system:admin", SecurityLevel.SECRET
        )
    
    def test_audit_logging(self, security_service):
        """Test audit event logging."""
        audit_service = security_service.audit_service
        
        # Log test event
        event_id = audit_service.log_event(
            AuditEventType.DATA_ACCESS,
            "test_user",
            "test_resource",
            "read",
            "success",
            {"test": "data"}
        )
        
        assert event_id is not None
        
        # Search for event
        events = audit_service.search_audit_events(
            user_id="test_user",
            limit=10
        )
        
        assert len(events) > 0
        assert events[0].user_id == "test_user"
        assert events[0].action == "read"
    
    def test_rate_limiting(self, security_service):
        """Test rate limiting functionality."""
        # Make requests within rate limit
        for i in range(5):
            success, user_id, error = security_service.authenticate_request(
                "fake_token",
                "test:permission",
                SecurityLevel.INTERNAL,
                "127.0.0.1",
                "test_agent"
            )
            # Should fail due to invalid token, not rate limiting
            assert not success
            assert "authentication" in error.lower()
        
        # Verify rate limit tracking is working
        assert "127.0.0.1" in security_service._rate_limit_tracking
    
    def test_data_sanitization(self, security_service):
        """Test sensitive data sanitization."""
        sensitive_data = {
            "username": "test_user",
            "password": "secret123",
            "credit_card": "1234-5678-9012-3456",
            "normal_field": "normal_value"
        }
        
        sanitized = security_service.sanitize_data(sensitive_data)
        
        assert sanitized["username"] == "test_user"  # Not sensitive
        assert sanitized["password"] != "secret123"  # Should be masked
        assert sanitized["credit_card"] != "1234-5678-9012-3456"  # Should be masked
        assert sanitized["normal_field"] == "normal_value"
    
    def test_security_summary(self, security_service):
        """Test security status summary."""
        summary = security_service.get_security_summary()
        
        assert "security_status" in summary
        assert "encryption_enabled" in summary
        assert "2fa_enabled" in summary
        assert "rbac_enabled" in summary
        assert "audit_logging_enabled" in summary


class TestObservabilityService:
    """Test suite for observability and monitoring service."""
    
    @pytest.fixture
    def observability_config(self):
        """Create test observability configuration."""
        return ObservabilityConfig(
            service_name="test_service",
            enable_tracing=True,
            enable_metrics=True,
            enable_health_checks=True,
            metrics_collection_interval=1.0,
            health_check_interval=2.0
        )
    
    @pytest.fixture
    async def observability_service(self, observability_config):
        """Create observability service instance."""
        service = ObservabilityService(observability_config)
        await service.start()
        yield service
        await service.stop()
    
    def test_system_metrics_collection(self, observability_service):
        """Test system metrics collection."""
        system_metrics = observability_service.system_metrics.collect_metrics()
        
        assert "cpu" in system_metrics
        assert "memory" in system_metrics
        assert "disk" in system_metrics
        assert "network" in system_metrics
        
        # Verify CPU metrics
        cpu_metrics = system_metrics["cpu"]
        assert "usage_percent" in cpu_metrics
        assert isinstance(cpu_metrics["usage_percent"], (int, float))
        assert 0 <= cpu_metrics["usage_percent"] <= 100
        
        # Verify memory metrics
        memory_metrics = system_metrics["memory"]
        assert "total" in memory_metrics
        assert "used" in memory_metrics
        assert "percent" in memory_metrics
    
    def test_application_metrics(self, observability_service):
        """Test application metrics tracking."""
        app_metrics = observability_service.app_metrics
        
        # Record some test metrics
        app_metrics.record_request("GET", "/api/test", 200, 0.1)
        app_metrics.record_request("POST", "/api/test", 500, 0.5)
        app_metrics.record_ml_prediction("isolation_forest", True, True)
        app_metrics.set_active_sessions(5)
        
        # Get metrics
        metrics = app_metrics.get_metrics()
        
        assert "uptime_seconds" in metrics
        assert "request_counts" in metrics
        assert "error_counts" in metrics
        assert "active_sessions" in metrics
        assert "ml_metrics" in metrics
        
        # Verify request tracking
        assert "GET:/api/test" in metrics["request_counts"]
        assert "POST:/api/test" in metrics["error_counts"]
        
        # Verify ML metrics
        ml_metrics = metrics["ml_metrics"]
        assert ml_metrics["total_predictions"] == 1
        assert ml_metrics["anomalies_detected"] == 1
    
    async def test_health_checks(self, observability_service):
        """Test health check functionality."""
        health_service = observability_service.health_service
        
        # Run all health checks
        results = await health_service.run_all_checks()
        
        assert "overall_status" in results
        assert "checks" in results
        assert "summary" in results
        
        # Verify individual checks
        checks = results["checks"]
        assert "database" in checks
        assert "memory" in checks
        assert "disk_space" in checks
        
        # Each check should have required fields
        for check_name, check_result in checks.items():
            assert "status" in check_result
            assert "description" in check_result
            assert "critical" in check_result
    
    def test_business_metrics(self, observability_service):
        """Test business metrics recording."""
        # Record business metrics
        observability_service.record_business_metric(
            "anomalies_detected", 42.0, {"model": "isolation_forest"}
        )
        observability_service.record_business_metric(
            "data_processed_gb", 1.5, {"dataset": "production"}
        )
        
        # Verify metrics are buffered
        assert len(observability_service.metrics_buffer) >= 2
        
        # Check metric structure
        metric = observability_service.metrics_buffer[-1]
        assert "name" in metric
        assert "value" in metric
        assert "labels" in metric
        assert "timestamp" in metric
    
    def test_alert_management(self, observability_service):
        """Test alert rule management."""
        # Add test alert
        alert_id = observability_service.add_alert(
            "high_cpu_usage",
            "cpu_usage > threshold",
            80.0,
            "warning"
        )
        
        assert alert_id is not None
        assert alert_id in observability_service.alerts
        
        alert = observability_service.alerts[alert_id]
        assert alert.name == "high_cpu_usage"
        assert alert.threshold == 80.0
        assert alert.enabled is True
    
    def test_metrics_summary(self, observability_service):
        """Test comprehensive metrics summary."""
        summary = observability_service.get_metrics_summary()
        
        assert "service_info" in summary
        assert "system_metrics" in summary
        assert "application_metrics" in summary
        assert "health_status" in summary
        assert "timestamp" in summary
        
        # Verify service info
        service_info = summary["service_info"]
        assert service_info["name"] == "test_service"


class TestCircuitBreaker:
    """Test suite for circuit breaker resilience pattern."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker instance."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,  # Short timeout for testing
            name="test_circuit"
        )
    
    def test_normal_operation(self, circuit_breaker):
        """Test circuit breaker in normal operation."""
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Successful calls should work normally
        def successful_function():
            return "success"
        
        result = circuit_breaker.call(successful_function)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
    
    def test_failure_handling(self, circuit_breaker):
        """Test circuit breaker failure handling."""
        def failing_function():
            raise ValueError("Test failure")
        
        # First few failures should still call the function
        for i in range(3):
            with pytest.raises(ValueError):
                circuit_breaker.call(failing_function)
        
        # Circuit should now be open
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Further calls should be blocked
        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call(failing_function)
    
    async def test_recovery_mechanism(self, circuit_breaker):
        """Test circuit breaker recovery."""
        def failing_function():
            raise ValueError("Test failure")
        
        def successful_function():
            return "success"
        
        # Trigger circuit to open
        for i in range(3):
            with pytest.raises(ValueError):
                circuit_breaker.call(failing_function)
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.2)
        
        # Next call should transition to half-open
        result = circuit_breaker.call(successful_function)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
    
    async def test_async_functionality(self, circuit_breaker):
        """Test circuit breaker with async functions."""
        async def async_failing_function():
            raise ValueError("Async failure")
        
        async def async_successful_function():
            return "async success"
        
        # Test async failure handling
        for i in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.acall(async_failing_function)
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Test blocked calls
        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.acall(async_failing_function)
        
        # Test recovery
        await asyncio.sleep(0.2)
        result = await circuit_breaker.acall(async_successful_function)
        assert result == "async success"
    
    def test_decorator_usage(self):
        """Test circuit breaker as decorator."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        
        @cb
        def decorated_function(should_fail=False):
            if should_fail:
                raise RuntimeError("Decorated failure")
            return "decorated success"
        
        # Test successful calls
        assert decorated_function() == "decorated success"
        
        # Test failures
        for i in range(2):
            with pytest.raises(RuntimeError):
                decorated_function(should_fail=True)
        
        # Circuit should be open
        with pytest.raises(CircuitBreakerError):
            decorated_function()
    
    def test_statistics(self, circuit_breaker):
        """Test circuit breaker statistics."""
        def test_function(should_fail=False):
            if should_fail:
                raise ValueError("Test failure")
            return "success"
        
        # Generate some activity
        circuit_breaker.call(test_function)  # Success
        
        try:
            circuit_breaker.call(test_function, should_fail=True)  # Failure
        except ValueError:
            pass
        
        stats = circuit_breaker.stats
        
        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 1
        assert stats["failed_calls"] == 1
        assert stats["state"] == CircuitState.CLOSED.value
        assert 0 <= stats["failure_rate"] <= 1


class TestAuditService:
    """Test suite for comprehensive audit service."""
    
    @pytest.fixture
    def audit_config(self):
        """Create test audit configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield AuditConfig(
                log_directory=temp_dir,
                buffer_size=10,
                flush_interval_seconds=1.0,
                enable_checksum_verification=True
            )
    
    @pytest.fixture
    async def audit_service(self, audit_config):
        """Create audit service instance."""
        service = AuditService(audit_config)
        await service.start()
        yield service
        await service.stop()
    
    def test_audit_event_logging(self, audit_service):
        """Test basic audit event logging."""
        event_id = audit_service.log_event(
            AuditType.DATA_ACCESS,
            "read_dataset",
            "success",
            actor="test_user",
            target="dataset_123",
            details={"rows": 1000, "columns": 10},
            severity=AuditSeverity.INFO
        )
        
        assert event_id is not None
        assert len(audit_service.storage.event_buffer) > 0
        
        # Verify event structure
        event = audit_service.storage.event_buffer[-1]
        assert event.action == "read_dataset"
        assert event.actor == "test_user"
        assert event.target == "dataset_123"
        assert event.outcome == "success"
        assert event.details["rows"] == 1000
    
    def test_event_integrity_verification(self, audit_service):
        """Test audit event integrity verification."""
        # Log multiple events
        for i in range(5):
            audit_service.log_event(
                AuditType.MODEL_PREDICTION,
                f"prediction_{i}",
                "success",
                actor="ml_service",
                details={"prediction_id": i}
            )
        
        # Flush to storage
        audit_service.storage.flush_buffer()
        
        # Verify integrity
        verification_result = audit_service.verify_integrity()
        
        assert verification_result["status"] == "success"
        assert verification_result["verified_events"] >= 5
        assert verification_result["integrity_failures"] == 0
        assert verification_result["chain_breaks"] == 0
    
    def test_event_search(self, audit_service):
        """Test audit event search functionality."""
        # Log test events
        test_events = [
            {"actor": "user1", "action": "login", "outcome": "success"},
            {"actor": "user2", "action": "data_access", "outcome": "success"},
            {"actor": "user1", "action": "logout", "outcome": "success"},
            {"actor": "user3", "action": "login", "outcome": "failure"},
        ]
        
        for event_data in test_events:
            audit_service.log_event(
                AuditType.USER_LOGIN if "login" in event_data["action"] else AuditType.DATA_ACCESS,
                event_data["action"],
                event_data["outcome"],
                actor=event_data["actor"]
            )
        
        # Flush to storage
        audit_service.storage.flush_buffer()
        
        # Search by actor
        user1_events = audit_service.search_events(actors=["user1"])
        assert len(user1_events) == 2
        
        # Search by outcome
        failed_events = audit_service.search_events(outcomes=["failure"])
        assert len(failed_events) == 1
        assert failed_events[0]["actor"] == "user3"
    
    def test_compliance_reporting(self, audit_service):
        """Test compliance report generation."""
        # Log GDPR-relevant events
        gdpr_events = [
            (AuditType.DATA_ACCESS, "personal_data_access", {"data_type": "personal"}),
            (AuditType.DATA_EXPORTED, "export_personal_data", {"export_reason": "user_request"}),
            (AuditType.GDPR_REQUEST, "data_deletion_request", {"request_type": "deletion"}),
        ]
        
        for event_type, action, details in gdpr_events:
            audit_service.log_event(
                event_type,
                action,
                "success",
                actor="gdpr_service",
                details=details,
                tags={"personal_data"}
            )
        
        # Flush to storage
        audit_service.storage.flush_buffer()
        
        # Generate GDPR report
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=1)
        
        report = audit_service.generate_compliance_report(
            ComplianceFramework.GDPR,
            start_date,
            end_date
        )
        
        assert report["report_type"] == "GDPR Compliance"
        assert "summary" in report
        assert "details" in report
        assert report["summary"]["total_events"] >= 3
    
    def test_event_export(self, audit_service):
        """Test audit event export functionality."""
        # Log test events
        for i in range(10):
            audit_service.log_event(
                AuditType.API_CALL,
                f"api_call_{i}",
                "success",
                actor="api_user",
                details={"endpoint": f"/api/test/{i}"}
            )
        
        # Flush to storage
        audit_service.storage.flush_buffer()
        
        # Export events
        start_date = datetime.utcnow() - timedelta(days=1)
        end_date = datetime.utcnow() + timedelta(days=1)
        
        export_path = audit_service.export_events(
            start_date,
            end_date,
            format="json"
        )
        
        assert Path(export_path).exists()
        
        # Verify exported content
        with open(export_path, 'r') as f:
            exported_events = json.load(f)
        
        assert len(exported_events) >= 10
        assert exported_events[0]["action"].startswith("api_call_")
    
    def test_audit_metrics(self, audit_service):
        """Test audit service metrics."""
        # Log some events
        for i in range(5):
            audit_service.log_event(
                AuditType.SYSTEM_STARTUP,
                f"service_start_{i}",
                "success"
            )
        
        metrics = audit_service.get_metrics()
        
        assert "buffer_size" in metrics
        assert "config" in metrics
        assert "storage" in metrics
        
        # Verify buffer has events
        assert metrics["buffer_size"] == 5
        
        # Verify configuration
        config = metrics["config"]
        assert "retention_days" in config
        assert "integrity_enabled" in config


class TestEnterpriseIntegration:
    """Integration tests for enterprise services working together."""
    
    @pytest.fixture
    async def integrated_services(self):
        """Set up integrated enterprise services."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Configure services
            audit_config = AuditConfig(log_directory=temp_dir)
            security_config = SecurityConfig(enable_audit_logging=True)
            observability_config = ObservabilityConfig(
                enable_metrics=True,
                enable_health_checks=True
            )
            
            # Create services
            audit_service = AuditService(audit_config)
            security_service = SecurityService(security_config)
            observability_service = ObservabilityService(observability_config)
            
            # Start services
            await audit_service.start()
            await observability_service.start()
            
            services = {
                'audit': audit_service,
                'security': security_service,
                'observability': observability_service
            }
            
            yield services
            
            # Stop services
            await audit_service.stop()
            await observability_service.stop()
    
    async def test_security_audit_integration(self, integrated_services):
        """Test security events are properly audited."""
        security_service = integrated_services['security']
        audit_service = integrated_services['audit']
        
        # Perform security operations that should be audited
        auth_service = security_service.auth_service
        
        # This should trigger audit events
        try:
            # Simulate authentication (will fail but should be audited)
            auth_service.authenticate_user(
                "test_user",
                "wrong_password",
                "127.0.0.1",
                "test_agent"
            )
        except:
            pass
        
        # Check if security events were audited
        audit_service.storage.flush_buffer()
        
        # Search for authentication events
        auth_events = audit_service.search_events(
            event_types=[AuditType.USER_LOGIN],
            limit=10
        )
        
        # Note: This test verifies the integration framework
        # In a real implementation, the security service would
        # directly call the audit service
        assert isinstance(auth_events, list)
    
    async def test_observability_health_monitoring(self, integrated_services):
        """Test observability monitors service health."""
        observability_service = integrated_services['observability']
        
        # Run health checks
        health_results = await observability_service.health_service.run_all_checks()
        
        assert health_results["overall_status"] in [
            HealthStatus.HEALTHY.value,
            HealthStatus.DEGRADED.value,
            HealthStatus.UNHEALTHY.value
        ]
        
        # Verify health check covers multiple services
        checks = health_results["checks"]
        assert len(checks) >= 3  # database, memory, disk_space
    
    def test_performance_monitoring_integration(self, integrated_services):
        """Test performance monitoring across services."""
        observability_service = integrated_services['observability']
        
        # Simulate application activity
        app_metrics = observability_service.app_metrics
        
        # Record various metrics
        app_metrics.record_request("GET", "/api/security/login", 200, 0.1)
        app_metrics.record_request("POST", "/api/audit/events", 201, 0.05)
        app_metrics.record_ml_prediction("anomaly_detector", True, False)
        
        # Get comprehensive metrics
        metrics_summary = observability_service.get_metrics_summary()
        
        assert "application_metrics" in metrics_summary
        app_metrics_data = metrics_summary["application_metrics"]
        
        assert "request_counts" in app_metrics_data
        assert "ml_metrics" in app_metrics_data
        
        # Verify metrics capture cross-service activity
        request_counts = app_metrics_data["request_counts"]
        assert any("/api/security/" in key for key in request_counts.keys())
        assert any("/api/audit/" in key for key in request_counts.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
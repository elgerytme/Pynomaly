"""
Comprehensive audit logging system for Web UI security events
Provides detailed logging, compliance reporting, and forensic capabilities
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

try:
    from pynomaly.core.config import get_settings
    from pynomaly.presentation.web.enhanced_auth import AuthenticationMethod, UserRole
except ImportError:
    # Fallback for testing
    def get_settings():
        from types import SimpleNamespace
        return SimpleNamespace(
            storage=SimpleNamespace(log_path=Path("/tmp")),
            audit=SimpleNamespace(
                enabled=True,
                retention_days=365,
                encrypt_logs=False,
                compliance_mode=False
            )
        )

    class AuthenticationMethod(Enum):
        PASSWORD = "password"

    class UserRole(Enum):
        ADMIN = "admin"


class AuditEventType(Enum):
    """Types of audit events"""
    # Authentication events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_LOGIN_FAILED = "user_login_failed"
    USER_ACCOUNT_LOCKED = "user_account_locked"
    USER_PASSWORD_CHANGED = "user_password_changed"
    USER_MFA_ENABLED = "user_mfa_enabled"
    USER_MFA_DISABLED = "user_mfa_disabled"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REMOVED = "role_removed"

    # Data access events
    DATA_VIEWED = "data_viewed"
    DATA_CREATED = "data_created"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    DATA_IMPORTED = "data_imported"

    # System events
    SYSTEM_CONFIG_CHANGED = "system_config_changed"
    SYSTEM_BACKUP_CREATED = "system_backup_created"
    SYSTEM_BACKUP_RESTORED = "system_backup_restored"
    SYSTEM_MAINTENANCE_START = "system_maintenance_start"
    SYSTEM_MAINTENANCE_END = "system_maintenance_end"

    # Security events
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"
    MALICIOUS_REQUEST = "malicious_request"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    WAF_TRIGGERED = "waf_triggered"

    # API events
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_ACCESS = "api_access"
    API_RATE_LIMITED = "api_rate_limited"

    # Model events
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_PREDICTION = "model_prediction"
    MODEL_DELETED = "model_deleted"


class AuditEventSeverity(Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "sox"  # Sarbanes-Oxley
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    SOC2 = "soc2"  # Service Organization Control 2
    ISO27001 = "iso27001"  # ISO/IEC 27001


@dataclass
class AuditEvent:
    """Comprehensive audit event structure"""
    event_id: str
    event_type: AuditEventType
    severity: AuditEventSeverity
    timestamp: datetime

    # User context
    user_id: str | None
    username: str | None
    user_email: str | None
    user_roles: list[UserRole]
    authentication_method: AuthenticationMethod | None
    session_id: str | None

    # Request context
    ip_address: str
    user_agent: str
    request_method: str
    request_path: str
    request_id: str | None

    # Event details
    resource_type: str | None
    resource_id: str | None
    action: str
    result: str  # success, failure, error

    # Additional data
    details: dict[str, Any]
    before_values: dict[str, Any] | None
    after_values: dict[str, Any] | None

    # Compliance
    compliance_frameworks: list[ComplianceFramework]
    retention_required: bool

    # Technical details
    response_code: int | None
    execution_time_ms: float | None
    error_message: str | None

    # Geolocation (if available)
    country: str | None
    region: str | None
    city: str | None


class AuditLogger:
    """Enhanced audit logging service"""

    def __init__(self):
        self.settings = get_settings()
        self.logger = self._setup_logger()
        self.event_buffer: list[AuditEvent] = []
        self.buffer_size = 1000

        # Compliance mappings
        self.compliance_mappings = self._setup_compliance_mappings()

        # Start background tasks
        self.start_background_tasks()

    def _setup_logger(self) -> structlog.stdlib.BoundLogger:
        """Setup structured audit logger"""
        # Configure audit-specific logger
        audit_log_file = self.settings.storage.log_path / "audit.log"

        # Ensure log directory exists
        audit_log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure structlog for audit logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger("pynomaly.audit")

    def _setup_compliance_mappings(self) -> dict[AuditEventType, list[ComplianceFramework]]:
        """Setup mappings between event types and compliance frameworks"""
        return {
            # Authentication events - relevant for most frameworks
            AuditEventType.USER_LOGIN: [ComplianceFramework.SOX, ComplianceFramework.GDPR, ComplianceFramework.SOC2],
            AuditEventType.USER_LOGOUT: [ComplianceFramework.SOX, ComplianceFramework.GDPR, ComplianceFramework.SOC2],
            AuditEventType.USER_LOGIN_FAILED: [ComplianceFramework.SOX, ComplianceFramework.GDPR, ComplianceFramework.SOC2],

            # Data access events - critical for data protection regulations
            AuditEventType.DATA_VIEWED: [ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
            AuditEventType.DATA_CREATED: [ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.SOX],
            AuditEventType.DATA_MODIFIED: [ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.SOX],
            AuditEventType.DATA_DELETED: [ComplianceFramework.GDPR, ComplianceFramework.HIPAA, ComplianceFramework.SOX],
            AuditEventType.DATA_EXPORTED: [ComplianceFramework.GDPR, ComplianceFramework.HIPAA],

            # System events - relevant for operational frameworks
            AuditEventType.SYSTEM_CONFIG_CHANGED: [ComplianceFramework.SOC2, ComplianceFramework.ISO27001],
            AuditEventType.PERMISSION_CHANGED: [ComplianceFramework.SOX, ComplianceFramework.SOC2, ComplianceFramework.ISO27001],

            # Security events - critical for all frameworks
            AuditEventType.SECURITY_VIOLATION: [framework for framework in ComplianceFramework],
            AuditEventType.SUSPICIOUS_ACTIVITY: [framework for framework in ComplianceFramework],
            AuditEventType.BRUTE_FORCE_ATTEMPT: [framework for framework in ComplianceFramework],
        }

    def start_background_tasks(self):
        """Start background audit tasks"""
        try:
            asyncio.create_task(self._flush_events_task())
            asyncio.create_task(self._cleanup_old_logs_task())
        except RuntimeError:
            # No event loop running
            pass

    async def _flush_events_task(self):
        """Background task to flush buffered events"""
        while True:
            try:
                if self.event_buffer:
                    await self._flush_events()
                await asyncio.sleep(30)  # Flush every 30 seconds
            except Exception as e:
                print(f"Error in audit flush task: {e}")
                await asyncio.sleep(60)

    async def _cleanup_old_logs_task(self):
        """Background task to clean up old audit logs"""
        while True:
            try:
                await self._cleanup_old_logs()
                await asyncio.sleep(86400)  # Cleanup daily
            except Exception as e:
                print(f"Error in audit cleanup task: {e}")
                await asyncio.sleep(86400)

    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        result: str,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        request_method: str = "GET",
        request_path: str = "/",
        user_id: str | None = None,
        username: str | None = None,
        user_email: str | None = None,
        user_roles: list[UserRole] | None = None,
        authentication_method: AuthenticationMethod | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        before_values: dict[str, Any] | None = None,
        after_values: dict[str, Any] | None = None,
        response_code: int | None = None,
        execution_time_ms: float | None = None,
        error_message: str | None = None,
        severity: AuditEventSeverity | None = None
    ) -> str:
        """Log audit event"""
        if not self.settings.audit.enabled:
            return ""

        # Determine severity if not provided
        if severity is None:
            severity = self._determine_severity(event_type, result)

        # Determine compliance frameworks
        compliance_frameworks = self.compliance_mappings.get(event_type, [])

        # Create audit event
        event = AuditEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),

            # User context
            user_id=user_id,
            username=username,
            user_email=user_email,
            user_roles=user_roles or [],
            authentication_method=authentication_method,
            session_id=session_id,

            # Request context
            ip_address=ip_address,
            user_agent=user_agent,
            request_method=request_method,
            request_path=request_path,
            request_id=request_id,

            # Event details
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,

            # Additional data
            details=details or {},
            before_values=before_values,
            after_values=after_values,

            # Compliance
            compliance_frameworks=compliance_frameworks,
            retention_required=len(compliance_frameworks) > 0,

            # Technical details
            response_code=response_code,
            execution_time_ms=execution_time_ms,
            error_message=error_message,

            # Geolocation (placeholder - would integrate with GeoIP service)
            country=None,
            region=None,
            city=None
        )

        # Add to buffer
        self.event_buffer.append(event)

        # Flush if buffer is full
        if len(self.event_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_events())

        # Log immediately for critical events
        if severity == AuditEventSeverity.CRITICAL:
            self._log_event_immediately(event)

        return event.event_id

    def _determine_severity(self, event_type: AuditEventType, result: str) -> AuditEventSeverity:
        """Determine event severity based on type and result"""
        # Critical events
        critical_events = [
            AuditEventType.SECURITY_VIOLATION,
            AuditEventType.BRUTE_FORCE_ATTEMPT,
            AuditEventType.USER_ACCOUNT_LOCKED,
            AuditEventType.SYSTEM_CONFIG_CHANGED,
            AuditEventType.DATA_DELETED
        ]

        # High severity events
        high_events = [
            AuditEventType.USER_LOGIN_FAILED,
            AuditEventType.ACCESS_DENIED,
            AuditEventType.SUSPICIOUS_ACTIVITY,
            AuditEventType.DATA_EXPORTED,
            AuditEventType.PERMISSION_CHANGED
        ]

        # Medium severity events
        medium_events = [
            AuditEventType.USER_LOGIN,
            AuditEventType.DATA_MODIFIED,
            AuditEventType.API_KEY_CREATED,
            AuditEventType.MODEL_DEPLOYED
        ]

        if event_type in critical_events or result == "failure":
            return AuditEventSeverity.CRITICAL
        elif event_type in high_events:
            return AuditEventSeverity.HIGH
        elif event_type in medium_events:
            return AuditEventSeverity.MEDIUM
        else:
            return AuditEventSeverity.LOW

    def _log_event_immediately(self, event: AuditEvent):
        """Log event immediately to audit log"""
        event_data = asdict(event)

        # Convert enums to strings for JSON serialization
        event_data["event_type"] = event.event_type.value
        event_data["severity"] = event.severity.value
        event_data["user_roles"] = [role.value for role in event.user_roles]
        event_data["authentication_method"] = event.authentication_method.value if event.authentication_method else None
        event_data["compliance_frameworks"] = [fw.value for fw in event.compliance_frameworks]
        event_data["timestamp"] = event.timestamp.isoformat()

        self.logger.info("Audit event", **event_data)

    async def _flush_events(self):
        """Flush buffered events to audit log"""
        if not self.event_buffer:
            return

        events_to_flush = self.event_buffer.copy()
        self.event_buffer.clear()

        for event in events_to_flush:
            self._log_event_immediately(event)

    async def _cleanup_old_logs(self):
        """Clean up old audit logs based on retention policy"""
        try:
            log_directory = self.settings.storage.log_path
            retention_days = getattr(self.settings.audit, 'retention_days', 365)
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

            # Find old log files
            for log_file in log_directory.glob("audit*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    # Archive or delete old logs
                    if self.settings.audit.compliance_mode:
                        # In compliance mode, archive instead of deleting
                        await self._archive_log_file(log_file)
                    else:
                        log_file.unlink()

        except Exception as e:
            print(f"Error cleaning up audit logs: {e}")

    async def _archive_log_file(self, log_file: Path):
        """Archive old log file for compliance"""
        archive_dir = log_file.parent / "archived"
        archive_dir.mkdir(exist_ok=True)

        archive_path = archive_dir / f"{log_file.name}.archived"

        if self.settings.audit.encrypt_logs:
            # Encrypt archived logs (placeholder - would implement actual encryption)
            await self._encrypt_log_file(log_file, archive_path)
        else:
            log_file.rename(archive_path)

    async def _encrypt_log_file(self, source: Path, destination: Path):
        """Encrypt log file for secure archival"""
        # Placeholder for log encryption implementation
        # In production, would use proper encryption like AES-256
        import shutil
        shutil.copy2(source, destination)
        source.unlink()

    def log_authentication_event(
        self,
        event_type: AuditEventType,
        user_id: str | None,
        username: str | None,
        ip_address: str,
        user_agent: str,
        result: str,
        authentication_method: AuthenticationMethod | None = None,
        error_message: str | None = None,
        session_id: str | None = None
    ) -> str:
        """Log authentication-specific event"""
        return self.log_event(
            event_type=event_type,
            action="authenticate",
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            user_id=user_id,
            username=username,
            authentication_method=authentication_method,
            session_id=session_id,
            error_message=error_message,
            resource_type="authentication_system"
        )

    def log_data_access_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        username: str,
        resource_type: str,
        resource_id: str,
        action: str,
        result: str,
        ip_address: str,
        request_path: str,
        before_values: dict[str, Any] | None = None,
        after_values: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None
    ) -> str:
        """Log data access event"""
        return self.log_event(
            event_type=event_type,
            action=action,
            result=result,
            user_id=user_id,
            username=username,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            request_path=request_path,
            before_values=before_values,
            after_values=after_values,
            details=details
        )

    def log_security_event(
        self,
        event_type: AuditEventType,
        ip_address: str,
        user_agent: str,
        request_path: str,
        severity: AuditEventSeverity,
        details: dict[str, Any],
        user_id: str | None = None,
        action: str = "security_check"
    ) -> str:
        """Log security event"""
        return self.log_event(
            event_type=event_type,
            action=action,
            result="blocked" if severity in [AuditEventSeverity.HIGH, AuditEventSeverity.CRITICAL] else "detected",
            ip_address=ip_address,
            user_agent=user_agent,
            request_path=request_path,
            user_id=user_id,
            severity=severity,
            details=details,
            resource_type="security_system"
        )

    def log_system_event(
        self,
        event_type: AuditEventType,
        action: str,
        result: str,
        user_id: str,
        username: str,
        details: dict[str, Any] | None = None,
        before_values: dict[str, Any] | None = None,
        after_values: dict[str, Any] | None = None
    ) -> str:
        """Log system configuration event"""
        return self.log_event(
            event_type=event_type,
            action=action,
            result=result,
            user_id=user_id,
            username=username,
            resource_type="system_configuration",
            details=details,
            before_values=before_values,
            after_values=after_values,
            severity=AuditEventSeverity.HIGH
        )

    def search_events(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        event_types: list[AuditEventType] | None = None,
        ip_address: str | None = None,
        resource_type: str | None = None,
        severity: AuditEventSeverity | None = None,
        limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Search audit events (simplified implementation)"""
        # In production, this would query a database or search service
        # For now, return recent buffered events that match criteria
        results = []

        for event in self.event_buffer:
            # Apply filters
            if start_time and event.timestamp < start_time:
                continue
            if end_time and event.timestamp > end_time:
                continue
            if user_id and event.user_id != user_id:
                continue
            if event_types and event.event_type not in event_types:
                continue
            if ip_address and event.ip_address != ip_address:
                continue
            if resource_type and event.resource_type != resource_type:
                continue
            if severity and event.severity != severity:
                continue

            results.append(asdict(event))

            if len(results) >= limit:
                break

        return results

    def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> dict[str, Any]:
        """Generate compliance report for specific framework"""
        # Filter events relevant to the compliance framework
        relevant_events = []

        for event in self.event_buffer:
            if (framework in event.compliance_frameworks and
                start_date <= event.timestamp <= end_date):
                relevant_events.append(event)

        # Generate report
        report = {
            "framework": framework.value,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(relevant_events),
            "events_by_type": {},
            "events_by_severity": {},
            "security_incidents": 0,
            "access_violations": 0,
            "data_access_events": 0,
            "summary": {}
        }

        # Analyze events
        for event in relevant_events:
            event_type = event.event_type.value
            severity = event.severity.value

            # Count by type
            report["events_by_type"][event_type] = report["events_by_type"].get(event_type, 0) + 1

            # Count by severity
            report["events_by_severity"][severity] = report["events_by_severity"].get(severity, 0) + 1

            # Count specific categories
            if event.event_type in [AuditEventType.SECURITY_VIOLATION, AuditEventType.SUSPICIOUS_ACTIVITY]:
                report["security_incidents"] += 1

            if event.event_type == AuditEventType.ACCESS_DENIED:
                report["access_violations"] += 1

            if event.event_type in [AuditEventType.DATA_VIEWED, AuditEventType.DATA_MODIFIED,
                                   AuditEventType.DATA_DELETED, AuditEventType.DATA_EXPORTED]:
                report["data_access_events"] += 1

        # Generate summary
        report["summary"] = {
            "compliance_status": "compliant" if report["security_incidents"] == 0 else "review_required",
            "risk_level": "high" if report["security_incidents"] > 5 else "low",
            "recommendations": self._generate_compliance_recommendations(framework, report)
        }

        return report

    def _generate_compliance_recommendations(
        self,
        framework: ComplianceFramework,
        report: dict[str, Any]
    ) -> list[str]:
        """Generate compliance recommendations based on report"""
        recommendations = []

        if report["security_incidents"] > 0:
            recommendations.append("Review and investigate all security incidents")

        if report["access_violations"] > 10:
            recommendations.append("Review access control policies and user permissions")

        if framework == ComplianceFramework.GDPR and report["data_access_events"] > 100:
            recommendations.append("Implement additional data access monitoring and controls")

        if not recommendations:
            recommendations.append("Continue current security practices")

        return recommendations


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# Convenience functions for common audit events
def log_user_login(user_id: str, username: str, ip_address: str, user_agent: str,
                  method: AuthenticationMethod, session_id: str) -> str:
    """Log successful user login"""
    return get_audit_logger().log_authentication_event(
        AuditEventType.USER_LOGIN,
        user_id=user_id,
        username=username,
        ip_address=ip_address,
        user_agent=user_agent,
        result="success",
        authentication_method=method,
        session_id=session_id
    )


def log_user_login_failed(username: str, ip_address: str, user_agent: str,
                         reason: str) -> str:
    """Log failed user login"""
    return get_audit_logger().log_authentication_event(
        AuditEventType.USER_LOGIN_FAILED,
        user_id=None,
        username=username,
        ip_address=ip_address,
        user_agent=user_agent,
        result="failure",
        error_message=reason
    )


def log_data_access(user_id: str, username: str, resource_type: str,
                   resource_id: str, action: str, ip_address: str,
                   request_path: str) -> str:
    """Log data access event"""
    event_type_map = {
        "view": AuditEventType.DATA_VIEWED,
        "create": AuditEventType.DATA_CREATED,
        "update": AuditEventType.DATA_MODIFIED,
        "delete": AuditEventType.DATA_DELETED,
        "export": AuditEventType.DATA_EXPORTED
    }

    event_type = event_type_map.get(action, AuditEventType.DATA_VIEWED)

    return get_audit_logger().log_data_access_event(
        event_type=event_type,
        user_id=user_id,
        username=username,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        result="success",
        ip_address=ip_address,
        request_path=request_path
    )


def log_security_violation(ip_address: str, user_agent: str, request_path: str,
                          violation_type: str, details: dict[str, Any]) -> str:
    """Log security violation"""
    return get_audit_logger().log_security_event(
        AuditEventType.SECURITY_VIOLATION,
        ip_address=ip_address,
        user_agent=user_agent,
        request_path=request_path,
        severity=AuditEventSeverity.CRITICAL,
        details={"violation_type": violation_type, **details}
    )

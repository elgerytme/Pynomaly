"""Enterprise audit logging and compliance system."""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import aiofiles

    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from pynomaly.shared.config import Config

# from ..monitoring.opentelemetry_service import get_telemetry_service
# from ..monitoring.distributed_tracing import get_distributed_tracer


# Simple stubs for monitoring
def get_telemetry_service():
    """Simple stub for monitoring."""
    return None


def get_distributed_tracer():
    """Simple stub for monitoring."""
    return None


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of auditable events."""

    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGE = "auth.password.change"
    MFA_ENABLED = "auth.mfa.enabled"
    MFA_DISABLED = "auth.mfa.disabled"

    # Authorization events
    ACCESS_GRANTED = "authz.access.granted"
    ACCESS_DENIED = "authz.access.denied"
    PERMISSION_CHANGED = "authz.permission.changed"
    ROLE_ASSIGNED = "authz.role.assigned"
    ROLE_REMOVED = "authz.role.removed"

    # Data events
    DATA_ACCESS = "data.access"
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    DATA_DELETE = "data.delete"
    DATA_MODIFY = "data.modify"
    SENSITIVE_DATA_ACCESS = "data.sensitive.access"

    # Model events
    MODEL_CREATE = "model.create"
    MODEL_TRAIN = "model.train"
    MODEL_DEPLOY = "model.deploy"
    MODEL_DELETE = "model.delete"
    MODEL_PREDICT = "model.predict"
    MODEL_EXPORT = "model.export"

    # System events
    SYSTEM_CONFIG_CHANGE = "system.config.change"
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    BACKUP_CREATE = "system.backup.create"
    BACKUP_RESTORE = "system.backup.restore"

    # Compliance events
    GDPR_REQUEST = "compliance.gdpr.request"
    DATA_RETENTION_DELETE = "compliance.retention.delete"
    AUDIT_LOG_ACCESS = "compliance.audit.access"
    COMPLIANCE_REPORT_GENERATED = "compliance.report.generated"

    # Security events
    SECURITY_VIOLATION = "security.violation"
    ANOMALY_DETECTED = "security.anomaly.detected"
    INTRUSION_ATTEMPT = "security.intrusion.attempt"
    CERTIFICATE_EXPIRED = "security.certificate.expired"


class Severity(Enum):
    """Event severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    SOC2 = "soc2"
    CCPA = "ccpa"


@dataclass
class AuditEvent:
    """Audit event data structure."""

    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: str
    outcome: str  # "success", "failure", "partial"
    severity: Severity
    details: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    tenant_id: Optional[str] = None
    compliance_tags: List[ComplianceFramework] = field(default_factory=list)
    data_classification: Optional[str] = None
    retention_period_days: int = 2555  # 7 years default
    hash_signature: Optional[str] = None

    def __post_init__(self):
        """Generate hash signature for integrity verification."""
        self.hash_signature = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate hash signature for event integrity."""
        # Create deterministic hash from core event data
        data_to_hash = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "resource": self.resource,
            "action": self.action,
            "outcome": self.outcome,
        }

        json_str = json.dumps(data_to_hash, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify event integrity using hash signature."""
        expected_hash = self._generate_hash()
        return expected_hash == self.hash_signature


@dataclass
class ComplianceRule:
    """Compliance rule definition."""

    rule_id: str
    framework: ComplianceFramework
    name: str
    description: str
    event_types: List[EventType]
    required_fields: List[str]
    retention_period_days: int
    notification_required: bool = False
    alert_on_violation: bool = True
    custom_validation: Optional[str] = None  # Python code for custom validation


class AuditStorage:
    """Abstract audit storage interface."""

    async def store_event(self, event: AuditEvent) -> bool:
        """Store audit event."""
        raise NotImplementedError

    async def retrieve_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[AuditEvent]:
        """Retrieve audit events."""
        raise NotImplementedError

    async def delete_expired_events(self, before_date: datetime) -> int:
        """Delete expired events."""
        raise NotImplementedError


class FileAuditStorage(AuditStorage):
    """File-based audit storage."""

    def __init__(self, storage_path: str, rotate_size_mb: int = 100):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.rotate_size_mb = rotate_size_mb
        self.current_file_path = (
            self.storage_path / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )

    async def store_event(self, event: AuditEvent) -> bool:
        """Store audit event to file."""
        try:
            event_json = json.dumps(asdict(event), default=str)

            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self.current_file_path, "a") as f:
                    await f.write(event_json + "\n")
            else:
                with open(self.current_file_path, "a") as f:
                    f.write(event_json + "\n")

            # Check if rotation is needed
            await self._check_rotation()
            return True

        except Exception as e:
            logger.error(f"Failed to store audit event: {e}")
            return False

    async def _check_rotation(self) -> None:
        """Check if log rotation is needed."""
        try:
            file_size_mb = self.current_file_path.stat().st_size / (1024 * 1024)
            if file_size_mb >= self.rotate_size_mb:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rotated_path = self.storage_path / f"audit_{timestamp}.jsonl"
                self.current_file_path.rename(rotated_path)
                self.current_file_path = (
                    self.storage_path
                    / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
                )
                logger.info(f"Rotated audit log to {rotated_path}")
        except Exception as e:
            logger.error(f"Failed to rotate audit log: {e}")

    async def retrieve_events(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[AuditEvent]:
        """Retrieve audit events from files."""
        events = []

        # Find relevant files based on date range
        relevant_files = []
        for file_path in self.storage_path.glob("audit_*.jsonl"):
            relevant_files.append(file_path)

        # Read and filter events
        for file_path in relevant_files:
            try:
                if AIOFILES_AVAILABLE:
                    async with aiofiles.open(file_path, "r") as f:
                        async for line in f:
                            if line.strip():
                                event_data = json.loads(line)
                                event = self._dict_to_audit_event(event_data)
                                if self._event_matches_criteria(
                                    event, start_time, end_time, filters
                                ):
                                    events.append(event)
                else:
                    with open(file_path, "r") as f:
                        for line in f:
                            if line.strip():
                                event_data = json.loads(line)
                                event = self._dict_to_audit_event(event_data)
                                if self._event_matches_criteria(
                                    event, start_time, end_time, filters
                                ):
                                    events.append(event)
            except Exception as e:
                logger.error(f"Failed to read audit file {file_path}: {e}")

        return sorted(events, key=lambda e: e.timestamp)

    def _dict_to_audit_event(self, data: Dict[str, Any]) -> AuditEvent:
        """Convert dictionary to AuditEvent object."""
        # Convert string timestamp back to datetime
        data["timestamp"] = datetime.fromisoformat(
            data["timestamp"].replace("Z", "+00:00")
        )
        data["event_type"] = EventType(data["event_type"])
        data["severity"] = Severity(data["severity"])
        data["compliance_tags"] = [
            ComplianceFramework(tag) for tag in data.get("compliance_tags", [])
        ]

        return AuditEvent(**data)

    def _event_matches_criteria(
        self,
        event: AuditEvent,
        start_time: datetime,
        end_time: datetime,
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if event matches retrieval criteria."""
        # Time range check
        if not (start_time <= event.timestamp <= end_time):
            return False

        # Additional filters
        if filters:
            for key, value in filters.items():
                if hasattr(event, key):
                    event_value = getattr(event, key)
                    if event_value != value:
                        return False

        return True

    async def delete_expired_events(self, before_date: datetime) -> int:
        """Delete expired events."""
        deleted_count = 0

        for file_path in self.storage_path.glob("audit_*.jsonl"):
            try:
                # Check file modification time
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < before_date:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted expired audit file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete audit file {file_path}: {e}")

        return deleted_count


class AuditSystem:
    """Enterprise audit logging and compliance system."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize audit system."""
        self.config = config or Config()
        self.telemetry = get_telemetry_service()
        self.tracer = get_distributed_tracer()

        # Configuration
        self.enabled = self.config.get("audit.enabled", True)
        self.storage_path = self.config.get("audit.storage_path", "./audit_logs")
        self.async_mode = self.config.get("audit.async_mode", True)

        # Storage backend
        self.storage = FileAuditStorage(
            storage_path=self.storage_path,
            rotate_size_mb=self.config.get("audit.rotate_size_mb", 100),
        )

        # Compliance rules
        self.compliance_rules: Dict[ComplianceFramework, List[ComplianceRule]] = {}
        self._load_compliance_rules()

        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._processor_task: Optional[asyncio.Task] = None
        self._processing_active = False

        # Metrics
        self.events_logged = 0
        self.events_failed = 0
        self.compliance_violations = 0

        logger.info("Audit system initialized")

    def _load_compliance_rules(self) -> None:
        """Load compliance rules for different frameworks."""
        # GDPR rules
        gdpr_rules = [
            ComplianceRule(
                rule_id="gdpr_data_access",
                framework=ComplianceFramework.GDPR,
                name="GDPR Data Access Logging",
                description="Log all personal data access events",
                event_types=[EventType.DATA_ACCESS, EventType.SENSITIVE_DATA_ACCESS],
                required_fields=["user_id", "resource", "timestamp"],
                retention_period_days=2555,  # 7 years
                notification_required=True,
            ),
            ComplianceRule(
                rule_id="gdpr_data_export",
                framework=ComplianceFramework.GDPR,
                name="GDPR Data Export Logging",
                description="Log all personal data export events",
                event_types=[EventType.DATA_EXPORT],
                required_fields=["user_id", "resource", "timestamp", "details"],
                retention_period_days=2555,
                alert_on_violation=True,
            ),
        ]
        self.compliance_rules[ComplianceFramework.GDPR] = gdpr_rules

        # HIPAA rules
        hipaa_rules = [
            ComplianceRule(
                rule_id="hipaa_phi_access",
                framework=ComplianceFramework.HIPAA,
                name="HIPAA PHI Access Logging",
                description="Log all PHI access events",
                event_types=[EventType.DATA_ACCESS, EventType.SENSITIVE_DATA_ACCESS],
                required_fields=["user_id", "resource", "timestamp", "ip_address"],
                retention_period_days=2190,  # 6 years
                notification_required=True,
            )
        ]
        self.compliance_rules[ComplianceFramework.HIPAA] = hipaa_rules

        # SOX rules
        sox_rules = [
            ComplianceRule(
                rule_id="sox_financial_data",
                framework=ComplianceFramework.SOX,
                name="SOX Financial Data Access",
                description="Log all financial data access",
                event_types=[
                    EventType.DATA_ACCESS,
                    EventType.DATA_MODIFY,
                    EventType.DATA_DELETE,
                ],
                required_fields=["user_id", "resource", "timestamp", "action"],
                retention_period_days=2555,  # 7 years
                alert_on_violation=True,
            )
        ]
        self.compliance_rules[ComplianceFramework.SOX] = sox_rules

    async def start_processing(self) -> None:
        """Start async event processing."""
        if not self.async_mode or self._processing_active:
            return

        self._processing_active = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Started audit event processing")

    async def stop_processing(self) -> None:
        """Stop async event processing."""
        if not self._processing_active:
            return

        self._processing_active = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        # Process remaining events
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                await self._store_event(event)
            except asyncio.QueueEmpty:
                break

        logger.info("Stopped audit event processing")

    async def _process_events(self) -> None:
        """Process events from the queue."""
        while self._processing_active:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._store_event(event)
                self.event_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")
                self.events_failed += 1

    async def log_event(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: str = "",
        outcome: str = "success",
        severity: Severity = Severity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        data_classification: Optional[str] = None,
        compliance_frameworks: Optional[List[ComplianceFramework]] = None,
    ) -> bool:
        """Log an audit event."""
        if not self.enabled:
            return True

        try:
            # Get trace context
            trace_id = self.tracer.get_current_trace_id()

            # Create audit event
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                resource=resource,
                action=action,
                outcome=outcome,
                severity=severity,
                details=details or {},
                trace_id=trace_id,
                tenant_id=tenant_id,
                compliance_tags=compliance_frameworks or [],
                data_classification=data_classification,
            )

            # Validate compliance requirements
            await self._validate_compliance(event)

            # Store event
            if self.async_mode:
                try:
                    self.event_queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("Audit event queue full, storing synchronously")
                    await self._store_event(event)
            else:
                await self._store_event(event)

            self.events_logged += 1

            # Record metrics
            self.telemetry.record_detection_metrics(
                duration=0,
                anomaly_count=1,
                algorithm="audit_system",
                tenant_id=tenant_id,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            self.events_failed += 1
            return False

    async def _validate_compliance(self, event: AuditEvent) -> None:
        """Validate event against compliance rules."""
        for framework in event.compliance_tags:
            if framework in self.compliance_rules:
                for rule in self.compliance_rules[framework]:
                    if event.event_type in rule.event_types:
                        await self._check_compliance_rule(event, rule)

    async def _check_compliance_rule(
        self, event: AuditEvent, rule: ComplianceRule
    ) -> None:
        """Check event against specific compliance rule."""
        violations = []

        # Check required fields
        for field in rule.required_fields:
            if not hasattr(event, field) or getattr(event, field) is None:
                violations.append(f"Missing required field: {field}")

        # Set retention period
        event.retention_period_days = rule.retention_period_days

        # Log violations
        if violations:
            self.compliance_violations += 1
            violation_details = {
                "rule_id": rule.rule_id,
                "framework": rule.framework.value,
                "violations": violations,
                "event_id": event.event_id,
            }

            logger.warning(f"Compliance violation detected: {violation_details}")

            if rule.alert_on_violation:
                await self._send_compliance_alert(rule, event, violations)

    async def _send_compliance_alert(
        self, rule: ComplianceRule, event: AuditEvent, violations: List[str]
    ) -> None:
        """Send compliance violation alert."""
        alert_details = {
            "type": "compliance_violation",
            "rule": rule.name,
            "framework": rule.framework.value,
            "event_type": event.event_type.value,
            "violations": violations,
            "timestamp": event.timestamp.isoformat(),
            "user_id": event.user_id,
            "resource": event.resource,
        }

        # This would integrate with your alerting system
        logger.critical(f"COMPLIANCE ALERT: {alert_details}")

    async def _store_event(self, event: AuditEvent) -> bool:
        """Store event using configured storage backend."""
        try:
            return await self.storage.store_event(event)
        except Exception as e:
            logger.error(f"Failed to store audit event {event.event_id}: {e}")
            return False

    async def search_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_types: Optional[List[EventType]] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        severity: Optional[Severity] = None,
        outcome: Optional[str] = None,
        compliance_framework: Optional[ComplianceFramework] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """Search audit events with filters."""
        try:
            # Build filters
            filters = {}
            if user_id:
                filters["user_id"] = user_id
            if resource:
                filters["resource"] = resource
            if severity:
                filters["severity"] = severity
            if outcome:
                filters["outcome"] = outcome

            # Retrieve events
            events = await self.storage.retrieve_events(start_time, end_time, filters)

            # Additional filtering
            if event_types:
                events = [e for e in events if e.event_type in event_types]

            if compliance_framework:
                events = [
                    e for e in events if compliance_framework in e.compliance_tags
                ]

            # Limit results
            return events[:limit]

        except Exception as e:
            logger.error(f"Failed to search audit events: {e}")
            return []

    async def generate_compliance_report(
        self, framework: ComplianceFramework, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specific framework."""
        try:
            # Get events for the framework
            events = await self.search_events(
                start_time=start_time, end_time=end_time, compliance_framework=framework
            )

            # Analyze events
            report = {
                "framework": framework.value,
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "summary": {
                    "total_events": len(events),
                    "event_types": {},
                    "users": set(),
                    "resources": set(),
                    "violations": 0,
                },
                "compliance_status": "compliant",
                "recommendations": [],
                "generated_at": datetime.now().isoformat(),
            }

            # Analyze events
            for event in events:
                # Count event types
                event_type = event.event_type.value
                if event_type not in report["summary"]["event_types"]:
                    report["summary"]["event_types"][event_type] = 0
                report["summary"]["event_types"][event_type] += 1

                # Track users and resources
                if event.user_id:
                    report["summary"]["users"].add(event.user_id)
                if event.resource:
                    report["summary"]["resources"].add(event.resource)

            # Convert sets to lists for JSON serialization
            report["summary"]["users"] = list(report["summary"]["users"])
            report["summary"]["resources"] = list(report["summary"]["resources"])

            # Add framework-specific analysis
            if framework == ComplianceFramework.GDPR:
                await self._add_gdpr_analysis(report, events)
            elif framework == ComplianceFramework.HIPAA:
                await self._add_hipaa_analysis(report, events)
            elif framework == ComplianceFramework.SOX:
                await self._add_sox_analysis(report, events)

            return report

        except Exception as e:
            logger.error(f"Failed to generate compliance report: {e}")
            return {"error": str(e)}

    async def _add_gdpr_analysis(
        self, report: Dict[str, Any], events: List[AuditEvent]
    ) -> None:
        """Add GDPR-specific analysis to report."""
        gdpr_events = {
            "data_access": len(
                [
                    e
                    for e in events
                    if e.event_type
                    in [EventType.DATA_ACCESS, EventType.SENSITIVE_DATA_ACCESS]
                ]
            ),
            "data_exports": len(
                [e for e in events if e.event_type == EventType.DATA_EXPORT]
            ),
            "data_deletions": len(
                [e for e in events if e.event_type == EventType.DATA_DELETE]
            ),
        }

        report["gdpr_analysis"] = gdpr_events

        # Check for potential violations
        if gdpr_events["data_exports"] > 100:  # Example threshold
            report["recommendations"].append(
                "High number of data exports detected - review data processing activities"
            )

    async def _add_hipaa_analysis(
        self, report: Dict[str, Any], events: List[AuditEvent]
    ) -> None:
        """Add HIPAA-specific analysis to report."""
        phi_access_events = [e for e in events if e.data_classification == "PHI"]

        report["hipaa_analysis"] = {
            "phi_access_count": len(phi_access_events),
            "unique_phi_accessors": len(
                set(e.user_id for e in phi_access_events if e.user_id)
            ),
        }

    async def _add_sox_analysis(
        self, report: Dict[str, Any], events: List[AuditEvent]
    ) -> None:
        """Add SOX-specific analysis to report."""
        financial_events = [e for e in events if e.data_classification == "financial"]

        report["sox_analysis"] = {
            "financial_data_events": len(financial_events),
            "modifications": len(
                [e for e in financial_events if e.event_type == EventType.DATA_MODIFY]
            ),
            "deletions": len(
                [e for e in financial_events if e.event_type == EventType.DATA_DELETE]
            ),
        }

    async def cleanup_expired_events(self) -> int:
        """Clean up expired audit events based on retention policies."""
        try:
            # Calculate cleanup date based on shortest retention period
            min_retention_days = min(
                [
                    rule.retention_period_days
                    for rules in self.compliance_rules.values()
                    for rule in rules
                ],
                default=2555,
            )

            cleanup_date = datetime.now() - timedelta(days=min_retention_days)

            deleted_count = await self.storage.delete_expired_events(cleanup_date)

            logger.info(f"Cleaned up {deleted_count} expired audit events")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired events: {e}")
            return 0

    def get_audit_metrics(self) -> Dict[str, Any]:
        """Get audit system metrics."""
        return {
            "events_logged": self.events_logged,
            "events_failed": self.events_failed,
            "compliance_violations": self.compliance_violations,
            "queue_size": self.event_queue.qsize() if self.async_mode else 0,
            "processing_active": self._processing_active,
            "enabled": self.enabled,
            "compliance_frameworks": [f.value for f in self.compliance_rules.keys()],
        }


# Global audit system instance
_audit_system: Optional[AuditSystem] = None


def get_audit_system(config: Optional[Config] = None) -> AuditSystem:
    """Get the global audit system instance."""
    global _audit_system
    if _audit_system is None:
        _audit_system = AuditSystem(config)
    return _audit_system


async def log_audit_event(
    event_type: EventType, action: str, outcome: str = "success", **kwargs
) -> bool:
    """Convenience function to log audit events."""
    audit_system = get_audit_system()
    return await audit_system.log_event(
        event_type=event_type, action=action, outcome=outcome, **kwargs
    )

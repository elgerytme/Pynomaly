"""
Compliance and audit logging domain entities.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from pynomaly.shared.types import UserId, TenantId


class AuditAction(str, Enum):
    """Types of auditable actions."""
    # User actions
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"
    USER_INVITED = "user_invited"
    USER_ROLE_CHANGED = "user_role_changed"

    # Data actions
    DATASET_CREATED = "dataset_created"
    DATASET_UPDATED = "dataset_updated"
    DATASET_DELETED = "dataset_deleted"
    DATASET_ACCESSED = "dataset_accessed"
    DATASET_EXPORTED = "dataset_exported"

    # Model actions
    MODEL_CREATED = "model_created"
    MODEL_TRAINED = "model_trained"
    MODEL_UPDATED = "model_updated"
    MODEL_DELETED = "model_deleted"
    MODEL_DEPLOYED = "model_deployed"
    MODEL_RETIRED = "model_retired"

    # Detection actions
    DETECTION_PERFORMED = "detection_performed"
    ANOMALY_DETECTED = "anomaly_detected"
    ALERT_TRIGGERED = "alert_triggered"
    ALERT_ACKNOWLEDGED = "alert_acknowledged"

    # System actions
    SYSTEM_CONFIGURATION_CHANGED = "system_configuration_changed"
    INTEGRATION_CREATED = "integration_created"
    INTEGRATION_UPDATED = "integration_updated"
    INTEGRATION_DELETED = "integration_deleted"

    # Admin actions
    TENANT_CREATED = "tenant_created"
    TENANT_UPDATED = "tenant_updated"
    TENANT_SUSPENDED = "tenant_suspended"
    PERMISSIONS_CHANGED = "permissions_changed"

    # Compliance actions
    GDPR_REQUEST = "gdpr_request"
    DATA_RETENTION_POLICY_APPLIED = "data_retention_policy_applied"
    ENCRYPTION_KEY_ROTATED = "encryption_key_rotated"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"              # General Data Protection Regulation
    HIPAA = "hipaa"            # Health Insurance Portability and Accountability Act
    SOX = "sox"                # Sarbanes-Oxley Act
    PCI_DSS = "pci_dss"        # Payment Card Industry Data Security Standard
    ISO_27001 = "iso_27001"    # Information Security Management
    SOC2 = "soc2"              # Service Organization Control 2
    CCPA = "ccpa"              # California Consumer Privacy Act
    PIPEDA = "pipeda"          # Personal Information Protection and Electronic Documents Act


class DataClassification(str, Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class RetentionPolicyStatus(str, Enum):
    """Status of data retention policies."""
    ACTIVE = "active"
    PENDING = "pending"
    EXPIRED = "expired"
    DELETED = "deleted"


@dataclass
class AuditEvent:
    """Individual audit event record."""
    id: str
    action: AuditAction
    severity: AuditSeverity
    timestamp: datetime
    tenant_id: TenantId
    user_id: Optional[UserId] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    outcome: str = "success"  # success, failure, error
    risk_score: int = 0  # 0-100 risk assessment
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)

    @property
    def is_high_risk(self) -> bool:
        """Check if this is a high-risk event."""
        return (
            self.risk_score >= 70 or
            self.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL] or
            self.action in [
                AuditAction.USER_DELETED,
                AuditAction.DATASET_DELETED,
                AuditAction.MODEL_DELETED,
                AuditAction.PERMISSIONS_CHANGED,
                AuditAction.SYSTEM_CONFIGURATION_CHANGED
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "action": self.action.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "tenant_id": str(self.tenant_id),
            "user_id": str(self.user_id) if self.user_id else None,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "outcome": self.outcome,
            "risk_score": self.risk_score,
            "compliance_frameworks": [f.value for f in self.compliance_frameworks]
        }


@dataclass
class DataRetentionPolicy:
    """Data retention policy definition."""
    id: str
    name: str
    description: str
    tenant_id: TenantId
    data_type: str
    classification: DataClassification
    retention_period_days: int
    compliance_frameworks: List[ComplianceFramework]
    auto_delete: bool = True
    archive_before_delete: bool = True
    status: RetentionPolicyStatus = RetentionPolicyStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[UserId] = None

    @property
    def is_expired(self) -> bool:
        """Check if policy has expired."""
        return self.status == RetentionPolicyStatus.EXPIRED

    def get_deletion_date(self, creation_date: datetime) -> datetime:
        """Calculate when data should be deleted based on policy."""
        from datetime import timedelta
        return creation_date + timedelta(days=self.retention_period_days)


@dataclass
class ComplianceRule:
    """Individual compliance rule."""
    id: str
    name: str
    description: str
    framework: ComplianceFramework
    rule_type: str  # e.g., "data_protection", "access_control", "audit_log"
    requirements: List[str]
    validation_criteria: Dict[str, Any]
    severity: AuditSeverity
    is_mandatory: bool = True
    implementation_guide: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComplianceCheck:
    """Result of a compliance check."""
    id: str
    rule_id: str
    tenant_id: TenantId
    check_timestamp: datetime
    status: str  # "compliant", "non_compliant", "warning", "not_applicable"
    details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_check_due: Optional[datetime] = None

    @property
    def is_compliant(self) -> bool:
        """Check if rule is compliant."""
        return self.status == "compliant"

    @property
    def needs_attention(self) -> bool:
        """Check if check needs attention."""
        return self.status in ["non_compliant", "warning"]


@dataclass
class GDPRRequest:
    """GDPR data subject request."""
    id: str
    request_type: str  # "access", "rectification", "erasure", "portability", "restriction", "objection"
    tenant_id: TenantId
    data_subject_id: str
    data_subject_email: str
    request_details: str
    submitted_at: datetime
    status: str = "pending"  # "pending", "in_progress", "completed", "rejected"
    assigned_to: Optional[UserId] = None
    completion_deadline: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    processed_at: Optional[datetime] = None
    notes: str = ""

    @property
    def is_overdue(self) -> bool:
        """Check if request is overdue (GDPR 30-day requirement)."""
        if not self.completion_deadline:
            return False
        return datetime.utcnow() > self.completion_deadline and self.status not in ["completed", "rejected"]


@dataclass
class EncryptionKey:
    """Encryption key metadata for audit purposes."""
    id: str
    key_name: str
    algorithm: str
    key_size: int
    tenant_id: TenantId
    purpose: str  # "data_encryption", "backup_encryption", "communication"
    created_at: datetime
    expires_at: Optional[datetime] = None
    rotated_at: Optional[datetime] = None
    status: str = "active"  # "active", "retired", "compromised"
    usage_count: int = 0

    @property
    def needs_rotation(self) -> bool:
        """Check if key needs rotation."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return True
        # Rotate after 90 days or 1M uses
        if self.usage_count > 1000000:
            return True
        from datetime import timedelta
        if datetime.utcnow() > self.created_at + timedelta(days=90):
            return True
        return False


@dataclass
class BackupRecord:
    """Backup operation audit record."""
    id: str
    backup_type: str  # "full", "incremental", "differential"
    tenant_id: TenantId
    data_types: List[str]
    backup_location: str
    encryption_key_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "in_progress"  # "in_progress", "completed", "failed"
    size_bytes: int = 0
    compressed_size_bytes: int = 0
    checksum: str = ""
    retention_until: Optional[datetime] = None

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.size_bytes == 0:
            return 0.0
        return self.compressed_size_bytes / self.size_bytes

    @property
    def is_expired(self) -> bool:
        """Check if backup is expired."""
        return self.retention_until and datetime.utcnow() > self.retention_until


@dataclass
class ComplianceReport:
    """Comprehensive compliance report."""
    id: str
    report_type: str  # "periodic", "incident", "audit_preparation"
    framework: ComplianceFramework
    tenant_id: TenantId
    reporting_period_start: datetime
    reporting_period_end: datetime
    generated_at: datetime
    generated_by: UserId

    # Summary statistics
    total_checks: int = 0
    compliant_checks: int = 0
    non_compliant_checks: int = 0
    warning_checks: int = 0

    # Detailed findings
    findings: List[ComplianceCheck] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)

    # Audit trail summary
    high_risk_events: int = 0
    total_audit_events: int = 0
    failed_operations: int = 0

    @property
    def compliance_score(self) -> float:
        """Calculate overall compliance score (0-100)."""
        if self.total_checks == 0:
            return 0.0
        return (self.compliant_checks / self.total_checks) * 100

    @property
    def risk_level(self) -> str:
        """Determine overall risk level."""
        score = self.compliance_score
        if score >= 95:
            return "low"
        elif score >= 85:
            return "medium"
        elif score >= 70:
            return "high"
        else:
            return "critical"


# Default compliance rules for different frameworks
DEFAULT_COMPLIANCE_RULES = {
    ComplianceFramework.GDPR: [
        ComplianceRule(
            id="gdpr_data_retention",
            name="Data Retention Limits",
            description="Personal data must not be kept longer than necessary",
            framework=ComplianceFramework.GDPR,
            rule_type="data_protection",
            requirements=[
                "Implement data retention policies",
                "Automatically delete expired data",
                "Document retention periods"
            ],
            validation_criteria={
                "max_retention_days": 2555,  # 7 years max
                "auto_delete_enabled": True,
                "policy_documented": True
            },
            severity=AuditSeverity.HIGH
        ),
        ComplianceRule(
            id="gdpr_data_encryption",
            name="Data Encryption at Rest",
            description="Personal data must be encrypted when stored",
            framework=ComplianceFramework.GDPR,
            rule_type="data_protection",
            requirements=[
                "Encrypt all personal data at rest",
                "Use strong encryption algorithms",
                "Manage encryption keys securely"
            ],
            validation_criteria={
                "encryption_enabled": True,
                "algorithm_strength": "AES-256",
                "key_rotation_enabled": True
            },
            severity=AuditSeverity.CRITICAL
        ),
        ComplianceRule(
            id="gdpr_audit_logging",
            name="Audit Trail Maintenance",
            description="Maintain comprehensive audit logs for data processing",
            framework=ComplianceFramework.GDPR,
            rule_type="audit_log",
            requirements=[
                "Log all data access and processing",
                "Maintain logs for required period",
                "Ensure log integrity and immutability"
            ],
            validation_criteria={
                "audit_logging_enabled": True,
                "log_retention_days": 2555,  # 7 years
                "log_integrity_protected": True
            },
            severity=AuditSeverity.HIGH
        )
    ],
    ComplianceFramework.HIPAA: [
        ComplianceRule(
            id="hipaa_access_control",
            name="Access Control and Authentication",
            description="Implement proper access controls for PHI",
            framework=ComplianceFramework.HIPAA,
            rule_type="access_control",
            requirements=[
                "Unique user identification",
                "Automatic logoff",
                "Encryption and decryption"
            ],
            validation_criteria={
                "unique_user_id": True,
                "auto_logoff_enabled": True,
                "phi_encrypted": True
            },
            severity=AuditSeverity.CRITICAL
        )
    ],
    ComplianceFramework.SOX: [
        ComplianceRule(
            id="sox_financial_controls",
            name="Financial Reporting Controls",
            description="Maintain internal controls over financial reporting",
            framework=ComplianceFramework.SOX,
            rule_type="financial_control",
            requirements=[
                "Document all financial data processing",
                "Maintain segregation of duties",
                "Regular control testing"
            ],
            validation_criteria={
                "financial_data_documented": True,
                "segregation_of_duties": True,
                "control_testing_regular": True
            },
            severity=AuditSeverity.HIGH
        )
    ]
}


def get_default_retention_policies() -> List[DataRetentionPolicy]:
    """Get default data retention policies for common data types."""
    return [
        DataRetentionPolicy(
            id="user_activity_logs",
            name="User Activity Logs",
            description="User login and activity logs",
            tenant_id="",  # Will be set per tenant
            data_type="user_activity",
            classification=DataClassification.INTERNAL,
            retention_period_days=2555,  # 7 years
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA]
        ),
        DataRetentionPolicy(
            id="audit_logs",
            name="Audit Logs",
            description="System audit and security logs",
            tenant_id="",
            data_type="audit_log",
            classification=DataClassification.CONFIDENTIAL,
            retention_period_days=2555,  # 7 years
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.SOX, ComplianceFramework.HIPAA]
        ),
        DataRetentionPolicy(
            id="detection_results",
            name="Anomaly Detection Results",
            description="Results from anomaly detection operations",
            tenant_id="",
            data_type="detection_result",
            classification=DataClassification.CONFIDENTIAL,
            retention_period_days=1825,  # 5 years
            compliance_frameworks=[ComplianceFramework.GDPR]
        ),
        DataRetentionPolicy(
            id="user_datasets",
            name="User Uploaded Datasets",
            description="Datasets uploaded by users for analysis",
            tenant_id="",
            data_type="user_dataset",
            classification=DataClassification.RESTRICTED,
            retention_period_days=365,  # 1 year default
            compliance_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA]
        )
    ]

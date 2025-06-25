"""Domain entities for security and compliance framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

import numpy as np


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST = "nist"
    CCPA = "ccpa"
    PIPEDA = "pipeda"
    LGPD = "lgpd"
    SOX = "sox"


class SecurityLevel(Enum):
    """Security classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PrivacyControl(Enum):
    """Privacy protection controls."""
    ANONYMIZATION = "anonymization"
    PSEUDONYMIZATION = "pseudonymization"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    K_ANONYMITY = "k_anonymity"
    L_DIVERSITY = "l_diversity"
    T_CLOSENESS = "t_closeness"
    DATA_MASKING = "data_masking"
    TOKENIZATION = "tokenization"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"
    PERSONAL = "personal"  # GDPR personal data
    SENSITIVE_PERSONAL = "sensitive_personal"  # GDPR special category
    PHI = "phi"  # HIPAA protected health information
    PCI = "pci"  # PCI cardholder data


class AuditLevel(Enum):
    """Audit event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IncidentSeverity(Enum):
    """Security incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessType(Enum):
    """Types of access permissions."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: UUID = field(default_factory=uuid4)
    policy_name: str = ""
    policy_version: str = "1.0"
    description: str = ""
    applicable_frameworks: List[ComplianceFramework] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    enforcement_level: SecurityLevel = SecurityLevel.MEDIUM
    created_date: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    effective_date: datetime = field(default_factory=datetime.utcnow)
    expiration_date: Optional[datetime] = None
    approved_by: Optional[str] = None
    
    def __post_init__(self):
        """Validate security policy."""
        if not self.policy_name:
            raise ValueError("Policy name cannot be empty")
        if self.expiration_date and self.expiration_date <= self.effective_date:
            raise ValueError("Expiration date must be after effective date")
    
    def is_active(self) -> bool:
        """Check if policy is currently active."""
        now = datetime.utcnow()
        if now < self.effective_date:
            return False
        if self.expiration_date and now > self.expiration_date:
            return False
        return True
    
    def applies_to_framework(self, framework: ComplianceFramework) -> bool:
        """Check if policy applies to compliance framework."""
        return framework in self.applicable_frameworks
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get policy summary."""
        return {
            "policy_id": str(self.policy_id),
            "name": self.policy_name,
            "version": self.policy_version,
            "enforcement_level": self.enforcement_level.value,
            "is_active": self.is_active(),
            "frameworks": [f.value for f in self.applicable_frameworks],
            "controls_count": len(self.controls),
            "effective_date": self.effective_date.isoformat(),
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None
        }


@dataclass
class AccessControl:
    """Access control configuration."""
    control_id: UUID = field(default_factory=uuid4)
    resource_pattern: str = ""
    allowed_users: Set[str] = field(default_factory=set)
    allowed_roles: Set[str] = field(default_factory=set)
    allowed_groups: Set[str] = field(default_factory=set)
    access_types: Set[AccessType] = field(default_factory=set)
    time_restrictions: Optional[Dict[str, Any]] = None
    ip_restrictions: Optional[List[str]] = None
    data_classification_level: DataClassification = DataClassification.INTERNAL
    require_mfa: bool = False
    session_timeout_minutes: int = 480  # 8 hours
    created_date: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate access control."""
        if not self.resource_pattern:
            raise ValueError("Resource pattern cannot be empty")
        if not any([self.allowed_users, self.allowed_roles, self.allowed_groups]):
            raise ValueError("At least one access permission must be specified")
        if self.session_timeout_minutes <= 0:
            raise ValueError("Session timeout must be positive")
    
    def check_user_access(
        self,
        user_id: str,
        user_roles: Set[str],
        user_groups: Set[str],
        access_type: AccessType
    ) -> bool:
        """Check if user has access."""
        # Check access type
        if access_type not in self.access_types:
            return False
        
        # Check user permissions
        if user_id in self.allowed_users:
            return True
        
        # Check role permissions
        if self.allowed_roles.intersection(user_roles):
            return True
        
        # Check group permissions
        if self.allowed_groups.intersection(user_groups):
            return True
        
        return False
    
    def get_access_summary(self) -> Dict[str, Any]:
        """Get access control summary."""
        return {
            "control_id": str(self.control_id),
            "resource_pattern": self.resource_pattern,
            "users_count": len(self.allowed_users),
            "roles_count": len(self.allowed_roles),
            "groups_count": len(self.allowed_groups),
            "access_types": [at.value for at in self.access_types],
            "data_classification": self.data_classification_level.value,
            "require_mfa": self.require_mfa,
            "session_timeout_minutes": self.session_timeout_minutes
        }


@dataclass
class AuditEvent:
    """Audit event for compliance logging."""
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""
    event_level: AuditLevel = AuditLevel.INFO
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    resource_accessed: Optional[str] = None
    action_performed: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    data_classification: Optional[DataClassification] = None
    
    def __post_init__(self):
        """Validate audit event."""
        if not self.event_type:
            raise ValueError("Event type cannot be empty")
        if not self.description:
            raise ValueError("Event description cannot be empty")
    
    def is_security_relevant(self) -> bool:
        """Check if event is security relevant."""
        security_events = [
            "authentication", "authorization", "data_access", "data_modification",
            "privilege_escalation", "security_violation", "encryption", "decryption"
        ]
        return any(keyword in self.event_type.lower() for keyword in security_events)
    
    def is_privacy_relevant(self) -> bool:
        """Check if event is privacy relevant."""
        privacy_events = [
            "personal_data", "data_subject", "consent", "data_processing",
            "data_deletion", "data_export", "anonymization", "pseudonymization"
        ]
        return any(keyword in self.event_type.lower() for keyword in privacy_events)
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit event summary."""
        return {
            "event_id": str(self.event_id),
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "level": self.event_level.value,
            "user_id": self.user_id,
            "description": self.description,
            "is_security_relevant": self.is_security_relevant(),
            "is_privacy_relevant": self.is_privacy_relevant(),
            "frameworks": [f.value for f in self.compliance_frameworks],
            "data_classification": self.data_classification.value if self.data_classification else None
        }


@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: UUID = field(default_factory=uuid4)
    incident_type: str = ""
    severity: IncidentSeverity = IncidentSeverity.MEDIUM
    status: str = "open"  # open, investigating, resolved, closed
    detection_timestamp: datetime = field(default_factory=datetime.utcnow)
    resolution_timestamp: Optional[datetime] = None
    description: str = ""
    affected_systems: List[str] = field(default_factory=list)
    affected_data_types: List[str] = field(default_factory=list)
    affected_users: List[str] = field(default_factory=list)
    detection_method: str = "manual"  # manual, automated, external
    root_cause: Optional[str] = None
    remediation_actions: List[str] = field(default_factory=list)
    prevention_measures: List[str] = field(default_factory=list)
    compliance_implications: List[ComplianceFramework] = field(default_factory=list)
    estimated_impact: Dict[str, Any] = field(default_factory=dict)
    assigned_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate security incident."""
        if not self.incident_type:
            raise ValueError("Incident type cannot be empty")
        if not self.description:
            raise ValueError("Incident description cannot be empty")
        if self.status not in ["open", "investigating", "resolved", "closed"]:
            raise ValueError("Invalid incident status")
    
    def is_data_breach(self) -> bool:
        """Check if incident is a data breach."""
        breach_types = [
            "data_breach", "unauthorized_access", "data_exfiltration",
            "data_leak", "privacy_violation"
        ]
        return any(bt in self.incident_type.lower() for bt in breach_types)
    
    def requires_notification(self) -> bool:
        """Check if incident requires regulatory notification."""
        if self.is_data_breach():
            return True
        if self.severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL]:
            return True
        if ComplianceFramework.GDPR in self.compliance_implications:
            return True
        return False
    
    def get_time_to_resolution(self) -> Optional[timedelta]:
        """Get time to resolution."""
        if self.resolution_timestamp:
            return self.resolution_timestamp - self.detection_timestamp
        return None
    
    def get_incident_summary(self) -> Dict[str, Any]:
        """Get incident summary."""
        return {
            "incident_id": str(self.incident_id),
            "type": self.incident_type,
            "severity": self.severity.value,
            "status": self.status,
            "detection_timestamp": self.detection_timestamp.isoformat(),
            "resolution_timestamp": self.resolution_timestamp.isoformat() if self.resolution_timestamp else None,
            "is_data_breach": self.is_data_breach(),
            "requires_notification": self.requires_notification(),
            "affected_systems_count": len(self.affected_systems),
            "affected_users_count": len(self.affected_users),
            "compliance_frameworks": [f.value for f in self.compliance_implications],
            "time_to_resolution": str(self.get_time_to_resolution()) if self.get_time_to_resolution() else None
        }


@dataclass
class ComplianceViolation:
    """Compliance framework violation."""
    violation_id: UUID = field(default_factory=uuid4)
    framework: ComplianceFramework = ComplianceFramework.SOC2
    control_id: str = ""
    control_name: str = ""
    violation_type: str = ""
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    detection_date: datetime = field(default_factory=datetime.utcnow)
    remediation_deadline: Optional[datetime] = None
    status: str = "open"  # open, acknowledged, remediated, accepted_risk
    remediation_plan: Optional[str] = None
    remediation_cost_estimate: Optional[float] = None
    business_impact: Optional[str] = None
    technical_details: Dict[str, Any] = field(default_factory=dict)
    assigned_to: Optional[str] = None
    
    def __post_init__(self):
        """Validate compliance violation."""
        if not self.control_id:
            raise ValueError("Control ID cannot be empty")
        if not self.description:
            raise ValueError("Violation description cannot be empty")
        if self.severity not in ["low", "medium", "high", "critical"]:
            raise ValueError("Invalid violation severity")
        if self.status not in ["open", "acknowledged", "remediated", "accepted_risk"]:
            raise ValueError("Invalid violation status")
    
    def is_overdue(self) -> bool:
        """Check if remediation is overdue."""
        if not self.remediation_deadline:
            return False
        return datetime.utcnow() > self.remediation_deadline
    
    def get_days_until_deadline(self) -> Optional[int]:
        """Get days until remediation deadline."""
        if not self.remediation_deadline:
            return None
        delta = self.remediation_deadline - datetime.utcnow()
        return delta.days
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get violation summary."""
        return {
            "violation_id": str(self.violation_id),
            "framework": self.framework.value,
            "control_id": self.control_id,
            "control_name": self.control_name,
            "severity": self.severity,
            "status": self.status,
            "detection_date": self.detection_date.isoformat(),
            "remediation_deadline": self.remediation_deadline.isoformat() if self.remediation_deadline else None,
            "is_overdue": self.is_overdue(),
            "days_until_deadline": self.get_days_until_deadline(),
            "has_remediation_plan": bool(self.remediation_plan),
            "estimated_cost": self.remediation_cost_estimate
        }


@dataclass
class ComplianceReport:
    """Compliance assessment report."""
    report_id: UUID = field(default_factory=uuid4)
    framework: ComplianceFramework = ComplianceFramework.SOC2
    assessment_date: datetime = field(default_factory=datetime.utcnow)
    assessment_period: Optional[timedelta] = None
    scope: Optional[str] = None
    assessor: Optional[str] = None
    overall_score: float = 0.0
    controls_assessed: List[str] = field(default_factory=list)
    violations: List[ComplianceViolation] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    remediation_timeline: Optional[Dict[str, datetime]] = None
    next_assessment_date: Optional[datetime] = None
    certification_status: Optional[str] = None
    
    def __post_init__(self):
        """Validate compliance report."""
        if not (0.0 <= self.overall_score <= 1.0):
            raise ValueError("Overall score must be between 0.0 and 1.0")
    
    def get_compliance_percentage(self) -> float:
        """Get compliance percentage."""
        return self.overall_score * 100
    
    def get_critical_violations(self) -> List[ComplianceViolation]:
        """Get critical violations."""
        return [v for v in self.violations if v.severity == "critical"]
    
    def get_high_violations(self) -> List[ComplianceViolation]:
        """Get high severity violations."""
        return [v for v in self.violations if v.severity == "high"]
    
    def is_compliant(self, threshold: float = 0.8) -> bool:
        """Check if compliant based on threshold."""
        return self.overall_score >= threshold and len(self.get_critical_violations()) == 0
    
    def get_compliance_grade(self) -> str:
        """Get compliance grade."""
        if self.overall_score >= 0.95:
            return "A+"
        elif self.overall_score >= 0.90:
            return "A"
        elif self.overall_score >= 0.85:
            return "B+"
        elif self.overall_score >= 0.80:
            return "B"
        elif self.overall_score >= 0.75:
            return "C+"
        elif self.overall_score >= 0.70:
            return "C"
        else:
            return "F"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "report_id": str(self.report_id),
            "framework": self.framework.value,
            "assessment_date": self.assessment_date.isoformat(),
            "overall_score": self.overall_score,
            "compliance_percentage": self.get_compliance_percentage(),
            "compliance_grade": self.get_compliance_grade(),
            "is_compliant": self.is_compliant(),
            "controls_assessed_count": len(self.controls_assessed),
            "total_violations": len(self.violations),
            "critical_violations": len(self.get_critical_violations()),
            "high_violations": len(self.get_high_violations()),
            "recommendations_count": len(self.recommendations),
            "certification_status": self.certification_status,
            "next_assessment": self.next_assessment_date.isoformat() if self.next_assessment_date else None
        }


@dataclass
class PrivacyAssessment:
    """Privacy impact assessment."""
    assessment_id: UUID = field(default_factory=uuid4)
    assessment_date: datetime = field(default_factory=datetime.utcnow)
    data_processing_description: str = ""
    data_types_processed: List[str] = field(default_factory=list)
    processing_purposes: List[str] = field(default_factory=list)
    data_subjects_affected: int = 0
    legal_basis: Optional[str] = None
    privacy_risks: List[Dict[str, Any]] = field(default_factory=list)
    mitigation_measures: List[str] = field(default_factory=list)
    privacy_controls_implemented: List[PrivacyControl] = field(default_factory=list)
    data_retention_period: Optional[timedelta] = None
    third_party_sharing: bool = False
    cross_border_transfers: bool = False
    overall_privacy_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate privacy assessment."""
        if not self.data_processing_description:
            raise ValueError("Data processing description cannot be empty")
        if not (0.0 <= self.overall_privacy_score <= 1.0):
            raise ValueError("Privacy score must be between 0.0 and 1.0")
    
    def has_high_risk(self) -> bool:
        """Check if assessment indicates high privacy risk."""
        high_risk_indicators = [
            self.overall_privacy_score < 0.6,
            len(self.privacy_risks) > 5,
            self.third_party_sharing and not self.legal_basis,
            self.cross_border_transfers,
            "sensitive_personal" in [dt.lower() for dt in self.data_types_processed]
        ]
        return any(high_risk_indicators)
    
    def requires_dpia(self) -> bool:
        """Check if Data Protection Impact Assessment is required."""
        # GDPR Article 35 criteria
        dpia_triggers = [
            "automated_decision_making" in self.processing_purposes,
            "large_scale_monitoring" in self.processing_purposes,
            "sensitive_personal" in self.data_types_processed,
            self.data_subjects_affected > 10000,
            self.has_high_risk()
        ]
        return any(dpia_triggers)
    
    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get privacy assessment summary."""
        return {
            "assessment_id": str(self.assessment_id),
            "assessment_date": self.assessment_date.isoformat(),
            "overall_privacy_score": self.overall_privacy_score,
            "has_high_risk": self.has_high_risk(),
            "requires_dpia": self.requires_dpia(),
            "data_types_count": len(self.data_types_processed),
            "data_subjects_affected": self.data_subjects_affected,
            "privacy_risks_count": len(self.privacy_risks),
            "mitigation_measures_count": len(self.mitigation_measures),
            "privacy_controls": [pc.value for pc in self.privacy_controls_implemented],
            "third_party_sharing": self.third_party_sharing,
            "cross_border_transfers": self.cross_border_transfers,
            "legal_basis": self.legal_basis
        }


@dataclass
class DataRetentionPolicy:
    """Data retention policy definition."""
    policy_id: UUID = field(default_factory=uuid4)
    policy_name: str = ""
    data_classification: DataClassification = DataClassification.INTERNAL
    retention_period: timedelta = field(default_factory=lambda: timedelta(days=365))
    deletion_method: str = "secure_deletion"  # secure_deletion, anonymization, archival
    legal_hold_exceptions: List[str] = field(default_factory=list)
    applicable_regulations: List[ComplianceFramework] = field(default_factory=list)
    business_justification: str = ""
    automated_enforcement: bool = True
    review_frequency: timedelta = field(default_factory=lambda: timedelta(days=365))
    created_date: datetime = field(default_factory=datetime.utcnow)
    last_review_date: Optional[datetime] = None
    next_review_date: Optional[datetime] = None
    approved_by: Optional[str] = None
    
    def __post_init__(self):
        """Validate data retention policy."""
        if not self.policy_name:
            raise ValueError("Policy name cannot be empty")
        if self.retention_period.total_seconds() <= 0:
            raise ValueError("Retention period must be positive")
        if self.deletion_method not in ["secure_deletion", "anonymization", "archival"]:
            raise ValueError("Invalid deletion method")
        
        # Set next review date if not provided
        if not self.next_review_date:
            self.next_review_date = self.created_date + self.review_frequency
    
    def is_due_for_review(self) -> bool:
        """Check if policy is due for review."""
        if not self.next_review_date:
            return True
        return datetime.utcnow() >= self.next_review_date
    
    def should_delete_data(self, data_creation_date: datetime) -> bool:
        """Check if data should be deleted based on policy."""
        deletion_date = data_creation_date + self.retention_period
        return datetime.utcnow() >= deletion_date
    
    def get_retention_summary(self) -> Dict[str, Any]:
        """Get retention policy summary."""
        return {
            "policy_id": str(self.policy_id),
            "policy_name": self.policy_name,
            "data_classification": self.data_classification.value,
            "retention_days": self.retention_period.days,
            "deletion_method": self.deletion_method,
            "automated_enforcement": self.automated_enforcement,
            "is_due_for_review": self.is_due_for_review(),
            "applicable_regulations": [r.value for r in self.applicable_regulations],
            "legal_hold_exceptions_count": len(self.legal_hold_exceptions),
            "next_review_date": self.next_review_date.isoformat() if self.next_review_date else None
        }


@dataclass
class EncryptionPolicy:
    """Encryption policy definition."""
    policy_id: UUID = field(default_factory=uuid4)
    policy_name: str = ""
    data_classification_levels: List[DataClassification] = field(default_factory=list)
    encryption_algorithm: str = "AES-256-GCM"
    key_length: int = 256
    key_rotation_frequency: timedelta = field(default_factory=lambda: timedelta(days=90))
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    encryption_in_processing: bool = False
    key_management_system: str = "internal"  # internal, hsm, cloud_kms
    compliance_requirements: List[ComplianceFramework] = field(default_factory=list)
    performance_impact_acceptable: bool = True
    created_date: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate encryption policy."""
        if not self.policy_name:
            raise ValueError("Policy name cannot be empty")
        if self.key_length not in [128, 192, 256]:
            raise ValueError("Key length must be 128, 192, or 256 bits")
        if not self.data_classification_levels:
            raise ValueError("At least one data classification level must be specified")
    
    def applies_to_data_classification(self, classification: DataClassification) -> bool:
        """Check if policy applies to data classification."""
        return classification in self.data_classification_levels
    
    def requires_key_rotation(self, last_rotation: datetime) -> bool:
        """Check if key rotation is required."""
        next_rotation = last_rotation + self.key_rotation_frequency
        return datetime.utcnow() >= next_rotation
    
    def get_encryption_summary(self) -> Dict[str, Any]:
        """Get encryption policy summary."""
        return {
            "policy_id": str(self.policy_id),
            "policy_name": self.policy_name,
            "encryption_algorithm": self.encryption_algorithm,
            "key_length": self.key_length,
            "key_rotation_days": self.key_rotation_frequency.days,
            "encryption_at_rest": self.encryption_at_rest,
            "encryption_in_transit": self.encryption_in_transit,
            "encryption_in_processing": self.encryption_in_processing,
            "key_management_system": self.key_management_system,
            "applicable_classifications": [dc.value for dc in self.data_classification_levels],
            "compliance_requirements": [cr.value for cr in self.compliance_requirements]
        }
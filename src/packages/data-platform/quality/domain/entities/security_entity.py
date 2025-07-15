"""Security entities for data quality operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4


class PIIType(Enum):
    """Types of personally identifiable information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    CUSTOM = "custom"


class PrivacyLevel(Enum):
    """Privacy protection levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    CCPA = "ccpa"
    PCI_DSS = "pci_dss"
    FDA_21_CFR_11 = "fda_21_cfr_11"
    FERPA = "ferpa"
    GLBA = "glba"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    PRIVACY_VIOLATION = "privacy_violation"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_BREACH = "security_breach"


@dataclass
class PIIDetectionResult:
    """Result of PII detection analysis."""
    entity_id: UUID = field(default_factory=uuid4)
    pii_type: PIIType = field()
    confidence: float = field()
    location: Dict[str, Any] = field()  # column, row, offset etc.
    value_sample: Optional[str] = field(default=None)
    masked_value: Optional[str] = field(default=None)
    detection_method: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate PII detection result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class ConsentRecord:
    """Record of user consent for data processing."""
    consent_id: UUID = field(default_factory=uuid4)
    subject_id: str = field()
    purpose: str = field()
    legal_basis: str = field()
    consent_given: bool = field()
    consent_date: datetime = field()
    expiry_date: Optional[datetime] = field(default=None)
    withdrawal_date: Optional[datetime] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if not self.consent_given:
            return False
        if self.withdrawal_date:
            return False
        if self.expiry_date and datetime.now() > self.expiry_date:
            return False
        return True


@dataclass
class PrivacyImpactAssessment:
    """Privacy impact assessment for quality operations."""
    assessment_id: UUID = field(default_factory=uuid4)
    operation_name: str = field()
    description: str = field()
    data_categories: List[str] = field(default_factory=list)
    processing_purposes: List[str] = field(default_factory=list)
    legal_basis: str = field()
    data_subjects: List[str] = field(default_factory=list)
    risk_level: str = field(default="medium")
    mitigation_measures: List[str] = field(default_factory=list)
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    assessor: str = field()
    assessment_date: datetime = field(default_factory=datetime.now)
    review_date: Optional[datetime] = field(default=None)
    approved: bool = field(default=False)
    approval_date: Optional[datetime] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event for auditing and monitoring."""
    event_id: UUID = field(default_factory=uuid4)
    event_type: SecurityEventType = field()
    severity: str = field(default="info")
    user_id: Optional[str] = field(default=None)
    session_id: Optional[str] = field(default=None)
    resource: Optional[str] = field(default=None)
    action: str = field()
    result: str = field()  # success, failure, denied
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = field(default=None)
    user_agent: Optional[str] = field(default=None)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPermission:
    """Access permission definition."""
    permission_id: UUID = field(default_factory=uuid4)
    resource: str = field()
    action: str = field()
    conditions: Dict[str, Any] = field(default_factory=dict)
    granted_by: str = field()
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if permission is currently valid."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


@dataclass
class UserRole:
    """User role definition."""
    role_id: UUID = field(default_factory=uuid4)
    name: str = field()
    description: str = field()
    permissions: Set[str] = field(default_factory=set)
    inherited_roles: Set[str] = field(default_factory=set)
    created_by: str = field()
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncryptionConfig:
    """Encryption configuration."""
    config_id: UUID = field(default_factory=uuid4)
    algorithm: str = field()
    key_length: int = field()
    mode: str = field()
    padding: Optional[str] = field(default=None)
    key_derivation: Optional[str] = field(default=None)
    salt_length: Optional[int] = field(default=None)
    iterations: Optional[int] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditRecord:
    """Comprehensive audit record."""
    audit_id: UUID = field(default_factory=uuid4)
    operation_id: Optional[UUID] = field(default=None)
    user_id: str = field()
    action: str = field()
    resource_type: str = field()
    resource_id: str = field()
    previous_state: Optional[Dict[str, Any]] = field(default=None)
    new_state: Optional[Dict[str, Any]] = field(default=None)
    changes: Dict[str, Any] = field(default_factory=dict)
    business_justification: Optional[str] = field(default=None)
    compliance_context: List[ComplianceFramework] = field(default_factory=list)
    risk_assessment: Optional[str] = field(default=None)
    ip_address: Optional[str] = field(default=None)
    user_agent: Optional[str] = field(default=None)
    session_id: Optional[str] = field(default=None)
    timestamp: datetime = field(default_factory=datetime.now)
    retention_until: Optional[datetime] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceAssessment:
    """Compliance assessment result."""
    assessment_id: UUID = field(default_factory=uuid4)
    framework: ComplianceFramework = field()
    scope: str = field()
    assessment_date: datetime = field(default_factory=datetime.now)
    assessor: str = field()
    overall_score: float = field()
    requirement_scores: Dict[str, float] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    next_assessment_date: datetime = field()
    approved: bool = field(default=False)
    approval_date: Optional[datetime] = field(default=None)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate compliance assessment."""
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError("Overall score must be between 0.0 and 1.0")


class SecurityPolicy(ABC):
    """Abstract base class for security policies."""
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate if the policy is satisfied."""
        pass
    
    @abstractmethod
    def get_requirements(self) -> List[str]:
        """Get list of requirements for this policy."""
        pass


@dataclass
class ThreatDetectionResult:
    """Result of threat detection analysis."""
    detection_id: UUID = field(default_factory=uuid4)
    threat_type: str = field()
    severity: str = field()
    confidence: float = field()
    description: str = field()
    indicators: List[str] = field(default_factory=list)
    affected_resources: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    detection_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate threat detection result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
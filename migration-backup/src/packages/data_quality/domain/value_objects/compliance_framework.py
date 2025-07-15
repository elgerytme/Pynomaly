"""Compliance framework value objects for regulatory compliance and monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, date, timedelta
from uuid import UUID, uuid4


class ComplianceType(str, Enum):
    """Types of compliance requirements."""
    DATA_PROTECTION = "data_protection"
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    INDUSTRY_SPECIFIC = "industry_specific"
    SECURITY = "security"
    OPERATIONAL = "operational"
    ENVIRONMENTAL = "environmental"


class ComplianceStatus(str, Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    PENDING_ASSESSMENT = "pending_assessment"
    EXEMPTED = "exempted"
    NOT_APPLICABLE = "not_applicable"


class AuditFrequency(str, Enum):
    """Frequency of compliance audits."""
    CONTINUOUS = "continuous"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"
    AD_HOC = "ad_hoc"


class ViolationSeverity(str, Enum):
    """Severity levels for compliance violations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class RemediationStatus(str, Enum):
    """Status of violation remediation."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_VERIFICATION = "pending_verification"
    RESOLVED = "resolved"
    ACCEPTED_RISK = "accepted_risk"
    FALSE_POSITIVE = "false_positive"


@dataclass(frozen=True)
class ComplianceRequirement:
    """Individual compliance requirement within a framework."""
    requirement_id: str
    name: str
    description: str
    category: str
    is_mandatory: bool = True
    
    # Implementation details
    implementation_guidance: str = ""
    technical_controls: List[str] = field(default_factory=list)
    administrative_controls: List[str] = field(default_factory=list)
    physical_controls: List[str] = field(default_factory=list)
    
    # Assessment criteria
    assessment_criteria: List[str] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    testing_procedures: List[str] = field(default_factory=list)
    
    # Compliance tracking
    audit_frequency: AuditFrequency = AuditFrequency.QUARTERLY
    automated_monitoring: bool = True
    risk_level: str = "medium"  # low, medium, high, critical
    
    # Regulatory references
    regulatory_citations: List[str] = field(default_factory=list)
    standards_references: List[str] = field(default_factory=list)
    
    @property
    def control_count(self) -> int:
        """Get total number of controls."""
        return (len(self.technical_controls) + 
                len(self.administrative_controls) + 
                len(self.physical_controls))
    
    @property
    def is_high_risk(self) -> bool:
        """Check if requirement is high risk."""
        return self.risk_level in ["high", "critical"]


@dataclass(frozen=True)
class ComplianceAssessment:
    """Assessment of compliance with a specific requirement."""
    assessment_id: str
    requirement_id: str
    assessor_id: str
    assessment_date: datetime
    status: ComplianceStatus
    
    # Assessment details
    assessment_method: str  # automated, manual, hybrid
    evidence_collected: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    score: Optional[float] = None  # 0-100 compliance score
    
    # Violations and gaps
    violations_found: List[str] = field(default_factory=list)
    gaps_identified: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Remediation
    remediation_required: bool = False
    remediation_deadline: Optional[datetime] = None
    remediation_plan: List[str] = field(default_factory=list)
    
    # Documentation
    assessment_notes: str = ""
    attachments: List[str] = field(default_factory=list)
    next_assessment_due: Optional[datetime] = None
    
    @property
    def is_compliant(self) -> bool:
        """Check if assessment shows compliance."""
        return self.status == ComplianceStatus.COMPLIANT
    
    @property
    def has_violations(self) -> bool:
        """Check if assessment found violations."""
        return len(self.violations_found) > 0
    
    @property
    def is_overdue_for_remediation(self) -> bool:
        """Check if remediation is overdue."""
        if not self.remediation_required or not self.remediation_deadline:
            return False
        return datetime.now() > self.remediation_deadline


@dataclass(frozen=True)
class ComplianceViolation:
    """Specific compliance violation instance."""
    violation_id: str
    requirement_id: str
    framework_id: str
    violation_type: str
    severity: ViolationSeverity
    
    # Violation details
    description: str
    discovered_date: datetime
    discovered_by: str
    discovery_method: str  # automated, audit, incident, self_reported
    
    # Impact assessment
    business_impact: str = "medium"  # low, medium, high, critical
    affected_systems: List[str] = field(default_factory=list)
    affected_data_subjects: Optional[int] = None
    potential_penalties: str = ""
    
    # Root cause
    root_cause: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    failure_mode: str = ""  # control_failure, process_gap, human_error, etc.
    
    # Remediation
    remediation_status: RemediationStatus = RemediationStatus.OPEN
    assigned_to: Optional[str] = None
    remediation_plan: List[str] = field(default_factory=list)
    remediation_deadline: Optional[datetime] = None
    resolution_date: Optional[datetime] = None
    
    # Documentation
    evidence: List[str] = field(default_factory=list)
    regulatory_notification_required: bool = False
    regulatory_notification_deadline: Optional[datetime] = None
    external_reporting_completed: bool = False
    
    @property
    def is_critical(self) -> bool:
        """Check if violation is critical severity."""
        return self.severity == ViolationSeverity.CRITICAL
    
    @property
    def requires_immediate_attention(self) -> bool:
        """Check if violation requires immediate attention."""
        return (self.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH] or
                self.regulatory_notification_required)
    
    @property
    def days_open(self) -> int:
        """Get number of days violation has been open."""
        end_date = self.resolution_date or datetime.now()
        return (end_date - self.discovered_date).days
    
    @property
    def is_resolved(self) -> bool:
        """Check if violation is resolved."""
        return self.remediation_status == RemediationStatus.RESOLVED


@dataclass(frozen=True)
class AuditTrail:
    """Audit trail entry for compliance monitoring."""
    entry_id: str
    timestamp: datetime
    event_type: str
    actor_id: str
    resource_id: str
    action: str
    
    # Event details
    event_description: str
    outcome: str  # success, failure, partial
    source_system: str = ""
    session_id: Optional[str] = None
    
    # Data context
    before_value: Optional[str] = None
    after_value: Optional[str] = None
    affected_fields: List[str] = field(default_factory=list)
    
    # Security context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    risk_score: Optional[float] = None
    
    # Compliance context
    compliance_frameworks: List[str] = field(default_factory=list)
    retention_period_days: int = 2557  # 7 years default
    is_immutable: bool = True
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_high_risk_event(self) -> bool:
        """Check if this is a high-risk audit event."""
        high_risk_actions = [
            "data_access", "permission_change", "policy_modification",
            "system_configuration_change", "data_export", "data_deletion"
        ]
        return self.action in high_risk_actions or (self.risk_score and self.risk_score > 0.8)
    
    @property
    def retention_expiry_date(self) -> datetime:
        """Get retention expiry date."""
        return self.timestamp + timedelta(days=self.retention_period_days)


@dataclass(frozen=True)
class ComplianceReport:
    """Comprehensive compliance report."""
    report_id: str
    framework_id: str
    reporting_period_start: date
    reporting_period_end: date
    generated_date: datetime
    generated_by: str
    
    # Overall compliance status
    overall_status: ComplianceStatus
    compliance_percentage: float
    requirements_assessed: int
    requirements_compliant: int
    requirements_non_compliant: int
    
    # Violations summary
    total_violations: int
    critical_violations: int
    high_violations: int
    medium_violations: int
    low_violations: int
    resolved_violations: int
    
    # Assessment summary
    assessments_conducted: List[ComplianceAssessment] = field(default_factory=list)
    violations_found: List[ComplianceViolation] = field(default_factory=list)
    improvement_trends: Dict[str, float] = field(default_factory=dict)
    
    # Risk assessment
    risk_score: float = 0.0
    high_risk_areas: List[str] = field(default_factory=list)
    remediation_priorities: List[str] = field(default_factory=list)
    
    # Recommendations
    executive_summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    
    # Attestations
    sign_offs: List[Dict[str, str]] = field(default_factory=list)
    external_validation: Optional[str] = None
    certification_status: Optional[str] = None
    
    @property
    def violation_rate(self) -> float:
        """Calculate violation rate."""
        if self.requirements_assessed == 0:
            return 0.0
        return (self.requirements_non_compliant / self.requirements_assessed) * 100
    
    @property
    def critical_violation_rate(self) -> float:
        """Calculate critical violation rate."""
        if self.total_violations == 0:
            return 0.0
        return (self.critical_violations / self.total_violations) * 100
    
    @property
    def is_satisfactory(self) -> bool:
        """Check if compliance is at satisfactory level."""
        return (self.compliance_percentage >= 90.0 and 
                self.critical_violations == 0 and
                self.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT])


@dataclass(frozen=True)
class ComplianceFrameworkDefinition:
    """Definition of a compliance framework."""
    framework_id: str
    name: str
    description: str
    compliance_type: ComplianceType
    version: str = "1.0.0"
    
    # Framework structure
    requirements: List[ComplianceRequirement] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    mandatory_requirements: List[str] = field(default_factory=list)
    
    # Implementation details
    implementation_timeline_days: int = 180
    certification_required: bool = False
    external_audit_required: bool = False
    self_assessment_allowed: bool = True
    
    # Regulatory information
    regulatory_authority: str = ""
    effective_date: Optional[date] = None
    jurisdiction: List[str] = field(default_factory=list)
    industry_sectors: List[str] = field(default_factory=list)
    
    # Framework metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    maintained_by: str = ""
    
    # Assessment configuration
    default_audit_frequency: AuditFrequency = AuditFrequency.ANNUALLY
    automated_monitoring_enabled: bool = True
    continuous_monitoring_required: bool = False
    
    @property
    def mandatory_requirement_count(self) -> int:
        """Get count of mandatory requirements."""
        return len([req for req in self.requirements if req.is_mandatory])
    
    @property
    def total_control_count(self) -> int:
        """Get total control count across all requirements."""
        return sum(req.control_count for req in self.requirements)
    
    @property
    def high_risk_requirement_count(self) -> int:
        """Get count of high-risk requirements."""
        return len([req for req in self.requirements if req.is_high_risk])
    
    def get_requirements_by_category(self, category: str) -> List[ComplianceRequirement]:
        """Get requirements for specific category."""
        return [req for req in self.requirements if req.category == category]
    
    def get_mandatory_requirements(self) -> List[ComplianceRequirement]:
        """Get mandatory requirements."""
        return [req for req in self.requirements if req.is_mandatory]
    
    @classmethod
    def create_gdpr_framework(cls) -> ComplianceFrameworkDefinition:
        """Create GDPR compliance framework."""
        requirements = [
            ComplianceRequirement(
                requirement_id="gdpr_art_5",
                name="Principles of Processing",
                description="Personal data must be processed lawfully, fairly and transparently",
                category="data_processing",
                is_mandatory=True,
                implementation_guidance="Implement clear data processing purposes and legal bases",
                technical_controls=["data_classification", "purpose_limitation", "consent_management"],
                audit_frequency=AuditFrequency.QUARTERLY,
                risk_level="high"
            ),
            ComplianceRequirement(
                requirement_id="gdpr_art_17",
                name="Right to Erasure",
                description="Data subjects have right to erasure of personal data",
                category="data_rights",
                is_mandatory=True,
                implementation_guidance="Implement data deletion capabilities and processes",
                technical_controls=["data_deletion", "right_to_erasure", "data_inventory"],
                audit_frequency=AuditFrequency.MONTHLY,
                risk_level="high"
            ),
            ComplianceRequirement(
                requirement_id="gdpr_art_32",
                name="Security of Processing",
                description="Implement appropriate security measures for personal data",
                category="security",
                is_mandatory=True,
                implementation_guidance="Implement encryption, access controls, and security monitoring",
                technical_controls=["encryption", "access_control", "security_monitoring"],
                audit_frequency=AuditFrequency.CONTINUOUS,
                risk_level="critical"
            )
        ]
        
        return cls(
            framework_id="gdpr_2018",
            name="General Data Protection Regulation (GDPR)",
            description="EU regulation on data protection and privacy",
            compliance_type=ComplianceType.DATA_PROTECTION,
            requirements=requirements,
            categories=["data_processing", "data_rights", "security", "governance"],
            regulatory_authority="European Data Protection Board",
            effective_date=date(2018, 5, 25),
            jurisdiction=["EU", "EEA"],
            certification_required=False,
            external_audit_required=True,
            continuous_monitoring_required=True
        )
    
    @classmethod
    def create_hipaa_framework(cls) -> ComplianceFrameworkDefinition:
        """Create HIPAA compliance framework."""
        requirements = [
            ComplianceRequirement(
                requirement_id="hipaa_164_308",
                name="Administrative Safeguards",
                description="Implement administrative safeguards for PHI",
                category="administrative",
                is_mandatory=True,
                implementation_guidance="Establish security officer and workforce training",
                administrative_controls=["security_officer", "workforce_training", "access_management"],
                audit_frequency=AuditFrequency.ANNUALLY,
                risk_level="high"
            ),
            ComplianceRequirement(
                requirement_id="hipaa_164_310",
                name="Physical Safeguards",
                description="Implement physical safeguards for PHI",
                category="physical",
                is_mandatory=True,
                implementation_guidance="Control physical access to systems containing PHI",
                physical_controls=["facility_access", "workstation_use", "device_controls"],
                audit_frequency=AuditFrequency.SEMI_ANNUALLY,
                risk_level="medium"
            ),
            ComplianceRequirement(
                requirement_id="hipaa_164_312",
                name="Technical Safeguards",
                description="Implement technical safeguards for PHI",
                category="technical",
                is_mandatory=True,
                implementation_guidance="Implement access control, audit controls, and transmission security",
                technical_controls=["access_control", "audit_controls", "integrity", "transmission_security"],
                audit_frequency=AuditFrequency.QUARTERLY,
                risk_level="high"
            )
        ]
        
        return cls(
            framework_id="hipaa_1996",
            name="Health Insurance Portability and Accountability Act (HIPAA)",
            description="US federal law for healthcare data protection",
            compliance_type=ComplianceType.HEALTHCARE,
            requirements=requirements,
            categories=["administrative", "physical", "technical"],
            regulatory_authority="Department of Health and Human Services",
            effective_date=date(1996, 8, 21),
            jurisdiction=["US"],
            industry_sectors=["healthcare"],
            certification_required=False,
            external_audit_required=True
        )
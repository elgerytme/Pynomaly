"""
Compliance domain entities for enterprise governance and regulatory compliance.
"""

from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    GDPR = "gdpr"
    ISO27001 = "iso27001"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    CCPA = "ccpa"
    FedRAMP = "fedramp"
    NIST = "nist"
    CSF = "csf"  # NIST Cybersecurity Framework


class ComplianceStatus(str, Enum):
    """Compliance status enumeration."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL_COMPLIANCE = "partial_compliance"
    UNDER_REVIEW = "under_review"
    REMEDIATION_IN_PROGRESS = "remediation_in_progress"
    NOT_APPLICABLE = "not_applicable"


class ControlStatus(str, Enum):
    """Control implementation status."""
    IMPLEMENTED = "implemented"
    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    PLANNED = "planned"
    DEFERRED = "deferred"
    NOT_APPLICABLE = "not_applicable"


class EvidenceType(str, Enum):
    """Types of compliance evidence."""
    DOCUMENT = "document"
    SCREENSHOT = "screenshot"
    LOG_EXPORT = "log_export"
    POLICY = "policy"
    PROCEDURE = "procedure"
    CONFIGURATION = "configuration"
    REPORT = "report"
    CERTIFICATE = "certificate"
    AUDIT_TRAIL = "audit_trail"


class ComplianceControl(BaseModel):
    """
    Individual compliance control within a framework.
    
    Represents a specific requirement or control that must be
    implemented to achieve compliance with a regulatory framework.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique control identifier")
    
    # Control identification
    framework: ComplianceFramework = Field(..., description="Compliance framework")
    control_id: str = Field(..., description="Framework-specific control ID")
    control_number: str = Field(..., description="Control number (e.g., CC6.1)")
    title: str = Field(..., description="Control title")
    description: str = Field(..., description="Detailed control description")
    
    # Control categorization
    category: str = Field(..., description="Control category")
    subcategory: Optional[str] = Field(None, description="Control subcategory")
    domain: str = Field(..., description="Control domain")
    
    # Implementation details
    status: ControlStatus = Field(default=ControlStatus.NOT_IMPLEMENTED)
    implementation_date: Optional[date] = Field(None, description="Date implemented")
    last_reviewed_date: Optional[date] = Field(None, description="Last review date")
    next_review_date: Optional[date] = Field(None, description="Next scheduled review")
    
    # Risk and priority
    risk_level: str = Field(..., description="Risk level (low, medium, high, critical)")
    priority: int = Field(default=3, ge=1, le=5, description="Implementation priority (1=highest)")
    
    # Implementation details
    implementation_notes: str = Field(default="", description="Implementation notes")
    remediation_plan: str = Field(default="", description="Remediation plan")
    responsible_party: Optional[str] = Field(None, description="Responsible person/team")
    
    # Evidence and validation
    evidence_required: List[EvidenceType] = Field(default_factory=list)
    evidence_collected: List[str] = Field(default_factory=list, description="Evidence file references")
    validation_method: str = Field(default="", description="How control is validated")
    
    # Automation
    automated_check: bool = Field(default=False, description="Has automated validation")
    automation_script: Optional[str] = Field(None, description="Automation script reference")
    last_automated_check: Optional[datetime] = Field(None, description="Last automated check")
    
    # Metadata
    tenant_id: Optional[UUID] = Field(None, description="Associated tenant")
    tags: List[str] = Field(default_factory=list, description="Control tags")
    external_references: List[str] = Field(default_factory=list, description="External references")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def is_compliant(self) -> bool:
        """Check if control is compliant."""
        return self.status == ControlStatus.IMPLEMENTED
    
    def is_overdue_for_review(self) -> bool:
        """Check if control is overdue for review."""
        if not self.next_review_date:
            return False
        return date.today() > self.next_review_date
    
    def update_status(self, status: ControlStatus, notes: str = "") -> None:
        """Update control status with notes."""
        self.status = status
        self.updated_at = datetime.utcnow()
        
        if status == ControlStatus.IMPLEMENTED and not self.implementation_date:
            self.implementation_date = date.today()
        
        if notes:
            self.implementation_notes += f"\n[{datetime.utcnow().isoformat()}] {notes}"
    
    def add_evidence(self, evidence_ref: str, evidence_type: EvidenceType) -> None:
        """Add evidence reference to control."""
        if evidence_ref not in self.evidence_collected:
            self.evidence_collected.append(evidence_ref)
            self.updated_at = datetime.utcnow()
    
    def schedule_review(self, months_ahead: int = 12) -> None:
        """Schedule next review date."""
        from dateutil.relativedelta import relativedelta
        self.next_review_date = date.today() + relativedelta(months=months_ahead)
        self.updated_at = datetime.utcnow()


class ComplianceAssessment(BaseModel):
    """
    Comprehensive compliance assessment for a specific framework.
    
    Tracks overall compliance status, control implementation,
    and assessment results for regulatory frameworks.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Assessment identifier")
    
    # Assessment identification
    tenant_id: UUID = Field(..., description="Tenant being assessed")
    framework: ComplianceFramework = Field(..., description="Compliance framework")
    assessment_name: str = Field(..., description="Assessment name")
    description: str = Field(..., description="Assessment description")
    
    # Assessment scope
    scope: str = Field(..., description="Assessment scope")
    systems_in_scope: List[str] = Field(default_factory=list)
    exclusions: List[str] = Field(default_factory=list)
    
    # Assessment timeline
    start_date: date = Field(..., description="Assessment start date")
    end_date: Optional[date] = Field(None, description="Assessment end date")
    report_due_date: Optional[date] = Field(None, description="Report due date")
    
    # Assessment team
    lead_assessor: str = Field(..., description="Lead assessor")
    assessment_team: List[str] = Field(default_factory=list)
    external_auditor: Optional[str] = Field(None, description="External auditor")
    
    # Controls and results
    total_controls: int = Field(default=0, description="Total controls assessed")
    controls_implemented: int = Field(default=0, description="Controls fully implemented")
    controls_partial: int = Field(default=0, description="Controls partially implemented")
    controls_not_implemented: int = Field(default=0, description="Controls not implemented")
    
    # Overall status
    overall_status: ComplianceStatus = Field(default=ComplianceStatus.UNDER_REVIEW)
    compliance_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_score: float = Field(default=0.0, description="Overall risk score")
    
    # Findings
    critical_findings: int = Field(default=0, description="Critical findings count")
    high_findings: int = Field(default=0, description="High risk findings count")
    medium_findings: int = Field(default=0, description="Medium risk findings count")
    low_findings: int = Field(default=0, description="Low risk findings count")
    
    # Documentation
    executive_summary: str = Field(default="", description="Executive summary")
    recommendations: List[str] = Field(default_factory=list)
    remediation_timeline: Optional[str] = Field(None, description="Remediation timeline")
    
    # Certification
    certification_achieved: bool = Field(default=False)
    certificate_number: Optional[str] = Field(None, description="Certificate number")
    certificate_expiry: Optional[date] = Field(None, description="Certificate expiry")
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    attachments: List[str] = Field(default_factory=list, description="Assessment artifacts")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="Assessment completion time")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    @validator('compliance_percentage')
    def calculate_compliance_percentage(cls, v, values):
        """Auto-calculate compliance percentage from control counts."""
        if 'total_controls' in values and values['total_controls'] > 0:
            implemented = values.get('controls_implemented', 0)
            return (implemented / values['total_controls']) * 100.0
        return v
    
    def update_control_counts(self, controls: List[ComplianceControl]) -> None:
        """Update control counts from control list."""
        self.total_controls = len(controls)
        self.controls_implemented = sum(1 for c in controls if c.status == ControlStatus.IMPLEMENTED)
        self.controls_partial = sum(1 for c in controls if c.status == ControlStatus.PARTIALLY_IMPLEMENTED)
        self.controls_not_implemented = sum(1 for c in controls if c.status == ControlStatus.NOT_IMPLEMENTED)
        
        # Update compliance percentage
        if self.total_controls > 0:
            self.compliance_percentage = (self.controls_implemented / self.total_controls) * 100.0
        
        # Update overall status based on compliance percentage
        if self.compliance_percentage >= 95.0:
            self.overall_status = ComplianceStatus.COMPLIANT
        elif self.compliance_percentage >= 80.0:
            self.overall_status = ComplianceStatus.PARTIAL_COMPLIANCE
        else:
            self.overall_status = ComplianceStatus.NON_COMPLIANT
        
        self.updated_at = datetime.utcnow()
    
    def add_finding(self, risk_level: str) -> None:
        """Add a finding to the assessment."""
        if risk_level.lower() == "critical":
            self.critical_findings += 1
        elif risk_level.lower() == "high":
            self.high_findings += 1
        elif risk_level.lower() == "medium":
            self.medium_findings += 1
        elif risk_level.lower() == "low":
            self.low_findings += 1
        
        self.updated_at = datetime.utcnow()
    
    def complete_assessment(self) -> None:
        """Mark assessment as completed."""
        self.end_date = date.today()
        self.completed_at = datetime.utcnow()
        
        # Set final status if not already set
        if self.overall_status == ComplianceStatus.UNDER_REVIEW:
            if self.compliance_percentage >= 95.0:
                self.overall_status = ComplianceStatus.COMPLIANT
            elif self.compliance_percentage >= 80.0:
                self.overall_status = ComplianceStatus.PARTIAL_COMPLIANCE
            else:
                self.overall_status = ComplianceStatus.NON_COMPLIANT
    
    def is_overdue(self) -> bool:
        """Check if assessment is overdue."""
        if not self.report_due_date:
            return False
        return date.today() > self.report_due_date and not self.completed_at
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary statistics."""
        return {
            "framework": self.framework,
            "overall_status": self.overall_status,
            "compliance_percentage": self.compliance_percentage,
            "total_controls": self.total_controls,
            "implemented": self.controls_implemented,
            "partial": self.controls_partial,
            "not_implemented": self.controls_not_implemented,
            "critical_findings": self.critical_findings,
            "high_findings": self.high_findings,
            "medium_findings": self.medium_findings,
            "low_findings": self.low_findings,
            "certification_achieved": self.certification_achieved,
            "is_overdue": self.is_overdue()
        }


class ComplianceReport(BaseModel):
    """
    Compliance reporting for regulatory requirements and internal governance.
    
    Generates comprehensive compliance reports with evidence,
    findings, and remediation recommendations.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Report identifier")
    
    # Report identification
    tenant_id: UUID = Field(..., description="Tenant ID")
    assessment_id: UUID = Field(..., description="Associated assessment ID")
    report_type: str = Field(..., description="Type of report")
    title: str = Field(..., description="Report title")
    
    # Report metadata
    framework: ComplianceFramework = Field(..., description="Compliance framework")
    report_period_start: date = Field(..., description="Report period start")
    report_period_end: date = Field(..., description="Report period end")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Report author
    generated_by: str = Field(..., description="Report generator")
    reviewed_by: Optional[str] = Field(None, description="Report reviewer")
    approved_by: Optional[str] = Field(None, description="Report approver")
    
    # Report content
    executive_summary: str = Field(..., description="Executive summary")
    methodology: str = Field(..., description="Assessment methodology")
    scope_and_limitations: str = Field(..., description="Report scope and limitations")
    
    # Findings and results
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    remediation_plan: str = Field(default="", description="Remediation plan")
    
    # Compliance metrics
    compliance_score: float = Field(..., ge=0.0, le=100.0, description="Overall compliance score")
    risk_rating: str = Field(..., description="Overall risk rating")
    control_effectiveness: str = Field(..., description="Control effectiveness rating")
    
    # Supporting evidence
    evidence_references: List[str] = Field(default_factory=list)
    attachments: List[str] = Field(default_factory=list)
    
    # Report status
    status: str = Field(default="draft", description="Report status")
    version: str = Field(default="1.0", description="Report version")
    
    # Distribution
    distribution_list: List[str] = Field(default_factory=list)
    confidentiality_level: str = Field(default="internal", description="Confidentiality classification")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    published_at: Optional[datetime] = Field(None, description="Report publication time")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def add_finding(
        self,
        control_id: str,
        title: str,
        description: str,
        risk_level: str,
        recommendation: str
    ) -> None:
        """Add a finding to the report."""
        finding = {
            "id": str(uuid4()),
            "control_id": control_id,
            "title": title,
            "description": description,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "identified_at": datetime.utcnow().isoformat()
        }
        self.findings.append(finding)
        self.updated_at = datetime.utcnow()
    
    def add_recommendation(
        self,
        priority: str,
        title: str,
        description: str,
        timeline: str,
        responsible_party: str
    ) -> None:
        """Add a recommendation to the report."""
        recommendation = {
            "id": str(uuid4()),
            "priority": priority,
            "title": title,
            "description": description,
            "timeline": timeline,
            "responsible_party": responsible_party,
            "created_at": datetime.utcnow().isoformat()
        }
        self.recommendations.append(recommendation)
        self.updated_at = datetime.utcnow()
    
    def finalize_report(self, approved_by: str) -> None:
        """Finalize and approve the report."""
        self.status = "approved"
        self.approved_by = approved_by
        self.published_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def get_findings_by_risk(self, risk_level: str) -> List[Dict[str, Any]]:
        """Get findings filtered by risk level."""
        return [f for f in self.findings if f.get("risk_level") == risk_level]
    
    def get_high_priority_recommendations(self) -> List[Dict[str, Any]]:
        """Get high priority recommendations."""
        return [r for r in self.recommendations if r.get("priority") in ["critical", "high"]]


class DataPrivacyRecord(BaseModel):
    """
    Data privacy and protection record for GDPR/CCPA compliance.
    
    Tracks data processing activities, consent, and privacy rights
    for personal data protection compliance.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Privacy record identifier")
    
    # Record identification
    tenant_id: UUID = Field(..., description="Tenant ID")
    data_subject_id: str = Field(..., description="Data subject identifier")
    record_type: str = Field(..., description="Type of privacy record")
    
    # Data processing details
    processing_purpose: str = Field(..., description="Purpose of data processing")
    data_categories: List[str] = Field(..., description="Categories of personal data")
    legal_basis: str = Field(..., description="Legal basis for processing")
    
    # Consent management
    consent_given: bool = Field(default=False, description="Consent status")
    consent_date: Optional[datetime] = Field(None, description="Consent given date")
    consent_withdrawn_date: Optional[datetime] = Field(None, description="Consent withdrawal date")
    consent_method: Optional[str] = Field(None, description="How consent was obtained")
    
    # Data retention
    retention_period: str = Field(..., description="Data retention period")
    retention_start_date: date = Field(..., description="Retention period start")
    scheduled_deletion_date: Optional[date] = Field(None, description="Scheduled deletion date")
    
    # Data sharing
    data_shared_with: List[str] = Field(default_factory=list, description="Third parties data shared with")
    transfer_countries: List[str] = Field(default_factory=list, description="Countries data transferred to")
    safeguards_in_place: List[str] = Field(default_factory=list, description="Transfer safeguards")
    
    # Rights exercised
    access_requests: List[datetime] = Field(default_factory=list, description="Data access requests")
    rectification_requests: List[datetime] = Field(default_factory=list, description="Data correction requests")
    erasure_requests: List[datetime] = Field(default_factory=list, description="Data deletion requests")
    portability_requests: List[datetime] = Field(default_factory=list, description="Data portability requests")
    
    # Data breach tracking
    breach_incidents: List[str] = Field(default_factory=list, description="Related breach incidents")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    
    def withdraw_consent(self) -> None:
        """Record consent withdrawal."""
        self.consent_given = False
        self.consent_withdrawn_date = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def is_retention_expired(self) -> bool:
        """Check if data retention period has expired."""
        if not self.scheduled_deletion_date:
            return False
        return date.today() >= self.scheduled_deletion_date
    
    def add_rights_request(self, request_type: str) -> None:
        """Add a data subject rights request."""
        now = datetime.utcnow()
        
        if request_type == "access":
            self.access_requests.append(now)
        elif request_type == "rectification":
            self.rectification_requests.append(now)
        elif request_type == "erasure":
            self.erasure_requests.append(now)
        elif request_type == "portability":
            self.portability_requests.append(now)
        
        self.updated_at = now
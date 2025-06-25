"""Domain entities for governance framework and audit management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class GovernanceLevel(Enum):
    """Governance maturity levels."""

    BASIC = "basic"
    MANAGED = "managed"
    DEFINED = "defined"
    QUANTITATIVELY_MANAGED = "quantitatively_managed"
    OPTIMIZING = "optimizing"


class AuditType(Enum):
    """Types of audits."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    REGULATORY = "regulatory"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    SECURITY = "security"
    PRIVACY = "privacy"


class ComplianceStatus(Enum):
    """Compliance status levels."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"
    UNDER_REVIEW = "under_review"
    REMEDIATION_REQUIRED = "remediation_required"


class DataClassification(Enum):
    """Data classification levels for governance."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class WorkflowStatus(Enum):
    """Workflow approval status."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    IMPLEMENTED = "implemented"
    CLOSED = "closed"


@dataclass
class GovernanceFramework:
    """Comprehensive governance framework definition."""

    framework_id: UUID = field(default_factory=uuid4)
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    governance_level: GovernanceLevel = GovernanceLevel.BASIC
    established_date: datetime = field(default_factory=datetime.utcnow)
    last_review_date: datetime | None = None
    next_review_date: datetime | None = None
    review_frequency_months: int = 12

    # Framework components
    policies: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)

    # Governance structure
    governing_body: str | None = None
    stakeholders: list[str] = field(default_factory=list)
    responsible_parties: dict[str, str] = field(default_factory=dict)

    # Scope and applicability
    applicable_domains: list[str] = field(default_factory=list)
    geographical_scope: list[str] = field(default_factory=list)
    regulatory_requirements: list[str] = field(default_factory=list)

    # Maturity assessment
    maturity_score: float = 0.0
    improvement_areas: list[str] = field(default_factory=list)
    action_plan: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate governance framework."""
        if not self.name:
            raise ValueError("Framework name cannot be empty")

        if not self.next_review_date:
            self.next_review_date = self.established_date + timedelta(
                days=30 * self.review_frequency_months
            )

    def is_due_for_review(self) -> bool:
        """Check if framework is due for review."""
        if not self.next_review_date:
            return True
        return datetime.utcnow() >= self.next_review_date

    def calculate_maturity_score(self) -> float:
        """Calculate governance maturity score."""
        components = [
            len(self.policies) > 0,
            len(self.procedures) > 0,
            len(self.controls) > 0,
            len(self.metrics) > 0,
            self.governing_body is not None,
            len(self.stakeholders) > 0,
            len(self.responsible_parties) > 0,
        ]

        base_score = sum(components) / len(components) * 100

        # Adjust based on governance level
        level_multipliers = {
            GovernanceLevel.BASIC: 0.6,
            GovernanceLevel.MANAGED: 0.7,
            GovernanceLevel.DEFINED: 0.8,
            GovernanceLevel.QUANTITATIVELY_MANAGED: 0.9,
            GovernanceLevel.OPTIMIZING: 1.0,
        }

        self.maturity_score = base_score * level_multipliers[self.governance_level]
        return self.maturity_score

    def update_review_date(self) -> None:
        """Update review dates after assessment."""
        self.last_review_date = datetime.utcnow()
        self.next_review_date = self.last_review_date + timedelta(
            days=30 * self.review_frequency_months
        )

    def get_framework_summary(self) -> dict[str, Any]:
        """Get framework summary."""
        return {
            "framework_id": str(self.framework_id),
            "name": self.name,
            "version": self.version,
            "governance_level": self.governance_level.value,
            "maturity_score": self.calculate_maturity_score(),
            "is_due_for_review": self.is_due_for_review(),
            "components_count": {
                "policies": len(self.policies),
                "procedures": len(self.procedures),
                "controls": len(self.controls),
                "metrics": len(self.metrics),
            },
            "stakeholders_count": len(self.stakeholders),
            "applicable_domains": self.applicable_domains,
            "regulatory_requirements": self.regulatory_requirements,
        }


@dataclass
class AuditEvidence:
    """Audit evidence and supporting documentation."""

    evidence_id: UUID = field(default_factory=uuid4)
    evidence_type: str = ""  # document, screenshot, log, interview, observation
    title: str = ""
    description: str = ""
    collected_by: str = ""
    collection_date: datetime = field(default_factory=datetime.utcnow)
    source_system: str | None = None
    file_path: str | None = None
    file_hash: str | None = None
    relevance_score: float = 1.0
    verification_status: str = "pending"  # pending, verified, disputed
    verified_by: str | None = None
    verification_date: datetime | None = None
    retention_period: timedelta = field(
        default_factory=lambda: timedelta(days=2555)
    )  # 7 years
    metadata: dict[str, Any] = field(default_factory=dict)

    def verify(self, verifier: str) -> None:
        """Verify audit evidence."""
        self.verification_status = "verified"
        self.verified_by = verifier
        self.verification_date = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if evidence has expired."""
        expiry_date = self.collection_date + self.retention_period
        return datetime.utcnow() > expiry_date

    def get_evidence_summary(self) -> dict[str, Any]:
        """Get evidence summary."""
        return {
            "evidence_id": str(self.evidence_id),
            "evidence_type": self.evidence_type,
            "title": self.title,
            "collected_by": self.collected_by,
            "collection_date": self.collection_date.isoformat(),
            "verification_status": self.verification_status,
            "relevance_score": self.relevance_score,
            "is_expired": self.is_expired(),
        }


@dataclass
class AuditFinding:
    """Audit finding with recommendations and remediation tracking."""

    finding_id: UUID = field(default_factory=uuid4)
    audit_id: UUID = field(default_factory=uuid4)
    finding_type: str = "observation"  # observation, deficiency, non_compliance, risk
    severity: str = "medium"  # low, medium, high, critical
    title: str = ""
    description: str = ""
    criteria: str = ""  # What was being tested against
    condition: str = ""  # What was actually found
    cause: str = ""  # Root cause analysis
    effect: str = ""  # Impact or potential impact
    recommendation: str = ""
    management_response: str | None = None

    # Risk assessment
    likelihood: str = "medium"  # low, medium, high
    impact: str = "medium"  # low, medium, high
    risk_rating: str = "medium"  # low, medium, high, critical

    # Remediation tracking
    remediation_plan: str | None = None
    target_completion_date: datetime | None = None
    actual_completion_date: datetime | None = None
    responsible_party: str | None = None
    status: str = "open"  # open, in_progress, resolved, closed, deferred

    # Evidence and references
    evidence: list[UUID] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    # Follow-up
    follow_up_required: bool = False
    follow_up_date: datetime | None = None

    def __post_init__(self):
        """Validate audit finding."""
        if not self.title:
            raise ValueError("Finding title cannot be empty")
        if not self.description:
            raise ValueError("Finding description cannot be empty")

    def calculate_risk_score(self) -> float:
        """Calculate quantitative risk score."""
        likelihood_scores = {"low": 1, "medium": 2, "high": 3}
        impact_scores = {"low": 1, "medium": 2, "high": 3}

        score = likelihood_scores.get(self.likelihood, 2) * impact_scores.get(
            self.impact, 2
        )

        # Update risk rating based on score
        if score <= 2:
            self.risk_rating = "low"
        elif score <= 4:
            self.risk_rating = "medium"
        elif score <= 6:
            self.risk_rating = "high"
        else:
            self.risk_rating = "critical"

        return score

    def is_overdue(self) -> bool:
        """Check if remediation is overdue."""
        if not self.target_completion_date:
            return False
        return datetime.utcnow() > self.target_completion_date and self.status not in [
            "resolved",
            "closed",
        ]

    def close_finding(self, closer: str, resolution_notes: str = "") -> None:
        """Close the audit finding."""
        self.status = "closed"
        self.actual_completion_date = datetime.utcnow()
        if resolution_notes:
            self.management_response = resolution_notes

    def get_finding_summary(self) -> dict[str, Any]:
        """Get finding summary."""
        return {
            "finding_id": str(self.finding_id),
            "finding_type": self.finding_type,
            "severity": self.severity,
            "title": self.title,
            "risk_rating": self.risk_rating,
            "risk_score": self.calculate_risk_score(),
            "status": self.status,
            "is_overdue": self.is_overdue(),
            "responsible_party": self.responsible_party,
            "target_completion": self.target_completion_date.isoformat()
            if self.target_completion_date
            else None,
            "evidence_count": len(self.evidence),
        }


@dataclass
class ComplianceAssessment:
    """Comprehensive compliance assessment."""

    assessment_id: UUID = field(default_factory=uuid4)
    assessment_name: str = ""
    framework_name: str = ""
    assessment_type: AuditType = AuditType.COMPLIANCE
    scope: str = ""
    assessor: str = ""
    assessment_date: datetime = field(default_factory=datetime.utcnow)
    completion_date: datetime | None = None

    # Assessment criteria
    controls_assessed: list[str] = field(default_factory=list)
    assessment_criteria: dict[str, Any] = field(default_factory=dict)
    sampling_methodology: str | None = None

    # Results
    overall_status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    compliance_score: float = 0.0
    findings: list[UUID] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Evidence and documentation
    evidence_collected: list[UUID] = field(default_factory=list)
    documentation_reviewed: list[str] = field(default_factory=list)
    interviews_conducted: list[str] = field(default_factory=list)

    # Follow-up
    next_assessment_date: datetime | None = None
    remediation_timeline: dict[str, datetime] = field(default_factory=dict)

    def __post_init__(self):
        """Validate compliance assessment."""
        if not self.assessment_name:
            raise ValueError("Assessment name cannot be empty")
        if not self.framework_name:
            raise ValueError("Framework name cannot be empty")

    def calculate_compliance_score(self, findings: list[AuditFinding]) -> float:
        """Calculate overall compliance score."""
        if not self.controls_assessed:
            return 0.0

        # Get findings for this assessment
        assessment_findings = [f for f in findings if f.audit_id == self.assessment_id]

        # Calculate score based on findings severity
        total_controls = len(self.controls_assessed)
        severity_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        total_deductions = 0
        for finding in assessment_findings:
            if finding.finding_type in ["deficiency", "non_compliance"]:
                weight = severity_weights.get(finding.severity, 2)
                total_deductions += weight

        # Calculate percentage score
        max_possible_deductions = total_controls * 4  # Assuming all could be critical
        if max_possible_deductions > 0:
            score = max(
                0,
                (max_possible_deductions - total_deductions)
                / max_possible_deductions
                * 100,
            )
        else:
            score = 100.0

        self.compliance_score = score

        # Update overall status
        if score >= 95:
            self.overall_status = ComplianceStatus.COMPLIANT
        elif score >= 80:
            self.overall_status = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            self.overall_status = ComplianceStatus.NON_COMPLIANT

        return score

    def complete_assessment(self) -> None:
        """Mark assessment as complete."""
        self.completion_date = datetime.utcnow()
        # Set next assessment date (typically annually)
        self.next_assessment_date = self.completion_date + timedelta(days=365)

    def get_assessment_summary(self) -> dict[str, Any]:
        """Get assessment summary."""
        return {
            "assessment_id": str(self.assessment_id),
            "assessment_name": self.assessment_name,
            "framework_name": self.framework_name,
            "assessment_type": self.assessment_type.value,
            "assessor": self.assessor,
            "assessment_date": self.assessment_date.isoformat(),
            "completion_date": self.completion_date.isoformat()
            if self.completion_date
            else None,
            "overall_status": self.overall_status.value,
            "compliance_score": self.compliance_score,
            "controls_assessed": len(self.controls_assessed),
            "findings_count": len(self.findings),
            "evidence_count": len(self.evidence_collected),
            "next_assessment": self.next_assessment_date.isoformat()
            if self.next_assessment_date
            else None,
        }


@dataclass
class DataLineage:
    """Data lineage tracking for governance."""

    lineage_id: UUID = field(default_factory=uuid4)
    data_asset_id: str = ""
    data_asset_name: str = ""
    data_classification: DataClassification = DataClassification.INTERNAL

    # Source information
    source_systems: list[str] = field(default_factory=list)
    source_datasets: list[str] = field(default_factory=list)
    extraction_method: str | None = None
    extraction_frequency: str | None = None

    # Transformation information
    transformation_steps: list[dict[str, Any]] = field(default_factory=list)
    business_rules: list[str] = field(default_factory=list)
    data_quality_rules: list[str] = field(default_factory=list)

    # Target information
    target_systems: list[str] = field(default_factory=list)
    target_datasets: list[str] = field(default_factory=list)
    usage_purposes: list[str] = field(default_factory=list)

    # Governance information
    data_owner: str | None = None
    data_steward: str | None = None
    retention_policy: str | None = None
    privacy_requirements: list[str] = field(default_factory=list)

    # Tracking information
    created_date: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    lineage_version: str = "1.0.0"

    def add_transformation(
        self,
        transformation_type: str,
        description: str,
        tool_used: str | None = None,
        business_rule: str | None = None,
    ) -> None:
        """Add transformation step to lineage."""
        transformation = {
            "step_id": str(uuid4()),
            "type": transformation_type,
            "description": description,
            "tool_used": tool_used,
            "business_rule": business_rule,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.transformation_steps.append(transformation)
        self.last_updated = datetime.utcnow()

    def get_lineage_path(self) -> list[str]:
        """Get simplified lineage path."""
        path = []
        path.extend(self.source_systems)
        path.append(f"Transformations({len(self.transformation_steps)})")
        path.extend(self.target_systems)
        return path

    def get_lineage_summary(self) -> dict[str, Any]:
        """Get lineage summary."""
        return {
            "lineage_id": str(self.lineage_id),
            "data_asset_name": self.data_asset_name,
            "data_classification": self.data_classification.value,
            "lineage_path": self.get_lineage_path(),
            "source_count": len(self.source_systems),
            "transformation_count": len(self.transformation_steps),
            "target_count": len(self.target_systems),
            "data_owner": self.data_owner,
            "data_steward": self.data_steward,
            "last_updated": self.last_updated.isoformat(),
            "lineage_version": self.lineage_version,
        }


@dataclass
class GovernanceWorkflow:
    """Governance workflow for approvals and processes."""

    workflow_id: UUID = field(default_factory=uuid4)
    workflow_name: str = ""
    workflow_type: str = ""  # approval, review, assessment, remediation
    description: str = ""

    # Workflow definition
    steps: list[dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    status: WorkflowStatus = WorkflowStatus.DRAFT

    # Participants
    initiator: str = ""
    assignees: list[str] = field(default_factory=list)
    approvers: list[str] = field(default_factory=list)
    reviewers: list[str] = field(default_factory=list)

    # Timing
    created_date: datetime = field(default_factory=datetime.utcnow)
    started_date: datetime | None = None
    due_date: datetime | None = None
    completed_date: datetime | None = None

    # Progress tracking
    completion_percentage: float = 0.0
    step_history: list[dict[str, Any]] = field(default_factory=list)
    comments: list[dict[str, Any]] = field(default_factory=list)

    # Escalation
    escalation_enabled: bool = True
    escalation_days: int = 7
    escalated: bool = False
    escalated_to: str | None = None
    escalation_date: datetime | None = None

    def start_workflow(self, starter: str) -> None:
        """Start the workflow."""
        self.status = WorkflowStatus.SUBMITTED
        self.started_date = datetime.utcnow()
        self.current_step = 0

        self.step_history.append(
            {
                "step": "workflow_started",
                "user": starter,
                "timestamp": datetime.utcnow().isoformat(),
                "action": "started",
            }
        )

    def advance_step(self, user: str, action: str, comments: str = "") -> bool:
        """Advance workflow to next step."""
        if self.current_step >= len(self.steps):
            return False

        # Record step completion
        self.step_history.append(
            {
                "step": self.current_step,
                "user": user,
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                "comments": comments,
            }
        )

        if comments:
            self.add_comment(user, comments)

        # Advance to next step
        self.current_step += 1
        self.completion_percentage = (self.current_step / len(self.steps)) * 100

        # Check if workflow is complete
        if self.current_step >= len(self.steps):
            self.complete_workflow()
            return True

        return False

    def complete_workflow(self) -> None:
        """Complete the workflow."""
        self.status = WorkflowStatus.IMPLEMENTED
        self.completed_date = datetime.utcnow()
        self.completion_percentage = 100.0

    def escalate_workflow(self, escalated_to: str, reason: str = "") -> None:
        """Escalate workflow to higher authority."""
        self.escalated = True
        self.escalated_to = escalated_to
        self.escalation_date = datetime.utcnow()
        self.status = WorkflowStatus.ESCALATED

        self.add_comment(
            "system", f"Workflow escalated to {escalated_to}. Reason: {reason}"
        )

    def add_comment(self, user: str, comment: str) -> None:
        """Add comment to workflow."""
        self.comments.append(
            {
                "user": user,
                "timestamp": datetime.utcnow().isoformat(),
                "comment": comment,
            }
        )

    def is_overdue(self) -> bool:
        """Check if workflow is overdue."""
        if not self.due_date or self.status in [
            WorkflowStatus.IMPLEMENTED,
            WorkflowStatus.CLOSED,
        ]:
            return False
        return datetime.utcnow() > self.due_date

    def should_escalate(self) -> bool:
        """Check if workflow should be escalated."""
        if not self.escalation_enabled or self.escalated:
            return False

        if not self.started_date:
            return False

        escalation_due = self.started_date + timedelta(days=self.escalation_days)
        return datetime.utcnow() > escalation_due and self.status in [
            WorkflowStatus.SUBMITTED,
            WorkflowStatus.UNDER_REVIEW,
        ]

    def get_workflow_summary(self) -> dict[str, Any]:
        """Get workflow summary."""
        return {
            "workflow_id": str(self.workflow_id),
            "workflow_name": self.workflow_name,
            "workflow_type": self.workflow_type,
            "status": self.status.value,
            "initiator": self.initiator,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "completion_percentage": self.completion_percentage,
            "created_date": self.created_date.isoformat(),
            "started_date": self.started_date.isoformat()
            if self.started_date
            else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "completed_date": self.completed_date.isoformat()
            if self.completed_date
            else None,
            "is_overdue": self.is_overdue(),
            "should_escalate": self.should_escalate(),
            "escalated": self.escalated,
            "comments_count": len(self.comments),
        }

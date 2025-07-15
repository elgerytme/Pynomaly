"""Quality Issue Domain Entities.

Contains entities for quality issues, remediation suggestions, and business impact.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid

from .quality_scores import MonetaryAmount


# Value Objects and Enums
@dataclass(frozen=True)
class IssueId:
    """Issue identifier value object."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SuggestionId:
    """Suggestion identifier value object."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


class QualityIssueType(Enum):
    """Types of quality issues."""
    MISSING_VALUES = "missing_values"
    DUPLICATE_RECORDS = "duplicate_records"
    INVALID_FORMAT = "invalid_format"
    OUT_OF_RANGE = "out_of_range"
    INCONSISTENT_DATA = "inconsistent_data"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    OUTLIER_DETECTION = "outlier_detection"
    PATTERN_VIOLATION = "pattern_violation"
    COMPLETENESS_ISSUE = "completeness_issue"
    ACCURACY_ISSUE = "accuracy_issue"
    TIMELINESS_ISSUE = "timeliness_issue"
    UNIQUENESS_VIOLATION = "uniqueness_violation"


class IssueStatus(Enum):
    """Status of quality issues."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    IGNORED = "ignored"
    DEFERRED = "deferred"


class ImpactLevel(Enum):
    """Business impact levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class ComplianceRisk(Enum):
    """Compliance risk levels."""
    SEVERE = "severe"
    MODERATE = "moderate"
    MINOR = "minor"
    NONE = "none"


class CustomerImpact(Enum):
    """Customer impact levels."""
    SERVICE_DISRUPTION = "service_disruption"
    DEGRADED_EXPERIENCE = "degraded_experience"
    MINOR_INCONVENIENCE = "minor_inconvenience"
    NO_IMPACT = "no_impact"


class OperationalImpact(Enum):
    """Operational impact levels."""
    SYSTEM_FAILURE = "system_failure"
    PROCESS_DISRUPTION = "process_disruption"
    EFFICIENCY_REDUCTION = "efficiency_reduction"
    MINIMAL_IMPACT = "minimal_impact"


class RemediationAction(Enum):
    """Types of remediation actions."""
    DATA_CLEANSING = "data_cleansing"
    RULE_MODIFICATION = "rule_modification"
    PROCESS_CHANGE = "process_change"
    SYSTEM_UPDATE = "system_update"
    TRAINING = "training"
    MONITORING = "monitoring"
    IGNORE = "ignore"
    ESCALATE = "escalate"


class EffortEstimate(Enum):
    """Effort estimation levels."""
    TRIVIAL = "trivial"        # < 1 hour
    MINOR = "minor"            # 1-4 hours
    MODERATE = "moderate"      # 1-3 days
    MAJOR = "major"            # 1-2 weeks
    EXTENSIVE = "extensive"    # > 2 weeks


class Priority(Enum):
    """Priority levels for remediation."""
    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKLOG = "backlog"


@dataclass(frozen=True)
class BusinessImpact:
    """Business impact assessment."""
    impact_level: ImpactLevel
    affected_processes: List[str]
    financial_impact: Optional[MonetaryAmount] = None
    compliance_risk: ComplianceRisk = ComplianceRisk.NONE
    customer_impact: CustomerImpact = CustomerImpact.NO_IMPACT
    operational_impact: OperationalImpact = OperationalImpact.MINIMAL_IMPACT
    
    # Additional impact details
    impact_description: str = ""
    affected_systems: List[str] = field(default_factory=list)
    affected_stakeholders: List[str] = field(default_factory=list)
    sla_violations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate business impact."""
        if not self.affected_processes:
            raise ValueError("At least one affected process must be specified")
        if self.financial_impact and self.financial_impact.to_float() < 0:
            raise ValueError("Financial impact cannot be negative")
    
    def get_impact_score(self) -> float:
        """Calculate overall impact score (0-1 scale)."""
        level_scores = {
            ImpactLevel.CRITICAL: 1.0,
            ImpactLevel.HIGH: 0.8,
            ImpactLevel.MEDIUM: 0.6,
            ImpactLevel.LOW: 0.4,
            ImpactLevel.MINIMAL: 0.2
        }
        
        compliance_scores = {
            ComplianceRisk.SEVERE: 0.3,
            ComplianceRisk.MODERATE: 0.2,
            ComplianceRisk.MINOR: 0.1,
            ComplianceRisk.NONE: 0.0
        }
        
        customer_scores = {
            CustomerImpact.SERVICE_DISRUPTION: 0.3,
            CustomerImpact.DEGRADED_EXPERIENCE: 0.2,
            CustomerImpact.MINOR_INCONVENIENCE: 0.1,
            CustomerImpact.NO_IMPACT: 0.0
        }
        
        base_score = level_scores.get(self.impact_level, 0.0)
        compliance_score = compliance_scores.get(self.compliance_risk, 0.0)
        customer_score = customer_scores.get(self.customer_impact, 0.0)
        
        return min(1.0, base_score + compliance_score + customer_score)
    
    def is_high_impact(self) -> bool:
        """Check if this is a high impact issue."""
        return self.impact_level in [ImpactLevel.CRITICAL, ImpactLevel.HIGH]
    
    def requires_immediate_attention(self) -> bool:
        """Check if issue requires immediate attention."""
        return (self.impact_level == ImpactLevel.CRITICAL or
                self.compliance_risk == ComplianceRisk.SEVERE or
                self.customer_impact == CustomerImpact.SERVICE_DISRUPTION)
    
    def get_impact_summary(self) -> Dict[str, Any]:
        """Get summary of business impact."""
        return {
            'impact_level': self.impact_level.value,
            'impact_score': self.get_impact_score(),
            'financial_impact': str(self.financial_impact) if self.financial_impact else None,
            'compliance_risk': self.compliance_risk.value,
            'customer_impact': self.customer_impact.value,
            'operational_impact': self.operational_impact.value,
            'affected_processes': self.affected_processes,
            'affected_systems': self.affected_systems,
            'sla_violations': self.sla_violations,
            'requires_immediate_attention': self.requires_immediate_attention(),
            'is_high_impact': self.is_high_impact()
        }


@dataclass(frozen=True)
class RemediationSuggestion:
    """Remediation suggestion for quality issues."""
    
    suggestion_id: SuggestionId
    issue_id: IssueId
    action_type: RemediationAction
    description: str
    implementation_steps: List[str]
    effort_estimate: EffortEstimate
    success_probability: float
    side_effects: List[str]
    priority: Priority
    
    # Additional suggestion details
    required_skills: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_cost: Optional[MonetaryAmount] = None
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate remediation suggestion."""
        if not self.description.strip():
            raise ValueError("Description cannot be empty")
        if not self.implementation_steps:
            raise ValueError("Implementation steps cannot be empty")
        if not (0.0 <= self.success_probability <= 1.0):
            raise ValueError("Success probability must be between 0 and 1")
        if self.approved_at and self.approved_at < self.created_at:
            raise ValueError("Approval time cannot be before creation time")
    
    def get_effort_hours(self) -> float:
        """Get estimated effort in hours."""
        effort_hours = {
            EffortEstimate.TRIVIAL: 0.5,
            EffortEstimate.MINOR: 2.5,
            EffortEstimate.MODERATE: 16.0,
            EffortEstimate.MAJOR: 60.0,
            EffortEstimate.EXTENSIVE: 120.0
        }
        return effort_hours.get(self.effort_estimate, 0.0)
    
    def get_priority_score(self) -> float:
        """Get priority score (0-1 scale)."""
        priority_scores = {
            Priority.IMMEDIATE: 1.0,
            Priority.HIGH: 0.8,
            Priority.MEDIUM: 0.6,
            Priority.LOW: 0.4,
            Priority.BACKLOG: 0.2
        }
        return priority_scores.get(self.priority, 0.0)
    
    def is_approved(self) -> bool:
        """Check if suggestion is approved."""
        return self.approved_by is not None and self.approved_at is not None
    
    def is_high_priority(self) -> bool:
        """Check if this is high priority."""
        return self.priority in [Priority.IMMEDIATE, Priority.HIGH]
    
    def get_suggestion_summary(self) -> Dict[str, Any]:
        """Get summary of remediation suggestion."""
        return {
            'suggestion_id': str(self.suggestion_id),
            'issue_id': str(self.issue_id),
            'action_type': self.action_type.value,
            'description': self.description,
            'effort_estimate': self.effort_estimate.value,
            'effort_hours': self.get_effort_hours(),
            'success_probability': self.success_probability,
            'priority': self.priority.value,
            'priority_score': self.get_priority_score(),
            'implementation_steps': len(self.implementation_steps),
            'side_effects': len(self.side_effects),
            'required_skills': self.required_skills,
            'required_tools': self.required_tools,
            'dependencies': len(self.dependencies),
            'estimated_cost': str(self.estimated_cost) if self.estimated_cost else None,
            'is_approved': self.is_approved(),
            'is_high_priority': self.is_high_priority(),
            'created_at': self.created_at.isoformat()
        }


# Main Domain Entity
@dataclass(frozen=True)
class QualityIssue:
    """Quality issue entity."""
    
    issue_id: IssueId
    issue_type: QualityIssueType
    severity: 'Severity'
    description: str
    affected_records: int
    affected_columns: List[str]
    root_cause: Optional[str]
    business_impact: BusinessImpact
    remediation_effort: EffortEstimate
    status: IssueStatus
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    
    # Additional issue details
    dataset_id: Optional['DatasetId'] = None
    rule_id: Optional['RuleId'] = None
    error_samples: List[str] = field(default_factory=list)
    
    # Tracking and assignment
    assigned_to: Optional[str] = None
    assigned_at: Optional[datetime] = None
    reporter: Optional[str] = None
    
    # Resolution details
    resolution_notes: Optional[str] = None
    resolution_method: Optional[RemediationAction] = None
    verification_status: Optional[str] = None
    
    def __post_init__(self):
        """Validate quality issue."""
        if not self.description.strip():
            raise ValueError("Description cannot be empty")
        if self.affected_records < 0:
            raise ValueError("Affected records cannot be negative")
        if not self.affected_columns:
            raise ValueError("At least one affected column must be specified")
        if self.resolved_at and self.resolved_at < self.detected_at:
            raise ValueError("Resolution time cannot be before detection time")
        if self.assigned_at and self.assigned_at < self.detected_at:
            raise ValueError("Assignment time cannot be before detection time")
    
    def is_resolved(self) -> bool:
        """Check if issue is resolved."""
        return self.status == IssueStatus.RESOLVED
    
    def is_critical(self) -> bool:
        """Check if issue is critical."""
        return self.severity.value == 'critical'
    
    def is_high_severity(self) -> bool:
        """Check if issue is high severity."""
        return self.severity.value in ['critical', 'high']
    
    def is_assigned(self) -> bool:
        """Check if issue is assigned."""
        return self.assigned_to is not None
    
    def get_age_hours(self) -> float:
        """Get age of issue in hours."""
        end_time = self.resolved_at or datetime.now()
        return (end_time - self.detected_at).total_seconds() / 3600
    
    def get_resolution_time_hours(self) -> Optional[float]:
        """Get resolution time in hours."""
        if self.resolved_at:
            return (self.resolved_at - self.detected_at).total_seconds() / 3600
        return None
    
    def get_impact_score(self) -> float:
        """Get overall impact score combining severity and business impact."""
        severity_scores = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'info': 0.2
        }
        
        severity_score = severity_scores.get(self.severity.value, 0.0)
        business_score = self.business_impact.get_impact_score()
        
        # Weighted combination
        return (severity_score * 0.6) + (business_score * 0.4)
    
    def get_issue_summary(self) -> Dict[str, Any]:
        """Get comprehensive issue summary."""
        return {
            'issue_id': str(self.issue_id),
            'issue_type': self.issue_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'status': self.status.value,
            'affected_records': self.affected_records,
            'affected_columns': self.affected_columns,
            'root_cause': self.root_cause,
            'remediation_effort': self.remediation_effort.value,
            'detected_at': self.detected_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'age_hours': self.get_age_hours(),
            'resolution_time_hours': self.get_resolution_time_hours(),
            'impact_score': self.get_impact_score(),
            'business_impact': self.business_impact.get_impact_summary(),
            'is_resolved': self.is_resolved(),
            'is_critical': self.is_critical(),
            'is_assigned': self.is_assigned(),
            'assigned_to': self.assigned_to,
            'dataset_id': str(self.dataset_id) if self.dataset_id else None,
            'rule_id': str(self.rule_id) if self.rule_id else None,
            'error_samples': len(self.error_samples),
            'resolution_method': self.resolution_method.value if self.resolution_method else None
        }
    
    def assign_to(self, assignee: str) -> 'QualityIssue':
        """Assign issue to someone."""
        return QualityIssue(
            issue_id=self.issue_id,
            issue_type=self.issue_type,
            severity=self.severity,
            description=self.description,
            affected_records=self.affected_records,
            affected_columns=self.affected_columns,
            root_cause=self.root_cause,
            business_impact=self.business_impact,
            remediation_effort=self.remediation_effort,
            status=IssueStatus.IN_PROGRESS,
            detected_at=self.detected_at,
            resolved_at=self.resolved_at,
            dataset_id=self.dataset_id,
            rule_id=self.rule_id,
            error_samples=self.error_samples,
            assigned_to=assignee,
            assigned_at=datetime.now(),
            reporter=self.reporter,
            resolution_notes=self.resolution_notes,
            resolution_method=self.resolution_method,
            verification_status=self.verification_status
        )
    
    def resolve(self, resolution_notes: str, resolution_method: RemediationAction) -> 'QualityIssue':
        """Resolve the issue."""
        return QualityIssue(
            issue_id=self.issue_id,
            issue_type=self.issue_type,
            severity=self.severity,
            description=self.description,
            affected_records=self.affected_records,
            affected_columns=self.affected_columns,
            root_cause=self.root_cause,
            business_impact=self.business_impact,
            remediation_effort=self.remediation_effort,
            status=IssueStatus.RESOLVED,
            detected_at=self.detected_at,
            resolved_at=datetime.now(),
            dataset_id=self.dataset_id,
            rule_id=self.rule_id,
            error_samples=self.error_samples,
            assigned_to=self.assigned_to,
            assigned_at=self.assigned_at,
            reporter=self.reporter,
            resolution_notes=resolution_notes,
            resolution_method=resolution_method,
            verification_status=self.verification_status
        )


# Import statements for forward references
from .validation_rule import Severity, RuleId
from .quality_profile import DatasetId
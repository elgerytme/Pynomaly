"""Data Quality Profile entity for comprehensive quality assessment results."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .quality_rule import ValidationResult, DatasetId, UserId
from ..value_objects.quality_scores import QualityScores
from ..value_objects.business_impact import BusinessImpact
from ..value_objects.remediation_suggestion import RemediationSuggestion


@dataclass(frozen=True)
class ProfileId:
    """Data Quality Profile identifier."""
    value: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class ProfileVersion:
    """Profile version identifier."""
    major: int = 1
    minor: int = 0
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"
    
    def increment_minor(self) -> 'ProfileVersion':
        """Create new version with incremented minor version."""
        return ProfileVersion(major=self.major, minor=self.minor + 1)
    
    def increment_major(self) -> 'ProfileVersion':
        """Create new version with incremented major version."""
        return ProfileVersion(major=self.major + 1, minor=0)


class QualityTrends(BaseModel):
    """Quality trends over time."""
    
    current_period_score: float
    previous_period_score: Optional[float] = None
    trend_direction: str = "stable"  # "improving", "declining", "stable"
    trend_magnitude: float = 0.0
    periods_analyzed: int = 1
    
    class Config:
        frozen = True
    
    def calculate_trend(self) -> None:
        """Calculate trend direction and magnitude."""
        if self.previous_period_score is None:
            return
        
        difference = self.current_period_score - self.previous_period_score
        self.trend_magnitude = abs(difference)
        
        if difference > 0.02:  # More than 2% improvement
            self.trend_direction = "improving"
        elif difference < -0.02:  # More than 2% decline
            self.trend_direction = "declining"
        else:
            self.trend_direction = "stable"


class QualityIssueType(str, Enum):
    """Quality issue type enumeration."""
    COMPLETENESS_VIOLATION = "completeness_violation"
    ACCURACY_VIOLATION = "accuracy_violation"
    CONSISTENCY_VIOLATION = "consistency_violation"
    VALIDITY_VIOLATION = "validity_violation"
    UNIQUENESS_VIOLATION = "uniqueness_violation"
    TIMELINESS_VIOLATION = "timeliness_violation"
    REFERENTIAL_INTEGRITY_VIOLATION = "referential_integrity_violation"
    FORMAT_VIOLATION = "format_violation"
    BUSINESS_RULE_VIOLATION = "business_rule_violation"
    STATISTICAL_ANOMALY = "statistical_anomaly"


class IssueStatus(str, Enum):
    """Quality issue status enumeration."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"
    SUPPRESSED = "suppressed"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class QualityIssueId:
    """Quality issue identifier."""
    value: UUID = field(default_factory=uuid4)


class QualityIssue(BaseModel):
    """Quality issue entity for tracking data quality problems."""
    
    issue_id: QualityIssueId = Field(default_factory=QualityIssueId)
    issue_type: QualityIssueType
    severity: str  # From quality_rule.Severity
    description: str
    affected_records: int
    affected_columns: List[str] = Field(default_factory=list)
    root_cause: Optional[str] = None
    business_impact: Optional[BusinessImpact] = None
    remediation_effort: Optional[str] = None  # EffortEstimate reference
    status: IssueStatus = IssueStatus.OPEN
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    assigned_to: Optional[UserId] = None
    
    # Associated validation rule and result
    source_rule_id: Optional[str] = None
    source_validation_id: Optional[UUID] = None
    
    # Resolution tracking
    resolution_notes: Optional[str] = None
    resolution_actions: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True
    
    def resolve(self, resolution_notes: str, resolved_by: UserId) -> None:
        """Mark issue as resolved."""
        if self.status == IssueStatus.RESOLVED:
            raise ValueError("Issue is already resolved")
        
        self.status = IssueStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.resolution_notes = resolution_notes
        self.assigned_to = resolved_by
    
    def close(self, closing_notes: Optional[str] = None) -> None:
        """Close the issue."""
        if self.status != IssueStatus.RESOLVED:
            raise ValueError("Issue must be resolved before closing")
        
        self.status = IssueStatus.CLOSED
        if closing_notes:
            self.resolution_notes = f"{self.resolution_notes}\nClosed: {closing_notes}"
    
    def suppress(self, reason: str) -> None:
        """Suppress the issue (false positive or acceptable)."""
        self.status = IssueStatus.SUPPRESSED
        self.resolution_notes = f"Suppressed: {reason}"
    
    def assign_to(self, user_id: UserId) -> None:
        """Assign issue to a user."""
        self.assigned_to = user_id
        if self.status == IssueStatus.OPEN:
            self.status = IssueStatus.IN_PROGRESS
    
    def add_resolution_action(self, action: str) -> None:
        """Add a resolution action to the issue."""
        self.resolution_actions.append(action)
    
    def get_age_in_days(self) -> int:
        """Get issue age in days."""
        return (datetime.utcnow() - self.detected_at).days
    
    def is_overdue(self, sla_days: int = 30) -> bool:
        """Check if issue is overdue based on SLA."""
        return self.get_age_in_days() > sla_days and self.status not in [IssueStatus.RESOLVED, IssueStatus.CLOSED]


class DataQualityProfile(BaseModel):
    """Comprehensive data quality profile aggregate root."""
    
    profile_id: ProfileId = Field(default_factory=ProfileId)
    dataset_id: DatasetId
    quality_scores: QualityScores
    validation_results: List[ValidationResult] = Field(default_factory=list)
    quality_issues: List[QualityIssue] = Field(default_factory=list)
    remediation_suggestions: List[RemediationSuggestion] = Field(default_factory=list)
    quality_trends: Optional[QualityTrends] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_assessed: datetime = Field(default_factory=datetime.utcnow)
    version: ProfileVersion = Field(default_factory=ProfileVersion)
    
    # Assessment metadata
    assessment_duration_seconds: Optional[float] = None
    rules_executed: int = 0
    total_records_assessed: int = 0
    
    # Ownership
    created_by: UserId
    assessment_engine_version: Optional[str] = None
    
    class Config:
        use_enum_values = True
    
    def add_validation_result(self, result: ValidationResult) -> None:
        """Add a validation result to the profile."""
        self.validation_results.append(result)
        self.rules_executed += 1
        self.last_assessed = datetime.utcnow()
    
    def add_quality_issue(self, issue: QualityIssue) -> None:
        """Add a quality issue to the profile."""
        self.quality_issues.append(issue)
    
    def add_remediation_suggestion(self, suggestion: RemediationSuggestion) -> None:
        """Add a remediation suggestion to the profile."""
        self.remediation_suggestions.append(suggestion)
    
    def get_open_issues(self) -> List[QualityIssue]:
        """Get all open quality issues."""
        return [issue for issue in self.quality_issues if issue.status == IssueStatus.OPEN]
    
    def get_critical_issues(self) -> List[QualityIssue]:
        """Get all critical quality issues."""
        return [issue for issue in self.quality_issues if issue.severity == "critical"]
    
    def get_failed_validations(self) -> List[ValidationResult]:
        """Get all failed validation results."""
        return [result for result in self.validation_results if result.status == "failed"]
    
    def get_overall_pass_rate(self) -> float:
        """Calculate overall pass rate across all validations."""
        if not self.validation_results:
            return 1.0
        
        total_records = sum(result.total_records for result in self.validation_results)
        passed_records = sum(result.records_passed for result in self.validation_results)
        
        return passed_records / total_records if total_records > 0 else 1.0
    
    def is_quality_acceptable(self, threshold: float = 0.80) -> bool:
        """Check if overall quality meets the threshold."""
        return self.quality_scores.overall_score >= threshold
    
    def requires_attention(self) -> bool:
        """Check if profile requires immediate attention."""
        conditions = [
            len(self.get_critical_issues()) > 0,
            self.quality_scores.overall_score < 0.70,
            len(self.get_open_issues()) > 10
        ]
        return any(conditions)
    
    def get_improvement_priority_areas(self) -> List[str]:
        """Get quality dimensions that need improvement."""
        failing_dimensions = self.quality_scores.get_failing_dimensions(threshold=0.85)
        return list(failing_dimensions.keys())
    
    def update_quality_scores(self, new_scores: QualityScores) -> None:
        """Update quality scores and increment version."""
        # Create trends if we have previous scores
        if self.quality_trends is None:
            self.quality_trends = QualityTrends(
                current_period_score=new_scores.overall_score,
                previous_period_score=self.quality_scores.overall_score
            )
        else:
            self.quality_trends.previous_period_score = self.quality_trends.current_period_score
            self.quality_trends.current_period_score = new_scores.overall_score
        
        self.quality_trends.calculate_trend()
        self.quality_scores = new_scores
        self.version = self.version.increment_minor()
        self.last_assessed = datetime.utcnow()
    
    def get_assessment_summary(self) -> dict:
        """Get summary of the quality assessment."""
        return {
            "profile_id": str(self.profile_id.value),
            "dataset_id": str(self.dataset_id.value),
            "overall_score": self.quality_scores.overall_score,
            "quality_grade": self.quality_scores.get_quality_grade(),
            "total_issues": len(self.quality_issues),
            "critical_issues": len(self.get_critical_issues()),
            "open_issues": len(self.get_open_issues()),
            "rules_executed": self.rules_executed,
            "records_assessed": self.total_records_assessed,
            "assessment_date": self.last_assessed.isoformat(),
            "requires_attention": self.requires_attention(),
            "improvement_areas": self.get_improvement_priority_areas()
        }
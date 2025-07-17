"""Quality Profile Domain Entity.

Contains the core DataQualityProfile entity and related value objects
for managing comprehensive data quality assessments.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import uuid

from .quality_scores import QualityScores, QualityTrends


# Value Objects
@dataclass(frozen=True)
class ProfileId:
    """Profile identifier value object."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DatasetId:
    """Dataset identifier value object."""
    value: str
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class JobId:
    """Job identifier value object."""
    value: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ProfileVersion:
    """Profile version value object."""
    major: int = 1
    minor: int = 0
    patch: int = 0
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def increment_patch(self) -> 'ProfileVersion':
        return ProfileVersion(self.major, self.minor, self.patch + 1)
    
    def increment_minor(self) -> 'ProfileVersion':
        return ProfileVersion(self.major, self.minor + 1, 0)
    
    def increment_major(self) -> 'ProfileVersion':
        return ProfileVersion(self.major + 1, 0, 0)


class QualityJobType(Enum):
    """Types of quality jobs."""
    VALIDATION = "validation"
    PROFILING = "profiling"
    CLEANSING = "cleansing"
    MONITORING = "monitoring"
    ASSESSMENT = "assessment"
    COMPARISON = "comparison"


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class QualityJobConfig:
    """Configuration for quality job execution."""
    job_type: QualityJobType
    enable_sampling: bool = True
    sample_size: int = 10000
    enable_parallel_processing: bool = True
    max_workers: int = 4
    timeout_seconds: int = 3600
    enable_caching: bool = True
    cache_ttl_seconds: int = 1800
    enable_detailed_logging: bool = False
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def with_custom_parameter(self, key: str, value: Any) -> 'QualityJobConfig':
        """Return new config with additional custom parameter."""
        new_params = self.custom_parameters.copy()
        new_params[key] = value
        return QualityJobConfig(
            job_type=self.job_type,
            enable_sampling=self.enable_sampling,
            sample_size=self.sample_size,
            enable_parallel_processing=self.enable_parallel_processing,
            max_workers=self.max_workers,
            timeout_seconds=self.timeout_seconds,
            enable_caching=self.enable_caching,
            cache_ttl_seconds=self.cache_ttl_seconds,
            enable_detailed_logging=self.enable_detailed_logging,
            custom_parameters=new_params
        )


@dataclass(frozen=True)
class JobMetrics:
    """Performance metrics for quality job execution."""
    execution_time_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    records_processed: int
    rules_executed: int
    validations_performed: int
    cache_hit_rate: float = 0.0
    parallel_efficiency: float = 0.0
    throughput_records_per_second: float = 0.0
    
    def __post_init__(self):
        # Calculate derived metrics
        if self.execution_time_seconds > 0:
            object.__setattr__(self, 'throughput_records_per_second', 
                             self.records_processed / self.execution_time_seconds)


@dataclass(frozen=True)
class DataSource:
    """Data source configuration."""
    source_type: str
    source_uri: str
    connection_params: Dict[str, Any] = field(default_factory=dict)
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None
    
    def __str__(self) -> str:
        if self.table_name:
            return f"{self.source_type}://{self.source_uri}/{self.table_name}"
        return f"{self.source_type}://{self.source_uri}"


# Main Domain Entities
@dataclass(frozen=True)
class DataQualityProfile:
    """Comprehensive data quality profile for a dataset."""
    
    profile_id: ProfileId
    dataset_id: DatasetId
    quality_scores: QualityScores
    validation_results: List['ValidationResult']
    quality_issues: List['QualityIssue']
    remediation_suggestions: List['RemediationSuggestion']
    quality_trends: QualityTrends
    created_at: datetime
    last_assessed: datetime
    version: ProfileVersion
    
    # Optional metadata
    dataset_name: Optional[str] = None
    dataset_description: Optional[str] = None
    data_source: Optional[DataSource] = None
    record_count: Optional[int] = None
    column_count: Optional[int] = None
    data_size_bytes: Optional[int] = None
    
    # Configuration used for assessment
    assessment_config: Optional[QualityJobConfig] = None
    assessment_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate profile consistency."""
        if self.last_assessed < self.created_at:
            raise ValueError("Last assessed time cannot be before creation time")
        
        if self.quality_scores.overall_score < 0 or self.quality_scores.overall_score > 1:
            raise ValueError("Overall quality score must be between 0 and 1")
    
    def get_critical_issues(self) -> List['QualityIssue']:
        """Get all critical quality issues."""
        return [issue for issue in self.quality_issues 
                if issue.severity.value == 'critical']
    
    def get_high_priority_issues(self) -> List['QualityIssue']:
        """Get all high priority issues."""
        return [issue for issue in self.quality_issues 
                if issue.severity.value in ['critical', 'high']]
    
    def get_issues_by_type(self, issue_type: 'QualityIssueType') -> List['QualityIssue']:
        """Get issues by specific type."""
        return [issue for issue in self.quality_issues 
                if issue.issue_type == issue_type]
    
    def get_failed_validations(self) -> List['ValidationResult']:
        """Get all failed validation results."""
        return [result for result in self.validation_results 
                if result.status.value == 'failed']
    
    def get_validation_success_rate(self) -> float:
        """Calculate overall validation success rate."""
        if not self.validation_results:
            return 0.0
        
        successful = sum(1 for result in self.validation_results 
                        if result.status.value == 'passed')
        return successful / len(self.validation_results)
    
    def has_quality_regression(self, previous_profile: 'DataQualityProfile') -> bool:
        """Check if quality has regressed compared to previous profile."""
        return self.quality_scores.overall_score < previous_profile.quality_scores.overall_score
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of quality metrics."""
        return {
            'profile_id': str(self.profile_id),
            'dataset_id': str(self.dataset_id),
            'overall_score': self.quality_scores.overall_score,
            'total_issues': len(self.quality_issues),
            'critical_issues': len(self.get_critical_issues()),
            'validation_success_rate': self.get_validation_success_rate(),
            'record_count': self.record_count,
            'column_count': self.column_count,
            'last_assessed': self.last_assessed.isoformat(),
            'version': str(self.version)
        }
    
    def update_version(self, increment_type: str = 'patch') -> 'DataQualityProfile':
        """Create new profile with incremented version."""
        if increment_type == 'major':
            new_version = self.version.increment_major()
        elif increment_type == 'minor':
            new_version = self.version.increment_minor()
        else:
            new_version = self.version.increment_patch()
        
        return DataQualityProfile(
            profile_id=self.profile_id,
            dataset_id=self.dataset_id,
            quality_scores=self.quality_scores,
            validation_results=self.validation_results,
            quality_issues=self.quality_issues,
            remediation_suggestions=self.remediation_suggestions,
            quality_trends=self.quality_trends,
            created_at=self.created_at,
            last_assessed=datetime.now(),
            version=new_version,
            dataset_name=self.dataset_name,
            dataset_description=self.dataset_description,
            data_source=self.data_source,
            record_count=self.record_count,
            column_count=self.column_count,
            data_size_bytes=self.data_size_bytes,
            assessment_config=self.assessment_config,
            assessment_metadata=self.assessment_metadata
        )


@dataclass(frozen=True)
class QualityJob:
    """Quality job entity for tracking job execution."""
    
    job_id: JobId
    job_type: QualityJobType
    dataset_source: DataSource
    rules_applied: List['RuleId']
    job_config: QualityJobConfig
    status: JobStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[DataQualityProfile] = None
    metrics: Optional[JobMetrics] = None
    
    # Error handling
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Progress tracking
    progress_percentage: float = 0.0
    current_step: str = "initialized"
    total_steps: int = 0
    completed_steps: int = 0
    
    def __post_init__(self):
        """Validate job state consistency."""
        if self.completed_at and self.completed_at < self.started_at:
            raise ValueError("Completion time cannot be before start time")
        
        if self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            if not self.completed_at:
                object.__setattr__(self, 'completed_at', datetime.now())
    
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == JobStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if job has completed successfully."""
        return self.status == JobStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if job has failed."""
        return self.status == JobStatus.FAILED
    
    def get_execution_duration(self) -> Optional[timedelta]:
        """Get job execution duration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        elif self.status == JobStatus.RUNNING:
            return datetime.now() - self.started_at
        return None
    
    def get_job_summary(self) -> Dict[str, Any]:
        """Get summary of job execution."""
        duration = self.get_execution_duration()
        return {
            'job_id': str(self.job_id),
            'job_type': self.job_type.value,
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': duration.total_seconds() if duration else None,
            'progress_percentage': self.progress_percentage,
            'current_step': self.current_step,
            'rules_applied': len(self.rules_applied),
            'has_error': bool(self.error_message),
            'has_results': bool(self.results)
        }
    
    def with_progress_update(self, progress: float, step: str) -> 'QualityJob':
        """Create new job with updated progress."""
        return QualityJob(
            job_id=self.job_id,
            job_type=self.job_type,
            dataset_source=self.dataset_source,
            rules_applied=self.rules_applied,
            job_config=self.job_config,
            status=self.status,
            started_at=self.started_at,
            completed_at=self.completed_at,
            results=self.results,
            metrics=self.metrics,
            error_message=self.error_message,
            error_details=self.error_details,
            progress_percentage=progress,
            current_step=step,
            total_steps=self.total_steps,
            completed_steps=self.completed_steps
        )
    
    def with_completion(self, results: DataQualityProfile, metrics: JobMetrics) -> 'QualityJob':
        """Create new job with completion status."""
        return QualityJob(
            job_id=self.job_id,
            job_type=self.job_type,
            dataset_source=self.dataset_source,
            rules_applied=self.rules_applied,
            job_config=self.job_config,
            status=JobStatus.COMPLETED,
            started_at=self.started_at,
            completed_at=datetime.now(),
            results=results,
            metrics=metrics,
            error_message=self.error_message,
            error_details=self.error_details,
            progress_percentage=100.0,
            current_step="completed",
            total_steps=self.total_steps,
            completed_steps=self.total_steps
        )
    
    def with_failure(self, error_message: str, error_details: Dict[str, Any] = None) -> 'QualityJob':
        """Create new job with failure status."""
        return QualityJob(
            job_id=self.job_id,
            job_type=self.job_type,
            dataset_source=self.dataset_source,
            rules_applied=self.rules_applied,
            job_config=self.job_config,
            status=JobStatus.FAILED,
            started_at=self.started_at,
            completed_at=datetime.now(),
            results=self.results,
            metrics=self.metrics,
            error_message=error_message,
            error_details=error_details or {},
            progress_percentage=self.progress_percentage,
            current_step="failed",
            total_steps=self.total_steps,
            completed_steps=self.completed_steps
        )
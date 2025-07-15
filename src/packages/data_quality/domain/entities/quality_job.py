"""Quality Job entity for tracking data quality assessment executions."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .quality_rule import RuleId, DatasetId, UserId
from .data_quality_profile import DataQualityProfile


@dataclass(frozen=True)
class JobId:
    """Quality Job identifier."""
    value: UUID = field(default_factory=uuid4)


class QualityJobType(str, Enum):
    """Quality job type enumeration."""
    FULL_ASSESSMENT = "full_assessment"
    INCREMENTAL_ASSESSMENT = "incremental_assessment"
    RULE_VALIDATION = "rule_validation"
    MONITORING_CHECK = "monitoring_check"
    BATCH_VALIDATION = "batch_validation"
    REAL_TIME_VALIDATION = "real_time_validation"
    SCHEDULED_ASSESSMENT = "scheduled_assessment"
    ON_DEMAND_ASSESSMENT = "on_demand_assessment"


class JobStatus(str, Enum):
    """Quality job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    PAUSED = "paused"


class DataSource(BaseModel):
    """Data source configuration for quality job."""
    
    source_type: str  # "database", "file", "stream", "api"
    connection_string: Optional[str] = None
    file_path: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None
    schema_name: Optional[str] = None
    
    # For streaming sources
    topic_name: Optional[str] = None
    partition: Optional[int] = None
    
    # Authentication
    credentials: Optional[Dict[str, Any]] = None
    
    class Config:
        frozen = True


class QualityJobConfig(BaseModel):
    """Configuration for quality job execution."""
    
    # Execution settings
    max_execution_time_minutes: int = 60
    memory_limit_mb: Optional[int] = None
    cpu_limit_cores: Optional[float] = None
    parallel_execution: bool = True
    max_parallel_rules: int = 10
    
    # Sampling settings
    enable_sampling: bool = False
    sample_size: Optional[int] = None
    sample_percentage: Optional[float] = None
    random_seed: Optional[int] = None
    
    # Output settings
    store_detailed_results: bool = True
    generate_report: bool = True
    send_notifications: bool = True
    notification_recipients: List[str] = Field(default_factory=list)
    
    # Quality thresholds for the job
    fail_on_critical_issues: bool = True
    fail_on_score_below: Optional[float] = None
    
    # Retry settings
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay_seconds: int = 60
    
    class Config:
        frozen = True


class JobMetrics(BaseModel):
    """Metrics collected during job execution."""
    
    # Performance metrics
    execution_time_seconds: float = 0.0
    memory_peak_mb: Optional[float] = None
    cpu_utilization_avg: Optional[float] = None
    
    # Data processing metrics
    records_processed: int = 0
    records_per_second: float = 0.0
    bytes_processed: Optional[int] = None
    
    # Rule execution metrics
    rules_executed: int = 0
    rules_passed: int = 0
    rules_failed: int = 0
    rules_with_errors: int = 0
    
    # Quality metrics
    overall_pass_rate: float = 0.0
    critical_issues_found: int = 0
    total_issues_found: int = 0
    
    # Infrastructure metrics
    nodes_used: Optional[int] = None
    worker_failures: int = 0
    retries_attempted: int = 0
    
    class Config:
        frozen = True
        
    def calculate_derived_metrics(self) -> None:
        """Calculate derived metrics."""
        if self.execution_time_seconds > 0 and self.records_processed > 0:
            self.records_per_second = self.records_processed / self.execution_time_seconds
        
        if self.rules_executed > 0:
            self.overall_pass_rate = self.rules_passed / self.rules_executed


class QualityJob(BaseModel):
    """Quality job entity for orchestrating data quality assessments."""
    
    job_id: JobId = Field(default_factory=JobId)
    job_type: QualityJobType
    dataset_source: DataSource
    rules_applied: List[RuleId] = Field(default_factory=list)
    job_config: QualityJobConfig = Field(default_factory=QualityJobConfig)
    status: JobStatus = JobStatus.PENDING
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    results: Optional[DataQualityProfile] = None
    metrics: JobMetrics = Field(default_factory=JobMetrics)
    
    # Error tracking
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Ownership and scheduling
    created_by: UserId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    parent_job_id: Optional[JobId] = None
    
    # Job context
    job_name: Optional[str] = None
    job_description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True
    
    def start(self) -> None:
        """Start the quality job execution."""
        if self.status != JobStatus.PENDING:
            raise ValueError(f"Cannot start job with status {self.status}")
        
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete(self, results: DataQualityProfile, metrics: JobMetrics) -> None:
        """Mark job as completed with results."""
        if self.status != JobStatus.RUNNING:
            raise ValueError(f"Cannot complete job with status {self.status}")
        
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.results = results
        self.metrics = metrics
        
        # Calculate execution time
        if self.started_at:
            duration = (self.completed_at - self.started_at).total_seconds()
            self.metrics.execution_time_seconds = duration
        
        self.metrics.calculate_derived_metrics()
    
    def fail(self, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.error_details = error_details or {}
    
    def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel the job execution."""
        if self.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise ValueError(f"Cannot cancel job with status {self.status}")
        
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        if reason:
            self.error_message = f"Cancelled: {reason}"
    
    def pause(self) -> None:
        """Pause the job execution."""
        if self.status != JobStatus.RUNNING:
            raise ValueError(f"Cannot pause job with status {self.status}")
        
        self.status = JobStatus.PAUSED
    
    def resume(self) -> None:
        """Resume paused job execution."""
        if self.status != JobStatus.PAUSED:
            raise ValueError(f"Cannot resume job with status {self.status}")
        
        self.status = JobStatus.RUNNING
    
    def timeout(self) -> None:
        """Mark job as timed out."""
        self.status = JobStatus.TIMEOUT
        self.completed_at = datetime.utcnow()
        self.error_message = f"Job exceeded maximum execution time of {self.job_config.max_execution_time_minutes} minutes"
    
    def get_duration_seconds(self) -> Optional[float]:
        """Get job execution duration in seconds."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == JobStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Check if job failed."""
        return self.status in [JobStatus.FAILED, JobStatus.TIMEOUT, JobStatus.CANCELLED]
    
    def should_retry(self) -> bool:
        """Check if job should be retried."""
        return (
            self.is_failed() and 
            self.job_config.retry_on_failure and 
            self.metrics.retries_attempted < self.job_config.max_retries
        )
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the job."""
        if tag not in self.tags:
            self.tags.append(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the job."""
        if tag in self.tags:
            self.tags.remove(tag)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the job."""
        duration = self.get_duration_seconds()
        
        return {
            "job_id": str(self.job_id.value),
            "status": self.status.value,
            "duration_seconds": duration,
            "records_processed": self.metrics.records_processed,
            "records_per_second": self.metrics.records_per_second,
            "rules_executed": self.metrics.rules_executed,
            "overall_pass_rate": self.metrics.overall_pass_rate,
            "critical_issues": self.metrics.critical_issues_found,
            "memory_peak_mb": self.metrics.memory_peak_mb,
            "cpu_utilization": self.metrics.cpu_utilization_avg,
            "retries": self.metrics.retries_attempted,
            "worker_failures": self.metrics.worker_failures
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary."""
        return {
            "job_info": {
                "id": str(self.job_id.value),
                "name": self.job_name,
                "type": self.job_type.value,
                "status": self.status.value,
                "created_by": str(self.created_by.value),
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None
            },
            "configuration": {
                "source_type": self.dataset_source.source_type,
                "rules_count": len(self.rules_applied),
                "parallel_execution": self.job_config.parallel_execution,
                "sampling_enabled": self.job_config.enable_sampling,
                "max_execution_time": self.job_config.max_execution_time_minutes
            },
            "performance": self.get_performance_summary(),
            "quality_results": {
                "overall_score": self.results.quality_scores.overall_score if self.results else None,
                "quality_grade": self.results.quality_scores.get_quality_grade() if self.results else None,
                "total_issues": len(self.results.quality_issues) if self.results else 0,
                "critical_issues": len(self.results.get_critical_issues()) if self.results else 0
            } if self.results else None,
            "error_info": {
                "message": self.error_message,
                "details": self.error_details
            } if self.error_message else None
        }
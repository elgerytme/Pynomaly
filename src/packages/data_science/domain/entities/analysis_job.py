"""Analysis Job entity for tracking data science workflows."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import Field, validator

from packages.core.domain.abstractions.base_entity import BaseEntity


class AnalysisType(str, Enum):
    """Types of data analysis."""
    
    EXPLORATORY = "exploratory"
    STATISTICAL = "statistical"
    PREDICTIVE = "predictive"
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PRESCRIPTIVE = "prescriptive"
    COMPARATIVE = "comparative"
    TREND_ANALYSIS = "trend_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_EVALUATION = "model_evaluation"


class JobStatus(str, Enum):
    """Analysis job execution status."""
    
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    RETRYING = "retrying"


class Priority(str, Enum):
    """Job execution priority."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AnalysisJob(BaseEntity):
    """Entity representing a data science analysis workflow.
    
    This entity manages the execution and tracking of data analysis tasks,
    from exploratory data analysis to complex statistical modeling.
    
    Attributes:
        name: Human-readable name for the analysis job
        analysis_type: Type of analysis being performed
        description: Detailed description of the analysis
        status: Current execution status
        priority: Execution priority level
        dataset_ids: List of datasets involved in the analysis
        target_variables: Variables being analyzed or predicted
        parameters: Analysis configuration parameters
        output_specifications: Desired outputs and formats
        progress_percentage: Current progress (0-100)
        started_at: When the job execution started
        completed_at: When the job execution completed
        estimated_duration_seconds: Estimated time to completion
        actual_duration_seconds: Actual execution time
        resource_requirements: Required compute resources
        error_message: Error details if job failed
        retry_count: Number of retry attempts
        max_retries: Maximum allowed retries
        results_uri: Location of analysis results
        intermediate_results: Partial results during execution
        dependencies: Other jobs this analysis depends on
        dependents: Jobs that depend on this analysis
        scheduled_at: When the job is scheduled to run
        executor_id: ID of the worker/executor running the job
        parent_job_id: Reference to parent job (for sub-analyses)
        workflow_id: Reference to larger workflow
        tags: Searchable tags for organization
        notifications: Notification settings for job events
    """
    
    name: str = Field(..., min_length=1, max_length=255)
    analysis_type: AnalysisType
    description: Optional[str] = Field(None, max_length=2000)
    status: JobStatus = Field(default=JobStatus.PENDING)
    priority: Priority = Field(default=Priority.NORMAL)
    
    # Data and targets
    dataset_ids: list[str] = Field(default_factory=list)
    target_variables: list[str] = Field(default_factory=list)
    
    # Configuration
    parameters: dict[str, Any] = Field(default_factory=dict)
    output_specifications: dict[str, Any] = Field(default_factory=dict)
    resource_requirements: dict[str, Any] = Field(default_factory=dict)
    
    # Execution tracking
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration_seconds: Optional[float] = Field(None, gt=0)
    actual_duration_seconds: Optional[float] = Field(None, ge=0)
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    
    # Results
    results_uri: Optional[str] = None
    intermediate_results: dict[str, Any] = Field(default_factory=dict)
    
    # Dependencies and scheduling
    dependencies: list[str] = Field(default_factory=list)
    dependents: list[str] = Field(default_factory=list)
    scheduled_at: Optional[datetime] = None
    
    # Execution context
    executor_id: Optional[str] = None
    parent_job_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Organization
    tags: list[str] = Field(default_factory=list)
    notifications: dict[str, Any] = Field(default_factory=dict)
    
    @validator('dataset_ids')
    def validate_dataset_ids(cls, v: list[str]) -> list[str]:
        """Validate dataset IDs."""
        if not v:
            raise ValueError("At least one dataset ID is required")
            
        for dataset_id in v:
            if not dataset_id.strip():
                raise ValueError("Dataset IDs cannot be empty")
                
        return v
    
    @validator('dependencies')
    def validate_dependencies(cls, v: list[str]) -> list[str]:
        """Validate job dependencies."""
        return [dep.strip() for dep in v if dep.strip()]
    
    @validator('tags')
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Validate tags list."""
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    def start_execution(self, executor_id: str) -> None:
        """Start job execution."""
        if self.status != JobStatus.QUEUED and self.status != JobStatus.PENDING:
            raise ValueError(f"Cannot start job in status: {self.status}")
            
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.executor_id = executor_id
        self.progress_percentage = 0.0
        self.mark_as_updated()
    
    def update_progress(self, percentage: float, 
                       intermediate_result: Optional[dict[str, Any]] = None) -> None:
        """Update job execution progress."""
        if self.status != JobStatus.RUNNING:
            raise ValueError("Can only update progress for running jobs")
            
        if not 0 <= percentage <= 100:
            raise ValueError("Progress percentage must be between 0 and 100")
            
        self.progress_percentage = percentage
        
        if intermediate_result:
            timestamp = datetime.utcnow().isoformat()
            self.intermediate_results[timestamp] = intermediate_result
            
        self.mark_as_updated()
    
    def complete_successfully(self, results_uri: str, 
                            final_results: Optional[dict[str, Any]] = None) -> None:
        """Mark job as successfully completed."""
        if self.status != JobStatus.RUNNING:
            raise ValueError("Can only complete running jobs")
            
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress_percentage = 100.0
        self.results_uri = results_uri
        
        if final_results:
            self.intermediate_results["final"] = final_results
            
        if self.started_at:
            self.actual_duration_seconds = (
                self.completed_at - self.started_at
            ).total_seconds()
            
        self.mark_as_updated()
    
    def fail_with_error(self, error_message: str, should_retry: bool = True) -> None:
        """Mark job as failed with error details."""
        if should_retry and self.retry_count < self.max_retries:
            self.status = JobStatus.RETRYING
            self.retry_count += 1
        else:
            self.status = JobStatus.FAILED
            
        self.error_message = error_message
        self.metadata["failed_at"] = datetime.utcnow().isoformat()
        self.mark_as_updated()
    
    def cancel_execution(self, reason: Optional[str] = None) -> None:
        """Cancel job execution."""
        if self.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            raise ValueError("Cannot cancel completed or failed jobs")
            
        self.status = JobStatus.CANCELLED
        
        if reason:
            self.metadata["cancellation_reason"] = reason
            
        self.metadata["cancelled_at"] = datetime.utcnow().isoformat()
        self.mark_as_updated()
    
    def pause_execution(self, reason: Optional[str] = None) -> None:
        """Pause job execution."""
        if self.status != JobStatus.RUNNING:
            raise ValueError("Can only pause running jobs")
            
        self.status = JobStatus.PAUSED
        
        if reason:
            self.metadata["pause_reason"] = reason
            
        self.metadata["paused_at"] = datetime.utcnow().isoformat()
        self.mark_as_updated()
    
    def resume_execution(self) -> None:
        """Resume paused job execution."""
        if self.status != JobStatus.PAUSED:
            raise ValueError("Can only resume paused jobs")
            
        self.status = JobStatus.RUNNING
        self.metadata["resumed_at"] = datetime.utcnow().isoformat()
        self.mark_as_updated()
    
    def add_dependency(self, job_id: str) -> None:
        """Add a job dependency."""
        if job_id not in self.dependencies:
            self.dependencies.append(job_id)
            self.mark_as_updated()
    
    def remove_dependency(self, job_id: str) -> None:
        """Remove a job dependency."""
        if job_id in self.dependencies:
            self.dependencies.remove(job_id)
            self.mark_as_updated()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the job."""
        tag = tag.strip().lower()
        if tag and tag not in self.tags:
            self.tags.append(tag)
            self.mark_as_updated()
    
    def estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining execution time in seconds."""
        if self.status != JobStatus.RUNNING or self.progress_percentage == 0:
            return self.estimated_duration_seconds
            
        if not self.started_at:
            return None
            
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        progress_ratio = self.progress_percentage / 100.0
        
        if progress_ratio > 0:
            estimated_total = elapsed / progress_ratio
            return max(0, estimated_total - elapsed)
            
        return None
    
    def is_ready_to_execute(self) -> bool:
        """Check if job is ready for execution."""
        return (
            self.status in [JobStatus.PENDING, JobStatus.QUEUED] and
            len(self.dataset_ids) > 0 and
            all(dep_id for dep_id in self.dependencies)  # All dependencies resolved
        )
    
    def is_blocking_others(self) -> bool:
        """Check if this job is blocking other jobs."""
        return len(self.dependents) > 0 and self.status != JobStatus.COMPLETED
    
    def get_execution_summary(self) -> dict[str, Any]:
        """Get execution summary for monitoring."""
        return {
            "job_id": str(self.id),
            "name": self.name,
            "analysis_type": self.analysis_type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "progress": self.progress_percentage,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "estimated_completion": self.estimate_remaining_time(),
            "retry_count": self.retry_count,
            "executor_id": self.executor_id,
            "dependencies_count": len(self.dependencies),
            "dependents_count": len(self.dependents),
        }
    
    def validate_invariants(self) -> None:
        """Validate domain invariants."""
        super().validate_invariants()
        
        # Business rule: Running jobs must have executor
        if self.status == JobStatus.RUNNING and not self.executor_id:
            raise ValueError("Running jobs must have an executor assigned")
        
        # Business rule: Completed jobs must have results
        if self.status == JobStatus.COMPLETED and not self.results_uri:
            raise ValueError("Completed jobs must have results URI")
        
        # Business rule: Failed jobs with retries must have retry count
        if self.status == JobStatus.FAILED and self.retry_count > self.max_retries:
            raise ValueError("Retry count cannot exceed maximum retries")
        
        # Business rule: Progress must be consistent with status
        if self.status == JobStatus.COMPLETED and self.progress_percentage != 100:
            raise ValueError("Completed jobs must have 100% progress")
        
        # Business rule: Started jobs must have start timestamp
        if self.status in [JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED]:
            if not self.started_at:
                raise ValueError("Executed jobs must have start timestamp")
"""Experiment entity for data science experimentation tracking."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import Field, validator

from packages.core.domain.abstractions.base_entity import BaseEntity
from packages.data_science.domain.value_objects.ml_model_metrics import MLModelMetrics
from packages.data_science.domain.value_objects.feature_importance import FeatureImportance


class Experiment(BaseEntity):
    """Entity representing a data science experiment.
    
    This entity tracks a complete data science experiment including
    dataset, algorithm, hyperparameters, results, and metadata.
    
    Attributes:
        experiment_id: Unique identifier for the experiment
        name: Human-readable experiment name
        description: Detailed description of the experiment
        status: Current status of the experiment
        created_at: When the experiment was created
        updated_at: When the experiment was last updated
        completed_at: When the experiment completed
        
        # Dataset information
        dataset_id: Identifier for the dataset used
        dataset_name: Name of the dataset
        dataset_version: Version of the dataset
        data_split: Information about train/validation/test splits
        
        # Algorithm configuration
        algorithm_name: Name of the algorithm/model used
        algorithm_version: Version of the algorithm
        hyperparameters: Algorithm hyperparameters
        preprocessing_steps: Data preprocessing pipeline
        
        # Execution environment
        environment: Execution environment details
        compute_resources: Computational resources used
        random_seed: Random seed for reproducibility
        
        # Results
        metrics: Model evaluation metrics
        feature_importance: Feature importance analysis
        model_artifacts: Paths to saved model artifacts
        predictions: Sample predictions or prediction metadata
        
        # Metadata
        tags: Experiment tags for organization
        notes: Additional notes or observations
        parent_experiment_id: Reference to parent experiment if applicable
        related_experiments: List of related experiment IDs
        
        # Tracking
        duration_seconds: Total experiment duration
        steps_completed: Number of completed steps
        total_steps: Total number of steps
        error_message: Error message if experiment failed
        
        # Collaboration
        created_by: User who created the experiment
        assigned_to: User assigned to the experiment
        reviewers: List of reviewers
        
        # Reproducibility
        code_version: Version of code used
        dependencies: Software dependencies
        data_hash: Hash of input data for integrity
        
        # Business context
        objective: Business objective of the experiment
        success_criteria: Success criteria definition
        business_impact: Measured business impact
        
        # Comparison and analysis
        baseline_experiment_id: Reference to baseline experiment
        comparison_results: Results of comparison with other experiments
        statistical_significance: Statistical significance of results
    """
    
    # Core identification
    experiment_id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    status: str = Field(default="created")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Dataset information
    dataset_id: str = Field(..., min_length=1)
    dataset_name: str = Field(..., min_length=1)
    dataset_version: str = Field(default="1.0.0")
    data_split: dict[str, Any] = Field(default_factory=dict)
    
    # Algorithm configuration
    algorithm_name: str = Field(..., min_length=1)
    algorithm_version: str = Field(default="1.0.0")
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing_steps: list[dict[str, Any]] = Field(default_factory=list)
    
    # Execution environment
    environment: dict[str, Any] = Field(default_factory=dict)
    compute_resources: dict[str, Any] = Field(default_factory=dict)
    random_seed: Optional[int] = Field(None, ge=0)
    
    # Results
    metrics: Optional[MLModelMetrics] = None
    feature_importance: Optional[FeatureImportance] = None
    model_artifacts: dict[str, str] = Field(default_factory=dict)
    predictions: dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    tags: list[str] = Field(default_factory=list)
    notes: str = Field(default="", max_length=5000)
    parent_experiment_id: Optional[UUID] = None
    related_experiments: list[UUID] = Field(default_factory=list)
    
    # Tracking
    duration_seconds: Optional[float] = Field(None, ge=0)
    steps_completed: int = Field(default=0, ge=0)
    total_steps: int = Field(default=1, gt=0)
    error_message: Optional[str] = None
    
    # Collaboration
    created_by: str = Field(..., min_length=1)
    assigned_to: Optional[str] = None
    reviewers: list[str] = Field(default_factory=list)
    
    # Reproducibility
    code_version: Optional[str] = None
    dependencies: dict[str, str] = Field(default_factory=dict)
    data_hash: Optional[str] = None
    
    # Business context
    objective: str = Field(default="", max_length=1000)
    success_criteria: list[str] = Field(default_factory=list)
    business_impact: dict[str, Any] = Field(default_factory=dict)
    
    # Comparison and analysis
    baseline_experiment_id: Optional[UUID] = None
    comparison_results: dict[str, Any] = Field(default_factory=dict)
    statistical_significance: Optional[dict[str, float]] = None
    
    @validator('status')
    def validate_status(cls, v: str) -> str:
        """Validate experiment status."""
        valid_statuses = {
            'created', 'queued', 'running', 'completed', 'failed', 
            'cancelled', 'paused', 'archived'
        }
        
        if v.lower() not in valid_statuses:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid_statuses}")
        
        return v.lower()
    
    @validator('updated_at')
    def validate_updated_at(cls, v: datetime, values: dict[str, Any]) -> datetime:
        """Ensure updated_at is not before created_at."""
        created_at = values.get('created_at')
        if created_at and v < created_at:
            raise ValueError("updated_at cannot be before created_at")
        return v
    
    @validator('completed_at')
    def validate_completed_at(cls, v: Optional[datetime], values: dict[str, Any]) -> Optional[datetime]:
        """Validate completion timestamp."""
        if v is not None:
            created_at = values.get('created_at')
            if created_at and v < created_at:
                raise ValueError("completed_at cannot be before created_at")
        return v
    
    @validator('steps_completed')
    def validate_steps_completed(cls, v: int, values: dict[str, Any]) -> int:
        """Validate steps completed against total steps."""
        total_steps = values.get('total_steps', 1)
        if v > total_steps:
            raise ValueError("steps_completed cannot exceed total_steps")
        return v
    
    def is_completed(self) -> bool:
        """Check if experiment is completed."""
        return self.status == 'completed'
    
    def is_running(self) -> bool:
        """Check if experiment is currently running."""
        return self.status == 'running'
    
    def is_failed(self) -> bool:
        """Check if experiment failed."""
        return self.status == 'failed'
    
    def get_progress_percentage(self) -> float:
        """Get experiment progress as percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.steps_completed / self.total_steps) * 100
    
    def get_duration_hours(self) -> Optional[float]:
        """Get experiment duration in hours."""
        if self.duration_seconds is None:
            return None
        return self.duration_seconds / 3600
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the experiment."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the experiment."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if experiment has a specific tag."""
        return tag in self.tags
    
    def start_experiment(self) -> None:
        """Mark experiment as started."""
        if self.status != 'created' and self.status != 'queued':
            raise ValueError(f"Cannot start experiment with status: {self.status}")
        
        self.status = 'running'
        self.updated_at = datetime.utcnow()
    
    def complete_experiment(self, final_metrics: Optional[MLModelMetrics] = None) -> None:
        """Mark experiment as completed."""
        if self.status != 'running':
            raise ValueError(f"Cannot complete experiment with status: {self.status}")
        
        self.status = 'completed'
        self.completed_at = datetime.utcnow()
        self.updated_at = self.completed_at
        self.steps_completed = self.total_steps
        
        if final_metrics:
            self.metrics = final_metrics
        
        # Calculate duration if not already set
        if self.duration_seconds is None:
            duration = self.completed_at - self.created_at
            self.duration_seconds = duration.total_seconds()
    
    def fail_experiment(self, error_message: str) -> None:
        """Mark experiment as failed."""
        if self.status not in ['running', 'queued']:
            raise ValueError(f"Cannot fail experiment with status: {self.status}")
        
        self.status = 'failed'
        self.error_message = error_message
        self.updated_at = datetime.utcnow()
        
        # Calculate duration if not already set
        if self.duration_seconds is None:
            duration = self.updated_at - self.created_at
            self.duration_seconds = duration.total_seconds()
    
    def update_progress(self, steps_completed: int, notes: str = "") -> None:
        """Update experiment progress."""
        if steps_completed > self.total_steps:
            raise ValueError("steps_completed cannot exceed total_steps")
        
        self.steps_completed = steps_completed
        self.updated_at = datetime.utcnow()
        
        if notes:
            self.notes = notes
    
    def set_metrics(self, metrics: MLModelMetrics) -> None:
        """Set experiment metrics."""
        self.metrics = metrics
        self.updated_at = datetime.utcnow()
    
    def set_feature_importance(self, feature_importance: FeatureImportance) -> None:
        """Set feature importance analysis."""
        self.feature_importance = feature_importance
        self.updated_at = datetime.utcnow()
    
    def compare_with_baseline(self, baseline_experiment: Experiment) -> dict[str, Any]:
        """Compare this experiment with a baseline experiment."""
        if not self.metrics or not baseline_experiment.metrics:
            return {"error": "Both experiments must have metrics for comparison"}
        
        comparison = {
            "experiment_id": str(self.experiment_id),
            "baseline_id": str(baseline_experiment.experiment_id),
            "comparison_timestamp": datetime.utcnow().isoformat(),
        }
        
        # Compare primary metrics
        self_primary = self.metrics.get_primary_metric()
        baseline_primary = baseline_experiment.metrics.get_primary_metric()
        
        if self_primary is not None and baseline_primary is not None:
            improvement = self_primary - baseline_primary
            improvement_pct = (improvement / baseline_primary) * 100 if baseline_primary != 0 else 0
            
            comparison.update({
                "primary_metric_improvement": improvement,
                "primary_metric_improvement_pct": improvement_pct,
                "is_improvement": improvement > 0,
                "current_score": self_primary,
                "baseline_score": baseline_primary
            })
        
        # Compare model scores
        self_model_score = self.metrics.calculate_model_score()
        baseline_model_score = baseline_experiment.metrics.calculate_model_score()
        
        comparison.update({
            "model_score_improvement": self_model_score - baseline_model_score,
            "current_model_score": self_model_score,
            "baseline_model_score": baseline_model_score
        })
        
        # Store comparison results
        self.comparison_results = comparison
        self.baseline_experiment_id = baseline_experiment.experiment_id
        
        return comparison
    
    def get_experiment_summary(self) -> dict[str, Any]:
        """Get comprehensive experiment summary."""
        summary = {
            "experiment_id": str(self.experiment_id),
            "name": self.name,
            "status": self.status,
            "progress": f"{self.get_progress_percentage():.1f}%",
            "algorithm": self.algorithm_name,
            "dataset": self.dataset_name,
            "created_by": self.created_by,
            "duration_hours": self.get_duration_hours(),
            "tags": self.tags,
        }
        
        # Add metrics summary if available
        if self.metrics:
            summary["metrics"] = {
                "primary_metric": self.metrics.get_primary_metric(),
                "model_score": self.metrics.calculate_model_score(),
                "model_type": self.metrics.model_type,
                "sample_size": self.metrics.sample_size
            }
        
        # Add feature importance summary if available
        if self.feature_importance:
            summary["feature_importance"] = {
                "method": self.feature_importance.method,
                "top_features": self.feature_importance.get_top_n_features(5),
                "total_features": len(self.feature_importance.feature_scores)
            }
        
        # Add business impact if available
        if self.business_impact:
            summary["business_impact"] = self.business_impact
        
        # Add error information if failed
        if self.is_failed() and self.error_message:
            summary["error"] = self.error_message
        
        return summary
    
    def to_dict(self) -> dict[str, Any]:
        """Convert experiment to dictionary representation."""
        data = super().to_dict()
        
        # Convert UUIDs to strings for serialization
        data["experiment_id"] = str(self.experiment_id)
        if self.parent_experiment_id:
            data["parent_experiment_id"] = str(self.parent_experiment_id)
        if self.baseline_experiment_id:
            data["baseline_experiment_id"] = str(self.baseline_experiment_id)
        
        data["related_experiments"] = [str(exp_id) for exp_id in self.related_experiments]
        
        # Convert datetime objects to ISO format
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        
        return data
    
    def __str__(self) -> str:
        """String representation of the experiment."""
        return f"Experiment(id={self.experiment_id}, name='{self.name}', status='{self.status}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"Experiment(experiment_id={self.experiment_id}, "
            f"name='{self.name}', status='{self.status}', "
            f"algorithm='{self.algorithm_name}', dataset='{self.dataset_name}', "
            f"progress={self.get_progress_percentage():.1f}%)"
        )
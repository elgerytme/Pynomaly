"""Transformation step value object."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """Types of transformation steps."""
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    DATA_CLEANING = "data_cleaning"
    MISSING_VALUE_IMPUTATION = "missing_value_imputation"
    OUTLIER_DETECTION = "outlier_detection"
    DUPLICATE_REMOVAL = "duplicate_removal"
    FEATURE_SCALING = "feature_scaling"
    CATEGORICAL_ENCODING = "categorical_encoding"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    DATA_SPLITTING = "data_splitting"
    DATA_EXPORT = "data_export"


class StepStatus(str, Enum):
    """Execution status of transformation steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TransformationStep(BaseModel):
    """
    Value object representing a single transformation step in a pipeline.
    
    Encapsulates the configuration, execution state, and results of
    an individual data transformation operation.
    """
    
    name: str
    step_type: StepType
    description: Optional[str] = None
    
    # Configuration
    parameters: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    optional: bool = False
    
    # Execution state
    status: StepStatus = StepStatus.PENDING
    order: int = 0
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    
    # Results
    input_records: Optional[int] = None
    output_records: Optional[int] = None
    input_features: Optional[int] = None
    output_features: Optional[int] = None
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
    
    def start_execution(self) -> None:
        """Mark step as started."""
        self.status = StepStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete_execution(
        self,
        input_records: int,
        output_records: int,
        input_features: int,
        output_features: int,
        execution_time: float
    ) -> None:
        """Mark step as completed with results."""
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.input_records = input_records
        self.output_records = output_records
        self.input_features = input_features
        self.output_features = output_features
        self.execution_time_seconds = execution_time
    
    def fail_execution(self, error_message: str) -> None:
        """Mark step as failed with error message."""
        self.status = StepStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
    
    def skip_execution(self, reason: str) -> None:
        """Skip step execution with reason."""
        self.status = StepStatus.SKIPPED
        self.completed_at = datetime.utcnow()
        self.metadata["skip_reason"] = reason
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message to the step."""
        self.warnings.append(warning)
    
    @property
    def is_completed(self) -> bool:
        """Check if step completed successfully."""
        return self.status == StepStatus.COMPLETED
    
    @property
    def has_failed(self) -> bool:
        """Check if step failed."""
        return self.status == StepStatus.FAILED
    
    @property
    def is_running(self) -> bool:
        """Check if step is currently running."""
        return self.status == StepStatus.RUNNING
    
    @property
    def was_skipped(self) -> bool:
        """Check if step was skipped."""
        return self.status == StepStatus.SKIPPED
    
    @property
    def records_changed(self) -> Optional[int]:
        """Calculate change in record count."""
        if self.input_records is None or self.output_records is None:
            return None
        return self.output_records - self.input_records
    
    @property
    def features_changed(self) -> Optional[int]:
        """Calculate change in feature count."""
        if self.input_features is None or self.output_features is None:
            return None
        return self.output_features - self.input_features
    
    @classmethod
    def create_data_loading_step(
        cls,
        name: str = "data_loading",
        source_path: str = "",
        **parameters: Any
    ) -> TransformationStep:
        """Create a data loading step."""
        return cls(
            name=name,
            step_type=StepType.DATA_LOADING,
            description="Load data from source",
            parameters={"source_path": source_path, **parameters},
            order=0
        )
    
    @classmethod
    def create_cleaning_step(
        cls,
        name: str = "data_cleaning",
        strategy: str = "auto",
        **parameters: Any
    ) -> TransformationStep:
        """Create a data cleaning step."""
        return cls(
            name=name,
            step_type=StepType.DATA_CLEANING,
            description="Clean and preprocess data",
            parameters={"strategy": strategy, **parameters},
            order=10
        )
    
    @classmethod
    def create_feature_engineering_step(
        cls,
        name: str = "feature_engineering",
        methods: List[str] = None,
        **parameters: Any
    ) -> TransformationStep:
        """Create a feature engineering step."""
        return cls(
            name=name,
            step_type=StepType.FEATURE_ENGINEERING,
            description="Generate new features",
            parameters={"methods": methods or [], **parameters},
            order=20
        )
    
    @classmethod
    def create_scaling_step(
        cls,
        name: str = "feature_scaling",
        method: str = "standard",
        **parameters: Any
    ) -> TransformationStep:
        """Create a feature scaling step."""
        return cls(
            name=name,
            step_type=StepType.FEATURE_SCALING,
            description="Scale numerical features",
            parameters={"method": method, **parameters},
            order=30
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary representation."""
        return {
            "name": self.name,
            "step_type": self.step_type,
            "description": self.description,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "optional": self.optional,
            "status": self.status,
            "order": self.order,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_seconds": self.execution_time_seconds,
            "input_records": self.input_records,
            "output_records": self.output_records,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }
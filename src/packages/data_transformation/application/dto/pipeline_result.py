"""Pipeline result data transfer object."""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

import pandas as pd
from pydantic import BaseModel, Field


class PipelineResult(BaseModel):
    """
    Data transfer object for pipeline execution results.
    
    Contains the transformed data along with execution metadata,
    performance metrics, and detailed information about the
    transformation process.
    """
    
    # Core result data
    data: Optional[pd.DataFrame] = None
    pipeline_id: UUID
    
    # Execution metrics
    records_processed: int = 0
    features_created: int = 0
    execution_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Step execution summary
    steps_completed: int = 0
    steps_failed: int = 0
    total_steps: int = 0
    
    # Detailed metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    # Performance statistics
    processing_rate_records_per_second: Optional[float] = None
    memory_efficiency_mb_per_record: Optional[float] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def __init__(self, **data: Any) -> None:
        """Initialize pipeline result and calculate derived metrics."""
        super().__init__(**data)
        self._calculate_derived_metrics()
    
    def _calculate_derived_metrics(self) -> None:
        """Calculate derived performance metrics."""
        # Calculate processing rate
        if self.execution_time_seconds > 0 and self.records_processed > 0:
            self.processing_rate_records_per_second = (
                self.records_processed / self.execution_time_seconds
            )
        
        # Calculate memory efficiency
        if self.records_processed > 0 and self.memory_usage_mb > 0:
            self.memory_efficiency_mb_per_record = (
                self.memory_usage_mb / self.records_processed
            )
        
        # Calculate total steps
        self.total_steps = self.steps_completed + self.steps_failed
    
    @property
    def success_rate(self) -> float:
        """Calculate step success rate."""
        if self.total_steps == 0:
            return 1.0
        return self.steps_completed / self.total_steps
    
    @property
    def has_data(self) -> bool:
        """Check if result contains data."""
        return self.data is not None and not self.data.empty
    
    @property
    def data_shape(self) -> Optional[tuple[int, int]]:
        """Get shape of result data."""
        if self.data is not None:
            return self.data.shape
        return None
    
    @property
    def feature_names(self) -> Optional[list[str]]:
        """Get list of feature names."""
        if self.data is not None and hasattr(self.data, 'columns'):
            return self.data.columns.tolist()
        return None
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline execution results."""
        return {
            "pipeline_id": str(self.pipeline_id),
            "success": self.steps_failed == 0,
            "records_processed": self.records_processed,
            "features_created": self.features_created,
            "execution_time_seconds": self.execution_time_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "processing_rate": self.processing_rate_records_per_second,
            "memory_efficiency": self.memory_efficiency_mb_per_record,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "success_rate": self.success_rate,
            "data_shape": self.data_shape,
            "has_warnings": len(self.warnings) > 0,
            "has_errors": len(self.errors) > 0,
            "warning_count": len(self.warnings),
            "error_count": len(self.errors)
        }
    
    def export_data(
        self,
        file_path: str,
        format: str = "csv",
        **kwargs: Any
    ) -> None:
        """
        Export the transformed data to a file.
        
        Args:
            file_path: Output file path
            format: Export format ('csv', 'parquet', 'json', 'excel')
            **kwargs: Additional export parameters
        """
        if self.data is None:
            raise ValueError("No data to export")
        
        if format.lower() == "csv":
            self.data.to_csv(file_path, index=False, **kwargs)
        elif format.lower() == "parquet":
            self.data.to_parquet(file_path, index=False, **kwargs)
        elif format.lower() == "json":
            self.data.to_json(file_path, orient="records", **kwargs)
        elif format.lower() == "excel":
            self.data.to_excel(file_path, index=False, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary (excluding data for serialization)."""
        result_dict = self.model_dump(exclude={"data"})
        result_dict["data_shape"] = self.data_shape
        result_dict["feature_names"] = self.feature_names
        return result_dict
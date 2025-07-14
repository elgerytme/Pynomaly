"""Pipeline result data transfer objects."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field


class StepResult(BaseModel):
    """Result data for individual pipeline steps."""
    
    step_id: str
    step_type: str
    status: str
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class ValidationResult(BaseModel):
    """Data validation result."""
    
    is_valid: bool
    score: float
    checks_passed: int
    total_checks: int
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class ExecutionMetrics(BaseModel):
    """Execution performance metrics."""
    
    total_execution_time: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    rows_processed: int
    columns_processed: Optional[int] = None
    data_size_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class PipelineResult(BaseModel):
    """
    Data transfer object for pipeline execution results.
    
    Contains the transformed data along with execution metadata,
    performance metrics, and detailed information about the
    transformation process.
    """
    
    # Core result data
    success: bool
    data: Optional[pd.DataFrame] = None
    pipeline_id: str
    execution_time: float
    
    # Step execution details
    steps_executed: List[StepResult] = Field(default_factory=list)
    
    # Optional components
    config: Optional[Any] = None
    validation_results: Optional[ValidationResult] = None
    metrics: Optional[ExecutionMetrics] = None
    error_message: Optional[str] = None
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "success": self.success,
            "pipeline_id": self.pipeline_id,
            "execution_time": self.execution_time,
            "steps_executed": [step.to_dict() for step in self.steps_executed],
            "error_message": self.error_message,
            "data_shape": list(self.data.shape) if self.data is not None else None
        }
        
        if self.validation_results:
            result["validation_results"] = self.validation_results.to_dict()
            
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
            
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution summary."""
        completed_steps = sum(1 for step in self.steps_executed if step.status == "completed")
        failed_steps = sum(1 for step in self.steps_executed if step.status == "failed")
        total_steps = len(self.steps_executed)
        
        return {
            "success": self.success,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "success_rate": completed_steps / total_steps if total_steps > 0 else 0.0,
            "execution_time": self.execution_time,
            "pipeline_id": self.pipeline_id
        }
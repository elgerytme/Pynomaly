"""Data Transfer Objects for machine learning pipeline operations."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID
from datetime import datetime


@dataclass(frozen=True)
class CreatePipelineRequestDTO:
    """Request DTO for creating ML pipeline."""
    name: str
    description: Optional[str]
    pipeline_type: str
    steps: List[Dict[str, Any]]
    user_id: UUID
    parameters: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None


@dataclass(frozen=True)
class CreatePipelineResponseDTO:
    """Response DTO for pipeline creation."""
    pipeline_id: UUID
    name: str
    status: str
    version_number: str
    created_at: datetime


@dataclass(frozen=True)
class ExecutePipelineRequestDTO:
    """Request DTO for executing ML pipeline."""
    pipeline_id: UUID
    execution_config: Optional[Dict[str, Any]] = None
    input_data: Optional[Dict[str, Any]] = None
    user_id: UUID


@dataclass(frozen=True)
class ExecutePipelineResponseDTO:
    """Response DTO for pipeline execution."""
    execution_id: str
    pipeline_id: UUID
    status: str
    started_at: datetime
    progress: float = 0.0
    current_step: Optional[str] = None


@dataclass(frozen=True)
class PipelineStatusRequestDTO:
    """Request DTO for getting pipeline status."""
    pipeline_id: UUID
    include_logs: bool = False
    include_metrics: bool = False


@dataclass(frozen=True)
class PipelineStatusResponseDTO:
    """Response DTO for pipeline status."""
    pipeline_id: UUID
    execution_id: Optional[str]
    status: str
    progress: float
    current_step: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    execution_time_seconds: Optional[float]
    step_statuses: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    logs: Optional[List[str]] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class TrainModelRequestDTO:
    """Request DTO for training ML model within pipeline."""
    pipeline_id: UUID
    model_config: Dict[str, Any]
    training_data: Any
    validation_data: Optional[Any] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    user_id: UUID


@dataclass(frozen=True)
class TrainModelResponseDTO:
    """Response DTO for model training."""
    model_id: UUID
    pipeline_id: UUID
    training_status: str
    performance_metrics: Optional[Dict[str, Any]] = None
    model_artifacts: Optional[Dict[str, str]] = None
    training_time_seconds: Optional[float] = None


@dataclass(frozen=True)
class ValidateModelRequestDTO:
    """Request DTO for model validation within pipeline."""
    model_id: UUID
    pipeline_id: UUID
    validation_data: Any
    validation_config: Optional[Dict[str, Any]] = None
    user_id: UUID


@dataclass(frozen=True)
class ValidateModelResponseDTO:
    """Response DTO for model validation."""
    model_id: UUID
    validation_status: str
    performance_metrics: Dict[str, Any]
    validation_report: Dict[str, Any]
    recommendations: Optional[List[str]] = None
    passed_validation: bool


@dataclass(frozen=True)
class DeployModelRequestDTO:
    """Request DTO for model deployment within pipeline."""
    model_id: UUID
    pipeline_id: UUID
    deployment_config: Dict[str, Any]
    user_id: UUID


@dataclass(frozen=True)
class DeployModelResponseDTO:
    """Response DTO for model deployment."""
    model_id: UUID
    deployment_id: str
    deployment_status: str
    endpoint_url: Optional[str] = None
    deployment_config: Dict[str, Any]
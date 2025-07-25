"""
Data Transfer Objects for cross-package communication.

This module defines stable DTOs that enable different domains to
communicate while maintaining proper boundaries and loose coupling.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


class DetectionStatus(Enum):
    """Status of anomaly detection operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataQualityStatus(Enum):
    """Status of data quality operations."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ModelStatus(Enum):
    """Status of ML model operations."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"


@dataclass
class BaseDTO(ABC):
    """Base class for all DTOs."""
    id: str
    created_at: datetime
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()


# Anomaly Detection DTOs
@dataclass
class DetectionRequest(BaseDTO):
    """Request for anomaly detection analysis."""
    dataset_id: str
    algorithm: str
    parameters: Dict[str, Any]
    priority: str = "normal"
    callback_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DetectionResult(BaseDTO):
    """Result of anomaly detection analysis."""
    request_id: str
    status: DetectionStatus
    anomalies_count: int
    anomaly_scores: List[float]
    anomaly_indices: List[int]
    confidence_scores: List[float]
    execution_time_ms: int
    algorithm_used: str
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# Data Quality DTOs
@dataclass
class DataQualityRequest(BaseDTO):
    """Request for data quality assessment."""
    dataset_id: str
    quality_rules: List[str]
    threshold: float = 0.8
    include_profiling: bool = True
    callback_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DataQualityResult(BaseDTO):
    """Result of data quality assessment."""
    request_id: str
    dataset_id: str
    status: DataQualityStatus
    overall_score: float
    rule_results: Dict[str, Dict[str, Any]]
    issues_found: List[Dict[str, Any]]
    recommendations: List[str]
    execution_time_ms: int
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ML Model DTOs
@dataclass
class ModelTrainingRequest(BaseDTO):
    """Request for ML model training."""
    model_type: str
    dataset_id: str
    training_config: Dict[str, Any]
    validation_split: float = 0.2
    hyperparameters: Optional[Dict[str, Any]] = None
    experiment_id: Optional[str] = None
    callback_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelTrainingResult(BaseDTO):
    """Result of ML model training."""
    request_id: str
    model_id: str
    status: ModelStatus
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    model_artifacts: Dict[str, str]
    training_time_ms: int
    experiment_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# Analytics DTOs
@dataclass
class AnalyticsRequest(BaseDTO):
    """Request for analytics processing."""
    query_type: str
    parameters: Dict[str, Any]
    time_range: Dict[str, datetime]
    aggregation_level: str = "daily"
    include_forecast: bool = False
    callback_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnalyticsResult(BaseDTO):
    """Result of analytics processing."""
    request_id: str
    query_type: str
    data: List[Dict[str, Any]]
    summary_stats: Dict[str, float]
    insights: List[str]
    forecast_data: Optional[List[Dict[str, Any]]] = None
    execution_time_ms: int
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# System Health DTOs
@dataclass
class HealthCheckRequest(BaseDTO):
    """Request for system health check."""
    component: str
    include_dependencies: bool = True
    timeout_ms: int = 5000
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HealthCheckResult(BaseDTO):
    """Result of system health check."""
    request_id: str
    component: str
    status: str
    response_time_ms: int
    dependencies: Dict[str, str]
    metrics: Dict[str, float]
    issues: List[str]
    metadata: Optional[Dict[str, Any]] = None
"""GraphQL types for Pynomaly API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

import strawberry
from strawberry.scalars import JSON


@strawberry.type
class User:
    """User type for GraphQL."""
    
    id: UUID
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    two_factor_enabled: bool = False


@strawberry.input
class UserInput:
    """User input for GraphQL mutations."""
    
    username: str
    email: str
    password: str
    roles: Optional[List[str]] = None


@strawberry.type
class Detector:
    """Detector type for GraphQL."""
    
    id: UUID
    name: str
    algorithm_name: str
    contamination_rate: float
    parameters: JSON
    metadata: JSON
    is_fitted: bool
    created_at: datetime
    updated_at: datetime


@strawberry.input
class DetectorInput:
    """Detector input for GraphQL mutations."""
    
    name: str
    algorithm_name: str
    contamination_rate: float
    parameters: Optional[JSON] = None
    metadata: Optional[JSON] = None


@strawberry.type
class AnomalyDetectionResult:
    """Anomaly detection result type for GraphQL."""
    
    id: UUID
    detector_id: UUID
    dataset_id: UUID
    anomaly_indices: List[int]
    anomaly_scores: List[float]
    threshold: float
    processing_time_ms: float
    algorithm_metadata: JSON
    created_at: datetime


@strawberry.input
class DetectionResultInput:
    """Detection result input for GraphQL mutations."""
    
    detector_id: UUID
    dataset_id: UUID
    data: JSON
    validate_features: bool = True
    save_results: bool = True


@strawberry.type
class DetectionJob:
    """Detection job type for GraphQL."""
    
    id: UUID
    detector_id: UUID
    dataset_id: UUID
    status: str
    progress: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[AnomalyDetectionResult] = None


@strawberry.input
class DetectionJobInput:
    """Detection job input for GraphQL mutations."""
    
    detector_id: UUID
    dataset_id: UUID
    parameters: Optional[JSON] = None


@strawberry.type
class Dataset:
    """Dataset type for GraphQL."""
    
    id: UUID
    name: str
    description: Optional[str] = None
    feature_names: List[str]
    sample_count: int
    feature_count: int
    created_at: datetime
    updated_at: datetime
    metadata: JSON


@strawberry.input
class DatasetInput:
    """Dataset input for GraphQL mutations."""
    
    name: str
    description: Optional[str] = None
    data: JSON
    metadata: Optional[JSON] = None


@strawberry.type
class TrainingJob:
    """Training job type for GraphQL."""
    
    id: UUID
    detector_id: UUID
    dataset_id: UUID
    status: str
    progress: float
    started_at: datetime
    completed_at: Optional[datetime] = None
    training_time_ms: Optional[float] = None
    validation_results: Optional[JSON] = None
    error_message: Optional[str] = None


@strawberry.input
class TrainingJobInput:
    """Training job input for GraphQL mutations."""
    
    detector_id: UUID
    dataset_id: UUID
    validate_data: bool = True
    save_model: bool = True
    parameters: Optional[JSON] = None


@strawberry.type
class ModelMetrics:
    """Model metrics type for GraphQL."""
    
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    custom_metrics: Optional[JSON] = None


@strawberry.type
class ModelVersion:
    """Model version type for GraphQL."""
    
    id: UUID
    detector_id: UUID
    version: str
    model_path: str
    metrics: Optional[ModelMetrics] = None
    created_at: datetime
    is_active: bool = False


@strawberry.type
class ApiKey:
    """API key type for GraphQL."""
    
    id: UUID
    key_id: str  # Don't expose the actual key
    name: str
    user_id: UUID
    permissions: List[str]
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None


@strawberry.input
class ApiKeyInput:
    """API key input for GraphQL mutations."""
    
    name: str
    permissions: List[str]
    expires_at: Optional[datetime] = None


@strawberry.type
class AuditLog:
    """Audit log type for GraphQL."""
    
    id: UUID
    user_id: Optional[UUID] = None
    action: str
    resource: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    success: bool
    details: JSON
    risk_score: int


@strawberry.type
class SystemHealth:
    """System health type for GraphQL."""
    
    status: str
    timestamp: datetime
    version: str
    services: JSON
    metrics: JSON


@strawberry.type
class SecurityMetrics:
    """Security metrics type for GraphQL."""
    
    total_users: int
    active_sessions: int
    api_keys_active: int
    failed_actions_24h: int
    high_risk_events_24h: int
    blocked_ips: int
    audit_logs_total: int


@strawberry.type
class PerformanceMetrics:
    """Performance metrics type for GraphQL."""
    
    avg_detection_time_ms: float
    avg_training_time_ms: float
    total_detections: int
    total_trainings: int
    success_rate: float
    error_rate: float
    throughput_per_hour: float


@strawberry.type
class WebSocketEvent:
    """WebSocket event type for GraphQL subscriptions."""
    
    event_type: str
    timestamp: datetime
    data: JSON
    user_id: Optional[UUID] = None


@strawberry.type
class JobProgress:
    """Job progress type for GraphQL subscriptions."""
    
    job_id: UUID
    job_type: str
    progress: float
    status: str
    message: Optional[str] = None
    timestamp: datetime


@strawberry.type
class DetectionAlert:
    """Detection alert type for GraphQL subscriptions."""
    
    id: UUID
    detector_id: UUID
    severity: str
    message: str
    anomaly_count: int
    timestamp: datetime
    metadata: JSON


# Error types
@strawberry.type
class ValidationError:
    """Validation error type."""
    
    field: str
    message: str
    code: str


@strawberry.type
class ApiError:
    """API error type."""
    
    message: str
    code: str
    details: Optional[JSON] = None


# Response types with error handling
@strawberry.type
class DetectorResponse:
    """Detector response with error handling."""
    
    detector: Optional[Detector] = None
    errors: Optional[List[ValidationError]] = None
    success: bool = True


@strawberry.type
class DetectionResponse:
    """Detection response with error handling."""
    
    result: Optional[AnomalyDetectionResult] = None
    job: Optional[DetectionJob] = None
    errors: Optional[List[ValidationError]] = None
    success: bool = True


@strawberry.type
class TrainingResponse:
    """Training response with error handling."""
    
    job: Optional[TrainingJob] = None
    errors: Optional[List[ValidationError]] = None
    success: bool = True


@strawberry.type
class UserResponse:
    """User response with error handling."""
    
    user: Optional[User] = None
    errors: Optional[List[ValidationError]] = None
    success: bool = True


@strawberry.type
class AuthResponse:
    """Authentication response."""
    
    token: Optional[str] = None
    refresh_token: Optional[str] = None
    user: Optional[User] = None
    expires_in: Optional[int] = None
    errors: Optional[List[ValidationError]] = None
    success: bool = True


@strawberry.input
class LoginInput:
    """Login input for authentication."""
    
    username: str
    password: str
    remember_me: bool = False


@strawberry.input
class RefreshTokenInput:
    """Refresh token input."""
    
    refresh_token: str


# Pagination types
@strawberry.type
class PageInfo:
    """Page information for pagination."""
    
    has_next_page: bool
    has_previous_page: bool
    start_cursor: Optional[str] = None
    end_cursor: Optional[str] = None


@strawberry.type
class DetectorConnection:
    """Detector connection for pagination."""
    
    edges: List[Detector]
    page_info: PageInfo
    total_count: int


@strawberry.type
class DetectionResultConnection:
    """Detection result connection for pagination."""
    
    edges: List[AnomalyDetectionResult]
    page_info: PageInfo
    total_count: int


@strawberry.type
class AuditLogConnection:
    """Audit log connection for pagination."""
    
    edges: List[AuditLog]
    page_info: PageInfo
    total_count: int


# Filter inputs
@strawberry.input
class DetectorFilter:
    """Detector filter input."""
    
    name: Optional[str] = None
    algorithm_name: Optional[str] = None
    is_fitted: Optional[bool] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


@strawberry.input
class DetectionResultFilter:
    """Detection result filter input."""
    
    detector_id: Optional[UUID] = None
    dataset_id: Optional[UUID] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    min_anomaly_count: Optional[int] = None
    max_anomaly_count: Optional[int] = None


@strawberry.input
class AuditLogFilter:
    """Audit log filter input."""
    
    user_id: Optional[UUID] = None
    action: Optional[str] = None
    resource: Optional[str] = None
    success: Optional[bool] = None
    min_risk_score: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


# Sorting inputs
@strawberry.enum
class SortOrder:
    """Sort order enumeration."""
    
    ASC = "asc"
    DESC = "desc"


@strawberry.input
class DetectorSort:
    """Detector sort input."""
    
    field: str
    order: SortOrder = SortOrder.ASC


@strawberry.input
class DetectionResultSort:
    """Detection result sort input."""
    
    field: str
    order: SortOrder = SortOrder.ASC


@strawberry.input
class AuditLogSort:
    """Audit log sort input."""
    
    field: str
    order: SortOrder = SortOrder.ASC
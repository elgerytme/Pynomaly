"""GraphQL type definitions for Pynomaly API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

import strawberry


@strawberry.enum
class UserRole(Enum):
    """User role enumeration."""
    VIEWER = "viewer"
    ANALYST = "analyst"
    DATA_SCIENTIST = "data_scientist"
    TENANT_ADMIN = "tenant_admin"
    SUPER_ADMIN = "super_admin"


@strawberry.enum
class DetectorAlgorithm(Enum):
    """Anomaly detection algorithm types."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    AUTOENCODER = "autoencoder"
    VAE = "vae"
    GAN = "gan"
    LSTM_AD = "lstm_ad"
    PROPHET = "prophet"
    ARIMA = "arima"


@strawberry.enum
class DetectorStatus(Enum):
    """Detector status enumeration."""
    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    PREDICTING = "predicting"
    STOPPED = "stopped"


@strawberry.enum
class DatasetStatus(Enum):
    """Dataset status enumeration."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


@strawberry.type
class User:
    """User type for GraphQL."""
    
    id: strawberry.ID
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    @strawberry.field
    async def datasets(self, info) -> List[Dataset]:
        """Get datasets owned by this user."""
        from pynomaly.presentation.api.graphql.resolvers.dataset_resolvers import resolve_user_datasets
        return await resolve_user_datasets(info, str(self.id))
    
    @strawberry.field
    async def detectors(self, info) -> List[Detector]:
        """Get detectors owned by this user."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_user_detectors
        return await resolve_user_detectors(info, str(self.id))


@strawberry.type
class Dataset:
    """Dataset type for GraphQL."""
    
    id: strawberry.ID
    name: str
    description: str
    status: DatasetStatus
    file_path: Optional[str] = None
    size_bytes: Optional[int] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    owner_id: strawberry.ID
    
    @strawberry.field
    async def owner(self, info) -> User:
        """Get the owner of this dataset."""
        from pynomaly.presentation.api.graphql.resolvers.user_resolvers import resolve_user_by_id
        return await resolve_user_by_id(info, str(self.owner_id))
    
    @strawberry.field
    async def detectors(self, info) -> List[Detector]:
        """Get detectors using this dataset."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_dataset_detectors
        return await resolve_dataset_detectors(info, str(self.id))
    
    @strawberry.field
    async def statistics(self, info) -> Optional[DatasetStatistics]:
        """Get dataset statistics."""
        from pynomaly.presentation.api.graphql.resolvers.dataset_resolvers import resolve_dataset_statistics
        return await resolve_dataset_statistics(info, str(self.id))


@strawberry.type
class DatasetStatistics:
    """Dataset statistics type."""
    
    total_rows: int
    total_columns: int
    numerical_columns: int
    categorical_columns: int
    missing_values: int
    memory_usage_mb: float
    summary: str


@strawberry.type
class Detector:
    """Anomaly detector type for GraphQL."""
    
    id: strawberry.ID
    name: str
    algorithm: DetectorAlgorithm
    status: DetectorStatus
    parameters: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    dataset_id: strawberry.ID
    owner_id: strawberry.ID
    model_id: Optional[strawberry.ID] = None
    
    @strawberry.field
    async def dataset(self, info) -> Dataset:
        """Get the dataset used by this detector."""
        from pynomaly.presentation.api.graphql.resolvers.dataset_resolvers import resolve_dataset_by_id
        return await resolve_dataset_by_id(info, str(self.dataset_id))
    
    @strawberry.field
    async def owner(self, info) -> User:
        """Get the owner of this detector."""
        from pynomaly.presentation.api.graphql.resolvers.user_resolvers import resolve_user_by_id
        return await resolve_user_by_id(info, str(self.owner_id))
    
    @strawberry.field
    async def model(self, info) -> Optional[Model]:
        """Get the trained model for this detector."""
        if not self.model_id:
            return None
        from pynomaly.presentation.api.graphql.resolvers.model_resolvers import resolve_model_by_id
        return await resolve_model_by_id(info, str(self.model_id))
    
    @strawberry.field
    async def training_history(self, info) -> List[TrainingResult]:
        """Get training history for this detector."""
        from pynomaly.presentation.api.graphql.resolvers.training_resolvers import resolve_detector_training_history
        return await resolve_detector_training_history(info, str(self.id))
    
    @strawberry.field
    async def detection_results(self, info, limit: int = 10) -> List[AnomalyDetectionResult]:
        """Get recent detection results for this detector."""
        from pynomaly.presentation.api.graphql.resolvers.detection_resolvers import resolve_detector_results
        return await resolve_detector_results(info, str(self.id), limit)


@strawberry.type
class Model:
    """Machine learning model type for GraphQL."""
    
    id: strawberry.ID
    name: str
    algorithm: DetectorAlgorithm
    version: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    training_time_seconds: Optional[float] = None
    model_size_bytes: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    detector_id: strawberry.ID
    
    @strawberry.field
    async def detector(self, info) -> Detector:
        """Get the detector that owns this model."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_detector_by_id
        return await resolve_detector_by_id(info, str(self.detector_id))
    
    @strawberry.field
    async def metrics(self, info) -> Optional[ModelMetrics]:
        """Get detailed model metrics."""
        from pynomaly.presentation.api.graphql.resolvers.model_resolvers import resolve_model_metrics
        return await resolve_model_metrics(info, str(self.id))


@strawberry.type
class ModelMetrics:
    """Model performance metrics type."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    confusion_matrix: Optional[str] = None  # JSON string
    feature_importance: Optional[str] = None  # JSON string
    training_loss: Optional[List[float]] = None
    validation_loss: Optional[List[float]] = None


@strawberry.type
class TrainingResult:
    """Training result type for GraphQL."""
    
    id: strawberry.ID
    detector_id: strawberry.ID
    status: str
    progress: float  # 0.0 to 1.0
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    current_step_number: Optional[int] = None
    error_message: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    @strawberry.field
    async def detector(self, info) -> Detector:
        """Get the detector being trained."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_detector_by_id
        return await resolve_detector_by_id(info, str(self.detector_id))
    
    @strawberry.field
    async def logs(self, info) -> List[str]:
        """Get training logs."""
        from pynomaly.presentation.api.graphql.resolvers.training_resolvers import resolve_training_logs
        return await resolve_training_logs(info, str(self.id))


@strawberry.type
class AnomalyDetectionResult:
    """Anomaly detection result type for GraphQL."""
    
    id: strawberry.ID
    detector_id: strawberry.ID
    is_anomaly: bool
    anomaly_score: float
    confidence: Optional[float] = None
    input_data: Optional[str] = None  # JSON string
    explanation: Optional[str] = None
    timestamp: datetime
    processing_time_ms: Optional[float] = None
    
    @strawberry.field
    async def detector(self, info) -> Detector:
        """Get the detector that produced this result."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_detector_by_id
        return await resolve_detector_by_id(info, str(self.detector_id))


@strawberry.type
class AnomalyAlert:
    """Anomaly alert type for real-time notifications."""
    
    id: strawberry.ID
    detector_id: strawberry.ID
    alert_type: str
    severity: str
    message: str
    data: Optional[str] = None  # JSON string
    timestamp: datetime
    acknowledged: bool = False
    
    @strawberry.field
    async def detector(self, info) -> Detector:
        """Get the detector that triggered this alert."""
        from pynomaly.presentation.api.graphql.resolvers.detector_resolvers import resolve_detector_by_id
        return await resolve_detector_by_id(info, str(self.detector_id))


@strawberry.type
class ApiError:
    """API error type for GraphQL."""
    
    code: str
    message: str
    details: Optional[str] = None
    timestamp: datetime


@strawberry.type
class PaginationInfo:
    """Pagination information type."""
    
    has_next_page: bool
    has_previous_page: bool
    total_count: int
    page_size: int
    current_page: int


@strawberry.type
class AuthResult:
    """Authentication result type."""
    
    access_token: str
    token_type: str
    expires_in: int
    user: User


# Input types for mutations

@strawberry.input
class CreateDatasetInput:
    """Input for creating a dataset."""
    
    name: str
    description: str
    data_url: Optional[str] = None


@strawberry.input
class UpdateDatasetInput:
    """Input for updating a dataset."""
    
    name: Optional[str] = None
    description: Optional[str] = None


@strawberry.input
class CreateDetectorInput:
    """Input for creating a detector."""
    
    name: str
    algorithm: DetectorAlgorithm
    dataset_id: str
    parameters: Optional[str] = None  # JSON string


@strawberry.input
class UpdateDetectorInput:
    """Input for updating a detector."""
    
    name: Optional[str] = None
    parameters: Optional[str] = None


@strawberry.input
class DetectionInput:
    """Input for anomaly detection."""
    
    data: Optional[str] = None  # JSON string
    dataset_id: Optional[str] = None


@strawberry.input
class LoginInput:
    """Input for user login."""
    
    username: str
    password: str


# Union types for responses

@strawberry.type
class DatasetResponse:
    """Response type for dataset operations."""
    
    dataset: Optional[Dataset] = None
    error: Optional[ApiError] = None


@strawberry.type
class DetectorResponse:
    """Response type for detector operations."""
    
    detector: Optional[Detector] = None
    error: Optional[ApiError] = None


@strawberry.type
class ModelResponse:
    """Response type for model operations."""
    
    model: Optional[Model] = None
    error: Optional[ApiError] = None


@strawberry.type
class TrainingResponse:
    """Response type for training operations."""
    
    training_result: Optional[TrainingResult] = None
    error: Optional[ApiError] = None


@strawberry.type
class DetectionResponse:
    """Response type for detection operations."""
    
    detection_result: Optional[AnomalyDetectionResult] = None
    error: Optional[ApiError] = None
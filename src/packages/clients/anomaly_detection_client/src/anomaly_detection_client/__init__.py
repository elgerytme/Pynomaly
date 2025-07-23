"""Official Python client for platform anomaly detection service."""

from anomaly_detection_client.client import AnomalyDetectionClient, AnomalyDetectionSyncClient
from anomaly_detection_client.models import (
    DetectionRequest,
    DetectionResponse,
    EnsembleDetectionRequest,
    EnsembleDetectionResponse,
    ModelInfo,
    TrainingRequest,
    TrainingResponse,
    PredictionRequest,
    PredictionResponse,
    AlgorithmInfo,
)
from sdk_core import (
    ClientConfig,
    Environment,
    # Exceptions
    SDKError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
    ServerError,
)

__version__ = "0.1.0"
__all__ = [
    # Clients
    "AnomalyDetectionClient",
    "AnomalyDetectionSyncClient",
    # Models
    "DetectionRequest",
    "DetectionResponse", 
    "EnsembleDetectionRequest",
    "EnsembleDetectionResponse",
    "ModelInfo",
    "TrainingRequest",
    "TrainingResponse",
    "PredictionRequest", 
    "PredictionResponse",
    "AlgorithmInfo",
    # Configuration
    "ClientConfig",
    "Environment",
    # Exceptions
    "SDKError",
    "AuthenticationError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
]
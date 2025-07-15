"""
Pynomaly SDK - Data Science Client Library

A comprehensive Python SDK for interacting with Pynomaly anomaly detection services.
Provides high-level APIs for data scientists and ML engineers.
"""

from .client import PynomalyClient
from .data_science import DataScienceAPI
from .models import (
    DetectorConfig,
    Dataset,
    DetectionResult,
    ExperimentConfig,
    ModelMetrics,
    TrainingJob
)
from .exceptions import (
    PynomalySDKError,
    AuthenticationError,
    ValidationError,
    APIError
)

__version__ = "0.1.0"
__all__ = [
    # Core client
    "PynomalyClient",
    
    # API interfaces
    "DataScienceAPI",
    
    # Data models
    "DetectorConfig",
    "Dataset", 
    "DetectionResult",
    "ExperimentConfig",
    "ModelMetrics",
    "TrainingJob",
    
    # Exceptions
    "PynomalySDKError",
    "AuthenticationError", 
    "ValidationError",
    "APIError",
]
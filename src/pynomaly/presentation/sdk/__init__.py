"""
Pynomaly Python SDK

A comprehensive Python SDK for programmatic access to the Pynomaly anomaly detection platform.
Provides both synchronous and asynchronous interfaces for all major operations.

Usage:
    from pynomaly.presentation.sdk import PynomaliClient, AsyncPynomaliClient

    # Synchronous client
    client = PynomaliClient(base_url="http://localhost:8000", api_key="your-api-key")

    # Asynchronous client
    async_client = AsyncPynomaliClient(base_url="http://localhost:8000", api_key="your-api-key")
"""

from .client import PynomaliClient

# Optional async client (requires aiohttp)
try:
    from .async_client import AsyncPynomaliClient
except ImportError:
    AsyncPynomaliClient = None
from .config import ClientConfig, SDKConfig
from .exceptions import (
    AuthenticationError,
    PynomaliSDKError,
    ResourceNotFoundError,
    ServerError,
    TimeoutError,
    ValidationError,
)
from .models import (
    AnomalyScore,
    Dataset,
    DetectionResult,
    Detector,
    ExperimentResult,
    PerformanceMetrics,
    TrainingJob,
)

__version__ = "1.0.0"
__author__ = "Pynomali Team"

__all__ = [
    # Main client classes
    "PynomaliClient",
    "AsyncPynomaliClient",
    # Configuration
    "SDKConfig",
    "ClientConfig",
    # Exceptions
    "PynomaliSDKError",
    "AuthenticationError",
    "ValidationError",
    "ResourceNotFoundError",
    "ServerError",
    "TimeoutError",
    # Models
    "Dataset",
    "Detector",
    "DetectionResult",
    "TrainingJob",
    "ExperimentResult",
    "AnomalyScore",
    "PerformanceMetrics",
]

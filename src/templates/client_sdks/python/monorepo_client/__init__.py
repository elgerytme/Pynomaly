"""
anomaly_detection Python SDK

Official Python client library for the anomaly_detection anomaly detection API.
This SDK provides convenient access to the anomaly detection API with full type support,
authentication handling, error management, and comprehensive documentation.

Features:
- Complete API coverage with type-safe client methods
- JWT and API Key authentication support
- Automatic retry logic with exponential backoff
- Rate limiting and request throttling
- Comprehensive error handling
- Async/await support
- Built-in logging and debugging

Example Usage:
    import asyncio
    from anomaly_detection_client import AnomalyDetectionClient

    async def main():
        # Initialize client
        client = AnomalyDetectionClient(
            base_url="https://api.anomaly_detection.com",
            api_key="your-api-key"
        )

        # Authenticate (if using JWT)
        # await client.auth.login("username", "password")

        # Detect anomalies
        result = await client.detection.detect(
            data=[1.0, 2.0, 3.0, 100.0, 4.0, 5.0],
            algorithm="isolation_forest",
            parameters={"contamination": 0.1}
        )

        print(f"Anomalies detected: {result.anomalies}")

        # Clean up
        await client.close()

    asyncio.run(main())
"""

__version__ = "1.0.0"
__author__ = "Anomaly Detection Team"
__email__ = "support@anomaly_detection.com"
__license__ = "MIT"

from .client import AsyncAnomalyDetectionClient, AnomalyDetectionClient
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    NetworkError,
    anomaly-detectionException,
    RateLimitError,
    ServerError,
    ValidationError,
)
from .models import (
    AuthToken,
    DatasetInfo,
    DetectionRequest,
    DetectionResponse,
    ModelInfo,
    TrainingRequest,
    TrainingResponse,
    User,
)

__all__ = [
    # Clients
    "AnomalyDetectionClient",
    "AsyncAnomalyDetectionClient",
    # Exceptions
    "anomaly-detectionException",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "ServerError",
    "NetworkError",
    "RateLimitError",
    # Models
    "DetectionRequest",
    "DetectionResponse",
    "TrainingRequest",
    "TrainingResponse",
    "ModelInfo",
    "DatasetInfo",
    "User",
    "AuthToken",
]

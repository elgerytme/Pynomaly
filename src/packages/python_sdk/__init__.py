"""
Pynomaly Python SDK Package

A comprehensive Python SDK package for the Pynomaly anomaly detection platform.
This package follows clean architecture principles with clear separation of concerns
across domain, application, infrastructure, and presentation layers.

Architecture:
- Domain: Core business entities, value objects, and domain services
- Application: Use cases, application services, and DTOs
- Infrastructure: External adapters, persistence, and integrations
- Presentation: CLI, API, and web interfaces

Usage:
    from python_sdk import PynomaliSDK
    
    sdk = PynomaliSDK(api_key="your-api-key")
    result = sdk.detect_anomalies(data, algorithm="isolation_forest")
"""

__version__ = "1.0.0"
__author__ = "Pynomaly Team"

# Import main SDK components when they are implemented
# from .application.services.sdk_service import PynomaliSDK
# from .domain.entities.detection_result import DetectionResult
# from .domain.value_objects.algorithm_config import AlgorithmConfig

__all__ = [
    # "PynomaliSDK",
    # "DetectionResult", 
    # "AlgorithmConfig"
]
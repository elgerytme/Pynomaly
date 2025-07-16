"""
Pynomaly Python SDK - Data Science Client Library

A comprehensive Python SDK for data science packages with intuitive APIs,
async support, and integration with popular data science libraries.
"""

from .client import PynomaliClient
from .application.services.detection_service import DetectionService
from .application.dto.detection_dto import DetectionRequestDTO, DetectionResponseDTO
from .domain.entities.detection_request import DetectionRequest
from .domain.value_objects.algorithm_config import AlgorithmConfig
from .domain.value_objects.detection_metadata import DetectionMetadata
from .infrastructure.adapters.pyod_algorithm_adapter import PyODAlgorithmAdapter
from .presentation.cli.detection_cli import DetectionCLI

__version__ = "1.0.0"
__author__ = "Pynomaly Team"
__email__ = "support@monorepo.com"

__all__ = [
    "PynomaliClient",
    "DetectionService",
    "DetectionRequestDTO",
    "DetectionResponseDTO", 
    "DetectionRequest",
    "AlgorithmConfig",
    "DetectionMetadata",
    "PyODAlgorithmAdapter",
    "DetectionCLI",
]

# SDK Configuration
DEFAULT_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "backoff_factor": 2,
    "verify_ssl": True,
    "connection_pool_size": 10,
}
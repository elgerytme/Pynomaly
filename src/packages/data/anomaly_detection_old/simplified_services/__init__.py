"""Simplified service layer for anomaly detection.

This module provides a clean, simplified service architecture that replaces
the over-engineered 126 service files with just the essential services needed
for production anomaly detection.
"""

from .core_detection_service import CoreDetectionService
from .automl_service import AutoMLService  
from .ensemble_service import EnsembleService
from .explainability_service import ExplainabilityService

__all__ = [
    "CoreDetectionService",
    "AutoMLService", 
    "EnsembleService",
    "ExplainabilityService",
]
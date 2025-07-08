"""Model drift detection domain entities (legacy module).

This module now imports from the main drift_detection module.
All entities have been consolidated into drift_detection.py.
"""

from __future__ import annotations

# Import all entities from the main drift_detection module
from .drift_detection import (
    DriftConfiguration,
    DriftDetectionMethod,
    DriftMonitor,
    DriftReport,
    DriftSeverity,
    DriftType,
    FeatureDrift,
)

# Legacy exports for backward compatibility
__all__ = [
    "DriftType",
    "DriftSeverity",
    "DriftDetectionMethod",
    "FeatureDrift",
    "DriftConfiguration",
    "DriftReport",
    "DriftMonitor",
]

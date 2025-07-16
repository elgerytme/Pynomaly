"""Domain Value Objects

Immutable value objects that represent concepts with value semantics.
"""

from .semantic_version import SemanticVersion
from .model_metrics import ModelMetrics
from .scaling_config import ScalingConfig

__all__ = [
    "SemanticVersion",
    "ModelMetrics", 
    "ScalingConfig",
]
"""
MLOps Package - ML Operations and Lifecycle Management

This package provides comprehensive MLOps capabilities including:
- Model versioning and registry
- Experiment tracking
- Model deployment and serving
- Monitoring and observability
- Pipeline orchestration
- Model governance
"""

__version__ = "0.1.0"
__author__ = "Anomaly Detection Team"
__email__ = "support@anomaly_detection.com"

# Core imports
from .core import (
    Model,
    Experiment,
    Pipeline,
    Deployment,
)

# Service imports
from .services import (
    ModelRegistryService,
    ExperimentTrackingService,
    DeploymentService,
    MonitoringService,
)

# Utilities
from .utils import (
    logger,
    config,
    metrics,
)

__all__ = [
    # Core entities
    "Model",
    "Experiment", 
    "Pipeline",
    "Deployment",
    
    # Services
    "ModelRegistryService",
    "ExperimentTrackingService",
    "DeploymentService",
    "MonitoringService",
    
    # Utilities
    "logger",
    "config",
    "metrics",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]
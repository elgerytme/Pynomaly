"""MLOps Package - ML Operations and Lifecycle Management.

This package provides comprehensive MLOps capabilities following DDD architecture:
- Model versioning and registry
- Experiment tracking and management
- Model deployment and serving
- Pipeline orchestration
- Model governance and compliance
"""

__version__ = "0.1.0"
__author__ = "AI Team"
__email__ = "ai-team@company.com"

# Domain layer exports
from .domain.entities.model import Model
from .domain.entities.model_version import ModelVersion
from .domain.entities.experiment import Experiment
from .domain.entities.pipeline import Pipeline
from .domain.entities.deployment import Deployment

# Domain services - using try/except to handle missing dependencies gracefully
try:
    from .domain.services.model_management_service import ModelManagementService
except ImportError:
    ModelManagementService = None

try:
    from .domain.services.experiment_tracking_service import ExperimentTrackingService
except ImportError:
    ExperimentTrackingService = None

# Application layer exports - using try/except for graceful degradation
try:
    from .application.use_cases.create_model_use_case import CreateModelUseCase
except ImportError:
    CreateModelUseCase = None

try:
    from .application.services.ml_lifecycle_service import MLLifecycleService  
except ImportError:
    MLLifecycleService = None

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    # Domain entities
    "Model",
    "ModelVersion",
    "Experiment", 
    "Pipeline",
    "Deployment",
    # Domain services (may be None if imports fail)
    "ModelManagementService",
    "ExperimentTrackingService",
    # Application use cases (may be None if imports fail)
    "CreateModelUseCase",
    # Application services (may be None if imports fail)
    "MLLifecycleService",
]
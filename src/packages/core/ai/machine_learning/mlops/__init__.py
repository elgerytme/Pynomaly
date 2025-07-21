"""Core MLOps package - domain logic only."""

__version__ = "0.3.0"

# Core MLOps domain functionality
try:
    from .src.mlops.domain.entities.model import Model, ModelType, ModelStage
    from .services.experiment_tracking_service import ExperimentTrackingService
    from .services.model_registry_service import ModelRegistryService
    
    __all__ = [
        "__version__",
        # Core entities
        "Model", "ModelType", "ModelStage",
        # Core services
        "ExperimentTrackingService", "ModelRegistryService",
    ]
except ImportError:
    # Graceful degradation if dependencies not available
    __all__ = ["__version__"]

# Note: Enterprise features and platform integrations have been moved to:
# - Enterprise services: packages/enterprise/
# - Platform integrations: packages/integrations/
# - Configuration compositions: packages/configurations/
#
# Use configuration packages to compose complete MLOps solutions:
#   from configurations.basic.mlops_basic import create_basic_mlops_config
#   from configurations.enterprise.mlops_enterprise import create_enterprise_mlops_config

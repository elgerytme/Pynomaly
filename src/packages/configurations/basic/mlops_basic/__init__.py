"""Basic open-source MLOps configuration."""

from pathlib import Path
from typing import Optional

# Use shared interfaces instead of direct cross-package imports
from shared.interfaces.mlops import ExperimentTrackingInterface, ModelRegistryInterface
from infrastructure.dependency_injection import ServiceRegistry, get_service_registry


class BasicMLOpsConfiguration:
    """
    Basic open-source MLOps configuration.
    
    Provides core MLOps functionality without enterprise features:
    - Local file-based experiment tracking
    - Local model registry
    - Basic monitoring via logging
    - No authentication
    - No multi-tenancy
    """
    
    def __init__(self, data_path: Optional[Path] = None, service_registry: ServiceRegistry = None):
        """Initialize basic MLOps configuration."""
        self.data_path = data_path or Path("./mlops_data")
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.service_registry = service_registry or get_service_registry()
    
    def create_experiment_tracker(self) -> ExperimentTrackingInterface:
        """Create experiment tracking service through dependency injection."""
        return self.service_registry.get(ExperimentTrackingInterface)
    
    def create_model_registry(self) -> ModelRegistryInterface:
        """Create model registry service through dependency injection."""
        return self.service_registry.get(ModelRegistryInterface)
    
    def create_mlops_services(self) -> dict:
        """Create all MLOps services."""
        return {
            "experiment_tracker": self.create_experiment_tracker(),
            "model_registry": self.create_model_registry(),
            "auth": None,  # No authentication in basic config
            "monitoring": None,  # Basic logging only
            "scalability": None,  # Single node only
        }


def create_basic_mlops_config(data_path: Optional[Path] = None) -> BasicMLOpsConfiguration:
    """Factory function for basic MLOps configuration."""
    return BasicMLOpsConfiguration(data_path)


__all__ = [
    "BasicMLOpsConfiguration",
    "create_basic_mlops_config"
]
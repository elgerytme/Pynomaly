"""Basic open-source MLOps configuration."""

from pathlib import Path
from typing import Optional

from ....core.ai.machine_learning.mlops.services.experiment_tracking_service import ExperimentTrackingService
from ....core.ai.machine_learning.mlops.services.model_registry_service import ModelRegistryService


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
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize basic MLOps configuration."""
        self.data_path = data_path or Path("./mlops_data")
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def create_experiment_tracker(self) -> ExperimentTrackingService:
        """Create experiment tracking service."""
        experiments_path = self.data_path / "experiments"
        return ExperimentTrackingService(experiments_path)
    
    def create_model_registry(self) -> ModelRegistryService:
        """Create model registry service."""
        registry_path = self.data_path / "models"
        return ModelRegistryService(registry_path)
    
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
"""ML Lifecycle application service.

Orchestrates the complete ML model lifecycle from training to deployment.
"""

from __future__ import annotations

from typing import Any, Dict
from uuid import UUID

from ...domain.services.model_management_service import ModelManagementService
from ...domain.services.experiment_tracking_service import ExperimentTrackingService
from ...domain.services.model_optimization_service import ModelOptimizationService


class MLLifecycleService:
    """Application service for ML lifecycle management."""
    
    def __init__(
        self,
        model_service: ModelManagementService,
        experiment_service: ExperimentTrackingService,
        optimization_service: ModelOptimizationService,
    ):
        """Initialize the ML lifecycle service.
        
        Args:
            model_service: Domain service for model management
            experiment_service: Domain service for experiment tracking
            optimization_service: Domain service for model optimization
        """
        self.model_service = model_service
        self.experiment_service = experiment_service
        self.optimization_service = optimization_service
    
    async def create_and_train_model(
        self,
        name: str,
        description: str,
        algorithm_config: Dict[str, Any],
        training_data_path: str,
        created_by: str,
    ) -> Dict[str, Any]:
        """Create a new model and start training.
        
        Args:
            name: Model name
            description: Model description
            algorithm_config: Algorithm configuration
            training_data_path: Path to training data
            created_by: User who created the model
            
        Returns:
            Model creation and training results
        """
        # Create experiment
        experiment = await self.experiment_service.create_experiment(
            name=f"{name}_training",
            description=f"Training experiment for {name}",
            created_by=created_by,
        )
        
        # Create model
        model = await self.model_service.create_model(
            name=name,
            description=description,
            model_type="supervised",  # Default
            algorithm_family=algorithm_config.get("algorithm", "unknown"),
            created_by=created_by,
        )
        
        # Start training with optimization
        training_result = await self.optimization_service.optimize_training(
            model_id=model.id,
            experiment_id=experiment.id,
            config=algorithm_config,
            data_path=training_data_path,
        )
        
        return {
            "model": model,
            "experiment": experiment,
            "training_result": training_result,
        }
    
    async def deploy_best_model(
        self,
        model_id: UUID,
        deployment_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deploy the best performing version of a model.
        
        Args:
            model_id: ID of the model to deploy
            deployment_config: Deployment configuration
            
        Returns:
            Deployment results
        """
        # Get best model version
        best_version = await self.model_service.get_best_model_version(model_id)
        
        if not best_version:
            raise ValueError(f"No trained versions found for model {model_id}")
        
        # Deploy model
        deployment_result = await self.model_service.deploy_model_version(
            model_version_id=best_version.id,
            config=deployment_config,
        )
        
        return {
            "model_version": best_version,
            "deployment": deployment_result,
        }
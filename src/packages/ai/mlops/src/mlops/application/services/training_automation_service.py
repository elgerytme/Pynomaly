"""Training automation service for MLOps."""

from typing import List, Dict, Any, Optional
from uuid import UUID

from ...domain.entities.model import Model
from ...domain.entities.experiment import Experiment
from ...domain.services.experiment_tracking_service import ExperimentTrackingService
from ...domain.services.model_management_service import ModelManagementService
from ...domain.services.model_optimization_service import ModelOptimizationService


class TrainingAutomationService:
    """Application service for automated training workflows."""
    
    def __init__(
        self,
        experiment_service: ExperimentTrackingService,
        model_service: ModelManagementService,
        optimization_service: ModelOptimizationService,
    ):
        self.experiment_service = experiment_service
        self.model_service = model_service
        self.optimization_service = optimization_service
    
    async def run_automated_training(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        optimization_config: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """Run automated training with optimization."""
        # Create experiment
        experiment = await self.experiment_service.create_experiment(
            name=training_config.get("experiment_name", "automated_training"),
            description="Automated training workflow",
            parameters=training_config,
        )
        
        # Create and train model
        model = await self.model_service.create_model(
            name=model_config["name"],
            model_type=model_config["type"],
            algorithm_family=model_config.get("algorithm_family", "supervised"),
            description=model_config.get("description", ""),
            created_by=training_config.get("created_by", "system"),
        )
        
        # Run optimization if configured
        if optimization_config:
            await self.optimization_service.optimize_hyperparameters(
                model_id=model.id,
                optimization_config=optimization_config,
            )
        
        # Update experiment with results
        await self.experiment_service.log_metrics(
            experiment_id=experiment.id,
            metrics={"model_id": str(model.id), "status": "completed"}
        )
        
        return experiment
    
    async def schedule_retraining(
        self,
        model_id: UUID,
        trigger_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Schedule model retraining based on triggers."""
        model = await self.model_service.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Create retraining schedule
        schedule_config = {
            "model_id": str(model_id),
            "trigger_type": trigger_config.get("type", "performance_degradation"),
            "threshold": trigger_config.get("threshold", 0.05),
            "schedule": trigger_config.get("schedule", "weekly"),
            "enabled": True,
        }
        
        return schedule_config
    
    async def run_batch_training(
        self,
        training_jobs: List[Dict[str, Any]],
    ) -> List[Experiment]:
        """Run multiple training jobs in batch."""
        experiments = []
        
        for job_config in training_jobs:
            experiment = await self.run_automated_training(
                model_config=job_config["model_config"],
                training_config=job_config["training_config"],
                optimization_config=job_config.get("optimization_config"),
            )
            experiments.append(experiment)
        
        return experiments
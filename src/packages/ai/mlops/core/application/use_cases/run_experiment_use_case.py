"""Run experiment use case."""

from typing import Dict, Any, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field
import time

from ...domain.entities.experiment import Experiment
from ...domain.services.experiment_tracking_service import ExperimentTrackingService
from ...domain.services.model_management_service import ModelManagementService
from ...domain.services.model_optimization_service import ModelOptimizationService


class RunExperimentRequest(BaseModel):
    """Request for running an experiment."""
    experiment_name: str
    description: str = ""
    model_configs: List[Dict[str, Any]]
    training_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    evaluation_metrics: List[str] = Field(default_factory=lambda: ["accuracy", "f1_score"])
    cross_validation_folds: int = Field(default=5, ge=2, le=10)
    optimization_enabled: bool = False
    optimization_config: Optional[Dict[str, Any]] = None
    parallel_execution: bool = False
    max_parallel_jobs: int = Field(default=2, ge=1, le=10)


class ExperimentResult(BaseModel):
    """Result for a single model in the experiment."""
    model_name: str
    model_config: Dict[str, Any]
    metrics: Dict[str, float]
    training_time_seconds: float
    model_path: str
    status: str


class RunExperimentResponse(BaseModel):
    """Response for experiment execution."""
    experiment_id: UUID
    experiment_name: str
    status: str
    results: List[ExperimentResult]
    best_model: str
    best_metrics: Dict[str, float]
    total_execution_time_seconds: float
    summary: Dict[str, Any]


class RunExperimentUseCase:
    """Use case for running ML experiments."""
    
    def __init__(
        self,
        experiment_service: ExperimentTrackingService,
        model_service: ModelManagementService,
        optimization_service: ModelOptimizationService,
    ):
        self.experiment_service = experiment_service
        self.model_service = model_service
        self.optimization_service = optimization_service
    
    async def execute(self, request: RunExperimentRequest) -> RunExperimentResponse:
        """Execute experiment with multiple model configurations."""
        start_time = time.time()
        
        # Create experiment
        experiment = await self.experiment_service.create_experiment(
            name=request.experiment_name,
            description=request.description,
            parameters={
                "model_configs": request.model_configs,
                "training_config": request.training_config,
                "dataset_config": request.dataset_config,
                "evaluation_metrics": request.evaluation_metrics,
                "cross_validation_folds": request.cross_validation_folds,
                "optimization_enabled": request.optimization_enabled,
            },
        )
        
        # Execute model training and evaluation
        if request.parallel_execution:
            results = await self._run_parallel_experiments(
                experiment, request
            )
        else:
            results = await self._run_sequential_experiments(
                experiment, request
            )
        
        # Determine best model
        best_model, best_metrics = self._find_best_model(
            results, request.evaluation_metrics
        )
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        # Generate experiment summary
        summary = self._generate_experiment_summary(results, total_time)
        
        # Log experiment completion
        await self.experiment_service.log_metrics(
            experiment_id=experiment.id,
            metrics={
                "best_model": best_model,
                "best_metrics": best_metrics,
                "total_models_tested": len(results),
                "execution_time_seconds": total_time,
            },
        )
        
        return RunExperimentResponse(
            experiment_id=experiment.id,
            experiment_name=request.experiment_name,
            status="completed",
            results=results,
            best_model=best_model,
            best_metrics=best_metrics,
            total_execution_time_seconds=total_time,
            summary=summary,
        )
    
    async def _run_sequential_experiments(
        self,
        experiment: Experiment,
        request: RunExperimentRequest,
    ) -> List[ExperimentResult]:
        """Run experiments sequentially."""
        results = []
        
        for i, model_config in enumerate(request.model_configs):
            result = await self._train_and_evaluate_model(
                experiment, model_config, request, f"model_{i+1}"
            )
            results.append(result)
            
            # Log intermediate result
            await self.experiment_service.log_metrics(
                experiment_id=experiment.id,
                metrics={f"model_{i+1}_metrics": result.metrics},
            )
        
        return results
    
    async def _run_parallel_experiments(
        self,
        experiment: Experiment,
        request: RunExperimentRequest,
    ) -> List[ExperimentResult]:
        """Run experiments in parallel (simplified implementation)."""
        # In a real implementation, this would use asyncio.gather or similar
        # For now, we'll simulate parallel execution with sequential calls
        return await self._run_sequential_experiments(experiment, request)
    
    async def _train_and_evaluate_model(
        self,
        experiment: Experiment,
        model_config: Dict[str, Any],
        request: RunExperimentRequest,
        model_name: str,
    ) -> ExperimentResult:
        """Train and evaluate a single model configuration."""
        start_time = time.time()
        
        # Create model
        model = await self.model_service.create_model(
            name=f"{request.experiment_name}_{model_name}",
            model_type=model_config.get("type", "classifier"),
            algorithm_family=model_config.get("algorithm_family", "supervised"),
            description=f"Model from experiment {request.experiment_name}",
            created_by="experiment_runner",
        )
        
        # Run optimization if enabled
        if request.optimization_enabled and request.optimization_config:
            await self.optimization_service.optimize_hyperparameters(
                model_id=model.id,
                optimization_config=request.optimization_config,
            )
        
        # Simulate training and evaluation
        # In real implementation, this would use actual ML frameworks
        metrics = self._simulate_model_training(model_config, request)
        
        training_time = time.time() - start_time
        
        return ExperimentResult(
            model_name=model_name,
            model_config=model_config,
            metrics=metrics,
            training_time_seconds=training_time,
            model_path=f"/experiments/{experiment.id}/models/{model_name}.pkl",
            status="completed",
        )
    
    def _simulate_model_training(
        self,
        model_config: Dict[str, Any],
        request: RunExperimentRequest,
    ) -> Dict[str, float]:
        """Simulate model training and return metrics."""
        import random
        
        # Generate realistic but random metrics
        base_accuracy = 0.85 + random.uniform(-0.1, 0.1)
        metrics = {
            "accuracy": max(0.0, min(1.0, base_accuracy)),
            "precision": max(0.0, min(1.0, base_accuracy + random.uniform(-0.05, 0.05))),
            "recall": max(0.0, min(1.0, base_accuracy + random.uniform(-0.05, 0.05))),
            "f1_score": max(0.0, min(1.0, base_accuracy + random.uniform(-0.03, 0.03))),
        }
        
        # Only return requested metrics
        return {
            metric: metrics.get(metric, 0.0)
            for metric in request.evaluation_metrics
            if metric in metrics
        }
    
    def _find_best_model(
        self,
        results: List[ExperimentResult],
        evaluation_metrics: List[str],
    ) -> tuple[str, Dict[str, float]]:
        """Find the best performing model based on evaluation metrics."""
        if not results:
            return "", {}
        
        # Use first metric as primary sorting criterion
        primary_metric = evaluation_metrics[0] if evaluation_metrics else "accuracy"
        
        best_result = max(
            results,
            key=lambda r: r.metrics.get(primary_metric, 0.0)
        )
        
        return best_result.model_name, best_result.metrics
    
    def _generate_experiment_summary(
        self,
        results: List[ExperimentResult],
        total_time: float,
    ) -> Dict[str, Any]:
        """Generate summary statistics for the experiment."""
        if not results:
            return {}
        
        # Calculate average metrics
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        avg_metrics = {}
        for metric in all_metrics:
            values = [r.metrics.get(metric, 0.0) for r in results]
            avg_metrics[f"avg_{metric}"] = sum(values) / len(values)
        
        return {
            "total_models_tested": len(results),
            "total_execution_time_seconds": total_time,
            "average_training_time_seconds": sum(r.training_time_seconds for r in results) / len(results),
            "successful_models": len([r for r in results if r.status == "completed"]),
            "failed_models": len([r for r in results if r.status != "completed"]),
            **avg_metrics,
        }
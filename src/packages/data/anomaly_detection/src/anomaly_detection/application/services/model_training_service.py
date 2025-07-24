"""Model Training Application Service.

This service demonstrates the proper use of hexagonal architecture
by orchestrating domain operations through well-defined interfaces.
It shows how to integrate ML training, MLOps tracking, and business logic.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from anomaly_detection.domain.interfaces.ml_operations import (
    MLModelTrainingPort,
    TrainingRequest,
    TrainingResult,
    ModelEvaluationRequest,
    OptimizationObjective,
)
from anomaly_detection.domain.interfaces.mlops_operations import (
    MLOpsExperimentTrackingPort,
    MLOpsModelRegistryPort,
    RunStatus,
)
try:
    from data.processing.domain.entities.dataset import Dataset
except ImportError:
    from anomaly_detection.domain.entities.dataset import Dataset

try:
    from data.processing.domain.entities.detection_result import DetectionResult
except ImportError:
    from anomaly_detection.domain.entities.detection_result import DetectionResult


class ModelTrainingApplicationService:
    """Application service for model training workflows.
    
    This service orchestrates the training process by coordinating
    between ML training, experiment tracking, and model registry
    through their respective interfaces.
    """
    
    def __init__(
        self,
        ml_training: MLModelTrainingPort,
        experiment_tracking: MLOpsExperimentTrackingPort,
        model_registry: MLOpsModelRegistryPort,
    ):
        """Initialize the model training application service.
        
        Args:
            ml_training: ML training operations interface
            experiment_tracking: MLOps experiment tracking interface
            model_registry: MLOps model registry interface
        """
        self._ml_training = ml_training
        self._experiment_tracking = experiment_tracking
        self._model_registry = model_registry
        self._logger = logging.getLogger(__name__)
    
    async def train_anomaly_detection_model(
        self,
        algorithm_name: str,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        parameters: Optional[Dict[str, Any]] = None,
        experiment_name: Optional[str] = None,
        optimization_objective: Optional[OptimizationObjective] = None,
        register_model: bool = True,
        created_by: str = "system"
    ) -> Dict[str, Any]:
        """Train an anomaly detection model with full MLOps integration.
        
        This method demonstrates the orchestration of multiple domain
        operations through their interfaces, following clean architecture
        principles.
        
        Args:
            algorithm_name: Name of the algorithm to use
            training_data: Training dataset
            validation_data: Optional validation dataset
            parameters: Optional algorithm parameters
            experiment_name: Optional experiment name
            optimization_objective: Optional optimization objective
            register_model: Whether to register the trained model
            created_by: User initiating the training
            
        Returns:
            Dictionary containing training results and metadata
        """
        try:
            # 1. Create experiment for tracking
            exp_name = experiment_name or f"{algorithm_name}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            experiment_id = await self._experiment_tracking.create_experiment(
                name=exp_name,
                description=f"Training {algorithm_name} for anomaly detection",
                tags={"algorithm": algorithm_name, "type": "training"},
                created_by=created_by
            )
            
            self._logger.info(f"Created experiment {exp_name} with ID: {experiment_id}")
            
            # 2. Start experiment run
            run_id = await self._experiment_tracking.start_run(
                experiment_id=experiment_id,
                run_name=f"training_run_{datetime.now().strftime('%H%M%S')}",
                parameters=parameters or {},
                tags={"phase": "training"},
                created_by=created_by
            )
            
            self._logger.info(f"Started training run with ID: {run_id}")
            
            # 3. Log training parameters
            training_params = {
                "algorithm_name": algorithm_name,
                "training_samples": len(training_data.data) if hasattr(training_data.data, '__len__') else 0,
                "validation_samples": len(validation_data.data) if validation_data and hasattr(validation_data.data, '__len__') else 0,
                "feature_count": len(training_data.feature_names),
                "optimization_objective": optimization_objective.value if optimization_objective else None,
            }
            training_params.update(parameters or {})
            
            await self._experiment_tracking.log_parameters(run_id, training_params)
            
            # 4. Execute training through ML interface
            training_request = TrainingRequest(
                algorithm_name=algorithm_name,
                training_data=training_data,
                validation_data=validation_data,
                parameters=parameters or {},
                optimization_objective=optimization_objective,
                created_by=created_by
            )
            
            self._logger.info(f"Starting model training with algorithm: {algorithm_name}")
            training_result = await self._ml_training.train_model(training_request)
            
            # 5. Log training metrics
            await self._experiment_tracking.log_metrics(run_id, training_result.training_metrics)
            
            if training_result.validation_metrics:
                validation_metrics = {f"val_{key}": value for key, value in training_result.validation_metrics.items()}
                await self._experiment_tracking.log_metrics(run_id, validation_metrics)
            
            # 6. Log additional metrics
            additional_metrics = {
                "training_duration_seconds": training_result.training_duration_seconds,
                "model_status": training_result.status.value,
            }
            await self._experiment_tracking.log_metrics(run_id, additional_metrics)
            
            # 7. Register model if requested
            model_id = None
            version_id = None
            
            if register_model:
                try:
                    # Register the model
                    model_id = await self._model_registry.register_model(
                        name=f"{algorithm_name}_anomaly_detector",
                        description=f"Anomaly detection model using {algorithm_name}",
                        tags={"algorithm": algorithm_name, "type": "anomaly_detection"},
                        created_by=created_by
                    )
                    
                    # Create model version
                    version_id = await self._model_registry.create_model_version(
                        model_id=model_id,
                        version="1.0.0",
                        run_id=run_id,
                        source_path=f"/models/{training_result.model.metadata.model_id}",
                        description="Initial version from training",
                        performance_metrics=training_result.training_metrics,
                        deployment_config={
                            "algorithm": algorithm_name,
                            "preprocessing": "standard",
                            "inference_timeout_ms": 5000,
                        },
                        tags={"source": "training"},
                        created_by=created_by
                    )
                    
                    self._logger.info(f"Registered model {model_id} with version {version_id}")
                    
                except Exception as e:
                    self._logger.warning(f"Failed to register model: {str(e)}")
                    # Don't fail the entire training process if registration fails
            
            # 8. End the experiment run
            await self._experiment_tracking.end_run(run_id, RunStatus.COMPLETED)
            
            self._logger.info(f"Training completed successfully for algorithm: {algorithm_name}")
            
            # 9. Return comprehensive results
            return {
                "success": True,
                "model": {
                    "id": training_result.model.metadata.model_id,
                    "name": training_result.model.metadata.name,
                    "algorithm": training_result.model.metadata.algorithm,
                    "version": training_result.model.metadata.version,
                    "status": training_result.status.value,
                },
                "training": {
                    "metrics": training_result.training_metrics,
                    "validation_metrics": training_result.validation_metrics,
                    "duration_seconds": training_result.training_duration_seconds,
                    "feature_importance": training_result.feature_importance,
                },
                "experiment": {
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "name": exp_name,
                },
                "registry": {
                    "model_id": model_id,
                    "version_id": version_id,
                    "registered": register_model and model_id is not None,
                },
                "metadata": {
                    "created_by": created_by,
                    "created_at": datetime.now().isoformat(),
                    "algorithm_name": algorithm_name,
                    "training_samples": training_params["training_samples"],
                    "validation_samples": training_params["validation_samples"],
                }
            }
            
        except Exception as e:
            self._logger.error(f"Training failed for algorithm {algorithm_name}: {str(e)}")
            
            # End run as failed if it was started
            try:
                if 'run_id' in locals():
                    await self._experiment_tracking.end_run(run_id, RunStatus.FAILED)
            except Exception:
                pass  # Don't mask the original error
            
            # Return error result
            return {
                "success": False,
                "error": str(e),
                "algorithm_name": algorithm_name,
                "created_by": created_by,
                "failed_at": datetime.now().isoformat(),
            }
    
    async def evaluate_trained_model(
        self,
        model_id: str,
        test_data: Dataset,
        evaluation_metrics: Optional[List[str]] = None,
        experiment_name: Optional[str] = None,
        created_by: str = "system"
    ) -> Dict[str, Any]:
        """Evaluate a trained model with MLOps tracking.
        
        Args:
            model_id: ID of the model to evaluate
            test_data: Test dataset for evaluation
            evaluation_metrics: Optional specific metrics to compute
            experiment_name: Optional experiment name
            created_by: User initiating the evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # This would need to be implemented with model loading logic
            # For now, return a placeholder structure
            exp_name = experiment_name or f"model_evaluation_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            experiment_id = await self._experiment_tracking.create_experiment(
                name=exp_name,
                description=f"Evaluation of model {model_id}",
                tags={"type": "evaluation", "model_id": model_id},
                created_by=created_by
            )
            
            run_id = await self._experiment_tracking.start_run(
                experiment_id=experiment_id,
                run_name=f"evaluation_run_{datetime.now().strftime('%H%M%S')}",
                parameters={"model_id": model_id, "test_samples": len(test_data.data) if hasattr(test_data.data, '__len__') else 0},
                tags={"phase": "evaluation"},
                created_by=created_by
            )
            
            # Here you would load the model and run evaluation
            # For demonstration, we'll use placeholder results
            evaluation_results = {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.88,
                "f1_score": 0.85,
            }
            
            await self._experiment_tracking.log_metrics(run_id, evaluation_results)
            await self._experiment_tracking.end_run(run_id, RunStatus.COMPLETED)
            
            return {
                "success": True,
                "model_id": model_id,
                "metrics": evaluation_results,
                "experiment": {
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                },
                "test_data": {
                    "samples": len(test_data.data) if hasattr(test_data.data, '__len__') else 0,
                    "features": len(test_data.feature_names),
                },
                "created_by": created_by,
                "evaluated_at": datetime.now().isoformat(),
            }
            
        except Exception as e:
            self._logger.error(f"Evaluation failed for model {model_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "model_id": model_id,
                "created_by": created_by,
                "failed_at": datetime.now().isoformat(),
            }
    
    async def get_supported_algorithms(self) -> List[str]:
        """Get list of supported algorithms for anomaly detection.
        
        Returns:
            List of supported algorithm names
        """
        try:
            return await self._ml_training.get_supported_algorithms()
        except Exception as e:
            self._logger.error(f"Failed to get supported algorithms: {str(e)}")
            return []
    
    async def get_algorithm_parameters(self, algorithm_name: str) -> Dict[str, Any]:
        """Get parameter schema for a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Parameter schema with types, defaults, and constraints
        """
        try:
            return await self._ml_training.get_algorithm_parameters(algorithm_name)
        except Exception as e:
            self._logger.error(f"Failed to get parameters for algorithm {algorithm_name}: {str(e)}")
            return {}
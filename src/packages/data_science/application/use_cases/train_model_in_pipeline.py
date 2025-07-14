"""Use case for training ML models within pipelines."""

from typing import Any, Dict, Optional
from uuid import uuid4
from datetime import datetime

from ..dto.ml_pipeline_dto import (
    TrainModelRequestDTO,
    TrainModelResponseDTO
)
from ...domain.entities.data_science_model import DataScienceModel, ModelType, ModelStatus
from ...domain.entities.machine_learning_pipeline import MachineLearningPipeline
from ...domain.repositories.data_science_model_repository import IDataScienceModelRepository
from ...domain.repositories.machine_learning_pipeline_repository import IMachineLearningPipelineRepository
from ...domain.value_objects.model_performance_metrics import ModelPerformanceMetrics, ModelTask


class TrainModelInPipelineUseCase:
    """Use case for training ML models within pipeline context."""
    
    def __init__(
        self,
        model_repository: IDataScienceModelRepository,
        pipeline_repository: IMachineLearningPipelineRepository
    ):
        self._model_repository = model_repository
        self._pipeline_repository = pipeline_repository
    
    async def execute(self, request: TrainModelRequestDTO) -> TrainModelResponseDTO:
        """Execute model training within pipeline use case.
        
        Args:
            request: Model training request parameters
            
        Returns:
            Model training response with training results
            
        Raises:
            ModelTrainingError: If model training fails
        """
        try:
            # Get pipeline context
            pipeline = await self._pipeline_repository.get_by_id(request.pipeline_id)
            if not pipeline:
                raise ValueError(f"Pipeline not found: {request.pipeline_id}")
            
            # Create model entity
            model = await self._create_model_entity(request, pipeline)
            
            # Start training
            model.start_training()
            await self._model_repository.save(model)
            
            # Log training start in pipeline
            await self._pipeline_repository.add_execution_log(
                request.pipeline_id,
                f"Started training model: {model.name}",
                level="INFO"
            )
            
            # Perform training
            training_results = await self._train_model(
                model, 
                request.training_data,
                request.validation_data,
                request.hyperparameters or {}
            )
            
            # Update model with training results
            performance_metrics = self._create_performance_metrics(training_results)
            model.complete_training(performance_metrics)
            
            # Update model artifacts
            model.update_artifacts(training_results.get("artifacts", {}))
            
            await self._model_repository.save(model)
            
            # Log training completion in pipeline
            await self._pipeline_repository.add_execution_log(
                request.pipeline_id,
                f"Completed training model: {model.name} (accuracy: {training_results.get('accuracy', 'N/A')})",
                level="INFO"
            )
            
            return TrainModelResponseDTO(
                model_id=model.model_id.value,
                pipeline_id=request.pipeline_id,
                training_status="completed",
                performance_metrics=training_results.get("metrics", {}),
                model_artifacts=training_results.get("artifacts", {}),
                training_time_seconds=model.training_duration_seconds
            )
            
        except Exception as e:
            # Mark model as failed if it exists
            if 'model' in locals():
                model.fail_training(str(e))
                await self._model_repository.save(model)
                
                # Log training failure in pipeline
                await self._pipeline_repository.add_execution_log(
                    request.pipeline_id,
                    f"Failed training model: {model.name} - {str(e)}",
                    level="ERROR"
                )
            
            return TrainModelResponseDTO(
                model_id=model.model_id.value if 'model' in locals() else uuid4(),
                pipeline_id=request.pipeline_id,
                training_status="failed",
                performance_metrics={},
                model_artifacts={},
                training_time_seconds=None
            )
    
    async def _create_model_entity(self, request: TrainModelRequestDTO, pipeline: MachineLearningPipeline) -> DataScienceModel:
        """Create model entity from training request."""
        model_config = request.model_config
        
        # Determine model type
        model_type = self._map_model_type(model_config.get("algorithm", "unknown"))
        
        # Generate model name if not provided
        model_name = model_config.get("name", f"model_{pipeline.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
        
        model = DataScienceModel(
            name=model_name,
            model_type=model_type,
            algorithm=model_config.get("algorithm", "unknown"),
            version_number="1.0.0",
            status=ModelStatus.TRAINING,
            description=model_config.get("description", f"Model trained in pipeline {pipeline.name}"),
            hyperparameters=request.hyperparameters or {},
            features=model_config.get("features", []),
            target_variable=model_config.get("target_variable"),
            training_dataset_info={
                "pipeline_id": str(request.pipeline_id),
                "training_config": model_config
            },
            business_context={
                "purpose": model_config.get("purpose", "Machine learning model"),
                "use_case": model_config.get("use_case", "Prediction"),
                "stakeholders": model_config.get("stakeholders", [])
            }
        )
        
        return model
    
    async def _train_model(
        self,
        model: DataScienceModel,
        training_data: Any,
        validation_data: Optional[Any],
        hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train the model with given data and hyperparameters."""
        # Mock training implementation
        # In real implementation, this would:
        # 1. Load the appropriate ML algorithm
        # 2. Prepare data for training
        # 3. Train the model with hyperparameters
        # 4. Validate on validation data
        # 5. Return training results and artifacts
        
        import asyncio
        import random
        
        # Simulate training time
        training_time = random.uniform(10, 30)
        await asyncio.sleep(min(training_time, 2))  # Cap sleep for demo
        
        # Mock training results
        accuracy = random.uniform(0.7, 0.95)
        precision = random.uniform(0.7, 0.95)
        recall = random.uniform(0.7, 0.95)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "training_loss": random.uniform(0.1, 0.5),
            "validation_loss": random.uniform(0.1, 0.5),
            "training_time_seconds": training_time,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "auc_roc": random.uniform(0.8, 0.98)
            },
            "artifacts": {
                "model_file": f"model_{model.name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl",
                "feature_importance": f"feature_importance_{model.name}.json",
                "training_metrics": f"training_metrics_{model.name}.json"
            }
        }
    
    def _create_performance_metrics(self, training_results: Dict[str, Any]) -> ModelPerformanceMetrics:
        """Create performance metrics from training results."""
        return ModelPerformanceMetrics(
            model_task=ModelTask.CLASSIFICATION,  # Default to classification
            accuracy=training_results.get("accuracy"),
            precision=training_results.get("precision"),
            recall=training_results.get("recall"),
            f1_score=training_results.get("f1_score"),
            auc_roc=training_results.get("metrics", {}).get("auc_roc"),
            log_loss=training_results.get("training_loss"),
            cross_validation_scores=[training_results.get("accuracy", 0.0)],  # Single score for demo
            training_time_seconds=training_results.get("training_time_seconds"),
            model_size_mb=random.uniform(1.0, 50.0),  # Mock model size
            inference_latency_ms=random.uniform(10.0, 100.0),  # Mock latency
            feature_importance_scores={f"feature_{i}": random.uniform(0.0, 1.0) for i in range(5)}
        )
    
    def _map_model_type(self, algorithm: str) -> ModelType:
        """Map algorithm string to model type enum."""
        algorithm_lower = algorithm.lower()
        
        if any(term in algorithm_lower for term in ["tree", "forest", "xgboost", "lightgbm", "catboost"]):
            return ModelType.TREE_BASED
        elif any(term in algorithm_lower for term in ["linear", "logistic", "regression"]):
            return ModelType.LINEAR
        elif any(term in algorithm_lower for term in ["neural", "deep", "cnn", "rnn", "lstm", "transformer"]):
            return ModelType.DEEP_LEARNING
        elif any(term in algorithm_lower for term in ["svm", "support"]):
            return ModelType.KERNEL_BASED
        elif any(term in algorithm_lower for term in ["ensemble", "voting", "bagging", "boosting"]):
            return ModelType.ENSEMBLE
        elif any(term in algorithm_lower for term in ["cluster", "kmeans", "dbscan"]):
            return ModelType.CLUSTERING
        else:
            return ModelType.OTHER
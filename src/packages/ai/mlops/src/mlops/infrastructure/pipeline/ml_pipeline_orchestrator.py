"""
End-to-End ML Pipeline Orchestrator

Provides comprehensive automation for ML workflows including data ingestion,
preprocessing, training, evaluation, deployment, and monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import uuid4
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from mlops.domain.entities.pipeline import (
    Pipeline, PipelineStage, PipelineRun, PipelineStatus,
    StageStatus, PipelineConfig, StageConfig
)
from mlops.application.services.model_registry_service import ModelRegistryService
from mlops.application.services.feature_store_service import FeatureStoreService
from mlops.application.services.experiment_tracking_service import ExperimentTrackingService
from mlops.infrastructure.monitoring.pipeline_monitor import PipelineMonitor


class PipelineType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    RETRAINING = "retraining"
    BATCH_PREDICTION = "batch_prediction"
    DATA_VALIDATION = "data_validation"


class TriggerType(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    DATA_DRIFT = "data_drift"
    MODEL_PERFORMANCE = "model_performance"
    EVENT_DRIVEN = "event_driven"


@dataclass
class StageResult:
    """Result of pipeline stage execution."""
    stage_name: str
    status: StageStatus
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: Optional[str] = None


class PipelineExecutionContext:
    """Context for pipeline execution with shared state."""
    
    def __init__(self, pipeline_id: str, run_id: str):
        self.pipeline_id = pipeline_id
        self.run_id = run_id
        self.shared_data: Dict[str, Any] = {}
        self.artifacts: Dict[str, str] = {}
        self.metrics: Dict[str, float] = {}
        self.created_at = datetime.utcnow()
    
    def set_data(self, key: str, value: Any) -> None:
        """Store data in shared context."""
        self.shared_data[key] = value
    
    def get_data(self, key: str, default: Any = None) -> Any:
        """Retrieve data from shared context."""
        return self.shared_data.get(key, default)
    
    def add_artifact(self, name: str, path: str) -> None:
        """Add artifact path to context."""
        self.artifacts[name] = path
    
    def add_metric(self, name: str, value: float) -> None:
        """Add metric to context."""
        self.metrics[name] = value


class MLPipelineOrchestrator:
    """Orchestrator for end-to-end ML pipelines."""
    
    def __init__(
        self,
        model_registry: ModelRegistryService,
        feature_store: FeatureStoreService,
        experiment_tracker: ExperimentTrackingService,
        pipeline_monitor: PipelineMonitor
    ):
        self.model_registry = model_registry
        self.feature_store = feature_store
        self.experiment_tracker = experiment_tracker
        self.pipeline_monitor = pipeline_monitor
        
        self.logger = logging.getLogger(__name__)
        
        # Pipeline registry
        self.pipelines: Dict[str, Pipeline] = {}
        self.active_runs: Dict[str, PipelineRun] = {}
        
        # Stage executors
        self.stage_executors: Dict[str, Callable] = {
            "data_ingestion": self._execute_data_ingestion,
            "data_validation": self._execute_data_validation,
            "feature_engineering": self._execute_feature_engineering,
            "model_training": self._execute_model_training,
            "model_evaluation": self._execute_model_evaluation,
            "model_validation": self._execute_model_validation,
            "model_deployment": self._execute_model_deployment,
            "batch_prediction": self._execute_batch_prediction,
            "monitoring_setup": self._execute_monitoring_setup,
        }
    
    async def register_pipeline(
        self,
        name: str,
        pipeline_type: PipelineType,
        stages: List[StageConfig],
        config: PipelineConfig,
        description: str = ""
    ) -> str:
        """Register a new ML pipeline."""
        pipeline_id = str(uuid4())
        
        pipeline = Pipeline(
            id=pipeline_id,
            name=name,
            pipeline_type=pipeline_type,
            stages=stages,
            config=config,
            description=description,
            status=PipelineStatus.READY,
            created_at=datetime.utcnow()
        )
        
        self.pipelines[pipeline_id] = pipeline
        
        self.logger.info(f"Registered pipeline '{name}' with ID: {pipeline_id}")
        
        # Register with monitoring
        await self.pipeline_monitor.register_pipeline(pipeline)
        
        return pipeline_id
    
    async def create_training_pipeline(
        self,
        name: str,
        data_source: Dict[str, Any],
        feature_config: Dict[str, Any],
        model_config: Dict[str, Any],
        validation_config: Dict[str, Any] = None,
        deployment_config: Dict[str, Any] = None
    ) -> str:
        """Create a complete training pipeline."""
        
        stages = [
            StageConfig(
                name="data_ingestion",
                stage_type="data_ingestion",
                config=data_source,
                dependencies=[],
                retry_config={"max_retries": 3, "retry_delay": 60}
            ),
            StageConfig(
                name="data_validation",
                stage_type="data_validation",
                config=validation_config or {},
                dependencies=["data_ingestion"],
                retry_config={"max_retries": 2, "retry_delay": 30}
            ),
            StageConfig(
                name="feature_engineering",
                stage_type="feature_engineering",
                config=feature_config,
                dependencies=["data_validation"],
                retry_config={"max_retries": 3, "retry_delay": 60}
            ),
            StageConfig(
                name="model_training",
                stage_type="model_training",
                config=model_config,
                dependencies=["feature_engineering"],
                retry_config={"max_retries": 2, "retry_delay": 120}
            ),
            StageConfig(
                name="model_evaluation",
                stage_type="model_evaluation",
                config={"evaluation_metrics": ["accuracy", "precision", "recall", "f1"]},
                dependencies=["model_training"],
                retry_config={"max_retries": 2, "retry_delay": 60}
            ),
            StageConfig(
                name="model_validation",
                stage_type="model_validation",
                config={"validation_threshold": 0.8},
                dependencies=["model_evaluation"],
                retry_config={"max_retries": 1, "retry_delay": 30}
            )
        ]
        
        # Add deployment stage if config provided
        if deployment_config:
            stages.append(
                StageConfig(
                    name="model_deployment",
                    stage_type="model_deployment",
                    config=deployment_config,
                    dependencies=["model_validation"],
                    retry_config={"max_retries": 3, "retry_delay": 60}
                )
            )
            
            # Add monitoring setup
            stages.append(
                StageConfig(
                    name="monitoring_setup",
                    stage_type="monitoring_setup",
                    config={"enable_drift_detection": True, "alert_thresholds": {}},
                    dependencies=["model_deployment"],
                    retry_config={"max_retries": 2, "retry_delay": 30}
                )
            )
        
        pipeline_config = PipelineConfig(
            max_parallel_stages=2,
            timeout_minutes=480,  # 8 hours
            failure_policy="stop_on_failure",
            notification_config={
                "on_success": True,
                "on_failure": True,
                "channels": ["email", "slack"]
            }
        )
        
        return await self.register_pipeline(
            name=name,
            pipeline_type=PipelineType.TRAINING,
            stages=stages,
            config=pipeline_config,
            description=f"Complete training pipeline for {name}"
        )
    
    async def create_inference_pipeline(
        self,
        name: str,
        model_id: str,
        input_source: Dict[str, Any],
        output_destination: Dict[str, Any],
        batch_size: int = 1000
    ) -> str:
        """Create a batch inference pipeline."""
        
        stages = [
            StageConfig(
                name="data_ingestion",
                stage_type="data_ingestion",
                config=input_source,
                dependencies=[],
                retry_config={"max_retries": 3, "retry_delay": 60}
            ),
            StageConfig(
                name="feature_engineering",
                stage_type="feature_engineering",
                config={"model_id": model_id, "inference_mode": True},
                dependencies=["data_ingestion"],
                retry_config={"max_retries": 3, "retry_delay": 60}
            ),
            StageConfig(
                name="batch_prediction",
                stage_type="batch_prediction",
                config={
                    "model_id": model_id,
                    "batch_size": batch_size,
                    "output_destination": output_destination
                },
                dependencies=["feature_engineering"],
                retry_config={"max_retries": 3, "retry_delay": 120}
            )
        ]
        
        pipeline_config = PipelineConfig(
            max_parallel_stages=1,
            timeout_minutes=240,  # 4 hours
            failure_policy="retry_failed_stages",
            notification_config={
                "on_success": True,
                "on_failure": True,
                "channels": ["email"]
            }
        )
        
        return await self.register_pipeline(
            name=name,
            pipeline_type=PipelineType.INFERENCE,
            stages=stages,
            config=pipeline_config,
            description=f"Batch inference pipeline for model {model_id}"
        )
    
    async def execute_pipeline(
        self,
        pipeline_id: str,
        trigger_type: TriggerType = TriggerType.MANUAL,
        parameters: Dict[str, Any] = None
    ) -> str:
        """Execute a registered pipeline."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self.pipelines[pipeline_id]
        run_id = str(uuid4())
        
        # Create pipeline run
        pipeline_run = PipelineRun(
            id=run_id,
            pipeline_id=pipeline_id,
            trigger_type=trigger_type,
            parameters=parameters or {},
            status=PipelineStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        self.active_runs[run_id] = pipeline_run
        
        self.logger.info(f"Starting pipeline execution: {pipeline.name} (run_id: {run_id})")
        
        # Start monitoring
        await self.pipeline_monitor.start_run_monitoring(pipeline_run)
        
        # Execute pipeline asynchronously
        asyncio.create_task(self._execute_pipeline_async(pipeline, pipeline_run))
        
        return run_id
    
    async def _execute_pipeline_async(self, pipeline: Pipeline, run: PipelineRun):
        """Execute pipeline stages asynchronously."""
        context = PipelineExecutionContext(pipeline.id, run.id)
        
        try:
            # Start experiment tracking
            experiment_run = await self.experiment_tracker.start_run(
                experiment_name=f"{pipeline.name}_pipeline",
                run_name=f"{pipeline.name}_{run.id[:8]}",
                tags={
                    "pipeline_id": pipeline.id,
                    "pipeline_type": pipeline.pipeline_type.value,
                    "run_id": run.id
                }
            )
            
            context.set_data("experiment_run_id", experiment_run.id)
            
            # Execute stages based on dependencies
            completed_stages = set()
            stage_results = {}
            
            while len(completed_stages) < len(pipeline.stages):
                # Find stages ready to execute
                ready_stages = []
                for stage in pipeline.stages:
                    if (stage.name not in completed_stages and 
                        all(dep in completed_stages for dep in stage.dependencies)):
                        ready_stages.append(stage)
                
                if not ready_stages:
                    # Check for circular dependencies or other issues
                    remaining_stages = [s.name for s in pipeline.stages if s.name not in completed_stages]
                    raise RuntimeError(f"No stages ready to execute. Remaining: {remaining_stages}")
                
                # Execute ready stages (respecting parallelism limits)
                max_parallel = pipeline.config.max_parallel_stages
                stage_batches = [ready_stages[i:i+max_parallel] for i in range(0, len(ready_stages), max_parallel)]
                
                for batch in stage_batches:
                    # Execute stages in parallel
                    tasks = []
                    for stage in batch:
                        task = asyncio.create_task(self._execute_stage(stage, context))
                        tasks.append((stage.name, task))
                    
                    # Wait for batch completion
                    for stage_name, task in tasks:
                        try:
                            result = await task
                            stage_results[stage_name] = result
                            completed_stages.add(stage_name)
                            
                            # Log stage completion
                            await self.experiment_tracker.log_metrics(
                                experiment_run.id,
                                {
                                    f"stage_{stage_name}_duration": result.execution_time,
                                    f"stage_{stage_name}_status": 1 if result.status == StageStatus.COMPLETED else 0
                                }
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Stage {stage_name} failed: {str(e)}")
                            stage_results[stage_name] = StageResult(
                                stage_name=stage_name,
                                status=StageStatus.FAILED,
                                error_message=str(e)
                            )
                            
                            # Handle failure based on pipeline policy
                            if pipeline.config.failure_policy == "stop_on_failure":
                                raise
                            # Otherwise continue with other stages
            
            # Pipeline completed successfully
            run.status = PipelineStatus.COMPLETED
            run.completed_at = datetime.utcnow()
            
            # Log final metrics
            await self.experiment_tracker.log_metrics(
                experiment_run.id,
                {
                    "pipeline_duration": (run.completed_at - run.started_at).total_seconds(),
                    "pipeline_success": 1,
                    "total_stages": len(pipeline.stages),
                    "successful_stages": len([r for r in stage_results.values() if r.status == StageStatus.COMPLETED])
                }
            )
            
            # End experiment
            await self.experiment_tracker.end_run(experiment_run.id, "FINISHED")
            
            self.logger.info(f"Pipeline {pipeline.name} completed successfully")
            
        except Exception as e:
            # Pipeline failed
            run.status = PipelineStatus.FAILED
            run.completed_at = datetime.utcnow()
            run.error_message = str(e)
            
            self.logger.error(f"Pipeline {pipeline.name} failed: {str(e)}")
            
            # Log failure
            if "experiment_run_id" in context.shared_data:
                await self.experiment_tracker.log_metrics(
                    context.get_data("experiment_run_id"),
                    {"pipeline_success": 0}
                )
                await self.experiment_tracker.end_run(
                    context.get_data("experiment_run_id"), "FAILED"
                )
        
        finally:
            # Stop monitoring
            await self.pipeline_monitor.stop_run_monitoring(run)
            
            # Clean up active runs
            if run.id in self.active_runs:
                del self.active_runs[run.id]
    
    async def _execute_stage(self, stage: StageConfig, context: PipelineExecutionContext) -> StageResult:
        """Execute a single pipeline stage."""
        start_time = datetime.utcnow()
        
        self.logger.info(f"Executing stage: {stage.name}")
        
        if stage.stage_type not in self.stage_executors:
            raise ValueError(f"Unknown stage type: {stage.stage_type}")
        
        executor = self.stage_executors[stage.stage_type]
        
        try:
            result = await executor(stage, context)
            result.execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            self.logger.info(f"Stage {stage.name} completed in {result.execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            self.logger.error(f"Stage {stage.name} failed after {execution_time:.2f}s: {str(e)}")
            
            return StageResult(
                stage_name=stage.name,
                status=StageStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _execute_data_ingestion(self, stage: StageConfig, context: PipelineExecutionContext) -> StageResult:
        """Execute data ingestion stage."""
        config = stage.config
        source_type = config.get("source_type", "file")
        
        if source_type == "file":
            file_path = config["file_path"]
            file_format = config.get("format", "csv")
            
            if file_format == "csv":
                data = pd.read_csv(file_path)
            elif file_format == "parquet":
                data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
        elif source_type == "database":
            # Database ingestion logic
            connection_string = config["connection_string"]
            query = config["query"]
            # Implementation would use actual database connector
            data = pd.DataFrame()  # Placeholder
            
        elif source_type == "api":
            # API ingestion logic
            endpoint = config["endpoint"]
            # Implementation would use HTTP client
            data = pd.DataFrame()  # Placeholder
            
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Store data in context
        context.set_data("raw_data", data)
        context.add_metric("ingested_rows", len(data))
        context.add_metric("ingested_columns", len(data.columns))
        
        return StageResult(
            stage_name=stage.name,
            status=StageStatus.COMPLETED,
            outputs={"data_shape": data.shape},
            metrics={"rows": len(data), "columns": len(data.columns)}
        )
    
    async def _execute_data_validation(self, stage: StageConfig, context: PipelineExecutionContext) -> StageResult:
        """Execute data validation stage."""
        data = context.get_data("raw_data")
        if data is None:
            raise ValueError("No data found in context for validation")
        
        config = stage.config
        validation_results = {}
        
        # Check for missing values
        missing_values = data.isnull().sum()
        validation_results["missing_values"] = missing_values.to_dict()
        
        # Check data types
        validation_results["data_types"] = data.dtypes.to_dict()
        
        # Check for duplicates
        duplicate_count = data.duplicated().sum()
        validation_results["duplicate_rows"] = int(duplicate_count)
        
        # Custom validation rules
        validation_rules = config.get("validation_rules", [])
        for rule in validation_rules:
            # Implement custom validation logic
            pass
        
        # Determine if validation passed
        max_missing_threshold = config.get("max_missing_percentage", 0.1)
        max_duplicates_threshold = config.get("max_duplicate_percentage", 0.05)
        
        missing_percentage = missing_values.sum() / (len(data) * len(data.columns))
        duplicate_percentage = duplicate_count / len(data)
        
        validation_passed = (
            missing_percentage <= max_missing_threshold and
            duplicate_percentage <= max_duplicates_threshold
        )
        
        if validation_passed:
            context.set_data("validated_data", data)
            status = StageStatus.COMPLETED
        else:
            status = StageStatus.FAILED
        
        return StageResult(
            stage_name=stage.name,
            status=status,
            outputs=validation_results,
            metrics={
                "missing_percentage": missing_percentage,
                "duplicate_percentage": duplicate_percentage,
                "validation_passed": 1 if validation_passed else 0
            }
        )
    
    async def _execute_feature_engineering(self, stage: StageConfig, context: PipelineExecutionContext) -> StageResult:
        """Execute feature engineering stage."""
        data = context.get_data("validated_data")
        if data is None:
            raise ValueError("No validated data found in context")
        
        config = stage.config
        inference_mode = config.get("inference_mode", False)
        
        if inference_mode:
            # Load feature transformations from feature store
            model_id = config["model_id"]
            feature_config = await self.feature_store.get_feature_config(model_id)
        else:
            # Define feature engineering steps
            feature_config = config.get("feature_config", {})
        
        # Apply feature transformations
        processed_data = data.copy()
        
        # Example transformations
        transformations = feature_config.get("transformations", [])
        for transform in transformations:
            transform_type = transform["type"]
            if transform_type == "scale":
                # Apply scaling
                pass
            elif transform_type == "encode":
                # Apply encoding
                pass
            # Add more transformation types
        
        # Store processed data
        context.set_data("processed_data", processed_data)
        
        # Save feature config if not inference mode
        if not inference_mode:
            await self.feature_store.save_feature_config(
                pipeline_id=context.pipeline_id,
                config=feature_config
            )
        
        return StageResult(
            stage_name=stage.name,
            status=StageStatus.COMPLETED,
            outputs={"processed_shape": processed_data.shape},
            metrics={"processed_features": len(processed_data.columns)}
        )
    
    async def _execute_model_training(self, stage: StageConfig, context: PipelineExecutionContext) -> StageResult:
        """Execute model training stage."""
        data = context.get_data("processed_data")
        if data is None:
            raise ValueError("No processed data found in context")
        
        config = stage.config
        model_type = config.get("model_type", "sklearn_random_forest")
        model_params = config.get("model_params", {})
        
        # Split data for training
        target_column = config["target_column"]
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Train model (simplified example)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_type == "sklearn_random_forest":
            model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Save model to registry
        model_version = await self.model_registry.register_model(
            name=f"pipeline_{context.pipeline_id}_model",
            model=model,
            metadata={
                "pipeline_id": context.pipeline_id,
                "run_id": context.run_id,
                "model_type": model_type,
                "training_data_shape": X_train.shape
            }
        )
        
        context.set_data("trained_model", model)
        context.set_data("model_version_id", model_version.id)
        context.set_data("test_data", (X_test, y_test))
        
        return StageResult(
            stage_name=stage.name,
            status=StageStatus.COMPLETED,
            outputs={"model_version_id": model_version.id},
            metrics={"training_samples": len(X_train)}
        )
    
    async def _execute_model_evaluation(self, stage: StageConfig, context: PipelineExecutionContext) -> StageResult:
        """Execute model evaluation stage."""
        model = context.get_data("trained_model")
        X_test, y_test = context.get_data("test_data")
        
        if model is None or X_test is None:
            raise ValueError("No trained model or test data found in context")
        
        config = stage.config
        evaluation_metrics = config.get("evaluation_metrics", ["accuracy"])
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {}
        if "accuracy" in evaluation_metrics:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
        if "precision" in evaluation_metrics:
            metrics["precision"] = precision_score(y_test, y_pred, average='weighted')
        if "recall" in evaluation_metrics:
            metrics["recall"] = recall_score(y_test, y_pred, average='weighted')
        if "f1" in evaluation_metrics:
            metrics["f1"] = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics to experiment tracker
        experiment_run_id = context.get_data("experiment_run_id")
        if experiment_run_id:
            await self.experiment_tracker.log_metrics(experiment_run_id, metrics)
        
        context.set_data("evaluation_metrics", metrics)
        
        return StageResult(
            stage_name=stage.name,
            status=StageStatus.COMPLETED,
            outputs=metrics,
            metrics=metrics
        )
    
    async def _execute_model_validation(self, stage: StageConfig, context: PipelineExecutionContext) -> StageResult:
        """Execute model validation stage."""
        metrics = context.get_data("evaluation_metrics")
        if metrics is None:
            raise ValueError("No evaluation metrics found in context")
        
        config = stage.config
        validation_threshold = config.get("validation_threshold", 0.8)
        
        # Check if model meets validation criteria
        primary_metric = config.get("primary_metric", "accuracy")
        primary_score = metrics.get(primary_metric, 0.0)
        
        validation_passed = primary_score >= validation_threshold
        
        if validation_passed:
            # Update model status in registry
            model_version_id = context.get_data("model_version_id")
            if model_version_id:
                await self.model_registry.update_model_status(
                    model_version_id, "validated"
                )
            status = StageStatus.COMPLETED
        else:
            status = StageStatus.FAILED
        
        return StageResult(
            stage_name=stage.name,
            status=status,
            outputs={"validation_passed": validation_passed, "primary_score": primary_score},
            metrics={"validation_passed": 1 if validation_passed else 0}
        )
    
    async def _execute_model_deployment(self, stage: StageConfig, context: PipelineExecutionContext) -> StageResult:
        """Execute model deployment stage."""
        model_version_id = context.get_data("model_version_id")
        if model_version_id is None:
            raise ValueError("No model version ID found in context")
        
        config = stage.config
        deployment_target = config.get("target", "production")
        
        # Deploy model to serving infrastructure
        deployment_result = await self.model_registry.deploy_model(
            model_version_id=model_version_id,
            deployment_target=deployment_target,
            deployment_config=config
        )
        
        context.set_data("deployment_id", deployment_result.deployment_id)
        
        return StageResult(
            stage_name=stage.name,
            status=StageStatus.COMPLETED,
            outputs={"deployment_id": deployment_result.deployment_id},
            metrics={"deployment_success": 1}
        )
    
    async def _execute_batch_prediction(self, stage: StageConfig, context: PipelineExecutionContext) -> StageResult:
        """Execute batch prediction stage."""
        data = context.get_data("processed_data")
        model_id = stage.config["model_id"]
        batch_size = stage.config.get("batch_size", 1000)
        
        if data is None:
            raise ValueError("No processed data found in context")
        
        # Load model from registry
        model = await self.model_registry.load_model(model_id)
        
        # Make batch predictions
        predictions = []
        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size]
            batch_predictions = model.predict(batch)
            predictions.extend(batch_predictions)
        
        # Save predictions
        output_config = stage.config.get("output_destination", {})
        if output_config.get("type") == "file":
            output_path = output_config["path"]
            prediction_df = pd.DataFrame({"predictions": predictions})
            prediction_df.to_csv(output_path, index=False)
            context.add_artifact("predictions", output_path)
        
        return StageResult(
            stage_name=stage.name,
            status=StageStatus.COMPLETED,
            outputs={"prediction_count": len(predictions)},
            metrics={"predictions_generated": len(predictions)}
        )
    
    async def _execute_monitoring_setup(self, stage: StageConfig, context: PipelineExecutionContext) -> StageResult:
        """Execute monitoring setup stage."""
        deployment_id = context.get_data("deployment_id")
        model_version_id = context.get_data("model_version_id")
        
        config = stage.config
        
        # Setup model monitoring
        monitoring_config = {
            "deployment_id": deployment_id,
            "model_version_id": model_version_id,
            "enable_drift_detection": config.get("enable_drift_detection", True),
            "alert_thresholds": config.get("alert_thresholds", {}),
            "monitoring_frequency": config.get("monitoring_frequency", "hourly")
        }
        
        await self.pipeline_monitor.setup_model_monitoring(monitoring_config)
        
        return StageResult(
            stage_name=stage.name,
            status=StageStatus.COMPLETED,
            outputs={"monitoring_enabled": True},
            metrics={"monitoring_setup_success": 1}
        )
    
    async def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get current status of a pipeline."""
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self.pipelines[pipeline_id]
        
        # Get active runs
        active_runs = [run for run in self.active_runs.values() if run.pipeline_id == pipeline_id]
        
        return {
            "pipeline_id": pipeline_id,
            "name": pipeline.name,
            "status": pipeline.status.value,
            "pipeline_type": pipeline.pipeline_type.value,
            "created_at": pipeline.created_at.isoformat(),
            "stage_count": len(pipeline.stages),
            "active_runs": len(active_runs),
            "last_run": active_runs[-1].started_at.isoformat() if active_runs else None
        }
    
    async def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Get status of a specific pipeline run."""
        if run_id not in self.active_runs:
            # Check completed runs in monitoring system
            return await self.pipeline_monitor.get_run_status(run_id)
        
        run = self.active_runs[run_id]
        
        return {
            "run_id": run_id,
            "pipeline_id": run.pipeline_id,
            "status": run.status.value,
            "trigger_type": run.trigger_type.value,
            "started_at": run.started_at.isoformat(),
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "error_message": run.error_message
        }
    
    async def cancel_run(self, run_id: str) -> bool:
        """Cancel a running pipeline."""
        if run_id not in self.active_runs:
            return False
        
        run = self.active_runs[run_id]
        run.status = PipelineStatus.CANCELLED
        run.completed_at = datetime.utcnow()
        
        self.logger.info(f"Pipeline run {run_id} cancelled")
        
        return True
    
    async def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all registered pipelines."""
        pipelines = []
        for pipeline in self.pipelines.values():
            pipelines.append({
                "id": pipeline.id,
                "name": pipeline.name,
                "pipeline_type": pipeline.pipeline_type.value,
                "status": pipeline.status.value,
                "created_at": pipeline.created_at.isoformat(),
                "stage_count": len(pipeline.stages)
            })
        
        return pipelines
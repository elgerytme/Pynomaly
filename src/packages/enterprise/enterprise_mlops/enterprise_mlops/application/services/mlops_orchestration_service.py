"""
MLOps Orchestration Service

Provides centralized orchestration and coordination of MLOps operations
including experiment tracking, model management, deployment, and monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from structlog import get_logger
from dependency_injector.wiring import inject, Provide

from ..domain.entities.mlops import (
    MLExperiment, MLModel, ModelDeployment, MLPipeline,
    ExperimentStatus, ModelStatus, DeploymentStatus, PipelineStatus
)
from ..infrastructure.mlops.mlflow.mlflow_integration import MLflowIntegration
from ..infrastructure.mlops.kubeflow.kubeflow_integration import KubeflowIntegration
from ..infrastructure.monitoring.datadog.datadog_integration import DatadogIntegration
from ..infrastructure.monitoring.newrelic.newrelic_integration import NewRelicIntegration

logger = get_logger(__name__)


class MLOpsOrchestrationService:
    """
    MLOps orchestration service.
    
    Coordinates MLOps operations across different platforms and tools
    including experiment tracking, model deployment, and monitoring.
    """
    
    def __init__(
        self,
        mlflow_integration: Optional[MLflowIntegration] = None,
        kubeflow_integration: Optional[KubeflowIntegration] = None,
        datadog_integration: Optional[DatadogIntegration] = None,
        newrelic_integration: Optional[NewRelicIntegration] = None
    ):
        self.mlflow = mlflow_integration
        self.kubeflow = kubeflow_integration
        self.datadog = datadog_integration
        self.newrelic = newrelic_integration
        self.logger = logger.bind(service="mlops_orchestration")
        
        self.logger.info("MLOpsOrchestrationService initialized")
    
    # Experiment Management
    
    async def create_experiment(
        self,
        experiment: MLExperiment,
        track_in_mlflow: bool = True,
        enable_monitoring: bool = True
    ) -> MLExperiment:
        """Create and register ML experiment across platforms."""
        self.logger.info("Creating ML experiment", name=experiment.name)
        
        try:
            # Track in MLflow if enabled
            if track_in_mlflow and self.mlflow:
                try:
                    mlflow_id = await self.mlflow.create_experiment(experiment)
                    experiment.external_experiment_id = mlflow_id
                    experiment.tracking_uri = self.mlflow.tracking_uri
                except Exception as e:
                    self.logger.warning("Failed to create MLflow experiment", error=str(e))
            
            # Enable monitoring if requested
            if enable_monitoring:
                await self._setup_experiment_monitoring(experiment)
            
            experiment.start_experiment()
            
            self.logger.info("ML experiment created successfully", id=experiment.id)
            return experiment
            
        except Exception as e:
            self.logger.error("Failed to create ML experiment", error=str(e))
            experiment.complete_experiment(ExperimentStatus.FAILED)
            raise
    
    async def start_experiment_run(
        self,
        experiment: MLExperiment,
        run_name: Optional[str] = None
    ) -> str:
        """Start experiment run with tracking."""
        self.logger.info("Starting experiment run", experiment=experiment.name)
        
        try:
            run_id = None
            
            # Start MLflow run
            if self.mlflow and experiment.external_experiment_id:
                run_id = await self.mlflow.start_run(experiment, run_name)
            
            # Send monitoring events
            if self.datadog:
                await self.datadog.create_model_deployment_event(
                    model_id=uuid4(),  # Placeholder for now
                    deployment_id=uuid4(),  # Placeholder for now
                    deployment_name=f"experiment-{experiment.name}",
                    status="started",
                    tenant_id=experiment.tenant_id,
                    details={"experiment_id": str(experiment.id)}
                )
            
            if self.newrelic:
                await self.newrelic.send_event(
                    "ExperimentStarted",
                    {
                        "experiment_id": str(experiment.id),
                        "experiment_name": experiment.name,
                        "tenant_id": str(experiment.tenant_id),
                        "run_id": run_id or "unknown"
                    }
                )
            
            return run_id or str(uuid4())
            
        except Exception as e:
            self.logger.error("Failed to start experiment run", error=str(e))
            raise
    
    async def log_experiment_metrics(
        self,
        experiment: MLExperiment,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log experiment metrics across platforms."""
        self.logger.debug("Logging experiment metrics", experiment=experiment.name, metrics=list(metrics.keys()))
        
        try:
            # Log to MLflow
            if self.mlflow:
                await self.mlflow.log_metrics(experiment, metrics, step)
            
            # Update experiment metrics
            for name, value in metrics.items():
                experiment.add_metric(name, value, step)
            
            # Send monitoring metrics
            if self.datadog:
                datadog_metrics = []
                for name, value in metrics.items():
                    datadog_metrics.append({
                        "name": f"pynomaly.experiment.{name}",
                        "value": value,
                        "type": "gauge",
                        "tags": [
                            f"experiment_id:{experiment.id}",
                            f"tenant_id:{experiment.tenant_id}",
                            "service:pynomaly",
                            "component:ml_experiment"
                        ]
                    })
                
                await self.datadog.send_metrics(datadog_metrics)
            
            if self.newrelic:
                await self.newrelic.send_metrics([
                    {
                        "name": f"pynomaly.experiment.{name}",
                        "value": value,
                        "type": "gauge",
                        "tags": {
                            "experiment_id": str(experiment.id),
                            "tenant_id": str(experiment.tenant_id),
                            "service": "pynomaly",
                            "component": "ml_experiment"
                        }
                    }
                    for name, value in metrics.items()
                ])
            
        except Exception as e:
            self.logger.error("Failed to log experiment metrics", error=str(e))
            raise
    
    async def complete_experiment(
        self,
        experiment: MLExperiment,
        status: ExperimentStatus = ExperimentStatus.COMPLETED
    ) -> None:
        """Complete experiment and update all platforms."""
        self.logger.info("Completing experiment", experiment=experiment.name, status=status)
        
        try:
            # Complete experiment
            experiment.complete_experiment(status)
            
            # Send completion events
            if self.datadog:
                await self.datadog.send_event(
                    "Experiment Completed",
                    f"Experiment {experiment.name} completed with status {status}",
                    alert_type="success" if status == ExperimentStatus.COMPLETED else "error",
                    tags=[
                        f"experiment_id:{experiment.id}",
                        f"tenant_id:{experiment.tenant_id}",
                        "service:pynomaly",
                        "component:ml_experiment"
                    ]
                )
            
            if self.newrelic:
                await self.newrelic.send_event(
                    "ExperimentCompleted",
                    {
                        "experiment_id": str(experiment.id),
                        "experiment_name": experiment.name,
                        "status": status.value,
                        "tenant_id": str(experiment.tenant_id),
                        "duration_seconds": experiment.duration_seconds
                    }
                )
            
        except Exception as e:
            self.logger.error("Failed to complete experiment", error=str(e))
            raise
    
    # Model Management
    
    async def register_model(
        self,
        model: MLModel,
        experiment: Optional[MLExperiment] = None,
        run_id: Optional[str] = None,
        artifact_path: str = "model"
    ) -> str:
        """Register model in model registry."""
        self.logger.info("Registering model", name=model.name, version=model.version)
        
        try:
            model_uri = None
            
            # Register in MLflow
            if self.mlflow and run_id:
                model_uri = await self.mlflow.register_model(model, run_id, artifact_path)
                model.model_uri = model_uri
            
            # Update model status
            model.status = ModelStatus.STAGING
            
            # Send monitoring events
            if self.datadog:
                await self.datadog.send_event(
                    "Model Registered",
                    f"Model {model.name} v{model.version} registered successfully",
                    alert_type="success",
                    tags=[
                        f"model_id:{model.id}",
                        f"tenant_id:{model.tenant_id}",
                        "service:pynomaly",
                        "component:model_registry"
                    ]
                )
            
            if self.newrelic:
                await self.newrelic.send_event(
                    "ModelRegistered",
                    {
                        "model_id": str(model.id),
                        "model_name": model.name,
                        "version": model.version,
                        "tenant_id": str(model.tenant_id),
                        "algorithm": model.algorithm,
                        "framework": model.framework
                    }
                )
            
            self.logger.info("Model registered successfully", model_uri=model_uri)
            return model_uri or f"models:/{model.name}/{model.version}"
            
        except Exception as e:
            self.logger.error("Failed to register model", error=str(e))
            raise
    
    async def deploy_model(
        self,
        deployment: ModelDeployment,
        model: MLModel,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Deploy model to target platform."""
        self.logger.info("Deploying model", deployment=deployment.name, platform=deployment.platform)
        
        try:
            deployment_name = None
            
            # Update deployment status
            deployment.update_status(DeploymentStatus.DEPLOYING)
            
            # Deploy based on platform
            if deployment.platform.lower() == "kubeflow" and self.kubeflow:
                deployment_name = await self.kubeflow.deploy_model_kserve(
                    deployment,
                    model.model_uri or f"models:/{model.name}/{model.version}",
                    deployment_config.get("serving_runtime", "sklearn") if deployment_config else "sklearn",
                    deployment_config.get("resources") if deployment_config else None
                )
            elif deployment.platform.lower() == "mlflow" and self.mlflow:
                deployment_name = await self.mlflow.deploy_model(
                    deployment,
                    model.model_uri or f"models:/{model.name}/{model.version}",
                    deployment_config
                )
            else:
                # Generic deployment configuration
                deployment_name = f"{deployment.name}-{deployment.environment}"
                deployment.endpoint_url = f"http://{deployment_name}.{deployment.environment}.svc.cluster.local"
            
            # Update model with deployment
            model.add_deployment(deployment.id)
            deployment.update_status(DeploymentStatus.DEPLOYED)
            
            # Setup deployment monitoring
            await self._setup_deployment_monitoring(deployment, model)
            
            # Send deployment events
            if self.datadog:
                await self.datadog.create_model_deployment_event(
                    model.id,
                    deployment.id,
                    deployment.name,
                    "deployed",
                    deployment.tenant_id,
                    {
                        "platform": deployment.platform,
                        "environment": deployment.environment,
                        "endpoint": deployment.endpoint_url
                    }
                )
            
            if self.newrelic:
                await self.newrelic.create_model_deployment_event(
                    model.id,
                    deployment.id,
                    deployment.name,
                    "deployed",
                    deployment.tenant_id,
                    {
                        "platform": deployment.platform,
                        "environment": deployment.environment,
                        "endpoint": deployment.endpoint_url
                    }
                )
            
            self.logger.info("Model deployed successfully", deployment_name=deployment_name)
            return deployment_name or deployment.name
            
        except Exception as e:
            deployment.update_status(DeploymentStatus.FAILED)
            self.logger.error("Failed to deploy model", error=str(e))
            raise
    
    # Pipeline Management
    
    async def create_pipeline(
        self,
        pipeline: MLPipeline,
        pipeline_function: Optional[callable] = None
    ) -> str:
        """Create ML pipeline."""
        self.logger.info("Creating ML pipeline", name=pipeline.name)
        
        try:
            pipeline_id = None
            
            # Create in Kubeflow if available
            if self.kubeflow and pipeline_function:
                pipeline_id = await self.kubeflow.create_pipeline(pipeline, pipeline_function)
                pipeline.external_pipeline_id = pipeline_id
            
            # Send monitoring events
            if self.datadog:
                await self.datadog.send_event(
                    "Pipeline Created",
                    f"ML pipeline {pipeline.name} created successfully",
                    alert_type="success",
                    tags=[
                        f"pipeline_id:{pipeline.id}",
                        f"tenant_id:{pipeline.tenant_id}",
                        "service:pynomaly",
                        "component:ml_pipeline"
                    ]
                )
            
            if self.newrelic:
                await self.newrelic.send_event(
                    "PipelineCreated",
                    {
                        "pipeline_id": str(pipeline.id),
                        "pipeline_name": pipeline.name,
                        "pipeline_type": pipeline.pipeline_type,
                        "tenant_id": str(pipeline.tenant_id)
                    }
                )
            
            self.logger.info("ML pipeline created successfully", external_id=pipeline_id)
            return pipeline_id or str(pipeline.id)
            
        except Exception as e:
            self.logger.error("Failed to create ML pipeline", error=str(e))
            raise
    
    async def run_pipeline(
        self,
        pipeline: MLPipeline,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute ML pipeline."""
        self.logger.info("Running ML pipeline", name=pipeline.name)
        
        try:
            # Start execution
            execution_id = pipeline.start_execution()
            
            run_id = None
            
            # Run in Kubeflow if available
            if self.kubeflow and pipeline.external_pipeline_id:
                run_id = await self.kubeflow.run_pipeline(
                    pipeline,
                    f"{pipeline.name}-{execution_id[:8]}",
                    parameters
                )
            
            # Send monitoring events
            if self.datadog:
                await self.datadog.send_event(
                    "Pipeline Started",
                    f"ML pipeline {pipeline.name} started execution",
                    alert_type="info",
                    tags=[
                        f"pipeline_id:{pipeline.id}",
                        f"execution_id:{execution_id}",
                        f"tenant_id:{pipeline.tenant_id}",
                        "service:pynomaly",
                        "component:ml_pipeline"
                    ]
                )
            
            if self.newrelic:
                await self.newrelic.send_event(
                    "PipelineStarted",
                    {
                        "pipeline_id": str(pipeline.id),
                        "execution_id": execution_id,
                        "pipeline_name": pipeline.name,
                        "tenant_id": str(pipeline.tenant_id),
                        "run_id": run_id or "unknown"
                    }
                )
            
            self.logger.info("ML pipeline started", execution_id=execution_id, run_id=run_id)
            return run_id or execution_id
            
        except Exception as e:
            pipeline.complete_execution(execution_id, success=False)
            self.logger.error("Failed to run ML pipeline", error=str(e))
            raise
    
    # Monitoring and Observability
    
    async def record_model_performance(
        self,
        deployment: ModelDeployment,
        metrics: Dict[str, float],
        additional_tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record model performance metrics."""
        self.logger.debug("Recording model performance", deployment=deployment.name)
        
        try:
            # Update deployment metrics
            deployment.record_request(
                success=metrics.get("success", True),
                latency_ms=metrics.get("latency_ms")
            )
            
            # Send to monitoring platforms
            if self.datadog:
                await self.datadog.send_model_performance_metrics(
                    deployment.model_id,
                    deployment.id,
                    metrics,
                    deployment.tenant_id,
                    tags=additional_tags
                )
            
            if self.newrelic:
                await self.newrelic.send_model_performance_metrics(
                    deployment.model_id,
                    deployment.id,
                    metrics,
                    deployment.tenant_id,
                    tags=additional_tags
                )
            
        except Exception as e:
            self.logger.error("Failed to record model performance", error=str(e))
            raise
    
    async def get_model_metrics_summary(
        self,
        model_id: UUID,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get comprehensive model metrics summary."""
        self.logger.debug("Getting model metrics summary", model_id=model_id)
        
        try:
            summary = {
                "model_id": str(model_id),
                "query_period_hours": hours_back,
                "datadog": None,
                "newrelic": None
            }
            
            # Get from Datadog
            if self.datadog:
                try:
                    summary["datadog"] = await self.datadog.get_model_metrics_summary(model_id, hours_back)
                except Exception as e:
                    self.logger.warning("Failed to get Datadog metrics", error=str(e))
            
            # Get from New Relic
            if self.newrelic:
                try:
                    summary["newrelic"] = await self.newrelic.get_model_metrics_summary(model_id, hours_back)
                except Exception as e:
                    self.logger.warning("Failed to get New Relic metrics", error=str(e))
            
            return summary
            
        except Exception as e:
            self.logger.error("Failed to get model metrics summary", error=str(e))
            raise
    
    # Private helper methods
    
    async def _setup_experiment_monitoring(self, experiment: MLExperiment) -> None:
        """Setup monitoring for experiment."""
        try:
            # Setup Datadog monitoring
            if self.datadog:
                # Create dashboard for experiment
                await self.datadog.create_model_monitoring_dashboard(
                    experiment.id,
                    experiment.name,
                    experiment.tenant_id
                )
            
            # Setup New Relic monitoring
            if self.newrelic:
                # Create alert policies if needed
                policy_id = await self.newrelic.create_alert_policy(
                    f"Pynomaly Experiment - {experiment.name}"
                )
                
                if policy_id:
                    await self.newrelic.create_anomaly_alert_condition(
                        policy_id,
                        f"High Error Rate - {experiment.name}",
                        "pynomaly.experiment.error_rate",
                        5.0,  # 5% error rate threshold
                        experiment.tenant_id
                    )
            
        except Exception as e:
            self.logger.warning("Failed to setup experiment monitoring", error=str(e))
    
    async def _setup_deployment_monitoring(self, deployment: ModelDeployment, model: MLModel) -> None:
        """Setup monitoring for model deployment."""
        try:
            # Setup Datadog monitoring
            if self.datadog:
                # Create monitoring dashboard
                await self.datadog.create_model_monitoring_dashboard(
                    model.id,
                    f"{model.name}-{deployment.environment}",
                    deployment.tenant_id
                )
                
                # Create anomaly detection monitor
                await self.datadog.create_anomaly_detection_monitor(
                    deployment.tenant_id,
                    f"{model.name}-{deployment.environment}",
                    5.0  # 5% anomaly threshold
                )
            
            # Setup New Relic monitoring
            if self.newrelic:
                # Create alert policy
                policy_id = await self.newrelic.create_alert_policy(
                    f"Pynomaly Model - {model.name} ({deployment.environment})"
                )
                
                if policy_id:
                    await self.newrelic.create_anomaly_alert_condition(
                        policy_id,
                        f"High Latency - {model.name}",
                        "pynomaly.model.latency",
                        1000.0,  # 1 second latency threshold
                        deployment.tenant_id,
                        f"{model.name}-{deployment.environment}"
                    )
            
        except Exception as e:
            self.logger.warning("Failed to setup deployment monitoring", error=str(e))
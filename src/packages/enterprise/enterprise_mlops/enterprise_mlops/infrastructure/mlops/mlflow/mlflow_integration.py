"""
MLflow Integration for Enterprise MLOps

Provides comprehensive integration with MLflow for experiment tracking,
model registry, and model deployment management.
"""

import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
import pickle
import json

from structlog import get_logger
import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment, Run, RegisteredModel, ModelVersion
from mlflow.store.artifact.artifact_repository import ArtifactRepository
import pandas as pd

from ...domain.entities.mlops import MLExperiment, MLModel, ModelDeployment, ExperimentStatus, ModelStatus

logger = get_logger(__name__)


class MLflowIntegration:
    """
    MLflow integration for enterprise MLOps.
    
    Provides comprehensive integration with MLflow including
    experiment tracking, model registry, and deployment management.
    """
    
    def __init__(
        self,
        tracking_uri: str,
        registry_uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None
    ):
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri
        self.username = username
        self.password = password
        self.token = token
        
        self.client = None
        self.logger = logger.bind(integration="mlflow")
        
        self.logger.info("MLflowIntegration initialized", tracking_uri=tracking_uri)
    
    async def connect(self) -> bool:
        """Establish connection to MLflow."""
        self.logger.info("Connecting to MLflow")
        
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Set registry URI if different
            if self.registry_uri != self.tracking_uri:
                mlflow.set_registry_uri(self.registry_uri)
            
            # Configure authentication
            if self.token:
                os.environ["MLFLOW_TRACKING_TOKEN"] = self.token
            elif self.username and self.password:
                os.environ["MLFLOW_TRACKING_USERNAME"] = self.username
                os.environ["MLFLOW_TRACKING_PASSWORD"] = self.password
            
            # Create MLflow client
            self.client = MlflowClient(
                tracking_uri=self.tracking_uri,
                registry_uri=self.registry_uri
            )
            
            # Test connection
            await self._test_connection()
            
            self.logger.info("Successfully connected to MLflow")
            return True
            
        except Exception as e:
            error_msg = f"Failed to connect to MLflow: {str(e)}"
            self.logger.error(error_msg)
            return False
    
    async def create_experiment(
        self,
        experiment: MLExperiment
    ) -> str:
        """Create MLflow experiment."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        self.logger.info("Creating MLflow experiment", name=experiment.name)
        
        try:
            # Check if experiment already exists
            existing_experiment = None
            try:
                existing_experiment = self.client.get_experiment_by_name(experiment.name)
            except Exception:
                pass
            
            if existing_experiment:
                mlflow_experiment_id = existing_experiment.experiment_id
                self.logger.info("Using existing experiment", experiment_id=mlflow_experiment_id)
            else:
                # Create new experiment
                mlflow_experiment_id = self.client.create_experiment(
                    name=experiment.name,
                    artifact_location=None,
                    tags=experiment.tags
                )
                self.logger.info("Created new experiment", experiment_id=mlflow_experiment_id)
            
            return mlflow_experiment_id
            
        except Exception as e:
            self.logger.error("Failed to create experiment", error=str(e))
            raise
    
    async def start_run(
        self,
        experiment: MLExperiment,
        run_name: Optional[str] = None
    ) -> str:
        """Start MLflow run for experiment."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        self.logger.info("Starting MLflow run", experiment=experiment.name)
        
        try:
            # Get or create experiment
            mlflow_experiment_id = experiment.external_experiment_id
            if not mlflow_experiment_id:
                mlflow_experiment_id = await self.create_experiment(experiment)
                experiment.external_experiment_id = mlflow_experiment_id
            
            # Start run
            with mlflow.start_run(
                experiment_id=mlflow_experiment_id,
                run_name=run_name or f"run-{experiment.id}"
            ) as run:
                mlflow_run_id = run.info.run_id
                
                # Log parameters
                if experiment.parameters:
                    mlflow.log_params(experiment.parameters)
                
                # Log hyperparameters as parameters
                if experiment.hyperparameters:
                    for key, value in experiment.hyperparameters.items():
                        mlflow.log_param(f"hp_{key}", value)
                
                # Set tags
                tags = dict(experiment.tags)
                tags.update({
                    "pynomaly.experiment_id": str(experiment.id),
                    "pynomaly.tenant_id": str(experiment.tenant_id),
                    "pynomaly.project": experiment.project_name
                })
                
                mlflow.set_tags(tags)
                
                self.logger.info("Started MLflow run", run_id=mlflow_run_id)
                return mlflow_run_id
            
        except Exception as e:
            self.logger.error("Failed to start run", error=str(e))
            raise
    
    async def log_metrics(
        self,
        experiment: MLExperiment,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log metrics to MLflow."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        self.logger.debug("Logging metrics to MLflow", metrics=list(metrics.keys()))
        
        try:
            # Get active run or start new one
            if mlflow.active_run():
                run_id = mlflow.active_run().info.run_id
            else:
                # Find run by experiment tags
                run_id = await self._find_run_by_experiment(experiment)
                if not run_id:
                    raise RuntimeError("No active run found for experiment")
            
            # Log metrics
            for name, value in metrics.items():
                self.client.log_metric(run_id, name, value, step=step)
            
            self.logger.debug("Metrics logged successfully")
            
        except Exception as e:
            self.logger.error("Failed to log metrics", error=str(e))
            raise
    
    async def log_artifacts(
        self,
        experiment: MLExperiment,
        artifacts: Dict[str, Any],
        artifact_path: Optional[str] = None
    ) -> None:
        """Log artifacts to MLflow."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        self.logger.debug("Logging artifacts to MLflow", artifacts=list(artifacts.keys()))
        
        try:
            # Get active run
            run_id = None
            if mlflow.active_run():
                run_id = mlflow.active_run().info.run_id
            else:
                run_id = await self._find_run_by_experiment(experiment)
            
            if not run_id:
                raise RuntimeError("No active run found for experiment")
            
            # Create temporary directory for artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                for artifact_name, artifact_data in artifacts.items():
                    artifact_file = os.path.join(temp_dir, artifact_name)
                    
                    # Save artifact based on type
                    if isinstance(artifact_data, dict):
                        # JSON data
                        with open(f"{artifact_file}.json", 'w') as f:
                            json.dump(artifact_data, f, indent=2)
                        self.client.log_artifact(run_id, f"{artifact_file}.json", artifact_path)
                    
                    elif isinstance(artifact_data, pd.DataFrame):
                        # DataFrame as CSV
                        csv_file = f"{artifact_file}.csv"
                        artifact_data.to_csv(csv_file, index=False)
                        self.client.log_artifact(run_id, csv_file, artifact_path)
                    
                    elif hasattr(artifact_data, 'save'):
                        # Model with save method (sklearn, etc.)
                        model_file = f"{artifact_file}.pkl"
                        with open(model_file, 'wb') as f:
                            pickle.dump(artifact_data, f)
                        self.client.log_artifact(run_id, model_file, artifact_path)
                    
                    else:
                        # Generic pickle
                        pickle_file = f"{artifact_file}.pkl"
                        with open(pickle_file, 'wb') as f:
                            pickle.dump(artifact_data, f)
                        self.client.log_artifact(run_id, pickle_file, artifact_path)
            
            self.logger.debug("Artifacts logged successfully")
            
        except Exception as e:
            self.logger.error("Failed to log artifacts", error=str(e))
            raise
    
    async def register_model(
        self,
        model: MLModel,
        run_id: str,
        artifact_path: str = "model"
    ) -> str:
        """Register model in MLflow Model Registry."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        self.logger.info("Registering model in MLflow", name=model.name, version=model.version)
        
        try:
            # Create registered model if it doesn't exist
            try:
                registered_model = self.client.get_registered_model(model.name)
            except Exception:
                registered_model = self.client.create_registered_model(
                    name=model.name,
                    tags={
                        "pynomaly.model_id": str(model.id),
                        "pynomaly.tenant_id": str(model.tenant_id),
                        "framework": model.framework
                    },
                    description=model.description
                )
            
            # Register model version
            model_version = self.client.create_model_version(
                name=model.name,
                source=f"runs:/{run_id}/{artifact_path}",
                run_id=run_id,
                tags={
                    "pynomaly.model_id": str(model.id),
                    "pynomaly.version": model.version,
                    "algorithm": model.algorithm
                },
                description=f"Model version {model.version}"
            )
            
            # Update model with MLflow information
            model_uri = f"models:/{model.name}/{model_version.version}"
            
            self.logger.info("Model registered successfully", 
                           name=model.name, 
                           version=model_version.version,
                           uri=model_uri)
            
            return model_uri
            
        except Exception as e:
            self.logger.error("Failed to register model", error=str(e))
            raise
    
    async def deploy_model(
        self,
        deployment: ModelDeployment,
        model_uri: str,
        deployment_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Deploy model using MLflow."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        self.logger.info("Deploying model with MLflow", deployment=deployment.name)
        
        try:
            # For now, this creates a deployment configuration
            # In practice, this would integrate with MLflow's deployment targets
            # (SageMaker, AzureML, etc.)
            
            deployment_name = f"{deployment.name}-{deployment.environment}"
            
            # Create deployment tags
            deployment_tags = {
                "pynomaly.deployment_id": str(deployment.id),
                "pynomaly.model_id": str(deployment.model_id),
                "pynomaly.tenant_id": str(deployment.tenant_id),
                "environment": deployment.environment,
                "platform": deployment.platform
            }
            
            # Set model version stage based on environment
            if deployment.environment.lower() == "production":
                stage = "Production"
            elif deployment.environment.lower() == "staging":
                stage = "Staging"
            else:
                stage = "None"
            
            # Parse model URI to get name and version
            if model_uri.startswith("models:/"):
                parts = model_uri.replace("models:/", "").split("/")
                model_name = parts[0]
                model_version = parts[1] if len(parts) > 1 else "latest"
                
                # Transition model to appropriate stage
                if stage != "None":
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=model_version,
                        stage=stage,
                        archive_existing_versions=False
                    )
            
            self.logger.info("Model deployment configured", name=deployment_name)
            return deployment_name
            
        except Exception as e:
            self.logger.error("Failed to deploy model", error=str(e))
            raise
    
    async def get_experiment_runs(
        self,
        experiment_id: str,
        max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get runs for an experiment."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                max_results=max_results
            )
            
            run_data = []
            for run in runs:
                run_info = {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000) if run.info.start_time else None,
                    "end_time": datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags
                }
                run_data.append(run_info)
            
            return run_data
            
        except Exception as e:
            self.logger.error("Failed to get experiment runs", error=str(e))
            raise
    
    async def get_model_versions(
        self,
        model_name: str
    ) -> List[Dict[str, Any]]:
        """Get all versions of a registered model."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        try:
            model_versions = self.client.get_latest_versions(
                name=model_name,
                stages=["Production", "Staging", "Archived", "None"]
            )
            
            version_data = []
            for version in model_versions:
                version_info = {
                    "version": version.version,
                    "stage": version.current_stage,
                    "creation_time": datetime.fromtimestamp(version.creation_timestamp / 1000),
                    "last_updated": datetime.fromtimestamp(version.last_updated_timestamp / 1000),
                    "description": version.description,
                    "source": version.source,
                    "run_id": version.run_id,
                    "status": version.status,
                    "tags": version.tags
                }
                version_data.append(version_info)
            
            return version_data
            
        except Exception as e:
            self.logger.error("Failed to get model versions", error=str(e))
            raise
    
    async def search_experiments(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """Search experiments with optional filter."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        try:
            experiments = self.client.search_experiments(
                filter_string=filter_string,
                max_results=max_results
            )
            
            experiment_data = []
            for exp in experiments:
                exp_info = {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage,
                    "creation_time": datetime.fromtimestamp(exp.creation_time / 1000) if exp.creation_time else None,
                    "last_update_time": datetime.fromtimestamp(exp.last_update_time / 1000) if exp.last_update_time else None,
                    "tags": exp.tags
                }
                experiment_data.append(exp_info)
            
            return experiment_data
            
        except Exception as e:
            self.logger.error("Failed to search experiments", error=str(e))
            raise
    
    async def delete_experiment(
        self,
        experiment_id: str
    ) -> bool:
        """Delete MLflow experiment."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        try:
            self.client.delete_experiment(experiment_id)
            self.logger.info("Experiment deleted", experiment_id=experiment_id)
            return True
            
        except Exception as e:
            self.logger.error("Failed to delete experiment", error=str(e))
            return False
    
    async def get_run_artifacts(
        self,
        run_id: str,
        artifact_path: str = ""
    ) -> List[Dict[str, Any]]:
        """Get artifacts for a run."""
        if not self.client:
            raise RuntimeError("Not connected to MLflow")
        
        try:
            artifacts = self.client.list_artifacts(run_id, artifact_path)
            
            artifact_data = []
            for artifact in artifacts:
                artifact_info = {
                    "path": artifact.path,
                    "is_dir": artifact.is_dir,
                    "file_size": artifact.file_size
                }
                artifact_data.append(artifact_info)
            
            return artifact_data
            
        except Exception as e:
            self.logger.error("Failed to get run artifacts", error=str(e))
            raise
    
    # Private helper methods
    
    async def _test_connection(self) -> None:
        """Test MLflow connection."""
        try:
            # Try to list experiments
            experiments = self.client.search_experiments(max_results=1)
            self.logger.info("Connection test successful")
        except Exception as e:
            raise RuntimeError(f"Connection test failed: {str(e)}")
    
    async def _find_run_by_experiment(self, experiment: MLExperiment) -> Optional[str]:
        """Find MLflow run ID for experiment."""
        try:
            if not experiment.external_experiment_id:
                return None
            
            # Search for runs with matching tags
            runs = self.client.search_runs(
                experiment_ids=[experiment.external_experiment_id],
                filter_string=f"tags.pynomaly.experiment_id = '{experiment.id}'",
                max_results=1
            )
            
            if runs:
                return runs[0].info.run_id
            
            return None
            
        except Exception:
            return None
    
    def __del__(self):
        """Cleanup on object destruction."""
        # MLflow client doesn't require explicit cleanup
        pass
"""MLOps service for experiment tracking and model lifecycle management."""

import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import joblib

from anomaly_detection.domain.entities.model_entity import ModelEntity
from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.domain.value_objects.algorithm_config import AlgorithmConfig
from anomaly_detection.infrastructure.repositories.model_repository import ModelRepository


@dataclass
class ExperimentConfig:
    """Configuration for an ML experiment."""
    experiment_name: str
    description: str
    tags: Dict[str, str]
    parameters: Dict[str, Any]
    created_by: str
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags,
            "parameters": self.parameters,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ExperimentRun:
    """Represents a single experiment run."""
    run_id: str
    experiment_id: str
    status: str  # running, completed, failed
    start_time: datetime
    end_time: Optional[datetime]
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, str]  # artifact_name -> artifact_path
    model_path: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "model_path": self.model_path
        }


@dataclass
class ModelVersion:
    """Represents a versioned model."""
    model_id: str
    version: int
    run_id: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    model_path: str
    status: str  # staging, production, archived
    deployment_config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "performance_metrics": self.performance_metrics,
            "validation_metrics": self.validation_metrics,
            "model_path": self.model_path,
            "status": self.status,
            "deployment_config": self.deployment_config
        }


class MLOpsService:
    """Service for MLOps operations including experiment tracking and model lifecycle management."""
    
    def __init__(self, model_repository: ModelRepository, tracking_backend: str = "local"):
        """Initialize MLOps service.
        
        Args:
            model_repository: Repository for model persistence
            tracking_backend: Backend for experiment tracking (local, mlflow, wandb)
        """
        self.model_repository = model_repository
        self.tracking_backend = tracking_backend
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage for local backend (replace with proper storage)
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._runs: Dict[str, ExperimentRun] = {}
        self._model_versions: Dict[str, List[ModelVersion]] = {}
        
        # Initialize tracking backend
        self._init_tracking_backend()
    
    def _init_tracking_backend(self):
        """Initialize the experiment tracking backend."""
        if self.tracking_backend == "mlflow":
            try:
                import mlflow
                self.mlflow = mlflow
                self.logger.info("Initialized MLflow tracking backend")
            except ImportError:
                self.logger.warning("MLflow not available, falling back to local tracking")
                self.tracking_backend = "local"
        
        elif self.tracking_backend == "wandb":
            try:
                import wandb
                self.wandb = wandb
                self.logger.info("Initialized Weights & Biases tracking backend")
            except ImportError:
                self.logger.warning("Weights & Biases not available, falling back to local tracking")
                self.tracking_backend = "local"
        
        if self.tracking_backend == "local":
            self.logger.info("Using local experiment tracking")
    
    def create_experiment(self, 
                         experiment_name: str,
                         description: str = "",
                         tags: Optional[Dict[str, str]] = None,
                         created_by: str = "system") -> str:
        """Create a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            description: Description of the experiment
            tags: Optional tags for the experiment
            created_by: User who created the experiment
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        experiment_config = ExperimentConfig(
            experiment_name=experiment_name,
            description=description,
            tags=tags or {},
            parameters={},
            created_by=created_by,
            created_at=datetime.now()
        )
        
        if self.tracking_backend == "mlflow":
            try:
                exp_id = self.mlflow.create_experiment(
                    name=experiment_name,
                    tags=tags
                )
                experiment_id = str(exp_id)
            except Exception as e:
                self.logger.error(f"Failed to create MLflow experiment: {e}")
        
        elif self.tracking_backend == "wandb":
            try:
                # W&B experiments are created implicitly with runs
                pass
            except Exception as e:
                self.logger.error(f"Failed to create W&B experiment: {e}")
        
        # Store locally regardless of backend
        self._experiments[experiment_id] = experiment_config
        
        self.logger.info(f"Created experiment '{experiment_name}' with ID: {experiment_id}")
        return experiment_id
    
    def start_run(self, 
                  experiment_id: str,
                  parameters: Optional[Dict[str, Any]] = None,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new experiment run.
        
        Args:
            experiment_id: ID of the experiment
            parameters: Parameters for this run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())
        
        run = ExperimentRun(
            run_id=run_id,
            experiment_id=experiment_id,
            status="running",
            start_time=datetime.now(),
            end_time=None,
            parameters=parameters or {},
            metrics={},
            artifacts={},
            model_path=None
        )
        
        if self.tracking_backend == "mlflow":
            try:
                self.mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=run_id,
                    tags=tags
                )
                if parameters:
                    self.mlflow.log_params(parameters)
            except Exception as e:
                self.logger.error(f"Failed to start MLflow run: {e}")
        
        elif self.tracking_backend == "wandb":
            try:
                experiment_name = self._experiments.get(experiment_id, {}).get("experiment_name", "default")
                self.wandb.init(
                    project=experiment_name,
                    name=run_id,
                    config=parameters,
                    tags=list(tags.values()) if tags else None
                )
            except Exception as e:
                self.logger.error(f"Failed to start W&B run: {e}")
        
        self._runs[run_id] = run
        
        self.logger.info(f"Started run {run_id} for experiment {experiment_id}")
        return run_id
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for a run.
        
        Args:
            run_id: ID of the run
            metrics: Metrics to log
            step: Optional step number
        """
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")
        
        run = self._runs[run_id]
        run.metrics.update(metrics)
        
        if self.tracking_backend == "mlflow":
            try:
                for metric_name, metric_value in metrics.items():
                    self.mlflow.log_metric(metric_name, metric_value, step=step)
            except Exception as e:
                self.logger.error(f"Failed to log MLflow metrics: {e}")
        
        elif self.tracking_backend == "wandb":
            try:
                log_dict = metrics.copy()
                if step is not None:
                    log_dict["step"] = step
                self.wandb.log(log_dict)
            except Exception as e:
                self.logger.error(f"Failed to log W&B metrics: {e}")
        
        self.logger.debug(f"Logged metrics for run {run_id}: {metrics}")
    
    def log_artifact(self, run_id: str, artifact_name: str, artifact_path: str):
        """Log an artifact for a run.
        
        Args:
            run_id: ID of the run
            artifact_name: Name of the artifact
            artifact_path: Path to the artifact
        """
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")
        
        run = self._runs[run_id]
        run.artifacts[artifact_name] = artifact_path
        
        if self.tracking_backend == "mlflow":
            try:
                self.mlflow.log_artifact(artifact_path)
            except Exception as e:
                self.logger.error(f"Failed to log MLflow artifact: {e}")
        
        elif self.tracking_backend == "wandb":
            try:
                self.wandb.save(artifact_path)
            except Exception as e:
                self.logger.error(f"Failed to log W&B artifact: {e}")
        
        self.logger.debug(f"Logged artifact '{artifact_name}' for run {run_id}")
    
    def log_model(self, run_id: str, model: Any, model_name: str = "model") -> str:
        """Log a trained model.
        
        Args:
            run_id: ID of the run
            model: The trained model object
            model_name: Name for the model
            
        Returns:
            Path where the model was saved
        """
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")
        
        # Create model directory
        model_dir = Path(f"models/{run_id}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        
        # Update run
        run = self._runs[run_id]
        run.model_path = str(model_path)
        
        # Log as artifact
        self.log_artifact(run_id, model_name, str(model_path))
        
        if self.tracking_backend == "mlflow":
            try:
                self.mlflow.sklearn.log_model(model, model_name)
            except Exception as e:
                self.logger.error(f"Failed to log MLflow model: {e}")
        
        self.logger.info(f"Logged model for run {run_id} at {model_path}")
        return str(model_path)
    
    def end_run(self, run_id: str, status: str = "completed"):
        """End an experiment run.
        
        Args:
            run_id: ID of the run
            status: Final status (completed, failed)
        """
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")
        
        run = self._runs[run_id]
        run.status = status
        run.end_time = datetime.now()
        
        if self.tracking_backend == "mlflow":
            try:
                self.mlflow.end_run()
            except Exception as e:
                self.logger.error(f"Failed to end MLflow run: {e}")
        
        elif self.tracking_backend == "wandb":
            try:
                self.wandb.finish()
            except Exception as e:
                self.logger.error(f"Failed to end W&B run: {e}")
        
        self.logger.info(f"Ended run {run_id} with status: {status}")
    
    def register_model_version(self,
                             model_id: str,
                             run_id: str,
                             performance_metrics: Dict[str, float],
                             validation_metrics: Optional[Dict[str, float]] = None,
                             deployment_config: Optional[Dict[str, Any]] = None) -> ModelVersion:
        """Register a new model version.
        
        Args:
            model_id: ID of the model
            run_id: ID of the run that produced this model
            performance_metrics: Performance metrics for the model
            validation_metrics: Validation metrics
            deployment_config: Configuration for deployment
            
        Returns:
            ModelVersion object
        """
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")
        
        run = self._runs[run_id]
        if not run.model_path:
            raise ValueError(f"No model logged for run {run_id}")
        
        # Determine next version number
        existing_versions = self._model_versions.get(model_id, [])
        next_version = len(existing_versions) + 1
        
        model_version = ModelVersion(
            model_id=model_id,
            version=next_version,
            run_id=run_id,
            created_at=datetime.now(),
            performance_metrics=performance_metrics,
            validation_metrics=validation_metrics or {},
            model_path=run.model_path,
            status="staging",
            deployment_config=deployment_config
        )
        
        # Store version
        if model_id not in self._model_versions:
            self._model_versions[model_id] = []
        self._model_versions[model_id].append(model_version)
        
        self.logger.info(f"Registered model version {next_version} for model {model_id}")
        return model_version
    
    def promote_model_version(self, model_id: str, version: int, target_stage: str = "production"):
        """Promote a model version to a different stage.
        
        Args:
            model_id: ID of the model
            version: Version number to promote
            target_stage: Target stage (staging, production, archived)
        """
        if model_id not in self._model_versions:
            raise ValueError(f"Model {model_id} not found")
        
        versions = self._model_versions[model_id]
        target_version = None
        
        for v in versions:
            if v.version == version:
                target_version = v
                break
        
        if not target_version:
            raise ValueError(f"Version {version} not found for model {model_id}")
        
        # If promoting to production, demote current production version
        if target_stage == "production":
            for v in versions:
                if v.status == "production":
                    v.status = "archived"
                    self.logger.info(f"Archived previous production version {v.version}")
        
        target_version.status = target_stage
        
        self.logger.info(f"Promoted model {model_id} version {version} to {target_stage}")
    
    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            List of model versions
        """
        return self._model_versions.get(model_id, [])
    
    def get_production_model(self, model_id: str) -> Optional[ModelVersion]:
        """Get the production version of a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Production model version or None
        """
        versions = self._model_versions.get(model_id, [])
        
        for version in versions:
            if version.status == "production":
                return version
        
        return None
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compare multiple experiment runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            Comparison data
        """
        comparison = {}
        
        for run_id in run_ids:
            if run_id in self._runs:
                run = self._runs[run_id]
                comparison[run_id] = {
                    "parameters": run.parameters,
                    "metrics": run.metrics,
                    "status": run.status,
                    "duration_minutes": self._calculate_run_duration(run)
                }
            else:
                comparison[run_id] = {"error": "Run not found"}
        
        return comparison
    
    def _calculate_run_duration(self, run: ExperimentRun) -> Optional[float]:
        """Calculate run duration in minutes.
        
        Args:
            run: ExperimentRun object
            
        Returns:
            Duration in minutes or None if run is still running
        """
        if run.end_time:
            duration = run.end_time - run.start_time
            return duration.total_seconds() / 60.0
        return None
    
    def get_experiment_runs(self, experiment_id: str) -> List[ExperimentRun]:
        """Get all runs for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            List of experiment runs
        """
        return [run for run in self._runs.values() if run.experiment_id == experiment_id]
    
    def search_runs(self, 
                   experiment_ids: Optional[List[str]] = None,
                   status: Optional[str] = None,
                   min_performance: Optional[Dict[str, float]] = None) -> List[ExperimentRun]:
        """Search for runs based on criteria.
        
        Args:
            experiment_ids: Filter by experiment IDs
            status: Filter by status
            min_performance: Minimum performance thresholds
            
        Returns:
            List of matching runs
        """
        matching_runs = []
        
        for run in self._runs.values():
            # Filter by experiment ID
            if experiment_ids and run.experiment_id not in experiment_ids:
                continue
            
            # Filter by status
            if status and run.status != status:
                continue
            
            # Filter by minimum performance
            if min_performance:
                meets_threshold = True
                for metric, threshold in min_performance.items():
                    if metric not in run.metrics or run.metrics[metric] < threshold:
                        meets_threshold = False
                        break
                if not meets_threshold:
                    continue
            
            matching_runs.append(run)
        
        return matching_runs
    
    def cleanup_old_experiments(self, days: int = 30):
        """Clean up old experiments and runs.
        
        Args:
            days: Number of days to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Clean up old experiments
        experiments_to_remove = []
        for exp_id, experiment in self._experiments.items():
            if experiment.created_at < cutoff_date:
                experiments_to_remove.append(exp_id)
        
        for exp_id in experiments_to_remove:
            del self._experiments[exp_id]
            self.logger.info(f"Cleaned up experiment {exp_id}")
        
        # Clean up old runs
        runs_to_remove = []
        for run_id, run in self._runs.items():
            if run.start_time < cutoff_date:
                runs_to_remove.append(run_id)
        
        for run_id in runs_to_remove:
            del self._runs[run_id]
            self.logger.info(f"Cleaned up run {run_id}")
        
        self.logger.info(f"Cleaned up {len(experiments_to_remove)} experiments and {len(runs_to_remove)} runs")
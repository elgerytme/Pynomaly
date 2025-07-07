"""MLflow integration for experiment tracking and model registry."""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel

from pynomaly.domain.entities import Detector, Dataset

logger = logging.getLogger(__name__)


class MLflowConfig(BaseModel):
    """MLflow configuration."""
    
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "pynomaly_anomaly_detection"
    model_registry_uri: Optional[str] = None
    artifact_location: Optional[str] = None
    enable_autolog: bool = True


class MLflowExperimentTracker:
    """MLflow experiment tracking for anomaly detection workflows."""
    
    def __init__(self, config: MLflowConfig):
        self.config = config
        self.mlflow_client = None
        self.experiment_id = None
        self._initialize_mlflow()
    
    def _initialize_mlflow(self):
        """Initialize MLflow client and experiment."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            
            if self.config.model_registry_uri:
                mlflow.set_registry_uri(self.config.model_registry_uri)
            
            # Initialize client
            self.mlflow_client = MlflowClient()
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
                if experiment:
                    self.experiment_id = experiment.experiment_id
                else:
                    self.experiment_id = mlflow.create_experiment(
                        name=self.config.experiment_name,
                        artifact_location=self.config.artifact_location
                    )
            except Exception:
                self.experiment_id = mlflow.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location
                )
            
            # Enable autologging
            if self.config.enable_autolog:
                mlflow.sklearn.autolog()
            
            logger.info(f"MLflow initialized: experiment_id={self.experiment_id}")
            
        except ImportError:
            logger.warning("MLflow not available, using mock implementation")
            self.mlflow_client = None
    
    async def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Start MLflow run."""
        try:
            import mlflow
            
            if not self.mlflow_client:
                return None
            
            # Start run
            run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags
            )
            
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None
    
    async def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        try:
            import mlflow
            
            if not self.mlflow_client:
                return
            
            # Convert non-string parameters
            str_params = {}
            for key, value in parameters.items():
                if isinstance(value, (dict, list)):
                    str_params[key] = str(value)
                else:
                    str_params[key] = value
            
            mlflow.log_params(str_params)
            logger.debug(f"Logged {len(str_params)} parameters to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log parameters to MLflow: {e}")
    
    async def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        try:
            import mlflow
            
            if not self.mlflow_client:
                return
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            
            logger.debug(f"Logged {len(metrics)} metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log metrics to MLflow: {e}")
    
    async def log_dataset(
        self,
        dataset: Dataset,
        name: str = "dataset",
        context: str = "training"
    ) -> None:
        """Log dataset information to MLflow."""
        try:
            import mlflow
            
            if not self.mlflow_client:
                return
            
            # Log dataset metadata
            dataset_info = {
                f"{name}_shape": f"{dataset.data.shape[0]}x{dataset.data.shape[1]}",
                f"{name}_features": len(dataset.feature_names),
                f"{name}_context": context
            }
            
            mlflow.log_params(dataset_info)
            
            # Log dataset statistics
            if dataset.data.size > 0:
                df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
                stats = {
                    f"{name}_mean": float(df.mean().mean()),
                    f"{name}_std": float(df.std().mean()),
                    f"{name}_null_count": int(df.isnull().sum().sum())
                }
                mlflow.log_metrics(stats)
            
            logger.debug(f"Logged dataset {name} to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log dataset to MLflow: {e}")
    
    async def log_detector(
        self,
        detector: Detector,
        model_name: str = "anomaly_detector"
    ) -> None:
        """Log detector model to MLflow."""
        try:
            import mlflow
            import mlflow.sklearn
            
            if not self.mlflow_client:
                return
            
            # Log detector parameters
            detector_params = {
                "detector_id": detector.detector_id,
                "algorithm": detector.algorithm,
                "version": detector.version,
                "is_trained": detector.is_trained
            }
            
            if detector.hyperparameters:
                detector_params.update({
                    f"param_{k}": v for k, v in detector.hyperparameters.items()
                })
            
            await self.log_parameters(detector_params)
            
            # Log model if it has sklearn-compatible interface
            if hasattr(detector, 'model') and detector.model is not None:
                try:
                    mlflow.sklearn.log_model(
                        detector.model,
                        model_name,
                        registered_model_name=f"{model_name}_{detector.algorithm}"
                    )
                    logger.info(f"Logged detector model {model_name} to MLflow")
                except Exception as e:
                    logger.warning(f"Could not log model with sklearn format: {e}")
                    # Fallback to pickle
                    with open(f"{model_name}.pkl", "wb") as f:
                        pickle.dump(detector, f)
                    mlflow.log_artifact(f"{model_name}.pkl")
                    os.remove(f"{model_name}.pkl")
            
        except Exception as e:
            logger.error(f"Failed to log detector to MLflow: {e}")
    
    async def log_anomaly_results(
        self,
        anomaly_scores: np.ndarray,
        anomalies: np.ndarray,
        dataset: Dataset,
        detector: Detector
    ) -> None:
        """Log anomaly detection results to MLflow."""
        try:
            import mlflow
            
            if not self.mlflow_client:
                return
            
            # Calculate metrics
            anomaly_count = int(np.sum(anomalies))
            total_samples = len(anomalies)
            anomaly_rate = anomaly_count / total_samples if total_samples > 0 else 0
            
            metrics = {
                "anomaly_count": anomaly_count,
                "total_samples": total_samples,
                "anomaly_rate": anomaly_rate,
                "mean_anomaly_score": float(np.mean(anomaly_scores)),
                "max_anomaly_score": float(np.max(anomaly_scores)),
                "min_anomaly_score": float(np.min(anomaly_scores)),
                "std_anomaly_score": float(np.std(anomaly_scores))
            }
            
            await self.log_metrics(metrics)
            
            # Log score distribution
            score_percentiles = {
                f"score_p{p}": float(np.percentile(anomaly_scores, p))
                for p in [25, 50, 75, 90, 95, 99]
            }
            await self.log_metrics(score_percentiles)
            
            # Save detailed results as artifact
            results_df = pd.DataFrame({
                'anomaly_score': anomaly_scores,
                'is_anomaly': anomalies
            })
            
            results_file = "anomaly_results.csv"
            results_df.to_csv(results_file, index=False)
            mlflow.log_artifact(results_file)
            os.remove(results_file)
            
            logger.info("Logged anomaly detection results to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log anomaly results to MLflow: {e}")
    
    async def log_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Log custom artifacts to MLflow."""
        try:
            import mlflow
            
            if not self.mlflow_client:
                return
            
            for name, content in artifacts.items():
                if isinstance(content, (pd.DataFrame,)):
                    file_path = f"{name}.csv"
                    content.to_csv(file_path, index=False)
                    mlflow.log_artifact(file_path)
                    os.remove(file_path)
                elif isinstance(content, dict):
                    import json
                    file_path = f"{name}.json"
                    with open(file_path, 'w') as f:
                        json.dump(content, f, indent=2, default=str)
                    mlflow.log_artifact(file_path)
                    os.remove(file_path)
                elif isinstance(content, str):
                    file_path = f"{name}.txt"
                    with open(file_path, 'w') as f:
                        f.write(content)
                    mlflow.log_artifact(file_path)
                    os.remove(file_path)
            
            logger.debug(f"Logged {len(artifacts)} artifacts to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log artifacts to MLflow: {e}")
    
    async def end_run(self, status: str = "FINISHED") -> None:
        """End MLflow run."""
        try:
            import mlflow
            
            if not self.mlflow_client:
                return
            
            mlflow.end_run(status=status)
            logger.info("Ended MLflow run")
            
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    async def get_run_info(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific run."""
        try:
            if not self.mlflow_client:
                return None
            
            run = self.mlflow_client.get_run(run_id)
            
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags
            }
            
        except Exception as e:
            logger.error(f"Failed to get run info: {e}")
            return None
    
    async def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for runs in the experiment."""
        try:
            if not self.mlflow_client:
                return []
            
            runs = self.mlflow_client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )
            
            return [
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "params": run.data.params,
                    "metrics": run.data.metrics,
                    "tags": run.data.tags
                }
                for run in runs
            ]
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []


class MLflowModelRegistry:
    """MLflow model registry for detector model management."""
    
    def __init__(self, config: MLflowConfig):
        self.config = config
        self.mlflow_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize MLflow client."""
        try:
            from mlflow.tracking import MlflowClient
            import mlflow
            
            mlflow.set_tracking_uri(self.config.tracking_uri)
            if self.config.model_registry_uri:
                mlflow.set_registry_uri(self.config.model_registry_uri)
            
            self.mlflow_client = MlflowClient()
            
        except ImportError:
            logger.warning("MLflow not available for model registry")
            self.mlflow_client = None
    
    async def register_model(
        self,
        model_uri: str,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Register model in MLflow model registry."""
        try:
            if not self.mlflow_client:
                return None
            
            model_version = self.mlflow_client.create_model_version(
                name=name,
                source=model_uri,
                description=description,
                tags=tags
            )
            
            logger.info(f"Registered model {name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    async def promote_model(
        self,
        name: str,
        version: str,
        stage: str = "Production"
    ) -> bool:
        """Promote model to a specific stage."""
        try:
            if not self.mlflow_client:
                return False
            
            self.mlflow_client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Promoted model {name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    async def get_latest_model(
        self,
        name: str,
        stage: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get latest model version from registry."""
        try:
            if not self.mlflow_client:
                return None
            
            if stage:
                latest_versions = self.mlflow_client.get_latest_versions(
                    name=name,
                    stages=[stage]
                )
            else:
                latest_versions = self.mlflow_client.get_latest_versions(name=name)
            
            if latest_versions:
                version = latest_versions[0]
                return {
                    "name": version.name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "source": version.source,
                    "run_id": version.run_id,
                    "tags": version.tags,
                    "description": version.description
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest model: {e}")
            return None
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        try:
            if not self.mlflow_client:
                return []
            
            models = self.mlflow_client.list_registered_models()
            
            return [
                {
                    "name": model.name,
                    "description": model.description,
                    "tags": model.tags,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "run_id": v.run_id
                        }
                        for v in model.latest_versions
                    ]
                }
                for model in models
            ]
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


# Integration with workflow orchestrator
class MLflowWorkflowIntegration:
    """Integration between MLflow and workflow orchestration."""
    
    def __init__(self, mlflow_config: MLflowConfig):
        self.experiment_tracker = MLflowExperimentTracker(mlflow_config)
        self.model_registry = MLflowModelRegistry(mlflow_config)
        self.current_run_id = None
    
    async def start_workflow_run(
        self,
        workflow_id: str,
        workflow_config: Dict[str, Any]
    ) -> Optional[str]:
        """Start MLflow run for workflow execution."""
        run_name = f"workflow_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        tags = {
            "workflow_id": workflow_id,
            "workflow_type": "anomaly_detection",
            "orchestrator": workflow_config.get("engine", "unknown")
        }
        
        self.current_run_id = await self.experiment_tracker.start_run(
            run_name=run_name,
            tags=tags
        )
        
        # Log workflow configuration
        if self.current_run_id:
            await self.experiment_tracker.log_parameters(workflow_config)
        
        return self.current_run_id
    
    async def log_task_result(
        self,
        task_id: str,
        task_result: Dict[str, Any]
    ) -> None:
        """Log task execution result."""
        if not self.current_run_id:
            return
        
        # Log task metrics
        if task_result.get("status") == "success":
            task_metrics = {f"task_{task_id}_success": 1}
            
            # Extract specific metrics based on task type
            if "anomalies_detected" in task_result:
                task_metrics.update({
                    f"task_{task_id}_anomalies": task_result["anomalies_detected"],
                    f"task_{task_id}_anomaly_rate": task_result.get("anomaly_rate", 0.0)
                })
            
            if "rows_ingested" in task_result:
                task_metrics[f"task_{task_id}_rows"] = task_result["rows_ingested"]
            
            await self.experiment_tracker.log_metrics(task_metrics)
        else:
            # Log failure
            await self.experiment_tracker.log_metrics({f"task_{task_id}_success": 0})
    
    async def end_workflow_run(self, workflow_status: str = "FINISHED") -> None:
        """End MLflow run for workflow."""
        if self.current_run_id:
            await self.experiment_tracker.end_run(status=workflow_status)
            self.current_run_id = None
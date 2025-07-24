"""Stub implementations for MLOps operations.

These stubs implement the MLOps operations interfaces but provide basic
functionality when the mlops package is not available.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from anomaly_detection.domain.interfaces.mlops_operations import (
    MLOpsExperimentTrackingPort,
    MLOpsModelRegistryPort,
    ExperimentMetadata,
    RunMetadata,
    ModelVersionMetadata,
    ExperimentStatus,
    RunStatus,
    ModelStage,
)


class MLOpsExperimentTrackingStub(MLOpsExperimentTrackingPort):
    """Stub implementation for MLOps experiment tracking.
    
    This stub provides basic functionality when the mlops package
    is not available. It maintains in-memory storage for demonstration.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            "Using MLOps experiment tracking stub. MLOps package not available. "
            "Install mlops package for full functionality."
        )
        
        # In-memory storage for stub functionality
        self._experiments: Dict[str, Dict[str, Any]] = {}
        self._runs: Dict[str, Dict[str, Any]] = {}
    
    async def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new experiment (stub implementation)."""
        experiment_id = str(uuid.uuid4())
        
        self._experiments[experiment_id] = {
            "id": experiment_id,
            "name": name,
            "description": description,
            "tags": tags or {},
            "created_by": created_by,
            "created_at": datetime.now().isoformat(),
            "status": ExperimentStatus.ACTIVE.value
        }
        
        self._logger.info(f"Stub: Created experiment '{name}' with ID: {experiment_id}")
        return experiment_id
    
    async def start_run(
        self,
        experiment_id: str,
        run_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Start a new experiment run (stub implementation)."""
        run_id = str(uuid.uuid4())
        
        self._runs[run_id] = {
            "id": run_id,
            "experiment_id": experiment_id,
            "name": run_name or f"run_{run_id[:8]}",
            "parameters": parameters or {},
            "tags": tags or {},
            "metrics": {},
            "artifacts": {},
            "created_by": created_by,
            "started_at": datetime.now().isoformat(),
            "ended_at": None,
            "status": RunStatus.RUNNING.value
        }
        
        self._logger.info(f"Stub: Started run '{run_name}' with ID: {run_id}")
        return run_id
    
    async def log_parameter(self, run_id: str, key: str, value: Any) -> None:
        """Log a single parameter for a run (stub implementation)."""
        if run_id in self._runs:
            self._runs[run_id]["parameters"][key] = value
            self._logger.debug(f"Stub: Logged parameter {key}={value} for run {run_id}")
        else:
            self._logger.warning(f"Stub: Run {run_id} not found for parameter logging")
    
    async def log_parameters(self, run_id: str, parameters: Dict[str, Any]) -> None:
        """Log parameters for a run (stub implementation)."""
        if run_id in self._runs:
            self._runs[run_id]["parameters"].update(parameters)
            self._logger.debug(f"Stub: Logged parameters for run {run_id}: {parameters}")
        else:
            self._logger.warning(f"Stub: Run {run_id} not found for parameter logging")
    
    async def log_metric(self, run_id: str, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a single metric for a run (stub implementation)."""
        if run_id in self._runs:
            self._runs[run_id]["metrics"][key] = value
            self._logger.debug(f"Stub: Logged metric {key}={value} for run {run_id}")
        else:
            self._logger.warning(f"Stub: Run {run_id} not found for metric logging")
    
    async def log_metrics(self, run_id: str, metrics: Dict[str, float]) -> None:
        """Log metrics for a run (stub implementation)."""
        if run_id in self._runs:
            self._runs[run_id]["metrics"].update(metrics)
            self._logger.debug(f"Stub: Logged metrics for run {run_id}: {metrics}")
        else:
            self._logger.warning(f"Stub: Run {run_id} not found for metric logging")
    
    async def log_artifact(self, run_id: str, artifact_name: str, artifact_path: str) -> None:
        """Log a single artifact for a run (stub implementation)."""
        if run_id in self._runs:
            self._runs[run_id]["artifacts"][artifact_name] = artifact_path
            self._logger.debug(f"Stub: Logged artifact {artifact_name} for run {run_id}")
        else:
            self._logger.warning(f"Stub: Run {run_id} not found for artifact logging")
    
    async def log_artifacts(self, run_id: str, artifacts: Dict[str, str]) -> None:
        """Log artifacts for a run (stub implementation)."""
        if run_id in self._runs:
            self._runs[run_id]["artifacts"].update(artifacts)
            self._logger.debug(f"Stub: Logged artifacts for run {run_id}: {artifacts}")
        else:
            self._logger.warning(f"Stub: Run {run_id} not found for artifact logging")
    
    async def end_run(self, run_id: str, status: RunStatus) -> None:
        """End an experiment run (stub implementation)."""
        if run_id in self._runs:
            self._runs[run_id]["status"] = status.value
            self._runs[run_id]["ended_at"] = datetime.now().isoformat()
            self._logger.info(f"Stub: Ended run {run_id} with status: {status.value}")
        else:
            self._logger.warning(f"Stub: Run {run_id} not found for ending")
    
    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Get experiment metadata (stub implementation)."""
        if experiment_id in self._experiments:
            exp_data = self._experiments[experiment_id]
            return ExperimentMetadata(
                experiment_id=exp_data["id"],
                name=exp_data["name"],
                description=exp_data["description"],
                tags=exp_data["tags"],
                created_by=exp_data["created_by"],
                created_at=datetime.fromisoformat(exp_data["created_at"]),
                status=ExperimentStatus(exp_data["status"])
            )
        return None
    
    async def get_run(self, run_id: str) -> Optional[RunMetadata]:
        """Get run metadata (stub implementation)."""
        if run_id in self._runs:
            run_data = self._runs[run_id]
            return RunMetadata(
                run_id=run_data["id"],
                experiment_id=run_data["experiment_id"],
                name=run_data["name"],
                parameters=run_data["parameters"],
                metrics=run_data["metrics"],
                artifacts=run_data["artifacts"],
                tags=run_data["tags"],
                created_by=run_data["created_by"],
                started_at=datetime.fromisoformat(run_data["started_at"]),
                ended_at=datetime.fromisoformat(run_data["ended_at"]) if run_data["ended_at"] else None,
                status=RunStatus(run_data["status"])
            )
        return None
    
    async def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        created_by: Optional[str] = None
    ) -> List[ExperimentMetadata]:
        """List experiments (stub implementation)."""
        experiments = []
        
        for exp_data in self._experiments.values():
            # Apply filters
            if status and ExperimentStatus(exp_data["status"]) != status:
                continue
            if created_by and exp_data["created_by"] != created_by:
                continue
            
            experiments.append(ExperimentMetadata(
                experiment_id=exp_data["id"],
                name=exp_data["name"],
                description=exp_data["description"],
                tags=exp_data["tags"],
                created_by=exp_data["created_by"],
                created_at=datetime.fromisoformat(exp_data["created_at"]),
                status=ExperimentStatus(exp_data["status"])
            ))
        
        return experiments
    
    async def list_runs(
        self,
        experiment_id: Optional[str] = None,
        status: Optional[RunStatus] = None
    ) -> List[RunMetadata]:
        """List runs (stub implementation)."""
        runs = []
        
        for run_data in self._runs.values():
            # Apply filters
            if experiment_id and run_data["experiment_id"] != experiment_id:
                continue
            if status and RunStatus(run_data["status"]) != status:
                continue
            
            runs.append(RunMetadata(
                run_id=run_data["id"],
                experiment_id=run_data["experiment_id"],
                name=run_data["name"],
                parameters=run_data["parameters"],
                metrics=run_data["metrics"],
                artifacts=run_data["artifacts"],
                tags=run_data["tags"],
                created_by=run_data["created_by"],
                started_at=datetime.fromisoformat(run_data["started_at"]),
                ended_at=datetime.fromisoformat(run_data["ended_at"]) if run_data["ended_at"] else None,
                status=RunStatus(run_data["status"])
            ))
        
        return runs


class MLOpsModelRegistryStub(MLOpsModelRegistryPort):
    """Stub implementation for MLOps model registry.
    
    This stub provides basic functionality when the mlops package
    is not available. It maintains in-memory storage for demonstration.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            "Using MLOps model registry stub. MLOps package not available. "
            "Install mlops package for full functionality."
        )
        
        # In-memory storage for stub functionality
        self._models: Dict[str, Dict[str, Any]] = {}
        self._model_versions: Dict[str, List[Dict[str, Any]]] = {}
    
    async def register_model(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Register a new model (stub implementation)."""
        model_id = str(uuid.uuid4())
        
        self._models[model_id] = {
            "id": model_id,
            "name": name,
            "description": description,
            "tags": tags or {},
            "created_by": created_by,
            "created_at": datetime.now().isoformat()
        }
        
        self._model_versions[model_id] = []
        
        self._logger.info(f"Stub: Registered model '{name}' with ID: {model_id}")
        return model_id
    
    async def create_model_version(
        self,
        model_id: str,
        version: str,
        run_id: str,
        source_path: str,
        description: str = "",
        performance_metrics: Optional[Dict[str, float]] = None,
        deployment_config: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new model version (stub implementation)."""
        version_id = str(uuid.uuid4())
        
        version_data = {
            "id": version_id,
            "model_id": model_id,
            "version": version,
            "run_id": run_id,
            "source_path": source_path,
            "description": description,
            "performance_metrics": performance_metrics or {},
            "deployment_config": deployment_config or {},
            "tags": tags or {},
            "created_by": created_by,
            "created_at": datetime.now().isoformat(),
            "stage": ModelStage.STAGING.value
        }
        
        if model_id in self._model_versions:
            self._model_versions[model_id].append(version_data)
        else:
            self._model_versions[model_id] = [version_data]
        
        self._logger.info(f"Stub: Created model version {version} for model {model_id}")
        return version_id
    
    async def transition_model_version_stage(
        self,
        model_id: str,
        version: str,
        stage: ModelStage
    ) -> None:
        """Transition model version to different stage (stub implementation)."""
        if model_id in self._model_versions:
            for version_data in self._model_versions[model_id]:
                if version_data["version"] == version:
                    version_data["stage"] = stage.value
                    self._logger.info(f"Stub: Transitioned model {model_id} version {version} to {stage.value}")
                    return
        
        self._logger.warning(f"Stub: Model version {version} not found for model {model_id}")
    
    async def get_model_version(
        self,
        model_id: str,
        version: str
    ) -> Optional[ModelVersionMetadata]:
        """Get model version metadata (stub implementation)."""
        if model_id in self._model_versions:
            for version_data in self._model_versions[model_id]:
                if version_data["version"] == version:
                    return ModelVersionMetadata(
                        version_id=version_data["id"],
                        model_id=version_data["model_id"],
                        version=version_data["version"],
                        run_id=version_data["run_id"],
                        source_path=version_data["source_path"],
                        description=version_data["description"],
                        performance_metrics=version_data["performance_metrics"],
                        deployment_config=version_data["deployment_config"],
                        tags=version_data["tags"],
                        created_by=version_data["created_by"],
                        created_at=datetime.fromisoformat(version_data["created_at"]),
                        stage=ModelStage(version_data["stage"])
                    )
        return None
    
    async def list_model_versions(
        self,
        model_id: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersionMetadata]:
        """List model versions (stub implementation)."""
        versions = []
        
        if model_id in self._model_versions:
            for version_data in self._model_versions[model_id]:
                # Apply filter
                if stage and ModelStage(version_data["stage"]) != stage:
                    continue
                
                versions.append(ModelVersionMetadata(
                    version_id=version_data["id"],
                    model_id=version_data["model_id"],
                    version=version_data["version"],
                    run_id=version_data["run_id"],
                    source_path=version_data["source_path"],
                    description=version_data["description"],
                    performance_metrics=version_data["performance_metrics"],
                    deployment_config=version_data["deployment_config"],
                    tags=version_data["tags"],
                    created_by=version_data["created_by"],
                    created_at=datetime.fromisoformat(version_data["created_at"]),
                    stage=ModelStage(version_data["stage"])
                ))
        
        return versions
    
    async def delete_model_version(self, model_id: str, version: str) -> bool:
        """Delete a model version (stub implementation)."""
        if model_id in self._model_versions:
            versions = self._model_versions[model_id]
            for i, version_data in enumerate(versions):
                if version_data["version"] == version:
                    del versions[i]
                    self._logger.info(f"Stub: Deleted model version {version} for model {model_id}")
                    return True
        
        return False
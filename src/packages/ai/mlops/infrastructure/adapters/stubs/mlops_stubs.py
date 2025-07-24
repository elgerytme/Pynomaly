"""Stub implementations for MLOps operations.

These stubs implement the MLOps operations interfaces but provide basic
functionality when the mlops package is not available.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai.mlops.domain.interfaces.mlops_operations import (
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
    """Stub implementation for MLOps experiment tracking operations.
    
    This stub provides basic functionality when the mlops package is not
    available. It logs warnings and maintains simple in-memory storage.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            "Using MLOps experiment tracking stub. MLOps package not available. "
            "Install mlops package for full functionality."
        )
        
        # Simple in-memory storage
        self._experiments: Dict[str, ExperimentMetadata] = {}
        self._runs: Dict[str, RunMetadata] = {}
    
    async def create_experiment(
        self, 
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Stub implementation of experiment creation."""
        experiment_id = str(uuid.uuid4())
        
        experiment = ExperimentMetadata(
            experiment_id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.ACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by=created_by,
            tags=tags or {},
            parameters={}
        )
        
        self._experiments[experiment_id] = experiment
        
        self._logger.info(f"Stub: Created experiment '{name}' with ID: {experiment_id}")
        return experiment_id
    
    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Stub implementation of experiment retrieval."""
        return self._experiments.get(experiment_id)
    
    async def list_experiments(
        self, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ExperimentMetadata]:
        """Stub implementation of experiment listing."""
        experiments = list(self._experiments.values())
        
        # Apply simple filtering if provided
        if filters:
            filtered_experiments = []
            for exp in experiments:
                match = True
                if "status" in filters and exp.status.value != filters["status"]:
                    match = False
                if "created_by" in filters and exp.created_by != filters["created_by"]:
                    match = False
                if match:
                    filtered_experiments.append(exp)
            experiments = filtered_experiments
        
        return experiments
    
    async def start_run(
        self,
        experiment_id: str,
        run_name: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Stub implementation of run creation."""
        run_id = str(uuid.uuid4())
        
        run = RunMetadata(
            run_id=run_id,
            experiment_id=experiment_id,
            name=run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status=RunStatus.RUNNING,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            created_by=created_by,
            parameters=parameters or {},
            metrics={},
            artifacts={},
            tags=tags or {}
        )
        
        self._runs[run_id] = run
        
        self._logger.info(f"Stub: Started run '{run.name}' with ID: {run_id}")
        return run_id
    
    async def end_run(self, run_id: str, status: RunStatus = RunStatus.COMPLETED) -> None:
        """Stub implementation of run ending."""
        if run_id in self._runs:
            run = self._runs[run_id]
            end_time = datetime.now()
            duration = end_time - run.start_time
            
            # Update run metadata
            updated_run = RunMetadata(
                run_id=run.run_id,
                experiment_id=run.experiment_id,
                name=run.name,
                status=status,
                start_time=run.start_time,
                end_time=end_time,
                duration_seconds=duration.total_seconds(),
                created_by=run.created_by,
                parameters=run.parameters,
                metrics=run.metrics,
                artifacts=run.artifacts,
                tags=run.tags
            )
            
            self._runs[run_id] = updated_run
            self._logger.info(f"Stub: Ended run {run_id} with status: {status.value}")
    
    async def log_parameter(self, run_id: str, key: str, value: Any) -> None:
        """Stub implementation of parameter logging."""
        if run_id in self._runs:
            self._runs[run_id].parameters[key] = value
            self._logger.debug(f"Stub: Logged parameter {key}={value} for run {run_id}")
    
    async def log_parameters(self, run_id: str, parameters: Dict[str, Any]) -> None:
        """Stub implementation of parameters logging."""
        if run_id in self._runs:
            self._runs[run_id].parameters.update(parameters)
            self._logger.debug(f"Stub: Logged {len(parameters)} parameters for run {run_id}")
    
    async def log_metric(
        self, 
        run_id: str, 
        key: str, 
        value: float, 
        step: Optional[int] = None
    ) -> None:
        """Stub implementation of metric logging."""
        if run_id in self._runs:
            metric_key = f"{key}_step_{step}" if step is not None else key
            self._runs[run_id].metrics[metric_key] = value
            self._logger.debug(f"Stub: Logged metric {key}={value} for run {run_id}")
    
    async def log_metrics(
        self, 
        run_id: str, 
        metrics: Dict[str, float], 
        step: Optional[int] = None
    ) -> None:
        """Stub implementation of metrics logging."""
        if run_id in self._runs:
            for key, value in metrics.items():
                metric_key = f"{key}_step_{step}" if step is not None else key
                self._runs[run_id].metrics[metric_key] = value
            self._logger.debug(f"Stub: Logged {len(metrics)} metrics for run {run_id}")
    
    async def log_artifact(
        self, 
        run_id: str, 
        artifact_path: str, 
        artifact_name: str = ""
    ) -> str:
        """Stub implementation of artifact logging."""
        if run_id in self._runs:
            artifact_uri = f"file://{artifact_path}"
            name = artifact_name or artifact_path.split("/")[-1]
            self._runs[run_id].artifacts[name] = artifact_uri
            self._logger.debug(f"Stub: Logged artifact {name} for run {run_id}")
            return artifact_uri
        return ""
    
    async def get_run(self, run_id: str) -> Optional[RunMetadata]:
        """Stub implementation of run retrieval."""
        return self._runs.get(run_id)
    
    async def list_runs(
        self, 
        experiment_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RunMetadata]:
        """Stub implementation of run listing."""
        runs = [run for run in self._runs.values() if run.experiment_id == experiment_id]
        
        # Apply simple filtering if provided
        if filters:
            filtered_runs = []
            for run in runs:
                match = True
                if "status" in filters and run.status.value != filters["status"]:
                    match = False
                if "created_by" in filters and run.created_by != filters["created_by"]:
                    match = False
                if match:
                    filtered_runs.append(run)
            runs = filtered_runs
        
        return runs


class MLOpsModelRegistryStub(MLOpsModelRegistryPort):
    """Stub implementation for MLOps model registry operations.
    
    This stub provides basic functionality when the mlops package is not
    available. It logs warnings and maintains simple in-memory storage.
    """
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.warning(
            "Using MLOps model registry stub. MLOps package not available. "
            "Install mlops package for full functionality."
        )
        
        # Simple in-memory storage
        self._models: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, ModelVersionMetadata] = {}
    
    async def register_model(
        self,
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Stub implementation of model registration."""
        model_id = str(uuid.uuid4())
        
        model_info = {
            "id": model_id,
            "name": name,
            "description": description,
            "tags": tags or {},
            "created_by": created_by,
            "created_at": datetime.now(),
        }
        
        self._models[model_id] = model_info
        
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
        """Stub implementation of model version creation."""
        version_id = str(uuid.uuid4())
        
        version_metadata = ModelVersionMetadata(
            version_id=version_id,
            model_id=model_id,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            run_id=run_id,
            source_path=source_path,
            performance_metrics=performance_metrics or {},
            deployment_config=deployment_config or {},
            tags=tags or {},
        )
        
        self._versions[version_id] = version_metadata
        
        self._logger.info(f"Stub: Created version {version} for model {model_id}")
        return version_id
    
    async def get_model_version(
        self, 
        model_id: str, 
        version: str
    ) -> Optional[ModelVersionMetadata]:
        """Stub implementation of model version retrieval."""
        for version_metadata in self._versions.values():
            if version_metadata.model_id == model_id and version_metadata.version == version:
                return version_metadata
        return None
    
    async def list_model_versions(
        self, 
        model_id: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelVersionMetadata]:
        """Stub implementation of model version listing."""
        versions = [v for v in self._versions.values() if v.model_id == model_id]
        
        if stage:
            versions = [v for v in versions if v.stage == stage]
        
        return versions
    
    async def transition_model_stage(
        self,
        model_id: str,
        version: str,
        stage: ModelStage,
        comment: str = "",
        archive_existing: bool = True
    ) -> None:
        """Stub implementation of model stage transition."""
        version_metadata = await self.get_model_version(model_id, version)
        
        if version_metadata:
            # Update the stage
            updated_metadata = ModelVersionMetadata(
                version_id=version_metadata.version_id,
                model_id=version_metadata.model_id,
                version=version_metadata.version,
                stage=stage,
                created_at=version_metadata.created_at,
                created_by=version_metadata.created_by,
                description=version_metadata.description,
                run_id=version_metadata.run_id,
                source_path=version_metadata.source_path,
                performance_metrics=version_metadata.performance_metrics,
                deployment_config=version_metadata.deployment_config,
                tags=version_metadata.tags,
            )
            
            self._versions[version_metadata.version_id] = updated_metadata
            
            self._logger.info(
                f"Stub: Transitioned model {model_id} version {version} to stage {stage.value}"
            )
    
    async def get_latest_model_version(
        self, 
        model_id: str, 
        stage: Optional[ModelStage] = None
    ) -> Optional[ModelVersionMetadata]:
        """Stub implementation of latest model version retrieval."""
        versions = await self.list_model_versions(model_id, stage)
        
        if not versions:
            return None
        
        # Return the most recently created version
        latest_version = max(versions, key=lambda v: v.created_at)
        return latest_version
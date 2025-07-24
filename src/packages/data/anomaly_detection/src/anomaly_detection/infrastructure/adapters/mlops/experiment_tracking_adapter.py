"""MLOps Experiment Tracking Adapter.

This adapter implements the MLOpsExperimentTrackingPort interface by integrating
with the mlops package. It translates between anomaly detection domain concepts
and the MLOps package's experiment tracking APIs.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from anomaly_detection.domain.interfaces.mlops_operations import (
    MLOpsExperimentTrackingPort,
    ExperimentMetadata,
    RunMetadata,
    ExperimentStatus,
    RunStatus,
    ExperimentCreationError,
    ExperimentRetrievalError,
    ExperimentQueryError,
    RunCreationError,
    RunUpdateError,
    RunRetrievalError,
    RunQueryError,
    ParameterLoggingError,
    MetricLoggingError,
    ArtifactLoggingError,
)

# MLOps package imports
try:
    from mlops.domain.services.experiment_tracking_service import ExperimentTrackingService
    from mlops.domain.entities.experiment import Experiment as MLOpsExperiment
    from mlops.domain.entities.experiment import ExperimentRun as MLOpsRun
    from mlops.application.use_cases.create_experiment_use_case import CreateExperimentUseCase
    from mlops.application.use_cases.run_experiment_use_case import RunExperimentUseCase
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False
    # Create type stubs when MLOps package is not available
    ExperimentTrackingService = Any
    MLOpsExperiment = Any
    MLOpsRun = Any
    CreateExperimentUseCase = Any
    RunExperimentUseCase = Any


class MLOpsExperimentTrackingAdapter(MLOpsExperimentTrackingPort):
    """Adapter for MLOps experiment tracking operations.
    
    This adapter integrates the anomaly detection domain with the
    MLOps package, providing experiment and run tracking capabilities.
    """

    def __init__(
        self,
        experiment_tracking_service: ExperimentTrackingService,
        create_experiment_use_case: CreateExperimentUseCase,
        run_experiment_use_case: RunExperimentUseCase,
    ):
        """Initialize the MLOps experiment tracking adapter.
        
        Args:
            experiment_tracking_service: Experiment tracking service from MLOps package
            create_experiment_use_case: Create experiment use case implementation
            run_experiment_use_case: Run experiment use case implementation
        """
        if not MLOPS_AVAILABLE:
            raise ImportError(
                "mlops package is not available. "
                "Please install it to use this adapter."
            )
        
        self._experiment_service = experiment_tracking_service
        self._create_experiment_use_case = create_experiment_use_case
        self._run_experiment_use_case = run_experiment_use_case
        self._logger = logging.getLogger(__name__)
        
        # Status mapping from MLOps to domain format
        self._experiment_status_mapping = {
            "active": ExperimentStatus.ACTIVE,
            "completed": ExperimentStatus.COMPLETED,
            "failed": ExperimentStatus.FAILED,
            "cancelled": ExperimentStatus.CANCELLED,
        }
        
        self._run_status_mapping = {
            "running": RunStatus.RUNNING,
            "completed": RunStatus.COMPLETED,
            "failed": RunStatus.FAILED,
            "killed": RunStatus.KILLED,
        }
        
        self._logger.info("MLOpsExperimentTrackingAdapter initialized successfully")

    async def create_experiment(
        self, 
        name: str,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new experiment for tracking.
        
        Args:
            name: Experiment name
            description: Experiment description
            tags: Optional tags for categorization
            created_by: User creating the experiment
            
        Returns:
            Unique experiment identifier
            
        Raises:
            ExperimentCreationError: If experiment creation fails
        """
        try:
            # Create request for MLOps package
            experiment_request = {
                "name": name,
                "description": description,
                "tags": tags or {},
                "created_by": created_by,
                "metadata": {
                    "source": "anomaly_detection",
                    "created_at": datetime.now().isoformat(),
                }
            }
            
            self._logger.info(f"Creating experiment: {name}")
            
            # Create experiment through MLOps package
            mlops_experiment = await self._create_experiment_use_case.execute(experiment_request)
            
            experiment_id = str(mlops_experiment.id)
            
            self._logger.info(f"Experiment created successfully with ID: {experiment_id}")
            
            return experiment_id
            
        except Exception as e:
            self._logger.error(f"Failed to create experiment '{name}': {str(e)}")
            raise ExperimentCreationError(f"Experiment creation failed: {str(e)}") from e

    async def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Retrieve experiment metadata.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            Experiment metadata if found, None otherwise
            
        Raises:
            ExperimentRetrievalError: If retrieval fails
        """
        try:
            # Get experiment from MLOps package
            mlops_experiment = await self._experiment_service.get_experiment(
                uuid.UUID(experiment_id)
            )
            
            if not mlops_experiment:
                return None
            
            # Convert to domain format
            experiment_metadata = self._convert_experiment_to_domain_format(mlops_experiment)
            
            return experiment_metadata
            
        except Exception as e:
            self._logger.error(f"Failed to retrieve experiment {experiment_id}: {str(e)}")
            raise ExperimentRetrievalError(f"Experiment retrieval failed: {str(e)}") from e

    async def list_experiments(
        self, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ExperimentMetadata]:
        """List experiments with optional filtering.
        
        Args:
            filters: Optional filters (status, created_by, tags, etc.)
            
        Returns:
            List of experiment metadata
            
        Raises:
            ExperimentQueryError: If listing fails
        """
        try:
            # Convert filters to MLOps package format
            mlops_filters = self._convert_filters_to_mlops_format(filters)
            
            # List experiments through MLOps package
            mlops_experiments = await self._experiment_service.list_experiments(mlops_filters)
            
            # Convert to domain format
            experiments = [
                self._convert_experiment_to_domain_format(exp)
                for exp in mlops_experiments
            ]
            
            self._logger.info(f"Retrieved {len(experiments)} experiments")
            
            return experiments
            
        except Exception as e:
            self._logger.error(f"Failed to list experiments: {str(e)}")
            raise ExperimentQueryError(f"Experiment listing failed: {str(e)}") from e

    async def start_run(
        self,
        experiment_id: str,
        run_name: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "system"
    ) -> str:
        """Start a new experiment run.
        
        Args:
            experiment_id: Parent experiment identifier
            run_name: Name for the run
            parameters: Initial parameters
            tags: Optional tags
            created_by: User starting the run
            
        Returns:
            Unique run identifier
            
        Raises:
            RunCreationError: If run creation fails
        """
        try:
            # Create run request for MLOps package
            run_request = {
                "experiment_id": uuid.UUID(experiment_id),
                "run_name": run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "parameters": parameters or {},
                "tags": tags or {},
                "created_by": created_by,
                "metadata": {
                    "source": "anomaly_detection",
                    "started_at": datetime.now().isoformat(),
                }
            }
            
            self._logger.info(f"Starting run for experiment: {experiment_id}")
            
            # Start run through MLOps package
            mlops_run = await self._run_experiment_use_case.start_run(run_request)
            
            run_id = str(mlops_run.id)
            
            self._logger.info(f"Run started successfully with ID: {run_id}")
            
            return run_id
            
        except Exception as e:
            self._logger.error(f"Failed to start run for experiment {experiment_id}: {str(e)}")
            raise RunCreationError(f"Run creation failed: {str(e)}") from e

    async def end_run(self, run_id: str, status: RunStatus = RunStatus.COMPLETED) -> None:
        """End an experiment run.
        
        Args:
            run_id: Run identifier
            status: Final status of the run
            
        Raises:
            RunUpdateError: If run ending fails
        """
        try:
            # Convert status to MLOps format
            mlops_status = self._convert_run_status_to_mlops_format(status)
            
            # End run through MLOps package
            await self._run_experiment_use_case.end_run(
                uuid.UUID(run_id), 
                mlops_status
            )
            
            self._logger.info(f"Run {run_id} ended with status: {status.value}")
            
        except Exception as e:
            self._logger.error(f"Failed to end run {run_id}: {str(e)}")
            raise RunUpdateError(f"Run ending failed: {str(e)}") from e

    async def log_parameter(self, run_id: str, key: str, value: Any) -> None:
        """Log a parameter for a run.
        
        Args:
            run_id: Run identifier
            key: Parameter name
            value: Parameter value
            
        Raises:
            ParameterLoggingError: If parameter logging fails
        """
        try:
            await self._experiment_service.log_parameter(
                uuid.UUID(run_id), key, value
            )
            
            self._logger.debug(f"Logged parameter {key}={value} for run {run_id}")
            
        except Exception as e:
            self._logger.error(f"Failed to log parameter for run {run_id}: {str(e)}")
            raise ParameterLoggingError(f"Parameter logging failed: {str(e)}") from e

    async def log_parameters(self, run_id: str, parameters: Dict[str, Any]) -> None:
        """Log multiple parameters for a run.
        
        Args:
            run_id: Run identifier
            parameters: Dictionary of parameters
            
        Raises:
            ParameterLoggingError: If parameter logging fails
        """
        try:
            await self._experiment_service.log_parameters(
                uuid.UUID(run_id), parameters
            )
            
            self._logger.debug(f"Logged {len(parameters)} parameters for run {run_id}")
            
        except Exception as e:
            self._logger.error(f"Failed to log parameters for run {run_id}: {str(e)}")
            raise ParameterLoggingError(f"Parameters logging failed: {str(e)}") from e

    async def log_metric(
        self, 
        run_id: str, 
        key: str, 
        value: float, 
        step: Optional[int] = None
    ) -> None:
        """Log a metric for a run.
        
        Args:
            run_id: Run identifier
            key: Metric name
            value: Metric value
            step: Optional step number
            
        Raises:
            MetricLoggingError: If metric logging fails
        """
        try:
            await self._experiment_service.log_metric(
                uuid.UUID(run_id), key, value, step
            )
            
            self._logger.debug(f"Logged metric {key}={value} for run {run_id}")
            
        except Exception as e:
            self._logger.error(f"Failed to log metric for run {run_id}: {str(e)}")
            raise MetricLoggingError(f"Metric logging failed: {str(e)}") from e

    async def log_metrics(
        self, 
        run_id: str, 
        metrics: Dict[str, float], 
        step: Optional[int] = None
    ) -> None:
        """Log multiple metrics for a run.
        
        Args:
            run_id: Run identifier
            metrics: Dictionary of metrics
            step: Optional step number
            
        Raises:
            MetricLoggingError: If metric logging fails
        """
        try:
            await self._experiment_service.log_metrics(
                uuid.UUID(run_id), metrics, step
            )
            
            self._logger.debug(f"Logged {len(metrics)} metrics for run {run_id}")
            
        except Exception as e:
            self._logger.error(f"Failed to log metrics for run {run_id}: {str(e)}")
            raise MetricLoggingError(f"Metrics logging failed: {str(e)}") from e

    async def log_artifact(
        self, 
        run_id: str, 
        artifact_path: str, 
        artifact_name: str = ""
    ) -> str:
        """Log an artifact for a run.
        
        Args:
            run_id: Run identifier
            artifact_path: Path to the artifact
            artifact_name: Optional name for the artifact
            
        Returns:
            URI of the logged artifact
            
        Raises:
            ArtifactLoggingError: If artifact logging fails
        """
        try:
            artifact_uri = await self._experiment_service.log_artifact(
                uuid.UUID(run_id), artifact_path, artifact_name
            )
            
            self._logger.debug(f"Logged artifact {artifact_name} for run {run_id}")
            
            return artifact_uri
            
        except Exception as e:
            self._logger.error(f"Failed to log artifact for run {run_id}: {str(e)}")
            raise ArtifactLoggingError(f"Artifact logging failed: {str(e)}") from e

    async def get_run(self, run_id: str) -> Optional[RunMetadata]:
        """Retrieve run metadata.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run metadata if found, None otherwise
            
        Raises:
            RunRetrievalError: If retrieval fails
        """
        try:
            # Get run from MLOps package
            mlops_run = await self._experiment_service.get_run(uuid.UUID(run_id))
            
            if not mlops_run:
                return None
            
            # Convert to domain format
            run_metadata = self._convert_run_to_domain_format(mlops_run)
            
            return run_metadata
            
        except Exception as e:
            self._logger.error(f"Failed to retrieve run {run_id}: {str(e)}")
            raise RunRetrievalError(f"Run retrieval failed: {str(e)}") from e

    async def list_runs(
        self, 
        experiment_id: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RunMetadata]:
        """List runs for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            filters: Optional filters (status, created_by, etc.)
            
        Returns:
            List of run metadata
            
        Raises:
            RunQueryError: If listing fails
        """
        try:
            # Convert filters to MLOps package format
            mlops_filters = self._convert_filters_to_mlops_format(filters)
            mlops_filters["experiment_id"] = uuid.UUID(experiment_id)
            
            # List runs through MLOps package
            mlops_runs = await self._experiment_service.list_runs(mlops_filters)
            
            # Convert to domain format
            runs = [
                self._convert_run_to_domain_format(run)
                for run in mlops_runs
            ]
            
            self._logger.info(f"Retrieved {len(runs)} runs for experiment {experiment_id}")
            
            return runs
            
        except Exception as e:
            self._logger.error(f"Failed to list runs for experiment {experiment_id}: {str(e)}")
            raise RunQueryError(f"Run listing failed: {str(e)}") from e

    def _convert_experiment_to_domain_format(self, mlops_experiment: MLOpsExperiment) -> ExperimentMetadata:
        """Convert MLOps experiment to domain format."""
        return ExperimentMetadata(
            experiment_id=str(mlops_experiment.id),
            name=mlops_experiment.name,
            description=mlops_experiment.description,
            status=self._experiment_status_mapping.get(
                mlops_experiment.status, ExperimentStatus.ACTIVE
            ),
            created_at=mlops_experiment.created_at,
            updated_at=mlops_experiment.updated_at,
            created_by=mlops_experiment.created_by,
            tags=getattr(mlops_experiment, 'tags', {}),
            parameters=getattr(mlops_experiment, 'parameters', {}),
        )

    def _convert_run_to_domain_format(self, mlops_run: MLOpsRun) -> RunMetadata:
        """Convert MLOps run to domain format."""
        duration_seconds = None
        if mlops_run.end_time and mlops_run.start_time:
            duration = mlops_run.end_time - mlops_run.start_time
            duration_seconds = duration.total_seconds()
        
        return RunMetadata(
            run_id=str(mlops_run.id),
            experiment_id=str(mlops_run.experiment_id),
            name=mlops_run.name,
            status=self._run_status_mapping.get(
                mlops_run.status, RunStatus.RUNNING
            ),
            start_time=mlops_run.start_time,
            end_time=mlops_run.end_time,
            duration_seconds=duration_seconds,
            created_by=mlops_run.created_by,
            parameters=getattr(mlops_run, 'parameters', {}),
            metrics=getattr(mlops_run, 'metrics', {}),
            artifacts=getattr(mlops_run, 'artifacts', {}),
            tags=getattr(mlops_run, 'tags', {}),
        )

    def _convert_filters_to_mlops_format(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert domain filters to MLOps package format."""
        if not filters:
            return {}
        
        mlops_filters = {}
        
        for key, value in filters.items():
            if key == "status" and isinstance(value, (ExperimentStatus, RunStatus)):
                mlops_filters[key] = value.value
            else:
                mlops_filters[key] = value
        
        return mlops_filters

    def _convert_run_status_to_mlops_format(self, status: RunStatus) -> str:
        """Convert domain run status to MLOps format."""
        status_mapping = {
            RunStatus.RUNNING: "running",
            RunStatus.COMPLETED: "completed",
            RunStatus.FAILED: "failed",
            RunStatus.KILLED: "killed",
        }
        
        return status_mapping.get(status, "completed")
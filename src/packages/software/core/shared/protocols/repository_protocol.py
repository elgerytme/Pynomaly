"""Repository protocol for infrastructure persistence."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import UUID

from monorepo.domain.entities import (
    Alert,
    AlertNotification,
    DataCollection,
    DetectionResult,
    Detector,
    Experiment,
    ExperimentRun,
    Processor,
    ModelVersion,
    Pipeline,
    PipelineRun,
)

T = TypeVar("T")


@runtime_checkable
class RepositoryProtocol(Protocol[T]):
    """Base protocol for repository implementations.

    This protocol defines the interface for persistence operations
    on domain entities.
    """

    async def save(self, entity: T) -> None:
        """Save an entity to the repository.

        Args:
            entity: The entity to save
        """
        ...

    async def find_by_id(self, entity_id: UUID) -> T | None:
        """Find an entity by its ID.

        Args:
            entity_id: The UUID of the entity

        Returns:
            The entity if found, None otherwise
        """
        ...

    async def find_all(self) -> list[T]:
        """Find all entities in the repository.

        Returns:
            List of all entities
        """
        ...

    async def delete(self, entity_id: UUID) -> bool:
        """Delete an entity by its ID.

        Args:
            entity_id: The UUID of the entity to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    async def exists(self, entity_id: UUID) -> bool:
        """Check if an entity exists.

        Args:
            entity_id: The UUID to check

        Returns:
            True if exists, False otherwise
        """
        ...

    async def count(self) -> int:
        """Count total number of entities.

        Returns:
            Number of entities in repository
        """
        ...


@runtime_checkable
class DetectorRepositoryProtocol(RepositoryProtocol[Detector], Protocol):
    """Protocol for detector repository implementations."""

    async def find_by_name(self, name: str) -> Detector | None:
        """Find a detector by name.

        Args:
            name: The detector name

        Returns:
            The detector if found
        """
        ...

    async def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
        """Find all detectors using a specific algorithm.

        Args:
            algorithm_name: Name of the algorithm

        Returns:
            List of detectors using that algorithm
        """
        ...

    async def find_fitted(self) -> list[Detector]:
        """Find all fitted detectors.

        Returns:
            List of fitted detectors
        """
        ...

    async def save_processor_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        """Save the trained processor artifact.

        Args:
            detector_id: ID of the detector
            artifact: Serialized processor data
        """
        ...

    async def load_processor_artifact(self, detector_id: UUID) -> bytes | None:
        """Load the trained processor artifact.

        Args:
            detector_id: ID of the detector

        Returns:
            Serialized processor data if found
        """
        ...


@runtime_checkable
class DatasetRepositoryProtocol(RepositoryProtocol[Dataset], Protocol):
    """Protocol for data_collection repository implementations."""

    async def find_by_name(self, name: str) -> DataCollection | None:
        """Find a data_collection by name.

        Args:
            name: The data_collection name

        Returns:
            The data_collection if found
        """
        ...

    async def find_by_metadata(self, key: str, value: Any) -> list[DataCollection]:
        """Find datasets by metadata key-value pair.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            List of matching datasets
        """
        ...

    async def save_data(self, data_collection_id: UUID, format: str = "parquet") -> str:
        """Save data_collection data to persistent storage.

        Args:
            data_collection_id: ID of the data_collection
            format: Storage format (parquet, csv, etc.)

        Returns:
            Path or URI where data was saved
        """
        ...

    async def load_data(self, data_collection_id: UUID) -> DataCollection | None:
        """Load data_collection with its data from storage.

        Args:
            data_collection_id: ID of the data_collection

        Returns:
            DataCollection with data if found
        """
        ...


@runtime_checkable
class DetectionResultRepositoryProtocol(RepositoryProtocol[DetectionResult], Protocol):
    """Protocol for processing result repository implementations."""

    async def find_by_detector(self, detector_id: UUID) -> list[DetectionResult]:
        """Find all results from a specific detector.

        Args:
            detector_id: ID of the detector

        Returns:
            List of processing results
        """
        ...

    async def find_by_data_collection(self, data_collection_id: UUID) -> list[DetectionResult]:
        """Find all results for a specific data_collection.

        Args:
            data_collection_id: ID of the data_collection

        Returns:
            List of processing results
        """
        ...

    async def find_recent(self, limit: int = 10) -> list[DetectionResult]:
        """Find most recent processing results.

        Args:
            limit: Maximum number of results

        Returns:
            List of recent results ordered by timestamp
        """
        ...

    async def get_summary_stats(self, result_id: UUID) -> dict[str, Any]:
        """Get summary statistics for a result.

        Args:
            result_id: ID of the processing result

        Returns:
            Dictionary of statistics
        """
        ...


@runtime_checkable
class ModelRepositoryProtocol(RepositoryProtocol[Model], Protocol):
    """Protocol for processor repository implementations."""

    async def find_by_name(self, name: str) -> list[Processor]:
        """Find models by name.

        Args:
            name: The processor name

        Returns:
            List of models with that name
        """
        ...

    async def find_by_stage(self, stage) -> list[Processor]:
        """Find models by stage.

        Args:
            stage: The processor stage

        Returns:
            List of models in that stage
        """
        ...

    async def find_by_type(self, processor_type) -> list[Processor]:
        """Find models by type.

        Args:
            processor_type: The processor type

        Returns:
            List of models of that type
        """
        ...


@runtime_checkable
class ModelVersionRepositoryProtocol(RepositoryProtocol[ModelVersion], Protocol):
    """Protocol for processor version repository implementations."""

    async def find_by_processor_id(self, processor_id: UUID) -> list[ModelVersion]:
        """Find all versions for a processor.

        Args:
            processor_id: ID of the processor

        Returns:
            List of processor versions
        """
        ...

    async def find_by_processor_and_version(
        self, processor_id: UUID, version
    ) -> ModelVersion | None:
        """Find a specific version of a processor.

        Args:
            processor_id: ID of the processor
            version: The version to find

        Returns:
            Processor version if found
        """
        ...


@runtime_checkable
class ExperimentRepositoryProtocol(RepositoryProtocol[Experiment], Protocol):
    """Protocol for experiment repository implementations."""

    async def find_by_name(self, name: str) -> list[Experiment]:
        """Find experiments by name.

        Args:
            name: The experiment name

        Returns:
            List of experiments with that name
        """
        ...

    async def find_by_status(self, status) -> list[Experiment]:
        """Find experiments by status.

        Args:
            status: The experiment status

        Returns:
            List of experiments with that status
        """
        ...

    async def find_by_type(self, experiment_type) -> list[Experiment]:
        """Find experiments by type.

        Args:
            experiment_type: The experiment type

        Returns:
            List of experiments of that type
        """
        ...


@runtime_checkable
class ExperimentRunRepositoryProtocol(RepositoryProtocol[ExperimentRun], Protocol):
    """Protocol for experiment run repository implementations."""

    async def find_by_experiment_id(self, experiment_id: UUID) -> list[ExperimentRun]:
        """Find all runs for an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            List of experiment runs
        """
        ...

    async def find_by_status(self, status: str) -> list[ExperimentRun]:
        """Find runs by status.

        Args:
            status: The run status

        Returns:
            List of runs with that status
        """
        ...


@runtime_checkable
class PipelineRepositoryProtocol(RepositoryProtocol[Pipeline], Protocol):
    """Protocol for pipeline repository implementations."""

    async def find_by_name(self, name: str) -> list[Pipeline]:
        """Find pipelines by name.

        Args:
            name: The pipeline name

        Returns:
            List of pipelines with that name
        """
        ...

    async def find_by_name_and_environment(
        self, name: str, environment: str
    ) -> list[Pipeline]:
        """Find pipelines by name and environment.

        Args:
            name: The pipeline name
            environment: The environment

        Returns:
            List of pipelines with that name and environment
        """
        ...

    async def find_by_status(self, status) -> list[Pipeline]:
        """Find pipelines by status.

        Args:
            status: The pipeline status

        Returns:
            List of pipelines with that status
        """
        ...

    async def find_by_type(self, pipeline_type) -> list[Pipeline]:
        """Find pipelines by type.

        Args:
            pipeline_type: The pipeline type

        Returns:
            List of pipelines of that type
        """
        ...


@runtime_checkable
class PipelineRunRepositoryProtocol(RepositoryProtocol[PipelineRun], Protocol):
    """Protocol for pipeline run repository implementations."""

    async def find_by_pipeline_id(self, pipeline_id: UUID) -> list[PipelineRun]:
        """Find all runs for a pipeline.

        Args:
            pipeline_id: ID of the pipeline

        Returns:
            List of pipeline runs
        """
        ...

    async def find_by_status(self, status: str) -> list[PipelineRun]:
        """Find runs by status.

        Args:
            status: The run status

        Returns:
            List of runs with that status
        """
        ...


@runtime_checkable
class AlertRepositoryProtocol(RepositoryProtocol[Alert], Protocol):
    """Protocol for alert repository implementations."""

    async def find_by_name(self, name: str) -> list[Alert]:
        """Find alerts by name.

        Args:
            name: The alert name

        Returns:
            List of alerts with that name
        """
        ...

    async def find_by_status(self, status) -> list[Alert]:
        """Find alerts by status.

        Args:
            status: The alert status

        Returns:
            List of alerts with that status
        """
        ...

    async def find_by_type(self, alert_type) -> list[Alert]:
        """Find alerts by type.

        Args:
            alert_type: The alert type

        Returns:
            List of alerts of that type
        """
        ...

    async def find_by_severity(self, severity) -> list[Alert]:
        """Find alerts by severity.

        Args:
            severity: The alert severity

        Returns:
            List of alerts with that severity
        """
        ...


@runtime_checkable
class AlertNotificationRepositoryProtocol(
    RepositoryProtocol[AlertNotification], Protocol
):
    """Protocol for alert notification repository implementations."""

    async def find_by_alert_id(self, alert_id: UUID) -> list[AlertNotification]:
        """Find all notifications for an alert.

        Args:
            alert_id: ID of the alert

        Returns:
            List of alert notifications
        """
        ...

    async def find_by_status(self, status: str) -> list[AlertNotification]:
        """Find notifications by status.

        Args:
            status: The notification status

        Returns:
            List of notifications with that status
        """
        ...

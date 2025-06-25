"""Repository protocol for infrastructure persistence."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable
from uuid import UUID

from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Model, ModelVersion, Experiment, ExperimentRun, Pipeline, PipelineRun, Alert, AlertNotification

T = TypeVar("T")


@runtime_checkable
class RepositoryProtocol(Protocol[T]):
    """Base protocol for repository implementations.
    
    This protocol defines the interface for persistence operations
    on domain entities.
    """
    
    def save(self, entity: T) -> None:
        """Save an entity to the repository.
        
        Args:
            entity: The entity to save
        """
        ...
    
    def find_by_id(self, entity_id: UUID) -> Optional[T]:
        """Find an entity by its ID.
        
        Args:
            entity_id: The UUID of the entity
            
        Returns:
            The entity if found, None otherwise
        """
        ...
    
    def find_all(self) -> List[T]:
        """Find all entities in the repository.
        
        Returns:
            List of all entities
        """
        ...
    
    def delete(self, entity_id: UUID) -> bool:
        """Delete an entity by its ID.
        
        Args:
            entity_id: The UUID of the entity to delete
            
        Returns:
            True if deleted, False if not found
        """
        ...
    
    def exists(self, entity_id: UUID) -> bool:
        """Check if an entity exists.
        
        Args:
            entity_id: The UUID to check
            
        Returns:
            True if exists, False otherwise
        """
        ...
    
    def count(self) -> int:
        """Count total number of entities.
        
        Returns:
            Number of entities in repository
        """
        ...


@runtime_checkable
class DetectorRepositoryProtocol(RepositoryProtocol[Detector], Protocol):
    """Protocol for detector repository implementations."""
    
    def find_by_name(self, name: str) -> Optional[Detector]:
        """Find a detector by name.
        
        Args:
            name: The detector name
            
        Returns:
            The detector if found
        """
        ...
    
    def find_by_algorithm(self, algorithm_name: str) -> List[Detector]:
        """Find all detectors using a specific algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            List of detectors using that algorithm
        """
        ...
    
    def find_fitted(self) -> List[Detector]:
        """Find all fitted detectors.
        
        Returns:
            List of fitted detectors
        """
        ...
    
    def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        """Save the trained model artifact.
        
        Args:
            detector_id: ID of the detector
            artifact: Serialized model data
        """
        ...
    
    def load_model_artifact(self, detector_id: UUID) -> Optional[bytes]:
        """Load the trained model artifact.
        
        Args:
            detector_id: ID of the detector
            
        Returns:
            Serialized model data if found
        """
        ...


@runtime_checkable
class DatasetRepositoryProtocol(RepositoryProtocol[Dataset], Protocol):
    """Protocol for dataset repository implementations."""
    
    def find_by_name(self, name: str) -> Optional[Dataset]:
        """Find a dataset by name.
        
        Args:
            name: The dataset name
            
        Returns:
            The dataset if found
        """
        ...
    
    def find_by_metadata(self, key: str, value: Any) -> List[Dataset]:
        """Find datasets by metadata key-value pair.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            List of matching datasets
        """
        ...
    
    def save_data(self, dataset_id: UUID, format: str = "parquet") -> str:
        """Save dataset data to persistent storage.
        
        Args:
            dataset_id: ID of the dataset
            format: Storage format (parquet, csv, etc.)
            
        Returns:
            Path or URI where data was saved
        """
        ...
    
    def load_data(self, dataset_id: UUID) -> Optional[Dataset]:
        """Load dataset with its data from storage.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dataset with data if found
        """
        ...


@runtime_checkable
class DetectionResultRepositoryProtocol(RepositoryProtocol[DetectionResult], Protocol):
    """Protocol for detection result repository implementations."""
    
    def find_by_detector(self, detector_id: UUID) -> List[DetectionResult]:
        """Find all results from a specific detector.
        
        Args:
            detector_id: ID of the detector
            
        Returns:
            List of detection results
        """
        ...
    
    def find_by_dataset(self, dataset_id: UUID) -> List[DetectionResult]:
        """Find all results for a specific dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            List of detection results
        """
        ...
    
    def find_recent(self, limit: int = 10) -> List[DetectionResult]:
        """Find most recent detection results.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of recent results ordered by timestamp
        """
        ...
    
    def get_summary_stats(self, result_id: UUID) -> Dict[str, Any]:
        """Get summary statistics for a result.
        
        Args:
            result_id: ID of the detection result
            
        Returns:
            Dictionary of statistics
        """
        ...


@runtime_checkable  
class ModelRepositoryProtocol(RepositoryProtocol[Model], Protocol):
    """Protocol for model repository implementations."""
    
    def find_by_name(self, name: str) -> List[Model]:
        """Find models by name.
        
        Args:
            name: The model name
            
        Returns:
            List of models with that name
        """
        ...
    
    def find_by_stage(self, stage) -> List[Model]:
        """Find models by stage.
        
        Args:
            stage: The model stage
            
        Returns:
            List of models in that stage
        """
        ...
    
    def find_by_type(self, model_type) -> List[Model]:
        """Find models by type.
        
        Args:
            model_type: The model type
            
        Returns:
            List of models of that type
        """
        ...


@runtime_checkable
class ModelVersionRepositoryProtocol(RepositoryProtocol[ModelVersion], Protocol):
    """Protocol for model version repository implementations."""
    
    def find_by_model_id(self, model_id: UUID) -> List[ModelVersion]:
        """Find all versions for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            List of model versions
        """
        ...
    
    def find_by_model_and_version(self, model_id: UUID, version) -> Optional[ModelVersion]:
        """Find a specific version of a model.
        
        Args:
            model_id: ID of the model
            version: The version to find
            
        Returns:
            Model version if found
        """
        ...


@runtime_checkable
class ExperimentRepositoryProtocol(RepositoryProtocol[Experiment], Protocol):
    """Protocol for experiment repository implementations."""
    
    def find_by_name(self, name: str) -> List[Experiment]:
        """Find experiments by name.
        
        Args:
            name: The experiment name
            
        Returns:
            List of experiments with that name
        """
        ...
    
    def find_by_status(self, status) -> List[Experiment]:
        """Find experiments by status.
        
        Args:
            status: The experiment status
            
        Returns:
            List of experiments with that status
        """
        ...
    
    def find_by_type(self, experiment_type) -> List[Experiment]:
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
    
    def find_by_experiment_id(self, experiment_id: UUID) -> List[ExperimentRun]:
        """Find all runs for an experiment.
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            List of experiment runs
        """
        ...
    
    def find_by_status(self, status: str) -> List[ExperimentRun]:
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
    
    def find_by_name(self, name: str) -> List[Pipeline]:
        """Find pipelines by name.
        
        Args:
            name: The pipeline name
            
        Returns:
            List of pipelines with that name
        """
        ...
    
    def find_by_name_and_environment(self, name: str, environment: str) -> List[Pipeline]:
        """Find pipelines by name and environment.
        
        Args:
            name: The pipeline name
            environment: The environment
            
        Returns:
            List of pipelines with that name and environment
        """
        ...
    
    def find_by_status(self, status) -> List[Pipeline]:
        """Find pipelines by status.
        
        Args:
            status: The pipeline status
            
        Returns:
            List of pipelines with that status
        """
        ...
    
    def find_by_type(self, pipeline_type) -> List[Pipeline]:
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
    
    def find_by_pipeline_id(self, pipeline_id: UUID) -> List[PipelineRun]:
        """Find all runs for a pipeline.
        
        Args:
            pipeline_id: ID of the pipeline
            
        Returns:
            List of pipeline runs
        """
        ...
    
    def find_by_status(self, status: str) -> List[PipelineRun]:
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
    
    def find_by_name(self, name: str) -> List[Alert]:
        """Find alerts by name.
        
        Args:
            name: The alert name
            
        Returns:
            List of alerts with that name
        """
        ...
    
    def find_by_status(self, status) -> List[Alert]:
        """Find alerts by status.
        
        Args:
            status: The alert status
            
        Returns:
            List of alerts with that status
        """
        ...
    
    def find_by_type(self, alert_type) -> List[Alert]:
        """Find alerts by type.
        
        Args:
            alert_type: The alert type
            
        Returns:
            List of alerts of that type
        """
        ...
    
    def find_by_severity(self, severity) -> List[Alert]:
        """Find alerts by severity.
        
        Args:
            severity: The alert severity
            
        Returns:
            List of alerts with that severity
        """
        ...


@runtime_checkable
class AlertNotificationRepositoryProtocol(RepositoryProtocol[AlertNotification], Protocol):
    """Protocol for alert notification repository implementations."""
    
    def find_by_alert_id(self, alert_id: UUID) -> List[AlertNotification]:
        """Find all notifications for an alert.
        
        Args:
            alert_id: ID of the alert
            
        Returns:
            List of alert notifications
        """
        ...
    
    def find_by_status(self, status: str) -> List[AlertNotification]:
        """Find notifications by status.
        
        Args:
            status: The notification status
            
        Returns:
            List of notifications with that status
        """
        ...
"""Repository protocol for infrastructure persistence."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable
from uuid import UUID

from pynomaly.domain.entities import Dataset, Detector, DetectionResult

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
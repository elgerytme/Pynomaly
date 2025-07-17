"""
Detection Request Entity

Represents a request for anomaly detection in the system.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..value_objects.algorithm_config import AlgorithmConfig
from ..value_objects.detection_metadata import DetectionMetadata


class DetectionRequest:
    """
    Entity representing an anomaly detection request.
    
    This entity encapsulates all information needed to perform
    anomaly detection, including data, algorithm configuration,
    and request metadata.
    """
    
    def __init__(
        self,
        data: List[float],
        algorithm_config: AlgorithmConfig,
        metadata: Optional[DetectionMetadata] = None,
        request_id: Optional[UUID] = None,
        created_at: Optional[datetime] = None
    ):
        self._id = request_id or uuid4()
        self._data = data.copy() if data else []
        self._algorithm_config = algorithm_config
        self._metadata = metadata or DetectionMetadata()
        self._created_at = created_at or datetime.utcnow()
        self._status = "pending"
        
    @property
    def id(self) -> UUID:
        """Unique identifier for the detection request."""
        return self._id
        
    @property
    def data(self) -> List[float]:
        """Input data for anomaly detection."""
        return self._data.copy()
        
    @property
    def algorithm_config(self) -> AlgorithmConfig:
        """Configuration for the detection algorithm."""
        return self._algorithm_config
        
    @property
    def metadata(self) -> DetectionMetadata:
        """Additional metadata for the request."""
        return self._metadata
        
    @property
    def created_at(self) -> datetime:
        """Timestamp when the request was created."""
        return self._created_at
        
    @property
    def status(self) -> str:
        """Current status of the detection request."""
        return self._status
        
    def mark_as_processing(self) -> None:
        """Mark the request as currently being processed."""
        self._status = "processing"
        
    def mark_as_completed(self) -> None:
        """Mark the request as completed."""
        self._status = "completed"
        
    def mark_as_failed(self, error: str) -> None:
        """Mark the request as failed with an error message."""
        self._status = "failed"
        self._error = error
        
    def validate(self) -> bool:
        """
        Validate the detection request.
        
        Returns:
            bool: True if the request is valid, False otherwise.
        """
        if not self._data:
            return False
            
        if len(self._data) < 2:
            return False
            
        if not all(isinstance(x, (int, float)) for x in self._data):
            return False
            
        return self._algorithm_config.is_valid()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entity to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the entity.
        """
        return {
            "id": str(self._id),
            "data": self._data,
            "algorithm_config": self._algorithm_config.to_dict(),
            "metadata": self._metadata.to_dict(),
            "created_at": self._created_at.isoformat(),
            "status": self._status
        }
        
    def __eq__(self, other: object) -> bool:
        """Check equality based on entity ID."""
        if not isinstance(other, DetectionRequest):
            return False
        return self._id == other._id
        
    def __hash__(self) -> int:
        """Hash based on entity ID."""
        return hash(self._id)
        
    def __repr__(self) -> str:
        """String representation of the entity."""
        return f"DetectionRequest(id={self._id}, status={self._status}, data_points={len(self._data)})"
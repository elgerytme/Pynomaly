"""Repository interface for detectors."""

from abc import ABC, abstractmethod
from typing import List, Optional

from .entities.detector import Detector


class DetectorRepository(ABC):
    """Abstract repository for anomaly detectors."""
    
    @abstractmethod
    async def save(self, detector: Detector) -> Detector:
        """Save a detector."""
        pass
    
    @abstractmethod
    async def get_by_id(self, detector_id: str) -> Optional[Detector]:
        """Get detector by ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> List[Detector]:
        """Get detectors by name."""
        pass
    
    @abstractmethod
    async def get_by_type(self, detector_type: str) -> List[Detector]:
        """Get detectors by type."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Detector]:
        """Get all detectors."""
        pass
    
    @abstractmethod
    async def delete(self, detector_id: str) -> bool:
        """Delete a detector."""
        pass
    
    @abstractmethod
    async def update(self, detector: Detector) -> Detector:
        """Update a detector."""
        pass
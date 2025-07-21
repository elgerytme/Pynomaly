"""In-memory repository implementation for detectors."""

from typing import Dict, List, Optional

from ...domain.entities.detector import Detector
from ...domain.repositories.detector_repository import DetectorRepository


class InMemoryDetectorRepository(DetectorRepository):
    """In-memory implementation of detector repository."""
    
    def __init__(self):
        self._detectors: Dict[str, Detector] = {}
    
    async def save(self, detector: Detector) -> Detector:
        """Save a detector."""
        self._detectors[detector.id] = detector
        return detector
    
    async def get_by_id(self, detector_id: str) -> Optional[Detector]:
        """Get detector by ID."""
        return self._detectors.get(detector_id)
    
    async def get_by_name(self, name: str) -> List[Detector]:
        """Get detectors by name."""
        return [
            detector for detector in self._detectors.values()
            if detector.name == name
        ]
    
    async def get_by_type(self, detector_type: str) -> List[Detector]:
        """Get detectors by type."""
        return [
            detector for detector in self._detectors.values()
            if detector.algorithm_type == detector_type
        ]
    
    async def get_all(self) -> List[Detector]:
        """Get all detectors."""
        return list(self._detectors.values())
    
    async def delete(self, detector_id: str) -> bool:
        """Delete a detector."""
        if detector_id in self._detectors:
            del self._detectors[detector_id]
            return True
        return False
    
    async def update(self, detector: Detector) -> Detector:
        """Update a detector."""
        if detector.id in self._detectors:
            self._detectors[detector.id] = detector
            return detector
        else:
            raise ValueError(f"Detector with ID {detector.id} not found")
    
    def clear(self) -> None:
        """Clear all detectors (for testing)."""
        self._detectors.clear()
    
    def count(self) -> int:
        """Get count of detectors."""
        return len(self._detectors)
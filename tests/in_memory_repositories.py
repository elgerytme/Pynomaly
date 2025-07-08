"""In-memory repositories for testing purposes."""

from typing import Dict, List, Optional
from uuid import UUID

try:
    from pynomaly.domain.entities.anomaly import Anomaly
    from pynomaly.domain.entities.detector import Detector
    from pynomaly.domain.entities.detection_result import DetectionResult
except ImportError:
    # Fallback for missing modules
    class Anomaly:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            if not hasattr(self, 'id'):
                from uuid import uuid4
                self.id = uuid4()
    class Detector:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            if not hasattr(self, 'id'):
                from uuid import uuid4
                self.id = uuid4()
    class DetectionResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            if not hasattr(self, 'id'):
                from uuid import uuid4
                self.id = uuid4()


class AnomalyRepository:
    """In-memory repository for Anomalies."""
    def __init__(self):
        self._storage: Dict[UUID, Anomaly] = {}

    def add(self, anomaly: Anomaly) -> None:
        """Add an anomaly to the repository."""
        self._storage[anomaly.id] = anomaly

    def get(self, anomaly_id: UUID) -> Optional[Anomaly]:
        """Get an anomaly by ID."""
        return self._storage.get(anomaly_id)

    def list(self) -> List[Anomaly]:
        """List all anomalies."""
        return list(self._storage.values())


class DetectorRepository:
    """In-memory repository for Detectors."""
    def __init__(self):
        self._storage: Dict[UUID, Detector] = {}

    def add(self, detector: Detector) -> None:
        """Add a detector to the repository."""
        self._storage[detector.id] = detector

    def get(self, detector_id: UUID) -> Optional[Detector]:
        """Get a detector by ID."""
        return self._storage.get(detector_id)

    def list(self) -> List[Detector]:
        """List all detectors."""
        return list(self._storage.values())


class DetectionResultRepository:
    """In-memory repository for DetectionResults."""
    def __init__(self):
        self._storage: Dict[UUID, DetectionResult] = {}

    def add(self, result: DetectionResult) -> None:
        """Add a detection result to the repository."""
        self._storage[result.id] = result

    def get(self, result_id: UUID) -> Optional[DetectionResult]:
        """Get a detection result by ID."""
        return self._storage.get(result_id)

    def list(self) -> List[DetectionResult]:
        """List all detection results."""
        return list(self._storage.values())

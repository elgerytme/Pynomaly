"""In-memory repositories for testing purposes."""

from uuid import UUID

try:
    from pynomaly.domain.entities.anomaly import Anomaly
    from pynomaly.domain.entities.detection_result import DetectionResult
    from pynomaly.domain.entities.detector import Detector
except ImportError:
    # Fallback for missing modules
    class Anomaly:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            if not hasattr(self, "id"):
                from uuid import uuid4

                self.id = uuid4()

    class Detector:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            if not hasattr(self, "id"):
                from uuid import uuid4

                self.id = uuid4()

    class DetectionResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            if not hasattr(self, "id"):
                from uuid import uuid4

                self.id = uuid4()


class AnomalyRepository:
    """In-memory repository for Anomalies."""

    def __init__(self):
        self._storage: dict[UUID, Anomaly] = {}

    def add(self, anomaly: Anomaly) -> None:
        """Add an anomaly to the repository."""
        self._storage[anomaly.id] = anomaly

    def get(self, anomaly_id: UUID) -> Anomaly | None:
        """Get an anomaly by ID."""
        return self._storage.get(anomaly_id)

    def list(self) -> list[Anomaly]:
        """List all anomalies."""
        return list(self._storage.values())


class DetectorRepository:
    """In-memory repository for Detectors."""

    def __init__(self):
        self._storage: dict[UUID, Detector] = {}

    def add(self, detector: Detector) -> None:
        """Add a detector to the repository."""
        self._storage[detector.id] = detector

    def get(self, detector_id: UUID) -> Detector | None:
        """Get a detector by ID."""
        return self._storage.get(detector_id)

    def list(self) -> list[Detector]:
        """List all detectors."""
        return list(self._storage.values())


class DetectionResultRepository:
    """In-memory repository for DetectionResults."""

    def __init__(self):
        self._storage: dict[UUID, DetectionResult] = {}

    def add(self, result: DetectionResult) -> None:
        """Add a detection result to the repository."""
        self._storage[result.id] = result

    def get(self, result_id: UUID) -> DetectionResult | None:
        """Get a detection result by ID."""
        return self._storage.get(result_id)

    def list(self) -> list[DetectionResult]:
        """List all detection results."""
        return list(self._storage.values())

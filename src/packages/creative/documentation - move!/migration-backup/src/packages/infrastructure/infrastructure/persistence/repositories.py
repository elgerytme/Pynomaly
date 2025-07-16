"""
In-memory repository implementations for testing and development.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from pynomaly.domain.entities import Dataset, DetectionResult, Detector

T = TypeVar("T")


class BaseRepository(Generic[T], ABC):
    """Base repository interface."""

    @abstractmethod
    def save(self, entity: T) -> None:
        """Save an entity."""
        pass

    @abstractmethod
    def find_by_id(self, entity_id: UUID) -> T | None:
        """Find entity by ID."""
        pass

    @abstractmethod
    def find_all(self) -> list[T]:
        """Find all entities."""
        pass

    @abstractmethod
    def delete(self, entity_id: UUID) -> None:
        """Delete entity by ID."""
        pass

    @abstractmethod
    def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total entities."""
        pass


class InMemoryDatasetRepository(BaseRepository[Dataset]):
    """In-memory dataset repository for testing."""

    def __init__(self):
        self._data: dict[UUID, Dataset] = {}

    def save(self, dataset: Dataset) -> None:
        """Save a dataset."""
        self._data[dataset.id] = dataset

    def find_by_id(self, dataset_id: UUID) -> Dataset | None:
        """Find dataset by ID."""
        return self._data.get(dataset_id)

    def find_all(self) -> list[Dataset]:
        """Find all datasets."""
        return list(self._data.values())

    def delete(self, dataset_id: UUID) -> None:
        """Delete dataset by ID."""
        if dataset_id in self._data:
            del self._data[dataset_id]

    def exists(self, dataset_id: UUID) -> bool:
        """Check if dataset exists."""
        return dataset_id in self._data

    def count(self) -> int:
        """Count total datasets."""
        return len(self._data)


class InMemoryDetectorRepository(BaseRepository[Detector]):
    """In-memory detector repository for testing."""

    def __init__(self):
        self._data: dict[UUID, Detector] = {}

    def save(self, detector: Detector) -> None:
        """Save a detector."""
        self._data[detector.id] = detector

    def find_by_id(self, detector_id: UUID) -> Detector | None:
        """Find detector by ID."""
        return self._data.get(detector_id)

    def find_all(self) -> list[Detector]:
        """Find all detectors."""
        return list(self._data.values())

    def delete(self, detector_id: UUID) -> None:
        """Delete detector by ID."""
        if detector_id in self._data:
            del self._data[detector_id]

    def exists(self, detector_id: UUID) -> bool:
        """Check if detector exists."""
        return detector_id in self._data

    def count(self) -> int:
        """Count total detectors."""
        return len(self._data)

    def find_by_algorithm(self, algorithm: str) -> list[Detector]:
        """Find detectors by algorithm."""
        return [
            detector
            for detector in self._data.values()
            if detector.algorithm == algorithm
        ]


class InMemoryResultRepository(BaseRepository[DetectionResult]):
    """In-memory detection result repository for testing."""

    def __init__(self):
        self._data: dict[UUID, DetectionResult] = {}

    def save(self, result: DetectionResult) -> None:
        """Save a detection result."""
        self._data[result.id] = result

    def find_by_id(self, result_id: UUID) -> DetectionResult | None:
        """Find result by ID."""
        return self._data.get(result_id)

    def find_all(self) -> list[DetectionResult]:
        """Find all results."""
        return list(self._data.values())

    def delete(self, result_id: UUID) -> None:
        """Delete result by ID."""
        if result_id in self._data:
            del self._data[result_id]

    def exists(self, result_id: UUID) -> bool:
        """Check if result exists."""
        return result_id in self._data

    def count(self) -> int:
        """Count total results."""
        return len(self._data)

    def find_by_detector(self, detector_id: UUID) -> list[DetectionResult]:
        """Find results by detector ID."""
        return [
            result
            for result in self._data.values()
            if result.detector_id == detector_id
        ]

    def find_by_dataset(self, dataset_id: UUID) -> list[DetectionResult]:
        """Find results by dataset ID."""
        return [
            result for result in self._data.values() if result.dataset_id == dataset_id
        ]

    def find_recent(self, limit: int = 10) -> list[DetectionResult]:
        """Find recent results."""
        # Sort by timestamp descending and limit
        sorted_results = sorted(
            self._data.values(), key=lambda r: r.timestamp, reverse=True
        )
        return sorted_results[:limit]

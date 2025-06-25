"""In-memory repository implementations."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.shared.protocols import (
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)


class InMemoryDetectorRepository(DetectorRepositoryProtocol):
    """In-memory implementation of detector repository."""

    def __init__(self):
        """Initialize empty repository."""
        self._storage: dict[UUID, Detector] = {}
        self._model_artifacts: dict[UUID, bytes] = {}
        self._name_index: dict[str, UUID] = {}

    def save(self, entity: Detector) -> None:
        """Save a detector to the repository."""
        self._storage[entity.id] = entity
        self._name_index[entity.name] = entity.id

    def find_by_id(self, entity_id: UUID) -> Detector | None:
        """Find a detector by its ID."""
        return self._storage.get(entity_id)

    def find_all(self) -> list[Detector]:
        """Find all detectors in the repository."""
        return list(self._storage.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete a detector by its ID."""
        if entity_id in self._storage:
            detector = self._storage[entity_id]
            del self._storage[entity_id]

            # Remove from name index
            if detector.name in self._name_index:
                del self._name_index[detector.name]

            # Remove model artifact if exists
            if entity_id in self._model_artifacts:
                del self._model_artifacts[entity_id]

            return True
        return False

    def exists(self, entity_id: UUID) -> bool:
        """Check if a detector exists."""
        return entity_id in self._storage

    def count(self) -> int:
        """Count total number of detectors."""
        return len(self._storage)

    def find_by_name(self, name: str) -> Detector | None:
        """Find a detector by name."""
        detector_id = self._name_index.get(name)
        if detector_id:
            return self._storage.get(detector_id)
        return None

    def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
        """Find all detectors using a specific algorithm."""
        return [
            detector
            for detector in self._storage.values()
            if detector.algorithm_name == algorithm_name
        ]

    def find_fitted(self) -> list[Detector]:
        """Find all fitted detectors."""
        return [detector for detector in self._storage.values() if detector.is_fitted]

    def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        """Save the trained model artifact."""
        self._model_artifacts[detector_id] = artifact

    def load_model_artifact(self, detector_id: UUID) -> bytes | None:
        """Load the trained model artifact."""
        return self._model_artifacts.get(detector_id)


class InMemoryDatasetRepository(DatasetRepositoryProtocol):
    """In-memory implementation of dataset repository."""

    def __init__(self):
        """Initialize empty repository."""
        self._storage: dict[UUID, Dataset] = {}
        self._name_index: dict[str, UUID] = {}
        self._data_storage: dict[UUID, Any] = {}

    def save(self, entity: Dataset) -> None:
        """Save a dataset to the repository."""
        self._storage[entity.id] = entity
        self._name_index[entity.name] = entity.id

    def find_by_id(self, entity_id: UUID) -> Dataset | None:
        """Find a dataset by its ID."""
        return self._storage.get(entity_id)

    def find_all(self) -> list[Dataset]:
        """Find all datasets in the repository."""
        return list(self._storage.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete a dataset by its ID."""
        if entity_id in self._storage:
            dataset = self._storage[entity_id]
            del self._storage[entity_id]

            # Remove from name index
            if dataset.name in self._name_index:
                del self._name_index[dataset.name]

            # Remove data if stored
            if entity_id in self._data_storage:
                del self._data_storage[entity_id]

            return True
        return False

    def exists(self, entity_id: UUID) -> bool:
        """Check if a dataset exists."""
        return entity_id in self._storage

    def count(self) -> int:
        """Count total number of datasets."""
        return len(self._storage)

    def find_by_name(self, name: str) -> Dataset | None:
        """Find a dataset by name."""
        dataset_id = self._name_index.get(name)
        if dataset_id:
            return self._storage.get(dataset_id)
        return None

    def find_by_metadata(self, key: str, value: Any) -> list[Dataset]:
        """Find datasets by metadata key-value pair."""
        return [
            dataset
            for dataset in self._storage.values()
            if dataset.metadata.get(key) == value
        ]

    def save_data(self, dataset_id: UUID, format: str = "parquet") -> str:
        """Save dataset data to persistent storage."""
        dataset = self._storage.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        # In-memory implementation just stores reference
        self._data_storage[dataset_id] = {
            "format": format,
            "data": dataset.data,
            "saved_at": datetime.utcnow(),
        }

        return f"memory://{dataset_id}.{format}"

    def load_data(self, dataset_id: UUID) -> Dataset | None:
        """Load dataset with its data from storage."""
        dataset = self._storage.get(dataset_id)
        if dataset and dataset_id in self._data_storage:
            # In-memory implementation already has data
            return dataset
        return None


class InMemoryResultRepository(DetectionResultRepositoryProtocol):
    """In-memory implementation of detection result repository."""

    def __init__(self):
        """Initialize empty repository."""
        self._storage: dict[UUID, DetectionResult] = {}
        self._detector_index: dict[UUID, list[UUID]] = {}
        self._dataset_index: dict[UUID, list[UUID]] = {}

    def save(self, entity: DetectionResult) -> None:
        """Save a detection result to the repository."""
        self._storage[entity.id] = entity

        # Update indices
        if entity.detector_id not in self._detector_index:
            self._detector_index[entity.detector_id] = []
        self._detector_index[entity.detector_id].append(entity.id)

        if entity.dataset_id not in self._dataset_index:
            self._dataset_index[entity.dataset_id] = []
        self._dataset_index[entity.dataset_id].append(entity.id)

    def find_by_id(self, entity_id: UUID) -> DetectionResult | None:
        """Find a detection result by its ID."""
        return self._storage.get(entity_id)

    def find_all(self) -> list[DetectionResult]:
        """Find all detection results in the repository."""
        return list(self._storage.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete a detection result by its ID."""
        if entity_id in self._storage:
            result = self._storage[entity_id]
            del self._storage[entity_id]

            # Remove from indices
            if result.detector_id in self._detector_index:
                self._detector_index[result.detector_id].remove(entity_id)
                if not self._detector_index[result.detector_id]:
                    del self._detector_index[result.detector_id]

            if result.dataset_id in self._dataset_index:
                self._dataset_index[result.dataset_id].remove(entity_id)
                if not self._dataset_index[result.dataset_id]:
                    del self._dataset_index[result.dataset_id]

            return True
        return False

    def exists(self, entity_id: UUID) -> bool:
        """Check if a detection result exists."""
        return entity_id in self._storage

    def count(self) -> int:
        """Count total number of detection results."""
        return len(self._storage)

    def find_by_detector(self, detector_id: UUID) -> list[DetectionResult]:
        """Find all results from a specific detector."""
        result_ids = self._detector_index.get(detector_id, [])
        return [
            self._storage[result_id]
            for result_id in result_ids
            if result_id in self._storage
        ]

    def find_by_dataset(self, dataset_id: UUID) -> list[DetectionResult]:
        """Find all results for a specific dataset."""
        result_ids = self._dataset_index.get(dataset_id, [])
        return [
            self._storage[result_id]
            for result_id in result_ids
            if result_id in self._storage
        ]

    def find_recent(self, limit: int = 10) -> list[DetectionResult]:
        """Find most recent detection results."""
        # Sort by timestamp
        sorted_results = sorted(
            self._storage.values(), key=lambda r: r.timestamp, reverse=True
        )
        return sorted_results[:limit]

    def get_summary_stats(self, result_id: UUID) -> dict[str, Any]:
        """Get summary statistics for a result."""
        result = self._storage.get(result_id)
        if not result:
            return {}

        return {
            "id": str(result.id),
            "detector_id": str(result.detector_id),
            "dataset_id": str(result.dataset_id),
            "timestamp": result.timestamp.isoformat(),
            "n_samples": result.n_samples,
            "n_anomalies": result.n_anomalies,
            "anomaly_rate": result.anomaly_rate,
            "threshold": result.threshold,
            "execution_time_ms": result.execution_time_ms,
            "score_statistics": result.score_statistics,
            "has_confidence_intervals": result.has_confidence_intervals,
        }

"""In-memory repository implementations for development and testing."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
from uuid import UUID

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.shared.protocols.repository_protocol import (
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)


class MemoryDetectorRepository(DetectorRepositoryProtocol):
    """In-memory detector repository implementation."""

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._detectors: dict[UUID, Detector] = {}
        self._model_artifacts: dict[UUID, bytes] = {}

    def save(self, entity: Detector) -> None:
        """Save detector to memory."""
        self._detectors[entity.id] = entity

    def find_by_id(self, entity_id: UUID) -> Detector | None:
        """Find detector by ID."""
        return self._detectors.get(entity_id)

    def find_all(self) -> list[Detector]:
        """Find all detectors."""
        return list(self._detectors.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete detector by ID."""
        if entity_id in self._detectors:
            del self._detectors[entity_id]
            if entity_id in self._model_artifacts:
                del self._model_artifacts[entity_id]
            return True
        return False

    def exists(self, entity_id: UUID) -> bool:
        """Check if detector exists."""
        return entity_id in self._detectors

    def count(self) -> int:
        """Count total detectors."""
        return len(self._detectors)

    def find_by_name(self, name: str) -> Detector | None:
        """Find detector by name."""
        for detector in self._detectors.values():
            if detector.name == name:
                return detector
        return None

    def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
        """Find detectors by algorithm name."""
        return [
            detector
            for detector in self._detectors.values()
            if detector.algorithm_name == algorithm_name
        ]

    def find_fitted(self) -> list[Detector]:
        """Find all fitted detectors."""
        return [detector for detector in self._detectors.values() if detector.is_fitted]

    def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        """Save model artifact."""
        self._model_artifacts[detector_id] = artifact

    def load_model_artifact(self, detector_id: UUID) -> bytes | None:
        """Load model artifact."""
        return self._model_artifacts.get(detector_id)


class MemoryDatasetRepository(DatasetRepositoryProtocol):
    """In-memory dataset repository implementation."""

    def __init__(self, storage_path: str = "data/datasets") -> None:
        """Initialize in-memory storage."""
        self._datasets: dict[UUID, Dataset] = {}
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, entity: Dataset) -> None:
        """Save dataset to memory."""
        self._datasets[entity.id] = entity

    def find_by_id(self, entity_id: UUID) -> Dataset | None:
        """Find dataset by ID."""
        return self._datasets.get(entity_id)

    def find_all(self) -> list[Dataset]:
        """Find all datasets."""
        return list(self._datasets.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete dataset by ID."""
        if entity_id in self._datasets:
            del self._datasets[entity_id]
            # Also delete stored data files
            data_file = self._storage_path / f"{entity_id}.parquet"
            if data_file.exists():
                data_file.unlink()
            return True
        return False

    def exists(self, entity_id: UUID) -> bool:
        """Check if dataset exists."""
        return entity_id in self._datasets

    def count(self) -> int:
        """Count total datasets."""
        return len(self._datasets)

    def find_by_name(self, name: str) -> Dataset | None:
        """Find dataset by name."""
        for dataset in self._datasets.values():
            if dataset.name == name:
                return dataset
        return None

    def find_by_metadata(self, key: str, value: Any) -> list[Dataset]:
        """Find datasets by metadata key-value pair."""
        return [
            dataset
            for dataset in self._datasets.values()
            if dataset.metadata.get(key) == value
        ]

    def save_data(self, dataset_id: UUID, format: str = "parquet") -> str:
        """Save dataset data to persistent storage."""
        dataset = self._datasets.get(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")

        if format == "parquet":
            file_path = self._storage_path / f"{dataset_id}.parquet"
            dataset.data.to_parquet(file_path)
        elif format == "csv":
            file_path = self._storage_path / f"{dataset_id}.csv"
            dataset.data.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return str(file_path)

    def load_data(self, dataset_id: UUID) -> Dataset | None:
        """Load dataset with data from storage."""
        dataset = self._datasets.get(dataset_id)
        if not dataset:
            return None

        # Try to load from parquet first, then CSV
        parquet_path = self._storage_path / f"{dataset_id}.parquet"
        csv_path = self._storage_path / f"{dataset_id}.csv"

        if parquet_path.exists():
            import pandas as pd

            dataset.data = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            import pandas as pd

            dataset.data = pd.read_csv(csv_path)

        return dataset


class MemoryDetectionResultRepository(DetectionResultRepositoryProtocol):
    """In-memory detection result repository implementation."""

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._results: dict[UUID, DetectionResult] = {}

    def save(self, entity: DetectionResult) -> None:
        """Save detection result to memory."""
        self._results[entity.id] = entity

    def find_by_id(self, entity_id: UUID) -> DetectionResult | None:
        """Find detection result by ID."""
        return self._results.get(entity_id)

    def find_all(self) -> list[DetectionResult]:
        """Find all detection results."""
        return list(self._results.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete detection result by ID."""
        if entity_id in self._results:
            del self._results[entity_id]
            return True
        return False

    def exists(self, entity_id: UUID) -> bool:
        """Check if detection result exists."""
        return entity_id in self._results

    def count(self) -> int:
        """Count total detection results."""
        return len(self._results)

    def find_by_detector(self, detector_id: UUID) -> list[DetectionResult]:
        """Find results by detector ID."""
        return [
            result
            for result in self._results.values()
            if result.detector_id == detector_id
        ]

    def find_by_dataset(self, dataset_id: UUID) -> list[DetectionResult]:
        """Find results by dataset ID."""
        return [
            result
            for result in self._results.values()
            if result.dataset_id == dataset_id
        ]

    def find_recent(self, limit: int = 10) -> list[DetectionResult]:
        """Find most recent detection results."""
        sorted_results = sorted(
            self._results.values(), key=lambda x: x.timestamp, reverse=True
        )
        return sorted_results[:limit]

    def get_summary_stats(self, result_id: UUID) -> dict[str, Any]:
        """Get summary statistics for a result."""
        result = self._results.get(result_id)
        if not result:
            return {}

        return {
            "id": str(result.id),
            "detector_id": str(result.detector_id),
            "dataset_id": str(result.dataset_id),
            "total_anomalies": len(result.anomalies),
            "total_samples": len(result.scores),
            "anomaly_rate": (
                len(result.anomalies) / len(result.scores) if result.scores else 0
            ),
            "execution_time_ms": result.execution_time_ms,
            "threshold": result.threshold,
            "timestamp": result.timestamp.isoformat(),
            "metadata": result.metadata,
        }


class FileSystemDetectorRepository(DetectorRepositoryProtocol):
    """File system-based detector repository implementation."""

    def __init__(self, storage_path: str = "data/detectors") -> None:
        """Initialize file system storage."""
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._models_path = self._storage_path / "models"
        self._models_path.mkdir(parents=True, exist_ok=True)

    def save(self, entity: Detector) -> None:
        """Save detector to file system."""
        file_path = self._storage_path / f"{entity.id}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(entity, f)

    def find_by_id(self, entity_id: UUID) -> Detector | None:
        """Find detector by ID."""
        file_path = self._storage_path / f"{entity_id}.pkl"
        if file_path.exists():
            with open(file_path, "rb") as f:
                return pickle.load(f)
        return None

    def find_all(self) -> list[Detector]:
        """Find all detectors."""
        detectors = []
        for file_path in self._storage_path.glob("*.pkl"):
            try:
                with open(file_path, "rb") as f:
                    detector = pickle.load(f)
                    detectors.append(detector)
            except Exception:
                continue  # Skip corrupted files
        return detectors

    def delete(self, entity_id: UUID) -> bool:
        """Delete detector by ID."""
        file_path = self._storage_path / f"{entity_id}.pkl"
        model_path = self._models_path / f"{entity_id}.pkl"

        deleted = False
        if file_path.exists():
            file_path.unlink()
            deleted = True
        if model_path.exists():
            model_path.unlink()
            deleted = True
        return deleted

    def exists(self, entity_id: UUID) -> bool:
        """Check if detector exists."""
        file_path = self._storage_path / f"{entity_id}.pkl"
        return file_path.exists()

    def count(self) -> int:
        """Count total detectors."""
        return len(list(self._storage_path.glob("*.pkl")))

    def find_by_name(self, name: str) -> Detector | None:
        """Find detector by name."""
        for detector in self.find_all():
            if detector.name == name:
                return detector
        return None

    def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
        """Find detectors by algorithm name."""
        return [
            detector
            for detector in self.find_all()
            if detector.algorithm_name == algorithm_name
        ]

    def find_fitted(self) -> list[Detector]:
        """Find all fitted detectors."""
        return [detector for detector in self.find_all() if detector.is_fitted]

    def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        """Save model artifact."""
        model_path = self._models_path / f"{detector_id}.pkl"
        with open(model_path, "wb") as f:
            f.write(artifact)

    def load_model_artifact(self, detector_id: UUID) -> bytes | None:
        """Load model artifact."""
        model_path = self._models_path / f"{detector_id}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                return f.read()
        return None

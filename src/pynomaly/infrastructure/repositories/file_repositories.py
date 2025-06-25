"""File-based repository implementations for simple persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.shared.protocols import (
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)


class FileDetectorRepository(DetectorRepositoryProtocol):
    """File-based implementation of detector repository using JSON."""

    def __init__(self, storage_path: str | Path = "storage"):
        """Initialize file repository.

        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.detectors_file = self.storage_path / "detectors.json"
        self.models_dir = self.storage_path / "models"
        self.models_dir.mkdir(exist_ok=True)

        self._load_data()

    def _load_data(self) -> None:
        """Load detectors from file."""
        if self.detectors_file.exists():
            try:
                with open(self.detectors_file) as f:
                    data = json.load(f)
                self._storage = {
                    UUID(k): self._dict_to_detector(v) for k, v in data.items()
                }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(
                    f"Warning: Failed to load detectors from {self.detectors_file}: {e}"
                )
                self._storage = {}
        else:
            self._storage = {}

    def _save_data(self) -> None:
        """Save detectors to file."""
        try:
            data = {str(k): self._detector_to_dict(v) for k, v in self._storage.items()}
            with open(self.detectors_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save detectors to {self.detectors_file}: {e}")

    def _detector_to_dict(self, detector: Detector) -> dict[str, Any]:
        """Convert detector to dictionary for JSON serialization."""
        return {
            "id": str(detector.id),
            "name": detector.name,
            "algorithm_name": detector.algorithm_name,
            "contamination_rate": detector.contamination_rate.value
            if detector.contamination_rate
            else None,
            "parameters": detector.parameters,
            "metadata": detector.metadata,
            "created_at": detector.created_at.isoformat(),
            "trained_at": detector.trained_at.isoformat()
            if detector.trained_at
            else None,
            "is_fitted": detector.is_fitted,
        }

    def _dict_to_detector(self, data: dict[str, Any]) -> Detector:
        """Convert dictionary to detector object."""
        # Handle contamination_rate
        contamination_rate = None
        if data.get("contamination_rate") is not None:
            contamination_rate = ContaminationRate(data["contamination_rate"])
        else:
            contamination_rate = ContaminationRate.auto()

        # Parse dates
        created_at = datetime.fromisoformat(data["created_at"])
        trained_at = None
        if data.get("trained_at"):
            trained_at = datetime.fromisoformat(data["trained_at"])

        return Detector(
            id=UUID(data["id"]),
            name=data["name"],
            algorithm_name=data["algorithm_name"],
            contamination_rate=contamination_rate,
            parameters=data.get("parameters", {}),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            trained_at=trained_at,
            is_fitted=data.get("is_fitted", False),
        )

    def save(self, entity: Detector) -> None:
        """Save a detector to the repository."""
        self._storage[entity.id] = entity
        self._save_data()

    def find_by_id(self, entity_id: UUID) -> Detector | None:
        """Find a detector by its ID."""
        return self._storage.get(entity_id)

    def find_all(self) -> list[Detector]:
        """Find all detectors in the repository."""
        return list(self._storage.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete a detector by its ID."""
        if entity_id in self._storage:
            del self._storage[entity_id]
            self._save_data()

            # Also remove model file if it exists
            model_file = self.models_dir / f"{entity_id}.model"
            if model_file.exists():
                model_file.unlink()

            return True
        return False

    def count(self) -> int:
        """Count total number of detectors."""
        return len(self._storage)

    def find_by_name(self, name: str) -> Detector | None:
        """Find a detector by name."""
        for detector in self._storage.values():
            if detector.name == name:
                return detector
        return None

    def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
        """Find detectors by algorithm name."""
        return [d for d in self._storage.values() if d.algorithm_name == algorithm_name]


class FileDatasetRepository(DatasetRepositoryProtocol):
    """File-based implementation of dataset repository using JSON."""

    def __init__(self, storage_path: str | Path = "storage"):
        """Initialize file repository."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.datasets_file = self.storage_path / "datasets.json"
        self._load_data()

    def _load_data(self) -> None:
        """Load datasets from file."""
        if self.datasets_file.exists():
            try:
                with open(self.datasets_file) as f:
                    data = json.load(f)
                self._storage = {
                    UUID(k): self._dict_to_dataset(v) for k, v in data.items()
                }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(
                    f"Warning: Failed to load datasets from {self.datasets_file}: {e}"
                )
                self._storage = {}
        else:
            self._storage = {}

    def _save_data(self) -> None:
        """Save datasets to file."""
        try:
            data = {str(k): self._dataset_to_dict(v) for k, v in self._storage.items()}
            with open(self.datasets_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save datasets to {self.datasets_file}: {e}")

    def _dataset_to_dict(self, dataset: Dataset) -> dict[str, Any]:
        """Convert dataset to dictionary for JSON serialization."""
        return {
            "id": str(dataset.id),
            "name": dataset.name,
            "source": dataset.source,
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "feature_names": dataset.feature_names,
            "metadata": dataset.metadata,
            "created_at": dataset.created_at.isoformat(),
        }

    def _dict_to_dataset(self, data: dict[str, Any]) -> Dataset:
        """Convert dictionary to dataset object."""
        created_at = datetime.fromisoformat(data["created_at"])

        return Dataset(
            id=UUID(data["id"]),
            name=data["name"],
            source=data["source"],
            n_samples=data["n_samples"],
            n_features=data["n_features"],
            feature_names=data.get("feature_names", []),
            metadata=data.get("metadata", {}),
            created_at=created_at,
        )

    def save(self, entity: Dataset) -> None:
        """Save a dataset to the repository."""
        self._storage[entity.id] = entity
        self._save_data()

    def find_by_id(self, entity_id: UUID) -> Dataset | None:
        """Find a dataset by its ID."""
        return self._storage.get(entity_id)

    def find_all(self) -> list[Dataset]:
        """Find all datasets in the repository."""
        return list(self._storage.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete a dataset by its ID."""
        if entity_id in self._storage:
            del self._storage[entity_id]
            self._save_data()
            return True
        return False

    def count(self) -> int:
        """Count total number of datasets."""
        return len(self._storage)


class FileResultRepository(DetectionResultRepositoryProtocol):
    """File-based implementation of detection result repository using JSON."""

    def __init__(self, storage_path: str | Path = "storage"):
        """Initialize file repository."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.results_file = self.storage_path / "results.json"
        self._load_data()

    def _load_data(self) -> None:
        """Load results from file."""
        if self.results_file.exists():
            try:
                with open(self.results_file) as f:
                    data = json.load(f)
                self._storage = {
                    UUID(k): self._dict_to_result(v) for k, v in data.items()
                }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Failed to load results from {self.results_file}: {e}")
                self._storage = {}
        else:
            self._storage = {}

    def _save_data(self) -> None:
        """Save results to file."""
        try:
            data = {str(k): self._result_to_dict(v) for k, v in self._storage.items()}
            with open(self.results_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save results to {self.results_file}: {e}")

    def _result_to_dict(self, result: DetectionResult) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        # Serialize anomalies
        anomalies_data = []
        for anomaly in result.anomalies:
            anomalies_data.append(
                {
                    "index": anomaly.index,
                    "score": anomaly.score.value,
                    "explanation": anomaly.explanation,
                }
            )

        # Serialize scores
        scores_data = []
        for score in result.scores:
            if hasattr(score, "value"):
                scores_data.append(score.value)
            else:
                scores_data.append(float(score))

        return {
            "id": str(result.id),
            "detector_id": str(result.detector_id),
            "dataset_id": str(result.dataset_id),
            "anomalies": anomalies_data,
            "scores": scores_data,
            "labels": result.labels.tolist()
            if hasattr(result.labels, "tolist")
            else list(result.labels),
            "threshold": result.threshold,
            "timestamp": result.timestamp.isoformat(),
            "metadata": result.metadata,
        }

    def _dict_to_result(self, data: dict[str, Any]) -> DetectionResult:
        """Convert dictionary to result object."""
        import numpy as np

        from pynomaly.domain.entities.anomaly import Anomaly
        from pynomaly.domain.value_objects import AnomalyScore

        timestamp = datetime.fromisoformat(data["timestamp"])

        # Convert anomalies data back to Anomaly objects
        anomalies = []
        if "anomalies" in data:
            for anomaly_data in data["anomalies"]:
                anomaly = Anomaly(
                    index=anomaly_data["index"],
                    score=AnomalyScore(anomaly_data["score"]),
                    explanation=anomaly_data.get("explanation", ""),
                )
                anomalies.append(anomaly)

        # Convert scores to AnomalyScore objects if they're not already
        scores = []
        if "scores" in data:
            for score_val in data["scores"]:
                if isinstance(score_val, dict):
                    scores.append(AnomalyScore(float(score_val["value"])))
                else:
                    # Handle both string and numeric scores
                    scores.append(AnomalyScore(float(score_val)))

        return DetectionResult(
            detector_id=UUID(data["detector_id"]),
            dataset_id=UUID(data["dataset_id"]),
            anomalies=anomalies,
            scores=scores,
            labels=np.array(data["labels"]),
            threshold=data["threshold"],
            id=UUID(data["id"]),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )

    def save(self, entity: DetectionResult) -> None:
        """Save a result to the repository."""
        self._storage[entity.id] = entity
        self._save_data()

    def find_by_id(self, entity_id: UUID) -> DetectionResult | None:
        """Find a result by its ID."""
        return self._storage.get(entity_id)

    def find_all(self) -> list[DetectionResult]:
        """Find all results in the repository."""
        return list(self._storage.values())

    def delete(self, entity_id: UUID) -> bool:
        """Delete a result by its ID."""
        if entity_id in self._storage:
            del self._storage[entity_id]
            self._save_data()
            return True
        return False

    def count(self) -> int:
        """Count total number of results."""
        return len(self._storage)

    def find_recent(self, limit: int = 10) -> list[DetectionResult]:
        """Find recent detection results."""
        results = sorted(
            self._storage.values(), key=lambda r: r.timestamp, reverse=True
        )
        return results[:limit]

    def find_by_detector(self, detector_id: UUID) -> list[DetectionResult]:
        """Find results by detector ID."""
        return [r for r in self._storage.values() if r.detector_id == detector_id]

    def find_by_dataset(self, dataset_id: UUID) -> list[DetectionResult]:
        """Find results by dataset ID."""
        return [r for r in self._storage.values() if r.dataset_id == dataset_id]

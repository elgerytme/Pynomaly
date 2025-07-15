"""Application layer repository protocols."""

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from ...domain.entities.dataset import Dataset
from ...domain.entities.detection_result import DetectionResult
from ...domain.entities.detector import Detector


@runtime_checkable
class ApplicationRepositoryProtocol(Protocol):
    """Base repository protocol for application layer."""

    def save(self, entity: Any) -> None:
        """Save an entity."""
        ...

    def find_by_id(self, entity_id: UUID) -> Any | None:
        """Find entity by ID."""
        ...

    def find_all(self) -> list[Any]:
        """Find all entities."""
        ...

    def delete(self, entity_id: UUID) -> bool:
        """Delete entity by ID."""
        ...


@runtime_checkable
class DetectorRepositoryProtocol(Protocol):
    """Repository protocol for detector entities."""

    def save(self, detector: Detector) -> None:
        """Save a detector."""
        ...

    def find_by_id(self, detector_id: UUID) -> Detector | None:
        """Find detector by ID."""
        ...

    def find_by_name(self, name: str) -> Detector | None:
        """Find detector by name."""
        ...

    def find_all(self) -> list[Detector]:
        """Find all detectors."""
        ...

    def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
        """Find detectors by algorithm name."""
        ...

    def delete(self, detector_id: UUID) -> bool:
        """Delete detector by ID."""
        ...


@runtime_checkable
class ModelRepositoryProtocol(Protocol):
    """Repository protocol for model persistence."""

    def save_model(
        self,
        model_id: str,
        model_data: bytes,
        metadata: dict[str, Any]
    ) -> None:
        """Save a model with metadata."""
        ...

    def load_model(self, model_id: str) -> tuple[bytes, dict[str, Any]] | None:
        """Load model data and metadata."""
        ...

    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        ...

    def list_models(self) -> list[dict[str, Any]]:
        """List all models with metadata."""
        ...


@runtime_checkable
class DatasetRepositoryProtocol(Protocol):
    """Repository protocol for dataset entities."""

    def save(self, dataset: Dataset) -> None:
        """Save a dataset."""
        ...

    def find_by_id(self, dataset_id: UUID) -> Dataset | None:
        """Find dataset by ID."""
        ...

    def find_by_name(self, name: str) -> Dataset | None:
        """Find dataset by name."""
        ...

    def find_all(self) -> list[Dataset]:
        """Find all datasets."""
        ...

    def delete(self, dataset_id: UUID) -> bool:
        """Delete dataset by ID."""
        ...


@runtime_checkable
class DetectionResultRepositoryProtocol(Protocol):
    """Repository protocol for detection results."""

    def save(self, result: DetectionResult) -> None:
        """Save a detection result."""
        ...

    def find_by_id(self, result_id: UUID) -> DetectionResult | None:
        """Find detection result by ID."""
        ...

    def find_by_dataset_id(self, dataset_id: UUID) -> list[DetectionResult]:
        """Find detection results by dataset ID."""
        ...

    def find_all(self) -> list[DetectionResult]:
        """Find all detection results."""
        ...

    def delete(self, result_id: UUID) -> bool:
        """Delete detection result by ID."""
        ...

"""Repository service providing unified access to all repositories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynomaly.shared.protocols.repository_protocol import (
        DatasetRepositoryProtocol,
        DetectionResultRepositoryProtocol,
        DetectorRepositoryProtocol,
    )


class RepositoryService:
    """Service providing unified access to all repositories."""

    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        dataset_repository: DatasetRepositoryProtocol,
        result_repository: DetectionResultRepositoryProtocol,
    ) -> None:
        """Initialize repository service.

        Args:
            detector_repository: Repository for detector entities
            dataset_repository: Repository for dataset entities
            result_repository: Repository for detection result entities
        """
        self._detector_repository = detector_repository
        self._dataset_repository = dataset_repository
        self._result_repository = result_repository

    @property
    def detectors(self) -> DetectorRepositoryProtocol:
        """Get detector repository."""
        return self._detector_repository

    @property
    def datasets(self) -> DatasetRepositoryProtocol:
        """Get dataset repository."""
        return self._dataset_repository

    @property
    def results(self) -> DetectionResultRepositoryProtocol:
        """Get detection result repository."""
        return self._result_repository

    def detector_repository(self) -> DetectorRepositoryProtocol:
        """Get detector repository (method interface)."""
        return self._detector_repository

    def dataset_repository(self) -> DatasetRepositoryProtocol:
        """Get dataset repository (method interface)."""
        return self._dataset_repository

    def result_repository(self) -> DetectionResultRepositoryProtocol:
        """Get detection result repository (method interface)."""
        return self._result_repository

    def get_storage_stats(self) -> dict[str, int]:
        """Get storage statistics across all repositories."""
        return {
            "detectors": self._detector_repository.count(),
            "datasets": self._dataset_repository.count(),
            "results": self._result_repository.count(),
        }

    def clear_all(self) -> dict[str, int]:
        """Clear all repositories and return counts of deleted items."""
        deleted_counts = {}

        # Clear results first (no dependencies)
        results = self._result_repository.find_all()
        for result in results:
            self._result_repository.delete(result.id)
        deleted_counts["results"] = len(results)

        # Clear datasets
        datasets = self._dataset_repository.find_all()
        for dataset in datasets:
            self._dataset_repository.delete(dataset.id)
        deleted_counts["datasets"] = len(datasets)

        # Clear detectors
        detectors = self._detector_repository.find_all()
        for detector in detectors:
            self._detector_repository.delete(detector.id)
        deleted_counts["detectors"] = len(detectors)

        return deleted_counts

    def health_check(self) -> dict[str, bool]:
        """Perform health check on all repositories."""
        health = {}

        try:
            self._detector_repository.count()
            health["detector_repository"] = True
        except Exception:
            health["detector_repository"] = False

        try:
            self._dataset_repository.count()
            health["dataset_repository"] = True
        except Exception:
            health["dataset_repository"] = False

        try:
            self._result_repository.count()
            health["result_repository"] = True
        except Exception:
            health["result_repository"] = False

        return health


def create_memory_repository_service() -> RepositoryService:
    """Create repository service with in-memory implementations."""
    from pynomaly.infrastructure.repositories.memory_repository import (
        MemoryDatasetRepository,
        MemoryDetectionResultRepository,
        MemoryDetectorRepository,
    )

    return RepositoryService(
        detector_repository=MemoryDetectorRepository(),
        dataset_repository=MemoryDatasetRepository(),
        result_repository=MemoryDetectionResultRepository(),
    )


def create_filesystem_repository_service(base_path: str = "data") -> RepositoryService:
    """Create repository service with file system implementations.

    Args:
        base_path: Base directory for file storage
    """
    from pynomaly.infrastructure.repositories.memory_repository import (
        FileSystemDetectorRepository,
        MemoryDatasetRepository,
        MemoryDetectionResultRepository,
    )

    return RepositoryService(
        detector_repository=FileSystemDetectorRepository(f"{base_path}/detectors"),
        dataset_repository=MemoryDatasetRepository(f"{base_path}/datasets"),
        result_repository=MemoryDetectionResultRepository(),
    )

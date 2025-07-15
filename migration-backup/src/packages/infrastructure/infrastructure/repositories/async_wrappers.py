"""Async wrapper classes for repository protocols.

This module provides async wrappers around synchronous repository implementations
to support application services that expect async repository methods.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.shared.protocols import (
    DatasetRepositoryProtocol,
    DetectionResultRepositoryProtocol,
    DetectorRepositoryProtocol,
)


class AsyncDetectorRepositoryWrapper:
    """Async wrapper for DetectorRepositoryProtocol implementations."""

    def __init__(self, sync_repository: DetectorRepositoryProtocol):
        """Initialize with a synchronous repository implementation.

        Args:
            sync_repository: The synchronous repository to wrap
        """
        self._sync_repo = sync_repository

    async def save(self, entity: Detector) -> None:
        """Save a detector to the repository."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.save, entity
        )

    async def find_by_id(self, entity_id: UUID) -> Detector | None:
        """Find a detector by its ID."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_by_id, entity_id
        )

    async def find_all(self) -> list[Detector]:
        """Find all detectors in the repository."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_all
        )

    async def delete(self, entity_id: UUID) -> bool:
        """Delete a detector by its ID."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.delete, entity_id
        )

    async def exists(self, entity_id: UUID) -> bool:
        """Check if a detector exists."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.exists, entity_id
        )

    async def count(self) -> int:
        """Count total number of detectors."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.count
        )

    async def find_by_name(self, name: str) -> Detector | None:
        """Find a detector by name."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_by_name, name
        )

    async def find_by_algorithm(self, algorithm_name: str) -> list[Detector]:
        """Find all detectors using a specific algorithm."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_by_algorithm, algorithm_name
        )

    async def find_fitted(self) -> list[Detector]:
        """Find all fitted detectors."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_fitted
        )

    async def save_model_artifact(self, detector_id: UUID, artifact: bytes) -> None:
        """Save the trained model artifact."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.save_model_artifact, detector_id, artifact
        )

    async def load_model_artifact(self, detector_id: UUID) -> bytes | None:
        """Load the trained model artifact."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.load_model_artifact, detector_id
        )

    # Compatibility methods for legacy code
    async def get(self, entity_id: UUID) -> Detector | None:
        """Alias for find_by_id for compatibility."""
        return await self.find_by_id(entity_id)

    async def get_by_id(self, entity_id: UUID) -> Detector | None:
        """Alias for find_by_id for compatibility."""
        return await self.find_by_id(entity_id)


class AsyncDatasetRepositoryWrapper:
    """Async wrapper for DatasetRepositoryProtocol implementations."""

    def __init__(self, sync_repository: DatasetRepositoryProtocol):
        """Initialize with a synchronous repository implementation.

        Args:
            sync_repository: The synchronous repository to wrap
        """
        self._sync_repo = sync_repository

    async def save(self, entity: Dataset) -> None:
        """Save a dataset to the repository."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.save, entity
        )

    async def find_by_id(self, entity_id: UUID) -> Dataset | None:
        """Find a dataset by its ID."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_by_id, entity_id
        )

    async def find_all(self) -> list[Dataset]:
        """Find all datasets in the repository."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_all
        )

    async def delete(self, entity_id: UUID) -> bool:
        """Delete a dataset by its ID."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.delete, entity_id
        )

    async def exists(self, entity_id: UUID) -> bool:
        """Check if a dataset exists."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.exists, entity_id
        )

    async def count(self) -> int:
        """Count total number of datasets."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.count
        )

    async def find_by_name(self, name: str) -> Dataset | None:
        """Find a dataset by name."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_by_name, name
        )

    async def find_by_metadata(self, key: str, value: Any) -> list[Dataset]:
        """Find datasets by metadata key-value pair."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_by_metadata, key, value
        )

    async def save_data(self, dataset_id: UUID, format: str = "parquet") -> str:
        """Save dataset data to persistent storage."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.save_data, dataset_id, format
        )

    async def load_data(self, dataset_id: UUID) -> Dataset | None:
        """Load dataset with its data from storage."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.load_data, dataset_id
        )

    # Compatibility methods for legacy code
    async def get(self, entity_id: UUID) -> Dataset | None:
        """Alias for find_by_id for compatibility."""
        return await self.find_by_id(entity_id)

    async def get_by_id(self, entity_id: UUID) -> Dataset | None:
        """Alias for find_by_id for compatibility."""
        return await self.find_by_id(entity_id)


class AsyncDetectionResultRepositoryWrapper:
    """Async wrapper for DetectionResultRepositoryProtocol implementations."""

    def __init__(self, sync_repository: DetectionResultRepositoryProtocol):
        """Initialize with a synchronous repository implementation.

        Args:
            sync_repository: The synchronous repository to wrap
        """
        self._sync_repo = sync_repository

    async def save(self, entity: DetectionResult) -> None:
        """Save a detection result to the repository."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.save, entity
        )

    async def find_by_id(self, entity_id: UUID) -> DetectionResult | None:
        """Find a detection result by its ID."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_by_id, entity_id
        )

    async def find_all(self) -> list[DetectionResult]:
        """Find all detection results in the repository."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_all
        )

    async def delete(self, entity_id: UUID) -> bool:
        """Delete a detection result by its ID."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.delete, entity_id
        )

    async def exists(self, entity_id: UUID) -> bool:
        """Check if a detection result exists."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.exists, entity_id
        )

    async def count(self) -> int:
        """Count total number of detection results."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.count
        )

    async def find_by_detector(self, detector_id: UUID) -> list[DetectionResult]:
        """Find all results from a specific detector."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_by_detector, detector_id
        )

    async def find_by_dataset(self, dataset_id: UUID) -> list[DetectionResult]:
        """Find all results for a specific dataset."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_by_dataset, dataset_id
        )

    async def find_recent(self, limit: int = 10) -> list[DetectionResult]:
        """Find most recent detection results."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.find_recent, limit
        )

    async def get_summary_stats(self, result_id: UUID) -> dict[str, Any]:
        """Get summary statistics for a result."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_repo.get_summary_stats, result_id
        )

    # Compatibility methods for legacy code
    async def get(self, entity_id: UUID) -> DetectionResult | None:
        """Alias for find_by_id for compatibility."""
        return await self.find_by_id(entity_id)

    async def get_by_id(self, entity_id: UUID) -> DetectionResult | None:
        """Alias for find_by_id for compatibility."""
        return await self.find_by_id(entity_id)

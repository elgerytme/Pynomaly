"""Dataset repository implementations - DEPRECATED.

This module is deprecated. Use DatasetRepositoryProtocol from
monorepo.shared.protocols.repository_protocol instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID
from warnings import warn

from monorepo.domain.entities.dataset import Dataset


class DatasetRepository(ABC):
    """Abstract dataset repository interface.

    DEPRECATED: Use DatasetRepositoryProtocol from monorepo.shared.protocols instead.
    This class exists only for backward compatibility and will be removed in v2.0.
    """

    def __init__(self):
        """Initialize repository with deprecation warning."""
        warn(
            "DatasetRepository is deprecated. Use DatasetRepositoryProtocol from "
            "monorepo.shared.protocols.repository_protocol instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    @abstractmethod
    async def save(self, dataset: Dataset) -> None:
        """Save a dataset."""
        pass

    @abstractmethod
    async def find_by_id(self, dataset_id: UUID) -> Dataset | None:
        """Find a dataset by ID."""
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> Dataset | None:
        """Find a dataset by name."""
        pass

    @abstractmethod
    async def find_all(self) -> list[Dataset]:
        """Find all datasets."""
        pass

    @abstractmethod
    async def delete(self, dataset_id: UUID) -> bool:
        """Delete a dataset."""
        pass

    @abstractmethod
    async def exists(self, dataset_id: UUID) -> bool:
        """Check if dataset exists."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Count total datasets."""
        pass

    @abstractmethod
    async def find_by_metadata(self, key: str, value: Any) -> list[Dataset]:
        """Find datasets by metadata key-value pair."""
        pass

    @abstractmethod
    async def save_data(self, dataset_id: UUID, format: str = "parquet") -> str:
        """Save dataset data to persistent storage."""
        pass

    @abstractmethod
    async def load_data(self, dataset_id: UUID) -> Dataset | None:
        """Load dataset with its data from storage."""
        pass

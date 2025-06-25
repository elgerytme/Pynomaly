"""Dataset repository implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pynomaly.domain.entities.dataset import Dataset


class DatasetRepository(ABC):
    """Abstract dataset repository interface."""

    @abstractmethod
    def save(self, dataset: Dataset) -> Dataset:
        """Save a dataset."""
        pass

    @abstractmethod
    def find_by_id(self, dataset_id: str) -> Dataset | None:
        """Find a dataset by ID."""
        pass

    @abstractmethod
    def find_by_name(self, name: str) -> Dataset | None:
        """Find a dataset by name."""
        pass

    @abstractmethod
    def find_all(self) -> list[Dataset]:
        """Find all datasets."""
        pass

    @abstractmethod
    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset."""
        pass

    @abstractmethod
    def exists(self, dataset_id: str) -> bool:
        """Check if dataset exists."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total datasets."""
        pass

    @abstractmethod
    def find_by_tags(self, tags: list[str]) -> list[Dataset]:
        """Find datasets by tags."""
        pass

    @abstractmethod
    def find_recent(self, limit: int = 10) -> list[Dataset]:
        """Find recently created datasets."""
        pass

    @abstractmethod
    def update_metadata(self, dataset_id: str, metadata: dict[str, Any]) -> bool:
        """Update dataset metadata."""
        pass

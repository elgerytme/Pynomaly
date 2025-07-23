
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.domain.entities.data_profile import DataProfile


class DataProfileRepository(ABC):
    """Port for data profile repository."""

    @abstractmethod
    def get_by_id(self, id: UUID) -> Optional[DataProfile]:
        """Get a data profile by its ID."""
        raise NotImplementedError

    @abstractmethod
    def save(self, data_profile: DataProfile) -> None:
        """Save a data profile."""
        raise NotImplementedError

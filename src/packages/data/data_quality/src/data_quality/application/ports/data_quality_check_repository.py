
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from ...domain.entities.data_quality_check import DataQualityCheck


class DataQualityCheckRepository(ABC):
    """Port for data quality check repository."""

    @abstractmethod
    def get_by_id(self, id: UUID) -> Optional[DataQualityCheck]:
        """Get a data quality check by its ID."""
        raise NotImplementedError

    @abstractmethod
    def save(self, data_quality_check: DataQualityCheck) -> None:
        """Save a data quality check."""
        raise NotImplementedError

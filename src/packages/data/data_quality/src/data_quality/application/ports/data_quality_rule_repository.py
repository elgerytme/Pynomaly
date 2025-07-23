
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.domain.entities.data_quality_rule import DataQualityRule


class DataQualityRuleRepository(ABC):
    """Port for data quality rule repository."""

    @abstractmethod
    def get_by_id(self, id: UUID) -> Optional[DataQualityRule]:
        """Get a data quality rule by its ID."""
        raise NotImplementedError

    @abstractmethod
    def save(self, data_quality_rule: DataQualityRule) -> None:
        """Save a data quality rule."""
        raise NotImplementedError

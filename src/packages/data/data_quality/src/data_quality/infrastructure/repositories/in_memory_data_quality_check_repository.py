
from typing import Dict, Optional
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.application.ports.data_quality_check_repository import DataQualityCheckRepository
from src.packages.data.data_quality.src.data_quality.domain.entities.data_quality_check import DataQualityCheck


class InMemoryDataQualityCheckRepository(DataQualityCheckRepository):
    """In-memory implementation of DataQualityCheckRepository."""

    def __init__(self):
        self.checks: Dict[UUID, DataQualityCheck] = {}

    def get_by_id(self, id: UUID) -> Optional[DataQualityCheck]:
        """Get a data quality check by its ID."""
        return self.checks.get(id)

    def save(self, data_quality_check: DataQualityCheck) -> None:
        """Save a data quality check."""
        self.checks[data_quality_check.id] = data_quality_check


from uuid import UUID

from src.packages.data.data_quality.src.data_quality.application.ports.data_quality_check_repository import DataQualityCheckRepository
from src.packages.data.data_quality.src.data_quality.domain.entities.data_quality_check import DataQualityCheck


class DataQualityCheckService:
    """Service for data quality checks."""

    def __init__(self, data_quality_check_repository: DataQualityCheckRepository):
        self.data_quality_check_repository = data_quality_check_repository

    def run_check(self, check_id: UUID) -> DataQualityCheck:
        """Run a data quality check."""
        data_quality_check = self.data_quality_check_repository.get_by_id(check_id)
        # In a real implementation, this would trigger the check execution
        # and update the check's status and result.
        return data_quality_check

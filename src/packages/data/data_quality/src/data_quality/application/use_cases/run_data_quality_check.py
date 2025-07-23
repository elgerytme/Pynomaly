
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.application.services.data_quality_check_service import DataQualityCheckService
from src.packages.data.data_quality.src.data_quality.domain.entities.data_quality_check import DataQualityCheck


class RunDataQualityCheckUseCase:
    """Use case for running a data quality check."""

    def __init__(self, data_quality_check_service: DataQualityCheckService):
        self.data_quality_check_service = data_quality_check_service

    def execute(self, check_id: UUID) -> DataQualityCheck:
        """Execute the use case."""
        return self.data_quality_check_service.run_check(check_id)

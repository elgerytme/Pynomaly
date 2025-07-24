
from uuid import UUID
from typing import Any, Dict

from ..services.data_quality_check_service import DataQualityCheckService
from ...domain.entities.data_quality_check import DataQualityCheck


class RunDataQualityCheckUseCase:
    """Use case for running a data quality check."""

    def __init__(self, data_quality_check_service: DataQualityCheckService):
        self.data_quality_check_service = data_quality_check_service

    def execute(self, check_id: UUID, source_config: Dict[str, Any]) -> DataQualityCheck:
        """Execute the use case.

        Args:
            check_id: The ID of the data quality check to run.
            source_config: Configuration for the data source adapter.

        Returns:
            The updated DataQualityCheck entity with the result.
        """
        return self.data_quality_check_service.run_check(check_id, source_config)

from typing import Any, Dict

from src.packages.data.data_quality.src.data_quality.application.services.data_profiling_service import DataProfilingService
from src.packages.data.data_quality.src.data_quality.domain.entities.data_profile import DataProfile
from src.packages.data.data_quality.src.data_quality.infrastructure.adapters.data_source_adapter import DataSourceAdapter


class CreateDataProfileUseCase:
    """Use case for creating a data profile."""

    def __init__(self, data_profiling_service: DataProfilingService):
        self.data_profiling_service = data_profiling_service

    def execute(self, dataset_name: str, data_source_adapter: DataSourceAdapter, source_config: Dict[str, Any]) -> DataProfile:
        """Execute the use case.

        Args:
            dataset_name: The name of the dataset to profile.
            data_source_adapter: An instance of a DataSourceAdapter to read the data.
            source_config: Configuration for the data source adapter.

        Returns:
            The created DataProfile entity.
        """
        return self.data_profiling_service.create_profile(dataset_name, data_source_adapter, source_config)
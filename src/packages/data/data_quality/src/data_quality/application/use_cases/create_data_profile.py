
from src.packages.data.data_quality.src.data_quality.application.services.data_profiling_service import DataProfilingService
from src.packages.data.data_quality.src.data_quality.domain.entities.data_profile import DataProfile


class CreateDataProfileUseCase:
    """Use case for creating a data profile."""

    def __init__(self, data_profiling_service: DataProfilingService):
        self.data_profiling_service = data_profiling_service

    def execute(self, dataset_name: str) -> DataProfile:
        """Execute the use case."""
        return self.data_profiling_service.create_profile(dataset_name)

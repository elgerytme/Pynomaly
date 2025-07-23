
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.application.ports.data_profile_repository import DataProfileRepository
from src.packages.data.data_quality.src.data_quality.domain.entities.data_profile import DataProfile


class DataProfilingService:
    """Service for data profiling."""

    def __init__(self, data_profile_repository: DataProfileRepository):
        self.data_profile_repository = data_profile_repository

    def create_profile(self, dataset_name: str) -> DataProfile:
        """Create a new data profile."""
        data_profile = DataProfile(dataset_name=dataset_name)
        self.data_profile_repository.save(data_profile)
        return data_profile

    def get_profile(self, profile_id: UUID) -> DataProfile:
        """Get a data profile by its ID."""
        return self.data_profile_repository.get_by_id(profile_id)

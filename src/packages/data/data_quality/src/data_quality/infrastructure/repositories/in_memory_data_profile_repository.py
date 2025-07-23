
from typing import Dict, Optional
from uuid import UUID

from src.packages.data.data_quality.src.data_quality.application.ports.data_profile_repository import DataProfileRepository
from src.packages.data.data_quality.src.data_quality.domain.entities.data_profile import DataProfile


class InMemoryDataProfileRepository(DataProfileRepository):
    """In-memory implementation of DataProfileRepository."""

    def __init__(self):
        self.profiles: Dict[UUID, DataProfile] = {}

    def get_by_id(self, id: UUID) -> Optional[DataProfile]:
        """Get a data profile by its ID."""
        return self.profiles.get(id)

    def save(self, data_profile: DataProfile) -> None:
        """Save a data profile."""
        self.profiles[data_profile.id] = data_profile

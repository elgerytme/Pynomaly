"""Data Profile repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.data_profile import DataProfile, ProfileId, DatasetId, ProfilingStatus


class DataProfileRepository(ABC):
    """Repository interface for data profile persistence."""
    
    @abstractmethod
    async def save(self, profile: DataProfile) -> None:
        """Save a data profile."""
        pass
    
    @abstractmethod
    async def get_by_id(self, profile_id: ProfileId) -> Optional[DataProfile]:
        """Get data profile by ID."""
        pass
    
    @abstractmethod
    async def get_by_dataset_id(self, dataset_id: DatasetId) -> List[DataProfile]:
        """Get all profiles for a dataset."""
        pass
    
    @abstractmethod
    async def get_latest_by_dataset_id(self, dataset_id: DatasetId) -> Optional[DataProfile]:
        """Get the latest profile for a dataset."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: ProfilingStatus) -> List[DataProfile]:
        """Get profiles by status."""
        pass
    
    @abstractmethod
    async def get_by_source_type(self, source_type: str) -> List[DataProfile]:
        """Get profiles by source type."""
        pass
    
    @abstractmethod
    async def delete(self, profile_id: ProfileId) -> None:
        """Delete a data profile."""
        pass
    
    @abstractmethod
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[DataProfile]:
        """List all data profiles with pagination."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count total number of profiles."""
        pass
    
    @abstractmethod
    async def search_by_column_name(self, column_name: str) -> List[DataProfile]:
        """Search profiles containing a specific column name."""
        pass
    
    @abstractmethod
    async def get_profiles_with_quality_issues(self, severity: Optional[str] = None) -> List[DataProfile]:
        """Get profiles that have quality issues."""
        pass
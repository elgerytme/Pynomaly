"""Data origin repository interface."""

from abc import abstractmethod
from typing import List, Optional
from uuid import UUID
from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from ..entities.data_origin import DataOrigin, OriginType


class DataOriginRepository(RepositoryInterface[DataOrigin]):
    """Repository interface for data origins."""
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[DataOrigin]:
        """Find origin by name."""
        pass
    
    @abstractmethod
    async def find_by_type(self, origin_type: OriginType) -> List[DataOrigin]:
        """Find origins by type."""
        pass
    
    @abstractmethod
    async def find_by_system_name(self, system_name: str) -> List[DataOrigin]:
        """Find origins by source system name."""
        pass
    
    @abstractmethod
    async def find_active_origins(self) -> List[DataOrigin]:
        """Find all active origins."""
        pass
    
    @abstractmethod
    async def find_trusted_origins(self) -> List[DataOrigin]:
        """Find all trusted origins."""
        pass
    
    @abstractmethod
    async def find_by_reliability_threshold(self, min_score: float) -> List[DataOrigin]:
        """Find origins meeting minimum reliability score."""
        pass
    
    @abstractmethod
    async def find_by_owner(self, owner: str) -> List[DataOrigin]:
        """Find origins by owner."""
        pass
    
    @abstractmethod
    async def find_frequently_accessed(self, min_frequency: int) -> List[DataOrigin]:
        """Find origins with high access frequency."""
        pass
    
    @abstractmethod
    async def find_by_tags(self, tags: List[str]) -> List[DataOrigin]:
        """Find origins with specific tags."""
        pass
    
    @abstractmethod
    async def update_access_stats(self, origin_id: UUID) -> None:
        """Update access statistics for an origin."""
        pass
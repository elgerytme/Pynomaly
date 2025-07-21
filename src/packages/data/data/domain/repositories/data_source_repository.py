"""Data source repository interface."""

from abc import abstractmethod
from typing import List, Optional
from uuid import UUID
from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from ..entities.data_source import DataSource, SourceStatus, AccessPattern


class DataSourceRepository(RepositoryInterface[DataSource]):
    """Repository interface for data sources."""
    
    @abstractmethod
    async def find_by_origin_id(self, origin_id: UUID) -> List[DataSource]:
        """Find sources by origin ID."""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[DataSource]:
        """Find source by name."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: SourceStatus) -> List[DataSource]:
        """Find sources by status."""
        pass
    
    @abstractmethod
    async def find_by_access_pattern(self, pattern: AccessPattern) -> List[DataSource]:
        """Find sources by access pattern."""
        pass
    
    @abstractmethod
    async def find_active_sources(self) -> List[DataSource]:
        """Find all active sources."""
        pass
    
    @abstractmethod
    async def find_error_sources(self) -> List[DataSource]:
        """Find sources with errors."""
        pass
    
    @abstractmethod
    async def find_by_format(self, format_type: str) -> List[DataSource]:
        """Find sources by data format."""
        pass
    
    @abstractmethod
    async def find_by_quality_threshold(self, min_score: float) -> List[DataSource]:
        """Find sources meeting minimum quality score."""
        pass
    
    @abstractmethod
    async def find_by_availability_threshold(self, min_score: float) -> List[DataSource]:
        """Find sources meeting minimum availability score."""
        pass
    
    @abstractmethod
    async def find_streaming_sources(self) -> List[DataSource]:
        """Find all streaming data sources."""
        pass
    
    @abstractmethod
    async def find_batch_sources(self) -> List[DataSource]:
        """Find all batch data sources."""
        pass
    
    @abstractmethod
    async def find_by_size_range(self, min_bytes: int, max_bytes: int) -> List[DataSource]:
        """Find sources within size range."""
        pass
    
    @abstractmethod
    async def find_recently_modified(self, hours: int) -> List[DataSource]:
        """Find sources modified within specified hours."""
        pass
    
    @abstractmethod
    async def find_by_consumer(self, consumer: str) -> List[DataSource]:
        """Find sources consumed by specified system."""
        pass
    
    @abstractmethod
    async def find_by_tags(self, tags: List[str]) -> List[DataSource]:
        """Find sources with specific tags."""
        pass
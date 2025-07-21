"""Data set repository interface."""

from abc import abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from ..entities.data_set import DataSet, DataSetType, DataSetStatus


class DataSetRepository(RepositoryInterface[DataSet]):
    """Repository interface for data sets."""
    
    @abstractmethod
    async def find_by_asset_id(self, asset_id: UUID) -> List[DataSet]:
        """Find datasets by parent asset ID."""
        pass
    
    @abstractmethod
    async def find_by_source_id(self, source_id: UUID) -> List[DataSet]:
        """Find datasets by source ID."""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[DataSet]:
        """Find dataset by name."""
        pass
    
    @abstractmethod
    async def find_by_type(self, dataset_type: DataSetType) -> List[DataSet]:
        """Find datasets by type."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: DataSetStatus) -> List[DataSet]:
        """Find datasets by status."""
        pass
    
    @abstractmethod
    async def find_active_datasets(self) -> List[DataSet]:
        """Find all active datasets."""
        pass
    
    @abstractmethod
    async def find_error_datasets(self) -> List[DataSet]:
        """Find datasets with errors."""
        pass
    
    @abstractmethod
    async def find_by_quality_threshold(self, min_score: float) -> List[DataSet]:
        """Find datasets meeting minimum quality score."""
        pass
    
    @abstractmethod
    async def find_by_freshness(self, max_hours: float) -> List[DataSet]:
        """Find datasets within freshness threshold."""
        pass
    
    @abstractmethod
    async def find_by_size_range(self, min_bytes: int, max_bytes: int) -> List[DataSet]:
        """Find datasets within size range."""
        pass
    
    @abstractmethod
    async def find_by_record_count_range(self, min_records: int, max_records: int) -> List[DataSet]:
        """Find datasets within record count range."""
        pass
    
    @abstractmethod
    async def find_recently_loaded(self, hours: int) -> List[DataSet]:
        """Find datasets loaded within specified hours."""
        pass
    
    @abstractmethod
    async def find_recently_validated(self, hours: int) -> List[DataSet]:
        """Find datasets validated within specified hours."""
        pass
    
    @abstractmethod
    async def find_requiring_validation(self) -> List[DataSet]:
        """Find datasets requiring validation."""
        pass
    
    @abstractmethod
    async def find_requiring_cleanup(self) -> List[DataSet]:
        """Find datasets requiring data cleanup."""
        pass
    
    @abstractmethod
    async def find_by_partition_column(self, column_name: str) -> List[DataSet]:
        """Find datasets partitioned by specific column."""
        pass
    
    @abstractmethod
    async def find_by_classification_level(self, sensitivity_level: str) -> List[DataSet]:
        """Find datasets by data classification level."""
        pass
    
    @abstractmethod
    async def find_with_quality_issues(self) -> List[DataSet]:
        """Find datasets with quality issues."""
        pass
    
    @abstractmethod
    async def find_archived_datasets(self) -> List[DataSet]:
        """Find archived datasets."""
        pass
    
    @abstractmethod
    async def find_by_tags(self, tags: List[str]) -> List[DataSet]:
        """Find datasets with specific tags."""
        pass
    
    @abstractmethod
    async def get_dataset_statistics(self, dataset_id: UUID) -> Dict[str, Any]:
        """Get comprehensive statistics for a dataset."""
        pass
    
    @abstractmethod
    async def get_quality_trend(self, dataset_id: UUID, days: int) -> List[Dict[str, Any]]:
        """Get quality score trend for a dataset."""
        pass
    
    @abstractmethod
    async def get_access_patterns(self, dataset_id: UUID, days: int) -> List[Dict[str, Any]]:
        """Get access patterns for a dataset."""
        pass
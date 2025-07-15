"""Repository interface for DatasetProfile entities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from packages.core.domain.abstractions.repository_interface import RepositoryInterface
from packages.data_science.domain.entities.dataset_profile import DatasetProfile


class DatasetProfileRepository(RepositoryInterface[DatasetProfile], ABC):
    """Repository interface for dataset profile persistence operations."""

    @abstractmethod
    async def find_by_dataset_id(self, dataset_id: str) -> Optional[DatasetProfile]:
        """Find dataset profile by dataset ID.
        
        Args:
            dataset_id: Dataset ID to search for
            
        Returns:
            DatasetProfile if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[DatasetProfile]:
        """Find dataset profile by name.
        
        Args:
            name: Dataset name to search for
            
        Returns:
            DatasetProfile if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_by_source(self, source: str) -> list[DatasetProfile]:
        """Find dataset profiles by data source.
        
        Args:
            source: Data source to search for
            
        Returns:
            List of dataset profiles from the specified source
        """
        pass

    @abstractmethod
    async def find_by_schema_hash(self, schema_hash: str) -> list[DatasetProfile]:
        """Find dataset profiles by schema hash.
        
        Args:
            schema_hash: Schema hash to search for
            
        Returns:
            List of dataset profiles with matching schema
        """
        pass

    @abstractmethod
    async def find_by_size_range(
        self, min_size: int, max_size: int
    ) -> list[DatasetProfile]:
        """Find dataset profiles by size range.
        
        Args:
            min_size: Minimum dataset size
            max_size: Maximum dataset size
            
        Returns:
            List of dataset profiles within the size range
        """
        pass

    @abstractmethod
    async def find_by_quality_score_range(
        self, min_score: float, max_score: float
    ) -> list[DatasetProfile]:
        """Find dataset profiles by quality score range.
        
        Args:
            min_score: Minimum quality score
            max_score: Maximum quality score
            
        Returns:
            List of dataset profiles within the quality score range
        """
        pass

    @abstractmethod
    async def find_by_tags(self, tags: list[str]) -> list[DatasetProfile]:
        """Find dataset profiles by tags.
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of dataset profiles containing any of the specified tags
        """
        pass

    @abstractmethod
    async def find_by_creation_date_range(
        self, start_date: datetime, end_date: datetime
    ) -> list[DatasetProfile]:
        """Find dataset profiles created within a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of dataset profiles created within the date range
        """
        pass

    @abstractmethod
    async def find_similar_datasets(
        self, dataset_profile_id: UUID, similarity_threshold: float = 0.8
    ) -> list[tuple[DatasetProfile, float]]:
        """Find datasets similar to the given dataset.
        
        Args:
            dataset_profile_id: Reference dataset profile ID
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of tuples (similar_dataset, similarity_score)
        """
        pass

    @abstractmethod
    async def find_datasets_with_missing_values(
        self, threshold_percentage: float = 5.0
    ) -> list[DatasetProfile]:
        """Find datasets with missing values above threshold.
        
        Args:
            threshold_percentage: Percentage threshold for missing values
            
        Returns:
            List of dataset profiles with significant missing values
        """
        pass

    @abstractmethod
    async def find_datasets_with_outliers(
        self, outlier_percentage_threshold: float = 5.0
    ) -> list[DatasetProfile]:
        """Find datasets with outliers above threshold.
        
        Args:
            outlier_percentage_threshold: Percentage threshold for outliers
            
        Returns:
            List of dataset profiles with significant outliers
        """
        pass

    @abstractmethod
    async def find_datasets_with_drift(
        self, drift_score_threshold: float = 0.1
    ) -> list[DatasetProfile]:
        """Find datasets with data drift above threshold.
        
        Args:
            drift_score_threshold: Drift score threshold
            
        Returns:
            List of dataset profiles with significant drift
        """
        pass

    @abstractmethod
    async def get_schema_evolution_history(
        self, dataset_id: str
    ) -> list[dict[str, Any]]:
        """Get schema evolution history for a dataset.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            List of schema evolution events
        """
        pass

    @abstractmethod
    async def get_quality_trends(
        self, dataset_id: str, days: int = 30
    ) -> list[dict[str, Any]]:
        """Get quality trends for a dataset over time.
        
        Args:
            dataset_id: Dataset ID
            days: Number of days to analyze
            
        Returns:
            List of quality trend data points
        """
        pass

    @abstractmethod
    async def find_datasets_by_column_names(
        self, column_names: list[str]
    ) -> list[DatasetProfile]:
        """Find datasets containing specific column names.
        
        Args:
            column_names: List of column names to search for
            
        Returns:
            List of dataset profiles containing the specified columns
        """
        pass

    @abstractmethod
    async def find_datasets_by_data_types(
        self, data_types: list[str]
    ) -> list[DatasetProfile]:
        """Find datasets containing specific data types.
        
        Args:
            data_types: List of data types to search for
            
        Returns:
            List of dataset profiles containing the specified data types
        """
        pass

    @abstractmethod
    async def archive_old_profiles(
        self, older_than_days: int = 90
    ) -> int:
        """Archive old dataset profiles.
        
        Args:
            older_than_days: Archive profiles older than this many days
            
        Returns:
            Number of profiles archived
        """
        pass

    @abstractmethod
    async def get_dataset_lineage(
        self, dataset_id: str
    ) -> dict[str, Any]:
        """Get dataset lineage information.
        
        Args:
            dataset_id: Dataset ID
            
        Returns:
            Dictionary containing lineage information
        """
        pass

    @abstractmethod
    async def validate_profile_integrity(
        self, profile_id: UUID
    ) -> dict[str, Any]:
        """Validate dataset profile integrity.
        
        Args:
            profile_id: Profile ID to validate
            
        Returns:
            Dictionary containing validation results
        """
        pass

    @abstractmethod
    async def get_profiling_recommendations(
        self, profile_id: UUID
    ) -> list[dict[str, Any]]:
        """Get recommendations for dataset improvements.
        
        Args:
            profile_id: Profile ID
            
        Returns:
            List of improvement recommendations
        """
        pass

    @abstractmethod
    async def bulk_update_quality_scores(
        self, profile_updates: dict[UUID, float]
    ) -> int:
        """Bulk update quality scores for multiple profiles.
        
        Args:
            profile_updates: Dictionary mapping profile IDs to new quality scores
            
        Returns:
            Number of profiles updated
        """
        pass

    @abstractmethod
    async def get_global_dataset_statistics(self) -> dict[str, Any]:
        """Get global statistics across all dataset profiles.
        
        Returns:
            Dictionary of global dataset statistics
        """
        pass
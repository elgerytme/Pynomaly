"""In-memory implementation of Data Profile Repository."""

from typing import Dict, List, Optional
from uuid import UUID
import structlog

from ...domain.entities.data_profile import (
    DataProfile, ProfileId, DatasetId, ProfilingStatus
)
from ...domain.repositories.data_profile_repository import DataProfileRepository

logger = structlog.get_logger(__name__)


class InMemoryDataProfileRepository(DataProfileRepository):
    """In-memory implementation for testing and development."""
    
    def __init__(self):
        self._profiles: Dict[UUID, DataProfile] = {}
        logger.info("Initialized in-memory data profile repository")
    
    async def save(self, profile: DataProfile) -> None:
        """Save a data profile."""
        self._profiles[profile.profile_id.value] = profile
        logger.debug(
            "Saved data profile",
            profile_id=str(profile.profile_id.value),
            status=profile.status.value
        )
    
    async def get_by_id(self, profile_id: ProfileId) -> Optional[DataProfile]:
        """Get data profile by ID."""
        profile = self._profiles.get(profile_id.value)
        if profile:
            logger.debug(
                "Retrieved data profile",
                profile_id=str(profile_id.value)
            )
        return profile
    
    async def get_by_dataset_id(self, dataset_id: DatasetId) -> List[DataProfile]:
        """Get all profiles for a dataset."""
        profiles = [
            profile for profile in self._profiles.values()
            if profile.dataset_id.value == dataset_id.value
        ]
        # Sort by creation time (most recent first)
        profiles.sort(key=lambda x: x.created_at, reverse=True)
        
        logger.debug(
            "Retrieved profiles by dataset",
            dataset_id=str(dataset_id.value),
            count=len(profiles)
        )
        return profiles
    
    async def get_latest_by_dataset_id(self, dataset_id: DatasetId) -> Optional[DataProfile]:
        """Get the latest profile for a dataset."""
        profiles = await self.get_by_dataset_id(dataset_id)
        return profiles[0] if profiles else None
    
    async def get_by_status(self, status: ProfilingStatus) -> List[DataProfile]:
        """Get profiles by status."""
        profiles = [
            profile for profile in self._profiles.values()
            if profile.status == status
        ]
        logger.debug(
            "Retrieved profiles by status",
            status=status.value,
            count=len(profiles)
        )
        return profiles
    
    async def get_by_source_type(self, source_type: str) -> List[DataProfile]:
        """Get profiles by source type."""
        profiles = [
            profile for profile in self._profiles.values()
            if profile.source_type == source_type
        ]
        logger.debug(
            "Retrieved profiles by source type",
            source_type=source_type,
            count=len(profiles)
        )
        return profiles
    
    async def delete(self, profile_id: ProfileId) -> None:
        """Delete a data profile."""
        if profile_id.value in self._profiles:
            del self._profiles[profile_id.value]
            logger.debug(
                "Deleted data profile",
                profile_id=str(profile_id.value)
            )
    
    async def list_all(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None
    ) -> List[DataProfile]:
        """List all data profiles with pagination."""
        profiles = list(self._profiles.values())
        
        # Sort by creation time (most recent first)
        profiles.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        if offset:
            profiles = profiles[offset:]
        if limit:
            profiles = profiles[:limit]
        
        logger.debug(
            "Listed all profiles",
            total_count=len(self._profiles),
            returned_count=len(profiles)
        )
        return profiles
    
    async def count(self) -> int:
        """Count total number of profiles."""
        count = len(self._profiles)
        logger.debug("Counted profiles", count=count)
        return count
    
    async def search_by_column_name(self, column_name: str) -> List[DataProfile]:
        """Search profiles containing a specific column name."""
        matching_profiles = []
        
        for profile in self._profiles.values():
            if profile.schema_profile:
                for column in profile.schema_profile.columns:
                    if column_name.lower() in column.column_name.lower():
                        matching_profiles.append(profile)
                        break
        
        logger.debug(
            "Searched profiles by column name",
            column_name=column_name,
            count=len(matching_profiles)
        )
        return matching_profiles
    
    async def get_profiles_with_quality_issues(
        self, 
        severity: Optional[str] = None
    ) -> List[DataProfile]:
        """Get profiles that have quality issues."""
        profiles_with_issues = []
        
        for profile in self._profiles.values():
            if profile.schema_profile:
                has_issues = False
                for column in profile.schema_profile.columns:
                    if column.quality_issues:
                        if severity:
                            # Check if any issue matches the severity
                            for issue in column.quality_issues:
                                if issue.severity == severity:
                                    has_issues = True
                                    break
                        else:
                            has_issues = True
                        
                        if has_issues:
                            break
                
                if has_issues:
                    profiles_with_issues.append(profile)
        
        logger.debug(
            "Retrieved profiles with quality issues",
            severity=severity,
            count=len(profiles_with_issues)
        )
        return profiles_with_issues
    
    # Additional utility methods for testing
    
    def clear(self) -> None:
        """Clear all profiles (for testing)."""
        self._profiles.clear()
        logger.debug("Cleared all data profiles")
    
    def get_all_sync(self) -> List[DataProfile]:
        """Get all profiles synchronously (for testing)."""
        return list(self._profiles.values())
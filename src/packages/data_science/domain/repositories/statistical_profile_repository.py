"""Statistical Profile repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from ..entities.statistical_profile import StatisticalProfile


class IStatisticalProfileRepository(ABC):
    """Repository interface for statistical profile persistence."""
    
    @abstractmethod
    async def save(self, profile: StatisticalProfile) -> None:
        """Save a statistical profile."""
        pass
    
    @abstractmethod
    async def get_by_id(self, profile_id: UUID) -> Optional[StatisticalProfile]:
        """Get profile by ID."""
        pass
    
    @abstractmethod
    async def get_by_dataset_id(self, dataset_id: UUID) -> List[StatisticalProfile]:
        """Get all profiles for a dataset."""
        pass
    
    @abstractmethod
    async def get_by_feature_name(self, feature_name: str) -> List[StatisticalProfile]:
        """Get profiles by feature name."""
        pass
    
    @abstractmethod
    async def get_by_profile_type(self, profile_type: str) -> List[StatisticalProfile]:
        """Get profiles by type."""
        pass
    
    @abstractmethod
    async def get_by_analysis_level(self, analysis_level: str) -> List[StatisticalProfile]:
        """Get profiles by analysis level."""
        pass
    
    @abstractmethod
    async def get_latest_profile(self, dataset_id: UUID, feature_name: Optional[str] = None) -> Optional[StatisticalProfile]:
        """Get latest profile for dataset/feature."""
        pass
    
    @abstractmethod
    async def get_profiles_by_date_range(self, start_date: datetime, end_date: datetime) -> List[StatisticalProfile]:
        """Get profiles created within date range."""
        pass
    
    @abstractmethod
    async def get_profiles_with_issues(self, issue_types: Optional[List[str]] = None) -> List[StatisticalProfile]:
        """Get profiles with data quality issues."""
        pass
    
    @abstractmethod
    async def get_profiles_by_quality_score(self, min_score: float, max_score: Optional[float] = None) -> List[StatisticalProfile]:
        """Get profiles by quality score range."""
        pass
    
    @abstractmethod
    async def compare_profiles(self, profile1_id: UUID, profile2_id: UUID) -> Dict[str, Any]:
        """Compare two statistical profiles."""
        pass
    
    @abstractmethod
    async def get_profile_history(self, dataset_id: UUID, feature_name: str) -> List[StatisticalProfile]:
        """Get profile history for a feature."""
        pass
    
    @abstractmethod
    async def search_profiles(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[StatisticalProfile]:
        """Search profiles by query and filters."""
        pass
    
    @abstractmethod
    async def update_quality_assessment(self, profile_id: UUID, quality_issues: List[str], 
                                      recommendations: List[str]) -> None:
        """Update profile quality assessment."""
        pass
    
    @abstractmethod
    async def add_annotation(self, profile_id: UUID, annotation: str, user_id: UUID) -> None:
        """Add annotation to profile."""
        pass
    
    @abstractmethod
    async def get_annotations(self, profile_id: UUID) -> List[Dict[str, Any]]:
        """Get profile annotations."""
        pass
    
    @abstractmethod
    async def mark_reviewed(self, profile_id: UUID, reviewer_id: UUID, notes: Optional[str] = None) -> None:
        """Mark profile as reviewed."""
        pass
    
    @abstractmethod
    async def get_unreviewed_profiles(self) -> List[StatisticalProfile]:
        """Get profiles pending review."""
        pass
    
    @abstractmethod
    async def delete(self, profile_id: UUID) -> None:
        """Delete a profile."""
        pass
    
    @abstractmethod
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[StatisticalProfile]:
        """List all profiles with pagination."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count total number of profiles."""
        pass
    
    @abstractmethod
    async def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all profiles."""
        pass
    
    @abstractmethod
    async def export_profiles(self, profile_ids: List[UUID], format_type: str = "json") -> bytes:
        """Export profiles in specified format."""
        pass
    
    @abstractmethod
    async def archive_old_profiles(self, days_old: int) -> int:
        """Archive profiles older than specified days."""
        pass
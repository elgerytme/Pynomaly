"""Statistical Analysis repository interface."""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.statistical_analysis import StatisticalAnalysis, StatisticalAnalysisId, DatasetId, UserId


class StatisticalAnalysisRepository(ABC):
    """Repository interface for statistical analysis persistence."""
    
    @abstractmethod
    async def save(self, analysis: StatisticalAnalysis) -> None:
        """Save a statistical analysis."""
        pass
    
    @abstractmethod
    async def get_by_id(self, analysis_id: StatisticalAnalysisId) -> Optional[StatisticalAnalysis]:
        """Get statistical analysis by ID."""
        pass
    
    @abstractmethod
    async def get_by_dataset_id(self, dataset_id: DatasetId) -> List[StatisticalAnalysis]:
        """Get all analyses for a dataset."""
        pass
    
    @abstractmethod
    async def get_by_user_id(self, user_id: UserId) -> List[StatisticalAnalysis]:
        """Get all analyses by a user."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str) -> List[StatisticalAnalysis]:
        """Get analyses by status."""
        pass
    
    @abstractmethod
    async def delete(self, analysis_id: StatisticalAnalysisId) -> None:
        """Delete a statistical analysis."""
        pass
    
    @abstractmethod
    async def list_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[StatisticalAnalysis]:
        """List all statistical analyses with pagination."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Count total number of analyses."""
        pass
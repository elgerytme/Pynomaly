"""In-memory implementation of Statistical Analysis Repository."""

from typing import Dict, List, Optional
from uuid import UUID
import structlog

from ...domain.entities.statistical_analysis import (
    StatisticalAnalysis, StatisticalAnalysisId, DatasetId, UserId
)
from ...domain.repositories.statistical_analysis_repository import StatisticalAnalysisRepository

logger = structlog.get_logger(__name__)


class InMemoryStatisticalAnalysisRepository(StatisticalAnalysisRepository):
    """In-memory implementation for testing and development."""
    
    def __init__(self):
        self._analyses: Dict[UUID, StatisticalAnalysis] = {}
        logger.info("Initialized in-memory statistical analysis repository")
    
    async def save(self, analysis: StatisticalAnalysis) -> None:
        """Save a statistical analysis."""
        self._analyses[analysis.analysis_id.value] = analysis
        logger.debug(
            "Saved statistical analysis",
            analysis_id=str(analysis.analysis_id.value),
            status=analysis.status
        )
    
    async def get_by_id(self, analysis_id: StatisticalAnalysisId) -> Optional[StatisticalAnalysis]:
        """Get statistical analysis by ID."""
        analysis = self._analyses.get(analysis_id.value)
        if analysis:
            logger.debug(
                "Retrieved statistical analysis",
                analysis_id=str(analysis_id.value)
            )
        return analysis
    
    async def get_by_dataset_id(self, dataset_id: DatasetId) -> List[StatisticalAnalysis]:
        """Get all analyses for a dataset."""
        analyses = [
            analysis for analysis in self._analyses.values()
            if analysis.dataset_id.value == dataset_id.value
        ]
        logger.debug(
            "Retrieved analyses by dataset",
            dataset_id=str(dataset_id.value),
            count=len(analyses)
        )
        return analyses
    
    async def get_by_user_id(self, user_id: UserId) -> List[StatisticalAnalysis]:
        """Get all analyses by a user."""
        analyses = [
            analysis for analysis in self._analyses.values()
            if analysis.user_id.value == user_id.value
        ]
        logger.debug(
            "Retrieved analyses by user",
            user_id=str(user_id.value),
            count=len(analyses)
        )
        return analyses
    
    async def get_by_status(self, status: str) -> List[StatisticalAnalysis]:
        """Get analyses by status."""
        analyses = [
            analysis for analysis in self._analyses.values()
            if analysis.status == status
        ]
        logger.debug(
            "Retrieved analyses by status",
            status=status,
            count=len(analyses)
        )
        return analyses
    
    async def delete(self, analysis_id: StatisticalAnalysisId) -> None:
        """Delete a statistical analysis."""
        if analysis_id.value in self._analyses:
            del self._analyses[analysis_id.value]
            logger.debug(
                "Deleted statistical analysis",
                analysis_id=str(analysis_id.value)
            )
    
    async def list_all(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None
    ) -> List[StatisticalAnalysis]:
        """List all statistical analyses with pagination."""
        analyses = list(self._analyses.values())
        
        # Sort by creation time (most recent first)
        analyses.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        if offset:
            analyses = analyses[offset:]
        if limit:
            analyses = analyses[:limit]
        
        logger.debug(
            "Listed all analyses",
            total_count=len(self._analyses),
            returned_count=len(analyses)
        )
        return analyses
    
    async def count(self) -> int:
        """Count total number of analyses."""
        count = len(self._analyses)
        logger.debug("Counted analyses", count=count)
        return count
    
    # Additional utility methods for testing
    
    def clear(self) -> None:
        """Clear all analyses (for testing)."""
        self._analyses.clear()
        logger.debug("Cleared all statistical analyses")
    
    def get_all_sync(self) -> List[StatisticalAnalysis]:
        """Get all analyses synchronously (for testing)."""
        return list(self._analyses.values())
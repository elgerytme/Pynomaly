"""Execute Statistical Analysis Use Case."""

from typing import Dict, Any, Optional
import pandas as pd
import structlog

from ...domain.entities.statistical_analysis import (
    StatisticalAnalysis, StatisticalAnalysisId, DatasetId, UserId, AnalysisType
)
from ...domain.repositories.statistical_analysis_repository import StatisticalAnalysisRepository

logger = structlog.get_logger(__name__)


class ExecuteStatisticalAnalysisUseCase:
    """Use case for executing statistical analysis."""
    
    def __init__(
        self,
        repository: StatisticalAnalysisRepository,
        statistical_service: Optional[Any] = None
    ):
        self.repository = repository
        self.statistical_service = statistical_service
    
    async def execute(
        self,
        dataset_id: DatasetId,
        user_id: UserId,
        analysis_type: AnalysisType,
        data: pd.DataFrame,
        feature_columns: List[str],
        analysis_params: Optional[Dict[str, Any]] = None
    ) -> StatisticalAnalysis:
        """Execute statistical analysis use case."""
        
        # Create new analysis entity
        analysis = StatisticalAnalysis(
            analysis_id=StatisticalAnalysisId(),
            dataset_id=dataset_id,
            user_id=user_id,
            analysis_type=analysis_type,
            feature_columns=feature_columns,
            analysis_params=analysis_params or {}
        )
        
        logger.info(
            "Executing statistical analysis",
            analysis_id=str(analysis.analysis_id.value),
            dataset_id=str(dataset_id.value),
            analysis_type=analysis_type.name
        )
        
        try:
            # Start the analysis
            analysis.start_analysis()
            await self.repository.save(analysis)
            
            # Use the statistical service if available
            if self.statistical_service:
                completed_analysis = await self.statistical_service.execute_analysis(
                    analysis, data
                )
                await self.repository.save(completed_analysis)
                return completed_analysis
            else:
                # Basic fallback implementation
                analysis.complete_analysis(
                    metrics=None,
                    tests=[],
                    insights=["Analysis completed without detailed statistics"]
                )
                await self.repository.save(analysis)
                return analysis
                
        except Exception as e:
            logger.error(
                "Statistical analysis failed",
                analysis_id=str(analysis.analysis_id.value),
                error=str(e)
            )
            analysis.fail_analysis(f"Analysis execution failed: {str(e)}")
            await self.repository.save(analysis)
            raise
    
    async def get_analysis_by_id(
        self, 
        analysis_id: StatisticalAnalysisId
    ) -> Optional[StatisticalAnalysis]:
        """Get analysis by ID."""
        return await self.repository.get_by_id(analysis_id)
    
    async def get_analyses_by_dataset(
        self, 
        dataset_id: DatasetId
    ) -> List[StatisticalAnalysis]:
        """Get all analyses for a dataset."""
        return await self.repository.get_by_dataset_id(dataset_id)
    
    async def get_analyses_by_user(
        self, 
        user_id: UserId
    ) -> List[StatisticalAnalysis]:
        """Get all analyses by a user."""
        return await self.repository.get_by_user_id(user_id)
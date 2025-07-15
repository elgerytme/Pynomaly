"""Use case for performing correlation analysis."""

from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime

from ..dto.statistical_analysis_dto import (
    CorrelationAnalysisRequestDTO,
    CorrelationAnalysisResponseDTO
)
from ...domain.entities.statistical_analysis import (
    StatisticalAnalysis,
    StatisticalAnalysisId,
    DatasetId,
    UserId,
    AnalysisType
)
from ...domain.repositories.statistical_analysis_repository import StatisticalAnalysisRepository
from ...domain.services.statistical_analysis_service import IStatisticalAnalysisService
from ...domain.value_objects.correlation_matrix import CorrelationMatrix, CorrelationType


class PerformCorrelationAnalysisUseCase:
    """Use case for performing correlation analysis."""
    
    def __init__(
        self,
        statistical_analysis_repository: StatisticalAnalysisRepository,
        statistical_analysis_service: IStatisticalAnalysisService
    ):
        self._repository = statistical_analysis_repository
        self._service = statistical_analysis_service
    
    async def execute(self, request: CorrelationAnalysisRequestDTO) -> CorrelationAnalysisResponseDTO:
        """Execute correlation analysis use case.
        
        Args:
            request: Correlation analysis request parameters
            
        Returns:
            Correlation analysis response with correlation matrix
            
        Raises:
            CorrelationAnalysisError: If analysis fails
        """
        try:
            analysis_id = StatisticalAnalysisId(uuid4())
            dataset_id = DatasetId(request.dataset_id)
            user_id = UserId(request.user_id)
            
            analysis_type = AnalysisType(
                name="correlation",
                description="Correlation analysis between features",
                requires_target=False
            )
            
            analysis = StatisticalAnalysis(
                analysis_id=analysis_id,
                dataset_id=dataset_id,
                user_id=user_id,
                analysis_type=analysis_type,
                feature_columns=request.features or [],
                analysis_params={
                    "correlation_method": request.correlation_method,
                    "significance_level": request.significance_level,
                    "include_p_values": request.include_p_values
                }
            )
            
            analysis.start_analysis()
            await self._repository.save(analysis)
            
            correlation_matrix = await self._compute_correlation_matrix(request)
            
            analysis.complete_analysis(
                metrics=None,
                tests=[],
                insights=self._generate_correlation_insights(correlation_matrix)
            )
            await self._repository.save(analysis)
            
            return CorrelationAnalysisResponseDTO(
                analysis_id=analysis_id.value,
                correlation_matrix=correlation_matrix.correlation_matrix,
                feature_names=correlation_matrix.features,
                p_value_matrix=correlation_matrix.p_value_matrix,
                significant_correlations=self._extract_significant_correlations(correlation_matrix),
                multicollinearity_warnings=self._check_multicollinearity(correlation_matrix)
            )
            
        except Exception as e:
            if 'analysis' in locals():
                analysis.fail_analysis(str(e))
                await self._repository.save(analysis)
            
            raise
    
    async def _compute_correlation_matrix(self, request: CorrelationAnalysisRequestDTO) -> CorrelationMatrix:
        """Compute correlation matrix for the dataset."""
        # Mock implementation - in real scenario, this would load data and compute correlations
        features = request.features or ["feature1", "feature2", "feature3"]
        
        # Mock correlation matrix
        correlation_data = [
            [1.0, 0.65, -0.23],
            [0.65, 1.0, 0.12], 
            [-0.23, 0.12, 1.0]
        ]
        
        # Mock p-values if requested
        p_values = None
        if request.include_p_values:
            p_values = [
                [0.0, 0.001, 0.15],
                [0.001, 0.0, 0.45],
                [0.15, 0.45, 0.0]
            ]
        
        correlation_type = CorrelationType.PEARSON
        if request.correlation_method == "spearman":
            correlation_type = CorrelationType.SPEARMAN
        elif request.correlation_method == "kendall":
            correlation_type = CorrelationType.KENDALL
        
        return CorrelationMatrix(
            features=features,
            correlation_matrix=correlation_data,
            correlation_type=correlation_type,
            p_value_matrix=p_values,
            significance_level=request.significance_level
        )
    
    def _extract_significant_correlations(self, correlation_matrix: CorrelationMatrix) -> List[Dict[str, Any]]:
        """Extract statistically significant correlations."""
        significant_pairs = []
        
        for i, feature1 in enumerate(correlation_matrix.features):
            for j, feature2 in enumerate(correlation_matrix.features):
                if i < j:  # Only upper triangle
                    correlation = correlation_matrix.correlation_matrix[i][j]
                    p_value = None
                    if correlation_matrix.p_value_matrix:
                        p_value = correlation_matrix.p_value_matrix[i][j]
                    
                    if p_value and p_value < correlation_matrix.significance_level:
                        significant_pairs.append({
                            "feature1": feature1,
                            "feature2": feature2,
                            "correlation": correlation,
                            "p_value": p_value,
                            "strength": correlation_matrix.get_correlation_strength(feature1, feature2).value
                        })
        
        return significant_pairs
    
    def _check_multicollinearity(self, correlation_matrix: CorrelationMatrix) -> List[str]:
        """Check for multicollinearity issues."""
        warnings = []
        
        high_correlations = correlation_matrix.get_highly_correlated_pairs(threshold=0.8)
        
        if high_correlations:
            warnings.append(f"High correlations detected: {len(high_correlations)} pairs above 0.8")
            
            for feature1, feature2, corr in high_correlations[:3]:  # Top 3
                warnings.append(f"Strong correlation: {feature1} - {feature2} (r={corr:.3f})")
        
        return warnings
    
    def _generate_correlation_insights(self, correlation_matrix: CorrelationMatrix) -> List[str]:
        """Generate insights from correlation analysis."""
        insights = []
        
        summary = correlation_matrix.get_correlation_summary()
        
        insights.append(f"Analyzed {summary['n_features']} features with {summary['total_pairs']} correlation pairs")
        insights.append(f"Average absolute correlation: {summary['statistics']['mean_correlation']:.3f}")
        
        # Check for strong correlations
        strong_count = summary['strength_distribution'].get('strong', 0) + \
                     summary['strength_distribution'].get('very_strong', 0)
        
        if strong_count > 0:
            insights.append(f"Found {strong_count} strong correlation pairs")
        
        # Check multicollinearity
        multicollinearity_info = correlation_matrix.detect_multicollinearity()
        if multicollinearity_info["has_multicollinearity"]:
            insights.append("Potential multicollinearity detected - consider feature selection")
        
        return insights
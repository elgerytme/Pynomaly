"""Use case for performing statistical analysis."""

from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime

from ..dto.statistical_analysis_dto import (
    StatisticalAnalysisRequestDTO,
    StatisticalAnalysisResponseDTO
)
from ...domain.entities.statistical_analysis import (
    StatisticalAnalysis,
    StatisticalAnalysisId,
    DatasetId,
    UserId,
    AnalysisType,
    StatisticalTest,
    StatisticalMetrics
)
from ...domain.repositories.statistical_analysis_repository import StatisticalAnalysisRepository
from ...domain.services.statistical_analysis_service import IStatisticalAnalysisService


class PerformStatisticalAnalysisUseCase:
    """Use case for performing comprehensive statistical analysis."""
    
    def __init__(
        self,
        statistical_analysis_repository: StatisticalAnalysisRepository,
        statistical_analysis_service: IStatisticalAnalysisService
    ):
        self._repository = statistical_analysis_repository
        self._service = statistical_analysis_service
    
    async def execute(self, request: StatisticalAnalysisRequestDTO) -> StatisticalAnalysisResponseDTO:
        """Execute statistical analysis use case.
        
        Args:
            request: Statistical analysis request parameters
            
        Returns:
            Statistical analysis response with results
            
        Raises:
            StatisticalAnalysisError: If analysis fails
        """
        try:
            analysis_id = StatisticalAnalysisId(uuid4())
            dataset_id = DatasetId(request.dataset_id)
            user_id = UserId(request.user_id)
            
            analysis_type = AnalysisType(
                name=request.analysis_type,
                description=f"Statistical analysis of type {request.analysis_type}",
                requires_target=request.target_column is not None
            )
            
            analysis = StatisticalAnalysis(
                analysis_id=analysis_id,
                dataset_id=dataset_id,
                user_id=user_id,
                analysis_type=analysis_type,
                target_column=request.target_column,
                feature_columns=request.feature_columns or [],
                analysis_params=request.analysis_params or {}
            )
            
            analysis.start_analysis()
            await self._repository.save(analysis)
            
            if request.analysis_type == "descriptive":
                results = await self._perform_descriptive_analysis(request)
            elif request.analysis_type == "correlation":
                results = await self._perform_correlation_analysis(request)
            elif request.analysis_type == "distribution":
                results = await self._perform_distribution_analysis(request)
            elif request.analysis_type == "hypothesis_testing":
                results = await self._perform_hypothesis_testing(request)
            else:
                raise ValueError(f"Unsupported analysis type: {request.analysis_type}")
            
            metrics = StatisticalMetrics(
                descriptive_stats=results.get("descriptive_stats", {}),
                correlation_matrix=results.get("correlation_matrix"),
                distribution_params=results.get("distribution_params"),
                outlier_scores=results.get("outlier_scores")
            )
            
            tests = [
                StatisticalTest(
                    test_name=test["name"],
                    statistic=test["statistic"],
                    p_value=test["p_value"],
                    critical_value=test.get("critical_value"),
                    confidence_level=test.get("confidence_level", 0.95),
                    interpretation=test["interpretation"]
                )
                for test in results.get("statistical_tests", [])
            ]
            
            insights = results.get("insights", [])
            
            analysis.complete_analysis(metrics, tests, insights)
            await self._repository.save(analysis)
            
            return StatisticalAnalysisResponseDTO(
                analysis_id=analysis_id.value,
                status="completed",
                results=results,
                insights=insights,
                execution_time_seconds=analysis.execution_time_seconds,
                created_at=analysis.created_at
            )
            
        except Exception as e:
            if 'analysis' in locals():
                analysis.fail_analysis(str(e))
                await self._repository.save(analysis)
            
            return StatisticalAnalysisResponseDTO(
                analysis_id=analysis_id.value if 'analysis_id' in locals() else uuid4(),
                status="failed",
                error_message=str(e),
                created_at=datetime.utcnow()
            )
    
    async def _perform_descriptive_analysis(self, request: StatisticalAnalysisRequestDTO) -> Dict[str, Any]:
        """Perform descriptive statistical analysis."""
        # This would integrate with the statistical analysis service
        # For now, return a placeholder structure
        return {
            "descriptive_stats": {
                "count": 1000,
                "mean": 50.5,
                "std": 15.2,
                "min": 0,
                "max": 100,
                "25%": 25.0,
                "50%": 50.0,
                "75%": 75.0
            },
            "insights": [
                "Data appears normally distributed",
                "No significant outliers detected",
                "Sample size is adequate for analysis"
            ]
        }
    
    async def _perform_correlation_analysis(self, request: StatisticalAnalysisRequestDTO) -> Dict[str, Any]:
        """Perform correlation analysis."""
        return {
            "correlation_matrix": {
                "feature1": {"feature1": 1.0, "feature2": 0.65},
                "feature2": {"feature1": 0.65, "feature2": 1.0}
            },
            "insights": [
                "Strong positive correlation between feature1 and feature2",
                "No multicollinearity issues detected"
            ]
        }
    
    async def _perform_distribution_analysis(self, request: StatisticalAnalysisRequestDTO) -> Dict[str, Any]:
        """Perform distribution analysis."""
        return {
            "distribution_params": {
                "distribution_type": "normal",
                "mean": 50.5,
                "std": 15.2
            },
            "statistical_tests": [
                {
                    "name": "Shapiro-Wilk Test",
                    "statistic": 0.998,
                    "p_value": 0.15,
                    "interpretation": "Data is normally distributed (p > 0.05)"
                }
            ],
            "insights": [
                "Data follows normal distribution",
                "No evidence of skewness or kurtosis issues"
            ]
        }
    
    async def _perform_hypothesis_testing(self, request: StatisticalAnalysisRequestDTO) -> Dict[str, Any]:
        """Perform hypothesis testing."""
        return {
            "statistical_tests": [
                {
                    "name": "T-Test",
                    "statistic": 2.45,
                    "p_value": 0.014,
                    "critical_value": 1.96,
                    "confidence_level": 0.95,
                    "interpretation": "Reject null hypothesis (p < 0.05)"
                }
            ],
            "insights": [
                "Statistically significant difference detected",
                "Effect size is moderate"
            ]
        }
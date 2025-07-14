"""Use case for performing distribution analysis."""

from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime

from ..dto.statistical_analysis_dto import (
    DistributionAnalysisRequestDTO,
    DistributionAnalysisResponseDTO
)
from ...domain.entities.statistical_analysis import (
    StatisticalAnalysis,
    StatisticalAnalysisId,
    DatasetId,
    UserId,
    AnalysisType,
    StatisticalTest
)
from ...domain.repositories.statistical_analysis_repository import StatisticalAnalysisRepository
from ...domain.services.statistical_analysis_service import IStatisticalAnalysisService
from ...domain.value_objects.data_distribution import DataDistribution, DistributionType, DistributionTest


class PerformDistributionAnalysisUseCase:
    """Use case for performing distribution analysis."""
    
    def __init__(
        self,
        statistical_analysis_repository: StatisticalAnalysisRepository,
        statistical_analysis_service: IStatisticalAnalysisService
    ):
        self._repository = statistical_analysis_repository
        self._service = statistical_analysis_service
    
    async def execute(self, request: DistributionAnalysisRequestDTO) -> DistributionAnalysisResponseDTO:
        """Execute distribution analysis use case.
        
        Args:
            request: Distribution analysis request parameters
            
        Returns:
            Distribution analysis response with distribution information
            
        Raises:
            DistributionAnalysisError: If analysis fails
        """
        try:
            analysis_id = StatisticalAnalysisId(uuid4())
            dataset_id = DatasetId(request.dataset_id)
            user_id = UserId(request.user_id)
            
            analysis_type = AnalysisType(
                name="distribution",
                description=f"Distribution analysis for feature {request.feature}",
                requires_target=False
            )
            
            analysis = StatisticalAnalysis(
                analysis_id=analysis_id,
                dataset_id=dataset_id,
                user_id=user_id,
                analysis_type=analysis_type,
                feature_columns=[request.feature],
                analysis_params={
                    "distribution_tests": request.distribution_tests or [],
                    "confidence_level": request.confidence_level
                }
            )
            
            analysis.start_analysis()
            await self._repository.save(analysis)
            
            distribution_analysis = await self._analyze_distribution(request)
            
            tests = self._create_statistical_tests(distribution_analysis)
            
            analysis.complete_analysis(
                metrics=None,
                tests=tests,
                insights=distribution_analysis["insights"]
            )
            await self._repository.save(analysis)
            
            return DistributionAnalysisResponseDTO(
                analysis_id=analysis_id.value,
                feature=request.feature,
                distribution_type=distribution_analysis["distribution_type"],
                distribution_parameters=distribution_analysis["parameters"],
                goodness_of_fit_tests=distribution_analysis["fit_tests"],
                normality_test_results=distribution_analysis["normality_tests"],
                outlier_analysis=distribution_analysis["outlier_analysis"]
            )
            
        except Exception as e:
            if 'analysis' in locals():
                analysis.fail_analysis(str(e))
                await self._repository.save(analysis)
            
            raise
    
    async def _analyze_distribution(self, request: DistributionAnalysisRequestDTO) -> Dict[str, Any]:
        """Analyze the distribution of the specified feature."""
        # Mock implementation - in real scenario, this would load data and analyze distribution
        
        # Mock distribution analysis results
        return {
            "distribution_type": "normal",
            "parameters": {
                "mean": 50.5,
                "std": 15.2,
                "variance": 231.04,
                "skewness": -0.05,
                "kurtosis": 2.98
            },
            "fit_tests": [
                {
                    "test_name": "Kolmogorov-Smirnov",
                    "statistic": 0.045,
                    "p_value": 0.234,
                    "critical_value": 0.063,
                    "result": "accept",
                    "interpretation": "Data fits normal distribution (p > 0.05)"
                },
                {
                    "test_name": "Anderson-Darling",
                    "statistic": 0.752,
                    "p_value": 0.051,
                    "critical_value": 0.787,
                    "result": "accept",
                    "interpretation": "Data fits normal distribution (p > 0.05)"
                }
            ],
            "normality_tests": {
                "shapiro_wilk": {
                    "statistic": 0.998,
                    "p_value": 0.156,
                    "result": "normal",
                    "interpretation": "Data is normally distributed"
                },
                "jarque_bera": {
                    "statistic": 2.45,
                    "p_value": 0.294,
                    "result": "normal",
                    "interpretation": "Data is normally distributed"
                },
                "dagostino": {
                    "statistic": 1.87,
                    "p_value": 0.171,
                    "result": "normal",
                    "interpretation": "Data is normally distributed"
                }
            },
            "outlier_analysis": {
                "method": "iqr",
                "outlier_count": 12,
                "outlier_percentage": 1.2,
                "lower_fence": 12.5,
                "upper_fence": 88.5,
                "outlier_indices": [45, 67, 123, 234, 456, 567, 678, 789, 890, 901, 912, 923]
            },
            "insights": [
                "Data follows a normal distribution",
                "Low skewness indicates symmetric distribution",
                "Kurtosis close to 3 indicates normal tail behavior",
                "1.2% outliers detected using IQR method",
                "Distribution parameters are stable"
            ]
        }
    
    def _create_statistical_tests(self, distribution_analysis: Dict[str, Any]) -> List[StatisticalTest]:
        """Create statistical test objects from analysis results."""
        tests = []
        
        # Add goodness-of-fit tests
        for test in distribution_analysis["fit_tests"]:
            tests.append(StatisticalTest(
                test_name=test["test_name"],
                statistic=test["statistic"],
                p_value=test["p_value"],
                critical_value=test.get("critical_value"),
                confidence_level=0.95,
                interpretation=test["interpretation"]
            ))
        
        # Add normality tests
        for test_name, test_result in distribution_analysis["normality_tests"].items():
            tests.append(StatisticalTest(
                test_name=test_name.replace("_", " ").title(),
                statistic=test_result["statistic"],
                p_value=test_result["p_value"],
                confidence_level=0.95,
                interpretation=test_result["interpretation"]
            ))
        
        return tests
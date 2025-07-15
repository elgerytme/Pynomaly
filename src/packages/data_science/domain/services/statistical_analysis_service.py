"""Statistical Analysis domain service interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from ..entities.analysis_job import AnalysisJob
from ..entities.statistical_profile import StatisticalProfile
from ..value_objects.statistical_metrics import StatisticalMetrics
from ..value_objects.correlation_matrix import CorrelationMatrix
from ..value_objects.data_distribution import DataDistribution


class IStatisticalAnalysisService(ABC):
    """Domain service for statistical analysis operations.
    
    This service orchestrates complex statistical analyses that involve
    multiple entities and require domain expertise beyond simple CRUD operations.
    """
    
    @abstractmethod
    async def perform_descriptive_analysis(self, dataset: Any, 
                                         analysis_config: Dict[str, Any]) -> StatisticalProfile:
        """Perform comprehensive descriptive statistical analysis.
        
        Args:
            dataset: The dataset to analyze
            analysis_config: Configuration for the analysis
            
        Returns:
            Statistical profile with descriptive statistics
            
        Raises:
            AnalysisError: If analysis fails
            ValidationError: If dataset or config is invalid
        """
        pass
    
    @abstractmethod
    async def perform_correlation_analysis(self, dataset: Any,
                                         features: Optional[List[str]] = None,
                                         method: str = "pearson") -> CorrelationMatrix:
        """Perform correlation analysis between features.
        
        Args:
            dataset: The dataset to analyze
            features: Specific features to analyze (all if None)
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            Correlation matrix with relationships and significance tests
            
        Raises:
            AnalysisError: If correlation analysis fails
        """
        pass
    
    @abstractmethod
    async def analyze_data_distribution(self, dataset: Any, 
                                      feature: str) -> DataDistribution:
        """Analyze the distribution of a specific feature.
        
        Args:
            dataset: The dataset to analyze
            feature: Feature name to analyze
            
        Returns:
            Data distribution analysis with fit tests and parameters
            
        Raises:
            AnalysisError: If distribution analysis fails
        """
        pass
    
    @abstractmethod
    async def detect_outliers(self, dataset: Any, 
                            features: Optional[List[str]] = None,
                            methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect outliers in the dataset using multiple methods.
        
        Args:
            dataset: The dataset to analyze
            features: Features to analyze for outliers (all if None)
            methods: Outlier detection methods to use
            
        Returns:
            Dictionary with outlier detection results per feature
            
        Raises:
            AnalysisError: If outlier detection fails
        """
        pass
    
    @abstractmethod
    async def perform_hypothesis_test(self, dataset: Any,
                                    test_type: str,
                                    test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical hypothesis testing.
        
        Args:
            dataset: The dataset to test
            test_type: Type of test (t_test, chi_square, anova, etc.)
            test_config: Configuration for the specific test
            
        Returns:
            Test results with statistics, p-values, and interpretation
            
        Raises:
            AnalysisError: If hypothesis test fails
        """
        pass
    
    @abstractmethod
    async def calculate_feature_statistics(self, dataset: Any,
                                         feature: str) -> StatisticalMetrics:
        """Calculate comprehensive statistics for a single feature.
        
        Args:
            dataset: The dataset to analyze
            feature: Feature name to calculate statistics for
            
        Returns:
            Comprehensive statistical metrics
            
        Raises:
            AnalysisError: If statistics calculation fails
        """
        pass
    
    @abstractmethod
    async def compare_distributions(self, dataset1: Any, dataset2: Any,
                                  feature: str) -> Dict[str, Any]:
        """Compare distributions between two datasets for a feature.
        
        Args:
            dataset1: First dataset
            dataset2: Second dataset
            feature: Feature to compare
            
        Returns:
            Comparison results with statistical tests and effect sizes
            
        Raises:
            AnalysisError: If distribution comparison fails
        """
        pass
    
    @abstractmethod
    async def analyze_time_series(self, dataset: Any,
                                timestamp_column: str,
                                value_column: str,
                                analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze time series data for trends, seasonality, and patterns.
        
        Args:
            dataset: The time series dataset
            timestamp_column: Column containing timestamps
            value_column: Column containing values to analyze
            analysis_config: Configuration for time series analysis
            
        Returns:
            Time series analysis results
            
        Raises:
            AnalysisError: If time series analysis fails
        """
        pass
    
    @abstractmethod
    async def validate_analysis_requirements(self, dataset: Any,
                                           analysis_type: str) -> Dict[str, Any]:
        """Validate that dataset meets requirements for analysis type.
        
        Args:
            dataset: The dataset to validate
            analysis_type: Type of analysis to validate for
            
        Returns:
            Validation results with requirements check
            
        Raises:
            ValidationError: If dataset doesn't meet requirements
        """
        pass
    
    @abstractmethod
    async def generate_analysis_report(self, analysis_job: AnalysisJob,
                                     results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report.
        
        Args:
            analysis_job: The analysis job that was executed
            results: Analysis results to include in report
            
        Returns:
            Formatted analysis report with insights and recommendations
            
        Raises:
            ReportGenerationError: If report generation fails
        """
        pass
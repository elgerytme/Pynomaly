"""Domain interfaces for data processing operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd

from data_quality.domain.entities.data_profile import DataProfile, ColumnProfile
from data_quality.domain.entities.data_quality_check import DataQualityCheck, CheckResult
from data_quality.domain.entities.data_quality_rule import DataQualityRule, RuleCondition


@dataclass
class DataProfilingRequest:
    """Request for data profiling operations."""
    data_source: str
    profile_config: Dict[str, Any]
    include_distributions: bool = True
    include_correlations: bool = False
    sample_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DataValidationRequest:
    """Request for data validation operations."""
    data_source: str
    rules: List[DataQualityRule]
    validation_config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StatisticalAnalysisRequest:
    """Request for statistical analysis operations."""
    data_source: str
    analysis_type: str
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class DataProfilingPort(ABC):
    """Port for data profiling operations."""
    
    @abstractmethod
    async def create_data_profile(
        self, 
        request: DataProfilingRequest
    ) -> DataProfile:
        """Create a comprehensive data profile from a data source.
        
        Args:
            request: Data profiling request configuration
            
        Returns:
            Complete data profile with statistics and metadata
        """
        pass
    
    @abstractmethod
    async def create_column_profile(
        self, 
        data_source: str, 
        column_name: str, 
        config: Dict[str, Any]
    ) -> ColumnProfile:
        """Create a detailed profile for a specific column.
        
        Args:
            data_source: Data source identifier
            column_name: Name of the column to profile
            config: Profiling configuration
            
        Returns:
            Detailed column profile
        """
        pass
    
    @abstractmethod
    async def update_profile_incrementally(
        self, 
        profile: DataProfile, 
        new_data_source: str
    ) -> DataProfile:
        """Update an existing profile with new data incrementally.
        
        Args:
            profile: Existing data profile
            new_data_source: New data to incorporate
            
        Returns:
            Updated data profile
        """
        pass
    
    @abstractmethod
    async def compare_profiles(
        self, 
        profile1: DataProfile, 
        profile2: DataProfile
    ) -> Dict[str, Any]:
        """Compare two data profiles and identify differences.
        
        Args:
            profile1: First data profile
            profile2: Second data profile
            
        Returns:
            Profile comparison results
        """
        pass
    
    @abstractmethod
    async def detect_schema_drift(
        self, 
        baseline_profile: DataProfile, 
        current_profile: DataProfile
    ) -> Dict[str, Any]:
        """Detect schema drift between two profiles.
        
        Args:
            baseline_profile: Baseline data profile
            current_profile: Current data profile
            
        Returns:
            Schema drift analysis results
        """
        pass


class DataValidationPort(ABC):
    """Port for data validation operations."""
    
    @abstractmethod
    async def validate_data_quality(
        self, 
        request: DataValidationRequest
    ) -> List[CheckResult]:
        """Validate data quality against a set of rules.
        
        Args:
            request: Data validation request
            
        Returns:
            List of validation results
        """
        pass
    
    @abstractmethod
    async def execute_quality_check(
        self, 
        data_source: str, 
        check: DataQualityCheck
    ) -> CheckResult:
        """Execute a single data quality check.
        
        Args:
            data_source: Data source to validate
            check: Quality check to execute
            
        Returns:
            Quality check result
        """
        pass
    
    @abstractmethod
    async def validate_business_rules(
        self, 
        data_source: str, 
        rules: List[DataQualityRule]
    ) -> List[CheckResult]:
        """Validate data against business rules.
        
        Args:
            data_source: Data source to validate
            rules: List of business rules
            
        Returns:
            List of validation results
        """
        pass
    
    @abstractmethod
    async def check_data_completeness(
        self, 
        data_source: str, 
        required_columns: List[str]
    ) -> Dict[str, Any]:
        """Check data completeness for required columns.
        
        Args:
            data_source: Data source to check
            required_columns: List of required column names
            
        Returns:
            Completeness analysis results
        """
        pass
    
    @abstractmethod
    async def validate_data_types(
        self, 
        data_source: str, 
        expected_schema: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validate data types against expected schema.
        
        Args:
            data_source: Data source to validate
            expected_schema: Expected data types for columns
            
        Returns:
            Data type validation results
        """
        pass


class StatisticalAnalysisPort(ABC):
    """Port for statistical analysis operations."""
    
    @abstractmethod
    async def calculate_descriptive_statistics(
        self, 
        data_source: str, 
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculate descriptive statistics for data.
        
        Args:
            data_source: Data source to analyze
            columns: Optional list of columns to analyze
            
        Returns:
            Descriptive statistics
        """
        pass
    
    @abstractmethod
    async def detect_outliers(
        self, 
        data_source: str, 
        method: str = "iqr",
        threshold: float = 1.5
    ) -> Dict[str, Any]:
        """Detect outliers in the data.
        
        Args:
            data_source: Data source to analyze
            method: Outlier detection method (iqr, zscore, isolation_forest)
            threshold: Threshold for outlier detection
            
        Returns:
            Outlier detection results
        """
        pass
    
    @abstractmethod
    async def calculate_correlations(
        self, 
        data_source: str, 
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """Calculate correlations between columns.
        
        Args:
            data_source: Data source to analyze
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            Correlation analysis results
        """
        pass
    
    @abstractmethod
    async def perform_distribution_analysis(
        self, 
        data_source: str, 
        column: str
    ) -> Dict[str, Any]:
        """Perform distribution analysis for a column.
        
        Args:
            data_source: Data source to analyze
            column: Column to analyze
            
        Returns:
            Distribution analysis results
        """
        pass
    
    @abstractmethod
    async def detect_data_drift(
        self, 
        baseline_data: str, 
        current_data: str, 
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect statistical data drift between datasets.
        
        Args:
            baseline_data: Baseline data source
            current_data: Current data source
            columns: Optional list of columns to compare
            
        Returns:
            Data drift analysis results
        """
        pass


class DataSamplingPort(ABC):
    """Port for data sampling operations."""
    
    @abstractmethod
    async def create_random_sample(
        self, 
        data_source: str, 
        sample_size: int, 
        random_seed: Optional[int] = None
    ) -> str:
        """Create a random sample from the data source.
        
        Args:
            data_source: Source data identifier
            sample_size: Number of rows to sample
            random_seed: Random seed for reproducibility
            
        Returns:
            Sample data identifier
        """
        pass
    
    @abstractmethod
    async def create_stratified_sample(
        self, 
        data_source: str, 
        strata_column: str, 
        sample_size: int
    ) -> str:
        """Create a stratified sample from the data source.
        
        Args:
            data_source: Source data identifier
            strata_column: Column to use for stratification
            sample_size: Number of rows to sample
            
        Returns:
            Sample data identifier
        """
        pass
    
    @abstractmethod
    async def create_time_based_sample(
        self, 
        data_source: str, 
        time_column: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> str:
        """Create a time-based sample from the data source.
        
        Args:
            data_source: Source data identifier
            time_column: Column containing timestamps
            start_time: Start time for sampling
            end_time: End time for sampling
            
        Returns:
            Sample data identifier
        """
        pass


class DataTransformationPort(ABC):
    """Port for data transformation operations."""
    
    @abstractmethod
    async def clean_data(
        self, 
        data_source: str, 
        cleaning_rules: Dict[str, Any]
    ) -> str:
        """Clean data according to specified rules.
        
        Args:
            data_source: Source data identifier
            cleaning_rules: Rules for data cleaning
            
        Returns:
            Cleaned data identifier
        """
        pass
    
    @abstractmethod
    async def normalize_data(
        self, 
        data_source: str, 
        normalization_config: Dict[str, Any]
    ) -> str:
        """Normalize data according to configuration.
        
        Args:
            data_source: Source data identifier
            normalization_config: Normalization configuration
            
        Returns:
            Normalized data identifier
        """
        pass
    
    @abstractmethod
    async def aggregate_data(
        self, 
        data_source: str, 
        aggregation_config: Dict[str, Any]
    ) -> str:
        """Aggregate data according to configuration.
        
        Args:
            data_source: Source data identifier
            aggregation_config: Aggregation configuration
            
        Returns:
            Aggregated data identifier
        """
        pass
    
    @abstractmethod
    async def filter_data(
        self, 
        data_source: str, 
        filter_conditions: List[Dict[str, Any]]
    ) -> str:
        """Filter data according to conditions.
        
        Args:
            data_source: Source data identifier
            filter_conditions: List of filter conditions
            
        Returns:
            Filtered data identifier
        """
        pass
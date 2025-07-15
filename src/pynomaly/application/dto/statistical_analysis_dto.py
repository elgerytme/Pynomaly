"""
Statistical Analysis Data Transfer Objects

DTOs for statistical analysis requests and responses.
"""

from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class AnalysisTypeEnum(str, Enum):
    """Supported analysis types."""
    DESCRIPTIVE = "descriptive"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    HYPOTHESIS_TESTING = "hypothesis_testing"


class CorrelationMethodEnum(str, Enum):
    """Correlation analysis methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class DistributionTypeEnum(str, Enum):
    """Distribution types for fitting."""
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"
    WEIBULL = "weibull"


class HypothesisTestTypeEnum(str, Enum):
    """Hypothesis test types."""
    T_TEST_ONE_SAMPLE = "t_test_one_sample"
    T_TEST_TWO_SAMPLE = "t_test_two_sample"
    T_TEST_PAIRED = "t_test_paired"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    KRUSKAL_WALLIS = "kruskal_wallis"


class OutlierMethodEnum(str, Enum):
    """Outlier detection methods."""
    IQR = "iqr"
    Z_SCORE = "z_score"
    MODIFIED_Z_SCORE = "modified_z_score"
    ISOLATION_FOREST = "isolation_forest"


# Base Request/Response Models

class StatisticalAnalysisRequestDTO(BaseModel):
    """Base statistical analysis request."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    user_id: UUID = Field(..., description="User identifier")
    analysis_type: AnalysisTypeEnum = Field(..., description="Type of analysis to perform")
    feature_columns: Optional[List[str]] = Field(None, description="Features to analyze")
    target_column: Optional[str] = Field(None, description="Target column for supervised analysis")
    analysis_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis parameters")


class StatisticalAnalysisResponseDTO(BaseModel):
    """Base statistical analysis response."""
    analysis_id: UUID = Field(..., description="Analysis identifier")
    status: str = Field(..., description="Analysis status")
    results: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    insights: Optional[List[str]] = Field(None, description="Generated insights")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    execution_time_seconds: Optional[float] = Field(None, description="Execution time in seconds")
    created_at: datetime = Field(..., description="Analysis creation timestamp")


# Descriptive Statistics DTOs

class DescriptiveStatsRequestDTO(BaseModel):
    """Request for descriptive statistics analysis."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_columns: Optional[List[str]] = Field(None, description="Features to analyze (all if not specified)")
    include_percentiles: bool = Field(True, description="Include percentile calculations")
    percentile_values: List[float] = Field(
        default=[5, 10, 25, 50, 75, 90, 95],
        description="Percentile values to calculate"
    )
    detect_outliers: bool = Field(True, description="Perform outlier detection")
    outlier_method: OutlierMethodEnum = Field(
        default=OutlierMethodEnum.IQR,
        description="Outlier detection method"
    )
    missing_value_analysis: bool = Field(True, description="Analyze missing values")
    
    @validator("percentile_values")
    def validate_percentiles(cls, v):
        if any(p < 0 or p > 100 for p in v):
            raise ValueError("Percentile values must be between 0 and 100")
        return sorted(v)


class DescriptiveStatsResponseDTO(BaseModel):
    """Response for descriptive statistics analysis."""
    analysis_id: UUID = Field(..., description="Analysis identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    descriptive_stats: Dict[str, Dict[str, float]] = Field(..., description="Descriptive statistics by feature")
    outlier_analysis: Optional[Dict[str, Any]] = Field(None, description="Outlier detection results")
    missing_value_analysis: Optional[Dict[str, Any]] = Field(None, description="Missing value analysis")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Analysis creation timestamp")


# Correlation Analysis DTOs

class CorrelationAnalysisRequestDTO(BaseModel):
    """Request for correlation analysis."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_columns: List[str] = Field(..., description="Features for correlation analysis")
    method: CorrelationMethodEnum = Field(
        default=CorrelationMethodEnum.PEARSON,
        description="Correlation method"
    )
    min_periods: int = Field(1, description="Minimum number of observations for correlation")
    significance_level: float = Field(0.05, description="Significance level for statistical tests")
    partial_correlation: bool = Field(False, description="Calculate partial correlations")
    clustering_enabled: bool = Field(True, description="Perform correlation clustering")
    
    @validator("feature_columns")
    def validate_feature_columns(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 features required for correlation analysis")
        return v
    
    @validator("significance_level")
    def validate_significance_level(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("Significance level must be between 0 and 1")
        return v


class CorrelationAnalysisResponseDTO(BaseModel):
    """Response for correlation analysis."""
    analysis_id: UUID = Field(..., description="Analysis identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(..., description="Correlation matrix")
    significance_matrix: Optional[Dict[str, Dict[str, float]]] = Field(None, description="P-value matrix")
    partial_correlations: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Partial correlations")
    correlation_clusters: List[List[str]] = Field(default_factory=list, description="Correlation clusters")
    method: CorrelationMethodEnum = Field(..., description="Correlation method used")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Analysis creation timestamp")


# Distribution Analysis DTOs

class DistributionAnalysisRequestDTO(BaseModel):
    """Request for distribution analysis."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_columns: List[str] = Field(..., description="Features for distribution analysis")
    distributions_to_test: List[DistributionTypeEnum] = Field(
        default=[
            DistributionTypeEnum.NORMAL,
            DistributionTypeEnum.EXPONENTIAL,
            DistributionTypeEnum.GAMMA,
            DistributionTypeEnum.LOGNORMAL
        ],
        description="Distribution types to test"
    )
    significance_level: float = Field(0.05, description="Significance level for goodness-of-fit tests")
    estimation_method: str = Field("mle", description="Parameter estimation method")
    goodness_of_fit_tests: List[str] = Field(
        default=["ks", "ad", "cvm"],
        description="Goodness-of-fit tests to perform"
    )
    bootstrap_samples: int = Field(1000, description="Bootstrap samples for confidence intervals")
    
    @validator("significance_level")
    def validate_significance_level(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("Significance level must be between 0 and 1")
        return v
    
    @validator("bootstrap_samples")
    def validate_bootstrap_samples(cls, v):
        if v < 100:
            raise ValueError("Bootstrap samples must be at least 100")
        return v


class DistributionAnalysisResponseDTO(BaseModel):
    """Response for distribution analysis."""
    analysis_id: UUID = Field(..., description="Analysis identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    fitted_distributions: Dict[str, Dict[str, Any]] = Field(..., description="Fitted distribution parameters")
    goodness_of_fit_tests: Dict[str, Dict[str, float]] = Field(..., description="Goodness-of-fit test results")
    best_fit_distribution: Dict[str, str] = Field(..., description="Best fit distribution for each feature")
    parameter_estimates: Dict[str, Dict[str, Any]] = Field(..., description="Parameter estimates with confidence intervals")
    visualization_data: Optional[Dict[str, Any]] = Field(None, description="Data for distribution plots")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Analysis creation timestamp")


# Hypothesis Testing DTOs

class HypothesisTestRequestDTO(BaseModel):
    """Request for hypothesis testing."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_columns: List[str] = Field(..., description="Features for hypothesis testing")
    target_column: Optional[str] = Field(None, description="Target column for group comparisons")
    test_type: HypothesisTestTypeEnum = Field(..., description="Type of hypothesis test")
    alternative_hypothesis: str = Field("two-sided", description="Alternative hypothesis")
    significance_level: float = Field(0.05, description="Significance level")
    effect_size_calculation: bool = Field(True, description="Calculate effect size")
    multiple_comparison_correction: Optional[str] = Field(None, description="Multiple comparison correction method")
    power_analysis: bool = Field(False, description="Perform power analysis")
    
    @validator("significance_level")
    def validate_significance_level(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("Significance level must be between 0 and 1")
        return v
    
    @validator("alternative_hypothesis")
    def validate_alternative_hypothesis(cls, v):
        if v not in ["two-sided", "less", "greater"]:
            raise ValueError("Alternative hypothesis must be 'two-sided', 'less', or 'greater'")
        return v


class HypothesisTestResponseDTO(BaseModel):
    """Response for hypothesis testing."""
    analysis_id: UUID = Field(..., description="Analysis identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    test_statistic: float = Field(..., description="Test statistic value")
    p_value: float = Field(..., description="P-value of the test")
    critical_value: Optional[float] = Field(None, description="Critical value")
    effect_size: Optional[float] = Field(None, description="Effect size measure")
    confidence_interval: Optional[List[float]] = Field(None, description="Confidence interval")
    power: Optional[float] = Field(None, description="Statistical power")
    test_decision: str = Field(..., description="Test decision (reject/fail to reject)")
    test_interpretation: str = Field(..., description="Human-readable interpretation")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    execution_time_seconds: float = Field(..., description="Execution time in seconds")
    created_at: datetime = Field(..., description="Analysis creation timestamp")


# Composite Analysis DTOs

class ComprehensiveAnalysisRequestDTO(BaseModel):
    """Request for comprehensive statistical analysis."""
    dataset_id: UUID = Field(..., description="Dataset identifier")
    feature_columns: Optional[List[str]] = Field(None, description="Features to analyze")
    analysis_types: List[AnalysisTypeEnum] = Field(..., description="Types of analyses to perform")
    analysis_params: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Parameters for each analysis type"
    )
    parallel_execution: bool = Field(True, description="Execute analyses in parallel")


class ComprehensiveAnalysisResponseDTO(BaseModel):
    """Response for comprehensive statistical analysis."""
    analysis_id: UUID = Field(..., description="Analysis identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    analysis_results: Dict[str, Any] = Field(..., description="Results for each analysis type")
    combined_insights: List[str] = Field(default_factory=list, description="Combined insights from all analyses")
    execution_summary: Dict[str, float] = Field(..., description="Execution time summary")
    total_execution_time_seconds: float = Field(..., description="Total execution time in seconds")
    created_at: datetime = Field(..., description="Analysis creation timestamp")


# Analysis Status and Management DTOs

class AnalysisStatusDTO(BaseModel):
    """Analysis status information."""
    analysis_id: UUID = Field(..., description="Analysis identifier")
    status: str = Field(..., description="Current status")
    progress_percentage: Optional[float] = Field(None, description="Progress percentage")
    current_step: Optional[str] = Field(None, description="Current processing step")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated completion time")
    created_at: datetime = Field(..., description="Analysis creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class AnalysisListDTO(BaseModel):
    """List of analyses with metadata."""
    analyses: List[StatisticalAnalysisResponseDTO] = Field(..., description="List of analyses")
    total_count: int = Field(..., description="Total number of analyses")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more results")


# Validation and Error DTOs

class AnalysisValidationErrorDTO(BaseModel):
    """Analysis validation error details."""
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    field_errors: Optional[Dict[str, str]] = Field(None, description="Field-specific errors")
    suggestions: Optional[List[str]] = Field(None, description="Suggestions for fixing the error")


class AnalysisConfigurationDTO(BaseModel):
    """Analysis configuration and capabilities."""
    supported_analysis_types: List[str] = Field(..., description="Supported analysis types")
    supported_correlation_methods: List[str] = Field(..., description="Supported correlation methods")
    supported_distributions: List[str] = Field(..., description="Supported distribution types")
    supported_hypothesis_tests: List[str] = Field(..., description="Supported hypothesis tests")
    default_parameters: Dict[str, Any] = Field(..., description="Default parameters for each analysis type")
    limitations: Dict[str, Any] = Field(..., description="Analysis limitations and constraints")
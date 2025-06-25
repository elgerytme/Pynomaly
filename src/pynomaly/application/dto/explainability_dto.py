"""Data Transfer Objects for explainability operations."""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Union, Literal, Tuple
from pydantic import BaseModel, Field, ConfigDict, validator
from datetime import datetime
from enum import Enum
from uuid import UUID


# ============================================================================
# Enums and Constants
# ============================================================================

class ExplanationMethod(str, Enum):
    """Supported explanation methods."""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION = "permutation"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    GRADCAM = "gradcam"
    ANCHORS = "anchors"
    CAPTUM = "captum"


class ExplanationType(str, Enum):
    """Types of explanations."""
    LOCAL = "local"
    GLOBAL = "global"
    COHORT = "cohort"
    COMPARATIVE = "comparative"


class BiasMetric(str, Enum):
    """Bias detection metrics."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    STATISTICAL_PARITY = "statistical_parity"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"
    CALIBRATION = "calibration"


class TrustMetric(str, Enum):
    """Trust assessment metrics."""
    CONSISTENCY = "consistency"
    STABILITY = "stability"
    FIDELITY = "fidelity"
    ROBUSTNESS = "robustness"
    PLAUSIBILITY = "plausibility"
    UNCERTAINTY = "uncertainty"


class VisualizationType(str, Enum):
    """Visualization types for explanations."""
    FEATURE_IMPORTANCE = "feature_importance"
    WATERFALL = "waterfall"
    FORCE_PLOT = "force_plot"
    DECISION_PLOT = "decision_plot"
    PARTIAL_DEPENDENCE = "partial_dependence"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    BAR_CHART = "bar_chart"
    BOX_PLOT = "box_plot"
    CORRELATION_MATRIX = "correlation_matrix"


# ============================================================================
# Configuration DTOs
# ============================================================================

class ExplanationConfigDTO(BaseModel):
    """Configuration for explanation generation."""
    model_config = ConfigDict(from_attributes=True)
    
    method: ExplanationMethod = Field(..., description="Explanation method to use")
    explanation_type: ExplanationType = Field(..., description="Type of explanation")
    max_features: int = Field(default=10, ge=1, le=100, description="Maximum features to include")
    background_samples: int = Field(default=100, ge=10, le=5000, description="Background samples for explanation")
    n_permutations: int = Field(default=100, ge=10, le=1000, description="Number of permutations for permutation-based methods")
    feature_names: Optional[List[str]] = Field(default=None, description="Feature names to use")
    categorical_features: Optional[List[str]] = Field(default=None, description="Categorical feature names")
    random_state: Optional[int] = Field(default=42, description="Random state for reproducibility")
    
    # SHAP-specific parameters
    shap_explainer_type: Optional[str] = Field(default="auto", description="SHAP explainer type")
    shap_check_additivity: bool = Field(default=False, description="Check SHAP additivity")
    
    # LIME-specific parameters
    lime_mode: Optional[str] = Field(default="tabular", description="LIME mode (tabular, text, image)")
    lime_kernel_width: Optional[float] = Field(default=None, description="LIME kernel width")
    lime_num_samples: int = Field(default=5000, ge=100, le=10000, description="LIME number of samples")
    
    # Advanced options
    compute_interactions: bool = Field(default=False, description="Compute feature interactions")
    include_confidence_intervals: bool = Field(default=False, description="Include confidence intervals")
    parallel_processing: bool = Field(default=True, description="Use parallel processing")
    cache_explanations: bool = Field(default=True, description="Cache explanation results")


class BiasAnalysisConfigDTO(BaseModel):
    """Configuration for bias analysis."""
    model_config = ConfigDict(from_attributes=True)
    
    protected_attributes: List[str] = Field(..., description="Protected attribute names")
    privileged_groups: Dict[str, List[Any]] = Field(..., description="Privileged group values for each protected attribute")
    metrics: List[BiasMetric] = Field(default_factory=lambda: [BiasMetric.DEMOGRAPHIC_PARITY], description="Bias metrics to compute")
    threshold: float = Field(default=0.5, ge=0, le=1, description="Decision threshold")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for bias tests")
    bootstrap_samples: int = Field(default=1000, ge=100, le=10000, description="Bootstrap samples for confidence intervals")
    min_group_size: int = Field(default=30, ge=10, le=1000, description="Minimum group size for analysis")


class TrustAssessmentConfigDTO(BaseModel):
    """Configuration for trust assessment."""
    model_config = ConfigDict(from_attributes=True)
    
    metrics: List[TrustMetric] = Field(default_factory=lambda: [TrustMetric.CONSISTENCY, TrustMetric.STABILITY], description="Trust metrics to compute")
    perturbation_ratio: float = Field(default=0.1, ge=0.01, le=0.5, description="Perturbation ratio for robustness testing")
    n_perturbations: int = Field(default=100, ge=10, le=1000, description="Number of perturbations")
    stability_samples: int = Field(default=50, ge=10, le=500, description="Samples for stability assessment")
    fidelity_samples: int = Field(default=100, ge=50, le=1000, description="Samples for fidelity assessment")
    uncertainty_method: str = Field(default="monte_carlo", description="Uncertainty quantification method")
    mc_dropout_samples: int = Field(default=100, ge=10, le=500, description="Monte Carlo dropout samples")


# ============================================================================
# Core Explanation DTOs
# ============================================================================

class FeatureContributionDTO(BaseModel):
    """DTO for feature contribution information."""
    model_config = ConfigDict(from_attributes=True)
    
    feature_name: str = Field(..., description="Name of the feature")
    value: float = Field(..., description="Feature value")
    contribution: float = Field(..., description="Contribution to prediction")
    importance: float = Field(..., description="Absolute importance score")
    rank: int = Field(..., description="Importance rank")
    description: Optional[str] = Field(default=None, description="Description of contribution")


class LocalExplanationDTO(BaseModel):
    """DTO for local explanation."""
    model_config = ConfigDict(from_attributes=True)
    
    instance_id: str = Field(..., description="Instance identifier")
    anomaly_score: float = Field(..., description="Anomaly score")
    prediction: str = Field(..., description="Prediction label")
    confidence: float = Field(..., description="Prediction confidence")
    feature_contributions: List[FeatureContributionDTO] = Field(..., description="Feature contributions")
    explanation_method: str = Field(..., description="Explanation method used")
    model_name: str = Field(..., description="Model name")
    timestamp: str = Field(..., description="Explanation timestamp")


class GlobalExplanationDTO(BaseModel):
    """DTO for global explanation."""
    model_config = ConfigDict(from_attributes=True)
    
    model_name: str = Field(..., description="Model name")
    feature_importances: Dict[str, float] = Field(..., description="Feature importance scores")
    top_features: List[str] = Field(..., description="Top important features")
    explanation_method: str = Field(..., description="Explanation method used")
    model_performance: Dict[str, float] = Field(..., description="Model performance metrics")
    timestamp: str = Field(..., description="Explanation timestamp")
    summary: str = Field(..., description="Explanation summary")


class CohortExplanationDTO(BaseModel):
    """DTO for cohort explanation."""
    model_config = ConfigDict(from_attributes=True)
    
    cohort_id: str = Field(..., description="Cohort identifier")
    cohort_description: str = Field(..., description="Cohort description")
    instance_count: int = Field(..., description="Number of instances in cohort")
    common_features: List[FeatureContributionDTO] = Field(..., description="Common feature contributions")
    explanation_method: str = Field(..., description="Explanation method used")
    model_name: str = Field(..., description="Model name")
    timestamp: str = Field(..., description="Explanation timestamp")


class ExplanationRequestDTO(BaseModel):
    """DTO for explanation request."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier")
    instance_data: Optional[Dict[str, Any]] = Field(default=None, description="Instance data for explanation")
    instance_indices: Optional[List[int]] = Field(default=None, description="Instance indices to explain")
    explanation_method: str = Field(default="shap", description="Explanation method")
    max_features: int = Field(default=10, ge=1, le=50, description="Maximum features to include")
    background_samples: int = Field(default=100, ge=10, le=1000, description="Background samples for explanation")
    include_cohort_analysis: bool = Field(default=False, description="Include cohort analysis")
    compare_methods: bool = Field(default=False, description="Compare multiple methods")


class MethodComparisonDTO(BaseModel):
    """DTO for method comparison results."""
    model_config = ConfigDict(from_attributes=True)
    
    method_name: str = Field(..., description="Method name")
    success: bool = Field(..., description="Whether explanation succeeded")
    explanation: Optional[Union[LocalExplanationDTO, GlobalExplanationDTO, CohortExplanationDTO]] = Field(
        default=None, description="Generated explanation"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")


class FeatureStatisticsDTO(BaseModel):
    """DTO for feature statistics."""
    model_config = ConfigDict(from_attributes=True)
    
    feature_name: str = Field(..., description="Feature name")
    mean_contribution: float = Field(..., description="Mean contribution")
    std_contribution: float = Field(..., description="Standard deviation of contribution")
    mean_importance: float = Field(..., description="Mean importance")
    std_importance: float = Field(..., description="Standard deviation of importance")
    mean_value: float = Field(..., description="Mean feature value")
    std_value: float = Field(..., description="Standard deviation of feature value")
    count: int = Field(..., description="Number of explanations")


class ExplanationResponseDTO(BaseModel):
    """DTO for explanation response."""
    model_config = ConfigDict(from_attributes=True)
    
    success: bool = Field(..., description="Whether explanation succeeded")
    explanations: Optional[Dict[str, Any]] = Field(default=None, description="Generated explanations")
    feature_rankings: Optional[List[tuple]] = Field(default=None, description="Feature importance rankings")
    cohort_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Cohort analysis results")
    method_comparison: Optional[Dict[str, MethodComparisonDTO]] = Field(default=None, description="Method comparison results")
    feature_statistics: Optional[Dict[str, FeatureStatisticsDTO]] = Field(default=None, description="Feature statistics")
    available_methods: Optional[List[str]] = Field(default=None, description="Available explanation methods")
    message: str = Field(..., description="Response message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Total execution time")


class ExplainInstanceRequestDTO(BaseModel):
    """DTO for single instance explanation request."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    instance_data: Dict[str, Any] = Field(..., description="Instance data to explain")
    explanation_method: str = Field(default="shap", description="Explanation method")
    max_features: int = Field(default=10, ge=1, le=50, description="Maximum features to show")


class ExplainModelRequestDTO(BaseModel):
    """DTO for global model explanation request."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    dataset_id: str = Field(..., description="Dataset identifier for background data")
    explanation_method: str = Field(default="shap", description="Explanation method")
    max_features: int = Field(default=10, ge=1, le=50, description="Maximum features to show")
    background_samples: int = Field(default=100, ge=10, le=1000, description="Background samples")


class ExplainCohortRequestDTO(BaseModel):
    """DTO for cohort explanation request."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    instance_indices: List[int] = Field(..., description="Indices of instances in cohort")
    explanation_method: str = Field(default="shap", description="Explanation method")
    max_features: int = Field(default=10, ge=1, le=50, description="Maximum features to show")
    cohort_name: Optional[str] = Field(default=None, description="Optional cohort name")


class CompareMethodsRequestDTO(BaseModel):
    """DTO for comparing explanation methods."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    instance_data: Optional[Dict[str, Any]] = Field(default=None, description="Instance data")
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier")
    instance_index: Optional[int] = Field(default=None, description="Instance index if using dataset")
    methods: List[str] = Field(..., description="Methods to compare")
    max_features: int = Field(default=10, ge=1, le=50, description="Maximum features to show")


class FeatureRankingDTO(BaseModel):
    """DTO for feature ranking information."""
    model_config = ConfigDict(from_attributes=True)
    
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., description="Average importance score")
    rank: int = Field(..., description="Feature rank")
    frequency: int = Field(..., description="Frequency in top features")
    variance: float = Field(..., description="Importance variance across explanations")


class ExplanationSummaryDTO(BaseModel):
    """DTO for explanation summary."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: str = Field(..., description="Detector identifier")
    dataset_id: Optional[str] = Field(default=None, description="Dataset identifier")
    total_explanations: int = Field(..., description="Total explanations generated")
    methods_used: List[str] = Field(..., description="Methods used")
    top_features: List[FeatureRankingDTO] = Field(..., description="Top feature rankings")
    execution_summary: Dict[str, float] = Field(..., description="Execution time summary")
    created_at: datetime = Field(default_factory=datetime.now, description="Summary creation time")
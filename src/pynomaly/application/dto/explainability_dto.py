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
    confidence_interval: Optional[Tuple[float, float]] = Field(default=None, description="Confidence interval for contribution")
    p_value: Optional[float] = Field(default=None, description="Statistical significance p-value")
    normalized_contribution: Optional[float] = Field(default=None, description="Normalized contribution (0-1)")


class FeatureInteractionDTO(BaseModel):
    """DTO for feature interaction information."""
    model_config = ConfigDict(from_attributes=True)
    
    feature_1: str = Field(..., description="First feature name")
    feature_2: str = Field(..., description="Second feature name")
    interaction_value: float = Field(..., description="Interaction contribution")
    interaction_strength: float = Field(..., description="Strength of interaction")
    statistical_significance: Optional[float] = Field(default=None, description="Statistical significance")


# ============================================================================
# Bias Analysis DTOs
# ============================================================================

class BiasMetricResultDTO(BaseModel):
    """DTO for individual bias metric result."""
    model_config = ConfigDict(from_attributes=True)
    
    metric_name: BiasMetric = Field(..., description="Bias metric name")
    overall_score: float = Field(..., description="Overall bias score")
    group_scores: Dict[str, float] = Field(..., description="Scores by protected group")
    disparity_ratio: float = Field(..., description="Ratio of max to min group performance")
    statistical_significance: Optional[float] = Field(default=None, description="Statistical significance p-value")
    confidence_interval: Optional[Tuple[float, float]] = Field(default=None, description="Confidence interval for overall score")
    interpretation: str = Field(..., description="Human-readable interpretation")
    is_biased: bool = Field(..., description="Whether bias is detected based on threshold")
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="Bias severity level")


class GroupComparisonDTO(BaseModel):
    """DTO for comparing groups in bias analysis."""
    model_config = ConfigDict(from_attributes=True)
    
    group_name: str = Field(..., description="Group identifier")
    group_size: int = Field(..., description="Number of samples in group")
    mean_prediction: float = Field(..., description="Mean prediction for group")
    std_prediction: float = Field(..., description="Standard deviation of predictions")
    positive_rate: float = Field(..., description="Positive prediction rate")
    false_positive_rate: Optional[float] = Field(default=None, description="False positive rate")
    false_negative_rate: Optional[float] = Field(default=None, description="False negative rate")
    accuracy: Optional[float] = Field(default=None, description="Group-specific accuracy")


class BiasAnalysisResultDTO(BaseModel):
    """DTO for comprehensive bias analysis results."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: UUID = Field(..., description="Detector identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    analysis_id: UUID = Field(..., description="Analysis identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    protected_attributes: List[str] = Field(..., description="Protected attributes analyzed")
    total_samples: int = Field(..., description="Total number of samples")
    group_comparisons: Dict[str, List[GroupComparisonDTO]] = Field(..., description="Group comparison results by attribute")
    bias_metrics: List[BiasMetricResultDTO] = Field(..., description="Bias metric results")
    
    overall_bias_score: float = Field(..., description="Overall bias score (0-1, lower is better)")
    is_fair: bool = Field(..., description="Whether model is considered fair")
    fairness_threshold: float = Field(..., description="Threshold used for fairness determination")
    
    recommendations: List[str] = Field(default_factory=list, description="Bias mitigation recommendations")
    mitigation_strategies: Dict[str, Any] = Field(default_factory=dict, description="Suggested mitigation strategies")
    
    execution_time: float = Field(..., description="Analysis execution time in seconds")
    configuration: BiasAnalysisConfigDTO = Field(..., description="Analysis configuration used")


# ============================================================================
# Trust Assessment DTOs
# ============================================================================

class TrustMetricResultDTO(BaseModel):
    """DTO for individual trust metric result."""
    model_config = ConfigDict(from_attributes=True)
    
    metric_name: TrustMetric = Field(..., description="Trust metric name")
    score: float = Field(ge=0, le=1, description="Trust metric score (0-1, higher is better)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed metric information")
    interpretation: str = Field(..., description="Human-readable interpretation")
    confidence_level: Literal["low", "medium", "high"] = Field(..., description="Confidence in metric assessment")
    
    # Metric-specific fields
    consistency_variance: Optional[float] = Field(default=None, description="Variance in explanations for consistency")
    stability_correlation: Optional[float] = Field(default=None, description="Correlation between perturbed explanations")
    fidelity_accuracy: Optional[float] = Field(default=None, description="Accuracy of explanation approximation")
    robustness_degradation: Optional[float] = Field(default=None, description="Performance degradation under perturbations")


class UncertaintyQuantificationDTO(BaseModel):
    """DTO for uncertainty quantification results."""
    model_config = ConfigDict(from_attributes=True)
    
    epistemic_uncertainty: float = Field(..., description="Model uncertainty (epistemic)")
    aleatoric_uncertainty: float = Field(..., description="Data uncertainty (aleatoric)")
    total_uncertainty: float = Field(..., description="Total uncertainty")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in prediction")
    prediction_interval: Tuple[float, float] = Field(..., description="Prediction interval")
    entropy: float = Field(..., description="Prediction entropy")
    
    # Method-specific uncertainty measures
    mc_dropout_variance: Optional[float] = Field(default=None, description="Monte Carlo dropout variance")
    ensemble_variance: Optional[float] = Field(default=None, description="Ensemble prediction variance")
    bayesian_uncertainty: Optional[float] = Field(default=None, description="Bayesian uncertainty estimate")


class TrustAssessmentResultDTO(BaseModel):
    """DTO for comprehensive trust assessment results."""
    model_config = ConfigDict(from_attributes=True)
    
    detector_id: UUID = Field(..., description="Detector identifier")
    dataset_id: Optional[UUID] = Field(default=None, description="Dataset identifier")
    assessment_id: UUID = Field(..., description="Assessment identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Assessment timestamp")
    
    trust_metrics: List[TrustMetricResultDTO] = Field(..., description="Individual trust metric results")
    uncertainty_quantification: Optional[UncertaintyQuantificationDTO] = Field(default=None, description="Uncertainty analysis")
    
    overall_trust_score: float = Field(ge=0, le=1, description="Overall trust score (0-1, higher is better)")
    trust_level: Literal["very_low", "low", "medium", "high", "very_high"] = Field(..., description="Categorical trust level")
    
    key_strengths: List[str] = Field(default_factory=list, description="Key trust strengths identified")
    key_concerns: List[str] = Field(default_factory=list, description="Key trust concerns identified")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Suggestions for improving trust")
    
    execution_time: float = Field(..., description="Assessment execution time in seconds")
    configuration: TrustAssessmentConfigDTO = Field(..., description="Assessment configuration used")


# ============================================================================
# Enhanced Explanation DTOs
# ============================================================================

class LocalExplanationDTO(BaseModel):
    """DTO for local explanation."""
    model_config = ConfigDict(from_attributes=True)
    
    instance_id: str = Field(..., description="Instance identifier")
    anomaly_score: float = Field(..., description="Anomaly score")
    prediction: str = Field(..., description="Prediction label")
    confidence: float = Field(..., description="Prediction confidence")
    feature_contributions: List[FeatureContributionDTO] = Field(..., description="Feature contributions")
    feature_interactions: Optional[List[FeatureInteractionDTO]] = Field(default=None, description="Feature interactions")
    explanation_method: ExplanationMethod = Field(..., description="Explanation method used")
    model_name: str = Field(..., description="Model name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Explanation timestamp")
    
    # Enhanced fields
    baseline_score: Optional[float] = Field(default=None, description="Baseline prediction score")
    counterfactual_examples: Optional[List[Dict[str, Any]]] = Field(default=None, description="Counterfactual examples")
    similar_instances: Optional[List[str]] = Field(default=None, description="Similar instance identifiers")
    explanation_quality_score: Optional[float] = Field(default=None, description="Quality of explanation (0-1)")
    uncertainty: Optional[UncertaintyQuantificationDTO] = Field(default=None, description="Uncertainty quantification")
    
    # Metadata
    computation_time: float = Field(..., description="Time to compute explanation (seconds)")
    explanation_config: Optional[ExplanationConfigDTO] = Field(default=None, description="Configuration used")


class GlobalExplanationDTO(BaseModel):
    """DTO for global explanation."""
    model_config = ConfigDict(from_attributes=True)
    
    model_name: str = Field(..., description="Model name")
    feature_importances: Dict[str, float] = Field(..., description="Feature importance scores")
    top_features: List[str] = Field(..., description="Top important features")
    explanation_method: ExplanationMethod = Field(..., description="Explanation method used")
    model_performance: Dict[str, float] = Field(..., description="Model performance metrics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Explanation timestamp")
    summary: str = Field(..., description="Explanation summary")
    
    # Enhanced fields
    feature_statistics: Optional[Dict[str, Any]] = Field(default=None, description="Feature statistics across dataset")
    feature_interactions_global: Optional[List[FeatureInteractionDTO]] = Field(default=None, description="Global feature interactions")
    partial_dependence_plots: Optional[Dict[str, Any]] = Field(default=None, description="Partial dependence plot data")
    decision_boundary_data: Optional[Dict[str, Any]] = Field(default=None, description="Decision boundary visualization data")
    
    # Model insights
    model_complexity_score: Optional[float] = Field(default=None, description="Model complexity score")
    interpretability_score: Optional[float] = Field(default=None, description="Overall interpretability score")
    fairness_assessment: Optional[BiasAnalysisResultDTO] = Field(default=None, description="Fairness assessment results")
    
    # Metadata
    samples_analyzed: int = Field(..., description="Number of samples used for global explanation")
    computation_time: float = Field(..., description="Time to compute explanation (seconds)")
    explanation_config: Optional[ExplanationConfigDTO] = Field(default=None, description="Configuration used")


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
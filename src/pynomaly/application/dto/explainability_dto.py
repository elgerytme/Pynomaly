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


# ============================================================================
# Visualization DTOs
# ============================================================================

class VisualizationDataDTO(BaseModel):
    """DTO for visualization data."""
    model_config = ConfigDict(from_attributes=True)
    
    visualization_type: VisualizationType = Field(..., description="Type of visualization")
    title: str = Field(..., description="Visualization title")
    data: Dict[str, Any] = Field(..., description="Visualization data")
    layout: Dict[str, Any] = Field(default_factory=dict, description="Layout configuration")
    style: Dict[str, Any] = Field(default_factory=dict, description="Style configuration")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    dimensions: Optional[Tuple[int, int]] = Field(default=None, description="Visualization dimensions (width, height)")
    interactive: bool = Field(default=True, description="Whether visualization is interactive")
    export_formats: List[str] = Field(default_factory=lambda: ["png", "svg", "html"], description="Supported export formats")


class ExplanationVisualizationDTO(BaseModel):
    """DTO for explanation visualization package."""
    model_config = ConfigDict(from_attributes=True)
    
    explanation_id: str = Field(..., description="Explanation identifier")
    visualizations: List[VisualizationDataDTO] = Field(..., description="List of visualizations")
    
    # Summary visualizations
    feature_importance_chart: Optional[VisualizationDataDTO] = Field(default=None, description="Feature importance chart")
    waterfall_chart: Optional[VisualizationDataDTO] = Field(default=None, description="Waterfall chart for contributions")
    force_plot: Optional[VisualizationDataDTO] = Field(default=None, description="SHAP force plot")
    decision_plot: Optional[VisualizationDataDTO] = Field(default=None, description="Decision plot")
    
    # Advanced visualizations
    partial_dependence_plots: Optional[List[VisualizationDataDTO]] = Field(default=None, description="Partial dependence plots")
    interaction_heatmap: Optional[VisualizationDataDTO] = Field(default=None, description="Feature interaction heatmap")
    correlation_matrix: Optional[VisualizationDataDTO] = Field(default=None, description="Feature correlation matrix")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    total_visualizations: int = Field(..., description="Total number of visualizations")


class ReportGenerationConfigDTO(BaseModel):
    """Configuration for report generation."""
    model_config = ConfigDict(from_attributes=True)
    
    report_type: Literal["summary", "detailed", "executive", "technical"] = Field(..., description="Type of report")
    include_visualizations: bool = Field(default=True, description="Include visualizations in report")
    include_raw_data: bool = Field(default=False, description="Include raw data in report")
    include_methodology: bool = Field(default=True, description="Include methodology section")
    include_recommendations: bool = Field(default=True, description="Include recommendations")
    
    # Formatting options
    format: Literal["html", "pdf", "markdown", "json"] = Field(default="html", description="Report format")
    template: Optional[str] = Field(default=None, description="Custom template to use")
    language: str = Field(default="en", description="Report language")
    
    # Content options
    max_features_in_summary: int = Field(default=10, ge=5, le=50, description="Maximum features in summary")
    include_statistical_tests: bool = Field(default=True, description="Include statistical significance tests")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="Confidence level for statistical tests")


class ExplanationReportDTO(BaseModel):
    """DTO for comprehensive explanation report."""
    model_config = ConfigDict(from_attributes=True)
    
    report_id: UUID = Field(..., description="Report identifier")
    title: str = Field(..., description="Report title")
    created_at: datetime = Field(default_factory=datetime.now, description="Report creation timestamp")
    generated_by: str = Field(..., description="User or system that generated the report")
    
    # Report content
    executive_summary: str = Field(..., description="Executive summary")
    methodology: str = Field(..., description="Methodology used")
    key_findings: List[str] = Field(..., description="Key findings")
    recommendations: List[str] = Field(..., description="Recommendations")
    
    # Explanations included
    local_explanations: Optional[List[LocalExplanationDTO]] = Field(default=None, description="Local explanations")
    global_explanation: Optional[GlobalExplanationDTO] = Field(default=None, description="Global explanation")
    bias_analysis: Optional[BiasAnalysisResultDTO] = Field(default=None, description="Bias analysis results")
    trust_assessment: Optional[TrustAssessmentResultDTO] = Field(default=None, description="Trust assessment results")
    
    # Visualizations
    visualizations: Optional[ExplanationVisualizationDTO] = Field(default=None, description="Report visualizations")
    
    # Metadata
    report_config: ReportGenerationConfigDTO = Field(..., description="Report generation configuration")
    file_paths: Optional[Dict[str, str]] = Field(default=None, description="Generated file paths by format")
    size_mb: Optional[float] = Field(default=None, description="Report size in MB")


# ============================================================================
# Audit and Feedback DTOs
# ============================================================================

class ExplanationAuditLogDTO(BaseModel):
    """DTO for explanation audit log entry."""
    model_config = ConfigDict(from_attributes=True)
    
    audit_id: UUID = Field(..., description="Audit entry identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Audit timestamp")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    
    # Action details
    action: Literal["generate", "view", "export", "feedback", "modify", "delete"] = Field(..., description="Action performed")
    explanation_id: str = Field(..., description="Explanation identifier")
    explanation_type: ExplanationType = Field(..., description="Type of explanation")
    
    # Context
    detector_id: UUID = Field(..., description="Detector identifier")
    dataset_id: Optional[UUID] = Field(default=None, description="Dataset identifier")
    instance_id: Optional[str] = Field(default=None, description="Instance identifier")
    
    # Details
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used")
    success: bool = Field(..., description="Whether action was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Execution time in seconds")
    
    # Metadata
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")
    request_id: Optional[str] = Field(default=None, description="Request identifier")


class ExplanationFeedbackDTO(BaseModel):
    """DTO for user feedback on explanations."""
    model_config = ConfigDict(from_attributes=True)
    
    feedback_id: UUID = Field(..., description="Feedback identifier")
    explanation_id: str = Field(..., description="Explanation identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Feedback timestamp")
    
    # Feedback content
    rating: int = Field(..., ge=1, le=5, description="Overall rating (1-5)")
    usefulness: int = Field(..., ge=1, le=5, description="Usefulness rating (1-5)")
    clarity: int = Field(..., ge=1, le=5, description="Clarity rating (1-5)")
    accuracy: int = Field(..., ge=1, le=5, description="Perceived accuracy rating (1-5)")
    completeness: int = Field(..., ge=1, le=5, description="Completeness rating (1-5)")
    
    # Detailed feedback
    comments: Optional[str] = Field(default=None, description="User comments")
    suggestions: Optional[str] = Field(default=None, description="User suggestions for improvement")
    missing_information: Optional[List[str]] = Field(default=None, description="Information user felt was missing")
    
    # Specific feedback
    most_helpful_features: Optional[List[str]] = Field(default=None, description="Most helpful features identified")
    least_helpful_features: Optional[List[str]] = Field(default=None, description="Least helpful features identified")
    trust_level: Optional[int] = Field(default=None, ge=1, le=5, description="Trust level in explanation (1-5)")
    
    # Context
    use_case: Optional[str] = Field(default=None, description="Use case or context")
    expertise_level: Optional[Literal["beginner", "intermediate", "expert"]] = Field(default=None, description="User expertise level")
    
    # Follow-up
    would_recommend: Optional[bool] = Field(default=None, description="Would recommend this explanation approach")
    contact_for_followup: bool = Field(default=False, description="User agrees to be contacted for follow-up")


class FeedbackSummaryDTO(BaseModel):
    """DTO for feedback summary and analytics."""
    model_config = ConfigDict(from_attributes=True)
    
    summary_id: UUID = Field(..., description="Summary identifier")
    period_start: datetime = Field(..., description="Summary period start")
    period_end: datetime = Field(..., description="Summary period end")
    created_at: datetime = Field(default_factory=datetime.now, description="Summary creation timestamp")
    
    # Aggregate statistics
    total_feedback_count: int = Field(..., description="Total feedback entries")
    average_rating: float = Field(..., description="Average overall rating")
    average_usefulness: float = Field(..., description="Average usefulness rating")
    average_clarity: float = Field(..., description="Average clarity rating")
    average_accuracy: float = Field(..., description="Average accuracy rating")
    average_completeness: float = Field(..., description="Average completeness rating")
    
    # Breakdown by explanation type
    feedback_by_type: Dict[str, Dict[str, float]] = Field(..., description="Feedback statistics by explanation type")
    feedback_by_method: Dict[str, Dict[str, float]] = Field(..., description="Feedback statistics by explanation method")
    
    # Common themes
    common_complaints: List[str] = Field(default_factory=list, description="Most common complaints")
    common_suggestions: List[str] = Field(default_factory=list, description="Most common suggestions")
    most_helpful_features: List[str] = Field(default_factory=list, description="Most frequently mentioned helpful features")
    least_helpful_features: List[str] = Field(default_factory=list, description="Most frequently mentioned unhelpful features")
    
    # Trends
    rating_trend: Dict[str, float] = Field(default_factory=dict, description="Rating trends over time")
    improvement_areas: List[str] = Field(default_factory=list, description="Identified areas for improvement")
    
    # Recommendations
    system_recommendations: List[str] = Field(default_factory=list, description="System improvement recommendations")
    priority_actions: List[str] = Field(default_factory=list, description="Priority actions based on feedback")


class CohortExplanationDTO(BaseModel):
    """DTO for cohort explanation."""
    model_config = ConfigDict(from_attributes=True)
    
    cohort_id: str = Field(..., description="Cohort identifier")
    cohort_description: str = Field(..., description="Cohort description")
    instance_count: int = Field(..., description="Number of instances in cohort")
    common_features: List[FeatureContributionDTO] = Field(..., description="Common feature contributions")
    explanation_method: ExplanationMethod = Field(..., description="Explanation method used")
    model_name: str = Field(..., description="Model name")
    timestamp: datetime = Field(default_factory=datetime.now, description="Explanation timestamp")
    
    # Enhanced fields
    cohort_characteristics: Optional[Dict[str, Any]] = Field(default=None, description="Statistical characteristics of cohort")
    similarity_score: Optional[float] = Field(default=None, description="Internal cohort similarity score")
    representative_instances: Optional[List[str]] = Field(default=None, description="Most representative instance IDs")
    outlier_instances: Optional[List[str]] = Field(default=None, description="Outlier instance IDs within cohort")
    
    # Comparison with other cohorts
    comparison_cohorts: Optional[List[str]] = Field(default=None, description="Other cohort IDs for comparison")
    distinguishing_features: Optional[List[FeatureContributionDTO]] = Field(default=None, description="Features that distinguish this cohort")
    
    # Metadata
    computation_time: float = Field(..., description="Time to compute explanation (seconds)")
    explanation_config: Optional[ExplanationConfigDTO] = Field(default=None, description="Configuration used")


# ============================================================================
# Enhanced Request DTOs
# ============================================================================

class ComprehensiveExplanationRequestDTO(BaseModel):
    """DTO for comprehensive explanation request including bias and trust analysis."""
    model_config = ConfigDict(from_attributes=True)
    
    # Core request
    detector_id: UUID = Field(..., description="Detector identifier")
    dataset_id: Optional[UUID] = Field(default=None, description="Dataset identifier")
    instance_data: Optional[Dict[str, Any]] = Field(default=None, description="Instance data for explanation")
    instance_indices: Optional[List[int]] = Field(default=None, description="Instance indices to explain")
    
    # Explanation configuration
    explanation_config: ExplanationConfigDTO = Field(..., description="Explanation configuration")
    
    # Analysis options
    include_bias_analysis: bool = Field(default=False, description="Include bias analysis")
    bias_config: Optional[BiasAnalysisConfigDTO] = Field(default=None, description="Bias analysis configuration")
    
    include_trust_assessment: bool = Field(default=False, description="Include trust assessment")
    trust_config: Optional[TrustAssessmentConfigDTO] = Field(default=None, description="Trust assessment configuration")
    
    # Visualization and reporting
    generate_visualizations: bool = Field(default=True, description="Generate visualizations")
    visualization_types: Optional[List[VisualizationType]] = Field(default=None, description="Specific visualization types to generate")
    
    generate_report: bool = Field(default=False, description="Generate comprehensive report")
    report_config: Optional[ReportGenerationConfigDTO] = Field(default=None, description="Report generation configuration")
    
    # Advanced options
    compare_methods: bool = Field(default=False, description="Compare multiple explanation methods")
    methods_to_compare: Optional[List[ExplanationMethod]] = Field(default=None, description="Specific methods to compare")
    include_cohort_analysis: bool = Field(default=False, description="Include cohort analysis")
    
    # Metadata
    request_id: Optional[UUID] = Field(default=None, description="Request identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    priority: Literal["low", "medium", "high", "urgent"] = Field(default="medium", description="Request priority")


class BatchExplanationRequestDTO(BaseModel):
    """DTO for batch explanation requests."""
    model_config = ConfigDict(from_attributes=True)
    
    batch_id: UUID = Field(..., description="Batch identifier")
    detector_id: UUID = Field(..., description="Detector identifier")
    dataset_id: UUID = Field(..., description="Dataset identifier")
    
    # Batch configuration
    instance_indices: Optional[List[int]] = Field(default=None, description="Specific instances to explain (if None, explain all)")
    batch_size: int = Field(default=100, ge=1, le=10000, description="Processing batch size")
    max_concurrent: int = Field(default=4, ge=1, le=16, description="Maximum concurrent processing")
    
    # Explanation configuration
    explanation_config: ExplanationConfigDTO = Field(..., description="Explanation configuration")
    
    # Processing options
    fail_on_error: bool = Field(default=False, description="Fail entire batch on single error")
    include_progress_callback: bool = Field(default=True, description="Include progress callbacks")
    
    # Output options
    save_intermediate_results: bool = Field(default=True, description="Save intermediate results")
    output_format: Literal["json", "parquet", "csv", "pickle"] = Field(default="json", description="Output format")
    compression: Optional[str] = Field(default="gzip", description="Compression type")
    
    # Metadata
    user_id: Optional[str] = Field(default=None, description="User identifier")
    priority: Literal["low", "medium", "high"] = Field(default="medium", description="Processing priority")
    estimated_duration: Optional[float] = Field(default=None, description="Estimated duration in seconds")


# ============================================================================
# Response DTOs
# ============================================================================

class ComprehensiveExplanationResponseDTO(BaseModel):
    """DTO for comprehensive explanation response."""
    model_config = ConfigDict(from_attributes=True)
    
    request_id: UUID = Field(..., description="Request identifier")
    success: bool = Field(..., description="Whether request was successful")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    # Core explanations
    local_explanations: Optional[List[LocalExplanationDTO]] = Field(default=None, description="Local explanations")
    global_explanation: Optional[GlobalExplanationDTO] = Field(default=None, description="Global explanation")
    cohort_explanations: Optional[List[CohortExplanationDTO]] = Field(default=None, description="Cohort explanations")
    
    # Analysis results
    bias_analysis: Optional[BiasAnalysisResultDTO] = Field(default=None, description="Bias analysis results")
    trust_assessment: Optional[TrustAssessmentResultDTO] = Field(default=None, description="Trust assessment results")
    
    # Method comparison
    method_comparison: Optional[Dict[str, Any]] = Field(default=None, description="Method comparison results")
    
    # Visualizations and reporting
    visualizations: Optional[ExplanationVisualizationDTO] = Field(default=None, description="Generated visualizations")
    report: Optional[ExplanationReportDTO] = Field(default=None, description="Generated report")
    
    # Metadata
    total_execution_time: float = Field(..., description="Total execution time in seconds")
    individual_timings: Dict[str, float] = Field(default_factory=dict, description="Individual component timings")
    memory_usage_mb: Optional[float] = Field(default=None, description="Peak memory usage in MB")
    
    # Status and errors
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: Optional[List[str]] = Field(default=None, description="Error messages")
    partial_success: bool = Field(default=False, description="Whether some parts succeeded despite errors")
    
    # Quality metrics
    explanation_quality_scores: Optional[Dict[str, float]] = Field(default=None, description="Quality scores for explanations")
    confidence_scores: Optional[Dict[str, float]] = Field(default=None, description="Confidence scores for explanations")


class BatchExplanationResponseDTO(BaseModel):
    """DTO for batch explanation response."""
    model_config = ConfigDict(from_attributes=True)
    
    batch_id: UUID = Field(..., description="Batch identifier")
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = Field(..., description="Batch status")
    created_at: datetime = Field(default_factory=datetime.now, description="Batch creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Batch start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Batch completion timestamp")
    
    # Progress information
    total_instances: int = Field(..., description="Total instances to process")
    processed_instances: int = Field(default=0, description="Number of processed instances")
    successful_instances: int = Field(default=0, description="Number of successfully processed instances")
    failed_instances: int = Field(default=0, description="Number of failed instances")
    
    # Results
    results_summary: Optional[Dict[str, Any]] = Field(default=None, description="Summary of results")
    output_paths: Optional[Dict[str, str]] = Field(default=None, description="Output file paths")
    
    # Performance metrics
    average_processing_time: Optional[float] = Field(default=None, description="Average processing time per instance")
    total_processing_time: Optional[float] = Field(default=None, description="Total processing time")
    peak_memory_usage_mb: Optional[float] = Field(default=None, description="Peak memory usage in MB")
    
    # Error handling
    error_summary: Optional[Dict[str, int]] = Field(default=None, description="Summary of errors by type")
    sample_errors: Optional[List[str]] = Field(default=None, description="Sample error messages")
    
    # Metadata
    configuration_used: Optional[Dict[str, Any]] = Field(default=None, description="Configuration used for processing")
    resource_usage: Optional[Dict[str, Any]] = Field(default=None, description="Resource usage information")


# ============================================================================
# Legacy DTOs (maintained for backward compatibility)
# ============================================================================

class ExplanationRequestDTO(BaseModel):
    """DTO for explanation request (legacy - use ComprehensiveExplanationRequestDTO for new implementations)."""
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
    """DTO for explanation response (legacy - use ComprehensiveExplanationResponseDTO for new implementations)."""
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


# ============================================================================
# Utility Functions for DTO Conversion
# ============================================================================

def convert_explanation_config_to_legacy(config: ExplanationConfigDTO) -> Dict[str, Any]:
    """Convert new ExplanationConfigDTO to legacy format."""
    return {
        "explanation_method": config.method.value,
        "max_features": config.max_features,
        "background_samples": config.background_samples,
        "n_permutations": config.n_permutations,
        "feature_names": config.feature_names,
        "random_state": config.random_state,
    }


def create_explanation_config_from_legacy(
    explanation_method: str = "shap",
    max_features: int = 10,
    background_samples: int = 100,
    **kwargs
) -> ExplanationConfigDTO:
    """Create ExplanationConfigDTO from legacy parameters."""
    return ExplanationConfigDTO(
        method=ExplanationMethod(explanation_method),
        explanation_type=ExplanationType.LOCAL,
        max_features=max_features,
        background_samples=background_samples,
        **kwargs
    )


def merge_feature_contributions(
    contributions: List[FeatureContributionDTO],
    method: str = "mean"
) -> List[FeatureContributionDTO]:
    """Merge multiple feature contributions using specified method."""
    if not contributions:
        return []
    
    feature_groups = {}
    for contrib in contributions:
        if contrib.feature_name not in feature_groups:
            feature_groups[contrib.feature_name] = []
        feature_groups[contrib.feature_name].append(contrib)
    
    merged = []
    for feature_name, group in feature_groups.items():
        if method == "mean":
            avg_contribution = sum(c.contribution for c in group) / len(group)
            avg_importance = sum(c.importance for c in group) / len(group)
            avg_value = sum(c.value for c in group) / len(group)
        elif method == "median":
            sorted_contrib = sorted(c.contribution for c in group)
            sorted_importance = sorted(c.importance for c in group)
            sorted_value = sorted(c.value for c in group)
            n = len(group)
            avg_contribution = sorted_contrib[n // 2]
            avg_importance = sorted_importance[n // 2]
            avg_value = sorted_value[n // 2]
        else:
            raise ValueError(f"Unsupported merge method: {method}")
        
        merged_contrib = FeatureContributionDTO(
            feature_name=feature_name,
            value=avg_value,
            contribution=avg_contribution,
            importance=avg_importance,
            rank=0,  # Will be recalculated
            description=f"Merged from {len(group)} contributions using {method}"
        )
        merged.append(merged_contrib)
    
    # Recalculate ranks
    merged.sort(key=lambda x: abs(x.importance), reverse=True)
    for i, contrib in enumerate(merged):
        contrib.rank = i + 1
    
    return merged


def calculate_explanation_quality_score(
    explanation: LocalExplanationDTO,
    trust_assessment: Optional[TrustAssessmentResultDTO] = None,
    bias_analysis: Optional[BiasAnalysisResultDTO] = None
) -> float:
    """Calculate overall quality score for an explanation."""
    base_score = 0.7  # Base score for having an explanation
    
    # Adjust based on confidence
    if explanation.confidence:
        confidence_bonus = (explanation.confidence - 0.5) * 0.2
        base_score += confidence_bonus
    
    # Adjust based on trust assessment
    if trust_assessment:
        trust_bonus = (trust_assessment.overall_trust_score - 0.5) * 0.3
        base_score += trust_bonus
    
    # Adjust based on bias analysis
    if bias_analysis:
        fairness_bonus = (1 - bias_analysis.overall_bias_score) * 0.2
        base_score += fairness_bonus
    
    # Adjust based on feature coverage
    if explanation.feature_contributions:
        feature_coverage = min(len(explanation.feature_contributions) / 10, 1.0)
        base_score += feature_coverage * 0.1
    
    # Ensure score is within bounds
    return max(0.0, min(1.0, base_score))


def create_visualization_config(
    visualization_types: Optional[List[VisualizationType]] = None,
    interactive: bool = True,
    export_formats: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create visualization configuration dictionary."""
    if visualization_types is None:
        visualization_types = [
            VisualizationType.FEATURE_IMPORTANCE,
            VisualizationType.WATERFALL,
            VisualizationType.FORCE_PLOT
        ]
    
    if export_formats is None:
        export_formats = ["png", "svg", "html"]
    
    return {
        "types": [vt.value for vt in visualization_types],
        "interactive": interactive,
        "export_formats": export_formats,
        "style": {
            "theme": "default",
            "color_scheme": "viridis",
            "font_size": 12,
            "figure_size": (10, 6)
        }
    }


def validate_explanation_request(request: ComprehensiveExplanationRequestDTO) -> List[str]:
    """Validate explanation request and return list of validation errors."""
    errors = []
    
    # Check required fields
    if not request.detector_id:
        errors.append("detector_id is required")
    
    # Check that either instance_data or dataset_id is provided
    if not request.instance_data and not request.dataset_id:
        errors.append("Either instance_data or dataset_id must be provided")
    
    # Validate bias analysis configuration
    if request.include_bias_analysis and not request.bias_config:
        errors.append("bias_config is required when include_bias_analysis is True")
    
    if request.bias_config:
        if not request.bias_config.protected_attributes:
            errors.append("protected_attributes must be specified in bias_config")
        if not request.bias_config.privileged_groups:
            errors.append("privileged_groups must be specified in bias_config")
    
    # Validate trust assessment configuration
    if request.include_trust_assessment and not request.trust_config:
        errors.append("trust_config is required when include_trust_assessment is True")
    
    # Validate method comparison
    if request.compare_methods and not request.methods_to_compare:
        errors.append("methods_to_compare must be specified when compare_methods is True")
    
    # Validate report generation
    if request.generate_report and not request.report_config:
        errors.append("report_config is required when generate_report is True")
    
    return errors


# ============================================================================
# Export all DTOs
# ============================================================================

__all__ = [
    # Enums
    "ExplanationMethod",
    "ExplanationType", 
    "BiasMetric",
    "TrustMetric",
    "VisualizationType",
    
    # Configuration DTOs
    "ExplanationConfigDTO",
    "BiasAnalysisConfigDTO",
    "TrustAssessmentConfigDTO",
    
    # Core DTOs
    "FeatureContributionDTO",
    "FeatureInteractionDTO",
    "LocalExplanationDTO",
    "GlobalExplanationDTO",
    "CohortExplanationDTO",
    
    # Bias Analysis DTOs
    "BiasMetricResultDTO",
    "GroupComparisonDTO",
    "BiasAnalysisResultDTO",
    
    # Trust Assessment DTOs
    "TrustMetricResultDTO",
    "UncertaintyQuantificationDTO",
    "TrustAssessmentResultDTO",
    
    # Visualization DTOs
    "VisualizationDataDTO",
    "ExplanationVisualizationDTO",
    "ReportGenerationConfigDTO",
    "ExplanationReportDTO",
    
    # Audit and Feedback DTOs
    "ExplanationAuditLogDTO",
    "ExplanationFeedbackDTO",
    "FeedbackSummaryDTO",
    
    # Request DTOs
    "ComprehensiveExplanationRequestDTO",
    "BatchExplanationRequestDTO",
    
    # Response DTOs
    "ComprehensiveExplanationResponseDTO",
    "BatchExplanationResponseDTO",
    
    # Legacy DTOs (for backward compatibility)
    "ExplanationRequestDTO",
    "MethodComparisonDTO",
    "FeatureStatisticsDTO",
    "ExplanationResponseDTO",
    "ExplainInstanceRequestDTO",
    "ExplainModelRequestDTO",
    "ExplainCohortRequestDTO",
    "CompareMethodsRequestDTO",
    "FeatureRankingDTO",
    "ExplanationSummaryDTO",
    
    # Utility Functions
    "convert_explanation_config_to_legacy",
    "create_explanation_config_from_legacy",
    "merge_feature_contributions",
    "calculate_explanation_quality_score",
    "create_visualization_config",
    "validate_explanation_request",
]
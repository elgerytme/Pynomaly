"""Data Transfer Objects for intelligent algorithm selection."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DatasetCharacteristicsDTO(BaseModel):
    """Dataset characteristics for algorithm selection."""

    model_config = ConfigDict(from_attributes=True)

    # Basic dimensions
    n_samples: int = Field(..., ge=0, description="Number of samples")
    n_features: int = Field(..., ge=0, description="Number of features")
    n_numeric_features: int = Field(..., ge=0, description="Number of numeric features")
    n_categorical_features: int = Field(
        ..., ge=0, description="Number of categorical features"
    )

    # Data quality metrics
    feature_density: float = Field(
        ..., ge=0, le=1, description="Ratio of non-zero values"
    )
    missing_value_ratio: float = Field(
        ..., ge=0, le=1, description="Ratio of missing values"
    )
    outlier_ratio: float = Field(..., ge=0, le=1, description="Ratio of outliers")

    # Statistical properties
    mean_feature_correlation: float = Field(
        ..., ge=0, le=1, description="Mean absolute correlation"
    )
    feature_variance_ratio: float = Field(
        ..., ge=0, description="Variance ratio across features"
    )
    data_dimensionality_ratio: float = Field(
        ..., ge=0, description="Features to samples ratio"
    )

    # Distribution characteristics
    skewness_mean: float = Field(..., description="Mean skewness across features")
    kurtosis_mean: float = Field(..., description="Mean kurtosis across features")
    class_imbalance: float = Field(..., ge=0, le=1, description="Class imbalance ratio")

    # Additional metadata
    estimated_complexity: str | None = Field(
        default=None, description="Estimated complexity level"
    )
    data_type: str | None = Field(default=None, description="Data type category")
    domain: str | None = Field(default=None, description="Application domain")


class AlgorithmPerformanceDTO(BaseModel):
    """Algorithm performance metrics."""

    model_config = ConfigDict(from_attributes=True)

    # Primary metrics
    primary_metric: float = Field(
        ..., ge=0, le=1, description="Primary performance metric"
    )
    secondary_metrics: dict[str, float] = Field(
        default_factory=dict, description="Secondary metrics"
    )

    # Resource usage
    training_time_seconds: float = Field(
        ..., ge=0, description="Training time in seconds"
    )
    memory_usage_mb: float = Field(..., ge=0, description="Memory usage in MB")
    prediction_time_ms: float = Field(
        default=0.0, ge=0, description="Prediction time in ms"
    )

    # Quality indicators
    stability_score: float = Field(
        default=0.0, ge=0, le=1, description="Model stability"
    )
    interpretability_score: float = Field(
        default=0.0, ge=0, le=1, description="Interpretability"
    )
    confidence_score: float = Field(
        default=0.0, ge=0, le=1, description="Prediction confidence"
    )

    # Cross-validation results
    cv_mean: float | None = Field(default=None, description="CV mean score")
    cv_std: float | None = Field(default=None, description="CV standard deviation")
    cv_scores: list[float] | None = Field(
        default=None, description="Individual CV scores"
    )


class OptimizationConstraintsDTO(BaseModel):
    """Constraints for algorithm optimization."""

    model_config = ConfigDict(from_attributes=True)

    # Resource constraints
    max_training_time_seconds: float | None = Field(
        default=None, gt=0, description="Max training time"
    )
    max_memory_mb: float | None = Field(
        default=None, gt=0, description="Max memory usage"
    )
    max_prediction_time_ms: float | None = Field(
        default=None, gt=0, description="Max prediction time"
    )

    # Performance requirements
    min_accuracy: float | None = Field(
        default=None, ge=0, le=1, description="Minimum accuracy"
    )
    min_interpretability: float | None = Field(
        default=None, ge=0, le=1, description="Min interpretability"
    )
    min_stability: float | None = Field(
        default=None, ge=0, le=1, description="Minimum stability"
    )

    # Deployment constraints
    require_online_prediction: bool = Field(
        default=False, description="Requires online prediction"
    )
    require_batch_prediction: bool = Field(
        default=True, description="Requires batch prediction"
    )
    require_interpretability: bool = Field(
        default=False, description="Requires interpretability"
    )

    # Environment constraints
    available_libraries: list[str] | None = Field(
        default=None, description="Available libraries"
    )
    gpu_available: bool = Field(default=False, description="GPU availability")
    distributed_computing: bool = Field(
        default=False, description="Distributed computing support"
    )


class MetaLearningConfigDTO(BaseModel):
    """Configuration for meta-learning."""

    model_config = ConfigDict(from_attributes=True)

    # Meta-learning settings
    enable_transfer_learning: bool = Field(
        default=True, description="Enable transfer learning"
    )
    similarity_threshold: float = Field(
        default=0.7, ge=0, le=1, description="Dataset similarity threshold"
    )
    min_historical_samples: int = Field(
        default=5, ge=1, description="Min historical samples required"
    )

    # Model settings
    meta_model_type: str = Field(default="random_forest", description="Meta-model type")
    cross_validation_folds: int = Field(
        default=5, ge=2, le=10, description="CV folds for evaluation"
    )

    # Learning settings
    learning_rate: float = Field(
        default=0.01, gt=0, le=1, description="Learning rate for updates"
    )
    forgetting_factor: float = Field(
        default=0.95, ge=0, le=1, description="Forgetting factor for old data"
    )

    # Advanced options
    ensemble_meta_models: bool = Field(
        default=False, description="Use ensemble of meta-models"
    )
    adaptive_weighting: bool = Field(
        default=True, description="Adaptive weighting based on recency"
    )


class SelectionRecommendationDTO(BaseModel):
    """Algorithm selection recommendation."""

    model_config = ConfigDict(from_attributes=True)

    # Recommendations
    recommended_algorithms: list[str] = Field(
        ..., description="List of recommended algorithms"
    )
    confidence_scores: dict[str, float] = Field(
        ..., description="Confidence scores for each algorithm"
    )
    reasoning: list[str] = Field(
        default_factory=list, description="Human-readable reasoning"
    )

    # Context
    dataset_characteristics: DatasetCharacteristicsDTO = Field(
        ..., description="Dataset characteristics"
    )
    selection_context: dict[str, Any] = Field(
        default_factory=dict, description="Selection context"
    )

    # Metadata
    timestamp: datetime = Field(..., description="Recommendation timestamp")
    recommendation_id: UUID | None = Field(
        default=None, description="Unique recommendation ID"
    )

    # Performance predictions
    predicted_performances: dict[str, float] | None = Field(
        default=None, description="Predicted performances"
    )
    uncertainty_estimates: dict[str, float] | None = Field(
        default=None, description="Uncertainty estimates"
    )


class AlgorithmBenchmarkDTO(BaseModel):
    """Algorithm benchmark results."""

    model_config = ConfigDict(from_attributes=True)

    # Algorithm identification
    algorithm_name: str = Field(..., description="Algorithm name")
    algorithm_version: str | None = Field(
        default=None, description="Algorithm version"
    )

    # Performance metrics
    mean_score: float = Field(..., ge=0, le=1, description="Mean performance score")
    std_score: float = Field(..., ge=0, description="Standard deviation of scores")
    cv_scores: list[float] = Field(..., description="Cross-validation scores")

    # Resource usage
    training_time_seconds: float = Field(..., ge=0, description="Training time")
    memory_usage_mb: float = Field(..., ge=0, description="Memory usage")
    prediction_time_ms: float = Field(default=0.0, ge=0, description="Prediction time")

    # Configuration
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameters used"
    )
    additional_metrics: dict[str, float] = Field(
        default_factory=dict, description="Additional metrics"
    )

    # Metadata
    benchmark_timestamp: datetime = Field(
        default_factory=datetime.now, description="Benchmark timestamp"
    )
    environment_info: dict[str, Any] | None = Field(
        default=None, description="Environment information"
    )


class SelectionHistoryDTO(BaseModel):
    """Historical algorithm selection entry."""

    model_config = ConfigDict(from_attributes=True)

    # Selection details
    dataset_characteristics: DatasetCharacteristicsDTO = Field(
        ..., description="Dataset characteristics"
    )
    selected_algorithm: str = Field(..., description="Selected algorithm")
    performance: AlgorithmPerformanceDTO = Field(
        ..., description="Achieved performance"
    )

    # Context
    selection_context: dict[str, Any] = Field(
        default_factory=dict, description="Selection context"
    )
    constraints_used: OptimizationConstraintsDTO | None = Field(
        default=None, description="Constraints applied"
    )

    # Metadata
    timestamp: datetime = Field(..., description="Selection timestamp")
    dataset_hash: str = Field(..., description="Dataset hash for identification")
    user_feedback: dict[str, Any] | None = Field(
        default=None, description="User feedback"
    )

    # Outcomes
    was_successful: bool = Field(
        default=True, description="Whether selection was successful"
    )
    lessons_learned: list[str] | None = Field(
        default=None, description="Lessons learned"
    )


class LearningInsightsDTO(BaseModel):
    """Insights from learning history."""

    model_config = ConfigDict(from_attributes=True)

    # Summary statistics
    total_selections: int = Field(..., ge=0, description="Total number of selections")
    unique_algorithms: int | None = Field(
        default=None, description="Number of unique algorithms used"
    )
    unique_datasets: int | None = Field(
        default=None, description="Number of unique datasets"
    )

    # Performance analysis
    algorithm_performance_stats: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Performance statistics by algorithm"
    )
    dataset_type_preferences: dict[str, list[str]] = Field(
        default_factory=dict, description="Algorithm preferences by dataset type"
    )

    # Trends
    performance_trends: dict[str, list[float]] = Field(
        default_factory=dict, description="Performance trends over time"
    )
    feature_importance_insights: dict[str, float] = Field(
        default_factory=dict, description="Feature importance for selection"
    )

    # Model quality
    meta_model_accuracy: float | None = Field(
        default=None, description="Meta-model accuracy"
    )
    recommendation_confidence: float = Field(
        default=0.0, ge=0, le=1, description="Overall confidence"
    )

    # Metadata
    generated_at: datetime = Field(..., description="Insights generation timestamp")
    analysis_period_days: int | None = Field(
        default=None, description="Analysis period in days"
    )


class AlgorithmComparisonDTO(BaseModel):
    """Comparison between algorithms."""

    model_config = ConfigDict(from_attributes=True)

    # Algorithms being compared
    algorithm_a: str = Field(..., description="First algorithm")
    algorithm_b: str = Field(..., description="Second algorithm")

    # Performance comparison
    performance_difference: float = Field(
        ..., description="Performance difference (A - B)"
    )
    statistical_significance: float | None = Field(
        default=None, description="Statistical significance p-value"
    )

    # Resource comparison
    time_difference_seconds: float = Field(
        default=0.0, description="Training time difference"
    )
    memory_difference_mb: float = Field(
        default=0.0, description="Memory usage difference"
    )

    # Qualitative comparison
    interpretability_comparison: str | None = Field(
        default=None, description="Interpretability comparison"
    )
    stability_comparison: str | None = Field(
        default=None, description="Stability comparison"
    )

    # Recommendation
    recommended_choice: str = Field(..., description="Recommended algorithm")
    recommendation_reasoning: list[str] = Field(
        default_factory=list, description="Reasoning for recommendation"
    )


class PerformancePredictionDTO(BaseModel):
    """Performance prediction for algorithm-dataset combination."""

    model_config = ConfigDict(from_attributes=True)

    # Prediction details
    algorithm: str = Field(..., description="Algorithm name")
    predicted_performance: float = Field(
        ..., ge=0, le=1, description="Predicted performance"
    )
    confidence_interval: Tuple[float, float] = Field(
        ..., description="Confidence interval"
    )

    # Prediction quality
    prediction_confidence: float = Field(
        ..., ge=0, le=1, description="Prediction confidence"
    )
    uncertainty_sources: list[str] = Field(
        default_factory=list, description="Sources of uncertainty"
    )

    # Supporting evidence
    similar_datasets_count: int = Field(
        default=0, ge=0, description="Number of similar datasets in history"
    )
    historical_performance_range: Tuple[float, float] | None = Field(
        default=None, description="Historical performance range for similar datasets"
    )

    # Metadata
    prediction_timestamp: datetime = Field(
        default_factory=datetime.now, description="Prediction timestamp"
    )
    model_version: str | None = Field(
        default=None, description="Prediction model version"
    )


class SelectionExplanationDTO(BaseModel):
    """Explanation for algorithm selection decision."""

    model_config = ConfigDict(from_attributes=True)

    # Main explanation
    primary_reason: str = Field(..., description="Primary reason for selection")
    supporting_reasons: list[str] = Field(
        default_factory=list, description="Supporting reasons"
    )

    # Evidence
    evidence_sources: list[str] = Field(
        default_factory=list, description="Sources of evidence"
    )
    confidence_factors: dict[str, float] = Field(
        default_factory=dict, description="Factors affecting confidence"
    )

    # Alternatives
    alternative_algorithms: list[str] = Field(
        default_factory=list, description="Alternative algorithms considered"
    )
    why_not_alternatives: dict[str, str] = Field(
        default_factory=dict, description="Why alternatives weren't chosen"
    )

    # Caveats and limitations
    assumptions: list[str] = Field(default_factory=list, description="Assumptions made")
    limitations: list[str] = Field(
        default_factory=list, description="Known limitations"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Additional recommendations"
    )


class SelectionRequestDTO(BaseModel):
    """Request for algorithm selection."""

    model_config = ConfigDict(from_attributes=True)

    # Dataset information
    dataset_id: UUID | None = Field(default=None, description="Dataset identifier")
    dataset_characteristics: DatasetCharacteristicsDTO | None = Field(
        default=None, description="Pre-computed dataset characteristics"
    )

    # Selection preferences
    optimization_goal: str = Field(
        default="performance", description="Optimization goal"
    )
    constraints: OptimizationConstraintsDTO | None = Field(
        default=None, description="Constraints"
    )

    # Advanced options
    require_explanation: bool = Field(default=True, description="Require explanation")
    include_alternatives: bool = Field(
        default=True, description="Include alternative recommendations"
    )
    use_meta_learning: bool = Field(default=True, description="Use meta-learning")

    # Context
    user_expertise: str | None = Field(
        default=None, description="User expertise level"
    )
    use_case: str | None = Field(default=None, description="Intended use case")
    deployment_environment: str | None = Field(
        default=None, description="Deployment environment"
    )

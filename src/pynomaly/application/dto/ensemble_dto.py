"""Data Transfer Objects for advanced ensemble methods and meta-learning."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EnsembleStrategyDTO(BaseModel):
    """DTO for ensemble strategy configuration."""

    name: str = Field(description="Strategy name")
    description: str = Field(description="Strategy description")
    requires_training: bool = Field(
        default=False, description="Whether strategy requires training"
    )
    supports_weights: bool = Field(
        default=True, description="Whether strategy supports weights"
    )
    complexity: str = Field(default="medium", description="Computational complexity")
    interpretability: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Interpretability score"
    )


class DiversityMetricsDTO(BaseModel):
    """DTO for diversity analysis metrics."""

    disagreement_measure: float = Field(
        description="Average disagreement between classifiers"
    )
    double_fault_measure: float = Field(description="Double fault diversity measure")
    q_statistic: float = Field(description="Q-statistic correlation measure")
    correlation_coefficient: float = Field(description="Correlation coefficient")
    kappa_statistic: float = Field(description="Kappa inter-rater agreement")
    entropy_measure: float = Field(description="Entropy-based diversity measure")
    overall_diversity: float = Field(description="Overall diversity score")


class MetaLearningKnowledgeDTO(BaseModel):
    """DTO for meta-learning knowledge representation."""

    dataset_characteristics: dict[str, Any] = Field(
        description="Dataset characteristics"
    )
    algorithm_performance: dict[str, float] = Field(
        description="Algorithm performance mapping"
    )
    ensemble_composition: list[str] = Field(description="Optimal ensemble composition")
    optimal_weights: dict[str, float] = Field(description="Optimal ensemble weights")
    diversity_requirements: dict[str, float] = Field(
        description="Diversity requirements"
    )
    performance_metrics: dict[str, float] = Field(description="Performance metrics")
    confidence_score: float = Field(description="Confidence in recommendation")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Knowledge creation time"
    )


class EnsembleConfigurationDTO(BaseModel):
    """DTO for ensemble creation and optimization configuration."""

    base_algorithms: list[str] = Field(description="Base algorithms for ensemble")
    ensemble_strategy: str = Field(
        default="voting", description="Ensemble combination strategy"
    )
    max_ensemble_size: int = Field(
        default=5, ge=2, le=10, description="Maximum ensemble size"
    )
    min_diversity_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum diversity threshold"
    )
    weight_optimization: bool = Field(
        default=True, description="Enable weight optimization"
    )
    diversity_weighting: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Diversity vs performance trade-off"
    )
    cross_validation_folds: int = Field(
        default=3, ge=2, le=10, description="CV folds for validation"
    )
    meta_learning_enabled: bool = Field(
        default=True, description="Enable meta-learning"
    )


class EnsemblePerformanceDTO(BaseModel):
    """DTO for ensemble performance metrics."""

    weighted_individual_performance: float = Field(
        description="Weighted average individual performance"
    )
    diversity_score: float = Field(description="Overall diversity score")
    estimated_ensemble_performance: float = Field(
        description="Estimated ensemble performance"
    )
    performance_improvement: float = Field(
        description="Performance improvement over individual"
    )
    confidence_score: float = Field(description="Confidence in performance estimate")


class EnsembleReportDTO(BaseModel):
    """DTO for comprehensive ensemble report."""

    ensemble_summary: dict[str, Any] = Field(description="Ensemble summary information")
    dataset_characteristics: dict[str, Any] = Field(
        description="Dataset characteristics"
    )
    individual_performance: dict[str, dict[str, float]] = Field(
        description="Individual detector performance"
    )
    diversity_analysis: DiversityMetricsDTO = Field(
        description="Diversity analysis results"
    )
    ensemble_weights: dict[str, float] = Field(description="Optimal ensemble weights")
    configuration: EnsembleConfigurationDTO = Field(
        description="Ensemble configuration"
    )
    performance_summary: EnsemblePerformanceDTO = Field(
        description="Performance summary"
    )
    recommendations: list[str] = Field(description="Improvement recommendations")
    meta_learning_insights: dict[str, Any] = Field(description="Meta-learning insights")


class AlgorithmCompatibilityDTO(BaseModel):
    """DTO for algorithm compatibility matrix."""

    algorithm_name: str = Field(description="Primary algorithm name")
    compatibility_scores: dict[str, float] = Field(
        description="Compatibility scores with other algorithms"
    )
    diversity_potential: float = Field(
        description="Potential for creating diverse ensembles"
    )
    recommended_combinations: list[str] = Field(
        description="Recommended algorithm combinations"
    )


class EnsembleOptimizationRequestDTO(BaseModel):
    """DTO for ensemble optimization request."""

    dataset_name: str = Field(description="Dataset name")
    algorithms: list[str] | None = Field(
        None, description="Algorithms to consider (None for auto-selection)"
    )
    configuration: EnsembleConfigurationDTO = Field(
        description="Ensemble configuration"
    )
    meta_learning_enabled: bool = Field(
        default=True, description="Enable meta-learning"
    )
    optimization_objectives: list[str] = Field(
        default=["performance", "diversity"], description="Optimization objectives"
    )


class EnsembleOptimizationResponseDTO(BaseModel):
    """DTO for ensemble optimization response."""

    request_id: str = Field(description="Request identifier")
    status: str = Field(description="Optimization status")
    ensemble_detectors: list[str] = Field(description="Selected ensemble detectors")
    ensemble_report: EnsembleReportDTO = Field(
        description="Comprehensive ensemble report"
    )
    optimization_time: float = Field(description="Total optimization time")
    meta_knowledge_updated: bool = Field(
        description="Whether meta-knowledge was updated"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


class MetaLearningInsightsDTO(BaseModel):
    """DTO for meta-learning insights and analytics."""

    knowledge_base_size: int = Field(description="Size of meta-learning knowledge base")
    most_popular_algorithms: list[tuple] = Field(
        description="Most popular algorithms with usage counts"
    )
    average_performance_by_algorithm: dict[str, float] = Field(
        description="Average performance by algorithm"
    )
    learning_confidence: float = Field(description="Overall learning confidence")
    diversity_patterns: dict[str, float] = Field(
        description="Observed diversity patterns"
    )
    performance_trends: dict[str, list[float]] = Field(
        description="Performance trends over time"
    )
    dataset_coverage: dict[str, int] = Field(
        description="Coverage of different dataset types"
    )


class EnsembleValidationDTO(BaseModel):
    """DTO for ensemble validation results."""

    validation_method: str = Field(description="Validation method used")
    cross_validation_scores: list[float] = Field(description="Cross-validation scores")
    diversity_validation: DiversityMetricsDTO = Field(
        description="Diversity validation metrics"
    )
    stability_score: float = Field(description="Ensemble stability score")
    robustness_score: float = Field(description="Ensemble robustness score")
    generalization_estimate: float = Field(
        description="Generalization performance estimate"
    )
    validation_warnings: list[str] = Field(description="Validation warnings")


class DynamicEnsembleDTO(BaseModel):
    """DTO for dynamic ensemble selection configuration."""

    selection_strategy: str = Field(
        default="competence_based", description="Dynamic selection strategy"
    )
    competence_measure: str = Field(
        default="accuracy", description="Competence measurement method"
    )
    neighborhood_size: int = Field(
        default=5, description="Neighborhood size for competence estimation"
    )
    competence_threshold: float = Field(
        default=0.6, description="Minimum competence threshold"
    )
    fallback_strategy: str = Field(
        default="best_overall", description="Fallback strategy"
    )


class EnsembleEvolutionDTO(BaseModel):
    """DTO for ensemble evolution and adaptation."""

    evolution_strategy: str = Field(
        description="Evolution strategy (genetic, gradient, etc.)"
    )
    mutation_rate: float = Field(default=0.1, description="Mutation rate for evolution")
    population_size: int = Field(
        default=20, description="Population size for genetic algorithms"
    )
    generations: int = Field(default=50, description="Number of generations")
    fitness_function: str = Field(
        default="weighted", description="Fitness function definition"
    )
    elitism_ratio: float = Field(
        default=0.2, description="Elitism ratio for genetic algorithms"
    )


class StakingConfigurationDTO(BaseModel):
    """DTO for stacking ensemble configuration."""

    meta_learner_algorithm: str = Field(
        default="LogisticRegression", description="Meta-learner algorithm"
    )
    base_level_cv_folds: int = Field(default=3, description="CV folds for base level")
    meta_level_cv_folds: int = Field(default=5, description="CV folds for meta level")
    feature_engineering: bool = Field(
        default=True, description="Enable feature engineering"
    )
    regularization: float = Field(default=0.1, description="Regularization parameter")


class EnsembleMonitoringDTO(BaseModel):
    """DTO for real-time ensemble monitoring."""

    monitoring_enabled: bool = Field(
        default=True, description="Enable real-time monitoring"
    )
    performance_tracking: bool = Field(
        default=True, description="Track performance metrics"
    )
    drift_detection: bool = Field(default=True, description="Enable drift detection")
    alert_thresholds: dict[str, float] = Field(
        description="Alert threshold configuration"
    )
    monitoring_interval: int = Field(
        default=3600, description="Monitoring interval in seconds"
    )
    metric_retention_days: int = Field(
        default=30, description="Metric retention period"
    )


class EnsembleBenchmarkDTO(BaseModel):
    """DTO for ensemble benchmarking results."""

    benchmark_name: str = Field(description="Benchmark name")
    dataset_characteristics: dict[str, Any] = Field(
        description="Benchmark dataset characteristics"
    )
    ensemble_configurations: list[EnsembleConfigurationDTO] = Field(
        description="Tested configurations"
    )
    performance_results: dict[str, float] = Field(description="Performance results")
    execution_times: dict[str, float] = Field(description="Execution times")
    resource_usage: dict[str, float] = Field(description="Resource usage metrics")
    best_configuration: EnsembleConfigurationDTO = Field(
        description="Best performing configuration"
    )
    improvement_over_single: float = Field(
        description="Improvement over single best detector"
    )


class MetaFeatureDTO(BaseModel):
    """DTO for meta-features used in meta-learning."""

    statistical_features: dict[str, float] = Field(
        description="Statistical meta-features"
    )
    information_theoretic_features: dict[str, float] = Field(
        description="Information-theoretic features"
    )
    model_based_features: dict[str, float] = Field(
        description="Model-based meta-features"
    )
    landmarking_features: dict[str, float] = Field(
        description="Landmarking meta-features"
    )
    complexity_features: dict[str, float] = Field(
        description="Data complexity features"
    )


class EnsembleExplanationDTO(BaseModel):
    """DTO for ensemble decision explanation."""

    ensemble_decision: str = Field(description="Final ensemble decision")
    individual_contributions: dict[str, float] = Field(
        description="Individual detector contributions"
    )
    weight_explanations: dict[str, str] = Field(
        description="Explanation for ensemble weights"
    )
    diversity_impact: str = Field(description="Impact of diversity on decision")
    confidence_factors: dict[str, float] = Field(
        description="Factors affecting confidence"
    )
    alternative_decisions: list[str] = Field(
        description="Alternative possible decisions"
    )


# Utility functions for DTO conversion
def create_default_ensemble_config() -> EnsembleConfigurationDTO:
    """Create default ensemble configuration."""
    return EnsembleConfigurationDTO(
        base_algorithms=["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
        ensemble_strategy="voting",
        max_ensemble_size=5,
        min_diversity_threshold=0.3,
        weight_optimization=True,
        diversity_weighting=0.3,
        cross_validation_folds=3,
        meta_learning_enabled=True,
    )


def create_diversity_metrics_from_dict(data: dict[str, float]) -> DiversityMetricsDTO:
    """Create diversity metrics DTO from dictionary."""
    return DiversityMetricsDTO(
        disagreement_measure=data.get("disagreement_measure", 0.0),
        double_fault_measure=data.get("double_fault_measure", 0.0),
        q_statistic=data.get("q_statistic", 0.0),
        correlation_coefficient=data.get("correlation_coefficient", 0.0),
        kappa_statistic=data.get("kappa_statistic", 0.0),
        entropy_measure=data.get("entropy_measure", 0.0),
        overall_diversity=data.get("overall_diversity", 0.0),
    )


def create_ensemble_performance_from_dict(
    data: dict[str, float],
) -> EnsemblePerformanceDTO:
    """Create ensemble performance DTO from dictionary."""
    return EnsemblePerformanceDTO(
        weighted_individual_performance=data.get(
            "weighted_individual_performance", 0.0
        ),
        diversity_score=data.get("diversity_score", 0.0),
        estimated_ensemble_performance=data.get("estimated_ensemble_performance", 0.0),
        performance_improvement=data.get("performance_improvement", 0.0),
        confidence_score=data.get("confidence_score", 0.0),
    )


# DTOs for Ensemble Detection Use Case


class EnsembleDetectionRequestDTO(BaseModel):
    """Request DTO for ensemble detection."""

    detector_ids: list[str] = Field(
        min_length=2,
        max_length=20,
        description="List of detector IDs to include in ensemble (2-20 detectors)",
    )
    data: list[list[float]] | list[dict[str, Any]] = Field(
        description="Input data as array of arrays or list of dictionaries"
    )
    voting_strategy: str = Field(
        default="dynamic_selection",
        description="Voting strategy for ensemble combination",
    )
    enable_dynamic_weighting: bool = Field(
        default=True, description="Enable performance-based dynamic weighting"
    )
    enable_uncertainty_estimation: bool = Field(
        default=True, description="Enable uncertainty estimation for predictions"
    )
    enable_explanation: bool = Field(
        default=True, description="Generate explanations for ensemble predictions"
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for cascaded voting",
    )
    consensus_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Agreement threshold for consensus voting",
    )
    max_processing_time: float | None = Field(
        default=None, gt=0, description="Maximum processing time in seconds"
    )
    enable_caching: bool = Field(
        default=True, description="Enable result caching for performance"
    )
    return_individual_results: bool = Field(
        default=False, description="Return individual detector results"
    )

    @field_validator("voting_strategy")
    @classmethod
    def validate_voting_strategy(cls, v):
        """Validate voting strategy."""
        valid_strategies = {
            "simple_average",
            "weighted_average",
            "bayesian_model_averaging",
            "rank_aggregation",
            "consensus_voting",
            "dynamic_selection",
            "uncertainty_weighted",
            "performance_weighted",
            "diversity_weighted",
            "adaptive_threshold",
            "robust_aggregation",
            "cascaded_voting",
        }
        if v not in valid_strategies:
            raise ValueError(
                f"Invalid voting strategy. Must be one of: {valid_strategies}"
            )
        return v

    @field_validator("data")
    @classmethod
    def validate_data(cls, v):
        """Validate input data format."""
        if not v:
            raise ValueError("Data cannot be empty")

        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("Data list cannot be empty")

            # Check if it's list of lists (array format)
            if isinstance(v[0], list):
                # Validate that all rows have same number of features
                expected_features = len(v[0])
                for i, row in enumerate(v):
                    if not isinstance(row, list):
                        raise ValueError(f"Row {i} is not a list")
                    if len(row) != expected_features:
                        raise ValueError(
                            f"Row {i} has {len(row)} features, expected {expected_features}"
                        )

            # Check if it's list of dicts
            elif isinstance(v[0], dict):
                # Validate that all dicts have same keys
                expected_keys = set(v[0].keys())
                for i, row in enumerate(v):
                    if not isinstance(row, dict):
                        raise ValueError(f"Row {i} is not a dictionary")
                    if set(row.keys()) != expected_keys:
                        raise ValueError(f"Row {i} has different keys than first row")

            else:
                raise ValueError("Data must be list of lists or list of dictionaries")

        return v


class EnsembleDetectionResponseDTO(BaseModel):
    """Response DTO for ensemble detection."""

    success: bool = Field(description="Whether the detection was successful")
    predictions: list[int] = Field(
        default_factory=list,
        description="Binary anomaly predictions (0=normal, 1=anomaly)",
    )
    anomaly_scores: list[float] = Field(
        default_factory=list, description="Continuous anomaly scores (0.0-1.0)"
    )
    confidence_scores: list[float] = Field(
        default_factory=list, description="Prediction confidence levels"
    )
    uncertainty_scores: list[float] = Field(
        default_factory=list, description="Prediction uncertainty estimates"
    )
    consensus_scores: list[float] = Field(
        default_factory=list, description="Agreement scores among detectors"
    )
    individual_results: dict[str, list[float]] | None = Field(
        default=None, description="Individual detector results (if requested)"
    )
    detector_weights: list[float] = Field(
        default_factory=list, description="Weights used for each detector"
    )
    voting_strategy_used: str = Field(
        default="", description="Actual voting strategy used"
    )
    ensemble_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Ensemble diversity and performance metrics"
    )
    explanations: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-prediction explanations"
    )
    processing_time: float = Field(
        default=0.0, description="Total processing time in seconds"
    )
    warnings: list[str] = Field(default_factory=list, description="Warning messages")
    error_message: str | None = Field(
        default=None, description="Error message if detection failed"
    )


class EnsembleOptimizationRequestDTO(BaseModel):
    """Request DTO for ensemble optimization."""

    detector_ids: list[str] = Field(
        min_length=2, description="Candidate detector IDs for ensemble"
    )
    validation_dataset_id: str = Field(
        description="Dataset ID for optimization validation"
    )
    optimization_objective: str = Field(
        default="f1_score", description="Primary optimization objective"
    )
    target_voting_strategies: list[str] = Field(
        default_factory=lambda: ["dynamic_selection"],
        description="Voting strategies to evaluate",
    )
    max_ensemble_size: int = Field(
        default=5, ge=2, le=10, description="Maximum detectors in final ensemble"
    )
    min_diversity_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum required diversity"
    )
    enable_pruning: bool = Field(
        default=True, description="Remove underperforming detectors"
    )
    enable_weight_optimization: bool = Field(
        default=True, description="Optimize detector weights"
    )
    cross_validation_folds: int = Field(
        default=5, ge=2, le=10, description="Number of cross-validation folds"
    )
    optimization_timeout: float = Field(
        default=300.0, gt=0, description="Optimization timeout in seconds"
    )
    random_state: int = Field(
        default=42, description="Random state for reproducibility"
    )

    @field_validator("optimization_objective")
    @classmethod
    def validate_optimization_objective(cls, v):
        """Validate optimization objective."""
        valid_objectives = {
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_score",
            "balanced_accuracy",
            "diversity",
            "stability",
            "efficiency",
        }
        if v not in valid_objectives:
            raise ValueError(
                f"Invalid optimization objective. Must be one of: {valid_objectives}"
            )
        return v

    @field_validator("target_voting_strategies")
    @classmethod
    def validate_target_strategies(cls, v):
        """Validate target voting strategies."""
        valid_strategies = {
            "simple_average",
            "weighted_average",
            "bayesian_model_averaging",
            "rank_aggregation",
            "consensus_voting",
            "dynamic_selection",
            "uncertainty_weighted",
            "performance_weighted",
            "diversity_weighted",
            "adaptive_threshold",
            "robust_aggregation",
            "cascaded_voting",
        }
        for strategy in v:
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid voting strategy: {strategy}")
        return v


class EnsembleOptimizationResponseDTO(BaseModel):
    """Response DTO for ensemble optimization."""

    success: bool = Field(description="Whether optimization was successful")
    optimized_detector_ids: list[str] = Field(
        default_factory=list, description="Optimized detector combination"
    )
    optimal_voting_strategy: str = Field(
        default="", description="Best voting strategy found"
    )
    optimal_weights: list[float] = Field(
        default_factory=list, description="Optimized detector weights"
    )
    ensemble_performance: dict[str, float] = Field(
        default_factory=dict, description="Performance metrics on validation data"
    )
    diversity_metrics: dict[str, float] = Field(
        default_factory=dict, description="Ensemble diversity analysis"
    )
    optimization_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Optimization process history"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Optimization recommendations"
    )
    optimization_time: float = Field(
        default=0.0, description="Total optimization time in seconds"
    )
    error_message: str | None = Field(
        default=None, description="Error message if optimization failed"
    )


class EnsembleStatusResponseDTO(BaseModel):
    """Response DTO for ensemble system status."""

    available_voting_strategies: list[dict[str, Any]] = Field(
        description="Available voting strategies with descriptions"
    )
    available_optimization_objectives: list[dict[str, Any]] = Field(
        description="Available optimization objectives with descriptions"
    )
    system_capabilities: dict[str, Any] = Field(
        description="System capabilities and limits"
    )
    system_statistics: dict[str, Any] = Field(description="Current system statistics")


class EnsembleMetricsResponseDTO(BaseModel):
    """Response DTO for ensemble performance metrics."""

    detector_performance_metrics: dict[str, dict[str, Any]] = Field(
        description="Individual detector performance metrics"
    )
    ensemble_statistics: dict[str, Any] = Field(description="Ensemble-level statistics")
    recent_optimizations: list[dict[str, Any]] = Field(
        description="Recent optimization runs"
    )
    system_health: dict[str, Any] = Field(description="System health indicators")

"""Data Transfer Objects for advanced ensemble methods and meta-learning."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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

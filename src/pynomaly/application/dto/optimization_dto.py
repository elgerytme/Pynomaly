"""Data Transfer Objects for advanced optimization and AutoML."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class OptimizationObjectiveDTO(BaseModel):
    """DTO for optimization objective configuration."""

    name: str = Field(description="Objective name (e.g., 'accuracy', 'speed')")
    weight: float = Field(gt=0.0, le=1.0, description="Objective weight (0-1)")
    direction: str = Field(
        pattern="^(maximize|minimize)$", description="Optimization direction"
    )
    threshold: float | None = Field(None, description="Minimum acceptable threshold")
    description: str = Field(default="", description="Objective description")


class ResourceConstraintsDTO(BaseModel):
    """DTO for optimization resource constraints."""

    max_time_seconds: int = Field(
        default=3600, ge=60, description="Maximum optimization time"
    )
    max_trials: int = Field(default=100, ge=10, description="Maximum number of trials")
    max_memory_mb: int = Field(default=4096, ge=512, description="Maximum memory usage")
    max_cpu_cores: int = Field(default=4, ge=1, description="Maximum CPU cores")
    gpu_available: bool = Field(default=False, description="GPU availability")
    prefer_speed: bool = Field(default=False, description="Prefer speed over accuracy")


class OptimizationConfigDTO(BaseModel):
    """DTO for complete optimization configuration."""

    algorithm_name: str = Field(description="Algorithm to optimize")
    objectives: list[OptimizationObjectiveDTO] = Field(
        description="Optimization objectives"
    )
    constraints: ResourceConstraintsDTO = Field(description="Resource constraints")
    enable_learning: bool = Field(
        default=True, description="Enable learning from history"
    )
    enable_distributed: bool = Field(
        default=False, description="Enable distributed optimization"
    )
    n_parallel_jobs: int = Field(default=1, ge=1, description="Number of parallel jobs")


class DatasetCharacteristicsDTO(BaseModel):
    """DTO for dataset characteristics analysis."""

    n_samples: int = Field(description="Number of samples")
    n_features: int = Field(description="Number of features")
    size_category: str = Field(description="Dataset size category")
    feature_types: dict[str, float] = Field(description="Feature type distribution")
    data_distribution: dict[str, float] = Field(description="Data distribution stats")
    sparsity: float = Field(description="Data sparsity ratio")
    correlation_structure: dict[str, float] = Field(description="Correlation analysis")
    outlier_characteristics: dict[str, float] = Field(description="Outlier patterns")


class OptimizationResultDTO(BaseModel):
    """DTO for optimization results."""

    best_parameters: dict[str, Any] = Field(description="Best parameters found")
    best_metrics: dict[str, float] = Field(description="Best metric values")
    optimization_time: float = Field(description="Total optimization time")
    total_trials: int = Field(description="Total trials executed")
    successful_trials: int = Field(description="Successfully completed trials")
    pareto_solutions: list[dict[str, Any]] = Field(
        description="Pareto optimal solutions"
    )


class OptimizationHistoryDTO(BaseModel):
    """DTO for optimization history entry."""

    algorithm_name: str = Field(description="Algorithm name")
    dataset_characteristics: DatasetCharacteristicsDTO = Field(
        description="Dataset characteristics"
    )
    best_parameters: dict[str, Any] = Field(description="Best parameters")
    performance_metrics: dict[str, float] = Field(description="Performance metrics")
    optimization_time: float = Field(description="Optimization time")
    resource_usage: dict[str, float] = Field(description="Resource usage")
    user_feedback: dict[str, Any] | None = Field(None, description="User feedback")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Optimization timestamp"
    )


class LearningInsightsDTO(BaseModel):
    """DTO for learning insights from optimization history."""

    algorithm_trends: dict[str, dict[str, Any]] = Field(
        description="Algorithm performance trends"
    )
    total_optimizations: int = Field(description="Total optimizations performed")
    learning_insights: list[str] = Field(description="Generated learning insights")
    performance_improvements: dict[str, float] = Field(
        description="Performance improvements by algorithm"
    )
    parameter_preferences: dict[str, dict[str, Any]] = Field(
        description="Learned parameter preferences"
    )


class AutoMLRequestDTO(BaseModel):
    """DTO for AutoML optimization request."""

    dataset_name: str = Field(description="Dataset name")
    algorithm_names: list[str] = Field(description="Algorithms to optimize")
    optimization_config: OptimizationConfigDTO = Field(
        description="Optimization configuration"
    )
    evaluation_mode: str = Field(
        default="cross_validation", description="Evaluation method"
    )
    output_format: str = Field(default="comprehensive", description="Output format")


class AutoMLResponseDTO(BaseModel):
    """DTO for AutoML optimization response."""

    request_id: str = Field(description="Request identifier")
    status: str = Field(description="Optimization status")
    results: list[OptimizationResultDTO] = Field(
        description="Optimization results per algorithm"
    )
    best_overall: OptimizationResultDTO = Field(description="Best overall result")
    execution_summary: dict[str, Any] = Field(description="Execution summary")
    recommendations: list[str] = Field(description="Generated recommendations")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Response timestamp"
    )


class EnsembleOptimizationDTO(BaseModel):
    """DTO for ensemble optimization configuration."""

    base_algorithms: list[str] = Field(description="Base algorithms for ensemble")
    ensemble_method: str = Field(
        default="voting", description="Ensemble combination method"
    )
    weight_optimization: bool = Field(
        default=True, description="Optimize ensemble weights"
    )
    diversity_threshold: float = Field(
        default=0.3, description="Minimum diversity threshold"
    )
    max_ensemble_size: int = Field(default=5, description="Maximum ensemble size")


class MetaLearningConfigDTO(BaseModel):
    """DTO for meta-learning configuration."""

    enable_meta_learning: bool = Field(default=True, description="Enable meta-learning")
    similarity_threshold: float = Field(
        default=0.7, description="Dataset similarity threshold"
    )
    learning_rate: float = Field(default=0.1, description="Meta-learning rate")
    memory_size: int = Field(default=1000, description="Meta-learning memory size")
    adaptation_strategy: str = Field(
        default="weighted", description="Adaptation strategy"
    )


class PerformancePredictionDTO(BaseModel):
    """DTO for performance prediction."""

    algorithm_name: str = Field(description="Algorithm name")
    dataset_characteristics: DatasetCharacteristicsDTO = Field(
        description="Dataset characteristics"
    )
    predicted_metrics: dict[str, float] = Field(
        description="Predicted performance metrics"
    )
    confidence_intervals: dict[str, list[float]] = Field(
        description="Prediction confidence intervals"
    )
    prediction_accuracy: float = Field(description="Historical prediction accuracy")
    risk_assessment: str = Field(description="Risk level assessment")


class OptimizationMonitoringDTO(BaseModel):
    """DTO for real-time optimization monitoring."""

    optimization_id: str = Field(description="Optimization identifier")
    current_trial: int = Field(description="Current trial number")
    total_trials: int = Field(description="Total planned trials")
    elapsed_time: float = Field(description="Elapsed time in seconds")
    estimated_remaining: float = Field(description="Estimated remaining time")
    current_best: dict[str, float] = Field(description="Current best metrics")
    trial_history: list[dict[str, Any]] = Field(description="Recent trial history")
    resource_usage: dict[str, float] = Field(description="Current resource usage")
    status: str = Field(description="Optimization status")


class HyperparameterSpaceDTO(BaseModel):
    """DTO for hyperparameter search space definition."""

    algorithm_name: str = Field(description="Algorithm name")
    parameter_definitions: dict[str, dict[str, Any]] = Field(
        description="Parameter definitions"
    )
    constraints: list[dict[str, Any]] = Field(description="Parameter constraints")
    prior_knowledge: dict[str, Any] | None = Field(
        None, description="Prior parameter knowledge"
    )
    custom_samplers: dict[str, str] | None = Field(
        None, description="Custom sampling strategies"
    )


class OptimizationProfileDTO(BaseModel):
    """DTO for optimization profile and preferences."""

    user_id: str = Field(description="User identifier")
    optimization_preferences: dict[str, Any] = Field(
        description="User optimization preferences"
    )
    algorithm_preferences: list[str] = Field(description="Preferred algorithms")
    resource_budget: ResourceConstraintsDTO = Field(
        description="Typical resource budget"
    )
    success_criteria: dict[str, float] = Field(
        description="Success criteria thresholds"
    )
    historical_performance: dict[str, float] = Field(
        description="Historical performance metrics"
    )


class BatchOptimizationDTO(BaseModel):
    """DTO for batch optimization across multiple datasets."""

    datasets: list[str] = Field(description="Dataset names for batch optimization")
    algorithms: list[str] = Field(description="Algorithms to optimize")
    shared_config: OptimizationConfigDTO = Field(
        description="Shared optimization configuration"
    )
    parallel_execution: bool = Field(
        default=True, description="Enable parallel execution"
    )
    result_aggregation: str = Field(
        default="average", description="Result aggregation method"
    )
    cross_dataset_learning: bool = Field(
        default=True, description="Enable cross-dataset learning"
    )


class OptimizationReportDTO(BaseModel):
    """DTO for comprehensive optimization report."""

    executive_summary: dict[str, Any] = Field(description="Executive summary")
    detailed_results: list[OptimizationResultDTO] = Field(
        description="Detailed results"
    )
    performance_analysis: dict[str, Any] = Field(description="Performance analysis")
    learning_insights: LearningInsightsDTO = Field(description="Learning insights")
    recommendations: list[str] = Field(description="Actionable recommendations")
    next_steps: list[str] = Field(description="Suggested next steps")
    metadata: dict[str, Any] = Field(description="Report metadata")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Report generation time"
    )


# Utility functions for DTO conversion
def create_default_objectives() -> list[OptimizationObjectiveDTO]:
    """Create default optimization objectives."""
    return [
        OptimizationObjectiveDTO(
            name="accuracy",
            weight=0.4,
            direction="maximize",
            description="Detection accuracy (AUC-ROC)",
        ),
        OptimizationObjectiveDTO(
            name="speed",
            weight=0.3,
            direction="maximize",
            description="Training and inference speed",
        ),
        OptimizationObjectiveDTO(
            name="interpretability",
            weight=0.2,
            direction="maximize",
            description="Model interpretability score",
        ),
        OptimizationObjectiveDTO(
            name="memory_efficiency",
            weight=0.1,
            direction="maximize",
            description="Memory usage efficiency",
        ),
    ]


def create_default_constraints() -> ResourceConstraintsDTO:
    """Create default resource constraints."""
    return ResourceConstraintsDTO(
        max_time_seconds=3600,  # 1 hour
        max_trials=100,
        max_memory_mb=4096,  # 4 GB
        max_cpu_cores=4,
        gpu_available=False,
        prefer_speed=False,
    )


def create_optimization_config(
    algorithm_name: str,
    objectives: list[OptimizationObjectiveDTO] | None = None,
    constraints: ResourceConstraintsDTO | None = None,
) -> OptimizationConfigDTO:
    """Create optimization configuration with defaults."""
    return OptimizationConfigDTO(
        algorithm_name=algorithm_name,
        objectives=objectives or create_default_objectives(),
        constraints=constraints or create_default_constraints(),
        enable_learning=True,
        enable_distributed=False,
        n_parallel_jobs=1,
    )

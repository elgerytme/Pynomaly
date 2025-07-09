"""Data Transfer Objects for AutoML operations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AlgorithmRecommendationRequestDTO(BaseModel):
    """DTO for algorithm recommendation request."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")
    dataset_id: str = Field(..., description="ID of the dataset for recommendation")
    max_recommendations: int = Field(
        default=5, ge=1, le=20, description="Maximum number of recommendations"
    )
    performance_priority: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Priority for performance (0-1)"
    )
    speed_priority: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Priority for speed (0-1)"
    )
    include_experimental: bool = Field(
        default=False, description="Include experimental algorithms"
    )
    exclude_algorithms: list[str] = Field(
        default_factory=list, description="Algorithms to exclude"
    )


class DatasetProfileDTO(BaseModel):
    """DTO for dataset profiling information."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")
    n_samples: int = Field(..., description="Number of samples in the dataset")
    n_features: int = Field(..., description="Number of features in the dataset")
    contamination_estimate: float = Field(
        ..., description="Estimated contamination rate"
    )
    feature_types: dict[str, str] = Field(..., description="Feature type mapping")
    missing_values_ratio: float = Field(..., description="Ratio of missing values")
    categorical_features: list[str] = Field(
        default_factory=list, description="List of categorical features"
    )
    numerical_features: list[str] = Field(
        default_factory=list, description="List of numerical features"
    )
    time_series_features: list[str] = Field(
        default_factory=list, description="List of time series features"
    )
    sparsity_ratio: float = Field(..., description="Data sparsity ratio")
    dimensionality_ratio: float = Field(
        ..., description="Dimensionality ratio (features/samples)"
    )
    dataset_size_mb: float = Field(..., description="Dataset size in megabytes")
    has_temporal_structure: bool = Field(
        default=False, description="Whether dataset has temporal structure"
    )
    has_graph_structure: bool = Field(
        default=False, description="Whether dataset has graph structure"
    )
    complexity_score: float = Field(..., description="Calculated complexity score")


class AlgorithmRecommendationDTO(BaseModel):
    """DTO for algorithm recommendation."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")
    algorithm_name: str = Field(..., description="Name of the recommended algorithm")
    score: float = Field(..., description="Suitability score for the algorithm")
    family: str = Field(..., description="Algorithm family")
    complexity_score: float = Field(..., description="Algorithm complexity score")
    recommended_params: dict[str, Any] = Field(
        ..., description="Recommended hyperparameters"
    )
    reasoning: list[str] = Field(
        default_factory=list, description="Reasoning for recommendation"
    )


class AutoMLRequestDTO(BaseModel):
    """DTO for AutoML optimization request."""
    
    model_config = ConfigDict(from_attributes=True, extra="forbid")
    dataset_id: str = Field(..., description="ID of the dataset to optimize on")
    objective: str = Field(default="auc", description="Optimization objective")
    max_algorithms: int = Field(
        default=3, ge=1, le=10, description="Maximum number of algorithms to try"
    )
    max_optimization_time: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Maximum optimization time in seconds",
    )
    n_trials: int = Field(
        default=100, ge=10, le=1000, description="Number of optimization trials"
    )
    enable_ensemble: bool = Field(
        default=True, description="Whether to enable ensemble creation"
    )
    detector_name: str | None = Field(
        default=None, description="Optional custom name for the detector"
    )
    cross_validation_folds: int = Field(
        default=3, ge=2, le=10, description="Number of cross-validation folds"
    )
    random_state: int = Field(
        default=42, description="Random state for reproducibility"
    )


class HyperparameterSpaceDTO(BaseModel):
    """DTO for hyperparameter search space."""

    model_config = ConfigDict(from_attributes=True)

    parameter_name: str = Field(..., description="Name of the hyperparameter")
    parameter_type: str = Field(
        ..., description="Type of parameter (float, int, categorical)"
    )
    low: float | None = Field(
        default=None, description="Lower bound for numerical parameters"
    )
    high: float | None = Field(
        default=None, description="Upper bound for numerical parameters"
    )
    choices: list[Any] | None = Field(
        default=None, description="Choices for categorical parameters"
    )
    log: bool = Field(default=False, description="Whether to use log scale")
    description: str = Field(..., description="Description of the parameter")


class OptimizationTrialDTO(BaseModel):
    """DTO for individual optimization trial."""

    model_config = ConfigDict(from_attributes=True)

    trial_number: int = Field(..., description="Trial number")
    parameters: dict[str, Any] = Field(..., description="Trial parameters")
    score: float = Field(..., description="Trial score")
    state: str = Field(..., description="Trial state (COMPLETE, FAIL, PRUNED)")
    duration: float = Field(..., description="Trial duration in seconds")
    algorithm: str = Field(..., description="Algorithm used in trial")


class EnsembleConfigDTO(BaseModel):
    """DTO for ensemble configuration."""

    model_config = ConfigDict(from_attributes=True)

    method: str = Field(..., description="Ensemble method")
    algorithms: list[dict[str, Any]] = Field(
        ..., description="List of algorithms in ensemble"
    )
    voting_strategy: str = Field(..., description="Voting strategy")
    normalize_scores: bool = Field(
        default=True, description="Whether to normalize scores"
    )
    weights: list[float] | None = Field(default=None, description="Algorithm weights")


class AutoMLResultDTO(BaseModel):
    """DTO for AutoML optimization result."""

    model_config = ConfigDict(from_attributes=True)

    best_algorithm: str = Field(..., description="Best performing algorithm")
    best_params: dict[str, Any] = Field(..., description="Best hyperparameters")
    best_score: float = Field(..., description="Best achieved score")
    optimization_time: float = Field(..., description="Total optimization time")
    trials_completed: int = Field(..., description="Number of completed trials")
    algorithm_rankings: list[tuple] = Field(
        ..., description="Algorithm performance rankings"
    )
    ensemble_config: EnsembleConfigDTO | None = Field(
        default=None, description="Ensemble configuration if created"
    )
    cross_validation_scores: list[float] | None = Field(
        default=None, description="Cross-validation scores"
    )
    feature_importance: dict[str, float] | None = Field(
        default=None, description="Feature importance scores"
    )
    optimization_history: list[OptimizationTrialDTO] | None = Field(
        default=None, description="Optimization trial history"
    )


class AutoMLResponseDTO(BaseModel):
    """DTO for AutoML operation response."""

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(..., description="Whether the operation was successful")
    detector_id: str | None = Field(
        default=None, description="ID of the created detector"
    )
    automl_result: AutoMLResultDTO | None = Field(
        default=None, description="AutoML optimization result"
    )
    dataset_profile: DatasetProfileDTO | None = Field(
        default=None, description="Dataset profiling information"
    )
    algorithm_recommendations: list[AlgorithmRecommendationDTO] | None = Field(
        default=None, description="Algorithm recommendations"
    )
    optimization_summary: dict[str, Any] | None = Field(
        default=None, description="Optimization summary"
    )
    message: str = Field(..., description="Response message")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Total execution time")


class AutoMLProfileRequestDTO(BaseModel):
    """DTO for dataset profiling request."""

    model_config = ConfigDict(from_attributes=True)

    dataset_id: str = Field(..., description="ID of the dataset to profile")
    include_recommendations: bool = Field(
        default=True, description="Whether to include algorithm recommendations"
    )
    max_recommendations: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of algorithm recommendations",
    )


class AutoMLProfileResponseDTO(BaseModel):
    """DTO for dataset profiling response."""

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(..., description="Whether the profiling was successful")
    dataset_profile: DatasetProfileDTO | None = Field(
        default=None, description="Dataset profile"
    )
    algorithm_recommendations: list[AlgorithmRecommendationDTO] | None = Field(
        default=None, description="Algorithm recommendations"
    )
    message: str = Field(..., description="Response message")
    error: str | None = Field(default=None, description="Error message if failed")
    execution_time: float = Field(..., description="Profiling execution time")


class HyperparameterOptimizationRequestDTO(BaseModel):
    """DTO for hyperparameter optimization request."""

    model_config = ConfigDict(from_attributes=True)

    dataset_id: str = Field(..., description="ID of the dataset")
    algorithm: str = Field(..., description="Algorithm to optimize")
    objective: str = Field(default="auc", description="Optimization objective")
    n_trials: int = Field(
        default=100, ge=10, le=1000, description="Number of optimization trials"
    )
    timeout: int = Field(
        default=3600, ge=60, le=86400, description="Optimization timeout in seconds"
    )
    direction: str = Field(default="maximize", description="Optimization direction")
    custom_param_space: dict[str, HyperparameterSpaceDTO] | None = Field(
        default=None, description="Custom parameter space"
    )
    cross_validation_folds: int = Field(
        default=3, ge=2, le=10, description="Number of CV folds"
    )
    random_state: int = Field(default=42, description="Random state")


class HyperparameterOptimizationResponseDTO(BaseModel):
    """DTO for hyperparameter optimization response."""

    model_config = ConfigDict(from_attributes=True)

    success: bool = Field(..., description="Whether optimization was successful")
    best_params: dict[str, Any] | None = Field(
        default=None, description="Best hyperparameters"
    )
    best_score: float | None = Field(default=None, description="Best score achieved")
    optimization_time: float = Field(..., description="Optimization time")
    trials_completed: int = Field(default=0, description="Number of trials completed")
    optimization_history: list[OptimizationTrialDTO] | None = Field(
        default=None, description="Trial history"
    )
    algorithm: str = Field(..., description="Algorithm that was optimized")
    objective: str = Field(..., description="Optimization objective used")
    message: str = Field(..., description="Response message")
    error: str | None = Field(default=None, description="Error message if failed")

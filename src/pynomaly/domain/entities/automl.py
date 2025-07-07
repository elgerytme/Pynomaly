"""AutoML domain entities for automated anomaly detection pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class OptimizationObjective(str, Enum):
    """AutoML optimization objectives."""
    
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    SPEED = "speed"
    MEMORY_USAGE = "memory_usage"
    BALANCED = "balanced"
    CUSTOM = "custom"


class SearchStrategy(str, Enum):
    """Hyperparameter search strategies."""
    
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"
    OPTUNA = "optuna"
    SUCCESSIVE_HALVING = "successive_halving"


class FeatureSelectionMethod(str, Enum):
    """Feature selection methods."""
    
    MUTUAL_INFORMATION = "mutual_information"
    CORRELATION = "correlation"
    VARIANCE_THRESHOLD = "variance_threshold"
    RECURSIVE_ELIMINATION = "recursive_elimination"
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"
    BORUTA = "boruta"
    SHAP_BASED = "shap_based"


class AutoMLStatus(str, Enum):
    """AutoML pipeline execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class HyperparameterSpace:
    """Hyperparameter search space definition."""
    
    parameter_name: str
    parameter_type: str  # int, float, categorical, boolean
    min_value: Any | None = None
    max_value: Any | None = None
    step: Any | None = None
    choices: list[Any] | None = None
    default_value: Any | None = None
    
    def __post_init__(self) -> None:
        """Validate hyperparameter space."""
        valid_types = {"int", "float", "categorical", "boolean"}
        if self.parameter_type not in valid_types:
            raise ValueError(f"Parameter type must be one of: {valid_types}")
        
        if self.parameter_type == "categorical":
            if not self.choices:
                raise ValueError("Categorical parameters must have choices")
        
        if self.parameter_type in ["int", "float"]:
            if self.min_value is None or self.max_value is None:
                raise ValueError("Numeric parameters must have min and max values")

    def validate_value(self, value: Any) -> bool:
        """Validate if a value is within the parameter space."""
        if self.parameter_type == "categorical":
            return value in self.choices
        elif self.parameter_type == "boolean":
            return isinstance(value, bool)
        elif self.parameter_type in ["int", "float"]:
            return self.min_value <= value <= self.max_value
        return False


@dataclass
class AlgorithmConfiguration:
    """Configuration for an anomaly detection algorithm."""
    
    algorithm_name: str
    algorithm_class: str
    hyperparameter_space: list[HyperparameterSpace]
    
    # Algorithm metadata
    supports_incremental: bool = False
    supports_streaming: bool = False
    memory_requirements: str = "medium"  # low, medium, high
    computational_complexity: str = "medium"  # low, medium, high
    
    # Default configuration
    default_parameters: dict[str, Any] = field(default_factory=dict)
    
    # Constraints
    min_samples: int = 100
    max_features: int | None = None
    
    def __post_init__(self) -> None:
        """Validate algorithm configuration."""
        if not self.algorithm_name.strip():
            raise ValueError("Algorithm name cannot be empty")
        if self.min_samples <= 0:
            raise ValueError("Minimum samples must be positive")

    def get_parameter_names(self) -> list[str]:
        """Get list of hyperparameter names."""
        return [hp.parameter_name for hp in self.hyperparameter_space]

    def validate_parameters(self, parameters: dict[str, Any]) -> bool:
        """Validate if parameters are within defined space."""
        param_spaces = {hp.parameter_name: hp for hp in self.hyperparameter_space}
        
        for param_name, value in parameters.items():
            if param_name not in param_spaces:
                return False
            if not param_spaces[param_name].validate_value(value):
                return False
        
        return True


@dataclass
class FeatureEngineeringStep:
    """Feature engineering step configuration."""
    
    step_name: str
    step_type: str  # scaling, encoding, selection, transformation
    method: str
    parameters: dict[str, Any] = field(default_factory=dict)
    
    # Execution order
    order: int = 0
    depends_on: list[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate feature engineering step."""
        valid_types = {"scaling", "encoding", "selection", "transformation", "imputation"}
        if self.step_type not in valid_types:
            raise ValueError(f"Step type must be one of: {valid_types}")


@dataclass
class AutoMLConfiguration:
    """Complete AutoML pipeline configuration."""
    
    # Basic configuration
    pipeline_name: str
    optimization_objective: OptimizationObjective
    search_strategy: SearchStrategy
    
    # Algorithm selection
    candidate_algorithms: list[AlgorithmConfiguration]
    algorithm_selection_strategy: str = "performance_based"  # performance_based, resource_based, ensemble
    
    # Feature engineering
    feature_engineering_steps: list[FeatureEngineeringStep] = field(default_factory=list)
    feature_selection_method: FeatureSelectionMethod | None = None
    max_features: int | None = None
    
    # Search configuration
    max_trials: int = 100
    max_time: timedelta = field(default_factory=lambda: timedelta(hours=4))
    early_stopping_patience: int = 10
    
    # Validation strategy
    validation_strategy: str = "time_series_split"  # time_series_split, stratified_kfold, hold_out
    validation_splits: int = 5
    test_size: float = 0.2
    
    # Resource constraints
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 4
    enable_gpu: bool = False
    
    # Advanced options
    enable_ensemble: bool = True
    ensemble_size: int = 5
    enable_model_explanation: bool = True
    enable_drift_detection: bool = True
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate AutoML configuration."""
        if not self.pipeline_name.strip():
            raise ValueError("Pipeline name cannot be empty")
        if not self.candidate_algorithms:
            raise ValueError("At least one candidate algorithm must be specified")
        if self.max_trials <= 0:
            raise ValueError("Max trials must be positive")
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("Test size must be between 0 and 1")
        if self.validation_splits <= 1:
            raise ValueError("Validation splits must be greater than 1")

    def get_total_search_space_size(self) -> int:
        """Estimate total hyperparameter search space size."""
        total_size = 0
        for algorithm in self.candidate_algorithms:
            algorithm_size = 1
            for hp in algorithm.hyperparameter_space:
                if hp.parameter_type == "categorical":
                    algorithm_size *= len(hp.choices)
                elif hp.parameter_type == "boolean":
                    algorithm_size *= 2
                elif hp.parameter_type in ["int", "float"]:
                    # Estimate based on range and step
                    if hp.step:
                        if hp.parameter_type == "int":
                            algorithm_size *= (hp.max_value - hp.min_value) // hp.step + 1
                        else:
                            algorithm_size *= int((hp.max_value - hp.min_value) / hp.step) + 1
                    else:
                        # Assume 10 values for continuous parameters
                        algorithm_size *= 10
            total_size += algorithm_size
        return total_size

    def estimate_runtime(self) -> timedelta:
        """Estimate total runtime based on configuration."""
        # Simple estimation based on trials and algorithms
        base_time_per_trial = timedelta(minutes=5)  # Base assumption
        
        complexity_multiplier = 1.0
        for algorithm in self.candidate_algorithms:
            if algorithm.computational_complexity == "high":
                complexity_multiplier = max(complexity_multiplier, 2.0)
            elif algorithm.computational_complexity == "low":
                complexity_multiplier = max(complexity_multiplier, 0.5)
        
        estimated_time = base_time_per_trial * self.max_trials * complexity_multiplier
        return min(estimated_time, self.max_time)


@dataclass
class TrialResult:
    """Result of a single AutoML trial."""
    
    trial_id: UUID
    algorithm_name: str
    hyperparameters: dict[str, Any]
    
    # Performance metrics
    performance_metrics: dict[str, float]
    objective_score: float
    
    # Resource usage
    training_time: timedelta
    memory_usage: float
    cpu_usage: float
    
    # Model metadata
    model_size: int | None = None
    model_complexity: dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    cross_validation_scores: list[float] = field(default_factory=list)
    validation_std: float | None = None
    
    # Additional information
    feature_importance: dict[str, float] = field(default_factory=dict)
    feature_count: int | None = None
    
    # Status
    trial_status: str = "completed"
    error_message: str | None = None
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate trial result."""
        if not (0.0 <= self.objective_score <= 1.0):
            raise ValueError("Objective score must be between 0.0 and 1.0")
        if self.memory_usage < 0:
            raise ValueError("Memory usage must be non-negative")

    def get_duration(self) -> timedelta:
        """Get trial duration."""
        end_time = self.completed_at or datetime.utcnow()
        return end_time - self.started_at

    def is_successful(self) -> bool:
        """Check if trial completed successfully."""
        return self.trial_status == "completed" and self.error_message is None


@dataclass
class AutoMLPipeline:
    """AutoML pipeline execution context."""
    
    # Identity
    pipeline_id: UUID
    configuration: AutoMLConfiguration
    
    # Execution state
    status: AutoMLStatus = AutoMLStatus.PENDING
    current_trial: int = 0
    
    # Results
    trial_results: list[TrialResult] = field(default_factory=list)
    best_trial: TrialResult | None = None
    best_model_config: dict[str, Any] = field(default_factory=dict)
    
    # Feature engineering results
    selected_features: list[str] = field(default_factory=list)
    feature_engineering_pipeline: list[dict[str, Any]] = field(default_factory=list)
    
    # Ensemble configuration
    ensemble_config: dict[str, Any] = field(default_factory=dict)
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    
    # Execution tracking
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress_percentage: float = 0.0
    
    # Error handling
    last_error: str | None = None
    failed_trials: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def start_pipeline(self) -> None:
        """Start the AutoML pipeline execution."""
        if self.status != AutoMLStatus.PENDING:
            raise ValueError(f"Cannot start pipeline in {self.status} status")
        
        self.status = AutoMLStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def add_trial_result(self, result: TrialResult) -> None:
        """Add trial result and update best trial if necessary."""
        self.trial_results.append(result)
        self.current_trial += 1
        
        if result.is_successful():
            if (self.best_trial is None or 
                result.objective_score > self.best_trial.objective_score):
                self.best_trial = result
                self.best_model_config = {
                    "algorithm": result.algorithm_name,
                    "hyperparameters": result.hyperparameters,
                    "performance": result.performance_metrics,
                }
        else:
            self.failed_trials += 1
        
        # Update progress
        self.progress_percentage = (self.current_trial / self.configuration.max_trials) * 100
        self.updated_at = datetime.utcnow()

    def complete_pipeline(self) -> None:
        """Mark pipeline as completed."""
        self.status = AutoMLStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.progress_percentage = 100.0

    def fail_pipeline(self, error_message: str) -> None:
        """Mark pipeline as failed."""
        self.status = AutoMLStatus.FAILED
        self.last_error = error_message
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def get_duration(self) -> timedelta | None:
        """Get pipeline execution duration."""
        if not self.started_at:
            return None
        
        end_time = self.completed_at or datetime.utcnow()
        return end_time - self.started_at

    def get_success_rate(self) -> float:
        """Get trial success rate."""
        if not self.trial_results:
            return 0.0
        
        successful_trials = sum(1 for result in self.trial_results if result.is_successful())
        return successful_trials / len(self.trial_results)

    def get_top_trials(self, n: int = 5) -> list[TrialResult]:
        """Get top N trials by objective score."""
        successful_trials = [result for result in self.trial_results if result.is_successful()]
        return sorted(successful_trials, key=lambda x: x.objective_score, reverse=True)[:n]

    def get_pipeline_summary(self) -> dict[str, Any]:
        """Get comprehensive pipeline summary."""
        return {
            "pipeline_id": str(self.pipeline_id),
            "status": self.status.value,
            "progress": self.progress_percentage,
            "trials_completed": len(self.trial_results),
            "trials_successful": len([r for r in self.trial_results if r.is_successful()]),
            "trials_failed": self.failed_trials,
            "success_rate": self.get_success_rate(),
            "best_score": self.best_trial.objective_score if self.best_trial else None,
            "best_algorithm": self.best_trial.algorithm_name if self.best_trial else None,
            "duration": self.get_duration().total_seconds() if self.get_duration() else None,
            "selected_features_count": len(self.selected_features),
            "configuration": {
                "objective": self.configuration.optimization_objective.value,
                "max_trials": self.configuration.max_trials,
                "algorithms": len(self.configuration.candidate_algorithms),
                "search_strategy": self.configuration.search_strategy.value,
            }
        }


@dataclass
class AutoMLQuery:
    """Query for AutoML pipelines and results."""
    
    # Pipeline filters
    pipeline_ids: list[UUID] | None = None
    statuses: list[AutoMLStatus] | None = None
    objectives: list[OptimizationObjective] | None = None
    
    # Performance filters
    min_objective_score: float | None = None
    min_trials: int | None = None
    min_success_rate: float | None = None
    
    # Time filters
    created_after: datetime | None = None
    created_before: datetime | None = None
    completed_after: datetime | None = None
    completed_before: datetime | None = None
    
    # Configuration filters
    algorithms: list[str] | None = None
    search_strategies: list[SearchStrategy] | None = None
    
    # Pagination
    limit: int = 50
    offset: int = 0
    sort_by: str = "created_at"
    sort_order: str = "desc"
    
    def __post_init__(self) -> None:
        """Validate AutoML query."""
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        if self.offset < 0:
            raise ValueError("Offset must be non-negative")
        if self.sort_order not in ["asc", "desc"]:
            raise ValueError("Sort order must be 'asc' or 'desc'")
        
        if (self.min_objective_score is not None and 
            not (0.0 <= self.min_objective_score <= 1.0)):
            raise ValueError("Min objective score must be between 0.0 and 1.0")


@dataclass
class AutoMLSummary:
    """Summary of AutoML pipeline executions."""
    
    total_pipelines: int
    pipelines_by_status: dict[str, int]
    pipelines_by_objective: dict[str, int]
    
    average_success_rate: float
    average_best_score: float
    total_trials_executed: int
    
    top_algorithms: list[dict[str, Any]] = field(default_factory=list)
    performance_trends: dict[str, list[float]] = field(default_factory=dict)
    
    summary_generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self) -> None:
        """Validate AutoML summary."""
        if self.total_pipelines < 0:
            raise ValueError("Total pipelines must be non-negative")
        if not (0.0 <= self.average_success_rate <= 1.0):
            raise ValueError("Average success rate must be between 0.0 and 1.0")
        if not (0.0 <= self.average_best_score <= 1.0):
            raise ValueError("Average best score must be between 0.0 and 1.0")
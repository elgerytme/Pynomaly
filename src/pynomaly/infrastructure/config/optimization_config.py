"""
Optimization Configuration

Configuration classes for hyperparameter optimization including
optimization strategies, resource constraints, and optimization objectives.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class OptimizationStrategy(Enum):
    """Available optimization strategies."""

    OPTUNA = "optuna"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    HYPERBAND = "hyperband"
    BOHB = "bohb"  # Bayesian Optimization and HyperBand
    EVOLUTIONARY = "evolutionary"


class SamplerType(Enum):
    """Optuna sampler types."""

    TPE = "tpe"  # Tree-structured Parzen Estimator
    RANDOM = "random"
    CMAES = "cmaes"  # Covariance Matrix Adaptation Evolution Strategy
    NSGA2 = "nsga2"  # Non-dominated Sorting Genetic Algorithm II
    QUASI_RANDOM = "quasi_random"
    GRID = "grid"


class PrunerType(Enum):
    """Optuna pruner types."""

    MEDIAN = "median"
    PERCENTILE = "percentile"
    SUCCESSIVE_HALVING = "successive_halving"
    HYPERBAND = "hyperband"
    THRESHOLD = "threshold"
    NONE = "none"


class ObjectiveDirection(Enum):
    """Optimization direction."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class ResourceConstraints:
    """Resource constraints for optimization."""

    # Time constraints
    max_time_seconds: int | None = None
    max_time_per_trial: int | None = None
    timeout_patience: int = 30  # seconds to wait for stuck trials

    # Resource limits
    max_memory_mb: int | None = None
    max_cpu_cores: int | None = None
    max_gpu_memory_mb: int | None = None

    # Concurrency
    n_jobs: int = 1
    max_concurrent_trials: int | None = None
    enable_parallel_execution: bool = True

    # Storage limits
    max_storage_mb: int | None = None
    max_log_size_mb: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_time_seconds": self.max_time_seconds,
            "max_time_per_trial": self.max_time_per_trial,
            "timeout_patience": self.timeout_patience,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_cores": self.max_cpu_cores,
            "max_gpu_memory_mb": self.max_gpu_memory_mb,
            "n_jobs": self.n_jobs,
            "max_concurrent_trials": self.max_concurrent_trials,
            "enable_parallel_execution": self.enable_parallel_execution,
            "max_storage_mb": self.max_storage_mb,
            "max_log_size_mb": self.max_log_size_mb,
        }


@dataclass
class SamplerConfig:
    """Configuration for optimization samplers."""

    sampler_type: SamplerType = SamplerType.TPE

    # TPE specific settings
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    gamma: float = 0.25

    # CMA-ES specific settings
    sigma0: float | None = None
    restart_strategy: str | None = None

    # Random/Quasi-random settings
    seed: int | None = None

    # NSGA-II specific settings
    population_size: int = 50
    mutation_prob: float | None = None
    crossover_prob: float = 0.9

    # Advanced settings
    warn_independent_sampling: bool = True
    multivariate: bool = False
    group: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sampler_type": self.sampler_type.value,
            "n_startup_trials": self.n_startup_trials,
            "n_ei_candidates": self.n_ei_candidates,
            "gamma": self.gamma,
            "sigma0": self.sigma0,
            "restart_strategy": self.restart_strategy,
            "seed": self.seed,
            "population_size": self.population_size,
            "mutation_prob": self.mutation_prob,
            "crossover_prob": self.crossover_prob,
            "warn_independent_sampling": self.warn_independent_sampling,
            "multivariate": self.multivariate,
            "group": self.group,
        }


@dataclass
class PrunerConfig:
    """Configuration for optimization pruners."""

    pruner_type: PrunerType = PrunerType.MEDIAN

    # Median pruner settings
    n_startup_trials: int = 5
    n_warmup_steps: int = 0
    interval_steps: int = 1

    # Percentile pruner settings
    percentile: float = 25.0

    # Successive halving settings
    min_resource: int = 1
    reduction_factor: int = 4
    min_early_stopping_rate: int = 0

    # Hyperband settings
    max_resource: int | None = None

    # Threshold pruner settings
    lower: float | None = None
    upper: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pruner_type": self.pruner_type.value,
            "n_startup_trials": self.n_startup_trials,
            "n_warmup_steps": self.n_warmup_steps,
            "interval_steps": self.interval_steps,
            "percentile": self.percentile,
            "min_resource": self.min_resource,
            "reduction_factor": self.reduction_factor,
            "min_early_stopping_rate": self.min_early_stopping_rate,
            "max_resource": self.max_resource,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass
class OptimizationObjective:
    """Configuration for optimization objectives."""

    name: str
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    weight: float = 1.0

    # For multi-objective optimization
    constraints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "direction": self.direction.value,
            "weight": self.weight,
            "constraints": self.constraints,
        }


@dataclass
class StudyConfig:
    """Configuration for optimization studies."""

    study_name: str | None = None
    storage_url: str | None = None  # Database URL for study persistence
    load_if_exists: bool = True

    # Multi-objective settings
    directions: list[ObjectiveDirection] = field(
        default_factory=lambda: [ObjectiveDirection.MAXIMIZE]
    )

    # Study pruning
    enable_study_pruning: bool = False
    study_pruning_threshold: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "study_name": self.study_name,
            "storage_url": self.storage_url,
            "load_if_exists": self.load_if_exists,
            "directions": [d.value for d in self.directions],
            "enable_study_pruning": self.enable_study_pruning,
            "study_pruning_threshold": self.study_pruning_threshold,
        }


@dataclass
class LoggingConfig:
    """Configuration for optimization logging."""

    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Path | None = None

    # Progress reporting
    enable_progress_bar: bool = True
    progress_update_interval: int = 10  # trials

    # Trial logging
    log_trial_details: bool = True
    log_intermediate_values: bool = True
    log_system_attributes: bool = False

    # Performance logging
    log_memory_usage: bool = True
    log_timing: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "log_file": str(self.log_file) if self.log_file else None,
            "enable_progress_bar": self.enable_progress_bar,
            "progress_update_interval": self.progress_update_interval,
            "log_trial_details": self.log_trial_details,
            "log_intermediate_values": self.log_intermediate_values,
            "log_system_attributes": self.log_system_attributes,
            "log_memory_usage": self.log_memory_usage,
            "log_timing": self.log_timing,
        }


@dataclass
class OptimizationConfig:
    """
    Complete configuration for hyperparameter optimization.

    Provides all configuration options for optimization including strategy,
    resource constraints, sampling, pruning, and logging settings.
    """

    # Core optimization settings
    strategy: OptimizationStrategy = OptimizationStrategy.OPTUNA
    n_trials: int = 100
    random_seed: int | None = 42

    # Objective configuration
    objectives: list[OptimizationObjective] = field(
        default_factory=lambda: [
            OptimizationObjective("score", ObjectiveDirection.MAXIMIZE)
        ]
    )

    # Component configurations
    resource_constraints: ResourceConstraints = field(
        default_factory=ResourceConstraints
    )
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)
    pruner_config: PrunerConfig = field(default_factory=PrunerConfig)
    study_config: StudyConfig = field(default_factory=StudyConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 0.001

    # Warm start
    enable_warm_start: bool = True
    warm_start_trials: list[dict[str, Any]] | None = None

    # Callbacks and hooks
    enable_callbacks: bool = True
    callback_config: dict[str, Any] = field(default_factory=dict)

    # Experimental features
    enable_experimental_features: bool = False
    experimental_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")

        if len(self.objectives) == 0:
            raise ValueError("At least one objective must be specified")

        # Adjust sampler settings based on strategy
        if self.strategy == OptimizationStrategy.RANDOM_SEARCH:
            self.sampler_config.sampler_type = SamplerType.RANDOM
        elif self.strategy == OptimizationStrategy.GRID_SEARCH:
            self.sampler_config.sampler_type = SamplerType.GRID

    @property
    def is_multi_objective(self) -> bool:
        """Check if this is multi-objective optimization."""
        return len(self.objectives) > 1

    @property
    def primary_objective(self) -> OptimizationObjective:
        """Get the primary optimization objective."""
        return self.objectives[0]

    def get_directions(self) -> list[ObjectiveDirection]:
        """Get optimization directions for all objectives."""
        return [obj.direction for obj in self.objectives]

    def add_objective(self, objective: OptimizationObjective) -> None:
        """Add an optimization objective."""
        self.objectives.append(objective)

    def set_resource_constraint(self, constraint_type: str, value: Any) -> None:
        """Set a resource constraint."""
        if hasattr(self.resource_constraints, constraint_type):
            setattr(self.resource_constraints, constraint_type, value)
        else:
            raise ValueError(f"Unknown resource constraint: {constraint_type}")

    def enable_distributed_optimization(
        self, n_workers: int, storage_url: str, study_name: str
    ) -> None:
        """Configure for distributed optimization."""
        self.resource_constraints.n_jobs = n_workers
        self.resource_constraints.enable_parallel_execution = True
        self.study_config.storage_url = storage_url
        self.study_config.study_name = study_name
        self.study_config.load_if_exists = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "strategy": self.strategy.value,
            "n_trials": self.n_trials,
            "random_seed": self.random_seed,
            "objectives": [obj.to_dict() for obj in self.objectives],
            "resource_constraints": self.resource_constraints.to_dict(),
            "sampler_config": self.sampler_config.to_dict(),
            "pruner_config": self.pruner_config.to_dict(),
            "study_config": self.study_config.to_dict(),
            "logging_config": self.logging_config.to_dict(),
            "enable_early_stopping": self.enable_early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "enable_warm_start": self.enable_warm_start,
            "warm_start_trials": self.warm_start_trials,
            "enable_callbacks": self.enable_callbacks,
            "callback_config": self.callback_config,
            "enable_experimental_features": self.enable_experimental_features,
            "experimental_config": self.experimental_config,
            "is_multi_objective": self.is_multi_objective,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationConfig":
        """Create from dictionary representation."""
        data = data.copy()

        # Convert enum fields
        if "strategy" in data:
            data["strategy"] = OptimizationStrategy(data["strategy"])

        # Convert objectives
        if "objectives" in data:
            objectives = []
            for obj_data in data["objectives"]:
                obj = OptimizationObjective(
                    name=obj_data["name"],
                    direction=ObjectiveDirection(obj_data["direction"]),
                    weight=obj_data.get("weight", 1.0),
                    constraints=obj_data.get("constraints", []),
                )
                objectives.append(obj)
            data["objectives"] = objectives

        # Convert component configurations
        if "resource_constraints" in data:
            data["resource_constraints"] = ResourceConstraints(
                **data["resource_constraints"]
            )

        if "sampler_config" in data:
            sampler_data = data["sampler_config"].copy()
            if "sampler_type" in sampler_data:
                sampler_data["sampler_type"] = SamplerType(sampler_data["sampler_type"])
            data["sampler_config"] = SamplerConfig(**sampler_data)

        if "pruner_config" in data:
            pruner_data = data["pruner_config"].copy()
            if "pruner_type" in pruner_data:
                pruner_data["pruner_type"] = PrunerType(pruner_data["pruner_type"])
            data["pruner_config"] = PrunerConfig(**pruner_data)

        if "study_config" in data:
            study_data = data["study_config"].copy()
            if "directions" in study_data:
                study_data["directions"] = [
                    ObjectiveDirection(d) for d in study_data["directions"]
                ]
            data["study_config"] = StudyConfig(**study_data)

        if "logging_config" in data:
            logging_data = data["logging_config"].copy()
            if "log_file" in logging_data and logging_data["log_file"]:
                logging_data["log_file"] = Path(logging_data["log_file"])
            data["logging_config"] = LoggingConfig(**logging_data)

        # Remove computed fields
        computed_fields = ["is_multi_objective"]
        for field in computed_fields:
            data.pop(field, None)

        return cls(**data)


# Predefined optimization configurations


def get_quick_optimization_config(n_trials: int = 50) -> OptimizationConfig:
    """Get configuration for quick optimization."""
    return OptimizationConfig(
        strategy=OptimizationStrategy.RANDOM_SEARCH,
        n_trials=n_trials,
        resource_constraints=ResourceConstraints(
            max_time_seconds=300, n_jobs=2  # 5 minutes
        ),
        enable_early_stopping=True,
        early_stopping_patience=10,
    )


def get_thorough_optimization_config(n_trials: int = 500) -> OptimizationConfig:
    """Get configuration for thorough optimization."""
    return OptimizationConfig(
        strategy=OptimizationStrategy.OPTUNA,
        n_trials=n_trials,
        sampler_config=SamplerConfig(sampler_type=SamplerType.TPE, n_startup_trials=20),
        pruner_config=PrunerConfig(
            pruner_type=PrunerType.HYPERBAND, n_startup_trials=10
        ),
        resource_constraints=ResourceConstraints(
            max_time_seconds=7200, n_jobs=4  # 2 hours
        ),
        enable_early_stopping=True,
        early_stopping_patience=50,
    )


def get_production_optimization_config(n_trials: int = 200) -> OptimizationConfig:
    """Get configuration for production optimization."""
    return OptimizationConfig(
        strategy=OptimizationStrategy.OPTUNA,
        n_trials=n_trials,
        sampler_config=SamplerConfig(
            sampler_type=SamplerType.TPE, n_startup_trials=15, multivariate=True
        ),
        pruner_config=PrunerConfig(pruner_type=PrunerType.MEDIAN, n_startup_trials=10),
        resource_constraints=ResourceConstraints(
            max_time_seconds=3600, max_memory_mb=8192, n_jobs=3  # 1 hour
        ),
        logging_config=LoggingConfig(
            enable_logging=True, log_trial_details=True, log_memory_usage=True
        ),
        enable_early_stopping=True,
        early_stopping_patience=25,
    )

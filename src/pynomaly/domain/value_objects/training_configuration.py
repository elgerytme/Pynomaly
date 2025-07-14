"""Training configuration value object for domain layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrainingPriorityLevel(Enum):
    """Training priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass(frozen=True)
class ResourceConstraints:
    """Resource constraints for training - domain value object."""

    max_training_time_seconds: int | None = None
    max_time_per_trial: int | None = None
    max_memory_mb: int | None = None
    max_cpu_cores: int | None = None
    max_gpu_memory_mb: int | None = None
    enable_gpu: bool = False
    n_jobs: int = 1
    max_concurrent_trials: int | None = None


@dataclass(frozen=True)
class AutoMLConfiguration:
    """AutoML configuration - domain value object."""

    enable_automl: bool = True
    optimization_objective: str = "auc"
    max_algorithms: int = 3
    enable_ensemble: bool = True
    ensemble_method: str = "voting"
    algorithm_whitelist: list[str] | None = None
    algorithm_blacklist: list[str] | None = None
    model_selection_strategy: str = "best_single_metric"
    metric_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class ValidationConfiguration:
    """Validation configuration - domain value object."""

    validation_strategy: str = "holdout"
    validation_split: float = 0.2
    cv_folds: int = 5
    cv_strategy: str = "stratified"
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_metric: str = "validation_score"
    early_stopping_threshold: float = 0.001
    enable_data_validation: bool = True
    min_samples_required: int = 100
    max_missing_ratio: float = 0.1


@dataclass(frozen=True)
class TrainingConfiguration:
    """
    Domain value object representing training configuration.
    
    This is a pure domain object with no dependencies on application or
    infrastructure layers, preventing circular dependencies.
    """

    experiment_name: str | None = None
    description: str | None = None
    tags: list[str] = field(default_factory=list)

    automl_config: AutoMLConfiguration = field(default_factory=AutoMLConfiguration)
    validation_config: ValidationConfiguration = field(default_factory=ValidationConfiguration)
    resource_constraints: ResourceConstraints = field(default_factory=ResourceConstraints)

    schedule_cron: str | None = None
    schedule_enabled: bool = False
    performance_monitoring_enabled: bool = True
    retrain_threshold: float = 0.05
    performance_window_days: int = 7
    auto_deploy_best_model: bool = False
    model_versioning_enabled: bool = True
    keep_model_versions: int = 5

    enable_model_explainability: bool = True
    enable_drift_detection: bool = True
    enable_feature_selection: bool = True

    priority: TrainingPriorityLevel = TrainingPriorityLevel.NORMAL

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": list(self.tags),
            "automl_config": {
                "enable_automl": self.automl_config.enable_automl,
                "optimization_objective": self.automl_config.optimization_objective,
                "max_algorithms": self.automl_config.max_algorithms,
                "enable_ensemble": self.automl_config.enable_ensemble,
                "ensemble_method": self.automl_config.ensemble_method,
                "algorithm_whitelist": self.automl_config.algorithm_whitelist,
                "algorithm_blacklist": self.automl_config.algorithm_blacklist,
                "model_selection_strategy": self.automl_config.model_selection_strategy,
                "metric_weights": self.automl_config.metric_weights,
            },
            "validation_config": {
                "validation_strategy": self.validation_config.validation_strategy,
                "validation_split": self.validation_config.validation_split,
                "cv_folds": self.validation_config.cv_folds,
                "cv_strategy": self.validation_config.cv_strategy,
                "enable_early_stopping": self.validation_config.enable_early_stopping,
                "early_stopping_patience": self.validation_config.early_stopping_patience,
                "early_stopping_metric": self.validation_config.early_stopping_metric,
                "early_stopping_threshold": self.validation_config.early_stopping_threshold,
                "enable_data_validation": self.validation_config.enable_data_validation,
                "min_samples_required": self.validation_config.min_samples_required,
                "max_missing_ratio": self.validation_config.max_missing_ratio,
            },
            "resource_constraints": {
                "max_training_time_seconds": self.resource_constraints.max_training_time_seconds,
                "max_time_per_trial": self.resource_constraints.max_time_per_trial,
                "max_memory_mb": self.resource_constraints.max_memory_mb,
                "max_cpu_cores": self.resource_constraints.max_cpu_cores,
                "max_gpu_memory_mb": self.resource_constraints.max_gpu_memory_mb,
                "enable_gpu": self.resource_constraints.enable_gpu,
                "n_jobs": self.resource_constraints.n_jobs,
                "max_concurrent_trials": self.resource_constraints.max_concurrent_trials,
            },
            "schedule_cron": self.schedule_cron,
            "schedule_enabled": self.schedule_enabled,
            "performance_monitoring_enabled": self.performance_monitoring_enabled,
            "retrain_threshold": self.retrain_threshold,
            "performance_window_days": self.performance_window_days,
            "auto_deploy_best_model": self.auto_deploy_best_model,
            "model_versioning_enabled": self.model_versioning_enabled,
            "keep_model_versions": self.keep_model_versions,
            "enable_model_explainability": self.enable_model_explainability,
            "enable_drift_detection": self.enable_drift_detection,
            "enable_feature_selection": self.enable_feature_selection,
            "priority": self.priority.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfiguration:
        """Create from dictionary representation."""
        data = data.copy()

        # Convert priority enum
        if "priority" in data:
            data["priority"] = TrainingPriorityLevel(data["priority"])

        # Convert nested configurations
        if "automl_config" in data:
            automl_data = data["automl_config"]
            data["automl_config"] = AutoMLConfiguration(**automl_data)

        if "validation_config" in data:
            validation_data = data["validation_config"]
            data["validation_config"] = ValidationConfiguration(**validation_data)

        if "resource_constraints" in data:
            resource_data = data["resource_constraints"]
            data["resource_constraints"] = ResourceConstraints(**resource_data)

        return cls(**data)

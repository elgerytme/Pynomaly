#!/usr/bin/env python3
"""
AutoML Pipeline Configuration Classes
Configuration and data structures for AutoML pipeline orchestration
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator


class PipelineStage(Enum):
    """AutoML pipeline stages"""

    INITIALIZATION = "initialization"
    DATA_VALIDATION = "data_validation"
    DATA_PROFILING = "data_profiling"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    ENSEMBLE_CREATION = "ensemble_creation"
    MODEL_VALIDATION = "model_validation"
    DEPLOYMENT_PREPARATION = "deployment_preparation"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineMode(Enum):
    """AutoML pipeline execution modes"""

    FAST = "fast"  # Quick baseline models
    BALANCED = "balanced"  # Balance between speed and performance
    THOROUGH = "thorough"  # Comprehensive search
    CUSTOM = "custom"  # User-defined configuration


@dataclass
class PipelineConfig:
    """Configuration for AutoML pipeline execution"""

    # Execution settings
    mode: PipelineMode = PipelineMode.BALANCED
    max_time_minutes: int = 30
    max_models_to_evaluate: int = 10
    cross_validation_folds: int = 5
    random_state: int = 42

    # Model selection
    algorithms: list[str] = field(
        default_factory=lambda: [
            "isolation_forest",
            "random_forest",
            "one_class_svm",
            "local_outlier_factor",
        ]
    )
    ensemble_methods: list[str] = field(
        default_factory=lambda: ["voting", "stacking", "bagging"]
    )

    # Data processing
    contamination_rate: float = 0.1
    max_features: int = 1000
    feature_selection_enabled: bool = True
    auto_feature_engineering: bool = True

    # Performance criteria
    min_model_performance: float = 0.7
    optimization_time_budget_minutes: int = 15

    # Advanced options
    enable_ensemble: bool = True
    enable_meta_learning: bool = False
    enable_early_stopping: bool = True
    enable_model_compression: bool = False

    # Resource limits
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 4

    # Output configuration
    artifacts_dir: Path = field(default_factory=lambda: Path("./automl_artifacts"))
    save_intermediate_results: bool = True
    generate_model_reports: bool = True

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Validate configuration
        if self.max_time_minutes <= 0:
            raise ValueError("max_time_minutes must be positive")

        if self.max_models_to_evaluate <= 0:
            raise ValueError("max_models_to_evaluate must be positive")

        if not 0 < self.contamination_rate < 1:
            raise ValueError("contamination_rate must be between 0 and 1")

        if self.min_model_performance < 0 or self.min_model_performance > 1:
            raise ValueError("min_model_performance must be between 0 and 1")

    @classmethod
    def for_mode(cls, mode: PipelineMode, **kwargs) -> "PipelineConfig":
        """Create configuration optimized for specific execution mode."""
        base_configs = {
            PipelineMode.FAST: {
                "max_time_minutes": 10,
                "max_models_to_evaluate": 3,
                "cross_validation_folds": 3,
                "optimization_time_budget_minutes": 5,
                "enable_ensemble": False,
                "auto_feature_engineering": False,
                "algorithms": ["isolation_forest", "random_forest"],
            },
            PipelineMode.BALANCED: {
                "max_time_minutes": 30,
                "max_models_to_evaluate": 5,
                "cross_validation_folds": 5,
                "optimization_time_budget_minutes": 15,
                "enable_ensemble": True,
                "auto_feature_engineering": True,
            },
            PipelineMode.THOROUGH: {
                "max_time_minutes": 120,
                "max_models_to_evaluate": 20,
                "cross_validation_folds": 10,
                "optimization_time_budget_minutes": 60,
                "enable_ensemble": True,
                "enable_meta_learning": True,
                "auto_feature_engineering": True,
            },
        }

        if mode in base_configs:
            config_dict = base_configs[mode].copy()
            config_dict.update(kwargs)
            return cls(mode=mode, **config_dict)
        else:
            return cls(mode=mode, **kwargs)


@dataclass
class PipelineStageResult:
    """Result of a single pipeline stage"""

    stage: PipelineStage
    success: bool
    duration_seconds: float
    outputs: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage": self.stage.value,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "outputs": self.outputs,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PipelineResult:
    """Complete AutoML pipeline execution result"""

    pipeline_id: str
    config: PipelineConfig
    start_time: datetime
    end_time: datetime | None = None
    current_stage: PipelineStage = PipelineStage.INITIALIZATION
    stage_results: dict[PipelineStage, PipelineStageResult] = field(
        default_factory=dict
    )

    # Model artifacts
    best_model: BaseEstimator | None = None
    best_model_params: dict[str, Any] = field(default_factory=dict)
    best_model_performance: dict[str, float] = field(default_factory=dict)
    ensemble_model: BaseEstimator | None = None
    ensemble_performance: dict[str, float] = field(default_factory=dict)

    # Data processing artifacts
    processed_data: pd.DataFrame | None = None
    selected_features: list[str] = field(default_factory=list)
    engineered_features: list[str] = field(default_factory=list)
    data_profile: dict[str, Any] = field(default_factory=dict)

    # Validation results
    cross_validation_scores: list[float] = field(default_factory=list)
    holdout_performance: dict[str, float] = field(default_factory=dict)

    # Deployment artifacts
    model_artifacts_path: str | None = None
    deployment_config: dict[str, Any] = field(default_factory=dict)

    # Analysis and recommendations
    production_readiness_score: float = 0.0
    improvement_recommendations: list[str] = field(default_factory=list)
    resource_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration_seconds(self) -> float:
        """Calculate total pipeline execution time."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def is_successful(self) -> bool:
        """Check if pipeline completed successfully."""
        return (
            self.current_stage == PipelineStage.COMPLETED
            and self.best_model is not None
        )

    @property
    def success_rate(self) -> float:
        """Calculate success rate of completed stages."""
        if not self.stage_results:
            return 0.0

        successful_stages = sum(
            1 for result in self.stage_results.values() if result.success
        )
        return successful_stages / len(self.stage_results)

    def get_stage_result(self, stage: PipelineStage) -> PipelineStageResult | None:
        """Get result for a specific stage."""
        return self.stage_results.get(stage)

    def to_summary_dict(self) -> dict[str, Any]:
        """Convert to summary dictionary for API responses."""
        return {
            "pipeline_id": self.pipeline_id,
            "total_duration_seconds": self.total_duration_seconds,
            "current_stage": self.current_stage.value,
            "is_successful": self.is_successful,
            "success_rate": self.success_rate,
            "best_model_performance": self.best_model_performance,
            "production_readiness_score": self.production_readiness_score,
            "feature_count": len(self.selected_features),
            "recommendations_count": len(self.improvement_recommendations),
        }
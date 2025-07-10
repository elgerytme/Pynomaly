#!/usr/bin/env python3
"""
Pipeline Domain Models - Core data structures for pipeline operations
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from sklearn.base import BaseEstimator

from pynomaly.application.services.automl_service import DatasetProfile


class PipelineStage(Enum):
    """Pipeline execution stages"""
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
    """Pipeline execution modes"""
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    CUSTOM = "custom"


@dataclass
class PipelineConfig:
    """Pipeline configuration settings"""
    mode: PipelineMode = PipelineMode.BALANCED
    train_test_split_ratio: float = 0.8
    validation_split_ratio: float = 0.2
    cross_validation_folds: int = 5
    enable_feature_engineering: bool = True
    max_feature_combinations: int = 100
    feature_selection_threshold: float = 0.95
    optimization_time_budget_minutes: int = 60
    max_models_to_evaluate: int = 20
    early_stopping_patience: int = 10
    enable_ensemble: bool = True
    max_ensemble_size: int = 5
    ensemble_selection_metric: str = "f1_score"
    min_model_performance: float = 0.7
    performance_improvement_threshold: float = 0.01
    max_memory_usage_gb: float = 8.0
    max_cpu_cores: int = 4
    save_intermediate_results: bool = True
    export_model_artifacts: bool = True
    generate_model_report: bool = True
    enable_meta_learning: bool = True
    enable_transfer_learning: bool = False
    enable_neural_architecture_search: bool = False


@dataclass
class PipelineStageResult:
    """Result of a single pipeline stage"""
    stage: PipelineStage
    status: str
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    error_message: str | None = None
    error_details: dict[str, Any] | None = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class PipelineResult:
    """Complete pipeline execution result"""
    pipeline_id: str
    config: PipelineConfig
    start_time: datetime
    end_time: datetime | None = None
    total_duration_seconds: float = 0.0
    final_stage: PipelineStage = PipelineStage.INITIALIZATION
    stage_results: dict[PipelineStage, PipelineStageResult] = field(default_factory=dict)
    dataset_profile: DatasetProfile | None = None
    data_quality_report: dict[str, Any] = field(default_factory=dict)
    best_model: BaseEstimator | None = None
    best_model_params: dict[str, Any] = field(default_factory=dict)
    best_model_performance: dict[str, float] = field(default_factory=dict)
    model_leaderboard: list[dict[str, Any]] = field(default_factory=list)
    ensemble_model: BaseEstimator | None = None
    ensemble_performance: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)
    engineered_features: list[str] = field(default_factory=list)
    cross_validation_scores: list[float] = field(default_factory=list)
    holdout_performance: dict[str, float] = field(default_factory=dict)
    model_artifacts_path: str | None = None
    deployment_config: dict[str, Any] = field(default_factory=dict)
    improvement_recommendations: list[str] = field(default_factory=list)
    production_readiness_score: float = 0.0
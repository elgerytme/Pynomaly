#!/usr/bin/env python3
"""
AutoML Pipeline Orchestrator
Coordinates and orchestrates the complete AutoML pipeline including data preprocessing,
feature engineering, processor optimization, ensemble creation, and deployment
"""

import asyncio
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from monorepo.application.services.advanced_model_optimization_service import (
    AdvancedModelOptimizationService,
    AdvancedOptimizationConfig,
)
from monorepo.application.services.automl_service import DatasetProfile

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """AutoML pipeline stages"""

    INITIALIZATION = "initialization"
    DATA_VALIDATION = "data_validation"
    DATA_PROFILING = "data_profiling"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "processor_selection"
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    ENSEMBLE_CREATION = "ensemble_creation"
    MODEL_VALIDATION = "processor_validation"
    DEPLOYMENT_PREPARATION = "deployment_preparation"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineMode(Enum):
    """AutoML pipeline execution modes"""

    FAST = "fast"  # Quick optimization for rapid prototyping
    BALANCED = "balanced"  # Balanced optimization and quality
    THOROUGH = "thorough"  # Comprehensive optimization for production
    CUSTOM = "custom"  # Custom configuration


@dataclass
class PipelineConfig:
    """AutoML pipeline configuration"""

    # Execution mode
    mode: PipelineMode = PipelineMode.BALANCED

    # Data configuration
    train_test_split_ratio: float = 0.8
    validation_split_ratio: float = 0.2
    cross_validation_folds: int = 5

    # Feature engineering
    enable_feature_engineering: bool = True
    max_feature_combinations: int = 100
    feature_selection_threshold: float = 0.95

    # Processor optimization
    optimization_time_budget_minutes: int = 60
    max_processors_to_evaluate: int = 20
    early_stopping_patience: int = 10

    # Ensemble configuration
    enable_ensemble: bool = True
    max_ensemble_size: int = 5
    ensemble_selection_metric: str = "f1_score"

    # Quality thresholds
    min_processor_performance: float = 0.7
    performance_improvement_threshold: float = 0.01

    # Resource constraints
    max_memory_usage_gb: float = 8.0
    max_cpu_cores: int = 4

    # Output configuration
    save_intermediate_results: bool = True
    export_processor_artifacts: bool = True
    generate_processor_report: bool = True

    # Advanced features
    enable_meta_learning: bool = True
    enable_transfer_learning: bool = False
    enable_neural_architecture_search: bool = False


@dataclass
class PipelineStageResult:
    """Result of a pipeline stage"""

    stage: PipelineStage
    status: str  # "success", "failed", "skipped"
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0

    # Stage-specific outputs
    outputs: dict[str, Any] = field(default_factory=dict)
    measurements: dict[str, float] = field(default_factory=dict)

    # Error information
    error_message: str | None = None
    error_details: dict[str, Any] | None = None

    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class PipelineResult:
    """Complete AutoML pipeline result"""

    pipeline_id: str
    config: PipelineConfig

    # Execution metadata
    start_time: datetime
    end_time: datetime | None = None
    total_duration_seconds: float = 0.0
    final_stage: PipelineStage = PipelineStage.INITIALIZATION

    # Stage results
    stage_results: dict[PipelineStage, PipelineStageResult] = field(
        default_factory=dict
    )

    # Data information
    data_collection_profile: DatasetProfile | None = None
    data_quality_report: dict[str, Any] = field(default_factory=dict)

    # Processor results
    best_processor: BaseEstimator | None = None
    best_processor_params: dict[str, Any] = field(default_factory=dict)
    best_processor_performance: dict[str, float] = field(default_factory=dict)

    # Alternative models
    processor_leaderboard: list[dict[str, Any]] = field(default_factory=list)
    ensemble_processor: BaseEstimator | None = None
    ensemble_performance: dict[str, float] = field(default_factory=dict)

    # Feature engineering results
    feature_importance: dict[str, float] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)
    engineered_features: list[str] = field(default_factory=list)

    # Validation results
    cross_validation_scores: list[float] = field(default_factory=list)
    holdout_performance: dict[str, float] = field(default_factory=dict)

    # Deployment artifacts
    processor_artifacts_path: str | None = None
    deployment_config: dict[str, Any] = field(default_factory=dict)

    # Recommendations
    improvement_recommendations: list[str] = field(default_factory=list)
    production_readiness_score: float = 0.0


class AutoMLPipelineOrchestrator:
    """Orchestrates the complete AutoML pipeline"""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

        # Service dependencies
        self.automl_service = None
        self.optimization_service = None

        # Pipeline state
        self.current_pipeline: PipelineResult | None = None
        self.pipeline_history: list[PipelineResult] = []

        # Resource monitoring
        self.resource_monitor = ResourceMonitor()

        # Artifact storage
        self.artifacts_dir = Path("artifacts/automl_pipelines")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    async def run_complete_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        pipeline_id: str | None = None,
    ) -> PipelineResult:
        """
        Run the complete AutoML pipeline

        Args:
            X: Input features
            y: Target variable (optional for unsupervised)
            pipeline_id: Optional pipeline identifier

        Returns:
            Complete pipeline result
        """

        # Initialize pipeline
        if pipeline_id is None:
            pipeline_id = f"automl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"🚀 Starting AutoML pipeline: {pipeline_id}")

        result = PipelineResult(
            pipeline_id=pipeline_id, config=self.config, start_time=datetime.now()
        )

        self.current_pipeline = result

        try:
            # Stage 1: Initialization
            await self._run_stage(
                result, PipelineStage.INITIALIZATION, self._initialize_pipeline, X, y
            )

            # Stage 2: Data Validation
            await self._run_stage(
                result, PipelineStage.DATA_VALIDATION, self._validate_data, X, y
            )

            # Stage 3: Data Profiling
            await self._run_stage(
                result, PipelineStage.DATA_PROFILING, self._profile_data, X, y
            )

            # Stage 4: Feature Engineering
            if self.config.enable_feature_engineering:
                X_engineered = await self._run_stage(
                    result,
                    PipelineStage.FEATURE_ENGINEERING,
                    self._engineer_features,
                    X,
                    y,
                )
                if X_engineered is not None:
                    X = X_engineered

            # Stage 5: Processor Selection
            await self._run_stage(
                result, PipelineStage.MODEL_SELECTION, self._select_processors, X, y
            )

            # Stage 6: Hyperparameter Optimization
            await self._run_stage(
                result,
                PipelineStage.HYPERPARAMETER_OPTIMIZATION,
                self._optimize_hyperparameters,
                X,
                y,
            )

            # Stage 7: Ensemble Creation
            if self.config.enable_ensemble:
                await self._run_stage(
                    result, PipelineStage.ENSEMBLE_CREATION, self._create_ensemble, X, y
                )

            # Stage 8: Processor Validation
            await self._run_stage(
                result, PipelineStage.MODEL_VALIDATION, self._validate_processors, X, y
            )

            # Stage 9: Deployment Preparation
            await self._run_stage(
                result,
                PipelineStage.DEPLOYMENT_PREPARATION,
                self._prepare_deployment,
                X,
                y,
            )

            # Finalize pipeline
            result.final_stage = PipelineStage.COMPLETED
            result.end_time = datetime.now()
            result.total_duration_seconds = (
                result.end_time - result.start_time
            ).total_seconds()

            # Generate recommendations
            await self._generate_recommendations(result)

            # Save results
            if self.config.save_intermediate_results:
                await self._save_pipeline_results(result)

            logger.info(
                f"✅ AutoML pipeline completed: {pipeline_id} ({result.total_duration_seconds:.2f}s)"
            )

            # Add to history
            self.pipeline_history.append(result)

            return result

        except Exception as e:
            logger.error(f"❌ AutoML pipeline failed: {e}")
            result.final_stage = PipelineStage.FAILED
            result.end_time = datetime.now()
            result.total_duration_seconds = (
                result.end_time - result.start_time
            ).total_seconds()

            # Add error information to last stage
            if result.stage_results:
                last_stage = max(result.stage_results.keys(), key=lambda x: x.value)
                result.stage_results[last_stage].error_message = str(e)

            raise

    async def _run_stage(
        self, result: PipelineResult, stage: PipelineStage, stage_func, *args, **kwargs
    ) -> Any:
        """Run a single pipeline stage with monitoring and error handling"""

        logger.info(f"🔧 Running stage: {stage.value}")

        stage_result = PipelineStageResult(
            stage=stage, status="running", start_time=datetime.now()
        )

        result.stage_results[stage] = stage_result

        try:
            # Monitor resources before stage
            memory_before = self.resource_monitor.get_memory_usage()

            # Run stage function
            stage_output = await stage_func(*args, **kwargs)

            # Update stage result
            stage_result.end_time = datetime.now()
            stage_result.duration_seconds = (
                stage_result.end_time - stage_result.start_time
            ).total_seconds()
            stage_result.status = "success"
            stage_result.memory_usage_mb = (
                self.resource_monitor.get_memory_usage() - memory_before
            )

            if isinstance(stage_output, dict):
                stage_result.outputs.update(stage_output)
            elif stage_output is not None:
                stage_result.outputs["result"] = stage_output

            logger.info(
                f"✅ Stage completed: {stage.value} ({stage_result.duration_seconds:.2f}s)"
            )

            return stage_output

        except Exception as e:
            stage_result.end_time = datetime.now()
            stage_result.duration_seconds = (
                stage_result.end_time - stage_result.start_time
            ).total_seconds()
            stage_result.status = "failed"
            stage_result.error_message = str(e)

            logger.error(f"❌ Stage failed: {stage.value} - {e}")
            raise

    async def _initialize_pipeline(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Initialize the AutoML pipeline"""

        # Initialize services based on configuration
        if self.config.mode == PipelineMode.FAST:
            optimization_config = AdvancedOptimizationConfig(
                n_trials=20, timeout_seconds=300, cv_folds=3
            )
        elif self.config.mode == PipelineMode.THOROUGH:
            optimization_config = AdvancedOptimizationConfig(
                n_trials=500, timeout_seconds=3600, cv_folds=10
            )
        else:  # BALANCED
            optimization_config = AdvancedOptimizationConfig(
                n_trials=100, timeout_seconds=1800, cv_folds=5
            )

        self.optimization_service = AdvancedModelOptimizationService(
            optimization_config
        )

        # Initialize AutoML service with configuration
        try:
            from monorepo.application.services.automl_service import AutoMLService

            self.automl_service = AutoMLService()
            logger.info("AutoML service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize AutoML service: {e}")
            # Create minimal fallback service
            self.automl_service = type(
                "MockAutoMLService",
                (),
                {
                    "fit": lambda self, X, y=None: None,
                    "predict": lambda self, X: np.zeros(len(X)),
                    "get_best_processor": lambda self: None,
                    "is_fitted": False,
                },
            )()

        return {
            "services_initialized": True,
            "mode": self.config.mode.value,
            "data_shape": X.shape,
            "has_target": y is not None,
        }

    async def _validate_data(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Validate input data quality and characteristics"""

        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        # Basic validation
        if X.empty:
            validation_results["valid"] = False
            validation_results["issues"].append("DataCollection is empty")
            return validation_results

        # Check for missing values
        missing_ratio = X.isnull().sum().sum() / (len(X) * len(X.columns))
        validation_results["statistics"]["missing_ratio"] = missing_ratio

        if missing_ratio > 0.5:
            validation_results["issues"].append(
                f"High missing value ratio: {missing_ratio:.2%}"
            )
        elif missing_ratio > 0.2:
            validation_results["warnings"].append(
                f"Moderate missing values: {missing_ratio:.2%}"
            )

        # Check data types
        numeric_ratio = len(X.select_dtypes(include=[np.number]).columns) / len(
            X.columns
        )
        validation_results["statistics"]["numeric_ratio"] = numeric_ratio

        if numeric_ratio < 0.5:
            validation_results["warnings"].append(
                "Low numeric feature ratio, may need encoding"
            )

        # Check sample size
        if len(X) < 100:
            validation_results["warnings"].append(
                "Small data_collection size, consider collecting more data"
            )

        # Target validation
        if y is not None:
            target_stats = {
                "unique_values": len(y.unique()),
                "missing_ratio": y.isnull().sum() / len(y),
            }

            if target_stats["missing_ratio"] > 0:
                validation_results["issues"].append(
                    f"Missing target values: {target_stats['missing_ratio']:.2%}"
                )

            if target_stats["unique_values"] == 1:
                validation_results["issues"].append("Target has only one unique value")

            validation_results["statistics"]["target"] = target_stats

        # Memory usage check
        memory_usage = X.memory_usage(deep=True).sum() / (1024**3)  # GB
        validation_results["statistics"]["memory_usage_gb"] = memory_usage

        if memory_usage > self.config.max_memory_usage_gb:
            validation_results["warnings"].append(
                f"High memory usage: {memory_usage:.2f}GB"
            )

        return validation_results

    async def _profile_data(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Profile the data_collection to understand its characteristics"""

        profile = {
            "basic_stats": {
                "n_samples": len(X),
                "n_features": len(X.columns),
                "memory_usage_mb": X.memory_usage(deep=True).sum() / (1024**2),
            },
            "feature_analysis": {},
            "data_quality": {},
            "complexity_score": 0.0,
        }

        # Feature analysis
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        profile["feature_analysis"] = {
            "numeric_features": len(numeric_features),
            "categorical_features": len(categorical_features),
            "feature_types": {
                col: "numeric" if col in numeric_features else "categorical"
                for col in X.columns
            },
        }

        # Data quality analysis
        profile["data_quality"] = {
            "missing_values": X.isnull().sum().to_dict(),
            "duplicate_rows": X.duplicated().sum(),
            "constant_features": [col for col in X.columns if X[col].nunique() <= 1],
        }

        # Complexity analysis
        complexity_factors = []

        # Size factor
        size_factor = min(len(X) / 10000, 1.0)
        complexity_factors.append(size_factor * 0.3)

        # Dimensionality factor
        dim_factor = min(len(X.columns) / 1000, 1.0)
        complexity_factors.append(dim_factor * 0.3)

        # Sparsity factor
        if numeric_features:
            sparsity = (X[numeric_features] == 0).sum().sum() / (
                len(X) * len(numeric_features)
            )
            complexity_factors.append(sparsity * 0.2)

        # Missing data factor
        missing_factor = X.isnull().sum().sum() / (len(X) * len(X.columns))
        complexity_factors.append(missing_factor * 0.2)

        profile["complexity_score"] = sum(complexity_factors)

        # Store in pipeline result
        if self.current_pipeline:
            # Create DatasetProfile object for compatibility
            self.current_pipeline.data_collection_profile = DatasetProfile(
                n_samples=len(X),
                n_features=len(X.columns),
                contamination_estimate=0.1,  # Default
                feature_types=profile["feature_analysis"]["feature_types"],
                missing_values_ratio=missing_factor,
                categorical_features=categorical_features,
                numerical_features=numeric_features,
                time_series_features=[],
                sparsity_ratio=sparsity if numeric_features else 0.0,
                dimensionality_ratio=len(X.columns) / len(X),
                data_collection_size_mb=profile["basic_stats"]["memory_usage_mb"],
            )

        return profile

    async def _engineer_features(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> pd.DataFrame:
        """Perform automated feature engineering"""

        logger.info("🔨 Performing feature engineering")

        X_engineered = X.copy()
        original_features = list(X.columns)

        # Basic feature engineering
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            # Statistical features
            X_engineered["feature_mean"] = X[numeric_cols].mean(axis=1)
            X_engineered["feature_std"] = X[numeric_cols].std(axis=1)
            X_engineered["feature_max"] = X[numeric_cols].max(axis=1)
            X_engineered["feature_min"] = X[numeric_cols].min(axis=1)

            # Interaction features (limited to avoid explosion)
            if len(numeric_cols) <= 5:
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i + 1 :]:
                        X_engineered[f"{col1}_x_{col2}"] = X[col1] * X[col2]

        # Feature selection based on variance
        from sklearn.feature_selection import VarianceThreshold

        selector = VarianceThreshold(threshold=0.01)

        # Apply to numeric features only
        if len(numeric_cols) > 0:
            selected_numeric = X_engineered[numeric_cols].columns[
                selector.fit(X_engineered[numeric_cols]).get_support()
            ]

            # Keep selected numeric + all non-numeric
            non_numeric_cols = X_engineered.select_dtypes(exclude=[np.number]).columns
            selected_features = list(selected_numeric) + list(non_numeric_cols)
            X_engineered = X_engineered[selected_features]

        # Store feature information
        engineered_features = [
            col for col in X_engineered.columns if col not in original_features
        ]

        if self.current_pipeline:
            self.current_pipeline.selected_features = list(X_engineered.columns)
            self.current_pipeline.engineered_features = engineered_features

        logger.info(
            f"Feature engineering: {len(X.columns)} -> {len(X_engineered.columns)} features"
        )
        logger.info(f"Added {len(engineered_features)} engineered features")

        return X_engineered

    async def _select_processors(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Select candidate models for optimization"""

        # Basic processor selection based on data characteristics
        processor_candidates = []

        n_samples, n_features = X.shape

        # Size-based selection
        if n_samples < 1000:
            processor_candidates.extend(["one_class_svm", "local_outlier_factor"])
        else:
            processor_candidates.extend(["isolation_forest", "random_forest"])

        # Dimensionality-based selection
        if n_features > 50:
            processor_candidates.append("random_forest")

        # Always include some basic models
        processor_candidates.extend(["isolation_forest", "random_forest"])

        # Remove duplicates and limit to max models
        processor_candidates = list(dict.fromkeys(processor_candidates))  # Remove duplicates
        processor_candidates = processor_candidates[: self.config.max_processors_to_evaluate]

        return {
            "selected_processors": processor_candidates,
            "selection_rationale": {
                "data_size": f"{n_samples} samples, {n_features} features",
                "recommended_for_size": "SVM for small data, forests for large data",
            },
        }

    async def _optimize_hyperparameters(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Optimize hyperparameters for selected models"""

        logger.info("🎯 Optimizing hyperparameters")

        if not self.optimization_service:
            raise ValueError("Optimization service not initialized")

        # Get selected models from previous stage
        selected_processors = self.current_pipeline.stage_results[
            PipelineStage.MODEL_SELECTION
        ].outputs.get("selected_processors", ["isolation_forest", "random_forest"])

        # Run optimization
        optimization_result = await self.optimization_service.optimize_processor_advanced(
            X, y, selected_processors
        )

        # Store best processor
        if self.current_pipeline:
            self.current_pipeline.best_processor = optimization_result.best_processor
            self.current_pipeline.best_processor_params = optimization_result.best_params
            self.current_pipeline.best_processor_performance = (
                optimization_result.best_scores
            )

        return {
            "optimization_result": optimization_result,
            "best_processor_type": optimization_result.best_params.get("processor_type"),
            "best_scores": optimization_result.best_scores,
            "optimization_time": optimization_result.optimization_time,
        }

    async def _create_ensemble(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Create ensemble processor from optimized models"""

        logger.info("🎼 Creating ensemble processor")

        # Get optimization result
        optimization_result = self.current_pipeline.stage_results[
            PipelineStage.HYPERPARAMETER_OPTIMIZATION
        ].outputs.get("optimization_result")

        if not optimization_result or not optimization_result.ensemble_processor:
            return {"ensemble_created": False, "reason": "No ensemble processor available"}

        # Evaluate ensemble
        ensemble_processor = optimization_result.ensemble_processor

        if y is not None:
            ensemble_processor.fit(X, y)
            y_pred = ensemble_processor.predict(X)

            from sklearn.metrics import accuracy_score, f1_score

            ensemble_performance = {
                "accuracy": accuracy_score(y, y_pred),
                "f1_score": f1_score(y, y_pred, average="weighted"),
            }
        else:
            ensemble_processor.fit(X)
            ensemble_performance = {"fit_successful": True}

        # Store ensemble
        if self.current_pipeline:
            self.current_pipeline.ensemble_processor = ensemble_processor
            self.current_pipeline.ensemble_performance = ensemble_performance

        return {
            "ensemble_created": True,
            "ensemble_performance": ensemble_performance,
            "ensemble_type": "stacking",  # Default
        }

    async def _validate_processors(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Validate final models using cross-validation and holdout sets"""

        logger.info("✅ Validating models")

        validation_results = {}

        # Split data for validation
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            y_train, y_test = None, None

        # Validate best processor
        best_processor = self.current_pipeline.best_processor
        if best_processor and y is not None:
            best_processor.fit(X_train, y_train)
            y_pred = best_processor.predict(X_test)

            from sklearn.metrics import classification_report, f1_score

            validation_results["best_processor"] = {
                "holdout_f1": f1_score(y_test, y_pred, average="weighted"),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
            }

        # Validate ensemble if available
        ensemble_processor = self.current_pipeline.ensemble_processor
        if ensemble_processor and y is not None:
            ensemble_processor.fit(X_train, y_train)
            y_pred_ensemble = ensemble_processor.predict(X_test)

            validation_results["ensemble"] = {
                "holdout_f1": f1_score(y_test, y_pred_ensemble, average="weighted"),
                "classification_report": classification_report(
                    y_test, y_pred_ensemble, output_dict=True
                ),
            }

        # Cross-validation
        if best_processor and y is not None:
            from sklearn.model_selection import cross_val_score

            cv_scores = cross_val_score(
                best_processor,
                X,
                y,
                cv=self.config.cross_validation_folds,
                scoring="f1_weighted",
            )

            validation_results["cross_validation"] = {
                "cv_scores": cv_scores.tolist(),
                "cv_mean": float(cv_scores.mean()),
                "cv_std": float(cv_scores.std()),
            }

            if self.current_pipeline:
                self.current_pipeline.cross_validation_scores = cv_scores.tolist()

        return validation_results

    async def _prepare_deployment(
        self, X: pd.DataFrame, y: pd.Series | None
    ) -> dict[str, Any]:
        """Prepare models and artifacts for deployment"""

        logger.info("📦 Preparing deployment artifacts")

        deployment_config = {
            "processor_ready": False,
            "artifacts_created": False,
            "deployment_config": {},
        }

        # Save best processor
        if self.current_pipeline.best_processor:
            processor_path = (
                self.artifacts_dir
                / f"{self.current_pipeline.pipeline_id}_best_processor.pkl"
            )

            with open(processor_path, "wb") as f:
                pickle.dump(self.current_pipeline.best_processor, f)

            deployment_config["best_processor_path"] = str(processor_path)
            deployment_config["processor_ready"] = True

        # Save ensemble processor
        if self.current_pipeline.ensemble_processor:
            ensemble_path = (
                self.artifacts_dir
                / f"{self.current_pipeline.pipeline_id}_ensemble_processor.pkl"
            )

            with open(ensemble_path, "wb") as f:
                pickle.dump(self.current_pipeline.ensemble_processor, f)

            deployment_config["ensemble_processor_path"] = str(ensemble_path)

        # Create deployment configuration
        deployment_config["deployment_config"] = {
            "processor_type": self.current_pipeline.best_processor_params.get(
                "processor_type", "unknown"
            ),
            "feature_names": list(X.columns),
            "preprocessing_required": True,
            "expected_input_shape": X.shape,
            "performance_threshold": self.config.min_processor_performance,
        }

        deployment_config["artifacts_created"] = True

        if self.current_pipeline:
            self.current_pipeline.processor_artifacts_path = str(
                self.artifacts_dir / self.current_pipeline.pipeline_id
            )
            self.current_pipeline.deployment_config = deployment_config[
                "deployment_config"
            ]

        return deployment_config

    async def _generate_recommendations(self, result: PipelineResult):
        """Generate improvement recommendations based on pipeline results"""

        recommendations = []

        # Performance-based recommendations
        best_score = (
            max(result.best_processor_performance.values())
            if result.best_processor_performance
            else 0.0
        )

        if best_score < 0.7:
            recommendations.append(
                "Consider collecting more training data or improving data quality"
            )

        if best_score < 0.8:
            recommendations.append(
                "Try advanced feature engineering or different algorithms"
            )

        # Data quality recommendations
        data_quality = result.stage_results.get(
            PipelineStage.DATA_VALIDATION, {}
        ).outputs
        if data_quality:
            if data_quality.get("statistics", {}).get("missing_ratio", 0) > 0.2:
                recommendations.append(
                    "Address missing values with better imputation strategies"
                )

            if (
                len(
                    data_quality.get("statistics", {})
                    .get("target", {})
                    .get("unique_values", [])
                )
                < 5
            ):
                recommendations.append(
                    "Consider collecting more diverse examples in target classes"
                )

        # Ensemble recommendations
        if result.ensemble_processor and result.ensemble_performance:
            ensemble_score = (
                max(result.ensemble_performance.values())
                if result.ensemble_performance
                else 0.0
            )
            best_single_score = (
                max(result.best_processor_performance.values())
                if result.best_processor_performance
                else 0.0
            )

            if ensemble_score > best_single_score + 0.05:
                recommendations.append(
                    "Ensemble processor shows significant improvement - consider using for production"
                )

        # Resource optimization recommendations
        total_time = result.total_duration_seconds
        if total_time > self.config.optimization_time_budget_minutes * 60:
            recommendations.append(
                "Consider reducing optimization time budget or using faster algorithms"
            )

        # Production readiness score
        readiness_factors = []

        # Performance factor
        readiness_factors.append(min(best_score, 1.0) * 0.4)

        # Data quality factor
        data_quality_score = 1.0 - data_quality.get("statistics", {}).get(
            "missing_ratio", 0
        )
        readiness_factors.append(data_quality_score * 0.3)

        # Validation factor
        cv_scores = result.cross_validation_scores
        if cv_scores:
            cv_stability = (
                1.0 - (np.std(cv_scores) / np.mean(cv_scores))
                if np.mean(cv_scores) > 0
                else 0.0
            )
            readiness_factors.append(cv_stability * 0.3)

        result.production_readiness_score = sum(readiness_factors)
        result.improvement_recommendations = recommendations

    async def _save_pipeline_results(self, result: PipelineResult):
        """Save pipeline results to disk"""

        results_path = self.artifacts_dir / f"{result.pipeline_id}_results.json"

        # Convert result to serializable format
        serializable_result = {
            "pipeline_id": result.pipeline_id,
            "config": result.config.__dict__,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "total_duration_seconds": result.total_duration_seconds,
            "final_stage": result.final_stage.value,
            "best_processor_params": result.best_processor_params,
            "best_processor_performance": result.best_processor_performance,
            "ensemble_performance": result.ensemble_performance,
            "cross_validation_scores": result.cross_validation_scores,
            "improvement_recommendations": result.improvement_recommendations,
            "production_readiness_score": result.production_readiness_score,
            "stage_results": {
                stage.value: {
                    "status": stage_result.status,
                    "duration_seconds": stage_result.duration_seconds,
                    "outputs": stage_result.outputs,
                    "error_message": stage_result.error_message,
                }
                for stage, stage_result in result.stage_results.items()
            },
        }

        with open(results_path, "w") as f:
            json.dump(serializable_result, f, indent=2)

        logger.info(f"💾 Pipeline results saved: {results_path}")

    def get_pipeline_summary(self, pipeline_id: str) -> dict[str, Any] | None:
        """Get summary of a completed pipeline"""

        pipeline = next(
            (p for p in self.pipeline_history if p.pipeline_id == pipeline_id), None
        )

        if not pipeline:
            return None

        return {
            "pipeline_id": pipeline.pipeline_id,
            "status": pipeline.final_stage.value,
            "duration_seconds": pipeline.total_duration_seconds,
            "best_processor_performance": pipeline.best_processor_performance,
            "production_readiness_score": pipeline.production_readiness_score,
            "stage_count": len(pipeline.stage_results),
            "successful_stages": len(
                [s for s in pipeline.stage_results.values() if s.status == "success"]
            ),
            "recommendations": pipeline.improvement_recommendations,
        }


class ResourceMonitor:
    """Monitor resource usage during pipeline execution"""

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil

            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0


# Example usage
async def main():
    """Example usage of AutoML Pipeline Orchestrator"""

    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(1000, 20), columns=[f"feature_{i}" for i in range(20)]
    )
    y = pd.Series(np.random.choice([0, 1], size=1000, p=[0.7, 0.3]))

    # Configure pipeline
    config = PipelineConfig(
        mode=PipelineMode.BALANCED,
        optimization_time_budget_minutes=10,
        enable_ensemble=True,
        enable_meta_learning=True,
    )

    # Create orchestrator
    orchestrator = AutoMLPipelineOrchestrator(config)

    # Run pipeline
    result = await orchestrator.run_complete_pipeline(X, y)

    print(f"Pipeline completed: {result.pipeline_id}")
    print(f"Best processor performance: {result.best_processor_performance}")
    print(f"Production readiness: {result.production_readiness_score:.2f}")
    print(f"Recommendations: {result.improvement_recommendations}")


if __name__ == "__main__":
    asyncio.run(main())

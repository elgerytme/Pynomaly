"""Enhanced AutoML service with advanced hyperparameter optimization.

This service extends the basic AutoML capabilities with state-of-the-art optimization techniques
including multi-objective optimization, meta-learning, and advanced acquisition functions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import AutoMLError

# Import base AutoML service
from .automl_service import AlgorithmConfig, AutoMLResult, AutoMLService, DatasetProfile

# Import advanced optimization components
try:
    from pynomaly.infrastructure.automl import (
        AcquisitionFunction,
        AdvancedHyperparameterOptimizer,
        AdvancedOptimizationConfig,
        EarlyStoppingConfig,
        MetaLearningConfig,
        MetaLearningStrategy,
        OptimizationConstraint,
        OptimizationResult,
        OptimizationStrategy,
    )
    from pynomaly.infrastructure.automl import (
        OptimizationObjective as AdvancedObjective,
    )

    ADVANCED_OPTIMIZER_AVAILABLE = True
except ImportError:
    ADVANCED_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAutoMLConfig:
    """Enhanced configuration for AutoML with advanced features."""

    # Basic optimization settings
    max_optimization_time: int = 3600
    n_trials: int = 100
    cv_folds: int = 3
    random_state: int = 42

    # Advanced optimization strategy
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT

    # Multi-objective optimization
    enable_multi_objective: bool = False
    objectives: list[str] = field(default_factory=lambda: ["auc", "training_time"])
    objective_weights: dict[str, float] = field(
        default_factory=lambda: {"auc": 0.8, "training_time": 0.2}
    )

    # Meta-learning settings
    enable_meta_learning: bool = True
    meta_learning_strategy: MetaLearningStrategy = MetaLearningStrategy.SIMILAR_DATASETS
    similarity_threshold: float = 0.8
    warm_start_trials: int = 10

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Parallelization
    enable_parallel: bool = True
    n_jobs: int = -1

    # Advanced features
    enable_ensemble_optimization: bool = True
    enable_feature_selection: bool = True
    enable_preprocessing_optimization: bool = True

    # Constraints
    max_training_time_per_trial: float = 300.0  # seconds
    max_memory_usage_mb: float = 4096.0
    min_performance_threshold: float = 0.6


@dataclass
class EnhancedAutoMLResult(AutoMLResult):
    """Enhanced AutoML result with additional metrics and insights."""

    # Advanced optimization metrics
    optimization_strategy_used: str = ""
    meta_learning_effectiveness: float | None = None
    warm_start_contribution: float | None = None

    # Multi-objective results
    pareto_front: list[dict[str, Any]] | None = None
    objective_trade_offs: dict[str, float] | None = None

    # Optimization insights
    exploration_score: float = 0.0
    exploitation_score: float = 0.0
    convergence_stability: float = 0.0
    parameter_sensitivity: dict[str, float] = field(default_factory=dict)

    # Performance analysis
    training_time_breakdown: dict[str, float] = field(default_factory=dict)
    memory_usage_analysis: dict[str, float] = field(default_factory=dict)

    # Recommendations
    optimization_recommendations: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


class EnhancedAutoMLService(AutoMLService):
    """Enhanced AutoML service with advanced optimization capabilities."""

    def __init__(
        self,
        detector_repository,
        dataset_repository,
        adapter_registry,
        config: EnhancedAutoMLConfig | None = None,
        storage_path: Path | None = None,
    ):
        """Initialize enhanced AutoML service.

        Args:
            detector_repository: Repository for detector storage
            dataset_repository: Repository for dataset access
            adapter_registry: Registry for algorithm adapters
            config: Enhanced AutoML configuration
            storage_path: Path for storing optimization artifacts
        """
        # Initialize base service
        super().__init__(
            detector_repository=detector_repository,
            dataset_repository=dataset_repository,
            adapter_registry=adapter_registry,
            max_optimization_time=config.max_optimization_time if config else 3600,
            n_trials=config.n_trials if config else 100,
            cv_folds=config.cv_folds if config else 3,
            random_state=config.random_state if config else 42,
        )

        self.config = config or EnhancedAutoMLConfig()
        self.storage_path = storage_path or Path("./automl_storage")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize advanced optimizer if available
        self.advanced_optimizer = None
        if ADVANCED_OPTIMIZER_AVAILABLE:
            self._initialize_advanced_optimizer()
        else:
            logger.warning(
                "Advanced optimizer not available. Using basic optimization."
            )

        # Optimization history for meta-learning
        self.optimization_history = []

    def _initialize_advanced_optimizer(self) -> None:
        """Initialize the advanced hyperparameter optimizer."""
        # Configure early stopping
        early_stopping = None
        if self.config.enable_early_stopping:
            early_stopping = EarlyStoppingConfig(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                restore_best_weights=True,
                monitor_metric="validation_score",
                mode="max",
            )

        # Configure meta-learning
        meta_learning = None
        if self.config.enable_meta_learning:
            meta_learning = MetaLearningConfig(
                strategy=self.config.meta_learning_strategy,
                similarity_threshold=self.config.similarity_threshold,
                max_historical_experiments=100,
                use_dataset_features=True,
                use_algorithm_features=True,
                warm_start_trials=self.config.warm_start_trials,
            )

        # Configure objectives for multi-objective optimization
        objectives = []
        if self.config.enable_multi_objective:
            for obj_name in self.config.objectives:
                weight = self.config.objective_weights.get(obj_name, 1.0)
                direction = (
                    "maximize"
                    if obj_name in ["auc", "precision", "recall", "f1_score"]
                    else "minimize"
                )

                objectives.append(
                    AdvancedObjective(name=obj_name, direction=direction, weight=weight)
                )

        # Configure constraints
        constraints = [
            OptimizationConstraint(
                name="training_time",
                constraint_type="upper_bound",
                value=self.config.max_training_time_per_trial,
            ),
            OptimizationConstraint(
                name="memory_usage",
                constraint_type="upper_bound",
                value=self.config.max_memory_usage_mb,
            ),
            OptimizationConstraint(
                name="performance",
                constraint_type="lower_bound",
                value=self.config.min_performance_threshold,
            ),
        ]

        # Create advanced optimization config
        optimization_config = AdvancedOptimizationConfig(
            strategy=self.config.optimization_strategy,
            acquisition_function=self.config.acquisition_function,
            n_trials=self.config.n_trials,
            timeout=self.config.max_optimization_time,
            n_jobs=self.config.n_jobs if self.config.enable_parallel else 1,
            random_state=self.config.random_state,
            objectives=objectives,
            constraints=constraints,
            early_stopping=early_stopping,
            meta_learning=meta_learning,
            enable_pruning=True,
            enable_parallel=self.config.enable_parallel,
            save_intermediate_results=True,
        )

        # Initialize optimizer
        self.advanced_optimizer = AdvancedHyperparameterOptimizer(
            config=optimization_config, storage_path=self.storage_path
        )

    async def advanced_optimize_hyperparameters(
        self, dataset_id: str, algorithm: str, objectives: list[str] | None = None
    ) -> EnhancedAutoMLResult:
        """Perform advanced hyperparameter optimization.

        Args:
            dataset_id: ID of the dataset
            algorithm: Algorithm name
            objectives: List of objectives to optimize

        Returns:
            Enhanced AutoML optimization result
        """
        if not ADVANCED_OPTIMIZER_AVAILABLE or not self.advanced_optimizer:
            # Fall back to basic optimization
            logger.warning(
                "Advanced optimizer not available. Using basic optimization."
            )
            basic_result = await self.optimize_hyperparameters(dataset_id, algorithm)
            return self._convert_basic_to_enhanced_result(basic_result)

        try:
            start_time = time.time()
            logger.info(
                f"Starting advanced hyperparameter optimization for {algorithm}"
            )

            # Get dataset and profile
            dataset = await self.dataset_repository.get(dataset_id)
            if not dataset:
                raise AutoMLError(f"Dataset {dataset_id} not found")

            profile = await self.profile_dataset(dataset_id)

            # Get algorithm configuration
            if algorithm not in self.algorithm_configs:
                raise AutoMLError(f"Algorithm {algorithm} not supported")

            config = self.algorithm_configs[algorithm]

            # Create objective function
            objective_function = self._create_advanced_objective_function(
                dataset, config, objectives or ["auc"]
            )

            # Prepare dataset profile for meta-learning
            dataset_profile_dict = self._convert_profile_to_dict(profile)

            # Run advanced optimization
            optimization_result = self.advanced_optimizer.optimize(
                objective_function=objective_function,
                parameter_space=config.param_space,
                dataset_profile=dataset_profile_dict,
            )

            # Convert to enhanced result
            enhanced_result = self._convert_optimization_result(
                optimization_result, algorithm, start_time
            )

            # Get optimization insights
            insights = self.advanced_optimizer.get_optimization_insights(
                optimization_result
            )
            enhanced_result.optimization_recommendations = insights.get(
                "recommendations", []
            )
            enhanced_result.next_steps = insights.get("next_steps", [])

            # Store in optimization history for meta-learning
            self._store_optimization_history(enhanced_result, dataset_profile_dict)

            logger.info(
                f"Advanced optimization completed in {enhanced_result.optimization_time:.2f}s"
            )
            return enhanced_result

        except Exception as e:
            logger.error(f"Advanced optimization failed: {str(e)}")
            raise AutoMLError(f"Advanced hyperparameter optimization failed: {str(e)}")

    def _create_advanced_objective_function(
        self, dataset: Dataset, config: AlgorithmConfig, objectives: list[str]
    ) -> callable:
        """Create advanced objective function for optimization."""

        def objective_function(
            params: dict[str, Any], trial=None, callbacks: list | None = None
        ) -> float | dict[str, float]:
            """Advanced objective function with multiple objectives."""
            try:
                time.time()

                # Create detector with parameters
                detector = Detector(
                    id=str(uuid4()),
                    name=f"trial_{config.name}",
                    algorithm=config.name,
                    hyperparameters=params,
                    is_fitted=False,
                )

                # Get adapter and prepare data
                adapter = self.adapter_registry.get_adapter(config.adapter_type)
                X = dataset.features.values

                # Train detector
                training_start = time.time()
                success = adapter.train(detector, X)
                training_time = time.time() - training_start

                if not success:
                    if len(objectives) == 1:
                        return 0.0
                    else:
                        return dict.fromkeys(objectives, 0.0)

                # Make predictions
                predictions, scores = adapter.predict(detector, X)

                # Calculate objectives
                results = {}
                contamination = params.get("contamination", 0.1)

                for objective in objectives:
                    if objective == "auc":
                        # Create synthetic labels for evaluation
                        y_true = self._create_synthetic_labels(X, contamination)
                        try:
                            from sklearn.metrics import roc_auc_score

                            if len(np.unique(y_true)) > 1:
                                results[objective] = roc_auc_score(y_true, scores)
                            else:
                                results[objective] = 0.5
                        except Exception:
                            results[objective] = 0.5

                    elif objective == "training_time":
                        # Normalize training time (lower is better)
                        max_time = self.config.max_training_time_per_trial
                        results[objective] = max(0.0, 1.0 - training_time / max_time)

                    elif objective == "memory_usage":
                        # Estimate memory usage (lower is better)
                        estimated_memory = X.nbytes / (1024 * 1024)  # MB
                        max_memory = self.config.max_memory_usage_mb
                        results[objective] = max(
                            0.0, 1.0 - estimated_memory / max_memory
                        )

                    elif objective == "detection_rate":
                        y_true = self._create_synthetic_labels(X, contamination)
                        if np.sum(y_true) > 0:
                            detected = np.sum((y_true == 1) & (predictions == 1))
                            results[objective] = detected / np.sum(y_true)
                        else:
                            results[objective] = 0.0

                    else:
                        # Default objective
                        results[objective] = np.mean(scores)

                # Handle early stopping callbacks
                if callbacks and trial:
                    for callback in callbacks:
                        should_stop = callback(0, results)
                        if should_stop:
                            break

                # Return single value for single objective, dict for multi-objective
                if len(objectives) == 1:
                    return results[objectives[0]]
                else:
                    return results

            except Exception as e:
                logger.warning(f"Objective function evaluation failed: {str(e)}")
                if len(objectives) == 1:
                    return 0.0
                else:
                    return dict.fromkeys(objectives, 0.0)

        return objective_function

    def _create_synthetic_labels(
        self, X: np.ndarray, contamination: float
    ) -> np.ndarray:
        """Create synthetic labels for evaluation."""
        n_samples = len(X)
        n_anomalies = int(n_samples * contamination)

        y_true = np.zeros(n_samples)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        y_true[anomaly_indices] = 1

        return y_true

    def _convert_profile_to_dict(self, profile: DatasetProfile) -> dict[str, Any]:
        """Convert dataset profile to dictionary for meta-learning."""
        return {
            "n_samples": profile.n_samples,
            "n_features": profile.n_features,
            "contamination_estimate": profile.contamination_estimate,
            "missing_values_ratio": profile.missing_values_ratio,
            "sparsity_ratio": profile.sparsity_ratio,
            "dimensionality_ratio": profile.dimensionality_ratio,
            "dataset_size_mb": profile.dataset_size_mb,
            "has_temporal_structure": profile.has_temporal_structure,
            "has_graph_structure": profile.has_graph_structure,
            "complexity_score": profile.complexity_score,
            "n_categorical_features": len(profile.categorical_features),
            "n_numerical_features": len(profile.numerical_features),
            "n_time_series_features": len(profile.time_series_features),
        }

    def _convert_optimization_result(
        self, optimization_result: OptimizationResult, algorithm: str, start_time: float
    ) -> EnhancedAutoMLResult:
        """Convert optimization result to enhanced AutoML result."""
        optimization_time = time.time() - start_time

        # Extract best parameters and score
        best_trial = optimization_result.best_trial
        best_params = best_trial.parameters

        # Get primary objective score
        primary_score = best_trial.objectives.get("auc", 0.0)
        if not primary_score and best_trial.objectives:
            primary_score = list(best_trial.objectives.values())[0]

        # Create enhanced result
        enhanced_result = EnhancedAutoMLResult(
            best_algorithm=algorithm,
            best_params=best_params,
            best_score=primary_score,
            optimization_time=optimization_time,
            trials_completed=optimization_result.n_trials_completed,
            algorithm_rankings=[(algorithm, primary_score)],
            # Enhanced fields
            optimization_strategy_used=self.config.optimization_strategy.value,
            meta_learning_effectiveness=optimization_result.warm_start_effectiveness,
            exploration_score=optimization_result.exploration_score,
            exploitation_score=optimization_result.exploitation_score,
            parameter_sensitivity={},
        )

        # Add multi-objective results if available
        if optimization_result.pareto_front:
            enhanced_result.pareto_front = [
                {"parameters": trial.parameters, "objectives": trial.objectives}
                for trial in optimization_result.pareto_front
            ]

        # Add convergence analysis
        enhanced_result.convergence_stability = self._calculate_convergence_stability(
            optimization_result.best_value_history
        )

        # Add training time breakdown
        enhanced_result.training_time_breakdown = {
            "optimization_time": optimization_time,
            "average_trial_time": optimization_time
            / max(optimization_result.n_trials_completed, 1),
            "total_trials": len(optimization_result.all_trials),
            "successful_trials": optimization_result.n_trials_completed,
            "pruned_trials": optimization_result.n_trials_pruned,
            "failed_trials": optimization_result.n_trials_failed,
        }

        return enhanced_result

    def _calculate_convergence_stability(
        self, best_value_history: list[float]
    ) -> float:
        """Calculate convergence stability metric."""
        if len(best_value_history) < 10:
            return 0.0

        # Calculate coefficient of variation in the last 25% of trials
        last_quarter = best_value_history[-len(best_value_history) // 4 :]
        if len(last_quarter) < 2:
            return 0.0

        mean_val = np.mean(last_quarter)
        if mean_val == 0:
            return 1.0

        cv = np.std(last_quarter) / mean_val
        stability = max(0.0, 1.0 - cv)

        return stability

    def _convert_basic_to_enhanced_result(
        self, basic_result: AutoMLResult
    ) -> EnhancedAutoMLResult:
        """Convert basic AutoML result to enhanced result."""
        return EnhancedAutoMLResult(
            best_algorithm=basic_result.best_algorithm,
            best_params=basic_result.best_params,
            best_score=basic_result.best_score,
            optimization_time=basic_result.optimization_time,
            trials_completed=basic_result.trials_completed,
            algorithm_rankings=basic_result.algorithm_rankings,
            ensemble_config=basic_result.ensemble_config,
            cross_validation_scores=basic_result.cross_validation_scores,
            feature_importance=basic_result.feature_importance,
            optimization_strategy_used="basic",
            exploration_score=0.5,
            exploitation_score=0.5,
            convergence_stability=0.5,
        )

    def _store_optimization_history(
        self, result: EnhancedAutoMLResult, dataset_profile: dict[str, Any]
    ) -> None:
        """Store optimization result in history for meta-learning."""
        history_entry = {
            "timestamp": time.time(),
            "algorithm": result.best_algorithm,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "optimization_time": result.optimization_time,
            "dataset_profile": dataset_profile,
            "optimization_strategy": result.optimization_strategy_used,
            "meta_learning_effectiveness": result.meta_learning_effectiveness,
        }

        self.optimization_history.append(history_entry)

        # Save to file for persistence
        history_file = self.storage_path / "optimization_history.json"
        try:
            import json

            existing_history = []
            if history_file.exists():
                with open(history_file) as f:
                    existing_history = json.load(f)

            existing_history.append(history_entry)

            # Keep only last 1000 entries
            if len(existing_history) > 1000:
                existing_history = existing_history[-1000:]

            with open(history_file, "w") as f:
                json.dump(existing_history, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save optimization history: {str(e)}")

    async def auto_select_and_optimize_advanced(
        self,
        dataset_id: str,
        objectives: list[str] | None = None,
        max_algorithms: int = 3,
        enable_ensemble: bool = True,
    ) -> EnhancedAutoMLResult:
        """Automatically select and optimize using advanced techniques.

        Args:
            dataset_id: ID of the dataset
            objectives: List of objectives to optimize
            max_algorithms: Maximum algorithms to try
            enable_ensemble: Whether to create ensemble if beneficial

        Returns:
            Enhanced AutoML result
        """
        try:
            start_time = time.time()
            logger.info(f"Starting advanced AutoML for dataset {dataset_id}")

            # Profile dataset
            profile = await self.profile_dataset(dataset_id)

            # Recommend algorithms
            recommended_algorithms = self.recommend_algorithms(profile, max_algorithms)

            # Optimize each recommended algorithm using advanced methods
            optimization_results = []
            for algorithm in recommended_algorithms:
                try:
                    result = await self.advanced_optimize_hyperparameters(
                        dataset_id, algorithm, objectives
                    )
                    optimization_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to optimize {algorithm}: {str(e)}")
                    continue

            if not optimization_results:
                raise AutoMLError("No algorithms could be successfully optimized")

            # Find best algorithm based on primary objective
            best_result = max(optimization_results, key=lambda x: x.best_score)

            # Create ensemble if enabled and beneficial
            if enable_ensemble and len(optimization_results) > 1:
                ensemble_config = self._create_advanced_ensemble_config(
                    optimization_results
                )
                best_result.ensemble_config = ensemble_config

            # Update algorithm rankings
            algorithm_rankings = [
                (result.best_algorithm, result.best_score)
                for result in optimization_results
            ]
            algorithm_rankings.sort(key=lambda x: x[1], reverse=True)
            best_result.algorithm_rankings = algorithm_rankings

            # Update total time
            total_time = time.time() - start_time
            best_result.optimization_time = total_time

            # Add final recommendations
            self._add_final_recommendations(best_result, optimization_results)

            logger.info(
                f"Advanced AutoML completed in {total_time:.2f}s. "
                f"Best algorithm: {best_result.best_algorithm} "
                f"(score: {best_result.best_score:.4f})"
            )

            return best_result

        except Exception as e:
            logger.error(f"Advanced AutoML failed: {str(e)}")
            raise AutoMLError(f"Advanced AutoML process failed: {str(e)}")

    def _create_advanced_ensemble_config(
        self, optimization_results: list[EnhancedAutoMLResult]
    ) -> dict[str, Any]:
        """Create advanced ensemble configuration."""
        # Select top algorithms for ensemble based on multiple criteria
        scored_results = []
        for result in optimization_results:
            # Score based on performance, diversity, and efficiency
            performance_score = result.best_score
            efficiency_score = 1.0 / max(result.optimization_time, 1.0)
            stability_score = result.convergence_stability

            combined_score = (
                performance_score * 0.6 + efficiency_score * 0.2 + stability_score * 0.2
            )

            scored_results.append((result, combined_score))

        # Sort by combined score and select top 3
        scored_results.sort(key=lambda x: x[1], reverse=True)
        top_results = [result for result, score in scored_results[:3]]

        # Calculate dynamic weights
        scores = [r.best_score for r in top_results]
        total_score = sum(scores)
        weights = (
            [score / total_score for score in scores]
            if total_score > 0
            else [1 / len(top_results)] * len(top_results)
        )

        ensemble_config = {
            "method": "advanced_weighted_voting",
            "algorithms": [
                {
                    "name": result.best_algorithm,
                    "params": result.best_params,
                    "weight": weight,
                    "performance_score": result.best_score,
                    "convergence_stability": result.convergence_stability,
                }
                for result, weight in zip(top_results, weights, strict=False)
            ],
            "voting_strategy": "soft",
            "normalize_scores": True,
            "dynamic_weighting": True,
            "confidence_boosting": True,
        }

        return ensemble_config

    def _add_final_recommendations(
        self, best_result: EnhancedAutoMLResult, all_results: list[EnhancedAutoMLResult]
    ) -> None:
        """Add final recommendations based on optimization results."""
        recommendations = []
        next_steps = []

        # Performance-based recommendations
        if best_result.best_score < 0.7:
            recommendations.append(
                "Consider collecting more training data or feature engineering"
            )

        if best_result.best_score > 0.9:
            recommendations.append(
                "Excellent performance achieved. Consider production deployment"
            )

        # Convergence-based recommendations
        if best_result.convergence_stability < 0.5:
            recommendations.append(
                "Low convergence stability. Consider increasing optimization time"
            )

        # Exploration/exploitation balance
        if best_result.exploration_score < 0.3:
            recommendations.append(
                "Low exploration detected. Consider different sampling strategy"
            )

        if best_result.exploitation_score < 0.3:
            recommendations.append(
                "Poor convergence. Consider tuning acquisition function"
            )

        # Meta-learning effectiveness
        if (
            best_result.meta_learning_effectiveness
            and best_result.meta_learning_effectiveness < 0.3
        ):
            recommendations.append(
                "Meta-learning not effective. Consider manual parameter initialization"
            )

        # Ensemble recommendations
        if len(all_results) > 1 and not best_result.ensemble_config:
            next_steps.append(
                "Consider creating ensemble from top performing algorithms"
            )

        # Algorithm-specific recommendations
        algorithm_performances = [r.best_score for r in all_results]
        if len(algorithm_performances) > 1:
            performance_variance = np.var(algorithm_performances)
            if performance_variance > 0.1:
                next_steps.append(
                    "High variance in algorithm performance. Data characteristics analysis recommended"
                )

        # Time efficiency recommendations
        avg_optimization_time = np.mean([r.optimization_time for r in all_results])
        if avg_optimization_time > self.config.max_optimization_time * 0.8:
            recommendations.append(
                "Optimization time near limit. Consider reducing n_trials or enabling parallel processing"
            )

        # Add to result
        best_result.optimization_recommendations.extend(recommendations)
        best_result.next_steps.extend(next_steps)

    def get_optimization_insights(self, result: EnhancedAutoMLResult) -> dict[str, Any]:
        """Get comprehensive optimization insights."""
        insights = {
            "performance_analysis": {
                "best_score": result.best_score,
                "score_category": self._categorize_score(result.best_score),
                "algorithm_rankings": result.algorithm_rankings,
                "ensemble_available": result.ensemble_config is not None,
            },
            "optimization_analysis": {
                "strategy_used": result.optimization_strategy_used,
                "exploration_score": result.exploration_score,
                "exploitation_score": result.exploitation_score,
                "convergence_stability": result.convergence_stability,
                "meta_learning_effectiveness": result.meta_learning_effectiveness,
            },
            "efficiency_analysis": {
                "total_optimization_time": result.optimization_time,
                "trials_completed": result.trials_completed,
                "time_per_trial": result.optimization_time
                / max(result.trials_completed, 1),
                "training_time_breakdown": result.training_time_breakdown,
            },
            "recommendations": result.optimization_recommendations,
            "next_steps": result.next_steps,
            "parameter_sensitivity": result.parameter_sensitivity,
        }

        return insights

    def _categorize_score(self, score: float) -> str:
        """Categorize performance score."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "fair"
        elif score >= 0.6:
            return "poor"
        else:
            return "very_poor"

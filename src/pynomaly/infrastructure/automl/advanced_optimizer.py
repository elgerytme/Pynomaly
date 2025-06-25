"""Advanced hyperparameter optimization with state-of-the-art techniques.

This module provides cutting-edge hyperparameter optimization capabilities including:
- Multi-objective optimization
- Bayesian optimization with advanced acquisition functions
- Population-based training
- Hyperband and BOHB algorithms
- Automated early stopping
- Meta-learning for warm starts
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Optional optimization libraries
try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner
    from optuna.samplers import CmaEsSampler, NSGAIISampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Advanced optimization strategies."""

    BAYESIAN = "bayesian"
    HYPERBAND = "hyperband"
    BOHB = "bohb"
    POPULATION_BASED = "population_based"
    MULTI_OBJECTIVE = "multi_objective"
    EVOLUTIONARY = "evolutionary"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"


class AcquisitionFunction(Enum):
    """Bayesian optimization acquisition functions."""

    EXPECTED_IMPROVEMENT = "expected_improvement"
    PROBABILITY_IMPROVEMENT = "probability_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    THOMPSON_SAMPLING = "thompson_sampling"


class MetaLearningStrategy(Enum):
    """Meta-learning strategies for warm starts."""

    SIMILAR_DATASETS = "similar_datasets"
    ALGORITHM_PERFORMANCE = "algorithm_performance"
    TRANSFER_LEARNING = "transfer_learning"
    COLLABORATIVE_FILTERING = "collaborative_filtering"


@dataclass
class OptimizationObjective:
    """Multi-objective optimization objective."""

    name: str
    direction: str  # "maximize" or "minimize"
    weight: float = 1.0
    threshold: float | None = None
    target_value: float | None = None


@dataclass
class OptimizationConstraint:
    """Optimization constraint."""

    name: str
    constraint_type: str  # "upper_bound", "lower_bound", "equality"
    value: float
    tolerance: float = 0.01


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""

    patience: int = 10
    min_delta: float = 0.001
    restore_best_weights: bool = True
    monitor_metric: str = "validation_score"
    mode: str = "max"  # "max" or "min"


@dataclass
class MetaLearningConfig:
    """Meta-learning configuration."""

    strategy: MetaLearningStrategy
    similarity_threshold: float = 0.8
    max_historical_experiments: int = 100
    use_dataset_features: bool = True
    use_algorithm_features: bool = True
    warm_start_trials: int = 10


@dataclass
class AdvancedOptimizationConfig:
    """Advanced optimization configuration."""

    strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT

    # Basic settings
    n_trials: int = 100
    timeout: int = 3600
    n_jobs: int = 1
    random_state: int = 42

    # Multi-objective settings
    objectives: list[OptimizationObjective] = field(default_factory=list)
    constraints: list[OptimizationConstraint] = field(default_factory=list)

    # Bayesian optimization settings
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    gamma: float = 0.25

    # Hyperband settings
    max_epochs: int = 100
    reduction_factor: int = 3

    # Population-based settings
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # Early stopping
    early_stopping: EarlyStoppingConfig | None = None

    # Meta-learning
    meta_learning: MetaLearningConfig | None = None

    # Advanced features
    enable_pruning: bool = True
    enable_parallel: bool = True
    save_intermediate_results: bool = True
    checkpoint_frequency: int = 10


@dataclass
class OptimizationTrial:
    """Individual optimization trial result."""

    trial_id: str
    parameters: dict[str, Any]
    objectives: dict[str, float]
    constraints: dict[str, float]
    state: str  # "complete", "pruned", "failed"
    duration: float
    iteration: int = 0
    intermediate_values: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Advanced optimization result."""

    best_trial: OptimizationTrial
    all_trials: list[OptimizationTrial]
    pareto_front: list[OptimizationTrial] | None = None

    # Optimization statistics
    total_time: float = 0.0
    n_trials_completed: int = 0
    n_trials_pruned: int = 0
    n_trials_failed: int = 0

    # Convergence information
    convergence_history: list[float] = field(default_factory=list)
    best_value_history: list[float] = field(default_factory=list)

    # Meta-learning information
    warm_start_effectiveness: float | None = None
    similar_datasets_used: list[str] = field(default_factory=list)

    # Advanced metrics
    hypervolume: float | None = None
    exploration_score: float = 0.0
    exploitation_score: float = 0.0


class AdvancedHyperparameterOptimizer:
    """Advanced hyperparameter optimizer with state-of-the-art techniques."""

    def __init__(
        self, config: AdvancedOptimizationConfig, storage_path: Path | None = None
    ):
        """Initialize the advanced optimizer.

        Args:
            config: Optimization configuration
            storage_path: Path for storing optimization artifacts
        """
        self.config = config
        self.storage_path = storage_path or Path("./optimization_storage")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Optimization state
        self.study = None
        self.optimization_history = []
        self.meta_learning_data = {}

        # Check dependencies
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Limited optimization capabilities.")

        if not SKLEARN_AVAILABLE:
            logger.warning(
                "Scikit-learn not available. Limited evaluation capabilities."
            )

        if not RAY_AVAILABLE and config.enable_parallel:
            logger.warning("Ray not available. Parallel optimization disabled.")
            self.config.enable_parallel = False

    def optimize(
        self,
        objective_function: Callable,
        parameter_space: dict[str, Any],
        dataset_profile: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Run advanced hyperparameter optimization.

        Args:
            objective_function: Function to optimize
            parameter_space: Parameter search space
            dataset_profile: Dataset characteristics for meta-learning

        Returns:
            Optimization result
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is required for advanced optimization")

        logger.info(f"Starting {self.config.strategy.value} optimization")
        start_time = time.time()

        try:
            # Initialize meta-learning if configured
            warm_start_params = None
            if self.config.meta_learning and dataset_profile:
                warm_start_params = self._get_warm_start_parameters(dataset_profile)

            # Create optimization study
            self.study = self._create_study(parameter_space, warm_start_params)

            # Run optimization
            self._run_optimization(objective_function, parameter_space)

            # Process results
            result = self._process_results(start_time)

            # Save optimization artifacts
            self._save_optimization_artifacts(result)

            logger.info(f"Optimization completed in {result.total_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise

    def _create_study(
        self,
        parameter_space: dict[str, Any],
        warm_start_params: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Create Optuna study with appropriate configuration."""
        # Choose sampler based on strategy
        if self.config.strategy == OptimizationStrategy.BAYESIAN:
            sampler = TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=self.config.n_ei_candidates,
                gamma=self.config.gamma,
                seed=self.config.random_state,
            )
        elif self.config.strategy == OptimizationStrategy.EVOLUTIONARY:
            sampler = CmaEsSampler(seed=self.config.random_state)
        elif self.config.strategy == OptimizationStrategy.MULTI_OBJECTIVE:
            sampler = NSGAIISampler(seed=self.config.random_state)
        else:
            sampler = TPESampler(seed=self.config.random_state)

        # Choose pruner
        pruner = None
        if self.config.enable_pruning:
            if self.config.strategy == OptimizationStrategy.HYPERBAND:
                pruner = HyperbandPruner(
                    min_resource=1,
                    max_resource=self.config.max_epochs,
                    reduction_factor=self.config.reduction_factor,
                )
            else:
                pruner = MedianPruner(
                    n_startup_trials=self.config.n_startup_trials, n_warmup_steps=10
                )

        # Determine study direction
        if self.config.objectives:
            if len(self.config.objectives) == 1:
                direction = self.config.objectives[0].direction
            else:
                # Multi-objective
                directions = [obj.direction for obj in self.config.objectives]
                direction = directions
        else:
            direction = "maximize"

        # Create study
        study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

        # Add warm start trials if available
        if warm_start_params:
            for params in warm_start_params:
                study.enqueue_trial(params)
            logger.info(f"Added {len(warm_start_params)} warm start trials")

        return study

    def _run_optimization(
        self, objective_function: Callable, parameter_space: dict[str, Any]
    ) -> None:
        """Run the optimization process."""
        if self.config.enable_parallel and RAY_AVAILABLE:
            self._run_parallel_optimization(objective_function, parameter_space)
        else:
            self._run_sequential_optimization(objective_function, parameter_space)

    def _run_sequential_optimization(
        self, objective_function: Callable, parameter_space: dict[str, Any]
    ) -> None:
        """Run sequential optimization."""

        def optuna_objective(trial):
            # Sample parameters
            params = self._sample_parameters(trial, parameter_space)

            # Add early stopping callback if configured
            callbacks = []
            if self.config.early_stopping:
                callbacks.append(self._create_early_stopping_callback(trial))

            # Evaluate objective
            try:
                if self.config.objectives and len(self.config.objectives) > 1:
                    # Multi-objective optimization
                    results = objective_function(
                        params, trial=trial, callbacks=callbacks
                    )
                    if isinstance(results, dict):
                        return [results[obj.name] for obj in self.config.objectives]
                    else:
                        return results
                else:
                    # Single objective
                    return objective_function(params, trial=trial, callbacks=callbacks)
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                if len(self.config.objectives) > 1:
                    return [float("-inf")] * len(self.config.objectives)
                else:
                    return float("-inf")

        # Run optimization
        self.study.optimize(
            optuna_objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            callbacks=[self._trial_callback]
            if self.config.save_intermediate_results
            else None,
        )

    def _run_parallel_optimization(
        self, objective_function: Callable, parameter_space: dict[str, Any]
    ) -> None:
        """Run parallel optimization using Ray Tune."""
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, falling back to sequential optimization")
            self._run_sequential_optimization(objective_function, parameter_space)
            return

        # Convert parameter space to Ray Tune format
        tune_space = self._convert_to_tune_space(parameter_space)

        # Choose scheduler
        if self.config.strategy == OptimizationStrategy.HYPERBAND:
            scheduler = HyperBandScheduler(
                metric="score",
                mode="max",
                max_t=self.config.max_epochs,
                reduction_factor=self.config.reduction_factor,
            )
        else:
            scheduler = ASHAScheduler(
                metric="score", mode="max", max_t=self.config.max_epochs
            )

        # Run Ray Tune
        analysis = tune.run(
            tune.with_parameters(objective_function),
            config=tune_space,
            num_samples=self.config.n_trials,
            scheduler=scheduler,
            resources_per_trial={"cpu": 1},
            local_dir=str(self.storage_path),
        )

        # Convert Ray results back to Optuna format
        self._convert_ray_results_to_optuna(analysis)

    def _sample_parameters(
        self, trial, parameter_space: dict[str, Any]
    ) -> dict[str, Any]:
        """Sample parameters from the search space."""
        params = {}

        for param_name, param_config in parameter_space.items():
            if param_config["type"] == "float":
                if param_config.get("log", False):
                    params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"], log=True
                    )
                else:
                    params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"]
                    )
            elif param_config["type"] == "int":
                if param_config.get("log", False):
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"], log=True
                    )
                else:
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif param_config["type"] == "uniform":
                params[param_name] = trial.suggest_uniform(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "discrete":
                params[param_name] = trial.suggest_discrete_uniform(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    param_config["step"],
                )

        return params

    def _create_early_stopping_callback(self, trial):
        """Create early stopping callback."""
        config = self.config.early_stopping

        class EarlyStoppingCallback:
            def __init__(self):
                self.best_value = (
                    float("-inf") if config.mode == "max" else float("inf")
                )
                self.patience_counter = 0
                self.stopped_epoch = 0

            def __call__(self, epoch: int, metrics: dict[str, float]):
                current_value = metrics.get(config.monitor_metric, 0.0)

                improved = False
                if config.mode == "max":
                    if current_value > self.best_value + config.min_delta:
                        self.best_value = current_value
                        improved = True
                else:
                    if current_value < self.best_value - config.min_delta:
                        self.best_value = current_value
                        improved = True

                if improved:
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Report intermediate value to Optuna
                trial.report(current_value, epoch)

                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Check early stopping
                if self.patience_counter >= config.patience:
                    self.stopped_epoch = epoch
                    logger.info(f"Early stopping at epoch {epoch}")
                    return True

                return False

        return EarlyStoppingCallback()

    def _trial_callback(self, study, trial):
        """Callback for trial completion."""
        if self.config.save_intermediate_results:
            trial_data = {
                "trial_id": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name,
                "duration": trial.duration.total_seconds() if trial.duration else 0,
                "intermediate_values": trial.intermediate_values,
            }

            # Save trial data
            trial_file = self.storage_path / f"trial_{trial.number}.json"
            with open(trial_file, "w") as f:
                json.dump(trial_data, f, indent=2)

    def _get_warm_start_parameters(
        self, dataset_profile: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get warm start parameters using meta-learning."""
        if not self.config.meta_learning:
            return []

        strategy = self.config.meta_learning.strategy

        if strategy == MetaLearningStrategy.SIMILAR_DATASETS:
            return self._get_similar_dataset_params(dataset_profile)
        elif strategy == MetaLearningStrategy.ALGORITHM_PERFORMANCE:
            return self._get_algorithm_performance_params(dataset_profile)
        elif strategy == MetaLearningStrategy.TRANSFER_LEARNING:
            return self._get_transfer_learning_params(dataset_profile)
        else:
            return []

    def _get_similar_dataset_params(
        self, dataset_profile: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get parameters from similar datasets."""
        # Load historical optimization data
        history_file = self.storage_path / "optimization_history.json"
        if not history_file.exists():
            return []

        with open(history_file) as f:
            history = json.load(f)

        # Find similar datasets
        similar_params = []
        for record in history:
            similarity = self._calculate_dataset_similarity(
                dataset_profile, record.get("dataset_profile", {})
            )

            if similarity > self.config.meta_learning.similarity_threshold:
                if "best_params" in record:
                    similar_params.append(record["best_params"])

        # Return top warm start trials
        return similar_params[: self.config.meta_learning.warm_start_trials]

    def _calculate_dataset_similarity(
        self, profile1: dict[str, Any], profile2: dict[str, Any]
    ) -> float:
        """Calculate similarity between dataset profiles."""
        if not profile1 or not profile2:
            return 0.0

        # Numerical features similarity
        numerical_features = [
            "n_samples",
            "n_features",
            "contamination_estimate",
            "missing_values_ratio",
            "sparsity_ratio",
            "dimensionality_ratio",
        ]

        similarities = []
        for feature in numerical_features:
            if feature in profile1 and feature in profile2:
                val1, val2 = profile1[feature], profile2[feature]
                if val1 == 0 and val2 == 0:
                    sim = 1.0
                elif val1 == 0 or val2 == 0:
                    sim = 0.0
                else:
                    # Relative similarity
                    sim = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2))
                similarities.append(sim)

        # Categorical features similarity
        categorical_features = ["has_temporal_structure", "has_graph_structure"]
        for feature in categorical_features:
            if feature in profile1 and feature in profile2:
                sim = 1.0 if profile1[feature] == profile2[feature] else 0.0
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def _get_algorithm_performance_params(
        self, dataset_profile: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get parameters based on algorithm performance patterns."""
        # This would implement more sophisticated meta-learning
        # For now, return empty list
        return []

    def _get_transfer_learning_params(
        self, dataset_profile: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get parameters using transfer learning from related domains."""
        # This would implement transfer learning techniques
        # For now, return empty list
        return []

    def _process_results(self, start_time: float) -> OptimizationResult:
        """Process optimization results."""
        total_time = time.time() - start_time

        # Get best trial
        best_trial_data = self.study.best_trial
        best_trial = OptimizationTrial(
            trial_id=str(best_trial_data.number),
            parameters=best_trial_data.params,
            objectives={"primary": best_trial_data.value}
            if best_trial_data.value is not None
            else {},
            constraints={},
            state=best_trial_data.state.name,
            duration=best_trial_data.duration.total_seconds()
            if best_trial_data.duration
            else 0,
            intermediate_values=list(best_trial_data.intermediate_values),
        )

        # Get all trials
        all_trials = []
        for trial_data in self.study.trials:
            trial = OptimizationTrial(
                trial_id=str(trial_data.number),
                parameters=trial_data.params,
                objectives={"primary": trial_data.value}
                if trial_data.value is not None
                else {},
                constraints={},
                state=trial_data.state.name,
                duration=trial_data.duration.total_seconds()
                if trial_data.duration
                else 0,
                intermediate_values=list(trial_data.intermediate_values),
            )
            all_trials.append(trial)

        # Calculate statistics
        completed_trials = [t for t in all_trials if t.state == "COMPLETE"]
        pruned_trials = [t for t in all_trials if t.state == "PRUNED"]
        failed_trials = [t for t in all_trials if t.state == "FAIL"]

        # Create convergence history
        convergence_history = []
        best_value_history = []
        best_so_far = float("-inf")

        for trial in completed_trials:
            if trial.objectives and "primary" in trial.objectives:
                value = trial.objectives["primary"]
                convergence_history.append(value)

                if value > best_so_far:
                    best_so_far = value
                best_value_history.append(best_so_far)

        # Calculate exploration/exploitation scores
        exploration_score = self._calculate_exploration_score(completed_trials)
        exploitation_score = self._calculate_exploitation_score(completed_trials)

        result = OptimizationResult(
            best_trial=best_trial,
            all_trials=all_trials,
            total_time=total_time,
            n_trials_completed=len(completed_trials),
            n_trials_pruned=len(pruned_trials),
            n_trials_failed=len(failed_trials),
            convergence_history=convergence_history,
            best_value_history=best_value_history,
            exploration_score=exploration_score,
            exploitation_score=exploitation_score,
        )

        return result

    def _calculate_exploration_score(self, trials: list[OptimizationTrial]) -> float:
        """Calculate exploration score based on parameter diversity."""
        if len(trials) < 2:
            return 0.0

        # Calculate parameter space coverage
        param_ranges = {}
        for trial in trials:
            for param, value in trial.parameters.items():
                if param not in param_ranges:
                    param_ranges[param] = [value, value]
                else:
                    param_ranges[param][0] = min(param_ranges[param][0], value)
                    param_ranges[param][1] = max(param_ranges[param][1], value)

        # Calculate normalized coverage
        coverages = []
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, (int, float)) and min_val != max_val:
                coverage = (max_val - min_val) / max(abs(max_val), abs(min_val), 1.0)
                coverages.append(min(coverage, 1.0))

        return np.mean(coverages) if coverages else 0.0

    def _calculate_exploitation_score(self, trials: list[OptimizationTrial]) -> float:
        """Calculate exploitation score based on convergence."""
        if len(trials) < 2:
            return 0.0

        # Calculate improvement rate in later trials
        objectives = [t.objectives.get("primary", 0) for t in trials if t.objectives]
        if len(objectives) < 2:
            return 0.0

        # Compare first half vs second half
        mid_point = len(objectives) // 2
        first_half_best = max(objectives[:mid_point])
        second_half_best = max(objectives[mid_point:])

        if first_half_best == 0:
            return 1.0 if second_half_best > 0 else 0.0

        improvement = (second_half_best - first_half_best) / abs(first_half_best)
        return max(0.0, min(1.0, improvement))

    def _save_optimization_artifacts(self, result: OptimizationResult) -> None:
        """Save optimization artifacts for future meta-learning."""
        # Save optimization result
        result_file = self.storage_path / f"optimization_result_{int(time.time())}.json"

        result_data = {
            "best_trial": {
                "parameters": result.best_trial.parameters,
                "objectives": result.best_trial.objectives,
                "state": result.best_trial.state,
                "duration": result.best_trial.duration,
            },
            "statistics": {
                "total_time": result.total_time,
                "n_trials_completed": result.n_trials_completed,
                "n_trials_pruned": result.n_trials_pruned,
                "n_trials_failed": result.n_trials_failed,
                "exploration_score": result.exploration_score,
                "exploitation_score": result.exploitation_score,
            },
            "convergence_history": result.convergence_history,
            "best_value_history": result.best_value_history,
            "config": {
                "strategy": self.config.strategy.value,
                "n_trials": self.config.n_trials,
                "timeout": self.config.timeout,
            },
        }

        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)

        # Save Optuna study for future reference
        if self.study:
            study_file = self.storage_path / f"study_{int(time.time())}.pkl"
            with open(study_file, "wb") as f:
                pickle.dump(self.study, f)

        logger.info(f"Optimization artifacts saved to {self.storage_path}")

    def _convert_to_tune_space(self, parameter_space: dict[str, Any]) -> dict[str, Any]:
        """Convert parameter space to Ray Tune format."""
        tune_space = {}

        for param_name, param_config in parameter_space.items():
            if param_config["type"] == "float":
                if param_config.get("log", False):
                    tune_space[param_name] = tune.loguniform(
                        param_config["low"], param_config["high"]
                    )
                else:
                    tune_space[param_name] = tune.uniform(
                        param_config["low"], param_config["high"]
                    )
            elif param_config["type"] == "int":
                tune_space[param_name] = tune.randint(
                    param_config["low"], param_config["high"] + 1
                )
            elif param_config["type"] == "categorical":
                tune_space[param_name] = tune.choice(param_config["choices"])

        return tune_space

    def _convert_ray_results_to_optuna(self, analysis) -> None:
        """Convert Ray Tune results back to Optuna format."""
        # This would convert Ray results to Optuna study format
        # For now, we'll keep them separate
        pass

    def get_optimization_insights(self, result: OptimizationResult) -> dict[str, Any]:
        """Get insights from optimization results."""
        insights = {
            "optimization_quality": {
                "exploration_score": result.exploration_score,
                "exploitation_score": result.exploitation_score,
                "convergence_stability": self._calculate_convergence_stability(result),
                "parameter_sensitivity": self._analyze_parameter_sensitivity(result),
            },
            "recommendations": [],
            "next_steps": [],
        }

        # Add recommendations based on results
        if result.exploration_score < 0.3:
            insights["recommendations"].append(
                "Low exploration detected. Consider increasing n_trials or using different sampling strategy."
            )

        if result.exploitation_score < 0.3:
            insights["recommendations"].append(
                "Poor convergence detected. Consider tuning acquisition function parameters."
            )

        if result.n_trials_pruned / result.n_trials_completed > 0.5:
            insights["recommendations"].append(
                "High pruning rate. Consider adjusting pruning strategy or early stopping criteria."
            )

        # Suggest next steps
        if result.best_trial.objectives.get("primary", 0) < 0.8:
            insights["next_steps"].append(
                "Consider feature engineering or data preprocessing"
            )

        insights["next_steps"].append(
            "Try ensemble methods with top performing configurations"
        )

        return insights

    def _calculate_convergence_stability(self, result: OptimizationResult) -> float:
        """Calculate convergence stability metric."""
        if len(result.best_value_history) < 10:
            return 0.0

        # Calculate coefficient of variation in the last portion
        last_values = result.best_value_history[-10:]
        if np.std(last_values) == 0:
            return 1.0

        cv = np.std(last_values) / np.mean(last_values)
        stability = max(0.0, 1.0 - cv)

        return stability

    def _analyze_parameter_sensitivity(
        self, result: OptimizationResult
    ) -> dict[str, float]:
        """Analyze parameter sensitivity."""
        if len(result.all_trials) < 10:
            return {}

        completed_trials = [
            t for t in result.all_trials if t.state == "COMPLETE" and t.objectives
        ]
        if len(completed_trials) < 10:
            return {}

        # Calculate correlation between parameters and objectives
        param_names = list(completed_trials[0].parameters.keys())
        sensitivity = {}

        for param_name in param_names:
            param_values = []
            objective_values = []

            for trial in completed_trials:
                if param_name in trial.parameters and "primary" in trial.objectives:
                    param_val = trial.parameters[param_name]
                    obj_val = trial.objectives["primary"]

                    # Only include numerical parameters
                    if isinstance(param_val, (int, float)):
                        param_values.append(param_val)
                        objective_values.append(obj_val)

            if len(param_values) > 1:
                correlation = np.corrcoef(param_values, objective_values)[0, 1]
                sensitivity[param_name] = (
                    abs(correlation) if not np.isnan(correlation) else 0.0
                )

        return sensitivity

"""Comprehensive AutoML service with advanced hyperparameter optimization strategies.

This service provides state-of-the-art automated machine learning capabilities including:
- Multiple optimization strategies (Random Search, Bayesian Optimization, Grid Search)
- Automated feature engineering and selection
- Ensemble methods and model combination
- Experiment tracking and model registry integration
- Automated model validation and evaluation
- Meta-learning for intelligent hyperparameter tuning
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# TODO: Create local Dataset entity
from monorepo.domain.exceptions import AutoMLError

# Optional optimization libraries
try:
    import optuna
    from optuna.pruners import MedianPruner, PatientPruner
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        RandomizedSearchCV,
        StratifiedKFold,
        cross_val_score,
    )
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import randint, uniform

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Suppress warnings during optimization
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Hyperparameter optimization strategies."""

    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GRID_SEARCH = "grid_search"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"
    BOHB = "bohb"
    MULTI_OBJECTIVE = "multi_objective"


class EnsembleMethod(Enum):
    """Ensemble methods for model combination."""

    VOTING = "voting"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    STACKING = "stacking"
    BLENDING = "blending"
    DYNAMIC_SELECTION = "dynamic_selection"


class FeatureEngineeringMethod(Enum):
    """Feature engineering methods."""

    POLYNOMIAL = "polynomial"
    INTERACTION = "interaction"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    DOMAIN_SPECIFIC = "domain_specific"
    AUTOMATED = "automated"


@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""

    strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_OPTIMIZATION
    max_trials: int = 100
    max_time_seconds: int = 3600
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"
    early_stopping_rounds: int = 10
    parallel_trials: int = 1
    random_state: int = 42
    enable_pruning: bool = True
    pruning_patience: int = 5

    # Advanced options
    multi_objective: bool = False
    objectives: list[str] = field(default_factory=lambda: ["accuracy", "speed"])
    constraints: dict[str, float] = field(default_factory=dict)
    meta_learning: bool = True
    warm_start: bool = True


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""

    methods: list[FeatureEngineeringMethod] = field(
        default_factory=lambda: [
            FeatureEngineeringMethod.STATISTICAL,
            FeatureEngineeringMethod.INTERACTION,
        ]
    )
    max_features: int = 1000
    selection_method: str = "univariate"
    selection_k: int = 100
    polynomial_degree: int = 2
    interaction_only: bool = False
    enable_scaling: bool = True
    handle_missing: str = "impute"  # "drop", "impute", "flag"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""

    methods: list[EnsembleMethod] = field(
        default_factory=lambda: [EnsembleMethod.VOTING, EnsembleMethod.STACKING]
    )
    max_models: int = 10
    min_models: int = 3
    diversity_threshold: float = 0.1
    performance_threshold: float = 0.8
    voting_strategy: str = "soft"  # "hard", "soft"
    stacking_cv: int = 5
    meta_learner: str = "linear"  # "linear", "tree", "neural"


@dataclass
class ExperimentResult:
    """Result of an optimization experiment."""

    experiment_id: str
    algorithm: str
    strategy: OptimizationStrategy
    best_params: dict[str, Any]
    best_score: float
    best_std: float
    cv_scores: list[float]
    optimization_time: float
    total_trials: int
    successful_trials: int
    feature_importance: dict[str, float]
    model_metadata: dict[str, Any]
    ensemble_config: dict[str, Any] | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetaLearningRecord:
    """Record for meta-learning system."""

    dataset_signature: str
    algorithm: str
    hyperparameters: dict[str, Any]
    performance: float
    optimization_time: float
    feature_count: int
    sample_count: int
    dataset_characteristics: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class ComprehensiveAutoMLService:
    """Comprehensive AutoML service with advanced capabilities."""

    def __init__(
        self,
        storage_path: Path = Path("./comprehensive_automl_storage"),
        experiment_tracking: bool = True,
        enable_parallel: bool = True,
        max_workers: int = 4,
    ):
        """Initialize comprehensive AutoML service.

        Args:
            storage_path: Path for storing experiments and meta-learning data
            experiment_tracking: Enable experiment tracking
            enable_parallel: Enable parallel optimization
            max_workers: Maximum number of parallel workers
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.experiment_tracking = experiment_tracking
        self.enable_parallel = enable_parallel
        self.max_workers = max_workers

        # Initialize storage
        self.experiments: list[ExperimentResult] = []
        self.meta_learning_records: list[MetaLearningRecord] = []

        # Load historical data
        self._load_historical_data()

        # Initialize algorithm configurations
        self.algorithm_configs = self._initialize_algorithm_configs()

        # Initialize feature engineering
        self.feature_engineer = FeatureEngineer()

        # Initialize ensemble manager
        self.ensemble_manager = EnsembleManager()

        logger.info("Comprehensive AutoML service initialized")

    def _initialize_algorithm_configs(self) -> dict[str, dict[str, Any]]:
        """Initialize algorithm configurations with parameter spaces."""
        return {
            "IsolationForest": {
                "param_space": {
                    "n_estimators": {"type": "int", "low": 50, "high": 500},
                    "max_samples": {
                        "type": "categorical",
                        "choices": ["auto", 0.5, 0.8, 1.0],
                    },
                    "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                    "max_features": {"type": "float", "low": 0.1, "high": 1.0},
                    "bootstrap": {"type": "categorical", "choices": [True, False]},
                },
                "grid_space": {
                    "n_estimators": [100, 200, 300],
                    "max_samples": ["auto", 0.5, 0.8],
                    "contamination": [0.05, 0.1, 0.2],
                    "max_features": [0.5, 0.8, 1.0],
                },
                "random_space": {
                    "n_estimators": randint(50, 500),
                    "max_samples": ["auto", 0.5, 0.8, 1.0],
                    "contamination": uniform(0.01, 0.49),
                    "max_features": uniform(0.1, 0.9),
                },
                "meta_features": ["n_estimators", "contamination", "max_features"],
                "complexity_score": 0.4,
                "interpretability_score": 0.8,
            },
            "LocalOutlierFactor": {
                "param_space": {
                    "n_neighbors": {"type": "int", "low": 5, "high": 50},
                    "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                    "algorithm": {
                        "type": "categorical",
                        "choices": ["auto", "ball_tree", "kd_tree", "brute"],
                    },
                    "leaf_size": {"type": "int", "low": 10, "high": 50},
                    "p": {"type": "categorical", "choices": [1, 2]},
                },
                "grid_space": {
                    "n_neighbors": [10, 20, 30],
                    "contamination": [0.05, 0.1, 0.2],
                    "algorithm": ["auto", "ball_tree"],
                    "leaf_size": [20, 30],
                },
                "random_space": {
                    "n_neighbors": randint(5, 50),
                    "contamination": uniform(0.01, 0.49),
                    "algorithm": ["auto", "ball_tree", "kd_tree"],
                    "leaf_size": randint(10, 50),
                },
                "meta_features": ["n_neighbors", "contamination", "algorithm"],
                "complexity_score": 0.7,
                "interpretability_score": 0.6,
            },
            "OneClassSVM": {
                "param_space": {
                    "kernel": {
                        "type": "categorical",
                        "choices": ["rbf", "linear", "poly", "sigmoid"],
                    },
                    "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
                    "nu": {"type": "float", "low": 0.01, "high": 0.5},
                    "degree": {"type": "int", "low": 2, "high": 5},
                    "coef0": {"type": "float", "low": -10, "high": 10},
                },
                "grid_space": {
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"],
                    "nu": [0.05, 0.1, 0.2],
                    "degree": [2, 3],
                },
                "random_space": {
                    "kernel": ["rbf", "linear", "poly"],
                    "gamma": ["scale", "auto"],
                    "nu": uniform(0.01, 0.49),
                    "degree": randint(2, 5),
                },
                "meta_features": ["kernel", "gamma", "nu"],
                "complexity_score": 0.9,
                "interpretability_score": 0.3,
            },
            "ECOD": {
                "param_space": {
                    "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                    "n_jobs": {"type": "int", "low": 1, "high": 4},
                },
                "grid_space": {
                    "contamination": [0.05, 0.1, 0.2],
                    "n_jobs": [1, 2],
                },
                "random_space": {
                    "contamination": uniform(0.01, 0.49),
                    "n_jobs": randint(1, 4),
                },
                "meta_features": ["contamination"],
                "complexity_score": 0.2,
                "interpretability_score": 0.9,
            },
            "COPOD": {
                "param_space": {
                    "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                    "n_jobs": {"type": "int", "low": 1, "high": 4},
                },
                "grid_space": {
                    "contamination": [0.05, 0.1, 0.2],
                    "n_jobs": [1, 2],
                },
                "random_space": {
                    "contamination": uniform(0.01, 0.49),
                    "n_jobs": randint(1, 4),
                },
                "meta_features": ["contamination"],
                "complexity_score": 0.3,
                "interpretability_score": 0.8,
            },
        }

    async def comprehensive_optimize(
        self,
        dataset: Dataset,
        algorithms: list[str] | None = None,
        config: OptimizationConfig | None = None,
        feature_config: FeatureEngineeringConfig | None = None,
        ensemble_config: EnsembleConfig | None = None,
    ) -> ExperimentResult:
        """Perform comprehensive optimization with all advanced features.

        Args:
            dataset: Dataset to optimize on
            algorithms: List of algorithms to try (None for automatic selection)
            config: Optimization configuration
            feature_config: Feature engineering configuration
            ensemble_config: Ensemble configuration

        Returns:
            Best experiment result
        """
        if not SKLEARN_AVAILABLE:
            raise AutoMLError("scikit-learn is required for comprehensive optimization")

        start_time = time.time()
        config = config or OptimizationConfig()
        feature_config = feature_config or FeatureEngineeringConfig()
        ensemble_config = ensemble_config or EnsembleConfig()

        logger.info("Starting comprehensive AutoML optimization")

        # Step 1: Feature Engineering
        logger.info("Step 1: Feature Engineering")
        engineered_dataset = await self._apply_feature_engineering(
            dataset, feature_config
        )

        # Step 2: Algorithm Selection
        logger.info("Step 2: Algorithm Selection")
        if algorithms is None:
            algorithms = self._select_algorithms(engineered_dataset)

        # Step 3: Hyperparameter Optimization
        logger.info("Step 3: Hyperparameter Optimization")
        optimization_results = await self._optimize_algorithms(
            engineered_dataset, algorithms, config
        )

        # Step 4: Ensemble Creation
        logger.info("Step 4: Ensemble Creation")
        ensemble_result = await self._create_ensemble(
            optimization_results, ensemble_config
        )

        # Step 5: Final Evaluation
        logger.info("Step 5: Final Evaluation")
        final_result = await self._evaluate_final_model(
            ensemble_result, engineered_dataset, config
        )

        optimization_time = time.time() - start_time

        # Create comprehensive result
        experiment_result = ExperimentResult(
            experiment_id=f"comprehensive_{int(time.time())}",
            algorithm=final_result.get("algorithm", "ensemble"),
            strategy=config.strategy,
            best_params=final_result.get("params", {}),
            best_score=final_result.get("score", 0.0),
            best_std=final_result.get("std", 0.0),
            cv_scores=final_result.get("cv_scores", []),
            optimization_time=optimization_time,
            total_trials=sum(r.get("total_trials", 0) for r in optimization_results),
            successful_trials=sum(
                r.get("successful_trials", 0) for r in optimization_results
            ),
            feature_importance=final_result.get("feature_importance", {}),
            model_metadata=final_result.get("metadata", {}),
            ensemble_config=final_result.get("ensemble_config"),
        )

        # Store experiment
        if self.experiment_tracking:
            self.experiments.append(experiment_result)
            self._save_experiment(experiment_result)

        # Update meta-learning
        if config.meta_learning:
            await self._update_meta_learning(dataset, experiment_result)

        logger.info(f"Comprehensive optimization completed in {optimization_time:.2f}s")
        return experiment_result

    async def _apply_feature_engineering(
        self, dataset: Dataset, config: FeatureEngineeringConfig
    ) -> Dataset:
        """Apply feature engineering to the dataset."""
        return await self.feature_engineer.engineer_features(dataset, config)

    def _select_algorithms(self, dataset: Dataset) -> list[str]:
        """Select algorithms based on dataset characteristics."""
        # Analyze dataset characteristics
        n_samples, n_features = dataset.data.shape

        selected_algorithms = []

        # Rule-based selection
        if n_samples < 1000:
            selected_algorithms.extend(["ECOD", "COPOD"])

        if n_features < 50:
            selected_algorithms.extend(["LocalOutlierFactor", "OneClassSVM"])

        if n_samples > 10000:
            selected_algorithms.extend(["IsolationForest"])

        # Always include some baseline algorithms
        if not selected_algorithms:
            selected_algorithms = ["IsolationForest", "ECOD", "LocalOutlierFactor"]

        # Use meta-learning for selection
        if self.meta_learning_records:
            meta_suggestions = self._get_meta_learning_suggestions(dataset)
            selected_algorithms.extend(meta_suggestions)

        return list(set(selected_algorithms))

    async def _optimize_algorithms(
        self, dataset: Dataset, algorithms: list[str], config: OptimizationConfig
    ) -> list[dict[str, Any]]:
        """Optimize hyperparameters for multiple algorithms."""
        results = []

        if self.enable_parallel and config.parallel_trials > 1:
            # Parallel optimization
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(algorithms))
            ) as executor:
                futures = {
                    executor.submit(
                        self._optimize_single_algorithm, dataset, algorithm, config
                    ): algorithm
                    for algorithm in algorithms
                }

                for future in as_completed(futures):
                    algorithm = futures[future]
                    try:
                        result = await future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to optimize {algorithm}: {e}")
        else:
            # Sequential optimization
            for algorithm in algorithms:
                try:
                    result = await self._optimize_single_algorithm(
                        dataset, algorithm, config
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to optimize {algorithm}: {e}")

        return results

    async def _optimize_single_algorithm(
        self, dataset: Dataset, algorithm: str, config: OptimizationConfig
    ) -> dict[str, Any]:
        """Optimize hyperparameters for a single algorithm."""
        if algorithm not in self.algorithm_configs:
            raise AutoMLError(f"Algorithm {algorithm} not supported")

        algorithm_config = self.algorithm_configs[algorithm]

        # Choose optimization strategy
        if config.strategy == OptimizationStrategy.RANDOM_SEARCH:
            return await self._random_search_optimization(
                dataset, algorithm, algorithm_config, config
            )
        elif config.strategy == OptimizationStrategy.GRID_SEARCH:
            return await self._grid_search_optimization(
                dataset, algorithm, algorithm_config, config
            )
        elif config.strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
            return await self._bayesian_optimization(
                dataset, algorithm, algorithm_config, config
            )
        elif config.strategy == OptimizationStrategy.HYPERBAND:
            return await self._hyperband_optimization(
                dataset, algorithm, algorithm_config, config
            )
        else:
            # Default to Bayesian optimization
            return await self._bayesian_optimization(
                dataset, algorithm, algorithm_config, config
            )

    async def _random_search_optimization(
        self,
        dataset: Dataset,
        algorithm: str,
        algorithm_config: dict,
        config: OptimizationConfig,
    ) -> dict[str, Any]:
        """Perform random search optimization."""
        if not SKLEARN_AVAILABLE:
            raise AutoMLError("scikit-learn is required for random search")

        # Create mock estimator for the algorithm
        estimator = self._create_estimator(algorithm, {})

        # Get parameter distribution
        param_dist = algorithm_config.get("random_space", {})
        if not param_dist:
            # Fallback to converting param_space
            param_dist = self._convert_param_space_to_random(
                algorithm_config["param_space"]
            )

        # Prepare data
        X = dataset.data.values
        y = self._generate_synthetic_labels(X, contamination=0.1)

        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=config.max_trials,
            cv=config.cv_folds,
            scoring=config.scoring_metric,
            n_jobs=config.parallel_trials,
            random_state=config.random_state,
        )

        start_time = time.time()
        random_search.fit(X, y)
        optimization_time = time.time() - start_time

        return {
            "algorithm": algorithm,
            "best_params": random_search.best_params_,
            "best_score": random_search.best_score_,
            "cv_scores": random_search.cv_results_["mean_test_score"],
            "optimization_time": optimization_time,
            "total_trials": len(random_search.cv_results_["mean_test_score"]),
            "successful_trials": len(random_search.cv_results_["mean_test_score"]),
        }

    async def _grid_search_optimization(
        self,
        dataset: Dataset,
        algorithm: str,
        algorithm_config: dict,
        config: OptimizationConfig,
    ) -> dict[str, Any]:
        """Perform grid search optimization."""
        if not SKLEARN_AVAILABLE:
            raise AutoMLError("scikit-learn is required for grid search")

        # Create mock estimator for the algorithm
        estimator = self._create_estimator(algorithm, {})

        # Get parameter grid
        param_grid = algorithm_config.get("grid_space", {})
        if not param_grid:
            # Fallback to converting param_space
            param_grid = self._convert_param_space_to_grid(
                algorithm_config["param_space"]
            )

        # Prepare data
        X = dataset.data.values
        y = self._generate_synthetic_labels(X, contamination=0.1)

        # Perform grid search
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=config.cv_folds,
            scoring=config.scoring_metric,
            n_jobs=config.parallel_trials,
        )

        start_time = time.time()
        grid_search.fit(X, y)
        optimization_time = time.time() - start_time

        return {
            "algorithm": algorithm,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_scores": grid_search.cv_results_["mean_test_score"],
            "optimization_time": optimization_time,
            "total_trials": len(grid_search.cv_results_["mean_test_score"]),
            "successful_trials": len(grid_search.cv_results_["mean_test_score"]),
        }

    async def _bayesian_optimization(
        self,
        dataset: Dataset,
        algorithm: str,
        algorithm_config: dict,
        config: OptimizationConfig,
    ) -> dict[str, Any]:
        """Perform Bayesian optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            raise AutoMLError("Optuna is required for Bayesian optimization")

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=config.random_state),
            pruner=MedianPruner(n_startup_trials=5) if config.enable_pruning else None,
        )

        # Define objective function
        def objective(trial):
            return self._optuna_objective(
                trial, dataset, algorithm, algorithm_config, config
            )

        # Optimize
        start_time = time.time()
        study.optimize(
            objective,
            n_trials=config.max_trials,
            timeout=config.max_time_seconds,
        )
        optimization_time = time.time() - start_time

        return {
            "algorithm": algorithm,
            "best_params": study.best_params,
            "best_score": study.best_value,
            "cv_scores": [
                trial.value for trial in study.trials if trial.value is not None
            ],
            "optimization_time": optimization_time,
            "total_trials": len(study.trials),
            "successful_trials": len([t for t in study.trials if t.value is not None]),
        }

    async def _hyperband_optimization(
        self,
        dataset: Dataset,
        algorithm: str,
        algorithm_config: dict,
        config: OptimizationConfig,
    ) -> dict[str, Any]:
        """Perform Hyperband optimization."""
        # For now, fallback to Bayesian optimization
        # Full Hyperband implementation would require more complex resource allocation
        return await self._bayesian_optimization(
            dataset, algorithm, algorithm_config, config
        )

    def _optuna_objective(
        self,
        trial,
        dataset: Dataset,
        algorithm: str,
        algorithm_config: dict,
        config: OptimizationConfig,
    ) -> float:
        """Objective function for Optuna optimization."""
        # Generate parameters
        params = {}
        param_space = algorithm_config["param_space"]

        for param_name, param_config in param_space.items():
            if param_config["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name, param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )

        # Create and evaluate model
        estimator = self._create_estimator(algorithm, params)

        # Prepare data
        X = dataset.data.values
        y = self._generate_synthetic_labels(
            X, contamination=params.get("contamination", 0.1)
        )

        # Perform cross-validation
        cv_scores = cross_val_score(
            estimator, X, y, cv=config.cv_folds, scoring=config.scoring_metric
        )

        return cv_scores.mean()

    async def _create_ensemble(
        self, optimization_results: list[dict[str, Any]], config: EnsembleConfig
    ) -> dict[str, Any]:
        """Create ensemble from optimization results."""
        return await self.ensemble_manager.create_ensemble(optimization_results, config)

    async def _evaluate_final_model(
        self,
        ensemble_result: dict[str, Any],
        dataset: Dataset,
        config: OptimizationConfig,
    ) -> dict[str, Any]:
        """Evaluate the final model/ensemble."""
        # This would involve comprehensive evaluation
        # For now, return the ensemble result
        return ensemble_result

    def _create_estimator(self, algorithm: str, params: dict[str, Any]):
        """Create estimator for the algorithm."""
        # This would integrate with the existing adapter system
        # For now, create mock estimators
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.svm import OneClassSVM

        if algorithm == "IsolationForest":
            return IsolationForest(**params)
        elif algorithm == "LocalOutlierFactor":
            return LocalOutlierFactor(**params)
        elif algorithm == "OneClassSVM":
            return OneClassSVM(**params)
        else:
            # Default to Isolation Forest
            return IsolationForest(**params)

    def _generate_synthetic_labels(
        self, X: np.ndarray, contamination: float = 0.1
    ) -> np.ndarray:
        """Generate synthetic labels for evaluation."""
        n_samples = X.shape[0]
        n_outliers = int(n_samples * contamination)

        # Create labels (0 = normal, 1 = outlier)
        y = np.zeros(n_samples)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        y[outlier_indices] = 1

        return y

    def _convert_param_space_to_random(
        self, param_space: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert parameter space to random search format."""
        if not SCIPY_AVAILABLE:
            return {}

        random_space = {}
        for param_name, param_config in param_space.items():
            if param_config["type"] == "int":
                random_space[param_name] = randint(
                    param_config["low"], param_config["high"]
                )
            elif param_config["type"] == "float":
                random_space[param_name] = uniform(
                    param_config["low"], param_config["high"] - param_config["low"]
                )
            elif param_config["type"] == "categorical":
                random_space[param_name] = param_config["choices"]

        return random_space

    def _convert_param_space_to_grid(
        self, param_space: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert parameter space to grid search format."""
        grid_space = {}
        for param_name, param_config in param_space.items():
            if param_config["type"] == "int":
                # Create 3-5 values in the range
                low, high = param_config["low"], param_config["high"]
                grid_space[param_name] = np.linspace(
                    low, high, min(5, high - low + 1), dtype=int
                ).tolist()
            elif param_config["type"] == "float":
                # Create 3-5 values in the range
                low, high = param_config["low"], param_config["high"]
                grid_space[param_name] = np.linspace(low, high, 5).tolist()
            elif param_config["type"] == "categorical":
                grid_space[param_name] = param_config["choices"]

        return grid_space

    def _get_meta_learning_suggestions(self, dataset: Dataset) -> list[str]:
        """Get algorithm suggestions from meta-learning."""
        # Simple meta-learning based on dataset characteristics
        n_samples, n_features = dataset.data.shape

        suggestions = []
        for record in self.meta_learning_records:
            # Check similarity
            size_similarity = abs(record.sample_count - n_samples) / max(
                record.sample_count, n_samples
            )
            feature_similarity = abs(record.feature_count - n_features) / max(
                record.feature_count, n_features
            )

            if size_similarity < 0.5 and feature_similarity < 0.5:
                suggestions.append(record.algorithm)

        return list(set(suggestions))

    async def _update_meta_learning(self, dataset: Dataset, result: ExperimentResult):
        """Update meta-learning records."""
        dataset_signature = self._compute_dataset_signature(dataset)

        record = MetaLearningRecord(
            dataset_signature=dataset_signature,
            algorithm=result.algorithm,
            hyperparameters=result.best_params,
            performance=result.best_score,
            optimization_time=result.optimization_time,
            feature_count=dataset.data.shape[1],
            sample_count=dataset.data.shape[0],
            dataset_characteristics=self._extract_dataset_characteristics(dataset),
        )

        self.meta_learning_records.append(record)
        self._save_meta_learning_record(record)

    def _compute_dataset_signature(self, dataset: Dataset) -> str:
        """Compute unique signature for dataset."""
        # Simple signature based on basic statistics
        data = dataset.data.values
        signature_parts = [
            f"shape_{data.shape[0]}_{data.shape[1]}",
            f"mean_{np.mean(data):.4f}",
            f"std_{np.std(data):.4f}",
            f"min_{np.min(data):.4f}",
            f"max_{np.max(data):.4f}",
        ]
        return "_".join(signature_parts)

    def _extract_dataset_characteristics(self, dataset: Dataset) -> dict[str, Any]:
        """Extract characteristics from dataset."""
        data = dataset.data.values
        return {
            "n_samples": data.shape[0],
            "n_features": data.shape[1],
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "sparsity": float(np.count_nonzero(data == 0) / data.size),
        }

    def _save_experiment(self, experiment: ExperimentResult):
        """Save experiment to storage."""
        experiment_file = (
            self.storage_path / f"experiment_{experiment.experiment_id}.json"
        )

        try:
            with open(experiment_file, "w") as f:
                json.dump(experiment.__dict__, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save experiment: {e}")

    def _save_meta_learning_record(self, record: MetaLearningRecord):
        """Save meta-learning record to storage."""
        records_file = self.storage_path / "meta_learning_records.json"

        try:
            # Load existing records
            if records_file.exists():
                with open(records_file) as f:
                    existing_records = json.load(f)
            else:
                existing_records = []

            # Add new record
            existing_records.append(record.__dict__)

            # Save updated records
            with open(records_file, "w") as f:
                json.dump(existing_records, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save meta-learning record: {e}")

    def _load_historical_data(self):
        """Load historical experiments and meta-learning data."""
        # Load experiments
        for experiment_file in self.storage_path.glob("experiment_*.json"):
            try:
                with open(experiment_file) as f:
                    experiment_data = json.load(f)
                # Convert back to ExperimentResult if needed
                # For now, just log that we found historical data
                logger.info(f"Found historical experiment: {experiment_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load experiment {experiment_file}: {e}")

        # Load meta-learning records
        records_file = self.storage_path / "meta_learning_records.json"
        if records_file.exists():
            try:
                with open(records_file) as f:
                    records_data = json.load(f)
                # Convert back to MetaLearningRecord if needed
                logger.info(f"Loaded {len(records_data)} meta-learning records")
            except Exception as e:
                logger.warning(f"Failed to load meta-learning records: {e}")

    def get_experiment_history(self) -> list[ExperimentResult]:
        """Get experiment history."""
        return self.experiments

    def get_meta_learning_insights(self) -> dict[str, Any]:
        """Get insights from meta-learning."""
        if not self.meta_learning_records:
            return {"message": "No meta-learning data available"}

        # Analyze algorithm performance
        algorithm_performance = {}
        for record in self.meta_learning_records:
            if record.algorithm not in algorithm_performance:
                algorithm_performance[record.algorithm] = []
            algorithm_performance[record.algorithm].append(record.performance)

        # Calculate statistics
        insights = {}
        for algorithm, performances in algorithm_performance.items():
            insights[algorithm] = {
                "mean_performance": np.mean(performances),
                "std_performance": np.std(performances),
                "best_performance": np.max(performances),
                "worst_performance": np.min(performances),
                "total_experiments": len(performances),
            }

        return insights


class FeatureEngineer:
    """Feature engineering component."""

    async def engineer_features(
        self, dataset: Dataset, config: FeatureEngineeringConfig
    ) -> Dataset:
        """Apply feature engineering to dataset."""
        logger.info("Applying feature engineering")

        data = dataset.data.copy()

        # Apply each method
        for method in config.methods:
            if method == FeatureEngineeringMethod.STATISTICAL:
                data = self._add_statistical_features(data)
            elif method == FeatureEngineeringMethod.INTERACTION:
                data = self._add_interaction_features(data, config)
            elif method == FeatureEngineeringMethod.POLYNOMIAL:
                data = self._add_polynomial_features(data, config)

        # Feature selection
        if config.selection_method == "univariate":
            data = self._select_features_univariate(data, config)

        # Feature scaling
        if config.enable_scaling:
            data = self._scale_features(data)

        # Create new dataset
        engineered_dataset = Dataset(
            name=f"{dataset.name}_engineered",
            data=data,
            features=data.columns.tolist() if hasattr(data, "columns") else None,
            metadata={**dataset.metadata, "feature_engineering": True},
        )

        logger.info(f"Feature engineering completed: {data.shape[1]} features")
        return engineered_dataset

    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        # Add row-wise statistics
        data["row_mean"] = data.mean(axis=1)
        data["row_std"] = data.std(axis=1)
        data["row_min"] = data.min(axis=1)
        data["row_max"] = data.max(axis=1)
        data["row_range"] = data["row_max"] - data["row_min"]

        return data

    def _add_interaction_features(
        self, data: pd.DataFrame, config: FeatureEngineeringConfig
    ) -> pd.DataFrame:
        """Add interaction features."""
        # Add pairwise interactions for top features
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        # Limit number of interactions
        max_interactions = min(50, len(numeric_cols) * (len(numeric_cols) - 1) // 2)
        interaction_count = 0

        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i + 1 :], i + 1):
                if interaction_count >= max_interactions:
                    break

                # Create interaction feature
                interaction_name = f"{col1}_x_{col2}"
                data[interaction_name] = data[col1] * data[col2]
                interaction_count += 1

            if interaction_count >= max_interactions:
                break

        return data

    def _add_polynomial_features(
        self, data: pd.DataFrame, config: FeatureEngineeringConfig
    ) -> pd.DataFrame:
        """Add polynomial features."""
        if not SKLEARN_AVAILABLE:
            return data

        from sklearn.preprocessing import PolynomialFeatures

        # Apply to first few columns to avoid explosion
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:10]
        poly_data = data[numeric_cols]

        poly = PolynomialFeatures(
            degree=config.polynomial_degree,
            interaction_only=config.interaction_only,
            include_bias=False,
        )

        poly_features = poly.fit_transform(poly_data)

        # Create new column names
        poly_feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]

        # Add polynomial features
        poly_df = pd.DataFrame(
            poly_features, columns=poly_feature_names, index=data.index
        )

        return pd.concat([data, poly_df], axis=1)

    def _select_features_univariate(
        self, data: pd.DataFrame, config: FeatureEngineeringConfig
    ) -> pd.DataFrame:
        """Select features using univariate selection."""
        if not SKLEARN_AVAILABLE:
            return data

        # Generate synthetic labels for selection
        y = np.random.choice([0, 1], size=len(data), p=[0.9, 0.1])

        # Apply feature selection
        selector = SelectKBest(
            score_func=f_classif, k=min(config.selection_k, data.shape[1])
        )
        selected_features = selector.fit_transform(data, y)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_columns = data.columns[selected_mask]

        return data[selected_columns]

    def _scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features."""
        if not SKLEARN_AVAILABLE:
            return data

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        return pd.DataFrame(scaled_data, columns=data.columns, index=data.index)


class EnsembleManager:
    """Ensemble management component."""

    async def create_ensemble(
        self, optimization_results: list[dict[str, Any]], config: EnsembleConfig
    ) -> dict[str, Any]:
        """Create ensemble from optimization results."""
        logger.info("Creating ensemble from optimization results")

        # Filter results by performance threshold
        good_results = [
            r
            for r in optimization_results
            if r.get("best_score", 0) >= config.performance_threshold
        ]

        if len(good_results) < config.min_models:
            logger.warning(f"Not enough good models ({len(good_results)}) for ensemble")
            # Return best single model
            best_result = max(
                optimization_results, key=lambda x: x.get("best_score", 0)
            )
            return {
                "algorithm": best_result["algorithm"],
                "params": best_result["best_params"],
                "score": best_result["best_score"],
                "cv_scores": best_result.get("cv_scores", []),
                "ensemble_config": None,
            }

        # Select diverse models
        selected_models = self._select_diverse_models(good_results, config)

        # Create ensemble configuration
        ensemble_config = {
            "method": config.methods[0].value,  # Use first method
            "models": selected_models,
            "voting_strategy": config.voting_strategy,
            "weights": self._calculate_ensemble_weights(selected_models),
        }

        # Calculate ensemble score (weighted average)
        ensemble_score = np.average(
            [model["best_score"] for model in selected_models],
            weights=ensemble_config["weights"],
        )

        return {
            "algorithm": "ensemble",
            "params": {},
            "score": ensemble_score,
            "cv_scores": [],
            "ensemble_config": ensemble_config,
            "metadata": {
                "ensemble_size": len(selected_models),
                "base_algorithms": [model["algorithm"] for model in selected_models],
            },
        }

    def _select_diverse_models(
        self, results: list[dict[str, Any]], config: EnsembleConfig
    ) -> list[dict[str, Any]]:
        """Select diverse models for ensemble."""
        # Sort by performance
        sorted_results = sorted(
            results, key=lambda x: x.get("best_score", 0), reverse=True
        )

        # Select top models up to max_models
        selected = []
        for result in sorted_results:
            if len(selected) >= config.max_models:
                break

            # Check diversity
            if self._is_diverse_enough(result, selected, config.diversity_threshold):
                selected.append(result)

        return selected

    def _is_diverse_enough(
        self,
        candidate: dict[str, Any],
        selected: list[dict[str, Any]],
        threshold: float,
    ) -> bool:
        """Check if candidate is diverse enough from selected models."""
        if not selected:
            return True

        # Simple diversity check based on algorithm type
        candidate_algo = candidate["algorithm"]
        selected_algos = [model["algorithm"] for model in selected]

        # If algorithm is already present, check parameter diversity
        if candidate_algo in selected_algos:
            # For now, consider it not diverse enough
            return False

        return True

    def _calculate_ensemble_weights(self, models: list[dict[str, Any]]) -> list[float]:
        """Calculate weights for ensemble models."""
        scores = [model.get("best_score", 0) for model in models]

        # Normalize scores to weights
        total_score = sum(scores)
        if total_score == 0:
            return [1.0 / len(models)] * len(models)

        weights = [score / total_score for score in scores]
        return weights

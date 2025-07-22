"""AutoML service for automated machine learning model selection and optimization."""

import itertools
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from optuna import Trial, create_study
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

from ...infrastructure.monitoring.distributed_tracing import trace_operation
from ..entities.dataset import Dataset
from ..entities.prediction_result import PredictionResult
from .advanced_prediction_service import (
    AlgorithmConfig,
    PredictionAlgorithm,
    EnsembleConfig,
    get_prediction_service,
)

logger = logging.getLogger(__name__)


class OptimizationMetric(Enum):
    """Metrics for model optimization."""

    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "auc"
    BALANCED_ACCURACY = "balanced_accuracy"
    ANOMALY_RATE = "anomaly_rate"
    EXECUTION_TIME = "execution_time"


class SearchStrategy(Enum):
    """Hyperparameter search strategies."""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class OptimizationConfig:
    """Configuration for AutoML optimization."""

    # Search strategy
    search_strategy: SearchStrategy = SearchStrategy.BAYESIAN_OPTIMIZATION
    max_trials: int = 100
    timeout_seconds: int | None = None

    # Optimization targets
    primary_metric: OptimizationMetric = OptimizationMetric.F1_SCORE
    secondary_metrics: list[OptimizationMetric] = field(
        default_factory=lambda: [OptimizationMetric.AUC]
    )

    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = "f1"

    # Algorithm selection
    algorithms_to_test: list[PredictionAlgorithm] | None = None
    ensemble_methods: bool = True
    max_ensemble_size: int = 5

    # Resource constraints
    max_memory_mb: int | None = None
    max_execution_time_per_trial: int | None = 300  # 5 minutes

    # Early stopping
    enable_pruning: bool = True
    patience: int = 20

    # Reproducibility
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class OptimizationResult:
    """Result of AutoML optimization."""

    best_algorithm: PredictionAlgorithm
    best_config: AlgorithmConfig
    best_score: float
    best_metrics: dict[str, float]

    # Optimization history
    trial_history: list[dict[str, Any]] = field(default_factory=list)
    total_trials: int = 0
    optimization_time: float = 0.0

    # Alternative configurations
    top_k_results: list[tuple[PredictionAlgorithm, AlgorithmConfig, float]] = field(
        default_factory=list
    )

    # Ensemble recommendation
    ensemble_config: EnsembleConfig | None = None
    ensemble_score: float | None = None


class AutoMLService:
    """AutoML service for automated machine learning optimization."""

    # JSON schemas for PyOD algorithms' search spaces
    PYOD_PARAMETER_SPACES = {
        "KNN": {
            "type": "object",
            "properties": {
                "n_neighbors": {"type": "integer", "enum": [3, 5, 10, 20]},
                "method": {"type": "string", "enum": ["largest", "median", "mean"]},
                "contamination": {"type": "number", "enum": [0.05, 0.1, 0.15]},
            },
            "required": ["n_neighbors", "method", "contamination"],
        },
        # Add more algorithms as needed
    }

    def __init__(self):
        """Initialize AutoML service."""
        self.prediction_service = get_prediction_service()
        self.optimization_history = []

        # Parameter search spaces for different algorithms
        self.parameter_spaces = self._initialize_parameter_spaces()

        logger.info("AutoML service initialized")

    def _initialize_parameter_spaces(self) -> dict[PredictionAlgorithm, dict[str, Any]]:
        """Initialize hyperparameter search spaces for algorithms."""
        spaces = {}

        # Isolation Forest
        spaces[PredictionAlgorithm.ISOLATION_FOREST] = {
            "n_estimators": [50, 100, 200, 300],
            "max_samples": ["auto", 0.5, 0.7, 0.9],
            "max_features": [0.5, 0.7, 0.9, 1.0],
            "contamination": [0.05, 0.1, 0.15, 0.2, 0.25],
        }

        # Local Outlier Factor
        spaces[PredictionAlgorithm.LOCAL_OUTLIER_FACTOR] = {
            "n_neighbors": [5, 10, 20, 30, 50],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
            "contamination": [0.05, 0.1, 0.15, 0.2, 0.25],
        }

        # One-Class SVM
        spaces[PredictionAlgorithm.ONE_CLASS_SVM] = {
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            "nu": [0.01, 0.05, 0.1, 0.2, 0.3],
            "degree": [2, 3, 4],  # For poly kernel
        }

        # DBSCAN
        spaces[PredictionAlgorithm.DBSCAN] = {
            "eps": [0.1, 0.3, 0.5, 0.7, 1.0, 1.5],
            "min_samples": [3, 5, 10, 15, 20],
            "metric": ["euclidean", "manhattan", "chebyshev"],
        }

        # KNN (PyOD)
        if True:  # Assuming PyOD might be available
            spaces[PredictionAlgorithm.KNN] = {
                "n_neighbors": [3, 5, 10, 15, 20, 30],
                "method": ["largest", "mean", "median"],
                "contamination": [0.05, 0.1, 0.15, 0.2, 0.25],
            }

            # AutoEncoder (PyOD)
            spaces[PredictionAlgorithm.AUTO_ENCODER] = {
                "hidden_neurons": [
                    [32, 16, 16, 32],
                    [64, 32, 32, 64],
                    [128, 64, 64, 128],
                ],
                "epochs": [50, 100, 200],
                "batch_size": [16, 32, 64],
                "dropout_rate": [0.1, 0.2, 0.3],
                "contamination": [0.05, 0.1, 0.15, 0.2, 0.25],
            }

        return spaces

    @trace_operation("automl_optimization")
    async def optimize_prediction(
        self,
        dataset: Dataset,
        optimization_config: OptimizationConfig | None = None,
        ground_truth: np.ndarray | None = None,
    ) -> OptimizationResult:
        """Automatically optimize machine learning predictions for the given dataset."""

        if optimization_config is None:
            optimization_config = OptimizationConfig()

        start_time = time.time()

        try:
            # Get available algorithms
            available_algorithms = (
                await self.prediction_service.get_available_algorithms()
            )

            # Filter algorithms if specified
            if optimization_config.algorithms_to_test:
                algorithms_to_test = [
                    alg
                    for alg in optimization_config.algorithms_to_test
                    if alg in available_algorithms
                ]
            else:
                # Exclude ensemble methods from individual optimization
                algorithms_to_test = [
                    alg
                    for alg in available_algorithms
                    if not alg.value.startswith("ensemble_")
                ]

            if not algorithms_to_test:
                raise ValueError("No algorithms available for optimization")

            logger.info(
                f"Starting AutoML optimization with {len(algorithms_to_test)} algorithms"
            )

            # Perform optimization based on strategy
            if optimization_config.search_strategy == SearchStrategy.GRID_SEARCH:
                result = await self._grid_search_optimization(
                    dataset, algorithms_to_test, optimization_config, ground_truth
                )
            elif optimization_config.search_strategy == SearchStrategy.RANDOM_SEARCH:
                result = await self._random_search_optimization(
                    dataset, algorithms_to_test, optimization_config, ground_truth
                )
            elif (
                optimization_config.search_strategy
                == SearchStrategy.BAYESIAN_OPTIMIZATION
            ):
                result = await self._bayesian_optimization(
                    dataset, algorithms_to_test, optimization_config, ground_truth
                )
            else:
                # Fallback to grid search
                result = await self._grid_search_optimization(
                    dataset, algorithms_to_test, optimization_config, ground_truth
                )

            # Optimize ensemble if requested
            if optimization_config.ensemble_methods and len(algorithms_to_test) >= 2:
                ensemble_result = await self._optimize_ensemble(
                    dataset,
                    algorithms_to_test,
                    optimization_config,
                    ground_truth,
                    result,
                )
                if ensemble_result and ensemble_result[1] > result.best_score:
                    result.ensemble_config = ensemble_result[0]
                    result.ensemble_score = ensemble_result[1]

            result.optimization_time = time.time() - start_time

            logger.info(
                f"AutoML optimization completed in {result.optimization_time:.2f}s"
            )
            logger.info(
                f"Best algorithm: {result.best_algorithm.value} (score: {result.best_score:.4f})"
            )

            return result

        except Exception as e:
            logger.error(f"Error in AutoML optimization: {e}")
            raise

    async def _grid_search_optimization(
        self,
        dataset: Dataset,
        algorithms: list[PredictionAlgorithm],
        config: OptimizationConfig,
        ground_truth: np.ndarray | None,
    ) -> OptimizationResult:
        """Perform grid search optimization."""

        best_score = -np.inf
        best_algorithm = None
        best_config = None
        best_metrics = {}
        trial_history = []

        trial_count = 0

        for algorithm in algorithms:
            logger.info(f"Grid searching algorithm: {algorithm.value}")

            # Get parameter space for this algorithm
            param_space = self.parameter_spaces.get(algorithm, {})
            if not param_space:
                continue

            # Generate parameter grid
            param_grid = ParameterGrid(param_space)

            for params in param_grid:
                if trial_count >= config.max_trials:
                    break

                try:
                    # Create algorithm config
                    algo_config = AlgorithmConfig(
                        algorithm=algorithm,
                        parameters=params.copy(),
                        random_state=config.random_state,
                    )

                    # Extract contamination if present
                    if "contamination" in params:
                        algo_config.contamination = params.pop("contamination")

                    # Evaluate configuration
                    score, metrics = await self._evaluate_configuration(
                        dataset, algo_config, config, ground_truth
                    )

                    # Track trial
                    trial_history.append(
                        {
                            "trial": trial_count,
                            "algorithm": algorithm.value,
                            "parameters": params,
                            "score": score,
                            "metrics": metrics,
                        }
                    )

                    # Update best if better
                    if score > best_score:
                        best_score = score
                        best_algorithm = algorithm
                        best_config = algo_config
                        best_metrics = metrics

                    trial_count += 1

                except Exception as e:
                    logger.warning(f"Trial failed for {algorithm.value}: {e}")
                    continue

        # Create top-k results
        top_k_results = []
        sorted_trials = sorted(trial_history, key=lambda x: x["score"], reverse=True)
        for trial in sorted_trials[:5]:  # Top 5
            algo = PredictionAlgorithm(trial["algorithm"])
            cfg = AlgorithmConfig(algorithm=algo, parameters=trial["parameters"])
            top_k_results.append((algo, cfg, trial["score"]))

        return OptimizationResult(
            best_algorithm=best_algorithm,
            best_config=best_config,
            best_score=best_score,
            best_metrics=best_metrics,
            trial_history=trial_history,
            total_trials=trial_count,
            top_k_results=top_k_results,
        )

    async def _random_search_optimization(
        self,
        dataset: Dataset,
        algorithms: list[PredictionAlgorithm],
        config: OptimizationConfig,
        ground_truth: np.ndarray | None,
    ) -> OptimizationResult:
        """Perform random search optimization."""

        best_score = -np.inf
        best_algorithm = None
        best_config = None
        best_metrics = {}
        trial_history = []

        np.random.seed(config.random_state)

        for trial in range(config.max_trials):
            # Randomly select algorithm
            algorithm = np.random.choice(algorithms)

            # Get parameter space
            param_space = self.parameter_spaces.get(algorithm, {})
            if not param_space:
                continue

            try:
                # Sample random parameters
                params = {}
                for param_name, param_values in param_space.items():
                    if isinstance(param_values, list):
                        params[param_name] = np.random.choice(param_values)
                    elif isinstance(param_values, tuple) and len(param_values) == 2:
                        # Assume (min, max) range
                        params[param_name] = np.random.uniform(
                            param_values[0], param_values[1]
                        )

                # Create algorithm config
                algo_config = AlgorithmConfig(
                    algorithm=algorithm,
                    parameters=params.copy(),
                    random_state=config.random_state,
                )

                # Extract contamination if present
                if "contamination" in params:
                    algo_config.contamination = params.pop("contamination")

                # Evaluate configuration
                score, metrics = await self._evaluate_configuration(
                    dataset, algo_config, config, ground_truth
                )

                # Track trial
                trial_history.append(
                    {
                        "trial": trial,
                        "algorithm": algorithm.value,
                        "parameters": params,
                        "score": score,
                        "metrics": metrics,
                    }
                )

                # Update best if better
                if score > best_score:
                    best_score = score
                    best_algorithm = algorithm
                    best_config = algo_config
                    best_metrics = metrics

            except Exception as e:
                logger.warning(f"Random search trial {trial} failed: {e}")
                continue

        # Create top-k results
        top_k_results = []
        sorted_trials = sorted(trial_history, key=lambda x: x["score"], reverse=True)
        for trial in sorted_trials[:5]:
            algo = PredictionAlgorithm(trial["algorithm"])
            cfg = AlgorithmConfig(algorithm=algo, parameters=trial["parameters"])
            top_k_results.append((algo, cfg, trial["score"]))

        return OptimizationResult(
            best_algorithm=best_algorithm,
            best_config=best_config,
            best_score=best_score,
            best_metrics=best_metrics,
            trial_history=trial_history,
            total_trials=len(trial_history),
            top_k_results=top_k_results,
        )

    async def _bayesian_optimization(
        self,
        dataset: Dataset,
        algorithms: list[PredictionAlgorithm],
        config: OptimizationConfig,
        ground_truth: np.ndarray | None,
    ) -> OptimizationResult:
        """Perform Bayesian optimization using Optuna."""

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to random search")
            return await self._random_search_optimization(
                dataset, algorithms, config, ground_truth
            )

        best_score = -np.inf
        best_algorithm = None
        best_config = None
        best_metrics = {}
        trial_history = []

        def objective(trial: Trial) -> float:
            nonlocal best_score, best_algorithm, best_config, best_metrics

            # Select algorithm
            algorithm_name = trial.suggest_categorical(
                "algorithm", [alg.value for alg in algorithms]
            )
            algorithm = PredictionAlgorithm(algorithm_name)

            # Get parameter space for this algorithm
            param_space = self.parameter_spaces.get(algorithm, {})

            # Suggest parameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        # Numeric values
                        params[param_name] = trial.suggest_categorical(
                            f"{algorithm_name}_{param_name}", param_values
                        )
                    else:
                        # Categorical values
                        params[param_name] = trial.suggest_categorical(
                            f"{algorithm_name}_{param_name}", param_values
                        )

            try:
                # Create algorithm config
                algo_config = AlgorithmConfig(
                    algorithm=algorithm,
                    parameters=params.copy(),
                    random_state=config.random_state,
                )

                # Extract contamination if present
                if "contamination" in params:
                    algo_config.contamination = params.pop("contamination")

                # Evaluate configuration (need to run sync)
                import asyncio

                loop = asyncio.get_event_loop()
                score, metrics = loop.run_until_complete(
                    self._evaluate_configuration(
                        dataset, algo_config, config, ground_truth
                    )
                )

                # Track trial
                trial_history.append(
                    {
                        "trial": trial.number,
                        "algorithm": algorithm.value,
                        "parameters": params,
                        "score": score,
                        "metrics": metrics,
                    }
                )

                # Update best if better
                if score > best_score:
                    best_score = score
                    best_algorithm = algorithm
                    best_config = algo_config
                    best_metrics = metrics

                return score

            except Exception as e:
                logger.warning(f"Bayesian optimization trial failed: {e}")
                return -np.inf

        # Create study
        study = create_study(
            direction="maximize",
            sampler=TPESampler(seed=config.random_state),
            pruner=MedianPruner() if config.enable_pruning else None,
        )

        # Optimize
        study.optimize(
            objective, n_trials=config.max_trials, timeout=config.timeout_seconds
        )

        # Create top-k results
        top_k_results = []
        sorted_trials = sorted(trial_history, key=lambda x: x["score"], reverse=True)
        for trial in sorted_trials[:5]:
            algo = PredictionAlgorithm(trial["algorithm"])
            cfg = AlgorithmConfig(algorithm=algo, parameters=trial["parameters"])
            top_k_results.append((algo, cfg, trial["score"]))

        return OptimizationResult(
            best_algorithm=best_algorithm,
            best_config=best_config,
            best_score=best_score,
            best_metrics=best_metrics,
            trial_history=trial_history,
            total_trials=len(trial_history),
            top_k_results=top_k_results,
        )

    async def _evaluate_configuration(
        self,
        dataset: Dataset,
        config: AlgorithmConfig,
        opt_config: OptimizationConfig,
        ground_truth: np.ndarray | None,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate a single algorithm configuration."""

        try:
            # Run prediction
            result = await self.prediction_service.predict(
                dataset, config.algorithm, config, ground_truth
            )

            # Extract metrics
            metrics = {}
            if "metrics" in result.metadata:
                metrics_obj = result.metadata["metrics"]
                metrics = {
                    "execution_time": metrics_obj.get("execution_time", 0.0),
                    "memory_usage": metrics_obj.get("memory_usage", 0.0),
                    "anomaly_rate": metrics_obj.get("anomaly_rate", 0.0),
                    "precision": metrics_obj.get("precision", 0.0),
                    "recall": metrics_obj.get("recall", 0.0),
                    "f1_score": metrics_obj.get("f1_score", 0.0),
                    "auc_score": metrics_obj.get("auc_score", 0.0),
                }

            # Calculate primary score
            if opt_config.primary_metric == OptimizationMetric.F1_SCORE:
                score = metrics.get("f1_score", 0.0)
            elif opt_config.primary_metric == OptimizationMetric.PRECISION:
                score = metrics.get("precision", 0.0)
            elif opt_config.primary_metric == OptimizationMetric.RECALL:
                score = metrics.get("recall", 0.0)
            elif opt_config.primary_metric == OptimizationMetric.AUC:
                score = metrics.get("auc_score", 0.0)
            elif opt_config.primary_metric == OptimizationMetric.ANOMALY_RATE:
                # For anomaly rate, we want something reasonable (not too high, not too low)
                rate = metrics.get("anomaly_rate", 0.0)
                # Penalty for rates outside reasonable range (5-25%)
                if 0.05 <= rate <= 0.25:
                    score = 1.0 - abs(rate - 0.1)  # Optimal around 10%
                else:
                    score = max(0.0, 1.0 - abs(rate - 0.1) * 2)
            elif opt_config.primary_metric == OptimizationMetric.EXECUTION_TIME:
                # For execution time, invert (lower is better)
                exec_time = metrics.get("execution_time", float("inf"))
                score = 1.0 / (1.0 + exec_time)
            else:
                # Default to F1 score
                score = metrics.get("f1_score", 0.0)

            return score, metrics

        except Exception as e:
            logger.warning(f"Configuration evaluation failed: {e}")
            return 0.0, {}

    async def _optimize_ensemble(
        self,
        dataset: Dataset,
        algorithms: list[PredictionAlgorithm],
        config: OptimizationConfig,
        ground_truth: np.ndarray | None,
        individual_results: OptimizationResult,
    ) -> tuple[EnsembleConfig, float] | None:
        """Optimize ensemble configuration."""

        logger.info("Optimizing ensemble configuration")

        # Select top algorithms for ensemble
        top_algorithms = [
            result[0]
            for result in individual_results.top_k_results[: config.max_ensemble_size]
        ]

        if len(top_algorithms) < 2:
            return None

        best_ensemble_score = -np.inf
        best_ensemble_config = None

        # Try different ensemble combinations
        for ensemble_size in range(
            2, min(len(top_algorithms) + 1, config.max_ensemble_size + 1)
        ):
            for algorithm_combo in itertools.combinations(
                top_algorithms, ensemble_size
            ):
                for combination_method in ["average", "voting", "weighted_average"]:
                    try:
                        # Create ensemble config
                        algorithm_configs = []
                        for alg in algorithm_combo:
                            # Use best config for this algorithm
                            best_config_for_alg = next(
                                (
                                    result[1]
                                    for result in individual_results.top_k_results
                                    if result[0] == alg
                                ),
                                AlgorithmConfig(algorithm=alg),
                            )
                            algorithm_configs.append(best_config_for_alg)

                        ensemble_config = EnsembleConfig(
                            algorithms=algorithm_configs,
                            combination_method=combination_method,
                        )

                        # Evaluate ensemble
                        result = await self.prediction_service.predict_ensemble(
                            dataset, ensemble_config, ground_truth
                        )

                        # Calculate score (using same metric as individual optimization)
                        if ground_truth is not None:
                            # Calculate metrics from result
                            predictions = np.zeros(result.total_samples)
                            scores = np.zeros(result.total_samples)

                            for anomaly in result.anomalies:
                                predictions[anomaly.index] = 1
                                scores[anomaly.index] = anomaly.score.value

                            if config.primary_metric == OptimizationMetric.F1_SCORE:
                                score = f1_score(ground_truth, predictions)
                            elif config.primary_metric == OptimizationMetric.AUC:
                                score = roc_auc_score(ground_truth, scores)
                            else:
                                score = f1_score(ground_truth, predictions)  # Default
                        else:
                            # Without ground truth, use anomaly rate as proxy
                            score = 1.0 - abs(result.contamination_rate.value - 0.1)

                        # Update best ensemble
                        if score > best_ensemble_score:
                            best_ensemble_score = score
                            best_ensemble_config = ensemble_config

                    except Exception as e:
                        logger.warning(f"Ensemble evaluation failed: {e}")
                        continue

        if best_ensemble_config:
            logger.info(f"Best ensemble score: {best_ensemble_score:.4f}")
            return best_ensemble_config, best_ensemble_score

        return None

    async def auto_select_algorithm(
        self,
        dataset: Dataset,
        ground_truth: np.ndarray | None = None,
        quick_mode: bool = False,
    ) -> tuple[PredictionAlgorithm, AlgorithmConfig]:
        """Automatically select the best algorithm for a dataset."""

        if quick_mode:
            # Quick heuristic-based selection
            return await self._quick_algorithm_selection(dataset)
        else:
            # Full optimization
            config = OptimizationConfig(
                max_trials=50,  # Reduced for faster selection
                search_strategy=SearchStrategy.RANDOM_SEARCH,
            )

            result = await self.optimize_prediction(dataset, config, ground_truth)
            return result.best_algorithm, result.best_config

    async def _quick_algorithm_selection(
        self, dataset: Dataset
    ) -> tuple[PredictionAlgorithm, AlgorithmConfig]:
        """Quick algorithm selection based on dataset characteristics."""

        # Get data characteristics
        if hasattr(dataset, "data"):
            data = dataset.data
            if isinstance(data, pd.DataFrame):
                n_samples, n_features = data.shape
                numeric_features = len(data.select_dtypes(include=[np.number]).columns)
            else:
                n_samples, n_features = (
                    data.shape if len(data.shape) == 2 else (len(data), 1)
                )
                numeric_features = n_features
        else:
            # Default assumptions
            n_samples, n_features = 1000, 10
            numeric_features = 10

        # Heuristic rules for algorithm selection
        if n_samples < 1000:
            # Small dataset - use simple algorithms
            if n_features <= 10:
                algorithm = PredictionAlgorithm.LOCAL_OUTLIER_FACTOR
            else:
                algorithm = PredictionAlgorithm.ISOLATION_FOREST
        elif n_samples > 100000:
            # Large dataset - use scalable algorithms
            algorithm = PredictionAlgorithm.ISOLATION_FOREST
        else:
            # Medium dataset - use balanced approach
            if n_features > 50:
                algorithm = PredictionAlgorithm.ISOLATION_FOREST
            else:
                algorithm = PredictionAlgorithm.LOCAL_OUTLIER_FACTOR

        # Create default config for selected algorithm
        config = AlgorithmConfig(algorithm=algorithm)

        return algorithm, config

    async def get_optimization_recommendations(
        self, dataset: Dataset, current_results: PredictionResult | None = None
    ) -> dict[str, Any]:
        """Get recommendations for improving prediction performance."""

        recommendations = {
            "data_preprocessing": [],
            "algorithm_suggestions": [],
            "parameter_tuning": [],
            "ensemble_methods": [],
            "general_tips": [],
        }

        # Analyze dataset characteristics
        if hasattr(dataset, "data"):
            data = dataset.data
            if isinstance(data, pd.DataFrame):
                n_samples, n_features = data.shape

                # Data preprocessing recommendations
                if data.isnull().sum().sum() > 0:
                    recommendations["data_preprocessing"].append(
                        "Consider handling missing values with imputation or removal"
                    )

                if n_features > 50:
                    recommendations["data_preprocessing"].append(
                        "Consider dimensionality reduction (PCA) for high-dimensional data"
                    )

                if data.select_dtypes(include=[np.number]).shape[1] < n_features:
                    recommendations["data_preprocessing"].append(
                        "Consider encoding categorical variables or feature selection"
                    )

                # Algorithm suggestions based on data characteristics
                if n_samples < 1000:
                    recommendations["algorithm_suggestions"].append(
                        "Try Local Outlier Factor (LOF) for small datasets"
                    )
                elif n_samples > 100000:
                    recommendations["algorithm_suggestions"].append(
                        "Try Isolation Forest for large datasets due to scalability"
                    )

                if n_features > 20:
                    recommendations["algorithm_suggestions"].append(
                        "Consider AutoEncoder for high-dimensional data"
                    )

        # Analyze current results if provided
        if current_results:
            if current_results.anomaly_count / current_results.total_samples > 0.3:
                recommendations["parameter_tuning"].append(
                    "Anomaly rate seems high - consider reducing contamination parameter"
                )
            elif current_results.anomaly_count / current_results.total_samples < 0.01:
                recommendations["parameter_tuning"].append(
                    "Anomaly rate seems low - consider increasing contamination parameter"
                )

            if current_results.execution_time > 60:  # More than 1 minute
                recommendations["parameter_tuning"].append(
                    "Consider reducing model complexity for faster execution"
                )

        # Ensemble recommendations
        recommendations["ensemble_methods"].append(
            "Try ensemble methods for improved robustness and accuracy"
        )
        recommendations["ensemble_methods"].append(
            "Combine complementary algorithms (e.g., Isolation Forest + LOF)"
        )

        # General tips
        recommendations["general_tips"].extend(
            [
                "Use cross-validation for robust performance estimation",
                "Consider domain-specific feature engineering",
                "Validate results with subject matter experts",
                "Monitor model performance over time for drift",
            ]
        )

        return recommendations


# Global AutoML service instance
_automl_service: AutoMLService | None = None


def get_automl_service() -> AutoMLService:
    """Get the global AutoML service instance."""
    global _automl_service
    if _automl_service is None:
        _automl_service = AutoMLService()
    return _automl_service

"""AutoML service for automated algorithm selection and hyperparameter optimization.

This module provides automated machine learning capabilities for anomaly detection,
including algorithm selection, hyperparameter optimization, and ensemble creation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import AutoMLError

# Optional optimization libraries
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    CmaEsSampler = None
    MedianPruner = None
    OPTUNA_AVAILABLE = False

try:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    roc_auc_score = None
    precision_score = None
    recall_score = None
    f1_score = None
    cross_val_score = None
    StratifiedKFold = None
    SKLEARN_AVAILABLE = False

try:
    import hyperopt
    from hyperopt import hp, fmin, tpe, Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    hyperopt = None
    hp = None
    fmin = None
    tpe = None
    Trials = None
    HYPEROPT_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for AutoML."""

    AUC = "auc"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    BALANCED_ACCURACY = "balanced_accuracy"
    DETECTION_RATE = "detection_rate"


class AlgorithmFamily(Enum):
    """Algorithm families for categorization."""

    STATISTICAL = "statistical"
    DISTANCE_BASED = "distance_based"
    DENSITY_BASED = "density_based"
    ISOLATION_BASED = "isolation_based"
    NEURAL_NETWORKS = "neural_networks"
    ENSEMBLE = "ensemble"
    GRAPH_BASED = "graph_based"


@dataclass
class AlgorithmConfig:
    """Configuration for an anomaly detection algorithm."""

    name: str
    family: AlgorithmFamily
    adapter_type: str
    default_params: dict[str, Any]
    param_space: dict[str, Any]
    complexity_score: float  # 0-1, where 1 is most complex
    training_time_factor: float  # Relative training time
    memory_factor: float  # Relative memory usage
    recommended_min_samples: int = 50
    recommended_max_samples: int = 100000
    supports_streaming: bool = False
    supports_categorical: bool = False


@dataclass
class AutoMLResult:
    """Result of AutoML optimization."""

    best_algorithm: str
    best_params: dict[str, Any]
    best_score: float
    optimization_time: float
    trials_completed: int
    algorithm_rankings: list[tuple[str, float]]
    ensemble_config: dict[str, Any] | None = None
    cross_validation_scores: list[float] | None = None
    feature_importance: dict[str, float] | None = None


@dataclass
class DatasetProfile:
    """Profile of a dataset for algorithm selection."""

    n_samples: int
    n_features: int
    contamination_estimate: float
    feature_types: dict[str, str]  # feature_name -> type
    missing_values_ratio: float
    categorical_features: list[str]
    numerical_features: list[str]
    time_series_features: list[str]
    sparsity_ratio: float
    dimensionality_ratio: float  # n_features / n_samples
    dataset_size_mb: float
    has_temporal_structure: bool = False
    has_graph_structure: bool = False
    complexity_score: float = field(init=False)

    def __post_init__(self):
        """Calculate complexity score after initialization."""
        self.complexity_score = self._calculate_complexity()

    def _calculate_complexity(self) -> float:
        """Calculate dataset complexity score."""
        # Normalize factors to 0-1 scale
        size_factor = min(self.n_samples / 10000, 1.0)
        dim_factor = min(self.n_features / 1000, 1.0)
        sparsity_factor = self.sparsity_ratio
        missing_factor = self.missing_values_ratio

        # Weight different complexity factors
        complexity = (
            size_factor * 0.3
            + dim_factor * 0.3
            + sparsity_factor * 0.2
            + missing_factor * 0.2
        )

        return min(complexity, 1.0)


class AutoMLService:
    """Automated machine learning service for anomaly detection."""

    def __init__(
        self,
        detector_repository,
        dataset_repository,
        adapter_registry,
        max_optimization_time: int = 3600,
        n_trials: int = 100,
        cv_folds: int = 3,
        random_state: int = 42,
    ):
        """Initialize AutoML service.

        Args:
            detector_repository: Repository for detector storage
            dataset_repository: Repository for dataset access
            adapter_registry: Registry for algorithm adapters
            max_optimization_time: Maximum optimization time in seconds
            n_trials: Maximum number of optimization trials
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.detector_repository = detector_repository
        self.dataset_repository = dataset_repository
        self.adapter_registry = adapter_registry
        self.max_optimization_time = max_optimization_time
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Algorithm configurations
        self.algorithm_configs = self._initialize_algorithm_configs()

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Limited optimization capabilities.")

        if not SKLEARN_AVAILABLE:
            logger.warning(
                "Scikit-learn not available. Limited evaluation capabilities."
            )
            
        if not HYPEROPT_AVAILABLE:
            logger.info("Hyperopt not available. Using only Optuna for optimization.")

    def _initialize_algorithm_configs(self) -> dict[str, AlgorithmConfig]:
        """Initialize algorithm configurations."""
        configs = {}

        # Statistical algorithms
        configs["ECOD"] = AlgorithmConfig(
            name="ECOD",
            family=AlgorithmFamily.STATISTICAL,
            adapter_type="pyod",
            default_params={"contamination": 0.1},
            param_space={"contamination": {"type": "float", "low": 0.01, "high": 0.5}},
            complexity_score=0.2,
            training_time_factor=0.3,
            memory_factor=0.2,
            recommended_max_samples=50000,
        )

        configs["COPOD"] = AlgorithmConfig(
            name="COPOD",
            family=AlgorithmFamily.STATISTICAL,
            adapter_type="pyod",
            default_params={"contamination": 0.1},
            param_space={"contamination": {"type": "float", "low": 0.01, "high": 0.5}},
            complexity_score=0.25,
            training_time_factor=0.4,
            memory_factor=0.3,
            recommended_max_samples=100000,
        )
        
        # Distance-based algorithms
        configs["LOF"] = AlgorithmConfig(
            name="LOF",
            family=AlgorithmFamily.DISTANCE_BASED,
            adapter_type="sklearn",
            default_params={"n_neighbors": 20, "contamination": 0.1},
            param_space={
                "n_neighbors": {"type": "int", "low": 5, "high": 50},
                "contamination": {"type": "float", "low": 0.01, "high": 0.5}
            },
            complexity_score=0.7,
            training_time_factor=0.8,
            memory_factor=0.9,
            recommended_max_samples=10000,
        )
        
        # Isolation-based algorithms
        configs["IsolationForest"] = AlgorithmConfig(
            name="IsolationForest",
            family=AlgorithmFamily.ISOLATION_BASED,
            adapter_type="sklearn",
            default_params={"n_estimators": 100, "contamination": 0.1},
            param_space={
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_samples": {"type": "categorical", "choices": ["auto", 256, 512, 1024]},
                "contamination": {"type": "float", "low": 0.01, "high": 0.5}
            },
            complexity_score=0.4,
            training_time_factor=0.3,
            memory_factor=0.4,
            recommended_max_samples=1000000,
            supports_streaming=True,
        )
        
        # SVM-based algorithms
        configs["OneClassSVM"] = AlgorithmConfig(
            name="OneClassSVM",
            family=AlgorithmFamily.DISTANCE_BASED,
            adapter_type="sklearn",
            default_params={"gamma": "scale", "nu": 0.1},
            param_space={
                "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
                "nu": {"type": "float", "low": 0.01, "high": 0.5},
                "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly", "sigmoid"]}
            },
            complexity_score=0.8,
            training_time_factor=0.9,
            memory_factor=0.8,
            recommended_max_samples=5000,
        )
        
        return configs
            adapter_type="pyod",
            default_params={"contamination": 0.1},
            param_space={"contamination": {"type": "float", "low": 0.01, "high": 0.5}},
            complexity_score=0.3,
            training_time_factor=0.4,
            memory_factor=0.3,
        )

        # Distance-based algorithms
        configs["KNN"] = AlgorithmConfig(
            name="KNN",
            family=AlgorithmFamily.DISTANCE_BASED,
            adapter_type="pyod",
            default_params={"contamination": 0.1, "n_neighbors": 5},
            param_space={
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                "n_neighbors": {"type": "int", "low": 3, "high": 50},
                "algorithm": {
                    "type": "categorical",
                    "choices": ["auto", "ball_tree", "kd_tree", "brute"],
                },
                "leaf_size": {"type": "int", "low": 10, "high": 100},
            },
            complexity_score=0.4,
            training_time_factor=0.6,
            memory_factor=0.5,
        )

        configs["LOF"] = AlgorithmConfig(
            name="LOF",
            family=AlgorithmFamily.DENSITY_BASED,
            adapter_type="pyod",
            default_params={"contamination": 0.1, "n_neighbors": 20},
            param_space={
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                "n_neighbors": {"type": "int", "low": 5, "high": 100},
                "algorithm": {
                    "type": "categorical",
                    "choices": ["auto", "ball_tree", "kd_tree", "brute"],
                },
                "leaf_size": {"type": "int", "low": 10, "high": 100},
            },
            complexity_score=0.5,
            training_time_factor=0.7,
            memory_factor=0.6,
        )

        # Isolation-based algorithms
        configs["IsolationForest"] = AlgorithmConfig(
            name="IsolationForest",
            family=AlgorithmFamily.ISOLATION_BASED,
            adapter_type="sklearn",
            default_params={"contamination": 0.1, "n_estimators": 100},
            param_space={
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                "n_estimators": {"type": "int", "low": 50, "high": 500},
                "max_samples": {
                    "type": "categorical",
                    "choices": ["auto", 256, 512, 1024],
                },
                "max_features": {"type": "float", "low": 0.1, "high": 1.0},
            },
            complexity_score=0.6,
            training_time_factor=0.5,
            memory_factor=0.4,
            supports_streaming=True,
        )

        # Neural networks
        configs["AutoEncoder"] = AlgorithmConfig(
            name="AutoEncoder",
            family=AlgorithmFamily.NEURAL_NETWORKS,
            adapter_type="pytorch",
            default_params={
                "contamination": 0.1,
                "hidden_dims": [64, 32],
                "epochs": 100,
                "learning_rate": 0.001,
            },
            param_space={
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                "hidden_dims": {
                    "type": "categorical",
                    "choices": [[32, 16], [64, 32], [128, 64, 32]],
                },
                "epochs": {"type": "int", "low": 50, "high": 300},
                "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
            },
            complexity_score=0.8,
            training_time_factor=1.0,
            memory_factor=0.8,
            recommended_min_samples=500,
        )

        configs["VAE"] = AlgorithmConfig(
            name="VAE",
            family=AlgorithmFamily.NEURAL_NETWORKS,
            adapter_type="pytorch",
            default_params={
                "contamination": 0.1,
                "latent_dim": 16,
                "hidden_dims": [64, 32],
                "epochs": 100,
                "learning_rate": 0.001,
                "beta": 1.0,
            },
            param_space={
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                "latent_dim": {"type": "int", "low": 8, "high": 64},
                "hidden_dims": {
                    "type": "categorical",
                    "choices": [[32, 16], [64, 32], [128, 64]],
                },
                "epochs": {"type": "int", "low": 50, "high": 300},
                "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01},
                "beta": {"type": "float", "low": 0.1, "high": 5.0},
            },
            complexity_score=0.9,
            training_time_factor=1.2,
            memory_factor=0.9,
            recommended_min_samples=1000,
        )

        # One-Class SVM
        configs["OneClassSVM"] = AlgorithmConfig(
            name="OneClassSVM",
            family=AlgorithmFamily.DISTANCE_BASED,
            adapter_type="sklearn",
            default_params={"contamination": 0.1, "kernel": "rbf"},
            param_space={
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
                "kernel": {
                    "type": "categorical",
                    "choices": ["rbf", "linear", "poly", "sigmoid"],
                },
                "gamma": {
                    "type": "categorical",
                    "choices": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
                },
                "nu": {"type": "float", "low": 0.01, "high": 1.0},
            },
            complexity_score=0.7,
            training_time_factor=0.8,
            memory_factor=0.7,
            recommended_max_samples=10000,
        )

        return configs

    async def profile_dataset(self, dataset_id: str) -> DatasetProfile:
        """Profile a dataset to understand its characteristics.

        Args:
            dataset_id: ID of the dataset to profile

        Returns:
            Dataset profile with characteristics
        """
        try:
            dataset = await self.dataset_repository.get(dataset_id)
            if not dataset:
                raise AutoMLError(f"Dataset {dataset_id} not found")

            logger.info(f"Profiling dataset {dataset_id}")

            # Load dataset features
            if dataset.features is None:
                raise AutoMLError("Dataset features are not available")

            df = dataset.features

            # Basic statistics
            n_samples, n_features = df.shape
            dataset_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

            # Feature type analysis
            feature_types = {}
            categorical_features = []
            numerical_features = []
            time_series_features = []

            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    feature_types[col] = "numerical"
                    numerical_features.append(col)
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    feature_types[col] = "datetime"
                    time_series_features.append(col)
                else:
                    feature_types[col] = "categorical"
                    categorical_features.append(col)

            # Data quality metrics
            missing_values_ratio = df.isnull().sum().sum() / (n_samples * n_features)

            # Sparsity analysis (for numerical features)
            if numerical_features:
                numerical_df = df[numerical_features]
                zero_ratio = (numerical_df == 0).sum().sum() / (
                    n_samples * len(numerical_features)
                )
                sparsity_ratio = zero_ratio
            else:
                sparsity_ratio = 0.0

            # Dimensionality ratio
            dimensionality_ratio = n_features / n_samples

            # Contamination estimation (basic heuristic)
            contamination_estimate = 0.1  # Default
            if numerical_features:
                # Use IQR method for contamination estimation
                numerical_data = df[numerical_features].select_dtypes(
                    include=[np.number]
                )
                if not numerical_data.empty:
                    q1 = numerical_data.quantile(0.25)
                    q3 = numerical_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outliers = (
                        (numerical_data < lower_bound) | (numerical_data > upper_bound)
                    ).any(axis=1)
                    contamination_estimate = min(outliers.sum() / n_samples, 0.5)

            # Temporal structure detection
            has_temporal_structure = len(time_series_features) > 0

            # Graph structure detection (heuristic)
            has_graph_structure = False
            if any(
                "edge" in col.lower() or "node" in col.lower() or "graph" in col.lower()
                for col in df.columns
            ):
                has_graph_structure = True

            profile = DatasetProfile(
                n_samples=n_samples,
                n_features=n_features,
                contamination_estimate=contamination_estimate,
                feature_types=feature_types,
                missing_values_ratio=missing_values_ratio,
                categorical_features=categorical_features,
                numerical_features=numerical_features,
                time_series_features=time_series_features,
                sparsity_ratio=sparsity_ratio,
                dimensionality_ratio=dimensionality_ratio,
                dataset_size_mb=dataset_size_mb,
                has_temporal_structure=has_temporal_structure,
                has_graph_structure=has_graph_structure,
            )

            logger.info(
                f"Dataset profiling completed. Complexity score: {profile.complexity_score:.3f}"
            )
            return profile

        except Exception as e:
            logger.error(f"Error profiling dataset: {str(e)}")
            raise AutoMLError(f"Dataset profiling failed: {str(e)}")

    def recommend_algorithms(
        self, profile: DatasetProfile, max_algorithms: int = 5
    ) -> list[str]:
        """Recommend algorithms based on dataset profile.

        Args:
            profile: Dataset profile
            max_algorithms: Maximum number of algorithms to recommend

        Returns:
            List of recommended algorithm names
        """
        try:
            logger.info(
                f"Recommending algorithms for dataset with {profile.n_samples} samples, "
                f"{profile.n_features} features"
            )

            algorithm_scores = {}

            for name, config in self.algorithm_configs.items():
                score = self._calculate_algorithm_score(config, profile)
                algorithm_scores[name] = score

            # Sort by score and return top algorithms
            sorted_algorithms = sorted(
                algorithm_scores.items(), key=lambda x: x[1], reverse=True
            )
            recommended = [name for name, score in sorted_algorithms[:max_algorithms]]

            logger.info(f"Recommended algorithms: {recommended}")
            return recommended

        except Exception as e:
            logger.error(f"Error recommending algorithms: {str(e)}")
            raise AutoMLError(f"Algorithm recommendation failed: {str(e)}")

    def _calculate_algorithm_score(
        self, config: AlgorithmConfig, profile: DatasetProfile
    ) -> float:
        """Calculate suitability score for an algorithm given a dataset profile."""
        score = 1.0

        # Sample size suitability
        if profile.n_samples < config.recommended_min_samples:
            score *= 0.5  # Penalize for too few samples
        elif profile.n_samples > config.recommended_max_samples:
            score *= 0.8  # Slight penalty for very large datasets

        # Complexity matching
        complexity_diff = abs(config.complexity_score - profile.complexity_score)
        score *= 1.0 - complexity_diff * 0.3

        # Categorical feature support
        if profile.categorical_features and not config.supports_categorical:
            score *= 0.7

        # Temporal structure
        if profile.has_temporal_structure and not config.supports_streaming:
            score *= 0.8

        # Memory considerations for large datasets
        if profile.dataset_size_mb > 1000:  # > 1GB
            score *= 1.0 - config.memory_factor * 0.3

        # Training time considerations
        if profile.n_samples > 10000:
            score *= 1.0 - config.training_time_factor * 0.2

        # Family-specific bonuses
        if config.family == AlgorithmFamily.STATISTICAL and profile.n_features < 50:
            score *= 1.1  # Statistical methods work well with moderate dimensions
        elif (
            config.family == AlgorithmFamily.NEURAL_NETWORKS
            and profile.n_samples > 1000
        ):
            score *= 1.1  # Neural networks need sufficient data
        elif (
            config.family == AlgorithmFamily.ISOLATION_BASED and profile.n_features > 10
        ):
            score *= 1.1  # Isolation methods work well in higher dimensions

        return max(score, 0.1)  # Minimum score to avoid zero

    async def optimize_hyperparameters(
        self,
        dataset_id: str,
        algorithm: str,
        objective: OptimizationObjective = OptimizationObjective.AUC,
        direction: str = "maximize",
    ) -> AutoMLResult:
        """Optimize hyperparameters for a specific algorithm.

        Args:
            dataset_id: ID of the dataset
            algorithm: Algorithm name
            objective: Optimization objective
            direction: "maximize" or "minimize"

        Returns:
            AutoML optimization result
        """
        if not OPTUNA_AVAILABLE:
            raise AutoMLError("Optuna is required for hyperparameter optimization")

        try:
            start_time = time.time()
            logger.info(f"Starting hyperparameter optimization for {algorithm}")

            # Get dataset and algorithm config
            dataset = await self.dataset_repository.get(dataset_id)
            if not dataset:
                raise AutoMLError(f"Dataset {dataset_id} not found")

            if algorithm not in self.algorithm_configs:
                raise AutoMLError(f"Algorithm {algorithm} not supported")

            config = self.algorithm_configs[algorithm]

            # Create Optuna study
            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )

            # Define objective function
            def objective_func(trial):
                return self._evaluate_trial(trial, dataset, config, objective)

            # Optimize
            study.optimize(
                objective_func,
                n_trials=self.n_trials,
                timeout=self.max_optimization_time,
            )

            optimization_time = time.time() - start_time

            # Prepare result
            best_params = study.best_params.copy()
            best_params["contamination"] = study.best_params.get("contamination", 0.1)

            result = AutoMLResult(
                best_algorithm=algorithm,
                best_params=best_params,
                best_score=study.best_value,
                optimization_time=optimization_time,
                trials_completed=len(study.trials),
                algorithm_rankings=[(algorithm, study.best_value)],
            )

            logger.info(
                f"Hyperparameter optimization completed in {optimization_time:.2f}s. "
                f"Best score: {study.best_value:.4f}"
            )

            return result

        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            raise AutoMLError(f"Hyperparameter optimization failed: {str(e)}")

    def _evaluate_trial(
        self,
        trial,
        dataset: Dataset,
        config: AlgorithmConfig,
        objective: OptimizationObjective,
    ) -> float:
        """Evaluate a single optimization trial."""
        try:
            # Sample hyperparameters
            params = {}
            for param_name, param_config in config.param_space.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )

            # Create detector with sampled parameters
            detector = Detector(
                id=str(uuid4()),
                name=f"trial_{config.name}",
                algorithm=config.name,
                hyperparameters=params,
                is_fitted=False,
            )

            # Get adapter and train
            adapter = self.adapter_registry.get_adapter(config.adapter_type)

            # Prepare data
            X = dataset.features.values

            # For simulation purposes, create synthetic labels
            # In practice, this would use semi-supervised evaluation or unsupervised metrics
            contamination = params.get("contamination", 0.1)
            n_anomalies = int(len(X) * contamination)
            y_true = np.zeros(len(X))

            # Simple synthetic anomaly generation for evaluation
            anomaly_indices = np.random.choice(len(X), n_anomalies, replace=False)
            y_true[anomaly_indices] = 1

            # Train and predict
            success = adapter.train(detector, X)
            if not success:
                return 0.0

            predictions, scores = adapter.predict(detector, X)

            # Calculate objective score
            if objective == OptimizationObjective.AUC and SKLEARN_AVAILABLE:
                if len(np.unique(y_true)) > 1:
                    score = roc_auc_score(y_true, scores)
                else:
                    score = 0.5
            elif objective == OptimizationObjective.DETECTION_RATE:
                if np.sum(y_true) > 0:
                    detected_anomalies = np.sum((y_true == 1) & (predictions == 1))
                    score = detected_anomalies / np.sum(y_true)
                else:
                    score = 0.0
            else:
                # Fallback to threshold-based score
                threshold = np.percentile(scores, (1 - contamination) * 100)
                pred_binary = (scores > threshold).astype(int)
                score = np.mean(pred_binary == y_true)

            return score

        except Exception as e:
            logger.warning(f"Trial evaluation failed: {str(e)}")
            return 0.0

    async def auto_select_and_optimize(
        self,
        dataset_id: str,
        objective: OptimizationObjective = OptimizationObjective.AUC,
        max_algorithms: int = 3,
        enable_ensemble: bool = True,
    ) -> AutoMLResult:
        """Automatically select and optimize the best algorithm.

        Args:
            dataset_id: ID of the dataset
            objective: Optimization objective
            max_algorithms: Maximum algorithms to try
            enable_ensemble: Whether to create ensemble if beneficial

        Returns:
            Complete AutoML result
        """
        try:
            start_time = time.time()
            logger.info(f"Starting AutoML for dataset {dataset_id}")

            # Profile dataset
            profile = await self.profile_dataset(dataset_id)

            # Recommend algorithms
            recommended_algorithms = self.recommend_algorithms(profile, max_algorithms)

            # Optimize each recommended algorithm
            optimization_results = []
            for algorithm in recommended_algorithms:
                try:
                    result = await self.optimize_hyperparameters(
                        dataset_id, algorithm, objective
                    )
                    optimization_results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to optimize {algorithm}: {str(e)}")
                    continue

            if not optimization_results:
                raise AutoMLError("No algorithms could be successfully optimized")

            # Find best algorithm
            best_result = max(optimization_results, key=lambda x: x.best_score)

            # Create algorithm rankings
            algorithm_rankings = [
                (result.best_algorithm, result.best_score)
                for result in optimization_results
            ]
            algorithm_rankings.sort(key=lambda x: x[1], reverse=True)

            # Create ensemble if enabled and beneficial
            ensemble_config = None
            if enable_ensemble and len(optimization_results) > 1:
                ensemble_config = self._create_ensemble_config(optimization_results)

            # Prepare final result
            total_time = time.time() - start_time

            final_result = AutoMLResult(
                best_algorithm=best_result.best_algorithm,
                best_params=best_result.best_params,
                best_score=best_result.best_score,
                optimization_time=total_time,
                trials_completed=sum(r.trials_completed for r in optimization_results),
                algorithm_rankings=algorithm_rankings,
                ensemble_config=ensemble_config,
            )

            logger.info(
                f"AutoML completed in {total_time:.2f}s. "
                f"Best algorithm: {best_result.best_algorithm} "
                f"(score: {best_result.best_score:.4f})"
            )

            return final_result

        except Exception as e:
            logger.error(f"AutoML failed: {str(e)}")
            raise AutoMLError(f"AutoML process failed: {str(e)}")

    def _create_ensemble_config(
        self, optimization_results: list[AutoMLResult]
    ) -> dict[str, Any]:
        """Create ensemble configuration from optimization results."""
        # Select top algorithms for ensemble
        top_results = sorted(
            optimization_results, key=lambda x: x.best_score, reverse=True
        )[:3]

        # Calculate weights based on performance
        scores = [r.best_score for r in top_results]
        total_score = sum(scores)
        weights = (
            [score / total_score for score in scores]
            if total_score > 0
            else [1 / len(scores)] * len(scores)
        )

        ensemble_config = {
            "method": "weighted_voting",
            "algorithms": [
                {
                    "name": result.best_algorithm,
                    "params": result.best_params,
                    "weight": weight,
                }
                for result, weight in zip(top_results, weights, strict=False)
            ],
            "voting_strategy": "soft",
            "normalize_scores": True,
        }

        return ensemble_config

    async def create_optimized_detector(
        self, automl_result: AutoMLResult, detector_name: str | None = None
    ) -> str:
        """Create a detector from AutoML results.

        Args:
            automl_result: Result from AutoML optimization
            detector_name: Optional custom name for the detector

        Returns:
            ID of the created detector
        """
        try:
            detector_id = str(uuid4())
            name = detector_name or f"AutoML_{automl_result.best_algorithm}"

            detector = Detector(
                id=detector_id,
                name=name,
                algorithm=automl_result.best_algorithm,
                hyperparameters=automl_result.best_params,
                is_fitted=False,
            )

            await self.detector_repository.save(detector)

            logger.info(
                f"Created optimized detector {detector_id} with algorithm {automl_result.best_algorithm}"
            )
            return detector_id

        except Exception as e:
            logger.error(f"Error creating optimized detector: {str(e)}")
            raise AutoMLError(f"Failed to create detector: {str(e)}")

    def get_optimization_summary(self, automl_result: AutoMLResult) -> dict[str, Any]:
        """Get a summary of the optimization process.

        Args:
            automl_result: AutoML result to summarize

        Returns:
            Optimization summary
        """
        summary = {
            "best_algorithm": automl_result.best_algorithm,
            "best_score": automl_result.best_score,
            "optimization_time_seconds": automl_result.optimization_time,
            "trials_completed": automl_result.trials_completed,
            "algorithm_rankings": automl_result.algorithm_rankings,
            "has_ensemble": automl_result.ensemble_config is not None,
            "recommendations": [],
        }

        # Add recommendations
        if automl_result.best_score < 0.7:
            summary["recommendations"].append("Consider collecting more training data")

        if automl_result.ensemble_config:
            summary["recommendations"].append(
                "Ensemble model created for improved performance"
            )

        if automl_result.optimization_time > self.max_optimization_time * 0.9:
            summary["recommendations"].append(
                "Consider increasing optimization time for better results"
            )

        return summary

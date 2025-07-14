"""
AutoML Service for Anomaly Detection

Comprehensive AutoML service providing automated feature engineering, algorithm recommendation,
and pipeline optimization specifically designed for anomaly detection use cases.

This addresses Issue #143: Phase 2.2: Data Science Package - Machine Learning Pipeline Framework
Component 5: AutoML Capabilities for Automated Feature Engineering and Algorithm Recommendation
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV,
    mutual_info_classif, f_classif, chi2
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, PolynomialFeatures
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import AutoMLError
from pynomaly.infrastructure.logging.structured_logger import StructuredLogger

# Optional optimization libraries
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import CmaEsSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    CmaEsSampler = None
    MedianPruner = None
    OPTUNA_AVAILABLE = False

try:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

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
    from hyperopt import Trials, fmin, hp, tpe

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
    TREE_BASED = "tree_based"


class DataCharacteristics(Enum):
    """Data characteristics that influence AutoML decisions."""
    
    HIGH_DIMENSIONAL = "high_dimensional"
    SPARSE_DATA = "sparse_data"
    TEMPORAL_DATA = "temporal_data"
    MIXED_TYPES = "mixed_types"
    MISSING_VALUES = "missing_values"
    IMBALANCED_CLASSES = "imbalanced_classes"
    SMALL_DATASET = "small_dataset"
    LARGE_DATASET = "large_dataset"
    CATEGORICAL_HEAVY = "categorical_heavy"
    NUMERICAL_HEAVY = "numerical_heavy"


class FeatureEngineeringStrategy(Enum):
    """Feature engineering strategies for different data types."""
    
    BASIC_PREPROCESSING = "basic_preprocessing"
    STATISTICAL_FEATURES = "statistical_features"
    TEMPORAL_FEATURES = "temporal_features"
    INTERACTION_FEATURES = "interaction_features"
    POLYNOMIAL_FEATURES = "polynomial_features"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    CLUSTERING_FEATURES = "clustering_features"
    DOMAIN_SPECIFIC = "domain_specific"


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
class FeatureEngineeringPipeline:
    """Feature engineering pipeline configuration."""
    
    pipeline_id: str
    strategies: List[FeatureEngineeringStrategy]
    transformers: Dict[str, Any]
    feature_selectors: Dict[str, Any]
    preprocessing_steps: List[Tuple[str, TransformerMixin]]
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    data_profile: Optional['DatasetProfile'] = None
    performance_score: Optional[float] = None


@dataclass
class AlgorithmRecommendation:
    """Algorithm recommendation with confidence score."""
    
    algorithm_name: str
    algorithm_family: AlgorithmFamily
    confidence_score: float
    recommended_parameters: Dict[str, Any]
    reasoning: List[str]
    
    # Performance estimates
    expected_performance: Optional[Dict[str, float]] = None
    training_time_estimate: Optional[float] = None
    memory_requirements: Optional[str] = None


@dataclass
class AutoMLConfiguration:
    """Configuration for AutoML pipeline."""
    
    # Feature engineering settings
    max_features_generated: int = 100
    feature_selection_methods: List[str] = field(default_factory=lambda: ["univariate", "recursive"])
    preprocessing_strategies: List[str] = field(default_factory=lambda: ["scaling", "encoding"])
    
    # Algorithm recommendation settings
    algorithm_families: List[AlgorithmFamily] = field(default_factory=lambda: [
        AlgorithmFamily.ISOLATION_BASED,
        AlgorithmFamily.DENSITY_BASED,
        AlgorithmFamily.ENSEMBLE
    ])
    max_algorithms_recommended: int = 5
    
    # Optimization settings
    optimization_budget: int = 100  # Number of trials
    optimization_timeout: int = 3600  # Seconds
    cross_validation_folds: int = 5
    
    # Performance criteria
    primary_metric: str = "roc_auc"
    secondary_metrics: List[str] = field(default_factory=lambda: ["precision", "recall", "f1_score"])
    
    # Resource constraints
    max_training_time: int = 1800  # Seconds per model
    max_memory_usage: str = "8GB"
    parallel_jobs: int = -1


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
    
    # Enhanced characteristics
    feature_correlations: Optional[np.ndarray] = None
    feature_importance_scores: Optional[Dict[str, float]] = None
    outlier_ratio: float = 0.0
    duplicate_ratio: float = 0.0
    skewness_scores: Optional[Dict[str, float]] = None
    kurtosis_scores: Optional[Dict[str, float]] = None
    is_temporal: bool = False
    temporal_patterns: Optional[Dict[str, Any]] = None
    data_characteristics: List[DataCharacteristics] = field(default_factory=list)
    recommended_strategies: List[FeatureEngineeringStrategy] = field(default_factory=list)
    recommended_algorithms: List[AlgorithmFamily] = field(default_factory=list)

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
    """Comprehensive AutoML service for anomaly detection."""

    def __init__(
        self,
        detector_repository=None,
        dataset_repository=None,
        adapter_registry=None,
        results_dir: str = "automl_results",
        cache_enabled: bool = True,
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
            results_dir: Directory for storing AutoML results
            cache_enabled: Whether to enable caching
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
        
        # Enhanced AutoML capabilities
        self.logger = StructuredLogger("automl_service")
        
        # Storage configuration
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration
        self.cache_enabled = cache_enabled

        # Algorithm configurations
        self.algorithm_configs = self._initialize_algorithm_configs()
        
        # Feature engineering components
        self._feature_generators = self._initialize_feature_generators()
        self._preprocessors = self._initialize_preprocessors()
        self._feature_selectors = self._initialize_feature_selectors()
        
        # Results cache
        self._profile_cache: Dict[str, DatasetProfile] = {}
        self._pipeline_cache: Dict[str, FeatureEngineeringPipeline] = {}

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Limited optimization capabilities.")

        if not SKLEARN_AVAILABLE:
            logger.warning(
                "Scikit-learn not available. Limited evaluation capabilities."
            )

        if not HYPEROPT_AVAILABLE:
            logger.info("Hyperopt not available. Using only Optuna for optimization.")
        
        self.logger.info("Enhanced AutoML service initialized")

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
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
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
                "max_samples": {
                    "type": "categorical",
                    "choices": ["auto", 256, 512, 1024],
                },
                "contamination": {"type": "float", "low": 0.01, "high": 0.5},
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
                "kernel": {
                    "type": "categorical",
                    "choices": ["rbf", "linear", "poly", "sigmoid"],
                },
            },
            complexity_score=0.8,
            training_time_factor=0.9,
            memory_factor=0.8,
            recommended_max_samples=5000,
        )

        return configs
    
    def _initialize_feature_generators(self) -> Dict[str, Any]:
        """Initialize feature generation components."""
        
        return {
            "polynomial": PolynomialFeatures,
            "statistical": None,  # Custom implementation
            "interactions": None,  # Custom implementation
            "clustering": None    # Custom implementation
        }
    
    def _initialize_preprocessors(self) -> Dict[str, Any]:
        """Initialize preprocessing components."""
        
        return {
            "standard_scaler": StandardScaler,
            "robust_scaler": RobustScaler,
            "minmax_scaler": MinMaxScaler,
            "maxabs_scaler": MaxAbsScaler,
            "power_transformer": PowerTransformer,
            "quantile_transformer": QuantileTransformer
        }
    
    def _initialize_feature_selectors(self) -> Dict[str, Any]:
        """Initialize feature selection components."""
        
        return {
            "select_k_best": SelectKBest,
            "select_percentile": SelectPercentile,
            "rfe": RFE,
            "rfecv": RFECV
        }

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

            # Enhanced data analysis
            # Statistical characteristics
            skewness_scores = {}
            kurtosis_scores = {}
            if numerical_features:
                for col in numerical_features:
                    try:
                        skewness_scores[col] = float(df[col].skew())
                        kurtosis_scores[col] = float(df[col].kurtosis())
                    except Exception:
                        skewness_scores[col] = 0.0
                        kurtosis_scores[col] = 0.0
            
            # Correlation analysis
            feature_correlations = None
            if len(numerical_features) > 1:
                try:
                    correlation_matrix = df[numerical_features].corr()
                    feature_correlations = correlation_matrix.values
                except Exception:
                    feature_correlations = None
            
            # Feature importance estimation (if labels available)
            feature_importance_scores = None
            if numerical_features:
                try:
                    feature_importance_scores = await self._estimate_feature_importance(
                        df[numerical_features]
                    )
                except Exception as e:
                    self.logger.warning(f"Feature importance estimation failed: {e}")
            
            # Outlier detection
            outlier_ratio = 0.0
            if numerical_features:
                try:
                    outlier_ratio = await self._estimate_outlier_ratio(df[numerical_features])
                except Exception as e:
                    self.logger.warning(f"Outlier estimation failed: {e}")
            
            # Duplicate analysis
            duplicate_ratio = df.duplicated().sum() / n_samples

            # Contamination estimation (basic heuristic)
            contamination_estimate = max(outlier_ratio, 0.1)  # Use outlier ratio or default
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

            # Temporal pattern detection
            is_temporal, temporal_patterns = await self._detect_temporal_patterns(df)
            
            # Data characteristics identification
            data_characteristics = await self._identify_data_characteristics(
                n_samples, n_features, feature_types, missing_values_ratio,
                sparsity_ratio, is_temporal
            )
            
            # Strategy recommendations
            recommended_strategies = await self._recommend_feature_strategies(
                data_characteristics, feature_types
            )
            
            # Algorithm recommendations
            recommended_algorithms = await self._recommend_algorithm_families(
                data_characteristics, n_samples, n_features
            )

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
                # Enhanced characteristics
                feature_correlations=feature_correlations,
                feature_importance_scores=feature_importance_scores,
                outlier_ratio=outlier_ratio,
                duplicate_ratio=duplicate_ratio,
                skewness_scores=skewness_scores,
                kurtosis_scores=kurtosis_scores,
                is_temporal=is_temporal,
                temporal_patterns=temporal_patterns,
                data_characteristics=data_characteristics,
                recommended_strategies=recommended_strategies,
                recommended_algorithms=recommended_algorithms,
            )

            logger.info(
                f"Dataset profiling completed. Complexity score: {profile.complexity_score:.3f}"
            )
            return profile

        except Exception as e:
            logger.error(f"Error profiling dataset: {str(e)}")
            raise AutoMLError(f"Dataset profiling failed: {str(e)}") from e

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
    
    # Enhanced AutoML helper methods
    async def _estimate_feature_importance(
        self,
        X: pd.DataFrame
    ) -> Dict[str, float]:
        """Estimate feature importance using statistical methods."""
        
        try:
            # Simple variance-based importance for unsupervised case
            feature_importance = {}
            for col in X.columns:
                try:
                    variance = X[col].var()
                    feature_importance[col] = float(variance)
                except Exception:
                    feature_importance[col] = 0.0
            
            # Normalize scores
            max_score = max(feature_importance.values()) if feature_importance.values() else 1
            if max_score > 0:
                feature_importance = {k: v / max_score for k, v in feature_importance.items()}
            
            return feature_importance
            
        except Exception:
            return {col: 0.0 for col in X.columns}
    
    async def _estimate_outlier_ratio(self, X: pd.DataFrame) -> float:
        """Estimate outlier ratio using Isolation Forest."""
        
        try:
            # Quick outlier detection
            iso_forest = IsolationForest(
                contamination='auto',
                random_state=42,
                n_estimators=50
            )
            outlier_predictions = iso_forest.fit_predict(X)
            outlier_ratio = (outlier_predictions == -1).sum() / len(outlier_predictions)
            
            return float(outlier_ratio)
            
        except Exception:
            return 0.0
    
    async def _detect_temporal_patterns(
        self,
        X: pd.DataFrame
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Detect temporal patterns in the data."""
        
        # Simple heuristics for temporal pattern detection
        is_temporal = False
        temporal_patterns = None
        
        # Check for datetime columns
        datetime_cols = X.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            is_temporal = True
            temporal_patterns = {"datetime_columns": datetime_cols}
        
        return is_temporal, temporal_patterns
    
    async def _identify_data_characteristics(
        self,
        n_samples: int,
        n_features: int,
        feature_types: Dict[str, str],
        missing_ratio: float,
        sparsity_ratio: float,
        is_temporal: bool
    ) -> List[DataCharacteristics]:
        """Identify key data characteristics."""
        
        characteristics = []
        
        # Size characteristics
        if n_samples < 1000:
            characteristics.append(DataCharacteristics.SMALL_DATASET)
        elif n_samples > 100000:
            characteristics.append(DataCharacteristics.LARGE_DATASET)
        
        # Dimensionality
        if n_features > 100:
            characteristics.append(DataCharacteristics.HIGH_DIMENSIONAL)
        
        # Sparsity
        if sparsity_ratio > 0.3:
            characteristics.append(DataCharacteristics.SPARSE_DATA)
        
        # Missing values
        if missing_ratio > 0.1:
            characteristics.append(DataCharacteristics.MISSING_VALUES)
        
        # Feature types
        numerical_ratio = sum(1 for t in feature_types.values() if t == 'numerical') / len(feature_types)
        categorical_ratio = sum(1 for t in feature_types.values() if t == 'categorical') / len(feature_types)
        
        if numerical_ratio > 0.7:
            characteristics.append(DataCharacteristics.NUMERICAL_HEAVY)
        elif categorical_ratio > 0.7:
            characteristics.append(DataCharacteristics.CATEGORICAL_HEAVY)
        else:
            characteristics.append(DataCharacteristics.MIXED_TYPES)
        
        # Temporal characteristics
        if is_temporal:
            characteristics.append(DataCharacteristics.TEMPORAL_DATA)
        
        return characteristics
    
    async def _recommend_feature_strategies(
        self,
        data_characteristics: List[DataCharacteristics],
        feature_types: Dict[str, str]
    ) -> List[FeatureEngineeringStrategy]:
        """Recommend feature engineering strategies."""
        
        strategies = []
        
        # Always include basic preprocessing
        strategies.append(FeatureEngineeringStrategy.BASIC_PREPROCESSING)
        
        # Statistical features for numerical data
        if DataCharacteristics.NUMERICAL_HEAVY in data_characteristics:
            strategies.append(FeatureEngineeringStrategy.STATISTICAL_FEATURES)
        
        # Temporal features for time series data
        if DataCharacteristics.TEMPORAL_DATA in data_characteristics:
            strategies.append(FeatureEngineeringStrategy.TEMPORAL_FEATURES)
        
        # Interaction features for small to medium datasets
        if DataCharacteristics.SMALL_DATASET in data_characteristics:
            strategies.append(FeatureEngineeringStrategy.INTERACTION_FEATURES)
            strategies.append(FeatureEngineeringStrategy.POLYNOMIAL_FEATURES)
        
        # Dimensionality reduction for high-dimensional data
        if DataCharacteristics.HIGH_DIMENSIONAL in data_characteristics:
            strategies.append(FeatureEngineeringStrategy.DIMENSIONALITY_REDUCTION)
        
        # Clustering features for large datasets
        if DataCharacteristics.LARGE_DATASET in data_characteristics:
            strategies.append(FeatureEngineeringStrategy.CLUSTERING_FEATURES)
        
        return strategies
    
    async def _recommend_algorithm_families(
        self,
        data_characteristics: List[DataCharacteristics],
        n_samples: int,
        n_features: int
    ) -> List[AlgorithmFamily]:
        """Recommend algorithm families based on data characteristics."""
        
        families = []
        
        # Isolation-based methods work well for most cases
        families.append(AlgorithmFamily.ISOLATION_BASED)
        
        # Density-based for medium-sized datasets
        if DataCharacteristics.SMALL_DATASET not in data_characteristics:
            families.append(AlgorithmFamily.DENSITY_BASED)
        
        # Ensemble methods for complex data
        if (DataCharacteristics.HIGH_DIMENSIONAL in data_characteristics or
            DataCharacteristics.MIXED_TYPES in data_characteristics):
            families.append(AlgorithmFamily.ENSEMBLE)
        
        # Tree-based methods for mixed data types
        if DataCharacteristics.MIXED_TYPES in data_characteristics:
            families.append(AlgorithmFamily.TREE_BASED)
        
        # Statistical methods for small, clean datasets
        if (DataCharacteristics.SMALL_DATASET in data_characteristics and
            DataCharacteristics.MISSING_VALUES not in data_characteristics):
            families.append(AlgorithmFamily.STATISTICAL)
        
        return families
    
    async def recommend_algorithms_enhanced(
        self,
        data_profile: DatasetProfile,
        config: Optional[AutoMLConfiguration] = None
    ) -> List[AlgorithmRecommendation]:
        """Enhanced algorithm recommendations with detailed analysis."""
        
        if config is None:
            config = AutoMLConfiguration()
        
        self.logger.info("Generating enhanced algorithm recommendations")
        
        try:
            recommendations = []
            
            # Score algorithms based on data characteristics
            algorithm_scores = await self._score_algorithms_enhanced(data_profile, config)
            
            # Sort by score and select top algorithms
            sorted_algorithms = sorted(
                algorithm_scores.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )
            
            for i, (algorithm_name, algo_data) in enumerate(sorted_algorithms[:config.max_algorithms_recommended]):
                # Get algorithm family
                algorithm_family = self._get_algorithm_family(algorithm_name)
                
                # Generate reasoning
                reasoning = await self._generate_algorithm_reasoning(
                    algorithm_name, data_profile, algo_data
                )
                
                # Get recommended parameters
                recommended_parameters = await self._get_recommended_parameters(
                    algorithm_name, data_profile
                )
                
                # Estimate performance and resource requirements
                expected_performance = await self._estimate_algorithm_performance(
                    algorithm_name, data_profile
                )
                
                training_time_estimate = await self._estimate_training_time(
                    algorithm_name, data_profile
                )
                
                memory_requirements = await self._estimate_memory_requirements(
                    algorithm_name, data_profile
                )
                
                recommendation = AlgorithmRecommendation(
                    algorithm_name=algorithm_name,
                    algorithm_family=algorithm_family,
                    confidence_score=algo_data["score"],
                    recommended_parameters=recommended_parameters,
                    reasoning=reasoning,
                    expected_performance=expected_performance,
                    training_time_estimate=training_time_estimate,
                    memory_requirements=memory_requirements
                )
                
                recommendations.append(recommendation)
            
            self.logger.info(f"Generated {len(recommendations)} enhanced algorithm recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Enhanced algorithm recommendation failed: {e}")
            raise AutoMLError(f"Enhanced algorithm recommendation failed: {str(e)}")
    
    async def _score_algorithms_enhanced(
        self,
        data_profile: DatasetProfile,
        config: AutoMLConfiguration
    ) -> Dict[str, Dict[str, Any]]:
        """Enhanced algorithm scoring based on data characteristics."""
        
        algorithm_scores = {}
        
        for algorithm_name, algorithm_config in self.algorithm_configs.items():
            score = 0.0
            reasoning = []
            
            # Base score from algorithm family preference
            if algorithm_config.family in data_profile.recommended_algorithms:
                score += 0.5
                reasoning.append(f"Recommended family: {algorithm_config.family.value}")
            
            # Data size compatibility
            if (algorithm_config.recommended_min_samples <= data_profile.n_samples <= 
                algorithm_config.recommended_max_samples):
                score += 0.2
                reasoning.append("Compatible with dataset size")
            
            # Feature dimensionality compatibility
            if data_profile.n_features > 100 and algorithm_config.complexity_score < 0.7:
                score += 0.15
                reasoning.append("Suitable for high-dimensional data")
            
            # Missing values tolerance
            if data_profile.missing_values_ratio > 0.1:
                if hasattr(algorithm_config, 'handles_missing') and algorithm_config.supports_categorical:
                    score += 0.1
                    reasoning.append("Handles missing values")
            
            # Sparsity tolerance
            if data_profile.sparsity_ratio > 0.3:
                if algorithm_config.family in [AlgorithmFamily.ISOLATION_BASED, AlgorithmFamily.STATISTICAL]:
                    score += 0.1
                    reasoning.append("Sparse data friendly")
            
            # Training speed preference for large datasets
            if data_profile.n_samples > 10000 and algorithm_config.training_time_factor < 0.5:
                score += 0.1
                reasoning.append("Fast training for large datasets")
            
            algorithm_scores[algorithm_name] = {
                "score": min(score, 1.0),  # Cap at 1.0
                "reasoning": reasoning,
                "algorithm_info": algorithm_config
            }
        
        return algorithm_scores
    
    def _get_algorithm_family(self, algorithm_name: str) -> AlgorithmFamily:
        """Get algorithm family for a given algorithm."""
        
        algorithm_config = self.algorithm_configs.get(algorithm_name)
        if algorithm_config:
            return algorithm_config.family
        return AlgorithmFamily.ISOLATION_BASED  # Default
    
    async def _generate_algorithm_reasoning(
        self,
        algorithm_name: str,
        data_profile: DatasetProfile,
        algo_data: Dict[str, Any]
    ) -> List[str]:
        """Generate reasoning for algorithm recommendation."""
        
        reasoning = algo_data.get("reasoning", [])
        
        # Add specific algorithm insights
        if algorithm_name == "IsolationForest":
            reasoning.append("Effective for unsupervised anomaly detection")
            if data_profile.n_samples > 1000:
                reasoning.append("Scales well with dataset size")
        elif algorithm_name == "LOF":
            reasoning.append("Good for density-based anomaly detection")
            if data_profile.n_features < 50:
                reasoning.append("Works well with moderate dimensionality")
        
        return reasoning
    
    async def _get_recommended_parameters(
        self,
        algorithm_name: str,
        data_profile: DatasetProfile
    ) -> Dict[str, Any]:
        """Get recommended parameter ranges for algorithm."""
        
        algorithm_config = self.algorithm_configs.get(algorithm_name)
        if algorithm_config:
            return algorithm_config.param_space
        
        # Default parameter space
        return {
            "contamination": {"type": "float", "low": 0.01, "high": 0.3}
        }
    
    async def _estimate_algorithm_performance(
        self,
        algorithm_name: str,
        data_profile: DatasetProfile
    ) -> Optional[Dict[str, float]]:
        """Estimate algorithm performance based on data characteristics."""
        
        # Performance estimates based on algorithm characteristics and data profile
        performance_estimates = {
            "IsolationForest": {
                "roc_auc": 0.8,
                "precision": 0.7,
                "recall": 0.75,
                "f1_score": 0.725
            },
            "LOF": {
                "roc_auc": 0.75,
                "precision": 0.65,
                "recall": 0.7,
                "f1_score": 0.675
            }
        }
        
        return performance_estimates.get(algorithm_name)
    
    async def _estimate_training_time(
        self,
        algorithm_name: str,
        data_profile: DatasetProfile
    ) -> Optional[float]:
        """Estimate training time in seconds."""
        
        algorithm_config = self.algorithm_configs.get(algorithm_name)
        if algorithm_config:
            base_time = algorithm_config.training_time_factor * 10  # Base time in seconds
            estimated_time = base_time * (data_profile.n_samples / 1000)
            return estimated_time
        
        return None
    
    async def _estimate_memory_requirements(
        self,
        algorithm_name: str,
        data_profile: DatasetProfile
    ) -> Optional[str]:
        """Estimate memory requirements."""
        
        algorithm_config = self.algorithm_configs.get(algorithm_name)
        if algorithm_config:
            if algorithm_config.memory_factor < 0.3:
                return "Low"
            elif algorithm_config.memory_factor < 0.7:
                return "Medium"
            else:
                return "High"
        
        return "Medium"
    
    async def get_automl_summary(self) -> Dict[str, Any]:
        """Get summary of AutoML service capabilities and cached results."""
        
        summary = {
            "service_info": {
                "algorithms_available": len(self.algorithm_configs),
                "feature_generators": len(self._feature_generators),
                "preprocessors": len(self._preprocessors),
                "feature_selectors": len(self._feature_selectors)
            },
            "cached_profiles": len(self._profile_cache),
            "cached_pipelines": len(self._pipeline_cache),
            "supported_algorithms": list(self.algorithm_configs.keys()),
            "supported_strategies": [strategy.value for strategy in FeatureEngineeringStrategy],
            "supported_characteristics": [char.value for char in DataCharacteristics]
        }
        
        return summary

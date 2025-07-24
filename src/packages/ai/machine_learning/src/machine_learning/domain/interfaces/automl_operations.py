"""AutoML Operations Interfaces (Ports).

This module defines the abstract interfaces for automated machine learning
operations that the machine learning domain requires. These interfaces
represent the "ports" in hexagonal architecture, defining contracts for
external AutoML libraries without coupling to specific implementations.

Following DDD principles, these interfaces belong to the domain layer and
define what the domain needs from external AutoML capabilities.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..entities.model import Model
from ..entities.dataset import Dataset
from ..entities.detection_result import DetectionResult


class OptimizationMetric(Enum):
    """Metrics for model optimization."""
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "auc"
    BALANCED_ACCURACY = "balanced_accuracy"
    ANOMALY_RATE = "anomaly_rate"
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"


class SearchStrategy(Enum):
    """Hyperparameter search strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    MULTI_OBJECTIVE = "multi_objective"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"


class AlgorithmType(Enum):
    """Types of algorithms for AutoML."""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    DBSCAN = "dbscan"
    KNN = "knn"
    AUTO_ENCODER = "auto_encoder"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    VARIATIONAL_AUTOENCODER = "vae"
    TRANSFORMER = "transformer"


class EnsembleMethod(Enum):
    """Ensemble combination methods."""
    AVERAGE = "average"
    VOTING = "voting"
    WEIGHTED_AVERAGE = "weighted_average"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"


@dataclass
class OptimizationConfig:
    """Configuration for AutoML optimization."""
    search_strategy: SearchStrategy = SearchStrategy.BAYESIAN_OPTIMIZATION
    max_trials: int = 100
    timeout_seconds: Optional[int] = None
    primary_metric: OptimizationMetric = OptimizationMetric.F1_SCORE
    secondary_metrics: List[OptimizationMetric] = None
    cv_folds: int = 5
    algorithms_to_test: Optional[List[AlgorithmType]] = None
    ensemble_methods: bool = True
    max_ensemble_size: int = 5
    max_memory_mb: Optional[int] = None
    max_execution_time_per_trial: Optional[int] = 300
    enable_pruning: bool = True
    patience: int = 20
    random_state: int = 42
    n_jobs: int = -1
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = [OptimizationMetric.AUC]


@dataclass
class AlgorithmConfig:
    """Configuration for a specific algorithm."""
    algorithm_type: AlgorithmType
    parameters: Dict[str, Any]
    contamination: float = 0.1
    random_state: Optional[int] = None
    preprocessing_config: Optional[Dict[str, Any]] = None


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    algorithm_configs: List[AlgorithmConfig]
    combination_method: EnsembleMethod
    weights: Optional[List[float]] = None
    voting_threshold: float = 0.5
    meta_learner_config: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationTrial:
    """Single optimization trial result."""
    trial_id: str
    algorithm_type: AlgorithmType
    parameters: Dict[str, Any]
    score: float
    metrics: Dict[str, float]
    execution_time_seconds: float
    memory_usage_mb: float
    status: str  # "completed", "failed", "pruned"
    error_message: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class OptimizationResult:
    """Result of AutoML optimization."""
    best_algorithm_type: AlgorithmType
    best_config: AlgorithmConfig
    best_score: float
    best_metrics: Dict[str, float]
    trial_history: List[OptimizationTrial]
    total_trials: int
    optimization_time_seconds: float
    top_k_results: List[tuple]  # (algorithm, config, score)
    ensemble_config: Optional[EnsembleConfig] = None
    ensemble_score: Optional[float] = None
    recommendations: Dict[str, List[str]] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.recommendations is None:
            self.recommendations = {}


class AutoMLOptimizationPort(ABC):
    """Port for AutoML optimization operations.
    
    This interface defines the contract for automated machine learning
    model selection, hyperparameter optimization, and ensemble creation.
    """

    @abstractmethod
    async def optimize_model(
        self,
        dataset: Dataset,
        optimization_config: OptimizationConfig,
        ground_truth: Optional[Any] = None
    ) -> OptimizationResult:
        """Automatically optimize model for the given dataset.
        
        Args:
            dataset: Training dataset
            optimization_config: Configuration for optimization process
            ground_truth: Optional ground truth labels for supervised evaluation
            
        Returns:
            Optimization result with best configuration and trial history
            
        Raises:
            OptimizationError: If optimization process fails
            ConfigurationError: If optimization configuration is invalid
        """
        pass

    @abstractmethod
    async def suggest_algorithms(
        self,
        dataset: Dataset,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[AlgorithmType]:
        """Suggest appropriate algorithms based on dataset characteristics.
        
        Args:
            dataset: Dataset for algorithm suggestion
            constraints: Optional constraints (time, memory, accuracy requirements)
            
        Returns:
            List of recommended algorithms in priority order
            
        Raises:
            SuggestionError: If algorithm suggestion fails
        """
        pass

    @abstractmethod
    async def evaluate_algorithm(
        self,
        algorithm_config: AlgorithmConfig,
        dataset: Dataset,
        evaluation_config: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> OptimizationTrial:
        """Evaluate a specific algorithm configuration.
        
        Args:
            algorithm_config: Algorithm configuration to evaluate
            dataset: Dataset for evaluation
            evaluation_config: Configuration for evaluation process
            ground_truth: Optional ground truth for supervised evaluation
            
        Returns:
            Trial result with performance metrics
            
        Raises:
            EvaluationError: If algorithm evaluation fails
        """
        pass

    @abstractmethod
    async def optimize_ensemble(
        self,
        individual_results: List[OptimizationTrial],
        dataset: Dataset,
        ensemble_config: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> EnsembleConfig:
        """Optimize ensemble configuration from individual algorithm results.
        
        Args:
            individual_results: Results from individual algorithm optimization
            dataset: Dataset for ensemble optimization
            ensemble_config: Configuration for ensemble optimization
            ground_truth: Optional ground truth for evaluation
            
        Returns:
            Optimized ensemble configuration
            
        Raises:
            EnsembleOptimizationError: If ensemble optimization fails
        """
        pass

    @abstractmethod
    async def get_parameter_space(
        self,
        algorithm_type: AlgorithmType
    ) -> Dict[str, Any]:
        """Get parameter search space for specific algorithm.
        
        Args:
            algorithm_type: Algorithm to get parameter space for
            
        Returns:
            Parameter space definition with ranges and types
            
        Raises:
            UnsupportedAlgorithmError: If algorithm is not supported
        """
        pass

    @abstractmethod
    async def get_supported_algorithms(self) -> List[AlgorithmType]:
        """Get list of supported algorithms.
        
        Returns:
            List of algorithm types supported by this implementation
        """
        pass


class ModelSelectionPort(ABC):
    """Port for automated model selection operations.
    
    This interface defines the contract for intelligent model selection
    based on dataset characteristics and requirements.
    """

    @abstractmethod
    async def select_best_model(
        self,
        dataset: Dataset,
        requirements: Dict[str, Any],
        quick_mode: bool = False
    ) -> tuple[AlgorithmType, AlgorithmConfig]:
        """Select the best model for given dataset and requirements.
        
        Args:
            dataset: Dataset for model selection
            requirements: Performance and resource requirements
            quick_mode: Whether to use fast heuristic-based selection
            
        Returns:
            Tuple of selected algorithm type and configuration
            
        Raises:
            ModelSelectionError: If model selection fails
        """
        pass

    @abstractmethod
    async def analyze_dataset_characteristics(
        self,
        dataset: Dataset
    ) -> Dict[str, Any]:
        """Analyze dataset characteristics for model selection.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dataset characteristics analysis
            
        Raises:
            DatasetAnalysisError: If dataset analysis fails
        """
        pass

    @abstractmethod
    async def get_model_recommendations(
        self,
        dataset: Dataset,
        current_performance: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[str]]:
        """Get recommendations for improving model performance.
        
        Args:
            dataset: Dataset for recommendations
            current_performance: Current model performance metrics
            
        Returns:
            Categorized recommendations for improvement
            
        Raises:
            RecommendationError: If recommendation generation fails
        """
        pass

    @abstractmethod
    async def compare_algorithms(
        self,
        algorithm_results: List[OptimizationTrial]
    ) -> Dict[str, Any]:
        """Compare multiple algorithm results and provide insights.
        
        Args:
            algorithm_results: List of algorithm evaluation results
            
        Returns:
            Comparison analysis with insights and recommendations
            
        Raises:
            ComparisonError: If algorithm comparison fails
        """
        pass


class HyperparameterOptimizationPort(ABC):
    """Port for hyperparameter optimization operations.
    
    This interface defines the contract for automated hyperparameter
    tuning using various optimization strategies.
    """

    @abstractmethod
    async def optimize_hyperparameters(
        self,
        algorithm_type: AlgorithmType,
        parameter_space: Dict[str, Any],
        objective_function: Callable,
        optimization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for specific algorithm.
        
        Args:
            algorithm_type: Algorithm to optimize hyperparameters for
            parameter_space: Search space for parameters
            objective_function: Function to optimize
            optimization_config: Configuration for optimization process
            
        Returns:
            Optimized hyperparameters with optimization history
            
        Raises:
            HyperparameterOptimizationError: If optimization fails
        """
        pass

    @abstractmethod
    async def suggest_parameter_ranges(
        self,
        algorithm_type: AlgorithmType,
        dataset_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest parameter ranges based on dataset characteristics.
        
        Args:
            algorithm_type: Algorithm to suggest parameters for
            dataset_characteristics: Characteristics of the dataset
            
        Returns:
            Suggested parameter ranges and defaults
            
        Raises:
            ParameterSuggestionError: If parameter suggestion fails
        """
        pass

    @abstractmethod
    async def validate_parameters(
        self,
        algorithm_type: AlgorithmType,
        parameters: Dict[str, Any]
    ) -> bool:
        """Validate parameter configuration for algorithm.
        
        Args:
            algorithm_type: Algorithm to validate parameters for
            parameters: Parameter configuration to validate
            
        Returns:
            True if parameters are valid, False otherwise
            
        Raises:
            ParameterValidationError: If validation process fails
        """
        pass


# Custom exceptions for AutoML operations
class AutoMLOperationError(Exception):
    """Base exception for AutoML operation errors."""
    pass


class OptimizationError(AutoMLOperationError):
    """Exception raised during optimization process."""
    pass


class ConfigurationError(AutoMLOperationError):
    """Exception raised for invalid configuration."""
    pass


class SuggestionError(AutoMLOperationError):
    """Exception raised during algorithm suggestion."""
    pass


class EvaluationError(AutoMLOperationError):
    """Exception raised during algorithm evaluation."""
    pass


class EnsembleOptimizationError(AutoMLOperationError):
    """Exception raised during ensemble optimization."""
    pass


class UnsupportedAlgorithmError(AutoMLOperationError):
    """Exception raised for unsupported algorithms."""
    pass


class ModelSelectionError(AutoMLOperationError):
    """Exception raised during model selection."""
    pass


class DatasetAnalysisError(AutoMLOperationError):
    """Exception raised during dataset analysis."""
    pass


class RecommendationError(AutoMLOperationError):
    """Exception raised during recommendation generation."""
    pass


class ComparisonError(AutoMLOperationError):
    """Exception raised during algorithm comparison."""
    pass


class HyperparameterOptimizationError(AutoMLOperationError):
    """Exception raised during hyperparameter optimization."""
    pass


class ParameterSuggestionError(AutoMLOperationError):
    """Exception raised during parameter suggestion."""
    pass


class ParameterValidationError(AutoMLOperationError):
    """Exception raised during parameter validation."""
    pass
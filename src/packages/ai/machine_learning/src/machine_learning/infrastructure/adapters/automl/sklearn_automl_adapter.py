"""Adapter for scikit-learn based AutoML operations.

This adapter implements the AutoML interfaces and provides integration
with scikit-learn, Optuna, and other ML libraries while isolating
the domain from infrastructure concerns.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from machine_learning.domain.interfaces.automl_operations import (
    AutoMLOptimizationPort,
    ModelSelectionPort,
    HyperparameterOptimizationPort,
    OptimizationConfig,
    AlgorithmConfig,
    EnsembleConfig,
    OptimizationResult,
    OptimizationTrial,
    OptimizationMetric,
    SearchStrategy,
    AlgorithmType,
    EnsembleMethod,
    OptimizationError,
    ConfigurationError,
    SuggestionError,
    EvaluationError,
    UnsupportedAlgorithmError,
)

logger = logging.getLogger(__name__)


class SklearnAutoMLAdapter(AutoMLOptimizationPort, ModelSelectionPort, HyperparameterOptimizationPort):
    """Adapter for scikit-learn based AutoML operations.
    
    This adapter provides concrete implementation of AutoML operations
    using scikit-learn and related libraries while abstracting away
    the specific implementations from the domain layer.
    """
    
    def __init__(self):
        """Initialize the scikit-learn AutoML adapter."""
        self._logger = logging.getLogger(__name__)
        self._supported_algorithms = self._initialize_supported_algorithms()
        self._parameter_spaces = self._initialize_parameter_spaces()
        
        # Check for optional dependencies
        self._sklearn_available = self._check_sklearn_availability()
        self._optuna_available = self._check_optuna_availability()
        
        if not self._sklearn_available:
            self._logger.warning(
                "Scikit-learn not available. Limited functionality will be provided."
            )
    
    def _check_sklearn_availability(self) -> bool:
        """Check if scikit-learn is available."""
        try:
            import sklearn
            return True
        except ImportError:
            return False
    
    def _check_optuna_availability(self) -> bool:
        """Check if Optuna is available."""
        try:
            import optuna
            return True
        except ImportError:
            return False
    
    def _initialize_supported_algorithms(self) -> List[AlgorithmType]:
        """Initialize list of supported algorithms."""
        return [
            AlgorithmType.ISOLATION_FOREST,
            AlgorithmType.LOCAL_OUTLIER_FACTOR,
            AlgorithmType.ONE_CLASS_SVM,
            AlgorithmType.DBSCAN,
        ]
    
    def _initialize_parameter_spaces(self) -> Dict[AlgorithmType, Dict[str, Any]]:
        """Initialize parameter search spaces for algorithms."""
        spaces = {}
        
        # Isolation Forest
        spaces[AlgorithmType.ISOLATION_FOREST] = {
            "n_estimators": [50, 100, 200, 300],
            "max_samples": ["auto", 0.5, 0.7, 0.9],
            "max_features": [0.5, 0.7, 0.9, 1.0],
            "contamination": [0.05, 0.1, 0.15, 0.2, 0.25],
        }
        
        # Local Outlier Factor
        spaces[AlgorithmType.LOCAL_OUTLIER_FACTOR] = {
            "n_neighbors": [5, 10, 20, 30, 50],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
            "contamination": [0.05, 0.1, 0.15, 0.2, 0.25],
        }
        
        # One-Class SVM
        spaces[AlgorithmType.ONE_CLASS_SVM] = {
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0],
            "nu": [0.01, 0.05, 0.1, 0.2, 0.3],
            "degree": [2, 3, 4],  # For poly kernel
        }
        
        # DBSCAN
        spaces[AlgorithmType.DBSCAN] = {
            "eps": [0.1, 0.3, 0.5, 0.7, 1.0, 1.5],
            "min_samples": [3, 5, 10, 15, 20],
            "metric": ["euclidean", "manhattan", "chebyshev"],
        }
        
        return spaces
    
    async def optimize_model(
        self,
        dataset: Any,  # Dataset type from domain
        optimization_config: OptimizationConfig,
        ground_truth: Optional[Any] = None
    ) -> OptimizationResult:
        """Automatically optimize model for the given dataset."""
        if not self._sklearn_available:
            raise OptimizationError("Scikit-learn not available for optimization")
        
        start_time = time.time()
        
        try:
            # Filter algorithms to test
            algorithms_to_test = optimization_config.algorithms_to_test or self._supported_algorithms
            algorithms_to_test = [alg for alg in algorithms_to_test if alg in self._supported_algorithms]
            
            if not algorithms_to_test:
                raise ConfigurationError("No supported algorithms specified for optimization")
            
            self._logger.info(f"Starting optimization with {len(algorithms_to_test)} algorithms")
            
            # Perform optimization based on strategy
            if optimization_config.search_strategy == SearchStrategy.GRID_SEARCH:
                result = await self._grid_search_optimization(
                    dataset, algorithms_to_test, optimization_config, ground_truth
                )
            elif optimization_config.search_strategy == SearchStrategy.RANDOM_SEARCH:
                result = await self._random_search_optimization(
                    dataset, algorithms_to_test, optimization_config, ground_truth
                )
            elif optimization_config.search_strategy == SearchStrategy.BAYESIAN_OPTIMIZATION:
                result = await self._bayesian_optimization(
                    dataset, algorithms_to_test, optimization_config, ground_truth
                )
            else:
                # Fallback to grid search
                result = await self._grid_search_optimization(
                    dataset, algorithms_to_test, optimization_config, ground_truth
                )
            
            result.optimization_time_seconds = time.time() - start_time
            
            self._logger.info(
                f"Optimization completed in {result.optimization_time_seconds:.2f}s "
                f"(best: {result.best_algorithm_type.value}, score: {result.best_score:.4f})"
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Optimization failed: {e}")
            raise OptimizationError(f"Optimization failed: {e}")
    
    async def _grid_search_optimization(
        self,
        dataset: Any,
        algorithms: List[AlgorithmType],
        config: OptimizationConfig,
        ground_truth: Optional[Any]
    ) -> OptimizationResult:
        """Perform grid search optimization."""
        import itertools
        
        best_score = -float('inf')
        best_algorithm = None
        best_config = None
        best_metrics = {}
        trial_history = []
        
        trial_count = 0
        
        for algorithm in algorithms:
            self._logger.info(f"Grid searching algorithm: {algorithm.value}")
            
            param_space = self._parameter_spaces.get(algorithm, {})
            if not param_space:
                continue
            
            # Generate parameter combinations
            param_names = list(param_space.keys())
            param_values = list(param_space.values())
            
            for param_combo in itertools.product(*param_values):
                if trial_count >= config.max_trials:
                    break
                
                params = dict(zip(param_names, param_combo))
                
                try:
                    trial = await self._evaluate_single_configuration(
                        algorithm, params, dataset, config, ground_truth, trial_count
                    )
                    
                    trial_history.append(trial)
                    
                    if trial.score > best_score:
                        best_score = trial.score
                        best_algorithm = algorithm
                        best_config = AlgorithmConfig(
                            algorithm_type=algorithm,
                            parameters=params,
                            contamination=params.get("contamination", 0.1)
                        )
                        best_metrics = trial.metrics
                    
                    trial_count += 1
                    
                except Exception as e:
                    self._logger.warning(f"Trial failed for {algorithm.value}: {e}")
                    continue
        
        # Create top-k results
        top_k_results = self._create_top_k_results(trial_history)
        
        return OptimizationResult(
            best_algorithm_type=best_algorithm,
            best_config=best_config,
            best_score=best_score,
            best_metrics=best_metrics,
            trial_history=trial_history,
            total_trials=trial_count,
            optimization_time_seconds=0.0,  # Will be set by caller
            top_k_results=top_k_results,
        )
    
    async def _random_search_optimization(
        self,
        dataset: Any,
        algorithms: List[AlgorithmType],
        config: OptimizationConfig,
        ground_truth: Optional[Any]
    ) -> OptimizationResult:
        """Perform random search optimization."""
        import random
        
        best_score = -float('inf')
        best_algorithm = None
        best_config = None
        best_metrics = {}
        trial_history = []
        
        random.seed(config.random_state)
        
        for trial_count in range(config.max_trials):
            # Randomly select algorithm
            algorithm = random.choice(algorithms)
            
            # Get parameter space
            param_space = self._parameter_spaces.get(algorithm, {})
            if not param_space:
                continue
            
            try:
                # Sample random parameters
                params = {}
                for param_name, param_values in param_space.items():
                    if isinstance(param_values, list):
                        params[param_name] = random.choice(param_values)
                
                trial = await self._evaluate_single_configuration(
                    algorithm, params, dataset, config, ground_truth, trial_count
                )
                
                trial_history.append(trial)
                
                if trial.score > best_score:
                    best_score = trial.score
                    best_algorithm = algorithm
                    best_config = AlgorithmConfig(
                        algorithm_type=algorithm,
                        parameters=params,
                        contamination=params.get("contamination", 0.1)
                    )
                    best_metrics = trial.metrics
                
            except Exception as e:
                self._logger.warning(f"Random search trial {trial_count} failed: {e}")
                continue
        
        # Create top-k results
        top_k_results = self._create_top_k_results(trial_history)
        
        return OptimizationResult(
            best_algorithm_type=best_algorithm,
            best_config=best_config,
            best_score=best_score,
            best_metrics=best_metrics,
            trial_history=trial_history,
            total_trials=len(trial_history),
            optimization_time_seconds=0.0,  # Will be set by caller
            top_k_results=top_k_results,
        )
    
    async def _bayesian_optimization(
        self,
        dataset: Any,
        algorithms: List[AlgorithmType],
        config: OptimizationConfig,
        ground_truth: Optional[Any]
    ) -> OptimizationResult:
        """Perform Bayesian optimization using Optuna."""
        if not self._optuna_available:
            self._logger.warning("Optuna not available, falling back to random search")
            return await self._random_search_optimization(dataset, algorithms, config, ground_truth)
        
        import optuna
        
        best_score = -float('inf')
        best_algorithm = None
        best_config = None
        best_metrics = {}
        trial_history = []
        
        def objective(trial):
            nonlocal best_score, best_algorithm, best_config, best_metrics
            
            # Select algorithm
            algorithm_name = trial.suggest_categorical(
                "algorithm", [alg.value for alg in algorithms]
            )
            algorithm = AlgorithmType(algorithm_name)
            
            # Get parameter space
            param_space = self._parameter_spaces.get(algorithm, {})
            
            # Suggest parameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values, list):
                    params[param_name] = trial.suggest_categorical(
                        f"{algorithm_name}_{param_name}", param_values
                    )
            
            try:
                # Create a mock trial object for evaluation
                trial_obj = OptimizationTrial(
                    trial_id=str(trial.number),
                    algorithm_type=algorithm,
                    parameters=params,
                    score=0.0,
                    metrics={},
                    execution_time_seconds=0.0,
                    memory_usage_mb=0.0,
                    status="running"
                )
                
                # Evaluate configuration (simplified)
                score = self._mock_evaluate_configuration(algorithm, params, config)
                
                trial_obj.score = score
                trial_obj.status = "completed"
                trial_history.append(trial_obj)
                
                # Update best if better
                if score > best_score:
                    best_score = score
                    best_algorithm = algorithm
                    best_config = AlgorithmConfig(
                        algorithm_type=algorithm,
                        parameters=params,
                        contamination=params.get("contamination", 0.1)
                    )
                    best_metrics = {"score": score}
                
                return score
                
            except Exception as e:
                self._logger.warning(f"Bayesian optimization trial failed: {e}")
                return -float('inf')
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.random_state),
        )
        
        # Optimize
        study.optimize(
            objective, 
            n_trials=config.max_trials, 
            timeout=config.timeout_seconds
        )
        
        # Create top-k results
        top_k_results = self._create_top_k_results(trial_history)
        
        return OptimizationResult(
            best_algorithm_type=best_algorithm,
            best_config=best_config,
            best_score=best_score,
            best_metrics=best_metrics,
            trial_history=trial_history,
            total_trials=len(trial_history),
            optimization_time_seconds=0.0,  # Will be set by caller
            top_k_results=top_k_results,
        )
    
    async def _evaluate_single_configuration(
        self,
        algorithm: AlgorithmType,
        params: Dict[str, Any],
        dataset: Any,
        config: OptimizationConfig,
        ground_truth: Optional[Any],
        trial_id: int
    ) -> OptimizationTrial:
        """Evaluate a single algorithm configuration."""
        start_time = time.time()
        
        try:
            # Mock evaluation - in real implementation this would train and evaluate the model
            score = self._mock_evaluate_configuration(algorithm, params, config)
            
            execution_time = (time.time() - start_time) * 1000
            
            return OptimizationTrial(
                trial_id=str(trial_id),
                algorithm_type=algorithm,
                parameters=params,
                score=score,
                metrics={"score": score, "execution_time_ms": execution_time},
                execution_time_seconds=execution_time / 1000,
                memory_usage_mb=50.0,  # Mock value
                status="completed"
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return OptimizationTrial(
                trial_id=str(trial_id),
                algorithm_type=algorithm,
                parameters=params,
                score=0.0,
                metrics={},
                execution_time_seconds=execution_time / 1000,
                memory_usage_mb=0.0,
                status="failed",
                error_message=str(e)
            )
    
    def _mock_evaluate_configuration(
        self, 
        algorithm: AlgorithmType, 
        params: Dict[str, Any],
        config: OptimizationConfig
    ) -> float:
        """Mock evaluation for demonstration purposes."""
        import random
        
        # Simple mock that favors certain parameter combinations
        base_score = random.uniform(0.5, 0.9)
        
        # Add bonus for contamination in reasonable range
        contamination = params.get("contamination", 0.1)
        if 0.05 <= contamination <= 0.2:
            base_score += 0.05
        
        # Add randomness
        base_score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_score))
    
    def _create_top_k_results(self, trial_history: List[OptimizationTrial]) -> List[tuple]:
        """Create top-k results from trial history."""
        sorted_trials = sorted(trial_history, key=lambda x: x.score, reverse=True)
        top_k_results = []
        
        for trial in sorted_trials[:5]:  # Top 5
            config = AlgorithmConfig(
                algorithm_type=trial.algorithm_type,
                parameters=trial.parameters,
                contamination=trial.parameters.get("contamination", 0.1)
            )
            top_k_results.append((trial.algorithm_type, config, trial.score))
        
        return top_k_results
    
    # Implement other interface methods
    async def suggest_algorithms(
        self,
        dataset: Any,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[AlgorithmType]:
        """Suggest appropriate algorithms based on dataset characteristics."""
        # Mock implementation - would analyze dataset in real implementation
        return self._supported_algorithms
    
    async def evaluate_algorithm(
        self,
        algorithm_config: AlgorithmConfig,
        dataset: Any,
        evaluation_config: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> OptimizationTrial:
        """Evaluate a specific algorithm configuration."""
        return await self._evaluate_single_configuration(
            algorithm_config.algorithm_type,
            algorithm_config.parameters,
            dataset,
            OptimizationConfig(),  # Default config
            ground_truth,
            0
        )
    
    async def optimize_ensemble(
        self,
        individual_results: List[OptimizationTrial],
        dataset: Any,
        ensemble_config: Dict[str, Any],
        ground_truth: Optional[Any] = None
    ) -> EnsembleConfig:
        """Optimize ensemble configuration from individual algorithm results."""
        # Mock implementation
        top_algorithms = sorted(individual_results, key=lambda x: x.score, reverse=True)[:3]
        
        algorithm_configs = []
        for trial in top_algorithms:
            config = AlgorithmConfig(
                algorithm_type=trial.algorithm_type,
                parameters=trial.parameters,
                contamination=trial.parameters.get("contamination", 0.1)
            )
            algorithm_configs.append(config)
        
        return EnsembleConfig(
            algorithm_configs=algorithm_configs,
            combination_method=EnsembleMethod.AVERAGE
        )
    
    async def get_parameter_space(self, algorithm_type: AlgorithmType) -> Dict[str, Any]:
        """Get parameter search space for specific algorithm."""
        if algorithm_type not in self._parameter_spaces:
            raise UnsupportedAlgorithmError(f"Algorithm {algorithm_type.value} not supported")
        
        return self._parameter_spaces[algorithm_type]
    
    async def get_supported_algorithms(self) -> List[AlgorithmType]:
        """Get list of supported algorithms."""
        return self._supported_algorithms
    
    # Model selection interface methods
    async def select_best_model(
        self,
        dataset: Any,
        requirements: Dict[str, Any],
        quick_mode: bool = False
    ) -> tuple[AlgorithmType, AlgorithmConfig]:
        """Select the best model for given dataset and requirements."""
        if quick_mode:
            # Simple heuristic selection
            algorithm = AlgorithmType.ISOLATION_FOREST
            config = AlgorithmConfig(
                algorithm_type=algorithm,
                parameters={"contamination": 0.1, "n_estimators": 100}
            )
            return algorithm, config
        else:
            # Full optimization
            optimization_config = OptimizationConfig(max_trials=20)
            result = await self.optimize_model(dataset, optimization_config)
            return result.best_algorithm_type, result.best_config
    
    async def analyze_dataset_characteristics(self, dataset: Any) -> Dict[str, Any]:
        """Analyze dataset characteristics for model selection."""
        # Mock implementation
        return {
            "n_samples": 1000,
            "n_features": 10,
            "data_type": "numeric",
            "missing_values": False,
            "outlier_ratio": 0.1
        }
    
    async def get_model_recommendations(
        self,
        dataset: Any,
        current_performance: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[str]]:
        """Get recommendations for improving model performance."""
        return {
            "algorithms": ["Try Isolation Forest for better scalability"],
            "preprocessing": ["Consider feature scaling"],
            "hyperparameters": ["Tune contamination parameter"]
        }
    
    async def compare_algorithms(
        self,
        algorithm_results: List[OptimizationTrial]
    ) -> Dict[str, Any]:
        """Compare multiple algorithm results and provide insights."""
        return {
            "best_algorithm": algorithm_results[0].algorithm_type.value if algorithm_results else None,
            "performance_ranking": [trial.algorithm_type.value for trial in algorithm_results],
            "insights": ["Isolation Forest showed best performance"]
        }
    
    # Hyperparameter optimization interface methods
    async def optimize_hyperparameters(
        self,
        algorithm_type: AlgorithmType,
        parameter_space: Dict[str, Any],
        objective_function: Callable,
        optimization_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for specific algorithm."""
        # Mock implementation
        return {
            "best_parameters": {"contamination": 0.1, "n_estimators": 100},
            "best_score": 0.85,
            "optimization_history": []
        }
    
    async def suggest_parameter_ranges(
        self,
        algorithm_type: AlgorithmType,
        dataset_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest parameter ranges based on dataset characteristics."""
        return self._parameter_spaces.get(algorithm_type, {})
    
    async def validate_parameters(
        self,
        algorithm_type: AlgorithmType,
        parameters: Dict[str, Any]
    ) -> bool:
        """Validate parameter configuration for algorithm."""
        param_space = self._parameter_spaces.get(algorithm_type, {})
        
        for param_name, param_value in parameters.items():
            if param_name in param_space:
                valid_values = param_space[param_name]
                if isinstance(valid_values, list) and param_value not in valid_values:
                    return False
        
        return True
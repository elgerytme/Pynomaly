"""
Hyperparameter Optimization Service

Advanced hyperparameter optimization service with support for multiple
optimization strategies, parallel execution, and intelligent search space
exploration for anomaly detection algorithms.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from pynomaly.domain.value_objects.hyperparameters import (
    HyperparameterSet, HyperparameterSpace, HyperparameterRange
)
from pynomaly.domain.entities.optimization_trial import OptimizationTrial, TrialStatus
from pynomaly.infrastructure.config.optimization_config import OptimizationConfig
from pynomaly.shared.exceptions import OptimizationError

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_trial: OptimizationTrial
    optimization_history: List[OptimizationTrial]
    total_trials: int
    optimization_time: float
    convergence_data: Dict[str, Any]


class HyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimizers."""
    
    @abstractmethod
    async def optimize(
        self,
        objective_function: Callable,
        search_space: HyperparameterSpace,
        n_trials: int,
        **kwargs
    ) -> OptimizationResult:
        """Optimize hyperparameters."""
        pass


class OptunaOptimizer(HyperparameterOptimizer):
    """Optuna-based hyperparameter optimizer with advanced features."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._import_optuna()
    
    def _import_optuna(self):
        """Import Optuna with error handling."""
        try:
            import optuna
            from optuna.samplers import (
                TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler
            )
            from optuna.pruners import (
                MedianPruner, PercentilePruner, SuccessiveHalvingPruner,
                HyperbandPruner
            )
            
            self.optuna = optuna
            self.samplers = {
                'tpe': TPESampler,
                'random': RandomSampler,
                'cmaes': CmaEsSampler,
                'nsga2': NSGAIISampler
            }
            self.pruners = {
                'median': MedianPruner,
                'percentile': PercentilePruner,
                'successive_halving': SuccessiveHalvingPruner,
                'hyperband': HyperbandPruner
            }
            
        except ImportError:
            raise OptimizationError(
                "Optuna not available. Install with: pip install optuna"
            )
    
    async def optimize(
        self,
        objective_function: Callable,
        search_space: HyperparameterSpace,
        n_trials: int,
        study_name: Optional[str] = None,
        direction: str = 'maximize',
        sampler_name: str = 'tpe',
        pruner_name: str = 'median',
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            objective_function: Function to optimize
            search_space: Hyperparameter search space
            n_trials: Number of optimization trials
            study_name: Name for the optimization study
            direction: Optimization direction ('maximize' or 'minimize')
            sampler_name: Sampler strategy ('tpe', 'random', 'cmaes', 'nsga2')
            pruner_name: Pruning strategy ('median', 'percentile', 'successive_halving', 'hyperband')
            
        Returns:
            OptimizationResult with best parameters and optimization history
        """
        start_time = time.time()
        
        # Create sampler
        sampler_class = self.samplers.get(sampler_name, self.samplers['tpe'])
        sampler = sampler_class(seed=self.config.random_seed)
        
        # Create pruner
        pruner_class = self.pruners.get(pruner_name, self.pruners['median'])
        pruner = pruner_class()
        
        # Create study
        study = self.optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # Define objective wrapper
        def objective_wrapper(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_range in search_space.parameters.items():
                params[param_name] = self._sample_parameter(trial, param_name, param_range)
            
            # Evaluate objective
            try:
                score = objective_function(params)
                
                # Handle pruning for multi-step optimization
                if hasattr(objective_function, 'intermediate_values'):
                    for step, value in enumerate(objective_function.intermediate_values):
                        trial.report(value, step)
                        if trial.should_prune():
                            raise self.optuna.TrialPruned()
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                # Return worst possible score for failed trials
                return 0.0 if direction == 'maximize' else float('inf')
        
        # Run optimization
        logger.info(f"Starting Optuna optimization with {n_trials} trials")
        study.optimize(objective_wrapper, n_trials=n_trials, n_jobs=self.config.n_jobs)
        
        optimization_time = time.time() - start_time
        
        # Convert trials to our format
        optimization_history = []
        for trial in study.trials:
            opt_trial = OptimizationTrial(
                trial_id=trial.number,
                parameters=HyperparameterSet(trial.params),
                score=trial.value if trial.value is not None else 0.0,
                status=self._convert_trial_state(trial.state),
                start_time=trial.datetime_start or datetime.utcnow(),
                end_time=trial.datetime_complete,
                duration=(trial.datetime_complete - trial.datetime_start).total_seconds() 
                         if trial.datetime_complete and trial.datetime_start else 0.0,
                intermediate_values=trial.intermediate_values,
                user_attrs=trial.user_attrs,
                system_attrs=trial.system_attrs
            )
            optimization_history.append(opt_trial)
        
        # Get best trial
        best_trial = max(optimization_history, key=lambda t: t.score)
        
        # Calculate convergence data
        convergence_data = self._calculate_convergence_data(optimization_history)
        
        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_trial=best_trial,
            optimization_history=optimization_history,
            total_trials=len(study.trials),
            optimization_time=optimization_time,
            convergence_data=convergence_data
        )
    
    def _sample_parameter(
        self,
        trial,
        param_name: str,
        param_range: HyperparameterRange
    ) -> Any:
        """Sample a parameter value using Optuna trial."""
        if param_range.type == 'categorical':
            return trial.suggest_categorical(param_name, param_range.choices)
            
        elif param_range.type == 'float':
            return trial.suggest_float(
                param_name,
                param_range.low,
                param_range.high,
                log=param_range.log,
                step=param_range.step
            )
            
        elif param_range.type == 'int':
            return trial.suggest_int(
                param_name,
                param_range.low,
                param_range.high,
                step=param_range.step,
                log=param_range.log
            )
            
        elif param_range.type == 'discrete':
            return trial.suggest_categorical(param_name, param_range.choices)
            
        else:
            raise OptimizationError(f"Unknown parameter type: {param_range.type}")
    
    def _convert_trial_state(self, optuna_state) -> TrialStatus:
        """Convert Optuna trial state to our TrialStatus."""
        state_mapping = {
            self.optuna.trial.TrialState.COMPLETE: TrialStatus.COMPLETED,
            self.optuna.trial.TrialState.PRUNED: TrialStatus.PRUNED,
            self.optuna.trial.TrialState.FAIL: TrialStatus.FAILED,
            self.optuna.trial.TrialState.RUNNING: TrialStatus.RUNNING,
            self.optuna.trial.TrialState.WAITING: TrialStatus.PENDING
        }
        return state_mapping.get(optuna_state, TrialStatus.FAILED)
    
    def _calculate_convergence_data(
        self,
        trials: List[OptimizationTrial]
    ) -> Dict[str, Any]:
        """Calculate convergence statistics."""
        scores = [trial.score for trial in trials if trial.status == TrialStatus.COMPLETED]
        
        if not scores:
            return {}
        
        # Best score progression
        best_scores = []
        current_best = scores[0]
        
        for score in scores:
            if score > current_best:
                current_best = score
            best_scores.append(current_best)
        
        # Convergence metrics
        final_scores = scores[-10:] if len(scores) >= 10 else scores
        convergence_variance = np.var(final_scores) if len(final_scores) > 1 else 0.0
        
        return {
            'best_score_progression': best_scores,
            'convergence_variance': convergence_variance,
            'improvement_rate': (scores[-1] - scores[0]) / len(scores) if len(scores) > 1 else 0.0,
            'plateau_detection': self._detect_plateau(best_scores),
            'early_stopping_point': self._find_early_stopping_point(best_scores)
        }
    
    def _detect_plateau(self, best_scores: List[float], window_size: int = 20) -> Dict[str, Any]:
        """Detect if optimization has plateaued."""
        if len(best_scores) < window_size:
            return {'plateaued': False, 'plateau_length': 0}
        
        recent_scores = best_scores[-window_size:]
        improvement = recent_scores[-1] - recent_scores[0]
        relative_improvement = improvement / abs(recent_scores[0]) if recent_scores[0] != 0 else 0
        
        plateaued = relative_improvement < 0.001  # Less than 0.1% improvement
        
        return {
            'plateaued': plateaued,
            'plateau_length': window_size if plateaued else 0,
            'recent_improvement': relative_improvement
        }
    
    def _find_early_stopping_point(
        self,
        best_scores: List[float],
        patience: int = 50
    ) -> Optional[int]:
        """Find optimal early stopping point."""
        if len(best_scores) < patience:
            return None
        
        best_score = max(best_scores)
        best_index = best_scores.index(best_score)
        
        # Check if we haven't improved for 'patience' trials after the best score
        if len(best_scores) - best_index >= patience:
            return best_index + patience
        
        return None


class GridSearchOptimizer(HyperparameterOptimizer):
    """Grid search optimizer for exhaustive parameter exploration."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    async def optimize(
        self,
        objective_function: Callable,
        search_space: HyperparameterSpace,
        n_trials: Optional[int] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize hyperparameters using grid search.
        
        Args:
            objective_function: Function to optimize
            search_space: Hyperparameter search space
            n_trials: Maximum number of trials (optional for grid search)
            
        Returns:
            OptimizationResult with best parameters and optimization history
        """
        start_time = time.time()
        
        # Generate parameter grid
        param_grid = self._generate_parameter_grid(search_space)
        
        # Limit trials if specified
        if n_trials and len(param_grid) > n_trials:
            # Sample randomly from grid
            import random
            random.shuffle(param_grid)
            param_grid = param_grid[:n_trials]
        
        logger.info(f"Starting grid search with {len(param_grid)} parameter combinations")
        
        # Evaluate all parameter combinations
        optimization_history = []
        best_score = float('-inf')
        best_params = None
        best_trial = None
        
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(objective_function, params): (i, params)
                for i, params in enumerate(param_grid)
            }
            
            # Collect results
            for future in as_completed(future_to_params):
                trial_id, params = future_to_params[future]
                
                try:
                    score = future.result()
                    status = TrialStatus.COMPLETED
                except Exception as e:
                    logger.warning(f"Trial {trial_id} failed: {e}")
                    score = 0.0
                    status = TrialStatus.FAILED
                
                # Create trial
                trial = OptimizationTrial(
                    trial_id=trial_id,
                    parameters=HyperparameterSet(params),
                    score=score,
                    status=status,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    duration=0.0  # Not tracking individual trial duration in grid search
                )
                
                optimization_history.append(trial)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_trial = trial
        
        optimization_time = time.time() - start_time
        
        # Sort history by trial ID
        optimization_history.sort(key=lambda t: t.trial_id)
        
        # Calculate convergence data
        convergence_data = self._calculate_convergence_data(optimization_history)
        
        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            best_trial=best_trial,
            optimization_history=optimization_history,
            total_trials=len(optimization_history),
            optimization_time=optimization_time,
            convergence_data=convergence_data
        )
    
    def _generate_parameter_grid(self, search_space: HyperparameterSpace) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        import itertools
        
        param_lists = {}
        
        for param_name, param_range in search_space.parameters.items():
            if param_range.type == 'categorical':
                param_lists[param_name] = param_range.choices
                
            elif param_range.type == 'discrete':
                param_lists[param_name] = param_range.choices
                
            elif param_range.type in ['float', 'int']:
                # Generate grid points
                if param_range.type == 'float':
                    if param_range.log:
                        values = np.logspace(
                            np.log10(param_range.low),
                            np.log10(param_range.high),
                            num=param_range.grid_size or 10
                        )
                    else:
                        values = np.linspace(
                            param_range.low,
                            param_range.high,
                            num=param_range.grid_size or 10
                        )
                else:  # int
                    values = np.linspace(
                        param_range.low,
                        param_range.high,
                        num=min(param_range.grid_size or 10, param_range.high - param_range.low + 1),
                        dtype=int
                    )
                
                param_lists[param_name] = values.tolist()
        
        # Generate all combinations
        param_names = list(param_lists.keys())
        param_values = list(param_lists.values())
        
        param_combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            param_combinations.append(param_dict)
        
        return param_combinations
    
    def _calculate_convergence_data(
        self,
        trials: List[OptimizationTrial]
    ) -> Dict[str, Any]:
        """Calculate convergence statistics for grid search."""
        scores = [trial.score for trial in trials if trial.status == TrialStatus.COMPLETED]
        
        if not scores:
            return {}
        
        return {
            'score_distribution': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'percentiles': {
                    '25': np.percentile(scores, 25),
                    '50': np.percentile(scores, 50),
                    '75': np.percentile(scores, 75)
                }
            },
            'exploration_completeness': 1.0,  # Grid search explores all combinations
            'parameter_sensitivity': self._analyze_parameter_sensitivity(trials)
        }
    
    def _analyze_parameter_sensitivity(
        self,
        trials: List[OptimizationTrial]
    ) -> Dict[str, float]:
        """Analyze sensitivity of each parameter."""
        sensitivity = {}
        
        if not trials:
            return sensitivity
        
        # Get all parameter names
        param_names = set()
        for trial in trials:
            param_names.update(trial.parameters.parameters.keys())
        
        for param_name in param_names:
            # Calculate correlation between parameter value and score
            param_values = []
            scores = []
            
            for trial in trials:
                if (trial.status == TrialStatus.COMPLETED and 
                    param_name in trial.parameters.parameters):
                    param_values.append(trial.parameters.parameters[param_name])
                    scores.append(trial.score)
            
            if len(param_values) > 1 and len(set(param_values)) > 1:
                # Convert categorical to numeric if needed
                if isinstance(param_values[0], str):
                    unique_values = list(set(param_values))
                    param_values = [unique_values.index(val) for val in param_values]
                
                correlation = np.corrcoef(param_values, scores)[0, 1]
                sensitivity[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                sensitivity[param_name] = 0.0
        
        return sensitivity


class RandomSearchOptimizer(HyperparameterOptimizer):
    """Random search optimizer for efficient parameter exploration."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    async def optimize(
        self,
        objective_function: Callable,
        search_space: HyperparameterSpace,
        n_trials: int,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize hyperparameters using random search.
        
        Args:
            objective_function: Function to optimize
            search_space: Hyperparameter search space
            n_trials: Number of random trials
            
        Returns:
            OptimizationResult with best parameters and optimization history
        """
        start_time = time.time()
        
        logger.info(f"Starting random search with {n_trials} trials")
        
        # Generate random parameter combinations
        param_combinations = []
        for i in range(n_trials):
            params = self._sample_random_parameters(search_space)
            param_combinations.append((i, params))
        
        # Evaluate parameter combinations
        optimization_history = []
        best_score = float('-inf')
        best_params = None
        best_trial = None
        
        # Use ThreadPoolExecutor for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(objective_function, params): (trial_id, params)
                for trial_id, params in param_combinations
            }
            
            # Collect results
            for future in as_completed(future_to_params):
                trial_id, params = future_to_params[future]
                
                try:
                    score = future.result()
                    status = TrialStatus.COMPLETED
                except Exception as e:
                    logger.warning(f"Trial {trial_id} failed: {e}")
                    score = 0.0
                    status = TrialStatus.FAILED
                
                # Create trial
                trial = OptimizationTrial(
                    trial_id=trial_id,
                    parameters=HyperparameterSet(params),
                    score=score,
                    status=status,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    duration=0.0
                )
                
                optimization_history.append(trial)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_trial = trial
        
        optimization_time = time.time() - start_time
        
        # Sort history by trial ID
        optimization_history.sort(key=lambda t: t.trial_id)
        
        # Calculate convergence data
        convergence_data = self._calculate_convergence_data(optimization_history)
        
        return OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            best_trial=best_trial,
            optimization_history=optimization_history,
            total_trials=len(optimization_history),
            optimization_time=optimization_time,
            convergence_data=convergence_data
        )
    
    def _sample_random_parameters(self, search_space: HyperparameterSpace) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        
        for param_name, param_range in search_space.parameters.items():
            if param_range.type == 'categorical':
                params[param_name] = np.random.choice(param_range.choices)
                
            elif param_range.type == 'discrete':
                params[param_name] = np.random.choice(param_range.choices)
                
            elif param_range.type == 'float':
                if param_range.log:
                    # Log-uniform sampling
                    log_low = np.log(param_range.low)
                    log_high = np.log(param_range.high)
                    params[param_name] = np.exp(np.random.uniform(log_low, log_high))
                else:
                    # Uniform sampling
                    params[param_name] = np.random.uniform(param_range.low, param_range.high)
                    
            elif param_range.type == 'int':
                if param_range.log:
                    # Log-uniform sampling for integers
                    log_low = np.log(param_range.low)
                    log_high = np.log(param_range.high)
                    params[param_name] = int(np.exp(np.random.uniform(log_low, log_high)))
                else:
                    # Uniform sampling
                    params[param_name] = np.random.randint(param_range.low, param_range.high + 1)
        
        return params
    
    def _calculate_convergence_data(
        self,
        trials: List[OptimizationTrial]
    ) -> Dict[str, Any]:
        """Calculate convergence statistics for random search."""
        scores = [trial.score for trial in trials if trial.status == TrialStatus.COMPLETED]
        
        if not scores:
            return {}
        
        # Best score progression
        best_scores = []
        current_best = float('-inf')
        
        for trial in sorted(trials, key=lambda t: t.trial_id):
            if trial.status == TrialStatus.COMPLETED and trial.score > current_best:
                current_best = trial.score
            best_scores.append(current_best)
        
        return {
            'best_score_progression': best_scores,
            'score_distribution': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'exploration_efficiency': self._calculate_exploration_efficiency(trials),
            'convergence_rate': self._calculate_convergence_rate(best_scores)
        }
    
    def _calculate_exploration_efficiency(
        self,
        trials: List[OptimizationTrial]
    ) -> float:
        """Calculate how efficiently the search space was explored."""
        # Simple metric: fraction of successful trials
        successful_trials = sum(1 for trial in trials if trial.status == TrialStatus.COMPLETED)
        return successful_trials / len(trials) if trials else 0.0
    
    def _calculate_convergence_rate(self, best_scores: List[float]) -> float:
        """Calculate the rate of convergence."""
        if len(best_scores) < 2:
            return 0.0
        
        # Calculate improvement rate
        improvements = []
        for i in range(1, len(best_scores)):
            if best_scores[i] > best_scores[i-1]:
                improvements.append(best_scores[i] - best_scores[i-1])
        
        return np.mean(improvements) if improvements else 0.0


class HyperparameterOptimizationService:
    """
    Main service for hyperparameter optimization with support for multiple strategies.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimizers = {
            'optuna': OptunaOptimizer(config),
            'grid_search': GridSearchOptimizer(config),
            'random_search': RandomSearchOptimizer(config)
        }
    
    async def optimize_hyperparameters(
        self,
        objective_function: Callable,
        search_space: HyperparameterSpace,
        strategy: str = 'optuna',
        n_trials: int = 100,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize hyperparameters using the specified strategy.
        
        Args:
            objective_function: Function to optimize (higher is better)
            search_space: Hyperparameter search space
            strategy: Optimization strategy ('optuna', 'grid_search', 'random_search')
            n_trials: Number of trials for optimization
            **kwargs: Additional arguments for the optimizer
            
        Returns:
            OptimizationResult with best parameters and optimization history
        """
        if strategy not in self.optimizers:
            raise OptimizationError(f"Unknown optimization strategy: {strategy}")
        
        optimizer = self.optimizers[strategy]
        
        logger.info(f"Starting hyperparameter optimization using {strategy}")
        result = await optimizer.optimize(
            objective_function=objective_function,
            search_space=search_space,
            n_trials=n_trials,
            **kwargs
        )
        
        logger.info(
            f"Optimization completed. Best score: {result.best_score:.4f} "
            f"in {result.optimization_time:.2f}s with {result.total_trials} trials"
        )
        
        return result
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available optimization strategies."""
        return list(self.optimizers.keys())
    
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """Get information about a specific optimization strategy."""
        strategy_info = {
            'optuna': {
                'name': 'Optuna',
                'description': 'Tree-structured Parzen Estimator with pruning',
                'strengths': ['Efficient for high-dimensional spaces', 'Automatic pruning', 'Multiple samplers'],
                'best_for': 'Most cases, especially complex search spaces',
                'supports_pruning': True,
                'supports_parallel': True
            },
            'grid_search': {
                'name': 'Grid Search',
                'description': 'Exhaustive search over parameter grid',
                'strengths': ['Guarantees finding global optimum in grid', 'Simple and interpretable'],
                'best_for': 'Small search spaces, thorough exploration',
                'supports_pruning': False,
                'supports_parallel': True
            },
            'random_search': {
                'name': 'Random Search',
                'description': 'Random sampling from parameter distributions',
                'strengths': ['Good baseline', 'Works well for high dimensions', 'Fast'],
                'best_for': 'Quick exploration, high-dimensional spaces',
                'supports_pruning': False,
                'supports_parallel': True
            }
        }
        
        return strategy_info.get(strategy, {})
    
    def suggest_strategy(
        self,
        search_space: HyperparameterSpace,
        max_trials: int,
        time_budget: Optional[float] = None
    ) -> str:
        """
        Suggest the best optimization strategy based on problem characteristics.
        
        Args:
            search_space: Hyperparameter search space
            max_trials: Maximum number of trials
            time_budget: Time budget in seconds (optional)
            
        Returns:
            Recommended optimization strategy
        """
        n_params = len(search_space.parameters)
        
        # Estimate grid size
        grid_size = 1
        for param_range in search_space.parameters.values():
            if param_range.type == 'categorical':
                grid_size *= len(param_range.choices)
            elif param_range.type == 'discrete':
                grid_size *= len(param_range.choices)
            else:
                grid_size *= param_range.grid_size or 10
        
        # Decision logic
        if grid_size <= max_trials and n_params <= 4:
            return 'grid_search'
        elif max_trials < 50 or (time_budget and time_budget < 300):
            return 'random_search'
        else:
            return 'optuna'
"""
Advanced Hyperparameter Optimization

Comprehensive hyperparameter optimization system with Bayesian optimization,
multi-fidelity methods, population-based training, and automated search spaces.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
import skopt
from skopt import gp_minimize, forest_minimize, dummy_minimize
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import structlog

from .advanced_automl_engine import AutoMLTask, OptimizationObjective


class OptimizationMethod(Enum):
    """Hyperparameter optimization methods."""
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    TPE = "tpe"  # Tree-structured Parzen Estimator
    CMA_ES = "cma_es"  # Covariance Matrix Adaptation
    POPULATION_BASED = "population_based"
    HYPERBAND = "hyperband"
    BOHB = "bohb"  # Bayesian Optimization and HyperBand
    MULTI_FIDELITY = "multi_fidelity"
    EVOLUTIONARY = "evolutionary"


class SearchSpaceType(Enum):
    """Types of hyperparameter search spaces."""
    CATEGORICAL = "categorical"
    INTEGER = "integer"
    FLOAT = "float"
    LOG_UNIFORM = "log_uniform"
    UNIFORM = "uniform"
    DISCRETE = "discrete"


class BudgetType(Enum):
    """Types of optimization budgets."""
    TIME_BUDGET = "time_budget"
    TRIAL_BUDGET = "trial_budget"
    EVALUATION_BUDGET = "evaluation_budget"
    RESOURCE_BUDGET = "resource_budget"


@dataclass
class SearchSpace:
    """Hyperparameter search space definition."""
    parameter_name: str
    space_type: SearchSpaceType
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    step: Optional[Union[int, float]] = None
    
    def to_optuna_suggest(self, trial: optuna.Trial) -> Any:
        """Convert to Optuna suggestion."""
        
        if self.space_type == SearchSpaceType.CATEGORICAL:
            return trial.suggest_categorical(self.parameter_name, self.choices)
        elif self.space_type == SearchSpaceType.INTEGER:
            return trial.suggest_int(self.parameter_name, self.low, self.high, step=self.step, log=self.log)
        elif self.space_type == SearchSpaceType.FLOAT:
            return trial.suggest_float(self.parameter_name, self.low, self.high, step=self.step, log=self.log)
        elif self.space_type == SearchSpaceType.LOG_UNIFORM:
            return trial.suggest_float(self.parameter_name, self.low, self.high, log=True)
        elif self.space_type == SearchSpaceType.UNIFORM:
            return trial.suggest_float(self.parameter_name, self.low, self.high, log=False)
        elif self.space_type == SearchSpaceType.DISCRETE:
            return trial.suggest_discrete_uniform(self.parameter_name, self.low, self.high, self.step)
        else:
            raise ValueError(f"Unknown search space type: {self.space_type}")
    
    def to_skopt_dimension(self):
        """Convert to scikit-optimize dimension."""
        
        if self.space_type == SearchSpaceType.CATEGORICAL:
            from skopt.space import Categorical
            return Categorical(self.choices, name=self.parameter_name)
        elif self.space_type == SearchSpaceType.INTEGER:
            from skopt.space import Integer
            return Integer(self.low, self.high, name=self.parameter_name)
        elif self.space_type == SearchSpaceType.FLOAT:
            from skopt.space import Real
            return Real(self.low, self.high, name=self.parameter_name)
        elif self.space_type == SearchSpaceType.LOG_UNIFORM:
            from skopt.space import Real
            return Real(self.low, self.high, prior='log-uniform', name=self.parameter_name)
        else:
            raise ValueError(f"Search space type {self.space_type} not supported for scikit-optimize")


@dataclass
class HPOConfig:
    """Configuration for hyperparameter optimization."""
    method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION
    task_type: AutoMLTask = AutoMLTask.BINARY_CLASSIFICATION
    optimization_objective: OptimizationObjective = OptimizationObjective.ACCURACY
    
    # Budget configuration
    budget_type: BudgetType = BudgetType.TRIAL_BUDGET
    max_trials: int = 100
    max_time_minutes: int = 60
    max_evaluations: int = 1000
    
    # Cross-validation
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 10
    min_improvement: float = 0.001
    
    # Multi-fidelity optimization
    enable_multi_fidelity: bool = True
    min_fidelity: float = 0.1  # Minimum data fraction
    max_fidelity: float = 1.0  # Maximum data fraction
    
    # Population-based training
    population_size: int = 20
    elite_fraction: float = 0.2
    mutation_rate: float = 0.8
    
    # Parallelization
    n_jobs: int = -1
    enable_distributed: bool = False
    
    # Advanced options
    enable_pruning: bool = True
    pruning_patience: int = 5
    enable_transfer_learning: bool = False
    warm_start_trials: int = 0


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    optimization_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION
    
    # Best results
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    best_trial_number: int = 0
    
    # Optimization history
    trial_history: List[Dict[str, Any]] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)
    
    # Performance metrics
    total_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0
    pruned_trials: int = 0
    
    # Time tracking
    optimization_time_seconds: float = 0.0
    average_trial_time_seconds: float = 0.0
    
    # Convergence analysis
    convergence_trial: Optional[int] = None
    improvement_rate: float = 0.0
    
    # Search space analysis
    parameter_importance: Dict[str, float] = field(default_factory=dict)
    correlation_matrix: Optional[np.ndarray] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    config: Optional[HPOConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimization_id": self.optimization_id,
            "method": self.method.value,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "best_trial_number": self.best_trial_number,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "failed_trials": self.failed_trials,
            "pruned_trials": self.pruned_trials,
            "optimization_time_seconds": self.optimization_time_seconds,
            "average_trial_time_seconds": self.average_trial_time_seconds,
            "convergence_trial": self.convergence_trial,
            "improvement_rate": self.improvement_rate,
            "parameter_importance": self.parameter_importance,
            "created_at": self.created_at.isoformat()
        }


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization engine."""
    
    def __init__(self, config: HPOConfig = None):
        self.config = config or HPOConfig()
        self.logger = structlog.get_logger(__name__)
        
        # Optimization state
        self.active_studies: Dict[str, optuna.Study] = {}
        self.optimization_results: Dict[str, OptimizationResult] = {}
        
        # Search space registry
        self.search_spaces: Dict[str, List[SearchSpace]] = {}
        
        # Optimization engines
        self.bayesian_optimizer = BayesianOptimizer(self.config)
        self.population_trainer = PopulationBasedTrainer(self.config)
        self.multi_fidelity_optimizer = MultiFidelityOptimizer(self.config)
        
        # Performance tracking
        self.performance_tracker = OptimizationPerformanceTracker()
        
        # Execution resources
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def optimize(self,
                      objective_function: Callable,
                      search_space: List[SearchSpace],
                      optimization_name: str = "") -> OptimizationResult:
        """Perform hyperparameter optimization."""
        
        optimization_id = str(uuid.uuid4())
        
        if not optimization_name:
            optimization_name = f"hpo_{optimization_id[:8]}"
        
        start_time = datetime.utcnow()
        
        self.logger.info(
            "Starting hyperparameter optimization",
            optimization_id=optimization_id,
            optimization_name=optimization_name,
            method=self.config.method.value,
            max_trials=self.config.max_trials
        )
        
        # Initialize result
        result = OptimizationResult(
            optimization_id=optimization_id,
            method=self.config.method,
            config=self.config
        )
        
        self.optimization_results[optimization_id] = result
        self.search_spaces[optimization_id] = search_space
        
        try:
            # Select optimization method
            if self.config.method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
                result = await self._bayesian_optimization(optimization_id, objective_function, search_space)
            elif self.config.method == OptimizationMethod.TPE:
                result = await self._tpe_optimization(optimization_id, objective_function, search_space)
            elif self.config.method == OptimizationMethod.POPULATION_BASED:
                result = await self._population_based_optimization(optimization_id, objective_function, search_space)
            elif self.config.method == OptimizationMethod.HYPERBAND:
                result = await self._hyperband_optimization(optimization_id, objective_function, search_space)
            elif self.config.method == OptimizationMethod.MULTI_FIDELITY:
                result = await self._multi_fidelity_optimization(optimization_id, objective_function, search_space)
            else:
                result = await self._random_optimization(optimization_id, objective_function, search_space)
            
            # Finalize result
            result.optimization_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            result.average_trial_time_seconds = (
                result.optimization_time_seconds / result.total_trials 
                if result.total_trials > 0 else 0
            )
            
            # Analyze convergence
            result.convergence_trial = self._analyze_convergence(result.score_history)
            result.improvement_rate = self._calculate_improvement_rate(result.score_history)
            
            # Parameter importance analysis
            if len(result.trial_history) > 10:
                result.parameter_importance = await self._analyze_parameter_importance(
                    result.trial_history, search_space
                )
            
            self.logger.info(
                "Hyperparameter optimization completed",
                optimization_id=optimization_id,
                best_score=result.best_score,
                total_trials=result.total_trials,
                optimization_time=result.optimization_time_seconds
            )
            
        except Exception as e:
            self.logger.error(
                "Hyperparameter optimization failed",
                optimization_id=optimization_id,
                error=str(e)
            )
            raise
        
        return result
    
    async def _bayesian_optimization(self,
                                   optimization_id: str,
                                   objective_function: Callable,
                                   search_space: List[SearchSpace]) -> OptimizationResult:
        """Perform Bayesian optimization using scikit-optimize."""
        
        result = self.optimization_results[optimization_id]
        
        # Convert search space to scikit-optimize format
        dimensions = [space.to_skopt_dimension() for space in search_space]
        
        # Objective function wrapper
        def skopt_objective(params):
            # Convert list to dictionary
            param_dict = {}
            for i, space in enumerate(search_space):
                param_dict[space.parameter_name] = params[i]
            
            try:
                score = objective_function(param_dict)
                
                # Record trial
                trial_info = {
                    "trial_number": len(result.trial_history),
                    "params": param_dict.copy(),
                    "score": score,
                    "status": "complete"
                }
                result.trial_history.append(trial_info)
                result.score_history.append(score)
                result.successful_trials += 1
                
                # Update best
                if self._is_better_score(score, result.best_score):
                    result.best_score = score
                    result.best_params = param_dict.copy()
                    result.best_trial_number = len(result.trial_history) - 1
                
                # For minimization (scikit-optimize minimizes by default)
                if self._is_maximization_objective():
                    return -score
                else:
                    return score
                
            except Exception as e:
                result.failed_trials += 1
                self.logger.warning(f"Trial failed: {str(e)}")
                return 1e6  # Large penalty for failed trials
        
        # Run optimization
        opt_result = gp_minimize(
            func=skopt_objective,
            dimensions=dimensions,
            n_calls=self.config.max_trials,
            n_initial_points=min(10, self.config.max_trials // 4),
            acq_func='gp_hedge',
            random_state=self.config.random_state
        )
        
        result.total_trials = len(result.trial_history)
        
        return result
    
    async def _tpe_optimization(self,
                              optimization_id: str,
                              objective_function: Callable,
                              search_space: List[SearchSpace]) -> OptimizationResult:
        """Perform Tree-structured Parzen Estimator optimization using Optuna."""
        
        result = self.optimization_results[optimization_id]
        
        # Create Optuna study
        direction = 'maximize' if self._is_maximization_objective() else 'minimize'
        
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner() if self.config.enable_pruning else optuna.pruners.NopPruner()
        )
        
        self.active_studies[optimization_id] = study
        
        # Objective function wrapper
        def optuna_objective(trial):
            # Generate parameters
            params = {}
            for space in search_space:
                params[space.parameter_name] = space.to_optuna_suggest(trial)
            
            try:
                score = objective_function(params)
                
                # Record trial
                trial_info = {
                    "trial_number": trial.number,
                    "params": params.copy(),
                    "score": score,
                    "status": "complete"
                }
                result.trial_history.append(trial_info)
                result.score_history.append(score)
                result.successful_trials += 1
                
                # Update best
                if self._is_better_score(score, result.best_score):
                    result.best_score = score
                    result.best_params = params.copy()
                    result.best_trial_number = trial.number
                
                return score
                
            except Exception as e:
                result.failed_trials += 1
                self.logger.warning(f"Trial {trial.number} failed: {str(e)}")
                raise optuna.TrialPruned()
        
        # Run optimization
        study.optimize(
            optuna_objective,
            n_trials=self.config.max_trials,
            timeout=self.config.max_time_minutes * 60 if self.config.budget_type == BudgetType.TIME_BUDGET else None
        )
        
        # Extract results
        result.total_trials = len(study.trials)
        result.pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        
        if study.best_trial:
            result.best_params = study.best_trial.params
            result.best_score = study.best_trial.value
            result.best_trial_number = study.best_trial.number
        
        return result
    
    async def _population_based_optimization(self,
                                           optimization_id: str,
                                           objective_function: Callable,
                                           search_space: List[SearchSpace]) -> OptimizationResult:
        """Perform population-based training optimization."""
        
        return await self.population_trainer.optimize(
            optimization_id, objective_function, search_space
        )
    
    async def _hyperband_optimization(self,
                                    optimization_id: str,
                                    objective_function: Callable,
                                    search_space: List[SearchSpace]) -> OptimizationResult:
        """Perform Hyperband optimization."""
        
        result = self.optimization_results[optimization_id]
        
        # Hyperband parameters
        max_iter = 81
        eta = 3
        
        # Generate initial configurations
        n_configs = max_iter
        configs = []
        
        for _ in range(n_configs):
            config = {}
            for space in search_space:
                if space.space_type == SearchSpaceType.CATEGORICAL:
                    config[space.parameter_name] = np.random.choice(space.choices)
                elif space.space_type == SearchSpaceType.INTEGER:
                    config[space.parameter_name] = np.random.randint(space.low, space.high + 1)
                elif space.space_type == SearchSpaceType.FLOAT:
                    if space.log:
                        config[space.parameter_name] = np.exp(np.random.uniform(
                            np.log(space.low), np.log(space.high)
                        ))
                    else:
                        config[space.parameter_name] = np.random.uniform(space.low, space.high)
            configs.append(config)
        
        # Hyperband successive halving
        for iteration in range(int(np.log(max_iter) / np.log(eta)) + 1):
            n_configs = len(configs)
            n_iterations = max_iter // (eta ** iteration)
            
            self.logger.info(
                f"Hyperband iteration {iteration + 1}: "
                f"{n_configs} configs, {n_iterations} iterations each"
            )
            
            # Evaluate configurations
            config_scores = []
            
            for config_idx, config in enumerate(configs):
                try:
                    # Add resource budget to config (e.g., training epochs)
                    config_with_budget = config.copy()
                    config_with_budget['_resource_budget'] = n_iterations
                    
                    score = objective_function(config_with_budget)
                    config_scores.append((score, config))
                    
                    # Record trial
                    trial_info = {
                        "trial_number": len(result.trial_history),
                        "params": config.copy(),
                        "score": score,
                        "status": "complete",
                        "resource_budget": n_iterations
                    }
                    result.trial_history.append(trial_info)
                    result.score_history.append(score)
                    result.successful_trials += 1
                    
                    # Update best
                    if self._is_better_score(score, result.best_score):
                        result.best_score = score
                        result.best_params = config.copy()
                        result.best_trial_number = len(result.trial_history) - 1
                    
                except Exception as e:
                    result.failed_trials += 1
                    config_scores.append((float('-inf') if self._is_maximization_objective() else float('inf'), config))
            
            # Select top configurations for next round
            config_scores.sort(key=lambda x: x[0], reverse=self._is_maximization_objective())
            n_survivors = max(1, n_configs // eta)
            configs = [config for _, config in config_scores[:n_survivors]]
        
        result.total_trials = len(result.trial_history)
        
        return result
    
    async def _multi_fidelity_optimization(self,
                                         optimization_id: str,
                                         objective_function: Callable,
                                         search_space: List[SearchSpace]) -> OptimizationResult:
        """Perform multi-fidelity optimization."""
        
        return await self.multi_fidelity_optimizer.optimize(
            optimization_id, objective_function, search_space
        )
    
    async def _random_optimization(self,
                                 optimization_id: str,
                                 objective_function: Callable,
                                 search_space: List[SearchSpace]) -> OptimizationResult:
        """Perform random search optimization."""
        
        result = self.optimization_results[optimization_id]
        
        for trial_num in range(self.config.max_trials):
            # Generate random parameters
            params = {}
            for space in search_space:
                if space.space_type == SearchSpaceType.CATEGORICAL:
                    params[space.parameter_name] = np.random.choice(space.choices)
                elif space.space_type == SearchSpaceType.INTEGER:
                    params[space.parameter_name] = np.random.randint(space.low, space.high + 1)
                elif space.space_type == SearchSpaceType.FLOAT:
                    if space.log:
                        params[space.parameter_name] = np.exp(np.random.uniform(
                            np.log(space.low), np.log(space.high)
                        ))
                    else:
                        params[space.parameter_name] = np.random.uniform(space.low, space.high)
            
            try:
                score = objective_function(params)
                
                # Record trial
                trial_info = {
                    "trial_number": trial_num,
                    "params": params.copy(),
                    "score": score,
                    "status": "complete"
                }
                result.trial_history.append(trial_info)
                result.score_history.append(score)
                result.successful_trials += 1
                
                # Update best
                if self._is_better_score(score, result.best_score):
                    result.best_score = score
                    result.best_params = params.copy()
                    result.best_trial_number = trial_num
                
            except Exception as e:
                result.failed_trials += 1
                self.logger.warning(f"Trial {trial_num} failed: {str(e)}")
        
        result.total_trials = self.config.max_trials
        
        return result
    
    def _is_maximization_objective(self) -> bool:
        """Check if objective should be maximized."""
        
        maximize_objectives = {
            OptimizationObjective.ACCURACY,
            OptimizationObjective.PRECISION,
            OptimizationObjective.RECALL,
            OptimizationObjective.F1_SCORE,
            OptimizationObjective.ROC_AUC,
            OptimizationObjective.R2_SCORE
        }
        
        return self.config.optimization_objective in maximize_objectives
    
    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        """Check if new score is better than current best."""
        
        if self._is_maximization_objective():
            return new_score > current_best
        else:
            return new_score < current_best
    
    def _analyze_convergence(self, score_history: List[float]) -> Optional[int]:
        """Analyze when optimization converged."""
        
        if len(score_history) < 10:
            return None
        
        # Find when improvement stopped
        best_scores = []
        current_best = score_history[0]
        
        for i, score in enumerate(score_history):
            if self._is_better_score(score, current_best):
                current_best = score
            best_scores.append(current_best)
        
        # Look for convergence (no improvement for several trials)
        convergence_window = min(20, len(best_scores) // 4)
        
        for i in range(convergence_window, len(best_scores)):
            window_start = i - convergence_window
            
            # Check if best score hasn't improved in window
            if all(best_scores[j] == best_scores[window_start] for j in range(window_start, i)):
                return window_start
        
        return None
    
    def _calculate_improvement_rate(self, score_history: List[float]) -> float:
        """Calculate rate of improvement over trials."""
        
        if len(score_history) < 2:
            return 0.0
        
        improvements = []
        best_so_far = score_history[0]
        
        for score in score_history[1:]:
            if self._is_better_score(score, best_so_far):
                if self._is_maximization_objective():
                    improvement = (score - best_so_far) / abs(best_so_far) if best_so_far != 0 else score
                else:
                    improvement = (best_so_far - score) / abs(best_so_far) if best_so_far != 0 else abs(score)
                
                improvements.append(improvement)
                best_so_far = score
        
        return np.mean(improvements) if improvements else 0.0
    
    async def _analyze_parameter_importance(self,
                                          trial_history: List[Dict[str, Any]],
                                          search_space: List[SearchSpace]) -> Dict[str, float]:
        """Analyze parameter importance using functional ANOVA."""
        
        try:
            # Extract parameters and scores
            params_data = []
            scores = []
            
            for trial in trial_history:
                if trial["status"] == "complete":
                    params_data.append(trial["params"])
                    scores.append(trial["score"])
            
            if len(params_data) < 10:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(params_data)
            
            # Calculate parameter importance using correlation
            importance = {}
            
            for param_name in df.columns:
                if df[param_name].dtype in ['int64', 'float64']:
                    # Numerical parameter
                    correlation = abs(df[param_name].corr(pd.Series(scores)))
                    importance[param_name] = correlation if not np.isnan(correlation) else 0.0
                else:
                    # Categorical parameter
                    # Use mutual information or ANOVA F-statistic
                    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
                    
                    # Encode categorical values
                    encoded_values = pd.Categorical(df[param_name]).codes
                    
                    if self.config.task_type == AutoMLTask.REGRESSION:
                        mi = mutual_info_regression(encoded_values.reshape(-1, 1), scores)
                    else:
                        mi = mutual_info_classif(encoded_values.reshape(-1, 1), scores)
                    
                    importance[param_name] = mi[0] if len(mi) > 0 else 0.0
            
            # Normalize importance scores
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze parameter importance: {str(e)}")
            return {}
    
    async def get_optimization_status(self, optimization_id: str) -> Dict[str, Any]:
        """Get status of ongoing optimization."""
        
        if optimization_id not in self.optimization_results:
            return {"error": f"Optimization {optimization_id} not found"}
        
        result = self.optimization_results[optimization_id]
        
        # Check if optimization is still running
        is_running = optimization_id in self.active_studies
        
        status = {
            "optimization_id": optimization_id,
            "status": "running" if is_running else "completed",
            "method": result.method.value,
            "total_trials": result.total_trials,
            "successful_trials": result.successful_trials,
            "failed_trials": result.failed_trials,
            "best_score": result.best_score,
            "best_params": result.best_params,
            "optimization_time_seconds": result.optimization_time_seconds,
            "convergence_trial": result.convergence_trial
        }
        
        return status
    
    async def stop_optimization(self, optimization_id: str) -> bool:
        """Stop ongoing optimization."""
        
        if optimization_id in self.active_studies:
            study = self.active_studies[optimization_id]
            study.stop()
            del self.active_studies[optimization_id]
            
            self.logger.info(f"Stopped optimization {optimization_id}")
            return True
        
        return False
    
    def create_search_space_from_model(self, model_class: Any) -> List[SearchSpace]:
        """Create search space automatically from model class."""
        
        search_space = []
        
        # Common hyperparameters for different model types
        model_name = model_class.__name__.lower()
        
        if 'randomforest' in model_name:
            search_space.extend([
                SearchSpace("n_estimators", SearchSpaceType.INTEGER, 10, 200),
                SearchSpace("max_depth", SearchSpaceType.INTEGER, 3, 20),
                SearchSpace("min_samples_split", SearchSpaceType.INTEGER, 2, 20),
                SearchSpace("min_samples_leaf", SearchSpaceType.INTEGER, 1, 10),
                SearchSpace("max_features", SearchSpaceType.CATEGORICAL, choices=["sqrt", "log2", None])
            ])
        
        elif 'svm' in model_name or 'svc' in model_name:
            search_space.extend([
                SearchSpace("C", SearchSpaceType.LOG_UNIFORM, 0.01, 100.0),
                SearchSpace("gamma", SearchSpaceType.CATEGORICAL, choices=["scale", "auto"]),
                SearchSpace("kernel", SearchSpaceType.CATEGORICAL, choices=["rbf", "poly", "sigmoid"])
            ])
        
        elif 'logistic' in model_name:
            search_space.extend([
                SearchSpace("C", SearchSpaceType.LOG_UNIFORM, 0.01, 100.0),
                SearchSpace("penalty", SearchSpaceType.CATEGORICAL, choices=["l1", "l2", "elasticnet"]),
                SearchSpace("solver", SearchSpaceType.CATEGORICAL, choices=["liblinear", "saga"])
            ])
        
        elif 'gradient' in model_name or 'gbm' in model_name or 'xgb' in model_name:
            search_space.extend([
                SearchSpace("n_estimators", SearchSpaceType.INTEGER, 50, 500),
                SearchSpace("learning_rate", SearchSpaceType.LOG_UNIFORM, 0.01, 0.3),
                SearchSpace("max_depth", SearchSpaceType.INTEGER, 3, 10),
                SearchSpace("subsample", SearchSpaceType.UNIFORM, 0.6, 1.0),
                SearchSpace("colsample_bytree", SearchSpaceType.UNIFORM, 0.6, 1.0)
            ])
        
        elif 'mlp' in model_name or 'neural' in model_name:
            search_space.extend([
                SearchSpace("hidden_layer_sizes", SearchSpaceType.CATEGORICAL, 
                           choices=[(50,), (100,), (50, 50), (100, 50), (100, 100)]),
                SearchSpace("alpha", SearchSpaceType.LOG_UNIFORM, 0.0001, 0.1),
                SearchSpace("learning_rate_init", SearchSpaceType.LOG_UNIFORM, 0.001, 0.1),
                SearchSpace("activation", SearchSpaceType.CATEGORICAL, choices=["relu", "tanh", "logistic"])
            ])
        
        return search_space


class BayesianOptimizer:
    """Bayesian optimization implementation."""
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    async def optimize(self,
                      objective_function: Callable,
                      search_space: List[SearchSpace],
                      n_calls: int = 100) -> Dict[str, Any]:
        """Perform Bayesian optimization."""
        
        # Implementation would use advanced Bayesian optimization
        # with Gaussian processes, acquisition functions, etc.
        
        return {
            "method": "bayesian_optimization",
            "best_params": {},
            "best_score": 0.0,
            "n_calls": n_calls
        }


class PopulationBasedTrainer:
    """Population-based training implementation."""
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    async def optimize(self,
                      optimization_id: str,
                      objective_function: Callable,
                      search_space: List[SearchSpace]) -> OptimizationResult:
        """Perform population-based training."""
        
        result = OptimizationResult(
            optimization_id=optimization_id,
            method=OptimizationMethod.POPULATION_BASED
        )
        
        # Initialize population
        population = []
        for _ in range(self.config.population_size):
            individual = self._generate_random_individual(search_space)
            population.append(individual)
        
        # Evaluate initial population
        for i, individual in enumerate(population):
            try:
                score = objective_function(individual['params'])
                individual['score'] = score
                individual['age'] = 0
                
                # Record trial
                trial_info = {
                    "trial_number": len(result.trial_history),
                    "params": individual['params'].copy(),
                    "score": score,
                    "status": "complete"
                }
                result.trial_history.append(trial_info)
                result.score_history.append(score)
                result.successful_trials += 1
                
                # Update best
                if self._is_better_score(score, result.best_score):
                    result.best_score = score
                    result.best_params = individual['params'].copy()
                    result.best_trial_number = len(result.trial_history) - 1
                
            except Exception as e:
                individual['score'] = float('-inf') if self._is_maximization_objective() else float('inf')
                result.failed_trials += 1
        
        # Evolution loop
        n_generations = self.config.max_trials // self.config.population_size
        
        for generation in range(n_generations):
            # Sort population by performance
            population.sort(key=lambda x: x['score'], reverse=self._is_maximization_objective())
            
            # Select elite individuals
            elite_size = int(self.config.elite_fraction * self.config.population_size)
            elite = population[:elite_size]
            
            # Exploit: copy elite individuals
            new_population = []
            for individual in elite:
                new_individual = {
                    'params': individual['params'].copy(),
                    'score': individual['score'],
                    'age': individual['age'] + 1
                }
                new_population.append(new_individual)
            
            # Explore: mutate and replace worst individuals
            worst_individuals = population[elite_size:]
            
            for i, individual in enumerate(worst_individuals):
                if random.random() < self.config.mutation_rate:
                    # Mutate parameters
                    new_params = self._mutate_individual(individual['params'], search_space)
                    
                    try:
                        score = objective_function(new_params)
                        
                        mutated_individual = {
                            'params': new_params,
                            'score': score,
                            'age': 0
                        }
                        new_population.append(mutated_individual)
                        
                        # Record trial
                        trial_info = {
                            "trial_number": len(result.trial_history),
                            "params": new_params.copy(),
                            "score": score,
                            "status": "complete"
                        }
                        result.trial_history.append(trial_info)
                        result.score_history.append(score)
                        result.successful_trials += 1
                        
                        # Update best
                        if self._is_better_score(score, result.best_score):
                            result.best_score = score
                            result.best_params = new_params.copy()
                            result.best_trial_number = len(result.trial_history) - 1
                        
                    except Exception as e:
                        # Keep original individual if mutation fails
                        new_population.append(individual)
                        result.failed_trials += 1
                else:
                    new_population.append(individual)
            
            population = new_population[:self.config.population_size]
        
        result.total_trials = len(result.trial_history)
        
        return result
    
    def _generate_random_individual(self, search_space: List[SearchSpace]) -> Dict[str, Any]:
        """Generate random individual in population."""
        
        params = {}
        
        for space in search_space:
            if space.space_type == SearchSpaceType.CATEGORICAL:
                params[space.parameter_name] = np.random.choice(space.choices)
            elif space.space_type == SearchSpaceType.INTEGER:
                params[space.parameter_name] = np.random.randint(space.low, space.high + 1)
            elif space.space_type == SearchSpaceType.FLOAT:
                if space.log:
                    params[space.parameter_name] = np.exp(np.random.uniform(
                        np.log(space.low), np.log(space.high)
                    ))
                else:
                    params[space.parameter_name] = np.random.uniform(space.low, space.high)
        
        return {'params': params, 'score': 0.0, 'age': 0}
    
    def _mutate_individual(self, params: Dict[str, Any], search_space: List[SearchSpace]) -> Dict[str, Any]:
        """Mutate individual parameters."""
        
        new_params = params.copy()
        
        # Mutate each parameter with some probability
        for space in search_space:
            if np.random.random() < 0.3:  # Mutation probability per parameter
                
                if space.space_type == SearchSpaceType.CATEGORICAL:
                    new_params[space.parameter_name] = np.random.choice(space.choices)
                
                elif space.space_type == SearchSpaceType.INTEGER:
                    current_value = params[space.parameter_name]
                    # Add noise
                    noise = np.random.randint(-2, 3)
                    new_value = max(space.low, min(space.high, current_value + noise))
                    new_params[space.parameter_name] = new_value
                
                elif space.space_type == SearchSpaceType.FLOAT:
                    current_value = params[space.parameter_name]
                    # Add Gaussian noise
                    if space.log:
                        log_value = np.log(current_value)
                        noise = np.random.normal(0, 0.1)
                        new_log_value = log_value + noise
                        new_value = np.exp(new_log_value)
                        new_value = max(space.low, min(space.high, new_value))
                    else:
                        noise = np.random.normal(0, (space.high - space.low) * 0.1)
                        new_value = current_value + noise
                        new_value = max(space.low, min(space.high, new_value))
                    
                    new_params[space.parameter_name] = new_value
        
        return new_params
    
    def _is_maximization_objective(self) -> bool:
        """Check if objective should be maximized."""
        
        maximize_objectives = {
            OptimizationObjective.ACCURACY,
            OptimizationObjective.PRECISION,
            OptimizationObjective.RECALL,
            OptimizationObjective.F1_SCORE,
            OptimizationObjective.ROC_AUC,
            OptimizationObjective.R2_SCORE
        }
        
        return self.config.optimization_objective in maximize_objectives
    
    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        """Check if new score is better than current best."""
        
        if self._is_maximization_objective():
            return new_score > current_best
        else:
            return new_score < current_best


class MultiFidelityOptimizer:
    """Multi-fidelity optimization implementation."""
    
    def __init__(self, config: HPOConfig):
        self.config = config
        self.logger = structlog.get_logger(__name__)
    
    async def optimize(self,
                      optimization_id: str,
                      objective_function: Callable,
                      search_space: List[SearchSpace]) -> OptimizationResult:
        """Perform multi-fidelity optimization."""
        
        result = OptimizationResult(
            optimization_id=optimization_id,
            method=OptimizationMethod.MULTI_FIDELITY
        )
        
        # Multi-fidelity optimization with successive halving
        # Start with many configurations at low fidelity
        # Progressively eliminate poor performers and increase fidelity
        
        fidelity_levels = [0.1, 0.3, 0.5, 0.7, 1.0]
        n_configs_per_level = [64, 32, 16, 8, 4]
        
        # Generate initial configurations
        configs = []
        for _ in range(n_configs_per_level[0]):
            config = {}
            for space in search_space:
                if space.space_type == SearchSpaceType.CATEGORICAL:
                    config[space.parameter_name] = np.random.choice(space.choices)
                elif space.space_type == SearchSpaceType.INTEGER:
                    config[space.parameter_name] = np.random.randint(space.low, space.high + 1)
                elif space.space_type == SearchSpaceType.FLOAT:
                    if space.log:
                        config[space.parameter_name] = np.exp(np.random.uniform(
                            np.log(space.low), np.log(space.high)
                        ))
                    else:
                        config[space.parameter_name] = np.random.uniform(space.low, space.high)
            configs.append(config)
        
        # Successive halving across fidelity levels
        for level_idx, (fidelity, n_configs) in enumerate(zip(fidelity_levels, n_configs_per_level)):
            if len(configs) == 0:
                break
            
            self.logger.info(
                f"Multi-fidelity level {level_idx + 1}/{len(fidelity_levels)}: "
                f"fidelity={fidelity}, configs={len(configs)}"
            )
            
            # Evaluate configurations at current fidelity
            config_scores = []
            
            for config in configs:
                try:
                    # Add fidelity information to config
                    config_with_fidelity = config.copy()
                    config_with_fidelity['_fidelity'] = fidelity
                    
                    score = objective_function(config_with_fidelity)
                    config_scores.append((score, config))
                    
                    # Record trial
                    trial_info = {
                        "trial_number": len(result.trial_history),
                        "params": config.copy(),
                        "score": score,
                        "status": "complete",
                        "fidelity": fidelity
                    }
                    result.trial_history.append(trial_info)
                    result.score_history.append(score)
                    result.successful_trials += 1
                    
                    # Update best (only for full fidelity)
                    if fidelity == 1.0 and self._is_better_score(score, result.best_score):
                        result.best_score = score
                        result.best_params = config.copy()
                        result.best_trial_number = len(result.trial_history) - 1
                    
                except Exception as e:
                    config_scores.append((
                        float('-inf') if self._is_maximization_objective() else float('inf'), 
                        config
                    ))
                    result.failed_trials += 1
            
            # Select top configurations for next fidelity level
            if level_idx < len(fidelity_levels) - 1:
                config_scores.sort(key=lambda x: x[0], reverse=self._is_maximization_objective())
                configs = [config for _, config in config_scores[:n_configs_per_level[level_idx + 1]]]
            else:
                configs = []
        
        result.total_trials = len(result.trial_history)
        
        return result
    
    def _is_maximization_objective(self) -> bool:
        """Check if objective should be maximized."""
        
        maximize_objectives = {
            OptimizationObjective.ACCURACY,
            OptimizationObjective.PRECISION,
            OptimizationObjective.RECALL,
            OptimizationObjective.F1_SCORE,
            OptimizationObjective.ROC_AUC,
            OptimizationObjective.R2_SCORE
        }
        
        return self.config.optimization_objective in maximize_objectives
    
    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        """Check if new score is better than current best."""
        
        if self._is_maximization_objective():
            return new_score > current_best
        else:
            return new_score < current_best


class OptimizationPerformanceTracker:
    """Tracks and analyzes optimization performance."""
    
    def __init__(self):
        self.performance_data: Dict[str, List[Dict[str, Any]]] = {}
    
    def record_performance(self, optimization_id: str, performance_data: Dict[str, Any]) -> None:
        """Record performance data for analysis."""
        
        if optimization_id not in self.performance_data:
            self.performance_data[optimization_id] = []
        
        performance_data['timestamp'] = datetime.utcnow()
        self.performance_data[optimization_id].append(performance_data)
    
    def analyze_convergence_patterns(self, optimization_results: List[OptimizationResult]) -> Dict[str, Any]:
        """Analyze convergence patterns across optimizations."""
        
        convergence_analysis = {
            "average_convergence_trial": 0.0,
            "convergence_rate_by_method": {},
            "improvement_patterns": {}
        }
        
        # Analyze convergence by method
        method_convergence = {}
        
        for result in optimization_results:
            method = result.method.value
            
            if method not in method_convergence:
                method_convergence[method] = []
            
            if result.convergence_trial is not None:
                method_convergence[method].append(result.convergence_trial)
        
        # Calculate statistics
        for method, convergence_trials in method_convergence.items():
            if convergence_trials:
                convergence_analysis["convergence_rate_by_method"][method] = {
                    "average_convergence_trial": np.mean(convergence_trials),
                    "median_convergence_trial": np.median(convergence_trials),
                    "convergence_rate": len(convergence_trials) / len([r for r in optimization_results if r.method.value == method])
                }
        
        return convergence_analysis
    
    def generate_performance_report(self, optimization_id: str) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        if optimization_id not in self.performance_data:
            return {"error": f"No performance data for optimization {optimization_id}"}
        
        data = self.performance_data[optimization_id]
        
        report = {
            "optimization_id": optimization_id,
            "total_data_points": len(data),
            "time_span": {
                "start": data[0]['timestamp'].isoformat() if data else None,
                "end": data[-1]['timestamp'].isoformat() if data else None
            },
            "performance_summary": self._summarize_performance_data(data)
        }
        
        return report
    
    def _summarize_performance_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize performance data."""
        
        if not data:
            return {}
        
        # Extract metrics
        scores = [d.get('score', 0) for d in data if 'score' in d]
        times = [d.get('time', 0) for d in data if 'time' in d]
        
        summary = {
            "score_statistics": {
                "mean": np.mean(scores) if scores else 0,
                "std": np.std(scores) if scores else 0,
                "min": np.min(scores) if scores else 0,
                "max": np.max(scores) if scores else 0
            },
            "time_statistics": {
                "mean": np.mean(times) if times else 0,
                "std": np.std(times) if times else 0,
                "total": np.sum(times) if times else 0
            }
        }
        
        return summary
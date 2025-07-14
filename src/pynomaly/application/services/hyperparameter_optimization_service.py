"""
Hyperparameter Optimization Service

Advanced hyperparameter optimization service using Optuna for Bayesian optimization
of anomaly detection models. Integrates with ML Pipeline Framework and Enhanced
Model Training Framework.

This addresses Issue #143: Phase 2.2: Data Science Package - Machine Learning Pipeline Framework
Component 3: Hyperparameter Optimization with Optuna
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import make_scorer

# Optional dependencies with graceful fallback
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner, PercentilePruner
    from optuna.study import StudyDirection
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from pynomaly.infrastructure.logging.structured_logger import StructuredLogger


class OptimizationObjective(Enum):
    """Optimization objectives for hyperparameter tuning."""
    
    MAXIMIZE_ROC_AUC = "maximize_roc_auc"
    MAXIMIZE_PRECISION = "maximize_precision"
    MAXIMIZE_RECALL = "maximize_recall"
    MAXIMIZE_F1_SCORE = "maximize_f1_score"
    MINIMIZE_FALSE_POSITIVES = "minimize_false_positives"
    MAXIMIZE_BALANCED_ACCURACY = "maximize_balanced_accuracy"
    CUSTOM = "custom"


class SamplerType(Enum):
    """Types of optimization samplers."""
    
    TPE = "tpe"  # Tree-structured Parzen Estimator
    CMAES = "cmaes"  # Covariance Matrix Adaptation Evolution Strategy
    RANDOM = "random"
    GRID = "grid"


class PrunerType(Enum):
    """Types of optimization pruners."""
    
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    PERCENTILE = "percentile"
    NONE = "none"


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_ROC_AUC
    sampler_type: SamplerType = SamplerType.TPE
    pruner_type: PrunerType = PrunerType.MEDIAN
    
    # Optimization parameters
    n_trials: int = 100
    timeout: Optional[int] = 3600  # seconds
    n_jobs: int = 1
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_scoring: Optional[str] = None
    cv_random_state: int = 42
    
    # Pruning parameters
    pruning_patience: int = 5
    pruning_percentile: float = 25.0
    
    # Early stopping
    early_stopping_rounds: Optional[int] = 20
    early_stopping_threshold: float = 1e-4
    
    # Study configuration
    study_name: Optional[str] = None
    storage: Optional[str] = None  # Database URL for distributed optimization
    load_if_exists: bool = True
    
    # Advanced options
    enable_pruning: bool = True
    enable_caching: bool = True
    max_memory_usage: Optional[int] = None  # MB
    
    # Custom objective function
    custom_objective_func: Optional[Callable] = None


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    
    study: Any  # optuna.Study object
    best_params: Dict[str, Any]
    best_value: float
    best_trial: Any
    
    # Optimization metadata
    n_trials: int
    optimization_time: float
    convergence_history: List[float]
    
    # Model performance
    cv_scores: Dict[str, List[float]]
    validation_scores: Dict[str, float]
    
    # Study analysis
    parameter_importance: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    
    # Additional metrics
    study_statistics: Dict[str, Any]
    pruned_trials: int
    failed_trials: int


@dataclass
class ParameterSpace:
    """Definition of hyperparameter search space."""
    
    # Continuous parameters
    float_params: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # name: (low, high)
    log_float_params: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # name: (low, high)
    
    # Discrete parameters
    int_params: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # name: (low, high)
    categorical_params: Dict[str, List[Any]] = field(default_factory=dict)  # name: choices
    
    # Conditional parameters
    conditional_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Custom parameter generators
    custom_params: Dict[str, Callable] = field(default_factory=dict)


class HyperparameterOptimizationService:
    """Advanced Hyperparameter Optimization Service using Optuna.
    
    Provides comprehensive hyperparameter optimization capabilities including:
    - Bayesian optimization with multiple sampling strategies
    - Advanced pruning for efficient search
    - Multi-objective optimization support
    - Distributed optimization capabilities
    - Visualization and analysis tools
    - Integration with anomaly detection models
    """
    
    def __init__(
        self,
        storage_dir: str = "optimization_results",
        enable_distributed: bool = False,
        storage_url: Optional[str] = None
    ):
        self.logger = StructuredLogger("hyperparameter_optimization")
        
        # Storage configuration
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.storage_url = storage_url
        self.enable_distributed = enable_distributed
        
        # Optimization tracking
        self.active_studies: Dict[str, Any] = {}
        self.optimization_history: Dict[str, OptimizationResult] = {}
        
        # Performance metrics
        self.optimization_metrics: Dict[str, Any] = {}
        
        # Validation
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available. Limited functionality.")
    
    async def optimize_hyperparameters(
        self,
        estimator_class: type,
        parameter_space: Union[ParameterSpace, Dict[str, Any]],
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        config: OptimizationConfig = None,
        study_name: Optional[str] = None
    ) -> OptimizationResult:
        """Optimize hyperparameters for a given estimator.
        
        Args:
            estimator_class: The estimator class to optimize
            parameter_space: Parameter search space definition
            X: Training features
            y: Training targets (optional for unsupervised)
            config: Optimization configuration
            study_name: Name for the optimization study
            
        Returns:
            Optimization results with best parameters and performance metrics
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization")
        
        if config is None:
            config = OptimizationConfig()
        
        if study_name is None:
            study_name = f"opt_{estimator_class.__name__}_{int(time.time())}"
        
        self.logger.info(f"Starting hyperparameter optimization: {study_name}")
        start_time = time.time()
        
        try:
            # Convert parameter space if needed
            if isinstance(parameter_space, dict):
                parameter_space = self._dict_to_parameter_space(parameter_space)
            
            # Create study
            study = await self._create_study(config, study_name)
            
            # Create objective function
            objective_func = self._create_objective_function(
                estimator_class, parameter_space, X, y, config
            )
            
            # Store active study
            self.active_studies[study_name] = study
            
            # Run optimization
            await self._run_optimization(study, objective_func, config)
            
            # Extract results
            result = await self._extract_optimization_results(
                study, config, time.time() - start_time
            )
            
            # Store results
            self.optimization_history[study_name] = result
            
            # Save results
            await self._save_optimization_results(study_name, result)
            
            self.logger.info(
                f"Optimization completed: {study_name} "
                f"(Best score: {result.best_value:.4f}, Time: {result.optimization_time:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization failed for {study_name}: {e}")
            raise
        finally:
            # Clean up
            if study_name in self.active_studies:
                del self.active_studies[study_name]
    
    async def multi_objective_optimization(
        self,
        estimator_class: type,
        parameter_space: Union[ParameterSpace, Dict[str, Any]],
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        objectives: List[OptimizationObjective] = None,
        config: OptimizationConfig = None,
        study_name: Optional[str] = None
    ) -> OptimizationResult:
        """Perform multi-objective hyperparameter optimization.
        
        Args:
            estimator_class: The estimator class to optimize
            parameter_space: Parameter search space definition
            X: Training features
            y: Training targets (optional for unsupervised)
            objectives: List of optimization objectives
            config: Optimization configuration
            study_name: Name for the optimization study
            
        Returns:
            Multi-objective optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for multi-objective optimization")
        
        if objectives is None:
            objectives = [
                OptimizationObjective.MAXIMIZE_ROC_AUC,
                OptimizationObjective.MAXIMIZE_F1_SCORE
            ]
        
        if config is None:
            config = OptimizationConfig()
        
        if study_name is None:
            study_name = f"multi_opt_{estimator_class.__name__}_{int(time.time())}"
        
        self.logger.info(f"Starting multi-objective optimization: {study_name}")
        
        try:
            # Create multi-objective study
            directions = [StudyDirection.MAXIMIZE] * len(objectives)
            study = optuna.create_study(
                directions=directions,
                study_name=study_name,
                storage=self.storage_url if self.enable_distributed else None,
                load_if_exists=config.load_if_exists,
                sampler=self._create_sampler(config.sampler_type),
                pruner=self._create_pruner(config.pruner_type) if config.enable_pruning else None
            )
            
            # Convert parameter space
            if isinstance(parameter_space, dict):
                parameter_space = self._dict_to_parameter_space(parameter_space)
            
            # Create multi-objective function
            objective_func = self._create_multi_objective_function(
                estimator_class, parameter_space, X, y, objectives, config
            )
            
            # Run optimization
            start_time = time.time()
            study.optimize(
                objective_func,
                n_trials=config.n_trials,
                timeout=config.timeout,
                n_jobs=config.n_jobs
            )
            
            # Extract Pareto optimal solutions
            pareto_trials = study.best_trials
            
            # Create result for best trial (using first objective)
            best_trial = pareto_trials[0] if pareto_trials else study.trials[0]
            
            result = OptimizationResult(
                study=study,
                best_params=best_trial.params,
                best_value=best_trial.values[0] if best_trial.values else 0.0,
                best_trial=best_trial,
                n_trials=len(study.trials),
                optimization_time=time.time() - start_time,
                convergence_history=[],
                cv_scores={},
                validation_scores={},
                parameter_importance={},
                optimization_history=[],
                study_statistics={
                    "pareto_solutions": len(pareto_trials),
                    "total_trials": len(study.trials),
                    "objectives": [obj.value for obj in objectives]
                },
                pruned_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                failed_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            )
            
            await self._save_optimization_results(study_name, result)
            
            self.logger.info(f"Multi-objective optimization completed: {study_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-objective optimization failed: {e}")
            raise
    
    async def distributed_optimization(
        self,
        estimator_class: type,
        parameter_space: Union[ParameterSpace, Dict[str, Any]],
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        config: OptimizationConfig = None,
        n_workers: int = 4,
        study_name: Optional[str] = None
    ) -> OptimizationResult:
        """Run distributed hyperparameter optimization.
        
        Args:
            estimator_class: The estimator class to optimize
            parameter_space: Parameter search space definition
            X: Training features
            y: Training targets (optional for unsupervised)
            config: Optimization configuration
            n_workers: Number of parallel workers
            study_name: Name for the optimization study
            
        Returns:
            Distributed optimization results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for distributed optimization")
        
        if not self.storage_url:
            raise ValueError("Storage URL required for distributed optimization")
        
        if config is None:
            config = OptimizationConfig()
        config.n_jobs = 1  # Each worker uses single process
        
        if study_name is None:
            study_name = f"dist_opt_{estimator_class.__name__}_{int(time.time())}"
        
        self.logger.info(f"Starting distributed optimization with {n_workers} workers: {study_name}")
        
        # Create shared study
        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage_url,
            load_if_exists=config.load_if_exists,
            sampler=self._create_sampler(config.sampler_type),
            pruner=self._create_pruner(config.pruner_type) if config.enable_pruning else None
        )
        
        # Convert parameter space
        if isinstance(parameter_space, dict):
            parameter_space = self._dict_to_parameter_space(parameter_space)
        
        # Create objective function
        objective_func = self._create_objective_function(
            estimator_class, parameter_space, X, y, config
        )
        
        # Run distributed optimization
        start_time = time.time()
        
        # Create worker tasks
        trials_per_worker = config.n_trials // n_workers
        tasks = []
        
        for worker_id in range(n_workers):
            worker_trials = trials_per_worker
            if worker_id == n_workers - 1:  # Last worker gets remaining trials
                worker_trials += config.n_trials % n_workers
            
            task = asyncio.create_task(
                self._run_worker_optimization(
                    study_name, objective_func, worker_trials, config, worker_id
                )
            )
            tasks.append(task)
        
        # Wait for all workers to complete
        await asyncio.gather(*tasks)
        
        # Reload study to get all results
        study = optuna.load_study(study_name=study_name, storage=self.storage_url)
        
        # Extract results
        result = await self._extract_optimization_results(
            study, config, time.time() - start_time
        )
        
        await self._save_optimization_results(study_name, result)
        
        self.logger.info(f"Distributed optimization completed: {study_name}")
        return result
    
    async def get_optimization_insights(self, study_name: str) -> Dict[str, Any]:
        """Get insights about an optimization study.
        
        Args:
            study_name: Name of the study
            
        Returns:
            Study insights and analysis
        """
        if study_name not in self.optimization_history:
            raise ValueError(f"Study {study_name} not found")
        
        result = self.optimization_history[study_name]
        study = result.study
        
        insights = {
            "study_name": study_name,
            "best_value": result.best_value,
            "best_params": result.best_params,
            "optimization_time": result.optimization_time,
            "n_trials": result.n_trials,
            "parameter_importance": result.parameter_importance,
            "convergence_analysis": self._analyze_convergence(study),
            "parameter_correlations": self._analyze_parameter_correlations(study),
            "trial_distribution": self._analyze_trial_distribution(study),
            "pruning_effectiveness": self._analyze_pruning_effectiveness(study),
            "optimization_efficiency": self._calculate_optimization_efficiency(study)
        }
        
        return insights
    
    async def generate_optimization_report(self, study_name: str, output_dir: str = None) -> str:
        """Generate comprehensive optimization report.
        
        Args:
            study_name: Name of the study
            output_dir: Output directory for report
            
        Returns:
            Path to generated report
        """
        if output_dir is None:
            output_dir = self.storage_dir
        
        insights = await self.get_optimization_insights(study_name)
        
        # Generate report
        report_path = Path(output_dir) / f"{study_name}_optimization_report.html"
        
        # Create HTML report
        html_content = self._create_html_report(insights)
        
        with open(report_path, "w") as f:
            f.write(html_content)
        
        # Generate visualizations if Plotly is available
        if PLOTLY_AVAILABLE:
            await self._generate_optimization_plots(study_name, output_dir)
        
        self.logger.info(f"Optimization report generated: {report_path}")
        return str(report_path)
    
    def _dict_to_parameter_space(self, param_dict: Dict[str, Any]) -> ParameterSpace:
        """Convert dictionary to ParameterSpace."""
        
        space = ParameterSpace()
        
        for param_name, param_config in param_dict.items():
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "float")
                
                if param_type == "float":
                    space.float_params[param_name] = (
                        param_config["low"], param_config["high"]
                    )
                elif param_type == "log_float":
                    space.log_float_params[param_name] = (
                        param_config["low"], param_config["high"]
                    )
                elif param_type == "int":
                    space.int_params[param_name] = (
                        param_config["low"], param_config["high"]
                    )
                elif param_type == "categorical":
                    space.categorical_params[param_name] = param_config["choices"]
            
            elif isinstance(param_config, (list, tuple)):
                if len(param_config) == 2 and all(isinstance(x, (int, float)) for x in param_config):
                    # Assume float range
                    space.float_params[param_name] = param_config
                else:
                    # Assume categorical
                    space.categorical_params[param_name] = param_config
        
        return space
    
    async def _create_study(self, config: OptimizationConfig, study_name: str):
        """Create Optuna study."""
        
        return optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=self.storage_url if self.enable_distributed else None,
            load_if_exists=config.load_if_exists,
            sampler=self._create_sampler(config.sampler_type),
            pruner=self._create_pruner(config.pruner_type) if config.enable_pruning else None
        )
    
    def _create_sampler(self, sampler_type: SamplerType):
        """Create Optuna sampler."""
        
        samplers = {
            SamplerType.TPE: TPESampler,
            SamplerType.CMAES: CmaEsSampler,
            SamplerType.RANDOM: RandomSampler
        }
        
        sampler_class = samplers.get(sampler_type, TPESampler)
        return sampler_class()
    
    def _create_pruner(self, pruner_type: PrunerType):
        """Create Optuna pruner."""
        
        pruners = {
            PrunerType.MEDIAN: MedianPruner,
            PrunerType.HYPERBAND: HyperbandPruner,
            PrunerType.PERCENTILE: lambda: PercentilePruner(25.0)
        }
        
        if pruner_type == PrunerType.NONE:
            return None
        
        pruner_factory = pruners.get(pruner_type, MedianPruner)
        return pruner_factory()
    
    def _create_objective_function(
        self,
        estimator_class: type,
        parameter_space: ParameterSpace,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]],
        config: OptimizationConfig
    ) -> Callable:
        """Create objective function for optimization."""
        
        def objective(trial):
            try:
                # Suggest hyperparameters
                params = self._suggest_parameters(trial, parameter_space)
                
                # Create estimator
                estimator = estimator_class(**params)
                
                # Evaluate with cross-validation
                if config.custom_objective_func:
                    score = config.custom_objective_func(estimator, X, y, trial)
                else:
                    score = self._evaluate_model(estimator, X, y, config)
                
                return score
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                # Return poor score for failed trials
                return 0.0
        
        return objective
    
    def _create_multi_objective_function(
        self,
        estimator_class: type,
        parameter_space: ParameterSpace,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]],
        objectives: List[OptimizationObjective],
        config: OptimizationConfig
    ) -> Callable:
        """Create multi-objective function."""
        
        def objective(trial):
            try:
                # Suggest hyperparameters
                params = self._suggest_parameters(trial, parameter_space)
                
                # Create estimator
                estimator = estimator_class(**params)
                
                # Evaluate for each objective
                scores = []
                for obj in objectives:
                    obj_config = OptimizationConfig(objective=obj)
                    score = self._evaluate_model(estimator, X, y, obj_config)
                    scores.append(score)
                
                return scores
                
            except Exception as e:
                self.logger.warning(f"Multi-objective trial failed: {e}")
                return [0.0] * len(objectives)
        
        return objective
    
    def _suggest_parameters(self, trial, parameter_space: ParameterSpace) -> Dict[str, Any]:
        """Suggest parameters for a trial."""
        
        params = {}
        
        # Float parameters
        for name, (low, high) in parameter_space.float_params.items():
            params[name] = trial.suggest_float(name, low, high)
        
        # Log float parameters
        for name, (low, high) in parameter_space.log_float_params.items():
            params[name] = trial.suggest_float(name, low, high, log=True)
        
        # Integer parameters
        for name, (low, high) in parameter_space.int_params.items():
            params[name] = trial.suggest_int(name, low, high)
        
        # Categorical parameters
        for name, choices in parameter_space.categorical_params.items():
            params[name] = trial.suggest_categorical(name, choices)
        
        # Custom parameters
        for name, generator in parameter_space.custom_params.items():
            params[name] = generator(trial)
        
        return params
    
    def _evaluate_model(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]],
        config: OptimizationConfig
    ) -> float:
        """Evaluate model performance."""
        
        try:
            # Set up cross-validation
            if y is not None:
                cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.cv_random_state)
            else:
                cv = config.cv_folds
            
            # Determine scoring
            scoring = self._get_scoring_function(config.objective, config.cv_scoring)
            
            # Perform cross-validation
            if scoring:
                scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=config.n_jobs)
            else:
                # For unsupervised models or custom scoring
                scores = self._evaluate_unsupervised(estimator, X, cv)
            
            return float(np.mean(scores))
            
        except Exception as e:
            self.logger.warning(f"Model evaluation failed: {e}")
            return 0.0
    
    def _get_scoring_function(self, objective: OptimizationObjective, custom_scoring: Optional[str]) -> Optional[str]:
        """Get appropriate scoring function for objective."""
        
        if custom_scoring:
            return custom_scoring
        
        scoring_map = {
            OptimizationObjective.MAXIMIZE_ROC_AUC: "roc_auc",
            OptimizationObjective.MAXIMIZE_PRECISION: "precision",
            OptimizationObjective.MAXIMIZE_RECALL: "recall",
            OptimizationObjective.MAXIMIZE_F1_SCORE: "f1",
            OptimizationObjective.MAXIMIZE_BALANCED_ACCURACY: "balanced_accuracy",
        }
        
        return scoring_map.get(objective)
    
    def _evaluate_unsupervised(self, estimator: BaseEstimator, X: Union[pd.DataFrame, np.ndarray], cv) -> np.ndarray:
        """Evaluate unsupervised model."""
        
        # Simple evaluation for unsupervised models
        scores = []
        
        try:
            from sklearn.model_selection import KFold
            
            if not hasattr(cv, 'split'):
                cv = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            for train_idx, test_idx in cv.split(X):
                X_train = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
                X_test = X[test_idx] if isinstance(X, np.ndarray) else X.iloc[test_idx]
                
                # Fit on train, predict on test
                estimator.fit(X_train)
                predictions = estimator.predict(X_test)
                
                # Simple scoring based on anomaly ratio
                anomaly_ratio = np.sum(predictions == -1) / len(predictions)
                ideal_ratio = 0.1
                score = 1.0 - abs(anomaly_ratio - ideal_ratio) / ideal_ratio
                scores.append(max(0.0, score))
            
            return np.array(scores)
            
        except Exception as e:
            self.logger.warning(f"Unsupervised evaluation failed: {e}")
            return np.array([0.5] * 5)
    
    async def _run_optimization(self, study, objective_func: Callable, config: OptimizationConfig):
        """Run optimization study."""
        
        study.optimize(
            objective_func,
            n_trials=config.n_trials,
            timeout=config.timeout,
            n_jobs=config.n_jobs
        )
    
    async def _run_worker_optimization(
        self,
        study_name: str,
        objective_func: Callable,
        n_trials: int,
        config: OptimizationConfig,
        worker_id: int
    ):
        """Run optimization for a single worker."""
        
        try:
            study = optuna.load_study(study_name=study_name, storage=self.storage_url)
            
            self.logger.info(f"Worker {worker_id} starting {n_trials} trials")
            
            study.optimize(
                objective_func,
                n_trials=n_trials,
                timeout=config.timeout
            )
            
            self.logger.info(f"Worker {worker_id} completed")
            
        except Exception as e:
            self.logger.error(f"Worker {worker_id} failed: {e}")
    
    async def _extract_optimization_results(
        self,
        study,
        config: OptimizationConfig,
        optimization_time: float
    ) -> OptimizationResult:
        """Extract optimization results from study."""
        
        # Basic results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        # Convergence history
        convergence_history = [trial.value for trial in study.trials if trial.value is not None]
        
        # Parameter importance
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except Exception:
            param_importance = {}
        
        # Optimization history
        optimization_history = [
            {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name
            }
            for trial in study.trials
        ]
        
        # Study statistics
        study_statistics = {
            "best_value": best_value,
            "n_trials": len(study.trials),
            "optimization_time": optimization_time,
            "trials_per_second": len(study.trials) / optimization_time if optimization_time > 0 else 0
        }
        
        # Count trial states
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        return OptimizationResult(
            study=study,
            best_params=best_params,
            best_value=best_value,
            best_trial=best_trial,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            convergence_history=convergence_history,
            cv_scores={},
            validation_scores={},
            parameter_importance=param_importance,
            optimization_history=optimization_history,
            study_statistics=study_statistics,
            pruned_trials=pruned_trials,
            failed_trials=failed_trials
        )
    
    async def _save_optimization_results(self, study_name: str, result: OptimizationResult):
        """Save optimization results."""
        
        try:
            # Save study
            if hasattr(result.study, 'trials'):
                study_path = self.storage_dir / f"{study_name}_study.json"
                study_data = {
                    "study_name": study_name,
                    "best_params": result.best_params,
                    "best_value": result.best_value,
                    "n_trials": result.n_trials,
                    "optimization_time": result.optimization_time,
                    "parameter_importance": result.parameter_importance,
                    "study_statistics": result.study_statistics
                }
                
                with open(study_path, "w") as f:
                    json.dump(study_data, f, indent=2)
            
            # Save detailed results
            results_path = self.storage_dir / f"{study_name}_results.joblib"
            joblib.dump(result, results_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save optimization results: {e}")
    
    def _analyze_convergence(self, study) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        
        values = [trial.value for trial in study.trials if trial.value is not None]
        
        if not values:
            return {"status": "no_data"}
        
        # Calculate convergence metrics
        best_so_far = []
        current_best = values[0]
        
        for value in values:
            current_best = max(current_best, value)
            best_so_far.append(current_best)
        
        # Convergence analysis
        convergence_analysis = {
            "converged": self._is_converged(best_so_far),
            "convergence_trial": self._find_convergence_trial(best_so_far),
            "improvement_rate": self._calculate_improvement_rate(best_so_far),
            "plateau_detection": self._detect_plateau(best_so_far),
            "final_best": current_best,
            "total_improvement": current_best - values[0] if values else 0
        }
        
        return convergence_analysis
    
    def _analyze_parameter_correlations(self, study) -> Dict[str, Any]:
        """Analyze parameter correlations with objective."""
        
        correlations = {}
        
        try:
            # Extract parameter values and objectives
            trials_data = []
            for trial in study.trials:
                if trial.value is not None:
                    trial_data = trial.params.copy()
                    trial_data['objective'] = trial.value
                    trials_data.append(trial_data)
            
            if trials_data:
                df = pd.DataFrame(trials_data)
                
                # Calculate correlations
                for param in df.columns:
                    if param != 'objective' and df[param].dtype in ['int64', 'float64']:
                        corr = df[param].corr(df['objective'])
                        if not np.isnan(corr):
                            correlations[param] = float(corr)
        
        except Exception as e:
            self.logger.warning(f"Could not analyze parameter correlations: {e}")
        
        return correlations
    
    def _analyze_trial_distribution(self, study) -> Dict[str, Any]:
        """Analyze distribution of trial values."""
        
        values = [trial.value for trial in study.trials if trial.value is not None]
        
        if not values:
            return {"status": "no_data"}
        
        values_array = np.array(values)
        
        return {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array)),
            "q25": float(np.percentile(values_array, 25)),
            "q75": float(np.percentile(values_array, 75)),
            "n_trials": len(values)
        }
    
    def _analyze_pruning_effectiveness(self, study) -> Dict[str, Any]:
        """Analyze effectiveness of pruning."""
        
        total_trials = len(study.trials)
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        complete_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        return {
            "total_trials": total_trials,
            "pruned_trials": pruned_trials,
            "complete_trials": complete_trials,
            "pruning_rate": pruned_trials / total_trials if total_trials > 0 else 0,
            "completion_rate": complete_trials / total_trials if total_trials > 0 else 0
        }
    
    def _calculate_optimization_efficiency(self, study) -> Dict[str, Any]:
        """Calculate optimization efficiency metrics."""
        
        values = [trial.value for trial in study.trials if trial.value is not None]
        
        if not values:
            return {"status": "no_data"}
        
        # Calculate improvement over time
        improvements = []
        best_so_far = values[0]
        
        for value in values[1:]:
            if value > best_so_far:
                improvements.append(value - best_so_far)
                best_so_far = value
            else:
                improvements.append(0)
        
        return {
            "total_improvements": len([imp for imp in improvements if imp > 0]),
            "average_improvement": float(np.mean(improvements)) if improvements else 0,
            "efficiency_score": len([imp for imp in improvements if imp > 0]) / len(improvements) if improvements else 0
        }
    
    def _is_converged(self, best_so_far: List[float], threshold: float = 1e-4, patience: int = 10) -> bool:
        """Check if optimization has converged."""
        
        if len(best_so_far) < patience:
            return False
        
        recent_values = best_so_far[-patience:]
        improvement = recent_values[-1] - recent_values[0]
        
        return improvement < threshold
    
    def _find_convergence_trial(self, best_so_far: List[float]) -> Optional[int]:
        """Find the trial where convergence occurred."""
        
        threshold = 1e-4
        patience = 10
        
        for i in range(patience, len(best_so_far)):
            window = best_so_far[i-patience:i]
            improvement = window[-1] - window[0]
            
            if improvement < threshold:
                return i
        
        return None
    
    def _calculate_improvement_rate(self, best_so_far: List[float]) -> float:
        """Calculate rate of improvement."""
        
        if len(best_so_far) < 2:
            return 0.0
        
        total_improvement = best_so_far[-1] - best_so_far[0]
        n_trials = len(best_so_far)
        
        return total_improvement / n_trials
    
    def _detect_plateau(self, best_so_far: List[float], min_length: int = 20) -> Dict[str, Any]:
        """Detect if optimization has plateaued."""
        
        if len(best_so_far) < min_length:
            return {"status": "insufficient_data"}
        
        recent_values = best_so_far[-min_length:]
        
        # Check if values are approximately constant
        std_dev = np.std(recent_values)
        mean_value = np.mean(recent_values)
        
        relative_std = std_dev / (abs(mean_value) + 1e-8)
        
        return {
            "is_plateau": relative_std < 0.01,
            "plateau_length": min_length,
            "relative_std": float(relative_std)
        }
    
    def _create_html_report(self, insights: Dict[str, Any]) -> str:
        """Create HTML optimization report."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Optimization Report - {insights['study_name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Hyperparameter Optimization Report</h1>
            
            <div class="section">
                <h2>Study Overview</h2>
                <div class="metric">Study Name: {insights['study_name']}</div>
                <div class="metric">Best Score: {insights['best_value']:.6f}</div>
                <div class="metric">Optimization Time: {insights['optimization_time']:.2f} seconds</div>
                <div class="metric">Total Trials: {insights['n_trials']}</div>
            </div>
            
            <div class="section">
                <h2>Best Parameters</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        for param, value in insights['best_params'].items():
            html += f"<tr><td>{param}</td><td>{value}</td></tr>"
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Parameter Importance</h2>
                <table>
                    <tr><th>Parameter</th><th>Importance</th></tr>
        """
        
        for param, importance in insights['parameter_importance'].items():
            html += f"<tr><td>{param}</td><td>{importance:.4f}</td></tr>"
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Optimization Efficiency</h2>
        """
        
        efficiency = insights['optimization_efficiency']
        html += f"""
                <div class="metric">Total Improvements: {efficiency.get('total_improvements', 0)}</div>
                <div class="metric">Efficiency Score: {efficiency.get('efficiency_score', 0):.4f}</div>
        """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    async def _generate_optimization_plots(self, study_name: str, output_dir: str):
        """Generate optimization visualization plots."""
        
        if not PLOTLY_AVAILABLE:
            return
        
        try:
            result = self.optimization_history[study_name]
            study = result.study
            
            # Optimization history plot
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(f"{output_dir}/{study_name}_optimization_history.html")
            
            # Parameter importance plot
            if result.parameter_importance:
                fig = optuna.visualization.plot_param_importances(study)
                fig.write_html(f"{output_dir}/{study_name}_param_importance.html")
            
            # Parallel coordinate plot
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_html(f"{output_dir}/{study_name}_parallel_coordinate.html")
            
        except Exception as e:
            self.logger.warning(f"Could not generate plots: {e}")
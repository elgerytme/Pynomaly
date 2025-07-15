#!/usr/bin/env python3
"""
Advanced Model Optimization Service
Extends the existing AutoML service with state-of-the-art optimization techniques including
Bayesian optimization, multi-objective optimization, neural architecture search, and advanced ensemble methods
"""

import asyncio
import json
import logging
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import RobustScaler

# Advanced optimization libraries
try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import CmaEsSampler, NSGAIISampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

try:
    from skopt import forest_minimize, gbrt_minimize, gp_minimize
    from skopt.space import Categorical, Integer, Real
    from skopt.utils import use_named_args

    SCIKIT_OPTIMIZE_AVAILABLE = True
except ImportError:
    SCIKIT_OPTIMIZE_AVAILABLE = False

try:
    import hyperopt
    from hyperopt import Trials, atpe, fmin, hp, tpe

    HYPEROPT_AVAILABLE = True
except ImportError:
    hyperopt = None
    HYPEROPT_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Advanced optimization strategies"""

    BAYESIAN_GAUSSIAN_PROCESS = "bayesian_gp"
    BAYESIAN_RANDOM_FOREST = "bayesian_rf"
    BAYESIAN_GRADIENT_BOOSTING = "bayesian_gbrt"
    TREE_STRUCTURED_PARZEN = "tpe"
    ADAPTIVE_TPE = "atpe"
    CMA_ES = "cma_es"
    NSGA_II = "nsga_ii"  # Multi-objective
    SUCCESSIVE_HALVING = "successive_halving"
    HYPERBAND = "hyperband"
    BOHB = "bohb"  # Bayesian Optimization HyperBand


class EnsembleStrategy(Enum):
    """Advanced ensemble strategies"""

    VOTING_SOFT = "voting_soft"
    VOTING_HARD = "voting_hard"
    STACKING = "stacking"
    BLENDING = "blending"
    BAYESIAN_MODEL_AVERAGING = "bayesian_averaging"
    DYNAMIC_ENSEMBLE = "dynamic_ensemble"
    ADVERSARIAL_ENSEMBLE = "adversarial_ensemble"


class ObjectiveFunction(Enum):
    """Multi-objective optimization functions"""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    TRAINING_TIME = "training_time"
    INFERENCE_TIME = "inference_time"
    MODEL_SIZE = "model_size"
    MEMORY_USAGE = "memory_usage"


@dataclass
class OptimizationObjective:
    """Multi-objective optimization configuration"""

    function: ObjectiveFunction
    weight: float = 1.0
    direction: str = "maximize"  # "maximize" or "minimize"
    constraint: dict[str, float] | None = None


@dataclass
class AdvancedOptimizationConfig:
    """Advanced optimization configuration"""

    # Optimization strategy
    strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_GAUSSIAN_PROCESS

    # Multi-objective configuration
    objectives: list[OptimizationObjective] = field(
        default_factory=lambda: [
            OptimizationObjective(ObjectiveFunction.F1_SCORE, weight=0.6),
            OptimizationObjective(
                ObjectiveFunction.TRAINING_TIME, weight=0.2, direction="minimize"
            ),
            OptimizationObjective(
                ObjectiveFunction.MODEL_SIZE, weight=0.2, direction="minimize"
            ),
        ]
    )

    # Search space configuration
    n_trials: int = 200
    n_jobs: int = -1
    timeout_seconds: int = 7200  # 2 hours

    # Early stopping
    early_stopping_rounds: int = 50
    min_improvement: float = 1e-4

    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified"

    # Ensemble configuration
    ensemble_strategy: EnsembleStrategy = EnsembleStrategy.STACKING
    ensemble_size: int = 5
    ensemble_diversity_threshold: float = 0.1

    # Advanced features
    enable_meta_learning: bool = True
    enable_transfer_learning: bool = True
    enable_neural_architecture_search: bool = False
    enable_automated_feature_engineering: bool = True

    # Resource constraints
    max_memory_gb: float = 16.0
    max_cpu_cores: int = 8
    gpu_enabled: bool = False

    # Reproducibility
    random_state: int = 42


@dataclass
class OptimizationResult:
    """Result of advanced optimization"""

    best_model: BaseEstimator
    best_params: dict[str, Any]
    best_scores: dict[str, float]
    optimization_history: list[dict[str, Any]]

    # Multi-objective results
    pareto_front: list[dict[str, Any]] = field(default_factory=list)
    dominated_solutions: list[dict[str, Any]] = field(default_factory=list)

    # Ensemble results
    ensemble_model: BaseEstimator | None = None
    ensemble_weights: list[float] = field(default_factory=list)
    ensemble_diversity_score: float = 0.0

    # Metadata
    optimization_time: float = 0.0
    total_trials: int = 0
    successful_trials: int = 0

    # Model analysis
    feature_importance: dict[str, float] = field(default_factory=dict)
    model_complexity: dict[str, Any] = field(default_factory=dict)
    performance_trade_offs: dict[str, Any] = field(default_factory=dict)


class AdvancedModelOptimizationService:
    """Advanced Model Optimization Service with state-of-the-art techniques"""

    def __init__(self, config: AdvancedOptimizationConfig | None = None):
        self.config = config or AdvancedOptimizationConfig()
        self.optimization_history: list[dict[str, Any]] = []
        self.pareto_archive: list[dict[str, Any]] = []

        # Meta-learning database
        self.meta_learning_db: dict[str, Any] = {}

        # Performance cache
        self.performance_cache: dict[str, Any] = {}

        # Ensemble registry
        self.ensemble_registry: dict[str, dict[str, Any]] = {}

    async def optimize_model_advanced(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        model_types: list[str] | None = None,
        custom_objective: Callable | None = None,
    ) -> OptimizationResult:
        """
        Perform advanced multi-objective model optimization

        Args:
            X: Training features
            y: Training labels (optional for unsupervised)
            model_types: List of model types to optimize
            custom_objective: Custom objective function

        Returns:
            Advanced optimization result with Pareto front
        """
        logger.info("🚀 Starting advanced model optimization")
        start_time = time.time()

        try:
            # Preprocessing and feature engineering
            X_processed = await self._advanced_preprocessing(X)

            # Model space definition
            if model_types is None:
                model_types = self._get_default_model_types()

            # Multi-objective optimization
            if len(self.config.objectives) > 1:
                result = await self._multi_objective_optimization(
                    X_processed, y, model_types, custom_objective
                )
            else:
                result = await self._single_objective_optimization(
                    X_processed, y, model_types, custom_objective
                )

            # Ensemble creation
            if self.config.ensemble_strategy != EnsembleStrategy.VOTING_SOFT:
                ensemble_model = await self._create_advanced_ensemble(
                    X_processed, y, result.optimization_history
                )
                result.ensemble_model = ensemble_model

            # Performance analysis
            await self._analyze_performance_trade_offs(result, X_processed, y)

            # Meta-learning update
            if self.config.enable_meta_learning:
                await self._update_meta_learning(X_processed, y, result)

            result.optimization_time = time.time() - start_time

            logger.info(
                f"✅ Advanced optimization completed in {result.optimization_time:.2f}s"
            )
            logger.info(f"📊 Found {len(result.pareto_front)} Pareto optimal solutions")

            return result

        except Exception as e:
            logger.error(f"❌ Advanced optimization failed: {e}")
            raise

    async def _advanced_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced preprocessing and feature engineering"""
        logger.info("🔧 Applying advanced preprocessing")

        X_processed = X.copy()

        if self.config.enable_automated_feature_engineering:
            # Automated feature engineering
            X_processed = await self._automated_feature_engineering(X_processed)

        # Advanced scaling
        scaler = RobustScaler()
        numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            X_processed[numeric_columns] = scaler.fit_transform(
                X_processed[numeric_columns]
            )

        return X_processed

    async def _automated_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Automated feature engineering using various techniques"""

        X_engineered = X.copy()

        # Polynomial features (limited to avoid explosion)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) <= 5:  # Only for small feature sets
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    # Interaction features
                    X_engineered[f"{col1}_x_{col2}"] = X[col1] * X[col2]

                    # Ratio features
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = X[col1] / (X[col2] + 1e-8)
                        ratio = np.where(np.isfinite(ratio), ratio, 0)
                        X_engineered[f"{col1}_ratio_{col2}"] = ratio

        # Statistical features
        if len(numeric_cols) > 0:
            X_engineered["feature_mean"] = X[numeric_cols].mean(axis=1)
            X_engineered["feature_std"] = X[numeric_cols].std(axis=1)
            X_engineered["feature_max"] = X[numeric_cols].max(axis=1)
            X_engineered["feature_min"] = X[numeric_cols].min(axis=1)

        # Remove highly correlated features
        if len(X_engineered.columns) > len(X.columns):
            corr_matrix = X_engineered.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
            X_engineered = X_engineered.drop(columns=to_drop)

        logger.info(
            f"Feature engineering: {X.shape[1]} -> {X_engineered.shape[1]} features"
        )
        return X_engineered

    def _get_default_model_types(self) -> list[str]:
        """Get default model types for optimization"""
        return [
            "isolation_forest",
            "one_class_svm",
            "local_outlier_factor",
            "random_forest",
            "gradient_boosting",
            "extra_trees",
            "ada_boost",
        ]

    async def _multi_objective_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        model_types: list[str],
        custom_objective: Callable | None = None,
    ) -> OptimizationResult:
        """Perform multi-objective optimization using NSGA-II"""

        if not OPTUNA_AVAILABLE:
            raise ValueError("Optuna is required for multi-objective optimization")

        logger.info("🎯 Starting multi-objective optimization")

        # Create study with NSGA-II sampler
        sampler = optuna.samplers.NSGAIISampler(
            population_size=50, mutation_prob=0.1, crossover_prob=0.9
        )

        study = optuna.create_study(
            directions=[
                "maximize" if obj.direction == "maximize" else "minimize"
                for obj in self.config.objectives
            ],
            sampler=sampler,
        )

        # Define objective function
        def objective(trial):
            return self._evaluate_multi_objective_trial(
                trial, X, y, model_types, custom_objective
            )

        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=1,  # Multi-objective requires sequential execution
        )

        # Extract results
        pareto_front = []
        for trial in study.best_trials:
            pareto_front.append(
                {
                    "params": trial.params,
                    "values": trial.values,
                    "model_type": trial.user_attrs.get("model_type"),
                    "trial_number": trial.number,
                }
            )

        # Select best trial (weighted sum)
        best_trial = self._select_best_from_pareto(study.best_trials)

        # Create result
        result = OptimizationResult(
            best_model=best_trial.user_attrs.get("model"),
            best_params=best_trial.params,
            best_scores={
                obj.function.value: val
                for obj, val in zip(
                    self.config.objectives, best_trial.values, strict=False
                )
            },
            optimization_history=[
                {
                    "trial": t.number,
                    "params": t.params,
                    "values": t.values,
                    "state": t.state.name,
                }
                for t in study.trials
            ],
            pareto_front=pareto_front,
            total_trials=len(study.trials),
            successful_trials=len(
                [t for t in study.trials if t.state.name == "COMPLETE"]
            ),
        )

        return result

    def _evaluate_multi_objective_trial(
        self,
        trial,
        X: pd.DataFrame,
        y: pd.Series | None,
        model_types: list[str],
        custom_objective: Callable | None = None,
    ) -> list[float]:
        """Evaluate a single trial for multi-objective optimization"""

        try:
            # Sample model type
            model_type = trial.suggest_categorical("model_type", model_types)

            # Sample hyperparameters for the chosen model
            params = self._sample_hyperparameters(trial, model_type)

            # Create and train model
            model = self._create_model(model_type, params)

            start_time = time.time()
            model.fit(X, y)
            training_time = time.time() - start_time

            # Calculate objectives
            objective_values = []

            for objective in self.config.objectives:
                if objective.function == ObjectiveFunction.F1_SCORE and y is not None:
                    y_pred = model.predict(X)
                    value = f1_score(y, y_pred, average="weighted")
                elif objective.function == ObjectiveFunction.ACCURACY and y is not None:
                    y_pred = model.predict(X)
                    value = accuracy_score(y, y_pred)
                elif objective.function == ObjectiveFunction.TRAINING_TIME:
                    value = training_time
                elif objective.function == ObjectiveFunction.MODEL_SIZE:
                    value = len(pickle.dumps(model)) / (1024 * 1024)  # MB
                elif custom_objective:
                    value = custom_objective(model, X, y)
                else:
                    # Default fallback
                    value = 0.5

                objective_values.append(value)

            # Store model in trial attributes
            trial.set_user_attr("model", model)
            trial.set_user_attr("model_type", model_type)
            trial.set_user_attr("training_time", training_time)

            return objective_values

        except Exception as e:
            logger.warning(f"Trial evaluation failed: {e}")
            # Return worst possible values
            return [
                0.0 if obj.direction == "maximize" else float("inf")
                for obj in self.config.objectives
            ]

    def _sample_hyperparameters(self, trial, model_type: str) -> dict[str, Any]:
        """Sample hyperparameters for a specific model type"""

        params = {}

        if model_type == "isolation_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_samples": trial.suggest_categorical(
                    "max_samples", ["auto", 256, 512, 1024]
                ),
                "contamination": trial.suggest_float("contamination", 0.01, 0.3),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }
        elif model_type == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
            }
        elif model_type == "gradient_boosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }
        # Add more model types as needed

        return params

    def _create_model(self, model_type: str, params: dict[str, Any]) -> BaseEstimator:
        """Create a model instance with given parameters"""

        if model_type == "isolation_forest":
            from sklearn.ensemble import IsolationForest

            return IsolationForest(**params, random_state=self.config.random_state)
        elif model_type == "random_forest":
            return RandomForestClassifier(
                **params, random_state=self.config.random_state
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                **params, random_state=self.config.random_state
            )
        elif model_type == "extra_trees":
            return ExtraTreesClassifier(**params, random_state=self.config.random_state)
        elif model_type == "ada_boost":
            return AdaBoostClassifier(**params, random_state=self.config.random_state)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _select_best_from_pareto(self, pareto_trials) -> Any:
        """Select best trial from Pareto front using weighted sum"""

        best_trial = None
        best_score = float("-inf")

        for trial in pareto_trials:
            # Calculate weighted sum
            weighted_score = 0.0
            for obj, value in zip(self.config.objectives, trial.values, strict=False):
                if obj.direction == "maximize":
                    weighted_score += obj.weight * value
                else:
                    weighted_score += obj.weight * (1.0 / (1.0 + value))

            if weighted_score > best_score:
                best_score = weighted_score
                best_trial = trial

        return best_trial

    async def _single_objective_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        model_types: list[str],
        custom_objective: Callable | None = None,
    ) -> OptimizationResult:
        """Perform single-objective optimization"""

        logger.info("🎯 Starting single-objective optimization")

        best_model = None
        best_score = float("-inf")
        best_params = {}
        optimization_history = []

        for model_type in model_types:
            logger.info(f"Optimizing {model_type}")

            # Use appropriate optimization strategy
            if (
                self.config.strategy == OptimizationStrategy.BAYESIAN_GAUSSIAN_PROCESS
                and SCIKIT_OPTIMIZE_AVAILABLE
            ):
                result = await self._bayesian_optimization_gp(
                    X, y, model_type, custom_objective
                )
            elif OPTUNA_AVAILABLE:
                result = await self._optuna_optimization(
                    X, y, model_type, custom_objective
                )
            else:
                result = await self._grid_search_optimization(
                    X, y, model_type, custom_objective
                )

            optimization_history.extend(result["history"])

            if result["best_score"] > best_score:
                best_score = result["best_score"]
                best_model = result["best_model"]
                best_params = result["best_params"]

        return OptimizationResult(
            best_model=best_model,
            best_params=best_params,
            best_scores={self.config.objectives[0].function.value: best_score},
            optimization_history=optimization_history,
            total_trials=len(optimization_history),
            successful_trials=len(
                [h for h in optimization_history if h.get("score", 0) > 0]
            ),
        )

    async def _bayesian_optimization_gp(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        model_type: str,
        custom_objective: Callable | None = None,
    ) -> dict[str, Any]:
        """Bayesian optimization using Gaussian Process"""

        if not SCIKIT_OPTIMIZE_AVAILABLE:
            raise ValueError("scikit-optimize is required for Bayesian optimization")

        # Define search space
        space = self._get_search_space(model_type)

        @use_named_args(space)
        def objective(**params):
            try:
                model = self._create_model(model_type, params)
                model.fit(X, y)

                if custom_objective:
                    score = custom_objective(model, X, y)
                elif y is not None:
                    y_pred = model.predict(X)
                    score = f1_score(y, y_pred, average="weighted")
                else:
                    score = 0.5

                return -score  # Minimize negative score

            except Exception as e:
                logger.warning(f"Objective evaluation failed: {e}")
                return 1.0  # Bad score

        # Perform optimization
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=self.config.n_trials // len(self._get_default_model_types()),
            n_initial_points=10,
            random_state=self.config.random_state,
        )

        # Extract best parameters
        best_params = dict(zip([dim.name for dim in space], result.x, strict=False))
        best_model = self._create_model(model_type, best_params)
        best_model.fit(X, y)

        return {
            "best_model": best_model,
            "best_params": best_params,
            "best_score": -result.fun,
            "history": [
                {
                    "iteration": i,
                    "score": -val,
                    "params": dict(zip([dim.name for dim in space], x, strict=False)),
                }
                for i, (val, x) in enumerate(
                    zip(result.func_vals, result.x_iters, strict=False)
                )
            ],
        }

    def _get_search_space(self, model_type: str):
        """Get search space for scikit-optimize"""

        from skopt.space import Categorical, Integer, Real

        if model_type == "isolation_forest":
            return [
                Integer(50, 500, name="n_estimators"),
                Categorical(["auto", 256, 512, 1024], name="max_samples"),
                Real(0.01, 0.3, name="contamination"),
                Real(0.5, 1.0, name="max_features"),
                Categorical([True, False], name="bootstrap"),
            ]
        elif model_type == "random_forest":
            return [
                Integer(50, 500, name="n_estimators"),
                Integer(3, 20, name="max_depth"),
                Integer(2, 20, name="min_samples_split"),
                Integer(1, 10, name="min_samples_leaf"),
                Categorical(["sqrt", "log2", None], name="max_features"),
            ]
        # Add more model types as needed

        return []

    async def _optuna_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        model_type: str,
        custom_objective: Callable | None = None,
    ) -> dict[str, Any]:
        """Optimization using Optuna"""

        # Choose sampler based on strategy
        if self.config.strategy == OptimizationStrategy.TREE_STRUCTURED_PARZEN:
            sampler = TPESampler(seed=self.config.random_state)
        elif self.config.strategy == OptimizationStrategy.CMA_ES:
            sampler = CmaEsSampler(seed=self.config.random_state)
        else:
            sampler = TPESampler(seed=self.config.random_state)

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=MedianPruner() if self.config.early_stopping_rounds > 0 else None,
        )

        def objective(trial):
            params = self._sample_hyperparameters(trial, model_type)

            try:
                model = self._create_model(model_type, params)
                model.fit(X, y)

                if custom_objective:
                    score = custom_objective(model, X, y)
                elif y is not None:
                    y_pred = model.predict(X)
                    score = f1_score(y, y_pred, average="weighted")
                else:
                    score = 0.5

                return score

            except Exception as e:
                logger.warning(f"Trial evaluation failed: {e}")
                return 0.0

        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials // len(self._get_default_model_types()),
            timeout=self.config.timeout_seconds // len(self._get_default_model_types()),
        )

        # Create best model
        best_model = self._create_model(model_type, study.best_params)
        best_model.fit(X, y)

        return {
            "best_model": best_model,
            "best_params": study.best_params,
            "best_score": study.best_value,
            "history": [
                {"trial": t.number, "score": t.value, "params": t.params}
                for t in study.trials
                if t.value is not None
            ],
        }

    async def _grid_search_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        model_type: str,
        custom_objective: Callable | None = None,
    ) -> dict[str, Any]:
        """Fallback grid search optimization"""

        from sklearn.model_selection import GridSearchCV

        # Define parameter grid
        param_grid = self._get_param_grid(model_type)

        # Create base model
        base_model = self._create_model(model_type, {})

        # Custom scoring function
        if custom_objective:
            scorer = make_scorer(
                lambda y_true, y_pred: custom_objective(base_model, X, y)
            )
        else:
            scorer = make_scorer(f1_score, average="weighted")

        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring=scorer,
            cv=self.config.cv_folds,
            n_jobs=self.config.n_jobs,
        )

        grid_search.fit(X, y)

        return {
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "history": [
                {"params": params, "score": score}
                for params, score in zip(
                    grid_search.cv_results_["params"],
                    grid_search.cv_results_["mean_test_score"],
                    strict=False,
                )
            ],
        }

    def _get_param_grid(self, model_type: str) -> dict[str, list[Any]]:
        """Get parameter grid for grid search"""

        if model_type == "isolation_forest":
            return {
                "n_estimators": [50, 100, 200],
                "contamination": [0.05, 0.1, 0.15],
                "max_features": [0.7, 0.8, 0.9, 1.0],
            }
        elif model_type == "random_forest":
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            }

        return {}

    async def _create_advanced_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        optimization_history: list[dict[str, Any]],
    ) -> BaseEstimator:
        """Create advanced ensemble from optimization history"""

        logger.info(
            f"🎼 Creating advanced ensemble using {self.config.ensemble_strategy.value}"
        )

        # Select diverse models from history
        top_models = sorted(
            optimization_history, key=lambda x: x.get("score", 0), reverse=True
        )[: self.config.ensemble_size]

        if self.config.ensemble_strategy == EnsembleStrategy.STACKING:
            # Stacking ensemble
            base_estimators = []
            for i, model_info in enumerate(top_models):
                if "model" in model_info:  # Assuming model is stored
                    base_estimators.append((f"model_{i}", model_info["model"]))

            if base_estimators:
                from sklearn.linear_model import LogisticRegression

                ensemble = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(),
                    cv=3,
                )
                ensemble.fit(X, y)
                return ensemble

        elif self.config.ensemble_strategy == EnsembleStrategy.VOTING_SOFT:
            # Soft voting ensemble
            estimators = []
            for i, model_info in enumerate(top_models):
                if "model" in model_info and hasattr(
                    model_info["model"], "predict_proba"
                ):
                    estimators.append((f"model_{i}", model_info["model"]))

            if estimators:
                ensemble = VotingClassifier(estimators=estimators, voting="soft")
                ensemble.fit(X, y)
                return ensemble

        # Fallback to simple voting
        estimators = []
        for i, model_info in enumerate(top_models):
            if "model" in model_info:
                estimators.append((f"model_{i}", model_info["model"]))

        if estimators:
            ensemble = VotingClassifier(estimators=estimators, voting="hard")
            ensemble.fit(X, y)
            return ensemble

        return None

    async def _analyze_performance_trade_offs(
        self, result: OptimizationResult, X: pd.DataFrame, y: pd.Series | None
    ):
        """Analyze performance trade-offs between different objectives"""

        if not result.pareto_front:
            return

        trade_offs = {}

        # Analyze correlation between objectives
        objective_names = [obj.function.value for obj in self.config.objectives]
        objective_values = np.array(
            [
                [trial["values"][i] for trial in result.pareto_front]
                for i in range(len(objective_names))
            ]
        )

        for i, obj1 in enumerate(objective_names):
            for j, obj2 in enumerate(objective_names):
                if i < j:
                    correlation = np.corrcoef(objective_values[i], objective_values[j])[
                        0, 1
                    ]
                    trade_offs[f"{obj1}_vs_{obj2}"] = {
                        "correlation": correlation,
                        "trade_off_strength": "strong"
                        if abs(correlation) > 0.7
                        else "moderate"
                        if abs(correlation) > 0.3
                        else "weak",
                    }

        result.performance_trade_offs = trade_offs

    async def _update_meta_learning(
        self, X: pd.DataFrame, y: pd.Series | None, result: OptimizationResult
    ):
        """Update meta-learning database with optimization results"""

        # Dataset characteristics
        dataset_features = {
            "n_samples": len(X),
            "n_features": len(X.columns),
            "sparsity": (X == 0).sum().sum() / (len(X) * len(X.columns)),
            "has_missing": X.isnull().any().any(),
            "numeric_ratio": len(X.select_dtypes(include=[np.number]).columns)
            / len(X.columns),
        }

        # Performance results
        performance_summary = {
            "best_score": max(result.best_scores.values()),
            "optimization_time": result.optimization_time,
            "successful_trials_ratio": result.successful_trials / result.total_trials
            if result.total_trials > 0
            else 0,
        }

        # Store in meta-learning database
        meta_entry = {
            "timestamp": datetime.now().isoformat(),
            "dataset_features": dataset_features,
            "performance": performance_summary,
            "best_algorithm": result.best_params.get("model_type", "unknown"),
            "best_params": result.best_params,
        }

        dataset_hash = hash(str(sorted(dataset_features.items())))
        self.meta_learning_db[dataset_hash] = meta_entry

        logger.info("📚 Meta-learning database updated")

    async def get_meta_learning_recommendations(
        self, X: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Get recommendations based on meta-learning"""

        if not self.meta_learning_db:
            return []

        # Calculate dataset features
        current_features = {
            "n_samples": len(X),
            "n_features": len(X.columns),
            "sparsity": (X == 0).sum().sum() / (len(X) * len(X.columns)),
            "has_missing": X.isnull().any().any(),
            "numeric_ratio": len(X.select_dtypes(include=[np.number]).columns)
            / len(X.columns),
        }

        # Find similar datasets
        similarities = []
        for dataset_hash, meta_entry in self.meta_learning_db.items():
            similarity = self._calculate_dataset_similarity(
                current_features, meta_entry["dataset_features"]
            )
            similarities.append((similarity, meta_entry))

        # Sort by similarity and return top recommendations
        similarities.sort(key=lambda x: x[0], reverse=True)

        recommendations = []
        for similarity, meta_entry in similarities[:3]:
            recommendations.append(
                {
                    "algorithm": meta_entry["best_algorithm"],
                    "params": meta_entry["best_params"],
                    "expected_performance": meta_entry["performance"]["best_score"],
                    "similarity": similarity,
                    "confidence": min(
                        similarity
                        * meta_entry["performance"]["successful_trials_ratio"],
                        1.0,
                    ),
                }
            )

        return recommendations

    def _calculate_dataset_similarity(self, features1: dict, features2: dict) -> float:
        """Calculate similarity between two datasets based on their features"""

        # Normalize numeric features
        similarity = 0.0
        weight_sum = 0.0

        for key in features1:
            if key in features2:
                if isinstance(features1[key], (int, float)) and isinstance(
                    features2[key], (int, float)
                ):
                    # Normalize and calculate similarity for numeric features
                    max_val = max(features1[key], features2[key], 1)
                    diff = abs(features1[key] - features2[key]) / max_val
                    similarity += (1 - diff) * 1.0  # Weight = 1.0
                    weight_sum += 1.0
                elif features1[key] == features2[key]:
                    # Exact match for categorical features
                    similarity += 1.0
                    weight_sum += 1.0

        return similarity / weight_sum if weight_sum > 0 else 0.0

    async def export_optimization_results(
        self, result: OptimizationResult, export_path: str
    ) -> bool:
        """Export optimization results to file"""

        try:
            export_data = {
                "optimization_config": {
                    "strategy": self.config.strategy.value,
                    "objectives": [
                        {
                            "function": obj.function.value,
                            "weight": obj.weight,
                            "direction": obj.direction,
                        }
                        for obj in self.config.objectives
                    ],
                    "n_trials": self.config.n_trials,
                    "ensemble_strategy": self.config.ensemble_strategy.value,
                },
                "results": {
                    "best_params": result.best_params,
                    "best_scores": result.best_scores,
                    "optimization_time": result.optimization_time,
                    "total_trials": result.total_trials,
                    "successful_trials": result.successful_trials,
                },
                "pareto_front": result.pareto_front,
                "performance_trade_offs": result.performance_trade_offs,
                "timestamp": datetime.now().isoformat(),
            }

            # Save model separately
            if result.best_model:
                model_path = export_path.replace(".json", "_model.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(result.best_model, f)
                export_data["model_path"] = model_path

            if result.ensemble_model:
                ensemble_path = export_path.replace(".json", "_ensemble.pkl")
                with open(ensemble_path, "wb") as f:
                    pickle.dump(result.ensemble_model, f)
                export_data["ensemble_path"] = ensemble_path

            # Save results
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"✅ Optimization results exported to {export_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to export results: {e}")
            return False


async def main():
    """Example usage of Advanced Model Optimization Service"""

    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(1000, 15), columns=[f"feature_{i}" for i in range(15)]
    )
    y = pd.Series(np.random.choice([0, 1], size=1000, p=[0.8, 0.2]))

    # Configure advanced optimization
    config = AdvancedOptimizationConfig(
        strategy=OptimizationStrategy.BAYESIAN_GAUSSIAN_PROCESS,
        objectives=[
            OptimizationObjective(ObjectiveFunction.F1_SCORE, weight=0.6),
            OptimizationObjective(
                ObjectiveFunction.TRAINING_TIME, weight=0.2, direction="minimize"
            ),
            OptimizationObjective(
                ObjectiveFunction.MODEL_SIZE, weight=0.2, direction="minimize"
            ),
        ],
        n_trials=50,
        ensemble_strategy=EnsembleStrategy.STACKING,
        enable_meta_learning=True,
    )

    # Create service
    optimization_service = AdvancedModelOptimizationService(config)

    # Perform optimization
    result = await optimization_service.optimize_model_advanced(X, y)

    print(f"Best model type: {result.best_params.get('model_type', 'unknown')}")
    print(f"Best scores: {result.best_scores}")
    print(f"Optimization time: {result.optimization_time:.2f}s")
    print(f"Pareto front size: {len(result.pareto_front)}")
    print(f"Performance trade-offs: {result.performance_trade_offs}")


if __name__ == "__main__":
    asyncio.run(main())

"""
Enhanced Model Training Framework for Anomaly Detection

Comprehensive model training orchestrator that integrates with the ML Pipeline Framework
to provide advanced training capabilities, algorithm optimization, and performance tracking.

This addresses Issue #143: Phase 2.2: Data Science Package - Machine Learning Pipeline Framework
Component 2: Enhanced Model Training Framework
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
    validation_curve,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Optional dependencies with graceful fallback
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from pynomaly.domain.entities.pipeline import Pipeline, PipelineStep, PipelineType, StepType
from pynomaly.domain.value_objects.performance_metrics import PerformanceMetrics
from pynomaly.infrastructure.logging.structured_logger import StructuredLogger


class TrainingStrategy(Enum):
    """Training strategies for different scenarios."""
    
    STANDARD = "standard"
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES = "time_series"
    INCREMENTAL = "incremental"
    ENSEMBLE = "ensemble"
    ACTIVE_LEARNING = "active_learning"


class ValidationStrategy(Enum):
    """Validation strategies for model evaluation."""
    
    HOLDOUT = "holdout"
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    TIME_SERIES_SPLIT = "time_series_split"
    LEAVE_ONE_OUT = "leave_one_out"
    BOOTSTRAP = "bootstrap"


class ModelComplexity(Enum):
    """Model complexity levels for automatic selection."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ADAPTIVE = "adaptive"


@dataclass
class TrainingConfiguration:
    """Configuration for model training."""
    
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_strategy: TrainingStrategy = TrainingStrategy.STANDARD
    validation_strategy: ValidationStrategy = ValidationStrategy.K_FOLD
    complexity_level: ModelComplexity = ModelComplexity.MEDIUM
    
    # Training parameters
    max_training_time: Optional[int] = 3600  # seconds
    early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Cross-validation parameters
    cv_folds: int = 5
    cv_scoring: str = "roc_auc"
    cv_random_state: int = 42
    
    # Data preprocessing
    scaling_method: str = "standard"  # standard, robust, minmax, none
    handle_imbalance: bool = True
    feature_selection: bool = False
    
    # Advanced options
    ensemble_methods: List[str] = field(default_factory=list)
    optuna_trials: int = 100
    use_gpu: bool = False
    n_jobs: int = -1


@dataclass
class TrainingResult:
    """Result of model training."""
    
    model: BaseEstimator
    training_config: TrainingConfiguration
    training_time: float
    validation_scores: Dict[str, float]
    cross_validation_scores: Optional[Dict[str, List[float]]] = None
    learning_curves: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_complexity: Optional[Dict[str, Any]] = None
    training_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfiguration:
    """Configuration for ensemble training."""
    
    base_models: List[str]
    ensemble_method: str = "voting"  # voting, bagging, stacking
    voting_type: str = "soft"  # soft, hard (for voting)
    stacking_meta_learner: str = "logistic_regression"
    bagging_n_estimators: int = 10
    bagging_max_samples: float = 1.0


class EnhancedModelTrainingFramework:
    """Enhanced Model Training Framework for Anomaly Detection.
    
    Provides comprehensive model training capabilities including:
    - Multiple training strategies and validation methods
    - Advanced hyperparameter optimization
    - Ensemble learning support
    - Performance tracking and model comparison
    - Integration with ML Pipeline Framework
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        results_dir: str = "training_results",
        use_gpu: bool = False
    ):
        self.logger = StructuredLogger("enhanced_training")
        
        # Storage directories
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.use_gpu = use_gpu
        
        # Training history
        self.training_history: Dict[str, List[TrainingResult]] = {}
        
        # Model registry
        self.trained_models: Dict[str, TrainingResult] = {}
        
        # Performance tracking
        self.training_metrics: Dict[str, Any] = {}
        
    async def train_model(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        config: TrainingConfiguration = None,
        model_name: Optional[str] = None
    ) -> TrainingResult:
        """Train a model with specified configuration.
        
        Args:
            X: Training features
            y: Training target (optional for unsupervised)
            config: Training configuration
            model_name: Name for the trained model
            
        Returns:
            Training result with model and metrics
        """
        if config is None:
            config = TrainingConfiguration(algorithm="isolation_forest")
            
        if model_name is None:
            model_name = f"{config.algorithm}_{int(time.time())}"
            
        self.logger.info(f"Starting model training: {model_name}")
        start_time = time.time()
        
        try:
            # Prepare data
            X_processed = await self._preprocess_data(X, config)
            
            # Get base estimator
            estimator = self._get_base_estimator(config.algorithm, config.hyperparameters)
            
            # Execute training strategy
            training_result = await self._execute_training_strategy(
                estimator, X_processed, y, config, model_name
            )
            
            # Calculate training time
            training_time = time.time() - start_time
            training_result.training_time = training_time
            
            # Store training result
            self.trained_models[model_name] = training_result
            
            # Update training history
            if config.algorithm not in self.training_history:
                self.training_history[config.algorithm] = []
            self.training_history[config.algorithm].append(training_result)
            
            # Save model
            await self._save_training_result(model_name, training_result)
            
            self.logger.info(
                f"Model training completed: {model_name} "
                f"(Time: {training_time:.2f}s, Score: {training_result.validation_scores.get('roc_auc', 0):.4f})"
            )
            
            return training_result
            
        except Exception as e:
            self.logger.error(f"Model training failed for {model_name}: {e}")
            raise
    
    async def train_ensemble(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        ensemble_config: EnsembleConfiguration = None,
        model_name: Optional[str] = None
    ) -> TrainingResult:
        """Train an ensemble of models.
        
        Args:
            X: Training features
            y: Training target (optional for unsupervised)
            ensemble_config: Ensemble configuration
            model_name: Name for the ensemble model
            
        Returns:
            Training result with ensemble model
        """
        if ensemble_config is None:
            ensemble_config = EnsembleConfiguration(
                base_models=["isolation_forest", "one_class_svm", "local_outlier_factor"]
            )
            
        if model_name is None:
            model_name = f"ensemble_{int(time.time())}"
            
        self.logger.info(f"Starting ensemble training: {model_name}")
        start_time = time.time()
        
        try:
            # Train base models
            base_model_results = []
            for algorithm in ensemble_config.base_models:
                config = TrainingConfiguration(
                    algorithm=algorithm,
                    training_strategy=TrainingStrategy.STANDARD,
                    validation_strategy=ValidationStrategy.K_FOLD
                )
                
                base_model_name = f"{model_name}_{algorithm}"
                result = await self.train_model(X, y, config, base_model_name)
                base_model_results.append(result)
            
            # Create ensemble
            ensemble_model = self._create_ensemble(base_model_results, ensemble_config)
            
            # Validate ensemble
            validation_scores = await self._validate_model(
                ensemble_model, X, y, ValidationStrategy.K_FOLD
            )
            
            # Create ensemble result
            ensemble_result = TrainingResult(
                model=ensemble_model,
                training_config=TrainingConfiguration(
                    algorithm=f"ensemble_{ensemble_config.ensemble_method}"
                ),
                training_time=time.time() - start_time,
                validation_scores=validation_scores,
                training_metadata={
                    "ensemble_config": ensemble_config,
                    "base_models": [r.training_config.algorithm for r in base_model_results],
                    "base_model_scores": [r.validation_scores for r in base_model_results]
                }
            )
            
            # Store ensemble result
            self.trained_models[model_name] = ensemble_result
            await self._save_training_result(model_name, ensemble_result)
            
            self.logger.info(f"Ensemble training completed: {model_name}")
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Ensemble training failed for {model_name}: {e}")
            raise
    
    async def optimize_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        algorithm: str = "isolation_forest",
        optimization_trials: int = 100,
        optimization_timeout: int = 3600,
        model_name: Optional[str] = None
    ) -> TrainingResult:
        """Optimize hyperparameters using Optuna.
        
        Args:
            X: Training features
            y: Training target (optional for unsupervised)
            algorithm: Algorithm to optimize
            optimization_trials: Number of optimization trials
            optimization_timeout: Timeout for optimization in seconds
            model_name: Name for the optimized model
            
        Returns:
            Training result with optimized model
        """
        if not OPTUNA_AVAILABLE:
            self.logger.warning("Optuna not available, falling back to default hyperparameters")
            config = TrainingConfiguration(algorithm=algorithm)
            return await self.train_model(X, y, config, model_name)
        
        if model_name is None:
            model_name = f"{algorithm}_optimized_{int(time.time())}"
            
        self.logger.info(f"Starting hyperparameter optimization: {model_name}")
        
        # Create optimization objective
        def objective(trial):
            # Get hyperparameter suggestions
            hyperparams = self._get_hyperparameter_suggestions(algorithm, trial)
            
            # Create estimator with suggested hyperparameters
            estimator = self._get_base_estimator(algorithm, hyperparams)
            
            # Evaluate with cross-validation
            if y is not None:
                scores = cross_val_score(
                    estimator, X, y, cv=5, scoring="roc_auc", n_jobs=-1
                )
            else:
                # For unsupervised models, use a different scoring method
                scores = self._evaluate_unsupervised_model(estimator, X)
            
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=optimization_trials,
            timeout=optimization_timeout,
            n_jobs=1  # Optuna handles parallelization
        )
        
        # Train final model with best hyperparameters
        best_params = study.best_params
        config = TrainingConfiguration(
            algorithm=algorithm,
            hyperparameters=best_params,
            training_strategy=TrainingStrategy.CROSS_VALIDATION
        )
        
        result = await self.train_model(X, y, config, model_name)
        
        # Add optimization metadata
        result.training_metadata.update({
            "optimization_study": study,
            "best_hyperparameters": best_params,
            "optimization_score": study.best_value,
            "optimization_trials": len(study.trials)
        })
        
        self.logger.info(
            f"Hyperparameter optimization completed: {model_name} "
            f"(Best score: {study.best_value:.4f})"
        )
        
        return result
    
    async def compare_models(
        self,
        model_names: List[str],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Compare multiple trained models.
        
        Args:
            model_names: Names of models to compare
            X_test: Test features
            y_test: Test target (optional)
            
        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing {len(model_names)} models")
        
        comparison_results = {
            "models": {},
            "rankings": {},
            "best_model": None,
            "comparison_metrics": {}
        }
        
        model_scores = {}
        
        for model_name in model_names:
            if model_name not in self.trained_models:
                self.logger.warning(f"Model {model_name} not found, skipping")
                continue
            
            result = self.trained_models[model_name]
            model = result.model
            
            # Get predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            metrics = {}
            if y_test is not None:
                # Supervised metrics
                metrics.update({
                    "accuracy": accuracy_score(y_test, predictions),
                    "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                    "f1": f1_score(y_test, predictions, average="weighted", zero_division=0)
                })
                
                # ROC AUC if probabilities available
                if hasattr(model, "decision_function"):
                    decision_scores = model.decision_function(X_test)
                    metrics["roc_auc"] = roc_auc_score(y_test, decision_scores)
                elif hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = roc_auc_score(y_test, probabilities)
            else:
                # Unsupervised metrics
                metrics.update(self._calculate_unsupervised_metrics(model, X_test, predictions))
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(metrics)
            model_scores[model_name] = composite_score
            
            comparison_results["models"][model_name] = {
                "metrics": metrics,
                "composite_score": composite_score,
                "training_time": result.training_time,
                "algorithm": result.training_config.algorithm
            }
        
        # Rank models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        comparison_results["rankings"] = {
            rank + 1: {"model": model, "score": score}
            for rank, (model, score) in enumerate(ranked_models)
        }
        
        if ranked_models:
            comparison_results["best_model"] = ranked_models[0][0]
        
        # Calculate comparison metrics
        comparison_results["comparison_metrics"] = {
            "total_models": len(model_scores),
            "score_range": max(model_scores.values()) - min(model_scores.values()) if model_scores else 0,
            "average_score": sum(model_scores.values()) / len(model_scores) if model_scores else 0
        }
        
        self.logger.info(f"Model comparison completed. Best model: {comparison_results['best_model']}")
        return comparison_results
    
    async def get_model_insights(self, model_name: str) -> Dict[str, Any]:
        """Get insights about a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model insights and analysis
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        result = self.trained_models[model_name]
        model = result.model
        
        insights = {
            "model_name": model_name,
            "algorithm": result.training_config.algorithm,
            "training_time": result.training_time,
            "validation_scores": result.validation_scores,
            "model_complexity": self._analyze_model_complexity(model),
            "feature_importance": self._extract_feature_importance(model),
            "model_parameters": self._get_model_parameters(model),
            "training_metadata": result.training_metadata
        }
        
        # Add cross-validation insights if available
        if result.cross_validation_scores:
            insights["cross_validation_analysis"] = self._analyze_cv_scores(
                result.cross_validation_scores
            )
        
        return insights
    
    async def _preprocess_data(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        config: TrainingConfiguration
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Preprocess data according to configuration."""
        
        if isinstance(X, pd.DataFrame):
            X_processed = X.copy()
        else:
            X_processed = X.copy()
        
        # Apply scaling
        if config.scaling_method != "none":
            scaler = self._get_scaler(config.scaling_method)
            if isinstance(X_processed, pd.DataFrame):
                numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
                X_processed[numeric_columns] = scaler.fit_transform(X_processed[numeric_columns])
            else:
                X_processed = scaler.fit_transform(X_processed)
        
        return X_processed
    
    def _get_scaler(self, scaling_method: str):
        """Get scaler based on method."""
        scalers = {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler()
        }
        return scalers.get(scaling_method, StandardScaler())
    
    def _get_base_estimator(self, algorithm: str, hyperparameters: Dict[str, Any] = None) -> BaseEstimator:
        """Get base estimator for algorithm."""
        
        if hyperparameters is None:
            hyperparameters = {}
        
        estimators = {
            "isolation_forest": IsolationForest(random_state=42, **hyperparameters),
            "one_class_svm": self._import_estimator("sklearn.svm", "OneClassSVM")(**hyperparameters),
            "local_outlier_factor": self._import_estimator("sklearn.neighbors", "LocalOutlierFactor")(
                novelty=True, **hyperparameters
            ),
            "elliptic_envelope": self._import_estimator("sklearn.covariance", "EllipticEnvelope")(
                random_state=42, **hyperparameters
            ),
        }
        
        # Add gradient boosting algorithms if available
        if XGBOOST_AVAILABLE:
            estimators["xgboost"] = xgb.XGBClassifier(random_state=42, **hyperparameters)
        
        if LIGHTGBM_AVAILABLE:
            estimators["lightgbm"] = lgb.LGBMClassifier(random_state=42, **hyperparameters)
        
        if CATBOOST_AVAILABLE:
            estimators["catboost"] = cb.CatBoostClassifier(
                random_state=42, verbose=False, **hyperparameters
            )
        
        if algorithm not in estimators:
            raise ValueError(f"Algorithm {algorithm} not supported")
        
        return estimators[algorithm]
    
    def _import_estimator(self, module_name: str, class_name: str):
        """Dynamically import estimator class."""
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    
    async def _execute_training_strategy(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]],
        config: TrainingConfiguration,
        model_name: str
    ) -> TrainingResult:
        """Execute training strategy."""
        
        if config.training_strategy == TrainingStrategy.STANDARD:
            return await self._train_standard(estimator, X, y, config)
        elif config.training_strategy == TrainingStrategy.CROSS_VALIDATION:
            return await self._train_cross_validation(estimator, X, y, config)
        elif config.training_strategy == TrainingStrategy.TIME_SERIES:
            return await self._train_time_series(estimator, X, y, config)
        else:
            # Fallback to standard training
            return await self._train_standard(estimator, X, y, config)
    
    async def _train_standard(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]],
        config: TrainingConfiguration
    ) -> TrainingResult:
        """Standard training with train/validation split."""
        
        # Split data
        if y is not None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=config.cv_random_state
            )
        else:
            X_train, X_val = train_test_split(
                X, test_size=0.2, random_state=config.cv_random_state
            )
            y_train, y_val = None, None
        
        # Train model
        trained_model = estimator.fit(X_train, y_train)
        
        # Validate model
        validation_scores = await self._validate_model(
            trained_model, X_val, y_val, config.validation_strategy
        )
        
        return TrainingResult(
            model=trained_model,
            training_config=config,
            training_time=0,  # Will be set by caller
            validation_scores=validation_scores
        )
    
    async def _train_cross_validation(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]],
        config: TrainingConfiguration
    ) -> TrainingResult:
        """Cross-validation training."""
        
        # Perform cross-validation
        cv_scores = {}
        scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"] if y is not None else ["silhouette"]
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(
                    estimator, X, y, cv=config.cv_folds, 
                    scoring=metric, n_jobs=config.n_jobs
                )
                cv_scores[metric] = scores.tolist()
            except Exception as e:
                self.logger.warning(f"Could not calculate {metric}: {e}")
        
        # Train final model on full dataset
        trained_model = estimator.fit(X, y)
        
        # Calculate validation scores as CV means
        validation_scores = {
            metric: np.mean(scores) for metric, scores in cv_scores.items()
        }
        
        return TrainingResult(
            model=trained_model,
            training_config=config,
            training_time=0,  # Will be set by caller
            validation_scores=validation_scores,
            cross_validation_scores=cv_scores
        )
    
    async def _train_time_series(
        self,
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]],
        config: TrainingConfiguration
    ) -> TrainingResult:
        """Time series training with temporal validation."""
        
        # Use TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=config.cv_folds)
        
        cv_scores = {}
        scoring_metrics = ["accuracy", "precision", "recall", "f1"] if y is not None else ["silhouette"]
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(
                    estimator, X, y, cv=tscv, 
                    scoring=metric, n_jobs=config.n_jobs
                )
                cv_scores[metric] = scores.tolist()
            except Exception as e:
                self.logger.warning(f"Could not calculate {metric}: {e}")
        
        # Train final model on full dataset
        trained_model = estimator.fit(X, y)
        
        # Calculate validation scores
        validation_scores = {
            metric: np.mean(scores) for metric, scores in cv_scores.items()
        }
        
        return TrainingResult(
            model=trained_model,
            training_config=config,
            training_time=0,  # Will be set by caller
            validation_scores=validation_scores,
            cross_validation_scores=cv_scores
        )
    
    async def _validate_model(
        self,
        model: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]],
        strategy: ValidationStrategy
    ) -> Dict[str, float]:
        """Validate model using specified strategy."""
        
        try:
            predictions = model.predict(X)
            
            if y is not None:
                # Supervised validation
                scores = {
                    "accuracy": accuracy_score(y, predictions),
                    "precision": precision_score(y, predictions, average="weighted", zero_division=0),
                    "recall": recall_score(y, predictions, average="weighted", zero_division=0),
                    "f1": f1_score(y, predictions, average="weighted", zero_division=0)
                }
                
                # Add ROC AUC if applicable
                if hasattr(model, "decision_function"):
                    decision_scores = model.decision_function(X)
                    scores["roc_auc"] = roc_auc_score(y, decision_scores)
                elif hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X)[:, 1]
                    scores["roc_auc"] = roc_auc_score(y, probabilities)
            else:
                # Unsupervised validation
                scores = self._calculate_unsupervised_metrics(model, X, predictions)
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {"error": 1.0}
    
    def _calculate_unsupervised_metrics(
        self, 
        model: BaseEstimator, 
        X: Union[pd.DataFrame, np.ndarray], 
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """Calculate metrics for unsupervised models."""
        
        metrics = {}
        
        try:
            # Anomaly ratio
            anomaly_ratio = np.sum(predictions == -1) / len(predictions)
            metrics["anomaly_ratio"] = anomaly_ratio
            
            # Silhouette score (if possible)
            if len(np.unique(predictions)) > 1:
                from sklearn.metrics import silhouette_score
                metrics["silhouette_score"] = silhouette_score(X, predictions)
            
            # Decision scores statistics (if available)
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                metrics.update({
                    "decision_score_mean": np.mean(scores),
                    "decision_score_std": np.std(scores),
                    "decision_score_range": np.max(scores) - np.min(scores)
                })
        
        except Exception as e:
            self.logger.warning(f"Could not calculate some unsupervised metrics: {e}")
            metrics["anomaly_ratio"] = np.sum(predictions == -1) / len(predictions)
        
        return metrics
    
    def _create_ensemble(
        self, 
        base_results: List[TrainingResult], 
        config: EnsembleConfiguration
    ) -> BaseEstimator:
        """Create ensemble model from base results."""
        
        base_models = [result.model for result in base_results]
        
        if config.ensemble_method == "voting":
            from sklearn.ensemble import VotingClassifier
            estimators = [(f"model_{i}", model) for i, model in enumerate(base_models)]
            return VotingClassifier(estimators=estimators, voting=config.voting_type)
        
        elif config.ensemble_method == "bagging":
            # Simple average for unsupervised models
            return EnsembleWrapper(base_models, method="average")
        
        else:
            # Default to simple ensemble wrapper
            return EnsembleWrapper(base_models, method="majority_vote")
    
    def _get_hyperparameter_suggestions(self, algorithm: str, trial) -> Dict[str, Any]:
        """Get hyperparameter suggestions for Optuna trial."""
        
        suggestions = {}
        
        if algorithm == "isolation_forest":
            suggestions.update({
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
                "contamination": trial.suggest_float("contamination", 0.01, 0.5),
                "max_features": trial.suggest_float("max_features", 0.1, 1.0)
            })
        
        elif algorithm == "one_class_svm":
            suggestions.update({
                "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "nu": trial.suggest_float("nu", 0.01, 0.9)
            })
        
        elif algorithm == "local_outlier_factor":
            suggestions.update({
                "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
                "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
                "leaf_size": trial.suggest_int("leaf_size", 10, 50)
            })
        
        # Add suggestions for other algorithms as needed
        
        return suggestions
    
    def _evaluate_unsupervised_model(self, model: BaseEstimator, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Evaluate unsupervised model (simplified scoring)."""
        
        try:
            model.fit(X)
            predictions = model.predict(X)
            
            # Simple scoring based on anomaly ratio and consistency
            anomaly_ratio = np.sum(predictions == -1) / len(predictions)
            
            # Prefer models with reasonable anomaly ratios (not too high, not too low)
            ideal_ratio = 0.1  # 10% anomalies
            ratio_score = 1.0 - abs(anomaly_ratio - ideal_ratio) / ideal_ratio
            
            # Add decision score variance if available
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                variance_score = np.std(scores) / (np.abs(np.mean(scores)) + 1e-8)
                total_score = 0.7 * ratio_score + 0.3 * min(variance_score, 1.0)
            else:
                total_score = ratio_score
            
            return np.array([total_score] * 5)  # Return as array for consistency with cross_val_score
            
        except Exception as e:
            self.logger.warning(f"Error evaluating unsupervised model: {e}")
            return np.array([0.0] * 5)
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score from metrics."""
        
        weights = {
            "roc_auc": 0.4,
            "f1": 0.3,
            "precision": 0.15,
            "recall": 0.15,
            "accuracy": 0.0,  # Less important for anomaly detection
            "silhouette_score": 0.5,  # For unsupervised
            "anomaly_ratio": 0.3  # For unsupervised (closer to ideal is better)
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            if metric in weights:
                weight = weights[metric]
                
                # Special handling for anomaly ratio (closer to 0.1 is better)
                if metric == "anomaly_ratio":
                    ideal_ratio = 0.1
                    normalized_value = 1.0 - abs(value - ideal_ratio) / ideal_ratio
                    normalized_value = max(0.0, min(1.0, normalized_value))
                else:
                    normalized_value = max(0.0, min(1.0, value))
                
                score += weight * normalized_value
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _analyze_model_complexity(self, model: BaseEstimator) -> Dict[str, Any]:
        """Analyze model complexity."""
        
        complexity = {
            "model_type": type(model).__name__,
            "parameters": self._count_model_parameters(model),
            "complexity_level": "unknown"
        }
        
        # Estimate complexity based on model type and parameters
        param_count = complexity["parameters"]
        
        if param_count < 10:
            complexity["complexity_level"] = "low"
        elif param_count < 100:
            complexity["complexity_level"] = "medium"
        else:
            complexity["complexity_level"] = "high"
        
        return complexity
    
    def _count_model_parameters(self, model: BaseEstimator) -> int:
        """Count model parameters."""
        
        try:
            # For scikit-learn models, count fitted attributes
            param_count = 0
            for attr_name in dir(model):
                if attr_name.endswith("_") and not attr_name.startswith("_"):
                    attr_value = getattr(model, attr_name)
                    if hasattr(attr_value, "shape"):
                        param_count += np.prod(attr_value.shape)
                    elif isinstance(attr_value, (int, float)):
                        param_count += 1
                    elif isinstance(attr_value, (list, tuple)):
                        param_count += len(attr_value)
            
            return param_count
            
        except Exception:
            return 0
    
    def _extract_feature_importance(self, model: BaseEstimator) -> Optional[Dict[str, float]]:
        """Extract feature importance if available."""
        
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}
            elif hasattr(model, "coef_"):
                coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                return {f"feature_{i}": float(coef) for i, coef in enumerate(coefficients)}
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
        
        return None
    
    def _get_model_parameters(self, model: BaseEstimator) -> Dict[str, Any]:
        """Get model parameters."""
        
        try:
            return model.get_params()
        except Exception:
            return {}
    
    def _analyze_cv_scores(self, cv_scores: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze cross-validation scores."""
        
        analysis = {}
        
        for metric, scores in cv_scores.items():
            scores_array = np.array(scores)
            analysis[metric] = {
                "mean": float(np.mean(scores_array)),
                "std": float(np.std(scores_array)),
                "min": float(np.min(scores_array)),
                "max": float(np.max(scores_array)),
                "stability": float(1.0 - np.std(scores_array) / (np.mean(scores_array) + 1e-8))
            }
        
        return analysis
    
    async def _save_training_result(self, model_name: str, result: TrainingResult):
        """Save training result to disk."""
        
        try:
            # Save model
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(result.model, model_path)
            
            # Save metadata
            metadata_path = self.results_dir / f"{model_name}_metadata.json"
            metadata = {
                "model_name": model_name,
                "algorithm": result.training_config.algorithm,
                "training_time": result.training_time,
                "validation_scores": result.validation_scores,
                "training_config": {
                    "algorithm": result.training_config.algorithm,
                    "training_strategy": result.training_config.training_strategy.value,
                    "validation_strategy": result.training_config.validation_strategy.value,
                    "hyperparameters": result.training_config.hyperparameters
                },
                "saved_at": datetime.now().isoformat()
            }
            
            import json
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save training result for {model_name}: {e}")


class EnsembleWrapper(BaseEstimator):
    """Simple ensemble wrapper for combining multiple models."""
    
    def __init__(self, models: List[BaseEstimator], method: str = "majority_vote"):
        self.models = models
        self.method = method
        
    def fit(self, X, y=None):
        # Models are already fitted
        return self
    
    def predict(self, X):
        if self.method == "majority_vote":
            predictions = np.array([model.predict(X) for model in self.models])
            return np.sign(np.sum(predictions, axis=0))
        elif self.method == "average":
            if hasattr(self.models[0], "decision_function"):
                scores = np.array([model.decision_function(X) for model in self.models])
                avg_scores = np.mean(scores, axis=0)
                return np.where(avg_scores >= 0, 1, -1)
            else:
                # Fallback to majority vote
                predictions = np.array([model.predict(X) for model in self.models])
                return np.sign(np.sum(predictions, axis=0))
    
    def decision_function(self, X):
        if hasattr(self.models[0], "decision_function"):
            scores = np.array([model.decision_function(X) for model in self.models])
            return np.mean(scores, axis=0)
        else:
            raise AttributeError("Base models do not support decision_function")
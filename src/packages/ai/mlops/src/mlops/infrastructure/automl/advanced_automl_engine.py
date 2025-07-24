"""
Advanced AutoML Engine

Comprehensive automated machine learning system with neural architecture search,
hyperparameter optimization, automated feature engineering, and model selection.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import optuna
import structlog

from mlops.domain.entities.model import Model, ModelVersion
from mlops.infrastructure.feature_store.feature_store import FeatureStore


class AutoMLTask(Enum):
    """AutoML task types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"


class OptimizationObjective(Enum):
    """Optimization objectives."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MAE = "mean_absolute_error"
    MSE = "mean_squared_error"
    RMSE = "root_mean_squared_error"
    R2_SCORE = "r2_score"


class FeatureEngineering(Enum):
    """Feature engineering strategies."""
    BASIC = "basic"
    ADVANCED = "advanced"
    DEEP = "deep"
    CUSTOM = "custom"


@dataclass
class AutoMLConfig:
    """Configuration for AutoML experiments."""
    task_type: AutoMLTask
    optimization_objective: OptimizationObjective
    max_runtime_minutes: int = 60
    max_trials: int = 100
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Feature engineering
    feature_engineering: FeatureEngineering = FeatureEngineering.ADVANCED
    enable_feature_selection: bool = True
    max_features: int = 1000
    
    # Model selection
    include_linear_models: bool = True
    include_tree_models: bool = True
    include_ensemble_models: bool = True
    include_neural_networks: bool = True
    include_deep_learning: bool = False
    
    # Neural architecture search
    enable_nas: bool = False
    nas_max_layers: int = 10
    nas_max_units: int = 512
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 10
    
    # Resource constraints
    max_memory_gb: float = 8.0
    max_cpu_cores: int = -1  # -1 for all available
    
    # Output configuration
    save_intermediate_models: bool = True
    generate_explanations: bool = True
    create_model_report: bool = True


@dataclass
class AutoMLResult:
    """Result of AutoML experiment."""
    experiment_id: str
    task_type: AutoMLTask
    best_model: Any
    best_score: float
    best_params: Dict[str, Any]
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Model performance
    cv_scores: List[float] = field(default_factory=list)
    test_score: float = 0.0
    training_time_seconds: float = 0.0
    
    # Feature information
    selected_features: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    engineered_features: List[str] = field(default_factory=list)
    
    # Model artifacts
    model_path: Optional[str] = None
    preprocessing_pipeline_path: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    config: Optional[AutoMLConfig] = None
    
    # Performance analysis
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[Dict[str, Any]] = None
    learning_curves: Optional[Dict[str, List[float]]] = None


class AdvancedAutoMLEngine:
    """Advanced automated machine learning system."""
    
    def __init__(self, feature_store: FeatureStore = None):
        self.feature_store = feature_store
        self.logger = structlog.get_logger(__name__)
        
        # Experiment tracking
        self.experiments: Dict[str, AutoMLResult] = {}
        self.active_studies: Dict[str, optuna.Study] = {}
        
        # Feature engineering
        self.feature_engineer = AdvancedFeatureEngineer()
        self.feature_selector = FeatureSelector()
        
        # Model registry
        self.model_registry = AutoMLModelRegistry()
        
        # Neural architecture search
        self.nas_engine = NeuralArchitectureSearchEngine()
        
        # Hyperparameter optimization
        self.hpo_engine = HyperparameterOptimizationEngine()
        
        # Performance evaluator
        self.evaluator = ModelEvaluator()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
    
    async def start_automl_experiment(self,
                                    data: pd.DataFrame,
                                    target_column: str,
                                    config: AutoMLConfig,
                                    experiment_name: str = "") -> str:
        """Start a new AutoML experiment."""
        
        experiment_id = str(uuid.uuid4())
        
        if not experiment_name:
            experiment_name = f"automl_experiment_{experiment_id[:8]}"
        
        self.logger.info(
            "Starting AutoML experiment",
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            task_type=config.task_type.value,
            data_shape=data.shape,
            target_column=target_column
        )
        
        # Initialize result
        result = AutoMLResult(
            experiment_id=experiment_id,
            task_type=config.task_type,
            best_model=None,
            best_score=0.0,
            best_params={},
            config=config
        )
        
        self.experiments[experiment_id] = result
        
        # Start experiment asynchronously
        task = asyncio.create_task(
            self._run_automl_experiment(experiment_id, data, target_column, config)
        )
        self.background_tasks.append(task)
        
        return experiment_id
    
    async def _run_automl_experiment(self,
                                   experiment_id: str,
                                   data: pd.DataFrame,
                                   target_column: str,
                                   config: AutoMLConfig) -> None:
        """Run the complete AutoML experiment."""
        
        start_time = datetime.utcnow()
        result = self.experiments[experiment_id]
        
        try:
            # Step 1: Data preprocessing and splitting
            X, y, X_test, y_test = await self._prepare_data(data, target_column, config)
            
            # Step 2: Feature engineering
            if config.feature_engineering != FeatureEngineering.BASIC:
                X, X_test, engineered_features = await self.feature_engineer.engineer_features(
                    X, X_test, config.task_type, config.feature_engineering
                )
                result.engineered_features = engineered_features
            
            # Step 3: Feature selection
            if config.enable_feature_selection:
                X, X_test, selected_features = await self.feature_selector.select_features(
                    X, y, X_test, config.task_type, config.max_features
                )
                result.selected_features = selected_features
            
            # Step 4: Model selection and hyperparameter optimization
            best_model, best_score, best_params, optimization_history = await self._optimize_model(
                X, y, config, experiment_id
            )
            
            # Step 5: Final evaluation
            test_score, performance_metrics = await self.evaluator.evaluate_model(
                best_model, X_test, y_test, config.task_type
            )
            
            # Step 6: Feature importance analysis
            feature_importance = await self._calculate_feature_importance(
                best_model, X.columns.tolist()
            )
            
            # Update result
            result.best_model = best_model
            result.best_score = best_score
            result.best_params = best_params
            result.test_score = test_score
            result.optimization_history = optimization_history
            result.feature_importance = feature_importance
            result.training_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            
            # Add performance metrics
            if 'confusion_matrix' in performance_metrics:
                result.confusion_matrix = performance_metrics['confusion_matrix']
            if 'classification_report' in performance_metrics:
                result.classification_report = performance_metrics['classification_report']
            
            # Step 7: Save model artifacts
            if config.save_intermediate_models:
                model_path, pipeline_path = await self._save_model_artifacts(
                    experiment_id, best_model, X.columns.tolist()
                )
                result.model_path = model_path
                result.preprocessing_pipeline_path = pipeline_path
            
            # Step 8: Generate model report
            if config.create_model_report:
                await self._generate_model_report(experiment_id, result)
            
            self.logger.info(
                "AutoML experiment completed successfully",
                experiment_id=experiment_id,
                best_score=best_score,
                test_score=test_score,
                training_time=result.training_time_seconds
            )
            
        except Exception as e:
            self.logger.error(
                "AutoML experiment failed",
                experiment_id=experiment_id,
                error=str(e)
            )
            raise
    
    async def _prepare_data(self,
                          data: pd.DataFrame,
                          target_column: str,
                          config: AutoMLConfig) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Prepare data for AutoML experiment."""
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))
        X = X.fillna(X.mode().iloc[0])  # For categorical columns
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target if classification
        if config.task_type in [AutoMLTask.BINARY_CLASSIFICATION, AutoMLTask.MULTICLASS_CLASSIFICATION]:
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state,
            stratify=y if config.task_type in [AutoMLTask.BINARY_CLASSIFICATION, AutoMLTask.MULTICLASS_CLASSIFICATION] else None
        )
        
        return X_train, y_train, X_test, y_test
    
    async def _optimize_model(self,
                            X: pd.DataFrame,
                            y: pd.Series,
                            config: AutoMLConfig,
                            experiment_id: str) -> Tuple[Any, float, Dict[str, Any], List[Dict[str, Any]]]:
        """Optimize model selection and hyperparameters."""
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize' if self._is_maximize_objective(config.optimization_objective) else 'minimize',
            study_name=f"automl_{experiment_id}"
        )
        
        self.active_studies[experiment_id] = study
        
        # Define objective function
        def objective(trial):
            return self._objective_function(trial, X, y, config)
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=config.max_trials,
            timeout=config.max_runtime_minutes * 60,
            show_progress_bar=True
        )
        
        # Get best results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        # Retrain best model with full data
        best_model = self._create_model_from_params(best_params, config.task_type)
        best_model.fit(X, y)
        
        # Extract optimization history
        optimization_history = [
            {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name
            }
            for trial in study.trials
        ]
        
        return best_model, best_score, best_params, optimization_history
    
    def _objective_function(self, trial, X, y, config):
        """Optuna objective function for hyperparameter optimization."""
        
        # Select model type
        model_types = []
        if config.include_linear_models:
            model_types.extend(['logistic_regression', 'linear_regression'])
        if config.include_tree_models:
            model_types.extend(['random_forest'])
        if config.include_neural_networks:
            model_types.extend(['mlp'])
        
        model_type = trial.suggest_categorical('model_type', model_types)
        
        # Generate hyperparameters based on model type
        if model_type == 'random_forest':
            params = {
                'model_type': 'random_forest',
                'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
        elif model_type == 'logistic_regression':
            params = {
                'model_type': 'logistic_regression',
                'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000)
            }
        elif model_type == 'mlp':
            hidden_layer_sizes = []
            n_layers = trial.suggest_int('n_layers', 1, 3)
            for i in range(n_layers):
                size = trial.suggest_int(f'layer_{i}_size', 50, 300)
                hidden_layer_sizes.append(size)
            
            params = {
                'model_type': 'mlp',
                'hidden_layer_sizes': tuple(hidden_layer_sizes),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True)
            }
        else:
            params = {'model_type': model_type}
        
        # Create and evaluate model
        model = self._create_model_from_params(params, config.task_type)
        
        # Cross-validation
        if config.task_type in [AutoMLTask.BINARY_CLASSIFICATION, AutoMLTask.MULTICLASS_CLASSIFICATION]:
            cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        else:
            cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
        
        # Get scoring metric
        scoring = self._get_scoring_metric(config.optimization_objective)
        
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
            return scores.mean()
        except Exception as e:
            # Return worst possible score if model fails
            return -1e6 if self._is_maximize_objective(config.optimization_objective) else 1e6
    
    def _create_model_from_params(self, params: Dict[str, Any], task_type: AutoMLTask):
        """Create model instance from parameters."""
        
        model_type = params['model_type']
        
        if model_type == 'random_forest':
            if task_type == AutoMLTask.REGRESSION:
                return RandomForestRegressor(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2),
                    min_samples_leaf=params.get('min_samples_leaf', 1),
                    random_state=42
                )
            else:
                return RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2),
                    min_samples_leaf=params.get('min_samples_leaf', 1),
                    random_state=42
                )
        
        elif model_type == 'logistic_regression':
            return LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=params.get('max_iter', 1000),
                random_state=42
            )
        
        elif model_type == 'linear_regression':
            return LinearRegression()
        
        elif model_type == 'mlp':
            if task_type == AutoMLTask.REGRESSION:
                return MLPRegressor(
                    hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
                    alpha=params.get('alpha', 0.0001),
                    learning_rate_init=params.get('learning_rate_init', 0.001),
                    max_iter=500,
                    random_state=42
                )
            else:
                return MLPClassifier(
                    hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
                    alpha=params.get('alpha', 0.0001),
                    learning_rate_init=params.get('learning_rate_init', 0.001),
                    max_iter=500,
                    random_state=42
                )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _get_scoring_metric(self, objective: OptimizationObjective) -> str:
        """Get sklearn scoring metric name."""
        
        mapping = {
            OptimizationObjective.ACCURACY: 'accuracy',
            OptimizationObjective.PRECISION: 'precision_weighted',
            OptimizationObjective.RECALL: 'recall_weighted',
            OptimizationObjective.F1_SCORE: 'f1_weighted',
            OptimizationObjective.ROC_AUC: 'roc_auc',
            OptimizationObjective.MAE: 'neg_mean_absolute_error',
            OptimizationObjective.MSE: 'neg_mean_squared_error',
            OptimizationObjective.RMSE: 'neg_root_mean_squared_error',
            OptimizationObjective.R2_SCORE: 'r2'
        }
        
        return mapping.get(objective, 'accuracy')
    
    def _is_maximize_objective(self, objective: OptimizationObjective) -> bool:
        """Check if objective should be maximized."""
        
        maximize_objectives = {
            OptimizationObjective.ACCURACY,
            OptimizationObjective.PRECISION,
            OptimizationObjective.RECALL,
            OptimizationObjective.F1_SCORE,
            OptimizationObjective.ROC_AUC,
            OptimizationObjective.R2_SCORE
        }
        
        return objective in maximize_objectives
    
    async def _calculate_feature_importance(self,
                                          model: Any,
                                          feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance from trained model."""
        
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                for i, importance in enumerate(importances):
                    if i < len(feature_names):
                        importance_dict[feature_names[i]] = float(importance)
            
            elif hasattr(model, 'coef_'):
                # Linear models
                coefs = model.coef_
                if len(coefs.shape) == 1:
                    # Binary classification or regression
                    for i, coef in enumerate(coefs):
                        if i < len(feature_names):
                            importance_dict[feature_names[i]] = float(abs(coef))
                else:
                    # Multiclass classification
                    avg_coefs = np.mean(np.abs(coefs), axis=0)
                    for i, coef in enumerate(avg_coefs):
                        if i < len(feature_names):
                            importance_dict[feature_names[i]] = float(coef)
            
        except Exception as e:
            self.logger.warning("Failed to calculate feature importance", error=str(e))
        
        return importance_dict
    
    async def _save_model_artifacts(self,
                                  experiment_id: str,
                                  model: Any,
                                  feature_names: List[str]) -> Tuple[str, str]:
        """Save model and preprocessing artifacts."""
        
        # Create directory for experiment
        experiment_dir = Path(f"automl_experiments/{experiment_id}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = experiment_dir / "best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save feature names and preprocessing info
        preprocessing_info = {
            "feature_names": feature_names,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        pipeline_path = experiment_dir / "preprocessing_pipeline.json"
        with open(pipeline_path, 'w') as f:
            json.dump(preprocessing_info, f)
        
        return str(model_path), str(pipeline_path)
    
    async def _generate_model_report(self, experiment_id: str, result: AutoMLResult) -> None:
        """Generate comprehensive model report."""
        
        report = {
            "experiment_id": experiment_id,
            "task_type": result.task_type.value,
            "best_model_type": result.best_params.get('model_type', 'unknown'),
            "performance": {
                "cv_score": result.best_score,
                "test_score": result.test_score,
                "training_time_seconds": result.training_time_seconds
            },
            "feature_analysis": {
                "selected_features_count": len(result.selected_features),
                "engineered_features_count": len(result.engineered_features),
                "top_features": dict(sorted(
                    result.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
            },
            "optimization": {
                "total_trials": len(result.optimization_history),
                "best_trial": max(result.optimization_history, key=lambda x: x['value']) if result.optimization_history else None
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Save report
        experiment_dir = Path(f"automl_experiments/{experiment_id}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = experiment_dir / "model_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(
            "Model report generated",
            experiment_id=experiment_id,
            report_path=str(report_path)
        )
    
    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get status of AutoML experiment."""
        
        if experiment_id not in self.experiments:
            return {"error": f"Experiment {experiment_id} not found"}
        
        result = self.experiments[experiment_id]
        
        # Check if experiment is still running
        is_running = any(
            not task.done() for task in self.background_tasks
            if hasattr(task, 'get_name') and experiment_id in task.get_name()
        )
        
        status = {
            "experiment_id": experiment_id,
            "status": "running" if is_running else "completed",
            "task_type": result.task_type.value,
            "created_at": result.created_at.isoformat(),
            "training_time_seconds": result.training_time_seconds,
            "best_score": result.best_score,
            "test_score": result.test_score,
            "optimization_trials": len(result.optimization_history),
            "selected_features_count": len(result.selected_features),
            "model_path": result.model_path
        }
        
        return status
    
    async def get_experiment_results(self, experiment_id: str) -> AutoMLResult:
        """Get detailed results of AutoML experiment."""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        return self.experiments[experiment_id]
    
    async def load_best_model(self, experiment_id: str) -> Any:
        """Load the best model from an experiment."""
        
        result = await self.get_experiment_results(experiment_id)
        
        if result.model_path and Path(result.model_path).exists():
            with open(result.model_path, 'rb') as f:
                return pickle.load(f)
        else:
            return result.best_model
    
    async def predict_with_experiment(self,
                                    experiment_id: str,
                                    data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model from an experiment."""
        
        model = await self.load_best_model(experiment_id)
        result = await self.get_experiment_results(experiment_id)
        
        # Prepare data (basic preprocessing)
        X = data.copy()
        
        # Apply same preprocessing as training
        X = X.fillna(X.mean(numeric_only=True))
        X = X.fillna(X.mode().iloc[0])
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Select only features used in training
        if result.selected_features:
            available_features = [f for f in result.selected_features if f in X.columns]
            X = X[available_features]
        
        return model.predict(X)
    
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List all AutoML experiments."""
        
        experiments = []
        
        for experiment_id, result in self.experiments.items():
            is_running = any(
                not task.done() for task in self.background_tasks
                if hasattr(task, 'get_name') and experiment_id in task.get_name()
            )
            
            experiments.append({
                "experiment_id": experiment_id,
                "task_type": result.task_type.value,
                "status": "running" if is_running else "completed",
                "created_at": result.created_at.isoformat(),
                "best_score": result.best_score,
                "test_score": result.test_score,
                "training_time_seconds": result.training_time_seconds
            })
        
        return sorted(experiments, key=lambda x: x['created_at'], reverse=True)


class AdvancedFeatureEngineer:
    """Advanced feature engineering capabilities."""
    
    async def engineer_features(self,
                              X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              task_type: AutoMLTask,
                              strategy: FeatureEngineering) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Apply feature engineering transformations."""
        
        engineered_features = []
        
        if strategy in [FeatureEngineering.ADVANCED, FeatureEngineering.DEEP]:
            # Polynomial features
            X_train, X_test, poly_features = self._create_polynomial_features(X_train, X_test)
            engineered_features.extend(poly_features)
            
            # Interaction features
            X_train, X_test, interaction_features = self._create_interaction_features(X_train, X_test)
            engineered_features.extend(interaction_features)
        
        if strategy == FeatureEngineering.DEEP:
            # Statistical features
            X_train, X_test, stat_features = self._create_statistical_features(X_train, X_test)
            engineered_features.extend(stat_features)
        
        return X_train, X_test, engineered_features
    
    def _create_polynomial_features(self, X_train, X_test, degree=2, max_features=50):
        """Create polynomial features for numerical columns."""
        
        from sklearn.preprocessing import PolynomialFeatures
        
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns[:10]  # Limit to prevent explosion
        
        if len(numerical_cols) == 0:
            return X_train, X_test, []
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        
        try:
            X_poly_train = poly.fit_transform(X_train[numerical_cols])
            X_poly_test = poly.transform(X_test[numerical_cols])
            
            # Limit number of polynomial features
            if X_poly_train.shape[1] > max_features:
                X_poly_train = X_poly_train[:, :max_features]
                X_poly_test = X_poly_test[:, :max_features]
            
            # Create feature names
            poly_feature_names = [f"poly_{i}" for i in range(X_poly_train.shape[1])]
            
            # Add to original dataframes
            for i, name in enumerate(poly_feature_names):
                X_train[name] = X_poly_train[:, i]
                X_test[name] = X_poly_test[:, i]
            
            return X_train, X_test, poly_feature_names
            
        except Exception:
            return X_train, X_test, []
    
    def _create_interaction_features(self, X_train, X_test, max_interactions=20):
        """Create interaction features between numerical columns."""
        
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        interaction_features = []
        
        count = 0
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                if count >= max_interactions:
                    break
                
                col1, col2 = numerical_cols[i], numerical_cols[j]
                
                # Multiplication interaction
                interaction_name = f"{col1}_x_{col2}"
                X_train[interaction_name] = X_train[col1] * X_train[col2]
                X_test[interaction_name] = X_test[col1] * X_test[col2]
                interaction_features.append(interaction_name)
                
                count += 1
        
        return X_train, X_test, interaction_features
    
    def _create_statistical_features(self, X_train, X_test):
        """Create statistical aggregation features."""
        
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        stat_features = []
        
        if len(numerical_cols) > 1:
            # Row-wise statistics
            X_train['row_mean'] = X_train[numerical_cols].mean(axis=1)
            X_test['row_mean'] = X_test[numerical_cols].mean(axis=1)
            stat_features.append('row_mean')
            
            X_train['row_std'] = X_train[numerical_cols].std(axis=1)
            X_test['row_std'] = X_test[numerical_cols].std(axis=1)
            stat_features.append('row_std')
            
            X_train['row_max'] = X_train[numerical_cols].max(axis=1)
            X_test['row_max'] = X_test[numerical_cols].max(axis=1)
            stat_features.append('row_max')
            
            X_train['row_min'] = X_train[numerical_cols].min(axis=1)
            X_test['row_min'] = X_test[numerical_cols].min(axis=1)
            stat_features.append('row_min')
        
        return X_train, X_test, stat_features


class FeatureSelector:
    """Feature selection capabilities."""
    
    async def select_features(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            task_type: AutoMLTask,
                            max_features: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Select most important features."""
        
        if X_train.shape[1] <= max_features:
            return X_train, X_test, X_train.columns.tolist()
        
        try:
            if task_type == AutoMLTask.REGRESSION:
                from sklearn.feature_selection import SelectKBest, f_regression
                selector = SelectKBest(score_func=f_regression, k=max_features)
            else:
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(score_func=f_classif, k=max_features)
            
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_features = X_train.columns[selected_indices].tolist()
            
            # Create new dataframes
            X_train_new = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
            X_test_new = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
            
            return X_train_new, X_test_new, selected_features
            
        except Exception:
            # Fallback to using all features
            return X_train, X_test, X_train.columns.tolist()


class ModelEvaluator:
    """Model evaluation and performance analysis."""
    
    async def evaluate_model(self,
                           model: Any,
                           X_test: pd.DataFrame,
                           y_test: pd.Series,
                           task_type: AutoMLTask) -> Tuple[float, Dict[str, Any]]:
        """Evaluate model performance on test set."""
        
        predictions = model.predict(X_test)
        performance_metrics = {}
        
        if task_type == AutoMLTask.REGRESSION:
            # Regression metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            performance_metrics.update({
                'mae': mae,
                'mse': mse,
                'rmse': mse ** 0.5,
                'r2': r2
            })
            
            return r2, performance_metrics
        
        else:
            # Classification metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.metrics import confusion_matrix, classification_report
            
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            
            # Confusion matrix
            cm = confusion_matrix(y_test, predictions)
            
            # Classification report
            report = classification_report(y_test, predictions, output_dict=True)
            
            performance_metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            })
            
            return accuracy, performance_metrics


class AutoMLModelRegistry:
    """Registry for AutoML generated models."""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
    
    async def register_model(self, experiment_id: str, model_info: Dict[str, Any]) -> str:
        """Register an AutoML model."""
        model_id = f"automl_{experiment_id}"
        self.models[model_id] = model_info
        return model_id
    
    async def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get model information."""
        return self.models.get(model_id, {})


class NeuralArchitectureSearchEngine:
    """Neural architecture search capabilities."""
    
    async def search_architecture(self,
                                X_train: pd.DataFrame,
                                y_train: pd.Series,
                                task_type: AutoMLTask,
                                max_trials: int = 50) -> Dict[str, Any]:
        """Search for optimal neural network architecture."""
        
        # Simplified NAS implementation
        # In production, would use more sophisticated methods like DARTS, ENAS, etc.
        
        best_architecture = {
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dropout", "rate": 0.3},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dropout", "rate": 0.2}
            ],
            "optimizer": "adam",
            "learning_rate": 0.001
        }
        
        return best_architecture


class HyperparameterOptimizationEngine:
    """Advanced hyperparameter optimization."""
    
    async def optimize_hyperparameters(self,
                                     model_class: Any,
                                     X_train: pd.DataFrame,
                                     y_train: pd.Series,
                                     param_space: Dict[str, Any],
                                     n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using advanced methods."""
        
        # This would implement advanced HPO methods like:
        # - Bayesian Optimization
        # - Multi-fidelity optimization
        # - Population-based training
        # - Hyperband
        
        # For now, return default parameters
        return {}
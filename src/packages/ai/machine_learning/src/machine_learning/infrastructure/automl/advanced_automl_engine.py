"""
Advanced AutoML Engine

Comprehensive automated machine learning system with neural architecture search,
hyperparameter optimization, feature engineering automation, and model selection.
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import structlog

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna

from machine_learning.domain.entities.model import Model, ModelStatus
from machine_learning.infrastructure.feature_store.feature_store import FeatureStore


class AutoMLTaskType(Enum):
    """Types of AutoML tasks."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


class OptimizationObjective(Enum):
    """Optimization objectives for AutoML."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    CUSTOM = "custom"


class ModelFamily(Enum):
    """Model families for AutoML search."""
    LINEAR_MODELS = "linear_models"
    TREE_MODELS = "tree_models"
    ENSEMBLE_MODELS = "ensemble_models"
    NEURAL_NETWORKS = "neural_networks"
    SVM_MODELS = "svm_models"
    BAYESIAN_MODELS = "bayesian_models"
    ALL = "all"


@dataclass
class AutoMLConfig:
    """Configuration for AutoML experiments."""
    task_type: AutoMLTaskType
    optimization_objective: OptimizationObjective
    max_training_time_hours: float = 2.0
    max_trials: int = 100
    cross_validation_folds: int = 5
    test_size: float = 0.2
    
    # Model search space
    model_families: List[ModelFamily] = field(default_factory=lambda: [ModelFamily.ALL])
    include_feature_engineering: bool = True
    include_feature_selection: bool = True
    include_neural_architecture_search: bool = False
    
    # Performance constraints
    max_model_size_mb: float = 100.0
    max_inference_time_ms: float = 100.0
    min_accuracy_threshold: float = 0.0
    
    # Resource constraints
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 4
    enable_gpu: bool = False
    
    # Advanced features
    enable_ensemble: bool = True
    enable_meta_learning: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Data preprocessing
    handle_missing_values: bool = True
    handle_categorical_features: bool = True
    enable_data_augmentation: bool = False
    
    # Interpretability
    require_interpretable_models: bool = False
    generate_feature_importance: bool = True
    
    # Reproducibility
    random_seed: int = 42


@dataclass
class ModelCandidate:
    """Individual model candidate in AutoML search."""
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_family: ModelFamily = ModelFamily.TREE_MODELS
    algorithm_name: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Feature engineering
    feature_transformations: List[str] = field(default_factory=list)
    selected_features: List[str] = field(default_factory=list)
    
    # Performance metrics
    cv_scores: List[float] = field(default_factory=list)
    mean_cv_score: float = 0.0
    std_cv_score: float = 0.0
    training_time_seconds: float = 0.0
    inference_time_ms: float = 0.0
    model_size_mb: float = 0.0
    
    # Model artifacts
    trained_model: Any = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Validation results
    test_score: Optional[float] = None
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Status
    status: str = "created"  # created, training, completed, failed
    error_message: Optional[str] = None
    
    def get_score(self) -> float:
        """Get the primary score for this candidate."""
        return self.mean_cv_score


@dataclass
class AutoMLExperiment:
    """AutoML experiment tracking."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    config: AutoMLConfig = field(default_factory=AutoMLConfig)
    
    # Data information
    dataset_shape: Tuple[int, int] = (0, 0)
    feature_names: List[str] = field(default_factory=list)
    target_name: str = ""
    
    # Experiment progress
    status: str = "created"  # created, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Model candidates
    candidates: List[ModelCandidate] = field(default_factory=list)
    best_candidate: Optional[ModelCandidate] = None
    
    # Results
    leaderboard: List[Dict[str, Any]] = field(default_factory=list)
    experiment_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Resource usage
    total_training_time_hours: float = 0.0
    peak_memory_usage_gb: float = 0.0
    
    def add_candidate(self, candidate: ModelCandidate) -> None:
        """Add a candidate to the experiment."""
        self.candidates.append(candidate)
        
        # Update best candidate
        if (not self.best_candidate or 
            candidate.get_score() > self.best_candidate.get_score()):
            self.best_candidate = candidate
    
    def get_leaderboard(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top performing candidates."""
        sorted_candidates = sorted(
            [c for c in self.candidates if c.status == "completed"],
            key=lambda c: c.get_score(),
            reverse=True
        )
        
        leaderboard = []
        for i, candidate in enumerate(sorted_candidates[:top_k]):
            leaderboard.append({
                "rank": i + 1,
                "candidate_id": candidate.candidate_id,
                "algorithm": candidate.algorithm_name,
                "score": candidate.get_score(),
                "cv_std": candidate.std_cv_score,
                "training_time": candidate.training_time_seconds,
                "inference_time_ms": candidate.inference_time_ms,
                "model_size_mb": candidate.model_size_mb
            })
        
        return leaderboard


class AdvancedAutoMLEngine:
    """Advanced AutoML engine with comprehensive capabilities."""
    
    def __init__(self, feature_store: FeatureStore = None):
        self.feature_store = feature_store
        self.logger = structlog.get_logger(__name__)
        
        # Experiment management
        self.experiments: Dict[str, AutoMLExperiment] = {}
        self.active_experiments: Dict[str, asyncio.Task] = {}
        
        # Model search spaces
        self.model_search_spaces = self._init_model_search_spaces()
        
        # Feature engineering pipelines
        self.feature_engineering_pipelines = FeatureEngineeringPipelines()
        
        # Neural Architecture Search
        self.nas_engine = NeuralArchitectureSearch()
        
        # Meta-learning
        self.meta_learner = MetaLearner()
        
        # Resource management
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Hyperparameter optimization
        self.hyperopt_studies: Dict[str, optuna.Study] = {}
    
    def _init_model_search_spaces(self) -> Dict[ModelFamily, Dict[str, Any]]:
        """Initialize model search spaces."""
        return {
            ModelFamily.LINEAR_MODELS: {
                "LogisticRegression": {
                    "C": optuna.distributions.FloatDistribution(0.01, 100.0, log=True),
                    "solver": optuna.distributions.CategoricalDistribution(["liblinear", "lbfgs"]),
                    "max_iter": optuna.distributions.IntDistribution(100, 1000)
                }
            },
            ModelFamily.TREE_MODELS: {
                "RandomForestClassifier": {
                    "n_estimators": optuna.distributions.IntDistribution(50, 500),
                    "max_depth": optuna.distributions.IntDistribution(3, 20),
                    "min_samples_split": optuna.distributions.IntDistribution(2, 20),
                    "min_samples_leaf": optuna.distributions.IntDistribution(1, 10)
                },
                "GradientBoostingClassifier": {
                    "n_estimators": optuna.distributions.IntDistribution(50, 300),
                    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.3),
                    "max_depth": optuna.distributions.IntDistribution(3, 10),
                    "subsample": optuna.distributions.FloatDistribution(0.6, 1.0)
                }
            },
            ModelFamily.SVM_MODELS: {
                "SVC": {
                    "C": optuna.distributions.FloatDistribution(0.1, 100.0, log=True),
                    "kernel": optuna.distributions.CategoricalDistribution(["rbf", "poly", "sigmoid"]),
                    "gamma": optuna.distributions.CategoricalDistribution(["scale", "auto"])
                }
            }
        }
    
    async def create_experiment(self,
                              name: str,
                              description: str,
                              config: AutoMLConfig) -> str:
        """Create a new AutoML experiment."""
        
        experiment = AutoMLExperiment(
            name=name,
            description=description,
            config=config
        )
        
        self.experiments[experiment.experiment_id] = experiment
        
        self.logger.info(
            "AutoML experiment created",
            experiment_id=experiment.experiment_id,
            name=name,
            task_type=config.task_type.value,
            max_trials=config.max_trials
        )
        
        return experiment.experiment_id
    
    async def start_experiment(self,
                             experiment_id: str,
                             X: pd.DataFrame,
                             y: pd.Series,
                             feature_names: List[str] = None) -> None:
        """Start AutoML experiment execution."""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status == "running":
            raise ValueError(f"Experiment {experiment_id} is already running")
        
        # Update experiment metadata
        experiment.dataset_shape = X.shape
        experiment.feature_names = feature_names or list(X.columns)
        experiment.target_name = y.name or "target"
        experiment.status = "running"
        experiment.started_at = datetime.utcnow()
        
        # Start experiment task
        task = asyncio.create_task(
            self._run_experiment(experiment, X, y)
        )
        self.active_experiments[experiment_id] = task
        
        self.logger.info(
            "AutoML experiment started",
            experiment_id=experiment_id,
            dataset_shape=X.shape,
            target_name=experiment.target_name
        )
    
    async def _run_experiment(self,
                            experiment: AutoMLExperiment,
                            X: pd.DataFrame,
                            y: pd.Series) -> None:
        """Run the complete AutoML experiment."""
        
        try:
            start_time = datetime.utcnow()
            
            # Data preprocessing
            X_processed, y_processed = await self._preprocess_data(
                X, y, experiment.config
            )
            
            # Feature engineering
            if experiment.config.include_feature_engineering:
                X_processed = await self._apply_feature_engineering(
                    X_processed, y_processed, experiment
                )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed,
                test_size=experiment.config.test_size,
                random_state=experiment.config.random_seed,
                stratify=y_processed if experiment.config.task_type in [
                    AutoMLTaskType.BINARY_CLASSIFICATION,
                    AutoMLTaskType.MULTICLASS_CLASSIFICATION
                ] else None
            )
            
            # Meta-learning: Get initial model recommendations
            if experiment.config.enable_meta_learning:
                recommended_models = await self.meta_learner.recommend_models(
                    X_train, y_train, experiment.config.task_type
                )
            else:
                recommended_models = self._get_default_models(experiment.config)
            
            # Create optimization study
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=experiment.config.random_seed)
            )
            self.hyperopt_studies[experiment.experiment_id] = study
            
            # Model search and optimization
            await self._search_models(
                experiment, study, X_train, y_train, X_test, y_test, recommended_models
            )
            
            # Neural Architecture Search (if enabled)
            if experiment.config.include_neural_architecture_search:
                await self._neural_architecture_search(
                    experiment, X_train, y_train, X_test, y_test
                )
            
            # Ensemble creation (if enabled)
            if experiment.config.enable_ensemble:
                await self._create_ensemble(experiment, X_train, y_train, X_test, y_test)
            
            # Final evaluation and selection
            await self._finalize_experiment(experiment, X_test, y_test)
            
            # Calculate total training time
            end_time = datetime.utcnow()
            experiment.total_training_time_hours = (end_time - start_time).total_seconds() / 3600
            
            experiment.status = "completed"
            experiment.completed_at = end_time
            
            self.logger.info(
                "AutoML experiment completed",
                experiment_id=experiment.experiment_id,
                best_score=experiment.best_candidate.get_score() if experiment.best_candidate else 0,
                total_candidates=len(experiment.candidates),
                training_time_hours=experiment.total_training_time_hours
            )
            
        except Exception as e:
            experiment.status = "failed"
            experiment.completed_at = datetime.utcnow()
            
            self.logger.error(
                "AutoML experiment failed",
                experiment_id=experiment.experiment_id,
                error=str(e)
            )
            raise
        
        finally:
            # Clean up
            if experiment.experiment_id in self.active_experiments:
                del self.active_experiments[experiment.experiment_id]
    
    async def _preprocess_data(self,
                             X: pd.DataFrame,
                             y: pd.Series,
                             config: AutoMLConfig) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data based on configuration."""
        
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Handle missing values
        if config.handle_missing_values:
            # Numeric columns: fill with median
            numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                X_processed[col].fillna(X_processed[col].median(), inplace=True)
            
            # Categorical columns: fill with mode
            categorical_columns = X_processed.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                X_processed[col].fillna(X_processed[col].mode()[0] if not X_processed[col].mode().empty else 'Unknown', inplace=True)
        
        # Handle categorical features
        if config.handle_categorical_features:
            categorical_columns = X_processed.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                if X_processed[col].nunique() > 50:  # High cardinality
                    # Use target encoding for high cardinality
                    X_processed[col] = self._target_encode(X_processed[col], y_processed)
                else:
                    # Use label encoding for low cardinality
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Handle target variable encoding
        if config.task_type in [AutoMLTaskType.BINARY_CLASSIFICATION, AutoMLTaskType.MULTICLASS_CLASSIFICATION]:
            if y_processed.dtype == 'object':
                le = LabelEncoder()
                y_processed = pd.Series(le.fit_transform(y_processed), index=y_processed.index)
        
        return X_processed, y_processed
    
    def _target_encode(self, categorical_series: pd.Series, target: pd.Series) -> pd.Series:
        """Apply target encoding to categorical series."""
        # Simple target encoding (in production, would use more sophisticated methods)
        mean_target = target.mean()
        encoding_map = target.groupby(categorical_series).mean()
        
        # Add smoothing to prevent overfitting
        smoothing_factor = 10
        counts = categorical_series.value_counts()
        
        for category in encoding_map.index:
            count = counts.get(category, 0)
            encoding_map[category] = (
                encoding_map[category] * count + mean_target * smoothing_factor
            ) / (count + smoothing_factor)
        
        return categorical_series.map(encoding_map).fillna(mean_target)
    
    async def _apply_feature_engineering(self,
                                       X: pd.DataFrame,
                                       y: pd.Series,
                                       experiment: AutoMLExperiment) -> pd.DataFrame:
        """Apply automated feature engineering."""
        
        X_engineered = await self.feature_engineering_pipelines.apply_transformations(
            X, y, experiment.config.task_type
        )
        
        # Feature selection
        if experiment.config.include_feature_selection:
            X_engineered = await self._select_features(X_engineered, y, experiment.config)
        
        return X_engineered
    
    async def _select_features(self,
                             X: pd.DataFrame,
                             y: pd.Series,
                             config: AutoMLConfig) -> pd.DataFrame:
        """Apply automated feature selection."""
        
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
        
        # Choose scoring function based on task type
        if config.task_type in [AutoMLTaskType.BINARY_CLASSIFICATION, AutoMLTaskType.MULTICLASS_CLASSIFICATION]:
            score_func = f_classif
        else:
            score_func = f_regression
        
        # Select top features (keep at most 50% of features)
        k = min(len(X.columns), max(10, len(X.columns) // 2))
        
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def _get_default_models(self, config: AutoMLConfig) -> List[str]:
        """Get default model recommendations."""
        
        if config.task_type == AutoMLTaskType.BINARY_CLASSIFICATION:
            return ["LogisticRegression", "RandomForestClassifier", "GradientBoostingClassifier"]
        elif config.task_type == AutoMLTaskType.MULTICLASS_CLASSIFICATION:
            return ["RandomForestClassifier", "GradientBoostingClassifier", "SVC"]
        elif config.task_type == AutoMLTaskType.REGRESSION:
            return ["RandomForestRegressor", "GradientBoostingRegressor", "LinearRegression"]
        else:
            return ["RandomForestClassifier"]
    
    async def _search_models(self,
                           experiment: AutoMLExperiment,
                           study: optuna.Study,
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_test: pd.DataFrame,
                           y_test: pd.Series,
                           recommended_models: List[str]) -> None:
        """Perform model search and hyperparameter optimization."""
        
        def objective(trial):
            # Select model
            model_name = trial.suggest_categorical("model", recommended_models)
            
            # Get hyperparameters for the selected model
            model_family = self._get_model_family(model_name)
            if model_family in self.model_search_spaces and model_name in self.model_search_spaces[model_family]:
                hyperparams = {}
                for param_name, distribution in self.model_search_spaces[model_family][model_name].items():
                    if isinstance(distribution, optuna.distributions.CategoricalDistribution):
                        hyperparams[param_name] = trial.suggest_categorical(param_name, distribution.choices)
                    elif isinstance(distribution, optuna.distributions.IntDistribution):
                        hyperparams[param_name] = trial.suggest_int(param_name, distribution.low, distribution.high)
                    elif isinstance(distribution, optuna.distributions.FloatDistribution):
                        hyperparams[param_name] = trial.suggest_float(
                            param_name, distribution.low, distribution.high, log=distribution.log
                        )
            else:
                hyperparams = {}
            
            # Create and train model
            model = self._create_model(model_name, hyperparams)
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=experiment.config.cross_validation_folds,
                    scoring=self._get_scoring_metric(experiment.config.optimization_objective),
                    n_jobs=1  # Avoid nested parallelism
                )
                
                mean_score = np.mean(cv_scores)
                
                # Create candidate
                candidate = ModelCandidate(
                    model_family=model_family,
                    algorithm_name=model_name,
                    hyperparameters=hyperparams,
                    cv_scores=cv_scores.tolist(),
                    mean_cv_score=mean_score,
                    std_cv_score=np.std(cv_scores),
                    status="completed"
                )
                
                # Train on full training set for final evaluation
                model.fit(X_train, y_train)
                candidate.trained_model = model
                
                # Test set evaluation
                y_pred = model.predict(X_test)
                test_score = self._calculate_score(
                    y_test, y_pred, experiment.config.optimization_objective
                )
                candidate.test_score = test_score
                
                # Add to experiment
                experiment.add_candidate(candidate)
                
                return mean_score
                
            except Exception as e:
                self.logger.warning(f"Model training failed: {str(e)}")
                return 0.0  # Return poor score for failed models
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=experiment.config.max_trials,
            timeout=experiment.config.max_training_time_hours * 3600
        )
    
    def _get_model_family(self, model_name: str) -> ModelFamily:
        """Get model family for a given model name."""
        if model_name in ["LogisticRegression", "LinearRegression", "Ridge", "Lasso"]:
            return ModelFamily.LINEAR_MODELS
        elif model_name in ["RandomForestClassifier", "RandomForestRegressor", "DecisionTreeClassifier"]:
            return ModelFamily.TREE_MODELS
        elif model_name in ["GradientBoostingClassifier", "GradientBoostingRegressor", "AdaBoostClassifier"]:
            return ModelFamily.ENSEMBLE_MODELS
        elif model_name in ["SVC", "SVR"]:
            return ModelFamily.SVM_MODELS
        else:
            return ModelFamily.LINEAR_MODELS
    
    def _create_model(self, model_name: str, hyperparams: Dict[str, Any]):
        """Create model instance with hyperparameters."""
        
        if model_name == "LogisticRegression":
            return LogisticRegression(random_state=42, **hyperparams)
        elif model_name == "RandomForestClassifier":
            return RandomForestClassifier(random_state=42, **hyperparams)
        elif model_name == "GradientBoostingClassifier":
            return GradientBoostingClassifier(random_state=42, **hyperparams)
        elif model_name == "SVC":
            return SVC(random_state=42, **hyperparams)
        else:
            # Default fallback
            return RandomForestClassifier(random_state=42)
    
    def _get_scoring_metric(self, objective: OptimizationObjective) -> str:
        """Get sklearn scoring metric string."""
        metric_map = {
            OptimizationObjective.ACCURACY: "accuracy",
            OptimizationObjective.PRECISION: "precision_weighted",
            OptimizationObjective.RECALL: "recall_weighted",
            OptimizationObjective.F1_SCORE: "f1_weighted",
            OptimizationObjective.ROC_AUC: "roc_auc",
            OptimizationObjective.MSE: "neg_mean_squared_error",
            OptimizationObjective.MAE: "neg_mean_absolute_error",
            OptimizationObjective.R2_SCORE: "r2"
        }
        return metric_map.get(objective, "accuracy")
    
    def _calculate_score(self, y_true, y_pred, objective: OptimizationObjective) -> float:
        """Calculate score for given objective."""
        
        if objective == OptimizationObjective.ACCURACY:
            return accuracy_score(y_true, y_pred)
        elif objective == OptimizationObjective.PRECISION:
            return precision_score(y_true, y_pred, average='weighted')
        elif objective == OptimizationObjective.RECALL:
            return recall_score(y_true, y_pred, average='weighted')
        elif objective == OptimizationObjective.F1_SCORE:
            return f1_score(y_true, y_pred, average='weighted')
        else:
            return accuracy_score(y_true, y_pred)
    
    async def _neural_architecture_search(self,
                                        experiment: AutoMLExperiment,
                                        X_train: pd.DataFrame,
                                        y_train: pd.Series,
                                        X_test: pd.DataFrame,
                                        y_test: pd.Series) -> None:
        """Perform Neural Architecture Search."""
        
        try:
            best_architecture = await self.nas_engine.search(
                X_train, y_train, X_test, y_test,
                experiment.config.task_type,
                max_trials=20,  # Limit NAS trials
                max_epochs=50
            )
            
            if best_architecture:
                candidate = ModelCandidate(
                    model_family=ModelFamily.NEURAL_NETWORKS,
                    algorithm_name="NAS_Neural_Network",
                    hyperparameters=best_architecture["architecture"],
                    mean_cv_score=best_architecture["score"],
                    status="completed"
                )
                
                experiment.add_candidate(candidate)
                
        except Exception as e:
            self.logger.warning(f"Neural Architecture Search failed: {str(e)}")
    
    async def _create_ensemble(self,
                             experiment: AutoMLExperiment,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_test: pd.DataFrame,
                             y_test: pd.Series) -> None:
        """Create ensemble from top performing models."""
        
        # Get top 5 models
        top_candidates = sorted(
            [c for c in experiment.candidates if c.status == "completed"],
            key=lambda c: c.get_score(),
            reverse=True
        )[:5]
        
        if len(top_candidates) < 2:
            return
        
        try:
            from sklearn.ensemble import VotingClassifier
            
            # Create voting ensemble
            estimators = []
            for i, candidate in enumerate(top_candidates):
                if candidate.trained_model is not None:
                    estimators.append((f"model_{i}", candidate.trained_model))
            
            if len(estimators) >= 2:
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
                ensemble.fit(X_train, y_train)
                
                # Evaluate ensemble
                y_pred = ensemble.predict(X_test)
                ensemble_score = self._calculate_score(
                    y_test, y_pred, experiment.config.optimization_objective
                )
                
                # Create ensemble candidate
                ensemble_candidate = ModelCandidate(
                    model_family=ModelFamily.ENSEMBLE_MODELS,
                    algorithm_name="Voting_Ensemble",
                    hyperparameters={"n_estimators": len(estimators)},
                    mean_cv_score=ensemble_score,
                    test_score=ensemble_score,
                    trained_model=ensemble,
                    status="completed"
                )
                
                experiment.add_candidate(ensemble_candidate)
                
        except Exception as e:
            self.logger.warning(f"Ensemble creation failed: {str(e)}")
    
    async def _finalize_experiment(self,
                                 experiment: AutoMLExperiment,
                                 X_test: pd.DataFrame,
                                 y_test: pd.Series) -> None:
        """Finalize experiment with summary and leaderboard."""
        
        # Generate leaderboard
        experiment.leaderboard = experiment.get_leaderboard()
        
        # Generate experiment summary
        completed_candidates = [c for c in experiment.candidates if c.status == "completed"]
        
        experiment.experiment_summary = {
            "total_candidates": len(experiment.candidates),
            "successful_candidates": len(completed_candidates),
            "best_score": experiment.best_candidate.get_score() if experiment.best_candidate else 0,
            "best_algorithm": experiment.best_candidate.algorithm_name if experiment.best_candidate else "None",
            "average_score": np.mean([c.get_score() for c in completed_candidates]) if completed_candidates else 0,
            "score_std": np.std([c.get_score() for c in completed_candidates]) if completed_candidates else 0,
            "model_family_distribution": self._get_model_family_distribution(completed_candidates)
        }
    
    def _get_model_family_distribution(self, candidates: List[ModelCandidate]) -> Dict[str, int]:
        """Get distribution of model families in candidates."""
        distribution = {}
        for candidate in candidates:
            family = candidate.model_family.value
            distribution[family] = distribution.get(family, 0) + 1
        return distribution
    
    async def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of an experiment."""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None,
            "total_candidates": len(experiment.candidates),
            "completed_candidates": len([c for c in experiment.candidates if c.status == "completed"]),
            "best_score": experiment.best_candidate.get_score() if experiment.best_candidate else 0,
            "best_algorithm": experiment.best_candidate.algorithm_name if experiment.best_candidate else "None",
            "training_time_hours": experiment.total_training_time_hours
        }
    
    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed results of an experiment."""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status,
            "config": {
                "task_type": experiment.config.task_type.value,
                "optimization_objective": experiment.config.optimization_objective.value,
                "max_trials": experiment.config.max_trials,
                "max_training_time_hours": experiment.config.max_training_time_hours
            },
            "data_info": {
                "dataset_shape": experiment.dataset_shape,
                "feature_names": experiment.feature_names,
                "target_name": experiment.target_name
            },
            "results": {
                "leaderboard": experiment.leaderboard,
                "summary": experiment.experiment_summary,
                "best_candidate": {
                    "algorithm": experiment.best_candidate.algorithm_name if experiment.best_candidate else None,
                    "score": experiment.best_candidate.get_score() if experiment.best_candidate else 0,
                    "hyperparameters": experiment.best_candidate.hyperparameters if experiment.best_candidate else {}
                }
            },
            "execution_info": {
                "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
                "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None,
                "total_training_time_hours": experiment.total_training_time_hours
            }
        }
    
    async def deploy_best_model(self, experiment_id: str) -> str:
        """Deploy the best model from an experiment."""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if not experiment.best_candidate or not experiment.best_candidate.trained_model:
            raise ValueError("No trained model available to deploy")
        
        # Create model entity
        model = Model(
            name=f"automl_{experiment.name}_{experiment.best_candidate.algorithm_name}",
            version="1.0.0",
            model_type=experiment.best_candidate.algorithm_name,
            framework="scikit-learn",
            status=ModelStatus.TRAINED,
            trained_model=experiment.best_candidate.trained_model,
            hyperparameters=experiment.best_candidate.hyperparameters,
            metrics=experiment.best_candidate.validation_metrics,
            feature_names=experiment.feature_names
        )
        
        # In production, would integrate with model registry and deployment system
        model_id = str(uuid.uuid4())
        
        self.logger.info(
            "AutoML model deployed",
            experiment_id=experiment_id,
            model_id=model_id,
            algorithm=experiment.best_candidate.algorithm_name,
            score=experiment.best_candidate.get_score()
        )
        
        return model_id


class FeatureEngineeringPipelines:
    """Automated feature engineering pipelines."""
    
    async def apply_transformations(self,
                                  X: pd.DataFrame,
                                  y: pd.Series,
                                  task_type: AutoMLTaskType) -> pd.DataFrame:
        """Apply automated feature engineering transformations."""
        
        X_transformed = X.copy()
        
        # Numeric feature transformations
        numeric_columns = X_transformed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Log transformation for skewed features
            if X_transformed[col].min() > 0:  # Only for positive values
                skewness = X_transformed[col].skew()
                if abs(skewness) > 1:  # Highly skewed
                    X_transformed[f"{col}_log"] = np.log1p(X_transformed[col])
            
            # Polynomial features for potentially non-linear relationships
            X_transformed[f"{col}_squared"] = X_transformed[col] ** 2
            
            # Binning for continuous variables
            X_transformed[f"{col}_binned"] = pd.cut(
                X_transformed[col], bins=5, labels=False
            )
        
        # Interaction features between top correlated features
        correlations = X_transformed[numeric_columns].corr().abs()
        
        # Find top correlated pairs
        for i, col1 in enumerate(correlations.columns):
            for j, col2 in enumerate(correlations.columns):
                if i < j and correlations.loc[col1, col2] > 0.5:
                    X_transformed[f"{col1}_x_{col2}"] = X_transformed[col1] * X_transformed[col2]
        
        return X_transformed


class NeuralArchitectureSearch:
    """Neural Architecture Search for automated neural network design."""
    
    async def search(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    task_type: AutoMLTaskType,
                    max_trials: int = 20,
                    max_epochs: int = 50) -> Optional[Dict[str, Any]]:
        """Perform neural architecture search."""
        
        try:
            # This is a simplified NAS implementation
            # In production, would use more sophisticated NAS algorithms like DARTS, ENAS, etc.
            
            best_architecture = None
            best_score = 0.0
            
            for trial in range(max_trials):
                # Generate random architecture
                architecture = self._generate_random_architecture(X_train.shape[1], task_type)
                
                # Train and evaluate
                score = await self._evaluate_architecture(
                    architecture, X_train, y_train, X_test, y_test, max_epochs
                )
                
                if score > best_score:
                    best_score = score
                    best_architecture = {
                        "architecture": architecture,
                        "score": score
                    }
            
            return best_architecture
            
        except Exception as e:
            logging.warning(f"NAS failed: {str(e)}")
            return None
    
    def _generate_random_architecture(self, input_size: int, task_type: AutoMLTaskType) -> Dict[str, Any]:
        """Generate a random neural network architecture."""
        
        # Random number of layers
        n_layers = random.randint(1, 4)
        
        # Random layer sizes
        layer_sizes = []
        current_size = input_size
        
        for _ in range(n_layers):
            layer_size = random.choice([32, 64, 128, 256])
            layer_sizes.append(layer_size)
            current_size = layer_size
        
        # Output layer size
        if task_type == AutoMLTaskType.BINARY_CLASSIFICATION:
            output_size = 1
        elif task_type == AutoMLTaskType.MULTICLASS_CLASSIFICATION:
            output_size = 10  # Assume max 10 classes for simplicity
        else:
            output_size = 1  # Regression
        
        return {
            "layer_sizes": layer_sizes,
            "output_size": output_size,
            "activation": random.choice(["relu", "tanh"]),
            "dropout_rate": random.uniform(0.1, 0.5),
            "learning_rate": random.uniform(0.001, 0.1)
        }
    
    async def _evaluate_architecture(self,
                                   architecture: Dict[str, Any],
                                   X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   X_test: pd.DataFrame,
                                   y_test: pd.Series,
                                   max_epochs: int) -> float:
        """Evaluate a neural network architecture."""
        
        try:
            # This would implement actual neural network training
            # For now, return a mock score
            return random.uniform(0.6, 0.9)
            
        except Exception:
            return 0.0


class MetaLearner:
    """Meta-learning system for model recommendations."""
    
    async def recommend_models(self,
                             X: pd.DataFrame,
                             y: pd.Series,
                             task_type: AutoMLTaskType) -> List[str]:
        """Recommend models based on dataset characteristics."""
        
        # Extract dataset meta-features
        meta_features = self._extract_meta_features(X, y)
        
        # Simple rule-based recommendations
        # In production, would use trained meta-models
        
        n_samples, n_features = X.shape
        
        recommendations = []
        
        if n_samples > 10000 and n_features < 100:
            # Large dataset, few features - linear models work well
            recommendations.extend(["LogisticRegression", "LinearRegression"])
        
        if n_features > n_samples:
            # High dimensional - regularized models
            recommendations.extend(["Ridge", "Lasso"])
        
        # Always include tree-based models
        recommendations.extend(["RandomForestClassifier", "GradientBoostingClassifier"])
        
        # Remove duplicates and ensure we have at least some models
        recommendations = list(set(recommendations))
        if not recommendations:
            recommendations = ["RandomForestClassifier", "LogisticRegression"]
        
        return recommendations
    
    def _extract_meta_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Extract meta-features from dataset."""
        
        n_samples, n_features = X.shape
        
        meta_features = {
            "n_samples": n_samples,
            "n_features": n_features,
            "ratio_samples_features": n_samples / n_features,
            "n_numeric_features": len(X.select_dtypes(include=[np.number]).columns),
            "n_categorical_features": len(X.select_dtypes(include=['object']).columns),
            "missing_value_ratio": X.isnull().sum().sum() / (n_samples * n_features),
            "target_entropy": self._calculate_entropy(y) if y.dtype == 'object' else 0,
            "class_imbalance": self._calculate_class_imbalance(y) if y.dtype == 'object' else 0
        }
        
        return meta_features
    
    def _calculate_entropy(self, y: pd.Series) -> float:
        """Calculate entropy of target variable."""
        value_counts = y.value_counts(normalize=True)
        entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
        return entropy
    
    def _calculate_class_imbalance(self, y: pd.Series) -> float:
        """Calculate class imbalance ratio."""
        value_counts = y.value_counts()
        if len(value_counts) < 2:
            return 0.0
        
        majority_class = value_counts.max()
        minority_class = value_counts.min()
        
        return majority_class / minority_class
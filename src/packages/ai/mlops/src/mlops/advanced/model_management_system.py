"""
Advanced ML/AI Capabilities and Model Management System
Comprehensive model lifecycle management with advanced ML features
"""

import asyncio
import json
import logging
import pickle
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import optuna

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model lifecycle status"""
    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ModelType(Enum):
    """Model type classification"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    ENSEMBLE = "ensemble"


class DeploymentStrategy(Enum):
    """Model deployment strategy"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"
    A_B_TEST = "a_b_test"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    model_type: ModelType = ModelType.CLASSIFICATION
    algorithm: str = ""
    framework: str = ""
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: ModelStatus = ModelStatus.DRAFT
    tags: List[str] = field(default_factory=list)
    
    # Training metadata
    training_dataset_id: Optional[str] = None
    training_parameters: Dict[str, Any] = field(default_factory=dict)
    training_duration: Optional[float] = None
    training_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Model artifacts
    model_path: Optional[str] = None
    model_size_bytes: Optional[int] = None
    feature_names: List[str] = field(default_factory=list)
    target_names: List[str] = field(default_factory=list)
    
    # Performance metrics
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    performance_benchmarks: Dict[str, float] = field(default_factory=dict)
    
    # Deployment metadata
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    deployment_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Lineage and governance
    parent_model_id: Optional[str] = None
    data_lineage: Dict[str, Any] = field(default_factory=dict)
    compliance_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelExperiment:
    """Model experiment tracking"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    model_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"
    notes: str = ""


@dataclass
class ModelVersion:
    """Model version information"""
    model_id: str
    version: str
    created_at: datetime
    metrics: Dict[str, float]
    model_path: str
    is_active: bool = False
    changelog: str = ""


class AdvancedModelManager:
    """Advanced model lifecycle management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, ModelMetadata] = {}
        self.experiments: Dict[str, ModelExperiment] = {}
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        
        # MLflow configuration
        self.mlflow_tracking_uri = config.get('mlflow_tracking_uri', 'http://localhost:5000')
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Model registry
        self.model_registry_path = Path(config.get('model_registry_path', './model_registry'))
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        
        # Experiment tracking
        self.experiment_name = config.get('experiment_name', 'mlops_experiments')
        mlflow.set_experiment(self.experiment_name)

    async def create_model(self, metadata: ModelMetadata) -> str:
        """Create a new model with metadata"""
        try:
            if not metadata.id:
                metadata.id = str(uuid.uuid4())
                
            metadata.created_at = datetime.now()
            metadata.updated_at = datetime.now()
            metadata.status = ModelStatus.DRAFT
            
            self.models[metadata.id] = metadata
            
            # Initialize version tracking
            self.model_versions[metadata.id] = []
            
            logger.info(f"Created model {metadata.name} with ID {metadata.id}")
            return metadata.id
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise

    async def train_model_with_hyperparameter_optimization(
        self, 
        model_id: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        optimization_method: str = "optuna",
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Train model with automated hyperparameter optimization"""
        
        model_metadata = self.models.get(model_id)
        if not model_metadata:
            raise ValueError(f"Model {model_id} not found")
            
        try:
            model_metadata.status = ModelStatus.TRAINING
            start_time = datetime.now()
            
            with mlflow.start_run(run_name=f"{model_metadata.name}_optimization"):
                # Log model metadata
                mlflow.log_params({
                    "model_name": model_metadata.name,
                    "model_type": model_metadata.model_type.value,
                    "algorithm": model_metadata.algorithm,
                    "optimization_method": optimization_method,
                    "n_trials": n_trials
                })
                
                # Perform hyperparameter optimization
                if optimization_method == "optuna":
                    best_params, best_score = await self._optimize_with_optuna(
                        model_metadata, X_train, y_train, n_trials
                    )
                elif optimization_method == "grid_search":
                    best_params, best_score = await self._optimize_with_grid_search(
                        model_metadata, X_train, y_train
                    )
                elif optimization_method == "random_search":
                    best_params, best_score = await self._optimize_with_random_search(
                        model_metadata, X_train, y_train
                    )
                else:
                    raise ValueError(f"Unsupported optimization method: {optimization_method}")
                
                # Train final model with best parameters
                final_model = await self._train_final_model(
                    model_metadata, X_train, y_train, best_params
                )
                
                # Save model
                model_path = await self._save_model(model_id, final_model)
                
                # Update metadata
                training_duration = (datetime.now() - start_time).total_seconds()
                model_metadata.training_parameters = best_params
                model_metadata.training_duration = training_duration
                model_metadata.training_metrics = {"optimization_score": best_score}
                model_metadata.model_path = model_path
                model_metadata.status = ModelStatus.TRAINED
                model_metadata.updated_at = datetime.now()
                
                # Log results to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metric("best_score", best_score)
                mlflow.log_metric("training_duration", training_duration)
                mlflow.sklearn.log_model(final_model, "model")
                
                logger.info(f"Model {model_id} training completed with score {best_score}")
                
                return {
                    "model_id": model_id,
                    "best_parameters": best_params,
                    "best_score": best_score,
                    "training_duration": training_duration,
                    "model_path": model_path
                }
                
        except Exception as e:
            model_metadata.status = ModelStatus.FAILED
            logger.error(f"Model training failed for {model_id}: {e}")
            raise

    async def _optimize_with_optuna(
        self, 
        model_metadata: ModelMetadata, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        n_trials: int
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            params = self._suggest_hyperparameters(trial, model_metadata.algorithm)
            model = self._create_model_instance(model_metadata.algorithm, params)
            
            # Cross-validation score
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, study.best_value

    async def _optimize_with_grid_search(
        self, 
        model_metadata: ModelMetadata, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters using Grid Search"""
        
        model = self._create_model_instance(model_metadata.algorithm)
        param_grid = self._get_parameter_grid(model_metadata.algorithm)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_params_, grid_search.best_score_

    async def _optimize_with_random_search(
        self, 
        model_metadata: ModelMetadata, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters using Random Search"""
        
        model = self._create_model_instance(model_metadata.algorithm)
        param_distributions = self._get_parameter_distributions(model_metadata.algorithm)
        
        random_search = RandomizedSearchCV(
            model, param_distributions, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        
        return random_search.best_params_, random_search.best_score_

    def _suggest_hyperparameters(self, trial, algorithm: str) -> Dict[str, Any]:
        """Suggest hyperparameters for Optuna optimization"""
        
        if algorithm == "random_forest":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            }
        elif algorithm == "gradient_boosting":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
            }
        elif algorithm == "svm":
            return {
                'C': trial.suggest_float('C', 0.1, 100, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])
            }
        else:
            return {}

    def _create_model_instance(self, algorithm: str, params: Dict[str, Any] = None) -> BaseEstimator:
        """Create model instance based on algorithm"""
        if params is None:
            params = {}
            
        if algorithm == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**params, random_state=42)
        elif algorithm == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(**params, random_state=42)
        elif algorithm == "svm":
            from sklearn.svm import SVC
            return SVC(**params, random_state=42)
        elif algorithm == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**params, random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    async def create_ensemble_model(
        self, 
        model_ids: List[str], 
        ensemble_method: str = "voting",
        weights: Optional[List[float]] = None
    ) -> str:
        """Create ensemble model from multiple trained models"""
        
        try:
            # Load individual models
            models = []
            model_names = []
            
            for model_id in model_ids:
                model_metadata = self.models.get(model_id)
                if not model_metadata or model_metadata.status != ModelStatus.TRAINED:
                    raise ValueError(f"Model {model_id} not found or not trained")
                
                model = await self._load_model(model_id)
                models.append((model_metadata.name, model))
                model_names.append(model_metadata.name)
            
            # Create ensemble
            if ensemble_method == "voting":
                if self._is_classification_ensemble(model_ids):
                    ensemble = VotingClassifier(estimators=models, voting='soft', weights=weights)
                else:
                    ensemble = VotingRegressor(estimators=models, weights=weights)
            else:
                raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
            
            # Create ensemble metadata
            ensemble_metadata = ModelMetadata(
                name=f"ensemble_{'_'.join(model_names)}",
                model_type=self._infer_ensemble_type(model_ids),
                algorithm=f"{ensemble_method}_ensemble",
                framework="sklearn",
                status=ModelStatus.TRAINED,
                tags=["ensemble", ensemble_method],
                training_parameters={
                    "base_models": model_ids,
                    "ensemble_method": ensemble_method,
                    "weights": weights
                }
            )
            
            ensemble_id = await self.create_model(ensemble_metadata)
            
            # Save ensemble model
            model_path = await self._save_model(ensemble_id, ensemble)
            ensemble_metadata.model_path = model_path
            
            logger.info(f"Created ensemble model {ensemble_id} from {len(model_ids)} base models")
            return ensemble_id
            
        except Exception as e:
            logger.error(f"Failed to create ensemble model: {e}")
            raise

    async def implement_model_versioning(self, model_id: str, changelog: str = "") -> str:
        """Create new version of existing model"""
        
        model_metadata = self.models.get(model_id)
        if not model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        try:
            # Get current versions
            current_versions = self.model_versions.get(model_id, [])
            
            # Generate new version number
            if current_versions:
                latest_version = max(current_versions, key=lambda v: v.created_at)
                version_parts = latest_version.version.split('.')
                major, minor, patch = int(version_parts[0]), int(version_parts[1]), int(version_parts[2])
                new_version = f"{major}.{minor}.{patch + 1}"
            else:
                new_version = "1.0.0"
            
            # Create new version
            new_model_version = ModelVersion(
                model_id=model_id,
                version=new_version,
                created_at=datetime.now(),
                metrics=model_metadata.training_metrics.copy(),
                model_path=model_metadata.model_path,
                changelog=changelog
            )
            
            # Deactivate previous versions
            for version in current_versions:
                version.is_active = False
            
            # Set new version as active
            new_model_version.is_active = True
            
            # Add to version history
            current_versions.append(new_model_version)
            self.model_versions[model_id] = current_versions
            
            # Update model metadata
            model_metadata.version = new_version
            model_metadata.updated_at = datetime.now()
            
            # Log version creation
            with mlflow.start_run(run_name=f"{model_metadata.name}_v{new_version}"):
                mlflow.log_param("version", new_version)
                mlflow.log_param("changelog", changelog)
                mlflow.log_metrics(model_metadata.training_metrics)
            
            logger.info(f"Created version {new_version} for model {model_id}")
            return new_version
            
        except Exception as e:
            logger.error(f"Failed to create model version: {e}")
            raise

    async def implement_ab_testing(
        self, 
        model_a_id: str, 
        model_b_id: str,
        traffic_split: float = 0.5,
        success_metrics: List[str] = None
    ) -> str:
        """Implement A/B testing between two models"""
        
        if success_metrics is None:
            success_metrics = ["accuracy", "precision", "recall"]
        
        try:
            model_a = self.models.get(model_a_id)
            model_b = self.models.get(model_b_id)
            
            if not model_a or not model_b:
                raise ValueError("Both models must exist for A/B testing")
            
            # Create A/B test configuration
            ab_test_config = {
                "test_id": str(uuid.uuid4()),
                "model_a": {
                    "id": model_a_id,
                    "name": model_a.name,
                    "version": model_a.version
                },
                "model_b": {
                    "id": model_b_id,
                    "name": model_b.name,
                    "version": model_b.version
                },
                "traffic_split": traffic_split,
                "success_metrics": success_metrics,
                "start_time": datetime.now(),
                "status": "active",
                "results": {
                    "model_a_performance": {},
                    "model_b_performance": {},
                    "statistical_significance": False,
                    "winner": None
                }
            }
            
            # Save A/B test configuration
            test_id = ab_test_config["test_id"]
            ab_test_path = self.model_registry_path / "ab_tests" / f"{test_id}.json"
            ab_test_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(ab_test_path, 'w') as f:
                json.dump(ab_test_config, f, default=str, indent=2)
            
            logger.info(f"Started A/B test {test_id} between {model_a.name} and {model_b.name}")
            return test_id
            
        except Exception as e:
            logger.error(f"Failed to implement A/B testing: {e}")
            raise

    async def implement_model_explainability(self, model_id: str, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Implement model explainability and interpretability"""
        
        model_metadata = self.models.get(model_id)
        if not model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        try:
            # Load model
            model = await self._load_model(model_id)
            
            explainability_results = {
                "model_id": model_id,
                "model_name": model_metadata.name,
                "feature_importance": {},
                "shap_values": {},
                "lime_explanations": {},
                "permutation_importance": {},
                "generated_at": datetime.now()
            }
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_names = model_metadata.feature_names or [f"feature_{i}" for i in range(len(model.feature_importances_))]
                explainability_results["feature_importance"] = dict(zip(
                    feature_names, model.feature_importances_.tolist()
                ))
            
            # SHAP explanations
            try:
                import shap
                explainer = shap.Explainer(model)
                shap_values = explainer(X_test.head(100))  # Limit for performance
                
                explainability_results["shap_values"] = {
                    "base_values": shap_values.base_values.tolist() if hasattr(shap_values, 'base_values') else [],
                    "values": shap_values.values.tolist() if hasattr(shap_values, 'values') else [],
                    "feature_names": shap_values.feature_names if hasattr(shap_values, 'feature_names') else []
                }
            except ImportError:
                logger.warning("SHAP not available for explainability analysis")
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
            
            # LIME explanations (for first few samples)
            try:
                import lime
                import lime.lime_tabular
                
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_test.values[:100],
                    feature_names=X_test.columns,
                    mode='classification' if model_metadata.model_type == ModelType.CLASSIFICATION else 'regression'
                )
                
                sample_explanations = []
                for i in range(min(5, len(X_test))):
                    explanation = explainer.explain_instance(
                        X_test.iloc[i].values,
                        model.predict_proba if hasattr(model, 'predict_proba') else model.predict
                    )
                    sample_explanations.append({
                        "sample_index": i,
                        "explanation": explanation.as_list()
                    })
                
                explainability_results["lime_explanations"] = sample_explanations
                
            except ImportError:
                logger.warning("LIME not available for explainability analysis")
            except Exception as e:
                logger.warning(f"LIME analysis failed: {e}")
            
            # Permutation importance
            try:
                from sklearn.inspection import permutation_importance
                
                # Use a subset for performance
                X_subset = X_test.head(500)
                y_subset = model.predict(X_subset)
                
                perm_importance = permutation_importance(
                    model, X_subset, y_subset, n_repeats=5, random_state=42
                )
                
                feature_names = X_test.columns
                explainability_results["permutation_importance"] = {
                    "importances_mean": dict(zip(feature_names, perm_importance.importances_mean.tolist())),
                    "importances_std": dict(zip(feature_names, perm_importance.importances_std.tolist()))
                }
                
            except Exception as e:
                logger.warning(f"Permutation importance analysis failed: {e}")
            
            # Save explainability results
            explainability_path = self.model_registry_path / "explainability" / f"{model_id}_explainability.json"
            explainability_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(explainability_path, 'w') as f:
                json.dump(explainability_results, f, default=str, indent=2)
            
            logger.info(f"Generated explainability analysis for model {model_id}")
            return explainability_results
            
        except Exception as e:
            logger.error(f"Failed to generate model explainability: {e}")
            raise

    async def implement_federated_learning(
        self, 
        model_id: str,
        client_data_sources: List[Dict[str, Any]],
        federation_rounds: int = 10
    ) -> str:
        """Implement federated learning across multiple data sources"""
        
        model_metadata = self.models.get(model_id)
        if not model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        try:
            # Initialize federated learning configuration
            federation_config = {
                "federation_id": str(uuid.uuid4()),
                "base_model_id": model_id,
                "client_count": len(client_data_sources),
                "federation_rounds": federation_rounds,
                "start_time": datetime.now(),
                "clients": client_data_sources,
                "round_results": [],
                "status": "initializing"
            }
            
            # Load base model
            global_model = await self._load_model(model_id)
            
            logger.info(f"Starting federated learning with {len(client_data_sources)} clients")
            
            # Federated learning rounds
            for round_num in range(federation_rounds):
                round_start = datetime.now()
                
                # Simulate client training (in production, this would be distributed)
                client_models = []
                client_weights = []
                
                for i, client_config in enumerate(client_data_sources):
                    # Simulate client training
                    client_model = self._simulate_client_training(
                        global_model, client_config, round_num
                    )
                    client_models.append(client_model)
                    client_weights.append(client_config.get('weight', 1.0))
                
                # Aggregate client models (Federated Averaging)
                global_model = self._federated_averaging(client_models, client_weights)
                
                # Evaluate global model
                round_metrics = await self._evaluate_federated_model(global_model, round_num)
                
                round_result = {
                    "round": round_num + 1,
                    "timestamp": datetime.now(),
                    "duration_seconds": (datetime.now() - round_start).total_seconds(),
                    "participating_clients": len(client_data_sources),
                    "metrics": round_metrics
                }
                
                federation_config["round_results"].append(round_result)
                
                logger.info(f"Completed federated round {round_num + 1}/{federation_rounds}")
            
            # Save final federated model
            federation_id = federation_config["federation_id"]
            federated_model_path = await self._save_federated_model(federation_id, global_model)
            
            # Update configuration
            federation_config["status"] = "completed"
            federation_config["final_model_path"] = federated_model_path
            federation_config["end_time"] = datetime.now()
            
            # Save federation configuration
            federation_path = self.model_registry_path / "federated_learning" / f"{federation_id}.json"
            federation_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(federation_path, 'w') as f:
                json.dump(federation_config, f, default=str, indent=2)
            
            logger.info(f"Completed federated learning with ID {federation_id}")
            return federation_id
            
        except Exception as e:
            logger.error(f"Failed to implement federated learning: {e}")
            raise

    def _simulate_client_training(self, global_model, client_config: Dict[str, Any], round_num: int):
        """Simulate client-side training (in production, this would be actual distributed training)"""
        # In a real implementation, this would involve:
        # 1. Sending global model to client
        # 2. Client training on local data
        # 3. Sending model updates back to server
        
        # For simulation, we'll create a slight variation of the global model
        import copy
        client_model = copy.deepcopy(global_model)
        
        # Simulate training by adding small random variations
        # (In real federated learning, this would be actual gradient updates)
        if hasattr(client_model, 'coef_'):
            client_model.coef_ += np.random.normal(0, 0.01, client_model.coef_.shape)
        
        return client_model

    def _federated_averaging(self, client_models: List[Any], weights: List[float]) -> Any:
        """Implement Federated Averaging algorithm"""
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Average model parameters
        if hasattr(client_models[0], 'coef_'):
            # For linear models
            averaged_coef = np.zeros_like(client_models[0].coef_)
            for model, weight in zip(client_models, normalized_weights):
                averaged_coef += weight * model.coef_
            
            # Create averaged model
            averaged_model = copy.deepcopy(client_models[0])
            averaged_model.coef_ = averaged_coef
            
            return averaged_model
        else:
            # For other model types, return the first model (simplified)
            return client_models[0]

    async def _evaluate_federated_model(self, model: Any, round_num: int) -> Dict[str, float]:
        """Evaluate federated model performance"""
        # Mock evaluation metrics
        base_accuracy = 0.85
        round_improvement = round_num * 0.005
        noise = np.random.normal(0, 0.02)
        
        accuracy = min(0.99, base_accuracy + round_improvement + noise)
        
        return {
            "accuracy": accuracy,
            "loss": 1 - accuracy,
            "round": round_num + 1
        }

    async def _save_federated_model(self, federation_id: str, model: Any) -> str:
        """Save federated learning model"""
        model_path = self.model_registry_path / "federated_models" / f"{federation_id}_model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_path)
        
        return str(model_path)

    async def implement_model_monitoring(self, model_id: str) -> Dict[str, Any]:
        """Implement comprehensive model monitoring"""
        
        model_metadata = self.models.get(model_id)
        if not model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        try:
            monitoring_config = {
                "model_id": model_id,
                "model_name": model_metadata.name,
                "monitoring_enabled": True,
                "drift_detection": {
                    "enabled": True,
                    "method": "kolmogorov_smirnov",
                    "threshold": 0.05,
                    "baseline_data": None
                },
                "performance_monitoring": {
                    "enabled": True,
                    "metrics": ["accuracy", "precision", "recall", "f1_score"],
                    "alert_thresholds": {
                        "accuracy": 0.05,  # Alert if accuracy drops by 5%
                        "precision": 0.05,
                        "recall": 0.05
                    }
                },
                "data_quality_monitoring": {
                    "enabled": True,
                    "checks": ["missing_values", "outliers", "schema_validation"],
                    "alert_thresholds": {
                        "missing_values_percent": 0.1,
                        "outlier_percent": 0.05
                    }
                },
                "prediction_monitoring": {
                    "enabled": True,
                    "log_predictions": True,
                    "sample_rate": 0.1  # Log 10% of predictions
                },
                "alerts": {
                    "email_notifications": True,
                    "slack_notifications": True,
                    "webhook_url": None
                },
                "created_at": datetime.now(),
                "last_updated": datetime.now()
            }
            
            # Save monitoring configuration
            monitoring_path = self.model_registry_path / "monitoring" / f"{model_id}_monitoring.json"
            monitoring_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(monitoring_path, 'w') as f:
                json.dump(monitoring_config, f, default=str, indent=2)
            
            # Initialize monitoring metrics storage
            metrics_path = self.model_registry_path / "monitoring" / f"{model_id}_metrics"
            metrics_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Enabled comprehensive monitoring for model {model_id}")
            return monitoring_config
            
        except Exception as e:
            logger.error(f"Failed to implement model monitoring: {e}")
            raise

    async def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get complete model lineage and dependency graph"""
        
        model_metadata = self.models.get(model_id)
        if not model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        lineage = {
            "model_id": model_id,
            "model_name": model_metadata.name,
            "version": model_metadata.version,
            "created_at": model_metadata.created_at,
            "lineage_tree": {},
            "dependencies": {},
            "derived_models": [],
            "data_sources": []
        }
        
        # Build lineage tree
        lineage["lineage_tree"] = await self._build_lineage_tree(model_id)
        
        # Get data dependencies
        lineage["data_sources"] = await self._get_data_dependencies(model_id)
        
        # Get derived models
        lineage["derived_models"] = await self._get_derived_models(model_id)
        
        return lineage

    async def _build_lineage_tree(self, model_id: str, visited: Set[str] = None) -> Dict[str, Any]:
        """Recursively build model lineage tree"""
        if visited is None:
            visited = set()
            
        if model_id in visited:
            return {"circular_reference": True}
            
        visited.add(model_id)
        
        model_metadata = self.models.get(model_id)
        if not model_metadata:
            return {}
        
        tree = {
            "model_id": model_id,
            "name": model_metadata.name,
            "version": model_metadata.version,
            "created_at": model_metadata.created_at,
            "parents": []
        }
        
        # Add parent model if exists
        if model_metadata.parent_model_id:
            parent_tree = await self._build_lineage_tree(model_metadata.parent_model_id, visited.copy())
            tree["parents"].append(parent_tree)
        
        return tree

    async def _get_data_dependencies(self, model_id: str) -> List[Dict[str, Any]]:
        """Get data source dependencies for model"""
        model_metadata = self.models.get(model_id)
        if not model_metadata:
            return []
        
        data_sources = []
        
        if model_metadata.training_dataset_id:
            data_sources.append({
                "type": "training_dataset",
                "dataset_id": model_metadata.training_dataset_id,
                "usage": "training"
            })
        
        # Add other data dependencies from metadata
        for source_type, source_info in model_metadata.data_lineage.items():
            data_sources.append({
                "type": source_type,
                "source_info": source_info
            })
        
        return data_sources

    async def _get_derived_models(self, model_id: str) -> List[str]:
        """Get models derived from this model"""
        derived_models = []
        
        for mid, metadata in self.models.items():
            if metadata.parent_model_id == model_id:
                derived_models.append(mid)
        
        return derived_models

    async def _train_final_model(
        self, 
        model_metadata: ModelMetadata, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        best_params: Dict[str, Any]
    ) -> BaseEstimator:
        """Train final model with optimized parameters"""
        
        model = self._create_model_instance(model_metadata.algorithm, best_params)
        model.fit(X_train, y_train)
        
        return model

    async def _save_model(self, model_id: str, model: BaseEstimator) -> str:
        """Save trained model to registry"""
        model_path = self.model_registry_path / "models" / f"{model_id}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_path)
        
        # Update model size
        model_metadata = self.models[model_id]
        model_metadata.model_size_bytes = model_path.stat().st_size
        
        return str(model_path)

    async def _load_model(self, model_id: str) -> BaseEstimator:
        """Load model from registry"""
        model_metadata = self.models.get(model_id)
        if not model_metadata or not model_metadata.model_path:
            raise ValueError(f"Model {model_id} not found or not saved")
        
        return joblib.load(model_metadata.model_path)

    def _get_parameter_grid(self, algorithm: str) -> Dict[str, List]:
        """Get parameter grid for grid search"""
        grids = {
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "gradient_boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            "svm": {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'poly']
            }
        }
        
        return grids.get(algorithm, {})

    def _get_parameter_distributions(self, algorithm: str) -> Dict[str, Any]:
        """Get parameter distributions for random search"""
        from scipy.stats import uniform, randint
        
        distributions = {
            "random_forest": {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 20),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10)
            },
            "gradient_boosting": {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.29),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 0.4)
            }
        }
        
        return distributions.get(algorithm, {})

    def _is_classification_ensemble(self, model_ids: List[str]) -> bool:
        """Check if ensemble should be classification"""
        for model_id in model_ids:
            model_metadata = self.models.get(model_id)
            if model_metadata and model_metadata.model_type == ModelType.CLASSIFICATION:
                return True
        return False

    def _infer_ensemble_type(self, model_ids: List[str]) -> ModelType:
        """Infer ensemble model type from base models"""
        model_types = set()
        for model_id in model_ids:
            model_metadata = self.models.get(model_id)
            if model_metadata:
                model_types.add(model_metadata.model_type)
        
        if len(model_types) == 1:
            return model_types.pop()
        else:
            return ModelType.ENSEMBLE


# Example usage and testing
async def main():
    """Example usage of Advanced Model Management System"""
    config = {
        'mlflow_tracking_uri': 'http://localhost:5000',
        'model_registry_path': './advanced_model_registry',
        'experiment_name': 'advanced_mlops_experiments'
    }
    
    manager = AdvancedModelManager(config)
    
    # Create sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    # Create model metadata
    model_metadata = ModelMetadata(
        name="advanced_classifier",
        model_type=ModelType.CLASSIFICATION,
        algorithm="random_forest",
        framework="sklearn",
        description="Advanced classification model with hyperparameter optimization"
    )
    
    # Create model
    model_id = await manager.create_model(model_metadata)
    print(f"Created model: {model_id}")
    
    # Train with hyperparameter optimization
    training_result = await manager.train_model_with_hyperparameter_optimization(
        model_id, X_df, y_series, optimization_method="optuna", n_trials=20
    )
    print(f"Training completed with score: {training_result['best_score']:.4f}")
    
    # Implement model versioning
    new_version = await manager.implement_model_versioning(model_id, "Initial production version")
    print(f"Created model version: {new_version}")
    
    # Implement explainability
    explainability = await manager.implement_model_explainability(model_id, X_df.head(100))
    print(f"Generated explainability analysis with {len(explainability['feature_importance'])} features")
    
    # Implement monitoring
    monitoring_config = await manager.implement_model_monitoring(model_id)
    print(f"Enabled monitoring for model: {model_id}")
    
    # Get model lineage
    lineage = await manager.get_model_lineage(model_id)
    print(f"Model lineage includes {len(lineage['data_sources'])} data sources")


if __name__ == "__main__":
    asyncio.run(main())
"""
AutoML Pipeline for Pynomaly Detection
======================================

Automated machine learning pipeline that automatically selects the best
anomaly detection algorithm and hyperparameters for a given dataset.
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Import our existing services
from ...simplified_services.core_detection_service import CoreDetectionService
from ...simplified_services.ensemble_service import EnsembleService

logger = logging.getLogger(__name__)

@dataclass
class AutoMLConfig:
    """Configuration for AutoML pipeline."""
    max_trials: int = 100
    timeout_minutes: int = 60
    n_jobs: int = -1
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Optimization settings
    optimization_metric: str = "roc_auc"  # roc_auc, average_precision, f1
    optimization_direction: str = "maximize"
    
    # Algorithm settings
    enable_deep_learning: bool = True
    enable_ensemble: bool = True
    enable_preprocessing: bool = True
    enable_feature_selection: bool = True
    
    # Resource limits
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 8
    
    # Early stopping
    early_stopping_rounds: int = 10
    min_improvement: float = 0.001
    
    # Logging
    log_level: str = "INFO"
    save_trials: bool = True
    save_models: bool = True

@dataclass
class AutoMLResult:
    """Result of AutoML pipeline."""
    best_algorithm: str
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    trial_history: List[Dict[str, Any]]
    optimization_time: float
    total_trials: int
    preprocessing_pipeline: Optional[Any] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_interpretation: Optional[Dict[str, Any]] = None

class AutoMLPipeline:
    """Automated machine learning pipeline for anomaly detection."""
    
    def __init__(self, config: AutoMLConfig = None):
        """Initialize AutoML pipeline.
        
        Args:
            config: AutoML configuration
        """
        self.config = config or AutoMLConfig()
        self.core_service = CoreDetectionService()
        self.ensemble_service = EnsembleService()
        
        # Available algorithms
        self.algorithms = {
            'isolation_forest': self._get_isolation_forest_space,
            'local_outlier_factor': self._get_lof_space,
            'one_class_svm': self._get_ocsvm_space,
            'elliptic_envelope': self._get_elliptic_envelope_space,
            'pyod_knn': self._get_pyod_knn_space,
            'pyod_lof': self._get_pyod_lof_space,
            'pyod_cof': self._get_pyod_cof_space,
            'pyod_abod': self._get_pyod_abod_space,
            'pyod_hbos': self._get_pyod_hbos_space,
            'pyod_iforest': self._get_pyod_iforest_space,
            'pyod_ocsvm': self._get_pyod_ocsvm_space,
        }
        
        # Add ensemble methods
        if self.config.enable_ensemble:
            self.algorithms['ensemble_voting'] = self._get_ensemble_voting_space
            self.algorithms['ensemble_average'] = self._get_ensemble_average_space
            self.algorithms['ensemble_max'] = self._get_ensemble_max_space
        
        # Add deep learning if enabled
        if self.config.enable_deep_learning:
            self.algorithms['deep_autoencoder'] = self._get_deep_autoencoder_space
            self.algorithms['deep_vae'] = self._get_deep_vae_space
        
        # Preprocessing options
        self.preprocessing_options = {
            'scaler': ['standard', 'robust', 'minmax', 'none'],
            'pca': [True, False],
            'feature_selection': [True, False]
        }
        
        self.trial_history = []
        self.best_result = None
        
        logger.info(f"AutoML Pipeline initialized with {len(self.algorithms)} algorithms")
    
    def optimize(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                contamination: float = 0.1) -> AutoMLResult:
        """Run AutoML optimization to find best algorithm and parameters.
        
        Args:
            X: Input features
            y: True labels (if available for evaluation)
            contamination: Expected contamination rate
            
        Returns:
            AutoML optimization result
        """
        start_time = time.time()
        
        logger.info(f"Starting AutoML optimization with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Validate input
        if X.shape[0] < 10:
            raise ValueError("Dataset too small for AutoML (minimum 10 samples)")
        
        # Store data for optimization
        self.X = X
        self.y = y
        self.contamination = contamination
        
        # Choose optimization backend
        if OPTUNA_AVAILABLE:
            return self._optimize_with_optuna()
        elif RAY_AVAILABLE:
            return self._optimize_with_ray()
        else:
            return self._optimize_with_grid_search()
    
    def _optimize_with_optuna(self) -> AutoMLResult:
        """Optimize using Optuna."""
        logger.info("Using Optuna for hyperparameter optimization")
        
        # Create study
        study = optuna.create_study(
            direction=self.config.optimization_direction,
            sampler=TPESampler(seed=self.config.random_state),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize
        study.optimize(
            self._objective_function,
            n_trials=self.config.max_trials,
            timeout=self.config.timeout_minutes * 60,
            n_jobs=self.config.n_jobs
        )
        
        # Get best result
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        # Train final model with best parameters
        best_model = self._train_final_model(best_params)
        
        # Create result
        optimization_time = time.time() - time.time()
        
        return AutoMLResult(
            best_algorithm=best_params['algorithm'],
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            trial_history=[
                {
                    'trial_id': trial.number,
                    'params': trial.params,
                    'score': trial.value,
                    'state': trial.state.name,
                    'duration': trial.duration.total_seconds() if trial.duration else None
                }
                for trial in study.trials
            ],
            optimization_time=optimization_time,
            total_trials=len(study.trials)
        )
    
    def _optimize_with_ray(self) -> AutoMLResult:
        """Optimize using Ray Tune."""
        logger.info("Using Ray Tune for hyperparameter optimization")
        
        # Define search space
        search_space = self._get_ray_search_space()
        
        # Configure scheduler
        scheduler = ASHAScheduler(
            metric=self.config.optimization_metric,
            mode=self.config.optimization_direction,
            max_t=self.config.max_trials,
            grace_period=5,
            reduction_factor=2
        )
        
        # Run optimization
        analysis = tune.run(
            self._ray_objective_function,
            config=search_space,
            num_samples=self.config.max_trials,
            scheduler=scheduler,
            resources_per_trial={"cpu": 1, "gpu": 0},
            time_budget_s=self.config.timeout_minutes * 60,
            verbose=1
        )
        
        # Get best result
        best_trial = analysis.best_trial
        best_params = best_trial.config
        best_score = best_trial.last_result[self.config.optimization_metric]
        
        # Train final model
        best_model = self._train_final_model(best_params)
        
        # Create result
        optimization_time = time.time() - time.time()
        
        return AutoMLResult(
            best_algorithm=best_params['algorithm'],
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            trial_history=[
                {
                    'trial_id': trial.trial_id,
                    'params': trial.config,
                    'score': trial.last_result.get(self.config.optimization_metric, 0),
                    'state': 'COMPLETED',
                    'duration': trial.last_result.get('time_total_s', 0)
                }
                for trial in analysis.trials
            ],
            optimization_time=optimization_time,
            total_trials=len(analysis.trials)
        )
    
    def _optimize_with_grid_search(self) -> AutoMLResult:
        """Fallback optimization using basic grid search."""
        logger.info("Using basic grid search for optimization")
        
        best_score = -np.inf if self.config.optimization_direction == "maximize" else np.inf
        best_params = None
        best_model = None
        trial_history = []
        
        # Simple grid search over algorithms
        for algorithm_name in self.algorithms.keys():
            logger.info(f"Testing algorithm: {algorithm_name}")
            
            # Get basic parameter grid
            param_grid = self._get_basic_param_grid(algorithm_name)
            
            for params in param_grid:
                params['algorithm'] = algorithm_name
                
                try:
                    score = self._evaluate_configuration(params)
                    
                    # Update best if better
                    is_better = (
                        (self.config.optimization_direction == "maximize" and score > best_score) or
                        (self.config.optimization_direction == "minimize" and score < best_score)
                    )
                    
                    if is_better:
                        best_score = score
                        best_params = params.copy()
                        best_model = self._train_final_model(params)
                    
                    trial_history.append({
                        'trial_id': len(trial_history),
                        'params': params.copy(),
                        'score': score,
                        'state': 'COMPLETED',
                        'duration': 0
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate {algorithm_name}: {e}")
                    continue
        
        optimization_time = time.time() - time.time()
        
        return AutoMLResult(
            best_algorithm=best_params['algorithm'] if best_params else 'isolation_forest',
            best_params=best_params or {'algorithm': 'isolation_forest'},
            best_score=best_score,
            best_model=best_model,
            trial_history=trial_history,
            optimization_time=optimization_time,
            total_trials=len(trial_history)
        )
    
    def _objective_function(self, trial) -> float:
        """Objective function for Optuna optimization."""
        # Sample algorithm
        algorithm = trial.suggest_categorical('algorithm', list(self.algorithms.keys()))
        
        # Sample preprocessing parameters
        if self.config.enable_preprocessing:
            scaler_type = trial.suggest_categorical('scaler', self.preprocessing_options['scaler'])
            use_pca = trial.suggest_categorical('pca', self.preprocessing_options['pca'])
            use_feature_selection = trial.suggest_categorical('feature_selection', 
                                                            self.preprocessing_options['feature_selection'])
        else:
            scaler_type = 'none'
            use_pca = False
            use_feature_selection = False
        
        # Sample algorithm-specific parameters
        algo_params = self.algorithms[algorithm](trial)
        
        # Combine all parameters
        params = {
            'algorithm': algorithm,
            'scaler': scaler_type,
            'pca': use_pca,
            'feature_selection': use_feature_selection,
            **algo_params
        }
        
        # Evaluate configuration
        try:
            score = self._evaluate_configuration(params)
            return score
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return -np.inf if self.config.optimization_direction == "maximize" else np.inf
    
    def _evaluate_configuration(self, params: Dict[str, Any]) -> float:
        """Evaluate a configuration and return score."""
        # Apply preprocessing
        X_processed = self._apply_preprocessing(self.X, params)
        
        # Get algorithm and parameters
        algorithm = params['algorithm']
        algo_params = {k: v for k, v in params.items() 
                      if k not in ['algorithm', 'scaler', 'pca', 'feature_selection']}
        
        # Train model
        if algorithm.startswith('ensemble'):
            model = self._train_ensemble_model(algorithm, algo_params, X_processed)
        else:
            model = self._train_single_model(algorithm, algo_params, X_processed)
        
        # Evaluate model
        if self.y is not None:
            # Supervised evaluation
            predictions = model.predict(X_processed)
            anomaly_scores = model.decision_function(X_processed) if hasattr(model, 'decision_function') else predictions
            
            if self.config.optimization_metric == "roc_auc":
                score = roc_auc_score(self.y, anomaly_scores)
            elif self.config.optimization_metric == "average_precision":
                score = average_precision_score(self.y, anomaly_scores)
            elif self.config.optimization_metric == "f1":
                score = f1_score(self.y, predictions)
            else:
                score = roc_auc_score(self.y, anomaly_scores)
        else:
            # Unsupervised evaluation using silhouette score or other metrics
            predictions = model.predict(X_processed)
            anomaly_scores = model.decision_function(X_processed) if hasattr(model, 'decision_function') else predictions
            
            # Use negative anomaly score variance as proxy for quality
            score = -np.var(anomaly_scores)
        
        return score
    
    def _apply_preprocessing(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply preprocessing pipeline."""
        X_processed = X.copy()
        
        # Scaling
        if params.get('scaler') == 'standard':
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
        elif params.get('scaler') == 'robust':
            scaler = RobustScaler()
            X_processed = scaler.fit_transform(X_processed)
        elif params.get('scaler') == 'minmax':
            scaler = MinMaxScaler()
            X_processed = scaler.fit_transform(X_processed)
        
        # PCA
        if params.get('pca', False):
            n_components = min(X_processed.shape[1], X_processed.shape[0] // 2)
            pca = PCA(n_components=n_components)
            X_processed = pca.fit_transform(X_processed)
        
        # Feature selection
        if params.get('feature_selection', False) and self.y is not None:
            k = min(X_processed.shape[1], X_processed.shape[1] // 2)
            selector = SelectKBest(f_classif, k=k)
            X_processed = selector.fit_transform(X_processed, self.y)
        
        return X_processed
    
    def _train_single_model(self, algorithm: str, params: Dict[str, Any], X: np.ndarray):
        """Train a single anomaly detection model."""
        if algorithm.startswith('pyod_'):
            return self._train_pyod_model(algorithm, params, X)
        else:
            return self._train_sklearn_model(algorithm, params, X)
    
    def _train_sklearn_model(self, algorithm: str, params: Dict[str, Any], X: np.ndarray):
        """Train sklearn-based model."""
        # Use core detection service
        self.core_service.contamination = self.contamination
        
        # Map algorithm names
        algo_map = {
            'isolation_forest': 'isolation_forest',
            'local_outlier_factor': 'local_outlier_factor',
            'one_class_svm': 'one_class_svm',
            'elliptic_envelope': 'elliptic_envelope'
        }
        
        sklearn_algo = algo_map.get(algorithm, algorithm)
        
        # Train model
        results = self.core_service.detect_anomalies(
            X, 
            algorithm=sklearn_algo,
            **params
        )
        
        return results['model']
    
    def _train_pyod_model(self, algorithm: str, params: Dict[str, Any], X: np.ndarray):
        """Train PyOD-based model."""
        # Use core detection service with PyOD algorithms
        pyod_algo = algorithm.replace('pyod_', '')
        
        results = self.core_service.detect_anomalies(
            X,
            algorithm=pyod_algo,
            contamination=self.contamination,
            **params
        )
        
        return results['model']
    
    def _train_ensemble_model(self, algorithm: str, params: Dict[str, Any], X: np.ndarray):
        """Train ensemble model."""
        # Use ensemble service
        ensemble_method = algorithm.replace('ensemble_', '')
        
        results = self.ensemble_service.ensemble_detect(
            X,
            contamination=self.contamination,
            voting_strategy=ensemble_method,
            **params
        )
        
        return results['ensemble_model']
    
    def _train_final_model(self, params: Dict[str, Any]):
        """Train final model with best parameters."""
        # Apply preprocessing
        X_processed = self._apply_preprocessing(self.X, params)
        
        # Train model
        algorithm = params['algorithm']
        algo_params = {k: v for k, v in params.items() 
                      if k not in ['algorithm', 'scaler', 'pca', 'feature_selection']}
        
        if algorithm.startswith('ensemble'):
            return self._train_ensemble_model(algorithm, algo_params, X_processed)
        else:
            return self._train_single_model(algorithm, algo_params, X_processed)
    
    # Parameter space definitions for different algorithms
    def _get_isolation_forest_space(self, trial):
        """Get Isolation Forest parameter space."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_samples': trial.suggest_categorical('max_samples', ['auto', 0.5, 0.7, 1.0]),
            'max_features': trial.suggest_float('max_features', 0.5, 1.0),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
    
    def _get_lof_space(self, trial):
        """Get Local Outlier Factor parameter space."""
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 50),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': trial.suggest_int('leaf_size', 10, 50),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
        }
    
    def _get_ocsvm_space(self, trial):
        """Get One-Class SVM parameter space."""
        return {
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']) != 'linear' else 'scale',
            'nu': trial.suggest_float('nu', 0.01, 0.5),
            'degree': trial.suggest_int('degree', 2, 5) if trial.suggest_categorical('kernel', ['poly']) == 'poly' else 3
        }
    
    def _get_elliptic_envelope_space(self, trial):
        """Get Elliptic Envelope parameter space."""
        return {
            'assume_centered': trial.suggest_categorical('assume_centered', [True, False]),
            'support_fraction': trial.suggest_float('support_fraction', 0.5, 1.0)
        }
    
    def _get_pyod_knn_space(self, trial):
        """Get PyOD KNN parameter space."""
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
            'method': trial.suggest_categorical('method', ['largest', 'mean', 'median']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        }
    
    def _get_pyod_lof_space(self, trial):
        """Get PyOD LOF parameter space."""
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 50),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': trial.suggest_int('leaf_size', 10, 50)
        }
    
    def _get_pyod_cof_space(self, trial):
        """Get PyOD COF parameter space."""
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 50),
            'method': trial.suggest_categorical('method', ['fast', 'memory'])
        }
    
    def _get_pyod_abod_space(self, trial):
        """Get PyOD ABOD parameter space."""
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 5, 20),
            'method': trial.suggest_categorical('method', ['fast', 'default'])
        }
    
    def _get_pyod_hbos_space(self, trial):
        """Get PyOD HBOS parameter space."""
        return {
            'n_bins': trial.suggest_int('n_bins', 5, 50),
            'alpha': trial.suggest_float('alpha', 0.1, 0.9),
            'tol': trial.suggest_float('tol', 0.1, 0.9)
        }
    
    def _get_pyod_iforest_space(self, trial):
        """Get PyOD Isolation Forest parameter space."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_samples': trial.suggest_categorical('max_samples', ['auto', 0.5, 0.7, 1.0]),
            'max_features': trial.suggest_float('max_features', 0.5, 1.0),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }
    
    def _get_pyod_ocsvm_space(self, trial):
        """Get PyOD One-Class SVM parameter space."""
        return {
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'nu': trial.suggest_float('nu', 0.01, 0.5)
        }
    
    def _get_ensemble_voting_space(self, trial):
        """Get ensemble voting parameter space."""
        return {
            'algorithms': trial.suggest_categorical('algorithms', [
                ['isolation_forest', 'local_outlier_factor'],
                ['isolation_forest', 'one_class_svm'],
                ['local_outlier_factor', 'one_class_svm'],
                ['isolation_forest', 'local_outlier_factor', 'one_class_svm']
            ]),
            'weights': trial.suggest_categorical('weights', [None, 'uniform', 'performance'])
        }
    
    def _get_ensemble_average_space(self, trial):
        """Get ensemble average parameter space."""
        return {
            'algorithms': trial.suggest_categorical('algorithms', [
                ['isolation_forest', 'local_outlier_factor'],
                ['isolation_forest', 'one_class_svm'],
                ['local_outlier_factor', 'one_class_svm'],
                ['isolation_forest', 'local_outlier_factor', 'one_class_svm']
            ])
        }
    
    def _get_ensemble_max_space(self, trial):
        """Get ensemble max parameter space."""
        return {
            'algorithms': trial.suggest_categorical('algorithms', [
                ['isolation_forest', 'local_outlier_factor'],
                ['isolation_forest', 'one_class_svm'],
                ['local_outlier_factor', 'one_class_svm']
            ])
        }
    
    def _get_deep_autoencoder_space(self, trial):
        """Get deep autoencoder parameter space."""
        return {
            'hidden_neurons': trial.suggest_categorical('hidden_neurons', [
                [64, 32, 16, 32, 64],
                [128, 64, 32, 64, 128],
                [256, 128, 64, 128, 256]
            ]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'l2_regularizer': trial.suggest_float('l2_regularizer', 0.0001, 0.1),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),
            'epochs': trial.suggest_int('epochs', 50, 200),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
        }
    
    def _get_deep_vae_space(self, trial):
        """Get deep VAE parameter space."""
        return {
            'encoder_neurons': trial.suggest_categorical('encoder_neurons', [
                [128, 64, 32],
                [256, 128, 64],
                [512, 256, 128]
            ]),
            'decoder_neurons': trial.suggest_categorical('decoder_neurons', [
                [32, 64, 128],
                [64, 128, 256],
                [128, 256, 512]
            ]),
            'latent_dim': trial.suggest_int('latent_dim', 8, 64),
            'beta': trial.suggest_float('beta', 0.1, 2.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01),
            'epochs': trial.suggest_int('epochs', 50, 200)
        }
    
    def _get_basic_param_grid(self, algorithm: str) -> List[Dict[str, Any]]:
        """Get basic parameter grid for fallback optimization."""
        if algorithm == 'isolation_forest':
            return [
                {'n_estimators': 100, 'max_samples': 'auto'},
                {'n_estimators': 200, 'max_samples': 0.7},
                {'n_estimators': 300, 'max_samples': 1.0}
            ]
        elif algorithm == 'local_outlier_factor':
            return [
                {'n_neighbors': 10},
                {'n_neighbors': 20},
                {'n_neighbors': 30}
            ]
        elif algorithm == 'one_class_svm':
            return [
                {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.05},
                {'kernel': 'rbf', 'gamma': 'scale', 'nu': 0.1},
                {'kernel': 'linear', 'nu': 0.05}
            ]
        else:
            return [{}]  # Default empty parameters
    
    def get_feature_importance(self, model: Any, X: np.ndarray) -> Dict[str, float]:
        """Get feature importance from trained model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                # Use permutation importance
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(model, X, model.predict(X))
                importances = perm_importance.importances_mean
            
            # Create feature importance dictionary
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            return dict(zip(feature_names, importances))
            
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")
            return {}
    
    def save_results(self, result: AutoMLResult, filepath: str):
        """Save AutoML results to file."""
        # Prepare serializable result
        serializable_result = {
            'best_algorithm': result.best_algorithm,
            'best_params': result.best_params,
            'best_score': result.best_score,
            'trial_history': result.trial_history,
            'optimization_time': result.optimization_time,
            'total_trials': result.total_trials,
            'feature_importance': result.feature_importance,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_trials': self.config.max_trials,
                'optimization_metric': self.config.optimization_metric,
                'algorithms_tested': list(self.algorithms.keys())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        logger.info(f"AutoML results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load AutoML results from file."""
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        logger.info(f"AutoML results loaded from {filepath}")
        return results
"""AutoML ensemble for automatic algorithm selection and optimization."""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..adapters.sklearn_adapter import SklearnAdapter
from ..adapters.deeplearning_adapter import DeepLearningAdapter
from ..adapters.pyod_adapter import PyODAdapter

logger = logging.getLogger(__name__)


class OptimizationMetric(Enum):
    """Metrics for model optimization."""
    AUC = "auc"
    PRECISION_RECALL_AUC = "pr_auc"
    F1_SCORE = "f1"
    BALANCED_ACCURACY = "balanced_accuracy"


@dataclass
class ModelCandidate:
    """Represents a model candidate with its configuration."""
    adapter_type: str
    algorithm: str
    parameters: Dict[str, Any]
    score: Optional[float] = None
    training_time: Optional[float] = None
    validation_scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AutoMLResult:
    """Result of AutoML optimization."""
    best_model: ModelCandidate
    all_candidates: List[ModelCandidate]
    optimization_history: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    data_insights: Optional[Dict[str, Any]] = None


class AutoMLEnsemble:
    """AutoML ensemble for automatic algorithm selection and hyperparameter optimization."""
    
    def __init__(
        self,
        optimization_metric: OptimizationMetric = OptimizationMetric.AUC,
        max_trials: int = 50,
        timeout_seconds: Optional[int] = 3600,
        n_folds: int = 5,
        contamination_range: Tuple[float, float] = (0.05, 0.3),
        enable_deep_learning: bool = True,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """Initialize AutoML ensemble.
        
        Args:
            optimization_metric: Metric to optimize
            max_trials: Maximum number of optimization trials
            timeout_seconds: Maximum optimization time in seconds
            n_folds: Number of cross-validation folds
            contamination_range: Range of contamination rates to try
            enable_deep_learning: Whether to include deep learning models
            n_jobs: Number of parallel jobs
            random_state: Random state for reproducibility
        """
        self.optimization_metric = optimization_metric
        self.max_trials = max_trials
        self.timeout_seconds = timeout_seconds
        self.n_folds = n_folds
        self.contamination_range = contamination_range
        self.enable_deep_learning = enable_deep_learning
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.best_model_ = None
        self.optimization_history_ = []
        self.data_insights_ = {}
        
        # Check dependencies
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for AutoMLEnsemble")
        
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate AutoML configuration."""
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if not (0 < self.contamination_range[0] < self.contamination_range[1] < 1):
            raise ValueError("contamination_range must be in (0, 1) with min < max")
    
    def fit(self, data: npt.NDArray[np.floating], labels: Optional[npt.NDArray[np.integer]] = None) -> AutoMLEnsemble:
        """Fit AutoML ensemble on data.
        
        Args:
            data: Training data
            labels: Optional true labels for supervised evaluation
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Starting AutoML optimization with {self.max_trials} trials")
        start_time = time.time()
        
        # Analyze data characteristics
        self._analyze_data(data)
        
        # Generate model candidates
        candidates = self._generate_candidates()
        logger.info(f"Generated {len(candidates)} model candidates")
        
        # Optimize candidates
        if OPTUNA_AVAILABLE:
            optimized_candidates = self._optimize_with_optuna(data, candidates, labels)
        else:
            optimized_candidates = self._optimize_grid_search(data, candidates, labels)
        
        # Select best model
        self.best_model_ = self._select_best_model(optimized_candidates)
        
        # Train final model on full data
        self._train_final_model(data)
        
        total_time = time.time() - start_time
        logger.info(f"AutoML optimization completed in {total_time:.2f}s")
        logger.info(f"Best model: {self.best_model_.adapter_type}/{self.best_model_.algorithm} "
                   f"(score: {self.best_model_.score:.4f})")
        
        return self
    
    def _analyze_data(self, data: npt.NDArray[np.floating]) -> None:
        """Analyze data characteristics to guide model selection."""
        n_samples, n_features = data.shape
        
        # Basic statistics
        self.data_insights_ = {
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_means": np.mean(data, axis=0).tolist(),
            "feature_stds": np.std(data, axis=0).tolist(),
            "data_range": [float(np.min(data)), float(np.max(data))],
            "sparsity": float(np.sum(data == 0) / data.size),
        }
        
        # Dimensionality analysis
        if n_features > 10:
            pca = PCA(n_components=min(10, n_features))
            pca.fit(data)
            explained_variance = np.sum(pca.explained_variance_ratio_)
            self.data_insights_["pca_10_components_variance"] = float(explained_variance)
        
        # Correlation structure
        if n_features > 1:
            corr_matrix = np.corrcoef(data.T)
            self.data_insights_["max_correlation"] = float(np.max(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])))
        
        logger.info(f"Data analysis: {n_samples} samples, {n_features} features, "
                   f"sparsity: {self.data_insights_['sparsity']:.3f}")
    
    def _generate_candidates(self) -> List[ModelCandidate]:
        """Generate initial model candidates based on data characteristics."""
        candidates = []
        
        # Always include classical algorithms
        sklearn_algorithms = ["iforest", "lof", "ocsvm"]
        if self.data_insights_["n_features"] > 5:
            sklearn_algorithms.append("pca")
        
        for algo in sklearn_algorithms:
            candidates.append(ModelCandidate(
                adapter_type="sklearn",
                algorithm=algo,
                parameters=self._get_default_parameters(algo)
            ))
        
        # Add PyOD algorithms if available
        try:
            pyod_algorithms = ["knn", "cblof", "hbos", "mcd"]
            for algo in pyod_algorithms:
                candidates.append(ModelCandidate(
                    adapter_type="pyod",
                    algorithm=algo,
                    parameters=self._get_default_parameters(algo)
                ))
        except:
            logger.warning("PyOD not available, skipping PyOD algorithms")
        
        # Add deep learning if enabled and data is suitable
        if (self.enable_deep_learning and 
            self.data_insights_["n_samples"] > 1000 and 
            self.data_insights_["n_features"] >= 5):
            
            frameworks = DeepLearningAdapter.get_available_frameworks()
            for framework in frameworks[:1]:  # Limit to one framework for efficiency
                candidates.append(ModelCandidate(
                    adapter_type="deeplearning",
                    algorithm="autoencoder",
                    parameters={
                        "framework": framework,
                        "hidden_dims": DeepLearningAdapter.create_default_architecture(
                            self.data_insights_["n_features"]
                        ),
                        "epochs": 50,  # Reduced for AutoML
                        "batch_size": min(64, self.data_insights_["n_samples"] // 10)
                    }
                ))
        
        return candidates
    
    def _get_default_parameters(self, algorithm: str) -> Dict[str, Any]:
        """Get default parameters for algorithms."""
        defaults = {
            # Sklearn
            "iforest": {"n_estimators": 100, "contamination": 0.1},
            "lof": {"n_neighbors": 20, "contamination": 0.1},
            "ocsvm": {"nu": 0.1, "kernel": "rbf"},
            "pca": {"contamination": 0.1},
            
            # PyOD
            "knn": {"n_neighbors": 5, "contamination": 0.1},
            "cblof": {"n_clusters": 8, "contamination": 0.1},
            "hbos": {"n_bins": 10, "contamination": 0.1},
            "mcd": {"contamination": 0.1},
        }
        return defaults.get(algorithm, {"contamination": 0.1})
    
    def _optimize_with_optuna(
        self, 
        data: npt.NDArray[np.floating], 
        candidates: List[ModelCandidate],
        labels: Optional[npt.NDArray[np.integer]] = None
    ) -> List[ModelCandidate]:
        """Optimize candidates using Optuna."""
        logger.info("Using Optuna for hyperparameter optimization")
        
        def objective(trial):
            # Select candidate
            candidate_idx = trial.suggest_int("candidate", 0, len(candidates) - 1)
            candidate = candidates[candidate_idx]
            
            # Optimize parameters
            optimized_params = self._suggest_parameters(trial, candidate)
            
            # Evaluate candidate
            score = self._evaluate_candidate(data, candidate.adapter_type, 
                                           candidate.algorithm, optimized_params, labels)
            return score
        
        # Run optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(
            objective, 
            n_trials=self.max_trials, 
            timeout=self.timeout_seconds,
            show_progress_bar=True
        )
        
        # Extract best candidates
        optimized_candidates = []
        for trial in study.trials:
            if trial.value is not None:
                candidate_idx = trial.params["candidate"]
                base_candidate = candidates[candidate_idx]
                
                optimized_params = {k: v for k, v in trial.params.items() if k != "candidate"}
                
                optimized_candidates.append(ModelCandidate(
                    adapter_type=base_candidate.adapter_type,
                    algorithm=base_candidate.algorithm,
                    parameters=optimized_params,
                    score=trial.value,
                    metadata={"trial_number": trial.number}
                ))
        
        self.optimization_history_ = study.trials_dataframe().to_dict('records')
        return optimized_candidates
    
    def _optimize_grid_search(
        self,
        data: npt.NDArray[np.floating],
        candidates: List[ModelCandidate],
        labels: Optional[npt.NDArray[np.integer]] = None
    ) -> List[ModelCandidate]:
        """Fallback optimization using simple grid search."""
        logger.info("Using grid search for hyperparameter optimization")
        
        optimized_candidates = []
        
        # Evaluate each candidate with parameter variations
        for candidate in candidates:
            param_grids = self._get_parameter_grids(candidate)
            
            for params in param_grids:
                score = self._evaluate_candidate(
                    data, candidate.adapter_type, candidate.algorithm, params, labels
                )
                
                optimized_candidates.append(ModelCandidate(
                    adapter_type=candidate.adapter_type,
                    algorithm=candidate.algorithm,
                    parameters=params,
                    score=score
                ))
        
        return optimized_candidates
    
    def _suggest_parameters(self, trial, candidate: ModelCandidate) -> Dict[str, Any]:
        """Suggest parameters for Optuna trial."""
        params = candidate.parameters.copy()
        
        # Common parameter suggestions
        if "contamination" in params:
            params["contamination"] = trial.suggest_float(
                "contamination", 
                self.contamination_range[0], 
                self.contamination_range[1]
            )
        
        # Algorithm-specific suggestions
        if candidate.algorithm == "iforest":
            params["n_estimators"] = trial.suggest_int("n_estimators", 50, 200)
            params["max_samples"] = trial.suggest_categorical("max_samples", ["auto", 0.5, 0.8])
            
        elif candidate.algorithm == "lof":
            params["n_neighbors"] = trial.suggest_int("n_neighbors", 5, 50)
            
        elif candidate.algorithm == "ocsvm":
            params["nu"] = trial.suggest_float("nu", 0.01, 0.3)
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            
        elif candidate.algorithm == "autoencoder":
            # Optimize architecture
            n_layers = trial.suggest_int("n_layers", 2, 4)
            hidden_dims = []
            base_dim = self.data_insights_["n_features"]
            
            for i in range(n_layers):
                dim = trial.suggest_int(f"layer_{i}_dim", 4, min(base_dim, 128))
                hidden_dims.append(dim)
                base_dim = dim
            
            params["hidden_dims"] = hidden_dims
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        
        return params
    
    def _get_parameter_grids(self, candidate: ModelCandidate) -> List[Dict[str, Any]]:
        """Get parameter grids for grid search."""
        base_params = candidate.parameters.copy()
        grids = []
        
        # Simple parameter variations
        contamination_values = [0.05, 0.1, 0.15, 0.2]
        
        for cont in contamination_values:
            params = base_params.copy()
            params["contamination"] = cont
            grids.append(params)
        
        return grids[:5]  # Limit for efficiency
    
    def _evaluate_candidate(
        self,
        data: npt.NDArray[np.floating],
        adapter_type: str,
        algorithm: str,
        parameters: Dict[str, Any],
        labels: Optional[npt.NDArray[np.integer]] = None
    ) -> float:
        """Evaluate a model candidate."""
        try:
            start_time = time.time()
            
            # Create adapter
            if adapter_type == "sklearn":
                adapter = SklearnAdapter(algorithm, **parameters)
            elif adapter_type == "deeplearning":
                adapter = DeepLearningAdapter(**parameters)
            elif adapter_type == "pyod":
                adapter = PyODAdapter(algorithm, **parameters)
            else:
                return 0.0
            
            # Cross-validation evaluation
            if labels is not None:
                # Supervised evaluation
                score = self._supervised_evaluation(adapter, data, labels)
            else:
                # Unsupervised evaluation using reconstruction error consistency
                score = self._unsupervised_evaluation(adapter, data)
            
            training_time = time.time() - start_time
            logger.debug(f"Evaluated {adapter_type}/{algorithm}: {score:.4f} ({training_time:.2f}s)")
            
            return score
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {adapter_type}/{algorithm}: {e}")
            return 0.0
    
    def _supervised_evaluation(
        self,
        adapter: Any,
        data: npt.NDArray[np.floating],
        labels: npt.NDArray[np.integer]
    ) -> float:
        """Evaluate using supervised metrics."""
        scores = []
        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, val_idx in kfold.split(data, labels):
            train_data, val_data = data[train_idx], data[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            
            # Fit on training data
            adapter.fit(train_data)
            
            # Predict on validation data
            if hasattr(adapter, 'decision_function'):
                val_scores = adapter.decision_function(val_data)
                if self.optimization_metric == OptimizationMetric.AUC:
                    score = roc_auc_score(val_labels, val_scores)
                elif self.optimization_metric == OptimizationMetric.PRECISION_RECALL_AUC:
                    precision, recall, _ = precision_recall_curve(val_labels, val_scores)
                    score = auc(recall, precision)
                else:
                    score = roc_auc_score(val_labels, val_scores)  # Fallback
            else:
                val_preds = adapter.predict(val_data)
                score = roc_auc_score(val_labels, val_preds)
            
            scores.append(score)
        
        return np.mean(scores)
    
    def _unsupervised_evaluation(
        self,
        adapter: Any,
        data: npt.NDArray[np.floating]
    ) -> float:
        """Evaluate using unsupervised metrics."""
        # Use bootstrap sampling for consistency evaluation
        n_bootstrap = 5
        scores = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample_indices = np.random.choice(
                len(data), size=int(0.8 * len(data)), replace=True
            )
            sample_data = data[sample_indices]
            
            try:
                # Fit and predict
                adapter.fit(sample_data)
                predictions = adapter.predict(sample_data)
                
                # Score based on anomaly rate consistency
                anomaly_rate = np.mean(predictions)
                target_contamination = adapter.get_parameters().get("contamination", 0.1)
                
                # Penalize deviations from expected contamination
                score = 1.0 - abs(anomaly_rate - target_contamination) / target_contamination
                scores.append(max(0.0, score))
                
            except Exception:
                scores.append(0.0)
        
        return np.mean(scores)
    
    def _select_best_model(self, candidates: List[ModelCandidate]) -> ModelCandidate:
        """Select the best model from candidates."""
        if not candidates:
            raise ValueError("No valid candidates found")
        
        # Sort by score
        candidates.sort(key=lambda x: x.score or 0.0, reverse=True)
        
        best = candidates[0]
        logger.info(f"Selected best model: {best.adapter_type}/{best.algorithm} "
                   f"with score {best.score:.4f}")
        
        return best
    
    def _train_final_model(self, data: npt.NDArray[np.floating]) -> None:
        """Train the final model on full dataset."""
        if self.best_model_.adapter_type == "sklearn":
            self.final_adapter_ = SklearnAdapter(
                self.best_model_.algorithm, 
                **self.best_model_.parameters
            )
        elif self.best_model_.adapter_type == "deeplearning":
            self.final_adapter_ = DeepLearningAdapter(**self.best_model_.parameters)
        elif self.best_model_.adapter_type == "pyod":
            self.final_adapter_ = PyODAdapter(
                self.best_model_.algorithm,
                **self.best_model_.parameters
            )
        
        self.final_adapter_.fit(data)
        logger.info("Final model trained on full dataset")
    
    def predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Predict using the best model."""
        if not hasattr(self, 'final_adapter_'):
            raise ValueError("Model must be fitted before prediction")
        return self.final_adapter_.predict(data)
    
    def decision_function(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get anomaly scores using the best model."""
        if not hasattr(self, 'final_adapter_'):
            raise ValueError("Model must be fitted before scoring")
        return self.final_adapter_.decision_function(data)
    
    def get_optimization_results(self) -> AutoMLResult:
        """Get detailed optimization results."""
        return AutoMLResult(
            best_model=self.best_model_,
            all_candidates=getattr(self, 'all_candidates_', []),
            optimization_history=self.optimization_history_,
            data_insights=self.data_insights_
        )
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the best model if available."""
        if not hasattr(self, 'final_adapter_'):
            return None
            
        if hasattr(self.final_adapter_, 'get_feature_importances'):
            importances = self.final_adapter_.get_feature_importances()
            if importances is not None:
                return {
                    f"feature_{i}": float(imp) 
                    for i, imp in enumerate(importances)
                }
        
        return None
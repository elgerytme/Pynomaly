"""Ensemble anomaly detection adapter combining multiple algorithms.

This adapter provides ensemble methods that combine predictions from multiple
base algorithms to improve detection accuracy and robustness.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import AdapterError, AlgorithmNotFoundError
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.shared.protocols import DetectorProtocol

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


class BaseEnsembleDetector:
    """Base class for ensemble detectors."""
    
    def __init__(self, base_detectors: List[Any], combination_method: str = 'average'):
        self.base_detectors = base_detectors
        self.combination_method = combination_method
        self.weights = None
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> None:
        """Fit all base detectors."""
        for detector in self.base_detectors:
            detector.fit(X)
        self.fitted = True
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble anomaly scores."""
        if not self.fitted:
            raise AdapterError("Ensemble must be fitted before prediction")
        
        all_scores = []
        for detector in self.base_detectors:
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X)
                # Convert to positive anomaly scores
                scores = -scores  # Higher values = more anomalous
            elif hasattr(detector, 'score_samples'):
                scores = -detector.score_samples(X)  # LOF returns negative scores
            else:
                # Fallback: use distance-based scoring
                scores = self._calculate_distance_scores(detector, X)
            
            # Normalize scores to [0, 1]
            scores = self._normalize_scores(scores)
            all_scores.append(scores)
        
        # Combine scores
        all_scores = np.array(all_scores).T  # Shape: (n_samples, n_detectors)
        
        if self.combination_method == 'average':
            ensemble_scores = np.mean(all_scores, axis=1)
        elif self.combination_method == 'weighted_average':
            if self.weights is None:
                self.weights = np.ones(len(self.base_detectors)) / len(self.base_detectors)
            ensemble_scores = np.average(all_scores, axis=1, weights=self.weights)
        elif self.combination_method == 'max':
            ensemble_scores = np.max(all_scores, axis=1)
        elif self.combination_method == 'min':
            ensemble_scores = np.min(all_scores, axis=1)
        elif self.combination_method == 'median':
            ensemble_scores = np.median(all_scores, axis=1)
        else:
            ensemble_scores = np.mean(all_scores, axis=1)
        
        return ensemble_scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return np.zeros_like(scores)
    
    def _calculate_distance_scores(self, detector: Any, X: np.ndarray) -> np.ndarray:
        """Calculate distance-based scores for detectors without scoring methods."""
        # This is a fallback method - in practice should not be needed for sklearn detectors
        return np.random.random(len(X))  # Placeholder


class VotingEnsemble(BaseEnsembleDetector):
    """Voting ensemble that combines binary predictions."""
    
    def __init__(self, base_detectors: List[Any], voting: str = 'hard', contamination: float = 0.1):
        super().__init__(base_detectors, 'voting')
        self.voting = voting  # 'hard' or 'soft'
        self.contamination = contamination
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using voting ensemble."""
        if self.voting == 'soft':
            # Use probability-based voting
            scores = self.predict_scores(X)
            threshold = np.percentile(scores, (1 - self.contamination) * 100)
            labels = (scores > threshold).astype(int)
            return labels, scores
        else:
            # Hard voting - majority rule
            all_predictions = []
            all_scores = []
            
            for detector in self.base_detectors:
                if hasattr(detector, 'predict'):
                    pred = detector.predict(X)
                    # Convert sklearn predictions (-1, 1) to (0, 1)
                    pred = np.where(pred == -1, 1, 0)
                else:
                    # Use decision function with threshold
                    scores = self.predict_scores(X)
                    threshold = np.percentile(scores, (1 - self.contamination) * 100)
                    pred = (scores > threshold).astype(int)
                
                all_predictions.append(pred)
            
            # Majority voting
            all_predictions = np.array(all_predictions).T
            ensemble_pred = np.round(np.mean(all_predictions, axis=1)).astype(int)
            
            # Calculate ensemble scores for confidence
            ensemble_scores = self.predict_scores(X)
            
            return ensemble_pred, ensemble_scores


class StackingEnsemble(BaseEnsembleDetector):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, base_detectors: List[Any], meta_detector: Any = None):
        super().__init__(base_detectors, 'stacking')
        self.meta_detector = meta_detector or IsolationForest(contamination=0.1, random_state=42)
        self.meta_fitted = False
    
    def fit(self, X: np.ndarray) -> None:
        """Fit base detectors and meta-learner."""
        # Fit base detectors
        super().fit(X)
        
        # Get base detector scores for meta-learning
        base_scores = []
        for detector in self.base_detectors:
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X)
            elif hasattr(detector, 'score_samples'):
                scores = detector.score_samples(X)
            else:
                scores = np.random.random(len(X))  # Fallback
            
            base_scores.append(self._normalize_scores(scores))
        
        # Stack base scores for meta-learner
        meta_features = np.column_stack(base_scores)
        
        # Fit meta-learner
        self.meta_detector.fit(meta_features)
        self.meta_fitted = True
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Get stacking ensemble scores."""
        if not self.meta_fitted:
            raise AdapterError("Stacking ensemble must be fitted before prediction")
        
        # Get base detector scores
        base_scores = []
        for detector in self.base_detectors:
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X)
            elif hasattr(detector, 'score_samples'):
                scores = detector.score_samples(X)
            else:
                scores = np.random.random(len(X))  # Fallback
            
            base_scores.append(self._normalize_scores(scores))
        
        # Stack base scores
        meta_features = np.column_stack(base_scores)
        
        # Get meta-learner scores
        if hasattr(self.meta_detector, 'decision_function'):
            ensemble_scores = -self.meta_detector.decision_function(meta_features)
        elif hasattr(self.meta_detector, 'score_samples'):
            ensemble_scores = -self.meta_detector.score_samples(meta_features)
        else:
            ensemble_scores = np.mean(meta_features, axis=1)  # Fallback
        
        return self._normalize_scores(ensemble_scores)


class AdaptiveEnsemble(BaseEnsembleDetector):
    """Adaptive ensemble that learns detector weights."""
    
    def __init__(self, base_detectors: List[Any], adaptation_method: str = 'performance'):
        super().__init__(base_detectors, 'adaptive')
        self.adaptation_method = adaptation_method
        self.performance_scores = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit base detectors and learn weights."""
        super().fit(X)
        
        if y is not None and self.adaptation_method == 'performance':
            # Learn weights based on individual detector performance
            self._learn_performance_weights(X, y)
        else:
            # Use diversity-based weighting
            self._learn_diversity_weights(X)
    
    def _learn_performance_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Learn weights based on detector performance."""
        performance_scores = []
        
        for detector in self.base_detectors:
            # Get detector scores
            if hasattr(detector, 'decision_function'):
                scores = -detector.decision_function(X)
            elif hasattr(detector, 'score_samples'):
                scores = -detector.score_samples(X)
            else:
                scores = np.random.random(len(X))
            
            scores = self._normalize_scores(scores)
            
            # Calculate performance (AUC if labels available)
            try:
                auc = roc_auc_score(y, scores)
                performance_scores.append(max(auc, 1 - auc))  # Handle inverted scores
            except ValueError:
                performance_scores.append(0.5)  # Default performance
        
        # Convert to weights (higher performance = higher weight)
        performance_scores = np.array(performance_scores)
        self.weights = performance_scores / np.sum(performance_scores)
        self.performance_scores = performance_scores
    
    def _learn_diversity_weights(self, X: np.ndarray) -> None:
        """Learn weights based on detector diversity."""
        all_scores = []
        
        for detector in self.base_detectors:
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X)
            elif hasattr(detector, 'score_samples'):
                scores = detector.score_samples(X)
            else:
                scores = np.random.random(len(X))
            
            all_scores.append(self._normalize_scores(scores))
        
        # Calculate diversity (negative correlation)
        all_scores = np.array(all_scores)
        diversity_scores = []
        
        for i, scores_i in enumerate(all_scores):
            diversity = 0
            for j, scores_j in enumerate(all_scores):
                if i != j:
                    corr = np.corrcoef(scores_i, scores_j)[0, 1]
                    if not np.isnan(corr):
                        diversity += (1 - abs(corr))  # Higher diversity for lower correlation
            
            diversity_scores.append(diversity / (len(all_scores) - 1))
        
        # Convert to weights
        diversity_scores = np.array(diversity_scores)
        self.weights = diversity_scores / np.sum(diversity_scores)


class EnsembleAdapter(DetectorProtocol):
    """Ensemble anomaly detection adapter."""
    
    _algorithm_map = {
        "VotingEnsemble": VotingEnsemble,
        "StackingEnsemble": StackingEnsemble,
        "AdaptiveEnsemble": AdaptiveEnsemble,
        "AverageEnsemble": BaseEnsembleDetector,
        "MaxEnsemble": BaseEnsembleDetector,
        "MedianEnsemble": BaseEnsembleDetector,
    }
    
    def __init__(self, detector: Detector):
        """Initialize ensemble adapter.
        
        Args:
            detector: Detector entity with ensemble configuration
        """
        self.detector = detector
        self._ensemble = None
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._init_algorithm()
    
    def _init_algorithm(self) -> None:
        """Initialize the ensemble algorithm."""
        if self.detector.algorithm_name not in self._algorithm_map:
            available = ", ".join(self._algorithm_map.keys())
            raise AlgorithmNotFoundError(
                f"Algorithm '{self.detector.algorithm_name}' not found. "
                f"Available ensemble algorithms: {available}"
            )
        
        params = self.detector.parameters.copy()
        
        # Create base detectors
        base_detectors = self._create_base_detectors(params)
        
        # Create ensemble
        try:
            if self.detector.algorithm_name == "VotingEnsemble":
                voting = params.get('voting', 'hard')
                contamination = params.get('contamination', 0.1)
                self._ensemble = VotingEnsemble(base_detectors, voting, contamination)
            
            elif self.detector.algorithm_name == "StackingEnsemble":
                meta_detector_type = params.get('meta_detector', 'IsolationForest')
                meta_detector = self._create_meta_detector(meta_detector_type, params)
                self._ensemble = StackingEnsemble(base_detectors, meta_detector)
            
            elif self.detector.algorithm_name == "AdaptiveEnsemble":
                adaptation_method = params.get('adaptation_method', 'performance')
                self._ensemble = AdaptiveEnsemble(base_detectors, adaptation_method)
            
            elif self.detector.algorithm_name == "AverageEnsemble":
                self._ensemble = BaseEnsembleDetector(base_detectors, 'average')
            
            elif self.detector.algorithm_name == "MaxEnsemble":
                self._ensemble = BaseEnsembleDetector(base_detectors, 'max')
            
            elif self.detector.algorithm_name == "MedianEnsemble":
                self._ensemble = BaseEnsembleDetector(base_detectors, 'median')
                
        except Exception as e:
            raise AdapterError(f"Failed to initialize ensemble {self.detector.algorithm_name}: {e}")
    
    def _create_base_detectors(self, params: Dict[str, Any]) -> List[Any]:
        """Create base detectors for the ensemble."""
        base_algorithms = params.get('base_algorithms', ['IsolationForest', 'LOF', 'OneClassSVM'])
        contamination = params.get('contamination', 0.1)
        
        detectors = []
        
        for alg_name in base_algorithms:
            try:
                if alg_name == 'IsolationForest':
                    detector = IsolationForest(
                        contamination=contamination,
                        random_state=42,
                        n_estimators=params.get('n_estimators', 100)
                    )
                
                elif alg_name == 'LOF':
                    detector = LocalOutlierFactor(
                        contamination=contamination,
                        n_neighbors=params.get('n_neighbors', 20),
                        novelty=True  # Enable predict method
                    )
                
                elif alg_name == 'OneClassSVM':
                    detector = OneClassSVM(
                        nu=contamination,
                        kernel=params.get('kernel', 'rbf'),
                        gamma=params.get('gamma', 'scale')
                    )
                
                elif alg_name == 'EllipticEnvelope':
                    detector = EllipticEnvelope(
                        contamination=contamination,
                        support_fraction=params.get('support_fraction', None)
                    )
                
                else:
                    logger.warning(f"Unknown base algorithm: {alg_name}, using IsolationForest")
                    detector = IsolationForest(contamination=contamination, random_state=42)
                
                detectors.append(detector)
                
            except Exception as e:
                logger.warning(f"Failed to create {alg_name}: {e}, skipping")
        
        if not detectors:
            # Fallback to default ensemble
            detectors = [
                IsolationForest(contamination=contamination, random_state=42),
                LocalOutlierFactor(contamination=contamination, novelty=True),
                OneClassSVM(nu=contamination)
            ]
        
        return detectors
    
    def _create_meta_detector(self, meta_type: str, params: Dict[str, Any]) -> Any:
        """Create meta-detector for stacking ensemble."""
        contamination = params.get('contamination', 0.1)
        
        if meta_type == 'IsolationForest':
            return IsolationForest(contamination=contamination, random_state=42)
        elif meta_type == 'OneClassSVM':
            return OneClassSVM(nu=contamination)
        elif meta_type == 'EllipticEnvelope':
            return EllipticEnvelope(contamination=contamination)
        else:
            return IsolationForest(contamination=contamination, random_state=42)
    
    def fit(self, dataset: Dataset) -> None:
        """Fit the ensemble detector.
        
        Args:
            dataset: Training dataset
        """
        try:
            # Prepare data
            X_train = self._prepare_data(dataset)
            
            # Scale data
            X_scaled = self._scaler.fit_transform(X_train)
            
            # Check if we have labels for adaptive ensemble
            y_train = None
            if (self.detector.algorithm_name == "AdaptiveEnsemble" and 
                dataset.target_column and 
                dataset.target_column in dataset.data.columns):
                y_train = dataset.data[dataset.target_column].values
            
            # Fit ensemble
            if y_train is not None:
                self._ensemble.fit(X_scaled, y_train)
            else:
                self._ensemble.fit(X_scaled)
            
            self.detector.is_fitted = True
            self._is_fitted = True
            logger.info(f"Successfully fitted ensemble detector: {self.detector.algorithm_name}")
            
        except Exception as e:
            raise AdapterError(f"Failed to fit ensemble model: {e}")
    
    def predict(self, dataset: Dataset) -> DetectionResult:
        """Predict anomalies using the ensemble.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Detection results with anomaly scores and labels
        """
        if not self._is_fitted or self._ensemble is None:
            raise AdapterError("Model must be fitted before prediction")
        
        try:
            # Prepare data
            X_test = self._prepare_data(dataset)
            X_scaled = self._scaler.transform(X_test)
            
            # Get ensemble predictions
            if isinstance(self._ensemble, VotingEnsemble):
                labels, scores = self._ensemble.predict(X_scaled)
            else:
                scores = self._ensemble.predict_scores(X_scaled)
                # Calculate threshold and labels
                contamination = self.detector.parameters.get('contamination', 0.1)
                threshold = np.percentile(scores, (1 - contamination) * 100)
                labels = (scores > threshold).astype(int)
            
            # Create anomaly scores
            anomaly_scores = [
                AnomalyScore(
                    value=float(score),
                    method=self.detector.algorithm_name
                )
                for score in scores
            ]
            
            # Create anomaly objects for detected anomalies
            from pynomaly.domain.entities.anomaly import Anomaly
            anomalies = []
            anomaly_indices = np.where(labels == 1)[0]
            threshold_val = np.percentile(scores, (1 - self.detector.parameters.get('contamination', 0.1)) * 100)
            
            for idx in anomaly_indices:
                # Get original data row for this anomaly
                X_test = self._prepare_data(dataset)
                data_row = {f"feature_{i}": float(X_test[idx, i]) for i in range(X_test.shape[1])}
                
                anomaly = Anomaly(
                    score=anomaly_scores[idx],
                    data_point=data_row,
                    detector_name=self.detector.algorithm_name,
                    metadata={"ensemble_detection": True, "row_index": int(idx)}
                )
                anomalies.append(anomaly)
            
            return DetectionResult(
                detector_id=self.detector.id,
                dataset_id=dataset.id,
                anomalies=anomalies,
                scores=anomaly_scores,
                labels=labels,
                threshold=float(threshold_val),
                metadata={
                    "algorithm": self.detector.algorithm_name,
                    "n_base_detectors": len(self._ensemble.base_detectors),
                    "base_algorithms": self.detector.parameters.get('base_algorithms', []),
                    "combination_method": getattr(self._ensemble, 'combination_method', 'unknown'),
                    "n_anomalies": int(np.sum(labels)),
                    "contamination_rate": float(np.sum(labels) / len(labels)),
                    "model_type": "ensemble",
                    "weights": getattr(self._ensemble, 'weights', None),
                    "performance_scores": getattr(self._ensemble, 'performance_scores', None)
                }
            )
            
        except Exception as e:
            raise AdapterError(f"Failed to predict with ensemble model: {e}")
    
    def _prepare_data(self, dataset: Dataset) -> np.ndarray:
        """Prepare data for ensemble model.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Numpy array of features
        """
        df = dataset.data
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        if dataset.target_column and dataset.target_column in numeric_cols:
            numeric_cols.remove(dataset.target_column)
        
        if not numeric_cols:
            raise AdapterError("No numeric features found in dataset")
        
        # Extract features and handle missing values
        X = df[numeric_cols].values
        
        # Simple imputation - replace NaN with column mean
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])
        
        return X
    
    def _calculate_confidence(self, score: float, threshold: float) -> float:
        """Calculate confidence score.
        
        Args:
            score: Anomaly score
            threshold: Detection threshold
            
        Returns:
            Confidence value between 0 and 1
        """
        if score <= threshold:
            return 1.0 - (score / threshold) * 0.5
        else:
            return 0.5 + min((score - threshold) / threshold * 0.5, 0.5)
    
    @classmethod
    def get_supported_algorithms(cls) -> List[str]:
        """Get list of supported ensemble algorithms.
        
        Returns:
            List of algorithm names
        """
        return list(cls._algorithm_map.keys())
    
    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> Dict[str, Any]:
        """Get information about a specific ensemble algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Algorithm metadata and parameters
        """
        if algorithm not in cls._algorithm_map:
            raise AlgorithmNotFoundError(f"Algorithm '{algorithm}' not found")
        
        info = {
            "VotingEnsemble": {
                "name": "Voting Ensemble",
                "type": "Ensemble",
                "description": "Combines predictions using majority voting (hard) or probability averaging (soft)",
                "parameters": {
                    "base_algorithms": {"type": "list", "default": ["IsolationForest", "LOF", "OneClassSVM"], "description": "List of base algorithms"},
                    "voting": {"type": "str", "default": "hard", "description": "Voting type (hard/soft)"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"},
                    "n_estimators": {"type": "int", "default": 100, "description": "Number of estimators for tree-based methods"}
                },
                "suitable_for": ["diverse_algorithms", "robust_detection", "interpretable_results"],
                "pros": ["Simple and interpretable", "Robust to individual detector failures", "Good baseline ensemble"],
                "cons": ["Equal weight to all detectors", "May not utilize detector strengths optimally"]
            },
            "StackingEnsemble": {
                "name": "Stacking Ensemble",
                "type": "Ensemble",
                "description": "Uses a meta-learner to combine base detector outputs optimally",
                "parameters": {
                    "base_algorithms": {"type": "list", "default": ["IsolationForest", "LOF", "OneClassSVM"], "description": "Base algorithms"},
                    "meta_detector": {"type": "str", "default": "IsolationForest", "description": "Meta-learner algorithm"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["complex_patterns", "optimal_combination", "high_accuracy_needs"],
                "pros": ["Learns optimal combination", "Can capture complex interactions", "Often highest accuracy"],
                "cons": ["More complex", "Requires more computation", "May overfit"]
            },
            "AdaptiveEnsemble": {
                "name": "Adaptive Ensemble",
                "type": "Ensemble",
                "description": "Adapts detector weights based on performance or diversity metrics",
                "parameters": {
                    "base_algorithms": {"type": "list", "default": ["IsolationForest", "LOF", "OneClassSVM"], "description": "Base algorithms"},
                    "adaptation_method": {"type": "str", "default": "performance", "description": "Adaptation method (performance/diversity)"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["labeled_data", "performance_optimization", "adaptive_systems"],
                "pros": ["Adapts to data characteristics", "Optimal weighting", "Good performance"],
                "cons": ["Requires labeled data for performance adaptation", "Complex weight learning"]
            },
            "AverageEnsemble": {
                "name": "Average Ensemble",
                "type": "Ensemble",
                "description": "Simple averaging of base detector scores",
                "parameters": {
                    "base_algorithms": {"type": "list", "default": ["IsolationForest", "LOF", "OneClassSVM"], "description": "Base algorithms"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["simple_combination", "baseline_ensemble", "fast_detection"],
                "pros": ["Very simple", "Fast computation", "Stable results"],
                "cons": ["Equal weight assumption", "May not be optimal"]
            },
            "MaxEnsemble": {
                "name": "Max Ensemble",
                "type": "Ensemble",
                "description": "Uses maximum score across base detectors (conservative approach)",
                "parameters": {
                    "base_algorithms": {"type": "list", "default": ["IsolationForest", "LOF", "OneClassSVM"], "description": "Base algorithms"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["conservative_detection", "high_precision_needs", "critical_systems"],
                "pros": ["Conservative approach", "Detects strong anomalies", "High precision"],
                "cons": ["May miss subtle anomalies", "Lower recall"]
            },
            "MedianEnsemble": {
                "name": "Median Ensemble",
                "type": "Ensemble",
                "description": "Uses median score across base detectors (robust to outliers)",
                "parameters": {
                    "base_algorithms": {"type": "list", "default": ["IsolationForest", "LOF", "OneClassSVM"], "description": "Base algorithms"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["robust_combination", "outlier_resistant", "stable_detection"],
                "pros": ["Robust to detector outliers", "Stable results", "Good balance"],
                "cons": ["May lose information", "Not optimal for all cases"]
            }
        }
        
        return info.get(algorithm, {})
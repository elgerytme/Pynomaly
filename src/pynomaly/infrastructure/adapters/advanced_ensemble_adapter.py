"""
Advanced ensemble methods for anomaly detection.

This module implements state-of-the-art ensemble techniques including:
- Dynamic ensemble selection
- Bayesian ensemble averaging
- Feature bagging ensembles
- Rotation forests for anomaly detection
- Multi-objective ensemble optimization
- Online adaptive ensembles
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import AdapterError, AlgorithmNotFoundError
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.shared.protocols import DetectorProtocol, EnsembleDetectorProtocol

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class EnsembleMetrics:
    """Metrics for ensemble evaluation and selection."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    diversity: float
    confidence: float
    
    def score(self) -> float:
        """Calculate overall ensemble score."""
        # Weighted combination of metrics
        return (0.3 * self.auc_score + 
                0.2 * self.f1_score + 
                0.2 * self.diversity + 
                0.15 * self.confidence + 
                0.15 * self.precision)


class DynamicEnsembleSelector:
    """Dynamic ensemble selection based on local competence."""
    
    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors
        self.base_detectors = []
        self.competence_regions = {}
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, base_detectors: List[Any]) -> None:
        """Fit the dynamic selector.
        
        Args:
            X: Training data
            y: Ground truth labels
            base_detectors: List of fitted base detectors
        """
        self.base_detectors = base_detectors
        
        # Calculate competence for each detector in each region
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=self.k_neighbors)
        nn.fit(X)
        
        # For each point, find k-nearest neighbors and evaluate detector performance
        for i, detector in enumerate(base_detectors):
            competence_scores = []
            
            for j in range(len(X)):
                # Find k-nearest neighbors
                distances, indices = nn.kneighbors(X[j:j+1])
                neighbor_indices = indices[0]
                
                # Get detector predictions for neighbors
                if hasattr(detector, 'decision_function'):
                    neighbor_scores = detector.decision_function(X[neighbor_indices])
                    neighbor_preds = (neighbor_scores > 0).astype(int)
                elif hasattr(detector, 'predict'):
                    neighbor_preds = detector.predict(X[neighbor_indices])
                    neighbor_preds = np.where(neighbor_preds == -1, 1, 0)
                else:
                    continue
                
                # Calculate local accuracy
                local_accuracy = np.mean(neighbor_preds == y[neighbor_indices])
                competence_scores.append(local_accuracy)
            
            self.competence_regions[i] = np.array(competence_scores)
        
        self._fitted = True
    
    def select_detectors(self, x: np.ndarray, X_train: np.ndarray) -> List[int]:
        """Select best detectors for a given point.
        
        Args:
            x: Query point
            X_train: Training data for neighbor search
            
        Returns:
            Indices of selected detectors
        """
        if not self._fitted:
            raise ValueError("Selector must be fitted before use")
        
        # Find nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=self.k_neighbors)
        nn.fit(X_train)
        
        distances, indices = nn.kneighbors(x.reshape(1, -1))
        neighbor_indices = indices[0]
        
        # Calculate competence for each detector
        detector_competences = []
        for i in range(len(self.base_detectors)):
            if i in self.competence_regions:
                competence = np.mean(self.competence_regions[i][neighbor_indices])
                detector_competences.append((i, competence))
        
        # Sort by competence and select top performers
        detector_competences.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 50% of detectors
        n_select = max(1, len(detector_competences) // 2)
        selected_indices = [idx for idx, _ in detector_competences[:n_select]]
        
        return selected_indices


class BayesianEnsemble:
    """Bayesian ensemble averaging with uncertainty quantification."""
    
    def __init__(self, prior_strength: float = 1.0):
        self.prior_strength = prior_strength
        self.detector_weights = None
        self.detector_uncertainties = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, base_detectors: List[Any]) -> None:
        """Fit Bayesian ensemble weights.
        
        Args:
            X: Training data
            y: Ground truth labels
            base_detectors: List of fitted base detectors
        """
        n_detectors = len(base_detectors)
        
        # Get predictions from all detectors
        detector_scores = []
        for detector in base_detectors:
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X)
            elif hasattr(detector, 'score_samples'):
                scores = -detector.score_samples(X)
            else:
                scores = np.random.random(len(X))
            
            # Normalize scores
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            detector_scores.append(scores)
        
        detector_scores = np.array(detector_scores).T
        
        # Bayesian weight estimation
        weights = []
        uncertainties = []
        
        for i in range(n_detectors):
            scores = detector_scores[:, i]
            
            # Calculate likelihood based on AUC
            try:
                auc = roc_auc_score(y, scores)
                auc = max(auc, 1 - auc)  # Handle inverted scores
            except ValueError:
                auc = 0.5
            
            # Bayesian weight: posterior mean
            # Prior: Beta(alpha=1, beta=1), Likelihood: Bernoulli(p=auc)
            alpha_post = self.prior_strength + auc * len(X)
            beta_post = self.prior_strength + (1 - auc) * len(X)
            
            weight = alpha_post / (alpha_post + beta_post)
            uncertainty = np.sqrt(alpha_post * beta_post / 
                                ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)))
            
            weights.append(weight)
            uncertainties.append(uncertainty)
        
        # Normalize weights
        weights = np.array(weights)
        self.detector_weights = weights / np.sum(weights)
        self.detector_uncertainties = np.array(uncertainties)
        self._fitted = True
    
    def predict_with_uncertainty(self, detector_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty quantification.
        
        Args:
            detector_scores: Scores from base detectors (n_samples, n_detectors)
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self._fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Weighted ensemble prediction
        ensemble_scores = np.average(detector_scores, axis=1, weights=self.detector_weights)
        
        # Propagate uncertainty
        ensemble_uncertainties = np.sqrt(
            np.sum((self.detector_uncertainties * detector_scores)**2, axis=1)
        )
        
        return ensemble_scores, ensemble_uncertainties


class FeatureBaggingEnsemble:
    """Feature bagging ensemble for high-dimensional anomaly detection."""
    
    def __init__(self, n_estimators: int = 10, max_features: float = 0.8):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.feature_subsets = []
        self.base_detectors = []
        self._fitted = False
    
    def fit(self, X: np.ndarray, base_detector_class: Any = IsolationForest) -> None:
        """Fit feature bagging ensemble.
        
        Args:
            X: Training data
            base_detector_class: Base detector class to use
        """
        n_features = X.shape[1]
        n_select = max(1, int(self.max_features * n_features))
        
        for i in range(self.n_estimators):
            # Random feature subset
            feature_indices = np.random.choice(
                n_features, size=n_select, replace=False
            )
            self.feature_subsets.append(feature_indices)
            
            # Train detector on feature subset
            X_subset = X[:, feature_indices]
            detector = base_detector_class(random_state=i)
            detector.fit(X_subset)
            self.base_detectors.append(detector)
        
        self._fitted = True
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores using feature bagging.
        
        Args:
            X: Test data
            
        Returns:
            Ensemble anomaly scores
        """
        if not self._fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        all_scores = []
        
        for i, detector in enumerate(self.base_detectors):
            feature_indices = self.feature_subsets[i]
            X_subset = X[:, feature_indices]
            
            if hasattr(detector, 'decision_function'):
                scores = -detector.decision_function(X_subset)
            elif hasattr(detector, 'score_samples'):
                scores = -detector.score_samples(X_subset)
            else:
                scores = np.random.random(len(X))
            
            all_scores.append(scores)
        
        # Average scores
        ensemble_scores = np.mean(all_scores, axis=0)
        return ensemble_scores


class RotationForestAnomalyDetector:
    """Rotation Forest adaptation for anomaly detection."""
    
    def __init__(self, n_estimators: int = 10, n_features_per_subset: int = 3):
        self.n_estimators = n_estimators
        self.n_features_per_subset = n_features_per_subset
        self.rotation_matrices = []
        self.base_detectors = []
        self.feature_subsets = []
        self._fitted = False
    
    def fit(self, X: np.ndarray) -> None:
        """Fit rotation forest ensemble.
        
        Args:
            X: Training data
        """
        n_features = X.shape[1]
        
        for i in range(self.n_estimators):
            # Create feature subsets
            n_subsets = max(1, n_features // self.n_features_per_subset)
            feature_indices = np.arange(n_features)
            np.random.shuffle(feature_indices)
            
            subsets = []
            for j in range(n_subsets):
                start_idx = j * self.n_features_per_subset
                end_idx = min((j + 1) * self.n_features_per_subset, n_features)
                subsets.append(feature_indices[start_idx:end_idx])
            
            self.feature_subsets.append(subsets)
            
            # Create rotation matrix
            rotation_matrix = np.eye(n_features)
            
            for subset in subsets:
                if len(subset) > 1:
                    # Apply PCA to subset
                    X_subset = X[:, subset]
                    pca = PCA(n_components=len(subset))
                    pca.fit(X_subset)
                    
                    # Update rotation matrix
                    rotation_matrix[np.ix_(subset, subset)] = pca.components_
            
            self.rotation_matrices.append(rotation_matrix)
            
            # Transform data and fit detector
            X_rotated = X @ rotation_matrix
            detector = IsolationForest(random_state=i)
            detector.fit(X_rotated)
            self.base_detectors.append(detector)
        
        self._fitted = True
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict using rotation forest ensemble.
        
        Args:
            X: Test data
            
        Returns:
            Ensemble anomaly scores
        """
        if not self._fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        all_scores = []
        
        for i, detector in enumerate(self.base_detectors):
            rotation_matrix = self.rotation_matrices[i]
            X_rotated = X @ rotation_matrix
            
            scores = -detector.decision_function(X_rotated)
            all_scores.append(scores)
        
        ensemble_scores = np.mean(all_scores, axis=0)
        return ensemble_scores


class MultiObjectiveEnsemble:
    """Multi-objective ensemble optimization."""
    
    def __init__(self, objectives: List[str] = None):
        self.objectives = objectives or ['accuracy', 'diversity', 'efficiency']
        self.pareto_solutions = []
        self.selected_ensemble = None
        self._fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, base_detectors: List[Any]) -> None:
        """Fit multi-objective ensemble.
        
        Args:
            X: Training data
            y: Ground truth labels
            base_detectors: List of base detectors
        """
        # Generate candidate ensembles (all possible combinations)
        n_detectors = len(base_detectors)
        candidate_ensembles = []
        
        # Generate subset combinations
        for size in range(2, n_detectors + 1):
            from itertools import combinations
            for subset in combinations(range(n_detectors), size):
                candidate_ensembles.append(list(subset))
        
        # Evaluate each candidate ensemble
        ensemble_scores = []
        
        for ensemble_indices in candidate_ensembles:
            # Create ensemble
            ensemble_detectors = [base_detectors[i] for i in ensemble_indices]
            
            # Evaluate objectives
            objectives_scores = self._evaluate_objectives(
                X, y, ensemble_detectors, ensemble_indices
            )
            
            ensemble_scores.append({
                'indices': ensemble_indices,
                'scores': objectives_scores,
                'detectors': ensemble_detectors
            })
        
        # Find Pareto-optimal solutions
        self.pareto_solutions = self._find_pareto_optimal(ensemble_scores)
        
        # Select best ensemble (using weighted sum for now)
        if self.pareto_solutions:
            weights = [1.0 / len(self.objectives)] * len(self.objectives)
            best_score = -np.inf
            
            for solution in self.pareto_solutions:
                weighted_score = np.sum([
                    w * s for w, s in zip(weights, solution['scores'])
                ])
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    self.selected_ensemble = solution
        
        self._fitted = True
    
    def _evaluate_objectives(self, X: np.ndarray, y: np.ndarray, 
                           ensemble_detectors: List[Any], 
                           indices: List[int]) -> List[float]:
        """Evaluate ensemble on multiple objectives.
        
        Args:
            X: Data
            y: Labels
            ensemble_detectors: Ensemble detectors
            indices: Detector indices
            
        Returns:
            Objective scores
        """
        scores = []
        
        # Get ensemble predictions
        all_scores = []
        for detector in ensemble_detectors:
            if hasattr(detector, 'decision_function'):
                det_scores = detector.decision_function(X)
            elif hasattr(detector, 'score_samples'):
                det_scores = -detector.score_samples(X)
            else:
                det_scores = np.random.random(len(X))
            
            all_scores.append(det_scores)
        
        ensemble_score = np.mean(all_scores, axis=0)
        
        for objective in self.objectives:
            if objective == 'accuracy':
                try:
                    auc = roc_auc_score(y, ensemble_score)
                    scores.append(max(auc, 1 - auc))
                except ValueError:
                    scores.append(0.5)
            
            elif objective == 'diversity':
                # Calculate pairwise correlation diversity
                if len(all_scores) > 1:
                    correlations = []
                    for i in range(len(all_scores)):
                        for j in range(i + 1, len(all_scores)):
                            corr = np.corrcoef(all_scores[i], all_scores[j])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                    
                    diversity = 1 - np.mean(correlations) if correlations else 0
                    scores.append(diversity)
                else:
                    scores.append(0)
            
            elif objective == 'efficiency':
                # Efficiency = 1 / ensemble_size (prefer smaller ensembles)
                efficiency = 1.0 / len(ensemble_detectors)
                scores.append(efficiency)
            
            else:
                scores.append(0.5)  # Default score
        
        return scores
    
    def _find_pareto_optimal(self, ensemble_scores: List[Dict]) -> List[Dict]:
        """Find Pareto-optimal ensemble solutions.
        
        Args:
            ensemble_scores: List of ensemble evaluations
            
        Returns:
            Pareto-optimal solutions
        """
        pareto_optimal = []
        
        for i, candidate in enumerate(ensemble_scores):
            is_dominated = False
            
            for j, other in enumerate(ensemble_scores):
                if i != j:
                    # Check if other dominates candidate
                    dominates = True
                    for k in range(len(candidate['scores'])):
                        if other['scores'][k] <= candidate['scores'][k]:
                            dominates = False
                            break
                    
                    if dominates:
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_optimal.append(candidate)
        
        return pareto_optimal
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict using selected ensemble.
        
        Args:
            X: Test data
            
        Returns:
            Ensemble predictions
        """
        if not self._fitted or self.selected_ensemble is None:
            raise ValueError("Ensemble must be fitted before prediction")
        
        ensemble_detectors = self.selected_ensemble['detectors']
        all_scores = []
        
        for detector in ensemble_detectors:
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X)
            elif hasattr(detector, 'score_samples'):
                scores = -detector.score_samples(X)
            else:
                scores = np.random.random(len(X))
            
            all_scores.append(scores)
        
        ensemble_scores = np.mean(all_scores, axis=0)
        return ensemble_scores


class OnlineAdaptiveEnsemble:
    """Online adaptive ensemble for streaming anomaly detection."""
    
    def __init__(self, max_detectors: int = 10, adaptation_rate: float = 0.1):
        self.max_detectors = max_detectors
        self.adaptation_rate = adaptation_rate
        self.detector_pool = []
        self.detector_weights = []
        self.detector_ages = []
        self.performance_history = []
        self._fitted = False
    
    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Incrementally fit the ensemble.
        
        Args:
            X: New batch of data
            y: Optional labels for supervised adaptation
        """
        # Initialize if needed
        if not self._fitted:
            self._initialize_ensemble(X)
        
        # Update existing detectors
        self._update_detectors(X, y)
        
        # Adapt ensemble weights
        if y is not None:
            self._adapt_weights(X, y)
        
        # Manage detector pool size
        self._manage_detector_pool()
        
        self._fitted = True
    
    def _initialize_ensemble(self, X: np.ndarray) -> None:
        """Initialize the ensemble with base detectors."""
        detector_classes = [
            IsolationForest,
            lambda: LocalOutlierFactor(novelty=True),
            OneClassSVM
        ]
        
        for i, detector_class in enumerate(detector_classes):
            detector = detector_class(random_state=i)
            detector.fit(X)
            
            self.detector_pool.append(detector)
            self.detector_weights.append(1.0 / len(detector_classes))
            self.detector_ages.append(0)
            self.performance_history.append([])
    
    def _update_detectors(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Update detectors with new data."""
        for i, detector in enumerate(self.detector_pool):
            # Age detectors
            self.detector_ages[i] += 1
            
            # Some detectors support partial_fit
            if hasattr(detector, 'partial_fit'):
                try:
                    detector.partial_fit(X)
                except Exception:
                    # Retrain if partial_fit fails
                    detector.fit(X)
            else:
                # For detectors without partial_fit, retrain periodically
                if self.detector_ages[i] % 10 == 0:  # Retrain every 10 batches
                    detector.fit(X)
    
    def _adapt_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Adapt detector weights based on performance."""
        for i, detector in enumerate(self.detector_pool):
            # Get detector predictions
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X)
            elif hasattr(detector, 'score_samples'):
                scores = -detector.score_samples(X)
            else:
                continue
            
            # Calculate performance
            try:
                auc = roc_auc_score(y, scores)
                performance = max(auc, 1 - auc)
            except ValueError:
                performance = 0.5
            
            self.performance_history[i].append(performance)
            
            # Keep only recent performance
            if len(self.performance_history[i]) > 10:
                self.performance_history[i] = self.performance_history[i][-10:]
            
            # Update weight using exponential moving average
            current_weight = self.detector_weights[i]
            recent_performance = np.mean(self.performance_history[i])
            
            self.detector_weights[i] = (
                (1 - self.adaptation_rate) * current_weight +
                self.adaptation_rate * recent_performance
            )
        
        # Normalize weights
        total_weight = sum(self.detector_weights)
        if total_weight > 0:
            self.detector_weights = [w / total_weight for w in self.detector_weights]
    
    def _manage_detector_pool(self) -> None:
        """Manage detector pool size and diversity."""
        if len(self.detector_pool) > self.max_detectors:
            # Remove worst performing detectors
            performance_scores = [
                np.mean(hist) if hist else 0.0 
                for hist in self.performance_history
            ]
            
            # Find worst detector
            worst_idx = np.argmin(performance_scores)
            
            # Remove worst detector
            del self.detector_pool[worst_idx]
            del self.detector_weights[worst_idx]
            del self.detector_ages[worst_idx]
            del self.performance_history[worst_idx]
    
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores.
        
        Args:
            X: Test data
            
        Returns:
            Ensemble anomaly scores
        """
        if not self._fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        all_scores = []
        valid_weights = []
        
        for i, detector in enumerate(self.detector_pool):
            try:
                if hasattr(detector, 'decision_function'):
                    scores = detector.decision_function(X)
                elif hasattr(detector, 'score_samples'):
                    scores = -detector.score_samples(X)
                else:
                    continue
                
                all_scores.append(scores)
                valid_weights.append(self.detector_weights[i])
            
            except Exception as e:
                logger.warning(f"Detector {i} failed: {e}")
                continue
        
        if not all_scores:
            raise ValueError("No valid detector predictions available")
        
        # Weighted ensemble prediction
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / np.sum(valid_weights)
        
        ensemble_scores = np.average(all_scores, axis=0, weights=valid_weights)
        return ensemble_scores


class AdvancedEnsembleAdapter(DetectorProtocol, EnsembleDetectorProtocol):
    """Advanced ensemble methods adapter implementing state-of-the-art techniques."""
    
    _algorithm_map = {
        "DynamicEnsemble": "dynamic",
        "BayesianEnsemble": "bayesian", 
        "FeatureBaggingEnsemble": "feature_bagging",
        "RotationForestEnsemble": "rotation_forest",
        "MultiObjectiveEnsemble": "multi_objective",
        "OnlineAdaptiveEnsemble": "online_adaptive"
    }
    
    def __init__(self, detector: Detector):
        """Initialize advanced ensemble adapter.
        
        Args:
            detector: Detector entity with ensemble configuration
        """
        self.detector = detector
        self._ensemble = None
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._base_detectors = []
        self._detector_weights = {}
        self._init_algorithm()
    
    def _init_algorithm(self) -> None:
        """Initialize the advanced ensemble algorithm."""
        if self.detector.algorithm_name not in self._algorithm_map:
            available = ", ".join(self._algorithm_map.keys())
            raise AlgorithmNotFoundError(
                f"Algorithm '{self.detector.algorithm_name}' not found. "
                f"Available advanced ensemble algorithms: {available}"
            )
        
        params = self.detector.parameters.copy()
        algorithm_type = self._algorithm_map[self.detector.algorithm_name]
        
        # Create base detectors
        self._base_detectors = self._create_base_detectors(params)
        
        # Initialize specific ensemble
        try:
            if algorithm_type == "dynamic":
                k_neighbors = params.get("k_neighbors", 5)
                self._ensemble = DynamicEnsembleSelector(k_neighbors)
            
            elif algorithm_type == "bayesian":
                prior_strength = params.get("prior_strength", 1.0)
                self._ensemble = BayesianEnsemble(prior_strength)
            
            elif algorithm_type == "feature_bagging":
                n_estimators = params.get("n_estimators", 10)
                max_features = params.get("max_features", 0.8)
                self._ensemble = FeatureBaggingEnsemble(n_estimators, max_features)
            
            elif algorithm_type == "rotation_forest":
                n_estimators = params.get("n_estimators", 10)
                n_features_per_subset = params.get("n_features_per_subset", 3)
                self._ensemble = RotationForestAnomalyDetector(
                    n_estimators, n_features_per_subset
                )
            
            elif algorithm_type == "multi_objective":
                objectives = params.get("objectives", ["accuracy", "diversity", "efficiency"])
                self._ensemble = MultiObjectiveEnsemble(objectives)
            
            elif algorithm_type == "online_adaptive":
                max_detectors = params.get("max_detectors", 10)
                adaptation_rate = params.get("adaptation_rate", 0.1)
                self._ensemble = OnlineAdaptiveEnsemble(max_detectors, adaptation_rate)
            
        except Exception as e:
            raise AdapterError(
                f"Failed to initialize advanced ensemble {self.detector.algorithm_name}: {e}"
            )
    
    def _create_base_detectors(self, params: Dict[str, Any]) -> List[Any]:
        """Create base detectors for the ensemble."""
        base_algorithms = params.get(
            "base_algorithms", ["IsolationForest", "LOF", "OneClassSVM", "EllipticEnvelope"]
        )
        contamination = params.get("contamination", 0.1)
        
        detectors = []
        
        for i, alg_name in enumerate(base_algorithms):
            try:
                if alg_name == "IsolationForest":
                    detector = IsolationForest(
                        contamination=contamination,
                        random_state=42 + i,
                        n_estimators=params.get("n_estimators", 100)
                    )
                
                elif alg_name == "LOF":
                    detector = LocalOutlierFactor(
                        contamination=contamination,
                        n_neighbors=params.get("n_neighbors", 20),
                        novelty=True
                    )
                
                elif alg_name == "OneClassSVM":
                    detector = OneClassSVM(
                        nu=contamination,
                        kernel=params.get("kernel", "rbf"),
                        gamma=params.get("gamma", "scale")
                    )
                
                elif alg_name == "EllipticEnvelope":
                    detector = EllipticEnvelope(
                        contamination=contamination,
                        support_fraction=params.get("support_fraction", None)
                    )
                
                else:
                    logger.warning(f"Unknown base algorithm: {alg_name}, using IsolationForest")
                    detector = IsolationForest(
                        contamination=contamination, 
                        random_state=42 + i
                    )
                
                detectors.append(detector)
                
            except Exception as e:
                logger.warning(f"Failed to create {alg_name}: {e}, skipping")
        
        if not detectors:
            # Fallback to default ensemble
            detectors = [
                IsolationForest(contamination=contamination, random_state=42),
                LocalOutlierFactor(contamination=contamination, novelty=True),
                OneClassSVM(nu=contamination),
                EllipticEnvelope(contamination=contamination)
            ]
        
        return detectors

    @property
    def name(self) -> str:
        """Get the name of the detector."""
        return self.detector.name

    @property
    def contamination_rate(self):
        """Get the contamination rate."""
        from pynomaly.domain.value_objects import ContaminationRate
        rate = self.detector.parameters.get("contamination", 0.1)
        return ContaminationRate(rate)

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._is_fitted

    @property
    def parameters(self) -> dict[str, Any]:
        """Get the current parameters of the detector."""
        return self.detector.parameters.copy()

    @property
    def base_detectors(self) -> list[DetectorProtocol]:
        """Get the base detectors in the ensemble."""
        return self._base_detectors

    def add_detector(self, detector: DetectorProtocol, weight: float = 1.0) -> None:
        """Add a detector to the ensemble."""
        self._base_detectors.append(detector)
        self._detector_weights[detector.name] = weight

    def remove_detector(self, detector_name: str) -> None:
        """Remove a detector from the ensemble."""
        self._base_detectors = [d for d in self._base_detectors if d.name != detector_name]
        if detector_name in self._detector_weights:
            del self._detector_weights[detector_name]

    def get_detector_weights(self) -> dict[str, float]:
        """Get weights of all detectors in the ensemble."""
        return self._detector_weights.copy()
    
    def fit(self, dataset: Dataset) -> None:
        """Fit the advanced ensemble detector."""
        try:
            # Prepare data
            X_train = self._prepare_data(dataset)
            X_scaled = self._scaler.fit_transform(X_train)
            
            # Get labels if available
            y_train = None
            if (dataset.target_column and 
                dataset.target_column in dataset.data.columns):
                y_train = dataset.data[dataset.target_column].values
            
            # Fit base detectors first
            for detector in self._base_detectors:
                detector.fit(X_scaled)
            
            # Fit ensemble using appropriate method
            algorithm_type = self._algorithm_map[self.detector.algorithm_name]
            
            if algorithm_type in ["dynamic", "bayesian", "multi_objective"]:
                if y_train is not None:
                    self._ensemble.fit(X_scaled, y_train, self._base_detectors)
                else:
                    logger.warning(f"{self.detector.algorithm_name} requires labels but none provided")
                    # Fallback: create synthetic labels based on isolation forest
                    if_detector = IsolationForest(contamination=0.1, random_state=42)
                    if_detector.fit(X_scaled)
                    y_synthetic = if_detector.predict(X_scaled)
                    y_synthetic = np.where(y_synthetic == -1, 1, 0)
                    self._ensemble.fit(X_scaled, y_synthetic, self._base_detectors)
            
            elif algorithm_type == "feature_bagging":
                base_detector_class = self.detector.parameters.get(
                    "base_detector_class", IsolationForest
                )
                self._ensemble.fit(X_scaled, base_detector_class)
            
            elif algorithm_type == "rotation_forest":
                self._ensemble.fit(X_scaled)
            
            elif algorithm_type == "online_adaptive":
                self._ensemble.partial_fit(X_scaled, y_train)
            
            self.detector.is_fitted = True
            self._is_fitted = True
            
            logger.info(
                f"Successfully fitted advanced ensemble detector: {self.detector.algorithm_name}"
            )
            
        except Exception as e:
            raise AdapterError(f"Failed to fit advanced ensemble model: {e}")
    
    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies using the advanced ensemble."""
        return self.predict(dataset)
    
    def predict(self, dataset: Dataset) -> DetectionResult:
        """Predict anomalies using the advanced ensemble."""
        if not self._is_fitted or self._ensemble is None:
            raise AdapterError("Model must be fitted before prediction")
        
        try:
            # Prepare data
            X_test = self._prepare_data(dataset)
            X_scaled = self._scaler.transform(X_test)
            
            # Get ensemble predictions
            algorithm_type = self._algorithm_map[self.detector.algorithm_name]
            
            if algorithm_type == "dynamic":
                # Dynamic selection needs original training data
                # For simplicity, use all detectors if training data not available
                selected_indices = list(range(len(self._base_detectors)))
                selected_detectors = [self._base_detectors[i] for i in selected_indices]
                
                all_scores = []
                for detector in selected_detectors:
                    if hasattr(detector, 'decision_function'):
                        scores = -detector.decision_function(X_scaled)
                    elif hasattr(detector, 'score_samples'):
                        scores = -detector.score_samples(X_scaled)
                    else:
                        scores = np.random.random(len(X_scaled))
                    all_scores.append(scores)
                
                ensemble_scores = np.mean(all_scores, axis=0)
                uncertainties = None
            
            elif algorithm_type == "bayesian":
                # Get predictions from base detectors
                detector_scores = []
                for detector in self._base_detectors:
                    if hasattr(detector, 'decision_function'):
                        scores = detector.decision_function(X_scaled)
                    elif hasattr(detector, 'score_samples'):
                        scores = -detector.score_samples(X_scaled)
                    else:
                        scores = np.random.random(len(X_scaled))
                    
                    # Normalize scores
                    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
                    detector_scores.append(scores)
                
                detector_scores = np.array(detector_scores).T
                ensemble_scores, uncertainties = self._ensemble.predict_with_uncertainty(
                    detector_scores
                )
            
            else:
                ensemble_scores = self._ensemble.predict_scores(X_scaled)
                uncertainties = None
            
            # Calculate threshold and labels
            contamination = self.detector.parameters.get("contamination", 0.1)
            threshold = np.percentile(ensemble_scores, (1 - contamination) * 100)
            labels = (ensemble_scores > threshold).astype(int)
            
            # Create anomaly scores
            anomaly_scores = [
                AnomalyScore(value=float(score), method=self.detector.algorithm_name)
                for score in ensemble_scores
            ]
            
            # Create anomaly objects
            from pynomaly.domain.entities.anomaly import Anomaly
            
            anomalies = []
            anomaly_indices = np.where(labels == 1)[0]
            
            for idx in anomaly_indices:
                X_test = self._prepare_data(dataset)
                data_row = {
                    f"feature_{i}": float(X_test[idx, i])
                    for i in range(X_test.shape[1])
                }
                
                metadata = {
                    "advanced_ensemble_detection": True,
                    "row_index": int(idx),
                    "algorithm_type": algorithm_type
                }
                
                if uncertainties is not None:
                    metadata["uncertainty"] = float(uncertainties[idx])
                
                anomaly = Anomaly(
                    score=anomaly_scores[idx],
                    data_point=data_row,
                    detector_name=self.detector.algorithm_name,
                    metadata=metadata
                )
                anomalies.append(anomaly)
            
            # Prepare metadata
            result_metadata = {
                "algorithm": self.detector.algorithm_name,
                "algorithm_type": algorithm_type,
                "n_base_detectors": len(self._base_detectors),
                "base_algorithms": self.detector.parameters.get("base_algorithms", []),
                "n_anomalies": int(np.sum(labels)),
                "contamination_rate": float(np.sum(labels) / len(labels)),
                "model_type": "advanced_ensemble"
            }
            
            if hasattr(self._ensemble, 'detector_weights'):
                result_metadata["detector_weights"] = self._ensemble.detector_weights
            
            if uncertainties is not None:
                result_metadata["has_uncertainty"] = True
                result_metadata["mean_uncertainty"] = float(np.mean(uncertainties))
            
            return DetectionResult(
                detector_id=self.detector.id,
                dataset_id=dataset.id,
                anomalies=anomalies,
                scores=anomaly_scores,
                labels=labels,
                threshold=float(threshold),
                metadata=result_metadata
            )
            
        except Exception as e:
            raise AdapterError(f"Failed to predict with advanced ensemble model: {e}")
    
    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores for the dataset."""
        result = self.predict(dataset)
        return result.scores
    
    def fit_detect(self, dataset: Dataset) -> DetectionResult:
        """Fit the detector and detect anomalies in one step."""
        self.fit(dataset)
        return self.detect(dataset)
    
    def get_params(self) -> dict[str, Any]:
        """Get parameters of the detector."""
        return self.detector.parameters.copy()
    
    def set_params(self, **params: Any) -> None:
        """Set parameters of the detector."""
        self.detector.parameters.update(params)
        # Reinitialize if needed
        if self._is_fitted:
            self._init_algorithm()
    
    def _prepare_data(self, dataset: Dataset) -> np.ndarray:
        """Prepare data for ensemble model."""
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
        
        # Simple imputation
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])
        
        return X
    
    @classmethod
    def get_supported_algorithms(cls) -> list[str]:
        """Get list of supported advanced ensemble algorithms."""
        return list(cls._algorithm_map.keys())
    
    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> dict[str, Any]:
        """Get information about a specific advanced ensemble algorithm."""
        if algorithm not in cls._algorithm_map:
            raise AlgorithmNotFoundError(f"Algorithm '{algorithm}' not found")
        
        info = {
            "DynamicEnsemble": {
                "name": "Dynamic Ensemble Selection",
                "type": "Advanced Ensemble",
                "description": "Selects best detectors based on local competence regions",
                "parameters": {
                    "k_neighbors": {"type": "int", "default": 5, "description": "Number of neighbors for competence calculation"},
                    "base_algorithms": {"type": "list", "default": ["IsolationForest", "LOF", "OneClassSVM"], "description": "Base algorithms"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["variable_data_patterns", "local_competence", "adaptive_selection"],
                "pros": ["Adapts to local data patterns", "High accuracy", "Intelligent selection"],
                "cons": ["Requires labeled data", "Computational overhead", "Complex setup"]
            },
            "BayesianEnsemble": {
                "name": "Bayesian Ensemble Averaging",
                "type": "Advanced Ensemble", 
                "description": "Uses Bayesian statistics to weight detectors and quantify uncertainty",
                "parameters": {
                    "prior_strength": {"type": "float", "default": 1.0, "description": "Strength of prior belief"},
                    "base_algorithms": {"type": "list", "default": ["IsolationForest", "LOF", "OneClassSVM"], "description": "Base algorithms"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["uncertainty_quantification", "probabilistic_reasoning", "risk_assessment"],
                "pros": ["Uncertainty quantification", "Principled weighting", "Robust to outliers"],
                "cons": ["Complex interpretation", "Requires labeled data", "Computational cost"]
            },
            "FeatureBaggingEnsemble": {
                "name": "Feature Bagging Ensemble",
                "type": "Advanced Ensemble",
                "description": "Creates diverse detectors using random feature subsets",
                "parameters": {
                    "n_estimators": {"type": "int", "default": 10, "description": "Number of estimators"},
                    "max_features": {"type": "float", "default": 0.8, "description": "Fraction of features per estimator"},
                    "base_detector_class": {"type": "class", "default": "IsolationForest", "description": "Base detector class"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["high_dimensional_data", "feature_diversity", "robust_detection"],
                "pros": ["Handles high dimensions", "Feature diversity", "Reduced overfitting"],
                "cons": ["May lose important features", "Parameter tuning needed"]
            },
            "RotationForestEnsemble": {
                "name": "Rotation Forest for Anomaly Detection",
                "type": "Advanced Ensemble",
                "description": "Uses feature rotation and PCA for diverse detectors",
                "parameters": {
                    "n_estimators": {"type": "int", "default": 10, "description": "Number of estimators"},
                    "n_features_per_subset": {"type": "int", "default": 3, "description": "Features per subset for rotation"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["feature_rotation", "dimensional_diversity", "complex_patterns"],
                "pros": ["Feature space diversity", "Handles correlations", "Good for complex data"],
                "cons": ["Complex implementation", "Parameter sensitive", "Computational overhead"]
            },
            "MultiObjectiveEnsemble": {
                "name": "Multi-Objective Ensemble Optimization",
                "type": "Advanced Ensemble",
                "description": "Optimizes ensemble for multiple objectives (accuracy, diversity, efficiency)",
                "parameters": {
                    "objectives": {"type": "list", "default": ["accuracy", "diversity", "efficiency"], "description": "Optimization objectives"},
                    "base_algorithms": {"type": "list", "default": ["IsolationForest", "LOF", "OneClassSVM"], "description": "Base algorithms"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["multi_criteria_optimization", "balanced_performance", "pareto_optimal"],
                "pros": ["Multi-objective optimization", "Pareto-optimal solutions", "Balanced trade-offs"],
                "cons": ["Complex optimization", "Requires labeled data", "Parameter selection"]
            },
            "OnlineAdaptiveEnsemble": {
                "name": "Online Adaptive Ensemble",
                "type": "Advanced Ensemble",
                "description": "Adapts ensemble weights in real-time for streaming data",
                "parameters": {
                    "max_detectors": {"type": "int", "default": 10, "description": "Maximum number of detectors"},
                    "adaptation_rate": {"type": "float", "default": 0.1, "description": "Learning rate for adaptation"},
                    "contamination": {"type": "float", "default": 0.1, "description": "Expected anomaly rate"}
                },
                "suitable_for": ["streaming_data", "concept_drift", "real_time_adaptation"],
                "pros": ["Online learning", "Adapts to drift", "Real-time updates"],
                "cons": ["Memory requirements", "Parameter tuning", "Complexity management"]
            }
        }
        
        return info.get(algorithm, {})
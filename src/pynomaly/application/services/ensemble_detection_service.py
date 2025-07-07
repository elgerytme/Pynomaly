"""Advanced ensemble methods and model stacking for anomaly detection."""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import numpy as np

from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.domain.exceptions import ValidationError
from pynomaly.domain.value_objects.algorithm_config import AlgorithmConfig
from pynomaly.domain.value_objects.score import Score

# Optional ML libraries
try:
    from sklearn.ensemble import VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class EnsembleMethod(Enum):
    """Ensemble combination methods."""
    
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTING = "majority_voting"
    SOFT_VOTING = "soft_voting"
    STACKING = "stacking"
    DYNAMIC_SELECTION = "dynamic_selection"
    BAYESIAN_MODEL_AVERAGING = "bayesian_model_averaging"


class WeightingStrategy(Enum):
    """Strategies for calculating ensemble weights."""
    
    EQUAL = "equal"
    PERFORMANCE_BASED = "performance_based"
    DIVERSITY_BASED = "diversity_based"
    CONFIDENCE_BASED = "confidence_based"
    ADAPTIVE = "adaptive"
    CROSS_VALIDATION = "cross_validation"


class DiversityMetric(Enum):
    """Diversity metrics for ensemble construction."""
    
    CORRELATION = "correlation"
    DISAGREEMENT = "disagreement"
    ENTROPY = "entropy"
    KL_DIVERGENCE = "kl_divergence"
    COSINE_SIMILARITY = "cosine_similarity"


@dataclass
class EnsembleMember:
    """Individual member of an ensemble."""
    
    detector: Detector
    weight: float = 1.0
    performance_score: float = 0.0
    diversity_score: float = 0.0
    confidence: float = 0.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def effective_weight(self) -> float:
        """Get effective weight considering if member is enabled."""
        return self.weight if self.enabled else 0.0


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    
    # Ensemble method configuration
    method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE
    weighting_strategy: WeightingStrategy = WeightingStrategy.PERFORMANCE_BASED
    diversity_metric: DiversityMetric = DiversityMetric.CORRELATION
    
    # Member selection
    min_members: int = 3
    max_members: int = 10
    diversity_threshold: float = 0.1
    performance_threshold: float = 0.5
    
    # Stacking configuration
    meta_learner: str = "logistic_regression"
    use_features_in_stacking: bool = True
    stacking_cv_folds: int = 5
    
    # Dynamic selection
    selection_window_size: int = 100
    adaptation_rate: float = 0.1
    
    # Weight optimization
    optimize_weights: bool = True
    weight_optimization_method: str = "differential_evolution"
    max_optimization_iterations: int = 100
    
    # Quality control
    enable_member_pruning: bool = True
    pruning_threshold: float = 0.05
    rebalance_frequency: int = 1000
    
    # Uncertainty quantification
    estimate_uncertainty: bool = True
    uncertainty_method: str = "ensemble_variance"
    bootstrap_samples: int = 100


@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    
    ensemble_id: UUID
    final_score: float
    final_prediction: int
    member_scores: Dict[str, float]
    member_predictions: Dict[str, int]
    member_weights: Dict[str, float]
    uncertainty: float = 0.0
    confidence: float = 0.0
    method_used: str = ""
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EnsemblePerformance:
    """Performance metrics for ensemble."""
    
    ensemble_score: float
    individual_scores: Dict[str, float]
    diversity_score: float
    improvement_over_best: float
    member_correlations: np.ndarray
    weight_distribution: Dict[str, float]
    stability_score: float
    robustness_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class EnsembleCombiner(ABC):
    """Abstract base class for ensemble combination methods."""
    
    @abstractmethod
    async def combine_predictions(
        self,
        member_scores: Dict[str, float],
        member_predictions: Dict[str, int],
        member_weights: Dict[str, float],
        features: Optional[np.ndarray] = None,
    ) -> Tuple[float, int, float]:
        """Combine member predictions into final result.
        
        Args:
            member_scores: Scores from ensemble members
            member_predictions: Predictions from ensemble members
            member_weights: Weights for ensemble members
            features: Optional input features for stacking
            
        Returns:
            Tuple of (final_score, final_prediction, confidence)
        """
        pass


class SimpleAverageCombiner(EnsembleCombiner):
    """Simple average combination method."""
    
    async def combine_predictions(
        self,
        member_scores: Dict[str, float],
        member_predictions: Dict[str, int],
        member_weights: Dict[str, float],
        features: Optional[np.ndarray] = None,
    ) -> Tuple[float, int, float]:
        
        if not member_scores:
            return 0.0, 0, 0.0
        
        # Simple average of scores
        scores = list(member_scores.values())
        final_score = np.mean(scores)
        
        # Majority vote for prediction
        predictions = list(member_predictions.values())
        final_prediction = int(np.round(np.mean(predictions)))
        
        # Confidence based on agreement
        agreement = np.mean([p == final_prediction for p in predictions])
        confidence = float(agreement)
        
        return final_score, final_prediction, confidence


class WeightedAverageCombiner(EnsembleCombiner):
    """Weighted average combination method."""
    
    async def combine_predictions(
        self,
        member_scores: Dict[str, float],
        member_predictions: Dict[str, int],
        member_weights: Dict[str, float],
        features: Optional[np.ndarray] = None,
    ) -> Tuple[float, int, float]:
        
        if not member_scores:
            return 0.0, 0, 0.0
        
        # Weighted average of scores
        weighted_score_sum = 0.0
        weighted_pred_sum = 0.0
        total_weight = 0.0
        
        for member_id in member_scores:
            weight = member_weights.get(member_id, 1.0)
            weighted_score_sum += member_scores[member_id] * weight
            weighted_pred_sum += member_predictions[member_id] * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0, 0, 0.0
        
        final_score = weighted_score_sum / total_weight
        final_prediction = int(np.round(weighted_pred_sum / total_weight))
        
        # Confidence based on weighted agreement
        weighted_agreement = sum(
            member_weights.get(member_id, 1.0) * (member_predictions[member_id] == final_prediction)
            for member_id in member_predictions
        ) / total_weight
        
        confidence = float(weighted_agreement)
        
        return final_score, final_prediction, confidence


class StackingCombiner(EnsembleCombiner):
    """Stacking (meta-learning) combination method."""
    
    def __init__(self, meta_learner: str = "logistic_regression"):
        self.meta_learner_name = meta_learner
        self.meta_learner = None
        self.is_trained = False
    
    def _create_meta_learner(self):
        """Create meta-learner model."""
        if not SKLEARN_AVAILABLE:
            raise ValidationError("scikit-learn required for stacking")
        
        if self.meta_learner_name == "logistic_regression":
            return LogisticRegression(random_state=42)
        elif self.meta_learner_name == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(random_state=42)
        else:
            return LogisticRegression(random_state=42)
    
    async def train_meta_learner(
        self,
        training_predictions: List[Dict[str, float]],
        training_labels: List[int],
        training_features: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Train the meta-learner on training data."""
        
        if not training_predictions:
            raise ValidationError("No training data provided for stacking")
        
        # Create meta-learner if not exists
        if self.meta_learner is None:
            self.meta_learner = self._create_meta_learner()
        
        # Prepare meta-features
        meta_features = []
        for i, pred_dict in enumerate(training_predictions):
            # Base learner predictions
            base_features = list(pred_dict.values())
            
            # Optionally include original features
            if training_features and i < len(training_features):
                if training_features[i] is not None:
                    original_features = training_features[i].flatten()
                    base_features.extend(original_features)
            
            meta_features.append(base_features)
        
        X_meta = np.array(meta_features)
        y_meta = np.array(training_labels)
        
        # Train meta-learner
        self.meta_learner.fit(X_meta, y_meta)
        self.is_trained = True
    
    async def combine_predictions(
        self,
        member_scores: Dict[str, float],
        member_predictions: Dict[str, int],
        member_weights: Dict[str, float],
        features: Optional[np.ndarray] = None,
    ) -> Tuple[float, int, float]:
        
        if not self.is_trained:
            # Fallback to weighted average if not trained
            combiner = WeightedAverageCombiner()
            return await combiner.combine_predictions(
                member_scores, member_predictions, member_weights, features
            )
        
        # Prepare meta-features
        base_features = list(member_scores.values())
        
        if features is not None:
            original_features = features.flatten()
            base_features.extend(original_features)
        
        X_meta = np.array(base_features).reshape(1, -1)
        
        # Get prediction from meta-learner
        try:
            prediction = self.meta_learner.predict(X_meta)[0]
            
            # Get prediction probabilities if available
            if hasattr(self.meta_learner, 'predict_proba'):
                probabilities = self.meta_learner.predict_proba(X_meta)[0]
                confidence = float(max(probabilities))
                # Use probability of anomaly class as score
                if len(probabilities) > 1:
                    final_score = float(probabilities[1])  # Assuming class 1 is anomaly
                else:
                    final_score = float(prediction)
            else:
                final_score = float(prediction)
                confidence = 0.7  # Default confidence
            
            return final_score, int(prediction), confidence
            
        except Exception as e:
            # Fallback to weighted average on error
            logging.warning(f"Stacking prediction failed: {e}")
            combiner = WeightedAverageCombiner()
            return await combiner.combine_predictions(
                member_scores, member_predictions, member_weights, features
            )


class DynamicSelectionCombiner(EnsembleCombiner):
    """Dynamic ensemble selection based on local competence."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: Dict[str, List[float]] = {}
    
    def update_performance(self, member_id: str, performance: float) -> None:
        """Update performance history for a member."""
        if member_id not in self.performance_history:
            self.performance_history[member_id] = []
        
        self.performance_history[member_id].append(performance)
        
        # Keep only recent history
        if len(self.performance_history[member_id]) > self.window_size:
            self.performance_history[member_id] = self.performance_history[member_id][-self.window_size:]
    
    async def combine_predictions(
        self,
        member_scores: Dict[str, float],
        member_predictions: Dict[str, int],
        member_weights: Dict[str, float],
        features: Optional[np.ndarray] = None,
    ) -> Tuple[float, int, float]:
        
        if not member_scores:
            return 0.0, 0, 0.0
        
        # Calculate dynamic weights based on recent performance
        dynamic_weights = {}
        for member_id in member_scores:
            if member_id in self.performance_history and self.performance_history[member_id]:
                recent_performance = np.mean(self.performance_history[member_id][-10:])
                dynamic_weights[member_id] = recent_performance
            else:
                dynamic_weights[member_id] = member_weights.get(member_id, 1.0)
        
        # Normalize weights
        total_weight = sum(dynamic_weights.values())
        if total_weight > 0:
            dynamic_weights = {k: v / total_weight for k, v in dynamic_weights.items()}
        
        # Use weighted average with dynamic weights
        combiner = WeightedAverageCombiner()
        return await combiner.combine_predictions(
            member_scores, member_predictions, dynamic_weights, features
        )


class EnsembleDetectionService:
    """Advanced ensemble detection service with multiple combination methods."""
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize ensemble detection service.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config or EnsembleConfig()
        self.logger = logging.getLogger(__name__)
        
        # Ensemble members
        self.members: List[EnsembleMember] = []
        self.ensemble_id = uuid4()
        
        # Combiners
        self._combiners = self._initialize_combiners()
        self._current_combiner = self._combiners[self.config.method]
        
        # Performance tracking
        self._performance_history: List[EnsemblePerformance] = []
        
        # Calibration data
        self._calibration_scores: List[np.ndarray] = []
        self._calibration_labels: List[np.ndarray] = []
    
    def _initialize_combiners(self) -> Dict[EnsembleMethod, EnsembleCombiner]:
        """Initialize ensemble combiners."""
        combiners = {
            EnsembleMethod.SIMPLE_AVERAGE: SimpleAverageCombiner(),
            EnsembleMethod.WEIGHTED_AVERAGE: WeightedAverageCombiner(),
            EnsembleMethod.STACKING: StackingCombiner(self.config.meta_learner),
            EnsembleMethod.DYNAMIC_SELECTION: DynamicSelectionCombiner(
                self.config.selection_window_size
            ),
        }
        
        # Add more combiners as needed
        combiners[EnsembleMethod.MAJORITY_VOTING] = SimpleAverageCombiner()
        combiners[EnsembleMethod.SOFT_VOTING] = WeightedAverageCombiner()
        
        return combiners
    
    async def add_member(
        self,
        detector: Detector,
        weight: float = 1.0,
        performance_score: float = 0.0,
    ) -> str:
        """Add a member to the ensemble.
        
        Args:
            detector: Detector to add
            weight: Initial weight for the member
            performance_score: Initial performance score
            
        Returns:
            Member ID
        """
        if len(self.members) >= self.config.max_members:
            raise ValidationError(f"Maximum ensemble members ({self.config.max_members}) reached")
        
        member = EnsembleMember(
            detector=detector,
            weight=weight,
            performance_score=performance_score,
        )
        
        self.members.append(member)
        self.logger.info(f"Added ensemble member: {detector.name}")
        
        # Rebalance weights if needed
        if self.config.optimize_weights:
            await self._optimize_weights()
        
        return str(detector.id)
    
    async def remove_member(self, detector_id: UUID) -> bool:
        """Remove a member from the ensemble.
        
        Args:
            detector_id: ID of detector to remove
            
        Returns:
            True if member was removed
        """
        for i, member in enumerate(self.members):
            if member.detector.id == detector_id:
                removed_member = self.members.pop(i)
                self.logger.info(f"Removed ensemble member: {removed_member.detector.name}")
                
                # Rebalance remaining members
                if self.config.optimize_weights and len(self.members) > 0:
                    await self._optimize_weights()
                
                return True
        
        return False
    
    async def predict(
        self,
        data: np.ndarray,
        return_individual: bool = False,
    ) -> Union[EnsembleResult, Tuple[EnsembleResult, Dict[str, Any]]]:
        """Make ensemble prediction.
        
        Args:
            data: Input data for prediction
            return_individual: Whether to return individual member results
            
        Returns:
            Ensemble result, optionally with individual results
        """
        if not self.members:
            raise ValidationError("No ensemble members available")
        
        start_time = datetime.utcnow()
        
        # Get predictions from all enabled members
        member_scores = {}
        member_predictions = {}
        member_weights = {}
        individual_results = {}
        
        for member in self.members:
            if not member.enabled:
                continue
            
            try:
                # Get prediction from member (placeholder)
                # In practice, this would use the actual detector
                score = np.random.random()
                prediction = int(score > 0.5)
                
                member_id = str(member.detector.id)
                member_scores[member_id] = score
                member_predictions[member_id] = prediction
                member_weights[member_id] = member.effective_weight
                
                if return_individual:
                    individual_results[member_id] = {
                        "detector_name": member.detector.name,
                        "score": score,
                        "prediction": prediction,
                        "weight": member.effective_weight,
                        "confidence": member.confidence,
                    }
                
            except Exception as e:
                self.logger.warning(f"Member {member.detector.name} prediction failed: {e}")
                continue
        
        if not member_scores:
            raise ValidationError("No successful predictions from ensemble members")
        
        # Combine predictions
        final_score, final_prediction, confidence = await self._current_combiner.combine_predictions(
            member_scores, member_predictions, member_weights, data
        )
        
        # Calculate uncertainty if enabled
        uncertainty = 0.0
        if self.config.estimate_uncertainty:
            uncertainty = await self._estimate_uncertainty(member_scores, member_predictions)
        
        # Create result
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        result = EnsembleResult(
            ensemble_id=self.ensemble_id,
            final_score=final_score,
            final_prediction=final_prediction,
            member_scores=member_scores,
            member_predictions=member_predictions,
            member_weights=member_weights,
            uncertainty=uncertainty,
            confidence=confidence,
            method_used=self.config.method.value,
            processing_time_ms=processing_time,
            metadata={
                "num_members": len(member_scores),
                "enabled_members": len([m for m in self.members if m.enabled]),
                "total_members": len(self.members),
            },
        )
        
        if return_individual:
            return result, individual_results
        return result
    
    async def train_ensemble(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
    ) -> EnsemblePerformance:
        """Train the ensemble on data.
        
        Args:
            training_data: Training dataset
            validation_data: Optional validation dataset
            
        Returns:
            Ensemble performance metrics
        """
        if not self.members:
            raise ValidationError("No ensemble members to train")
        
        self.logger.info(f"Training ensemble with {len(self.members)} members")
        
        # Train individual members (placeholder)
        # In practice, this would train actual detectors
        
        # Prepare data for stacking if needed
        if self.config.method == EnsembleMethod.STACKING:
            await self._prepare_stacking_data(training_data)
        
        # Optimize weights
        if self.config.optimize_weights:
            await self._optimize_weights(training_data, validation_data)
        
        # Calculate diversity scores
        await self._calculate_diversity_scores(training_data)
        
        # Evaluate ensemble performance
        performance = await self._evaluate_ensemble_performance(validation_data or training_data)
        
        self._performance_history.append(performance)
        
        self.logger.info(f"Ensemble training completed. Performance: {performance.ensemble_score:.3f}")
        
        return performance
    
    async def _prepare_stacking_data(self, training_data: Dataset) -> None:
        """Prepare data for stacking meta-learner."""
        
        if self.config.method != EnsembleMethod.STACKING:
            return
        
        stacking_combiner = self._combiners[EnsembleMethod.STACKING]
        if not isinstance(stacking_combiner, StackingCombiner):
            return
        
        # Generate cross-validation predictions for stacking
        # This is a placeholder - in practice, you'd use actual CV
        
        X = training_data.data if hasattr(training_data, 'data') else np.random.randn(100, 10)
        # Generate synthetic labels for demonstration
        y = np.random.randint(0, 2, len(X))
        
        # Collect predictions from all members
        training_predictions = []
        training_features = []
        
        for i in range(len(X)):
            member_preds = {}
            for member in self.members:
                if member.enabled:
                    # Placeholder prediction
                    score = np.random.random()
                    member_preds[str(member.detector.id)] = score
            
            training_predictions.append(member_preds)
            
            if self.config.use_features_in_stacking:
                training_features.append(X[i])
            else:
                training_features.append(None)
        
        # Train meta-learner
        await stacking_combiner.train_meta_learner(
            training_predictions, y.tolist(), training_features
        )
    
    async def _optimize_weights(
        self,
        training_data: Optional[Dataset] = None,
        validation_data: Optional[Dataset] = None,
    ) -> None:
        """Optimize ensemble weights."""
        
        if self.config.weighting_strategy == WeightingStrategy.EQUAL:
            # Set equal weights
            for member in self.members:
                member.weight = 1.0 / len(self.members) if member.enabled else 0.0
        
        elif self.config.weighting_strategy == WeightingStrategy.PERFORMANCE_BASED:
            # Weight based on individual performance
            total_performance = sum(m.performance_score for m in self.members if m.enabled)
            
            if total_performance > 0:
                for member in self.members:
                    if member.enabled:
                        member.weight = member.performance_score / total_performance
                    else:
                        member.weight = 0.0
            else:
                # Fallback to equal weights
                for member in self.members:
                    member.weight = 1.0 / len(self.members) if member.enabled else 0.0
        
        elif self.config.weighting_strategy == WeightingStrategy.DIVERSITY_BASED:
            # Weight based on diversity contribution
            await self._calculate_diversity_based_weights()
        
        elif self.config.weighting_strategy == WeightingStrategy.ADAPTIVE:
            # Adaptive weights based on recent performance
            await self._calculate_adaptive_weights()
        
        # Normalize weights
        total_weight = sum(m.weight for m in self.members if m.enabled)
        if total_weight > 0:
            for member in self.members:
                if member.enabled:
                    member.weight /= total_weight
    
    async def _calculate_diversity_scores(self, dataset: Dataset) -> None:
        """Calculate diversity scores for ensemble members."""
        
        if len(self.members) < 2:
            return
        
        # Get predictions from all members
        member_predictions = {}
        
        for member in self.members:
            if member.enabled:
                # Placeholder: generate synthetic predictions
                predictions = np.random.random(100)
                member_predictions[str(member.detector.id)] = predictions
        
        # Calculate pairwise diversity
        member_ids = list(member_predictions.keys())
        n_members = len(member_ids)
        
        if n_members < 2:
            return
        
        # Calculate correlation-based diversity
        correlation_matrix = np.zeros((n_members, n_members))
        
        for i, id1 in enumerate(member_ids):
            for j, id2 in enumerate(member_ids):
                if i != j:
                    corr = np.corrcoef(
                        member_predictions[id1],
                        member_predictions[id2]
                    )[0, 1]
                    correlation_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0.0
        
        # Update diversity scores
        for i, member in enumerate(self.members):
            if member.enabled and str(member.detector.id) in member_ids:
                member_idx = member_ids.index(str(member.detector.id))
                # Diversity is inverse of average correlation with other members
                avg_correlation = np.mean(correlation_matrix[member_idx, :])
                member.diversity_score = 1.0 - avg_correlation
    
    async def _calculate_diversity_based_weights(self) -> None:
        """Calculate weights based on diversity contribution."""
        
        total_diversity = sum(m.diversity_score for m in self.members if m.enabled)
        
        if total_diversity > 0:
            for member in self.members:
                if member.enabled:
                    # Combine performance and diversity
                    combined_score = (
                        member.performance_score * 0.7 + 
                        member.diversity_score * 0.3
                    )
                    member.weight = combined_score
                else:
                    member.weight = 0.0
        else:
            # Fallback to performance-based
            for member in self.members:
                member.weight = member.performance_score if member.enabled else 0.0
    
    async def _calculate_adaptive_weights(self) -> None:
        """Calculate adaptive weights based on recent performance."""
        
        # This would use recent performance history in practice
        # For now, use a simple adaptation
        
        for member in self.members:
            if member.enabled:
                # Simulate recent performance adaptation
                recent_performance = member.performance_score * np.random.uniform(0.8, 1.2)
                
                # Exponential moving average
                alpha = self.config.adaptation_rate
                member.weight = alpha * recent_performance + (1 - alpha) * member.weight
            else:
                member.weight = 0.0
    
    async def _estimate_uncertainty(
        self,
        member_scores: Dict[str, float],
        member_predictions: Dict[str, int],
    ) -> float:
        """Estimate prediction uncertainty."""
        
        if self.config.uncertainty_method == "ensemble_variance":
            # Variance of member scores
            scores = list(member_scores.values())
            if len(scores) > 1:
                return float(np.var(scores))
            else:
                return 0.0
        
        elif self.config.uncertainty_method == "prediction_entropy":
            # Entropy of prediction distribution
            predictions = list(member_predictions.values())
            if predictions:
                unique, counts = np.unique(predictions, return_counts=True)
                probabilities = counts / len(predictions)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
                return float(entropy)
            else:
                return 0.0
        
        else:
            # Default: standard deviation of scores
            scores = list(member_scores.values())
            return float(np.std(scores)) if len(scores) > 1 else 0.0
    
    async def _evaluate_ensemble_performance(self, dataset: Dataset) -> EnsemblePerformance:
        """Evaluate ensemble performance."""
        
        # Get ensemble predictions (placeholder)
        X = dataset.data if hasattr(dataset, 'data') else np.random.randn(100, 10)
        ensemble_scores = []
        individual_scores = {}
        
        # Simulate ensemble and individual predictions
        for i in range(len(X)):
            result = await self.predict(X[i:i+1])
            ensemble_scores.append(result.final_score)
        
        # Simulate individual member performance
        for member in self.members:
            if member.enabled:
                individual_scores[str(member.detector.id)] = member.performance_score
        
        # Calculate diversity metrics
        member_correlations = np.eye(len(self.members))  # Placeholder
        
        # Calculate improvement over best individual
        best_individual_score = max(individual_scores.values()) if individual_scores else 0.0
        ensemble_score = np.mean(ensemble_scores) if ensemble_scores else 0.0
        improvement = ensemble_score - best_individual_score
        
        # Calculate weight distribution
        weight_distribution = {
            str(m.detector.id): m.weight 
            for m in self.members if m.enabled
        }
        
        # Calculate stability (placeholder)
        stability_score = 0.9  # Would be calculated from actual predictions
        
        # Calculate robustness (placeholder)
        robustness_score = 0.8  # Would be calculated from noise resistance tests
        
        return EnsemblePerformance(
            ensemble_score=ensemble_score,
            individual_scores=individual_scores,
            diversity_score=np.mean([m.diversity_score for m in self.members if m.enabled]),
            improvement_over_best=improvement,
            member_correlations=member_correlations,
            weight_distribution=weight_distribution,
            stability_score=stability_score,
            robustness_score=robustness_score,
        )
    
    async def get_ensemble_info(self) -> Dict[str, Any]:
        """Get ensemble information and statistics."""
        
        enabled_members = [m for m in self.members if m.enabled]
        
        return {
            "ensemble_id": str(self.ensemble_id),
            "total_members": len(self.members),
            "enabled_members": len(enabled_members),
            "method": self.config.method.value,
            "weighting_strategy": self.config.weighting_strategy.value,
            "members": [
                {
                    "detector_id": str(m.detector.id),
                    "detector_name": m.detector.name,
                    "algorithm": m.detector.algorithm_config.name,
                    "weight": m.weight,
                    "performance_score": m.performance_score,
                    "diversity_score": m.diversity_score,
                    "enabled": m.enabled,
                }
                for m in self.members
            ],
            "performance_history": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "ensemble_score": p.ensemble_score,
                    "diversity_score": p.diversity_score,
                    "improvement": p.improvement_over_best,
                }
                for p in self._performance_history[-10:]  # Last 10 performances
            ],
            "configuration": {
                "min_members": self.config.min_members,
                "max_members": self.config.max_members,
                "diversity_threshold": self.config.diversity_threshold,
                "performance_threshold": self.config.performance_threshold,
                "optimize_weights": self.config.optimize_weights,
                "estimate_uncertainty": self.config.estimate_uncertainty,
            },
        }
    
    async def prune_weak_members(self) -> List[str]:
        """Remove underperforming members from ensemble.
        
        Returns:
            List of removed member IDs
        """
        if not self.config.enable_member_pruning:
            return []
        
        removed_members = []
        
        # Identify weak members
        for member in self.members[:]:  # Copy list to allow modification
            if (member.enabled and 
                member.performance_score < self.config.pruning_threshold and
                len([m for m in self.members if m.enabled]) > self.config.min_members):
                
                member.enabled = False
                removed_members.append(str(member.detector.id))
                self.logger.info(f"Pruned weak member: {member.detector.name}")
        
        # Rebalance weights after pruning
        if removed_members and self.config.optimize_weights:
            await self._optimize_weights()
        
        return removed_members


class EnsembleFactory:
    """Factory for creating ensemble detection services."""
    
    @staticmethod
    def create_voting_ensemble(
        detectors: List[Detector],
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
    ) -> EnsembleDetectionService:
        """Create a voting ensemble.
        
        Args:
            detectors: List of detectors to ensemble
            method: Voting method to use
            
        Returns:
            Configured ensemble service
        """
        config = EnsembleConfig(
            method=method,
            weighting_strategy=WeightingStrategy.PERFORMANCE_BASED,
            optimize_weights=True,
        )
        
        ensemble = EnsembleDetectionService(config)
        
        # Add all detectors with equal initial weights
        for detector in detectors:
            asyncio.create_task(ensemble.add_member(detector, weight=1.0))
        
        return ensemble
    
    @staticmethod
    def create_stacking_ensemble(
        detectors: List[Detector],
        meta_learner: str = "logistic_regression",
    ) -> EnsembleDetectionService:
        """Create a stacking ensemble.
        
        Args:
            detectors: List of base detectors
            meta_learner: Meta-learner algorithm
            
        Returns:
            Configured stacking ensemble
        """
        config = EnsembleConfig(
            method=EnsembleMethod.STACKING,
            meta_learner=meta_learner,
            use_features_in_stacking=True,
            stacking_cv_folds=5,
        )
        
        ensemble = EnsembleDetectionService(config)
        
        # Add all detectors
        for detector in detectors:
            asyncio.create_task(ensemble.add_member(detector))
        
        return ensemble
    
    @staticmethod
    def create_dynamic_ensemble(
        detectors: List[Detector],
        selection_window: int = 100,
    ) -> EnsembleDetectionService:
        """Create a dynamic selection ensemble.
        
        Args:
            detectors: List of detectors
            selection_window: Window size for performance tracking
            
        Returns:
            Configured dynamic ensemble
        """
        config = EnsembleConfig(
            method=EnsembleMethod.DYNAMIC_SELECTION,
            weighting_strategy=WeightingStrategy.ADAPTIVE,
            selection_window_size=selection_window,
            adaptation_rate=0.1,
        )
        
        ensemble = EnsembleDetectionService(config)
        
        # Add all detectors
        for detector in detectors:
            asyncio.create_task(ensemble.add_member(detector))
        
        return ensemble
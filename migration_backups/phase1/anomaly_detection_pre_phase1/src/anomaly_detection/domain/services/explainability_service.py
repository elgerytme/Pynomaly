"""Explainability service for anomaly detection models."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from .detection_service import DetectionService

logger = logging.getLogger(__name__)


class ExplainerType(Enum):
    """Types of explainers available."""
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION = "permutation"


@dataclass
class ExplanationResult:
    """Result of model explanation."""
    explainer_type: str
    feature_names: List[str]
    feature_importance: Dict[str, float]
    explanation_values: Optional[np.ndarray] = None
    base_value: Optional[float] = None
    data_sample: Optional[List[float]] = None
    is_anomaly: bool = False
    prediction_confidence: Optional[float] = None
    top_features: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


class ExplainabilityService:
    """Service for explaining anomaly detection model predictions."""
    
    def __init__(self, detection_service: DetectionService = None):
        """Initialize explainability service.
        
        Args:
            detection_service: Detection service instance
        """
        self.detection_service = detection_service or DetectionService()
        self._explainers: Dict[str, Any] = {}
        
        # Check available explainers
        self.available_explainers = []
        if SHAP_AVAILABLE:
            self.available_explainers.append(ExplainerType.SHAP)
        if LIME_AVAILABLE:
            self.available_explainers.append(ExplainerType.LIME)
        
        # Always available
        self.available_explainers.extend([
            ExplainerType.FEATURE_IMPORTANCE,
            ExplainerType.PERMUTATION
        ])
        
        logger.info(f"Explainability service initialized with {len(self.available_explainers)} explainers")
    
    def explain_prediction(
        self,
        sample: Union[np.ndarray, List[float]],
        algorithm: str,
        explainer_type: ExplainerType = ExplainerType.FEATURE_IMPORTANCE,
        training_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> ExplanationResult:
        """Explain a single prediction.
        
        Args:
            sample: Data sample to explain
            algorithm: Algorithm used for prediction
            explainer_type: Type of explainer to use
            training_data: Training data for LIME explainer
            feature_names: Names of features
            **kwargs: Additional parameters for explainers
            
        Returns:
            ExplanationResult with explanation details
        """
        # Convert sample to numpy array
        if isinstance(sample, list):
            sample = np.array(sample)
        
        if len(sample.shape) == 1:
            sample = sample.reshape(1, -1)
        
        # Get prediction
        result = self.detection_service.predict(sample, algorithm)
        is_anomaly = bool(result.predictions[0] == -1)
        confidence = float(result.confidence_scores[0]) if result.confidence_scores is not None else None
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(sample.shape[1])]
        
        # Choose explanation method
        if explainer_type == ExplainerType.SHAP and SHAP_AVAILABLE:
            return self._explain_with_shap(sample, algorithm, feature_names, is_anomaly, confidence, **kwargs)
        elif explainer_type == ExplainerType.LIME and LIME_AVAILABLE:
            if training_data is None:
                raise ValueError("Training data required for LIME explainer")
            return self._explain_with_lime(sample, algorithm, training_data, feature_names, is_anomaly, confidence, **kwargs)
        elif explainer_type == ExplainerType.PERMUTATION:
            return self._explain_with_permutation(sample, algorithm, feature_names, is_anomaly, confidence, **kwargs)
        else:
            # Default to feature importance
            return self._explain_with_feature_importance(sample, algorithm, feature_names, is_anomaly, confidence, **kwargs)
    
    def _explain_with_shap(
        self,
        sample: np.ndarray,
        algorithm: str,
        feature_names: List[str],
        is_anomaly: bool,
        confidence: Optional[float],
        **kwargs
    ) -> ExplanationResult:
        """Explain using SHAP."""
        try:
            # Get the fitted model
            model = self.detection_service._fitted_models.get(algorithm)
            if model is None:
                raise ValueError(f"Model for algorithm '{algorithm}' not fitted")
            
            # Create SHAP explainer
            explainer_key = f"{algorithm}_shap"
            if explainer_key not in self._explainers:
                # Use different explainer based on model type
                if hasattr(model, 'decision_function'):
                    # For models with decision_function (like SVM, Isolation Forest)
                    self._explainers[explainer_key] = shap.Explainer(model.decision_function)
                else:
                    # Fallback to basic explainer
                    self._explainers[explainer_key] = shap.Explainer(model.predict)
            
            explainer = self._explainers[explainer_key]
            
            # Get SHAP values
            shap_values = explainer(sample)
            
            # Extract values for the first (and only) sample
            explanation_values = shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0]
            base_value = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0.0
            
            # Calculate feature importance
            feature_importance = {}
            for i, name in enumerate(feature_names):
                feature_importance[name] = float(np.abs(explanation_values[i]))
            
            # Get top features
            top_features = self._get_top_features(feature_importance, sample[0], limit=5)
            
            return ExplanationResult(
                explainer_type="shap",
                feature_names=feature_names,
                feature_importance=feature_importance,
                explanation_values=explanation_values,
                base_value=float(base_value),
                data_sample=sample[0].tolist(),
                is_anomaly=is_anomaly,
                prediction_confidence=confidence,
                top_features=top_features,
                metadata={"shap_available": True}
            )
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            # Fallback to feature importance
            return self._explain_with_feature_importance(sample, algorithm, feature_names, is_anomaly, confidence)
    
    def _explain_with_lime(
        self,
        sample: np.ndarray,
        algorithm: str,
        training_data: np.ndarray,
        feature_names: List[str],
        is_anomaly: bool,
        confidence: Optional[float],
        **kwargs
    ) -> ExplanationResult:
        """Explain using LIME."""
        try:
            # Get the fitted model
            model = self.detection_service._fitted_models.get(algorithm)
            if model is None:
                raise ValueError(f"Model for algorithm '{algorithm}' not fitted")
            
            # Create LIME explainer
            explainer_key = f"{algorithm}_lime"
            if explainer_key not in self._explainers:
                self._explainers[explainer_key] = LimeTabularExplainer(
                    training_data,
                    feature_names=feature_names,
                    mode='regression',  # Anomaly scores are continuous
                    discretize_continuous=True
                )
            
            explainer = self._explainers[explainer_key]
            
            # Create prediction function
            def predict_fn(x):
                try:
                    if hasattr(model, 'decision_function'):
                        return model.decision_function(x)
                    else:
                        # For models without decision_function, use predict and convert
                        predictions = model.predict(x)
                        return predictions.astype(float)
                except:
                    return np.zeros(len(x))
            
            # Get LIME explanation
            explanation = explainer.explain_instance(
                sample[0],
                predict_fn,
                num_features=min(len(feature_names), kwargs.get('num_features', 10))
            )
            
            # Extract feature importance
            feature_importance = {}
            explanation_values = np.zeros(len(feature_names))
            
            for feature_idx, importance in explanation.as_list():
                if feature_idx < len(feature_names):
                    feature_name = feature_names[feature_idx]
                    feature_importance[feature_name] = float(abs(importance))
                    explanation_values[feature_idx] = importance
            
            # Fill remaining features with zero importance
            for i, name in enumerate(feature_names):
                if name not in feature_importance:
                    feature_importance[name] = 0.0
            
            # Get top features
            top_features = self._get_top_features(feature_importance, sample[0], limit=5)
            
            return ExplanationResult(
                explainer_type="lime",
                feature_names=feature_names,
                feature_importance=feature_importance,
                explanation_values=explanation_values,
                base_value=None,
                data_sample=sample[0].tolist(),
                is_anomaly=is_anomaly,
                prediction_confidence=confidence,
                top_features=top_features,
                metadata={"lime_available": True}
            )
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            # Fallback to feature importance
            return self._explain_with_feature_importance(sample, algorithm, feature_names, is_anomaly, confidence)
    
    def _explain_with_permutation(
        self,
        sample: np.ndarray,
        algorithm: str,
        feature_names: List[str],
        is_anomaly: bool,
        confidence: Optional[float],
        **kwargs
    ) -> ExplanationResult:
        """Explain using permutation importance."""
        try:
            # Get the fitted model
            model = self.detection_service._fitted_models.get(algorithm)
            if model is None:
                raise ValueError(f"Model for algorithm '{algorithm}' not fitted")
            
            # Get original prediction
            if hasattr(model, 'decision_function'):
                original_score = model.decision_function(sample)[0]
            else:
                original_score = float(model.predict(sample)[0])
            
            # Calculate permutation importance
            feature_importance = {}
            explanation_values = np.zeros(sample.shape[1])
            
            for i, feature_name in enumerate(feature_names):
                # Create permuted sample
                permuted_sample = sample.copy()
                
                # Permute this feature (set to random value from normal distribution)
                permuted_sample[0, i] = np.random.normal(0, 1)
                
                # Get new prediction
                if hasattr(model, 'decision_function'):
                    new_score = model.decision_function(permuted_sample)[0]
                else:
                    new_score = float(model.predict(permuted_sample)[0])
                
                # Importance is the change in prediction
                importance = abs(original_score - new_score)
                feature_importance[feature_name] = float(importance)
                explanation_values[i] = importance
            
            # Get top features
            top_features = self._get_top_features(feature_importance, sample[0], limit=5)
            
            return ExplanationResult(
                explainer_type="permutation",
                feature_names=feature_names,
                feature_importance=feature_importance,
                explanation_values=explanation_values,
                base_value=float(original_score),
                data_sample=sample[0].tolist(),
                is_anomaly=is_anomaly,
                prediction_confidence=confidence,
                top_features=top_features,
                metadata={"method": "permutation_importance"}
            )
            
        except Exception as e:
            logger.error(f"Permutation explanation failed: {e}")
            # Fallback to feature importance
            return self._explain_with_feature_importance(sample, algorithm, feature_names, is_anomaly, confidence)
    
    def _explain_with_feature_importance(
        self,
        sample: np.ndarray,
        algorithm: str,
        feature_names: List[str],
        is_anomaly: bool,
        confidence: Optional[float],
        **kwargs
    ) -> ExplanationResult:
        """Explain using simple feature importance based on magnitude."""
        try:
            # Simple feature importance based on absolute values
            feature_importance = {}
            sample_values = sample[0]
            
            # Normalize sample values to get relative importance
            max_abs_value = np.max(np.abs(sample_values))
            if max_abs_value > 0:
                normalized_values = np.abs(sample_values) / max_abs_value
            else:
                normalized_values = np.zeros_like(sample_values)
            
            for i, feature_name in enumerate(feature_names):
                feature_importance[feature_name] = float(normalized_values[i])
            
            # Get top features
            top_features = self._get_top_features(feature_importance, sample[0], limit=5)
            
            return ExplanationResult(
                explainer_type="feature_importance",
                feature_names=feature_names,
                feature_importance=feature_importance,
                explanation_values=normalized_values,
                base_value=0.0,
                data_sample=sample[0].tolist(),
                is_anomaly=is_anomaly,
                prediction_confidence=confidence,
                top_features=top_features,
                metadata={"method": "magnitude_based"}
            )
            
        except Exception as e:
            logger.error(f"Feature importance explanation failed: {e}")
            raise
    
    def _get_top_features(
        self,
        feature_importance: Dict[str, float],
        sample_values: np.ndarray,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get top contributing features."""
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = []
        for i, (feature_name, importance) in enumerate(sorted_features[:limit]):
            # Find feature index
            feature_idx = next(
                (j for j, name in enumerate(feature_importance.keys()) if name == feature_name),
                i
            )
            
            top_features.append({
                "feature_name": feature_name,
                "importance": importance,
                "value": float(sample_values[feature_idx]) if feature_idx < len(sample_values) else 0.0,
                "rank": i + 1
            })
        
        return top_features
    
    def explain_batch(
        self,
        samples: np.ndarray,
        algorithm: str,
        explainer_type: ExplainerType = ExplainerType.FEATURE_IMPORTANCE,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> List[ExplanationResult]:
        """Explain multiple predictions.
        
        Args:
            samples: Batch of samples
            algorithm: Algorithm used
            explainer_type: Type of explainer
            feature_names: Feature names
            **kwargs: Additional parameters
            
        Returns:
            List of ExplanationResult objects
        """
        results = []
        for sample in samples:
            result = self.explain_prediction(
                sample,
                algorithm,
                explainer_type,
                feature_names=feature_names,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def get_global_feature_importance(
        self,
        algorithm: str,
        training_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """Get global feature importance across multiple samples.
        
        Args:
            algorithm: Algorithm to analyze
            training_data: Training data to sample from
            feature_names: Feature names
            n_samples: Number of samples to analyze
            
        Returns:
            Dictionary of average feature importance
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(training_data.shape[1])]
        
        # Sample random subset
        sample_indices = np.random.choice(
            len(training_data),
            size=min(n_samples, len(training_data)),
            replace=False
        )
        samples = training_data[sample_indices]
        
        # Get explanations for all samples
        explanations = self.explain_batch(samples, algorithm, feature_names=feature_names)
        
        # Average feature importance
        global_importance = {name: 0.0 for name in feature_names}
        valid_explanations = 0
        
        for explanation in explanations:
            if explanation.feature_importance:
                valid_explanations += 1
                for feature_name, importance in explanation.feature_importance.items():
                    global_importance[feature_name] += importance
        
        # Average the importance scores
        if valid_explanations > 0:
            for feature_name in global_importance:
                global_importance[feature_name] /= valid_explanations
        
        return global_importance
    
    def get_available_explainers(self) -> List[str]:
        """Get list of available explainer types."""
        return [explainer.value for explainer in self.available_explainers]
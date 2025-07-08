"""Anomaly classification interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol

from pynomaly.domain.entities.anomaly import Anomaly


class SeverityClassifier(Protocol):
    """Protocol for severity classifiers."""
    
    def classify_severity(self, anomaly: Anomaly) -> str:
        """Classify anomaly severity.
        
        Args:
            anomaly: The anomaly to classify
            
        Returns:
            Severity level (e.g., 'low', 'medium', 'high', 'critical')
        """
        ...


class TypeClassifier(Protocol):
    """Protocol for type classifiers."""
    
    def classify_type(self, anomaly: Anomaly) -> str:
        """Classify anomaly type.
        
        Args:
            anomaly: The anomaly to classify
            
        Returns:
            Anomaly type (e.g., 'point', 'collective', 'contextual')
        """
        ...


class DefaultSeverityClassifier:
    """Default severity classifier based on anomaly score."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """Initialize with custom thresholds.
        
        Args:
            thresholds: Custom severity thresholds
        """
        self.thresholds = thresholds or {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.0
        }
    
    def classify_severity(self, anomaly: Anomaly) -> str:
        """Classify anomaly severity based on score."""
        # Extract score value
        score_value = anomaly.score.value if hasattr(anomaly.score, 'value') else anomaly.score
        
        # Sort thresholds by value (descending) to ensure correct classification
        sorted_thresholds = sorted(self.thresholds.items(), key=lambda x: x[1], reverse=True)
        
        for severity, threshold in sorted_thresholds:
            if score_value >= threshold:
                return severity
        
        return 'low'  # Fallback


class DefaultTypeClassifier:
    """Default type classifier based on data characteristics."""
    
    def __init__(self):
        """Initialize the type classifier."""
        pass
    
    def classify_type(self, anomaly: Anomaly) -> str:
        """Classify anomaly type based on data characteristics."""
        # Simple heuristic-based classification
        data_point = anomaly.data_point
        
        # Check if it's a collective anomaly (multiple features affected)
        if isinstance(data_point, dict) and len(data_point) > 3:
            # If multiple features are present, it could be collective
            return 'collective'
        
        # Check metadata for contextual clues
        if anomaly.metadata.get('temporal_context'):
            return 'contextual'
        
        # Default to point anomaly
        return 'point'


class MLSeverityClassifier:
    """ML-based severity classifier wrapper."""
    
    def __init__(self, ml_classifier: Optional[Any] = None):
        """Initialize with ML classifier.
        
        Args:
            ml_classifier: ML classifier instance (e.g., from domain.services.ml_severity_classifier)
        """
        self.ml_classifier = ml_classifier
        self.fallback_classifier = DefaultSeverityClassifier()
    
    def classify_severity(self, anomaly: Anomaly) -> str:
        """Classify using ML model with fallback to default classifier."""
        if self.ml_classifier is None:
            return self.fallback_classifier.classify_severity(anomaly)
        
        try:
            # Try to use ML classifier
            score_value = anomaly.score.value if hasattr(anomaly.score, 'value') else anomaly.score
            
            # Extract features for ML model
            features = {
                'score': float(score_value),
                'volatility': anomaly.metadata.get('volatility', 0.5),
                'seasonality': anomaly.metadata.get('seasonality', 0.5)
            }
            
            # Use ML classifier
            severity_score = self.ml_classifier.predict_severity(features)
            
            # Convert score to severity level
            if severity_score >= 0.9:
                return 'critical'
            elif severity_score >= 0.7:
                return 'high'
            elif severity_score >= 0.5:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            # Fall back to default classifier
            return self.fallback_classifier.classify_severity(anomaly)


class BatchProcessingSeverityClassifier:
    """Severity classifier optimized for batch processing (A-002)."""
    
    def __init__(self, base_classifier: Optional[SeverityClassifier] = None):
        """Initialize with base classifier.
        
        Args:
            base_classifier: Base classifier to use for actual classification
        """
        self.base_classifier = base_classifier or DefaultSeverityClassifier()
        self._batch_cache = {}
    
    def classify_severity(self, anomaly: Anomaly) -> str:
        """Classify with batch optimization."""
        # Simple caching for batch processing
        cache_key = self._get_cache_key(anomaly)
        
        if cache_key in self._batch_cache:
            return self._batch_cache[cache_key]
        
        result = self.base_classifier.classify_severity(anomaly)
        self._batch_cache[cache_key] = result
        
        return result
    
    def _get_cache_key(self, anomaly: Anomaly) -> str:
        """Generate cache key for anomaly."""
        score_value = anomaly.score.value if hasattr(anomaly.score, 'value') else anomaly.score
        return f"severity_{score_value:.3f}_{anomaly.detector_name}"
    
    def clear_cache(self):
        """Clear the batch cache."""
        self._batch_cache.clear()


class DashboardTypeClassifier:
    """Type classifier optimized for dashboard display (P-001)."""
    
    def __init__(self, base_classifier: Optional[TypeClassifier] = None):
        """Initialize with base classifier.
        
        Args:
            base_classifier: Base classifier to use for actual classification
        """
        self.base_classifier = base_classifier or DefaultTypeClassifier()
    
    def classify_type(self, anomaly: Anomaly) -> str:
        """Classify with dashboard-friendly categories."""
        base_type = self.base_classifier.classify_type(anomaly)
        
        # Map to dashboard-friendly names
        type_mapping = {
            'point': 'Point Anomaly',
            'collective': 'Pattern Anomaly',
            'contextual': 'Context Anomaly'
        }
        
        return type_mapping.get(base_type, base_type)

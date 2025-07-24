"""Unit tests for Anomaly entity."""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any

from anomaly_detection.domain.entities.anomaly import Anomaly, AnomalyType, AnomalySeverity


class TestAnomalyEntity:
    """Test suite for Anomaly entity."""
    
    def test_anomaly_creation_basic(self):
        """Test basic anomaly creation."""
        anomaly = Anomaly(
            index=42,
            confidence_score=0.85
        )
        
        assert anomaly.index == 42
        assert anomaly.confidence_score == 0.85
        assert anomaly.feature_contributions is None
        assert anomaly.explanation is None
        assert anomaly.anomaly_type == AnomalyType.UNKNOWN
        assert anomaly.severity is not None  # Should be calculated automatically
    
    def test_anomaly_creation_with_features(self):
        """Test anomaly creation with feature values."""
        feature_values = np.array([45.2, 1050.5, 0.05])
        feature_contributions = {
            'temperature': 0.6,
            'pressure': 0.3,
            'flow_rate': 0.1
        }
        
        anomaly = Anomaly(
            index=10,
            confidence_score=0.95,
            feature_values=feature_values,
            feature_contributions=feature_contributions
        )
        
        assert np.array_equal(anomaly.feature_values, feature_values)
        assert anomaly.feature_contributions == feature_contributions
    
    def test_anomaly_creation_with_explanation(self):
        """Test anomaly creation with explanation."""
        explanation = "High temperature and pressure values detected"
        
        anomaly = Anomaly(
            index=5,
            confidence_score=0.78,
            explanation=explanation
        )
        
        assert anomaly.explanation == explanation
    
    def test_anomaly_severity_calculation(self):
        """Test automatic severity calculation."""
        # Critical severity (>= 0.9)
        anomaly1 = Anomaly(index=0, confidence_score=0.95)
        assert anomaly1.severity == AnomalySeverity.CRITICAL
        
        # High severity (>= 0.75)
        anomaly2 = Anomaly(index=0, confidence_score=0.8)
        assert anomaly2.severity == AnomalySeverity.HIGH
        
        # Medium severity (>= 0.5)
        anomaly3 = Anomaly(index=0, confidence_score=0.6)
        assert anomaly3.severity == AnomalySeverity.MEDIUM
        
        # Low severity (< 0.5)
        anomaly4 = Anomaly(index=0, confidence_score=0.3)
        assert anomaly4.severity == AnomalySeverity.LOW
    
    def test_anomaly_severity_override(self):
        """Test manual severity override."""
        anomaly = Anomaly(
            index=0,
            confidence_score=0.3,  # Would normally be LOW
            severity=AnomalySeverity.HIGH  # Override to HIGH
        )
        assert anomaly.severity == AnomalySeverity.HIGH
    
    def test_get_top_contributing_features(self):
        """Test getting top contributing features."""
        feature_contributions = {
            'feature1': 0.5,
            'feature2': -0.3,
            'feature3': 0.8,
            'feature4': 0.1,
            'feature5': -0.6
        }
        
        anomaly = Anomaly(
            index=0,
            confidence_score=0.8,
            feature_contributions=feature_contributions
        )
        
        top_features = anomaly.get_top_contributing_features(3)
        
        assert len(top_features) == 3
        assert top_features[0] == ('feature3', 0.8)  # Highest absolute value
        assert top_features[1] == ('feature5', -0.6)  # Second highest absolute value
        assert top_features[2] == ('feature1', 0.5)   # Third highest absolute value
    
    def test_get_top_contributing_features_empty(self):
        """Test getting top features when none exist."""
        anomaly = Anomaly(index=0, confidence_score=0.8)
        top_features = anomaly.get_top_contributing_features(5)
        assert top_features == []
    
    def test_is_high_priority(self):
        """Test high priority detection."""
        # High priority (HIGH severity)
        anomaly1 = Anomaly(index=0, confidence_score=0.8)  # HIGH severity
        assert anomaly1.is_high_priority() is True
        
        # High priority (CRITICAL severity)
        anomaly2 = Anomaly(index=0, confidence_score=0.95)  # CRITICAL severity
        assert anomaly2.is_high_priority() is True
        
        # Not high priority (MEDIUM severity)
        anomaly3 = Anomaly(index=0, confidence_score=0.6)  # MEDIUM severity
        assert anomaly3.is_high_priority() is False
        
        # Not high priority (LOW severity)
        anomaly4 = Anomaly(index=0, confidence_score=0.3)  # LOW severity
        assert anomaly4.is_high_priority() is False
    
    def test_anomaly_to_dict(self):
        """Test conversion to dictionary."""
        feature_values = np.array([1.0, 2.0, 3.0])
        feature_contributions = {'f1': 0.5, 'f2': 0.3}
        explanation = "Test explanation"
        
        anomaly = Anomaly(
            index=42,
            confidence_score=0.85,
            anomaly_type=AnomalyType.POINT,
            feature_values=feature_values,
            feature_contributions=feature_contributions,
            explanation=explanation
        )
        
        result = anomaly.to_dict()
        
        assert isinstance(result, dict)
        assert result['index'] == 42
        assert result['confidence_score'] == 0.85
        assert result['anomaly_type'] == 'point'
        assert result['feature_values'] == [1.0, 2.0, 3.0]
        assert result['feature_contributions'] == feature_contributions
        assert result['explanation'] == explanation
        assert 'severity' in result
        assert 'is_high_priority' in result
        assert 'timestamp' in result
    
    def test_anomaly_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'index': 10,
            'confidence_score': 0.75,
            'anomaly_type': 'contextual',
            'severity': 'high',
            'feature_values': [1.0, 2.0],
            'feature_contributions': {'temp': 0.6},
            'explanation': 'Test anomaly'
        }
        
        anomaly = Anomaly.from_dict(data)
        
        assert anomaly.index == 10
        assert anomaly.confidence_score == 0.75
        assert anomaly.anomaly_type == AnomalyType.CONTEXTUAL
        assert anomaly.severity == AnomalySeverity.HIGH
        assert np.array_equal(anomaly.feature_values, np.array([1.0, 2.0]))
        assert anomaly.feature_contributions == {'temp': 0.6}
        assert anomaly.explanation == 'Test anomaly'
    
    def test_anomaly_from_dict_minimal(self):
        """Test creation from minimal dictionary."""
        data = {
            'index': 5,
            'confidence_score': 0.6
        }
        
        anomaly = Anomaly.from_dict(data)
        
        assert anomaly.index == 5
        assert anomaly.confidence_score == 0.6
        assert anomaly.anomaly_type == AnomalyType.UNKNOWN
        assert anomaly.severity is not None  # Should be calculated
    
    def test_anomaly_equality(self):
        """Test anomaly equality comparison."""
        anomaly1 = Anomaly(index=1, confidence_score=0.8)
        anomaly2 = Anomaly(index=1, confidence_score=0.8)
        anomaly3 = Anomaly(index=2, confidence_score=0.8)
        
        # Note: dataclass equality is based on all fields
        # Since timestamp is auto-generated, they won't be equal unless we set it explicitly
        timestamp = datetime.utcnow()
        anomaly1.timestamp = timestamp
        anomaly2.timestamp = timestamp
        
        assert anomaly1 == anomaly2
        assert anomaly1 != anomaly3
    
    def test_anomaly_repr(self):
        """Test string representation."""
        anomaly = Anomaly(index=42, confidence_score=0.85)
        
        repr_str = repr(anomaly)
        str_str = str(anomaly)
        
        assert 'Anomaly' in repr_str
        assert 'index=42' in repr_str
        assert 'confidence=0.85' in repr_str
        assert repr_str == str_str
    
    def test_anomaly_sorting(self):
        """Test anomaly sorting by confidence score."""
        anomalies = [
            Anomaly(index=1, confidence_score=0.6),
            Anomaly(index=2, confidence_score=0.9),
            Anomaly(index=3, confidence_score=0.3)
        ]
        
        # Sort by confidence score (descending)
        sorted_anomalies = sorted(anomalies, key=lambda a: a.confidence_score, reverse=True)
        
        assert sorted_anomalies[0].confidence_score == 0.9
        assert sorted_anomalies[1].confidence_score == 0.6
        assert sorted_anomalies[2].confidence_score == 0.3
    
    def test_anomaly_with_nan_features(self):
        """Test anomaly handling with NaN feature values."""
        feature_values = np.array([1.0, np.nan, 3.0])
        
        anomaly = Anomaly(
            index=0,
            confidence_score=0.8,
            feature_values=feature_values
        )
        
        assert len(anomaly.feature_values) == 3
        assert np.isnan(anomaly.feature_values[1])
        assert anomaly.feature_values[0] == 1.0
        assert anomaly.feature_values[2] == 3.0
    
    def test_anomaly_metadata(self):
        """Test anomaly metadata handling."""
        metadata = {
            'source': 'sensor_1',
            'algorithm': 'isolation_forest',
            'version': '1.0'
        }
        
        anomaly = Anomaly(
            index=0,
            confidence_score=0.8,
            metadata=metadata
        )
        
        assert anomaly.metadata == metadata
        assert anomaly.metadata['source'] == 'sensor_1'
    
    def test_anomaly_batch_creation(self):
        """Test creating multiple anomalies efficiently."""
        indices = [1, 2, 3, 4, 5]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        anomalies = [
            Anomaly(index=idx, confidence_score=score)
            for idx, score in zip(indices, scores)
        ]
        
        assert len(anomalies) == 5
        assert all(isinstance(a, Anomaly) for a in anomalies)
        assert [a.index for a in anomalies] == indices
        assert [a.confidence_score for a in anomalies] == scores
    
    def test_anomaly_timestamp_auto_generation(self):
        """Test automatic timestamp generation."""
        before = datetime.utcnow()
        anomaly = Anomaly(index=0, confidence_score=0.8)
        after = datetime.utcnow()
        
        assert before <= anomaly.timestamp <= after
    
    def test_anomaly_timestamp_override(self):
        """Test manual timestamp setting."""
        custom_timestamp = datetime(2023, 1, 1, 12, 0, 0)
        
        anomaly = Anomaly(
            index=0,
            confidence_score=0.8,
            timestamp=custom_timestamp
        )
        
        assert anomaly.timestamp == custom_timestamp
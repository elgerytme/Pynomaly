"""Unit tests for Anomaly entity."""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any

from anomaly_detection.domain.entities.anomaly import Anomaly


class TestAnomalyEntity:
    """Test suite for Anomaly entity."""
    
    def test_anomaly_creation_basic(self):
        """Test basic anomaly creation."""
        anomaly = Anomaly(
            index=42,
            score=0.85,
            confidence=0.92
        )
        
        assert anomaly.index == 42
        assert anomaly.score == 0.85
        assert anomaly.confidence == 0.92
        assert anomaly.features == {}
        assert anomaly.explanation is None
    
    def test_anomaly_creation_with_features(self):
        """Test anomaly creation with feature values."""
        features = {
            'temperature': 45.2,
            'pressure': 1050.5,
            'flow_rate': 0.05
        }
        
        anomaly = Anomaly(
            index=10,
            score=0.95,
            confidence=0.88,
            features=features
        )
        
        assert anomaly.features == features
        assert anomaly.features['temperature'] == 45.2
    
    def test_anomaly_creation_with_explanation(self):
        """Test anomaly creation with explanation."""
        explanation = {
            'type': 'feature_importance',
            'top_features': ['temperature', 'pressure'],
            'contributions': {'temperature': 0.6, 'pressure': 0.3}
        }
        
        anomaly = Anomaly(
            index=5,
            score=0.78,
            confidence=0.85,
            explanation=explanation
        )
        
        assert anomaly.explanation == explanation
        assert anomaly.explanation['type'] == 'feature_importance'
    
    def test_anomaly_validation_index(self):
        """Test index validation."""
        # Valid indices
        anomaly1 = Anomaly(index=0, score=0.5, confidence=0.5)
        assert anomaly1.index == 0
        
        anomaly2 = Anomaly(index=999999, score=0.5, confidence=0.5)
        assert anomaly2.index == 999999
        
        # Invalid indices
        with pytest.raises(ValueError, match="Index must be non-negative"):
            Anomaly(index=-1, score=0.5, confidence=0.5)
    
    def test_anomaly_validation_score(self):
        """Test score validation."""
        # Valid scores
        anomaly1 = Anomaly(index=0, score=0.0, confidence=0.5)
        assert anomaly1.score == 0.0
        
        anomaly2 = Anomaly(index=0, score=1.0, confidence=0.5)
        assert anomaly2.score == 1.0
        
        # Invalid scores
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            Anomaly(index=0, score=-0.1, confidence=0.5)
        
        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            Anomaly(index=0, score=1.1, confidence=0.5)
    
    def test_anomaly_validation_confidence(self):
        """Test confidence validation."""
        # Valid confidence
        anomaly1 = Anomaly(index=0, score=0.5, confidence=0.0)
        assert anomaly1.confidence == 0.0
        
        anomaly2 = Anomaly(index=0, score=0.5, confidence=1.0)
        assert anomaly2.confidence == 1.0
        
        # Invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Anomaly(index=0, score=0.5, confidence=-0.1)
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Anomaly(index=0, score=0.5, confidence=1.1)
    
    def test_anomaly_severity_property(self):
        """Test severity property calculation."""
        # Low severity
        anomaly1 = Anomaly(index=0, score=0.2, confidence=0.3)
        assert anomaly1.severity == 'low'
        
        # Medium severity  
        anomaly2 = Anomaly(index=0, score=0.6, confidence=0.7)
        assert anomaly2.severity == 'medium'
        
        # High severity
        anomaly3 = Anomaly(index=0, score=0.9, confidence=0.95)
        assert anomaly3.severity == 'high'
        
        # Edge cases
        anomaly4 = Anomaly(index=0, score=0.5, confidence=0.5)
        assert anomaly4.severity == 'low'
        
        anomaly5 = Anomaly(index=0, score=0.7, confidence=0.7)
        assert anomaly5.severity == 'medium'
    
    def test_anomaly_to_dict(self):
        """Test conversion to dictionary."""
        features = {'f1': 1.0, 'f2': 2.0}
        explanation = {'type': 'test', 'data': [1, 2, 3]}
        
        anomaly = Anomaly(
            index=42,
            score=0.85,
            confidence=0.92,
            features=features,
            explanation=explanation
        )
        
        result = anomaly.to_dict()
        
        assert isinstance(result, dict)
        assert result['index'] == 42
        assert result['score'] == 0.85
        assert result['confidence'] == 0.92
        assert result['features'] == features
        assert result['explanation'] == explanation
        assert result['severity'] == 'high'
    
    def test_anomaly_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'index': 10,
            'score': 0.75,
            'confidence': 0.88,
            'features': {'temp': 100},
            'explanation': {'reason': 'outlier'}
        }
        
        anomaly = Anomaly.from_dict(data)
        
        assert anomaly.index == 10
        assert anomaly.score == 0.75
        assert anomaly.confidence == 0.88
        assert anomaly.features == {'temp': 100}
        assert anomaly.explanation == {'reason': 'outlier'}
    
    def test_anomaly_from_dict_minimal(self):
        """Test creation from minimal dictionary."""
        data = {
            'index': 5,
            'score': 0.6,
            'confidence': 0.7
        }
        
        anomaly = Anomaly.from_dict(data)
        
        assert anomaly.index == 5
        assert anomaly.score == 0.6
        assert anomaly.confidence == 0.7
        assert anomaly.features == {}
        assert anomaly.explanation is None
    
    def test_anomaly_from_dict_validation(self):
        """Test validation when creating from dictionary."""
        # Missing required fields
        with pytest.raises(KeyError):
            Anomaly.from_dict({'index': 0, 'score': 0.5})
        
        with pytest.raises(KeyError):
            Anomaly.from_dict({'index': 0, 'confidence': 0.5})
        
        with pytest.raises(KeyError):
            Anomaly.from_dict({'score': 0.5, 'confidence': 0.5})
        
        # Invalid values
        with pytest.raises(ValueError):
            Anomaly.from_dict({'index': -1, 'score': 0.5, 'confidence': 0.5})
    
    def test_anomaly_equality(self):
        """Test anomaly equality comparison."""
        anomaly1 = Anomaly(index=42, score=0.85, confidence=0.92)
        anomaly2 = Anomaly(index=42, score=0.85, confidence=0.92)
        anomaly3 = Anomaly(index=43, score=0.85, confidence=0.92)
        
        assert anomaly1 == anomaly2
        assert anomaly1 != anomaly3
        assert anomaly1 != "not an anomaly"
    
    def test_anomaly_hash(self):
        """Test anomaly hashing."""
        anomaly1 = Anomaly(index=42, score=0.85, confidence=0.92)
        anomaly2 = Anomaly(index=42, score=0.85, confidence=0.92)
        anomaly3 = Anomaly(index=43, score=0.85, confidence=0.92)
        
        assert hash(anomaly1) == hash(anomaly2)
        assert hash(anomaly1) != hash(anomaly3)
        
        # Can be used in sets
        anomaly_set = {anomaly1, anomaly2, anomaly3}
        assert len(anomaly_set) == 2
    
    def test_anomaly_repr(self):
        """Test string representation."""
        anomaly = Anomaly(
            index=42,
            score=0.85,
            confidence=0.92
        )
        
        repr_str = repr(anomaly)
        assert 'Anomaly' in repr_str
        assert '42' in repr_str
        assert '0.85' in repr_str
        assert '0.92' in repr_str
    
    def test_anomaly_str(self):
        """Test human-readable string."""
        anomaly = Anomaly(
            index=42,
            score=0.85,
            confidence=0.92
        )
        
        str_repr = str(anomaly)
        assert 'index 42' in str_repr
        assert 'score=0.85' in str_repr
        assert 'confidence=0.92' in str_repr
        assert 'severity=high' in str_repr
    
    def test_anomaly_sorting(self):
        """Test sorting anomalies by score."""
        anomalies = [
            Anomaly(index=0, score=0.5, confidence=0.8),
            Anomaly(index=1, score=0.9, confidence=0.7),
            Anomaly(index=2, score=0.3, confidence=0.9),
            Anomaly(index=3, score=0.7, confidence=0.6)
        ]
        
        # Sort by score (descending)
        sorted_anomalies = sorted(anomalies, key=lambda a: a.score, reverse=True)
        
        assert sorted_anomalies[0].index == 1  # Highest score
        assert sorted_anomalies[-1].index == 2  # Lowest score
        
        # Sort by confidence
        sorted_by_conf = sorted(anomalies, key=lambda a: a.confidence, reverse=True)
        assert sorted_by_conf[0].index == 2  # Highest confidence
    
    def test_anomaly_with_nan_features(self):
        """Test handling of NaN in features."""
        features = {
            'valid': 1.0,
            'invalid': float('nan'),
            'inf': float('inf')
        }
        
        # Should accept NaN/Inf but mark them
        anomaly = Anomaly(
            index=0,
            score=0.5,
            confidence=0.8,
            features=features
        )
        
        assert np.isnan(anomaly.features['invalid'])
        assert np.isinf(anomaly.features['inf'])
    
    def test_anomaly_immutability(self):
        """Test that anomaly attributes are protected."""
        anomaly = Anomaly(index=42, score=0.85, confidence=0.92)
        
        # Direct assignment should raise AttributeError
        with pytest.raises(AttributeError):
            anomaly.index = 100
        
        with pytest.raises(AttributeError):
            anomaly.score = 0.99
        
        # Features dict should be a copy
        features = {'temp': 100}
        anomaly2 = Anomaly(index=0, score=0.5, confidence=0.5, features=features)
        features['temp'] = 200  # Modify original
        assert anomaly2.features['temp'] == 100  # Should not change
    
    def test_anomaly_metadata(self):
        """Test anomaly with metadata."""
        anomaly = Anomaly(
            index=42,
            score=0.85,
            confidence=0.92,
            metadata={
                'timestamp': datetime.now(),
                'algorithm': 'isolation_forest',
                'dataset': 'sensor_data'
            }
        )
        
        assert 'timestamp' in anomaly.metadata
        assert anomaly.metadata['algorithm'] == 'isolation_forest'
    
    def test_anomaly_batch_creation(self):
        """Test creating multiple anomalies efficiently."""
        # Simulate batch detection results
        indices = [10, 25, 42, 67, 89]
        scores = [0.9, 0.85, 0.92, 0.88, 0.95]
        confidences = [0.8, 0.9, 0.95, 0.85, 0.92]
        
        anomalies = [
            Anomaly(idx, score, conf)
            for idx, score, conf in zip(indices, scores, confidences)
        ]
        
        assert len(anomalies) == 5
        assert all(isinstance(a, Anomaly) for a in anomalies)
        
        # Get high severity anomalies
        high_severity = [a for a in anomalies if a.severity == 'high']
        assert len(high_severity) >= 3  # Most should be high given the scores
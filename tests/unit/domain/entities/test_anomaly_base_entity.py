"""Tests for Anomaly entity with BaseEntity inheritance."""

import pytest
from datetime import datetime
from uuid import UUID

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore


class TestAnomalyBaseEntity:
    """Test cases for Anomaly entity inheriting from BaseEntity."""

    def test_anomaly_inherits_base_entity(self):
        """Test that Anomaly properly inherits from BaseEntity."""
        anomaly = Anomaly(
            score=0.8,
            data_point={"feature1": 1.0, "feature2": 2.0},
            detector_name="test_detector"
        )
        
        # Should have BaseEntity attributes
        assert isinstance(anomaly.id, UUID)
        assert isinstance(anomaly.created_at, datetime)
        assert isinstance(anomaly.updated_at, datetime)
        assert anomaly.version == 1
        assert isinstance(anomaly.metadata, dict)
        
        # Should have Anomaly-specific attributes
        assert anomaly.score == 0.8
        assert anomaly.data_point == {"feature1": 1.0, "feature2": 2.0}
        assert anomaly.detector_name == "test_detector"

    def test_anomaly_validation(self):
        """Test anomaly validation."""
        # Valid anomaly
        anomaly = Anomaly(
            score=0.7,
            data_point={"feature": 1.0},
            detector_name="detector"
        )
        assert anomaly.score == 0.7

        # Invalid score type
        with pytest.raises(TypeError):
            Anomaly(
                score="invalid",
                data_point={"feature": 1.0},
                detector_name="detector"
            )

        # Empty detector name
        with pytest.raises(ValueError):
            Anomaly(
                score=0.7,
                data_point={"feature": 1.0},
                detector_name=""
            )

        # Invalid data point type
        with pytest.raises(TypeError):
            Anomaly(
                score=0.7,
                data_point="not_a_dict",
                detector_name="detector"
            )

    def test_anomaly_with_anomaly_score(self):
        """Test anomaly with AnomalyScore value object."""
        score = AnomalyScore(0.9)
        anomaly = Anomaly(
            score=score,
            data_point={"feature": 1.0},
            detector_name="detector"
        )
        
        assert anomaly.score == score

    def test_anomaly_severity_calculation(self):
        """Test severity calculation."""
        test_cases = [
            (0.95, "critical"),
            (0.8, "high"),
            (0.6, "medium"),
            (0.3, "low"),
        ]
        
        for score_value, expected_severity in test_cases:
            anomaly = Anomaly(
                score=score_value,
                data_point={"feature": 1.0},
                detector_name="detector"
            )
            assert anomaly.severity == expected_severity

    def test_anomaly_with_anomaly_score_severity(self):
        """Test severity calculation with AnomalyScore."""
        score = AnomalyScore(0.85)
        anomaly = Anomaly(
            score=score,
            data_point={"feature": 1.0},
            detector_name="detector"
        )
        
        assert anomaly.severity == "high"

    def test_anomaly_add_metadata(self):
        """Test adding metadata to anomaly."""
        anomaly = Anomaly(
            score=0.7,
            data_point={"feature": 1.0},
            detector_name="detector"
        )
        
        anomaly.add_metadata("source", "test")
        anomaly.add_metadata("confidence", 0.9)
        
        assert anomaly.metadata["source"] == "test"
        assert anomaly.metadata["confidence"] == 0.9

    def test_anomaly_to_dict(self):
        """Test converting anomaly to dictionary."""
        anomaly = Anomaly(
            score=0.8,
            data_point={"feature1": 1.0, "feature2": 2.0},
            detector_name="test_detector",
            explanation="Test explanation"
        )
        
        result = anomaly.to_dict()
        
        assert result["score"] == 0.8
        assert result["detector_name"] == "test_detector"
        assert result["data_point"] == {"feature1": 1.0, "feature2": 2.0}
        assert result["severity"] == "high"
        assert result["explanation"] == "Test explanation"
        assert "id" in result
        assert "timestamp" in result

    def test_anomaly_equality_based_on_id(self):
        """Test that anomaly equality is based on ID from BaseEntity."""
        data_point = {"feature": 1.0}
        
        anomaly1 = Anomaly(
            score=0.7,
            data_point=data_point,
            detector_name="detector1"
        )
        
        anomaly2 = Anomaly(
            score=0.8,
            data_point=data_point,
            detector_name="detector2",
            id=anomaly1.id  # Same ID
        )
        
        anomaly3 = Anomaly(
            score=0.7,
            data_point=data_point,
            detector_name="detector1"
            # Different ID
        )
        
        assert anomaly1 == anomaly2  # Same ID
        assert anomaly1 != anomaly3  # Different ID

    def test_anomaly_base_entity_methods(self):
        """Test that BaseEntity methods work with Anomaly."""
        anomaly = Anomaly(
            score=0.7,
            data_point={"feature": 1.0},
            detector_name="detector"
        )
        
        # Test is_new
        assert anomaly.is_new()
        
        # Test mark_as_updated
        original_version = anomaly.version
        original_updated_at = anomaly.updated_at
        
        import time
        time.sleep(0.01)
        
        anomaly.mark_as_updated()
        
        assert anomaly.version == original_version + 1
        assert anomaly.updated_at > original_updated_at
        assert not anomaly.is_new()
        
        # Test validate_invariants
        anomaly.validate_invariants()  # Should not raise

    def test_anomaly_repr_includes_base_entity_info(self):
        """Test that repr includes BaseEntity information."""
        anomaly = Anomaly(
            score=0.7,
            data_point={"feature": 1.0},
            detector_name="detector"
        )
        
        repr_str = repr(anomaly)
        assert "Anomaly" in repr_str
        assert str(anomaly.id) in repr_str

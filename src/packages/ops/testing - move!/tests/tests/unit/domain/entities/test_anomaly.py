"""Comprehensive tests for Anomaly domain entity."""

import json
from datetime import datetime
from uuid import uuid4

import pytest

from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects import AnomalyScore


class TestAnomalyInitialization:
    """Test anomaly initialization and validation."""

    def test_anomaly_initialization_with_float_score(self):
        """Test anomaly initialization with float score."""
        data_point = {"feature1": 1.5, "feature2": 2.0}
        anomaly = Anomaly(
            score=0.8, data_point=data_point, detector_name="Test Detector"
        )

        assert anomaly.score == 0.8
        assert anomaly.data_point == data_point
        assert anomaly.detector_name == "Test Detector"
        assert isinstance(anomaly.id, type(uuid4()))
        assert isinstance(anomaly.timestamp, datetime)
        assert anomaly.metadata == {}
        assert anomaly.explanation is None

    def test_anomaly_initialization_with_anomaly_score(self):
        """Test anomaly initialization with AnomalyScore object."""
        score = AnomalyScore(value=0.75, threshold=0.5)
        data_point = {"feature1": 1.5, "feature2": 2.0}

        anomaly = Anomaly(
            score=score, data_point=data_point, detector_name="Test Detector"
        )

        assert anomaly.score == score
        assert anomaly.data_point == data_point
        assert anomaly.detector_name == "Test Detector"

    def test_anomaly_initialization_with_all_fields(self):
        """Test anomaly initialization with all fields specified."""
        score = AnomalyScore(value=0.9, threshold=0.6)
        data_point = {"feature1": 100, "feature2": 200}
        metadata = {"source": "test", "index": 5}
        explanation = "High deviation from normal pattern"

        anomaly = Anomaly(
            score=score,
            data_point=data_point,
            detector_name="Advanced Detector",
            metadata=metadata,
            explanation=explanation,
        )

        assert anomaly.score == score
        assert anomaly.data_point == data_point
        assert anomaly.detector_name == "Advanced Detector"
        assert anomaly.metadata == metadata
        assert anomaly.explanation == explanation

    def test_anomaly_initialization_with_integer_score(self):
        """Test anomaly initialization with integer score."""
        anomaly = Anomaly(
            score=1, data_point={"value": 42}, detector_name="Test Detector"
        )

        assert anomaly.score == 1
        assert isinstance(anomaly.score, int)

    def test_anomaly_validation_invalid_score_type(self):
        """Test anomaly validation with invalid score type."""
        with pytest.raises(TypeError, match="Score must be a number or AnomalyScore"):
            Anomaly(
                score="invalid",
                data_point={"feature1": 1.0},
                detector_name="Test Detector",
            )

    def test_anomaly_validation_empty_detector_name(self):
        """Test anomaly validation with empty detector name."""
        with pytest.raises(ValueError, match="Detector name cannot be empty"):
            Anomaly(score=0.8, data_point={"feature1": 1.0}, detector_name="")

    def test_anomaly_validation_invalid_data_point_type(self):
        """Test anomaly validation with invalid data point type."""
        with pytest.raises(TypeError, match="Data point must be a dictionary"):
            Anomaly(
                score=0.8,
                data_point=[1, 2, 3],  # Should be dict
                detector_name="Test Detector",
            )

    def test_anomaly_validation_none_data_point(self):
        """Test anomaly validation with None data point."""
        with pytest.raises(TypeError, match="Data point must be a dictionary"):
            Anomaly(score=0.8, data_point=None, detector_name="Test Detector")


class TestAnomalySeverity:
    """Test anomaly severity classification."""

    def test_severity_critical(self):
        """Test critical severity classification."""
        anomaly = Anomaly(
            score=0.95, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly.severity == "critical"

    def test_severity_high(self):
        """Test high severity classification."""
        anomaly = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly.severity == "high"

    def test_severity_medium(self):
        """Test medium severity classification."""
        anomaly = Anomaly(
            score=0.6, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly.severity == "medium"

    def test_severity_low(self):
        """Test low severity classification."""
        anomaly = Anomaly(
            score=0.3, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly.severity == "low"

    def test_severity_with_anomaly_score_object(self):
        """Test severity classification with AnomalyScore object."""
        score = AnomalyScore(value=0.85, threshold=0.5)
        anomaly = Anomaly(
            score=score, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly.severity == "high"

    def test_severity_boundary_values(self):
        """Test severity classification at boundary values."""
        # Test exact boundary values
        anomaly_critical = Anomaly(
            score=0.9, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly_critical.severity == "critical"

        anomaly_high = Anomaly(
            score=0.7, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly_high.severity == "high"

        anomaly_medium = Anomaly(
            score=0.5, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly_medium.severity == "medium"

        # Test just above boundary
        anomaly_above_critical = Anomaly(
            score=0.9001, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly_above_critical.severity == "critical"

        # Test just below boundary
        anomaly_below_critical = Anomaly(
            score=0.8999, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly_below_critical.severity == "high"


class TestAnomalyMethods:
    """Test anomaly methods."""

    def test_add_metadata(self):
        """Test adding metadata to anomaly."""
        anomaly = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )

        anomaly.add_metadata("source", "test")
        anomaly.add_metadata("index", 42)

        assert anomaly.metadata["source"] == "test"
        assert anomaly.metadata["index"] == 42

    def test_add_metadata_overwrite(self):
        """Test overwriting existing metadata."""
        anomaly = Anomaly(
            score=0.8,
            data_point={"feature1": 1.0},
            detector_name="Test Detector",
            metadata={"version": "1.0"},
        )

        anomaly.add_metadata("version", "2.0")
        assert anomaly.metadata["version"] == "2.0"

    def test_add_metadata_preserves_existing(self):
        """Test that adding metadata preserves existing entries."""
        anomaly = Anomaly(
            score=0.8,
            data_point={"feature1": 1.0},
            detector_name="Test Detector",
            metadata={"existing": "value"},
        )

        anomaly.add_metadata("new", "value")
        assert anomaly.metadata["existing"] == "value"
        assert anomaly.metadata["new"] == "value"

    def test_to_dict_with_float_score(self):
        """Test converting anomaly to dictionary with float score."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        data_point = {"feature1": 1.5, "feature2": 2.0}
        metadata = {"source": "test", "index": 5}

        anomaly = Anomaly(
            score=0.8,
            data_point=data_point,
            detector_name="Test Detector",
            metadata=metadata,
            explanation="Test explanation",
        )
        anomaly.timestamp = timestamp  # Set for predictable testing

        result = anomaly.to_dict()

        assert result["score"] == 0.8
        assert result["detector_name"] == "Test Detector"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["data_point"] == data_point
        assert result["metadata"] == metadata
        assert result["severity"] == "high"
        assert result["explanation"] == "Test explanation"
        assert "id" in result

    def test_to_dict_with_anomaly_score(self):
        """Test converting anomaly to dictionary with AnomalyScore object."""
        score = AnomalyScore(value=0.75, threshold=0.5)
        data_point = {"feature1": 1.5}

        anomaly = Anomaly(
            score=score, data_point=data_point, detector_name="Test Detector"
        )

        result = anomaly.to_dict()

        assert result["score"] == 0.75
        assert result["severity"] == "high"

    def test_to_dict_with_none_explanation(self):
        """Test converting anomaly to dictionary with None explanation."""
        anomaly = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )

        result = anomaly.to_dict()
        assert result["explanation"] is None

    def test_to_dict_serializable(self):
        """Test that to_dict produces JSON-serializable output."""
        anomaly = Anomaly(
            score=0.8,
            data_point={"feature1": 1.0, "feature2": "text"},
            detector_name="Test Detector",
            metadata={"nested": {"key": "value"}},
        )

        result = anomaly.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should be deserializable
        deserialized = json.loads(json_str)
        assert deserialized["score"] == 0.8
        assert deserialized["detector_name"] == "Test Detector"


class TestAnomalyEquality:
    """Test anomaly equality and hashing."""

    def test_equality_same_id(self):
        """Test equality with same ID."""
        anomaly_id = uuid4()

        anomaly1 = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        anomaly1.id = anomaly_id

        anomaly2 = Anomaly(
            score=0.9,  # Different score
            data_point={"feature2": 2.0},  # Different data point
            detector_name="Different Detector",  # Different detector
        )
        anomaly2.id = anomaly_id

        assert anomaly1 == anomaly2

    def test_equality_different_id(self):
        """Test equality with different IDs."""
        anomaly1 = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )

        anomaly2 = Anomaly(
            score=0.8,  # Same score
            data_point={"feature1": 1.0},  # Same data point
            detector_name="Test Detector",  # Same detector
        )

        assert anomaly1 != anomaly2  # Different IDs

    def test_equality_with_non_anomaly(self):
        """Test equality with non-Anomaly object."""
        anomaly = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )

        assert anomaly != "not an anomaly"
        assert anomaly != 42
        assert anomaly != None

    def test_hash_consistency(self):
        """Test that hash is consistent and based on ID."""
        anomaly = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )

        # Hash should be consistent
        hash1 = hash(anomaly)
        hash2 = hash(anomaly)
        assert hash1 == hash2

        # Hash should be based on ID
        assert hash(anomaly) == hash(anomaly.id)

    def test_hash_different_for_different_anomalies(self):
        """Test that different anomalies have different hashes."""
        anomaly1 = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )

        anomaly2 = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )

        assert hash(anomaly1) != hash(anomaly2)

    def test_set_operations(self):
        """Test anomaly in set operations."""
        anomaly1 = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )

        anomaly2 = Anomaly(
            score=0.9, data_point={"feature2": 2.0}, detector_name="Test Detector"
        )

        anomaly3 = Anomaly(
            score=0.7, data_point={"feature3": 3.0}, detector_name="Test Detector"
        )

        # Test set creation
        anomaly_set = {anomaly1, anomaly2, anomaly3}
        assert len(anomaly_set) == 3

        # Test membership
        assert anomaly1 in anomaly_set
        assert anomaly2 in anomaly_set
        assert anomaly3 in anomaly_set

        # Test duplicate handling
        anomaly_set.add(anomaly1)  # Add duplicate
        assert len(anomaly_set) == 3  # Should still be 3


class TestAnomalyEdgeCases:
    """Test anomaly edge cases and error conditions."""

    def test_anomaly_with_complex_data_point(self):
        """Test anomaly with complex data point structure."""
        complex_data = {
            "numeric": 42.5,
            "string": "text_value",
            "nested": {"level1": {"level2": "deep_value"}},
            "list": [1, 2, 3],
            "boolean": True,
            "null_value": None,
        }

        anomaly = Anomaly(
            score=0.8, data_point=complex_data, detector_name="Test Detector"
        )

        assert anomaly.data_point == complex_data

        # Test that to_dict preserves complex structure
        result = anomaly.to_dict()
        assert result["data_point"] == complex_data

    def test_anomaly_with_empty_data_point(self):
        """Test anomaly with empty data point."""
        anomaly = Anomaly(score=0.8, data_point={}, detector_name="Test Detector")

        assert anomaly.data_point == {}
        assert anomaly.severity == "high"

    def test_anomaly_with_large_metadata(self):
        """Test anomaly with large metadata structure."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}

        anomaly = Anomaly(
            score=0.8,
            data_point={"feature1": 1.0},
            detector_name="Test Detector",
            metadata=large_metadata,
        )

        assert len(anomaly.metadata) == 1000
        assert anomaly.metadata["key_500"] == "value_500"

    def test_anomaly_with_unicode_strings(self):
        """Test anomaly with unicode strings."""
        unicode_data = {
            "unicode_feature": "测试数据",
            "emoji": "🔥💻🚀",
            "mixed": "Hello 世界 🌍",
        }

        anomaly = Anomaly(
            score=0.8,
            data_point=unicode_data,
            detector_name="Unicode Detector 测试",
            explanation="Unicode explanation 解释",
        )

        assert anomaly.data_point == unicode_data
        assert anomaly.detector_name == "Unicode Detector 测试"
        assert anomaly.explanation == "Unicode explanation 解释"

        # Test that to_dict preserves unicode
        result = anomaly.to_dict()
        assert result["data_point"] == unicode_data

    def test_anomaly_score_boundary_edge_cases(self):
        """Test anomaly score at exact boundary values."""
        # Test exactly at critical boundary
        anomaly_critical_exact = Anomaly(
            score=0.9, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly_critical_exact.severity == "critical"

        # Test just below critical
        anomaly_below_critical = Anomaly(
            score=0.8999999, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly_below_critical.severity == "high"

        # Test zero score
        anomaly_zero = Anomaly(
            score=0.0, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )
        assert anomaly_zero.severity == "low"

    def test_anomaly_with_extreme_values(self):
        """Test anomaly with extreme values in data point."""
        extreme_data = {
            "very_large": 1e100,
            "very_small": 1e-100,
            "negative": -1e50,
            "infinity": float("inf"),
            "neg_infinity": float("-inf"),
            "not_a_number": float("nan"),
        }

        anomaly = Anomaly(
            score=0.8, data_point=extreme_data, detector_name="Test Detector"
        )

        assert anomaly.data_point == extreme_data

        # Test that to_dict handles extreme values
        result = anomaly.to_dict()
        assert result["data_point"]["very_large"] == 1e100
        assert result["data_point"]["very_small"] == 1e-100
        assert result["data_point"]["negative"] == -1e50

    def test_anomaly_timestamp_immutability(self):
        """Test that modifying timestamp after creation doesn't affect original."""
        anomaly = Anomaly(
            score=0.8, data_point={"feature1": 1.0}, detector_name="Test Detector"
        )

        original_timestamp = anomaly.timestamp

        # Modify timestamp
        new_timestamp = datetime(2023, 1, 1, 12, 0, 0)
        anomaly.timestamp = new_timestamp

        # Verify timestamp was changed
        assert anomaly.timestamp == new_timestamp
        assert anomaly.timestamp != original_timestamp

    def test_anomaly_metadata_mutability(self):
        """Test that metadata can be modified after creation."""
        original_metadata = {"key1": "value1"}
        anomaly = Anomaly(
            score=0.8,
            data_point={"feature1": 1.0},
            detector_name="Test Detector",
            metadata=original_metadata,
        )

        # Modify metadata directly
        anomaly.metadata["key2"] = "value2"

        # Verify metadata was modified
        assert anomaly.metadata["key1"] == "value1"
        assert anomaly.metadata["key2"] == "value2"

        # Verify original dict was not modified
        assert "key2" not in original_metadata

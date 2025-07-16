"""
Comprehensive tests for AnomalyScore value object.

This module tests the AnomalyScore value object to ensure proper validation,
behavior, and immutability across all use cases.
"""

import json
import math
from unittest.mock import Mock

import pytest

from monorepo.domain.exceptions import ValidationError
from monorepo.domain.value_objects import AnomalyScore


class MockConfidenceInterval:
    """Mock confidence interval for testing."""

    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper

    def contains(self, value: float) -> bool:
        """Check if value is within interval."""
        return self.lower <= value <= self.upper


class TestAnomalyScoreCreation:
    """Test AnomalyScore creation and validation."""

    def test_basic_creation(self):
        """Test basic anomaly score creation."""
        score = AnomalyScore(value=0.8)

        assert score.value == 0.8
        assert score.threshold == 0.5  # default
        assert score.metadata == {}  # default
        assert score.confidence_interval is None
        assert score.method is None

    def test_creation_with_all_parameters(self):
        """Test creation with all parameters."""
        metadata = {"algorithm": "isolation_forest", "version": "1.0"}
        confidence_interval = MockConfidenceInterval(0.7, 0.9)

        score = AnomalyScore(
            value=0.8,
            threshold=0.6,
            metadata=metadata,
            confidence_interval=confidence_interval,
            method="statistical",
        )

        assert score.value == 0.8
        assert score.threshold == 0.6
        assert score.metadata == metadata
        assert score.metadata is not metadata  # should be a copy
        assert score.confidence_interval == confidence_interval
        assert score.method == "statistical"

    def test_metadata_deep_copy(self):
        """Test that metadata is deep copied."""
        original_metadata = {"nested": {"key": "value"}}
        score = AnomalyScore(value=0.5, metadata=original_metadata)

        # Modify original metadata
        original_metadata["nested"]["key"] = "modified"

        # Score metadata should be unchanged
        assert score.metadata["nested"]["key"] == "value"

    def test_none_metadata_becomes_empty_dict(self):
        """Test that None metadata becomes empty dict."""
        score = AnomalyScore(value=0.5, metadata=None)

        assert score.metadata == {}
        assert isinstance(score.metadata, dict)


class TestAnomalyScoreValidation:
    """Test validation of AnomalyScore parameters."""

    def test_valid_score_values(self):
        """Test valid score values."""
        valid_values = [0.0, 0.5, 1.0, 0.123, 0.999]

        for value in valid_values:
            score = AnomalyScore(value=value)
            assert score.value == value

    def test_invalid_score_values(self):
        """Test invalid score values."""
        invalid_values = [-0.1, 1.1, 2.0, -1.0, math.nan, math.inf, -math.inf]

        for value in invalid_values:
            with pytest.raises(ValidationError):
                AnomalyScore(value=value)

    def test_non_numeric_score_value(self):
        """Test non-numeric score values."""
        invalid_values = ["0.5", None, [], {}]

        for value in invalid_values:
            with pytest.raises(ValidationError, match="Score value must be numeric"):
                AnomalyScore(value=value)

    def test_valid_threshold_values(self):
        """Test valid threshold values."""
        valid_thresholds = [0.0, 0.5, 1.0, 0.1, 0.9]

        for threshold in valid_thresholds:
            score = AnomalyScore(value=0.5, threshold=threshold)
            assert score.threshold == threshold

    def test_invalid_threshold_values(self):
        """Test invalid threshold values."""
        invalid_thresholds = [-0.1, 1.1, 2.0, -1.0, math.nan, math.inf]

        for threshold in invalid_thresholds:
            with pytest.raises(ValidationError):
                AnomalyScore(value=0.5, threshold=threshold)

    def test_non_numeric_threshold(self):
        """Test non-numeric threshold values."""
        invalid_thresholds = ["0.5", None, [], {}]

        for threshold in invalid_thresholds:
            with pytest.raises(ValidationError, match="Threshold must be numeric"):
                AnomalyScore(value=0.5, threshold=threshold)

    def test_invalid_metadata_type(self):
        """Test invalid metadata types."""
        invalid_metadata = ["string", 123, [], None]

        for metadata in invalid_metadata[:-1]:  # Exclude None
            with pytest.raises(ValidationError, match="Metadata must be a dictionary"):
                AnomalyScore(value=0.5, metadata=metadata)

    def test_valid_confidence_interval(self):
        """Test valid confidence interval."""
        confidence_interval = MockConfidenceInterval(0.6, 0.9)
        score = AnomalyScore(value=0.8, confidence_interval=confidence_interval)

        assert score.confidence_interval == confidence_interval

    def test_invalid_confidence_interval_missing_methods(self):
        """Test confidence interval missing required methods."""
        # Create mocks that don't have the required attributes
        mock_missing_all = Mock()
        # Remove the attributes if they exist
        if hasattr(mock_missing_all, "contains"):
            delattr(mock_missing_all, "contains")
        if hasattr(mock_missing_all, "lower"):
            delattr(mock_missing_all, "lower")
        if hasattr(mock_missing_all, "upper"):
            delattr(mock_missing_all, "upper")

        with pytest.raises(
            ValidationError,
            match="Confidence interval must have 'contains', 'lower', and 'upper' attributes",
        ):
            AnomalyScore(value=0.5, confidence_interval=mock_missing_all)

    def test_score_outside_confidence_interval(self):
        """Test score outside confidence interval."""
        confidence_interval = MockConfidenceInterval(0.1, 0.3)

        with pytest.raises(
            ValidationError, match="Score value.*must be within confidence interval"
        ):
            AnomalyScore(value=0.8, confidence_interval=confidence_interval)


class TestAnomalyScoreBehavior:
    """Test AnomalyScore behavior and methods."""

    def test_is_anomaly_above_threshold(self):
        """Test is_anomaly when score is above threshold."""
        score = AnomalyScore(value=0.8, threshold=0.5)
        assert score.is_anomaly() is True

    def test_is_anomaly_below_threshold(self):
        """Test is_anomaly when score is below threshold."""
        score = AnomalyScore(value=0.3, threshold=0.5)
        assert score.is_anomaly() is False

    def test_is_anomaly_equal_threshold(self):
        """Test is_anomaly when score equals threshold."""
        score = AnomalyScore(value=0.5, threshold=0.5)
        assert score.is_anomaly() is False  # Equal is not greater

    def test_confidence_level_calculation(self):
        """Test confidence level calculation."""
        test_cases = [
            (0.8, 0.5, 0.3),  # Above threshold: |0.8 - 0.5| = 0.3
            (0.2, 0.5, 0.3),  # Below threshold: |0.2 - 0.5| = 0.3
            (0.5, 0.5, 0.0),  # Equal to threshold: |0.5 - 0.5| = 0.0
        ]

        for value, threshold, expected in test_cases:
            score = AnomalyScore(value=value, threshold=threshold)
            assert abs(score.confidence_level() - expected) < 1e-10

    def test_is_confident_with_interval(self):
        """Test is_confident with confidence interval."""
        confidence_interval = MockConfidenceInterval(0.7, 0.9)
        score = AnomalyScore(value=0.8, confidence_interval=confidence_interval)

        assert score.is_confident is True

    def test_is_confident_without_interval(self):
        """Test is_confident without confidence interval."""
        score = AnomalyScore(value=0.8)

        assert score.is_confident is False

    def test_confidence_width_with_interval(self):
        """Test confidence width calculation."""
        confidence_interval = MockConfidenceInterval(0.7, 0.9)
        score = AnomalyScore(value=0.8, confidence_interval=confidence_interval)

        assert abs(score.confidence_width - 0.2) < 1e-10

    def test_confidence_width_without_interval(self):
        """Test confidence width without interval."""
        score = AnomalyScore(value=0.8)

        assert score.confidence_width is None

    def test_confidence_bounds_with_interval(self):
        """Test confidence bounds with interval."""
        confidence_interval = MockConfidenceInterval(0.7, 0.9)
        score = AnomalyScore(value=0.8, confidence_interval=confidence_interval)

        assert score.confidence_lower == 0.7
        assert score.confidence_upper == 0.9

    def test_confidence_bounds_without_interval(self):
        """Test confidence bounds without interval."""
        score = AnomalyScore(value=0.8)

        assert score.confidence_lower is None
        assert score.confidence_upper is None

    def test_is_valid_method(self):
        """Test is_valid method."""
        valid_score = AnomalyScore(value=0.8)
        assert valid_score.is_valid() is True

        # Test with edge cases that would be valid
        edge_cases = [
            AnomalyScore(value=0.0),
            AnomalyScore(value=1.0),
            AnomalyScore(value=0.5),
        ]

        for score in edge_cases:
            assert score.is_valid() is True

    def test_exceeds_threshold_method(self):
        """Test exceeds_threshold method."""
        score = AnomalyScore(value=0.8)

        assert score.exceeds_threshold(0.5) is True
        assert score.exceeds_threshold(0.8) is False
        assert score.exceeds_threshold(0.9) is False


class TestAnomalyScoreComparison:
    """Test AnomalyScore comparison operations."""

    def test_equality_same_scores(self):
        """Test equality of identical scores."""
        score1 = AnomalyScore(value=0.8, threshold=0.5, metadata={"key": "value"})
        score2 = AnomalyScore(value=0.8, threshold=0.5, metadata={"key": "value"})

        assert score1 == score2

    def test_equality_different_values(self):
        """Test inequality of different values."""
        score1 = AnomalyScore(value=0.8)
        score2 = AnomalyScore(value=0.7)

        assert score1 != score2

    def test_equality_different_thresholds(self):
        """Test inequality of different thresholds."""
        score1 = AnomalyScore(value=0.8, threshold=0.5)
        score2 = AnomalyScore(value=0.8, threshold=0.6)

        assert score1 != score2

    def test_equality_different_metadata(self):
        """Test inequality of different metadata."""
        score1 = AnomalyScore(value=0.8, metadata={"key": "value1"})
        score2 = AnomalyScore(value=0.8, metadata={"key": "value2"})

        assert score1 != score2

    def test_equality_with_non_score(self):
        """Test equality with non-AnomalyScore objects."""
        score = AnomalyScore(value=0.8)

        assert score != 0.8
        assert score != "0.8"
        assert score != None
        assert score != []

    def test_comparison_with_other_scores(self):
        """Test comparison operations with other scores."""
        score1 = AnomalyScore(value=0.3)
        score2 = AnomalyScore(value=0.7)
        score3 = AnomalyScore(value=0.7)

        assert score1 < score2
        assert score1 <= score2
        assert score2 > score1
        assert score2 >= score1
        assert score2 <= score3
        assert score2 >= score3

    def test_comparison_with_numbers(self):
        """Test comparison operations with numbers."""
        score = AnomalyScore(value=0.7)

        assert score > 0.5
        assert score >= 0.7
        assert score < 0.9
        assert score <= 0.7
        assert not (score > 0.7)
        assert not (score < 0.7)

    def test_comparison_with_invalid_types(self):
        """Test comparison with invalid types."""
        score = AnomalyScore(value=0.7)

        with pytest.raises(TypeError):
            score < "0.5"

        with pytest.raises(TypeError):
            score > []


class TestAnomalyScoreSerialization:
    """Test AnomalyScore serialization and deserialization."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        score = AnomalyScore(value=0.8, threshold=0.6, metadata={"key": "value"})

        result = score.to_dict()

        expected = {"value": 0.8, "threshold": 0.6, "metadata": {"key": "value"}}
        assert result == expected

    def test_to_dict_metadata_copy(self):
        """Test that to_dict returns metadata copy."""
        metadata = {"key": "value"}
        score = AnomalyScore(value=0.8, metadata=metadata)

        result = score.to_dict()
        result["metadata"]["key"] = "modified"

        # Original score metadata should be unchanged
        assert score.metadata["key"] == "value"

    def test_from_dict_basic(self):
        """Test basic from_dict creation."""
        data = {"value": 0.8, "threshold": 0.6, "metadata": {"key": "value"}}

        score = AnomalyScore.from_dict(data)

        assert score.value == 0.8
        assert score.threshold == 0.6
        assert score.metadata == {"key": "value"}

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {"value": 0.8}

        score = AnomalyScore.from_dict(data)

        assert score.value == 0.8
        assert score.threshold == 0.5  # default
        assert score.metadata == {}  # default

    def test_from_dict_missing_value(self):
        """Test from_dict with missing value."""
        data = {"threshold": 0.6}

        with pytest.raises(ValidationError, match="Missing required field: value"):
            AnomalyScore.from_dict(data)

    def test_to_json_conversion(self):
        """Test JSON serialization."""
        score = AnomalyScore(value=0.8, threshold=0.6, metadata={"key": "value"})

        json_str = score.to_json()
        parsed = json.loads(json_str)

        expected = {"value": 0.8, "threshold": 0.6, "metadata": {"key": "value"}}
        assert parsed == expected

    def test_from_json_conversion(self):
        """Test JSON deserialization."""
        json_str = '{"value": 0.8, "threshold": 0.6, "metadata": {"key": "value"}}'

        score = AnomalyScore.from_json(json_str)

        assert score.value == 0.8
        assert score.threshold == 0.6
        assert score.metadata == {"key": "value"}

    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        original = AnomalyScore(value=0.8, threshold=0.6, metadata={"key": "value"})

        json_str = original.to_json()
        reconstructed = AnomalyScore.from_json(json_str)

        assert original == reconstructed


class TestAnomalyScoreStringRepresentation:
    """Test AnomalyScore string representations."""

    def test_str_representation(self):
        """Test string representation."""
        score = AnomalyScore(value=0.8)

        assert str(score) == "0.8"

    def test_repr_representation(self):
        """Test detailed representation."""
        score = AnomalyScore(value=0.8, threshold=0.6, metadata={"key": "value"})

        repr_str = repr(score)

        assert "AnomalyScore" in repr_str
        assert "value=0.8" in repr_str
        assert "threshold=0.6" in repr_str
        assert "metadata=" in repr_str

    def test_bool_conversion_nonzero(self):
        """Test boolean conversion for non-zero scores."""
        score = AnomalyScore(value=0.8)

        assert bool(score) is True

    def test_bool_conversion_zero(self):
        """Test boolean conversion for zero scores."""
        score = AnomalyScore(value=0.0)

        assert bool(score) is False


class TestAnomalyScoreHashing:
    """Test AnomalyScore hashing behavior."""

    def test_hash_equal_scores(self):
        """Test that equal scores have equal hashes."""
        score1 = AnomalyScore(value=0.8, threshold=0.5, metadata={"key": "value"})
        score2 = AnomalyScore(value=0.8, threshold=0.5, metadata={"key": "value"})

        assert hash(score1) == hash(score2)

    def test_hash_different_scores(self):
        """Test that different scores have different hashes."""
        score1 = AnomalyScore(value=0.8)
        score2 = AnomalyScore(value=0.7)

        assert hash(score1) != hash(score2)

    def test_hash_set_usage(self):
        """Test using scores in sets."""
        score1 = AnomalyScore(value=0.8)
        score2 = AnomalyScore(value=0.7)
        score3 = AnomalyScore(value=0.8)  # Same as score1

        score_set = {score1, score2, score3}

        assert len(score_set) == 2  # score1 and score3 are the same
        assert score1 in score_set
        assert score2 in score_set

    def test_hash_dict_key_usage(self):
        """Test using scores as dictionary keys."""
        score1 = AnomalyScore(value=0.8)
        score2 = AnomalyScore(value=0.7)

        score_dict = {score1: "high", score2: "medium"}

        assert len(score_dict) == 2
        assert score_dict[score1] == "high"
        assert score_dict[score2] == "medium"


class TestAnomalyScoreImmutability:
    """Test AnomalyScore immutability."""

    def test_frozen_dataclass(self):
        """Test that AnomalyScore is frozen."""
        score = AnomalyScore(value=0.8)

        with pytest.raises(AttributeError):
            score.value = 0.9

        with pytest.raises(AttributeError):
            score.threshold = 0.6

    def test_metadata_mutation_protection(self):
        """Test that metadata cannot be mutated through the score."""
        original_metadata = {"key": "value"}
        score = AnomalyScore(value=0.8, metadata=original_metadata)

        # Modifying original metadata should not affect score
        original_metadata["key"] = "modified"
        assert score.metadata["key"] == "value"

        # Note: The AnomalyScore metadata property doesn't return a copy for direct access,
        # so we test that the original creation made a deep copy
        assert score.metadata == {"key": "value"}


class TestAnomalyScoreEdgeCases:
    """Test AnomalyScore edge cases and boundary conditions."""

    def test_boundary_values(self):
        """Test boundary values for score and threshold."""
        # Test exact boundaries
        boundary_cases = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ]

        for value, threshold in boundary_cases:
            score = AnomalyScore(value=value, threshold=threshold)
            assert score.value == value
            assert score.threshold == threshold

    def test_floating_point_precision(self):
        """Test floating point precision handling."""
        # Test with high precision values
        precise_value = 0.123456789012345
        score = AnomalyScore(value=precise_value)

        assert score.value == precise_value

    def test_large_metadata(self):
        """Test with large metadata dictionaries."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
        score = AnomalyScore(value=0.5, metadata=large_metadata)

        assert len(score.metadata) == 1000
        assert score.metadata["key_500"] == "value_500"

    def test_nested_metadata(self):
        """Test with deeply nested metadata."""
        nested_metadata = {
            "level1": {"level2": {"level3": {"level4": {"key": "deep_value"}}}}
        }
        score = AnomalyScore(value=0.5, metadata=nested_metadata)

        assert (
            score.metadata["level1"]["level2"]["level3"]["level4"]["key"]
            == "deep_value"
        )

    def test_unicode_in_metadata(self):
        """Test with unicode characters in metadata."""
        unicode_metadata = {
            "emoji": "🚨",
            "chinese": "异常检测",
            "arabic": "كشف الشذوذ",
            "special_chars": "åäö",
        }
        score = AnomalyScore(value=0.5, metadata=unicode_metadata)

        assert score.metadata["emoji"] == "🚨"
        assert score.metadata["chinese"] == "异常检测"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

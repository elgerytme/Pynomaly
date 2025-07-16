"""
Comprehensive unit tests for AnomalyScore value object.

This test suite covers:
- Value object creation and validation
- Equality and comparison operations
- Serialization/deserialization
- Edge cases and error conditions
- Performance characteristics
- Type safety and invariants
"""

import json

import pytest

from pynomaly.domain.exceptions import ValidationError
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore


class TestAnomalyScoreCreation:
    """Test creation and validation of AnomalyScore objects."""

    def test_create_valid_score(self):
        """Test creating valid anomaly scores."""
        score = AnomalyScore(0.5)
        assert score.value == 0.5
        assert score.is_anomaly() is False

    def test_create_score_at_boundaries(self):
        """Test creation at boundary values."""
        min_score = AnomalyScore(0.0)
        assert min_score.value == 0.0
        assert min_score.is_anomaly() is False

        max_score = AnomalyScore(1.0)
        assert max_score.value == 1.0
        assert max_score.is_anomaly() is True

    def test_create_score_with_threshold(self):
        """Test creation with custom threshold."""
        score = AnomalyScore(0.7, threshold=0.6)
        assert score.value == 0.7
        assert score.threshold == 0.6
        assert score.is_anomaly() is True

    def test_default_threshold(self):
        """Test default threshold value."""
        score = AnomalyScore(0.5)
        assert score.threshold == 0.5

    def test_create_score_with_metadata(self):
        """Test creation with metadata."""
        metadata = {"model": "IsolationForest", "confidence": 0.95}
        score = AnomalyScore(0.8, metadata=metadata)
        assert score.metadata == metadata
        assert score.metadata["model"] == "IsolationForest"

    def test_invalid_score_values(self):
        """Test validation of invalid score values."""
        with pytest.raises(ValidationError):
            AnomalyScore(-0.1)

        with pytest.raises(ValidationError):
            AnomalyScore(1.1)

        with pytest.raises(ValidationError):
            AnomalyScore(float("inf"))

        with pytest.raises(ValidationError):
            AnomalyScore(float("nan"))

    def test_invalid_threshold_values(self):
        """Test validation of invalid threshold values."""
        with pytest.raises(ValidationError):
            AnomalyScore(0.5, threshold=-0.1)

        with pytest.raises(ValidationError):
            AnomalyScore(0.5, threshold=1.1)

        with pytest.raises(ValidationError):
            AnomalyScore(0.5, threshold=float("inf"))

        with pytest.raises(ValidationError):
            AnomalyScore(0.5, threshold=float("nan"))

    def test_invalid_types(self):
        """Test validation of invalid types."""
        with pytest.raises(ValidationError):
            AnomalyScore("0.5")

        with pytest.raises(ValidationError):
            AnomalyScore(None)

        with pytest.raises(ValidationError):
            AnomalyScore(0.5, threshold="0.5")


class TestAnomalyScoreComparison:
    """Test comparison operations for AnomalyScore objects."""

    def test_equality(self):
        """Test equality comparison."""
        score1 = AnomalyScore(0.7)
        score2 = AnomalyScore(0.7)
        score3 = AnomalyScore(0.8)

        assert score1 == score2
        assert score1 != score3

    def test_equality_with_different_thresholds(self):
        """Test equality with different thresholds."""
        score1 = AnomalyScore(0.7, threshold=0.5)
        score2 = AnomalyScore(0.7, threshold=0.6)

        # Scores with different thresholds should not be equal
        assert score1 != score2

    def test_equality_with_different_metadata(self):
        """Test equality with different metadata."""
        score1 = AnomalyScore(0.7, metadata={"model": "A"})
        score2 = AnomalyScore(0.7, metadata={"model": "B"})

        # Scores with different metadata should not be equal
        assert score1 != score2

    def test_less_than_comparison(self):
        """Test less than comparison."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)

        assert score1 < score2
        assert not score2 < score1

    def test_less_than_equal_comparison(self):
        """Test less than or equal comparison."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)
        score3 = AnomalyScore(0.7)

        assert score1 <= score2
        assert score2 <= score3
        assert not score2 <= score1

    def test_greater_than_comparison(self):
        """Test greater than comparison."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)

        assert score2 > score1
        assert not score1 > score2

    def test_greater_than_equal_comparison(self):
        """Test greater than or equal comparison."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)
        score3 = AnomalyScore(0.7)

        assert score2 >= score1
        assert score2 >= score3
        assert not score1 >= score2

    def test_sorting(self):
        """Test sorting of anomaly scores."""
        scores = [
            AnomalyScore(0.8),
            AnomalyScore(0.2),
            AnomalyScore(0.5),
            AnomalyScore(0.9),
            AnomalyScore(0.1),
        ]

        sorted_scores = sorted(scores)
        expected_values = [0.1, 0.2, 0.5, 0.8, 0.9]

        for i, score in enumerate(sorted_scores):
            assert score.value == expected_values[i]


class TestAnomalyScoreClassification:
    """Test anomaly classification methods."""

    def test_is_anomaly_default_threshold(self):
        """Test is_anomaly with default threshold."""
        normal_score = AnomalyScore(0.3)
        anomaly_score = AnomalyScore(0.7)

        assert normal_score.is_anomaly() is False
        assert anomaly_score.is_anomaly() is True

    def test_is_anomaly_custom_threshold(self):
        """Test is_anomaly with custom threshold."""
        score = AnomalyScore(0.6, threshold=0.8)
        assert score.is_anomaly() is False

        score = AnomalyScore(0.9, threshold=0.8)
        assert score.is_anomaly() is True

    def test_is_anomaly_boundary_cases(self):
        """Test is_anomaly at boundary values."""
        score_at_threshold = AnomalyScore(0.5, threshold=0.5)
        assert score_at_threshold.is_anomaly() is False

        score_above_threshold = AnomalyScore(0.50001, threshold=0.5)
        assert score_above_threshold.is_anomaly() is True

    def test_confidence_level(self):
        """Test confidence level calculation."""
        # Score significantly below threshold
        low_score = AnomalyScore(0.1, threshold=0.5)
        assert low_score.confidence_level() == 0.4

        # Score at threshold
        threshold_score = AnomalyScore(0.5, threshold=0.5)
        assert threshold_score.confidence_level() == 0.0

        # Score significantly above threshold
        high_score = AnomalyScore(0.9, threshold=0.5)
        assert high_score.confidence_level() == 0.4

    def test_confidence_level_edge_cases(self):
        """Test confidence level at edge cases."""
        # Minimum score
        min_score = AnomalyScore(0.0, threshold=0.5)
        assert min_score.confidence_level() == 0.5

        # Maximum score
        max_score = AnomalyScore(1.0, threshold=0.5)
        assert max_score.confidence_level() == 0.5


class TestAnomalyScoreSerialization:
    """Test serialization and deserialization of AnomalyScore objects."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        score = AnomalyScore(0.7)
        result = score.to_dict()

        expected = {"value": 0.7, "threshold": 0.5, "metadata": {}}

        assert result == expected

    def test_to_dict_with_metadata(self):
        """Test to_dict with metadata."""
        metadata = {"model": "IsolationForest", "confidence": 0.95}
        score = AnomalyScore(0.7, metadata=metadata)
        result = score.to_dict()

        expected = {"value": 0.7, "threshold": 0.5, "metadata": metadata}

        assert result == expected

    def test_from_dict_basic(self):
        """Test basic from_dict creation."""
        data = {"value": 0.7, "threshold": 0.6, "metadata": {}}

        score = AnomalyScore.from_dict(data)
        assert score.value == 0.7
        assert score.threshold == 0.6
        assert score.metadata == {}

    def test_from_dict_with_metadata(self):
        """Test from_dict with metadata."""
        metadata = {"model": "IsolationForest", "confidence": 0.95}
        data = {"value": 0.7, "threshold": 0.6, "metadata": metadata}

        score = AnomalyScore.from_dict(data)
        assert score.value == 0.7
        assert score.threshold == 0.6
        assert score.metadata == metadata

    def test_from_dict_missing_required_fields(self):
        """Test from_dict with missing required fields."""
        with pytest.raises(ValidationError):
            AnomalyScore.from_dict({})

        with pytest.raises(ValidationError):
            AnomalyScore.from_dict({"threshold": 0.5})

    def test_from_dict_invalid_values(self):
        """Test from_dict with invalid values."""
        with pytest.raises(ValidationError):
            AnomalyScore.from_dict({"value": -0.1, "threshold": 0.5})

        with pytest.raises(ValidationError):
            AnomalyScore.from_dict({"value": 0.7, "threshold": 1.1})

    def test_json_serialization(self):
        """Test JSON serialization."""
        metadata = {"model": "IsolationForest", "confidence": 0.95}
        score = AnomalyScore(0.7, threshold=0.6, metadata=metadata)

        json_str = json.dumps(score.to_dict())
        data = json.loads(json_str)

        reconstructed = AnomalyScore.from_dict(data)
        assert reconstructed == score

    def test_round_trip_serialization(self):
        """Test round-trip serialization."""
        original = AnomalyScore(0.7, threshold=0.6, metadata={"test": "value"})
        serialized = original.to_dict()
        reconstructed = AnomalyScore.from_dict(serialized)

        assert reconstructed == original
        assert reconstructed.value == original.value
        assert reconstructed.threshold == original.threshold
        assert reconstructed.metadata == original.metadata


class TestAnomalyScoreUtilityMethods:
    """Test utility methods of AnomalyScore objects."""

    def test_repr(self):
        """Test string representation."""
        score = AnomalyScore(0.7)
        repr_str = repr(score)

        assert "AnomalyScore" in repr_str
        assert "0.7" in repr_str
        assert "0.5" in repr_str  # threshold

    def test_str(self):
        """Test string conversion."""
        score = AnomalyScore(0.7)
        str_repr = str(score)

        assert "0.7" in str_repr

    def test_hash(self):
        """Test hash functionality."""
        score1 = AnomalyScore(0.7)
        score2 = AnomalyScore(0.7)
        score3 = AnomalyScore(0.8)

        assert hash(score1) == hash(score2)
        assert hash(score1) != hash(score3)

        # Test in set
        score_set = {score1, score2, score3}
        assert len(score_set) == 2  # score1 and score2 are the same

    def test_bool_conversion(self):
        """Test boolean conversion."""
        zero_score = AnomalyScore(0.0)
        non_zero_score = AnomalyScore(0.7)

        assert bool(zero_score) is False
        assert bool(non_zero_score) is True


class TestAnomalyScoreEdgeCases:
    """Test edge cases and error conditions."""

    def test_floating_point_precision(self):
        """Test handling of floating-point precision issues."""
        # Test values very close to boundaries
        score = AnomalyScore(0.0000001)
        assert score.value == 0.0000001
        assert score.is_anomaly() is False

        score = AnomalyScore(0.9999999)
        assert score.value == 0.9999999
        assert score.is_anomaly() is True

    def test_very_small_differences(self):
        """Test handling of very small differences."""
        score1 = AnomalyScore(0.5000001)
        score2 = AnomalyScore(0.5000002)

        assert score1 != score2
        assert score1 < score2

    def test_metadata_mutation(self):
        """Test that metadata cannot be mutated externally."""
        original_metadata = {"model": "test"}
        score = AnomalyScore(0.7, metadata=original_metadata)

        # Modify original metadata
        original_metadata["model"] = "modified"

        # Score metadata should remain unchanged
        assert score.metadata["model"] == "test"

    def test_large_metadata(self):
        """Test handling of large metadata objects."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
        score = AnomalyScore(0.7, metadata=large_metadata)

        assert len(score.metadata) == 1000
        assert score.metadata["key_500"] == "value_500"


class TestAnomalyScorePerformance:
    """Test performance characteristics of AnomalyScore operations."""

    def test_creation_performance(self):
        """Test performance of creating many AnomalyScore objects."""
        import time

        start_time = time.time()
        scores = [AnomalyScore(i / 10000) for i in range(10000)]
        end_time = time.time()

        # Should complete in reasonable time (less than 1 second)
        assert end_time - start_time < 1.0
        assert len(scores) == 10000

    def test_comparison_performance(self):
        """Test performance of comparison operations."""
        import time

        scores = [AnomalyScore(i / 10000) for i in range(1000)]

        start_time = time.time()
        sorted_scores = sorted(scores)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 1.0
        assert len(sorted_scores) == 1000

    def test_serialization_performance(self):
        """Test performance of serialization operations."""
        import time

        scores = [AnomalyScore(i / 1000, metadata={"index": i}) for i in range(1000)]

        start_time = time.time()
        serialized = [score.to_dict() for score in scores]
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 1.0
        assert len(serialized) == 1000

        start_time = time.time()
        reconstructed = [AnomalyScore.from_dict(data) for data in serialized]
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 1.0
        assert len(reconstructed) == 1000


class TestAnomalyScoreIntegration:
    """Test integration scenarios with other components."""

    def test_with_different_models(self):
        """Test AnomalyScore with different model types."""
        models = ["IsolationForest", "LOF", "OneClassSVM", "AutoEncoder"]

        for model in models:
            score = AnomalyScore(0.7, metadata={"model": model})
            assert score.metadata["model"] == model
            assert score.is_anomaly() is True

    def test_batch_processing(self):
        """Test batch processing of scores."""
        raw_scores = [0.1, 0.3, 0.7, 0.9, 0.2]
        threshold = 0.5

        scores = [AnomalyScore(score, threshold=threshold) for score in raw_scores]
        anomalies = [score for score in scores if score.is_anomaly()]

        assert len(anomalies) == 2  # 0.7 and 0.9
        assert all(score.value > threshold for score in anomalies)

    def test_score_aggregation(self):
        """Test aggregation of multiple scores."""
        scores = [
            AnomalyScore(0.3),
            AnomalyScore(0.7),
            AnomalyScore(0.9),
            AnomalyScore(0.1),
        ]

        # Calculate average
        avg_score = sum(score.value for score in scores) / len(scores)
        assert avg_score == 0.5

        # Find maximum
        max_score = max(scores)
        assert max_score.value == 0.9

        # Find minimum
        min_score = min(scores)
        assert min_score.value == 0.1


class TestAnomalyScoreTypeSafety:
    """Test type safety and invariants."""

    def test_immutability(self):
        """Test that AnomalyScore objects are immutable."""
        score = AnomalyScore(0.7)

        # These should not be possible (would require property setters)
        with pytest.raises(AttributeError):
            score.value = 0.8

        with pytest.raises(AttributeError):
            score.threshold = 0.6

    def test_type_annotations(self):
        """Test that type annotations are correct."""
        score = AnomalyScore(0.7)

        # Check that value is float
        assert isinstance(score.value, float)
        assert isinstance(score.threshold, float)
        assert isinstance(score.metadata, dict)

    def test_method_return_types(self):
        """Test that methods return correct types."""
        score = AnomalyScore(0.7)

        assert isinstance(score.is_anomaly(), bool)
        assert isinstance(score.confidence_level(), float)
        assert isinstance(score.to_dict(), dict)
        assert isinstance(str(score), str)
        assert isinstance(repr(score), str)

    def test_comparison_type_safety(self):
        """Test type safety in comparisons."""
        score = AnomalyScore(0.7)

        # Should not be able to compare with non-AnomalyScore objects
        with pytest.raises(TypeError):
            score < 0.5

        with pytest.raises(TypeError):
            score > "0.8"

        with pytest.raises(TypeError):
            score == 0.7

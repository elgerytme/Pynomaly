"""Tests for AnomalyScore value object - corrected version matching actual implementation."""

import pytest
from unittest.mock import Mock

from pynomaly.domain.value_objects.anomaly_score import AnomalyScore
from pynomaly.domain.value_objects.confidence_interval import ConfidenceInterval
from pynomaly.domain.exceptions import InvalidValueError


class TestAnomalyScoreCreation:
    """Test creation and validation of AnomalyScore objects."""

    def test_create_valid_score_basic(self):
        """Test creating basic valid anomaly score."""
        score = AnomalyScore(0.5)
        assert score.value == 0.5
        assert score.confidence_interval is None
        assert score.method is None

    def test_create_score_with_confidence_interval(self):
        """Test creating score with confidence interval."""
        mock_ci = Mock(spec=ConfidenceInterval)
        mock_ci.contains.return_value = True
        mock_ci.lower = 0.3
        mock_ci.upper = 0.7
        
        score = AnomalyScore(0.5, confidence_interval=mock_ci)
        assert score.value == 0.5
        assert score.confidence_interval == mock_ci

    def test_create_score_with_method(self):
        """Test creating score with method."""
        score = AnomalyScore(0.5, method="IsolationForest")
        assert score.value == 0.5
        assert score.method == "IsolationForest"

    def test_create_score_complete(self):
        """Test creating score with all parameters."""
        mock_ci = Mock(spec=ConfidenceInterval)
        mock_ci.contains.return_value = True
        
        score = AnomalyScore(0.7, confidence_interval=mock_ci, method="LOF")
        assert score.value == 0.7
        assert score.confidence_interval == mock_ci
        assert score.method == "LOF"

    def test_boundary_values(self):
        """Test boundary values 0.0 and 1.0."""
        min_score = AnomalyScore(0.0)
        assert min_score.value == 0.0
        
        max_score = AnomalyScore(1.0)
        assert max_score.value == 1.0

    def test_invalid_value_types(self):
        """Test validation of invalid value types."""
        with pytest.raises(InvalidValueError, match="Score value must be numeric"):
            AnomalyScore("0.5")
        
        with pytest.raises(InvalidValueError, match="Score value must be numeric"):
            AnomalyScore(None)
        
        with pytest.raises(InvalidValueError, match="Score value must be numeric"):
            AnomalyScore([0.5])

    def test_invalid_value_range(self):
        """Test validation of values outside 0-1 range."""
        with pytest.raises(InvalidValueError, match="Score value must be between 0 and 1"):
            AnomalyScore(-0.1)
        
        with pytest.raises(InvalidValueError, match="Score value must be between 0 and 1"):
            AnomalyScore(1.1)
        
        with pytest.raises(InvalidValueError, match="Score value must be between 0 and 1"):
            AnomalyScore(2.0)

    def test_invalid_confidence_interval(self):
        """Test validation when score is outside confidence interval."""
        mock_ci = Mock(spec=ConfidenceInterval)
        mock_ci.contains.return_value = False
        mock_ci.lower = 0.3
        mock_ci.upper = 0.7
        
        with pytest.raises(InvalidValueError, match="Score value.*must be within confidence interval"):
            AnomalyScore(0.8, confidence_interval=mock_ci)


class TestAnomalyScoreProperties:
    """Test properties of AnomalyScore objects."""

    def test_is_valid_normal_values(self):
        """Test is_valid with normal values."""
        score = AnomalyScore(0.5)
        assert score.is_valid() is True
        
        score = AnomalyScore(0.0)
        assert score.is_valid() is True
        
        score = AnomalyScore(1.0)
        assert score.is_valid() is True

    def test_is_confident_without_interval(self):
        """Test is_confident when no confidence interval."""
        score = AnomalyScore(0.5)
        assert score.is_confident is False

    def test_is_confident_with_interval(self):
        """Test is_confident when confidence interval exists."""
        mock_ci = Mock(spec=ConfidenceInterval)
        mock_ci.contains.return_value = True
        
        score = AnomalyScore(0.5, confidence_interval=mock_ci)
        assert score.is_confident is True

    def test_confidence_width_without_interval(self):
        """Test confidence_width when no confidence interval."""
        score = AnomalyScore(0.5)
        assert score.confidence_width is None

    def test_confidence_width_with_interval(self):
        """Test confidence_width with confidence interval."""
        mock_ci = Mock(spec=ConfidenceInterval)
        mock_ci.contains.return_value = True
        mock_ci.width.return_value = 0.4
        
        score = AnomalyScore(0.5, confidence_interval=mock_ci)
        assert score.confidence_width == 0.4

    def test_confidence_bounds_without_interval(self):
        """Test confidence bounds when no interval."""
        score = AnomalyScore(0.5)
        assert score.confidence_lower is None
        assert score.confidence_upper is None

    def test_confidence_bounds_with_interval(self):
        """Test confidence bounds with interval."""
        mock_ci = Mock(spec=ConfidenceInterval)
        mock_ci.contains.return_value = True
        mock_ci.lower = 0.3
        mock_ci.upper = 0.7
        
        score = AnomalyScore(0.5, confidence_interval=mock_ci)
        assert score.confidence_lower == 0.3
        assert score.confidence_upper == 0.7


class TestAnomalyScoreMethods:
    """Test methods of AnomalyScore objects."""

    def test_exceeds_threshold(self):
        """Test exceeds_threshold method."""
        score = AnomalyScore(0.7)
        
        assert score.exceeds_threshold(0.5) is True
        assert score.exceeds_threshold(0.7) is False
        assert score.exceeds_threshold(0.8) is False

    def test_exceeds_threshold_boundary(self):
        """Test exceeds_threshold at boundary."""
        score = AnomalyScore(0.5)
        
        assert score.exceeds_threshold(0.5) is False
        assert score.exceeds_threshold(0.4999999) is True
        assert score.exceeds_threshold(0.5000001) is False

    def test_str_representation(self):
        """Test string representation."""
        score = AnomalyScore(0.7)
        assert str(score) == "0.7"
        
        score = AnomalyScore(0.0)
        assert str(score) == "0.0"
        
        score = AnomalyScore(1.0)
        assert str(score) == "1.0"


class TestAnomalyScoreComparisons:
    """Test comparison operations."""

    def test_less_than_score(self):
        """Test less than comparison with other AnomalyScore."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)
        
        assert score1 < score2
        assert not score2 < score1
        assert not score1 < score1

    def test_less_than_numeric(self):
        """Test less than comparison with numeric values."""
        score = AnomalyScore(0.5)
        
        assert score < 0.7
        assert score < 1.0
        assert not score < 0.3
        assert not score < 0.5

    def test_less_than_equal_score(self):
        """Test less than or equal comparison with AnomalyScore."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)
        score3 = AnomalyScore(0.3)
        
        assert score1 <= score2
        assert score1 <= score3
        assert not score2 <= score1

    def test_less_than_equal_numeric(self):
        """Test less than or equal comparison with numeric values."""
        score = AnomalyScore(0.5)
        
        assert score <= 0.5
        assert score <= 0.7
        assert not score <= 0.3

    def test_greater_than_score(self):
        """Test greater than comparison with AnomalyScore."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)
        
        assert score2 > score1
        assert not score1 > score2
        assert not score1 > score1

    def test_greater_than_numeric(self):
        """Test greater than comparison with numeric values."""
        score = AnomalyScore(0.5)
        
        assert score > 0.3
        assert score > 0.0
        assert not score > 0.7
        assert not score > 0.5

    def test_greater_than_equal_score(self):
        """Test greater than or equal comparison with AnomalyScore."""
        score1 = AnomalyScore(0.3)
        score2 = AnomalyScore(0.7)
        score3 = AnomalyScore(0.7)
        
        assert score2 >= score1
        assert score2 >= score3
        assert not score1 >= score2

    def test_greater_than_equal_numeric(self):
        """Test greater than or equal comparison with numeric values."""
        score = AnomalyScore(0.5)
        
        assert score >= 0.5
        assert score >= 0.3
        assert not score >= 0.7

    def test_comparison_with_invalid_types(self):
        """Test comparison with invalid types returns NotImplemented."""
        score = AnomalyScore(0.5)
        
        assert score.__lt__("0.3") is NotImplemented
        assert score.__le__([0.3]) is NotImplemented
        assert score.__gt__(None) is NotImplemented
        assert score.__ge__({}) is NotImplemented

    def test_sorting(self):
        """Test sorting of AnomalyScore objects."""
        scores = [
            AnomalyScore(0.8),
            AnomalyScore(0.2),
            AnomalyScore(0.5),
            AnomalyScore(0.9),
            AnomalyScore(0.1)
        ]
        
        sorted_scores = sorted(scores)
        expected_values = [0.1, 0.2, 0.5, 0.8, 0.9]
        
        for i, score in enumerate(sorted_scores):
            assert score.value == expected_values[i]


class TestAnomalyScoreImmutability:
    """Test immutability of AnomalyScore objects."""

    def test_dataclass_frozen(self):
        """Test that dataclass is frozen (immutable)."""
        score = AnomalyScore(0.5)
        
        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            score.value = 0.7
        
        with pytest.raises(AttributeError):
            score.confidence_interval = Mock()
        
        with pytest.raises(AttributeError):
            score.method = "new_method"

    def test_equality_based_on_content(self):
        """Test equality is based on content, not identity."""
        score1 = AnomalyScore(0.5, method="test")
        score2 = AnomalyScore(0.5, method="test")
        score3 = AnomalyScore(0.5, method="other")
        
        assert score1 == score2
        assert score1 != score3
        assert score1 is not score2  # Different objects

    def test_hashable(self):
        """Test that AnomalyScore objects are hashable."""
        score1 = AnomalyScore(0.5, method="test")
        score2 = AnomalyScore(0.5, method="test")
        score3 = AnomalyScore(0.7, method="test")
        
        # Can be used in sets
        score_set = {score1, score2, score3}
        assert len(score_set) == 2  # score1 and score2 are equal
        
        # Hash consistency
        assert hash(score1) == hash(score2)
        assert hash(score1) != hash(score3)


class TestAnomalyScoreEdgeCases:
    """Test edge cases and error conditions."""

    def test_floating_point_precision(self):
        """Test handling of floating-point precision."""
        score = AnomalyScore(0.9999999999999999)
        assert score.value == 0.9999999999999999
        assert score.is_valid()

    def test_very_small_values(self):
        """Test very small valid values."""
        score = AnomalyScore(1e-10)
        assert score.value == 1e-10
        assert score.is_valid()
        assert score.exceeds_threshold(0.0) is True

    def test_confidence_interval_edge_cases(self):
        """Test edge cases with confidence intervals."""
        # Test when confidence interval exactly matches value
        mock_ci = Mock(spec=ConfidenceInterval)
        mock_ci.contains.return_value = True
        mock_ci.lower = 0.5
        mock_ci.upper = 0.5
        mock_ci.width.return_value = 0.0
        
        score = AnomalyScore(0.5, confidence_interval=mock_ci)
        assert score.confidence_width == 0.0
        assert score.confidence_lower == 0.5
        assert score.confidence_upper == 0.5

    def test_method_with_various_types(self):
        """Test method parameter with various string types."""
        score1 = AnomalyScore(0.5, method="")
        assert score1.method == ""
        
        score2 = AnomalyScore(0.5, method="Very Long Method Name With Spaces")
        assert score2.method == "Very Long Method Name With Spaces"

    def test_comparison_edge_cases(self):
        """Test comparison edge cases."""
        score = AnomalyScore(0.5)
        
        # Test with edge numeric values
        assert score < float('inf')
        assert score > float('-inf')
        assert not score < float('nan')  # NaN comparisons are False
        assert not score > float('nan')


class TestAnomalyScoreIntegration:
    """Test integration with confidence intervals."""

    def test_with_real_confidence_interval(self):
        """Test with actual ConfidenceInterval if available."""
        # This would test with real ConfidenceInterval objects
        # For now, using mocks since we need to implement ConfidenceInterval tests first
        mock_ci = Mock(spec=ConfidenceInterval)
        mock_ci.contains.return_value = True
        mock_ci.lower = 0.4
        mock_ci.upper = 0.6
        mock_ci.width.return_value = 0.2
        
        score = AnomalyScore(0.5, confidence_interval=mock_ci, method="test_method")
        
        assert score.is_confident
        assert score.confidence_width == 0.2
        assert score.confidence_lower == 0.4
        assert score.confidence_upper == 0.6
        
        # Verify confidence interval method was called
        mock_ci.contains.assert_called_once_with(0.5)

    def test_multiple_scores_with_intervals(self):
        """Test multiple scores with different confidence intervals."""
        scores = []
        
        for i, val in enumerate([0.2, 0.5, 0.8]):
            mock_ci = Mock(spec=ConfidenceInterval)
            mock_ci.contains.return_value = True
            mock_ci.lower = val - 0.1
            mock_ci.upper = val + 0.1
            mock_ci.width.return_value = 0.2
            
            score = AnomalyScore(val, confidence_interval=mock_ci, method=f"method_{i}")
            scores.append(score)
        
        # All should be confident
        assert all(score.is_confident for score in scores)
        
        # Should be sortable
        sorted_scores = sorted(scores)
        assert sorted_scores[0].value == 0.2
        assert sorted_scores[1].value == 0.5
        assert sorted_scores[2].value == 0.8
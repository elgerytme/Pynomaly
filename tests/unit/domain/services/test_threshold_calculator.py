"""Comprehensive tests for ThresholdCalculator domain service."""

import numpy as np
import pytest

from pynomaly.domain.services.threshold_calculator import ThresholdCalculator
from pynomaly.domain.value_objects import ContaminationRate


class TestThresholdCalculatorByContamination:
    """Test threshold calculation by contamination rate."""

    def test_calculate_by_contamination_basic(self):
        """Test basic contamination-based threshold calculation."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        contamination_rate = ContaminationRate(0.3)  # 30% contamination

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        # With 10 samples and 30% contamination, expect 3 anomalies
        # Threshold should be 0.8 (3rd highest score)
        assert threshold == 0.8

    def test_calculate_by_contamination_sorted_scores(self):
        """Test that scores are properly sorted internally."""
        scores = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 1.0]
        contamination_rate = ContaminationRate(0.2)  # 20% contamination

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        # With 10 samples and 20% contamination, expect 2 anomalies
        # Threshold should be 0.9 (2nd highest score)
        assert threshold == 0.9

    def test_calculate_by_contamination_no_anomalies(self):
        """Test contamination rate resulting in no anomalies."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        contamination_rate = ContaminationRate(0.0)  # No contamination

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        # Should return max score + 0.1 (no anomalies)
        assert threshold == 0.6  # 0.5 + 0.1

    def test_calculate_by_contamination_all_anomalies(self):
        """Test contamination rate resulting in all anomalies."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        contamination_rate = ContaminationRate(1.0)  # All contamination

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        # Should return min score - 0.1 (all anomalies)
        assert threshold == 0.0  # 0.1 - 0.1

    def test_calculate_by_contamination_single_sample(self):
        """Test threshold calculation with single sample."""
        scores = [0.5]
        contamination_rate = ContaminationRate(0.1)  # 10% contamination

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        # With 1 sample and 10% contamination, expect 0 anomalies
        assert threshold == 0.6  # 0.5 + 0.1

    def test_calculate_by_contamination_empty_scores(self):
        """Test threshold calculation with empty scores."""
        scores = []
        contamination_rate = ContaminationRate(0.1)

        with pytest.raises(
            ValueError, match="Cannot calculate threshold from empty scores"
        ):
            ThresholdCalculator.calculate_by_contamination(scores, contamination_rate)

    def test_calculate_by_contamination_identical_scores(self):
        """Test threshold calculation with identical scores."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        contamination_rate = ContaminationRate(0.4)  # 40% contamination

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        # With 5 samples and 40% contamination, expect 2 anomalies
        # All scores are identical, so threshold should be 0.5
        assert threshold == 0.5

    def test_calculate_by_contamination_rounding_behavior(self):
        """Test threshold calculation with rounding behavior."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        contamination_rate = ContaminationRate(0.15)  # 15% contamination

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        # With 7 samples and 15% contamination, expect 1 anomaly (7 * 0.15 = 1.05, rounded to 1)
        # Threshold should be 0.7 (1st highest score)
        assert threshold == 0.7


class TestThresholdCalculatorByPercentile:
    """Test threshold calculation by percentile."""

    def test_calculate_by_percentile_basic(self):
        """Test basic percentile-based threshold calculation."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        threshold = ThresholdCalculator.calculate_by_percentile(scores, 90)

        # 90th percentile should be 0.9
        assert threshold == 0.9

    def test_calculate_by_percentile_median(self):
        """Test percentile calculation at median."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        threshold = ThresholdCalculator.calculate_by_percentile(scores, 50)

        # 50th percentile should be 0.55
        assert threshold == 0.55

    def test_calculate_by_percentile_extreme_values(self):
        """Test percentile calculation at extreme values."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        threshold_0 = ThresholdCalculator.calculate_by_percentile(scores, 0)
        threshold_100 = ThresholdCalculator.calculate_by_percentile(scores, 100)

        assert threshold_0 == 0.1
        assert threshold_100 == 1.0

    def test_calculate_by_percentile_empty_scores(self):
        """Test percentile calculation with empty scores."""
        scores = []

        with pytest.raises(
            ValueError, match="Cannot calculate threshold from empty scores"
        ):
            ThresholdCalculator.calculate_by_percentile(scores, 90)

    def test_calculate_by_percentile_invalid_percentile(self):
        """Test percentile calculation with invalid percentile."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]

        with pytest.raises(ValueError, match="Percentile must be in \\[0, 100\\]"):
            ThresholdCalculator.calculate_by_percentile(scores, -10)

        with pytest.raises(ValueError, match="Percentile must be in \\[0, 100\\]"):
            ThresholdCalculator.calculate_by_percentile(scores, 150)

    def test_calculate_by_percentile_single_value(self):
        """Test percentile calculation with single value."""
        scores = [0.5]

        threshold = ThresholdCalculator.calculate_by_percentile(scores, 50)

        assert threshold == 0.5

    def test_calculate_by_percentile_identical_values(self):
        """Test percentile calculation with identical values."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]

        threshold = ThresholdCalculator.calculate_by_percentile(scores, 90)

        assert threshold == 0.5

    def test_calculate_by_percentile_float_return_type(self):
        """Test that percentile calculation returns float."""
        scores = [1, 2, 3, 4, 5]  # Integer scores

        threshold = ThresholdCalculator.calculate_by_percentile(scores, 80)

        assert isinstance(threshold, float)


class TestThresholdCalculatorByIQR:
    """Test threshold calculation by IQR method."""

    def test_calculate_by_iqr_basic(self):
        """Test basic IQR-based threshold calculation."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        threshold = ThresholdCalculator.calculate_by_iqr(scores)

        # Q1 = 0.325, Q3 = 0.775, IQR = 0.45
        # Threshold = 0.775 + 1.5 * 0.45 = 1.45
        expected_threshold = 0.775 + 1.5 * 0.45
        assert abs(threshold - expected_threshold) < 0.001

    def test_calculate_by_iqr_custom_multiplier(self):
        """Test IQR calculation with custom multiplier."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        threshold = ThresholdCalculator.calculate_by_iqr(scores, multiplier=3.0)

        # Q1 = 0.325, Q3 = 0.775, IQR = 0.45
        # Threshold = 0.775 + 3.0 * 0.45 = 2.125
        expected_threshold = 0.775 + 3.0 * 0.45
        assert abs(threshold - expected_threshold) < 0.001

    def test_calculate_by_iqr_empty_scores(self):
        """Test IQR calculation with empty scores."""
        scores = []

        with pytest.raises(
            ValueError, match="Cannot calculate threshold from empty scores"
        ):
            ThresholdCalculator.calculate_by_iqr(scores)

    def test_calculate_by_iqr_single_value(self):
        """Test IQR calculation with single value."""
        scores = [0.5]

        threshold = ThresholdCalculator.calculate_by_iqr(scores)

        # Q1 = Q3 = 0.5, IQR = 0
        # Threshold = 0.5 + 1.5 * 0 = 0.5
        assert threshold == 0.5

    def test_calculate_by_iqr_identical_values(self):
        """Test IQR calculation with identical values."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]

        threshold = ThresholdCalculator.calculate_by_iqr(scores)

        # Q1 = Q3 = 0.5, IQR = 0
        # Threshold = 0.5 + 1.5 * 0 = 0.5
        assert threshold == 0.5

    def test_calculate_by_iqr_small_dataset(self):
        """Test IQR calculation with small dataset."""
        scores = [0.1, 0.5, 0.9]

        threshold = ThresholdCalculator.calculate_by_iqr(scores)

        # Q1 = 0.3, Q3 = 0.7, IQR = 0.4
        # Threshold = 0.7 + 1.5 * 0.4 = 1.3
        expected_threshold = 0.7 + 1.5 * 0.4
        assert abs(threshold - expected_threshold) < 0.001

    def test_calculate_by_iqr_float_return_type(self):
        """Test that IQR calculation returns float."""
        scores = [1, 2, 3, 4, 5]  # Integer scores

        threshold = ThresholdCalculator.calculate_by_iqr(scores)

        assert isinstance(threshold, float)


class TestThresholdCalculatorByMAD:
    """Test threshold calculation by MAD method."""

    def test_calculate_by_mad_basic(self):
        """Test basic MAD-based threshold calculation."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        threshold = ThresholdCalculator.calculate_by_mad(scores)

        # Median = 0.55, MAD = median of |scores - 0.55|
        # MAD = 0.25, MAD_std = 0.25 * 1.4826 = 0.37065
        # Threshold = 0.55 + 3.0 * 0.37065 = 1.661...
        median = 0.55
        mad = 0.25
        mad_std = mad * 1.4826
        expected_threshold = median + 3.0 * mad_std
        assert abs(threshold - expected_threshold) < 0.001

    def test_calculate_by_mad_custom_threshold_factor(self):
        """Test MAD calculation with custom threshold factor."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        threshold = ThresholdCalculator.calculate_by_mad(scores, threshold_factor=2.0)

        median = 0.55
        mad = 0.25
        mad_std = mad * 1.4826
        expected_threshold = median + 2.0 * mad_std
        assert abs(threshold - expected_threshold) < 0.001

    def test_calculate_by_mad_empty_scores(self):
        """Test MAD calculation with empty scores."""
        scores = []

        with pytest.raises(
            ValueError, match="Cannot calculate threshold from empty scores"
        ):
            ThresholdCalculator.calculate_by_mad(scores)

    def test_calculate_by_mad_single_value(self):
        """Test MAD calculation with single value."""
        scores = [0.5]

        threshold = ThresholdCalculator.calculate_by_mad(scores)

        # Median = 0.5, MAD = 0 (no deviation)
        # Threshold = 0.5 + 3.0 * 0 = 0.5
        assert threshold == 0.5

    def test_calculate_by_mad_identical_values(self):
        """Test MAD calculation with identical values."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]

        threshold = ThresholdCalculator.calculate_by_mad(scores)

        # Median = 0.5, MAD = 0 (no deviation)
        # Threshold = 0.5 + 3.0 * 0 = 0.5
        assert threshold == 0.5

    def test_calculate_by_mad_outliers(self):
        """Test MAD calculation with outliers."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 5.0]  # 5.0 is outlier

        threshold = ThresholdCalculator.calculate_by_mad(scores)

        # MAD is robust to outliers, so threshold should be reasonable
        assert 0.5 < threshold < 5.0

    def test_calculate_by_mad_float_return_type(self):
        """Test that MAD calculation returns float."""
        scores = [1, 2, 3, 4, 5]  # Integer scores

        threshold = ThresholdCalculator.calculate_by_mad(scores)

        assert isinstance(threshold, float)


class TestThresholdCalculatorDynamicThreshold:
    """Test dynamic threshold calculation."""

    def test_calculate_dynamic_threshold_knee_method(self):
        """Test dynamic threshold calculation with knee method."""
        # Create scores with clear knee point
        scores = [1.0, 0.9, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1]

        threshold, contamination = ThresholdCalculator.calculate_dynamic_threshold(
            scores, min_anomalies=2, max_contamination=0.5, method="knee"
        )

        assert isinstance(threshold, float)
        assert isinstance(contamination, float)
        assert 0.0 <= contamination <= 0.5
        assert threshold in scores  # Should be one of the score values

    def test_calculate_dynamic_threshold_gap_method(self):
        """Test dynamic threshold calculation with gap method."""
        # Create scores with clear gap
        scores = [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]  # Gap between 0.8 and 0.3

        threshold, contamination = ThresholdCalculator.calculate_dynamic_threshold(
            scores, min_anomalies=2, max_contamination=0.5, method="gap"
        )

        assert isinstance(threshold, float)
        assert isinstance(contamination, float)
        assert 0.0 <= contamination <= 0.5
        # Threshold should be between 0.8 and 0.3
        assert 0.3 < threshold < 0.8

    def test_calculate_dynamic_threshold_slope_method(self):
        """Test dynamic threshold calculation with slope method."""
        scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        threshold, contamination = ThresholdCalculator.calculate_dynamic_threshold(
            scores, min_anomalies=2, max_contamination=0.5, method="slope"
        )

        assert isinstance(threshold, float)
        assert isinstance(contamination, float)
        assert 0.0 <= contamination <= 0.5
        assert threshold in scores  # Should be one of the score values

    def test_calculate_dynamic_threshold_empty_scores(self):
        """Test dynamic threshold calculation with empty scores."""
        scores = []

        with pytest.raises(
            ValueError, match="Cannot calculate threshold from empty scores"
        ):
            ThresholdCalculator.calculate_dynamic_threshold(scores)

    def test_calculate_dynamic_threshold_invalid_method(self):
        """Test dynamic threshold calculation with invalid method."""
        scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

        with pytest.raises(ValueError, match="Unknown dynamic threshold method"):
            ThresholdCalculator.calculate_dynamic_threshold(scores, method="invalid")

    def test_calculate_dynamic_threshold_min_anomalies_constraint(self):
        """Test that minimum anomalies constraint is respected."""
        scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

        threshold, contamination = ThresholdCalculator.calculate_dynamic_threshold(
            scores, min_anomalies=3, max_contamination=0.5, method="knee"
        )

        # Should detect at least 3 anomalies
        n_anomalies = contamination * len(scores)
        assert n_anomalies >= 3

    def test_calculate_dynamic_threshold_max_contamination_constraint(self):
        """Test that maximum contamination constraint is respected."""
        scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        threshold, contamination = ThresholdCalculator.calculate_dynamic_threshold(
            scores, min_anomalies=1, max_contamination=0.3, method="knee"
        )

        # Should not exceed 30% contamination
        assert contamination <= 0.3

    def test_calculate_dynamic_threshold_single_score(self):
        """Test dynamic threshold calculation with single score."""
        scores = [0.5]

        threshold, contamination = ThresholdCalculator.calculate_dynamic_threshold(
            scores, min_anomalies=1, max_contamination=0.5, method="knee"
        )

        assert threshold == 0.5
        assert contamination == 1.0  # 1 anomaly out of 1 sample

    def test_calculate_dynamic_threshold_identical_scores(self):
        """Test dynamic threshold calculation with identical scores."""
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]

        threshold, contamination = ThresholdCalculator.calculate_dynamic_threshold(
            scores, min_anomalies=2, max_contamination=0.5, method="knee"
        )

        assert threshold == 0.5
        assert contamination >= 0.4  # At least 2 out of 5


class TestThresholdCalculatorPrivateMethods:
    """Test private helper methods."""

    def test_find_knee_point_basic(self):
        """Test knee point finding."""
        sorted_scores = [1.0, 0.9, 0.8, 0.4, 0.3, 0.2, 0.1]

        threshold, n_anomalies = ThresholdCalculator._find_knee_point(
            sorted_scores, min_anomalies=2, max_anomalies=5
        )

        assert isinstance(threshold, float)
        assert isinstance(n_anomalies, int)
        assert 2 <= n_anomalies <= 5
        assert threshold in sorted_scores

    def test_find_knee_point_edge_case(self):
        """Test knee point finding with edge case."""
        sorted_scores = [1.0, 0.9, 0.8, 0.7, 0.6]

        threshold, n_anomalies = ThresholdCalculator._find_knee_point(
            sorted_scores, min_anomalies=3, max_anomalies=3
        )

        # When min == max, should return min
        assert n_anomalies == 3
        assert threshold == 0.8  # 3rd score (index 2)

    def test_find_largest_gap_basic(self):
        """Test largest gap finding."""
        sorted_scores = [1.0, 0.9, 0.8, 0.3, 0.2, 0.1]  # Gap between 0.8 and 0.3

        threshold, n_anomalies = ThresholdCalculator._find_largest_gap(
            sorted_scores, min_anomalies=2, max_anomalies=4
        )

        assert isinstance(threshold, float)
        assert isinstance(n_anomalies, int)
        assert 2 <= n_anomalies <= 4
        # Threshold should be between 0.8 and 0.3
        assert 0.3 < threshold < 0.8

    def test_find_largest_gap_no_gaps(self):
        """Test largest gap finding with no clear gaps."""
        sorted_scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

        threshold, n_anomalies = ThresholdCalculator._find_largest_gap(
            sorted_scores, min_anomalies=2, max_anomalies=4
        )

        assert isinstance(threshold, float)
        assert isinstance(n_anomalies, int)
        assert 2 <= n_anomalies <= 4

    def test_find_slope_change_basic(self):
        """Test slope change finding."""
        # Create scores with slope change
        sorted_scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

        threshold, n_anomalies = ThresholdCalculator._find_slope_change(
            sorted_scores, min_anomalies=2, max_anomalies=6
        )

        assert isinstance(threshold, float)
        assert isinstance(n_anomalies, int)
        assert 2 <= n_anomalies <= 6
        assert threshold in sorted_scores

    def test_find_slope_change_edge_case(self):
        """Test slope change finding with edge case."""
        sorted_scores = [1.0, 0.9, 0.8]

        threshold, n_anomalies = ThresholdCalculator._find_slope_change(
            sorted_scores, min_anomalies=1, max_anomalies=2
        )

        # With too few points, should return minimum
        assert n_anomalies == 1
        assert threshold == 1.0  # First score (index 0)

    def test_find_slope_change_insufficient_data(self):
        """Test slope change finding with insufficient data."""
        sorted_scores = [1.0, 0.9]

        threshold, n_anomalies = ThresholdCalculator._find_slope_change(
            sorted_scores, min_anomalies=1, max_anomalies=1
        )

        assert n_anomalies == 1
        assert threshold == 1.0


class TestThresholdCalculatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_scores(self):
        """Test with very small score values."""
        scores = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        contamination_rate = ContaminationRate(0.2)

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        assert threshold > 0
        assert threshold == 5e-10  # Top score

    def test_very_large_scores(self):
        """Test with very large score values."""
        scores = [1e10, 2e10, 3e10, 4e10, 5e10]
        contamination_rate = ContaminationRate(0.2)

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        assert threshold == 5e10  # Top score

    def test_negative_scores(self):
        """Test with negative score values."""
        scores = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        contamination_rate = ContaminationRate(0.3)

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        # Should work with negative scores
        assert threshold == 0.2  # 3rd highest score

    def test_mixed_positive_negative_scores(self):
        """Test with mixed positive and negative scores."""
        scores = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9]

        threshold = ThresholdCalculator.calculate_by_percentile(scores, 75)

        # Should handle mixed signs correctly
        assert threshold > 0

    def test_duplicate_scores_contamination(self):
        """Test contamination calculation with duplicate scores."""
        scores = [0.1, 0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 1.0]
        contamination_rate = ContaminationRate(0.3)

        threshold = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )

        # Should handle duplicates correctly
        assert threshold in scores

    def test_all_methods_with_same_data(self):
        """Test that all methods work with the same dataset."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        contamination_rate = ContaminationRate(0.2)

        # All methods should work without errors
        threshold_contamination = ThresholdCalculator.calculate_by_contamination(
            scores, contamination_rate
        )
        threshold_percentile = ThresholdCalculator.calculate_by_percentile(scores, 80)
        threshold_iqr = ThresholdCalculator.calculate_by_iqr(scores)
        threshold_mad = ThresholdCalculator.calculate_by_mad(scores)
        threshold_dynamic, _ = ThresholdCalculator.calculate_dynamic_threshold(
            scores, method="knee"
        )

        # All should return valid thresholds
        assert all(
            isinstance(t, float)
            for t in [
                threshold_contamination,
                threshold_percentile,
                threshold_iqr,
                threshold_mad,
                threshold_dynamic,
            ]
        )

    def test_performance_with_large_dataset(self):
        """Test performance with large dataset."""
        # Create large dataset
        np.random.seed(42)
        large_scores = np.random.random(10000).tolist()
        contamination_rate = ContaminationRate(0.1)

        # Should handle large datasets efficiently
        threshold = ThresholdCalculator.calculate_by_contamination(
            large_scores, contamination_rate
        )

        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_numeric_stability(self):
        """Test numeric stability with extreme values."""
        scores = [1e-100, 1e-50, 1e-10, 1e10, 1e50, 1e100]

        # All methods should handle extreme values
        threshold_percentile = ThresholdCalculator.calculate_by_percentile(scores, 50)
        threshold_iqr = ThresholdCalculator.calculate_by_iqr(scores)
        threshold_mad = ThresholdCalculator.calculate_by_mad(scores)

        # Should return finite values
        assert all(
            np.isfinite(t) for t in [threshold_percentile, threshold_iqr, threshold_mad]
        )

    def test_type_consistency(self):
        """Test that all methods return consistent types."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        contamination_rate = ContaminationRate(0.2)

        # All methods should return float
        results = [
            ThresholdCalculator.calculate_by_contamination(scores, contamination_rate),
            ThresholdCalculator.calculate_by_percentile(scores, 80),
            ThresholdCalculator.calculate_by_iqr(scores),
            ThresholdCalculator.calculate_by_mad(scores),
        ]

        dynamic_threshold, dynamic_contamination = (
            ThresholdCalculator.calculate_dynamic_threshold(scores, method="knee")
        )

        assert all(isinstance(r, float) for r in results)
        assert isinstance(dynamic_threshold, float)
        assert isinstance(dynamic_contamination, float)

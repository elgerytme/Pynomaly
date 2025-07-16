"""Mutation tests for domain layer business logic."""

import pytest

from monorepo.domain.entities import Anomaly, Dataset, DetectionResult
from monorepo.domain.exceptions import InvalidValueError
from monorepo.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
    ThresholdConfig,
)


class TestContaminationRateMutations:
    """Test contamination rate business logic against mutations."""

    def test_contamination_rate_boundary_validation(self):
        """Test that contamination rate properly validates boundaries."""
        # Valid values should work
        valid_rates = [0.0, 0.1, 0.25, 0.5]
        for rate in valid_rates:
            contamination = ContaminationRate(rate)
            assert contamination.value == rate
            assert 0.0 <= contamination.value <= 0.5

        # Invalid values should raise exceptions
        invalid_rates = [-0.1, 0.6, 1.0, 1.5, -1.0]
        for rate in invalid_rates:
            with pytest.raises(InvalidValueError):
                ContaminationRate(rate)

    def test_contamination_rate_percentage_calculation(self):
        """Test contamination rate percentage calculation logic."""
        test_cases = [(0.0, 0.0), (0.1, 10.0), (0.25, 25.0), (0.5, 50.0)]

        for rate, expected_percentage in test_cases:
            contamination = ContaminationRate(rate)
            # This test ensures the multiplication by 100 is correct
            assert contamination.as_percentage() == expected_percentage

    def test_contamination_rate_auto_creation(self):
        """Test auto-creation of contamination rate."""
        auto_rate = ContaminationRate.auto()
        # Verify default is 0.1 (10%)
        assert auto_rate.value == 0.1
        assert auto_rate.as_percentage() == 10.0


class TestAnomalyScoreMutations:
    """Test anomaly score business logic against mutations."""

    def test_anomaly_score_normalization(self):
        """Test that scores are properly normalized between 0 and 1."""
        valid_scores = [0.0, 0.5, 1.0, 0.333, 0.999]
        for score in valid_scores:
            anomaly_score = AnomalyScore(score)
            assert 0.0 <= anomaly_score.value <= 1.0
            assert anomaly_score.value == score

    def test_anomaly_score_invalid_values(self):
        """Test that invalid scores are rejected."""
        invalid_scores = [-0.1, 1.1, -1.0, 2.0, float("inf"), float("nan")]
        for score in invalid_scores:
            with pytest.raises(InvalidValueError):
                AnomalyScore(score)

    def test_anomaly_score_comparison_logic(self):
        """Test anomaly score comparison operations."""
        low_score = AnomalyScore(0.2)
        high_score = AnomalyScore(0.8)
        equal_score = AnomalyScore(0.2)

        # Test comparison operators
        assert low_score < high_score
        assert high_score > low_score
        assert low_score == equal_score
        assert low_score <= high_score
        assert high_score >= low_score
        assert low_score != high_score


class TestThresholdConfigMutations:
    """Test threshold configuration business logic."""

    def test_threshold_config_method_validation(self):
        """Test that only valid threshold methods are accepted."""
        valid_methods = ["contamination", "percentile", "fixed"]

        for method in valid_methods:
            config = ThresholdConfig(method=method)
            assert config.method == method

        # Invalid methods should raise exceptions
        invalid_methods = ["invalid", "auto", "dynamic", ""]
        for method in invalid_methods:
            with pytest.raises(InvalidValueError):
                ThresholdConfig(method=method)

    def test_threshold_config_percentile_validation(self):
        """Test percentile validation logic."""
        # Valid percentiles
        valid_percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
        for percentile in valid_percentiles:
            config = ThresholdConfig(method="percentile", percentile=percentile)
            assert config.percentile == percentile

        # Invalid percentiles
        invalid_percentiles = [-1, 101, -10, 150]
        for percentile in invalid_percentiles:
            with pytest.raises(InvalidValueError):
                ThresholdConfig(method="percentile", percentile=percentile)

    def test_threshold_config_fixed_value_validation(self):
        """Test fixed threshold value validation."""
        # Valid fixed values
        valid_values = [0.0, 0.5, 1.0, 0.85]
        for value in valid_values:
            config = ThresholdConfig(method="fixed", value=value)
            assert config.value == value

        # Invalid fixed values
        invalid_values = [-0.1, 1.1, -1.0, 2.0]
        for value in invalid_values:
            with pytest.raises(InvalidValueError):
                ThresholdConfig(method="fixed", value=value)


class TestConfidenceIntervalMutations:
    """Test confidence interval calculation logic."""

    def test_confidence_interval_calculation(self):
        """Test confidence interval boundary calculations."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        confidence_levels = [0.90, 0.95, 0.99]

        for confidence_level in confidence_levels:
            ci = ConfidenceInterval.from_scores(scores, confidence_level)

            # Verify that lower bound <= upper bound
            assert ci.lower_bound <= ci.upper_bound

            # Verify confidence level is stored correctly
            assert ci.confidence_level == confidence_level

            # Verify bounds are within valid score range
            assert 0.0 <= ci.lower_bound <= 1.0
            assert 0.0 <= ci.upper_bound <= 1.0

    def test_confidence_interval_edge_cases(self):
        """Test confidence interval edge cases."""
        # Test with single value
        single_score = [0.5]
        ci = ConfidenceInterval.from_scores(single_score, 0.95)
        assert ci.lower_bound == ci.upper_bound == 0.5

        # Test with identical values
        identical_scores = [0.7] * 10
        ci = ConfidenceInterval.from_scores(identical_scores, 0.95)
        assert ci.lower_bound == ci.upper_bound == 0.7

    def test_confidence_interval_contains_logic(self):
        """Test confidence interval containment logic."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ci = ConfidenceInterval.from_scores(scores, 0.95)

        # Test values within interval
        test_values = [
            ci.lower_bound,
            ci.upper_bound,
            (ci.lower_bound + ci.upper_bound) / 2,
        ]
        for value in test_values:
            assert ci.contains(value)

        # Test values outside interval (if bounds are not extreme)
        if ci.lower_bound > 0.0:
            assert not ci.contains(ci.lower_bound - 0.01)
        if ci.upper_bound < 1.0:
            assert not ci.contains(ci.upper_bound + 0.01)


class TestDatasetMutations:
    """Test dataset business logic mutations."""

    def test_dataset_feature_extraction(self):
        """Test dataset feature extraction logic."""
        import pandas as pd

        # Create test dataset with mixed types
        data = pd.DataFrame(
            {
                "numeric_1": [1.0, 2.0, 3.0],
                "numeric_2": [10, 20, 30],
                "categorical": ["A", "B", "C"],
                "boolean": [True, False, True],
                "target": [0, 1, 0],
            }
        )

        dataset = Dataset(name="test_dataset", data=data, target_column="target")

        # Test numeric feature extraction
        numeric_features = dataset.get_numeric_features()
        expected_numeric = ["numeric_1", "numeric_2"]
        assert set(numeric_features) == set(expected_numeric)

        # Verify features exclude target column
        assert "target" not in numeric_features

        # Test feature data extraction
        feature_data = dataset.features
        assert list(feature_data.columns) == expected_numeric
        assert len(feature_data) == 3

    def test_dataset_sample_count_logic(self):
        """Test dataset sample counting logic."""
        import pandas as pd

        # Test with different sizes
        sizes = [1, 10, 100, 1000]
        for size in sizes:
            data = pd.DataFrame({"feature": range(size), "target": [0] * size})

            dataset = Dataset(
                name=f"test_dataset_{size}", data=data, target_column="target"
            )

            # Verify sample count is correct
            assert dataset.n_samples == size
            assert len(dataset.data) == size


class TestAnomalyMutations:
    """Test anomaly entity business logic."""

    def test_anomaly_score_assignment(self):
        """Test anomaly score assignment logic."""
        test_scores = [0.0, 0.5, 0.9, 1.0]

        for score_value in test_scores:
            score = AnomalyScore(score_value)
            anomaly = Anomaly(
                index=1,
                score=score,
                timestamp=None,
                feature_names=["feature_1", "feature_2"],
            )

            # Verify score is correctly assigned
            assert anomaly.score.value == score_value
            assert anomaly.score == score

    def test_anomaly_severity_classification(self):
        """Test anomaly severity classification logic."""
        # Test severity boundaries
        test_cases = [
            (0.0, "low"),
            (0.3, "low"),
            (0.6, "medium"),
            (0.8, "high"),
            (1.0, "high"),
        ]

        for score_value, expected_severity in test_cases:
            score = AnomalyScore(score_value)
            anomaly = Anomaly(index=1, score=score, timestamp=None, feature_names=[])

            # Verify severity classification logic
            severity = anomaly.get_severity()
            assert severity == expected_severity


class TestDetectionResultMutations:
    """Test detection result business logic."""

    def test_anomaly_rate_calculation(self):
        """Test anomaly rate calculation logic."""
        # Create test anomalies
        anomalies = [
            Anomaly(index=i, score=AnomalyScore(0.8), timestamp=None, feature_names=[])
            for i in range(5)
        ]

        # Test different total sample counts
        total_samples = [10, 20, 50, 100]

        for n_samples in total_samples:
            result = DetectionResult(
                id="test_id",
                detector_id="detector_id",
                dataset_id="dataset_id",
                anomalies=anomalies,
                n_anomalies=len(anomalies),
                anomaly_rate=len(anomalies) / n_samples,
                threshold=0.7,
                execution_time=1.0,
            )

            # Verify rate calculation
            expected_rate = len(anomalies) / n_samples
            assert result.anomaly_rate == expected_rate
            assert 0.0 <= result.anomaly_rate <= 1.0

    def test_anomaly_count_consistency(self):
        """Test consistency between anomaly list and count."""
        anomaly_counts = [0, 1, 5, 10, 50]

        for count in anomaly_counts:
            anomalies = [
                Anomaly(
                    index=i, score=AnomalyScore(0.8), timestamp=None, feature_names=[]
                )
                for i in range(count)
            ]

            result = DetectionResult(
                id="test_id",
                detector_id="detector_id",
                dataset_id="dataset_id",
                anomalies=anomalies,
                n_anomalies=count,
                anomaly_rate=count / 100,
                threshold=0.7,
                execution_time=1.0,
            )

            # Verify consistency
            assert len(result.anomalies) == result.n_anomalies
            assert len(result.anomalies) == count

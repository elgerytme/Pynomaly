"""Critical path tests designed to catch mutations in core business logic."""

from __future__ import annotations

import numpy as np
import pytest

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import InvalidValueError
from pynomaly.domain.services import AnomalyScorer, ThresholdCalculator
from pynomaly.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
)


class TestCriticalDomainLogic:
    """Tests targeting critical domain logic mutations."""

    def test_contamination_rate_boundary_validation(self):
        """Test contamination rate validation catches boundary mutations."""
        # Test exact boundaries that mutants might break
        ContaminationRate(0.0)  # Should work
        ContaminationRate(0.5)  # Should work

        # These should fail - mutations might change comparison operators
        with pytest.raises(InvalidValueError):
            ContaminationRate(-0.0001)  # Just below minimum

        with pytest.raises(InvalidValueError):
            ContaminationRate(0.5001)  # Just above maximum

        with pytest.raises(InvalidValueError):
            ContaminationRate(-1.0)  # Far below minimum

        with pytest.raises(InvalidValueError):
            ContaminationRate(1.0)  # At maximum + 0.5

    def test_anomaly_score_boundary_validation(self):
        """Test anomaly score validation catches boundary mutations."""
        # Test boundaries
        AnomalyScore(0.0)  # Should work
        AnomalyScore(1.0)  # Should work

        # Test just outside boundaries
        with pytest.raises(InvalidValueError):
            AnomalyScore(-0.0001)

        with pytest.raises(InvalidValueError):
            AnomalyScore(1.0001)

    def test_confidence_interval_ordering_mutation(self):
        """Test confidence interval ordering logic."""
        # Valid interval
        ci = ConfidenceInterval(lower=0.2, upper=0.8, confidence_level=0.95)
        assert ci.width() == 0.6

        # Test that mutation breaking lower <= upper check is caught
        with pytest.raises(InvalidValueError):
            ConfidenceInterval(lower=0.8, upper=0.2, confidence_level=0.95)

        # Test edge case where lower == upper
        ci_equal = ConfidenceInterval(lower=0.5, upper=0.5, confidence_level=0.95)
        assert ci_equal.width() == 0.0

    def test_percentage_calculation_mutation(self):
        """Test percentage calculation against mutations."""
        rate = ContaminationRate(0.25)

        # Mutations might change multiplication operator or constant
        assert rate.as_percentage() == 25.0

        # Edge cases
        zero_rate = ContaminationRate(0.0)
        assert zero_rate.as_percentage() == 0.0

        max_rate = ContaminationRate(0.5)
        assert max_rate.as_percentage() == 50.0

    def test_threshold_calculation_logic_mutations(self):
        """Test threshold calculation against critical mutations."""
        calculator = ThresholdCalculator()

        # Test with known data
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # 10% contamination should give threshold around 90th percentile
        threshold = calculator.calculate_threshold(scores, contamination=0.1)

        # Mutations might break the percentile calculation
        expected_threshold = np.percentile(scores, 90)
        assert abs(threshold - expected_threshold) < 1e-10

        # Test edge cases that mutations might break
        all_same_scores = np.array([0.5] * 10)
        threshold_same = calculator.calculate_threshold(
            all_same_scores, contamination=0.1
        )
        assert threshold_same == 0.5

        # Test with 0% contamination (should be highest score)
        threshold_zero = calculator.calculate_threshold(scores, contamination=0.0)
        assert threshold_zero >= max(scores) - 1e-10

    def test_anomaly_detection_scoring_mutations(self):
        """Test anomaly scoring logic against mutations."""
        scorer = AnomalyScorer()

        # Create test data with clear pattern
        normal_data = np.random.RandomState(42).normal(0, 1, (50, 2))
        outlier_data = np.random.RandomState(42).normal(10, 1, (5, 2))  # Far outliers
        data = np.vstack([normal_data, outlier_data])

        scores = scorer.compute_scores(data, contamination=0.1)

        # Critical assertions that mutations should break
        assert len(scores) == len(data), "Score count mutation"
        assert np.all(scores >= 0), "Score range lower bound mutation"
        assert np.all(scores <= 1), "Score range upper bound mutation"
        assert np.all(np.isfinite(scores)), "Score finiteness mutation"

        # Outliers should generally have higher scores
        outlier_scores = scores[-5:]  # Last 5 are outliers
        normal_scores = scores[:-5]  # First 50 are normal

        # This could catch mutations in scoring logic
        assert np.mean(outlier_scores) > np.mean(normal_scores)

    def test_dataset_validation_mutations(self):
        """Test dataset validation logic against mutations."""
        # Valid dataset
        features = np.random.random((10, 3))
        targets = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])

        dataset = Dataset(name="test", features=features, targets=targets)

        # Test that mutations in shape validation are caught
        assert dataset.features.shape == (10, 3)
        assert len(dataset.targets) == 10

        # Test invalid shapes that mutations might allow
        with pytest.raises((ValueError, AssertionError)):
            Dataset(
                name="test", features=features, targets=np.array([0, 1])
            )  # Wrong length

    def test_detector_contamination_usage_mutations(self):
        """Test detector contamination usage against mutations."""
        contamination = ContaminationRate(0.15)
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination=contamination,
        )

        # Test that contamination is used correctly
        assert detector.contamination.value == 0.15
        assert detector.contamination.as_percentage() == 15.0

        # Test boundary contamination values
        min_detector = Detector(
            name="min_detector", algorithm="test", contamination=ContaminationRate(0.0)
        )
        assert min_detector.contamination.value == 0.0

        max_detector = Detector(
            name="max_detector", algorithm="test", contamination=ContaminationRate(0.5)
        )
        assert max_detector.contamination.value == 0.5


class TestCriticalBusinessRules:
    """Test business rules that mutations commonly break."""

    def test_anomaly_count_consistency(self):
        """Test anomaly count consistency rules."""
        # Create detection result
        features = np.random.random((100, 3))
        scores = np.random.random(100)

        # Create some anomalies based on high scores
        high_score_indices = np.where(scores > 0.8)[0]
        anomalies = [
            Anomaly(score=AnomalyScore(scores[i]), index=int(i))
            for i in high_score_indices
        ]

        detector = Detector(
            name="test", algorithm="test", contamination=ContaminationRate(0.1)
        )

        dataset = Dataset(name="test", features=features)

        result = DetectionResult(
            detector=detector, dataset=dataset, anomalies=anomalies, scores=scores
        )

        # Business rule: anomaly indices should be valid
        for anomaly in result.anomalies:
            assert 0 <= anomaly.index < len(features)

        # Business rule: anomaly scores should match score array
        for anomaly in result.anomalies:
            assert abs(anomaly.score.value - scores[anomaly.index]) < 1e-10

        # Business rule: no duplicate anomaly indices
        indices = [a.index for a in result.anomalies]
        assert len(indices) == len(set(indices))

    def test_score_threshold_relationship(self):
        """Test score-threshold relationship that mutations break."""
        calculator = ThresholdCalculator()
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Test different contamination rates
        for contamination in [0.2, 0.4, 0.6]:
            threshold = calculator.calculate_threshold(scores, contamination)

            # Count scores above threshold
            n_above = np.sum(scores > threshold)
            expected_n = int(len(scores) * contamination)

            # Allow tolerance for discrete effects
            tolerance = max(1, len(scores) * 0.1)
            assert abs(n_above - expected_n) <= tolerance

    def test_ensemble_score_aggregation_mutations(self):
        """Test ensemble aggregation logic against mutations."""
        from pynomaly.domain.services import EnsembleAggregator

        aggregator = EnsembleAggregator()

        # Test data
        scores1 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        scores2 = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        scores3 = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

        all_scores = [scores1, scores2, scores3]

        # Test mean aggregation
        mean_scores = aggregator.aggregate(all_scores, method="mean")
        expected_mean = (scores1 + scores2 + scores3) / 3

        assert np.allclose(mean_scores, expected_mean), "Mean aggregation mutation"

        # Test max aggregation
        max_scores = aggregator.aggregate(all_scores, method="max")
        expected_max = np.maximum(np.maximum(scores1, scores2), scores3)

        assert np.allclose(max_scores, expected_max), "Max aggregation mutation"

        # Test that aggregated scores maintain bounds
        assert np.all(mean_scores >= 0), "Aggregated score lower bound"
        assert np.all(mean_scores <= 1), "Aggregated score upper bound"
        assert np.all(max_scores >= 0), "Max score lower bound"
        assert np.all(max_scores <= 1), "Max score upper bound"


class TestMutationSensitiveEdgeCases:
    """Test edge cases that are particularly sensitive to mutations."""

    def test_zero_contamination_edge_case(self):
        """Test zero contamination edge case."""
        calculator = ThresholdCalculator()
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Zero contamination should give very high threshold
        threshold = calculator.calculate_threshold(scores, contamination=0.0)

        # Should be at or above maximum score
        assert threshold >= np.max(scores) - 1e-10

        # No scores should be above threshold (or at most due to ties)
        n_above = np.sum(scores > threshold)
        assert n_above == 0

    def test_maximum_contamination_edge_case(self):
        """Test maximum contamination edge case."""
        calculator = ThresholdCalculator()
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        # Maximum contamination (0.5) should give median threshold
        threshold = calculator.calculate_threshold(scores, contamination=0.5)

        # Should be around median
        expected_threshold = np.percentile(scores, 50)
        assert abs(threshold - expected_threshold) < 1e-10

    def test_single_value_arrays(self):
        """Test single value arrays that mutations often break."""
        calculator = ThresholdCalculator()

        # All same values
        same_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        threshold = calculator.calculate_threshold(same_scores, contamination=0.2)

        # Threshold should be the single value
        assert threshold == 0.5

        # Single element array
        single_score = np.array([0.7])
        threshold_single = calculator.calculate_threshold(
            single_score, contamination=0.0
        )
        assert threshold_single == 0.7

    def test_floating_point_precision_mutations(self):
        """Test floating point precision that mutations can break."""
        # Test very close values
        rate1 = ContaminationRate(0.1)
        rate2 = ContaminationRate(0.1 + 1e-15)  # Within floating point precision

        # Should be considered equal
        assert abs(rate1.value - rate2.value) < 1e-10

        # Test percentage calculations with precision
        assert abs(rate1.as_percentage() - 10.0) < 1e-10

    def test_array_indexing_mutations(self):
        """Test array indexing that off-by-one mutations break."""
        # Create anomaly with specific index
        anomaly = Anomaly(score=AnomalyScore(0.8), index=5)

        # Test that index is exactly what we set
        assert anomaly.index == 5

        # Test with dataset bounds
        features = np.random.random((10, 2))
        Dataset(name="test", features=features)

        # Valid indices
        for i in range(len(features)):
            anomaly = Anomaly(score=AnomalyScore(0.5), index=i)
            assert 0 <= anomaly.index < len(features)

        # Test boundary cases
        first_anomaly = Anomaly(score=AnomalyScore(0.5), index=0)
        assert first_anomaly.index == 0

        last_anomaly = Anomaly(score=AnomalyScore(0.5), index=len(features) - 1)
        assert last_anomaly.index == len(features) - 1

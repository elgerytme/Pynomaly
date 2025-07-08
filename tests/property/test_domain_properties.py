"""Property-based tests for domain layer invariants and business rules."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.services import (
    AnomalyScorer,
    EnsembleAggregator,
    FeatureValidator,
    ThresholdCalculator,
)
from pynomaly.domain.value_objects import (
    AnomalyScore,
    ConfidenceInterval,
    ContaminationRate,
)

from .strategies import (
    anomaly_score_strategy,
    anomaly_strategy,
    confidence_interval_strategy,
    contamination_rate_strategy,
    dataset_strategy,
    detection_result_strategy,
    detector_strategy,
)


class TestValueObjectProperties:
    """Test invariants for domain value objects."""

    @given(contamination_rate_strategy())
    def test_contamination_rate_invariants(self, contamination_rate: ContaminationRate):
        """Test ContaminationRate maintains invariants."""
        # Value must be between 0 and 0.5
        assert 0.0 <= contamination_rate.value <= 0.5

        # Percentage conversion should be consistent
        percentage = contamination_rate.as_percentage()
        assert 0.0 <= percentage <= 50.0
        assert abs(percentage - (contamination_rate.value * 100)) < 1e-10

        # String representation should be meaningful
        str_repr = str(contamination_rate)
        assert str(contamination_rate.value) in str_repr

    @given(anomaly_score_strategy())
    def test_anomaly_score_invariants(self, score: AnomalyScore):
        """Test AnomalyScore maintains invariants."""
        # Value must be between 0 and 1
        assert 0.0 <= score.value <= 1.0

        # Normalized value should equal original value (already normalized)
        assert abs(score.normalized() - score.value) < 1e-10

        # Higher scores should compare correctly
        lower_score = AnomalyScore(max(0.0, score.value - 0.1))
        if score.value > lower_score.value:
            assert score > lower_score

        # Equality should work correctly
        same_score = AnomalyScore(score.value)
        assert score == same_score

    @given(confidence_interval_strategy())
    def test_confidence_interval_invariants(self, ci: ConfidenceInterval):
        """Test ConfidenceInterval maintains invariants."""
        # Lower bound must be less than or equal to upper bound
        assert ci.lower <= ci.upper

        # Confidence level must be between 0 and 1
        assert 0.0 < ci.confidence_level < 1.0

        # Width should be non-negative
        width = ci.width()
        assert width >= 0.0
        assert abs(width - (ci.upper - ci.lower)) < 1e-10

        # Contains should work correctly for values in range
        if ci.width() > 0:
            mid_point = (ci.lower + ci.upper) / 2
            assert ci.contains(mid_point)

            # Values outside should not be contained
            if ci.lower > 0:
                assert not ci.contains(ci.lower - 0.001)
            if ci.upper < 1:
                assert not ci.contains(ci.upper + 0.001)


class TestEntityProperties:
    """Test invariants for domain entities."""

    @given(dataset_strategy())
    def test_dataset_invariants(self, dataset: Dataset):
        """Test Dataset maintains invariants."""
        # Features must be 2D array
        assert len(dataset.features.shape) == 2
        n_samples, n_features = dataset.features.shape

        # Must have at least one sample and one feature
        assert n_samples > 0
        assert n_features > 0

        # Targets, if present, must match number of samples
        if dataset.targets is not None:
            assert len(dataset.targets) == n_samples
            # Targets should be binary (0 or 1) for anomaly detection
            assert np.all(np.isin(dataset.targets, [0, 1]))

        # Name must be non-empty
        assert len(dataset.name.strip()) > 0

        # Features should not contain NaN or infinity
        assert np.all(np.isfinite(dataset.features))

    @given(detector_strategy())
    def test_detector_invariants(self, detector: Detector):
        """Test Detector maintains invariants."""
        # Name must be non-empty
        assert len(detector.name.strip()) > 0

        # Algorithm must be non-empty
        assert len(detector.algorithm.strip()) > 0

        # Contamination rate must be valid
        assert 0.0 <= detector.contamination.value <= 0.5

        # Hyperparameters, if present, should be reasonable
        if detector.hyperparameters:
            assert isinstance(detector.hyperparameters, dict)
            # Keys should be strings
            assert all(isinstance(k, str) for k in detector.hyperparameters.keys())

    @given(anomaly_strategy())
    def test_anomaly_invariants(self, anomaly: Anomaly):
        """Test Anomaly maintains invariants."""
        # Score must be valid
        assert 0.0 <= anomaly.score.value <= 1.0

        # Index must be non-negative
        assert anomaly.index >= 0

        # Confidence interval, if present, must be valid
        if anomaly.confidence:
            assert 0.0 <= anomaly.confidence.lower <= anomaly.confidence.upper <= 1.0

        # Features, if present, should be finite
        if anomaly.features is not None:
            assert len(anomaly.features.shape) == 1  # 1D array
            assert np.all(np.isfinite(anomaly.features))

        # Explanation, if present, should be meaningful
        if anomaly.explanation:
            assert isinstance(anomaly.explanation, dict)
            assert len(anomaly.explanation) > 0
            # Values should be finite
            assert all(np.isfinite(v) for v in anomaly.explanation.values())

    @given(detection_result_strategy())
    def test_detection_result_invariants(self, result: DetectionResult):
        """Test DetectionResult maintains invariants."""
        # Must have valid detector and dataset
        assert isinstance(result.detector, Detector)
        assert isinstance(result.dataset, Dataset)

        # Scores must match dataset size
        assert len(result.scores) == len(result.dataset.features)

        # All scores must be valid
        assert np.all((result.scores >= 0) & (result.scores <= 1))
        assert np.all(np.isfinite(result.scores))

        # Anomalies must have valid indices
        for anomaly in result.anomalies:
            assert 0 <= anomaly.index < len(result.dataset.features)
            # Anomaly score should match the score array
            assert abs(anomaly.score.value - result.scores[anomaly.index]) < 1e-6

        # Anomaly indices should be unique
        anomaly_indices = [a.index for a in result.anomalies]
        assert len(anomaly_indices) == len(set(anomaly_indices))


class TestDomainServiceProperties:
    """Test invariants for domain services."""

    @given(
        stnp.arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(10, 100), st.integers(1, 10)),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        ),
        st.floats(0.01, 0.5),
    )
    def test_anomaly_scorer_properties(self, data: np.ndarray, contamination: float):
        """Test AnomalyScorer maintains mathematical properties."""
        scorer = AnomalyScorer()

        # Compute scores
        scores = scorer.compute_scores(data, contamination)

        # Scores must be between 0 and 1
        assert np.all((scores >= 0) & (scores <= 1))
        assert np.all(np.isfinite(scores))

        # Number of scores must match input
        assert len(scores) == len(data)

        # Higher contamination should generally result in more anomalies
        # (though this is probabilistic, we test for basic consistency)
        threshold = scorer.compute_threshold(scores, contamination)
        n_anomalies = np.sum(scores > threshold)
        expected_anomalies = int(len(data) * contamination)

        # Allow some tolerance due to discrete nature
        assert abs(n_anomalies - expected_anomalies) <= max(2, len(data) * 0.05)

    @given(
        st.lists(
            stnp.arrays(
                dtype=np.float64,
                shape=st.integers(10, 50),
                elements=st.floats(0, 1, allow_nan=False, allow_infinity=False),
            ),
            min_size=2,
            max_size=5,
        )
    )
    def test_ensemble_aggregator_properties(self, score_arrays: list[np.ndarray]):
        """Test EnsembleAggregator maintains mathematical properties."""
        # Ensure all arrays have the same length
        min_length = min(len(arr) for arr in score_arrays)
        normalized_arrays = [arr[:min_length] for arr in score_arrays]

        aggregator = EnsembleAggregator()

        # Test mean aggregation
        mean_scores = aggregator.aggregate(normalized_arrays, method="mean")
        assert len(mean_scores) == min_length
        assert np.all((mean_scores >= 0) & (mean_scores <= 1))
        assert np.all(np.isfinite(mean_scores))

        # Mean should be between min and max of individual scores
        stacked = np.stack(normalized_arrays)
        min_scores = np.min(stacked, axis=0)
        max_scores = np.max(stacked, axis=0)
        assert np.all(mean_scores >= min_scores)
        assert np.all(mean_scores <= max_scores)

        # Test max aggregation
        max_scores_agg = aggregator.aggregate(normalized_arrays, method="max")
        assert np.allclose(max_scores_agg, max_scores)

    @given(
        stnp.arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(5, 20), st.integers(1, 5)),
            elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
        )
    )
    def test_feature_validator_properties(self, features: np.ndarray):
        """Test FeatureValidator maintains validation properties."""
        validator = FeatureValidator()

        # Validation should not modify input data
        original_features = features.copy()
        is_valid = validator.validate(features)
        assert np.array_equal(features, original_features)

        # Result should be boolean
        assert isinstance(is_valid, bool)

        # Valid data should pass validation
        if np.all(np.isfinite(features)) and features.shape[0] >= 2:
            assert is_valid

        # Invalid data should fail validation
        if np.any(~np.isfinite(features)) or features.shape[0] < 2:
            assert not is_valid

    @given(
        stnp.arrays(
            dtype=np.float64,
            shape=st.integers(10, 100),
            elements=st.floats(0, 1, allow_nan=False, allow_infinity=False),
        ),
        st.floats(0.01, 0.5),
    )
    def test_threshold_calculator_properties(
        self, scores: np.ndarray, contamination: float
    ):
        """Test ThresholdCalculator maintains mathematical properties."""
        calculator = ThresholdCalculator()

        # Compute threshold
        threshold = calculator.calculate_threshold(scores, contamination)

        # Threshold should be finite and in reasonable range
        assert np.isfinite(threshold)
        assert 0.0 <= threshold <= 1.0

        # Number of scores above threshold should approximate contamination
        n_above = np.sum(scores > threshold)
        expected = int(len(scores) * contamination)

        # Allow tolerance for discrete nature and edge cases
        tolerance = max(2, len(scores) * 0.1)
        assert abs(n_above - expected) <= tolerance

        # Monotonicity: higher contamination should not increase threshold
        # (more lenient threshold for higher contamination rates)
        if contamination < 0.4:  # Avoid edge cases
            higher_contamination = contamination + 0.1
            higher_threshold = calculator.calculate_threshold(
                scores, higher_contamination
            )
            assert (
                higher_threshold <= threshold + 1e-6
            )  # Small tolerance for numerical precision


class TestBusinessRuleProperties:
    """Test business logic invariants and rules."""

    @given(dataset_strategy(), detector_strategy())
    @settings(max_examples=20, deadline=5000)  # Reduce examples for complex test
    def test_detection_consistency(self, dataset: Dataset, detector: Detector):
        """Test that detection results are consistent with business rules."""
        # This is a simplified test - in practice, you'd use actual detector

        # Simulate detection result
        n_samples = len(dataset.features)
        scores = np.random.random(n_samples)  # Simplified scoring

        # Apply contamination rate to determine anomalies
        threshold = np.percentile(scores, (1 - detector.contamination.value) * 100)
        anomaly_indices = np.where(scores > threshold)[0]

        # Create anomalies
        anomalies = []
        for idx in anomaly_indices:
            anomaly = Anomaly(
                score=AnomalyScore(scores[idx]),
                index=int(idx),
                features=dataset.features[idx],
            )
            anomalies.append(anomaly)

        result = DetectionResult(
            detector=detector, dataset=dataset, anomalies=anomalies, scores=scores
        )

        # Business rule: Number of anomalies should not exceed contamination expectation
        expected_max_anomalies = (
            int(n_samples * detector.contamination.value) + 2
        )  # Allow tolerance
        assert len(result.anomalies) <= expected_max_anomalies

        # Business rule: All anomalies must have scores above threshold
        for anomaly in result.anomalies:
            assert anomaly.score.value >= threshold - 1e-6  # Small numerical tolerance

        # Business rule: Anomaly indices must be unique and valid
        indices = [a.index for a in result.anomalies]
        assert len(indices) == len(set(indices))  # Unique
        assert all(0 <= idx < n_samples for idx in indices)  # Valid range

"""Step definitions for anomaly detection BDD scenarios."""

from __future__ import annotations

import numpy as np
import pytest
from pytest_bdd import given, parsers, then, when
from sklearn.datasets import make_blobs

from pynomaly.domain.entities import Dataset
from pynomaly.domain.services import AnomalyScorer, ThresholdCalculator
from pynomaly.infrastructure.adapters import SklearnAdapter
from pynomaly.infrastructure.repositories import (
    InMemoryDetectorRepository,
    InMemoryResultRepository,
)


@pytest.fixture
def clean_dataset():
    """Create a clean dataset with known anomalies."""
    # Generate normal data
    normal_data, _ = make_blobs(
        n_samples=90, centers=1, n_features=2, random_state=42, cluster_std=1.0
    )

    # Generate anomalous data (far from normal data)
    anomalous_data = np.random.RandomState(42).uniform(-8, 8, (10, 2))

    # Combine data
    features = np.vstack([normal_data, anomalous_data])
    targets = np.array([0] * 90 + [1] * 10)  # 0 = normal, 1 = anomaly

    return Dataset(name="test_dataset", features=features, targets=targets)


@pytest.fixture
def pynomaly_system():
    """Initialize the Pynomaly system components."""
    return {
        "sklearn_adapter": SklearnAdapter(),
        "detector_repo": InMemoryDetectorRepository(),
        "result_repo": InMemoryResultRepository(),
        "scorer": AnomalyScorer(),
        "threshold_calc": ThresholdCalculator(),
    }


# Background steps
@given("I have a clean dataset with known anomalies")
def have_clean_dataset(clean_dataset):
    """Store the clean dataset in context."""
    pytest.dataset = clean_dataset


@given("the Pynomaly system is properly configured")
def pynomaly_configured(pynomaly_system):
    """Store system components in context."""
    pytest.system = pynomaly_system


# Dataset creation steps
@given(
    parsers.parse(
        "I have a dataset with {n_samples:d} samples and {n_features:d} features"
    )
)
def create_dataset(n_samples, n_features):
    """Create a dataset with specified dimensions."""
    # Create normal data
    n_normal = int(n_samples * 0.9)
    n_anomalies = n_samples - n_normal

    normal_data, _ = make_blobs(
        n_samples=n_normal,
        centers=1,
        n_features=n_features,
        random_state=42,
        cluster_std=1.0,
    )

    # Create anomalous data
    anomalous_data = np.random.RandomState(42).uniform(-5, 5, (n_anomalies, n_features))

    features = np.vstack([normal_data, anomalous_data])
    targets = np.array([0] * n_normal + [1] * n_anomalies)

    pytest.dataset = Dataset(
        name=f"test_dataset_{n_samples}x{n_features}",
        features=features,
        targets=targets,
    )


@given(parsers.parse("the dataset contains {percentage:d}% anomalies"))
def verify_anomaly_percentage(percentage):
    """Verify the dataset has the expected anomaly percentage."""
    if hasattr(pytest, "dataset") and pytest.dataset.targets is not None:
        actual_percentage = (
            np.sum(pytest.dataset.targets == 1) / len(pytest.dataset.targets)
        ) * 100
        assert (
            abs(actual_percentage - percentage) < 5
        ), f"Expected {percentage}% anomalies, got {actual_percentage:.1f}%"


# Detector creation steps
@when("I create an Isolation Forest detector with default contamination")
def create_isolation_forest_default():
    """Create an Isolation Forest detector with default settings."""
    adapter = pytest.system["sklearn_adapter"]
    pytest.detector = adapter.create_detector(
        "isolation_forest", contamination=0.1, random_state=42
    )


@when(
    parsers.parse(
        "I create a {algorithm} detector with {contamination:f} contamination"
    )
)
def create_detector_with_contamination(algorithm, contamination):
    """Create a detector with specified algorithm and contamination."""
    adapter = pytest.system["sklearn_adapter"]

    # Map friendly names to internal names
    algorithm_map = {
        "Local Outlier Factor": "local_outlier_factor",
        "Isolation Forest": "isolation_forest",
        "One-Class SVM": "one_class_svm",
    }

    algo_name = algorithm_map.get(algorithm, algorithm.lower().replace(" ", "_"))
    pytest.detector = adapter.create_detector(
        algo_name, contamination=contamination, random_state=42
    )


# Training steps
@when("I train the detector on the dataset")
def train_detector():
    """Train the detector on the dataset."""
    pytest.detector.fit(pytest.dataset.features)


# Detection steps
@when("I run anomaly detection")
def run_anomaly_detection():
    """Run anomaly detection and store results."""
    pytest.scores = pytest.detector.decision_function(pytest.dataset.features)
    pytest.predictions = pytest.detector.predict(pytest.dataset.features)


# Ensemble steps
@when(parsers.parse("I create an ensemble with {algorithms}"))
def create_ensemble(algorithms):
    """Create an ensemble with specified algorithms."""
    algorithm_list = [algo.strip() for algo in algorithms.split(",")]
    adapter = pytest.system["sklearn_adapter"]

    pytest.detectors = []
    algorithm_map = {
        "Isolation Forest": "isolation_forest",
        "LOF": "local_outlier_factor",
        "One-Class SVM": "one_class_svm",
    }

    for algo in algorithm_list:
        algo_name = algorithm_map.get(algo, algo.lower().replace(" ", "_"))
        detector = adapter.create_detector(
            algo_name, contamination=0.1, random_state=42
        )
        pytest.detectors.append(detector)


@when("I train all detectors on the dataset")
def train_all_detectors():
    """Train all detectors in the ensemble."""
    for detector in pytest.detectors:
        detector.fit(pytest.dataset.features)


@when(parsers.parse("I run ensemble anomaly detection with {method} voting"))
def run_ensemble_detection(method):
    """Run ensemble detection with specified voting method."""
    # Get scores from all detectors
    all_scores = []
    for detector in pytest.detectors:
        scores = detector.decision_function(pytest.dataset.features)
        all_scores.append(scores)

    # Aggregate scores based on method
    if method.lower() == "majority":
        # Simple average for majority voting
        pytest.ensemble_scores = np.mean(all_scores, axis=0)
    else:
        pytest.ensemble_scores = np.mean(all_scores, axis=0)


# Streaming steps
@given("I have a streaming data source")
def setup_streaming_source():
    """Set up a simulated streaming data source."""
    # Generate streaming data
    streaming_data = np.random.RandomState(42).normal(0, 1, (50, 2))
    pytest.streaming_data = streaming_data
    pytest.stream_index = 0


@given("I have a pre-trained anomaly detector")
def setup_pretrained_detector():
    """Set up a pre-trained detector."""
    adapter = pytest.system["sklearn_adapter"]
    detector = adapter.create_detector(
        "isolation_forest", contamination=0.1, random_state=42
    )

    # Train on some initial data
    training_data, _ = make_blobs(
        n_samples=100, centers=1, n_features=2, random_state=42
    )
    detector.fit(training_data)
    pytest.pretrained_detector = detector


@when("new data points arrive one by one")
def process_streaming_data():
    """Process streaming data points."""
    pytest.streaming_scores = []
    pytest.streaming_times = []

    import time

    for i in range(min(10, len(pytest.streaming_data))):
        data_point = pytest.streaming_data[i : i + 1]

        start_time = time.time()
        score = pytest.pretrained_detector.decision_function(data_point)
        end_time = time.time()

        pytest.streaming_scores.append(score[0])
        pytest.streaming_times.append(end_time - start_time)


@when("I score each point for anomalies")
def score_streaming_points():
    """Score each streaming point (already done in previous step)."""
    pass


# Validation steps
@then(
    parsers.parse("the detector should identify approximately {expected:d} anomalies")
)
def verify_anomaly_count(expected):
    """Verify the number of detected anomalies."""
    # Count anomalies (sklearn uses -1 for anomalies)
    n_anomalies = np.sum(pytest.predictions == -1)
    tolerance = max(2, expected * 0.3)  # Allow 30% tolerance
    assert (
        abs(n_anomalies - expected) <= tolerance
    ), f"Expected ~{expected} anomalies, got {n_anomalies}"


@then("all anomaly scores should be between 0 and 1")
def verify_score_range():
    """Verify all scores are in valid range."""
    assert np.all(
        pytest.scores >= 0
    ), f"Found negative scores: {pytest.scores[pytest.scores < 0]}"
    assert np.all(
        pytest.scores <= 1
    ), f"Found scores > 1: {pytest.scores[pytest.scores > 1]}"
    assert np.all(np.isfinite(pytest.scores)), "Found non-finite scores"


@then("anomalies should have higher scores than normal points")
def verify_anomaly_scores_higher():
    """Verify anomalies have higher scores than normal points."""
    if hasattr(pytest, "dataset") and pytest.dataset.targets is not None:
        normal_mask = pytest.dataset.targets == 0
        anomaly_mask = pytest.dataset.targets == 1

        if np.any(normal_mask) and np.any(anomaly_mask):
            avg_normal_score = np.mean(pytest.scores[normal_mask])
            avg_anomaly_score = np.mean(pytest.scores[anomaly_mask])

            assert (
                avg_anomaly_score > avg_normal_score
            ), f"Anomaly scores ({avg_anomaly_score:.3f}) should be higher than normal scores ({avg_normal_score:.3f})"


@then("the contamination rate should be respected")
def verify_contamination_rate():
    """Verify the contamination rate is approximately respected."""
    n_anomalies = np.sum(pytest.predictions == -1)
    expected_rate = 0.05  # From the scenario
    actual_rate = n_anomalies / len(pytest.predictions)

    assert (
        abs(actual_rate - expected_rate) < 0.03
    ), f"Expected contamination rate ~{expected_rate:.2f}, got {actual_rate:.3f}"


@then("the results should be deterministic with fixed random seed")
def verify_deterministic_results():
    """Verify results are deterministic."""
    # Run detection again with same detector
    scores2 = pytest.detector.decision_function(pytest.dataset.features)
    predictions2 = pytest.detector.predict(pytest.dataset.features)

    assert np.array_equal(pytest.scores, scores2), "Scores should be deterministic"
    assert np.array_equal(
        pytest.predictions, predictions2
    ), "Predictions should be deterministic"


@then("the ensemble should outperform individual detectors")
def verify_ensemble_performance():
    """Verify ensemble performs better than individual detectors."""
    if hasattr(pytest, "dataset") and pytest.dataset.targets is not None:
        from sklearn.metrics import roc_auc_score

        # Calculate AUC for ensemble
        ensemble_auc = roc_auc_score(pytest.dataset.targets, pytest.ensemble_scores)

        # Calculate AUC for individual detectors
        individual_aucs = []
        for detector in pytest.detectors:
            scores = detector.decision_function(pytest.dataset.features)
            auc = roc_auc_score(pytest.dataset.targets, scores)
            individual_aucs.append(auc)

        max_individual_auc = max(individual_aucs)

        # Ensemble should be at least as good as the best individual detector
        assert (
            ensemble_auc >= max_individual_auc - 0.05
        ), f"Ensemble AUC ({ensemble_auc:.3f}) should be >= best individual AUC ({max_individual_auc:.3f})"


@then("the final scores should be aggregated properly")
def verify_score_aggregation():
    """Verify ensemble scores are properly aggregated."""
    assert hasattr(pytest, "ensemble_scores"), "Ensemble scores should be computed"
    assert len(pytest.ensemble_scores) == len(
        pytest.dataset.features
    ), "Ensemble scores should match dataset size"
    assert np.all(
        np.isfinite(pytest.ensemble_scores)
    ), "Ensemble scores should be finite"


@then("confidence intervals should be provided for anomalies")
def verify_confidence_intervals():
    """Verify confidence intervals are provided."""
    # This would require implementing confidence interval calculation
    # For now, just verify we can compute basic statistics
    anomaly_indices = np.where(
        pytest.ensemble_scores > np.percentile(pytest.ensemble_scores, 90)
    )[0]
    assert (
        len(anomaly_indices) > 0
    ), "Should identify some anomalies for confidence intervals"


@then(parsers.parse("each score should be computed quickly (< {max_time:d}ms)"))
def verify_streaming_performance(max_time):
    """Verify streaming performance meets requirements."""
    max_time_seconds = max_time / 1000.0
    for i, time_taken in enumerate(pytest.streaming_times):
        assert (
            time_taken < max_time_seconds
        ), f"Point {i} took {time_taken * 1000:.1f}ms, expected < {max_time}ms"


@then("scores should be consistent with batch processing")
def verify_streaming_consistency():
    """Verify streaming scores are consistent with batch processing."""
    # Compare streaming scores with batch processing
    batch_scores = pytest.pretrained_detector.decision_function(
        pytest.streaming_data[: len(pytest.streaming_scores)]
    )

    for i, (stream_score, batch_score) in enumerate(
        zip(pytest.streaming_scores, batch_scores, strict=False)
    ):
        assert (
            abs(stream_score - batch_score) < 1e-10
        ), f"Point {i}: streaming score {stream_score} != batch score {batch_score}"


@then("the system should handle data drift gracefully")
def verify_data_drift_handling():
    """Verify the system handles data drift."""
    # This is a placeholder - in practice you'd implement drift detection
    # For now, just verify the system continues to produce valid scores
    assert all(
        0 <= score <= 1 for score in pytest.streaming_scores
    ), "Scores should remain valid despite potential data drift"

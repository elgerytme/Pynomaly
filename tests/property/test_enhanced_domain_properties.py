"""Enhanced property-based testing for domain entities and value objects."""

from datetime import datetime

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

# Import domain entities and value objects
from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import (
    InvalidAnomalyScoreError,
    InvalidContaminationRateError,
)
from pynomaly.domain.value_objects import (
    AnomalyScore,
    ContaminationRate,
)


# Custom strategies for domain testing
@st.composite
def valid_contamination_rates(draw):
    """Generate valid contamination rate values."""
    return draw(
        st.floats(min_value=0.001, max_value=0.999, exclude_min=True, exclude_max=True)
    )


@st.composite
def valid_anomaly_scores(draw):
    """Generate valid anomaly score values."""
    score = draw(
        st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        )
    )
    confidence = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    return {"score": score, "confidence": confidence}


@st.composite
def valid_dataset_data(draw):
    """Generate valid dataset data structures."""
    n_samples = draw(st.integers(min_value=10, max_value=1000))
    n_features = draw(st.integers(min_value=1, max_value=50))

    # Generate realistic data
    data = draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=-100.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n_features,
                max_size=n_features,
            ),
            min_size=n_samples,
            max_size=n_samples,
        )
    )

    feature_names = [f"feature_{i}" for i in range(n_features)]

    return {
        "data": data,
        "feature_names": feature_names,
        "n_samples": n_samples,
        "n_features": n_features,
    }


@st.composite
def valid_detector_configs(draw):
    """Generate valid detector configuration dictionaries."""
    algorithm = draw(
        st.sampled_from(
            [
                "IsolationForest",
                "LocalOutlierFactor",
                "OneClassSVM",
                "EllipticEnvelope",
                "KNN",
                "AutoEncoder",
            ]
        )
    )

    contamination = draw(valid_contamination_rates())

    base_config = {
        "algorithm": algorithm,
        "contamination": contamination,
        "random_state": draw(st.one_of(st.none(), st.integers(0, 2**31 - 1))),
    }

    # Add algorithm-specific parameters
    if algorithm == "IsolationForest":
        base_config.update(
            {
                "n_estimators": draw(st.integers(10, 200)),
                "max_samples": draw(
                    st.one_of(st.integers(10, 1000), st.floats(0.1, 1.0))
                ),
            }
        )
    elif algorithm == "LocalOutlierFactor":
        base_config.update(
            {
                "n_neighbors": draw(st.integers(1, 50)),
                "algorithm": draw(
                    st.sampled_from(["auto", "ball_tree", "kd_tree", "brute"])
                ),
            }
        )
    elif algorithm == "OneClassSVM":
        base_config.update(
            {
                "kernel": draw(st.sampled_from(["linear", "poly", "rbf", "sigmoid"])),
                "nu": draw(st.floats(0.01, 0.99)),
            }
        )

    return base_config


class TestAnomalyScoreProperties:
    """Property-based tests for AnomalyScore value object."""

    @given(score_data=valid_anomaly_scores())
    @settings(max_examples=100)
    def test_anomaly_score_creation_properties(self, score_data):
        """Test AnomalyScore creation with various valid inputs."""
        score = AnomalyScore(
            value=score_data["score"], confidence=score_data["confidence"]
        )

        # Property 1: Value preservation
        assert score.value == score_data["score"]
        assert score.confidence == score_data["confidence"]

        # Property 2: Confidence bounds
        assert 0.0 <= score.confidence <= 1.0

        # Property 3: Score should be finite
        assert np.isfinite(score.value)

        # Property 4: String representation should be meaningful
        str_repr = str(score)
        assert "AnomalyScore" in str_repr
        assert (
            str(score_data["score"]) in str_repr
            or f"{score_data['score']:.4f}" in str_repr
        )

    @given(score1=valid_anomaly_scores(), score2=valid_anomaly_scores())
    @settings(max_examples=50)
    def test_anomaly_score_comparison_properties(self, score1, score2):
        """Test AnomalyScore comparison operations."""
        as1 = AnomalyScore(value=score1["score"], confidence=score1["confidence"])
        as2 = AnomalyScore(value=score2["score"], confidence=score2["confidence"])

        # Property 1: Equality is reflexive
        assert as1 == as1

        # Property 2: Equality is symmetric
        if as1 == as2:
            assert as2 == as1

        # Property 3: Comparison consistency
        if score1["score"] < score2["score"]:
            assert as1 < as2
        elif score1["score"] > score2["score"]:
            assert as1 > as2
        else:
            # Equal scores - compare by confidence
            if score1["confidence"] < score2["confidence"]:
                assert as1 < as2
            elif score1["confidence"] > score2["confidence"]:
                assert as1 > as2
            else:
                assert as1 == as2

    @given(scores=st.lists(valid_anomaly_scores(), min_size=1, max_size=20))
    @settings(max_examples=30)
    def test_anomaly_score_aggregation_properties(self, scores):
        """Test properties when aggregating multiple AnomalyScore objects."""
        anomaly_scores = [
            AnomalyScore(value=s["score"], confidence=s["confidence"]) for s in scores
        ]

        # Property 1: Can sort scores
        sorted_scores = sorted(anomaly_scores)
        assert len(sorted_scores) == len(anomaly_scores)

        # Property 2: Sorting is consistent
        if len(sorted_scores) > 1:
            for i in range(len(sorted_scores) - 1):
                assert sorted_scores[i] <= sorted_scores[i + 1]

        # Property 3: Min/max operations work correctly
        min_score = min(anomaly_scores)
        max_score = max(anomaly_scores)

        assert min_score in anomaly_scores
        assert max_score in anomaly_scores
        assert min_score <= max_score

    @given(
        invalid_confidence=st.one_of(
            st.floats(min_value=-10.0, max_value=-0.001),
            st.floats(min_value=1.001, max_value=10.0),
            st.just(float("nan")),
            st.just(float("inf")),
        )
    )
    def test_anomaly_score_validation_properties(self, invalid_confidence):
        """Test AnomalyScore validation with invalid inputs."""
        # Property: Invalid confidence should raise appropriate error
        with pytest.raises((InvalidAnomalyScoreError, ValueError)):
            AnomalyScore(value=0.5, confidence=invalid_confidence)

    @given(
        score=st.floats(allow_nan=True, allow_infinity=True),
        confidence=st.floats(0.0, 1.0),
    )
    def test_anomaly_score_robustness_properties(self, score, confidence):
        """Test AnomalyScore robustness to edge cases."""
        if np.isnan(score) or np.isinf(score):
            # Property: NaN or infinite scores should be rejected
            with pytest.raises((InvalidAnomalyScoreError, ValueError)):
                AnomalyScore(value=score, confidence=confidence)
        else:
            # Property: Valid finite scores should be accepted
            anomaly_score = AnomalyScore(value=score, confidence=confidence)
            assert anomaly_score.value == score
            assert anomaly_score.confidence == confidence


class TestContaminationRateProperties:
    """Property-based tests for ContaminationRate value object."""

    @given(rate=valid_contamination_rates())
    @settings(max_examples=100)
    def test_contamination_rate_creation_properties(self, rate):
        """Test ContaminationRate creation with valid rates."""
        contamination = ContaminationRate(rate)

        # Property 1: Value preservation
        assert contamination.value == rate

        # Property 2: Rate bounds
        assert 0.0 < contamination.value < 1.0

        # Property 3: Percentage calculation
        percentage = contamination.as_percentage()
        assert 0.0 < percentage < 100.0
        assert abs(percentage - (rate * 100)) < 1e-10

        # Property 4: String representation includes percentage
        str_repr = str(contamination)
        assert "%" in str_repr

    @given(rate1=valid_contamination_rates(), rate2=valid_contamination_rates())
    @settings(max_examples=50)
    def test_contamination_rate_comparison_properties(self, rate1, rate2):
        """Test ContaminationRate comparison operations."""
        cr1 = ContaminationRate(rate1)
        cr2 = ContaminationRate(rate2)

        # Property 1: Comparison consistency with underlying values
        if rate1 < rate2:
            assert cr1 < cr2
        elif rate1 > rate2:
            assert cr1 > cr2
        else:
            assert cr1 == cr2

        # Property 2: Reflexivity
        assert cr1 == cr1

        # Property 3: Symmetry
        if cr1 == cr2:
            assert cr2 == cr1

    @given(
        invalid_rate=st.one_of(
            st.floats(max_value=0.0),
            st.floats(min_value=1.0),
            st.just(float("nan")),
            st.just(float("inf")),
        )
    )
    def test_contamination_rate_validation_properties(self, invalid_rate):
        """Test ContaminationRate validation with invalid rates."""
        # Property: Invalid rates should raise appropriate error
        with pytest.raises((InvalidContaminationRateError, ValueError)):
            ContaminationRate(invalid_rate)

    @given(rates=st.lists(valid_contamination_rates(), min_size=2, max_size=10))
    @settings(max_examples=20)
    def test_contamination_rate_operations_properties(self, rates):
        """Test mathematical operations on ContaminationRate."""
        contamination_rates = [ContaminationRate(rate) for rate in rates]

        # Property 1: Can find minimum and maximum
        min_rate = min(contamination_rates)
        max_rate = max(contamination_rates)

        assert min_rate.value == min(rates)
        assert max_rate.value == max(rates)

        # Property 2: Average calculation preserves bounds
        avg_value = sum(rates) / len(rates)
        if 0.0 < avg_value < 1.0:
            avg_rate = ContaminationRate(avg_value)
            assert 0.0 < avg_rate.value < 1.0


class TestDatasetProperties:
    """Property-based tests for Dataset entity."""

    @given(
        dataset_data=valid_dataset_data(),
        name=st.text(min_size=1, max_size=100),
        description=st.one_of(st.none(), st.text(max_size=500)),
    )
    @settings(max_examples=50)
    def test_dataset_creation_properties(self, dataset_data, name, description):
        """Test Dataset creation with various valid inputs."""
        assume(name.strip())  # Ensure name is not just whitespace

        dataset = Dataset(
            name=name.strip(),
            data=dataset_data["data"],
            feature_names=dataset_data["feature_names"],
            description=description,
        )

        # Property 1: Basic attribute preservation
        assert dataset.name == name.strip()
        assert dataset.description == description

        # Property 2: Data shape consistency
        assert dataset.n_samples == dataset_data["n_samples"]
        assert dataset.n_features == dataset_data["n_features"]
        assert len(dataset.feature_names) == dataset_data["n_features"]

        # Property 3: Data integrity
        assert len(dataset.data) == dataset_data["n_samples"]
        for row in dataset.data:
            assert len(row) == dataset_data["n_features"]

        # Property 4: ID generation
        assert dataset.id is not None
        assert isinstance(dataset.id, str)

        # Property 5: Creation timestamp
        assert dataset.created_at is not None
        assert isinstance(dataset.created_at, datetime)

    @given(
        base_data=valid_dataset_data(),
        subset_size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=30)
    def test_dataset_subset_properties(self, base_data, subset_size):
        """Test Dataset subset operations."""
        assume(subset_size <= base_data["n_samples"])

        original_dataset = Dataset(
            name="Original",
            data=base_data["data"],
            feature_names=base_data["feature_names"],
        )

        # Create subset
        subset_indices = list(range(subset_size))
        subset_data = [base_data["data"][i] for i in subset_indices]

        subset_dataset = Dataset(
            name="Subset", data=subset_data, feature_names=base_data["feature_names"]
        )

        # Property 1: Subset has correct dimensions
        assert subset_dataset.n_samples == subset_size
        assert subset_dataset.n_features == base_data["n_features"]

        # Property 2: Feature names are preserved
        assert subset_dataset.feature_names == original_dataset.feature_names

        # Property 3: Subset data matches original data
        for i, row in enumerate(subset_dataset.data):
            assert row == original_dataset.data[i]

    @given(data1=valid_dataset_data(), data2=valid_dataset_data())
    @settings(max_examples=20)
    def test_dataset_comparison_properties(self, data1, data2):
        """Test Dataset comparison and equality properties."""
        dataset1a = Dataset(
            name="Dataset1", data=data1["data"], feature_names=data1["feature_names"]
        )

        dataset1b = Dataset(
            name="Dataset1", data=data1["data"], feature_names=data1["feature_names"]
        )

        dataset2 = Dataset(
            name="Dataset2", data=data2["data"], feature_names=data2["feature_names"]
        )

        # Property 1: Datasets with same data should be considered equivalent
        # (even if they have different IDs)
        assert dataset1a.n_samples == dataset1b.n_samples
        assert dataset1a.n_features == dataset1b.n_features
        assert dataset1a.feature_names == dataset1b.feature_names

        # Property 2: Different datasets should have different characteristics
        if (
            data1["data"] != data2["data"]
            or data1["feature_names"] != data2["feature_names"]
        ):
            assert (
                dataset1a.n_samples != dataset2.n_samples
                or dataset1a.n_features != dataset2.n_features
                or dataset1a.feature_names != dataset2.feature_names
            )


class TestDetectorProperties:
    """Property-based tests for Detector entity."""

    @given(config=valid_detector_configs(), name=st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_detector_creation_properties(self, config, name):
        """Test Detector creation with various configurations."""
        assume(name.strip())

        detector = Detector(
            name=name.strip(), algorithm=config["algorithm"], parameters=config
        )

        # Property 1: Basic attribute preservation
        assert detector.name == name.strip()
        assert detector.algorithm == config["algorithm"]

        # Property 2: Parameters preservation
        for key, value in config.items():
            assert detector.parameters[key] == value

        # Property 3: Initial state
        assert not detector.is_fitted
        assert detector.training_data is None
        assert detector.performance_metrics is None

        # Property 4: ID and timestamp generation
        assert detector.id is not None
        assert isinstance(detector.id, str)
        assert detector.created_at is not None
        assert isinstance(detector.created_at, datetime)

    @given(config=valid_detector_configs(), training_data=valid_dataset_data())
    @settings(max_examples=30)
    def test_detector_training_properties(self, config, training_data):
        """Test Detector training state properties."""
        detector = Detector(
            name="Test Detector", algorithm=config["algorithm"], parameters=config
        )

        # Simulate training
        detector.training_data = training_data["data"]
        detector.is_fitted = True
        detector.fitted_at = datetime.now()

        # Property 1: Training state consistency
        assert detector.is_fitted
        assert detector.training_data is not None
        assert detector.fitted_at is not None

        # Property 2: Training data dimensions
        assert len(detector.training_data) == training_data["n_samples"]
        for row in detector.training_data:
            assert len(row) == training_data["n_features"]

        # Property 3: Fitted timestamp should be after creation
        assert detector.fitted_at >= detector.created_at

    @given(
        config=valid_detector_configs(),
        metrics_data=st.dictionaries(
            keys=st.sampled_from(
                ["precision", "recall", "f1_score", "accuracy", "auc"]
            ),
            values=st.floats(min_value=0.0, max_value=1.0),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_detector_performance_properties(self, config, metrics_data):
        """Test Detector performance metrics properties."""
        detector = Detector(
            name="Test Detector", algorithm=config["algorithm"], parameters=config
        )

        # Set performance metrics
        detector.performance_metrics = metrics_data

        # Property 1: Metrics preservation
        for metric, value in metrics_data.items():
            assert detector.performance_metrics[metric] == value

        # Property 2: Metric values are in valid range
        for value in detector.performance_metrics.values():
            assert 0.0 <= value <= 1.0

        # Property 3: Can compute average performance
        if detector.performance_metrics:
            avg_performance = sum(detector.performance_metrics.values()) / len(
                detector.performance_metrics
            )
            assert 0.0 <= avg_performance <= 1.0


class TestDetectionResultProperties:
    """Property-based tests for DetectionResult entity."""

    @given(
        n_samples=st.integers(min_value=10, max_value=500),
        detector_config=valid_detector_configs(),
    )
    @settings(max_examples=30)
    def test_detection_result_creation_properties(self, n_samples, detector_config):
        """Test DetectionResult creation properties."""
        # Generate mock predictions and scores
        predictions = np.random.choice([-1, 1], size=n_samples)
        scores = np.random.uniform(-2, 2, size=n_samples)

        # Create anomaly scores
        anomaly_scores = [
            AnomalyScore(value=score, confidence=np.random.uniform(0.5, 1.0))
            for score in scores
        ]

        # Create detector
        detector = Detector(
            name="Test Detector",
            algorithm=detector_config["algorithm"],
            parameters=detector_config,
        )

        # Create detection result
        result = DetectionResult(
            detector=detector,
            predictions=predictions.tolist(),
            anomaly_scores=anomaly_scores,
        )

        # Property 1: Basic attribute preservation
        assert result.detector == detector
        assert len(result.predictions) == n_samples
        assert len(result.anomaly_scores) == n_samples

        # Property 2: Predictions are valid
        assert all(pred in [-1, 1] for pred in result.predictions)

        # Property 3: Anomaly scores match predictions count
        assert len(result.anomaly_scores) == len(result.predictions)

        # Property 4: ID and timestamp generation
        assert result.id is not None
        assert isinstance(result.id, str)
        assert result.created_at is not None

        # Property 5: Statistics calculation
        anomaly_count = result.anomaly_count
        total_count = result.total_samples
        anomaly_rate = result.anomaly_rate

        assert anomaly_count == sum(1 for pred in predictions if pred == -1)
        assert total_count == n_samples
        assert abs(anomaly_rate - (anomaly_count / total_count)) < 1e-10

    @given(
        predictions=st.lists(st.sampled_from([-1, 1]), min_size=1, max_size=100),
        score_range=st.tuples(
            st.floats(min_value=-10.0, max_value=0.0),
            st.floats(min_value=0.0, max_value=10.0),
        ),
    )
    @settings(max_examples=30)
    def test_detection_result_statistics_properties(self, predictions, score_range):
        """Test DetectionResult statistical properties."""
        min_score, max_score = score_range
        assume(min_score < max_score)

        # Generate scores correlated with predictions
        scores = []
        for pred in predictions:
            if pred == -1:  # Anomaly
                score = np.random.uniform(min_score, (min_score + max_score) / 2)
            else:  # Normal
                score = np.random.uniform((min_score + max_score) / 2, max_score)
            scores.append(score)

        anomaly_scores = [
            AnomalyScore(value=score, confidence=np.random.uniform(0.6, 1.0))
            for score in scores
        ]

        # Create mock detector
        detector = Detector(
            name="Test Detector",
            algorithm="IsolationForest",
            parameters={"contamination": 0.1},
        )

        result = DetectionResult(
            detector=detector, predictions=predictions, anomaly_scores=anomaly_scores
        )

        # Property 1: Anomaly count consistency
        expected_anomaly_count = sum(1 for pred in predictions if pred == -1)
        assert result.anomaly_count == expected_anomaly_count

        # Property 2: Rate calculation consistency
        expected_rate = expected_anomaly_count / len(predictions)
        assert abs(result.anomaly_rate - expected_rate) < 1e-10

        # Property 3: Score statistics
        if result.anomaly_count > 0:
            anomaly_indices = [i for i, pred in enumerate(predictions) if pred == -1]
            anomaly_score_values = [
                result.anomaly_scores[i].value for i in anomaly_indices
            ]

            # Anomalies should generally have lower scores
            if len(anomaly_score_values) > 1:
                avg_anomaly_score = np.mean(anomaly_score_values)
                normal_indices = [i for i, pred in enumerate(predictions) if pred == 1]
                if normal_indices:
                    normal_score_values = [
                        result.anomaly_scores[i].value for i in normal_indices
                    ]
                    avg_normal_score = np.mean(normal_score_values)

                    # This should hold given our score generation logic
                    assert avg_anomaly_score <= avg_normal_score

    @given(
        results_count=st.integers(min_value=2, max_value=10),
        base_predictions=st.lists(st.sampled_from([-1, 1]), min_size=10, max_size=50),
    )
    @settings(max_examples=20)
    def test_detection_result_comparison_properties(
        self, results_count, base_predictions
    ):
        """Test comparison properties between multiple DetectionResult objects."""
        results = []

        for i in range(results_count):
            # Vary the anomaly rate for each result
            anomaly_prob = 0.1 + (i * 0.1)  # From 0.1 to results_count * 0.1

            predictions = []
            scores = []

            for base_pred in base_predictions:
                # Introduce some variation in predictions
                if np.random.random() < 0.1:  # 10% chance to flip
                    pred = -base_pred
                else:
                    pred = base_pred

                predictions.append(pred)

                # Generate score based on prediction
                if pred == -1:
                    score = np.random.uniform(-2, 0)
                else:
                    score = np.random.uniform(0, 2)
                scores.append(score)

            anomaly_scores = [
                AnomalyScore(value=score, confidence=np.random.uniform(0.7, 1.0))
                for score in scores
            ]

            detector = Detector(
                name=f"Detector_{i}",
                algorithm="IsolationForest",
                parameters={"contamination": anomaly_prob},
            )

            result = DetectionResult(
                detector=detector,
                predictions=predictions,
                anomaly_scores=anomaly_scores,
            )

            results.append(result)

        # Property 1: Results can be compared by anomaly rate
        sorted_results = sorted(results, key=lambda r: r.anomaly_rate)

        for i in range(len(sorted_results) - 1):
            assert sorted_results[i].anomaly_rate <= sorted_results[i + 1].anomaly_rate

        # Property 2: Each result maintains internal consistency
        for result in results:
            assert result.total_samples == len(base_predictions)
            assert result.anomaly_count <= result.total_samples
            assert 0.0 <= result.anomaly_rate <= 1.0

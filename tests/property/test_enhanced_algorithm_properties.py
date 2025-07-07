"""Enhanced property-based testing for anomaly detection algorithms."""

import warnings
from typing import Any, Dict, List, Tuple, Union
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


# Custom strategies for anomaly detection testing
@st.composite
def anomaly_datasets(
    draw, min_samples=10, max_samples=1000, min_features=1, max_features=20
):
    """Generate realistic anomaly detection datasets."""
    n_samples = draw(st.integers(min_samples, max_samples))
    n_features = draw(st.integers(min_features, max_features))
    contamination = draw(st.floats(0.01, 0.5))

    # Generate normal data
    normal_data = draw(
        arrays(
            dtype=np.float64,
            shape=(int(n_samples * (1 - contamination)), n_features),
            elements=st.floats(-3, 3, allow_nan=False, allow_infinity=False),
        )
    )

    # Generate anomalous data (more extreme values)
    n_anomalies = n_samples - len(normal_data)
    if n_anomalies > 0:
        anomaly_data = draw(
            arrays(
                dtype=np.float64,
                shape=(n_anomalies, n_features),
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            )
        )

        # Combine normal and anomalous data
        data = np.vstack([normal_data, anomaly_data])
        labels = np.concatenate(
            [np.ones(len(normal_data)), -np.ones(len(anomaly_data))]
        )
    else:
        data = normal_data
        labels = np.ones(len(normal_data))

    # Shuffle data
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]

    return {
        "data": data,
        "labels": labels,
        "contamination": contamination,
        "n_samples": n_samples,
        "n_features": n_features,
    }


@st.composite
def algorithm_parameters(draw, algorithm_name: str):
    """Generate valid parameters for different algorithms."""
    if algorithm_name == "IsolationForest":
        return {
            "n_estimators": draw(st.integers(10, 200)),
            "max_samples": draw(st.one_of(st.integers(10, 1000), st.floats(0.1, 1.0))),
            "contamination": draw(st.floats(0.01, 0.5)),
            "max_features": draw(st.one_of(st.integers(1, 10), st.floats(0.1, 1.0))),
            "bootstrap": draw(st.booleans()),
            "random_state": draw(st.one_of(st.none(), st.integers(0, 2**31 - 1))),
        }
    elif algorithm_name == "LocalOutlierFactor":
        return {
            "n_neighbors": draw(st.integers(1, 50)),
            "algorithm": draw(
                st.sampled_from(["auto", "ball_tree", "kd_tree", "brute"])
            ),
            "leaf_size": draw(st.integers(10, 50)),
            "contamination": draw(st.floats(0.01, 0.5)),
            "novelty": draw(st.booleans()),
        }
    elif algorithm_name == "OneClassSVM":
        return {
            "kernel": draw(st.sampled_from(["linear", "poly", "rbf", "sigmoid"])),
            "degree": draw(st.integers(2, 5)),
            "gamma": draw(
                st.one_of(st.sampled_from(["scale", "auto"]), st.floats(1e-4, 1e-1))
            ),
            "nu": draw(st.floats(0.01, 0.99)),
            "shrinking": draw(st.booleans()),
            "cache_size": draw(st.integers(100, 1000)),
        }
    else:
        return {}


@st.composite
def contamination_rates(draw):
    """Generate valid contamination rates."""
    return draw(st.floats(0.001, 0.5))


class TestEnhancedAlgorithmProperties:
    """Enhanced property-based tests for anomaly detection algorithms."""

    @given(dataset=anomaly_datasets(min_samples=20, max_samples=500))
    @settings(
        max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_isolation_forest_properties(self, dataset):
        """Test IsolationForest algorithm properties with various datasets."""
        from sklearn.ensemble import IsolationForest

        data = dataset["data"]
        contamination = dataset["contamination"]

        # Skip if dataset is too small or has invalid contamination
        assume(len(data) >= 10)
        assume(0.01 <= contamination <= 0.5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=50,  # Smaller for faster testing
            )

            detector.fit(data)
            predictions = detector.predict(data)
            scores = detector.decision_function(data)

            # Property 1: Output shapes should match input
            assert len(predictions) == len(data)
            assert len(scores) == len(data)

            # Property 2: Predictions should be binary (-1 or 1)
            assert set(np.unique(predictions)).issubset({-1, 1})

            # Property 3: Decision scores should be finite
            assert np.all(np.isfinite(scores))

            # Property 4: Contamination rate should be approximately respected
            anomaly_rate = np.sum(predictions == -1) / len(predictions)
            # Allow some tolerance due to algorithm approximation
            assert abs(anomaly_rate - contamination) <= 0.2

            # Property 5: Anomalies should have lower scores than normal points
            if len(np.unique(predictions)) == 2:  # Both classes present
                normal_scores = scores[predictions == 1]
                anomaly_scores = scores[predictions == -1]

                if len(normal_scores) > 0 and len(anomaly_scores) > 0:
                    assert np.mean(anomaly_scores) < np.mean(normal_scores)

    @given(dataset=anomaly_datasets(min_samples=20, max_samples=200))
    @settings(
        max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_local_outlier_factor_properties(self, dataset):
        """Test LocalOutlierFactor algorithm properties."""
        from sklearn.neighbors import LocalOutlierFactor

        data = dataset["data"]
        contamination = dataset["contamination"]

        assume(len(data) >= 10)
        assume(0.01 <= contamination <= 0.5)
        assume(data.shape[1] >= 1)

        # Ensure we have enough neighbors
        n_neighbors = min(20, len(data) - 1)
        assume(n_neighbors >= 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            detector = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=contamination, novelty=False
            )

            predictions = detector.fit_predict(data)
            scores = detector.negative_outlier_factor_

            # Property 1: Output shapes should match input
            assert len(predictions) == len(data)
            assert len(scores) == len(data)

            # Property 2: Predictions should be binary (-1 or 1)
            assert set(np.unique(predictions)).issubset({-1, 1})

            # Property 3: Scores should be finite and negative
            assert np.all(np.isfinite(scores))
            assert np.all(scores <= 0)  # LOF scores are negative

            # Property 4: Contamination rate should be approximately respected
            anomaly_rate = np.sum(predictions == -1) / len(predictions)
            assert abs(anomaly_rate - contamination) <= 0.2

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(50, 300), st.integers(2, 10)),
            elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
        ),
        contamination=contamination_rates(),
    )
    @settings(
        max_examples=20, deadline=15000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_one_class_svm_properties(self, data, contamination):
        """Test OneClassSVM algorithm properties."""
        from sklearn.svm import OneClassSVM

        assume(len(data) >= 20)
        assume(0.01 <= contamination <= 0.5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            detector = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")

            detector.fit(data)
            predictions = detector.predict(data)
            scores = detector.decision_function(data)

            # Property 1: Output shapes should match input
            assert len(predictions) == len(data)
            assert len(scores) == len(data)

            # Property 2: Predictions should be binary (-1 or 1)
            assert set(np.unique(predictions)).issubset({-1, 1})

            # Property 3: Decision scores should be finite
            assert np.all(np.isfinite(scores))

            # Property 4: Nu parameter should roughly control outlier fraction
            anomaly_rate = np.sum(predictions == -1) / len(predictions)
            # OneClassSVM's nu is an upper bound on fraction of outliers
            assert anomaly_rate <= contamination + 0.2

    @given(
        algorithms=st.lists(
            st.sampled_from(["IsolationForest", "LocalOutlierFactor"]),
            min_size=2,
            max_size=3,
            unique=True,
        ),
        dataset=anomaly_datasets(min_samples=50, max_samples=200),
    )
    @settings(
        max_examples=20, deadline=20000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_ensemble_properties(self, algorithms, dataset):
        """Test ensemble algorithm properties."""
        data = dataset["data"]
        contamination = dataset["contamination"]

        assume(len(data) >= 50)
        assume(0.01 <= contamination <= 0.3)

        detectors = []
        all_predictions = []
        all_scores = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for algorithm in algorithms:
                if algorithm == "IsolationForest":
                    from sklearn.ensemble import IsolationForest

                    detector = IsolationForest(
                        contamination=contamination, random_state=42, n_estimators=30
                    )
                elif algorithm == "LocalOutlierFactor":
                    from sklearn.neighbors import LocalOutlierFactor

                    n_neighbors = min(20, len(data) - 1)
                    detector = LocalOutlierFactor(
                        n_neighbors=n_neighbors,
                        contamination=contamination,
                        novelty=False,
                    )

                detectors.append(detector)

                if algorithm == "LocalOutlierFactor":
                    predictions = detector.fit_predict(data)
                    scores = detector.negative_outlier_factor_
                else:
                    detector.fit(data)
                    predictions = detector.predict(data)
                    scores = detector.decision_function(data)

                all_predictions.append(predictions)
                all_scores.append(scores)

            # Property 1: All detectors should produce same-shaped outputs
            assert all(len(pred) == len(data) for pred in all_predictions)
            assert all(len(score) == len(data) for score in all_scores)

            # Property 2: Ensemble voting should be consistent
            ensemble_predictions = np.array(all_predictions)
            majority_vote = np.sign(np.sum(ensemble_predictions, axis=0))

            # Majority vote should be binary
            assert set(np.unique(majority_vote)).issubset({-1, 0, 1})

            # Property 3: Ensemble should roughly respect contamination rate
            if 0 not in majority_vote:  # No ties
                ensemble_anomaly_rate = np.sum(majority_vote == -1) / len(majority_vote)
                assert abs(ensemble_anomaly_rate - contamination) <= 0.3

    @given(
        original_data=arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(50, 200), st.integers(2, 8)),
            elements=st.floats(-3, 3, allow_nan=False, allow_infinity=False),
        ),
        noise_level=st.floats(0.01, 0.5),
    )
    @settings(
        max_examples=15, deadline=10000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_robustness_properties(self, original_data, noise_level):
        """Test algorithm robustness to data perturbations."""
        from sklearn.ensemble import IsolationForest

        assume(len(original_data) >= 50)

        # Add noise to create perturbed data
        noise = np.random.normal(0, noise_level, original_data.shape)
        perturbed_data = original_data + noise

        contamination = 0.1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Train on original data
            detector_original = IsolationForest(
                contamination=contamination, random_state=42, n_estimators=50
            )
            detector_original.fit(original_data)
            scores_original = detector_original.decision_function(original_data)

            # Train on perturbed data
            detector_perturbed = IsolationForest(
                contamination=contamination, random_state=42, n_estimators=50
            )
            detector_perturbed.fit(perturbed_data)
            scores_perturbed = detector_perturbed.decision_function(perturbed_data)

            # Property 1: Scores should be correlated despite noise
            correlation = np.corrcoef(scores_original, scores_perturbed)[0, 1]

            # With small noise, correlation should be reasonably high
            if noise_level < 0.2:
                assert correlation > 0.3  # Allow for some variation due to randomness

            # Property 2: Extreme scores should remain relatively stable
            # Top 10% most normal points should overlap significantly
            top_normal_original = np.argsort(scores_original)[
                -int(0.1 * len(scores_original)) :
            ]
            top_normal_perturbed = np.argsort(scores_perturbed)[
                -int(0.1 * len(scores_perturbed)) :
            ]

            overlap = len(set(top_normal_original) & set(top_normal_perturbed))
            overlap_rate = overlap / len(top_normal_original)

            # With small noise, should have reasonable overlap
            if noise_level < 0.1:
                assert overlap_rate > 0.2

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(100, 500), st.integers(2, 10)),
            elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
        ),
        test_fraction=st.floats(0.1, 0.5),
    )
    @settings(
        max_examples=10, deadline=15000, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_generalization_properties(self, data, test_fraction):
        """Test algorithm generalization properties."""
        from sklearn.ensemble import IsolationForest
        from sklearn.model_selection import train_test_split

        assume(len(data) >= 100)

        # Split data into train and test
        train_data, test_data = train_test_split(
            data, test_size=test_fraction, random_state=42
        )

        assume(len(train_data) >= 20)
        assume(len(test_data) >= 10)

        contamination = 0.1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Train on training data
            detector = IsolationForest(
                contamination=contamination, random_state=42, n_estimators=50
            )
            detector.fit(train_data)

            # Test on both training and test data
            train_scores = detector.decision_function(train_data)
            test_scores = detector.decision_function(test_data)

            train_predictions = detector.predict(train_data)
            test_predictions = detector.predict(test_data)

            # Property 1: Predictions should be valid on both sets
            assert len(train_scores) == len(train_data)
            assert len(test_scores) == len(test_data)
            assert set(np.unique(train_predictions)).issubset({-1, 1})
            assert set(np.unique(test_predictions)).issubset({-1, 1})

            # Property 2: Score distributions should be somewhat similar
            # (allowing for natural variation between train/test)
            train_mean = np.mean(train_scores)
            test_mean = np.mean(test_scores)

            # Means shouldn't be drastically different
            assert abs(train_mean - test_mean) < 2.0

            # Property 3: Anomaly rates should be reasonable on both sets
            train_anomaly_rate = np.sum(train_predictions == -1) / len(
                train_predictions
            )
            test_anomaly_rate = np.sum(test_predictions == -1) / len(test_predictions)

            # Both should be within reasonable bounds
            assert 0.0 <= train_anomaly_rate <= 0.5
            assert 0.0 <= test_anomaly_rate <= 0.5

    @given(
        n_samples=st.integers(20, 200),
        n_features=st.integers(1, 10),
        random_state=st.integers(0, 1000),
    )
    @settings(max_examples=20, deadline=5000)
    def test_reproducibility_properties(self, n_samples, n_features, random_state):
        """Test algorithm reproducibility with fixed random state."""
        from sklearn.ensemble import IsolationForest

        # Generate deterministic data
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features)

        contamination = 0.1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Run algorithm twice with same random state
            detector1 = IsolationForest(
                contamination=contamination, random_state=random_state, n_estimators=30
            )
            detector1.fit(data)
            scores1 = detector1.decision_function(data)
            predictions1 = detector1.predict(data)

            detector2 = IsolationForest(
                contamination=contamination, random_state=random_state, n_estimators=30
            )
            detector2.fit(data)
            scores2 = detector2.decision_function(data)
            predictions2 = detector2.predict(data)

            # Property 1: Results should be identical with same random state
            np.testing.assert_array_equal(predictions1, predictions2)
            np.testing.assert_allclose(scores1, scores2, rtol=1e-10)

            # Property 2: Results should differ with different random state
            detector3 = IsolationForest(
                contamination=contamination,
                random_state=random_state + 1,
                n_estimators=30,
            )
            detector3.fit(data)
            scores3 = detector3.decision_function(data)

            # Should be different (with very high probability)
            assert not np.allclose(scores1, scores3, rtol=1e-5)


class TestDomainSpecificProperties:
    """Property-based tests for domain-specific anomaly detection scenarios."""

    @given(
        time_series_length=st.integers(50, 200),
        anomaly_positions=st.lists(
            st.integers(10, 40), min_size=1, max_size=5, unique=True
        ),
    )
    @settings(max_examples=15, deadline=10000)
    def test_time_series_properties(self, time_series_length, anomaly_positions):
        """Test properties on time series data."""
        from sklearn.ensemble import IsolationForest

        # Generate time series with known anomalies
        time_series = np.sin(np.linspace(0, 4 * np.pi, time_series_length))

        # Add anomalies at specified positions
        for pos in anomaly_positions:
            if pos < len(time_series):
                time_series[pos] += np.random.uniform(2, 5)  # Add spike

        # Convert to features (sliding window approach)
        window_size = 5
        assume(time_series_length > window_size)

        features = []
        for i in range(len(time_series) - window_size + 1):
            features.append(time_series[i : i + window_size])

        features = np.array(features)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            detector = IsolationForest(
                contamination=0.2, random_state=42, n_estimators=50
            )
            detector.fit(features)
            predictions = detector.predict(features)
            scores = detector.decision_function(features)

            # Property 1: Should detect some anomalies
            anomaly_count = np.sum(predictions == -1)
            assert anomaly_count > 0

            # Property 2: Anomalies should have lower scores
            if anomaly_count > 0 and anomaly_count < len(predictions):
                normal_scores = scores[predictions == 1]
                anomaly_scores = scores[predictions == -1]
                assert np.mean(anomaly_scores) < np.mean(normal_scores)

    @given(
        n_clusters=st.integers(2, 5),
        cluster_size=st.integers(20, 50),
        outlier_fraction=st.floats(0.05, 0.2),
    )
    @settings(max_examples=10, deadline=15000)
    def test_clustered_data_properties(
        self, n_clusters, cluster_size, outlier_fraction
    ):
        """Test properties on clustered data with outliers."""
        from sklearn.datasets import make_blobs
        from sklearn.ensemble import IsolationForest

        # Generate clustered data
        n_samples = n_clusters * cluster_size
        X, _ = make_blobs(
            n_samples=n_samples, centers=n_clusters, cluster_std=1.0, random_state=42
        )

        # Add outliers
        n_outliers = int(n_samples * outlier_fraction)
        outliers = np.random.uniform(-10, 10, size=(n_outliers, X.shape[1]))

        # Combine data
        data = np.vstack([X, outliers])
        true_labels = np.concatenate([np.ones(n_samples), -np.ones(n_outliers)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            detector = IsolationForest(
                contamination=outlier_fraction, random_state=42, n_estimators=50
            )
            detector.fit(data)
            predictions = detector.predict(data)
            scores = detector.decision_function(data)

            # Property 1: Should identify reasonable number of anomalies
            detected_anomalies = np.sum(predictions == -1)
            expected_anomalies = int(len(data) * outlier_fraction)

            # Allow some tolerance
            assert abs(detected_anomalies - expected_anomalies) <= len(data) * 0.1

            # Property 2: True outliers should tend to have lower scores
            true_outlier_indices = np.where(true_labels == -1)[0]
            true_normal_indices = np.where(true_labels == 1)[0]

            if len(true_outlier_indices) > 0 and len(true_normal_indices) > 0:
                outlier_scores = scores[true_outlier_indices]
                normal_scores = scores[true_normal_indices]

                # Most outliers should have lower scores than median normal score
                median_normal_score = np.median(normal_scores)
                low_score_outliers = np.sum(outlier_scores < median_normal_score)

                assert low_score_outliers > len(outlier_scores) * 0.3

    @given(data_dimension=st.integers(2, 15), n_samples=st.integers(50, 200))
    @settings(max_examples=10, deadline=10000)
    def test_high_dimensional_properties(self, data_dimension, n_samples):
        """Test properties in high-dimensional spaces."""
        from sklearn.ensemble import IsolationForest

        # Generate high-dimensional data
        data = np.random.multivariate_normal(
            mean=np.zeros(data_dimension), cov=np.eye(data_dimension), size=n_samples
        )

        # Add some outliers in random dimensions
        n_outliers = max(1, n_samples // 20)
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)

        for idx in outlier_indices:
            # Make outlier extreme in random dimensions
            extreme_dims = np.random.choice(
                data_dimension, size=max(1, data_dimension // 3), replace=False
            )
            data[idx, extreme_dims] += np.random.uniform(5, 10, size=len(extreme_dims))

        contamination = n_outliers / n_samples

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,  # More trees for high-dimensional data
            )
            detector.fit(data)
            predictions = detector.predict(data)
            scores = detector.decision_function(data)

            # Property 1: Should handle high-dimensional data without errors
            assert len(predictions) == n_samples
            assert len(scores) == n_samples
            assert np.all(np.isfinite(scores))

            # Property 2: Should detect some anomalies
            detected_anomalies = np.sum(predictions == -1)
            assert detected_anomalies > 0

            # Property 3: In high dimensions, algorithm should still be somewhat effective
            # Check that detected anomalies include some true outliers
            detected_outlier_indices = np.where(predictions == -1)[0]
            true_outlier_detection = len(
                set(outlier_indices) & set(detected_outlier_indices)
            )

            # Should detect at least some true outliers
            assert true_outlier_detection > 0

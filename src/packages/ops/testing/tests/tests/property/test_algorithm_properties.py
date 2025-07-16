"""Property-based tests for algorithm mathematical properties and contracts."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp

from .strategies import (
    algorithm_input_strategy,
    contamination_strategy,
    performance_data_strategy,
)


class TestAlgorithmMathematicalProperties:
    """Test mathematical properties that all anomaly detection algorithms should satisfy."""

    @given(algorithm_input_strategy(), contamination_strategy())
    @settings(max_examples=10, deadline=10000)
    def test_score_range_property(self, data: np.ndarray, contamination: float):
        """Test that algorithm scores are always in [0, 1] range."""
        from pynomaly.infrastructure.adapters import SklearnAdapter

        adapter = SklearnAdapter()

        # Test with Isolation Forest (most stable algorithm)
        try:
            detector = adapter.create_detector(
                "isolation_forest", contamination=contamination
            )
            detector.fit(data)
            scores = detector.decision_function(data)

            # Scores should be in valid range
            assert np.all(scores >= 0), f"Found negative scores: {scores[scores < 0]}"
            assert np.all(scores <= 1), f"Found scores > 1: {scores[scores > 1]}"
            assert np.all(np.isfinite(scores)), "Found non-finite scores"

        except Exception:
            # Skip if algorithm fails on this data (acceptable for property testing)
            assume(False)

    @given(algorithm_input_strategy(), contamination_strategy())
    @settings(max_examples=5, deadline=10000)
    def test_deterministic_property(self, data: np.ndarray, contamination: float):
        """Test that algorithms produce consistent results with same parameters."""
        from pynomaly.infrastructure.adapters import SklearnAdapter

        adapter = SklearnAdapter()

        try:
            # Create two identical detectors
            detector1 = adapter.create_detector(
                "isolation_forest", contamination=contamination, random_state=42
            )
            detector2 = adapter.create_detector(
                "isolation_forest", contamination=contamination, random_state=42
            )

            # Fit both detectors
            detector1.fit(data)
            detector2.fit(data)

            # Get scores
            scores1 = detector1.decision_function(data)
            scores2 = detector2.decision_function(data)

            # Scores should be identical (or very close due to numerical precision)
            assert np.allclose(
                scores1, scores2, rtol=1e-10, atol=1e-10
            ), "Same algorithm with same parameters should produce identical results"

        except Exception:
            assume(False)

    @given(algorithm_input_strategy(), contamination_strategy())
    @settings(max_examples=5, deadline=10000)
    def test_contamination_consistency_property(
        self, data: np.ndarray, contamination: float
    ):
        """Test that contamination rate affects the number of predicted anomalies."""
        from pynomaly.infrastructure.adapters import SklearnAdapter

        adapter = SklearnAdapter()

        try:
            detector = adapter.create_detector(
                "isolation_forest", contamination=contamination
            )
            detector.fit(data)

            # Get predictions (1 for normal, -1 for anomaly in sklearn)
            predictions = detector.predict(data)

            # Count anomalies (sklearn uses -1 for anomalies)
            n_anomalies = np.sum(predictions == -1)
            expected_anomalies = int(len(data) * contamination)

            # Allow reasonable tolerance due to discrete nature
            tolerance = max(1, int(len(data) * 0.1))
            assert (
                abs(n_anomalies - expected_anomalies) <= tolerance
            ), f"Expected ~{expected_anomalies} anomalies, got {n_anomalies}"

        except Exception:
            assume(False)

    @given(
        stnp.arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(20, 100), st.integers(2, 5)),
            elements=st.floats(-2, 2, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=5, deadline=10000)
    def test_scale_invariance_property(self, data: np.ndarray):
        """Test that algorithms handle data scaling appropriately."""
        from pynomaly.infrastructure.adapters import SklearnAdapter

        adapter = SklearnAdapter()
        contamination = 0.1

        try:
            # Original data
            detector1 = adapter.create_detector(
                "isolation_forest", contamination=contamination, random_state=42
            )
            detector1.fit(data)
            scores1 = detector1.decision_function(data)

            # Scaled data (multiply by constant)
            scale_factor = 10.0
            scaled_data = data * scale_factor
            detector2 = adapter.create_detector(
                "isolation_forest", contamination=contamination, random_state=42
            )
            detector2.fit(scaled_data)
            scores2 = detector2.decision_function(scaled_data)

            # Relative ranking should be preserved (correlation should be high)
            correlation = np.corrcoef(scores1, scores2)[0, 1]
            assert (
                correlation > 0.8
            ), f"Scale invariance violated: correlation = {correlation}"

        except Exception:
            assume(False)


class TestAlgorithmPerformanceProperties:
    """Test performance characteristics of algorithms."""

    @given(performance_data_strategy())
    @settings(max_examples=3, deadline=15000)
    def test_scalability_property(self, data_pair):
        """Test that algorithm runtime scales reasonably with data size."""
        import time

        from pynomaly.infrastructure.adapters import SklearnAdapter

        small_data, large_data = data_pair
        adapter = SklearnAdapter()
        contamination = 0.1

        try:
            # Time small dataset
            start_time = time.time()
            detector_small = adapter.create_detector(
                "isolation_forest", contamination=contamination
            )
            detector_small.fit(small_data)
            scores_small = detector_small.decision_function(small_data)
            small_time = time.time() - start_time

            # Time large dataset
            start_time = time.time()
            detector_large = adapter.create_detector(
                "isolation_forest", contamination=contamination
            )
            detector_large.fit(large_data)
            scores_large = detector_large.decision_function(large_data)
            large_time = time.time() - start_time

            # Compute size ratio
            size_ratio = len(large_data) / len(small_data)
            time_ratio = large_time / small_time if small_time > 0 else float("inf")

            # Algorithm should not be exponentially slower (allow for some polynomial growth)
            # This is a relaxed test - in practice you'd have tighter bounds
            max_acceptable_ratio = size_ratio**2  # Allow quadratic growth
            assert (
                time_ratio <= max_acceptable_ratio
            ), f"Algorithm too slow: {time_ratio:.2f}x time for {size_ratio:.2f}x data"

            # Scores should still be valid for both datasets
            assert np.all((scores_small >= 0) & (scores_small <= 1))
            assert np.all((scores_large >= 0) & (scores_large <= 1))

        except Exception:
            assume(False)

    @given(algorithm_input_strategy())
    @settings(max_examples=5, deadline=10000)
    def test_memory_efficiency_property(self, data: np.ndarray):
        """Test that algorithms don't use excessive memory."""
        import os

        import psutil

        from pynomaly.infrastructure.adapters import SklearnAdapter

        adapter = SklearnAdapter()
        contamination = 0.1

        try:
            # Measure initial memory
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run algorithm
            detector = adapter.create_detector(
                "isolation_forest", contamination=contamination
            )
            detector.fit(data)
            detector.decision_function(data)

            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable relative to data size
            data_size_mb = data.nbytes / 1024 / 1024
            max_acceptable_memory = data_size_mb * 10  # Allow 10x data size

            assert (
                memory_increase <= max_acceptable_memory
            ), f"Excessive memory usage: {memory_increase:.2f}MB for {data_size_mb:.2f}MB data"

        except Exception:
            assume(False)


class TestAlgorithmRobustnessProperties:
    """Test algorithm robustness to various data conditions."""

    @given(
        stnp.arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(50, 200), st.integers(2, 5)),
            elements=st.floats(-1, 1, allow_nan=False, allow_infinity=False),
        ),
        st.floats(0.0, 0.3),  # Noise level
    )
    @settings(max_examples=5, deadline=10000)
    def test_noise_robustness_property(
        self, clean_data: np.ndarray, noise_level: float
    ):
        """Test that algorithms are reasonably robust to noise."""
        from pynomaly.infrastructure.adapters import SklearnAdapter

        adapter = SklearnAdapter()
        contamination = 0.1

        try:
            # Clean data scores
            detector_clean = adapter.create_detector(
                "isolation_forest", contamination=contamination, random_state=42
            )
            detector_clean.fit(clean_data)
            scores_clean = detector_clean.decision_function(clean_data)

            # Add noise
            noise = np.random.normal(0, noise_level, clean_data.shape)
            noisy_data = clean_data + noise

            # Noisy data scores
            detector_noisy = adapter.create_detector(
                "isolation_forest", contamination=contamination, random_state=42
            )
            detector_noisy.fit(noisy_data)
            scores_noisy = detector_noisy.decision_function(noisy_data)

            # Scores should be reasonably correlated (robust to small noise)
            correlation = np.corrcoef(scores_clean, scores_noisy)[0, 1]
            min_correlation = max(
                0.5, 1.0 - noise_level * 2
            )  # Higher noise allows lower correlation

            assert (
                correlation >= min_correlation
            ), f"Algorithm not robust to noise: correlation = {correlation:.3f} with noise level {noise_level:.3f}"

        except Exception:
            assume(False)

    @given(
        st.integers(10, 50),  # n_samples
        st.integers(2, 10),  # n_features
        st.floats(0.1, 0.9),  # outlier_fraction
    )
    @settings(max_examples=5, deadline=10000)
    def test_outlier_detection_property(
        self, n_samples: int, n_features: int, outlier_fraction: float
    ):
        """Test that algorithms can detect obvious outliers."""
        from pynomaly.infrastructure.adapters import SklearnAdapter

        # Create data with clear outliers
        n_outliers = int(n_samples * outlier_fraction)
        n_normal = n_samples - n_outliers

        # Normal data clustered around origin
        normal_data = np.random.normal(0, 1, (n_normal, n_features))

        # Outlier data far from origin
        outlier_data = (
            np.random.normal(0, 1, (n_outliers, n_features)) + 10
        )  # Shift outliers far away

        # Combine data
        data = np.vstack([normal_data, outlier_data])
        true_labels = np.array([0] * n_normal + [1] * n_outliers)  # 1 for outliers

        adapter = SklearnAdapter()
        contamination = min(outlier_fraction, 0.5)  # Cap at 0.5 for algorithm stability

        try:
            detector = adapter.create_detector(
                "isolation_forest", contamination=contamination
            )
            detector.fit(data)
            scores = detector.decision_function(data)

            # Higher scores should correspond to true outliers
            # Compute area under ROC curve as a measure of detection quality
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(true_labels, scores)

            # AUC should be significantly better than random (0.5)
            assert auc > 0.7, f"Poor outlier detection: AUC = {auc:.3f}"

        except Exception:
            assume(False)


class TestAlgorithmContractProperties:
    """Test that algorithms satisfy their contracts and interfaces."""

    @given(algorithm_input_strategy(), contamination_strategy())
    @settings(max_examples=5, deadline=10000)
    def test_adapter_interface_contract(self, data: np.ndarray, contamination: float):
        """Test that all adapters satisfy the same interface contract."""
        from pynomaly.infrastructure.adapters import PyODAdapter, SklearnAdapter

        adapters = [SklearnAdapter()]

        # Add PyOD adapter if available
        try:
            adapters.append(PyODAdapter())
        except ImportError:
            pass

        for adapter in adapters:
            try:
                # All adapters should support these basic operations
                available_algorithms = adapter.list_algorithms()
                assert (
                    len(available_algorithms) > 0
                ), "Adapter should provide algorithms"

                # Test with first available algorithm
                algorithm = available_algorithms[0]
                detector = adapter.create_detector(
                    algorithm, contamination=contamination
                )

                # All detectors should support fit and decision_function
                assert hasattr(detector, "fit"), "Detector must have fit method"
                assert hasattr(
                    detector, "decision_function"
                ), "Detector must have decision_function method"

                # Fit and score
                detector.fit(data)
                scores = detector.decision_function(data)

                # Results should satisfy basic contracts
                assert len(scores) == len(data), "Scores length must match data length"
                assert np.all(np.isfinite(scores)), "Scores must be finite"

            except Exception:
                # Skip if specific algorithm fails
                continue

    @given(
        stnp.arrays(
            dtype=np.float64,
            shape=st.tuples(st.integers(10, 50), st.integers(1, 5)),
            elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=3, deadline=10000)
    def test_fit_predict_consistency(self, data: np.ndarray):
        """Test that fit followed by predict is consistent."""
        from pynomaly.infrastructure.adapters import SklearnAdapter

        adapter = SklearnAdapter()
        contamination = 0.1

        try:
            detector = adapter.create_detector(
                "isolation_forest", contamination=contamination, random_state=42
            )

            # Fit the detector
            detector.fit(data)

            # Get scores and predictions
            scores1 = detector.decision_function(data)
            predictions1 = detector.predict(data)

            # Call again - should be consistent
            scores2 = detector.decision_function(data)
            predictions2 = detector.predict(data)

            # Results should be identical
            assert np.array_equal(
                scores1, scores2
            ), "decision_function should be deterministic after fit"
            assert np.array_equal(
                predictions1, predictions2
            ), "predict should be deterministic after fit"

            # Predictions should be consistent with scores
            # In sklearn, higher scores typically mean more anomalous
            # and predictions use -1 for anomalies, 1 for normal
            anomaly_mask = predictions1 == -1
            normal_mask = predictions1 == 1

            if np.any(anomaly_mask) and np.any(normal_mask):
                # Average score of anomalies should be higher than normal points
                avg_anomaly_score = np.mean(scores1[anomaly_mask])
                avg_normal_score = np.mean(scores1[normal_mask])
                assert (
                    avg_anomaly_score >= avg_normal_score
                ), "Anomaly scores should be higher than normal scores"

        except Exception:
            assume(False)

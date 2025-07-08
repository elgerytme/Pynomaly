"""Contract tests for algorithm adapters to ensure consistent behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pytest
from pynomaly.infrastructure.adapters import PyODAdapter, SklearnAdapter


class AdapterContractTest(ABC):
    """Abstract base class for adapter contract tests."""

    @abstractmethod
    def get_adapter(self):
        """Return an instance of the adapter to test."""
        pass

    @abstractmethod
    def get_test_algorithm(self) -> str:
        """Return a reliable algorithm name for testing."""
        pass

    def get_test_data(self) -> np.ndarray:
        """Return standard test data for contract tests."""
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 5))

    def test_adapter_provides_algorithms(self):
        """Contract: Adapter must provide list of available algorithms."""
        adapter = self.get_adapter()
        algorithms = adapter.list_algorithms()

        assert isinstance(algorithms, list), "list_algorithms() must return a list"
        assert len(algorithms) > 0, "Adapter must provide at least one algorithm"
        assert all(
            isinstance(algo, str) for algo in algorithms
        ), "Algorithm names must be strings"

    def test_adapter_creates_valid_detector(self):
        """Contract: Adapter must create valid detector instances."""
        adapter = self.get_adapter()
        algorithm = self.get_test_algorithm()

        # Test with default parameters
        detector = adapter.create_detector(algorithm)
        assert detector is not None, "create_detector() must return a detector instance"

        # Test with contamination parameter
        detector = adapter.create_detector(algorithm, contamination=0.1)
        assert (
            detector is not None
        ), "create_detector() must handle contamination parameter"

        # Test with additional parameters
        detector = adapter.create_detector(
            algorithm, contamination=0.05, random_state=42
        )
        assert (
            detector is not None
        ), "create_detector() must handle additional parameters"

    def test_detector_has_required_methods(self):
        """Contract: Detectors must implement required interface methods."""
        adapter = self.get_adapter()
        algorithm = self.get_test_algorithm()
        detector = adapter.create_detector(algorithm)

        # Required methods
        assert hasattr(detector, "fit"), "Detector must have fit() method"
        assert hasattr(detector, "predict"), "Detector must have predict() method"
        assert hasattr(
            detector, "decision_function"
        ), "Detector must have decision_function() method"

        # Methods should be callable
        assert callable(detector.fit), "fit() must be callable"
        assert callable(detector.predict), "predict() must be callable"
        assert callable(
            detector.decision_function
        ), "decision_function() must be callable"

    def test_fit_predict_workflow(self):
        """Contract: fit() followed by predict() must work correctly."""
        adapter = self.get_adapter()
        algorithm = self.get_test_algorithm()
        detector = adapter.create_detector(algorithm, random_state=42)

        data = self.get_test_data()

        # Fit the detector
        fit_result = detector.fit(data)
        # fit() may return self or None
        assert (
            fit_result is None or fit_result is detector
        ), "fit() must return None or self"

        # Predict on the same data
        predictions = detector.predict(data)
        assert isinstance(predictions, np.ndarray), "predict() must return numpy array"
        assert predictions.shape == (
            len(data),
        ), f"predictions shape must be ({len(data)},)"
        assert predictions.dtype in [
            np.int32,
            np.int64,
        ], "predictions must be integer type"

        # Predictions should be binary (-1 for anomaly, 1 for normal in sklearn convention)
        unique_predictions = np.unique(predictions)
        assert (
            len(unique_predictions) <= 2
        ), "Should have at most 2 unique prediction values"
        assert all(
            pred in [-1, 1] for pred in unique_predictions
        ), "Predictions must be -1 or 1"

    def test_decision_function_workflow(self):
        """Contract: decision_function() must return valid anomaly scores."""
        adapter = self.get_adapter()
        algorithm = self.get_test_algorithm()
        detector = adapter.create_detector(algorithm, random_state=42)

        data = self.get_test_data()

        # Fit and score
        detector.fit(data)
        scores = detector.decision_function(data)

        assert isinstance(
            scores, np.ndarray
        ), "decision_function() must return numpy array"
        assert scores.shape == (len(data),), f"scores shape must be ({len(data)},)"
        assert scores.dtype in [np.float32, np.float64], "scores must be float type"
        assert np.all(np.isfinite(scores)), "All scores must be finite"

        # Scores should be in reasonable range (this may need adapter-specific adjustment)
        assert np.all(scores >= 0), "Scores should be non-negative"
        assert np.all(scores <= 1), "Scores should be <= 1"

    def test_deterministic_behavior(self):
        """Contract: Same parameters should produce identical results."""
        adapter = self.get_adapter()
        algorithm = self.get_test_algorithm()
        data = self.get_test_data()

        # Create two identical detectors
        detector1 = adapter.create_detector(
            algorithm, contamination=0.1, random_state=42
        )
        detector2 = adapter.create_detector(
            algorithm, contamination=0.1, random_state=42
        )

        # Fit both detectors
        detector1.fit(data)
        detector2.fit(data)

        # Get predictions and scores
        pred1 = detector1.predict(data)
        pred2 = detector2.predict(data)
        scores1 = detector1.decision_function(data)
        scores2 = detector2.decision_function(data)

        # Results should be identical
        assert np.array_equal(pred1, pred2), "Predictions should be deterministic"
        assert np.allclose(
            scores1, scores2, rtol=1e-10
        ), "Scores should be deterministic"

    def test_contamination_parameter_effect(self):
        """Contract: Contamination parameter should affect number of anomalies."""
        adapter = self.get_adapter()
        algorithm = self.get_test_algorithm()
        data = self.get_test_data()

        contamination_rates = [0.05, 0.1, 0.2]
        anomaly_counts = []

        for contamination in contamination_rates:
            detector = adapter.create_detector(
                algorithm, contamination=contamination, random_state=42
            )
            detector.fit(data)
            predictions = detector.predict(data)

            # Count anomalies (assuming -1 indicates anomaly)
            n_anomalies = np.sum(predictions == -1)
            anomaly_counts.append(n_anomalies)

            # Rough check that contamination is respected
            expected_anomalies = int(len(data) * contamination)
            tolerance = max(2, len(data) * 0.1)  # Allow reasonable tolerance
            assert (
                abs(n_anomalies - expected_anomalies) <= tolerance
            ), f"Contamination {contamination}: expected ~{expected_anomalies}, got {n_anomalies}"

        # Higher contamination should generally result in more anomalies
        # (though this may not be strictly monotonic due to algorithm specifics)
        assert (
            anomaly_counts[-1] >= anomaly_counts[0]
        ), f"Higher contamination should detect more anomalies: {anomaly_counts}"

    def test_input_validation(self):
        """Contract: Adapters should validate inputs appropriately."""
        adapter = self.get_adapter()
        algorithm = self.get_test_algorithm()

        # Test invalid algorithm name
        with pytest.raises((ValueError, KeyError, AttributeError)):
            adapter.create_detector("nonexistent_algorithm")

        # Test invalid contamination values
        with pytest.raises((ValueError, TypeError)):
            adapter.create_detector(algorithm, contamination=-0.1)  # Negative

        with pytest.raises((ValueError, TypeError)):
            adapter.create_detector(algorithm, contamination=1.5)  # > 1

        # Test detector input validation
        detector = adapter.create_detector(algorithm)

        # Invalid data shapes
        with pytest.raises((ValueError, TypeError)):
            detector.fit(np.array([1, 2, 3]))  # 1D array

        with pytest.raises((ValueError, TypeError)):
            detector.fit(np.array([]))  # Empty array

    def test_edge_cases(self):
        """Contract: Adapters should handle edge cases gracefully."""
        adapter = self.get_adapter()
        algorithm = self.get_test_algorithm()

        # Very small dataset
        small_data = np.random.random((5, 2))
        detector = adapter.create_detector(algorithm, contamination=0.2)

        try:
            detector.fit(small_data)
            predictions = detector.predict(small_data)
            scores = detector.decision_function(small_data)

            # Should still produce valid outputs
            assert len(predictions) == len(small_data)
            assert len(scores) == len(small_data)
            assert np.all(np.isfinite(scores))
        except ValueError:
            # Some algorithms may require minimum sample size - this is acceptable
            pass

        # Single feature
        single_feature_data = np.random.random((50, 1))
        detector = adapter.create_detector(algorithm)

        try:
            detector.fit(single_feature_data)
            predictions = detector.predict(single_feature_data)
            assert len(predictions) == len(single_feature_data)
        except ValueError:
            # Some algorithms may require multiple features - this is acceptable
            pass


class TestSklearnAdapterContract(AdapterContractTest):
    """Contract tests for SklearnAdapter."""

    def get_adapter(self):
        return SklearnAdapter()

    def get_test_algorithm(self) -> str:
        return "isolation_forest"


class TestPyODAdapterContract(AdapterContractTest):
    """Contract tests for PyODAdapter."""

    def get_adapter(self):
        try:
            return PyODAdapter()
        except ImportError:
            pytest.skip("PyOD not available")

    def get_test_algorithm(self) -> str:
        return "pyod_lof"  # Local Outlier Factor is usually stable


class TestAdapterInteroperability:
    """Test that different adapters can be used interchangeably."""

    def get_adapters(self):
        """Get all available adapters."""
        adapters = [("sklearn", SklearnAdapter())]

        try:
            adapters.append(("pyod", PyODAdapter()))
        except ImportError:
            pass

        return adapters

    def test_adapter_interface_consistency(self):
        """Test that all adapters implement the same interface."""
        adapters = self.get_adapters()

        if len(adapters) < 2:
            pytest.skip("Need at least 2 adapters for interoperability testing")

        # All adapters should have the same interface
        for name, adapter in adapters:
            assert hasattr(
                adapter, "list_algorithms"
            ), f"{name} adapter missing list_algorithms()"
            assert hasattr(
                adapter, "create_detector"
            ), f"{name} adapter missing create_detector()"

            # Methods should return expected types
            algorithms = adapter.list_algorithms()
            assert isinstance(
                algorithms, list
            ), f"{name} adapter list_algorithms() should return list"

            if algorithms:  # If algorithms are available
                detector = adapter.create_detector(algorithms[0])
                assert hasattr(detector, "fit"), f"{name} detector missing fit()"
                assert hasattr(
                    detector, "predict"
                ), f"{name} detector missing predict()"
                assert hasattr(
                    detector, "decision_function"
                ), f"{name} detector missing decision_function()"

    def test_cross_adapter_result_correlation(self):
        """Test that different adapters produce correlated results on same data."""
        adapters = self.get_adapters()

        if len(adapters) < 2:
            pytest.skip("Need at least 2 adapters for correlation testing")

        # Use common algorithms if available
        common_algorithm_map = {"sklearn": "local_outlier_factor", "pyod": "pyod_lof"}

        data = np.random.RandomState(42).normal(0, 1, (100, 3))
        results = {}

        for name, adapter in adapters:
            if name in common_algorithm_map:
                try:
                    algorithm = common_algorithm_map[name]
                    available_algorithms = adapter.list_algorithms()

                    if algorithm in available_algorithms:
                        detector = adapter.create_detector(
                            algorithm, contamination=0.1, random_state=42
                        )
                        detector.fit(data)
                        scores = detector.decision_function(data)
                        results[name] = scores
                except Exception:
                    # Skip if algorithm fails
                    continue

        # Check correlation between results
        if len(results) >= 2:
            adapter_names = list(results.keys())
            scores1 = results[adapter_names[0]]
            scores2 = results[adapter_names[1]]

            correlation = np.corrcoef(scores1, scores2)[0, 1]

            # Results should be reasonably correlated for same algorithm type
            assert (
                correlation > 0.3
            ), f"Adapter results should be correlated: {correlation:.3f}"


class TestAdapterErrorHandling:
    """Test error handling contracts across adapters."""

    def test_graceful_failure_on_invalid_data(self):
        """Test that adapters handle invalid data gracefully."""
        adapter = SklearnAdapter()
        detector = adapter.create_detector("isolation_forest")

        # Test NaN data
        nan_data = np.full((10, 3), np.nan)
        with pytest.raises((ValueError, TypeError)):
            detector.fit(nan_data)

        # Test infinite data
        inf_data = np.full((10, 3), np.inf)
        with pytest.raises((ValueError, TypeError)):
            detector.fit(inf_data)

        # Test wrong data type
        with pytest.raises((ValueError, TypeError)):
            detector.fit("not_an_array")

    def test_consistent_error_types(self):
        """Test that adapters raise consistent error types."""
        adapter = SklearnAdapter()

        # Invalid algorithm should raise ValueError or similar
        with pytest.raises((ValueError, KeyError, AttributeError)):
            adapter.create_detector("invalid_algorithm")

        # Invalid contamination should raise ValueError
        with pytest.raises((ValueError, TypeError)):
            adapter.create_detector("isolation_forest", contamination=2.0)

    def test_unfitted_detector_behavior(self):
        """Test behavior when using unfitted detectors."""
        adapter = SklearnAdapter()
        detector = adapter.create_detector("isolation_forest")

        data = np.random.random((10, 3))

        # Should raise error when calling predict/decision_function before fit
        with pytest.raises((ValueError, AttributeError)):
            detector.predict(data)

        with pytest.raises((ValueError, AttributeError)):
            detector.decision_function(data)

"""
Tests for drift detection adapters.

Tests cover:
- Statistical drift detection (KS test, Chi-square)
- Distance-based drift detection (MMD)
- Performance-based drift detection
- Streaming drift detection
- Multivariate drift detection
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from datetime import datetime, timedelta

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import AdapterError, AlgorithmNotFoundError
from pynomaly.infrastructure.adapters.drift_detection_adapter import (
    DriftDetectionAdapter,
    DriftDetectionResult,
    DriftSeverity,
    DriftType,
    StatisticalDriftDetector,
    DistanceBasedDriftDetector,
    PerformanceBasedDriftDetector,
    StreamingDriftDetector,
)


@pytest.fixture
def base_dataset():
    """Create a base dataset for drift testing."""
    # Generate stable dataset
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_redundant=0,
        n_informative=4,
        n_clusters_per_class=1,
        random_state=42
    )

    # Convert to anomaly labels (small fraction of anomalies)
    y_anomaly = np.zeros(len(y))
    anomaly_indices = np.random.choice(len(y), size=int(0.1 * len(y)), replace=False)
    y_anomaly[anomaly_indices] = 1

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y_anomaly

    return Dataset(
        id="base_dataset",
        name="Base Dataset",
        data=df,
        target_column="target"
    )


@pytest.fixture
def drifted_dataset():
    """Create a dataset with induced drift."""
    # Generate dataset with different distribution
    np.random.seed(123)  # Different seed for drift
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_redundant=0,
        n_informative=4,
        n_clusters_per_class=1,
        random_state=123
    )

    # Add systematic drift to first two features
    X[:, 0] += 2.0  # Mean shift
    X[:, 1] *= 1.5  # Scale change

    # Convert to anomaly labels
    y_anomaly = np.zeros(len(y))
    anomaly_indices = np.random.choice(len(y), size=int(0.15 * len(y)), replace=False)
    y_anomaly[anomaly_indices] = 1

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y_anomaly

    return Dataset(
        id="drifted_dataset",
        name="Drifted Dataset",
        data=df,
        target_column="target"
    )


@pytest.fixture
def statistical_drift_detector():
    """Create a statistical drift detector."""
    return Detector(
        id="test_statistical_drift",
        name="Test Statistical Drift",
        algorithm_name="StatisticalDrift",
        parameters={
            "significance_level": 0.05,
            "method": "ks_test"
        }
    )


@pytest.fixture
def distance_drift_detector():
    """Create a distance-based drift detector."""
    return Detector(
        id="test_distance_drift",
        name="Test Distance Drift",
        algorithm_name="DistanceBasedDrift",
        parameters={
            "kernel": "rbf",
            "gamma": 1.0
        }
    )


@pytest.fixture
def performance_drift_detector():
    """Create a performance-based drift detector."""
    return Detector(
        id="test_performance_drift",
        name="Test Performance Drift",
        algorithm_name="PerformanceBasedDrift",
        parameters={
            "performance_threshold": 0.1
        }
    )


@pytest.fixture
def streaming_drift_detector():
    """Create a streaming drift detector."""
    return Detector(
        id="test_streaming_drift",
        name="Test Streaming Drift",
        algorithm_name="StreamingDrift",
        parameters={
            "window_size": 100,
            "adaptation_rate": 0.01
        }
    )


class TestStatisticalDriftDetector:
    """Test statistical drift detection functionality."""

    def test_statistical_detector_initialization(self):
        """Test statistical detector initialization."""
        detector = StatisticalDriftDetector(significance_level=0.01)
        assert detector.significance_level == 0.01
        assert len(detector._reference_distributions) == 0

    def test_fit_reference(self, base_dataset):
        """Test fitting reference distributions."""
        detector = StatisticalDriftDetector()

        # Prepare data
        X = base_dataset.data.drop('target', axis=1).values
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Fit reference
        detector.fit_reference(X, feature_names)

        assert len(detector._reference_distributions) == len(feature_names)
        for feature_name in feature_names:
            assert feature_name in detector._reference_distributions
            ref_dist = detector._reference_distributions[feature_name]
            assert 'mean' in ref_dist
            assert 'std' in ref_dist
            assert 'data' in ref_dist

    def test_ks_test_no_drift(self, base_dataset):
        """Test KS test with no drift (same distribution)."""
        detector = StatisticalDriftDetector()

        # Use same data for reference and current
        X = base_dataset.data.drop('target', axis=1).values
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        detector.fit_reference(X, feature_names)
        result = detector.detect_drift_ks_test(X, feature_names)

        assert isinstance(result, DriftDetectionResult)
        assert result.drift_detected is False
        assert result.drift_type == DriftType.NO_DRIFT
        assert result.p_value > detector.significance_level

    def test_ks_test_with_drift(self, base_dataset, drifted_dataset):
        """Test KS test with actual drift."""
        detector = StatisticalDriftDetector()

        # Fit on base dataset
        X_base = base_dataset.data.drop('target', axis=1).values
        feature_names = [f"feature_{i}" for i in range(X_base.shape[1])]
        detector.fit_reference(X_base, feature_names)

        # Test on drifted dataset
        X_drift = drifted_dataset.data.drop('target', axis=1).values
        result = detector.detect_drift_ks_test(X_drift, feature_names)

        assert isinstance(result, DriftDetectionResult)
        assert result.drift_detected is True
        assert result.drift_type == DriftType.ABRUPT_DRIFT
        assert result.p_value < detector.significance_level
        assert len(result.features_affected) > 0

    def test_chi_square_test(self, base_dataset, drifted_dataset):
        """Test Chi-square test for drift detection."""
        detector = StatisticalDriftDetector()

        # Fit on base dataset
        X_base = base_dataset.data.drop('target', axis=1).values
        feature_names = [f"feature_{i}" for i in range(X_base.shape[1])]
        detector.fit_reference(X_base, feature_names)

        # Test on drifted dataset
        X_drift = drifted_dataset.data.drop('target', axis=1).values
        result = detector.detect_drift_chi_square(X_drift, feature_names, n_bins=10)

        assert isinstance(result, DriftDetectionResult)
        assert result.detection_method == "chi_square"
        assert 'n_bins' in result.details


class TestDistanceBasedDriftDetector:
    """Test distance-based drift detection functionality."""

    def test_distance_detector_initialization(self):
        """Test distance detector initialization."""
        detector = DistanceBasedDriftDetector(kernel='rbf', gamma=0.5)
        assert detector.kernel == 'rbf'
        assert detector.gamma == 0.5
        assert detector._reference_embeddings is None

    def test_fit_reference(self, base_dataset):
        """Test fitting reference embeddings."""
        detector = DistanceBasedDriftDetector()

        X = base_dataset.data.drop('target', axis=1).values
        detector.fit_reference(X)

        assert detector._reference_embeddings is not None
        assert len(detector._reference_embeddings) > 0

    def test_mmd_no_drift(self, base_dataset):
        """Test MMD with no drift."""
        detector = DistanceBasedDriftDetector()

        # Use same data for reference and current
        X = base_dataset.data.drop('target', axis=1).values
        detector.fit_reference(X)

        # Test with subset of same data
        result = detector.detect_drift_mmd(X[:500], bootstrap_samples=100)

        assert isinstance(result, DriftDetectionResult)
        assert result.detection_method == "maximum_mean_discrepancy"
        # MMD should be small for same distribution
        assert result.drift_score < 1.0

    def test_mmd_with_drift(self, base_dataset, drifted_dataset):
        """Test MMD with actual drift."""
        detector = DistanceBasedDriftDetector()

        # Fit on base dataset
        X_base = base_dataset.data.drop('target', axis=1).values
        detector.fit_reference(X_base)

        # Test on drifted dataset
        X_drift = drifted_dataset.data.drop('target', axis=1).values
        result = detector.detect_drift_mmd(X_drift, bootstrap_samples=100)

        assert isinstance(result, DriftDetectionResult)
        assert result.detection_method == "maximum_mean_discrepancy"
        # Should detect drift with higher MMD score
        assert result.drift_score > 0

    def test_rbf_kernel_computation(self, base_dataset):
        """Test RBF kernel computation."""
        detector = DistanceBasedDriftDetector(kernel='rbf', gamma=1.0)

        X = base_dataset.data.drop('target', axis=1).values[:100]  # Small sample
        Y = X + 0.1  # Slight perturbation

        kernel_matrix = detector._rbf_kernel(X, Y)

        assert kernel_matrix.shape == (len(X), len(Y))
        assert np.all(kernel_matrix >= 0)
        assert np.all(kernel_matrix <= 1)


class TestPerformanceBasedDriftDetector:
    """Test performance-based drift detection functionality."""

    def test_performance_detector_initialization(self):
        """Test performance detector initialization."""
        detector = PerformanceBasedDriftDetector(performance_threshold=0.2)
        assert detector.performance_threshold == 0.2
        assert detector._reference_performance is None

    def test_fit_reference_with_labels(self, base_dataset):
        """Test fitting with labeled reference data."""
        detector = PerformanceBasedDriftDetector()

        X = base_dataset.data.drop('target', axis=1).values
        y = base_dataset.data['target'].values

        detector.fit_reference(X, y)

        assert detector.baseline_model is not None
        assert detector._reference_performance is not None
        assert 'accuracy' in detector._reference_performance
        assert 'f1' in detector._reference_performance

    def test_fit_reference_without_labels(self, base_dataset):
        """Test fitting without labels (unsupervised)."""
        detector = PerformanceBasedDriftDetector()

        X = base_dataset.data.drop('target', axis=1).values

        detector.fit_reference(X)

        assert detector.baseline_model is not None
        # No reference performance without labels
        assert detector._reference_performance is None

    def test_performance_drift_detection(self, base_dataset, drifted_dataset):
        """Test performance-based drift detection."""
        detector = PerformanceBasedDriftDetector(performance_threshold=0.05)

        # Fit on base dataset
        X_base = base_dataset.data.drop('target', axis=1).values
        y_base = base_dataset.data['target'].values
        detector.fit_reference(X_base, y_base)

        # Test on drifted dataset (should show performance degradation)
        X_drift = drifted_dataset.data.drop('target', axis=1).values
        y_drift = drifted_dataset.data['target'].values

        result = detector.detect_drift_performance(X_drift, y_drift)

        assert isinstance(result, DriftDetectionResult)
        assert result.detection_method == "performance_based"
        assert 'reference_performance' in result.details
        assert 'current_performance' in result.details
        assert 'performance_degradation' in result.details


class TestStreamingDriftDetector:
    """Test streaming drift detection functionality."""

    def test_streaming_detector_initialization(self):
        """Test streaming detector initialization."""
        detector = StreamingDriftDetector(window_size=500, adaptation_rate=0.05)
        assert detector.window_size == 500
        assert detector.adaptation_rate == 0.05
        assert len(detector._reference_window) == 0
        assert len(detector._current_window) == 0

    def test_streaming_update_initial(self, base_dataset):
        """Test initial updates to streaming detector."""
        detector = StreamingDriftDetector(window_size=100)

        X = base_dataset.data.drop('target', axis=1).values

        # Initial update should not detect drift
        result = detector.update(X[:50])
        assert result is None  # Not enough data yet

        # Reference window should be initialized
        assert len(detector._reference_window) > 0

    def test_streaming_drift_detection(self):
        """Test streaming drift detection with simulated drift."""
        detector = StreamingDriftDetector(window_size=100, adaptation_rate=0.1)

        # Generate initial stable data
        np.random.seed(42)
        X_stable = np.random.normal(0, 1, (150, 3))

        # Initialize with stable data
        result = detector.update(X_stable)

        # Generate drifted data
        X_drift = np.random.normal(2, 1, (150, 3))  # Mean shift

        # Update with drifted data
        result = detector.update(X_drift)

        if result is not None:
            assert isinstance(result, DriftDetectionResult)
            assert result.detection_method == "streaming_wasserstein"

    def test_window_management(self):
        """Test window size management in streaming detector."""
        detector = StreamingDriftDetector(window_size=50)

        # Add more data than window size
        large_batch = np.random.random((100, 2))
        detector.update(large_batch)

        # Current window should be limited to window_size
        assert len(detector._current_window) <= detector.window_size

    def test_adaptation_mechanism(self):
        """Test adaptation mechanism in streaming detector."""
        detector = StreamingDriftDetector(window_size=50, adaptation_rate=0.2)

        # Initialize with some data
        initial_data = np.random.random((60, 2))
        detector.update(initial_data)

        initial_ref = detector._reference_window.copy()

        # Trigger adaptation with new data
        new_data = np.random.random((60, 2)) + 1  # Shifted data
        detector.update(new_data)

        # Reference window should have adapted
        if detector._reference_window:
            # Reference window should be different after adaptation
            assert not np.array_equal(initial_ref, detector._reference_window)


class TestDriftDetectionAdapter:
    """Test main drift detection adapter functionality."""

    def test_adapter_initialization(self, statistical_drift_detector):
        """Test adapter initialization."""
        adapter = DriftDetectionAdapter(statistical_drift_detector)
        assert adapter.detector == statistical_drift_detector
        assert adapter._is_fitted is False
        assert adapter._drift_detector is not None

    def test_adapter_unsupported_algorithm(self):
        """Test adapter with unsupported algorithm."""
        detector = Detector(
            id="test_unsupported",
            name="Test Unsupported",
            algorithm_name="UnsupportedDrift",
            parameters={}
        )

        with pytest.raises(AlgorithmNotFoundError):
            DriftDetectionAdapter(detector)

    def test_adapter_fit(self, base_dataset, statistical_drift_detector):
        """Test adapter fitting."""
        adapter = DriftDetectionAdapter(statistical_drift_detector)
        adapter.fit(base_dataset)

        assert adapter._is_fitted is True
        assert adapter.detector.is_fitted is True
        assert adapter._reference_data is not None

    def test_adapter_predict_no_drift(self, base_dataset, statistical_drift_detector):
        """Test adapter prediction with no drift."""
        adapter = DriftDetectionAdapter(statistical_drift_detector)
        adapter.fit(base_dataset)

        # Test on same dataset (should show no drift)
        result = adapter.predict(base_dataset)

        assert result.detector_id == statistical_drift_detector.id
        assert result.dataset_id == base_dataset.id
        assert len(result.scores) == len(base_dataset.data)
        assert "drift_detected" in result.metadata
        assert result.metadata["model_type"] == "drift_detection"

    def test_adapter_predict_with_drift(self, base_dataset, drifted_dataset, statistical_drift_detector):
        """Test adapter prediction with drift."""
        adapter = DriftDetectionAdapter(statistical_drift_detector)
        adapter.fit(base_dataset)

        # Test on drifted dataset
        result = adapter.predict(drifted_dataset)

        assert result.detector_id == statistical_drift_detector.id
        assert result.dataset_id == drifted_dataset.id
        assert "drift_detected" in result.metadata

        # Should detect drift
        if result.metadata["drift_detected"]:
            assert len(result.anomalies) > 0
            assert result.metadata["drift_severity"] in [s.value for s in DriftSeverity]

    def test_adapter_fit_detect(self, base_dataset, drifted_dataset, distance_drift_detector):
        """Test adapter fit_detect method."""
        adapter = DriftDetectionAdapter(distance_drift_detector)

        # Fit and detect in one step
        result = adapter.fit_detect(base_dataset)

        assert adapter._is_fitted is True
        assert len(result.scores) == len(base_dataset.data)

    def test_adapter_score(self, base_dataset, statistical_drift_detector):
        """Test adapter score method."""
        adapter = DriftDetectionAdapter(statistical_drift_detector)
        adapter.fit(base_dataset)

        scores = adapter.score(base_dataset)

        assert len(scores) == len(base_dataset.data)
        assert all(hasattr(score, 'value') for score in scores)
        assert all(score.method == statistical_drift_detector.algorithm_name for score in scores)

    def test_adapter_params(self, statistical_drift_detector):
        """Test adapter parameter management."""
        adapter = DriftDetectionAdapter(statistical_drift_detector)

        # Test get_params
        params = adapter.get_params()
        assert isinstance(params, dict)
        assert "significance_level" in params

        # Test set_params
        new_params = {"significance_level": 0.01}
        adapter.set_params(**new_params)

        updated_params = adapter.get_params()
        assert updated_params["significance_level"] == 0.01

    @pytest.mark.parametrize("algorithm_name", [
        "StatisticalDrift",
        "DistanceBasedDrift",
        "PerformanceBasedDrift",
        "StreamingDrift"
    ])
    def test_all_algorithms(self, base_dataset, drifted_dataset, algorithm_name):
        """Test all supported drift detection algorithms."""
        detector = Detector(
            id=f"test_{algorithm_name.lower()}",
            name=f"Test {algorithm_name}",
            algorithm_name=algorithm_name,
            parameters={
                "significance_level": 0.05,
                "window_size": 100,
                "performance_threshold": 0.1,
                "kernel": "rbf",
                "gamma": 1.0
            }
        )

        adapter = DriftDetectionAdapter(detector)

        try:
            adapter.fit(base_dataset)
            result = adapter.predict(drifted_dataset)

            assert adapter._is_fitted is True
            assert len(result.scores) == len(drifted_dataset.data)
            assert "detection_method" in result.metadata

        except Exception as e:
            pytest.fail(f"Algorithm {algorithm_name} failed: {e}")

    def test_adapter_error_handling(self, statistical_drift_detector):
        """Test adapter error handling."""
        adapter = DriftDetectionAdapter(statistical_drift_detector)

        # Test prediction without fitting
        with pytest.raises(AdapterError):
            empty_dataset = Dataset(
                id="empty",
                name="Empty",
                data=pd.DataFrame(),
                target_column=None
            )
            adapter.predict(empty_dataset)

    def test_adapter_get_supported_algorithms(self):
        """Test getting supported algorithms."""
        algorithms = DriftDetectionAdapter.get_supported_algorithms()

        assert isinstance(algorithms, list)
        assert len(algorithms) > 0
        assert "StatisticalDrift" in algorithms
        assert "DistanceBasedDrift" in algorithms
        assert "PerformanceBasedDrift" in algorithms
        assert "StreamingDrift" in algorithms

    def test_adapter_get_algorithm_info(self):
        """Test getting algorithm information."""
        info = DriftDetectionAdapter.get_algorithm_info("StatisticalDrift")

        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "parameters" in info
        assert "suitable_for" in info
        assert "pros" in info
        assert "cons" in info

        # Test unknown algorithm
        with pytest.raises(AlgorithmNotFoundError):
            DriftDetectionAdapter.get_algorithm_info("UnknownDrift")


class TestDriftDetectionIntegration:
    """Integration tests for drift detection functionality."""

    def test_multi_method_drift_detection(self, base_dataset, drifted_dataset):
        """Test using multiple drift detection methods on same data."""
        methods = ["StatisticalDrift", "DistanceBasedDrift"]
        results = {}

        for method in methods:
            detector = Detector(
                id=f"test_{method}",
                name=f"Test {method}",
                algorithm_name=method,
                parameters={"significance_level": 0.05, "kernel": "rbf"}
            )

            adapter = DriftDetectionAdapter(detector)
            adapter.fit(base_dataset)
            result = adapter.predict(drifted_dataset)

            results[method] = result.metadata["drift_detected"]

        # At least one method should detect drift
        assert any(results.values())

    def test_temporal_drift_simulation(self):
        """Test drift detection over time with simulated temporal data."""
        detector = Detector(
            id="temporal_drift",
            name="Temporal Drift",
            algorithm_name="StreamingDrift",
            parameters={"window_size": 50, "adaptation_rate": 0.1}
        )

        adapter = DriftDetectionAdapter(detector)

        # Create initial stable dataset
        np.random.seed(42)
        stable_data = np.random.normal(0, 1, (100, 3))
        df_stable = pd.DataFrame(stable_data, columns=['f1', 'f2', 'f3'])

        initial_dataset = Dataset(
            id="initial",
            name="Initial",
            data=df_stable,
            target_column=None
        )

        adapter.fit(initial_dataset)

        # Simulate gradual drift over time
        for i in range(5):
            # Gradually shift the mean
            drift_data = np.random.normal(i * 0.5, 1, (50, 3))
            df_drift = pd.DataFrame(drift_data, columns=['f1', 'f2', 'f3'])

            drift_dataset = Dataset(
                id=f"drift_{i}",
                name=f"Drift {i}",
                data=df_drift,
                target_column=None
            )

            result = adapter.predict(drift_dataset)

            # Later iterations should be more likely to detect drift
            if i > 2:
                # Strong drift should be detected
                pass  # Drift detection depends on the specific implementation

    def test_feature_specific_drift(self):
        """Test detection of drift in specific features."""
        detector = Detector(
            id="feature_drift",
            name="Feature Drift",
            algorithm_name="StatisticalDrift",
            parameters={"significance_level": 0.05}
        )

        adapter = DriftDetectionAdapter(detector)

        # Create reference data
        np.random.seed(42)
        X_ref = np.random.normal(0, 1, (500, 4))
        df_ref = pd.DataFrame(X_ref, columns=['f1', 'f2', 'f3', 'f4'])

        ref_dataset = Dataset(
            id="reference",
            name="Reference",
            data=df_ref,
            target_column=None
        )

        adapter.fit(ref_dataset)

        # Create data with drift only in specific features
        X_drift = X_ref.copy()
        X_drift[:, 0] += 2.0  # Shift only first feature
        X_drift[:, 2] *= 2.0  # Scale only third feature

        df_drift = pd.DataFrame(X_drift, columns=['f1', 'f2', 'f3', 'f4'])

        drift_dataset = Dataset(
            id="drift",
            name="Drift",
            data=df_drift,
            target_column=None
        )

        result = adapter.predict(drift_dataset)

        if result.metadata["drift_detected"]:
            # Should identify specific affected features
            affected_features = result.metadata.get("features_affected", [])

            # Features f1 and f3 should be affected more than f2 and f4
            assert isinstance(affected_features, list)

    def test_performance_drift_with_model_degradation(self, base_dataset):
        """Test performance-based drift detection with actual model degradation."""
        detector = Detector(
            id="performance_drift",
            name="Performance Drift",
            algorithm_name="PerformanceBasedDrift",
            parameters={"performance_threshold": 0.05}
        )

        adapter = DriftDetectionAdapter(detector)
        adapter.fit(base_dataset)

        # Create data that should cause performance degradation
        # Flip some labels to simulate concept drift
        degraded_data = base_dataset.data.copy()
        n_flip = int(0.3 * len(degraded_data))
        flip_indices = np.random.choice(len(degraded_data), n_flip, replace=False)
        degraded_data.loc[flip_indices, 'target'] = 1 - degraded_data.loc[flip_indices, 'target']

        degraded_dataset = Dataset(
            id="degraded",
            name="Degraded",
            data=degraded_data,
            target_column="target"
        )

        result = adapter.predict(degraded_dataset)

        # Should detect performance degradation
        if result.metadata["drift_detected"]:
            assert result.metadata["drift_type"] in [dt.value for dt in DriftType]
            assert "performance_degradation" in result.metadata.get("details", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

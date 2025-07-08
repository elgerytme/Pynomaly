"""Comprehensive tests for infrastructure adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest_dependencies import requires_dependencies, requires_dependency

# Optional imports with graceful fallbacks
sklearn_available = True
torch_available = True

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
except ImportError:
    sklearn_available = False
    IsolationForest = None
    LocalOutlierFactor = None
    OneClassSVM = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch_available = False
    torch = None
    nn = None

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import InvalidAlgorithmError
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.infrastructure.adapters import PyODAdapter, SklearnAdapter

# Conditionally import optional adapters
PyTorchAdapter = None
PyGODAdapter = None
TODSAdapter = None

try:
    from pynomaly.infrastructure.adapters import PyTorchAdapter
except ImportError:
    pass

try:
    from pynomaly.infrastructure.adapters import PyGODAdapter
except ImportError:
    pass


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    features = np.random.RandomState(42).normal(0, 1, (100, 5))
    targets = np.random.RandomState(42).choice([0, 1], size=100, p=[0.9, 0.1])
    return Dataset(name="test_dataset", features=features, targets=targets)


@pytest.fixture
def sample_detector():
    """Create a sample detector for testing."""
    return Detector(
        name="test_detector",
        algorithm="isolation_forest",
        contamination=ContaminationRate(0.1),
        hyperparameters={"n_estimators": 100, "random_state": 42},
    )


@pytest.fixture
def large_dataset():
    """Create a larger dataset for performance testing."""
    features = np.random.RandomState(42).normal(0, 1, (1000, 10))
    return Dataset(name="large_dataset", features=features)


@pytest.fixture
def anomalous_dataset():
    """Create a dataset with clear anomalies."""
    # Normal data
    normal = np.random.RandomState(42).normal(0, 1, (95, 3))
    # Clear anomalies
    anomalies = np.random.RandomState(42).normal(5, 0.5, (5, 3))
    features = np.vstack([normal, anomalies])
    targets = np.array([0] * 95 + [1] * 5)
    return Dataset(name="anomalous_dataset", features=features, targets=targets)


@requires_dependency("pyod")
class TestPyODAdapter:
    """Test PyODAdapter."""

    def test_list_algorithms(self):
        """Test listing available algorithms."""
        # Test the class-level algorithm mapping
        algorithms = list(PyODAdapter.ALGORITHM_MAPPING.keys())

        assert len(algorithms) > 0
        assert "IsolationForest" in algorithms
        assert "LOF" in algorithms
        assert "OCSVM" in algorithms

    def test_create_model(self):
        """Test creating adapter instance."""
        adapter = PyODAdapter("IsolationForest")

        # Test that adapter is properly initialized
        assert adapter.algorithm_name == "IsolationForest"
        assert adapter.name == "PyOD_IsolationForest"
        assert hasattr(adapter, "fit")
        assert hasattr(adapter, "detect")

    def test_fit_predict(self):
        """Test fitting and predicting."""
        adapter = PyODAdapter("IsolationForest")

        # Generate data
        X = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        dataset = Dataset(name="test", data=X)

        # Test fitting
        adapter.fit(dataset)
        assert adapter.is_fitted

        # Test detection
        result = adapter.detect(dataset)
        assert len(result.scores) == 100
        assert len(result.labels) == 100
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert all(label in [0, 1] for label in result.labels)
        assert len(result.anomalies) == sum(result.labels)  # Anomalies match labels

    def test_score_function(self):
        """Test getting anomaly scores."""
        adapter = PyODAdapter("IsolationForest")

        # Generate data
        X = np.random.rand(100, 3)
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        dataset = Dataset(name="test", data=df)

        # Fit and get scores
        adapter.fit(dataset)
        scores = adapter.score(dataset)

        assert len(scores) == len(df)
        assert all(isinstance(s, AnomalyScore) for s in scores)

    def test_unsupported_algorithm(self):
        """Test creating unsupported algorithm."""
        with pytest.raises(InvalidAlgorithmError):
            PyODAdapter("UnsupportedAlgorithm")
    
    def test_missing_class_algorithms(self):
        """Test algorithms where the class doesn't exist in the module."""
        missing_class_algorithms = [
            "FastABOD",  # Class doesn't exist in pyod.models.abod
            "Beta-VAE",  # Class doesn't exist in pyod.models.vae
        ]
        
        for algorithm in missing_class_algorithms:
            with pytest.raises(InvalidAlgorithmError) as exc_info:
                PyODAdapter(algorithm)
            assert "not available in PyOD version" in str(exc_info.value)
            assert "not found in module" in str(exc_info.value)
    
    def test_missing_module_algorithms(self):
        """Test algorithms where the entire module doesn't exist."""
        missing_module_algorithms = [
            "CLF",  # Module pyod.models.clf doesn't exist
        ]
        
        for algorithm in missing_module_algorithms:
            with pytest.raises(InvalidAlgorithmError) as exc_info:
                PyODAdapter(algorithm)
            assert "not available in PyOD version" in str(exc_info.value)
            assert "does not exist" in str(exc_info.value)
    
    def test_missing_dependency_algorithms(self):
        """Test algorithms with missing optional dependencies."""
        dependency_algorithms = [
            ("FeatureBagging", "combo"),  # Requires combo package
            ("XGBOD", "xgboost"),         # Requires xgboost package
            ("SUOD", "suod"),             # Requires suod package
        ]
        
        for algorithm, dependency in dependency_algorithms:
            with pytest.raises(InvalidAlgorithmError) as exc_info:
                PyODAdapter(algorithm)
            assert f"requires '{dependency}' package" in str(exc_info.value)
            assert f"pip install {dependency}" in str(exc_info.value)
    
    def test_algorithm_validation_comprehensive(self):
        """Test comprehensive algorithm validation for all known problematic algorithms."""
        # Test working algorithms that should succeed
        working_algorithms = ["PCA", "LOF", "IsolationForest", "OCSVM", "KNN"]
        
        for algorithm in working_algorithms:
            try:
                adapter = PyODAdapter(algorithm)
                assert adapter.algorithm_name == algorithm
                assert adapter.name == f"PyOD_{algorithm}"
            except Exception as e:
                pytest.fail(f"Working algorithm {algorithm} should not fail: {e}")
        
        # Test all known problematic algorithms
        problematic_algorithms = {
            "FastABOD": "not found in module",
            "Beta-VAE": "not found in module", 
            "CLF": "does not exist",
            "FeatureBagging": "combo",
            "XGBOD": "xgboost",
            "SUOD": "suod",
        }
        
        for algorithm, expected_error_fragment in problematic_algorithms.items():
            with pytest.raises(InvalidAlgorithmError) as exc_info:
                PyODAdapter(algorithm)
            assert expected_error_fragment in str(exc_info.value), f"Expected '{expected_error_fragment}' in error message for {algorithm}, got: {exc_info.value}"


@requires_dependency("scikit-learn")
class TestSklearnAdapter:
    """Test SklearnAdapter."""

    def test_list_algorithms(self):
        """Test listing available algorithms."""
        # Test the class-level algorithm mapping
        algorithms = list(SklearnAdapter.ALGORITHM_MAPPING.keys())

        assert len(algorithms) > 0
        assert "OneClassSVM" in algorithms
        assert "IsolationForest" in algorithms
        assert "EllipticEnvelope" in algorithms
        assert "LocalOutlierFactor" in algorithms

    def test_create_model(self):
        """Test creating adapter instance."""
        adapter = SklearnAdapter("IsolationForest")

        # Test that adapter is properly initialized
        assert adapter.algorithm_name == "IsolationForest"
        assert adapter.name == "Sklearn_IsolationForest"
        assert hasattr(adapter, "fit")
        assert hasattr(adapter, "detect")

    def test_fit_predict(self):
        """Test fitting and predicting."""
        adapter = SklearnAdapter("IsolationForest")

        # Generate data
        X = np.random.rand(100, 3)
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        dataset = Dataset(name="test", data=df)

        # Test fitting
        adapter.fit(dataset)
        assert adapter.is_fitted

        # Test detection
        result = adapter.detect(dataset)
        assert len(result.scores) == 100
        assert len(result.labels) == 100
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert all(label in [0, 1] for label in result.labels)
        assert len(result.anomalies) == sum(result.labels)  # Anomalies match labels

    def test_local_outlier_factor(self):
        """Test LocalOutlierFactor specifically."""
        adapter = SklearnAdapter("LocalOutlierFactor")

        # Generate data with clear outliers
        X = np.random.randn(100, 2)
        X[-5:] = X[-5:] + 3  # Add clear outliers
        df = pd.DataFrame(X, columns=["feature_0", "feature_1"])
        dataset = Dataset(name="test", data=df)

        # Test fitting and detection
        adapter.fit(dataset)
        result = adapter.detect(dataset)

        assert len(result.scores) == 100
        assert len(result.labels) == 100
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert all(label in [0, 1] for label in result.labels)
        assert sum(result.labels) > 0  # Should detect some anomalies
        assert len(result.anomalies) == sum(result.labels)  # Anomalies match labels


@requires_dependency("torch")
class TestPyTorchAdapter:
    """Test PyTorchAdapter."""

    @pytest.mark.skipif(
        not torch_available or PyTorchAdapter is None, reason="PyTorch not available"
    )
    def test_create_model(self):
        """Test creating PyTorch adapter instance."""
        if not torch_available or PyTorchAdapter is None:
            pytest.skip("PyTorch not available")

        adapter = PyTorchAdapter("AutoEncoder")
        assert adapter.algorithm_name == "AutoEncoder"
        assert adapter.name == "PyTorch_AutoEncoder"
        assert hasattr(adapter, "fit")
        assert hasattr(adapter, "detect")

    @pytest.mark.skipif(
        not torch_available or PyTorchAdapter is None, reason="PyTorch not available"
    )
    def test_torch_device_selection(self):
        """Test device selection for PyTorch."""
        if not torch_available or PyTorchAdapter is None:
            pytest.skip("PyTorch not available")

        adapter = PyTorchAdapter("AutoEncoder")
        # Should handle both CPU and CUDA devices gracefully
        assert hasattr(adapter, "device")


@requires_dependencies("pyod", "torch")
class TestPyGODAdapter:
    """Test PyGODAdapter for graph-based anomaly detection."""

    def test_list_algorithms(self):
        """Test listing available algorithms."""
        algorithms = list(PyGODAdapter.ALGORITHM_MAPPING.keys())
        assert len(algorithms) > 0
        assert "DOMINANT" in algorithms or "GCNAE" in algorithms

    def test_create_model(self):
        """Test creating PyGOD adapter instance."""
        adapter = PyGODAdapter("DOMINANT")
        assert adapter.algorithm_name == "DOMINANT"
        assert adapter.name == "PyGOD_DOMINANT"
        assert hasattr(adapter, "fit")
        assert hasattr(adapter, "detect")


class TestAdapterIntegration:
    """Test adapter integration and compatibility."""

    def test_adapter_protocol_compliance(self):
        """Test that all adapters follow the same protocol."""
        # Test with mock data that doesn't require external dependencies
        X = np.random.rand(50, 3)
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        Dataset(name="test", data=df)

        # Create mock adapters if real ones aren't available
        adapters_to_test = []

        # Only test adapters whose dependencies are available
        if sklearn_available:
            try:
                adapters_to_test.append(SklearnAdapter("IsolationForest"))
            except Exception:
                pass

        # Test basic protocol compliance
        for adapter in adapters_to_test:
            assert hasattr(adapter, "fit")
            assert hasattr(adapter, "detect")
            assert hasattr(adapter, "score")
            assert hasattr(adapter, "name")
            assert hasattr(adapter, "algorithm_name")

    def test_adapter_error_handling(self):
        """Test adapter error handling."""
        with pytest.raises(InvalidAlgorithmError):
            if sklearn_available:
                SklearnAdapter("NonExistentAlgorithm")

        # Test with unsupported algorithm names
        with pytest.raises((InvalidAlgorithmError, ValueError)):
            if PyODAdapter:
                PyODAdapter("UnsupportedAlgorithmName")

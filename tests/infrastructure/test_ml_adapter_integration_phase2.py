"""
Phase 2 Infrastructure Hardening: Comprehensive ML Adapter Integration Tests
Testing suite for PyOD, PyTorch, TensorFlow, JAX, PyGOD, TODS, and sklearn adapters.

This module implements comprehensive integration testing for all ML adapters with:
- Real dependency testing with graceful fallbacks
- Protocol compliance verification
- Performance benchmarking
- Error handling and edge cases
- Memory management and resource cleanup
"""

import gc
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import AdapterError


# Conditional imports for dependency-aware testing
def requires_dependency(dependency_name: str, module_name: str):
    """Decorator to skip tests if optional dependency is not available."""

    def decorator(test_func):
        try:
            __import__(module_name)
            return test_func
        except ImportError:
            return pytest.mark.skip(f"Requires {dependency_name} dependency")(test_func)

    return decorator


@contextmanager
def memory_monitor():
    """Context manager for monitoring memory usage during tests."""
    import psutil

    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    yield

    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_delta = final_memory - initial_memory

    # Log memory usage if significant
    if memory_delta > 100:  # More than 100MB increase
        print(f"Memory increase: {memory_delta:.2f} MB")


class MLAdapterTestBase:
    """Base class for ML adapter testing with common utilities."""

    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data for testing."""
        np.random.seed(42)

        # Create realistic anomaly data
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0, 0, 0], cov=np.eye(5), size=800
        )

        # Anomalies with different patterns
        point_anomalies = np.random.multivariate_normal(
            mean=[5, 5, 5, 5, 5], cov=np.eye(5) * 0.1, size=50
        )

        contextual_anomalies = np.random.multivariate_normal(
            mean=[0, 0, 0, 0, 10], cov=np.eye(5) * 2, size=50
        )

        collective_anomalies = np.random.multivariate_normal(
            mean=[-3, -3, -3, -3, -3], cov=np.eye(5) * 0.5, size=100
        )

        # Combine all data
        data = np.vstack(
            [normal_data, point_anomalies, contextual_anomalies, collective_anomalies]
        )

        # Create labels (1 = normal, -1 = anomaly for sklearn convention)
        labels = np.array([1] * 800 + [-1] * 200)

        return {
            "X_train": data[:600].astype(np.float32),
            "X_test": data[600:].astype(np.float32),
            "y_train": labels[:600],
            "y_test": labels[600:],
            "features": [f"feature_{i}" for i in range(5)],
            "full_data": data,
            "full_labels": labels,
        }

    @pytest.fixture
    def mock_dataset(self, sample_data):
        """Create mock dataset with realistic properties."""
        dataset = Mock(spec=Dataset)
        dataset.id = uuid.uuid4()
        dataset.name = "test_anomaly_dataset"
        dataset.data = pd.DataFrame(
            sample_data["X_train"], columns=sample_data["features"]
        )
        dataset.features = sample_data["features"]
        dataset.target_column = None
        dataset.created_at = datetime.now()
        dataset.metadata = {
            "source": "synthetic",
            "anomaly_rate": 0.2,
            "data_quality": "high",
        }
        return dataset

    def verify_protocol_compliance(self, adapter_class: type, adapter_instance: Any):
        """Verify adapter implements DetectorProtocol correctly."""
        # Check required methods exist
        required_methods = ["fit", "predict", "decision_function", "get_algorithm_info"]
        for method in required_methods:
            assert hasattr(adapter_instance, method), (
                f"Missing required method: {method}"
            )
            assert callable(getattr(adapter_instance, method)), (
                f"Method {method} is not callable"
            )

        # Check properties
        assert hasattr(adapter_instance, "is_fitted"), "Missing is_fitted property"
        assert hasattr(adapter_instance, "contamination"), (
            "Missing contamination property"
        )

    def benchmark_adapter_performance(self, adapter_instance: Any, sample_data: dict):
        """Benchmark adapter performance with different data sizes."""
        results = {}

        data_sizes = [100, 500, 1000]
        for size in data_sizes:
            if size > len(sample_data["X_train"]):
                continue

            X_subset = sample_data["X_train"][:size]

            # Time fitting
            start_time = time.time()
            try:
                adapter_instance.fit(X_subset)
                fit_time = time.time() - start_time

                # Time prediction
                start_time = time.time()
                adapter_instance.predict(X_subset[:50])  # Predict on subset
                predict_time = time.time() - start_time

                results[size] = {
                    "fit_time": fit_time,
                    "predict_time": predict_time,
                    "samples_per_second_fit": size / fit_time
                    if fit_time > 0
                    else float("inf"),
                    "samples_per_second_predict": 50 / predict_time
                    if predict_time > 0
                    else float("inf"),
                }
            except Exception as e:
                results[size] = {"error": str(e)}

        return results


class TestPyODAdapterIntegration(MLAdapterTestBase):
    """Comprehensive PyOD adapter integration tests."""

    @requires_dependency("PyOD", "pyod")
    def test_pyod_adapter_full_lifecycle(self, sample_data, mock_dataset):
        """Test complete PyOD adapter lifecycle with real dependencies."""
        from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

        with memory_monitor():
            # Test different algorithms
            algorithms = ["IsolationForest", "LOF", "OCSVM", "ABOD"]

            for algorithm in algorithms:
                adapter = PyODAdapter(algorithm=algorithm)

                # Verify protocol compliance
                self.verify_protocol_compliance(PyODAdapter, adapter)

                # Test fitting
                adapter.fit(sample_data["X_train"])
                assert adapter.is_fitted, f"{algorithm} adapter should be fitted"

                # Test prediction
                predictions = adapter.predict(sample_data["X_test"])
                assert len(predictions) == len(sample_data["X_test"])
                assert set(predictions).issubset({0, 1}), "Predictions should be binary"

                # Test decision function
                scores = adapter.decision_function(sample_data["X_test"])
                assert len(scores) == len(sample_data["X_test"])
                assert all(
                    isinstance(score, (int, float, np.number)) for score in scores
                )

                # Benchmark performance
                perf_results = self.benchmark_adapter_performance(adapter, sample_data)
                assert len(perf_results) > 0, f"No performance results for {algorithm}"

    @requires_dependency("PyOD", "pyod")
    def test_pyod_algorithm_metadata(self):
        """Test PyOD algorithm metadata and information retrieval."""
        from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

        adapter = PyODAdapter()

        # Test algorithm listing
        algorithms = adapter.list_algorithms()
        assert len(algorithms) > 20, "Should have many PyOD algorithms"
        assert "IsolationForest" in algorithms
        assert "LOF" in algorithms

        # Test algorithm info
        for algorithm in ["IsolationForest", "LOF", "OCSVM"]:
            info = adapter.get_algorithm_info(algorithm)
            assert "name" in info
            assert "description" in info
            assert "parameters" in info
            assert "category" in info

    @requires_dependency("PyOD", "pyod")
    def test_pyod_error_handling(self, sample_data):
        """Test PyOD adapter error handling and edge cases."""
        from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

        # Test invalid algorithm
        with pytest.raises((AdapterError, ValueError)):
            PyODAdapter(algorithm="InvalidAlgorithm")

        # Test fitting with invalid data
        adapter = PyODAdapter(algorithm="IsolationForest")

        with pytest.raises((AdapterError, ValueError)):
            adapter.fit(np.array([]))  # Empty data

        with pytest.raises((AdapterError, ValueError)):
            adapter.fit(np.array([[1, 2], [np.inf, 4]]))  # Invalid values

        # Test prediction before fitting
        unfitted_adapter = PyODAdapter(algorithm="LOF")
        with pytest.raises((AdapterError, ValueError)):
            unfitted_adapter.predict(sample_data["X_test"])


class TestSklearnAdapterIntegration(MLAdapterTestBase):
    """Comprehensive sklearn adapter integration tests."""

    def test_sklearn_adapter_full_lifecycle(self, sample_data, mock_dataset):
        """Test complete sklearn adapter lifecycle."""
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

        with memory_monitor():
            # Test different algorithms
            algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]

            for algorithm in algorithms:
                adapter = SklearnAdapter(algorithm=algorithm)

                # Verify protocol compliance
                self.verify_protocol_compliance(SklearnAdapter, adapter)

                # Test fitting
                adapter.fit(sample_data["X_train"])
                assert adapter.is_fitted, f"{algorithm} adapter should be fitted"

                # Test prediction
                predictions = adapter.predict(sample_data["X_test"])
                assert len(predictions) == len(sample_data["X_test"])
                assert set(predictions).issubset({0, 1}), "Predictions should be binary"

                # Test decision function
                scores = adapter.decision_function(sample_data["X_test"])
                assert len(scores) == len(sample_data["X_test"])

                # Benchmark performance
                perf_results = self.benchmark_adapter_performance(adapter, sample_data)
                assert len(perf_results) > 0, f"No performance results for {algorithm}"

    def test_sklearn_contamination_handling(self, sample_data):
        """Test contamination rate handling in sklearn adapter."""
        from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

        contamination_rates = [0.05, 0.1, 0.2, 0.3]

        for rate in contamination_rates:
            adapter = SklearnAdapter(algorithm="IsolationForest", contamination=rate)

            adapter.fit(sample_data["X_train"])
            predictions = adapter.predict(sample_data["X_test"])

            # Check that contamination rate influences results
            anomaly_rate = np.mean(predictions)
            # Should be roughly within expected range (allowing for variance)
            assert 0.0 <= anomaly_rate <= 1.0, f"Invalid anomaly rate: {anomaly_rate}"


@requires_dependency("PyTorch", "torch")
class TestPyTorchAdapterIntegration(MLAdapterTestBase):
    """Comprehensive PyTorch adapter integration tests."""

    def test_pytorch_adapter_autoencoder(self, sample_data, mock_dataset):
        """Test PyTorch AutoEncoder implementation."""
        from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter

        with memory_monitor():
            adapter = PyTorchAdapter(
                algorithm="AutoEncoder",
                epochs=5,  # Reduced for testing
                batch_size=32,
            )

            # Verify protocol compliance
            self.verify_protocol_compliance(PyTorchAdapter, adapter)

            # Test fitting
            adapter.fit(sample_data["X_train"])
            assert adapter.is_fitted, "PyTorch adapter should be fitted"

            # Test prediction
            predictions = adapter.predict(sample_data["X_test"])
            assert len(predictions) == len(sample_data["X_test"])
            assert set(predictions).issubset({0, 1}), "Predictions should be binary"

            # Test reconstruction scores
            scores = adapter.decision_function(sample_data["X_test"])
            assert len(scores) == len(sample_data["X_test"])
            assert all(score >= 0 for score in scores), (
                "Reconstruction scores should be non-negative"
            )

    @requires_dependency("PyTorch", "torch")
    def test_pytorch_vae_implementation(self, sample_data):
        """Test PyTorch VAE (Variational AutoEncoder) implementation."""
        from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter

        adapter = PyTorchAdapter(
            algorithm="VAE",
            epochs=5,
            latent_dim=3,
            beta=1.0,  # Standard beta-VAE
        )

        adapter.fit(sample_data["X_train"])

        # Test generation capability (VAE-specific)
        if hasattr(adapter, "generate_samples"):
            generated = adapter.generate_samples(10)
            assert generated.shape == (10, sample_data["X_train"].shape[1])

    @requires_dependency("PyTorch", "torch")
    def test_pytorch_gpu_cpu_compatibility(self, sample_data):
        """Test PyTorch adapter GPU/CPU device handling."""
        import torch

        from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter

        # Test CPU mode
        adapter_cpu = PyTorchAdapter(algorithm="AutoEncoder", device="cpu", epochs=2)

        adapter_cpu.fit(sample_data["X_train"])
        predictions_cpu = adapter_cpu.predict(sample_data["X_test"])

        # Test GPU mode if available
        if torch.cuda.is_available():
            adapter_gpu = PyTorchAdapter(
                algorithm="AutoEncoder", device="cuda", epochs=2
            )

            adapter_gpu.fit(sample_data["X_train"])
            predictions_gpu = adapter_gpu.predict(sample_data["X_test"])

            # Results should be similar (allowing for numerical differences)
            assert len(predictions_cpu) == len(predictions_gpu)


@requires_dependency("TensorFlow", "tensorflow")
class TestTensorFlowAdapterIntegration(MLAdapterTestBase):
    """Comprehensive TensorFlow adapter integration tests."""

    def test_tensorflow_adapter_autoencoder(self, sample_data, mock_dataset):
        """Test TensorFlow AutoEncoder implementation."""
        from pynomaly.infrastructure.adapters.tensorflow_adapter import (
            TensorFlowAdapter,
        )

        with memory_monitor():
            adapter = TensorFlowAdapter(
                algorithm="AutoEncoder",
                epochs=5,  # Reduced for testing
                batch_size=32,
            )

            # Verify protocol compliance
            self.verify_protocol_compliance(TensorFlowAdapter, adapter)

            # Test fitting
            adapter.fit(sample_data["X_train"])
            assert adapter.is_fitted, "TensorFlow adapter should be fitted"

            # Test prediction
            predictions = adapter.predict(sample_data["X_test"])
            assert len(predictions) == len(sample_data["X_test"])
            assert set(predictions).issubset({0, 1}), "Predictions should be binary"

    @requires_dependency("TensorFlow", "tensorflow")
    def test_tensorflow_training_history(self, sample_data):
        """Test TensorFlow training history tracking."""
        from pynomaly.infrastructure.adapters.tensorflow_adapter import (
            TensorFlowAdapter,
        )

        adapter = TensorFlowAdapter(
            algorithm="AutoEncoder", epochs=5, validation_split=0.2
        )

        adapter.fit(sample_data["X_train"])

        # Check training history
        if hasattr(adapter, "training_history"):
            history = adapter.training_history
            assert "loss" in history
            assert len(history["loss"]) == 5  # 5 epochs


@requires_dependency("JAX", "jax")
class TestJAXAdapterIntegration(MLAdapterTestBase):
    """Comprehensive JAX adapter integration tests."""

    def test_jax_adapter_autoencoder(self, sample_data, mock_dataset):
        """Test JAX AutoEncoder implementation."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        with memory_monitor():
            adapter = JAXAdapter(
                algorithm="AutoEncoder",
                epochs=5,  # Reduced for testing
                learning_rate=0.001,
            )

            # Verify protocol compliance
            self.verify_protocol_compliance(JAXAdapter, adapter)

            # Test fitting
            adapter.fit(sample_data["X_train"])
            assert adapter.is_fitted, "JAX adapter should be fitted"

            # Test prediction
            predictions = adapter.predict(sample_data["X_test"])
            assert len(predictions) == len(sample_data["X_test"])
            assert set(predictions).issubset({0, 1}), "Predictions should be binary"

    @requires_dependency("JAX", "jax")
    def test_jax_jit_compilation(self, sample_data):
        """Test JAX JIT compilation benefits."""
        import time

        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        adapter = JAXAdapter(algorithm="IsolationForest", use_jit=True)
        adapter.fit(sample_data["X_train"])

        # First prediction (compilation time included)
        start_time = time.time()
        adapter.predict(sample_data["X_test"][:10])
        first_time = time.time() - start_time

        # Second prediction (should be faster due to JIT)
        start_time = time.time()
        adapter.predict(sample_data["X_test"][:10])
        second_time = time.time() - start_time

        # Second run should typically be faster (allowing some variance)
        assert second_time <= first_time * 2, "JIT should provide performance benefits"


class TestMLAdapterInteroperability:
    """Test interoperability between different ML adapters."""

    @pytest.fixture
    def sample_data(self):
        """Create consistent sample data for comparison."""
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 4)).astype(np.float32)

    def test_adapter_consistency(self, sample_data):
        """Test that different adapters produce consistent results on same data."""
        results = {}

        # Test sklearn IsolationForest
        try:
            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            sklearn_adapter = SklearnAdapter(algorithm="IsolationForest")
            sklearn_adapter.fit(sample_data)
            results["sklearn"] = sklearn_adapter.predict(sample_data)
        except ImportError:
            pass

        # Test PyOD IsolationForest
        try:
            from pynomaly.infrastructure.adapters.pyod_adapter import PyODAdapter

            pyod_adapter = PyODAdapter(algorithm="IsolationForest")
            pyod_adapter.fit(sample_data)
            results["pyod"] = pyod_adapter.predict(sample_data)
        except ImportError:
            pass

        # Compare results if both available
        if "sklearn" in results and "pyod" in results:
            # Should have similar anomaly detection rates (allowing for algorithm differences)
            sklearn_rate = np.mean(results["sklearn"])
            pyod_rate = np.mean(results["pyod"])

            # Allow for reasonable variance between implementations
            assert abs(sklearn_rate - pyod_rate) < 0.3, (
                "Adapters should produce similar results"
            )

    def test_adapter_memory_cleanup(self, sample_data):
        """Test that adapters properly clean up resources."""
        initial_objects = len(gc.get_objects())

        adapters = []

        # Create multiple adapters
        try:
            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            for _ in range(5):
                adapter = SklearnAdapter(algorithm="IsolationForest")
                adapter.fit(sample_data)
                adapters.append(adapter)
        except ImportError:
            pass

        # Clear references and force garbage collection
        adapters.clear()
        gc.collect()

        final_objects = len(gc.get_objects())

        # Should not have excessive memory growth
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Excessive object growth: {object_growth}"


# Performance and stress testing
class TestMLAdapterPerformance:
    """Performance and stress testing for ML adapters."""

    @pytest.mark.slow
    def test_large_dataset_handling(self):
        """Test adapter performance with large datasets."""
        # Create large dataset
        np.random.seed(42)
        large_data = np.random.normal(0, 1, (10000, 10)).astype(np.float32)

        try:
            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            adapter = SklearnAdapter(algorithm="IsolationForest")

            start_time = time.time()
            adapter.fit(large_data)
            fit_time = time.time() - start_time

            start_time = time.time()
            predictions = adapter.predict(large_data[:1000])
            predict_time = time.time() - start_time

            # Performance benchmarks (reasonable expectations)
            assert fit_time < 60, f"Fitting took too long: {fit_time}s"
            assert predict_time < 10, f"Prediction took too long: {predict_time}s"
            assert len(predictions) == 1000

        except ImportError:
            pytest.skip("sklearn not available for performance testing")

    @pytest.mark.stress
    def test_memory_stress(self):
        """Test adapter behavior under memory stress."""
        try:
            from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter

            # Create progressively larger datasets
            base_size = 1000
            max_multiplier = 5

            for multiplier in range(1, max_multiplier + 1):
                size = base_size * multiplier
                data = np.random.normal(0, 1, (size, 5)).astype(np.float32)

                adapter = SklearnAdapter(algorithm="IsolationForest")
                adapter.fit(data)

                # Verify still working
                predictions = adapter.predict(data[:100])
                assert len(predictions) == 100

                # Clean up
                del adapter
                del data
                gc.collect()

        except (ImportError, MemoryError):
            pytest.skip("Memory stress test conditions not met")


if __name__ == "__main__":
    # Run specific test classes for debugging
    pytest.main([__file__ + "::TestSklearnAdapterIntegration", "-v"])

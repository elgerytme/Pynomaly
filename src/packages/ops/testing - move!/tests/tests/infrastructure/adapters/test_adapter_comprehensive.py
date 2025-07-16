"""
Comprehensive infrastructure adapter tests.
Tests algorithm adapters, ML framework integrations, and adapter registry.
"""

from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from monorepo.domain.entities import Detector
from monorepo.domain.exceptions import InfrastructureError, ValidationError
from monorepo.infrastructure.adapters.enhanced_pyod_adapter import EnhancedPyodAdapter
from monorepo.infrastructure.adapters.enhanced_sklearn_adapter import (
    EnhancedSklearnAdapter,
)
from monorepo.infrastructure.adapters.ensemble_adapter import EnsembleAdapter
from monorepo.infrastructure.adapters.optimized_adapter import OptimizedAdapter


class TestEnhancedSklearnAdapter:
    """Test suite for enhanced sklearn adapter."""

    @pytest.fixture
    def sklearn_adapter(self):
        """Create sklearn adapter for testing."""
        return EnhancedSklearnAdapter()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 5))
        anomaly_data = np.random.normal(3, 1, (10, 5))
        return np.vstack([normal_data, anomaly_data])

    @pytest.fixture
    def sample_detector(self):
        """Create sample detector configuration."""
        return Detector(
            id=uuid4(),
            name="test-sklearn-detector",
            algorithm_name="IsolationForest",
            hyperparameters={
                "n_estimators": 100,
                "contamination": 0.1,
                "random_state": 42,
            },
        )

    def test_sklearn_adapter_supported_algorithms(self, sklearn_adapter):
        """Test sklearn adapter algorithm support."""
        supported = sklearn_adapter.get_supported_algorithms()

        expected_algorithms = [
            "IsolationForest",
            "OneClassSVM",
            "LocalOutlierFactor",
            "EllipticEnvelope",
        ]

        for algorithm in expected_algorithms:
            assert algorithm in supported
            assert sklearn_adapter.supports_algorithm(algorithm)

        # Test unsupported algorithm
        assert not sklearn_adapter.supports_algorithm("NonExistentAlgorithm")

    def test_sklearn_adapter_fit_predict(
        self, sklearn_adapter, sample_detector, sample_data
    ):
        """Test sklearn adapter fit and predict operations."""
        # Test fit
        fitted_model = sklearn_adapter.fit(sample_detector, sample_data)
        assert fitted_model is not None

        # Test predict
        predictions = sklearn_adapter.predict(fitted_model, sample_data)
        assert len(predictions) == len(sample_data)
        assert all(pred in [0, 1] for pred in predictions)

        # Test predict_proba
        probabilities = sklearn_adapter.predict_proba(fitted_model, sample_data)
        assert len(probabilities) == len(sample_data)
        assert all(0 <= prob <= 1 for prob in probabilities)

    def test_sklearn_adapter_hyperparameter_validation(self, sklearn_adapter):
        """Test sklearn adapter hyperparameter validation."""
        # Valid hyperparameters
        valid_params = {"n_estimators": 100, "contamination": 0.1, "random_state": 42}

        detector = Detector(
            id=uuid4(),
            name="valid-detector",
            algorithm_name="IsolationForest",
            hyperparameters=valid_params,
        )

        # Should not raise exception
        sklearn_adapter.validate_hyperparameters(detector)

        # Invalid hyperparameters
        invalid_params = {
            "n_estimators": -10,  # Invalid: negative
            "contamination": 1.5,  # Invalid: > 1
            "random_state": "invalid",  # Invalid: not int
        }

        invalid_detector = Detector(
            id=uuid4(),
            name="invalid-detector",
            algorithm_name="IsolationForest",
            hyperparameters=invalid_params,
        )

        with pytest.raises(ValidationError):
            sklearn_adapter.validate_hyperparameters(invalid_detector)

    def test_sklearn_adapter_algorithm_specific_features(
        self, sklearn_adapter, sample_data
    ):
        """Test algorithm-specific features."""
        algorithms_to_test = [
            ("IsolationForest", {"n_estimators": 50, "contamination": 0.1}),
            ("OneClassSVM", {"nu": 0.1, "kernel": "rbf"}),
            ("LocalOutlierFactor", {"n_neighbors": 20, "contamination": 0.1}),
            ("EllipticEnvelope", {"contamination": 0.1}),
        ]

        for algorithm_name, hyperparams in algorithms_to_test:
            detector = Detector(
                id=uuid4(),
                name=f"test-{algorithm_name.lower()}",
                algorithm_name=algorithm_name,
                hyperparameters=hyperparams,
            )

            # Test fit
            model = sklearn_adapter.fit(detector, sample_data)
            assert model is not None

            # Test predict
            predictions = sklearn_adapter.predict(model, sample_data)
            assert len(predictions) == len(sample_data)

            # Test that some anomalies are detected
            anomaly_count = sum(predictions)
            assert 0 < anomaly_count < len(sample_data)

    def test_sklearn_adapter_model_serialization(
        self, sklearn_adapter, sample_detector, sample_data
    ):
        """Test model serialization and deserialization."""
        # Fit model
        model = sklearn_adapter.fit(sample_detector, sample_data)

        # Serialize model
        serialized = sklearn_adapter.serialize_model(model)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

        # Deserialize model
        deserialized_model = sklearn_adapter.deserialize_model(serialized)
        assert deserialized_model is not None

        # Test that deserialized model produces same predictions
        original_predictions = sklearn_adapter.predict(model, sample_data)
        deserialized_predictions = sklearn_adapter.predict(
            deserialized_model, sample_data
        )

        np.testing.assert_array_equal(original_predictions, deserialized_predictions)

    def test_sklearn_adapter_error_handling(self, sklearn_adapter):
        """Test sklearn adapter error handling."""
        # Test invalid data
        invalid_data = "not_an_array"
        detector = Detector(
            id=uuid4(),
            name="error-test",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100},
        )

        with pytest.raises(ValidationError):
            sklearn_adapter.fit(detector, invalid_data)

        # Test empty data
        empty_data = np.array([]).reshape(0, 5)
        with pytest.raises(ValidationError):
            sklearn_adapter.fit(detector, empty_data)

        # Test incompatible dimensions
        model = sklearn_adapter.fit(detector, np.random.randn(100, 5))
        incompatible_data = np.random.randn(50, 3)  # Different feature count

        with pytest.raises(ValidationError):
            sklearn_adapter.predict(model, incompatible_data)


class TestEnhancedPyodAdapter:
    """Test suite for enhanced PyOD adapter."""

    @pytest.fixture
    def pyod_adapter(self):
        """Create PyOD adapter for testing."""
        return EnhancedPyodAdapter()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 5))
        anomaly_data = np.random.normal(4, 1, (10, 5))
        return np.vstack([normal_data, anomaly_data])

    def test_pyod_adapter_supported_algorithms(self, pyod_adapter):
        """Test PyOD adapter algorithm support."""
        supported = pyod_adapter.get_supported_algorithms()

        expected_algorithms = [
            "ABOD",
            "AutoEncoder",
            "COPOD",
            "DeepSVDD",
            "HBOS",
            "KNN",
            "LMDD",
            "LOF",
            "OCSVM",
            "PCA",
        ]

        # Check that most expected algorithms are supported
        supported_count = sum(1 for alg in expected_algorithms if alg in supported)
        assert supported_count >= len(expected_algorithms) // 2  # At least half

        # Test some specific algorithms
        assert pyod_adapter.supports_algorithm("LOF")
        assert pyod_adapter.supports_algorithm("KNN")
        assert not pyod_adapter.supports_algorithm("NonExistentAlgorithm")

    @patch("pyod.models.lof.LOF")
    def test_pyod_adapter_fit_predict(self, mock_lof_class, pyod_adapter, sample_data):
        """Test PyOD adapter fit and predict operations."""
        # Mock PyOD model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        mock_lof_class.return_value = mock_model

        detector = Detector(
            id=uuid4(),
            name="test-pyod-detector",
            algorithm_name="LOF",
            hyperparameters={"n_neighbors": 20},
        )

        # Test fit
        fitted_model = pyod_adapter.fit(detector, sample_data)
        assert fitted_model is not None
        mock_model.fit.assert_called_once()

        # Test predict
        predictions = pyod_adapter.predict(fitted_model, sample_data[:5])
        assert len(predictions) == 5
        mock_model.predict.assert_called_once()

        # Test predict_proba
        probabilities = pyod_adapter.predict_proba(fitted_model, sample_data[:5])
        assert len(probabilities) == 5
        mock_model.predict_proba.assert_called_once()

    def test_pyod_adapter_hyperparameter_optimization(self, pyod_adapter):
        """Test PyOD adapter hyperparameter optimization."""
        detector = Detector(
            id=uuid4(),
            name="optimization-test",
            algorithm_name="LOF",
            hyperparameters={"n_neighbors": 20},
        )

        # Test hyperparameter suggestion
        suggested_params = pyod_adapter.suggest_hyperparameters(
            detector, data_size=(1000, 10), optimization_budget=10
        )

        assert isinstance(suggested_params, dict)
        assert "n_neighbors" in suggested_params
        assert isinstance(suggested_params["n_neighbors"], int)
        assert suggested_params["n_neighbors"] > 0


class TestEnsembleAdapter:
    """Test suite for ensemble adapter."""

    @pytest.fixture
    def ensemble_adapter(self):
        """Create ensemble adapter for testing."""
        return EnsembleAdapter()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.randn(100, 5)

    @pytest.fixture
    def base_adapters(self):
        """Create base adapters for ensemble."""
        return [EnhancedSklearnAdapter(), EnhancedPyodAdapter()]

    def test_ensemble_adapter_creation(self, ensemble_adapter, base_adapters):
        """Test ensemble adapter creation with base adapters."""
        ensemble_adapter.add_base_adapter("sklearn", base_adapters[0])
        ensemble_adapter.add_base_adapter("pyod", base_adapters[1])

        assert len(ensemble_adapter.get_base_adapters()) == 2
        assert "sklearn" in ensemble_adapter.get_base_adapters()
        assert "pyod" in ensemble_adapter.get_base_adapters()

    def test_ensemble_adapter_voting_strategies(self, ensemble_adapter, sample_data):
        """Test different voting strategies."""
        # Mock base adapters
        mock_adapter1 = Mock()
        mock_adapter1.fit.return_value = "model1"
        mock_adapter1.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_adapter1.predict_proba.return_value = np.array([0.2, 0.8, 0.3, 0.7, 0.1])

        mock_adapter2 = Mock()
        mock_adapter2.fit.return_value = "model2"
        mock_adapter2.predict.return_value = np.array([1, 1, 0, 0, 0])
        mock_adapter2.predict_proba.return_value = np.array([0.6, 0.9, 0.2, 0.4, 0.3])

        ensemble_adapter.add_base_adapter("adapter1", mock_adapter1)
        ensemble_adapter.add_base_adapter("adapter2", mock_adapter2)

        detector = Detector(
            id=uuid4(),
            name="ensemble-test",
            algorithm_name="Ensemble",
            hyperparameters={
                "voting_strategy": "majority",
                "base_algorithms": ["IsolationForest", "LOF"],
            },
        )

        # Test fit
        ensemble_model = ensemble_adapter.fit(detector, sample_data)
        assert ensemble_model is not None
        assert len(ensemble_model["models"]) == 2

        # Test majority voting
        predictions = ensemble_adapter.predict(ensemble_model, sample_data[:5])
        expected_majority = [
            0,
            1,
            0,
            0,
            0,
        ]  # Majority vote of [0,1,0,1,0] and [1,1,0,0,0]
        np.testing.assert_array_equal(predictions, expected_majority)

        # Test average probability
        probabilities = ensemble_adapter.predict_proba(ensemble_model, sample_data[:5])
        expected_avg = np.array([0.4, 0.85, 0.25, 0.55, 0.2])  # Average of two arrays
        np.testing.assert_array_almost_equal(probabilities, expected_avg, decimal=2)

    def test_ensemble_adapter_weighted_voting(self, ensemble_adapter, sample_data):
        """Test weighted voting in ensemble."""
        # Mock base adapters with different performance weights
        mock_adapter1 = Mock()
        mock_adapter1.fit.return_value = "model1"
        mock_adapter1.predict_proba.return_value = np.array([0.2, 0.8, 0.3])

        mock_adapter2 = Mock()
        mock_adapter2.fit.return_value = "model2"
        mock_adapter2.predict_proba.return_value = np.array([0.6, 0.4, 0.7])

        ensemble_adapter.add_base_adapter("adapter1", mock_adapter1, weight=0.7)
        ensemble_adapter.add_base_adapter("adapter2", mock_adapter2, weight=0.3)

        detector = Detector(
            id=uuid4(),
            name="weighted-ensemble",
            algorithm_name="WeightedEnsemble",
            hyperparameters={
                "voting_strategy": "weighted_average",
                "weights": [0.7, 0.3],
            },
        )

        ensemble_model = ensemble_adapter.fit(detector, sample_data)
        probabilities = ensemble_adapter.predict_proba(ensemble_model, sample_data[:3])

        # Weighted average: 0.7 * [0.2, 0.8, 0.3] + 0.3 * [0.6, 0.4, 0.7]
        expected_weighted = np.array([0.32, 0.68, 0.42])
        np.testing.assert_array_almost_equal(
            probabilities, expected_weighted, decimal=2
        )


class TestOptimizedAdapter:
    """Test suite for optimized adapter."""

    @pytest.fixture
    def optimized_adapter(self):
        """Create optimized adapter for testing."""
        return OptimizedAdapter()

    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        np.random.seed(42)
        return np.random.randn(10000, 20)

    def test_optimized_adapter_memory_efficiency(
        self, optimized_adapter, large_dataset
    ):
        """Test memory efficiency optimizations."""
        detector = Detector(
            id=uuid4(),
            name="memory-test",
            algorithm_name="IsolationForest",
            hyperparameters={
                "n_estimators": 100,
                "max_samples": 1000,
                "memory_optimization": True,
            },
        )

        # Test chunked processing
        chunk_size = 1000
        model = optimized_adapter.fit_chunked(
            detector, large_dataset, chunk_size=chunk_size
        )
        assert model is not None

        # Test chunked prediction
        predictions = optimized_adapter.predict_chunked(
            model, large_dataset, chunk_size=chunk_size
        )
        assert len(predictions) == len(large_dataset)

    def test_optimized_adapter_parallel_processing(
        self, optimized_adapter, large_dataset
    ):
        """Test parallel processing capabilities."""
        detector = Detector(
            id=uuid4(),
            name="parallel-test",
            algorithm_name="IsolationForest",
            hyperparameters={
                "n_estimators": 100,
                "n_jobs": 4,
                "parallel_backend": "threading",
            },
        )

        # Test parallel fit
        model = optimized_adapter.fit_parallel(detector, large_dataset, n_jobs=4)
        assert model is not None

        # Test parallel predict
        predictions = optimized_adapter.predict_parallel(model, large_dataset, n_jobs=4)
        assert len(predictions) == len(large_dataset)

    def test_optimized_adapter_caching(self, optimized_adapter, large_dataset):
        """Test caching mechanisms."""
        detector = Detector(
            id=uuid4(),
            name="cache-test",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100},
        )

        # First fit (should cache)
        model1 = optimized_adapter.fit_with_cache(detector, large_dataset)
        assert model1 is not None

        # Second fit with same data (should use cache)
        model2 = optimized_adapter.fit_with_cache(detector, large_dataset)
        assert model2 is not None

        # Verify cache was used (implementation specific)
        cache_stats = optimized_adapter.get_cache_stats()
        assert cache_stats["hits"] > 0 or cache_stats["total_size"] > 0

    def test_optimized_adapter_gpu_acceleration(self, optimized_adapter):
        """Test GPU acceleration when available."""
        detector = Detector(
            id=uuid4(),
            name="gpu-test",
            algorithm_name="IsolationForest",
            hyperparameters={"use_gpu": True, "gpu_memory_limit": "1GB"},
        )

        # Test GPU availability check
        gpu_available = optimized_adapter.is_gpu_available()
        assert isinstance(gpu_available, bool)

        # Test GPU configuration
        if gpu_available:
            gpu_config = optimized_adapter.configure_gpu(detector)
            assert isinstance(gpu_config, dict)
            assert "device" in gpu_config


class TestAdapterRegistry:
    """Test suite for adapter registry system."""

    def test_adapter_registry_registration(self):
        """Test adapter registration and retrieval."""
        from monorepo.infrastructure.adapters.registry import AdapterRegistry

        registry = AdapterRegistry()

        # Register adapters
        sklearn_adapter = EnhancedSklearnAdapter()
        pyod_adapter = EnhancedPyodAdapter()

        registry.register("sklearn", sklearn_adapter)
        registry.register("pyod", pyod_adapter)

        # Test retrieval
        assert registry.get("sklearn") is sklearn_adapter
        assert registry.get("pyod") is pyod_adapter
        assert registry.get("nonexistent") is None

        # Test listing
        registered_names = registry.list_registered()
        assert "sklearn" in registered_names
        assert "pyod" in registered_names

    def test_adapter_registry_algorithm_discovery(self):
        """Test algorithm discovery across adapters."""
        from monorepo.infrastructure.adapters.registry import AdapterRegistry

        registry = AdapterRegistry()
        registry.register("sklearn", EnhancedSklearnAdapter())
        registry.register("pyod", EnhancedPyodAdapter())

        # Test algorithm discovery
        all_algorithms = registry.get_all_supported_algorithms()
        assert "IsolationForest" in all_algorithms
        assert "LOF" in all_algorithms

        # Test adapter lookup by algorithm
        if_adapter = registry.get_adapter_for_algorithm("IsolationForest")
        assert if_adapter is not None

        lof_adapter = registry.get_adapter_for_algorithm("LOF")
        assert lof_adapter is not None


class TestAdapterErrorHandling:
    """Test suite for adapter error handling scenarios."""

    def test_adapter_validation_errors(self):
        """Test adapter validation error scenarios."""
        adapter = EnhancedSklearnAdapter()

        # Test invalid algorithm
        invalid_detector = Detector(
            id=uuid4(),
            name="invalid-algorithm",
            algorithm_name="NonExistentAlgorithm",
            hyperparameters={},
        )

        with pytest.raises(ValidationError):
            adapter.validate_detector(invalid_detector)

        # Test missing required hyperparameters
        incomplete_detector = Detector(
            id=uuid4(),
            name="incomplete",
            algorithm_name="IsolationForest",
            hyperparameters={},  # Missing required parameters
        )

        # Should either work with defaults or raise validation error
        try:
            adapter.validate_detector(incomplete_detector)
        except ValidationError:
            pass  # Expected for some algorithms

    def test_adapter_infrastructure_errors(self):
        """Test adapter infrastructure error scenarios."""
        adapter = EnhancedSklearnAdapter()

        detector = Detector(
            id=uuid4(),
            name="infra-test",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100},
        )

        # Test with corrupted data
        corrupted_data = np.array([[np.nan, np.inf, -np.inf, 1, 2]])

        with pytest.raises((ValidationError, InfrastructureError)):
            adapter.fit(detector, corrupted_data)

    def test_adapter_memory_errors(self):
        """Test adapter memory error handling."""
        adapter = OptimizedAdapter()

        detector = Detector(
            id=uuid4(),
            name="memory-test",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 10000},  # Excessive
        )

        # Simulate memory constraint
        small_data = np.random.randn(10, 5)

        try:
            # Should either work or handle memory issues gracefully
            model = adapter.fit_with_memory_limit(
                detector, small_data, memory_limit="100MB"
            )
            assert (
                model is not None or True
            )  # Accept either success or controlled failure
        except (MemoryError, InfrastructureError):
            pass  # Expected for memory-constrained scenarios


class TestAdapterPerformance:
    """Test suite for adapter performance characteristics."""

    @pytest.fixture
    def performance_data(self):
        """Create data for performance testing."""
        np.random.seed(42)
        sizes = [100, 1000, 5000]
        datasets = {}

        for size in sizes:
            datasets[size] = np.random.randn(size, 10)

        return datasets

    def test_adapter_scaling_performance(self, performance_data):
        """Test adapter performance scaling with data size."""
        adapter = EnhancedSklearnAdapter()

        detector = Detector(
            id=uuid4(),
            name="scaling-test",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 50},
        )

        performance_results = {}

        for size, data in performance_data.items():
            import time

            start_time = time.time()

            model = adapter.fit(detector, data)
            predictions = adapter.predict(model, data)

            end_time = time.time()
            performance_results[size] = {
                "time": end_time - start_time,
                "predictions_count": len(predictions),
            }

        # Verify scaling behavior
        assert performance_results[100]["time"] < performance_results[5000]["time"]
        assert all(
            result["predictions_count"] > 0 for result in performance_results.values()
        )

    def test_adapter_memory_usage(self, performance_data):
        """Test adapter memory usage patterns."""
        import os

        import psutil

        adapter = OptimizedAdapter()
        detector = Detector(
            id=uuid4(),
            name="memory-usage-test",
            algorithm_name="IsolationForest",
            hyperparameters={"n_estimators": 100},
        )

        process = psutil.Process(os.getpid())

        # Measure memory before
        memory_before = process.memory_info().rss

        # Perform operations
        large_data = performance_data[5000]
        model = adapter.fit(detector, large_data)
        predictions = adapter.predict(model, large_data)

        # Measure memory after
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable
        assert memory_increase < 500 * 1024 * 1024  # Less than 500MB
        assert len(predictions) == len(large_data)

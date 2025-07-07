"""Comprehensive tests for JAX adapter."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import (
    DetectorNotFittedError,
    FittingError,
    InvalidAlgorithmError,
)
from pynomaly.domain.value_objects import ContaminationRate


# Test with optional dependency handling
def requires_jax(test_func):
    """Decorator to skip tests if JAX is not available."""
    try:
        import jax
        import jaxlib
        import optax

        return test_func
    except ImportError:
        return pytest.mark.skip(reason="JAX dependencies not available")(test_func)


@requires_jax
class TestJAXAdapterComprehensive:
    """Comprehensive tests for JAX adapter with all algorithms."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        # Generate synthetic anomaly detection data
        np.random.seed(42)

        # Normal data
        normal_data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 200)

        # Anomaly data
        anomaly_data = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 20)

        # Combine data
        all_data = np.vstack([normal_data, anomaly_data])

        # Create DataFrame
        df = pd.DataFrame(all_data, columns=["feature_1", "feature_2"])

        return Dataset(name="test_dataset", data=df)

    @pytest.fixture
    def large_dataset(self):
        """Create a larger dataset for performance testing."""
        np.random.seed(123)

        # Generate 1000 samples with 10 features
        normal_data = np.random.randn(900, 10)
        anomaly_data = np.random.randn(100, 10) + 3  # Shifted anomalies

        all_data = np.vstack([normal_data, anomaly_data])
        df = pd.DataFrame(all_data, columns=[f"feature_{i}" for i in range(10)])

        return Dataset(name="large_test_dataset", data=df)

    @pytest.mark.parametrize(
        "algorithm_name", ["AutoEncoder", "VAE", "IsolationForest", "OCSVM", "LOF"]
    )
    def test_jax_adapter_initialization(self, algorithm_name):
        """Test JAX adapter initialization for all algorithms."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        adapter = JAXAdapter(
            algorithm_name=algorithm_name, contamination_rate=ContaminationRate(0.1)
        )

        assert adapter.algorithm_name == algorithm_name
        assert adapter.contamination_rate.value == 0.1
        assert not adapter.is_fitted
        assert adapter.get_model_info()["framework"] == "JAX"

    def test_jax_adapter_invalid_algorithm(self):
        """Test JAX adapter with invalid algorithm name."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        with pytest.raises(InvalidAlgorithmError):
            JAXAdapter(
                algorithm_name="NonExistentAlgorithm",
                contamination_rate=ContaminationRate(0.1),
            )

    @requires_jax
    def test_autoencoder_training_and_prediction(self, sample_dataset):
        """Test AutoEncoder training and prediction."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        adapter = JAXAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=10,  # Reduced for testing
            learning_rate=0.01,
            hidden_dims=[4, 2],
            encoding_dim=1,
        )

        # Test fitting
        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        # Test prediction
        result = adapter.predict(sample_dataset)

        assert result is not None
        assert result.n_samples == len(sample_dataset.data)
        assert result.anomaly_rate <= 0.2  # Should be reasonable
        assert len(result.anomalies) > 0  # Should detect some anomalies

        # Test model info
        model_info = adapter.get_model_info()
        assert model_info["algorithm"] == "AutoEncoder"
        assert model_info["is_fitted"] is True
        assert "total_params" in model_info

    @requires_jax
    def test_vae_training_and_prediction(self, sample_dataset):
        """Test VAE training and prediction."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        adapter = JAXAdapter(
            algorithm_name="VAE",
            contamination_rate=ContaminationRate(0.15),
            epochs=10,  # Reduced for testing
            learning_rate=0.01,
            hidden_dims=[4],
            latent_dim=2,
            beta=1.0,
        )

        # Test fitting
        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        # Test prediction
        result = adapter.predict(sample_dataset)

        assert result is not None
        assert result.n_samples == len(sample_dataset.data)
        assert result.anomaly_rate <= 0.25  # Should be reasonable

        # Test model info
        model_info = adapter.get_model_info()
        assert model_info["algorithm"] == "VAE"
        assert model_info["beta"] == 1.0
        assert "total_params" in model_info

    @requires_jax
    def test_isolation_forest_training_and_prediction(self, sample_dataset):
        """Test IsolationForest training and prediction."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        adapter = JAXAdapter(
            algorithm_name="IsolationForest",
            contamination_rate=ContaminationRate(0.1),
            n_trees=50,  # Reduced for testing
            max_depth=8,
        )

        # Test fitting
        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        # Test prediction
        result = adapter.predict(sample_dataset)

        assert result is not None
        assert result.n_samples == len(sample_dataset.data)
        assert result.anomaly_rate <= 0.2  # Should be reasonable

        # Test model info
        model_info = adapter.get_model_info()
        assert model_info["algorithm"] == "IsolationForest"
        assert model_info["n_trees"] == 50
        assert model_info["max_depth"] == 8

    @requires_jax
    def test_ocsvm_training_and_prediction(self, sample_dataset):
        """Test OCSVM training and prediction."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        adapter = JAXAdapter(
            algorithm_name="OCSVM",
            contamination_rate=ContaminationRate(0.1),
            gamma=0.1,
            nu=0.05,
        )

        # Test fitting
        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        # Test prediction
        result = adapter.predict(sample_dataset)

        assert result is not None
        assert result.n_samples == len(sample_dataset.data)
        assert result.anomaly_rate <= 0.2  # Should be reasonable

        # Test model info
        model_info = adapter.get_model_info()
        assert model_info["algorithm"] == "OCSVM"
        assert model_info["gamma"] == 0.1
        assert model_info["nu"] == 0.05
        assert "n_support_vectors" in model_info

    @requires_jax
    def test_lof_training_and_prediction(self, sample_dataset):
        """Test LOF training and prediction."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        adapter = JAXAdapter(
            algorithm_name="LOF",
            contamination_rate=ContaminationRate(0.1),
            n_neighbors=10,
        )

        # Test fitting
        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        # Test prediction
        result = adapter.predict(sample_dataset)

        assert result is not None
        assert result.n_samples == len(sample_dataset.data)
        assert result.anomaly_rate <= 0.2  # Should be reasonable

        # Test model info
        model_info = adapter.get_model_info()
        assert model_info["algorithm"] == "LOF"
        assert model_info["n_neighbors"] == 10
        assert "k_used" in model_info
        assert "n_training_samples" in model_info
        assert model_info["n_training_samples"] == len(sample_dataset.data)

    @requires_jax
    def test_prediction_without_fitting(self, sample_dataset):
        """Test prediction without fitting raises appropriate error."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        adapter = JAXAdapter(
            algorithm_name="AutoEncoder", contamination_rate=ContaminationRate(0.1)
        )

        with pytest.raises(DetectorNotFittedError):
            adapter.predict(sample_dataset)

    @requires_jax
    def test_performance_with_large_dataset(self, large_dataset):
        """Test performance with larger dataset."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        adapter = JAXAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=5,  # Reduced for testing performance
            batch_size=64,
            hidden_dims=[8, 4],
            encoding_dim=2,
        )

        # Test fitting performance
        import time

        start_time = time.time()
        adapter.fit(large_dataset)
        fit_time = time.time() - start_time

        assert adapter.is_fitted
        assert fit_time < 30.0  # Should complete within reasonable time

        # Test prediction performance
        start_time = time.time()
        result = adapter.predict(large_dataset)
        predict_time = time.time() - start_time

        assert predict_time < 5.0  # Should be fast for prediction
        assert result.n_samples == 1000

    @requires_jax
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        # Test with empty dataset
        empty_df = pd.DataFrame()
        empty_dataset = Dataset(name="empty", data=empty_df)

        adapter = JAXAdapter(
            algorithm_name="AutoEncoder", contamination_rate=ContaminationRate(0.1)
        )

        with pytest.raises((ValueError, FittingError)):
            adapter.fit(empty_dataset)

        # Test with single sample dataset
        single_df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        single_dataset = Dataset(name="single", data=single_df)

        adapter = JAXAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=1,
        )

        # Should not crash with single sample
        try:
            adapter.fit(single_dataset)
            result = adapter.predict(single_dataset)
            assert result.n_samples == 1
        except (ValueError, FittingError):
            # It's acceptable to fail with single sample
            pass

    @requires_jax
    def test_different_contamination_rates(self, sample_dataset):
        """Test different contamination rates."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        contamination_rates = [0.05, 0.1, 0.2, 0.3]

        for rate in contamination_rates:
            adapter = JAXAdapter(
                algorithm_name="IsolationForest",
                contamination_rate=ContaminationRate(rate),
                n_trees=20,  # Reduced for testing
            )

            adapter.fit(sample_dataset)
            result = adapter.predict(sample_dataset)

            # Higher contamination rate should generally lead to more detected anomalies
            assert result.anomaly_rate >= 0.0
            assert result.anomaly_rate <= rate * 2  # Allow some flexibility

    @requires_jax
    def test_reproducibility_with_random_seed(self, sample_dataset):
        """Test reproducibility with fixed random seed."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        # Create two identical adapters with same seed
        adapter1 = JAXAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=5,
            random_seed=42,
        )

        adapter2 = JAXAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=5,
            random_seed=42,
        )

        # Fit both adapters
        adapter1.fit(sample_dataset)
        adapter2.fit(sample_dataset)

        # Predictions should be similar (though not necessarily identical due to JAX's randomness)
        result1 = adapter1.predict(sample_dataset)
        result2 = adapter2.predict(sample_dataset)

        # Results should be in similar range
        assert abs(result1.anomaly_rate - result2.anomaly_rate) < 0.1

    @requires_jax
    def test_algorithm_parameter_variations(self, sample_dataset):
        """Test different algorithm parameter configurations."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        # Test AutoEncoder with different architectures
        configurations = [
            {"hidden_dims": [4], "encoding_dim": 1},
            {"hidden_dims": [6, 3], "encoding_dim": 2},
            {"hidden_dims": [8, 4, 2], "encoding_dim": 1},
        ]

        for config in configurations:
            adapter = JAXAdapter(
                algorithm_name="AutoEncoder",
                contamination_rate=ContaminationRate(0.1),
                epochs=3,  # Reduced for testing
                **config,
            )

            adapter.fit(sample_dataset)
            result = adapter.predict(sample_dataset)

            assert result is not None
            assert result.n_samples > 0

            model_info = adapter.get_model_info()
            assert model_info["hidden_dims"] == config["hidden_dims"]
            assert model_info["encoding_dim"] == config["encoding_dim"]

    def test_jax_availability_handling(self):
        """Test handling when JAX is not available."""
        # Mock JAX as unavailable
        with patch("pynomaly.infrastructure.adapters.jax_adapter.HAS_JAX", False):
            from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

            with pytest.raises(ImportError, match="JAX is not installed"):
                JAXAdapter(
                    algorithm_name="AutoEncoder",
                    contamination_rate=ContaminationRate(0.1),
                )

    @requires_jax
    def test_list_available_algorithms(self):
        """Test listing available algorithms."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        algorithms = JAXAdapter.list_available_algorithms()

        assert isinstance(algorithms, list)
        assert len(algorithms) > 0
        assert "AutoEncoder" in algorithms
        assert "VAE" in algorithms
        assert "IsolationForest" in algorithms

    @requires_jax
    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        # Test AutoEncoder info
        ae_info = JAXAdapter.get_algorithm_info("AutoEncoder")
        assert ae_info["type"] == "Neural Network"
        assert ae_info["unsupervised"] is True
        assert ae_info["gpu_support"] is True
        assert ae_info["jit_compiled"] is True
        assert "parameters" in ae_info

        # Test VAE info
        vae_info = JAXAdapter.get_algorithm_info("VAE")
        assert vae_info["type"] == "Neural Network"
        assert "latent_dim" in vae_info["parameters"]
        assert "beta" in vae_info["parameters"]

        # Test invalid algorithm
        with pytest.raises(InvalidAlgorithmError):
            JAXAdapter.get_algorithm_info("NonExistentAlgorithm")

    @requires_jax
    def test_memory_efficiency(self, large_dataset):
        """Test memory efficiency with large dataset."""
        import os

        import psutil

        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        adapter = JAXAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=3,
            batch_size=128,  # Larger batch size for efficiency
            hidden_dims=[6, 3],
            encoding_dim=2,
        )

        # Fit and predict
        adapter.fit(large_dataset)
        result = adapter.predict(large_dataset)

        # Check memory usage didn't explode
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert memory_increase < 500  # Should not use more than 500MB additional
        assert result.n_samples == 1000

    @requires_jax
    def test_batch_processing_consistency(self, sample_dataset):
        """Test that batch processing gives consistent results."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        # Test with different batch sizes
        batch_sizes = [16, 32, 64]
        results = []

        for batch_size in batch_sizes:
            adapter = JAXAdapter(
                algorithm_name="AutoEncoder",
                contamination_rate=ContaminationRate(0.1),
                epochs=5,
                batch_size=batch_size,
                random_seed=42,  # Fixed seed for consistency
            )

            adapter.fit(sample_dataset)
            result = adapter.predict(sample_dataset)
            results.append(result.anomaly_rate)

        # Results should be reasonably consistent across batch sizes
        max_diff = max(results) - min(results)
        assert max_diff < 0.2  # Allow some variation but not too much

    @requires_jax
    def test_feature_scaling_robustness(self, sample_dataset):
        """Test robustness to different feature scales."""
        from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter

        # Create datasets with different scales
        original_data = sample_dataset.data.copy()
        scaled_data = original_data * 1000  # Scale up
        tiny_data = original_data * 0.001  # Scale down

        datasets = [
            Dataset(name="original", data=original_data),
            Dataset(name="scaled", data=scaled_data),
            Dataset(name="tiny", data=tiny_data),
        ]

        results = []

        for dataset in datasets:
            adapter = JAXAdapter(
                algorithm_name="AutoEncoder",
                contamination_rate=ContaminationRate(0.1),
                epochs=5,
                random_seed=42,
            )

            adapter.fit(dataset)
            result = adapter.predict(dataset)
            results.append(result.anomaly_rate)

        # Should handle different scales reasonably well
        # (JAX adapter normalizes data, so results should be somewhat consistent)
        max_diff = max(results) - min(results)
        assert max_diff < 0.3  # Allow some variation but adapter should be robust

"""Test TensorFlow adapter functionality."""

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import AdapterError, InvalidAlgorithmError
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters.tensorflow_adapter import (
    HAS_TENSORFLOW,
    TensorFlowAdapter,
)


class TestTensorFlowAdapter:
    """Test TensorFlow adapter functionality."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(0, 1, 100),
                "feature3": np.random.normal(0, 1, 100),
            }
        )
        return Dataset(name="test_data", data=data)

    @pytest.mark.skipif(HAS_TENSORFLOW, reason="Testing without TensorFlow")
    def test_tensorflow_unavailable_raises_error(self):
        """Test that adapter raises error when TensorFlow is not available."""
        with pytest.raises(AdapterError, match="TensorFlow is not available"):
            TensorFlowAdapter(algorithm_name="AutoEncoder")

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_init_with_valid_algorithm(self):
        """Test initialization with valid algorithm."""
        adapter = TensorFlowAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=10,
            batch_size=32,
        )

        assert adapter.algorithm_name == "AutoEncoder"
        assert adapter.name == "TensorFlow_AutoEncoder"
        assert adapter.contamination_rate.value == 0.1
        assert not adapter.is_fitted
        assert adapter.requires_fitting
        assert not adapter.supports_streaming
        assert adapter.epochs == 10
        assert adapter.batch_size == 32

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_init_with_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with pytest.raises(InvalidAlgorithmError):
            TensorFlowAdapter(algorithm_name="InvalidAlgorithm")

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_get_supported_algorithms(self):
        """Test getting supported algorithms."""
        algorithms = TensorFlowAdapter.get_supported_algorithms()

        expected = ["AutoEncoder", "VAE", "DeepSVDD"]
        assert all(alg in algorithms for alg in expected)

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        info = TensorFlowAdapter.get_algorithm_info("AutoEncoder")

        assert (
            info["description"]
            == "Deep autoencoder for anomaly detection using reconstruction error"
        )
        assert info["type"] == "Neural Network"
        assert info["gpu_support"] == True
        assert info["distributed_training"] == True
        assert "parameters" in info
        assert "encoding_dim" in info["parameters"]

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_device_detection(self):
        """Test device detection functionality."""
        adapter = TensorFlowAdapter(algorithm_name="AutoEncoder")

        # Should detect CPU or GPU
        assert adapter.device_type in ["CPU", "GPU"]
        print(f"Detected device: {adapter.device_type}")

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_fit_and_predict_autoencoder(self, sample_dataset):
        """Test fitting and prediction with AutoEncoder."""
        adapter = TensorFlowAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=3,  # Small number for fast testing
            batch_size=32,
            encoding_dim=8,
            hidden_layers=[16, 12],
            validation_split=0.1,
        )

        # Test fit
        adapter.fit(sample_dataset)
        assert adapter.is_fitted
        assert adapter.model is not None

        # Test predict
        result = adapter.detect(sample_dataset)

        assert len(result.scores) == len(sample_dataset.data)
        assert len(result.labels) == len(sample_dataset.data)
        assert result.threshold > 0
        assert "algorithm" in result.metadata
        assert result.metadata["algorithm"] == "AutoEncoder"
        assert result.metadata["framework"] == "tensorflow"

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_score_method(self, sample_dataset):
        """Test score method."""
        adapter = TensorFlowAdapter(
            algorithm_name="AutoEncoder", epochs=2, encoding_dim=4, hidden_layers=[8]
        )

        adapter.fit(sample_dataset)
        scores = adapter.score(sample_dataset)

        assert len(scores) == len(sample_dataset.data)
        assert all(hasattr(score, "value") for score in scores)
        assert all(hasattr(score, "confidence") for score in scores)

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_fit_detect_method(self, sample_dataset):
        """Test fit_detect method."""
        adapter = TensorFlowAdapter(
            algorithm_name="VAE", epochs=2, latent_dim=4, hidden_layers=[8]
        )

        result = adapter.fit_detect(sample_dataset)

        assert adapter.is_fitted
        assert len(result.scores) == len(sample_dataset.data)

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_params_methods(self):
        """Test get_params and set_params methods."""
        adapter = TensorFlowAdapter(
            algorithm_name="AutoEncoder",
            epochs=100,
            learning_rate=0.001,
            encoding_dim=16,
        )

        params = adapter.get_params()
        assert params["epochs"] == 100
        assert params["learning_rate"] == 0.001
        assert params["encoding_dim"] == 16

        adapter.set_params(epochs=50, batch_size=64, encoding_dim=32)
        updated_params = adapter.get_params()
        assert updated_params["epochs"] == 50
        assert updated_params["batch_size"] == 64
        assert updated_params["encoding_dim"] == 32

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_predict_without_fitting_raises_error(self, sample_dataset):
        """Test that prediction without fitting raises error."""
        adapter = TensorFlowAdapter(algorithm_name="AutoEncoder")

        with pytest.raises(Exception):  # Should be DetectorNotFittedError
            adapter.detect(sample_dataset)

        with pytest.raises(Exception):  # Should be DetectorNotFittedError
            adapter.score(sample_dataset)

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_vae_algorithm(self, sample_dataset):
        """Test VAE algorithm specifically."""
        adapter = TensorFlowAdapter(
            algorithm_name="VAE",
            epochs=2,
            latent_dim=4,
            hidden_layers=[8, 6],
            beta=0.5,  # VAE specific parameter
        )

        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        result = adapter.detect(sample_dataset)
        assert len(result.scores) == len(sample_dataset.data)

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_deep_svdd_algorithm(self, sample_dataset):
        """Test DeepSVDD algorithm specifically."""
        adapter = TensorFlowAdapter(
            algorithm_name="DeepSVDD", epochs=2, output_dim=4, hidden_layers=[8]
        )

        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        result = adapter.detect(sample_dataset)
        assert len(result.scores) == len(sample_dataset.data)

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_get_model_info(self, sample_dataset):
        """Test getting model information."""
        adapter = TensorFlowAdapter(
            algorithm_name="AutoEncoder", epochs=2, encoding_dim=4
        )

        # Before fitting
        info = adapter.get_model_info()
        assert info["is_fitted"] == False
        assert info["algorithm"] == "AutoEncoder"
        assert info["has_tensorflow"] == True

        # After fitting
        adapter.fit(sample_dataset)
        info = adapter.get_model_info()
        assert info["is_fitted"] == True
        assert "total_params" in info
        assert "training_epochs" in info

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_early_stopping(self, sample_dataset):
        """Test early stopping functionality."""
        adapter = TensorFlowAdapter(
            algorithm_name="AutoEncoder",
            epochs=50,  # Large number to test early stopping
            early_stopping_patience=3,
            validation_split=0.2,
            encoding_dim=4,
        )

        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        # Should have stopped early
        info = adapter.get_model_info()
        assert info["training_epochs"] < 50  # Should stop before 50 epochs

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_confidence_calculation(self):
        """Test confidence calculation."""
        adapter = TensorFlowAdapter(algorithm_name="AutoEncoder")
        adapter.threshold_value = 0.5

        # Test below threshold
        confidence = adapter._calculate_confidence(0.3)
        assert 0.5 < confidence <= 1.0

        # Test above threshold
        confidence = adapter._calculate_confidence(0.7)
        assert 0.5 <= confidence <= 1.0

    def test_prepare_data_method(self, sample_dataset):
        """Test data preparation method."""
        if not HAS_TENSORFLOW:
            pytest.skip("TensorFlow not available")

        adapter = TensorFlowAdapter(algorithm_name="AutoEncoder")
        prepared_data = adapter._prepare_data(sample_dataset)

        # Should return TensorFlow tensor or numpy array
        assert hasattr(prepared_data, "shape")
        assert prepared_data.shape[0] == len(sample_dataset.data)
        assert prepared_data.shape[1] == 3  # 3 features

    @pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
    def test_list_available_algorithms(self):
        """Test listing available algorithms."""
        algorithms = TensorFlowAdapter.list_available_algorithms()
        assert isinstance(algorithms, list)
        assert len(algorithms) > 0

    def test_list_available_algorithms_without_tensorflow(self):
        """Test listing algorithms when TensorFlow is not available."""
        if HAS_TENSORFLOW:
            pytest.skip("TensorFlow is available")

        algorithms = TensorFlowAdapter.list_available_algorithms()
        assert algorithms == []


@pytest.mark.skipif(HAS_TENSORFLOW, reason="Testing TensorFlow unavailable scenario")
def test_tensorflow_unavailable_error():
    """Test error when TensorFlow is not available."""
    from pynomaly.domain.exceptions import AdapterError
    from pynomaly.infrastructure.adapters.tensorflow_adapter import TensorFlowAdapter

    with pytest.raises(AdapterError, match="TensorFlow is not available"):
        TensorFlowAdapter(algorithm_name="AutoEncoder")

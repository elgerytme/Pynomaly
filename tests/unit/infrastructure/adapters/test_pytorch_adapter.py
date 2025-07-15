"""Test PyTorch adapter functionality."""

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import AdapterError, AlgorithmNotFoundError
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters.pytorch_adapter import (
    TORCH_AVAILABLE,
    PyTorchAdapter,
)


class TestPyTorchAdapter:
    """Test PyTorch adapter functionality."""

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

    @pytest.mark.skipif(TORCH_AVAILABLE, reason="Testing without PyTorch")
    def test_pytorch_unavailable_raises_error(self):
        """Test that adapter raises error when PyTorch is not available."""
        with pytest.raises(AdapterError, match="PyTorch is not available"):
            PyTorchAdapter(algorithm_name="AutoEncoder")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_init_with_valid_algorithm(self):
        """Test initialization with valid algorithm."""
        adapter = PyTorchAdapter(
            algorithm_name="AutoEncoder", contamination_rate=ContaminationRate(0.1)
        )

        assert adapter.algorithm_name == "AutoEncoder"
        assert adapter.name == "PyTorch_AutoEncoder"
        assert adapter.contamination_rate.value == 0.1
        assert not adapter.is_fitted
        assert adapter.requires_fitting
        assert not adapter.supports_streaming

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_init_with_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with pytest.raises(AlgorithmNotFoundError):
            PyTorchAdapter(algorithm_name="InvalidAlgorithm")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_supported_algorithms(self):
        """Test getting supported algorithms."""
        algorithms = PyTorchAdapter.get_supported_algorithms()

        expected = ["AutoEncoder", "VAE", "DeepSVDD", "DAGMM", "LSTMAutoEncoder"]
        assert all(alg in algorithms for alg in expected)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_algorithm_info(self):
        """Test getting algorithm information."""
        info = PyTorchAdapter.get_algorithm_info("AutoEncoder")

        assert info["name"] == "AutoEncoder"
        assert info["type"] == "Deep Learning"
        assert "parameters" in info
        assert "hidden_dims" in info["parameters"]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_fit_and_predict_autoencoder(self, sample_dataset):
        """Test fitting and prediction with AutoEncoder."""
        adapter = PyTorchAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=5,  # Small number for fast testing
            batch_size=32,
            hidden_dims=[10, 5],
            latent_dim=3,
        )

        # Test fit
        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        # Test predict
        result = adapter.detect(sample_dataset)

        assert len(result.scores) == len(sample_dataset.data)
        assert len(result.labels) == len(sample_dataset.data)
        assert result.threshold > 0
        assert "algorithm" in result.metadata
        assert result.metadata["algorithm"] == "AutoEncoder"

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_score_method(self, sample_dataset):
        """Test score method."""
        adapter = PyTorchAdapter(
            algorithm_name="AutoEncoder", epochs=3, hidden_dims=[8], latent_dim=2
        )

        adapter.fit(sample_dataset)
        scores = adapter.score(sample_dataset)

        assert len(scores) == len(sample_dataset.data)
        assert all(hasattr(score, "value") for score in scores)
        assert all(hasattr(score, "confidence") for score in scores)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_fit_detect_method(self, sample_dataset):
        """Test fit_detect method."""
        adapter = PyTorchAdapter(
            algorithm_name="VAE", epochs=3, hidden_dims=[8], latent_dim=2
        )

        result = adapter.fit_detect(sample_dataset)

        assert adapter.is_fitted
        assert len(result.scores) == len(sample_dataset.data)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_params_methods(self):
        """Test get_params and set_params methods."""
        adapter = PyTorchAdapter(
            algorithm_name="AutoEncoder", epochs=100, learning_rate=0.001
        )

        params = adapter.get_params()
        assert params["epochs"] == 100
        assert params["learning_rate"] == 0.001

        adapter.set_params(epochs=50, batch_size=64)
        updated_params = adapter.get_params()
        assert updated_params["epochs"] == 50
        assert updated_params["batch_size"] == 64

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_predict_without_fitting_raises_error(self, sample_dataset):
        """Test that prediction without fitting raises error."""
        adapter = PyTorchAdapter(algorithm_name="AutoEncoder")

        with pytest.raises(AdapterError, match="Model must be fitted"):
            adapter.detect(sample_dataset)

        with pytest.raises(AdapterError, match="Model must be fitted"):
            adapter.score(sample_dataset)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_dagmm_algorithm(self, sample_dataset):
        """Test DAGMM algorithm specifically."""
        adapter = PyTorchAdapter(
            algorithm_name="DAGMM",
            epochs=3,
            hidden_dims=[6, 4],
            latent_dim=2,
            n_gmm=2,  # DAGMM specific parameter
        )

        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        result = adapter.detect(sample_dataset)
        assert len(result.scores) == len(sample_dataset.data)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_deep_svdd_algorithm(self, sample_dataset):
        """Test DeepSVDD algorithm specifically."""
        adapter = PyTorchAdapter(
            algorithm_name="DeepSVDD", epochs=3, hidden_dims=[8], latent_dim=4
        )

        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        result = adapter.detect(sample_dataset)
        assert len(result.scores) == len(sample_dataset.data)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_autoencoder_algorithm(self, sample_dataset):
        """Test LSTM AutoEncoder algorithm specifically."""
        adapter = PyTorchAdapter(
            algorithm_name="LSTMAutoEncoder",
            epochs=3,
            hidden_dim=16,
            num_layers=1,
            sequence_length=5,
            dropout=0.1,
        )

        adapter.fit(sample_dataset)
        assert adapter.is_fitted

        result = adapter.detect(sample_dataset)
        assert len(result.scores) == len(sample_dataset.data)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_lstm_algorithm_info(self):
        """Test getting LSTM algorithm information."""
        info = PyTorchAdapter.get_algorithm_info("LSTMAutoEncoder")

        assert info["name"] == "LSTM AutoEncoder"
        assert info["type"] == "Deep Learning"
        assert "parameters" in info
        assert "hidden_dim" in info["parameters"]
        assert "sequence_length" in info["parameters"]
        assert "temporal_patterns" in info["suitable_for"]

    def test_prepare_data_method(self, sample_dataset):
        """Test data preparation method works without PyTorch."""
        # Create a simple mock adapter to test data preparation
        from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter

        if not TORCH_AVAILABLE:
            # Skip testing _prepare_data when PyTorch is not available
            # since the adapter won't initialize
            pytest.skip("PyTorch not available")

        adapter = PyTorchAdapter(algorithm_name="AutoEncoder")
        prepared_data = adapter._prepare_data(sample_dataset)

        assert isinstance(prepared_data, np.ndarray)
        assert prepared_data.shape[0] == len(sample_dataset.data)
        assert prepared_data.shape[1] == 3  # 3 features

        # Check that data is standardized (approximately mean 0, std 1)
        assert abs(np.mean(prepared_data)) < 0.1
        assert abs(np.std(prepared_data) - 1.0) < 0.1

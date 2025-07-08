"""Integration tests for PyTorchAdapter with AutoEncoder-based anomaly detection."""

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from pynomaly.infrastructure.adapters.algorithm_factory import AlgorithmFactory


# Optional PyTorch imports with fallbacks
try:
    import torch
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    PYTORCH_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    torch = None
    PYTORCH_AVAILABLE = False
    SKLEARN_AVAILABLE = False


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
@pytest.mark.torch
class TestPyTorchAdapter:
    """Test PyTorchAdapter with AutoEncoder-based anomaly detection."""

    @pytest.fixture
    def mnist_subset(self):
        """Create a subset of MNIST data for testing."""
        # Fetch MNIST data (small subset for testing)
        mnist = fetch_openml('mnist_784', version=1, data_home='./data', as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        
        # Take a small subset for fast testing
        subset_size = 2000
        X_subset = X[:subset_size]
        y_subset = y[:subset_size]
        
        # Normalize data
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_subset)
        
        return X_normalized, y_subset

    @pytest.fixture
    def anomalous_data(self, mnist_subset):
        """Create normal and anomalous data from MNIST subset."""
        X, y = mnist_subset
        
        # Use digits 0-7 as normal, 8-9 as anomalies
        normal_mask = y < 8
        anomaly_mask = y >= 8
        
        X_normal = X[normal_mask]
        X_anomaly = X[anomaly_mask]
        
        # Create training data (only normal)
        X_train = X_normal[:1200]  # Use first 1200 normal samples for training
        
        # Create test data (mix of normal and anomalies)
        X_test_normal = X_normal[1200:1400]  # 200 normal samples
        X_test_anomaly = X_anomaly[:50]       # 50 anomalous samples
        
        # Combine test data
        X_test = np.vstack([X_test_normal, X_test_anomaly])
        y_test = np.hstack([np.zeros(len(X_test_normal)), np.ones(len(X_test_anomaly))])
        
        return X_train, X_test, y_test

    def test_pytorch_adapter_creation(self):
        """Test PyTorchAdapter can be created through factory."""
        factory = AlgorithmFactory()
        
        # Test creating autoencoder
        detector = factory.create_detector(
            algorithm_name="autoencoder",
            library="pytorch"
        )
        
        assert detector is not None
        assert detector.algorithm_name == "pytorch_autoencoder"

    def test_autoencoder_training_cpu(self, anomalous_data):
        """Test AutoEncoder training on CPU within 5 minutes."""
        X_train, X_test, y_test = anomalous_data
        
        factory = AlgorithmFactory()
        
        # Create autoencoder with minimal config for fast training
        detector = factory.create_detector(
            algorithm_name="autoencoder",
            library="pytorch",
            model_config={
                "hidden_dims": [32, 16],  # Smaller network for speed
                "latent_dim": 8,
                "epochs": 50,             # Fewer epochs for speed
                "batch_size": 64,
                "early_stopping_patience": 5,
                "contamination": 0.2      # Expected 20% anomalies
            }
        )
        
        # Measure training time
        start_time = time.time()
        detector.train(X_train)
        training_time = time.time() - start_time
        
        # Should finish within 5 minutes (300 seconds)
        assert training_time < 300, f"Training took {training_time:.2f}s, expected < 300s"
        
        # Test prediction
        predictions = detector.predict(X_test)
        scores = detector.decision_function(X_test)
        
        # Verify outputs
        assert len(predictions) == len(X_test)
        assert len(scores) == len(X_test)
        assert predictions.dtype == int
        assert np.all((predictions == 0) | (predictions == 1))
        
        # Calculate anomaly detection rate
        anomaly_indices = np.where(y_test == 1)[0]
        detected_anomalies = np.sum(predictions[anomaly_indices])
        detection_rate = detected_anomalies / len(anomaly_indices)
        
        # Should detect at least 80% of injected anomalies
        assert detection_rate >= 0.8, f"Detection rate {detection_rate:.2f} < 0.8"
        
        print(f"Training time: {training_time:.2f}s")
        print(f"Anomaly detection rate: {detection_rate:.2f}")

    def test_save_load_functionality(self, anomalous_data):
        """Test save and load functionality."""
        X_train, X_test, y_test = anomalous_data
        
        factory = AlgorithmFactory()
        
        # Create and train detector
        detector = factory.create_detector(
            algorithm_name="autoencoder",
            library="pytorch",
            model_config={
                "hidden_dims": [16, 8],
                "latent_dim": 4,
                "epochs": 10,  # Very few epochs for speed
                "batch_size": 32
            }
        )
        
        detector.train(X_train[:200])  # Use even smaller dataset for speed
        
        # Get predictions before saving
        predictions_before = detector.predict(X_test[:50])
        scores_before = detector.decision_function(X_test[:50])
        
        # Save model
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pth"
            detector.save(model_path)
            
            # Create new detector and load
            new_detector = factory.create_detector(
                algorithm_name="autoencoder",
                library="pytorch"
            )
            new_detector.load(model_path)
            
            # Get predictions after loading
            predictions_after = new_detector.predict(X_test[:50])
            scores_after = new_detector.decision_function(X_test[:50])
            
            # Should be identical
            np.testing.assert_array_equal(predictions_before, predictions_after)
            np.testing.assert_array_almost_equal(scores_before, scores_after, decimal=5)

    def test_vae_algorithm(self, anomalous_data):
        """Test VAE algorithm variant."""
        X_train, X_test, y_test = anomalous_data
        
        factory = AlgorithmFactory()
        
        detector = factory.create_detector(
            algorithm_name="vae",
            library="pytorch",
            model_config={
                "encoder_dims": [32, 16],
                "latent_dim": 8,
                "decoder_dims": [16, 32],
                "epochs": 20,
                "batch_size": 64,
                "beta": 1.0
            }
        )
        
        # Train and test
        detector.train(X_train[:500])  # Smaller dataset for speed
        predictions = detector.predict(X_test[:100])
        
        assert len(predictions) == 100
        assert detector.algorithm_name == "pytorch_vae"

    def test_cpu_fallback(self, anomalous_data):
        """Test that adapter works on CPU even when CUDA is available."""
        X_train, X_test, y_test = anomalous_data
        
        factory = AlgorithmFactory()
        
        # Explicitly request CPU device
        detector = factory.create_detector(
            algorithm_name="autoencoder",
            library="pytorch",
            device="cpu",
            model_config={
                "hidden_dims": [16],
                "latent_dim": 4,
                "epochs": 5,
                "batch_size": 32
            }
        )
        
        # Should work on CPU
        detector.train(X_train[:200])
        predictions = detector.predict(X_test[:50])
        
        assert len(predictions) == 50
        # Verify it's actually using CPU
        assert str(detector.device) == "cpu"

    def test_model_info(self, anomalous_data):
        """Test model information retrieval."""
        X_train, _, _ = anomalous_data
        
        factory = AlgorithmFactory()
        
        detector = factory.create_detector(
            algorithm_name="autoencoder",
            library="pytorch",
            model_config={"epochs": 5}
        )
        
        # Before training
        info_before = detector.get_model_info()
        assert info_before["is_trained"] is False
        assert "total_parameters" not in info_before
        
        # After training
        detector.train(X_train[:100])
        info_after = detector.get_model_info()
        assert info_after["is_trained"] is True
        assert "total_parameters" in info_after
        assert info_after["total_parameters"] > 0

    def test_factory_lists_pytorch_algorithms(self):
        """Test that factory correctly lists PyTorch algorithms."""
        factory = AlgorithmFactory()
        
        all_algorithms = factory.list_all_algorithms()
        pytorch_algorithms = factory.list_algorithms_for_library("pytorch")
        
        assert "autoencoder" in all_algorithms
        assert "vae" in all_algorithms
        assert "lstm" in all_algorithms
        
        assert "autoencoder" in pytorch_algorithms
        assert "vae" in pytorch_algorithms
        assert "lstm" in pytorch_algorithms

    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        factory = AlgorithmFactory()
        
        # Test invalid algorithm
        with pytest.raises(Exception):
            factory.create_detector(
                algorithm_name="invalid_algorithm",
                library="pytorch"
            )
        
        # Test training without data
        detector = factory.create_detector(
            algorithm_name="autoencoder",
            library="pytorch"
        )
        
        with pytest.raises(Exception):
            detector.predict(np.random.randn(10, 5))  # Predict before training

    def test_multiple_algorithms_performance(self, anomalous_data):
        """Test performance comparison of multiple PyTorch algorithms."""
        X_train, X_test, y_test = anomalous_data
        
        factory = AlgorithmFactory()
        algorithms = ["autoencoder", "vae"]
        results = {}
        
        for algorithm in algorithms:
            detector = factory.create_detector(
                algorithm_name=algorithm,
                library="pytorch",
                model_config={
                    "epochs": 15,
                    "batch_size": 64,
                    "contamination": 0.2
                }
            )
            
            start_time = time.time()
            detector.train(X_train[:800])
            training_time = time.time() - start_time
            
            predictions = detector.predict(X_test)
            detection_rate = np.sum(predictions[y_test == 1]) / np.sum(y_test == 1)
            
            results[algorithm] = {
                "training_time": training_time,
                "detection_rate": detection_rate
            }
            
            # Each should finish reasonably quickly and detect anomalies
            assert training_time < 120  # 2 minutes max
            assert detection_rate > 0.5  # At least 50% detection
        
        print(f"Performance results: {results}")


@pytest.mark.skipif(PYTORCH_AVAILABLE, reason="Testing PyTorch unavailable scenario")
class TestPyTorchAdapterWithoutPyTorch:
    """Test PyTorchAdapter behavior when PyTorch is not available."""
    
    def test_factory_without_pytorch(self):
        """Test factory behavior when PyTorch is not available."""
        factory = AlgorithmFactory()
        
        # Should not include PyTorch algorithms
        all_algorithms = factory.list_all_algorithms()
        assert "autoencoder" not in all_algorithms
        
        # Should return empty list for PyTorch library
        pytorch_algorithms = factory.list_algorithms_for_library("pytorch")
        assert pytorch_algorithms == []
        
        # Should raise error when trying to create PyTorch detector
        with pytest.raises(Exception):
            factory.create_detector(
                algorithm_name="autoencoder",
                library="pytorch"
            )

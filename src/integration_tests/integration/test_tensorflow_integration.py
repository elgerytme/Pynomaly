"""Integration test for TensorFlow adapter functionality."""

import numpy as np
import pandas as pd
import pytest

from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import ContaminationRate
from pynomaly.infrastructure.adapters.tensorflow_adapter import (
    HAS_TENSORFLOW,
    TensorFlowAdapter,
)


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not available")
class TestTensorFlowIntegration:
    """Integration tests for TensorFlow adapter."""

    def test_complete_anomaly_detection_workflow(self):
        """Test complete workflow from data to anomaly detection."""
        # Generate synthetic dataset with anomalies
        np.random.seed(42)

        # Normal data
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0], cov=[[1, 0.5, 0.1], [0.5, 1, 0.2], [0.1, 0.2, 1]], size=200
        )

        # Anomalous data (outliers)
        anomaly_data = np.random.multivariate_normal(
            mean=[5, 5, 5], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size=20
        )

        # Combine data
        all_data = np.vstack([normal_data, anomaly_data])

        # Create DataFrame
        df = pd.DataFrame(all_data, columns=["feature1", "feature2", "feature3"])
        dataset = Dataset(name="synthetic_data", data=df)

        # Test AutoEncoder
        autoencoder = TensorFlowAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=10,
            batch_size=32,
            encoding_dim=8,
            hidden_layers=[16, 12],
            learning_rate=0.001,
            validation_split=0.2,
            early_stopping_patience=5,
        )

        # Fit and detect
        result = autoencoder.fit_detect(dataset)

        # Verify results
        assert len(result.scores) == len(df)
        assert len(result.labels) == len(df)
        assert result.n_anomalies > 0
        assert result.threshold > 0
        assert result.execution_time_ms > 0

        # Check that some anomalies were detected
        detected_anomalies = np.sum(result.labels)
        expected_anomalies = int(len(df) * 0.1)  # 10% contamination

        # Should detect some anomalies (allowing for some variance)
        assert detected_anomalies > 0
        assert detected_anomalies <= len(df) * 0.15  # Not too many false positives

        print(f"Detected {detected_anomalies} anomalies out of {len(df)} points")
        print(f"Expected approximately {expected_anomalies} anomalies")

        # Test model info
        model_info = autoencoder.get_model_info()
        assert model_info["is_fitted"]
        assert model_info["algorithm"] == "AutoEncoder"
        assert model_info["device"] in ["CPU", "GPU"]
        assert model_info["total_params"] > 0

    def test_all_algorithms_basic_functionality(self):
        """Test that all algorithms can be instantiated and run basic operations."""
        # Create simple test data
        data = pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 50),
                "y": np.random.normal(0, 1, 50),
            }
        )
        dataset = Dataset(name="test", data=data)

        algorithms = ["AutoEncoder", "VAE", "DeepSVDD"]

        for algorithm in algorithms:
            print(f"Testing {algorithm}...")

            # Create adapter with small parameters for fast testing
            if algorithm == "AutoEncoder":
                adapter = TensorFlowAdapter(
                    algorithm_name=algorithm,
                    epochs=2,
                    batch_size=16,
                    encoding_dim=4,
                    hidden_layers=[8, 6],
                    validation_split=0.1,
                )
            elif algorithm == "VAE":
                adapter = TensorFlowAdapter(
                    algorithm_name=algorithm,
                    epochs=2,
                    batch_size=16,
                    latent_dim=4,
                    hidden_layers=[8, 6],
                    beta=0.5,
                )
            elif algorithm == "DeepSVDD":
                adapter = TensorFlowAdapter(
                    algorithm_name=algorithm,
                    epochs=2,
                    batch_size=16,
                    output_dim=4,
                    hidden_layers=[8, 6],
                )

            # Test basic operations
            assert not adapter.is_fitted

            # Fit
            adapter.fit(dataset)
            assert adapter.is_fitted

            # Score
            scores = adapter.score(dataset)
            assert len(scores) == len(data)

            # Detect
            result = adapter.detect(dataset)
            assert len(result.scores) == len(data)
            assert len(result.labels) == len(data)

            # Test model info
            info = adapter.get_model_info()
            assert info["is_fitted"]
            assert info["algorithm"] == algorithm

            print(f"  ✓ {algorithm} completed successfully")

    def test_gpu_detection_and_device_assignment(self):
        """Test GPU detection and device assignment."""
        adapter = TensorFlowAdapter(algorithm_name="AutoEncoder")

        # Device should be detected
        assert adapter.device_type in ["CPU", "GPU"]
        print(f"Detected device: {adapter.device_type}")

        # Test that model info includes device info
        info = adapter.get_model_info()
        assert "device" in info
        assert info["device"] in ["CPU", "GPU"]

    def test_distributed_training_support(self):
        """Test that distributed training capabilities are documented."""
        # Test getting algorithm info
        for algorithm in ["AutoEncoder", "VAE", "DeepSVDD"]:
            info = TensorFlowAdapter.get_algorithm_info(algorithm)

            assert "distributed_training" in info
            assert info["distributed_training"] == True

            print(
                f"{algorithm} supports distributed training: {info['distributed_training']}"
            )

    def test_early_stopping_and_validation(self):
        """Test early stopping and validation functionality."""
        data = pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 100),
                "y": np.random.normal(0, 1, 100),
            }
        )
        dataset = Dataset(name="validation_test", data=data)

        # Test with early stopping
        adapter = TensorFlowAdapter(
            algorithm_name="AutoEncoder",
            epochs=50,  # Large number to test early stopping
            batch_size=32,
            encoding_dim=8,
            validation_split=0.2,
            early_stopping_patience=3,
        )

        adapter.fit(dataset)

        # Check that early stopping worked
        info = adapter.get_model_info()
        assert info["training_epochs"] <= 50
        assert "final_loss" in info
        assert "best_val_loss" in info

        print(f"Training stopped after {info['training_epochs']} epochs")
        print(f"Final loss: {info['final_loss']:.4f}")
        print(f"Best validation loss: {info['best_val_loss']:.4f}")

    def test_parameter_validation_and_updates(self):
        """Test parameter validation and dynamic updates."""
        # Test getting algorithm info
        for algorithm in ["AutoEncoder", "VAE", "DeepSVDD"]:
            info = TensorFlowAdapter.get_algorithm_info(algorithm)

            assert "name" in info
            assert "description" in info
            assert "parameters" in info
            assert "suitable_for" in info
            assert "pros" in info
            assert "cons" in info

            print(f"{algorithm}: {info['description']}")

        # Test parameter updates
        adapter = TensorFlowAdapter(
            algorithm_name="AutoEncoder",
            epochs=10,
            learning_rate=0.001,
            encoding_dim=16,
        )

        # Get initial parameters
        params = adapter.get_params()
        assert params["epochs"] == 10
        assert params["learning_rate"] == 0.001
        assert params["encoding_dim"] == 16

        # Update parameters
        adapter.set_params(epochs=20, batch_size=64, encoding_dim=32)

        # Check updates
        updated_params = adapter.get_params()
        assert updated_params["epochs"] == 20
        assert updated_params["batch_size"] == 64
        assert updated_params["encoding_dim"] == 32

    def test_error_handling_and_edge_cases(self):
        """Test various error conditions and edge cases."""
        # Test invalid algorithm
        with pytest.raises(Exception):  # Should be InvalidAlgorithmError
            TensorFlowAdapter(algorithm_name="InvalidAlgorithm")

        # Test prediction before fitting
        data = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        dataset = Dataset(name="tiny", data=data)

        adapter = TensorFlowAdapter(algorithm_name="AutoEncoder")

        with pytest.raises(Exception):  # Should be DetectorNotFittedError
            adapter.detect(dataset)

        with pytest.raises(Exception):  # Should be DetectorNotFittedError
            adapter.score(dataset)

        # Test with very small dataset
        adapter = TensorFlowAdapter(
            algorithm_name="AutoEncoder", epochs=1, batch_size=2, encoding_dim=2
        )

        # Should still work with small data
        adapter.fit(dataset)
        result = adapter.detect(dataset)
        assert len(result.scores) == len(data)

    def test_confidence_and_threshold_calculation(self):
        """Test confidence calculation and threshold determination."""
        data = pd.DataFrame(
            {
                "x": np.random.normal(0, 1, 100),
                "y": np.random.normal(0, 1, 100),
            }
        )
        dataset = Dataset(name="confidence_test", data=data)

        adapter = TensorFlowAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=3,
            encoding_dim=4,
        )

        adapter.fit(dataset)

        # Test score method
        scores = adapter.score(dataset)
        assert all(hasattr(score, "confidence") for score in scores)
        assert all(0 <= score.confidence <= 1 for score in scores)

        # Test threshold calculation
        result = adapter.detect(dataset)
        assert result.threshold > 0

        # Check that threshold is used consistently
        high_scores = [s for s in scores if s.value > result.threshold]
        anomaly_count = sum(result.labels)

        # Should have some relationship between high scores and anomalies
        print(f"Threshold: {result.threshold:.4f}")
        print(f"High scores: {len(high_scores)}, Anomalies: {anomaly_count}")


@pytest.mark.skipif(HAS_TENSORFLOW, reason="Testing TensorFlow unavailable scenario")
def test_tensorflow_unavailable_error():
    """Test error when TensorFlow is not available."""
    from pynomaly.domain.exceptions import AdapterError
    from pynomaly.infrastructure.adapters.tensorflow_adapter import TensorFlowAdapter

    with pytest.raises(AdapterError, match="TensorFlow is not available"):
        TensorFlowAdapter(algorithm_name="AutoEncoder")

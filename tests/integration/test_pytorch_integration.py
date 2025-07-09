"""Integration test for PyTorch adapter functionality."""

import pytest
import pandas as pd
import numpy as np

from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter, TORCH_AVAILABLE
from pynomaly.domain.entities import Dataset
from pynomaly.domain.value_objects import ContaminationRate


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchIntegration:
    """Integration tests for PyTorch adapter."""

    def test_complete_anomaly_detection_workflow(self):
        """Test complete workflow from data to anomaly detection."""
        # Generate synthetic dataset with anomalies
        np.random.seed(42)
        
        # Normal data
        normal_data = np.random.multivariate_normal(
            mean=[0, 0, 0], 
            cov=[[1, 0.5, 0.1], [0.5, 1, 0.2], [0.1, 0.2, 1]], 
            size=200
        )
        
        # Anomalous data (outliers)
        anomaly_data = np.random.multivariate_normal(
            mean=[5, 5, 5], 
            cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
            size=20
        )
        
        # Combine data
        all_data = np.vstack([normal_data, anomaly_data])
        
        # Create DataFrame
        df = pd.DataFrame(all_data, columns=['feature1', 'feature2', 'feature3'])
        dataset = Dataset(name="synthetic_data", data=df)
        
        # Test AutoEncoder
        autoencoder = PyTorchAdapter(
            algorithm_name="AutoEncoder",
            contamination_rate=ContaminationRate(0.1),
            epochs=10,
            batch_size=32,
            hidden_dims=[16, 8],
            latent_dim=4,
            learning_rate=0.001
        )
        
        # Fit and detect
        result = autoencoder.fit_detect(dataset)
        
        # Verify results
        assert len(result.scores) == len(df)
        assert len(result.labels) == len(df)
        assert result.n_anomalies > 0
        assert result.threshold > 0
        
        # Check that some anomalies were detected
        detected_anomalies = np.sum(result.labels)
        expected_anomalies = int(len(df) * 0.1)  # 10% contamination
        
        # Should detect some anomalies (allowing for some variance)
        assert detected_anomalies > 0
        assert detected_anomalies <= len(df) * 0.15  # Not too many false positives
        
        print(f"Detected {detected_anomalies} anomalies out of {len(df)} points")
        print(f"Expected approximately {expected_anomalies} anomalies")

    def test_all_algorithms_basic_functionality(self):
        """Test that all algorithms can be instantiated and run basic operations."""
        # Create simple test data
        data = pd.DataFrame({
            'x': np.random.normal(0, 1, 50),
            'y': np.random.normal(0, 1, 50),
        })
        dataset = Dataset(name="test", data=data)
        
        algorithms = ["AutoEncoder", "VAE", "DeepSVDD", "DAGMM"]
        
        for algorithm in algorithms:
            print(f"Testing {algorithm}...")
            
            # Create adapter with small parameters for fast testing
            if algorithm == "DAGMM":
                adapter = PyTorchAdapter(
                    algorithm_name=algorithm,
                    epochs=2,
                    batch_size=16,
                    hidden_dims=[8, 4],
                    latent_dim=2,
                    n_gmm=2
                )
            else:
                adapter = PyTorchAdapter(
                    algorithm_name=algorithm,
                    epochs=2,
                    batch_size=16,
                    hidden_dims=[8, 4],
                    latent_dim=2
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
            
            print(f"  ✓ {algorithm} completed successfully")

    def test_gpu_detection(self):
        """Test GPU detection and device assignment."""
        adapter = PyTorchAdapter(algorithm_name="AutoEncoder")
        
        # Device should be set (CPU in this environment)
        assert hasattr(adapter, '_device')
        print(f"Device: {adapter._device}")
        
        # Should not crash when checking CUDA availability
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

    def test_parameter_validation_and_info(self):
        """Test parameter validation and algorithm information."""
        # Test getting algorithm info
        for algorithm in ["AutoEncoder", "VAE", "DeepSVDD", "DAGMM"]:
            info = PyTorchAdapter.get_algorithm_info(algorithm)
            
            assert "name" in info
            assert "type" in info
            assert "description" in info
            assert "parameters" in info
            assert "suitable_for" in info
            
            print(f"{algorithm}: {info['description']}")

    def test_error_handling(self):
        """Test various error conditions."""
        data = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
        dataset = Dataset(name="tiny", data=data)
        
        # Test invalid algorithm
        with pytest.raises(Exception):  # Should be AlgorithmNotFoundError
            PyTorchAdapter(algorithm_name="InvalidAlgorithm")
        
        # Test prediction before fitting
        adapter = PyTorchAdapter(algorithm_name="AutoEncoder")
        
        with pytest.raises(Exception):  # Should be AdapterError
            adapter.detect(dataset)
            
        with pytest.raises(Exception):  # Should be AdapterError
            adapter.score(dataset)


@pytest.mark.skipif(TORCH_AVAILABLE, reason="Testing PyTorch unavailable scenario")
def test_pytorch_unavailable_error():
    """Test error when PyTorch is not available."""
    from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
    from pynomaly.domain.exceptions import AdapterError
    
    with pytest.raises(AdapterError, match="PyTorch is not available"):
        PyTorchAdapter(algorithm_name="AutoEncoder")
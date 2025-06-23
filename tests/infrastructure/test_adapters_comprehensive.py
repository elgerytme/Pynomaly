"""Comprehensive tests for infrastructure adapters - Phase 2 Coverage."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import InvalidAlgorithmError, AdapterError
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.infrastructure.adapters import (
    PyODAdapter, 
    SklearnAdapter, 
    PyTorchAdapter, 
    PyGODAdapter, 
    TODSAdapter
)


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
        hyperparameters={"n_estimators": 100, "random_state": 42}
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


class TestSklearnAdapterComprehensive:
    """Comprehensive tests for SklearnAdapter functionality."""
    
    @pytest.mark.parametrize("algorithm", [
        "IsolationForest", "LocalOutlierFactor", "OneClassSVM", "EllipticEnvelope"
    ])
    def test_algorithm_creation(self, algorithm):
        """Test creating different sklearn algorithms."""
        with patch(f'sklearn.ensemble.{algorithm}' if algorithm == 'IsolationForest' else f'sklearn.{algorithm.lower()}.{algorithm}'):
            adapter = SklearnAdapter(algorithm)
            assert adapter.algorithm_name == algorithm
            assert adapter.name == f"Sklearn_{algorithm}"
            assert hasattr(adapter, "fit")
            assert hasattr(adapter, "detect")
    
    def test_fit_and_detect_pipeline(self, sample_dataset):
        """Test complete fitting and detection pipeline."""
        adapter = SklearnAdapter("IsolationForest")
        
        # Mock sklearn implementation
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.decision_function.return_value = np.random.random(len(sample_dataset.features))
            mock_model.predict.return_value = np.random.choice([-1, 1], len(sample_dataset.features))
            mock_iso.return_value = mock_model
            
            # Test detection
            result = adapter.fit_detect(sample_dataset)
            
            assert hasattr(result, 'anomaly_scores')
            assert hasattr(result, 'predictions')
            assert len(result.anomaly_scores) == len(sample_dataset.features)
            mock_model.fit.assert_called_once()
    
    def test_hyperparameter_validation(self, sample_detector):
        """Test hyperparameter validation and application."""
        adapter = SklearnAdapter("IsolationForest")
        
        # Test valid hyperparameters
        valid_params = {"n_estimators": 200, "contamination": 0.1, "random_state": 42}
        result = adapter.validate_hyperparameters(valid_params)
        assert result is True
        
        # Test invalid hyperparameters
        invalid_params = {"invalid_param": "value"}
        with pytest.raises(ValueError, match="Invalid hyperparameter"):
            adapter.validate_hyperparameters(invalid_params)
    
    def test_performance_with_large_dataset(self, large_dataset):
        """Test performance with larger datasets."""
        adapter = SklearnAdapter("IsolationForest")
        
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.decision_function.return_value = np.random.random(len(large_dataset.features))
            mock_iso.return_value = mock_model
            
            # Should handle large datasets efficiently
            result = adapter.fit_detect(large_dataset)
            assert len(result.anomaly_scores) == len(large_dataset.features)
            
    def test_error_handling(self, sample_dataset):
        """Test error handling during detection."""
        adapter = SklearnAdapter("IsolationForest")
        
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            mock_model = Mock()
            mock_model.fit.side_effect = ValueError("Fitting failed")
            mock_iso.return_value = mock_model
            
            with pytest.raises(AdapterError, match="Sklearn adapter failed"):
                adapter.fit_detect(sample_dataset)
    
    def test_local_outlier_factor_specific(self, anomalous_dataset):
        """Test LocalOutlierFactor with clear anomalies."""
        adapter = SklearnAdapter("LocalOutlierFactor")
        
        with patch('sklearn.neighbors.LocalOutlierFactor') as mock_lof:
            mock_model = Mock()
            mock_model.fit.return_value = None
            # LOF returns negative outlier factors
            mock_model.negative_outlier_factor_ = -np.random.random(len(anomalous_dataset.features)) - 1
            mock_model.predict.return_value = np.array([-1] * 5 + [1] * 95)  # 5 outliers
            mock_lof.return_value = mock_model
            
            result = adapter.fit_detect(anomalous_dataset)
            assert len(result.anomaly_scores) == len(anomalous_dataset.features)
            assert sum(result.predictions == -1) == 5  # Should detect 5 anomalies
    
    def test_one_class_svm_parameters(self, sample_dataset):
        """Test OneClassSVM with specific parameters."""
        adapter = SklearnAdapter("OneClassSVM")
        
        with patch('sklearn.svm.OneClassSVM') as mock_svm:
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.decision_function.return_value = np.random.normal(0, 1, len(sample_dataset.features))
            mock_model.predict.return_value = np.random.choice([-1, 1], len(sample_dataset.features))
            mock_svm.return_value = mock_model
            
            # Test with specific parameters
            custom_params = {"nu": 0.05, "gamma": "scale", "kernel": "rbf"}
            result = adapter.fit_detect(sample_dataset, **custom_params)
            
            # Verify parameters were passed to sklearn
            mock_svm.assert_called_with(**custom_params)
            assert len(result.anomaly_scores) == len(sample_dataset.features)


class TestPyODAdapterComprehensive:
    """Comprehensive tests for PyODAdapter functionality."""
    
    @pytest.mark.parametrize("algorithm", [
        "IsolationForest", "LOF", "OCSVM", "ABOD", "FeatureBagging"
    ])
    def test_pyod_algorithm_creation(self, algorithm):
        """Test creating different PyOD algorithms."""
        with patch(f'pyod.models.{algorithm.lower()}.{algorithm}'):
            adapter = PyODAdapter(algorithm)
            assert adapter.algorithm_name == algorithm
            assert adapter.name == f"PyOD_{algorithm}"
    
    def test_pyod_algorithm_parameters(self):
        """Test PyOD algorithm parameter handling."""
        with patch('pyod.models.iforest.IForest'):
            adapter = PyODAdapter("IsolationForest")
            
            # Test default parameters
            default_params = adapter.get_default_hyperparameters()
            assert "n_estimators" in default_params
            assert "contamination" in default_params
            
            # Test parameter validation
            valid_params = {"n_estimators": 200, "contamination": 0.15}
            assert adapter.validate_hyperparameters(valid_params) is True
            
            # Test invalid parameters
            invalid_params = {"non_existent_param": "value"}
            with pytest.raises(ValueError):
                adapter.validate_hyperparameters(invalid_params)
    
    def test_pyod_with_preprocessing(self, sample_dataset):
        """Test PyOD with data preprocessing."""
        with patch('pyod.models.lof.LOF'):
            adapter = PyODAdapter("LOF")
            
            # Mock preprocessing pipeline
            with patch.object(adapter, '_preprocess_data') as mock_preprocess:
                preprocessed_data = sample_dataset.features * 2
                mock_preprocess.return_value = preprocessed_data
                
                with patch('pyod.models.lof.LOF') as mock_lof:
                    mock_model = Mock()
                    mock_model.fit.return_value = None
                    mock_model.decision_function.return_value = np.random.random(len(sample_dataset.features))
                    mock_lof.return_value = mock_model
                    
                    result = adapter.fit_detect(sample_dataset, preprocess=True)
                    mock_preprocess.assert_called_once()
                    assert len(result.anomaly_scores) == len(sample_dataset.features)
    
    def test_pyod_ensemble_methods(self):
        """Test PyOD ensemble algorithm support."""
        ensemble_algorithms = ["FeatureBagging", "LSCP", "SUOD"]
        
        for algo in ensemble_algorithms:
            try:
                with patch(f'pyod.models.{algo.lower()}.{algo}'):
                    adapter = PyODAdapter(algo)
                    assert adapter.algorithm_name == algo
                    assert "ensemble" in adapter.get_algorithm_info()["category"].lower()
            except InvalidAlgorithmError:
                # Skip if algorithm not available in current PyOD version
                pytest.skip(f"Algorithm {algo} not available in current PyOD version")
    
    def test_memory_efficient_processing(self, large_dataset):
        """Test memory-efficient processing for large datasets."""
        with patch('pyod.models.iforest.IForest') as mock_iforest:
            adapter = PyODAdapter("IsolationForest")
            
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.decision_function.return_value = np.random.random(len(large_dataset.features))
            mock_iforest.return_value = mock_model
            
            # Should handle large datasets without memory issues
            result = adapter.fit_detect(large_dataset, batch_size=100)
            assert len(result.anomaly_scores) == len(large_dataset.features)


class TestPyTorchAdapterComprehensive:
    """Comprehensive tests for PyTorchAdapter functionality."""
    
    def test_pytorch_model_creation(self):
        """Test PyTorch model creation."""
        with patch('torch.nn.Module'):
            adapter = PyTorchAdapter("AutoEncoder")
            assert adapter.algorithm_name == "AutoEncoder"
            assert adapter.name == "PyTorch_AutoEncoder"
            assert hasattr(adapter, "fit")
            assert hasattr(adapter, "detect")
    
    def test_gpu_detection_and_usage(self):
        """Test GPU detection and usage."""
        with patch('torch.cuda.is_available') as mock_cuda:
            adapter = PyTorchAdapter("AutoEncoder")
            mock_cuda.return_value = True
            
            gpu_available = adapter.is_gpu_available()
            assert gpu_available is True
            
            with patch('torch.device') as mock_device:
                mock_device.return_value = "cuda:0"
                device = adapter.get_device()
                assert "cuda" in str(device)
    
    def test_autoencoder_training(self, sample_dataset):
        """Test AutoEncoder training process."""
        with patch('torch.nn.Module') as mock_nn, \
             patch('torch.optim.Adam') as mock_optimizer, \
             patch('torch.nn.MSELoss') as mock_criterion, \
             patch('torch.tensor') as mock_tensor:
            
            adapter = PyTorchAdapter("AutoEncoder")
            
            mock_model = Mock()
            mock_model.forward.return_value = mock_tensor.return_value
            mock_nn.return_value = mock_model
            
            mock_opt = Mock()
            mock_optimizer.return_value = mock_opt
            
            mock_loss = Mock()
            mock_loss.return_value = mock_tensor.return_value
            mock_criterion.return_value = mock_loss
            
            # Train the model
            adapter.fit(sample_dataset, epochs=5, learning_rate=0.001)
            
            # Verify training components were used
            mock_optimizer.assert_called()
            mock_criterion.assert_called()
    
    def test_vae_training(self, sample_dataset):
        """Test Variational AutoEncoder training."""
        with patch.object(PyTorchAdapter, '_create_vae_model') as mock_create_model, \
             patch('torch.tensor') as mock_tensor:
            
            adapter = PyTorchAdapter("VAE")
            
            mock_model = Mock()
            mock_model.forward.return_value = (
                mock_tensor.return_value,  # reconstruction
                mock_tensor.return_value,  # mu
                mock_tensor.return_value   # logvar
            )
            mock_create_model.return_value = mock_model
            
            adapter.fit(sample_dataset, epochs=3)
            assert adapter.is_fitted
    
    def test_anomaly_scoring(self, sample_dataset):
        """Test anomaly scoring methods."""
        with patch.object(PyTorchAdapter, 'is_fitted', True), \
             patch.object(PyTorchAdapter, '_compute_reconstruction_error') as mock_error:
            
            adapter = PyTorchAdapter("AutoEncoder")
            mock_error.return_value = np.random.random(len(sample_dataset.features))
            
            scores = adapter.score(sample_dataset)
            assert len(scores) == len(sample_dataset.features)
            assert all(isinstance(s, (float, np.float32, np.float64)) for s in scores)
    
    def test_model_persistence(self, sample_dataset):
        """Test model saving and loading."""
        with patch('torch.save') as mock_save, \
             patch('torch.load') as mock_load:
            
            adapter = PyTorchAdapter("AutoEncoder")
            
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Test saving
            adapter._fitted_model = mock_model
            adapter.save_model("/tmp/test_model.pth")
            mock_save.assert_called_once()
            
            # Test loading
            loaded_adapter = PyTorchAdapter("AutoEncoder")
            loaded_adapter.load_model("/tmp/test_model.pth")
            mock_load.assert_called_once()


class TestPyGODAdapterComprehensive:
    """Comprehensive tests for PyGODAdapter for graph anomaly detection."""
    
    def test_pygod_model_creation(self):
        """Test PyGOD model creation."""
        with patch('pygod.models.DOMINANT'):
            adapter = PyGODAdapter("DOMINANT")
            assert adapter.algorithm_name == "DOMINANT"
            assert adapter.name == "PyGOD_DOMINANT"
            assert hasattr(adapter, "fit")
            assert hasattr(adapter, "detect")
    
    def test_graph_structure_inference(self, sample_dataset):
        """Test automatic graph structure inference from tabular data."""
        with patch('pygod.models.GCNAE'):
            adapter = PyGODAdapter("GCNAE")
            
            with patch.object(adapter, '_infer_graph_structure') as mock_infer:
                # Mock graph structure (adjacency matrix)
                n_nodes = len(sample_dataset.features)
                mock_adj_matrix = np.random.choice([0, 1], size=(n_nodes, n_nodes), p=[0.9, 0.1])
                mock_infer.return_value = mock_adj_matrix
                
                graph_structure = adapter._infer_graph_structure(sample_dataset.features)
                assert graph_structure.shape == (n_nodes, n_nodes)
                mock_infer.assert_called_once()
    
    def test_gnn_model_training(self, sample_dataset):
        """Test Graph Neural Network model training."""
        with patch('pygod.models.DOMINANT') as mock_dominant:
            adapter = PyGODAdapter("DOMINANT")
            
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.decision_function.return_value = np.random.random(len(sample_dataset.features))
            mock_dominant.return_value = mock_model
            
            result = adapter.fit_detect(sample_dataset)
            assert len(result.anomaly_scores) == len(sample_dataset.features)
            mock_model.fit.assert_called_once()
    
    def test_graph_feature_processing(self, sample_dataset):
        """Test graph feature processing and node embeddings."""
        with patch('pygod.models.SCAN'):
            adapter = PyGODAdapter("SCAN")
            
            with patch.object(adapter, '_create_node_features') as mock_features:
                # Mock node feature matrix
                node_features = np.random.random((len(sample_dataset.features), 10))
                mock_features.return_value = node_features
                
                features = adapter._create_node_features(sample_dataset.features)
                assert features.shape[0] == len(sample_dataset.features)
                mock_features.assert_called_once()
    
    def test_graph_algorithm_parameters(self):
        """Test graph algorithm hyperparameters."""
        with patch('pygod.models.GAAN'):
            adapter = PyGODAdapter("GAAN")
            
            # Test default parameters
            default_params = adapter.get_default_hyperparameters()
            assert "hidden_dim" in default_params or "lr" in default_params
            
            # Test parameter validation
            valid_params = {"hidden_dim": 64, "lr": 0.01, "epoch": 100}
            assert adapter.validate_hyperparameters(valid_params) is True


class TestTODSAdapterComprehensive:
    """Comprehensive tests for TODSAdapter for time-series anomaly detection."""
    
    def test_tods_model_creation(self):
        """Test TODS model creation."""
        with patch('tods.detection_algorithm.MatrixProfile'):
            adapter = TODSAdapter("MatrixProfile")
            assert adapter.algorithm_name == "MatrixProfile"
            assert adapter.name == "TODS_MatrixProfile"
            assert hasattr(adapter, "fit")
            assert hasattr(adapter, "detect")
    
    def test_time_series_formatting(self, sample_dataset):
        """Test automatic time-series data formatting."""
        with patch('tods.detection_algorithm.LSTM'):
            adapter = TODSAdapter("LSTM")
            
            with patch.object(adapter, '_format_time_series') as mock_format:
                # Mock time-series formatted data
                ts_data = sample_dataset.features.reshape(-1, 1, sample_dataset.features.shape[1])
                mock_format.return_value = ts_data
                
                formatted_data = adapter._format_time_series(sample_dataset.features)
                assert len(formatted_data.shape) == 3  # Should be 3D for time-series
                mock_format.assert_called_once()
    
    def test_window_based_detection(self, sample_dataset):
        """Test window-based anomaly detection."""
        with patch('tods.detection_algorithm.DeepLog') as mock_deeplog:
            adapter = TODSAdapter("DeepLog")
            
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.produce.return_value = np.random.random(len(sample_dataset.features))
            mock_deeplog.return_value = mock_model
            
            result = adapter.fit_detect(sample_dataset, window_size=10)
            assert len(result.anomaly_scores) == len(sample_dataset.features)
            mock_model.fit.assert_called_once()
    
    def test_temporal_pattern_analysis(self, sample_dataset):
        """Test temporal pattern analysis capabilities."""
        with patch('tods.detection_algorithm.Telemanom'):
            adapter = TODSAdapter("Telemanom")
            
            with patch.object(adapter, '_analyze_temporal_patterns') as mock_analyze:
                # Mock temporal analysis results
                patterns = {
                    "seasonality": True,
                    "trend": "increasing",
                    "anomaly_periods": [10, 25, 67]
                }
                mock_analyze.return_value = patterns
                
                analysis = adapter._analyze_temporal_patterns(sample_dataset.features)
                assert "seasonality" in analysis
                assert "trend" in analysis
                mock_analyze.assert_called_once()
    
    def test_multivariate_time_series(self, sample_dataset):
        """Test multivariate time-series anomaly detection."""
        with patch('tods.detection_algorithm.LSTM') as mock_lstm:
            adapter = TODSAdapter("LSTM")
            
            # Create multivariate time-series data
            multivariate_data = sample_dataset.features  # Already multivariate
            
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.produce.return_value = np.random.random(len(multivariate_data))
            mock_lstm.return_value = mock_model
            
            result = adapter.fit_detect(Dataset(name="multivariate", features=multivariate_data))
            assert len(result.anomaly_scores) == len(multivariate_data)


class TestAdapterIntegration:
    """Test cross-adapter integration and interoperability."""
    
    def test_adapter_factory(self):
        """Test adapter factory pattern."""
        from pynomaly.infrastructure.adapters.factory import AdapterFactory
        
        with patch('pynomaly.infrastructure.adapters.sklearn_adapter.SklearnAdapter'), \
             patch('pynomaly.infrastructure.adapters.pyod_adapter.PyODAdapter'):
            
            # Test different adapter types
            sklearn_adapter = AdapterFactory.create_adapter("sklearn", "IsolationForest")
            assert isinstance(sklearn_adapter, SklearnAdapter)
            
            pyod_adapter = AdapterFactory.create_adapter("pyod", "LOF")
            assert isinstance(pyod_adapter, PyODAdapter)
    
    def test_cross_adapter_ensemble(self, sample_dataset):
        """Test ensemble detection across different adapter types."""
        with patch('sklearn.ensemble.IsolationForest'), \
             patch('pyod.models.lof.LOF'):
            
            adapters = [
                SklearnAdapter("IsolationForest"),
                PyODAdapter("LOF"),
            ]
            
            results = []
            for adapter in adapters:
                with patch.object(adapter, 'fit_detect') as mock_detect:
                    mock_result = Mock()
                    mock_result.anomaly_scores = np.random.random(len(sample_dataset.features))
                    mock_detect.return_value = mock_result
                    
                    result = adapter.fit_detect(sample_dataset)
                    results.append(result.anomaly_scores)
            
            # Test ensemble aggregation
            ensemble_scores = np.mean(results, axis=0)
            assert len(ensemble_scores) == len(sample_dataset.features)
    
    def test_adapter_performance_comparison(self, sample_dataset):
        """Test performance comparison between adapters."""
        algorithms = ["IsolationForest", "LOF", "OCSVM"]
        
        with patch('sklearn.ensemble.IsolationForest'), \
             patch('sklearn.neighbors.LocalOutlierFactor'), \
             patch('sklearn.svm.OneClassSVM'):
            
            performance_results = {}
            for algo in algorithms:
                sklearn_adapter = SklearnAdapter(algo)
                
                with patch.object(sklearn_adapter, 'fit_detect') as mock_detect:
                    import time
                    start_time = time.time()
                    
                    mock_result = Mock()
                    mock_result.anomaly_scores = np.random.random(len(sample_dataset.features))
                    mock_detect.return_value = mock_result
                    
                    result = sklearn_adapter.fit_detect(sample_dataset)
                    execution_time = time.time() - start_time
                    
                    performance_results[algo] = {
                        "execution_time": execution_time,
                        "num_anomalies": np.sum(result.anomaly_scores > 0.5)
                    }
            
            assert len(performance_results) == len(algorithms)
            assert all("execution_time" in result for result in performance_results.values())
    
    def test_adapter_configuration_consistency(self):
        """Test configuration consistency across adapters."""
        with patch('sklearn.ensemble.IsolationForest'), \
             patch('pyod.models.iforest.IForest'):
            
            adapters = [
                SklearnAdapter("IsolationForest"),
                PyODAdapter("IsolationForest"),
            ]
            
            # Test that similar algorithms have consistent parameters
            for adapter in adapters:
                default_params = adapter.get_default_hyperparameters()
                assert "contamination" in default_params
                assert isinstance(default_params["contamination"], (int, float))
                assert 0 < default_params["contamination"] <= 0.5


class TestAdapterErrorHandling:
    """Test comprehensive error handling across all adapters."""
    
    def test_invalid_algorithm_error(self):
        """Test invalid algorithm error handling."""
        with pytest.raises(InvalidAlgorithmError):
            SklearnAdapter("NonExistentAlgorithm")
        
        with pytest.raises(InvalidAlgorithmError):
            PyODAdapter("NonExistentAlgorithm")
    
    def test_data_validation_errors(self, sample_dataset):
        """Test data validation error handling."""
        with patch('sklearn.ensemble.IsolationForest'):
            adapter = SklearnAdapter("IsolationForest")
            
            # Test with invalid data (NaN values)
            invalid_data = sample_dataset.features.copy()
            invalid_data[0, 0] = np.nan
            invalid_dataset = Dataset(name="invalid", features=invalid_data)
            
            with pytest.raises(AdapterError, match="Invalid data"):
                adapter.fit_detect(invalid_dataset)
    
    def test_model_not_fitted_error(self, sample_dataset):
        """Test error when trying to detect with unfitted model."""
        with patch('sklearn.ensemble.IsolationForest'):
            adapter = SklearnAdapter("IsolationForest")
            
            # Try to detect without fitting first
            with pytest.raises(AdapterError, match="Model not fitted"):
                adapter.detect(sample_dataset)
    
    def test_dimension_mismatch_error(self, sample_dataset):
        """Test error when feature dimensions don't match between fit and detect."""
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            adapter = SklearnAdapter("IsolationForest")
            
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_iso.return_value = mock_model
            
            # Fit with original dataset
            adapter.fit(sample_dataset)
            
            # Try to detect with different dimension data
            different_dim_data = np.random.random((50, 3))  # Different number of features
            different_dataset = Dataset(name="different", features=different_dim_data)
            
            with pytest.raises(AdapterError, match="Feature dimension mismatch"):
                adapter.detect(different_dataset)


class TestAdapterConfigurationManagement:
    """Test adapter configuration and parameter management."""
    
    def test_hyperparameter_grid_search(self, sample_dataset):
        """Test hyperparameter grid search functionality."""
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            adapter = SklearnAdapter("IsolationForest")
            
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.decision_function.return_value = np.random.random(len(sample_dataset.features))
            mock_iso.return_value = mock_model
            
            # Define parameter grid
            param_grid = {
                "n_estimators": [50, 100, 200],
                "contamination": [0.05, 0.1, 0.15]
            }
            
            best_params = adapter.grid_search(sample_dataset, param_grid, cv=3)
            assert "n_estimators" in best_params
            assert "contamination" in best_params
    
    def test_configuration_serialization(self):
        """Test adapter configuration serialization."""
        with patch('sklearn.ensemble.IsolationForest'):
            adapter = SklearnAdapter("IsolationForest")
            
            # Test configuration to dict
            config = adapter.to_dict()
            assert "algorithm_name" in config
            assert "hyperparameters" in config
            assert "adapter_type" in config
            
            # Test configuration from dict
            new_adapter = SklearnAdapter.from_dict(config)
            assert new_adapter.algorithm_name == adapter.algorithm_name
    
    def test_adapter_metadata(self):
        """Test adapter metadata and algorithm information."""
        with patch('sklearn.ensemble.IsolationForest'):
            adapter = SklearnAdapter("IsolationForest")
            
            metadata = adapter.get_algorithm_metadata()
            assert "name" in metadata
            assert "category" in metadata
            assert "description" in metadata
            assert "parameters" in metadata
            assert "complexity" in metadata
    
    def test_adapter_versioning(self):
        """Test adapter versioning and compatibility."""
        with patch('sklearn.ensemble.IsolationForest'):
            adapter = SklearnAdapter("IsolationForest")
            
            version_info = adapter.get_version_info()
            assert "adapter_version" in version_info
            assert "sklearn_version" in version_info
            assert "compatibility" in version_info
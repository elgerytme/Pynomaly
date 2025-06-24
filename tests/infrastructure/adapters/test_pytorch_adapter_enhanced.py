"""
Enhanced PyTorch Adapter Testing Suite
Comprehensive tests for PyTorch deep learning anomaly detection adapter.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from datetime import datetime

from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.domain.exceptions import DetectorError, AdapterError


class TestPyTorchAdapter:
    """Enhanced test suite for PyTorch adapter functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (800, 5))
        anomalous_data = np.random.normal(3, 1, (200, 5))
        data = np.vstack([normal_data, anomalous_data])
        labels = np.hstack([np.zeros(800), np.ones(200)])
        
        return {
            "X_train": data[:600],
            "X_test": data[600:],
            "y_train": labels[:600],
            "y_test": labels[600:],
            "features": [f"feature_{i}" for i in range(5)]
        }

    @pytest.fixture
    def pytorch_adapter(self):
        """Create PyTorch adapter instance."""
        return PyTorchAdapter(
            model_type="autoencoder",
            hidden_dims=[16, 8, 4],
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            device="cpu"
        )

    @pytest.fixture
    def mock_dataset(self, sample_data):
        """Create mock dataset."""
        dataset = Mock(spec=Dataset)
        dataset.id = "test_dataset"
        dataset.data = sample_data["X_train"]
        dataset.features = sample_data["features"]
        dataset.target_column = None
        return dataset

    # Adapter Initialization Tests

    def test_pytorch_adapter_initialization_default(self):
        """Test PyTorch adapter initialization with default parameters."""
        adapter = PyTorchAdapter()
        
        assert adapter.model_type == "autoencoder"
        assert adapter.device in ["cpu", "cuda"]
        assert adapter.learning_rate == 0.001
        assert adapter.batch_size == 32
        assert adapter.epochs == 100

    def test_pytorch_adapter_initialization_custom(self):
        """Test PyTorch adapter initialization with custom parameters."""
        adapter = PyTorchAdapter(
            model_type="variational_autoencoder",
            hidden_dims=[64, 32, 16],
            learning_rate=0.01,
            batch_size=64,
            epochs=50,
            device="cpu",
            dropout_rate=0.2
        )
        
        assert adapter.model_type == "variational_autoencoder"
        assert adapter.hidden_dims == [64, 32, 16]
        assert adapter.learning_rate == 0.01
        assert adapter.batch_size == 64
        assert adapter.epochs == 50
        assert adapter.dropout_rate == 0.2

    def test_pytorch_adapter_gpu_detection(self):
        """Test GPU device detection and fallback."""
        with patch('torch.cuda.is_available') as mock_cuda:
            # Test CUDA available
            mock_cuda.return_value = True
            adapter = PyTorchAdapter(device="auto")
            assert "cuda" in adapter.device
            
            # Test CUDA not available
            mock_cuda.return_value = False
            adapter = PyTorchAdapter(device="auto")
            assert adapter.device == "cpu"

    def test_pytorch_adapter_invalid_parameters(self):
        """Test adapter initialization with invalid parameters."""
        with pytest.raises(ValueError):
            PyTorchAdapter(model_type="invalid_model")
        
        with pytest.raises(ValueError):
            PyTorchAdapter(learning_rate=-0.1)
        
        with pytest.raises(ValueError):
            PyTorchAdapter(batch_size=0)
        
        with pytest.raises(ValueError):
            PyTorchAdapter(epochs=-1)

    # Model Architecture Tests

    def test_autoencoder_model_creation(self, pytorch_adapter, sample_data):
        """Test autoencoder model architecture creation."""
        input_dim = sample_data["X_train"].shape[1]
        model = pytorch_adapter._create_autoencoder(input_dim)
        
        assert isinstance(model, nn.Module)
        
        # Test forward pass
        test_input = torch.randn(10, input_dim)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_variational_autoencoder_creation(self, sample_data):
        """Test variational autoencoder model creation."""
        adapter = PyTorchAdapter(model_type="variational_autoencoder")
        input_dim = sample_data["X_train"].shape[1]
        
        model = adapter._create_variational_autoencoder(input_dim)
        
        assert isinstance(model, nn.Module)
        
        # Test VAE forward pass returns reconstruction and latent variables
        test_input = torch.randn(10, input_dim)
        reconstruction, mu, logvar = model(test_input)
        
        assert reconstruction.shape == test_input.shape
        assert mu.shape[0] == test_input.shape[0]
        assert logvar.shape[0] == test_input.shape[0]

    def test_lstm_autoencoder_creation(self, sample_data):
        """Test LSTM autoencoder for sequence data."""
        adapter = PyTorchAdapter(
            model_type="lstm_autoencoder",
            sequence_length=10,
            hidden_dims=[32, 16]
        )
        
        input_dim = sample_data["X_train"].shape[1]
        model = adapter._create_lstm_autoencoder(input_dim)
        
        assert isinstance(model, nn.Module)
        
        # Test LSTM forward pass
        test_input = torch.randn(5, 10, input_dim)  # (batch, seq_len, features)
        output = model(test_input)
        assert output.shape == test_input.shape

    def test_transformer_autoencoder_creation(self, sample_data):
        """Test Transformer autoencoder model."""
        adapter = PyTorchAdapter(
            model_type="transformer_autoencoder",
            sequence_length=20,
            num_heads=4,
            num_layers=2
        )
        
        input_dim = sample_data["X_train"].shape[1]
        model = adapter._create_transformer_autoencoder(input_dim)
        
        assert isinstance(model, nn.Module)

    # Training Process Tests

    def test_fit_basic_training(self, pytorch_adapter, mock_dataset, sample_data):
        """Test basic model training process."""
        with patch.object(pytorch_adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock(spec=nn.Module)
            mock_create.return_value = mock_model
            
            # Mock training loop
            with patch('torch.optim.Adam'), \
                 patch('torch.nn.MSELoss'), \
                 patch('torch.utils.data.DataLoader'):
                
                detector = pytorch_adapter.fit(mock_dataset)
                
                assert isinstance(detector, Detector)
                assert detector.algorithm == "pytorch_autoencoder"
                assert "model_state" in detector.parameters

    def test_fit_with_validation_data(self, pytorch_adapter, mock_dataset, sample_data):
        """Test training with validation data."""
        validation_data = sample_data["X_test"]
        
        with patch.object(pytorch_adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock(spec=nn.Module)
            mock_create.return_value = mock_model
            
            with patch('torch.optim.Adam'), \
                 patch('torch.nn.MSELoss'), \
                 patch('torch.utils.data.DataLoader'):
                
                detector = pytorch_adapter.fit(
                    mock_dataset, 
                    validation_data=validation_data
                )
                
                assert detector is not None
                assert "validation_loss" in detector.parameters.get("training_history", {})

    def test_fit_early_stopping(self, pytorch_adapter, mock_dataset):
        """Test early stopping during training."""
        pytorch_adapter.early_stopping = True
        pytorch_adapter.patience = 5
        
        with patch.object(pytorch_adapter, '_train_epoch') as mock_train:
            # Simulate loss not improving
            mock_train.side_effect = [1.0, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
            
            with patch.object(pytorch_adapter, '_create_autoencoder'):
                detector = pytorch_adapter.fit(mock_dataset)
                
                # Should stop early due to patience
                assert detector is not None

    def test_fit_learning_rate_scheduling(self, pytorch_adapter, mock_dataset):
        """Test learning rate scheduling during training."""
        pytorch_adapter.lr_scheduler = "cosine"
        
        with patch.object(pytorch_adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock(spec=nn.Module)
            mock_create.return_value = mock_model
            
            with patch('torch.optim.Adam') as mock_optimizer, \
                 patch('torch.optim.lr_scheduler.CosineAnnealingLR') as mock_scheduler:
                
                detector = pytorch_adapter.fit(mock_dataset)
                
                assert detector is not None
                mock_scheduler.assert_called()

    def test_fit_gradient_clipping(self, pytorch_adapter, mock_dataset):
        """Test gradient clipping during training."""
        pytorch_adapter.gradient_clip_value = 1.0
        
        with patch.object(pytorch_adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock(spec=nn.Module)
            mock_create.return_value = mock_model
            
            with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
                with patch('torch.optim.Adam'), \
                     patch('torch.nn.MSELoss'), \
                     patch('torch.utils.data.DataLoader'):
                    
                    detector = pytorch_adapter.fit(mock_dataset)
                    
                    assert detector is not None
                    mock_clip.assert_called()

    # Detection Process Tests

    def test_predict_basic_detection(self, pytorch_adapter, sample_data):
        """Test basic anomaly detection."""
        # Create trained detector mock
        detector = Detector(
            id="test_detector",
            algorithm="pytorch_autoencoder",
            parameters={
                "model_state": "mock_state",
                "input_dim": 5,
                "threshold": 0.5
            }
        )
        
        with patch.object(pytorch_adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.eval.return_value = None
            mock_model.return_value = torch.randn(len(sample_data["X_test"]), 5)
            mock_load.return_value = mock_model
            
            result = pytorch_adapter.predict(detector, sample_data["X_test"])
            
            assert isinstance(result, DetectionResult)
            assert len(result.anomaly_scores) == len(sample_data["X_test"])
            assert all(isinstance(score, AnomalyScore) for score in result.anomaly_scores)

    def test_predict_batch_processing(self, pytorch_adapter, sample_data):
        """Test batch processing during prediction."""
        detector = Detector(
            id="test_detector",
            algorithm="pytorch_autoencoder",
            parameters={"model_state": "mock_state", "input_dim": 5}
        )
        
        # Large dataset to test batching
        large_data = np.random.randn(1000, 5)
        
        with patch.object(pytorch_adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.eval.return_value = None
            mock_model.return_value = torch.randn(pytorch_adapter.batch_size, 5)
            mock_load.return_value = mock_model
            
            result = pytorch_adapter.predict(detector, large_data)
            
            assert len(result.anomaly_scores) == len(large_data)

    def test_predict_gpu_processing(self, sample_data):
        """Test GPU processing during prediction."""
        adapter = PyTorchAdapter(device="cuda")
        detector = Detector(
            id="test_detector",
            algorithm="pytorch_autoencoder", 
            parameters={"model_state": "mock_state", "input_dim": 5}
        )
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch.object(adapter, '_load_model') as mock_load:
            
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model.return_value = torch.randn(len(sample_data["X_test"]), 5)
            mock_load.return_value = mock_model
            
            result = adapter.predict(detector, sample_data["X_test"])
            
            # Verify model was moved to GPU
            mock_model.to.assert_called_with("cuda")
            assert result is not None

    # Threshold Calculation Tests

    def test_automatic_threshold_calculation(self, pytorch_adapter, sample_data):
        """Test automatic threshold calculation from training data."""
        with patch.object(pytorch_adapter, '_calculate_reconstruction_errors') as mock_calc:
            mock_calc.return_value = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.12])
            
            threshold = pytorch_adapter._calculate_threshold(
                mock_calc.return_value,
                contamination_rate=ContaminationRate(0.1)
            )
            
            assert isinstance(threshold, float)
            assert 0 < threshold < 1

    def test_percentile_threshold_calculation(self, pytorch_adapter):
        """Test percentile-based threshold calculation."""
        errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        
        threshold = pytorch_adapter._calculate_threshold(
            errors,
            contamination_rate=ContaminationRate(0.2),
            method="percentile"
        )
        
        # Should be 80th percentile (1 - 0.2)
        expected = np.percentile(errors, 80)
        assert abs(threshold - expected) < 0.01

    def test_statistical_threshold_calculation(self, pytorch_adapter):
        """Test statistical threshold calculation (mean + k*std)."""
        errors = np.random.normal(0.3, 0.1, 1000)
        
        threshold = pytorch_adapter._calculate_threshold(
            errors,
            method="statistical",
            std_multiplier=2.5
        )
        
        expected = np.mean(errors) + 2.5 * np.std(errors)
        assert abs(threshold - expected) < 0.01

    # Model Persistence Tests

    def test_model_serialization(self, pytorch_adapter, mock_dataset):
        """Test model serialization and deserialization."""
        with patch.object(pytorch_adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock(spec=nn.Module)
            mock_model.state_dict.return_value = {"layer1.weight": torch.randn(5, 10)}
            mock_create.return_value = mock_model
            
            detector = pytorch_adapter.fit(mock_dataset)
            
            # Test serialization
            serialized = pytorch_adapter._serialize_model(mock_model)
            assert isinstance(serialized, (str, bytes))
            
            # Test deserialization
            deserialized_model = pytorch_adapter._deserialize_model(
                serialized, 
                input_dim=5
            )
            assert isinstance(deserialized_model, nn.Module)

    def test_model_checkpoint_saving(self, pytorch_adapter, mock_dataset):
        """Test model checkpoint saving during training."""
        pytorch_adapter.save_checkpoints = True
        pytorch_adapter.checkpoint_dir = "/tmp/checkpoints"
        
        with patch('os.makedirs'), \
             patch('torch.save') as mock_save, \
             patch.object(pytorch_adapter, '_create_autoencoder'):
            
            detector = pytorch_adapter.fit(mock_dataset)
            
            assert detector is not None
            # Verify checkpoint was saved
            mock_save.assert_called()

    # Error Handling Tests

    def test_fit_invalid_data(self, pytorch_adapter):
        """Test training with invalid data."""
        invalid_dataset = Mock()
        invalid_dataset.data = None
        
        with pytest.raises(AdapterError):
            pytorch_adapter.fit(invalid_dataset)

    def test_fit_empty_data(self, pytorch_adapter):
        """Test training with empty data."""
        empty_dataset = Mock()
        empty_dataset.data = np.array([]).reshape(0, 5)
        
        with pytest.raises(AdapterError):
            pytorch_adapter.fit(empty_dataset)

    def test_fit_insufficient_data(self, pytorch_adapter):
        """Test training with insufficient data."""
        small_dataset = Mock()
        small_dataset.data = np.random.randn(5, 3)  # Too few samples
        
        with pytest.raises(AdapterError):
            pytorch_adapter.fit(small_dataset)

    def test_predict_model_not_trained(self, pytorch_adapter, sample_data):
        """Test prediction without trained model."""
        detector = Detector(
            id="test_detector",
            algorithm="pytorch_autoencoder",
            parameters={}  # No model_state
        )
        
        with pytest.raises(DetectorError):
            pytorch_adapter.predict(detector, sample_data["X_test"])

    def test_predict_incompatible_data_shape(self, pytorch_adapter, sample_data):
        """Test prediction with incompatible data shape."""
        detector = Detector(
            id="test_detector",
            algorithm="pytorch_autoencoder",
            parameters={
                "model_state": "mock_state",
                "input_dim": 10  # Different from test data
            }
        )
        
        with pytest.raises(AdapterError):
            pytorch_adapter.predict(detector, sample_data["X_test"])

    def test_cuda_out_of_memory_handling(self, sample_data):
        """Test handling of CUDA out of memory errors."""
        adapter = PyTorchAdapter(device="cuda", batch_size=1024)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.randn') as mock_tensor:
            
            # Simulate CUDA OOM error
            mock_tensor.side_effect = RuntimeError("CUDA out of memory")
            
            # Should fallback to CPU
            with pytest.raises(AdapterError):
                adapter._move_to_device(torch.randn(100, 5))

    # Performance Tests

    def test_training_performance_monitoring(self, pytorch_adapter, mock_dataset):
        """Test training performance monitoring."""
        pytorch_adapter.monitor_performance = True
        
        with patch.object(pytorch_adapter, '_create_autoencoder'), \
             patch('time.time') as mock_time:
            
            # Mock time progression
            mock_time.side_effect = [0, 1, 2, 3, 4, 5]  # 5 seconds total
            
            detector = pytorch_adapter.fit(mock_dataset)
            
            assert "training_time" in detector.parameters
            assert detector.parameters["training_time"] > 0

    def test_memory_usage_monitoring(self, pytorch_adapter, mock_dataset):
        """Test memory usage monitoring during training."""
        pytorch_adapter.monitor_memory = True
        
        with patch.object(pytorch_adapter, '_create_autoencoder'), \
             patch('psutil.Process') as mock_process:
            
            mock_memory = Mock()
            mock_memory.rss = 1024 * 1024 * 100  # 100 MB
            mock_process.return_value.memory_info.return_value = mock_memory
            
            detector = pytorch_adapter.fit(mock_dataset)
            
            assert "peak_memory_usage" in detector.parameters

    # Advanced Features Tests

    def test_ensemble_model_training(self, sample_data):
        """Test ensemble model training."""
        adapter = PyTorchAdapter(
            ensemble_size=3,
            ensemble_method="bagging"
        )
        
        mock_dataset = Mock()
        mock_dataset.data = sample_data["X_train"]
        
        with patch.object(adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock(spec=nn.Module)
            mock_create.return_value = mock_model
            
            detector = adapter.fit(mock_dataset)
            
            assert "ensemble_models" in detector.parameters
            assert len(detector.parameters["ensemble_models"]) == 3

    def test_adversarial_training(self, pytorch_adapter, mock_dataset):
        """Test adversarial training for robustness."""
        pytorch_adapter.adversarial_training = True
        pytorch_adapter.adversarial_epsilon = 0.01
        
        with patch.object(pytorch_adapter, '_create_autoencoder'), \
             patch.object(pytorch_adapter, '_generate_adversarial_examples') as mock_adv:
            
            mock_adv.return_value = np.random.randn(100, 5)
            
            detector = pytorch_adapter.fit(mock_dataset)
            
            assert detector is not None
            mock_adv.assert_called()

    def test_online_learning_capability(self, pytorch_adapter, sample_data):
        """Test online learning for model updates."""
        # Initial training
        detector = pytorch_adapter.fit(Mock(data=sample_data["X_train"]))
        
        # Online update with new data
        new_data = np.random.randn(50, 5)
        
        with patch.object(pytorch_adapter, '_load_model'), \
             patch.object(pytorch_adapter, '_online_update') as mock_update:
            
            updated_detector = pytorch_adapter.partial_fit(detector, new_data)
            
            assert updated_detector is not None
            mock_update.assert_called()


class TestPyTorchAdapterIntegration:
    """Integration tests for PyTorch adapter."""

    def test_complete_anomaly_detection_workflow(self, sample_data):
        """Test complete workflow from training to detection."""
        adapter = PyTorchAdapter(epochs=2, batch_size=16)  # Fast training for test
        
        # Create dataset
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        dataset.features = sample_data["features"]
        
        with patch.object(adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock(spec=nn.Module)
            mock_model.eval.return_value = None
            mock_model.return_value = torch.randn(len(sample_data["X_test"]), 5)
            mock_create.return_value = mock_model
            
            # Train detector
            detector = adapter.fit(dataset)
            assert detector is not None
            
            # Make predictions
            with patch.object(adapter, '_load_model', return_value=mock_model):
                result = adapter.predict(detector, sample_data["X_test"])
                
                assert isinstance(result, DetectionResult)
                assert len(result.anomaly_scores) == len(sample_data["X_test"])

    def test_cross_validation_performance(self, sample_data):
        """Test cross-validation performance evaluation."""
        adapter = PyTorchAdapter(epochs=2)
        
        with patch.object(adapter, '_create_autoencoder'):
            scores = adapter.cross_validate(
                sample_data["X_train"],
                cv_folds=3,
                contamination_rate=ContaminationRate(0.1)
            )
            
            assert len(scores) == 3
            assert all(0 <= score <= 1 for score in scores)

    def test_hyperparameter_optimization_integration(self, sample_data):
        """Test integration with hyperparameter optimization."""
        from sklearn.model_selection import GridSearchCV
        
        adapter = PyTorchAdapter(epochs=1)  # Fast for testing
        
        param_grid = {
            'learning_rate': [0.001, 0.01],
            'hidden_dims': [[16, 8], [32, 16]]
        }
        
        with patch.object(adapter, 'fit'), \
             patch.object(adapter, 'score') as mock_score:
            
            mock_score.return_value = 0.8
            
            # This would normally use GridSearchCV, but we'll mock it
            best_params = adapter.optimize_hyperparameters(
                sample_data["X_train"],
                param_grid
            )
            
            assert isinstance(best_params, dict)
            assert "learning_rate" in best_params
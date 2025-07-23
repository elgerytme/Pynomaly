"""Unit tests for DeepLearningAdapter."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Dict

from anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter import (
    DeepLearningAdapter
)


class MockTensorFlowModel:
    """Mock TensorFlow model for testing."""
    
    def __init__(self):
        self.compiled = False
        self.trained = False
    
    def compile(self, optimizer, loss):
        """Mock compile method."""
        self.compiled = True
    
    def fit(self, x, y, epochs, batch_size, verbose, validation_split, callbacks):
        """Mock fit method."""
        if not self.compiled:
            raise ValueError("Model must be compiled before training")
        self.trained = True
    
    def predict(self, data, verbose=0):
        """Mock predict method."""
        if not self.trained:
            raise ValueError("Model not trained")
        # Return reconstructed data with some noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, data.shape)
        return data + noise
    
    def summary(self):
        """Mock summary method."""
        print("Model: Mock TensorFlow Autoencoder")
        print("Total params: 1000")


class MockPyTorchModel:
    """Mock PyTorch model for testing."""
    
    def __init__(self, input_dim, hidden_dims):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.training = True
        self.parameters_list = [Mock() for _ in range(4)]  # Mock parameters
    
    def train(self):
        """Set to training mode."""
        self.training = True
    
    def eval(self):
        """Set to evaluation mode."""
        self.training = False
    
    def parameters(self):
        """Mock parameters method."""
        return self.parameters_list
    
    def forward(self, x):
        """Mock forward pass."""
        # Return input with some noise as reconstruction
        noise = torch.randn_like(x) * 0.1
        return x + noise
    
    def __call__(self, x):
        """Make model callable."""
        return self.forward(x)
    
    def __str__(self):
        return f"MockPyTorchModel(input_dim={self.input_dim})"


class TestDeepLearningAdapter:
    """Test suite for DeepLearningAdapter."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.randn(100, 10).astype(np.float64)
    
    @pytest.fixture
    def small_data(self):
        """Create small dataset for testing."""
        np.random.seed(42)
        return np.random.randn(20, 5).astype(np.float64)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', False)
    def test_initialization_tensorflow_not_available(self):
        """Test initialization when TensorFlow is not available."""
        with pytest.raises(ImportError) as exc_info:
            DeepLearningAdapter(framework="tensorflow")
        
        assert "TensorFlow is required for tensorflow framework" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', False)
    def test_initialization_pytorch_not_available(self):
        """Test initialization when PyTorch is not available."""
        with pytest.raises(ImportError) as exc_info:
            DeepLearningAdapter(framework="pytorch")
        
        assert "PyTorch is required for pytorch framework" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_initialization_tensorflow_defaults(self):
        """Test initialization with TensorFlow defaults."""
        adapter = DeepLearningAdapter(framework="tensorflow")
        
        assert adapter.framework == "tensorflow"
        assert adapter.hidden_dims == [64, 32, 16]
        assert adapter.contamination == 0.1
        assert adapter.epochs == 100
        assert adapter.batch_size == 32
        assert adapter.learning_rate == 0.001
        assert adapter.model is None
        assert adapter._threshold is None
        assert adapter._fitted is False
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', True)
    def test_initialization_pytorch_custom_params(self):
        """Test initialization with PyTorch and custom parameters."""
        adapter = DeepLearningAdapter(
            framework="pytorch",
            hidden_dims=[32, 16],
            contamination=0.05,
            epochs=50,
            batch_size=16,
            learning_rate=0.0001
        )
        
        assert adapter.framework == "pytorch"
        assert adapter.hidden_dims == [32, 16]
        assert adapter.contamination == 0.05
        assert adapter.epochs == 50
        assert adapter.batch_size == 16
        assert adapter.learning_rate == 0.0001
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.tf.keras')
    def test_fit_tensorflow_success(self, mock_keras, sample_data):
        """Test successful fitting with TensorFlow."""
        # Setup mocks
        mock_model = MockTensorFlowModel()
        mock_keras.Model.return_value = mock_model
        mock_keras.layers.Input.return_value = Mock()
        mock_keras.layers.Dense.return_value = Mock()
        mock_keras.optimizers.Adam.return_value = Mock()
        mock_keras.callbacks.EarlyStopping.return_value = Mock()
        
        adapter = DeepLearningAdapter(framework="tensorflow", epochs=5)
        
        # Mock the _calculate_threshold method to avoid calling decision_function
        with patch.object(adapter, '_calculate_threshold'):
            result = adapter.fit(sample_data)
        
        assert result is adapter  # Method chaining
        assert adapter._fitted is True
        assert adapter.model is mock_model
        assert mock_model.compiled is True
        assert mock_model.trained is True
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.torch')
    def test_fit_pytorch_success(self, mock_torch, sample_data):
        """Test successful fitting with PyTorch."""
        # Setup torch mocks
        mock_torch.FloatTensor.return_value = Mock()
        mock_torch.utils.data.TensorDataset.return_value = Mock()
        
        # Mock dataloader to return some batches
        mock_batch = (Mock(), Mock())
        mock_dataloader = Mock()
        mock_dataloader.__iter__.return_value = iter([mock_batch, mock_batch])
        mock_dataloader.__len__.return_value = 2
        mock_torch.utils.data.DataLoader.return_value = mock_dataloader
        
        # Mock optimizer and criterion
        mock_torch.optim.Adam.return_value = Mock()
        mock_torch.nn.MSELoss.return_value = Mock(return_value=Mock(item=Mock(return_value=0.1)))
        
        # Mock model creation by patching the inner class
        with patch.object(DeepLearningAdapter, '_fit_pytorch') as mock_fit:
            adapter = DeepLearningAdapter(framework="pytorch", epochs=5)
            
            # Mock the _calculate_threshold method
            with patch.object(adapter, '_calculate_threshold'):
                adapter.fit(sample_data)
            
            mock_fit.assert_called_once_with(sample_data)
    
    def test_fit_unknown_framework(self, sample_data):
        """Test fitting with unknown framework."""
        # Create adapter with unknown framework (bypassing init validation)
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter.framework = "unknown"
        adapter._fitted = False
        
        with pytest.raises(ValueError) as exc_info:
            adapter.fit(sample_data)
        
        assert "Unknown framework: unknown" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_predict_not_fitted(self, sample_data):
        """Test prediction when model is not fitted."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter._fitted = False
        
        with pytest.raises(ValueError) as exc_info:
            adapter.predict(sample_data)
        
        assert "Model must be fitted before prediction" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_predict_success(self, sample_data):
        """Test successful prediction."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter._fitted = True
        adapter._threshold = 0.5
        
        # Mock decision_function to return some scores
        with patch.object(adapter, 'decision_function', return_value=np.array([0.3, 0.7, 0.2, 0.9])):
            predictions = adapter.predict(np.random.randn(4, 5))
        
        # Scores [0.3, 0.7, 0.2, 0.9] > threshold 0.5 = [0, 1, 0, 1]
        expected = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(predictions, expected)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_decision_function_not_fitted(self, sample_data):
        """Test decision function when model is not fitted."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter._fitted = False
        
        with pytest.raises(ValueError) as exc_info:
            adapter.decision_function(sample_data)
        
        assert "Model must be fitted before scoring" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_decision_function_tensorflow(self, sample_data):
        """Test decision function with TensorFlow."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter.framework = "tensorflow"
        adapter._fitted = True
        adapter.model = MockTensorFlowModel()
        adapter.model.trained = True
        
        scores = adapter.decision_function(sample_data[:5])  # Use smaller data
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 5
        assert all(score >= 0 for score in scores)  # MSE should be non-negative
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.torch')
    def test_decision_function_pytorch(self, mock_torch, sample_data):
        """Test decision function with PyTorch."""
        # Setup torch mocks
        mock_tensor = Mock()
        mock_torch.FloatTensor.return_value = mock_tensor
        mock_torch.mean.return_value = Mock()
        mock_torch.square.return_value = Mock()
        
        # Mock the return value of the computation
        mock_result = Mock()
        mock_result.numpy.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_torch.mean.return_value = mock_result
        
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter.framework = "pytorch"
        adapter._fitted = True
        adapter.model = Mock()
        adapter.model.eval = Mock()
        adapter.model.return_value = mock_tensor
        
        with mock_torch.no_grad():
            scores = adapter.decision_function(sample_data[:5])
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 5
    
    def test_decision_function_unknown_framework(self, sample_data):
        """Test decision function with unknown framework."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter.framework = "unknown"
        adapter._fitted = True
        
        with pytest.raises(ValueError) as exc_info:
            adapter.decision_function(sample_data)
        
        assert "Unknown framework: unknown" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_fit_predict(self, sample_data):
        """Test fit_predict method."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        
        # Mock fit and predict methods
        with patch.object(adapter, 'fit', return_value=adapter) as mock_fit:
            with patch.object(adapter, 'predict', return_value=np.array([0, 1, 0])) as mock_predict:
                result = adapter.fit_predict(sample_data)
        
        mock_fit.assert_called_once_with(sample_data)
        mock_predict.assert_called_once_with(sample_data)
        np.testing.assert_array_equal(result, np.array([0, 1, 0]))
    
    def test_calculate_threshold(self):
        """Test threshold calculation."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter.contamination = 0.1
        
        # Mock decision_function to return known scores
        with patch.object(adapter, 'decision_function', return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])):
            adapter._calculate_threshold(np.random.randn(10, 5))
        
        # With contamination=0.1, threshold should be at 90th percentile
        # For scores [0.1, 0.2, ..., 1.0], 90th percentile is 0.9
        assert adapter._threshold == 0.9
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_get_model_summary_not_fitted(self):
        """Test get_model_summary when model is not fitted."""
        adapter = DeepLearningAdapter(framework="tensorflow")
        
        summary = adapter.get_model_summary()
        assert summary == "Model not fitted"
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_get_model_summary_tensorflow(self):
        """Test get_model_summary with TensorFlow."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter.framework = "tensorflow"
        adapter._fitted = True
        adapter.model = MockTensorFlowModel()
        
        summary = adapter.get_model_summary()
        
        assert "Mock TensorFlow Autoencoder" in summary
        assert "Total params: 1000" in summary
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', True)
    def test_get_model_summary_pytorch(self):
        """Test get_model_summary with PyTorch."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter.framework = "pytorch"
        adapter._fitted = True
        adapter.model = MockPyTorchModel(10, [32, 16])
        
        summary = adapter.get_model_summary()
        
        assert "MockPyTorchModel" in summary
        assert "input_dim=10" in summary
    
    def test_get_model_summary_unknown_framework(self):
        """Test get_model_summary with unknown framework."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter.framework = "unknown"
        adapter._fitted = True
        
        summary = adapter.get_model_summary()
        assert summary == "Unknown framework"
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_get_parameters(self):
        """Test get_parameters method."""
        adapter = DeepLearningAdapter(
            framework="tensorflow",
            hidden_dims=[32, 16],
            contamination=0.05,
            epochs=50,
            batch_size=16,
            learning_rate=0.0001
        )
        adapter._threshold = 0.75
        
        params = adapter.get_parameters()
        
        assert params["framework"] == "tensorflow"
        assert params["hidden_dims"] == [32, 16]
        assert params["contamination"] == 0.05
        assert params["epochs"] == 50
        assert params["batch_size"] == 16
        assert params["learning_rate"] == 0.0001
        assert params["threshold"] == 0.75
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', False)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', False)
    def test_get_available_frameworks_none_available(self):
        """Test get_available_frameworks when no frameworks are available."""
        frameworks = DeepLearningAdapter.get_available_frameworks()
        assert frameworks == []
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', False)
    def test_get_available_frameworks_tensorflow_only(self):
        """Test get_available_frameworks when only TensorFlow is available."""
        frameworks = DeepLearningAdapter.get_available_frameworks()
        assert frameworks == ["tensorflow"]
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', False)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', True)
    def test_get_available_frameworks_pytorch_only(self):
        """Test get_available_frameworks when only PyTorch is available."""
        frameworks = DeepLearningAdapter.get_available_frameworks()
        assert frameworks == ["pytorch"]
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', True)
    def test_get_available_frameworks_both_available(self):
        """Test get_available_frameworks when both frameworks are available."""
        frameworks = DeepLearningAdapter.get_available_frameworks()
        assert frameworks == ["tensorflow", "pytorch"]
    
    @pytest.mark.parametrize("input_dim,expected", [
        (5, [8, 4]),
        (25, [32, 16, 8]),
        (75, [64, 32, 16]),
        (150, [128, 64, 32, 16]),
    ])
    def test_create_default_architecture(self, input_dim, expected):
        """Test create_default_architecture with different input dimensions."""
        architecture = DeepLearningAdapter.create_default_architecture(input_dim)
        assert architecture == expected
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_str_representation(self):
        """Test string representation."""
        adapter = DeepLearningAdapter(framework="tensorflow", hidden_dims=[32, 16])
        
        str_repr = str(adapter)
        assert "DeepLearningAdapter" in str_repr
        assert "framework='tensorflow'" in str_repr
        assert "hidden_dims=[32, 16]" in str_repr
        assert "fitted=False" in str_repr
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_repr_representation(self):
        """Test detailed string representation."""
        adapter = DeepLearningAdapter(
            framework="tensorflow",
            hidden_dims=[32, 16],
            contamination=0.05,
            epochs=50,
            batch_size=16
        )
        
        repr_str = repr(adapter)
        assert "DeepLearningAdapter" in repr_str
        assert "framework='tensorflow'" in repr_str
        assert "hidden_dims=[32, 16]" in repr_str
        assert "contamination=0.05" in repr_str
        assert "epochs=50" in repr_str
        assert "batch_size=16" in repr_str
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.tf.keras')
    def test_tensorflow_architecture_building(self, mock_keras, sample_data):
        """Test TensorFlow autoencoder architecture building."""
        # Setup detailed mocks for architecture verification
        mock_input = Mock()
        mock_dense_layers = [Mock() for _ in range(6)]  # 3 encoder + 3 decoder layers
        mock_keras.layers.Input.return_value = mock_input
        mock_keras.layers.Dense.side_effect = mock_dense_layers
        mock_keras.Model.return_value = MockTensorFlowModel()
        mock_keras.optimizers.Adam.return_value = Mock()
        mock_keras.callbacks.EarlyStopping.return_value = Mock()
        
        adapter = DeepLearningAdapter(
            framework="tensorflow",
            hidden_dims=[8, 4, 2],
            epochs=1  # Quick training
        )
        
        with patch.object(adapter, '_calculate_threshold'):
            adapter.fit(sample_data)
        
        # Verify Input layer was created with correct shape
        mock_keras.layers.Input.assert_called_once_with(shape=(10,))  # sample_data has 10 features
        
        # Verify Dense layers were created (3 encoder + 3 decoder)
        assert mock_keras.layers.Dense.call_count == 6
        
        # Check encoder layers (with ReLU activation)
        encoder_calls = mock_keras.layers.Dense.call_args_list[:3]
        assert encoder_calls[0][0] == (8,)
        assert encoder_calls[0][1]['activation'] == 'relu'
        assert encoder_calls[1][0] == (4,)
        assert encoder_calls[1][1]['activation'] == 'relu'
        assert encoder_calls[2][0] == (2,)
        assert encoder_calls[2][1]['activation'] == 'relu'
        
        # Check decoder layers (first 2 with ReLU, last with linear)
        decoder_calls = mock_keras.layers.Dense.call_args_list[3:]
        assert decoder_calls[0][0] == (4,)
        assert decoder_calls[0][1]['activation'] == 'relu'
        assert decoder_calls[1][0] == (8,)
        assert decoder_calls[1][1]['activation'] == 'relu'
        assert decoder_calls[2][0] == (10,)  # Back to input dimension
        assert decoder_calls[2][1]['activation'] == 'linear'
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.torch')
    def test_pytorch_training_loop(self, mock_torch, sample_data):
        """Test PyTorch training loop execution."""
        # Mock torch components
        mock_tensor = Mock()
        mock_torch.FloatTensor.return_value = mock_tensor
        mock_torch.utils.data.TensorDataset.return_value = Mock()
        
        # Create mock dataloader with specific batches
        mock_batch1 = (Mock(), Mock())
        mock_batch2 = (Mock(), Mock())
        mock_dataloader = Mock()
        mock_dataloader.__iter__.return_value = iter([mock_batch1, mock_batch2])
        mock_dataloader.__len__.return_value = 2
        mock_torch.utils.data.DataLoader.return_value = mock_dataloader
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_torch.optim.Adam.return_value = mock_optimizer
        
        # Mock loss function
        mock_loss_value = Mock()
        mock_loss_value.item.return_value = 0.1
        mock_loss_value.backward = Mock()
        mock_criterion = Mock(return_value=mock_loss_value)
        mock_torch.nn.MSELoss.return_value = mock_criterion
        
        adapter = DeepLearningAdapter(framework="pytorch", epochs=2, hidden_dims=[4, 2])
        
        with patch.object(adapter, '_calculate_threshold'):
            adapter.fit(sample_data)
        
        # Training should have been called
        assert adapter.model is not None
        
        # Verify optimizer was called (2 epochs * 2 batches = 4 times)
        assert mock_optimizer.zero_grad.call_count == 4
        assert mock_optimizer.step.call_count == 4
        assert mock_loss_value.backward.call_count == 4
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TF_AVAILABLE', True)
    def test_score_tensorflow_mse_calculation(self):
        """Test TensorFlow MSE calculation for scoring."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter.framework = "tensorflow"
        adapter._fitted = True
        
        # Create mock model that returns slightly different reconstruction
        mock_model = Mock()
        input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        reconstructed_data = np.array([[1.1, 1.9, 3.1], [3.9, 5.1, 5.9]])
        mock_model.predict.return_value = reconstructed_data
        
        adapter.model = mock_model
        
        scores = adapter._score_tensorflow(input_data)
        
        # Calculate expected MSE manually
        expected_scores = np.mean(np.square(input_data - reconstructed_data), axis=1)
        np.testing.assert_array_almost_equal(scores, expected_scores)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.TORCH_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.deeplearning_adapter.torch')
    def test_score_pytorch_mse_calculation(self, mock_torch):
        """Test PyTorch MSE calculation for scoring."""
        adapter = DeepLearningAdapter.__new__(DeepLearningAdapter)
        adapter.framework = "pytorch"
        adapter._fitted = True
        
        # Setup torch mocks
        input_tensor = Mock()
        reconstructed_tensor = Mock()
        mock_torch.FloatTensor.return_value = input_tensor
        
        # Mock model
        mock_model = Mock()
        mock_model.return_value = reconstructed_tensor
        mock_model.eval = Mock()
        adapter.model = mock_model
        
        # Mock torch operations
        mock_squared_diff = Mock()
        mock_torch.square.return_value = mock_squared_diff
        
        mock_mean_result = Mock()
        expected_scores = np.array([0.1, 0.2, 0.3])
        mock_mean_result.numpy.return_value = expected_scores
        mock_torch.mean.return_value = mock_mean_result
        
        with mock_torch.no_grad():
            scores = adapter._score_pytorch(np.random.randn(3, 5))
        
        # Verify model was set to eval mode
        mock_model.eval.assert_called_once()
        
        # Verify torch operations were called
        mock_torch.FloatTensor.assert_called_once()
        mock_torch.square.assert_called_once()
        mock_torch.mean.assert_called_once()
        
        np.testing.assert_array_equal(scores, expected_scores)
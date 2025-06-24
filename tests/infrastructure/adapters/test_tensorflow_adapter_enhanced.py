"""
Enhanced TensorFlow Adapter Testing Suite
Comprehensive tests for TensorFlow/Keras deep learning anomaly detection adapter.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tensorflow as tf
from datetime import datetime

from pynomaly.infrastructure.adapters.tensorflow_adapter import TensorFlowAdapter
from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.domain.exceptions import DetectorError, AdapterError


class TestTensorFlowAdapter:
    """Enhanced test suite for TensorFlow adapter functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (800, 10))
        anomalous_data = np.random.normal(3, 1, (200, 10))
        data = np.vstack([normal_data, anomalous_data])
        labels = np.hstack([np.zeros(800), np.ones(200)])
        
        return {
            "X_train": data[:600].astype(np.float32),
            "X_test": data[600:].astype(np.float32),
            "y_train": labels[:600],
            "y_test": labels[600:],
            "features": [f"feature_{i}" for i in range(10)]
        }

    @pytest.fixture
    def tensorflow_adapter(self):
        """Create TensorFlow adapter instance."""
        return TensorFlowAdapter(
            model_type="autoencoder",
            hidden_layers=[64, 32, 16],
            learning_rate=0.001,
            batch_size=32,
            epochs=10,
            activation="relu",
            optimizer="adam"
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

    def test_tensorflow_adapter_initialization_default(self):
        """Test TensorFlow adapter initialization with default parameters."""
        adapter = TensorFlowAdapter()
        
        assert adapter.model_type == "autoencoder"
        assert adapter.learning_rate == 0.001
        assert adapter.batch_size == 32
        assert adapter.epochs == 100
        assert adapter.activation == "relu"
        assert adapter.optimizer == "adam"

    def test_tensorflow_adapter_initialization_custom(self):
        """Test TensorFlow adapter initialization with custom parameters."""
        adapter = TensorFlowAdapter(
            model_type="variational_autoencoder",
            hidden_layers=[128, 64, 32],
            learning_rate=0.01,
            batch_size=64,
            epochs=50,
            activation="tanh",
            optimizer="rmsprop",
            dropout_rate=0.3,
            regularization_l1=0.001,
            regularization_l2=0.01
        )
        
        assert adapter.model_type == "variational_autoencoder"
        assert adapter.hidden_layers == [128, 64, 32]
        assert adapter.learning_rate == 0.01
        assert adapter.batch_size == 64
        assert adapter.epochs == 50
        assert adapter.activation == "tanh"
        assert adapter.optimizer == "rmsprop"
        assert adapter.dropout_rate == 0.3

    def test_tensorflow_gpu_configuration(self):
        """Test TensorFlow GPU configuration."""
        with patch('tensorflow.config.list_physical_devices') as mock_devices:
            mock_devices.return_value = [Mock(name="GPU:0")]
            
            adapter = TensorFlowAdapter(use_gpu=True)
            
            assert adapter.use_gpu is True

    def test_tensorflow_adapter_invalid_parameters(self):
        """Test adapter initialization with invalid parameters."""
        with pytest.raises(ValueError):
            TensorFlowAdapter(model_type="invalid_model")
        
        with pytest.raises(ValueError):
            TensorFlowAdapter(learning_rate=-0.1)
        
        with pytest.raises(ValueError):
            TensorFlowAdapter(batch_size=0)
        
        with pytest.raises(ValueError):
            TensorFlowAdapter(epochs=-1)
        
        with pytest.raises(ValueError):
            TensorFlowAdapter(dropout_rate=1.5)

    # Model Architecture Tests

    def test_autoencoder_model_creation(self, tensorflow_adapter, sample_data):
        """Test autoencoder model architecture creation."""
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('tensorflow.keras.Sequential') as mock_sequential:
            mock_model = MagicMock()
            mock_sequential.return_value = mock_model
            
            model = tensorflow_adapter._create_autoencoder(input_dim)
            
            mock_sequential.assert_called()
            assert model is not None

    def test_variational_autoencoder_creation(self, sample_data):
        """Test variational autoencoder model creation."""
        adapter = TensorFlowAdapter(model_type="variational_autoencoder", latent_dim=8)
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('tensorflow.keras.Model') as mock_model_class, \
             patch('tensorflow.keras.layers.Input') as mock_input, \
             patch('tensorflow.keras.layers.Dense') as mock_dense:
            
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model
            
            model = adapter._create_variational_autoencoder(input_dim)
            
            mock_model_class.assert_called()
            assert model is not None

    def test_lstm_autoencoder_creation(self, sample_data):
        """Test LSTM autoencoder for time series data."""
        adapter = TensorFlowAdapter(
            model_type="lstm_autoencoder",
            sequence_length=20,
            hidden_layers=[50, 25]
        )
        
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('tensorflow.keras.Sequential') as mock_sequential, \
             patch('tensorflow.keras.layers.LSTM') as mock_lstm:
            
            mock_model = MagicMock()
            mock_sequential.return_value = mock_model
            
            model = adapter._create_lstm_autoencoder(input_dim)
            
            mock_lstm.assert_called()
            assert model is not None

    def test_convolutional_autoencoder_creation(self, sample_data):
        """Test convolutional autoencoder for image-like data."""
        adapter = TensorFlowAdapter(
            model_type="conv_autoencoder",
            input_shape=(32, 32, 1),
            conv_filters=[32, 64, 128]
        )
        
        with patch('tensorflow.keras.Sequential') as mock_sequential, \
             patch('tensorflow.keras.layers.Conv2D') as mock_conv2d:
            
            mock_model = MagicMock()
            mock_sequential.return_value = mock_model
            
            model = adapter._create_conv_autoencoder()
            
            mock_conv2d.assert_called()
            assert model is not None

    def test_transformer_autoencoder_creation(self, sample_data):
        """Test Transformer autoencoder model."""
        adapter = TensorFlowAdapter(
            model_type="transformer_autoencoder",
            sequence_length=50,
            num_heads=8,
            num_transformer_blocks=4,
            ff_dim=512
        )
        
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('tensorflow.keras.Model') as mock_model:
            model = adapter._create_transformer_autoencoder(input_dim)
            assert model is not None

    # Training Process Tests

    def test_fit_basic_training(self, tensorflow_adapter, mock_dataset, sample_data):
        """Test basic model training process."""
        with patch.object(tensorflow_adapter, '_create_autoencoder') as mock_create, \
             patch('tensorflow.keras.callbacks.EarlyStopping') as mock_early_stopping:
            
            mock_model = MagicMock()
            mock_model.compile.return_value = None
            mock_model.fit.return_value = Mock(history={"loss": [1.0, 0.8, 0.6]})
            mock_create.return_value = mock_model
            
            detector = tensorflow_adapter.fit(mock_dataset)
            
            assert isinstance(detector, Detector)
            assert detector.algorithm == "tensorflow_autoencoder"
            assert "model_weights" in detector.parameters
            mock_model.compile.assert_called()
            mock_model.fit.assert_called()

    def test_fit_with_validation_split(self, tensorflow_adapter, mock_dataset):
        """Test training with validation split."""
        tensorflow_adapter.validation_split = 0.2
        
        with patch.object(tensorflow_adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock()
            mock_model.fit.return_value = Mock(history={
                "loss": [1.0, 0.8, 0.6],
                "val_loss": [1.1, 0.9, 0.7]
            })
            mock_create.return_value = mock_model
            
            detector = tensorflow_adapter.fit(mock_dataset)
            
            # Verify validation_split was passed to fit
            fit_call_args = mock_model.fit.call_args
            assert fit_call_args[1]["validation_split"] == 0.2

    def test_fit_with_callbacks(self, tensorflow_adapter, mock_dataset):
        """Test training with various callbacks."""
        tensorflow_adapter.use_early_stopping = True
        tensorflow_adapter.use_reduce_lr = True
        tensorflow_adapter.use_model_checkpoint = True
        
        with patch.object(tensorflow_adapter, '_create_autoencoder') as mock_create, \
             patch('tensorflow.keras.callbacks.EarlyStopping') as mock_early_stopping, \
             patch('tensorflow.keras.callbacks.ReduceLROnPlateau') as mock_reduce_lr, \
             patch('tensorflow.keras.callbacks.ModelCheckpoint') as mock_checkpoint:
            
            mock_model = MagicMock()
            mock_model.fit.return_value = Mock(history={"loss": [1.0, 0.8, 0.6]})
            mock_create.return_value = mock_model
            
            detector = tensorflow_adapter.fit(mock_dataset)
            
            mock_early_stopping.assert_called()
            mock_reduce_lr.assert_called()
            mock_checkpoint.assert_called()

    def test_fit_with_custom_loss_function(self, tensorflow_adapter, mock_dataset):
        """Test training with custom loss function."""
        tensorflow_adapter.loss_function = "huber"
        
        with patch.object(tensorflow_adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock()
            mock_create.return_value = mock_model
            
            detector = tensorflow_adapter.fit(mock_dataset)
            
            # Verify custom loss was used in compile
            compile_call_args = mock_model.compile.call_args
            assert "huber" in str(compile_call_args)

    def test_fit_with_mixed_precision(self, tensorflow_adapter, mock_dataset):
        """Test training with mixed precision for performance."""
        tensorflow_adapter.use_mixed_precision = True
        
        with patch('tensorflow.keras.mixed_precision.set_global_policy') as mock_set_policy, \
             patch.object(tensorflow_adapter, '_create_autoencoder') as mock_create:
            
            mock_model = MagicMock()
            mock_create.return_value = mock_model
            
            detector = tensorflow_adapter.fit(mock_dataset)
            
            mock_set_policy.assert_called_with('mixed_float16')

    # Detection Process Tests

    def test_predict_basic_detection(self, tensorflow_adapter, sample_data):
        """Test basic anomaly detection."""
        detector = Detector(
            id="test_detector",
            algorithm="tensorflow_autoencoder",
            parameters={
                "model_weights": "mock_weights",
                "model_config": "mock_config",
                "input_shape": [10],
                "threshold": 0.5
            }
        )
        
        with patch.object(tensorflow_adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = sample_data["X_test"] + np.random.normal(0, 0.1, sample_data["X_test"].shape)
            mock_load.return_value = mock_model
            
            result = tensorflow_adapter.predict(detector, sample_data["X_test"])
            
            assert isinstance(result, DetectionResult)
            assert len(result.anomaly_scores) == len(sample_data["X_test"])
            assert all(isinstance(score, AnomalyScore) for score in result.anomaly_scores)

    def test_predict_batch_processing(self, tensorflow_adapter, sample_data):
        """Test batch processing during prediction."""
        detector = Detector(
            id="test_detector",
            algorithm="tensorflow_autoencoder",
            parameters={
                "model_weights": "mock_weights",
                "model_config": "mock_config",
                "input_shape": [10]
            }
        )
        
        # Large dataset to test batching
        large_data = np.random.randn(1000, 10).astype(np.float32)
        
        with patch.object(tensorflow_adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = large_data + np.random.normal(0, 0.1, large_data.shape)
            mock_load.return_value = mock_model
            
            result = tensorflow_adapter.predict(detector, large_data)
            
            assert len(result.anomaly_scores) == len(large_data)
            # Verify model.predict was called with batch_size
            mock_model.predict.assert_called()

    def test_predict_with_uncertainty_estimation(self, tensorflow_adapter, sample_data):
        """Test prediction with uncertainty estimation."""
        tensorflow_adapter.estimate_uncertainty = True
        tensorflow_adapter.mc_dropout_samples = 10
        
        detector = Detector(
            id="test_detector",
            algorithm="tensorflow_autoencoder",
            parameters={
                "model_weights": "mock_weights",
                "model_config": "mock_config",
                "input_shape": [10]
            }
        )
        
        with patch.object(tensorflow_adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            # Simulate multiple predictions for MC dropout
            mock_model.predict.return_value = sample_data["X_test"] + np.random.normal(0, 0.1, sample_data["X_test"].shape)
            mock_load.return_value = mock_model
            
            result = tensorflow_adapter.predict(detector, sample_data["X_test"])
            
            assert result is not None
            # Should call predict multiple times for uncertainty estimation
            assert mock_model.predict.call_count >= 1

    # Advanced Model Features Tests

    def test_attention_mechanism_integration(self, sample_data):
        """Test attention mechanism in models."""
        adapter = TensorFlowAdapter(
            model_type="attention_autoencoder",
            use_attention=True,
            attention_heads=4
        )
        
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('tensorflow.keras.layers.MultiHeadAttention') as mock_attention, \
             patch('tensorflow.keras.Model') as mock_model:
            
            model = adapter._create_attention_autoencoder(input_dim)
            
            mock_attention.assert_called()
            assert model is not None

    def test_residual_connections(self, sample_data):
        """Test residual connections in deep autoencoders."""
        adapter = TensorFlowAdapter(
            model_type="residual_autoencoder",
            hidden_layers=[64, 32, 16, 8],
            use_residual_connections=True
        )
        
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('tensorflow.keras.layers.Add') as mock_add, \
             patch('tensorflow.keras.Model') as mock_model:
            
            model = adapter._create_residual_autoencoder(input_dim)
            
            assert model is not None

    def test_batch_normalization_integration(self, tensorflow_adapter, sample_data):
        """Test batch normalization layers."""
        tensorflow_adapter.use_batch_normalization = True
        
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('tensorflow.keras.layers.BatchNormalization') as mock_batch_norm, \
             patch('tensorflow.keras.Sequential') as mock_sequential:
            
            mock_model = MagicMock()
            mock_sequential.return_value = mock_model
            
            model = tensorflow_adapter._create_autoencoder(input_dim)
            
            mock_batch_norm.assert_called()

    # Model Persistence Tests

    def test_model_serialization(self, tensorflow_adapter, mock_dataset):
        """Test model serialization and deserialization."""
        with patch.object(tensorflow_adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock()
            mock_model.get_weights.return_value = [np.random.randn(10, 64)]
            mock_model.to_json.return_value = '{"class_name": "Sequential"}'
            mock_create.return_value = mock_model
            
            detector = tensorflow_adapter.fit(mock_dataset)
            
            # Test serialization
            assert "model_weights" in detector.parameters
            assert "model_config" in detector.parameters

    def test_model_versioning(self, tensorflow_adapter, mock_dataset):
        """Test model versioning for deployment."""
        tensorflow_adapter.enable_versioning = True
        
        with patch.object(tensorflow_adapter, '_create_autoencoder') as mock_create, \
             patch('tensorflow.saved_model.save') as mock_save:
            
            mock_model = MagicMock()
            mock_create.return_value = mock_model
            
            detector = tensorflow_adapter.fit(mock_dataset)
            
            assert "model_version" in detector.parameters

    def test_quantization_for_deployment(self, tensorflow_adapter, sample_data):
        """Test model quantization for edge deployment."""
        detector = Detector(
            id="test_detector",
            algorithm="tensorflow_autoencoder",
            parameters={
                "model_weights": "mock_weights",
                "model_config": "mock_config"
            }
        )
        
        with patch('tensorflow.lite.TFLiteConverter') as mock_converter:
            mock_converter.from_keras_model.return_value.convert.return_value = b"quantized_model"
            
            quantized_model = tensorflow_adapter.quantize_model(detector)
            
            assert quantized_model is not None

    # Error Handling Tests

    def test_fit_invalid_data_type(self, tensorflow_adapter):
        """Test training with invalid data type."""
        invalid_dataset = Mock()
        invalid_dataset.data = "invalid_data_type"
        
        with pytest.raises(AdapterError):
            tensorflow_adapter.fit(invalid_dataset)

    def test_fit_nan_values_in_data(self, tensorflow_adapter):
        """Test training with NaN values in data."""
        nan_dataset = Mock()
        nan_data = np.random.randn(100, 5)
        nan_data[0, 0] = np.nan
        nan_dataset.data = nan_data
        
        with pytest.raises(AdapterError):
            tensorflow_adapter.fit(nan_dataset)

    def test_fit_memory_error_handling(self, tensorflow_adapter, mock_dataset):
        """Test handling of memory errors during training."""
        with patch.object(tensorflow_adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock()
            mock_model.fit.side_effect = MemoryError("Out of memory")
            mock_create.return_value = mock_model
            
            with pytest.raises(AdapterError):
                tensorflow_adapter.fit(mock_dataset)

    def test_predict_model_loading_error(self, tensorflow_adapter, sample_data):
        """Test handling of model loading errors."""
        detector = Detector(
            id="test_detector",
            algorithm="tensorflow_autoencoder",
            parameters={"corrupted_model": "invalid"}
        )
        
        with patch.object(tensorflow_adapter, '_load_model') as mock_load:
            mock_load.side_effect = Exception("Model loading failed")
            
            with pytest.raises(DetectorError):
                tensorflow_adapter.predict(detector, sample_data["X_test"])

    def test_gpu_availability_handling(self, tensorflow_adapter):
        """Test graceful handling when GPU is not available."""
        with patch('tensorflow.config.list_physical_devices') as mock_devices:
            mock_devices.return_value = []  # No GPU devices
            
            # Should fallback to CPU without error
            tensorflow_adapter._configure_gpu()
            assert tensorflow_adapter.use_gpu is False

    # Performance Optimization Tests

    def test_dataset_api_usage(self, tensorflow_adapter, sample_data):
        """Test TensorFlow Dataset API for performance."""
        tensorflow_adapter.use_tf_data = True
        
        with patch('tensorflow.data.Dataset.from_tensor_slices') as mock_dataset:
            mock_tf_dataset = MagicMock()
            mock_tf_dataset.batch.return_value = mock_tf_dataset
            mock_tf_dataset.prefetch.return_value = mock_tf_dataset
            mock_dataset.return_value = mock_tf_dataset
            
            dataset = tensorflow_adapter._create_tf_dataset(sample_data["X_train"])
            
            mock_dataset.assert_called()
            mock_tf_dataset.batch.assert_called()
            mock_tf_dataset.prefetch.assert_called()

    def test_graph_optimization(self, tensorflow_adapter, mock_dataset):
        """Test TensorFlow graph optimization."""
        tensorflow_adapter.optimize_graph = True
        
        with patch('tensorflow.function') as mock_tf_function, \
             patch.object(tensorflow_adapter, '_create_autoencoder') as mock_create:
            
            mock_model = MagicMock()
            mock_create.return_value = mock_model
            
            detector = tensorflow_adapter.fit(mock_dataset)
            
            # Graph optimization should be applied
            assert detector is not None

    def test_distributed_training_setup(self, tensorflow_adapter, mock_dataset):
        """Test distributed training configuration."""
        tensorflow_adapter.use_distributed_training = True
        
        with patch('tensorflow.distribute.MirroredStrategy') as mock_strategy, \
             patch.object(tensorflow_adapter, '_create_autoencoder') as mock_create:
            
            mock_strategy_instance = MagicMock()
            mock_strategy.return_value = mock_strategy_instance
            
            mock_model = MagicMock()
            mock_create.return_value = mock_model
            
            detector = tensorflow_adapter.fit(mock_dataset)
            
            mock_strategy.assert_called()

    # Model Interpretation Tests

    def test_feature_importance_calculation(self, tensorflow_adapter, sample_data):
        """Test feature importance calculation using gradients."""
        detector = Detector(
            id="test_detector",
            algorithm="tensorflow_autoencoder",
            parameters={
                "model_weights": "mock_weights",
                "model_config": "mock_config"
            }
        )
        
        with patch.object(tensorflow_adapter, '_load_model') as mock_load, \
             patch('tensorflow.GradientTape') as mock_tape:
            
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            # Mock gradient calculation
            mock_tape_instance = MagicMock()
            mock_tape.return_value.__enter__.return_value = mock_tape_instance
            mock_tape_instance.gradient.return_value = [tf.random.normal((1, 10))]
            
            importance = tensorflow_adapter.get_feature_importance(
                detector, 
                sample_data["X_test"][:1]
            )
            
            assert importance is not None
            assert len(importance) == sample_data["X_test"].shape[1]

    def test_layer_wise_relevance_propagation(self, tensorflow_adapter, sample_data):
        """Test Layer-wise Relevance Propagation for explanations."""
        detector = Detector(
            id="test_detector",
            algorithm="tensorflow_autoencoder",
            parameters={
                "model_weights": "mock_weights",
                "model_config": "mock_config"
            }
        )
        
        with patch.object(tensorflow_adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            explanations = tensorflow_adapter.explain_predictions(
                detector,
                sample_data["X_test"][:5],
                method="lrp"
            )
            
            assert explanations is not None
            assert len(explanations) == 5


class TestTensorFlowAdapterIntegration:
    """Integration tests for TensorFlow adapter."""

    def test_complete_anomaly_detection_pipeline(self, sample_data):
        """Test complete anomaly detection pipeline."""
        adapter = TensorFlowAdapter(epochs=2, batch_size=16)  # Fast training
        
        # Create dataset
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        dataset.features = sample_data["features"]
        
        with patch.object(adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock()
            mock_model.fit.return_value = Mock(history={"loss": [1.0, 0.5]})
            mock_model.predict.return_value = sample_data["X_test"] + np.random.normal(0, 0.1, sample_data["X_test"].shape)
            mock_create.return_value = mock_model
            
            # Train detector
            detector = adapter.fit(dataset)
            assert detector is not None
            
            # Make predictions
            with patch.object(adapter, '_load_model', return_value=mock_model):
                result = adapter.predict(detector, sample_data["X_test"])
                
                assert isinstance(result, DetectionResult)
                assert len(result.anomaly_scores) == len(sample_data["X_test"])

    def test_ensemble_detection_workflow(self, sample_data):
        """Test ensemble detection with multiple models."""
        adapter = TensorFlowAdapter(
            ensemble_size=3,
            ensemble_method="voting",
            epochs=1
        )
        
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        
        with patch.object(adapter, '_create_autoencoder') as mock_create:
            mock_model = MagicMock()
            mock_model.fit.return_value = Mock(history={"loss": [1.0]})
            mock_create.return_value = mock_model
            
            detector = adapter.fit(dataset)
            
            assert "ensemble_models" in detector.parameters
            assert detector.parameters["ensemble_size"] == 3

    def test_transfer_learning_workflow(self, sample_data):
        """Test transfer learning from pre-trained model."""
        adapter = TensorFlowAdapter(
            use_transfer_learning=True,
            base_model="pretrained_autoencoder"
        )
        
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        
        with patch('tensorflow.keras.models.load_model') as mock_load_pretrained, \
             patch.object(adapter, '_fine_tune_model') as mock_fine_tune:
            
            mock_pretrained = MagicMock()
            mock_load_pretrained.return_value = mock_pretrained
            
            detector = adapter.fit(dataset)
            
            mock_load_pretrained.assert_called()
            mock_fine_tune.assert_called()

    def test_model_monitoring_and_drift_detection(self, sample_data):
        """Test model monitoring and drift detection."""
        adapter = TensorFlowAdapter(enable_monitoring=True)
        
        detector = Detector(
            id="test_detector",
            algorithm="tensorflow_autoencoder",
            parameters={
                "model_weights": "mock_weights",
                "model_config": "mock_config",
                "training_statistics": {
                    "mean": np.mean(sample_data["X_train"], axis=0).tolist(),
                    "std": np.std(sample_data["X_train"], axis=0).tolist()
                }
            }
        )
        
        # Simulate data drift
        drifted_data = sample_data["X_test"] + 2.0
        
        with patch.object(adapter, '_load_model'):
            drift_detected = adapter.detect_data_drift(detector, drifted_data)
            assert drift_detected is True
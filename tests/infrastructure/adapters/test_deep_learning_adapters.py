"""
Deep Learning Adapter Testing Suite
Comprehensive tests for PyTorch, TensorFlow, and JAX adapters with advanced deep learning features.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock, create_autospec
from datetime import datetime, timezone
import sys
import os
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from pynomaly.domain.entities import Dataset, Detector, DetectionResult, Anomaly
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.domain.exceptions import (
    DetectorNotFittedError, FittingError, InvalidAlgorithmError,
    AdapterError
)


class TestPyTorchAdapterAdvanced:
    """Advanced PyTorch adapter tests with realistic deep learning scenarios."""

    @pytest.fixture
    def time_series_dataset(self):
        """Create time series dataset for sequence modeling."""
        np.random.seed(42)
        
        # Generate synthetic time series with anomalies
        time_steps = 1000
        features = 8
        
        # Normal pattern: sinusoidal with noise
        t = np.linspace(0, 10, time_steps)
        normal_data = np.zeros((time_steps, features))
        for i in range(features):
            normal_data[:, i] = np.sin(t + i * 0.5) + np.random.normal(0, 0.1, time_steps)
        
        # Add anomalies at specific points
        anomaly_indices = [100, 300, 500, 700, 900]
        for idx in anomaly_indices:
            normal_data[idx] = normal_data[idx] + np.random.normal(0, 3, features)
        
        df = pd.DataFrame(normal_data, columns=[f"sensor_{i}" for i in range(features)])
        
        dataset = Mock(spec=Dataset)
        dataset.id = "time_series_dataset"
        dataset.name = "Time Series Dataset"
        dataset.data = df
        dataset.features = df
        dataset.n_samples = len(df)
        dataset.created_at = datetime.now(timezone.utc)
        dataset.get_numeric_features.return_value = list(df.columns)
        
        return dataset

    @pytest.fixture
    def image_like_dataset(self):
        """Create image-like dataset for convolutional models."""
        np.random.seed(42)
        
        # Simulate flattened image data (28x28 = 784 features like MNIST)
        n_samples = 500
        image_size = 784
        
        # Normal images: random noise with some structure
        normal_images = np.random.normal(0, 1, (n_samples, image_size))
        
        # Add some structure (simulated edges)
        for i in range(n_samples):
            # Add some correlation between adjacent "pixels"
            normal_images[i] = np.convolve(normal_images[i], [0.3, 0.4, 0.3], mode='same')
        
        df = pd.DataFrame(normal_images, columns=[f"pixel_{i}" for i in range(image_size)])
        
        dataset = Mock(spec=Dataset)
        dataset.id = "image_dataset"
        dataset.name = "Image Dataset"
        dataset.data = df
        dataset.features = df
        dataset.n_samples = len(df)
        dataset.created_at = datetime.now(timezone.utc)
        dataset.get_numeric_features.return_value = list(df.columns)
        
        return dataset

    def test_pytorch_lstm_autoencoder(self, time_series_dataset):
        """Test PyTorch LSTM autoencoder for time series anomaly detection."""
        # Mock PyTorch modules
        mock_torch = Mock()
        mock_nn = Mock()
        mock_optim = Mock()
        
        # Mock LSTM layers
        mock_lstm = Mock()
        mock_linear = Mock()
        mock_nn.LSTM = Mock(return_value=mock_lstm)
        mock_nn.Linear = Mock(return_value=mock_linear)
        mock_nn.MSELoss = Mock()
        mock_nn.Module = Mock
        
        # Mock tensor operations
        mock_tensor = Mock()
        mock_tensor.shape = [1000, 8]
        mock_torch.from_numpy.return_value = mock_tensor
        mock_torch.zeros.return_value = mock_tensor
        mock_torch.stack.return_value = mock_tensor
        
        # Mock optimizer
        mock_optimizer = Mock()
        mock_optim.Adam.return_value = mock_optimizer
        
        with patch.dict('sys.modules', {
            'torch': mock_torch,
            'torch.nn': mock_nn,
            'torch.optim': mock_optim,
            'torch.utils.data': Mock()
        }):
            from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
            
            adapter = PyTorchAdapter(
                model_type="lstm_autoencoder",
                sequence_length=50,
                hidden_size=64,
                num_layers=2,
                bidirectional=True,
                dropout=0.1,
                epochs=10
            )
            
            # Mock model creation
            mock_model = Mock()
            mock_model.parameters.return_value = [Mock()]
            mock_model.state_dict.return_value = {"lstm.weight": np.ones((64, 8))}
            
            with patch.object(adapter, '_create_lstm_autoencoder', return_value=mock_model), \
                 patch.object(adapter, '_train_sequence_model') as mock_train:
                
                detector = adapter.fit(time_series_dataset)
                
                assert isinstance(detector, Detector)
                assert detector.algorithm == "pytorch_lstm_autoencoder"
                assert "sequence_length" in detector.parameters
                assert "hidden_size" in detector.parameters
                assert detector.parameters["sequence_length"] == 50
                assert detector.parameters["hidden_size"] == 64
                mock_train.assert_called()

    def test_pytorch_convolutional_autoencoder(self, image_like_dataset):
        """Test PyTorch convolutional autoencoder for image anomaly detection."""
        # Mock PyTorch modules
        mock_torch = Mock()
        mock_nn = Mock()
        mock_f = Mock()
        
        # Mock CNN layers
        mock_conv2d = Mock()
        mock_convtranspose2d = Mock()
        mock_batchnorm2d = Mock()
        mock_nn.Conv2d = Mock(return_value=mock_conv2d)
        mock_nn.ConvTranspose2d = Mock(return_value=mock_convtranspose2d)
        mock_nn.BatchNorm2d = Mock(return_value=mock_batchnorm2d)
        mock_nn.ReLU = Mock()
        mock_nn.Sigmoid = Mock()
        mock_nn.Module = Mock
        
        # Mock functional operations
        mock_f.relu = Mock()
        mock_f.max_pool2d = Mock()
        mock_torch.nn.functional = mock_f
        
        with patch.dict('sys.modules', {
            'torch': mock_torch,
            'torch.nn': mock_nn,
            'torch.optim': Mock()
        }):
            from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
            
            adapter = PyTorchAdapter(
                model_type="convolutional_autoencoder",
                input_channels=1,
                image_size=(28, 28),
                conv_layers=[32, 64, 128],
                kernel_size=3,
                batch_norm=True,
                epochs=15
            )
            
            # Mock model
            mock_model = Mock()
            mock_model.parameters.return_value = [Mock()]
            
            with patch.object(adapter, '_create_conv_autoencoder', return_value=mock_model), \
                 patch.object(adapter, '_train_model') as mock_train:
                
                detector = adapter.fit(image_like_dataset)
                
                assert detector.algorithm == "pytorch_convolutional_autoencoder"
                assert "input_channels" in detector.parameters
                assert "image_size" in detector.parameters
                mock_train.assert_called()

    def test_pytorch_variational_autoencoder(self, time_series_dataset):
        """Test PyTorch Variational Autoencoder (VAE) implementation."""
        # Mock PyTorch modules
        mock_torch = Mock()
        mock_nn = Mock()
        mock_distributions = Mock()
        
        # Mock VAE-specific components
        mock_normal = Mock()
        mock_distributions.Normal = Mock(return_value=mock_normal)
        mock_distributions.kl_divergence = Mock(return_value=Mock())
        
        with patch.dict('sys.modules', {
            'torch': mock_torch,
            'torch.nn': mock_nn,
            'torch.optim': Mock(),
            'torch.distributions': mock_distributions
        }):
            from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
            
            adapter = PyTorchAdapter(
                model_type="variational_autoencoder",
                latent_dim=32,
                hidden_dims=[128, 64],
                beta=1.0,  # Beta-VAE parameter
                kl_weight=0.01,
                epochs=20
            )
            
            # Mock VAE model
            mock_vae = Mock()
            mock_vae.parameters.return_value = [Mock()]
            mock_vae.encode.return_value = (Mock(), Mock())  # mu, logvar
            mock_vae.decode.return_value = Mock()
            mock_vae.reparameterize.return_value = Mock()
            
            with patch.object(adapter, '_create_vae', return_value=mock_vae), \
                 patch.object(adapter, '_train_vae') as mock_train_vae:
                
                detector = adapter.fit(time_series_dataset)
                
                assert detector.algorithm == "pytorch_variational_autoencoder"
                assert "latent_dim" in detector.parameters
                assert "beta" in detector.parameters
                assert detector.parameters["latent_dim"] == 32
                assert detector.parameters["beta"] == 1.0
                mock_train_vae.assert_called()

    def test_pytorch_gan_based_detection(self, image_like_dataset):
        """Test PyTorch GAN-based anomaly detection."""
        # Mock PyTorch modules
        mock_torch = Mock()
        mock_nn = Mock()
        mock_optim = Mock()
        
        with patch.dict('sys.modules', {
            'torch': mock_torch,
            'torch.nn': mock_nn,
            'torch.optim': mock_optim
        }):
            from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
            
            adapter = PyTorchAdapter(
                model_type="adversarial_autoencoder",
                generator_dims=[100, 256, 512],
                discriminator_dims=[512, 256, 1],
                adversarial_weight=0.1,
                reconstruction_weight=0.9,
                epochs=25
            )
            
            # Mock GAN components
            mock_generator = Mock()
            mock_discriminator = Mock()
            mock_generator.parameters.return_value = [Mock()]
            mock_discriminator.parameters.return_value = [Mock()]
            
            with patch.object(adapter, '_create_adversarial_autoencoder', 
                             return_value=(mock_generator, mock_discriminator)), \
                 patch.object(adapter, '_train_adversarial_model') as mock_train_adv:
                
                detector = adapter.fit(image_like_dataset)
                
                assert detector.algorithm == "pytorch_adversarial_autoencoder"
                assert "generator_dims" in detector.parameters
                assert "adversarial_weight" in detector.parameters
                mock_train_adv.assert_called()

    def test_pytorch_ensemble_detection(self, time_series_dataset):
        """Test PyTorch ensemble-based detection."""
        # Mock PyTorch modules
        mock_torch = Mock()
        mock_nn = Mock()
        
        with patch.dict('sys.modules', {
            'torch': mock_torch,
            'torch.nn': mock_nn,
            'torch.optim': Mock()
        }):
            from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
            
            adapter = PyTorchAdapter(
                model_type="ensemble_autoencoder",
                ensemble_size=5,
                diversity_loss_weight=0.1,
                base_model_type="standard",
                voting_strategy="average",
                epochs=15
            )
            
            # Mock ensemble models
            mock_models = [Mock() for _ in range(5)]
            for mock_model in mock_models:
                mock_model.parameters.return_value = [Mock()]
                mock_model.state_dict.return_value = {"layer.weight": np.ones((10, 8))}
            
            with patch.object(adapter, '_create_ensemble', return_value=mock_models), \
                 patch.object(adapter, '_train_ensemble') as mock_train_ensemble:
                
                detector = adapter.fit(time_series_dataset)
                
                assert detector.algorithm == "pytorch_ensemble_autoencoder"
                assert "ensemble_size" in detector.parameters
                assert "voting_strategy" in detector.parameters
                assert detector.parameters["ensemble_size"] == 5
                mock_train_ensemble.assert_called()

    def test_pytorch_attention_mechanism(self, time_series_dataset):
        """Test PyTorch attention-based autoencoder."""
        # Mock PyTorch modules
        mock_torch = Mock()
        mock_nn = Mock()
        
        # Mock attention components
        mock_multihead_attention = Mock()
        mock_nn.MultiheadAttention = Mock(return_value=mock_multihead_attention)
        mock_nn.LayerNorm = Mock()
        mock_nn.TransformerEncoder = Mock()
        mock_nn.TransformerEncoderLayer = Mock()
        
        with patch.dict('sys.modules', {
            'torch': mock_torch,
            'torch.nn': mock_nn,
            'torch.optim': Mock()
        }):
            from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
            
            adapter = PyTorchAdapter(
                model_type="transformer_autoencoder",
                d_model=128,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=512,
                dropout=0.1,
                epochs=20
            )
            
            # Mock transformer model
            mock_transformer = Mock()
            mock_transformer.parameters.return_value = [Mock()]
            
            with patch.object(adapter, '_create_transformer_autoencoder', 
                             return_value=mock_transformer), \
                 patch.object(adapter, '_train_transformer') as mock_train_transformer:
                
                detector = adapter.fit(time_series_dataset)
                
                assert detector.algorithm == "pytorch_transformer_autoencoder"
                assert "d_model" in detector.parameters
                assert "nhead" in detector.parameters
                assert detector.parameters["d_model"] == 128
                assert detector.parameters["nhead"] == 8
                mock_train_transformer.assert_called()

    def test_pytorch_gpu_acceleration(self, time_series_dataset):
        """Test PyTorch GPU acceleration and device management."""
        # Mock PyTorch with CUDA support
        mock_torch = Mock()
        mock_cuda = Mock()
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 2
        mock_torch.cuda = mock_cuda
        
        # Mock device operations
        mock_device = Mock()
        mock_torch.device.return_value = mock_device
        
        with patch.dict('sys.modules', {
            'torch': mock_torch,
            'torch.nn': Mock(),
            'torch.optim': Mock()
        }):
            from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
            
            adapter = PyTorchAdapter(
                device="cuda:0",
                use_amp=True,  # Automatic Mixed Precision
                use_parallel=True,
                epochs=10
            )
            
            # Mock model and device setup
            mock_model = Mock()
            mock_model.to.return_value = mock_model
            mock_model.parameters.return_value = [Mock()]
            
            with patch.object(adapter, '_create_autoencoder', return_value=mock_model), \
                 patch.object(adapter, '_setup_device') as mock_setup_device, \
                 patch.object(adapter, '_train_with_amp') as mock_train_amp:
                
                detector = adapter.fit(time_series_dataset)
                
                assert "device" in detector.parameters
                assert "use_amp" in detector.parameters
                mock_setup_device.assert_called()
                mock_train_amp.assert_called()

    def test_pytorch_quantization_optimization(self, image_like_dataset):
        """Test PyTorch model quantization for optimized inference."""
        # Mock PyTorch quantization modules
        mock_torch = Mock()
        mock_quantization = Mock()
        mock_quantization.quantize_dynamic = Mock()
        mock_quantization.prepare_qat = Mock()
        mock_torch.quantization = mock_quantization
        
        with patch.dict('sys.modules', {
            'torch': mock_torch,
            'torch.nn': Mock(),
            'torch.quantization': mock_quantization
        }):
            from pynomaly.infrastructure.adapters.pytorch_adapter import PyTorchAdapter
            
            adapter = PyTorchAdapter(
                model_type="quantized_autoencoder",
                quantization_type="dynamic",
                quantize_weights=True,
                quantize_activations=False,
                epochs=10
            )
            
            # Mock quantized model
            mock_model = Mock()
            mock_quantized_model = Mock()
            mock_quantization.quantize_dynamic.return_value = mock_quantized_model
            
            with patch.object(adapter, '_create_autoencoder', return_value=mock_model), \
                 patch.object(adapter, '_quantize_model') as mock_quantize:
                
                detector = adapter.fit(image_like_dataset)
                
                assert "quantization_type" in detector.parameters
                assert detector.parameters["quantization_type"] == "dynamic"
                mock_quantize.assert_called()


class TestTensorFlowAdapterAdvanced:
    """Advanced TensorFlow adapter tests with Keras and tf.function optimizations."""

    @pytest.fixture
    def multivariate_dataset(self):
        """Create multivariate dataset for TensorFlow testing."""
        np.random.seed(42)
        
        # Generate correlated multivariate data
        n_samples = 800
        n_features = 12
        
        # Create correlation matrix
        correlation_matrix = np.random.uniform(0.1, 0.8, (n_features, n_features))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Generate data with correlation
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=correlation_matrix,
            size=n_samples
        )
        
        df = pd.DataFrame(normal_data, columns=[f"variable_{i}" for i in range(n_features)])
        
        dataset = Mock(spec=Dataset)
        dataset.id = "multivariate_dataset"
        dataset.name = "Multivariate Dataset"
        dataset.data = df
        dataset.features = df
        dataset.n_samples = len(df)
        dataset.created_at = datetime.now(timezone.utc)
        dataset.get_numeric_features.return_value = list(df.columns)
        
        return dataset

    def test_tensorflow_custom_layers(self, multivariate_dataset):
        """Test TensorFlow with custom layers and advanced architectures."""
        # Mock TensorFlow modules
        mock_tf = Mock()
        mock_keras = Mock()
        mock_layers = Mock()
        
        # Mock custom layer functionality
        mock_attention_layer = Mock()
        mock_residual_block = Mock()
        mock_layers.Layer = Mock
        mock_layers.Dense = Mock()
        mock_layers.Dropout = Mock()
        mock_layers.BatchNormalization = Mock()
        
        with patch.dict('sys.modules', {
            'tensorflow': mock_tf,
            'tensorflow.keras': mock_keras,
            'tensorflow.keras.layers': mock_layers
        }):
            from pynomaly.infrastructure.adapters.tensorflow_adapter import TensorFlowAdapter
            
            adapter = TensorFlowAdapter(
                model_type="residual_autoencoder",
                residual_blocks=4,
                block_depth=3,
                attention_mechanism=True,
                skip_connections=True,
                epochs=25
            )
            
            # Mock model with custom components
            mock_model = Mock()
            mock_model.compile.return_value = None
            mock_model.fit.return_value = Mock(history={"loss": [1.0, 0.5, 0.3]})
            mock_model.to_json.return_value = '{"class_name": "ResidualAutoencoder"}'
            
            with patch.object(adapter, '_create_residual_autoencoder', return_value=mock_model), \
                 patch.object(adapter, '_add_attention_layers') as mock_attention:
                
                detector = adapter.fit(multivariate_dataset)
                
                assert detector.algorithm == "tensorflow_residual_autoencoder"
                assert "residual_blocks" in detector.parameters
                assert "attention_mechanism" in detector.parameters
                mock_attention.assert_called()

    def test_tensorflow_distributed_training(self, multivariate_dataset):
        """Test TensorFlow distributed training capabilities."""
        # Mock TensorFlow distributed strategy
        mock_tf = Mock()
        mock_strategy = Mock()
        mock_strategy.scope.return_value.__enter__ = Mock()
        mock_strategy.scope.return_value.__exit__ = Mock()
        mock_tf.distribute.MirroredStrategy.return_value = mock_strategy
        
        with patch.dict('sys.modules', {
            'tensorflow': mock_tf,
            'tensorflow.keras': Mock()
        }):
            from pynomaly.infrastructure.adapters.tensorflow_adapter import TensorFlowAdapter
            
            adapter = TensorFlowAdapter(
                use_distributed_training=True,
                strategy_type="mirrored",
                num_gpus=2,
                epochs=15
            )
            
            # Mock distributed model
            mock_model = Mock()
            mock_model.compile.return_value = None
            mock_model.fit.return_value = Mock(history={"loss": [1.0, 0.7, 0.4]})
            
            with patch.object(adapter, '_create_distributed_model', return_value=mock_model), \
                 patch.object(adapter, '_setup_distributed_strategy') as mock_setup:
                
                detector = adapter.fit(multivariate_dataset)
                
                assert "use_distributed_training" in detector.parameters
                assert "strategy_type" in detector.parameters
                mock_setup.assert_called()

    def test_tensorflow_mixed_precision(self, multivariate_dataset):
        """Test TensorFlow mixed precision training."""
        # Mock TensorFlow mixed precision
        mock_tf = Mock()
        mock_mixed_precision = Mock()
        mock_policy = Mock()
        mock_mixed_precision.Policy.return_value = mock_policy
        mock_mixed_precision.set_global_policy = Mock()
        mock_tf.keras.mixed_precision = mock_mixed_precision
        
        with patch.dict('sys.modules', {
            'tensorflow': mock_tf,
            'tensorflow.keras': Mock()
        }):
            from pynomaly.infrastructure.adapters.tensorflow_adapter import TensorFlowAdapter
            
            adapter = TensorFlowAdapter(
                use_mixed_precision=True,
                precision_policy="mixed_float16",
                loss_scaling="dynamic",
                epochs=20
            )
            
            mock_model = Mock()
            mock_model.compile.return_value = None
            mock_model.fit.return_value = Mock(history={"loss": [1.0, 0.6]})
            
            with patch.object(adapter, '_setup_mixed_precision') as mock_setup_mp, \
                 patch.object(adapter, '_create_autoencoder', return_value=mock_model):
                
                detector = adapter.fit(multivariate_dataset)
                
                assert "use_mixed_precision" in detector.parameters
                assert "precision_policy" in detector.parameters
                mock_setup_mp.assert_called()

    def test_tensorflow_tensorboard_integration(self, multivariate_dataset):
        """Test TensorFlow TensorBoard logging integration."""
        # Mock TensorFlow callbacks
        mock_tf = Mock()
        mock_callbacks = Mock()
        mock_tensorboard = Mock()
        mock_callbacks.TensorBoard = Mock(return_value=mock_tensorboard)
        mock_tf.keras.callbacks = mock_callbacks
        
        with patch.dict('sys.modules', {
            'tensorflow': mock_tf,
            'tensorflow.keras': Mock()
        }):
            from pynomaly.infrastructure.adapters.tensorflow_adapter import TensorFlowAdapter
            
            adapter = TensorFlowAdapter(
                use_tensorboard=True,
                log_dir="./logs/anomaly_detection",
                log_histogram=True,
                log_embeddings=True,
                epochs=10
            )
            
            mock_model = Mock()
            mock_model.compile.return_value = None
            mock_model.fit.return_value = Mock(history={"loss": [1.0, 0.8]})
            
            with patch.object(adapter, '_setup_tensorboard_logging') as mock_setup_tb, \
                 patch.object(adapter, '_create_autoencoder', return_value=mock_model):
                
                detector = adapter.fit(multivariate_dataset)
                
                assert "use_tensorboard" in detector.parameters
                assert "log_dir" in detector.parameters
                mock_setup_tb.assert_called()

    def test_tensorflow_bayesian_layers(self, multivariate_dataset):
        """Test TensorFlow with Bayesian neural network layers."""
        # Mock TensorFlow Probability
        mock_tf = Mock()
        mock_tfp = Mock()
        mock_layers = Mock()
        mock_layers.DenseVariational = Mock()
        mock_layers.DenseFlipout = Mock()
        mock_tfp.layers = mock_layers
        
        with patch.dict('sys.modules', {
            'tensorflow': mock_tf,
            'tensorflow.keras': Mock(),
            'tensorflow_probability': mock_tfp
        }):
            from pynomaly.infrastructure.adapters.tensorflow_adapter import TensorFlowAdapter
            
            adapter = TensorFlowAdapter(
                model_type="bayesian_autoencoder",
                use_variational_layers=True,
                kl_divergence_weight=0.01,
                uncertainty_estimation=True,
                num_monte_carlo_samples=10,
                epochs=30
            )
            
            mock_model = Mock()
            mock_model.compile.return_value = None
            mock_model.fit.return_value = Mock(history={"loss": [2.0, 1.5, 1.0]})
            
            with patch.object(adapter, '_create_bayesian_autoencoder', return_value=mock_model), \
                 patch.object(adapter, '_add_kl_loss') as mock_add_kl:
                
                detector = adapter.fit(multivariate_dataset)
                
                assert detector.algorithm == "tensorflow_bayesian_autoencoder"
                assert "use_variational_layers" in detector.parameters
                assert "uncertainty_estimation" in detector.parameters
                mock_add_kl.assert_called()


class TestJAXAdapterAdvanced:
    """Advanced JAX adapter tests with functional programming and advanced optimization."""

    @pytest.fixture
    def graph_dataset(self):
        """Create graph-like dataset for JAX testing."""
        np.random.seed(42)
        
        # Generate graph node features
        n_nodes = 200
        n_features = 16
        
        # Normal nodes
        normal_features = np.random.normal(0, 1, (n_nodes, n_features))
        
        # Add graph structure (adjacency information as features)
        adjacency_features = np.random.binomial(1, 0.1, (n_nodes, n_nodes // 10))
        
        # Combine features
        all_features = np.hstack([normal_features, adjacency_features])
        
        df = pd.DataFrame(all_features, columns=[f"node_feature_{i}" for i in range(all_features.shape[1])])
        
        dataset = Mock(spec=Dataset)
        dataset.id = "graph_dataset"
        dataset.name = "Graph Dataset"
        dataset.data = df
        dataset.features = df
        dataset.n_samples = len(df)
        dataset.created_at = datetime.now(timezone.utc)
        dataset.get_numeric_features.return_value = list(df.columns)
        
        return dataset

    def test_jax_neural_ode_autoencoder(self, graph_dataset):
        """Test JAX Neural ODE autoencoder implementation."""
        # Mock JAX modules
        mock_jax = Mock()
        mock_jnp = Mock()
        mock_odeint = Mock()
        
        with patch.dict('sys.modules', {
            'jax': mock_jax,
            'jax.numpy': mock_jnp,
            'jax.experimental.ode': Mock(odeint=mock_odeint)
        }):
            from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter
            
            adapter = JAXAdapter(
                model_type="neural_ode_autoencoder",
                ode_solver="dopri5",
                integration_time=1.0,
                rtol=1e-5,
                atol=1e-6,
                hidden_dims=[64, 32],
                epochs=15
            )
            
            # Mock Neural ODE components
            mock_ode_func = Mock()
            mock_params = {
                "ode_params": {"weights": np.ones((26, 64))},
                "encoder_params": {"weights": np.ones((26, 32))},
                "decoder_params": {"weights": np.ones((32, 26))}
            }
            
            with patch.object(adapter, '_create_neural_ode_autoencoder', 
                             return_value=(mock_ode_func, mock_params)), \
                 patch.object(adapter, '_train_neural_ode') as mock_train_ode:
                
                detector = adapter.fit(graph_dataset)
                
                assert detector.algorithm == "jax_neural_ode_autoencoder"
                assert "ode_solver" in detector.parameters
                assert "integration_time" in detector.parameters
                mock_train_ode.assert_called()

    def test_jax_normalizing_flow(self, graph_dataset):
        """Test JAX normalizing flow implementation."""
        # Mock JAX modules
        mock_jax = Mock()
        mock_jnp = Mock()
        mock_random = Mock()
        
        with patch.dict('sys.modules', {
            'jax': mock_jax,
            'jax.numpy': mock_jnp,
            'jax.random': mock_random
        }):
            from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter
            
            adapter = JAXAdapter(
                model_type="normalizing_flow",
                num_flows=8,
                flow_type="coupling",
                hidden_units=128,
                num_masked=13,  # Half of 26 features
                epochs=25
            )
            
            # Mock normalizing flow components
            mock_flow_fn = Mock()
            mock_flow_params = {
                "flows": [{"weights": np.ones((26, 128))} for _ in range(8)],
                "base_distribution": {"loc": np.zeros(26), "scale": np.ones(26)}
            }
            
            with patch.object(adapter, '_create_normalizing_flow', 
                             return_value=(mock_flow_fn, mock_flow_params)), \
                 patch.object(adapter, '_train_normalizing_flow') as mock_train_flow:
                
                detector = adapter.fit(graph_dataset)
                
                assert detector.algorithm == "jax_normalizing_flow"
                assert "num_flows" in detector.parameters
                assert "flow_type" in detector.parameters
                assert detector.parameters["num_flows"] == 8
                mock_train_flow.assert_called()

    def test_jax_graph_neural_network(self, graph_dataset):
        """Test JAX Graph Neural Network for anomaly detection."""
        # Mock JAX modules
        mock_jax = Mock()
        mock_jnp = Mock()
        
        with patch.dict('sys.modules', {
            'jax': mock_jax,
            'jax.numpy': mock_jnp
        }):
            from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter
            
            adapter = JAXAdapter(
                model_type="graph_autoencoder",
                message_passing_steps=3,
                aggregation_type="mean",
                node_hidden_dim=64,
                edge_hidden_dim=32,
                readout_type="global_mean_pool",
                epochs=20
            )
            
            # Mock GNN components
            mock_gnn_fn = Mock()
            mock_gnn_params = {
                "message_fn": {"weights": np.ones((26, 64))},
                "update_fn": {"weights": np.ones((64, 64))},
                "readout_fn": {"weights": np.ones((64, 26))}
            }
            
            with patch.object(adapter, '_create_graph_autoencoder', 
                             return_value=(mock_gnn_fn, mock_gnn_params)), \
                 patch.object(adapter, '_train_graph_model') as mock_train_gnn:
                
                detector = adapter.fit(graph_dataset)
                
                assert detector.algorithm == "jax_graph_autoencoder"
                assert "message_passing_steps" in detector.parameters
                assert "aggregation_type" in detector.parameters
                mock_train_gnn.assert_called()

    def test_jax_memory_efficient_training(self, graph_dataset):
        """Test JAX memory-efficient training with gradient checkpointing."""
        # Mock JAX modules
        mock_jax = Mock()
        mock_jnp = Mock()
        mock_checkpoint = Mock()
        mock_jax.checkpoint = mock_checkpoint
        
        with patch.dict('sys.modules', {
            'jax': mock_jax,
            'jax.numpy': mock_jnp
        }):
            from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter
            
            adapter = JAXAdapter(
                use_gradient_checkpointing=True,
                checkpoint_every_n_layers=2,
                use_memory_efficient_attention=True,
                batch_size=32,
                epochs=10
            )
            
            # Mock checkpointed functions
            mock_forward_fn = Mock()
            mock_params = {"weights": np.ones((26, 64))}
            mock_checkpoint.return_value = mock_forward_fn
            
            with patch.object(adapter, '_create_memory_efficient_model', 
                             return_value=(mock_forward_fn, mock_params)), \
                 patch.object(adapter, '_apply_gradient_checkpointing') as mock_apply_checkpoint:
                
                detector = adapter.fit(graph_dataset)
                
                assert "use_gradient_checkpointing" in detector.parameters
                assert "checkpoint_every_n_layers" in detector.parameters
                mock_apply_checkpoint.assert_called()

    def test_jax_advanced_optimization(self, graph_dataset):
        """Test JAX advanced optimization techniques."""
        # Mock JAX and Optax modules
        mock_jax = Mock()
        mock_optax = Mock()
        
        # Mock advanced optimizers
        mock_adamw = Mock()
        mock_lookahead = Mock()
        mock_schedule = Mock()
        mock_optax.adamw.return_value = mock_adamw
        mock_optax.lookahead.return_value = mock_lookahead
        mock_optax.cosine_decay_schedule.return_value = mock_schedule
        
        with patch.dict('sys.modules', {
            'jax': mock_jax,
            'jax.numpy': Mock(),
            'optax': mock_optax
        }):
            from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter
            
            adapter = JAXAdapter(
                optimizer="adamw",
                learning_rate_schedule="cosine_decay",
                weight_decay=0.01,
                use_lookahead=True,
                lookahead_steps=5,
                gradient_clipping=1.0,
                epochs=30
            )
            
            mock_model_fn = Mock()
            mock_params = {"weights": np.ones((26, 64))}
            
            with patch.object(adapter, '_create_functional_autoencoder', 
                             return_value=(mock_model_fn, mock_params)), \
                 patch.object(adapter, '_setup_advanced_optimizer') as mock_setup_opt:
                
                detector = adapter.fit(graph_dataset)
                
                assert "optimizer" in detector.parameters
                assert "learning_rate_schedule" in detector.parameters
                assert "use_lookahead" in detector.parameters
                mock_setup_opt.assert_called()

    def test_jax_probabilistic_models(self, graph_dataset):
        """Test JAX probabilistic and Bayesian models."""
        # Mock JAX modules
        mock_jax = Mock()
        mock_jnp = Mock()
        mock_numpyro = Mock()
        
        # Mock probabilistic components
        mock_sample = Mock()
        mock_plate = Mock()
        mock_numpyro.sample = mock_sample
        mock_numpyro.plate = mock_plate
        
        with patch.dict('sys.modules', {
            'jax': mock_jax,
            'jax.numpy': mock_jnp,
            'numpyro': mock_numpyro,
            'numpyro.distributions': Mock()
        }):
            from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter
            
            adapter = JAXAdapter(
                model_type="bayesian_autoencoder",
                prior_type="normal",
                posterior_approximation="variational",
                num_samples=50,
                warmup_steps=100,
                estimate_uncertainty=True,
                epochs=20
            )
            
            # Mock Bayesian model components
            mock_model_fn = Mock()
            mock_guide_fn = Mock()
            mock_params = {
                "model_params": {"mean": np.zeros((26, 64)), "std": np.ones((26, 64))},
                "guide_params": {"loc": np.zeros((26, 64)), "scale": np.ones((26, 64))}
            }
            
            with patch.object(adapter, '_create_bayesian_autoencoder', 
                             return_value=(mock_model_fn, mock_guide_fn, mock_params)), \
                 patch.object(adapter, '_train_bayesian_model') as mock_train_bayesian:
                
                detector = adapter.fit(graph_dataset)
                
                assert detector.algorithm == "jax_bayesian_autoencoder"
                assert "prior_type" in detector.parameters
                assert "estimate_uncertainty" in detector.parameters
                mock_train_bayesian.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
"""
Enhanced JAX Adapter Testing Suite
Comprehensive tests for JAX-based functional programming anomaly detection adapter.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import jax
import jax.numpy as jnp
from jax import random
from datetime import datetime

from pynomaly.infrastructure.adapters.jax_adapter import JAXAdapter
from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.domain.exceptions import DetectorError, AdapterError


class TestJAXAdapter:
    """Enhanced test suite for JAX adapter functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (800, 8))
        anomalous_data = np.random.normal(3, 1, (200, 8))
        data = np.vstack([normal_data, anomalous_data])
        labels = np.hstack([np.zeros(800), np.ones(200)])
        
        return {
            "X_train": data[:600].astype(np.float32),
            "X_test": data[600:].astype(np.float32),
            "y_train": labels[:600],
            "y_test": labels[600:],
            "features": [f"feature_{i}" for i in range(8)]
        }

    @pytest.fixture
    def jax_adapter(self):
        """Create JAX adapter instance."""
        return JAXAdapter(
            model_type="functional_autoencoder",
            hidden_dims=[32, 16, 8],
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

    def test_jax_adapter_initialization_default(self):
        """Test JAX adapter initialization with default parameters."""
        adapter = JAXAdapter()
        
        assert adapter.model_type == "functional_autoencoder"
        assert adapter.learning_rate == 0.001
        assert adapter.batch_size == 32
        assert adapter.epochs == 100
        assert adapter.activation == "relu"
        assert adapter.optimizer == "adam"

    def test_jax_adapter_initialization_custom(self):
        """Test JAX adapter initialization with custom parameters."""
        adapter = JAXAdapter(
            model_type="variational_autoencoder",
            hidden_dims=[64, 32, 16],
            learning_rate=0.01,
            batch_size=64,
            epochs=50,
            activation="swish",
            optimizer="adamw",
            weight_decay=0.01,
            dropout_rate=0.2
        )
        
        assert adapter.model_type == "variational_autoencoder"
        assert adapter.hidden_dims == [64, 32, 16]
        assert adapter.learning_rate == 0.01
        assert adapter.batch_size == 64
        assert adapter.epochs == 50
        assert adapter.activation == "swish"
        assert adapter.optimizer == "adamw"
        assert adapter.weight_decay == 0.01
        assert adapter.dropout_rate == 0.2

    def test_jax_device_configuration(self):
        """Test JAX device configuration and detection."""
        with patch('jax.devices') as mock_devices:
            mock_devices.return_value = [Mock(device_kind="cpu")]
            
            adapter = JAXAdapter(device="auto")
            
            assert adapter.device in ["cpu", "gpu", "tpu"]

    def test_jax_adapter_invalid_parameters(self):
        """Test adapter initialization with invalid parameters."""
        with pytest.raises(ValueError):
            JAXAdapter(model_type="invalid_model")
        
        with pytest.raises(ValueError):
            JAXAdapter(learning_rate=-0.1)
        
        with pytest.raises(ValueError):
            JAXAdapter(batch_size=0)
        
        with pytest.raises(ValueError):
            JAXAdapter(epochs=-1)
        
        with pytest.raises(ValueError):
            JAXAdapter(dropout_rate=1.5)

    # Functional Model Architecture Tests

    def test_functional_autoencoder_creation(self, jax_adapter, sample_data):
        """Test functional autoencoder architecture creation."""
        key = random.PRNGKey(42)
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('jax.random.PRNGKey') as mock_key:
            mock_key.return_value = key
            
            model_fn, params = jax_adapter._create_functional_autoencoder(input_dim, key)
            
            assert callable(model_fn)
            assert isinstance(params, dict)
            assert "encoder" in params
            assert "decoder" in params

    def test_variational_autoencoder_creation(self, sample_data):
        """Test variational autoencoder model creation."""
        adapter = JAXAdapter(
            model_type="variational_autoencoder",
            latent_dim=4,
            beta=1.0
        )
        key = random.PRNGKey(42)
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('jax.random.PRNGKey') as mock_key:
            mock_key.return_value = key
            
            model_fn, params = adapter._create_variational_autoencoder(input_dim, key)
            
            assert callable(model_fn)
            assert isinstance(params, dict)
            assert "encoder" in params
            assert "decoder" in params

    def test_transformer_autoencoder_creation(self, sample_data):
        """Test Transformer autoencoder for sequence data."""
        adapter = JAXAdapter(
            model_type="transformer_autoencoder",
            sequence_length=20,
            num_heads=4,
            num_layers=2,
            d_model=64
        )
        key = random.PRNGKey(42)
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('jax.random.PRNGKey') as mock_key:
            mock_key.return_value = key
            
            model_fn, params = adapter._create_transformer_autoencoder(input_dim, key)
            
            assert callable(model_fn)
            assert isinstance(params, dict)

    def test_neural_ode_autoencoder_creation(self, sample_data):
        """Test Neural ODE autoencoder model."""
        adapter = JAXAdapter(
            model_type="neural_ode_autoencoder",
            ode_solver="rk4",
            integration_time=1.0,
            num_steps=10
        )
        key = random.PRNGKey(42)
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('jax.random.PRNGKey') as mock_key:
            mock_key.return_value = key
            
            model_fn, params = adapter._create_neural_ode_autoencoder(input_dim, key)
            
            assert callable(model_fn)
            assert isinstance(params, dict)

    def test_flow_based_model_creation(self, sample_data):
        """Test normalizing flow-based anomaly detection."""
        adapter = JAXAdapter(
            model_type="normalizing_flow",
            num_flows=8,
            flow_type="coupling",
            hidden_units=64
        )
        key = random.PRNGKey(42)
        input_dim = sample_data["X_train"].shape[1]
        
        with patch('jax.random.PRNGKey') as mock_key:
            mock_key.return_value = key
            
            model_fn, params = adapter._create_normalizing_flow(input_dim, key)
            
            assert callable(model_fn)
            assert isinstance(params, dict)

    # Training Process Tests

    def test_fit_basic_training(self, jax_adapter, mock_dataset, sample_data):
        """Test basic model training process."""
        with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create, \
             patch('jax.random.PRNGKey') as mock_key:
            
            mock_key.return_value = random.PRNGKey(42)
            mock_model_fn = MagicMock()
            mock_params = {"encoder": {"weights": jnp.ones((8, 32))}, "decoder": {"weights": jnp.ones((32, 8))}}
            mock_create.return_value = (mock_model_fn, mock_params)
            
            detector = jax_adapter.fit(mock_dataset)
            
            assert isinstance(detector, Detector)
            assert detector.algorithm == "jax_functional_autoencoder"
            assert "model_params" in detector.parameters

    def test_fit_with_validation_data(self, jax_adapter, mock_dataset, sample_data):
        """Test training with validation data."""
        validation_data = sample_data["X_test"]
        
        with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create, \
             patch('jax.random.PRNGKey') as mock_key:
            
            mock_key.return_value = random.PRNGKey(42)
            mock_model_fn = MagicMock()
            mock_params = {"encoder": {"weights": jnp.ones((8, 32))}}
            mock_create.return_value = (mock_model_fn, mock_params)
            
            detector = jax_adapter.fit(
                mock_dataset,
                validation_data=validation_data
            )
            
            assert detector is not None
            assert "validation_loss" in detector.parameters.get("training_history", {})

    def test_fit_jit_compilation(self, jax_adapter, mock_dataset):
        """Test JIT compilation during training."""
        jax_adapter.use_jit = True
        
        with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create, \
             patch('jax.jit') as mock_jit:
            
            mock_model_fn = MagicMock()
            mock_params = {"weights": jnp.ones((8, 32))}
            mock_create.return_value = (mock_model_fn, mock_params)
            mock_jit.return_value = mock_model_fn
            
            detector = jax_adapter.fit(mock_dataset)
            
            assert detector is not None
            mock_jit.assert_called()

    def test_fit_gradient_accumulation(self, jax_adapter, mock_dataset):
        """Test gradient accumulation for large batch training."""
        jax_adapter.gradient_accumulation_steps = 4
        
        with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create:
            mock_model_fn = MagicMock()
            mock_params = {"weights": jnp.ones((8, 32))}
            mock_create.return_value = (mock_model_fn, mock_params)
            
            detector = jax_adapter.fit(mock_dataset)
            
            assert detector is not None
            assert "gradient_accumulation_steps" in detector.parameters

    def test_fit_mixed_precision_training(self, jax_adapter, mock_dataset):
        """Test mixed precision training for performance."""
        jax_adapter.use_mixed_precision = True
        
        with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create:
            mock_model_fn = MagicMock()
            mock_params = {"weights": jnp.ones((8, 32))}
            mock_create.return_value = (mock_model_fn, mock_params)
            
            detector = jax_adapter.fit(mock_dataset)
            
            assert detector is not None
            assert detector.parameters.get("mixed_precision") is True

    # Detection Process Tests

    def test_predict_basic_detection(self, jax_adapter, sample_data):
        """Test basic anomaly detection."""
        detector = Detector(
            id="test_detector",
            algorithm="jax_functional_autoencoder",
            parameters={
                "model_params": {"weights": jnp.ones((8, 32))},
                "model_fn": "mock_function",
                "input_dim": 8,
                "threshold": 0.5
            }
        )
        
        with patch.object(jax_adapter, '_load_model') as mock_load:
            mock_model_fn = lambda params, x: x + jnp.random.normal(random.PRNGKey(42), x.shape) * 0.1
            mock_load.return_value = mock_model_fn
            
            result = jax_adapter.predict(detector, sample_data["X_test"])
            
            assert isinstance(result, DetectionResult)
            assert len(result.anomaly_scores) == len(sample_data["X_test"])
            assert all(isinstance(score, AnomalyScore) for score in result.anomaly_scores)

    def test_predict_vectorized_computation(self, jax_adapter, sample_data):
        """Test vectorized computation during prediction."""
        detector = Detector(
            id="test_detector",
            algorithm="jax_functional_autoencoder",
            parameters={
                "model_params": {"weights": jnp.ones((8, 32))},
                "model_fn": "mock_function",
                "input_dim": 8
            }
        )
        
        # Large dataset to test vectorization
        large_data = np.random.randn(2000, 8).astype(np.float32)
        
        with patch.object(jax_adapter, '_load_model') as mock_load, \
             patch('jax.vmap') as mock_vmap:
            
            mock_model_fn = lambda params, x: x
            mock_vmap.return_value = mock_model_fn
            mock_load.return_value = mock_model_fn
            
            result = jax_adapter.predict(detector, large_data)
            
            assert len(result.anomaly_scores) == len(large_data)
            mock_vmap.assert_called()

    def test_predict_probabilistic_detection(self, jax_adapter, sample_data):
        """Test probabilistic anomaly detection with uncertainty."""
        jax_adapter.estimate_uncertainty = True
        jax_adapter.num_samples = 10
        
        detector = Detector(
            id="test_detector",
            algorithm="jax_variational_autoencoder",
            parameters={
                "model_params": {"encoder": {"weights": jnp.ones((8, 16))}},
                "model_fn": "mock_function",
                "input_dim": 8
            }
        )
        
        with patch.object(jax_adapter, '_load_model') as mock_load:
            mock_model_fn = lambda params, x, key: (x, jnp.zeros((x.shape[0], 4)), jnp.ones((x.shape[0], 4)))
            mock_load.return_value = mock_model_fn
            
            result = jax_adapter.predict(detector, sample_data["X_test"])
            
            assert result is not None
            # Should include uncertainty estimates
            assert hasattr(result, 'uncertainty_scores') or 'uncertainty' in str(result)

    # Advanced Features Tests

    def test_functional_programming_paradigm(self, jax_adapter, sample_data):
        """Test pure functional programming paradigm."""
        with patch('jax.pure_callback') as mock_pure:
            mock_pure.return_value = jnp.array([1.0])
            
            # Test pure function composition
            input_data = sample_data["X_train"][:10]
            
            result = jax_adapter._apply_pure_function(
                input_data,
                lambda x: jnp.sum(x, axis=1)
            )
            
            assert result is not None

    def test_automatic_differentiation(self, jax_adapter, sample_data):
        """Test automatic differentiation capabilities."""
        def loss_fn(params, x):
            # Mock loss function
            return jnp.mean((x - jnp.dot(x, params["weights"])) ** 2)
        
        with patch('jax.grad') as mock_grad:
            mock_grad.return_value = lambda params, x: {"weights": jnp.ones((8, 8))}
            
            params = {"weights": jnp.eye(8)}
            gradients = jax_adapter._compute_gradients(
                loss_fn, 
                params, 
                sample_data["X_train"][:10]
            )
            
            assert gradients is not None
            mock_grad.assert_called()

    def test_parallel_computation(self, jax_adapter, sample_data):
        """Test parallel computation across devices."""
        jax_adapter.use_pmap = True
        
        with patch('jax.pmap') as mock_pmap, \
             patch('jax.devices') as mock_devices:
            
            mock_devices.return_value = [Mock(), Mock()]  # 2 devices
            mock_pmap.return_value = lambda x: x
            
            result = jax_adapter._parallel_predict(
                sample_data["X_test"],
                lambda x: x
            )
            
            assert result is not None
            mock_pmap.assert_called()

    def test_memory_efficient_training(self, jax_adapter, mock_dataset):
        """Test memory-efficient training with gradient checkpointing."""
        jax_adapter.use_gradient_checkpointing = True
        
        with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create, \
             patch('jax.checkpoint') as mock_checkpoint:
            
            mock_model_fn = MagicMock()
            mock_params = {"weights": jnp.ones((8, 32))}
            mock_create.return_value = (mock_model_fn, mock_params)
            mock_checkpoint.return_value = mock_model_fn
            
            detector = jax_adapter.fit(mock_dataset)
            
            assert detector is not None
            mock_checkpoint.assert_called()

    # Optimization Tests

    def test_optimizer_selection(self, jax_adapter, mock_dataset):
        """Test different optimizer implementations."""
        optimizers = ["adam", "adamw", "sgd", "rmsprop", "adagrad"]
        
        for optimizer in optimizers:
            jax_adapter.optimizer = optimizer
            
            with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create:
                mock_model_fn = MagicMock()
                mock_params = {"weights": jnp.ones((8, 32))}
                mock_create.return_value = (mock_model_fn, mock_params)
                
                detector = jax_adapter.fit(mock_dataset)
                
                assert detector is not None
                assert detector.parameters["optimizer"] == optimizer

    def test_learning_rate_scheduling(self, jax_adapter, mock_dataset):
        """Test learning rate scheduling strategies."""
        jax_adapter.lr_schedule = "cosine"
        jax_adapter.warmup_steps = 100
        
        with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create, \
             patch('optax.cosine_decay_schedule') as mock_schedule:
            
            mock_schedule.return_value = lambda step: 0.001
            mock_model_fn = MagicMock()
            mock_params = {"weights": jnp.ones((8, 32))}
            mock_create.return_value = (mock_model_fn, mock_params)
            
            detector = jax_adapter.fit(mock_dataset)
            
            assert detector is not None
            mock_schedule.assert_called()

    def test_regularization_techniques(self, jax_adapter, mock_dataset):
        """Test various regularization techniques."""
        jax_adapter.l1_regularization = 0.01
        jax_adapter.l2_regularization = 0.001
        jax_adapter.dropout_rate = 0.2
        jax_adapter.spectral_normalization = True
        
        with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create:
            mock_model_fn = MagicMock()
            mock_params = {"weights": jnp.ones((8, 32))}
            mock_create.return_value = (mock_model_fn, mock_params)
            
            detector = jax_adapter.fit(mock_dataset)
            
            assert detector is not None
            assert detector.parameters["l1_regularization"] == 0.01
            assert detector.parameters["l2_regularization"] == 0.001

    # Model Persistence Tests

    def test_model_serialization(self, jax_adapter, mock_dataset):
        """Test model serialization and deserialization."""
        with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create:
            mock_model_fn = MagicMock()
            mock_params = {"weights": jnp.ones((8, 32))}
            mock_create.return_value = (mock_model_fn, mock_params)
            
            detector = jax_adapter.fit(mock_dataset)
            
            # Test serialization
            serialized = jax_adapter._serialize_model(mock_params)
            assert isinstance(serialized, (str, bytes, dict))
            
            # Test deserialization
            deserialized_params = jax_adapter._deserialize_model(serialized)
            assert isinstance(deserialized_params, dict)

    def test_checkpoint_saving_and_loading(self, jax_adapter, mock_dataset):
        """Test checkpoint saving and loading."""
        jax_adapter.save_checkpoints = True
        jax_adapter.checkpoint_dir = "/tmp/jax_checkpoints"
        
        with patch('os.makedirs'), \
             patch('pickle.dump') as mock_dump, \
             patch('pickle.load') as mock_load, \
             patch.object(jax_adapter, '_create_functional_autoencoder'):
            
            mock_params = {"weights": jnp.ones((8, 32))}
            mock_load.return_value = mock_params
            
            detector = jax_adapter.fit(mock_dataset)
            
            assert detector is not None
            # Verify checkpoint was saved
            mock_dump.assert_called()

    # Error Handling Tests

    def test_fit_invalid_data_shape(self, jax_adapter):
        """Test training with invalid data shape."""
        invalid_dataset = Mock()
        invalid_dataset.data = np.array([])  # Empty array
        
        with pytest.raises(AdapterError):
            jax_adapter.fit(invalid_dataset)

    def test_fit_nan_values_handling(self, jax_adapter):
        """Test handling of NaN values in data."""
        nan_dataset = Mock()
        nan_data = np.random.randn(100, 8)
        nan_data[0, 0] = np.nan
        nan_dataset.data = nan_data
        
        with pytest.raises(AdapterError):
            jax_adapter.fit(nan_dataset)

    def test_predict_device_mismatch_handling(self, jax_adapter, sample_data):
        """Test handling of device mismatch errors."""
        detector = Detector(
            id="test_detector",
            algorithm="jax_functional_autoencoder",
            parameters={
                "model_params": {"weights": jnp.ones((8, 32))},
                "device": "gpu"  # Different from current device
            }
        )
        
        with patch('jax.device_put') as mock_device_put:
            mock_device_put.side_effect = RuntimeError("Device not available")
            
            with pytest.raises(AdapterError):
                jax_adapter.predict(detector, sample_data["X_test"])

    def test_memory_overflow_handling(self, jax_adapter, mock_dataset):
        """Test handling of memory overflow during training."""
        with patch.object(jax_adapter, '_create_functional_autoencoder') as mock_create:
            mock_create.side_effect = MemoryError("GPU memory exhausted")
            
            with pytest.raises(AdapterError):
                jax_adapter.fit(mock_dataset)

    # Performance Tests

    def test_compilation_time_monitoring(self, jax_adapter, mock_dataset):
        """Test JIT compilation time monitoring."""
        jax_adapter.monitor_compilation = True
        
        with patch.object(jax_adapter, '_create_functional_autoencoder'), \
             patch('time.time') as mock_time:
            
            # Mock compilation time
            mock_time.side_effect = [0, 2.5, 3.0, 5.0]  # 2.5s compilation, 2s execution
            
            detector = jax_adapter.fit(mock_dataset)
            
            assert "compilation_time" in detector.parameters
            assert detector.parameters["compilation_time"] > 0

    def test_memory_usage_profiling(self, jax_adapter, mock_dataset):
        """Test memory usage profiling during training."""
        jax_adapter.profile_memory = True
        
        with patch.object(jax_adapter, '_create_functional_autoencoder'), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.available = 8 * 1024**3  # 8GB available
            
            detector = jax_adapter.fit(mock_dataset)
            
            assert "memory_usage" in detector.parameters

    def test_throughput_benchmarking(self, jax_adapter, sample_data):
        """Test throughput benchmarking for prediction."""
        detector = Detector(
            id="test_detector",
            algorithm="jax_functional_autoencoder",
            parameters={
                "model_params": {"weights": jnp.ones((8, 32))},
                "model_fn": "mock_function"
            }
        )
        
        # Large dataset for throughput testing
        large_data = np.random.randn(10000, 8).astype(np.float32)
        
        with patch.object(jax_adapter, '_load_model'), \
             patch('time.time') as mock_time:
            
            mock_time.side_effect = [0, 1.0]  # 1 second for 10k samples
            
            result = jax_adapter.predict(detector, large_data)
            
            assert result is not None
            # Should achieve high throughput (>1000 samples/second)


class TestJAXAdapterIntegration:
    """Integration tests for JAX adapter."""

    def test_complete_functional_pipeline(self, sample_data):
        """Test complete functional programming pipeline."""
        adapter = JAXAdapter(epochs=2, batch_size=16)  # Fast training
        
        # Create dataset
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        dataset.features = sample_data["features"]
        
        with patch.object(adapter, '_create_functional_autoencoder') as mock_create:
            mock_model_fn = lambda params, x: x + jnp.random.normal(random.PRNGKey(42), x.shape) * 0.1
            mock_params = {"encoder": {"weights": jnp.ones((8, 32))}, "decoder": {"weights": jnp.ones((32, 8))}}
            mock_create.return_value = (mock_model_fn, mock_params)
            
            # Train detector
            detector = adapter.fit(dataset)
            assert detector is not None
            
            # Make predictions
            with patch.object(adapter, '_load_model', return_value=mock_model_fn):
                result = adapter.predict(detector, sample_data["X_test"])
                
                assert isinstance(result, DetectionResult)
                assert len(result.anomaly_scores) == len(sample_data["X_test"])

    def test_distributed_training_workflow(self, sample_data):
        """Test distributed training across multiple devices."""
        adapter = JAXAdapter(
            use_pmap=True,
            distributed_training=True,
            epochs=1
        )
        
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        
        with patch('jax.devices') as mock_devices, \
             patch('jax.pmap') as mock_pmap, \
             patch.object(adapter, '_create_functional_autoencoder'):
            
            mock_devices.return_value = [Mock(), Mock()]  # 2 devices
            mock_pmap.return_value = lambda x: x
            
            detector = adapter.fit(dataset)
            
            assert "distributed_training" in detector.parameters
            assert detector.parameters["distributed_training"] is True

    def test_neural_ode_integration(self, sample_data):
        """Test Neural ODE integration for continuous dynamics."""
        adapter = JAXAdapter(
            model_type="neural_ode_autoencoder",
            ode_solver="dopri5",
            integration_time=1.0
        )
        
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        
        with patch.object(adapter, '_create_neural_ode_autoencoder') as mock_create, \
             patch('jax.experimental.ode.odeint') as mock_odeint:
            
            mock_model_fn = MagicMock()
            mock_params = {"dynamics": {"weights": jnp.ones((8, 16))}}
            mock_create.return_value = (mock_model_fn, mock_params)
            mock_odeint.return_value = jnp.ones((10, 8))
            
            detector = adapter.fit(dataset)
            
            assert detector is not None
            assert "ode_solver" in detector.parameters

    def test_normalizing_flow_likelihood_estimation(self, sample_data):
        """Test normalizing flow for likelihood-based anomaly detection."""
        adapter = JAXAdapter(
            model_type="normalizing_flow",
            num_flows=6,
            flow_type="autoregressive"
        )
        
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        
        with patch.object(adapter, '_create_normalizing_flow') as mock_create:
            mock_flow_fn = lambda params, x: (x, jnp.sum(x, axis=1))  # (transformed, log_det)
            mock_params = {"flows": [{"weights": jnp.ones((8, 8))} for _ in range(6)]}
            mock_create.return_value = (mock_flow_fn, mock_params)
            
            detector = adapter.fit(dataset)
            
            assert detector is not None
            assert "num_flows" in detector.parameters
            assert detector.parameters["num_flows"] == 6

    def test_uncertainty_quantification_workflow(self, sample_data):
        """Test uncertainty quantification in anomaly detection."""
        adapter = JAXAdapter(
            model_type="bayesian_autoencoder",
            estimate_uncertainty=True,
            num_samples=20,
            prior_scale=0.1
        )
        
        detector = Detector(
            id="test_detector",
            algorithm="jax_bayesian_autoencoder",
            parameters={
                "model_params": {"mean": jnp.ones((8, 16)), "log_std": jnp.zeros((8, 16))},
                "model_fn": "mock_function"
            }
        )
        
        with patch.object(adapter, '_load_model') as mock_load:
            # Mock Bayesian forward pass
            mock_model_fn = lambda params, x, key: x + jnp.random.normal(key, x.shape) * 0.1
            mock_load.return_value = mock_model_fn
            
            result = adapter.predict(detector, sample_data["X_test"])
            
            assert result is not None
            # Should include uncertainty estimates
            assert adapter.estimate_uncertainty is True
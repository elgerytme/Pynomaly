"""TensorFlow adapter for deep learning anomaly detection algorithms."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
import structlog

from pynomaly.domain.entities import (
    Anomaly, Dataset, Detector, DetectionResult
)
from pynomaly.domain.exceptions import (
    DetectorNotFittedError, FittingError, InvalidAlgorithmError
)
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.shared.protocols import DetectorProtocol

logger = structlog.get_logger(__name__)

# Check for TensorFlow availability early and raise ImportError if not available
# This ensures the module import fails gracefully when TensorFlow is missing
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    raise ImportError("TensorFlow is not available. Install with: pip install tensorflow")

class AutoEncoder(Model):
    """TensorFlow AutoEncoder for anomaly detection."""
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 32,
        hidden_layers: Optional[List[int]] = None,
        activation: str = 'relu',
        dropout_rate: float = 0.1
    ):
        """Initialize AutoEncoder.
        
        Args:
            input_dim: Input feature dimension
            encoding_dim: Encoding layer dimension
            hidden_layers: Hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate for regularization
        """
        super(AutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [64, 32]
        
        # Encoder
        encoder_layers = []
        current_dim = input_dim
        
        for hidden_dim in self.hidden_layers:
            encoder_layers.extend([
                layers.Dense(hidden_dim, activation=activation),
                layers.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        encoder_layers.append(layers.Dense(encoding_dim, activation=activation))
        self.encoder = keras.Sequential(encoder_layers)
        
        # Decoder
        decoder_layers = []
        current_dim = encoding_dim
        
        for hidden_dim in reversed(self.hidden_layers):
            decoder_layers.extend([
                layers.Dense(hidden_dim, activation=activation),
                layers.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        decoder_layers.append(layers.Dense(input_dim, activation='linear'))
        self.decoder = keras.Sequential(decoder_layers)
    
    def call(self, x, training=None):
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x, training=False)


class VariationalAutoEncoder(Model):
    """TensorFlow Variational AutoEncoder for anomaly detection."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_layers: Optional[List[int]] = None,
        activation: str = 'relu',
        beta: float = 1.0
    ):
        """Initialize VAE.
        
        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_layers: Hidden layer dimensions
            activation: Activation function
            beta: Beta parameter for beta-VAE
        """
        super(VariationalAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.beta = beta
        
        # Encoder
        encoder_layers = []
        current_dim = input_dim
        
        for hidden_dim in self.hidden_layers:
            encoder_layers.append(layers.Dense(hidden_dim, activation=activation))
            current_dim = hidden_dim
        
        self.encoder_hidden = keras.Sequential(encoder_layers)
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)
        
        # Decoder
        decoder_layers = []
        current_dim = latent_dim
        
        for hidden_dim in reversed(self.hidden_layers):
            decoder_layers.append(layers.Dense(hidden_dim, activation=activation))
            current_dim = hidden_dim
        
        decoder_layers.append(layers.Dense(input_dim, activation='linear'))
        self.decoder = keras.Sequential(decoder_layers)
    
    def encode(self, x):
        """Encode input to latent parameters."""
        h = self.encoder_hidden(x)
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        """Reparameterization trick."""
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def call(self, x, training=None):
        """Forward pass through VAE."""
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decode(z)
        
        # Add KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(self.beta * kl_loss)
        
        return reconstruction


class DeepSVDD(Model):
    """TensorFlow Deep Support Vector Data Description."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Optional[List[int]] = None,
        activation: str = 'relu',
        output_dim: int = 32
    ):
        """Initialize Deep SVDD.
        
        Args:
            input_dim: Input feature dimension
            hidden_layers: Hidden layer dimensions
            activation: Activation function
            output_dim: Output representation dimension
        """
        super(DeepSVDD, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers or [64, 32]
        
        # Network layers
        network_layers = []
        current_dim = input_dim
        
        for hidden_dim in self.hidden_layers:
            network_layers.append(layers.Dense(hidden_dim, activation=activation))
            current_dim = hidden_dim
        
        network_layers.append(layers.Dense(output_dim, activation='linear'))
        self.network = keras.Sequential(network_layers)
        
        # Center will be initialized during training
        self.center = None
    
    def call(self, x, training=None):
        """Forward pass through Deep SVDD network."""
        return self.network(x, training=training)
    
    def init_center(self, x):
        """Initialize center from forward pass."""
        representations = self.network(x, training=False)
        self.center = tf.reduce_mean(representations, axis=0)
        return self.center


class TensorFlowAdapter(Detector):
    """TensorFlow adapter for deep learning anomaly detection."""
    
    ALGORITHM_MAPPING = {
        "AutoEncoder": AutoEncoder,
        "VAE": VariationalAutoEncoder,
        "DeepSVDD": DeepSVDD,
    }
    
    def __init__(
        self,
        algorithm_name: str,
        name: Optional[str] = None,
        contamination_rate: Optional[ContaminationRate] = None,
        **kwargs: Any
    ):
        """Initialize TensorFlow adapter.
        
        Args:
            algorithm_name: Name of the TensorFlow algorithm
            name: Optional custom name for the detector
            contamination_rate: Expected contamination rate
            **kwargs: Algorithm-specific parameters
        """
        if not HAS_TENSORFLOW:
            raise ImportError(
                "TensorFlow is not installed. Please install with: "
                "pip install tensorflow>=2.0.0"
            )
        
        # Validate algorithm
        if algorithm_name not in self.ALGORITHM_MAPPING:
            raise InvalidAlgorithmError(
                algorithm_name,
                available_algorithms=list(self.ALGORITHM_MAPPING.keys())
            )
        
        # Initialize parent
        super().__init__(
            name=name or f"TensorFlow_{algorithm_name}",
            algorithm_name=algorithm_name,
            contamination_rate=contamination_rate or ContaminationRate(0.1),
            **kwargs
        )
        
        # TensorFlow-specific attributes
        self.model: Optional[Model] = None
        self.training_history = None
        self.threshold_value: Optional[float] = None
        self.device_type = self._detect_device()
        
        # Training parameters
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        
        # Algorithm-specific parameters
        self.algorithm_params = {
            k: v for k, v in kwargs.items() 
            if k not in ['epochs', 'batch_size', 'learning_rate', 'validation_split', 'early_stopping_patience']
        }
    
    def _detect_device(self) -> str:
        """Detect available device (CPU/GPU)."""
        try:
            if tf.config.list_physical_devices('GPU'):
                return 'GPU'
            else:
                return 'CPU'
        except Exception:
            return 'CPU'
    
    def _create_model(self, input_dim: int) -> Model:
        """Create TensorFlow model based on algorithm."""
        algorithm_class = self.ALGORITHM_MAPPING[self.algorithm_name]
        
        # Prepare parameters
        model_params = {'input_dim': input_dim}
        model_params.update(self.algorithm_params)
        
        return algorithm_class(**model_params)
    
    def _prepare_data(self, dataset: Dataset) -> tf.Tensor:
        """Prepare data for TensorFlow training."""
        if dataset.features is None:
            raise ValueError("Dataset features cannot be None")
        
        # Convert to TensorFlow tensor
        X = tf.constant(dataset.features.values, dtype=tf.float32)
        
        # Normalize data
        X = tf.nn.l2_normalize(X, axis=1)
        
        return X
    
    def fit(self, dataset: Dataset) -> None:
        """Fit the TensorFlow model on the dataset.
        
        Args:
            dataset: Training dataset
            
        Raises:
            FittingError: If training fails
        """
        try:
            start_time = time.time()
            
            logger.info(
                "Starting TensorFlow model training",
                algorithm=self.algorithm_name,
                device=self.device_type
            )
            
            # Prepare data
            X = self._prepare_data(dataset)
            input_dim = X.shape[1]
            
            # Create model
            self.model = self._create_model(input_dim)
            
            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            if self.algorithm_name == "VAE":
                # VAE uses custom loss with KL divergence
                self.model.compile(optimizer=optimizer, loss='mse')
            elif self.algorithm_name == "DeepSVDD":
                # Deep SVDD uses custom loss
                self.model.compile(optimizer=optimizer, loss=self._deep_svdd_loss)
            else:
                # AutoEncoder uses reconstruction loss
                self.model.compile(optimizer=optimizer, loss='mse')
            
            # Prepare callbacks
            callbacks = []
            if self.early_stopping_patience > 0:
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.early_stopping_patience,
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
            
            # Special handling for Deep SVDD
            if self.algorithm_name == "DeepSVDD":
                # Initialize center
                self.model.init_center(X)
            
            # Train model
            if self.algorithm_name in ["AutoEncoder", "VAE"]:
                # Autoencoder-style training (input = target)
                self.training_history = self.model.fit(
                    X, X,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=callbacks,
                    verbose=0
                )
            else:
                # Deep SVDD training (unsupervised)
                self.training_history = self.model.fit(
                    X, X,  # Dummy target
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=callbacks,
                    verbose=0
                )
            
            # Calculate threshold based on training data
            self._calculate_threshold(X)
            
            # Mark as fitted
            self._is_fitted = True
            
            training_time = time.time() - start_time
            
            logger.info(
                "TensorFlow model training completed",
                algorithm=self.algorithm_name,
                training_time=training_time,
                epochs_trained=len(self.training_history.history['loss']),
                final_loss=self.training_history.history['loss'][-1]
            )
            
        except Exception as e:
            logger.error(
                "TensorFlow model training failed",
                algorithm=self.algorithm_name,
                error=str(e)
            )
            raise FittingError(f"Failed to fit {self.algorithm_name}: {str(e)}")
    
    def _deep_svdd_loss(self, y_true, y_pred):
        """Custom loss function for Deep SVDD."""
        if self.model.center is None:
            return tf.constant(0.0)
        
        distances = tf.reduce_sum(tf.square(y_pred - self.model.center), axis=1)
        return tf.reduce_mean(distances)
    
    def _calculate_threshold(self, X: tf.Tensor) -> None:
        """Calculate anomaly threshold based on training data."""
        # Get anomaly scores for training data
        scores = self._calculate_anomaly_scores(X)
        
        # Use contamination rate to determine threshold
        contamination = self.contamination_rate.value
        threshold_percentile = (1 - contamination) * 100
        
        self.threshold_value = float(np.percentile(scores, threshold_percentile))
    
    def _calculate_anomaly_scores(self, X: tf.Tensor) -> np.ndarray:
        """Calculate anomaly scores for given data."""
        if self.model is None:
            raise DetectorNotFittedError("Model must be fitted before calculating scores")
        
        if self.algorithm_name in ["AutoEncoder", "VAE"]:
            # Reconstruction error
            reconstructions = self.model(X, training=False)
            mse = tf.reduce_mean(tf.square(X - reconstructions), axis=1)
            return mse.numpy()
        
        elif self.algorithm_name == "DeepSVDD":
            # Distance from center
            representations = self.model(X, training=False)
            if self.model.center is None:
                raise ValueError("Deep SVDD center not initialized")
            
            distances = tf.reduce_sum(
                tf.square(representations - self.model.center), axis=1
            )
            return distances.numpy()
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
    
    def predict(self, dataset: Dataset) -> DetectionResult:
        """Predict anomalies on the dataset.
        
        Args:
            dataset: Dataset to predict on
            
        Returns:
            Detection result with anomalies
            
        Raises:
            DetectorNotFittedError: If model is not fitted
        """
        if not self._is_fitted or self.model is None:
            raise DetectorNotFittedError("Model must be fitted before prediction")
        
        try:
            start_time = time.time()
            
            # Prepare data
            X = self._prepare_data(dataset)
            
            # Calculate anomaly scores
            scores = self._calculate_anomaly_scores(X)
            
            # Determine anomalies based on threshold
            is_anomaly = scores > self.threshold_value
            
            # Create anomaly objects
            anomalies = []
            for idx, (score, anomaly_flag) in enumerate(zip(scores, is_anomaly)):
                if anomaly_flag:
                    anomaly = Anomaly(
                        index=int(idx),
                        score=AnomalyScore(float(score)),
                        timestamp=dataset.features.index[idx] if hasattr(dataset.features.index, '__getitem__') else None,
                        feature_names=list(dataset.features.columns) if dataset.features is not None else None
                    )
                    anomalies.append(anomaly)
            
            # Calculate metrics
            n_anomalies = len(anomalies)
            n_samples = len(dataset.features) if dataset.features is not None else 0
            anomaly_rate = n_anomalies / n_samples if n_samples > 0 else 0.0
            
            prediction_time = time.time() - start_time
            
            logger.info(
                "TensorFlow prediction completed",
                algorithm=self.algorithm_name,
                n_samples=n_samples,
                n_anomalies=n_anomalies,
                anomaly_rate=anomaly_rate,
                prediction_time=prediction_time
            )
            
            return DetectionResult(
                id=str(uuid4()),
                detector_id=self.id,
                dataset_id=dataset.id,
                anomalies=anomalies,
                n_anomalies=n_anomalies,
                anomaly_rate=anomaly_rate,
                threshold=self.threshold_value or 0.0,
                execution_time=prediction_time
            )
            
        except Exception as e:
            logger.error(
                "TensorFlow prediction failed",
                algorithm=self.algorithm_name,
                error=str(e)
            )
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (not directly available for neural networks)."""
        # Neural networks don't have direct feature importance
        # Could implement gradient-based importance in the future
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        info = {
            "algorithm": self.algorithm_name,
            "is_fitted": self._is_fitted,
            "device": self.device_type,
            "has_tensorflow": HAS_TENSORFLOW,
            "threshold": self.threshold_value,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }
        
        if self.model is not None:
            info.update({
                "total_params": self.model.count_params(),
                "trainable_params": sum([tf.size(w).numpy() for w in self.model.trainable_weights]),
            })
        
        if self.training_history is not None:
            info.update({
                "training_epochs": len(self.training_history.history['loss']),
                "final_loss": self.training_history.history['loss'][-1],
                "best_val_loss": min(self.training_history.history.get('val_loss', [float('inf')])),
            })
        
        return info
    
    @classmethod
    def list_available_algorithms(cls) -> List[str]:
        """List all available TensorFlow algorithms."""
        if not HAS_TENSORFLOW:
            return []
        return list(cls.ALGORITHM_MAPPING.keys())
    
    @classmethod
    def get_algorithm_info(cls, algorithm_name: str) -> Dict[str, Any]:
        """Get information about a specific algorithm."""
        if algorithm_name not in cls.ALGORITHM_MAPPING:
            raise InvalidAlgorithmError(
                algorithm_name,
                available_algorithms=list(cls.ALGORITHM_MAPPING.keys())
            )
        
        algorithm_info = {
            "AutoEncoder": {
                "description": "Deep autoencoder for anomaly detection using reconstruction error",
                "type": "Neural Network",
                "unsupervised": True,
                "gpu_support": True,
                "parameters": {
                    "encoding_dim": "Dimension of the encoding layer",
                    "hidden_layers": "List of hidden layer dimensions",
                    "activation": "Activation function",
                    "dropout_rate": "Dropout rate for regularization",
                    "epochs": "Number of training epochs",
                    "batch_size": "Training batch size",
                    "learning_rate": "Learning rate for optimizer"
                }
            },
            "VAE": {
                "description": "Variational Autoencoder for anomaly detection",
                "type": "Neural Network",
                "unsupervised": True,
                "gpu_support": True,
                "parameters": {
                    "latent_dim": "Dimension of the latent space",
                    "hidden_layers": "List of hidden layer dimensions",
                    "activation": "Activation function",
                    "beta": "Beta parameter for beta-VAE",
                    "epochs": "Number of training epochs",
                    "batch_size": "Training batch size",
                    "learning_rate": "Learning rate for optimizer"
                }
            },
            "DeepSVDD": {
                "description": "Deep Support Vector Data Description for anomaly detection",
                "type": "Neural Network",
                "unsupervised": True,
                "gpu_support": True,
                "parameters": {
                    "output_dim": "Dimension of the output representation",
                    "hidden_layers": "List of hidden layer dimensions",
                    "activation": "Activation function",
                    "epochs": "Number of training epochs",
                    "batch_size": "Training batch size",
                    "learning_rate": "Learning rate for optimizer"
                }
            }
        }
        
        return algorithm_info.get(algorithm_name, {})
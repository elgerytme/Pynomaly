"""TensorFlow adapter for deep learning anomaly detection algorithms."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import numpy as np
import structlog

from pynomaly.domain.entities import Anomaly, Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import (
    AdapterError,
    DetectorNotFittedError,
    FittingError,
    InvalidAlgorithmError,
)
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate

logger = structlog.get_logger(__name__)

# Check for TensorFlow availability with graceful fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Model, layers
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    # Create dummy classes to avoid import errors
    class tf:
        @staticmethod
        def constant(data, dtype=None):
            return data
        @staticmethod
        def reduce_mean(x, axis=None):
            return 0
        @staticmethod
        def reduce_sum(x, axis=None):
            return 0
        @staticmethod
        def square(x):
            return x
        @staticmethod
        def exp(x):
            return x
        @staticmethod
        def shape(x):
            return x.shape if hasattr(x, 'shape') else (1,)
        class random:
            @staticmethod
            def normal(shape):
                return 0
        class config:
            @staticmethod
            def list_physical_devices(device_type):
                return []
    
    class Model:
        def __init__(self):
            pass
        def compile(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            return MockHistory()
        def count_params(self):
            return 0
        @property
        def trainable_weights(self):
            return []
    
    class MockHistory:
        def __init__(self):
            self.history = {"loss": [1.0, 0.5, 0.3]}
    
    class layers:
        @staticmethod
        def Dense(*args, **kwargs):
            return None
        @staticmethod
        def Dropout(*args, **kwargs):
            return None
    
    class keras:
        @staticmethod
        def Sequential(*args):
            return Model()
        class optimizers:
            @staticmethod
            def Adam(*args, **kwargs):
                return None
    
    class EarlyStopping:
        def __init__(self, *args, **kwargs):
            pass


class AutoEncoder(Model):
    """TensorFlow AutoEncoder for anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 32,
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
        dropout_rate: float = 0.1,
    ):
        """Initialize AutoEncoder.

        Args:
            input_dim: Input feature dimension
            encoding_dim: Encoding layer dimension
            hidden_layers: Hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers or [64, 32]

        # Encoder
        encoder_layers = []

        for hidden_dim in self.hidden_layers:
            encoder_layers.extend(
                [
                    layers.Dense(hidden_dim, activation=activation),
                    layers.Dropout(dropout_rate),
                ]
            )

        encoder_layers.append(layers.Dense(encoding_dim, activation=activation))
        self.encoder = keras.Sequential(encoder_layers)

        # Decoder
        decoder_layers = []

        for hidden_dim in reversed(self.hidden_layers):
            decoder_layers.extend(
                [
                    layers.Dense(hidden_dim, activation=activation),
                    layers.Dropout(dropout_rate),
                ]
            )

        decoder_layers.append(layers.Dense(input_dim, activation="linear"))
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
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
        beta: float = 1.0,
    ):
        """Initialize VAE.

        Args:
            input_dim: Input feature dimension
            latent_dim: Latent space dimension
            hidden_layers: Hidden layer dimensions
            activation: Activation function
            beta: Beta parameter for beta-VAE
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers or [64, 32]
        self.beta = beta

        # Encoder
        encoder_layers = []

        for hidden_dim in self.hidden_layers:
            encoder_layers.append(layers.Dense(hidden_dim, activation=activation))

        self.encoder_hidden = keras.Sequential(encoder_layers)
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

        # Decoder
        decoder_layers = []

        for hidden_dim in reversed(self.hidden_layers):
            decoder_layers.append(layers.Dense(hidden_dim, activation=activation))

        decoder_layers.append(layers.Dense(input_dim, activation="linear"))
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
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
        output_dim: int = 32,
    ):
        """Initialize Deep SVDD.

        Args:
            input_dim: Input feature dimension
            hidden_layers: Hidden layer dimensions
            activation: Activation function
            output_dim: Output representation dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers or [64, 32]

        # Network layers
        network_layers = []

        for hidden_dim in self.hidden_layers:
            network_layers.append(layers.Dense(hidden_dim, activation=activation))

        network_layers.append(layers.Dense(output_dim, activation="linear"))
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
        name: str | None = None,
        contamination_rate: ContaminationRate | None = None,
        **kwargs: Any,
    ):
        """Initialize TensorFlow adapter.

        Args:
            algorithm_name: Name of the TensorFlow algorithm
            name: Optional custom name for the detector
            contamination_rate: Expected contamination rate
            **kwargs: Algorithm-specific parameters
        """
        if not HAS_TENSORFLOW:
            raise AdapterError(
                "TensorFlow is not available. Please install with: "
                "pip install tensorflow>=2.0.0"
            )

        # Validate algorithm
        if algorithm_name not in self.ALGORITHM_MAPPING:
            raise InvalidAlgorithmError(
                algorithm_name, available_algorithms=list(self.ALGORITHM_MAPPING.keys())
            )

        # Initialize parent
        super().__init__(
            name=name or f"TensorFlow_{algorithm_name}",
            algorithm_name=algorithm_name,
            contamination_rate=contamination_rate or ContaminationRate(0.1),
            **kwargs,
        )

        # TensorFlow-specific attributes
        self.model: Model | None = None
        self.training_history = None
        self.threshold_value: float | None = None
        self.device_type = self._detect_device()

        # Training parameters
        self.epochs = kwargs.get("epochs", 100)
        self.batch_size = kwargs.get("batch_size", 32)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.validation_split = kwargs.get("validation_split", 0.2)
        self.early_stopping_patience = kwargs.get("early_stopping_patience", 10)

        # Algorithm-specific parameters
        self.algorithm_params = {
            k: v
            for k, v in kwargs.items()
            if k
            not in [
                "epochs",
                "batch_size",
                "learning_rate",
                "validation_split",
                "early_stopping_patience",
            ]
        }

    def _detect_device(self) -> str:
        """Detect available device (CPU/GPU)."""
        try:
            if tf.config.list_physical_devices("GPU"):
                return "GPU"
            else:
                return "CPU"
        except Exception:
            return "CPU"

    def _create_model(self, input_dim: int) -> Model:
        """Create TensorFlow model based on algorithm."""
        algorithm_class = self.ALGORITHM_MAPPING[self.algorithm_name]

        # Prepare parameters
        model_params = {"input_dim": input_dim}
        model_params.update(self.algorithm_params)

        return algorithm_class(**model_params)

    def _prepare_data(self, dataset: Dataset) -> tf.Tensor:
        """Prepare data for TensorFlow training."""
        df = dataset.data
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target column if present
        if dataset.target_column and dataset.target_column in numeric_cols:
            numeric_cols.remove(dataset.target_column)
        
        if not numeric_cols:
            raise AdapterError("No numeric features found in dataset")
        
        # Extract features and handle missing values
        X = df[numeric_cols].values
        
        # Simple imputation - replace NaN with column mean
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])
        
        # Standardize features
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-6  # Add small constant to avoid division by zero
        X = (X - mean) / std
        
        # Convert to TensorFlow tensor
        return tf.constant(X, dtype=tf.float32)

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
                device=self.device_type,
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
                self.model.compile(optimizer=optimizer, loss="mse")
            elif self.algorithm_name == "DeepSVDD":
                # Deep SVDD uses custom loss
                self.model.compile(optimizer=optimizer, loss=self._deep_svdd_loss)
            else:
                # AutoEncoder uses reconstruction loss
                self.model.compile(optimizer=optimizer, loss="mse")

            # Prepare callbacks
            callbacks = []
            if self.early_stopping_patience > 0:
                early_stopping = EarlyStopping(
                    monitor="val_loss",
                    patience=self.early_stopping_patience,
                    restore_best_weights=True,
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
                    X,
                    X,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=callbacks,
                    verbose=0,
                )
            else:
                # Deep SVDD training (unsupervised)
                self.training_history = self.model.fit(
                    X,
                    X,  # Dummy target
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=callbacks,
                    verbose=0,
                )

            # Calculate threshold based on training data
            self._calculate_threshold(X)

            # Mark as fitted
            self.is_fitted = True

            training_time = time.time() - start_time

            logger.info(
                "TensorFlow model training completed",
                algorithm=self.algorithm_name,
                training_time=training_time,
                epochs_trained=len(self.training_history.history["loss"]),
                final_loss=self.training_history.history["loss"][-1],
            )

        except Exception as e:
            logger.error(
                "TensorFlow model training failed",
                algorithm=self.algorithm_name,
                error=str(e),
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
            raise DetectorNotFittedError(
                "Model must be fitted before calculating scores"
            )

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
        if not self.is_fitted or self.model is None:
            raise DetectorNotFittedError("Model must be fitted before prediction")

        try:
            start_time = time.time()

            # Prepare data
            X = self._prepare_data(dataset)

            # Calculate anomaly scores
            scores = self._calculate_anomaly_scores(X)

            # Determine anomalies based on threshold
            is_anomaly = scores > self.threshold_value

            # Create anomaly scores
            anomaly_scores = [
                AnomalyScore(
                    value=float(score),
                    confidence=self._calculate_confidence(score),
                )
                for score in scores
            ]

            # Create labels
            labels = is_anomaly.astype(int) if hasattr(is_anomaly, 'astype') else [int(x) for x in is_anomaly]

            # Create anomaly objects for detected anomalies
            anomalies = []
            for idx, (score, anomaly_flag) in enumerate(
                zip(scores, is_anomaly, strict=False)
            ):
                if anomaly_flag:
                    anomaly = Anomaly(
                        id=uuid4(),
                        data_point_index=int(idx),
                        score=AnomalyScore(float(score)),
                        features=dataset.data.iloc[idx].to_dict(),
                        timestamp=None,
                        explanation=f"TensorFlow {self.algorithm_name} detected anomaly with score {score:.3f}"
                    )
                    anomalies.append(anomaly)

            # Calculate metrics
            n_anomalies = len(anomalies)
            n_samples = len(dataset.data)

            prediction_time = time.time() - start_time

            logger.info(
                "TensorFlow prediction completed",
                algorithm=self.algorithm_name,
                n_samples=n_samples,
                n_anomalies=n_anomalies,
                prediction_time=prediction_time,
            )

            return DetectionResult(
                detector_id=self.id,
                dataset_name=dataset.name,
                scores=anomaly_scores,
                labels=labels,
                anomalies=anomalies,
                threshold=self.threshold_value or 0.0,
                execution_time_ms=prediction_time * 1000,
                metadata={
                    "algorithm": self.algorithm_name,
                    "device": self.device_type,
                    "n_samples": n_samples,
                    "n_anomalies": n_anomalies,
                    "model_type": "deep_learning",
                    "framework": "tensorflow",
                }
            )

        except Exception as e:
            logger.error(
                "TensorFlow prediction failed",
                algorithm=self.algorithm_name,
                error=str(e),
            )
            raise

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies in the dataset.

        Args:
            dataset: Dataset to analyze

        Returns:
            Detection result containing anomalies, scores, and labels
        """
        return self.predict(dataset)

    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores for the dataset.

        Args:
            dataset: Dataset to score

        Returns:
            List of anomaly scores
        """
        if not self.is_fitted or self.model is None:
            raise DetectorNotFittedError("Model must be fitted before scoring")

        try:
            # Prepare data
            X = self._prepare_data(dataset)

            # Calculate anomaly scores
            scores = self._calculate_anomaly_scores(X)

            # Create AnomalyScore objects
            return [
                AnomalyScore(
                    value=float(score),
                    confidence=self._calculate_confidence(score),
                )
                for score in scores
            ]

        except Exception as e:
            raise AdapterError(f"Failed to score with TensorFlow model: {e}")

    def fit_detect(self, dataset: Dataset) -> DetectionResult:
        """Fit the detector and detect anomalies in one step.

        Args:
            dataset: Dataset to fit and analyze

        Returns:
            Detection result
        """
        self.fit(dataset)
        return self.detect(dataset)

    def get_params(self) -> dict[str, Any]:
        """Get parameters of the detector.

        Returns:
            Dictionary of parameters
        """
        return {
            "algorithm_name": self.algorithm_name,
            "contamination_rate": self.contamination_rate.value,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            **self.algorithm_params,
        }

    def set_params(self, **params: Any) -> None:
        """Set parameters of the detector.

        Args:
            **params: Parameters to set
        """
        # Update training parameters
        if "epochs" in params:
            self.epochs = params["epochs"]
        if "batch_size" in params:
            self.batch_size = params["batch_size"]
        if "learning_rate" in params:
            self.learning_rate = params["learning_rate"]
        if "validation_split" in params:
            self.validation_split = params["validation_split"]
        if "early_stopping_patience" in params:
            self.early_stopping_patience = params["early_stopping_patience"]
        
        # Update contamination rate
        if "contamination_rate" in params:
            self.contamination_rate = ContaminationRate(params["contamination_rate"])
        
        # Update algorithm parameters
        algorithm_param_keys = [
            "encoding_dim", "hidden_layers", "activation", "dropout_rate",
            "latent_dim", "beta", "output_dim"
        ]
        for key in algorithm_param_keys:
            if key in params:
                self.algorithm_params[key] = params[key]

    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence score for anomaly.

        Args:
            score: Anomaly score

        Returns:
            Confidence value between 0 and 1
        """
        if self.threshold_value is None:
            return 0.5
        
        # Simple confidence calculation based on distance from threshold
        if score <= self.threshold_value:
            return 1.0 - (score / self.threshold_value) * 0.5
        else:
            return 0.5 + min((score - self.threshold_value) / self.threshold_value * 0.5, 0.5)

    @property
    def supports_streaming(self) -> bool:
        """Whether this detector supports streaming detection."""
        return False

    @property
    def requires_fitting(self) -> bool:
        """Whether this detector requires fitting before detection."""
        return True

    def get_feature_importance(self) -> dict[str, float] | None:
        """Get feature importance (not directly available for neural networks)."""
        # Neural networks don't have direct feature importance
        # Could implement gradient-based importance in the future
        return None

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the trained model."""
        info = {
            "algorithm": self.algorithm_name,
            "is_fitted": self.is_fitted,
            "device": self.device_type,
            "has_tensorflow": HAS_TENSORFLOW,
            "threshold": self.threshold_value,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }

        if self.model is not None:
            info.update(
                {
                    "total_params": self.model.count_params(),
                    "trainable_params": sum(
                        [tf.size(w).numpy() for w in self.model.trainable_weights]
                    ),
                }
            )

        if self.training_history is not None:
            info.update(
                {
                    "training_epochs": len(self.training_history.history["loss"]),
                    "final_loss": self.training_history.history["loss"][-1],
                    "best_val_loss": min(
                        self.training_history.history.get("val_loss", [float("inf")])
                    ),
                }
            )

        return info

    @classmethod
    def list_available_algorithms(cls) -> list[str]:
        """List all available TensorFlow algorithms."""
        if not HAS_TENSORFLOW:
            return []
        return list(cls.ALGORITHM_MAPPING.keys())

    @classmethod
    def get_algorithm_info(cls, algorithm_name: str) -> dict[str, Any]:
        """Get information about a specific algorithm."""
        if algorithm_name not in cls.ALGORITHM_MAPPING:
            raise InvalidAlgorithmError(
                algorithm_name, available_algorithms=list(cls.ALGORITHM_MAPPING.keys())
            )

        algorithm_info = {
            "AutoEncoder": {
                "description": "Deep autoencoder for anomaly detection using reconstruction error",
                "type": "Neural Network",
                "unsupervised": True,
                "gpu_support": True,
                "distributed_training": True,
                "parameters": {
                    "encoding_dim": "Dimension of the encoding layer",
                    "hidden_layers": "List of hidden layer dimensions",
                    "activation": "Activation function",
                    "dropout_rate": "Dropout rate for regularization",
                    "epochs": "Number of training epochs",
                    "batch_size": "Training batch size",
                    "learning_rate": "Learning rate for optimizer",
                },
                "suitable_for": ["tabular_data", "high_dimensional", "reconstruction_based"],
                "pros": ["Fast training", "Good for tabular data", "Interpretable"],
                "cons": ["May overfit", "Requires parameter tuning"]
            },
            "VAE": {
                "description": "Variational Autoencoder for anomaly detection",
                "type": "Neural Network",
                "unsupervised": True,
                "gpu_support": True,
                "distributed_training": True,
                "parameters": {
                    "latent_dim": "Dimension of the latent space",
                    "hidden_layers": "List of hidden layer dimensions",
                    "activation": "Activation function",
                    "beta": "Beta parameter for beta-VAE",
                    "epochs": "Number of training epochs",
                    "batch_size": "Training batch size",
                    "learning_rate": "Learning rate for optimizer",
                },
                "suitable_for": ["probabilistic_modeling", "generative_tasks"],
                "pros": ["Principled probabilistic approach", "Can generate samples"],
                "cons": ["More complex", "Requires careful hyperparameter tuning"]
            },
            "DeepSVDD": {
                "description": "Deep Support Vector Data Description for anomaly detection",
                "type": "Neural Network",
                "unsupervised": True,
                "gpu_support": True,
                "distributed_training": True,
                "parameters": {
                    "output_dim": "Dimension of the output representation",
                    "hidden_layers": "List of hidden layer dimensions",
                    "activation": "Activation function",
                    "epochs": "Number of training epochs",
                    "batch_size": "Training batch size",
                    "learning_rate": "Learning rate for optimizer",
                },
                "suitable_for": ["one_class_classification", "compact_representations"],
                "pros": ["Theoretically motivated", "Good for one-class problems"],
                "cons": ["Sensitive to initialization", "May require large datasets"]
            },
        }

        return algorithm_info.get(algorithm_name, {})

    @classmethod
    def get_supported_algorithms(cls) -> list[str]:
        """Get list of supported TensorFlow algorithms.

        Returns:
            List of algorithm names
        """
        return cls.list_available_algorithms()

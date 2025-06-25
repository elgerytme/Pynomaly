"""TensorFlow/Keras-based deep learning adapter for anomaly detection."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.shared.protocols import DetectorProtocol

# Optional TensorFlow imports with fallbacks
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    tf = None
    keras = None
    layers = None
    Model = None
    EarlyStopping = None
    ReduceLROnPlateau = None
    TENSORFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class KerasAutoEncoderConfig(BaseModel):
    """Configuration for Keras AutoEncoder."""
    
    input_dim: int = Field(description="Input dimension")
    encoder_dims: List[int] = Field(default=[128, 64, 32], description="Encoder layer dimensions")
    latent_dim: int = Field(default=16, description="Latent space dimension")
    activation: str = Field(default="relu", description="Activation function")
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.9, description="Dropout rate")
    use_batch_norm: bool = Field(default=True, description="Use batch normalization")
    
    # Training parameters
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    epochs: int = Field(default=100, ge=1, description="Training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size")
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5, description="Validation split")
    early_stopping_patience: int = Field(default=15, ge=1, description="Early stopping patience")
    
    # Anomaly detection
    contamination: float = Field(default=0.1, gt=0.0, lt=1.0, description="Contamination rate")
    loss_function: str = Field(default="mse", description="Loss function")


class KerasVAEConfig(BaseModel):
    """Configuration for Keras Variational AutoEncoder."""
    
    input_dim: int = Field(description="Input dimension")
    encoder_dims: List[int] = Field(default=[256, 128], description="Encoder dimensions")
    latent_dim: int = Field(default=32, description="Latent dimension")
    decoder_dims: List[int] = Field(default=[128, 256], description="Decoder dimensions")
    
    # VAE specific
    beta: float = Field(default=1.0, ge=0.0, description="Beta parameter for beta-VAE")
    learning_rate: float = Field(default=0.0005, gt=0.0, description="Learning rate")
    epochs: int = Field(default=150, ge=1, description="Training epochs")
    batch_size: int = Field(default=64, ge=1, description="Batch size")
    validation_split: float = Field(default=0.2, description="Validation split")
    
    # Detection
    contamination: float = Field(default=0.1, gt=0.0, lt=1.0, description="Contamination rate")
    use_reconstruction_error: bool = Field(default=True, description="Use reconstruction error")
    use_kl_divergence: bool = Field(default=False, description="Include KL divergence in score")


class KerasLSTMConfig(BaseModel):
    """Configuration for Keras LSTM AutoEncoder."""
    
    input_dim: int = Field(description="Input feature dimension")
    sequence_length: int = Field(default=10, ge=1, description="Sequence length")
    lstm_units: List[int] = Field(default=[64, 32], description="LSTM layer units")
    dropout: float = Field(default=0.2, ge=0.0, le=0.9, description="Dropout rate")
    recurrent_dropout: float = Field(default=0.2, ge=0.0, le=0.9, description="Recurrent dropout")
    
    # Training
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    epochs: int = Field(default=100, ge=1, description="Training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size")
    validation_split: float = Field(default=0.2, description="Validation split")
    
    # Detection
    contamination: float = Field(default=0.1, gt=0.0, lt=1.0, description="Contamination rate")
    prediction_steps: int = Field(default=1, ge=1, description="Steps to predict ahead")


class TransformerConfig(BaseModel):
    """Configuration for Transformer-based anomaly detection."""
    
    input_dim: int = Field(description="Input dimension")
    sequence_length: int = Field(default=100, ge=1, description="Sequence length")
    d_model: int = Field(default=128, description="Model dimension")
    num_heads: int = Field(default=8, description="Number of attention heads")
    num_layers: int = Field(default=4, description="Number of transformer layers")
    dff: int = Field(default=512, description="Feed-forward dimension")
    dropout_rate: float = Field(default=0.1, description="Dropout rate")
    
    # Training
    learning_rate: float = Field(default=0.0001, gt=0.0, description="Learning rate")
    epochs: int = Field(default=50, ge=1, description="Training epochs")
    batch_size: int = Field(default=16, ge=1, description="Batch size")
    
    # Detection
    contamination: float = Field(default=0.1, gt=0.0, lt=1.0, description="Contamination rate")


def create_autoencoder_model(config: KerasAutoEncoderConfig) -> Model:
    """Create Keras AutoEncoder model."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")
    
    # Input layer
    input_layer = keras.Input(shape=(config.input_dim,))
    
    # Encoder
    x = input_layer
    for dim in config.encoder_dims:
        x = layers.Dense(dim, activation=config.activation)(x)
        if config.use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout_rate)(x)
    
    # Latent space
    latent = layers.Dense(config.latent_dim, activation=config.activation, name='latent')(x)
    
    # Decoder
    x = latent
    for dim in reversed(config.encoder_dims):
        x = layers.Dense(dim, activation=config.activation)(x)
        if config.use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.dropout_rate)(x)
    
    # Output layer
    output = layers.Dense(config.input_dim, activation='linear', name='output')(x)
    
    # Create model
    autoencoder = Model(input_layer, output, name='autoencoder')
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    autoencoder.compile(optimizer=optimizer, loss=config.loss_function, metrics=['mae'])
    
    return autoencoder


def create_vae_model(config: KerasVAEConfig) -> Tuple[Model, Model, Model]:
    """Create Keras VAE model (encoder, decoder, full VAE)."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")
    
    # Encoder
    encoder_input = keras.Input(shape=(config.input_dim,))
    x = encoder_input
    
    for dim in config.encoder_dims:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    
    # Latent space
    z_mean = layers.Dense(config.latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(config.latent_dim, name='z_log_var')(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    decoder_input = keras.Input(shape=(config.latent_dim,))
    x = decoder_input
    
    for dim in config.decoder_dims:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
    
    decoder_output = layers.Dense(config.input_dim, activation='linear')(x)
    decoder = Model(decoder_input, decoder_output, name='decoder')
    
    # Full VAE
    vae_output = decoder(z)
    vae = Model(encoder_input, vae_output, name='vae')
    
    # VAE loss
    def vae_loss(y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        return reconstruction_loss + config.beta * kl_loss
    
    vae.compile(optimizer=keras.optimizers.Adam(config.learning_rate), loss=vae_loss)
    
    return encoder, decoder, vae


def create_lstm_autoencoder(config: KerasLSTMConfig) -> Model:
    """Create LSTM AutoEncoder model."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")
    
    # Input
    input_layer = keras.Input(shape=(config.sequence_length, config.input_dim))
    
    # Encoder LSTM layers
    x = input_layer
    for i, units in enumerate(config.lstm_units):
        return_sequences = i < len(config.lstm_units) - 1
        x = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout
        )(x)
    
    # Repeat vector to decode
    x = layers.RepeatVector(config.sequence_length)(x)
    
    # Decoder LSTM layers
    for units in reversed(config.lstm_units):
        x = layers.LSTM(
            units,
            return_sequences=True,
            dropout=config.dropout,
            recurrent_dropout=config.recurrent_dropout
        )(x)
    
    # Output layer
    output = layers.TimeDistributed(layers.Dense(config.input_dim))(x)
    
    # Create model
    model = Model(input_layer, output, name='lstm_autoencoder')
    
    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def create_transformer_model(config: TransformerConfig) -> Model:
    """Create Transformer-based anomaly detection model."""
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")
    
    # Multi-head attention layer
    class MultiHeadSelfAttention(layers.Layer):
        def __init__(self, d_model, num_heads):
            super(MultiHeadSelfAttention, self).__init__()
            self.num_heads = num_heads
            self.d_model = d_model
            
            assert d_model % self.num_heads == 0
            
            self.depth = d_model // self.num_heads
            
            self.wq = layers.Dense(d_model)
            self.wk = layers.Dense(d_model)
            self.wv = layers.Dense(d_model)
            
            self.dense = layers.Dense(d_model)
        
        def call(self, x):
            batch_size = tf.shape(x)[0]
            
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)
            
            q = tf.reshape(q, (batch_size, -1, self.num_heads, self.depth))
            q = tf.transpose(q, perm=[0, 2, 1, 3])
            
            k = tf.reshape(k, (batch_size, -1, self.num_heads, self.depth))
            k = tf.transpose(k, perm=[0, 2, 1, 3])
            
            v = tf.reshape(v, (batch_size, -1, self.num_heads, self.depth))
            v = tf.transpose(v, perm=[0, 2, 1, 3])
            
            attention = tf.matmul(q, k, transpose_b=True)
            attention = attention / tf.math.sqrt(tf.cast(self.depth, tf.float32))
            attention = tf.nn.softmax(attention, axis=-1)
            
            output = tf.matmul(attention, v)
            output = tf.transpose(output, perm=[0, 2, 1, 3])
            output = tf.reshape(output, (batch_size, -1, self.d_model))
            
            return self.dense(output)
    
    # Transformer block
    def transformer_block(x, d_model, num_heads, dff, dropout_rate):
        # Multi-head attention
        attn_output = MultiHeadSelfAttention(d_model, num_heads)(x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed forward
        ffn_output = layers.Dense(dff, activation='relu')(out1)
        ffn_output = layers.Dense(d_model)(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        
        return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    
    # Model architecture
    inputs = keras.Input(shape=(config.sequence_length, config.input_dim))
    
    # Embedding and positional encoding
    x = layers.Dense(config.d_model)(inputs)
    
    # Transformer layers
    for _ in range(config.num_layers):
        x = transformer_block(x, config.d_model, config.num_heads, config.dff, config.dropout_rate)
    
    # Global average pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(config.d_model // 2, activation='relu')(x)
    x = layers.Dropout(config.dropout_rate)(x)
    outputs = layers.Dense(config.input_dim)(x)
    
    model = Model(inputs, outputs, name='transformer_anomaly_detector')
    
    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


class TensorFlowAdapter(DetectorProtocol):
    """TensorFlow/Keras-based deep learning adapter."""
    
    def __init__(
        self,
        algorithm: str = "autoencoder",
        model_config: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        gpu_memory_growth: bool = True
    ):
        """Initialize TensorFlow adapter.
        
        Args:
            algorithm: Algorithm type ('autoencoder', 'vae', 'lstm', 'transformer')
            model_config: Algorithm-specific configuration
            random_state: Random seed
            gpu_memory_growth: Enable GPU memory growth
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for TensorFlowAdapter. "
                "Install with: pip install tensorflow"
            )
        
        self.algorithm = algorithm
        self.model_config = model_config or {}
        self.random_state = random_state
        
        # Configure TensorFlow
        if gpu_memory_growth:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"GPU configuration failed: {e}")
        
        # Set random seeds
        if random_state is not None:
            tf.random.set_seed(random_state)
        
        # Model components
        self.model = None
        self.encoder_model = None
        self.decoder_model = None
        self.scaler = None
        self.threshold = None
        self.is_trained = False
        
        logger.info(f"Initialized TensorFlowAdapter with {algorithm}")
    
    @property
    def algorithm_name(self) -> str:
        """Get algorithm name."""
        return f"tensorflow_{self.algorithm}"
    
    @property
    def algorithm_params(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        return {
            "algorithm": self.algorithm,
            "model_config": self.model_config,
            "random_state": self.random_state
        }
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TensorFlowAdapter':
        """Train the model."""
        try:
            # Preprocess data
            X_processed = self._preprocess_data(X)
            
            # Create and train model
            if self.algorithm == "autoencoder":
                self._fit_autoencoder(X_processed)
            elif self.algorithm == "vae":
                self._fit_vae(X_processed)
            elif self.algorithm == "lstm":
                self._fit_lstm(X_processed)
            elif self.algorithm == "transformer":
                self._fit_transformer(X_processed)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            # Calculate threshold
            self._calculate_threshold(X_processed)
            
            self.is_trained = True
            logger.info(f"TensorFlow {self.algorithm} model trained successfully")
            
            return self
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Failed to train TensorFlow model: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_processed = self._preprocess_data(X, fit_scaler=False)
        
        if self.algorithm == "autoencoder":
            scores = self._autoencoder_scores(X_processed)
        elif self.algorithm == "vae":
            scores = self._vae_scores(X_processed)
        elif self.algorithm == "lstm":
            scores = self._lstm_scores(X_processed)
        elif self.algorithm == "transformer":
            scores = self._transformer_scores(X_processed)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return scores
    
    def _preprocess_data(self, X: np.ndarray, fit_scaler: bool = True) -> np.ndarray:
        """Preprocess data."""
        if hasattr(X, 'values'):
            X = X.values
        
        # Normalize data
        if fit_scaler:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not fitted")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled.astype(np.float32)
    
    def _fit_autoencoder(self, X: np.ndarray):
        """Train AutoEncoder."""
        config = KerasAutoEncoderConfig(
            input_dim=X.shape[1],
            **self.model_config
        )
        
        self.model = create_autoencoder_model(config)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train
        history = self.model.fit(
            X, X,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        logger.info(f"AutoEncoder training completed. Final loss: {history.history['loss'][-1]:.6f}")
    
    def _fit_vae(self, X: np.ndarray):
        """Train VAE."""
        config = KerasVAEConfig(
            input_dim=X.shape[1],
            **self.model_config
        )
        
        self.encoder_model, self.decoder_model, self.model = create_vae_model(config)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8)
        ]
        
        history = self.model.fit(
            X, X,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        logger.info(f"VAE training completed. Final loss: {history.history['loss'][-1]:.6f}")
    
    def _fit_lstm(self, X: np.ndarray):
        """Train LSTM AutoEncoder."""
        config = KerasLSTMConfig(
            input_dim=X.shape[1],
            **self.model_config
        )
        
        # Create sequences
        X_sequences = self._create_sequences(X, config.sequence_length)
        
        self.model = create_lstm_autoencoder(config)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=7)
        ]
        
        history = self.model.fit(
            X_sequences, X_sequences,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        logger.info(f"LSTM training completed. Final loss: {history.history['loss'][-1]:.6f}")
    
    def _fit_transformer(self, X: np.ndarray):
        """Train Transformer model."""
        config = TransformerConfig(
            input_dim=X.shape[1],
            **self.model_config
        )
        
        # Create sequences
        X_sequences = self._create_sequences(X, config.sequence_length)
        
        self.model = create_transformer_model(config)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5)
        ]
        
        # For transformer, predict next step
        y_sequences = X_sequences[:, 1:, :]  # Shift by one step
        X_sequences = X_sequences[:, :-1, :]  # Remove last step
        
        history = self.model.fit(
            X_sequences, y_sequences,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_split=config.validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        logger.info(f"Transformer training completed. Final loss: {history.history['loss'][-1]:.6f}")
    
    def _create_sequences(self, X: np.ndarray, sequence_length: int) -> np.ndarray:
        """Create sequences for time series models."""
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
        return np.array(sequences)
    
    def _autoencoder_scores(self, X: np.ndarray) -> np.ndarray:
        """Calculate reconstruction error for AutoEncoder."""
        reconstructed = self.model.predict(X, verbose=0)
        scores = np.mean((X - reconstructed) ** 2, axis=1)
        return scores
    
    def _vae_scores(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores for VAE."""
        reconstructed = self.model.predict(X, verbose=0)
        recon_error = np.mean((X - reconstructed) ** 2, axis=1)
        
        if hasattr(self.model, 'config') and self.model.config.use_kl_divergence:
            # Calculate KL divergence
            z_mean, z_log_var, _ = self.encoder_model.predict(X, verbose=0)
            kl_loss = -0.5 * np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1)
            scores = recon_error + 0.1 * kl_loss
        else:
            scores = recon_error
        
        return scores
    
    def _lstm_scores(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction error for LSTM."""
        config = KerasLSTMConfig(**self.model_config)
        
        X_sequences = self._create_sequences(X, config.sequence_length)
        reconstructed = self.model.predict(X_sequences, verbose=0)
        
        # Calculate reconstruction error
        scores = np.mean((X_sequences - reconstructed) ** 2, axis=(1, 2))
        
        # Pad scores to match original length
        padding = np.zeros(config.sequence_length - 1)
        scores = np.concatenate([padding, scores])
        
        return scores
    
    def _transformer_scores(self, X: np.ndarray) -> np.ndarray:
        """Calculate prediction error for Transformer."""
        config = TransformerConfig(**self.model_config)
        
        X_sequences = self._create_sequences(X, config.sequence_length)
        
        # Predict next step
        X_input = X_sequences[:, :-1, :]
        y_true = X_sequences[:, 1:, :]
        y_pred = self.model.predict(X_input, verbose=0)
        
        # Calculate prediction error
        scores = np.mean((y_true - y_pred) ** 2, axis=(1, 2))
        
        # Pad scores
        padding = np.zeros(config.sequence_length - 1)
        scores = np.concatenate([padding, scores])
        
        return scores
    
    def _calculate_threshold(self, X: np.ndarray):
        """Calculate anomaly threshold."""
        scores = self.decision_function(X)
        
        # Get contamination rate from config
        contamination = 0.1
        if hasattr(self, 'model_config'):
            contamination = self.model_config.get('contamination', 0.1)
        
        self.threshold = np.percentile(scores, (1 - contamination) * 100)
        logger.info(f"Calculated threshold: {self.threshold:.6f}")
    
    def save_model(self, path: Union[str, Path]):
        """Save trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save main model
        self.model.save(path / "model.h5")
        
        # Save additional models for VAE
        if self.encoder_model:
            self.encoder_model.save(path / "encoder.h5")
        if self.decoder_model:
            self.decoder_model.save(path / "decoder.h5")
        
        # Save metadata
        metadata = {
            "algorithm": self.algorithm,
            "model_config": self.model_config,
            "threshold": float(self.threshold),
            "random_state": self.random_state,
            "scaler_params": {
                "mean_": self.scaler.mean_.tolist(),
                "scale_": self.scaler.scale_.tolist()
            } if self.scaler else None
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]):
        """Load trained model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        self.algorithm = metadata["algorithm"]
        self.model_config = metadata["model_config"]
        self.threshold = metadata["threshold"]
        self.random_state = metadata.get("random_state")
        
        # Restore scaler
        if metadata.get("scaler_params"):
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(metadata["scaler_params"]["mean_"])
            self.scaler.scale_ = np.array(metadata["scaler_params"]["scale_"])
        
        # Load models
        self.model = keras.models.load_model(path / "model.h5")
        
        if (path / "encoder.h5").exists():
            self.encoder_model = keras.models.load_model(path / "encoder.h5")
        if (path / "decoder.h5").exists():
            self.decoder_model = keras.models.load_model(path / "decoder.h5")
        
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
    
    async def async_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'TensorFlowAdapter':
        """Asynchronous training."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fit, X, y)
    
    async def async_predict(self, X: np.ndarray) -> np.ndarray:
        """Asynchronous prediction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "algorithm": self.algorithm,
            "is_trained": self.is_trained,
            "model_config": self.model_config,
            "threshold": self.threshold
        }
        
        if self.model and self.is_trained:
            info.update({
                "total_parameters": self.model.count_params(),
                "model_summary": str(self.model.summary())
            })
        
        return info
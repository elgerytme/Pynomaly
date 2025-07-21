"""
Deep Learning Anomaly Detector for Pynomaly Detection
======================================================

Advanced neural network-based anomaly detection using TensorFlow/PyTorch.
Includes autoencoders, variational autoencoders, and other deep learning models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

@dataclass
class DeepLearningConfig:
    """Configuration for deep learning models."""
    framework: str = "tensorflow"  # tensorflow or pytorch
    model_type: str = "autoencoder"  # autoencoder, vae, lstm_ae, cnn_ae
    
    # Architecture
    hidden_neurons: List[int] = None
    latent_dim: int = 32
    dropout_rate: float = 0.2
    activation: str = "relu"
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    # Regularization
    l1_reg: float = 0.0
    l2_reg: float = 0.001
    early_stopping: bool = True
    patience: int = 10
    
    # VAE specific
    beta: float = 1.0  # Beta-VAE parameter
    
    # Preprocessing
    normalize: bool = True
    scaler_type: str = "standard"  # standard, minmax
    
    # Anomaly detection
    contamination: float = 0.1
    threshold_method: str = "percentile"  # percentile, std_dev, auto
    
    def __post_init__(self):
        if self.hidden_neurons is None:
            self.hidden_neurons = [128, 64, 32, 64, 128]

class BaseDeepAnomalyDetector(ABC):
    """Base class for deep learning anomaly detectors."""
    
    def __init__(self, config: DeepLearningConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.threshold = None
        self.history = None
        self.is_fitted = False
        
    @abstractmethod
    def _build_model(self, input_dim: int) -> Any:
        """Build the neural network model."""
        pass
    
    @abstractmethod
    def _train_model(self, X: np.ndarray, X_val: np.ndarray = None) -> Any:
        """Train the model."""
        pass
    
    @abstractmethod
    def _compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        pass
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the anomaly detector."""
        # Preprocessing
        X_scaled = self._preprocess_data(X, fit=True)
        
        # Split data for validation
        if self.config.validation_split > 0:
            X_train, X_val = train_test_split(
                X_scaled, 
                test_size=self.config.validation_split, 
                random_state=42
            )
        else:
            X_train, X_val = X_scaled, None
        
        # Build model
        input_dim = X_train.shape[1]
        self.model = self._build_model(input_dim)
        
        # Train model
        self.history = self._train_model(X_train, X_val)
        
        # Compute threshold
        anomaly_scores = self._compute_anomaly_scores(X_train)
        self.threshold = self._compute_threshold(anomaly_scores)
        
        self.is_fitted = True
        logger.info(f"Deep anomaly detector fitted with threshold: {self.threshold:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for anomaly, 0 for normal)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        anomaly_scores = self.decision_function(X)
        return (anomaly_scores > self.threshold).astype(int)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self._preprocess_data(X, fit=False)
        return self._compute_anomaly_scores(X_scaled)
    
    def _preprocess_data(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocess data with scaling."""
        if not self.config.normalize:
            return X
        
        if fit:
            if self.config.scaler_type == "standard":
                self.scaler = StandardScaler()
            elif self.config.scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")
            
            return self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted")
            return self.scaler.transform(X)
    
    def _compute_threshold(self, anomaly_scores: np.ndarray) -> float:
        """Compute anomaly threshold."""
        if self.config.threshold_method == "percentile":
            return np.percentile(anomaly_scores, 100 * (1 - self.config.contamination))
        elif self.config.threshold_method == "std_dev":
            mean_score = np.mean(anomaly_scores)
            std_score = np.std(anomaly_scores)
            return mean_score + 2 * std_score
        elif self.config.threshold_method == "auto":
            # Use IQR method
            q75, q25 = np.percentile(anomaly_scores, [75, 25])
            iqr = q75 - q25
            return q75 + 1.5 * iqr
        else:
            raise ValueError(f"Unknown threshold method: {self.config.threshold_method}")

class TensorFlowAnomalyDetector(BaseDeepAnomalyDetector):
    """TensorFlow-based deep anomaly detector."""
    
    def __init__(self, config: DeepLearningConfig):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for TensorFlow-based detection")
        
        super().__init__(config)
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def _build_model(self, input_dim: int) -> keras.Model:
        """Build TensorFlow model."""
        if self.config.model_type == "autoencoder":
            return self._build_autoencoder(input_dim)
        elif self.config.model_type == "vae":
            return self._build_vae(input_dim)
        elif self.config.model_type == "lstm_ae":
            return self._build_lstm_autoencoder(input_dim)
        elif self.config.model_type == "cnn_ae":
            return self._build_cnn_autoencoder(input_dim)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _build_autoencoder(self, input_dim: int) -> keras.Model:
        """Build standard autoencoder."""
        # Input layer
        input_layer = layers.Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for neurons in self.config.hidden_neurons[:len(self.config.hidden_neurons)//2]:
            encoded = layers.Dense(
                neurons, 
                activation=self.config.activation,
                kernel_regularizer=keras.regularizers.l1_l2(
                    l1=self.config.l1_reg, 
                    l2=self.config.l2_reg
                )
            )(encoded)
            encoded = layers.Dropout(self.config.dropout_rate)(encoded)
        
        # Latent space
        latent = layers.Dense(
            self.config.latent_dim, 
            activation=self.config.activation,
            name="latent"
        )(encoded)
        
        # Decoder
        decoded = latent
        for neurons in reversed(self.config.hidden_neurons[len(self.config.hidden_neurons)//2:]):
            decoded = layers.Dense(
                neurons, 
                activation=self.config.activation,
                kernel_regularizer=keras.regularizers.l1_l2(
                    l1=self.config.l1_reg, 
                    l2=self.config.l2_reg
                )
            )(decoded)
            decoded = layers.Dropout(self.config.dropout_rate)(decoded)
        
        # Output layer
        output_layer = layers.Dense(input_dim, activation="linear")(decoded)
        
        # Create model
        autoencoder = keras.Model(input_layer, output_layer, name="autoencoder")
        
        # Compile
        autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        
        return autoencoder
    
    def _build_vae(self, input_dim: int) -> keras.Model:
        """Build Variational Autoencoder."""
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        
        h = encoder_input
        for neurons in self.config.hidden_neurons[:len(self.config.hidden_neurons)//2]:
            h = layers.Dense(neurons, activation=self.config.activation)(h)
            h = layers.Dropout(self.config.dropout_rate)(h)
        
        # Latent space parameters
        z_mean = layers.Dense(self.config.latent_dim, name="z_mean")(h)
        z_log_var = layers.Dense(self.config.latent_dim, name="z_log_var")(h)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
        
        # Decoder
        decoder_h = z
        for neurons in reversed(self.config.hidden_neurons[len(self.config.hidden_neurons)//2:]):
            decoder_h = layers.Dense(neurons, activation=self.config.activation)(decoder_h)
            decoder_h = layers.Dropout(self.config.dropout_rate)(decoder_h)
        
        decoder_output = layers.Dense(input_dim, activation="linear")(decoder_h)
        
        # VAE model
        vae = keras.Model(encoder_input, decoder_output, name="vae")
        
        # Custom loss function
        def vae_loss(y_true, y_pred):
            reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            return reconstruction_loss + self.config.beta * kl_loss
        
        vae.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss=vae_loss
        )
        
        return vae
    
    def _build_lstm_autoencoder(self, input_dim: int) -> keras.Model:
        """Build LSTM autoencoder for time series data."""
        # Reshape for LSTM (assuming time series)
        timesteps = min(10, input_dim)
        features = input_dim // timesteps
        
        # Input
        input_layer = layers.Input(shape=(timesteps, features))
        
        # Encoder
        encoded = layers.LSTM(
            self.config.hidden_neurons[0], 
            return_sequences=True, 
            dropout=self.config.dropout_rate
        )(input_layer)
        encoded = layers.LSTM(
            self.config.latent_dim, 
            return_sequences=False, 
            dropout=self.config.dropout_rate
        )(encoded)
        
        # Decoder
        decoded = layers.RepeatVector(timesteps)(encoded)
        decoded = layers.LSTM(
            self.config.latent_dim, 
            return_sequences=True, 
            dropout=self.config.dropout_rate
        )(decoded)
        decoded = layers.LSTM(
            self.config.hidden_neurons[0], 
            return_sequences=True, 
            dropout=self.config.dropout_rate
        )(decoded)
        
        # Output
        output_layer = layers.TimeDistributed(layers.Dense(features))(decoded)
        
        # Model
        lstm_ae = keras.Model(input_layer, output_layer, name="lstm_autoencoder")
        
        lstm_ae.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse"
        )
        
        return lstm_ae
    
    def _build_cnn_autoencoder(self, input_dim: int) -> keras.Model:
        """Build CNN autoencoder for high-dimensional data."""
        # Reshape for CNN (assuming square image-like data)
        img_size = int(np.sqrt(input_dim))
        if img_size * img_size != input_dim:
            img_size = int(np.sqrt(input_dim)) + 1
            padding_size = img_size * img_size - input_dim
        else:
            padding_size = 0
        
        # Input
        input_layer = layers.Input(shape=(img_size, img_size, 1))
        
        # Encoder
        x = layers.Conv2D(32, (3, 3), activation=self.config.activation, padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation=self.config.activation, padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder
        x = layers.Conv2D(16, (3, 3), activation=self.config.activation, padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation=self.config.activation, padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)
        
        # Model
        cnn_ae = keras.Model(input_layer, decoded, name="cnn_autoencoder")
        
        cnn_ae.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse"
        )
        
        return cnn_ae
    
    def _train_model(self, X: np.ndarray, X_val: np.ndarray = None) -> Any:
        """Train TensorFlow model."""
        # Prepare callbacks
        callback_list = []
        
        if self.config.early_stopping:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.config.patience,
                restore_best_weights=True
            )
            callback_list.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        callback_list.append(reduce_lr)
        
        # Prepare data for specific model types
        if self.config.model_type == "lstm_ae":
            X_train = self._prepare_lstm_data(X)
            X_val = self._prepare_lstm_data(X_val) if X_val is not None else None
        elif self.config.model_type == "cnn_ae":
            X_train = self._prepare_cnn_data(X)
            X_val = self._prepare_cnn_data(X_val) if X_val is not None else None
        else:
            X_train = X
        
        # Train model
        history = self.model.fit(
            X_train, X_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_val, X_val) if X_val is not None else None,
            callbacks=callback_list,
            verbose=1
        )
        
        return history
    
    def _prepare_lstm_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for LSTM model."""
        if X is None:
            return None
        
        timesteps = min(10, X.shape[1])
        features = X.shape[1] // timesteps
        
        # Reshape and pad if necessary
        reshaped_size = timesteps * features
        if X.shape[1] != reshaped_size:
            # Pad or truncate
            if X.shape[1] < reshaped_size:
                padding = np.zeros((X.shape[0], reshaped_size - X.shape[1]))
                X = np.hstack([X, padding])
            else:
                X = X[:, :reshaped_size]
        
        return X.reshape(X.shape[0], timesteps, features)
    
    def _prepare_cnn_data(self, X: np.ndarray) -> np.ndarray:
        """Prepare data for CNN model."""
        if X is None:
            return None
        
        img_size = int(np.sqrt(X.shape[1]))
        if img_size * img_size != X.shape[1]:
            img_size = int(np.sqrt(X.shape[1])) + 1
            padding_size = img_size * img_size - X.shape[1]
            
            # Pad data
            padding = np.zeros((X.shape[0], padding_size))
            X = np.hstack([X, padding])
        
        return X.reshape(X.shape[0], img_size, img_size, 1)
    
    def _compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores using reconstruction error."""
        # Prepare data for specific model types
        if self.config.model_type == "lstm_ae":
            X_prepared = self._prepare_lstm_data(X)
        elif self.config.model_type == "cnn_ae":
            X_prepared = self._prepare_cnn_data(X)
        else:
            X_prepared = X
        
        # Get predictions
        predictions = self.model.predict(X_prepared, verbose=0)
        
        # Reshape predictions back to original shape if needed
        if self.config.model_type in ["lstm_ae", "cnn_ae"]:
            predictions = predictions.reshape(predictions.shape[0], -1)
            if predictions.shape[1] > X.shape[1]:
                predictions = predictions[:, :X.shape[1]]
        
        # Compute reconstruction error
        mse = np.mean((X - predictions) ** 2, axis=1)
        
        return mse

class PyTorchAnomalyDetector(BaseDeepAnomalyDetector):
    """PyTorch-based deep anomaly detector."""
    
    def __init__(self, config: DeepLearningConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PyTorch-based detection")
        
        super().__init__(config)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
    
    def _build_model(self, input_dim: int) -> nn.Module:
        """Build PyTorch model."""
        if self.config.model_type == "autoencoder":
            return self._build_autoencoder(input_dim)
        elif self.config.model_type == "vae":
            return self._build_vae(input_dim)
        else:
            raise ValueError(f"Model type {self.config.model_type} not implemented for PyTorch")
    
    def _build_autoencoder(self, input_dim: int) -> nn.Module:
        """Build PyTorch autoencoder."""
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_neurons, latent_dim, dropout_rate):
                super().__init__()
                
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                
                for neurons in hidden_neurons[:len(hidden_neurons)//2]:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, neurons),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_dim = neurons
                
                encoder_layers.append(nn.Linear(prev_dim, latent_dim))
                self.encoder = nn.Sequential(*encoder_layers)
                
                # Decoder
                decoder_layers = []
                prev_dim = latent_dim
                
                for neurons in reversed(hidden_neurons[len(hidden_neurons)//2:]):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, neurons),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_dim = neurons
                
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        model = Autoencoder(
            input_dim, 
            self.config.hidden_neurons, 
            self.config.latent_dim, 
            self.config.dropout_rate
        )
        
        return model.to(self.device)
    
    def _build_vae(self, input_dim: int) -> nn.Module:
        """Build PyTorch VAE."""
        class VAE(nn.Module):
            def __init__(self, input_dim, hidden_neurons, latent_dim, dropout_rate):
                super().__init__()
                
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                
                for neurons in hidden_neurons[:len(hidden_neurons)//2]:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, neurons),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_dim = neurons
                
                self.encoder = nn.Sequential(*encoder_layers)
                
                # Latent space
                self.fc_mu = nn.Linear(prev_dim, latent_dim)
                self.fc_logvar = nn.Linear(prev_dim, latent_dim)
                
                # Decoder
                decoder_layers = []
                prev_dim = latent_dim
                
                for neurons in reversed(hidden_neurons[len(hidden_neurons)//2:]):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, neurons),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ])
                    prev_dim = neurons
                
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
            
            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_logvar(h)
            
            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                return self.decoder(z)
            
            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar
        
        model = VAE(
            input_dim, 
            self.config.hidden_neurons, 
            self.config.latent_dim, 
            self.config.dropout_rate
        )
        
        return model.to(self.device)
    
    def _train_model(self, X: np.ndarray, X_val: np.ndarray = None) -> Dict[str, List[float]]:
        """Train PyTorch model."""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                if self.config.model_type == "vae":
                    recon_batch, mu, logvar = self.model(batch_x)
                    loss = self._vae_loss(recon_batch, batch_y, mu, logvar)
                else:
                    recon_batch = self.model(batch_x)
                    loss = F.mse_loss(recon_batch, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                    if self.config.model_type == "vae":
                        val_recon, val_mu, val_logvar = self.model(X_val_tensor)
                        val_loss = self._vae_loss(val_recon, X_val_tensor, val_mu, val_logvar)
                    else:
                        val_recon = self.model(X_val_tensor)
                        val_loss = F.mse_loss(val_recon, X_val_tensor)
                    
                    history['val_loss'].append(val_loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return history
    
    def _vae_loss(self, recon_x, x, mu, logvar):
        """VAE loss function."""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.config.beta * kl_loss
    
    def _compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores using reconstruction error."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            if self.config.model_type == "vae":
                recon_x, _, _ = self.model(X_tensor)
            else:
                recon_x = self.model(X_tensor)
            
            # Compute reconstruction error
            mse = torch.mean((X_tensor - recon_x) ** 2, dim=1)
            
            return mse.cpu().numpy()

class DeepAnomalyDetector:
    """Main deep learning anomaly detector that handles both TensorFlow and PyTorch."""
    
    def __init__(self, config: DeepLearningConfig = None):
        """Initialize deep anomaly detector.
        
        Args:
            config: Deep learning configuration
        """
        self.config = config or DeepLearningConfig()
        
        # Choose framework
        if self.config.framework == "tensorflow":
            if not TF_AVAILABLE:
                logger.warning("TensorFlow not available, falling back to PyTorch")
                self.config.framework = "pytorch"
            else:
                self.detector = TensorFlowAnomalyDetector(self.config)
        
        if self.config.framework == "pytorch":
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available, falling back to TensorFlow")
                self.config.framework = "tensorflow"
                self.detector = TensorFlowAnomalyDetector(self.config)
            else:
                self.detector = PyTorchAnomalyDetector(self.config)
        
        if not hasattr(self, 'detector'):
            raise ImportError("Neither TensorFlow nor PyTorch is available")
        
        logger.info(f"Deep anomaly detector initialized with {self.config.framework} backend")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the deep anomaly detector."""
        return self.detector.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        return self.detector.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        return self.detector.decision_function(X)
    
    def get_config(self) -> DeepLearningConfig:
        """Get configuration."""
        return self.config
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "framework": self.config.framework,
            "model_type": self.config.model_type,
            "architecture": self.config.hidden_neurons,
            "latent_dim": self.config.latent_dim,
            "is_fitted": self.detector.is_fitted,
            "threshold": self.detector.threshold
        }
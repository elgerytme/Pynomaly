"""PyTorch-based deep learning adapter for anomaly detection."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from pynomaly.shared.protocols import DetectorProtocol

# Optional PyTorch imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    PYTORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    PYTORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


class AutoEncoderConfig(BaseModel):
    """Configuration for AutoEncoder neural network."""

    input_dim: int = Field(description="Input dimension (number of features)")
    hidden_dims: list[int] = Field(
        default=[64, 32, 16], description="Hidden layer dimensions"
    )
    latent_dim: int = Field(default=8, description="Latent space dimension")
    activation: str = Field(default="relu", description="Activation function")
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.9, description="Dropout rate")
    batch_norm: bool = Field(default=True, description="Use batch normalization")

    # Training parameters
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    epochs: int = Field(default=100, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size")
    early_stopping_patience: int = Field(
        default=10, ge=1, description="Early stopping patience"
    )

    # Anomaly detection parameters
    contamination: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Expected contamination rate"
    )
    threshold_strategy: str = Field(
        default="percentile", description="Threshold selection strategy"
    )


class VAEConfig(BaseModel):
    """Configuration for Variational AutoEncoder."""

    input_dim: int = Field(description="Input dimension")
    encoder_dims: list[int] = Field(default=[128, 64], description="Encoder dimensions")
    latent_dim: int = Field(default=16, description="Latent space dimension")
    decoder_dims: list[int] = Field(default=[64, 128], description="Decoder dimensions")

    # VAE-specific parameters
    beta: float = Field(
        default=1.0, ge=0.0, description="Beta parameter for KL divergence"
    )
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    epochs: int = Field(default=150, ge=1, description="Training epochs")
    batch_size: int = Field(default=64, ge=1, description="Batch size")

    # Anomaly detection
    contamination: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Contamination rate"
    )
    use_reconstruction_error: bool = Field(
        default=True, description="Use reconstruction error for detection"
    )
    use_kl_divergence: bool = Field(
        default=True, description="Include KL divergence in anomaly score"
    )


class LSTMConfig(BaseModel):
    """Configuration for LSTM-based anomaly detection."""

    input_dim: int = Field(description="Input feature dimension")
    sequence_length: int = Field(
        default=10, ge=1, description="Sequence length for time series"
    )
    hidden_dim: int = Field(default=64, ge=1, description="LSTM hidden dimension")
    num_layers: int = Field(default=2, ge=1, description="Number of LSTM layers")
    dropout: float = Field(default=0.2, ge=0.0, le=0.9, description="Dropout rate")

    # Training parameters
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    epochs: int = Field(default=100, ge=1, description="Training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size")

    # Detection parameters
    contamination: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Contamination rate"
    )
    prediction_window: int = Field(
        default=1, ge=1, description="Prediction window size"
    )


if PYTORCH_AVAILABLE:

    class AutoEncoder(nn.Module):
        """AutoEncoder neural network for anomaly detection."""

        def __init__(self, config: AutoEncoderConfig):
            super().__init__()
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch is required for AutoEncoder")

            self.config = config

            # Build encoder
            encoder_layers = []
            in_dim = config.input_dim

            for hidden_dim in config.hidden_dims:
                encoder_layers.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        (
                            nn.BatchNorm1d(hidden_dim)
                            if config.batch_norm
                            else nn.Identity()
                        ),
                        self._get_activation(config.activation),
                        nn.Dropout(config.dropout_rate),
                    ]
                )
                in_dim = hidden_dim

            # Latent layer
            encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            # Build decoder
            decoder_layers = []
            in_dim = config.latent_dim

            for hidden_dim in reversed(config.hidden_dims):
                decoder_layers.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        (
                            nn.BatchNorm1d(hidden_dim)
                            if config.batch_norm
                            else nn.Identity()
                        ),
                        self._get_activation(config.activation),
                        nn.Dropout(config.dropout_rate),
                    ]
                )
                in_dim = hidden_dim

            # Output layer
            decoder_layers.append(nn.Linear(in_dim, config.input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def _get_activation(self, activation: str):
            """Get activation function."""
            activations = {
                "relu": nn.ReLU(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "leaky_relu": nn.LeakyReLU(),
                "elu": nn.ELU(),
            }
            return activations.get(activation, nn.ReLU())

        def forward(self, x):
            """Forward pass."""
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

        def encode(self, x):
            """Encode input to latent space."""
            return self.encoder(x)

else:
    # Fallback class when PyTorch is not available
    class AutoEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for AutoEncoder. Install with: pip install torch"
            )


if PYTORCH_AVAILABLE:

    class VAE(nn.Module):
        """Variational AutoEncoder for anomaly detection."""

        def __init__(self, config: VAEConfig):
            super().__init__()
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch is required for VAE")

            self.config = config

            # Encoder
            encoder_layers = []
            in_dim = config.input_dim

            for hidden_dim in config.encoder_dims:
                encoder_layers.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim),
                    ]
                )
                in_dim = hidden_dim

            self.encoder = nn.Sequential(*encoder_layers)
            self.fc_mu = nn.Linear(in_dim, config.latent_dim)
            self.fc_logvar = nn.Linear(in_dim, config.latent_dim)

            # Decoder
            decoder_layers = []
            in_dim = config.latent_dim

            for hidden_dim in config.decoder_dims:
                decoder_layers.extend(
                    [
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim),
                    ]
                )
                in_dim = hidden_dim

            decoder_layers.append(nn.Linear(in_dim, config.input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def encode(self, x):
            """Encode input to latent distribution parameters."""
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            """Reparameterization trick."""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            """Decode latent representation."""
            return self.decoder(z)

        def forward(self, x):
            """Forward pass."""
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar

else:
    # Fallback class when PyTorch is not available
    class VAE:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for VAE. Install with: pip install torch"
            )


if PYTORCH_AVAILABLE:

    class LSTMAutoEncoder(nn.Module):
        """LSTM-based AutoEncoder for time series anomaly detection."""

        def __init__(self, config: LSTMConfig):
            super().__init__()
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch is required for LSTM")

            self.config = config

            # Encoder LSTM
            self.encoder_lstm = nn.LSTM(
                config.input_dim,
                config.hidden_dim,
                config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0,
            )

            # Decoder LSTM
            self.decoder_lstm = nn.LSTM(
                config.hidden_dim,
                config.hidden_dim,
                config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0,
            )

            # Output layer
            self.output_layer = nn.Linear(config.hidden_dim, config.input_dim)
            self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Forward pass for sequence reconstruction."""
        batch_size, seq_len, _ = x.shape

        # Encode
        encoded, (hidden, cell) = self.encoder_lstm(x)

        # Use the last hidden state for decoding
        context = encoded[:, -1:, :]  # (batch_size, 1, hidden_dim)

        # Decode
        decoder_input = context.repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(decoder_input, (hidden, cell))

        # Output projection
        output = self.output_layer(self.dropout(decoded))

        return output

else:
    # Fallback classes when PyTorch is not available
    class LSTMAutoEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for LSTMAutoEncoder. Install with: pip install torch"
            )


class PyTorchAdapter(DetectorProtocol):
    """PyTorch-based deep learning adapter for anomaly detection."""

    def __init__(
        self,
        algorithm_name: str = "autoencoder",
        name: str | None = None,
        contamination_rate: float = 0.1,
        device: str | None = None,
        model_config: dict[str, Any] | None = None,
        random_state: int | None = None,
        algorithm: str | None = None,  # Backward compatibility
    ):
        """Initialize PyTorch adapter.

        Args:
            algorithm_name: Deep learning algorithm ('autoencoder', 'vae', 'lstm')
            name: Name for this detector instance
            contamination_rate: Expected contamination rate
            device: PyTorch device ('cpu', 'cuda', 'auto')
            model_config: Model-specific configuration
            random_state: Random seed for reproducibility
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for PyTorchAdapter. "
                "Install with: pip install torch torchvision"
            )

        # Handle backward compatibility
        if algorithm is not None:
            algorithm_name = algorithm

        self.algorithm = algorithm_name
        self._name = name or f"PyTorch_{algorithm_name}"
        self._contamination_rate = contamination_rate
        self.model_config = model_config or {}
        self.random_state = random_state

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # Set random seeds
        if random_state is not None:
            torch.manual_seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_state)

        # Model components
        self.model = None
        self.scaler = None
        self.threshold = None
        self.is_trained = False

        logger.info(f"Initialized PyTorchAdapter with {algorithm} on {self.device}")

    def train(self, X: np.ndarray, y: np.ndarray | None = None) -> PyTorchAdapter:
        return self.fit(X, y)

    def save(self, path: str | Path):
        self.save_model(path)

    def load(self, path: str | Path):
        self.load_model(path)

    @property
    def name(self) -> str:
        """Get the name of the detector."""
        return self._name

    @property
    def contamination_rate(self) -> float:
        """Get the contamination rate."""
        return self._contamination_rate

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self.is_trained

    @property
    def parameters(self) -> dict[str, Any]:
        """Get the current parameters of the detector."""
        return self.algorithm_params

    @property
    def algorithm_name(self) -> str:
        """Get algorithm name."""
        return f"pytorch_{self.algorithm}"

    @property
    def algorithm_params(self) -> dict[str, Any]:
        """Get algorithm parameters."""
        return {
            "algorithm": self.algorithm,
            "device": str(self.device),
            "model_config": self.model_config,
            "random_state": self.random_state,
            "contamination_rate": self._contamination_rate,
        }

    def get_params(self) -> dict[str, Any]:
        """Get parameters of the detector."""
        return self.algorithm_params

    def set_params(self, **params: Any) -> None:
        """Set parameters of the detector."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.model_config:
                self.model_config[key] = value

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> PyTorchAdapter:
        """Train the deep learning model."""
        try:
            # Data preprocessing
            X_tensor = self._preprocess_data(X)

            # Create model based on algorithm
            if self.algorithm == "autoencoder":
                self._fit_autoencoder(X_tensor)
            elif self.algorithm == "vae":
                self._fit_vae(X_tensor)
            elif self.algorithm == "lstm":
                self._fit_lstm(X_tensor)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            # Calculate anomaly threshold
            self._calculate_threshold(X_tensor)

            self.is_trained = True
            logger.info(f"PyTorch {self.algorithm} model trained successfully")

            return self

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Failed to train PyTorch model: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for anomaly, 0 for normal)."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        scores = self.decision_function(X)
        return (scores > self.threshold).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        X_tensor = self._preprocess_data(X, fit_scaler=False)

        self.model.eval()
        with torch.no_grad():
            if self.algorithm == "autoencoder":
                scores = self._autoencoder_scores(X_tensor)
            elif self.algorithm == "vae":
                scores = self._vae_scores(X_tensor)
            elif self.algorithm == "lstm":
                scores = self._lstm_scores(X_tensor)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return scores.cpu().numpy()

    def _preprocess_data(self, X: np.ndarray, fit_scaler: bool = True) -> torch.Tensor:
        """Preprocess data for neural network."""
        # Handle pandas DataFrame
        if hasattr(X, "values"):
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

        # Convert to tensor
        return torch.FloatTensor(X_scaled).to(self.device)

    def _fit_autoencoder(self, X_tensor: torch.Tensor):
        """Train AutoEncoder model."""
        # Create configuration with contamination
        model_config = self.model_config.copy()
        model_config["contamination"] = model_config.get("contamination", self._contamination_rate)
        config = AutoEncoderConfig(input_dim=X_tensor.shape[1], **model_config)

        # Initialize model
        self.model = AutoEncoder(config).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        # Create data loader
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(config.epochs):
            epoch_loss = 0.0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch + 1}/{config.epochs}, Loss: {avg_loss:.6f}")

    def _fit_vae(self, X_tensor: torch.Tensor):
        """Train VAE model."""
        model_config = self.model_config.copy()
        model_config["contamination"] = model_config.get("contamination", self._contamination_rate)
        config = VAEConfig(input_dim=X_tensor.shape[1], **model_config)

        self.model = VAE(config).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(config.epochs):
            epoch_loss = 0.0

            for batch_x, _ in dataloader:
                optimizer.zero_grad()

                # Forward pass
                recon, mu, logvar = self.model(batch_x)

                # VAE loss
                recon_loss = nn.functional.mse_loss(recon, batch_x, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + config.beta * kl_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.debug(
                    f"VAE Epoch {epoch + 1}/{config.epochs}, Loss: {avg_loss:.6f}"
                )

    def _fit_lstm(self, X_tensor: torch.Tensor):
        """Train LSTM model for time series."""
        model_config = self.model_config.copy()
        model_config["contamination"] = model_config.get("contamination", self._contamination_rate)
        config = LSTMConfig(input_dim=X_tensor.shape[1], **model_config)

        # Reshape for time series
        X_sequences = self._create_sequences(X_tensor, config.sequence_length)

        self.model = LSTMAutoEncoder(config).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_sequences, X_sequences)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(config.epochs):
            epoch_loss = 0.0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()

                # Forward pass
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_y)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.debug(
                    f"LSTM Epoch {epoch + 1}/{config.epochs}, Loss: {avg_loss:.6f}"
                )

    def _create_sequences(self, X: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """Create sequences for LSTM training."""
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i : i + sequence_length])
        return torch.stack(sequences)

    def _autoencoder_scores(self, X_tensor: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction error for AutoEncoder."""
        reconstructed = self.model(X_tensor)
        scores = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
        return scores

    def _vae_scores(self, X_tensor: torch.Tensor) -> torch.Tensor:
        """Calculate anomaly scores for VAE."""
        recon, mu, logvar = self.model(X_tensor)

        # Reconstruction error
        recon_error = torch.mean((X_tensor - recon) ** 2, dim=1)

        # KL divergence (optional)
        if self.model.config.use_kl_divergence:
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            scores = recon_error + 0.1 * kl_div
        else:
            scores = recon_error

        return scores

    def _lstm_scores(self, X_tensor: torch.Tensor) -> torch.Tensor:
        """Calculate prediction error for LSTM."""
        config = self.model.config

        # Create sequences
        X_sequences = self._create_sequences(X_tensor, config.sequence_length)

        # Get reconstruction
        reconstructed = self.model(X_sequences)

        # Calculate reconstruction error
        scores = torch.mean((X_sequences - reconstructed) ** 2, dim=(1, 2))

        # Pad scores to match original length
        padding = torch.zeros(config.sequence_length - 1, device=self.device)
        scores = torch.cat([padding, scores])

        return scores

    def _calculate_threshold(self, X_tensor: torch.Tensor):
        """Calculate anomaly threshold based on training data."""
        # Calculate scores directly without using decision_function to avoid circular dependency
        self.model.eval()
        with torch.no_grad():
            if self.algorithm == "autoencoder":
                scores = self._autoencoder_scores(X_tensor)
            elif self.algorithm == "vae":
                scores = self._vae_scores(X_tensor)
            elif self.algorithm == "lstm":
                scores = self._lstm_scores(X_tensor)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

        scores_np = scores.cpu().numpy()

        if hasattr(self.model, "config"):
            contamination = self.model.config.contamination
        else:
            contamination = self._contamination_rate

        # Use percentile-based threshold
        self.threshold = np.percentile(scores_np, (1 - contamination) * 100)

        logger.info(f"Calculated anomaly threshold: {self.threshold:.6f}")

    def save_model(self, path: str | Path):
        """Save trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get complete model config including input_dim
        complete_config = self.model_config.copy()
        if hasattr(self.model, 'config'):
            complete_config['input_dim'] = self.model.config.input_dim

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "model_config": complete_config,
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "scaler_params": (
                {
                    "mean_": self.scaler.mean_.tolist(),
                    "scale_": self.scaler.scale_.tolist(),
                }
                if self.scaler
                else None
            ),
            "device": str(self.device),
            "random_state": self.random_state,
        }

        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str | Path):
        """Load trained model."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load with weights_only=False for compatibility with older PyTorch versions
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Restore configuration
        self.algorithm = checkpoint["algorithm"]
        self.model_config = checkpoint["model_config"]
        self.threshold = checkpoint["threshold"]
        self.random_state = checkpoint.get("random_state")

        # Restore scaler
        if checkpoint.get("scaler_params"):
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(checkpoint["scaler_params"]["mean_"])
            self.scaler.scale_ = np.array(checkpoint["scaler_params"]["scale_"])

        # Recreate and load model
        if self.algorithm == "autoencoder":
            config = AutoEncoderConfig(**self.model_config)
            self.model = AutoEncoder(config).to(self.device)
        elif self.algorithm == "vae":
            config = VAEConfig(**self.model_config)
            self.model = VAE(config).to(self.device)
        elif self.algorithm == "lstm":
            config = LSTMConfig(**self.model_config)
            self.model = LSTMAutoEncoder(config).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_trained = True

        logger.info(f"Model loaded from {path}")

    async def async_fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> PyTorchAdapter:
        """Asynchronous training (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fit, X, y)

    async def async_predict(self, X: np.ndarray) -> np.ndarray:
        """Asynchronous prediction."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, X)

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        info = {
            "algorithm": self.algorithm,
            "device": str(self.device),
            "is_trained": self.is_trained,
            "model_config": self.model_config,
            "threshold": self.threshold,
        }

        if self.model and self.is_trained:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )

            info.update(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_architecture": str(self.model),
                }
            )

        return info

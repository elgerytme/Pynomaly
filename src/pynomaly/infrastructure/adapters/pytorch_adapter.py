"""PyTorch adapter for deep learning anomaly detection models.

This module provides PyTorch-based deep learning models for anomaly detection,
including autoencoders, VAEs, GANs, and custom architectures.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Create dummy classes to avoid import errors
    class F:
        @staticmethod
        def mse_loss(*args, **kwargs):
            return 0

        @staticmethod
        def cosine_similarity(*args, **kwargs):
            return 0

    class nn:
        class Module:
            def __init__(self):
                pass

            def to(self, device):
                return self

            def train(self):
                pass

            def eval(self):
                pass

            def parameters(self):
                return []

        @staticmethod
        def Linear(*args, **kwargs):
            return nn.Module()

        @staticmethod
        def ReLU(*args, **kwargs):
            return nn.Module()

        @staticmethod
        def Tanh(*args, **kwargs):
            return nn.Module()

        @staticmethod
        def BatchNorm1d(*args, **kwargs):
            return nn.Module()

        @staticmethod
        def Dropout(*args, **kwargs):
            return nn.Module()

        @staticmethod
        def Softmax(*args, **kwargs):
            return nn.Module()

        @staticmethod
        def Sequential(*args, **kwargs):
            return nn.Module()

        class functional:
            @staticmethod
            def mse_loss(*args, **kwargs):
                return 0

            @staticmethod
            def cosine_similarity(*args, **kwargs):
                return 0

    class torch:
        @staticmethod
        def device(device_str):
            return "cpu"

        @staticmethod
        def cuda():
            class CUDA:
                @staticmethod
                def is_available():
                    return False

            return CUDA()

        @staticmethod
        def FloatTensor(*args):
            return None

        @staticmethod
        def zeros(*args):
            return None

        @staticmethod
        def eye(*args):
            return None

        @staticmethod
        def randn_like(*args):
            return None

        @staticmethod
        def exp(*args):
            return None

        @staticmethod
        def sum(*args, **kwargs):
            return None

        @staticmethod
        def mean(*args, **kwargs):
            return None

        @staticmethod
        def inverse(*args):
            return None

        @staticmethod
        def logdet(*args):
            return None


from pynomaly.domain.entities import Dataset, DetectionResult
from pynomaly.domain.exceptions import (
    AdapterError,
    AlgorithmNotFoundError,
)
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    BaseAnomalyModelBase = nn.Module
else:
    BaseAnomalyModelBase = object


class BaseAnomalyModel(BaseAnomalyModelBase):
    """Base class for PyTorch anomaly detection models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Model output tensor
        """
        # Default implementation - should be overridden by subclasses
        return x

    def loss_function(
        self, x: torch.Tensor, recon: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Calculate loss function for anomaly detection.

        Args:
            x: Original input tensor
            recon: Reconstructed tensor
            **kwargs: Additional arguments

        Returns:
            Loss tensor
        """
        # Default MSE loss for reconstruction
        return torch.nn.functional.mse_loss(recon, x, reduction="none")

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate anomaly scores for input data.

        Args:
            x: Input tensor

        Returns:
            Anomaly scores tensor
        """
        with torch.no_grad():
            # Forward pass to get reconstruction
            recon = self.forward(x)
            # Calculate reconstruction error as anomaly score
            loss = self.loss_function(x, recon)
            # Return mean loss across features for each sample
            return torch.mean(loss, dim=1)


class AutoEncoder(BaseAnomalyModel):
    """Vanilla Autoencoder for anomaly detection."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)]
            )
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)]
            )
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def loss_function(
        self, x: torch.Tensor, recon: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return F.mse_loss(recon, x, reduction="mean")

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(x)
            scores = torch.mean((x - recon) ** 2, dim=1)
        return scores


class VariationalAutoEncoder(BaseAnomalyModel):
    """Variational Autoencoder (VAE) for anomaly detection."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)]
            )
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)]
            )
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Total loss
        return recon_loss + beta * kl_loss

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            recon_error = torch.mean((x - recon) ** 2, dim=1)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            scores = recon_error + 0.1 * kl_div  # Weight can be adjusted
        return scores


class DeepSVDD(BaseAnomalyModel):
    """Deep Support Vector Data Description for anomaly detection."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()

        # Network layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, h_dim), nn.ReLU(), nn.BatchNorm1d(h_dim)]
            )
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))

        self.network = nn.Sequential(*layers)
        self.center = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def init_center(self, train_loader: DataLoader, eps: float = 0.1):
        """Initialize hypersphere center using training data."""
        n_samples = 0
        self.center = torch.zeros(self.network[-1].out_features)

        with torch.no_grad():
            for data, _ in train_loader:
                outputs = self.forward(data)
                n_samples += outputs.shape[0]
                self.center += torch.sum(outputs, dim=0)

        self.center /= n_samples

        # Avoid center too close to zero
        self.center[(torch.abs(self.center) < eps) & (self.center < 0)] = -eps
        self.center[(torch.abs(self.center) < eps) & (self.center >= 0)] = eps

    def loss_function(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.forward(x)
        dist = torch.sum((outputs - self.center) ** 2, dim=1)
        return torch.mean(dist)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.forward(x)
            scores = torch.sum((outputs - self.center) ** 2, dim=1)
        return scores


class DAGMM(BaseAnomalyModel):
    """Deep Autoencoding Gaussian Mixture Model for anomaly detection."""

    def __init__(
        self, input_dim: int, hidden_dims: list[int], latent_dim: int, n_gmm: int = 4
    ):
        super().__init__()

        # Compression network (autoencoder)
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.Tanh()])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.Tanh()])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Estimation network
        self.estimation = nn.Sequential(
            nn.Linear(latent_dim + 2, 10),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(10, n_gmm),
            nn.Softmax(dim=1),
        )

        self.n_gmm = n_gmm
        self.register_buffer("phi", torch.zeros(n_gmm))
        self.register_buffer("mu", torch.zeros(n_gmm, latent_dim + 2))
        self.register_buffer("cov", torch.zeros(n_gmm, latent_dim + 2, latent_dim + 2))

    def forward(self, x: torch.Tensor):
        # Compression
        z = self.encoder(x)
        x_recon = self.decoder(z)

        # Reconstruction error features
        recon_error = torch.sum((x - x_recon) ** 2, dim=1, keepdim=True)
        recon_cosine = F.cosine_similarity(x, x_recon, dim=1).unsqueeze(1)

        # Concatenate z with error features
        z_error = torch.cat([z, recon_error, recon_cosine], dim=1)

        # Estimation
        gamma = self.estimation(z_error)

        return x_recon, z_error, gamma

    def compute_gmm_params(self, z: torch.Tensor, gamma: torch.Tensor):
        """Update GMM parameters."""
        n = gamma.size(0)

        # Phi: mixture weights
        self.phi = torch.sum(gamma, dim=0) / n

        # Mu: means
        self.mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / torch.sum(
            gamma, dim=0
        ).unsqueeze(-1)

        # Cov: covariances
        z_centered = z.unsqueeze(1) - self.mu.unsqueeze(0)
        self.cov = torch.sum(
            gamma.unsqueeze(-1).unsqueeze(-1)
            * z_centered.unsqueeze(-1)
            * z_centered.unsqueeze(-2),
            dim=0,
        ) / torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        # Add small identity for numerical stability
        self.cov += 0.005 * torch.eye(self.cov.size(1))

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        z: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        # Reconstruction loss
        recon_loss = torch.mean(torch.sum((x - x_recon) ** 2, dim=1))

        # Energy loss (negative log-likelihood of GMM)
        energy_loss = torch.mean(torch.sum(gamma * self.energy(z), dim=1))

        return recon_loss + 0.1 * energy_loss

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """Compute energy for each sample under GMM."""
        n_samples = z.size(0)
        z.size(1)

        # Compute log probabilities for each component
        energy = torch.zeros(n_samples, self.n_gmm)

        for k in range(self.n_gmm):
            diff = z - self.mu[k].unsqueeze(0)
            inv_cov = torch.inverse(self.cov[k])
            exp_term = -0.5 * torch.sum(
                torch.sum(diff.unsqueeze(-1) * inv_cov.unsqueeze(0), dim=-2) * diff,
                dim=-1,
            )
            det_term = -0.5 * torch.logdet(2 * np.pi * self.cov[k])
            energy[:, k] = exp_term + det_term

        return -energy  # Negative log-likelihood

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, z, gamma = self.forward(x)
            scores = torch.sum(gamma * self.energy(z), dim=1)
        return scores


class LSTMAutoEncoder(BaseAnomalyModel):
    """LSTM-based AutoEncoder for time series anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        sequence_length: int = 10,
        dropout: float = 0.2,
    ):
        """Initialize LSTM AutoEncoder.
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            sequence_length: Length of input sequences
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def loss_function(
        self, x: torch.Tensor, recon: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Calculate MSE loss for sequence reconstruction."""
        return F.mse_loss(recon, x, reduction="mean")

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate anomaly scores based on reconstruction error."""
        with torch.no_grad():
            recon = self.forward(x)
            # Calculate reconstruction error per sequence
            scores = torch.mean(torch.mean((x - recon) ** 2, dim=2), dim=1)
        return scores


class PyTorchAdapter:
    """Adapter for PyTorch-based deep learning anomaly detection models.

    This adapter implements DetectorProtocol and maintains clean architecture
    by keeping infrastructure concerns separate from domain logic.
    """

    _algorithm_map = {
        "AutoEncoder": AutoEncoder,
        "VAE": VariationalAutoEncoder,
        "DeepSVDD": DeepSVDD,
        "DAGMM": DAGMM,
        "LSTMAutoEncoder": LSTMAutoEncoder,
    }

    def __init__(
        self,
        algorithm_name: str,
        name: str | None = None,
        contamination_rate: ContaminationRate | None = None,
        **kwargs: Any,
    ):
        """Initialize PyTorch adapter.

        Args:
            algorithm_name: Name of the PyTorch algorithm
            name: Optional custom name for the detector
            contamination_rate: Expected contamination rate
            **kwargs: Algorithm-specific parameters
        """
        # Check PyTorch availability
        if not TORCH_AVAILABLE:
            raise AdapterError(
                "PyTorch is not available. Please install PyTorch to use deep learning models."
            )

        # Validate algorithm
        if algorithm_name not in self._algorithm_map:
            available = ", ".join(self._algorithm_map.keys())
            raise AlgorithmNotFoundError(
                f"Algorithm '{algorithm_name}' not found. "
                f"Available algorithms: {available}"
            )

        # Infrastructure state (no domain entity composition)
        self._name = name or f"PyTorch_{algorithm_name}"
        self._algorithm_name = algorithm_name
        self._contamination_rate = contamination_rate or ContaminationRate(0.1)
        self._parameters = kwargs
        self._is_fitted = False

        self._model: BaseAnomalyModel | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_class = self._algorithm_map[algorithm_name]

    # DetectorProtocol properties
    @property
    def name(self) -> str:
        """Get the name of the detector."""
        return self._name

    @property
    def contamination_rate(self) -> ContaminationRate:
        """Get the contamination rate."""
        return self._contamination_rate

    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._is_fitted

    @property
    def parameters(self) -> dict[str, Any]:
        """Get the current parameters of the detector."""
        return self._parameters

    @property
    def algorithm_name(self) -> str:
        """Get the algorithm name."""
        return self._algorithm_name

    @property
    def supports_streaming(self) -> bool:
        """Whether this detector supports streaming detection."""
        return False

    @property
    def requires_fitting(self) -> bool:
        """Whether this detector requires fitting before detection."""
        return True

    def fit(self, dataset: Dataset) -> None:
        """Train the deep learning model on the dataset.

        Args:
            dataset: Training dataset
        """
        try:
            # Prepare data
            X_train = self._prepare_data(dataset)
            input_dim = X_train.shape[1]

            # Create model with known input dimension
            params = self.parameters.copy()
            hidden_dims = params.get("hidden_dims", [128, 64, 32])
            latent_dim = params.get("latent_dim", 16)

            if self.algorithm_name == "DAGMM":
                n_gmm = params.get("n_gmm", 4)
                self._model = self._model_class(
                    input_dim, hidden_dims, latent_dim, n_gmm
                )
            elif self.algorithm_name == "LSTMAutoEncoder":
                hidden_dim = params.get("hidden_dim", 64)
                num_layers = params.get("num_layers", 2)
                sequence_length = params.get("sequence_length", 10)
                dropout = params.get("dropout", 0.2)
                self._model = self._model_class(
                    input_dim, hidden_dim, num_layers, sequence_length, dropout
                )
            else:
                self._model = self._model_class(input_dim, hidden_dims, latent_dim)

            self._model.to(self._device)

            # Training parameters
            batch_size = params.get("batch_size", 256)
            epochs = params.get("epochs", 100)
            learning_rate = params.get("learning_rate", 0.001)

            # Create data loader
            tensor_x = torch.FloatTensor(X_train).to(self._device)
            train_dataset = TensorDataset(
                tensor_x, tensor_x
            )  # Using input as target for reconstruction
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            # Optimizer
            optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

            # Special initialization for DeepSVDD
            if isinstance(self._model, DeepSVDD):
                self._model.init_center(train_loader)

            # Training loop
            self._model.train()
            for epoch in range(epochs):
                total_loss = 0.0

                for batch_x, _ in train_loader:
                    optimizer.zero_grad()

                    if isinstance(self._model, AutoEncoder):
                        recon = self._model(batch_x)
                        loss = self._model.loss_function(batch_x, recon)

                    elif isinstance(self._model, VariationalAutoEncoder):
                        recon, mu, logvar = self._model(batch_x)
                        beta = params.get("beta", 1.0)
                        loss = self._model.loss_function(
                            batch_x, recon, mu, logvar, beta
                        )

                    elif isinstance(self._model, DeepSVDD):
                        loss = self._model.loss_function(batch_x)

                    elif isinstance(self._model, DAGMM):
                        recon, z, gamma = self._model(batch_x)
                        self._model.compute_gmm_params(z, gamma)
                        loss = self._model.loss_function(batch_x, recon, z, gamma)

                    elif isinstance(self._model, LSTMAutoEncoder):
                        # For LSTM, need to reshape data to sequences
                        sequence_length = self._model.sequence_length
                        if batch_x.shape[0] >= sequence_length:
                            # Create sequences from batch
                            sequences = self._create_sequences(batch_x, sequence_length)
                            recon = self._model(sequences)
                            loss = self._model.loss_function(sequences, recon)
                        else:
                            # Skip if batch is too small
                            continue

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / len(train_loader)
                    logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            self._is_fitted = True
            logger.info(f"Successfully trained PyTorch {self.algorithm_name}")

        except Exception as e:
            raise AdapterError(f"Failed to train PyTorch model: {e}")

    def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies in the dataset.

        Args:
            dataset: The dataset to analyze

        Returns:
            Detection result containing anomalies, scores, and labels
        """
        return self.predict(dataset)

    def score(self, dataset: Dataset) -> list[AnomalyScore]:
        """Calculate anomaly scores for the dataset.

        Args:
            dataset: The dataset to score

        Returns:
            List of anomaly scores
        """
        if not self.is_fitted:
            raise AdapterError("Model must be fitted before scoring")

        try:
            # Prepare data
            X_test = self._prepare_data(dataset)
            tensor_x = torch.FloatTensor(X_test).to(self._device)

            # Get anomaly scores
            self._model.eval()
            with torch.no_grad():
                scores = self._model.anomaly_score(tensor_x)
                scores = scores.cpu().numpy()

            # Normalize scores
            min_score = np.min(scores)
            max_score = np.max(scores)

            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)

            # Calculate threshold for confidence
            contamination = self.parameters.get("contamination", 0.1)
            threshold = np.percentile(normalized_scores, (1 - contamination) * 100)

            # Create anomaly scores
            return [
                AnomalyScore(
                    value=float(score),
                    confidence=self._calculate_confidence(score, threshold),
                )
                for score in normalized_scores
            ]

        except Exception as e:
            raise AdapterError(f"Failed to score with PyTorch model: {e}")

    def fit_detect(self, dataset: Dataset) -> DetectionResult:
        """Fit the detector and detect anomalies in one step.

        Args:
            dataset: The dataset to fit and analyze

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
        return self.parameters

    def set_params(self, **params: Any) -> None:
        """Set parameters of the detector.

        Args:
            **params: Parameters to set
        """
        self._parameters.update(params)
        # Re-initialize if algorithm changed
        if "algorithm_name" in params:
            self._algorithm_name = params["algorithm_name"]
            if params["algorithm_name"] not in self._algorithm_map:
                available = ", ".join(self._algorithm_map.keys())
                raise AlgorithmNotFoundError(
                    f"Algorithm '{params['algorithm_name']}' not found. "
                    f"Available algorithms: {available}"
                )
            self._model_class = self._algorithm_map[params["algorithm_name"]]
            # Reset fitting status if algorithm changed
            self._is_fitted = False
            self._model = None

    def predict(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies using the trained model.

        Args:
            dataset: Dataset to analyze

        Returns:
            Detection results with anomaly scores and labels
        """
        if not self.is_fitted or self._model is None:
            raise AdapterError("Model must be fitted before prediction")

        try:
            # Prepare data
            X_test = self._prepare_data(dataset)
            tensor_x = torch.FloatTensor(X_test).to(self._device)

            # Get anomaly scores
            self._model.eval()
            with torch.no_grad():
                scores = self._model.anomaly_score(tensor_x)
                scores = scores.cpu().numpy()

            # Normalize scores
            min_score = np.min(scores)
            max_score = np.max(scores)

            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.zeros_like(scores)

            # Calculate threshold
            contamination = self.parameters.get("contamination", 0.1)
            threshold = np.percentile(normalized_scores, (1 - contamination) * 100)

            # Create labels
            labels = (normalized_scores > threshold).astype(int)

            # Create anomaly scores
            anomaly_scores = [
                AnomalyScore(
                    value=float(score),
                    confidence=self._calculate_confidence(score, threshold),
                )
                for score in normalized_scores
            ]

            return DetectionResult(
                detector_id=self.name,
                dataset_name=dataset.name,
                scores=anomaly_scores,
                labels=labels,
                threshold=float(threshold),
                execution_time_ms=0.0,  # Could add timing if needed
                anomalies=[],  # Could create Anomaly objects for detected points
                metadata={
                    "algorithm": self.algorithm_name,
                    "threshold": float(threshold),
                    "n_anomalies": int(np.sum(labels)),
                    "contamination_rate": float(np.sum(labels) / len(labels)),
                    "model_type": "deep_learning",
                    "device": str(self._device),
                },
            )

        except Exception as e:
            raise AdapterError(f"Failed to predict with PyTorch model: {e}")

    def _prepare_data(self, dataset: Dataset) -> np.ndarray:
        """Prepare data for PyTorch model.

        Args:
            dataset: Input dataset

        Returns:
            Numpy array of features

        Raises:
            AdapterError: If data preparation fails
        """
        try:
            df = dataset.data

            # Validate dataset
            if df is None or df.empty:
                raise AdapterError("Dataset is empty or None")

            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

            # Remove target column if present
            if dataset.target_column and dataset.target_column in numeric_cols:
                numeric_cols.remove(dataset.target_column)

            if not numeric_cols:
                raise AdapterError("No numeric features found in dataset")

            # Extract features and handle missing values
            X = df[numeric_cols].values

            # Validate data shape
            if X.shape[0] == 0:
                raise AdapterError("Dataset has no rows")
            if X.shape[1] == 0:
                raise AdapterError("Dataset has no features")

            # Check for invalid values
            if not np.isfinite(X).all():
                # Simple imputation - replace NaN/inf with column mean
                for col_idx in range(X.shape[1]):
                    col_data = X[:, col_idx]
                    finite_mask = np.isfinite(col_data)

                    if not finite_mask.any():
                        # If entire column is non-finite, fill with zeros
                        X[:, col_idx] = 0.0
                    else:
                        # Replace non-finite values with column mean
                        col_mean = np.mean(col_data[finite_mask])
                        X[~finite_mask, col_idx] = col_mean

            # Standardize features
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)

            # Avoid division by zero
            std = np.where(std == 0, 1e-6, std)
            X = (X - mean) / std

            # Final validation
            if not np.isfinite(X).all():
                raise AdapterError(
                    "Data still contains non-finite values after preprocessing"
                )

            return X

        except Exception as e:
            if isinstance(e, AdapterError):
                raise
            raise AdapterError(f"Data preparation failed: {e}")

    def _calculate_confidence(self, score: float, threshold: float) -> float:
        """Calculate confidence score for anomaly.

        Args:
            score: Anomaly score
            threshold: Detection threshold

        Returns:
            Confidence value between 0 and 1
        """
        if score <= threshold:
            return 1.0 - (score / threshold) * 0.5
        else:
            return 0.5 + min((score - threshold) / threshold * 0.5, 0.5)

    def _create_sequences(self, data: torch.Tensor, sequence_length: int) -> torch.Tensor:
        """Create sequences for LSTM training.

        Args:
            data: Input data tensor of shape (batch_size, features)
            sequence_length: Length of sequences to create

        Returns:
            Tensor of shape (num_sequences, sequence_length, features)
        """
        if data.shape[0] < sequence_length:
            raise ValueError(f"Data length {data.shape[0]} is less than sequence length {sequence_length}")
        
        sequences = []
        for i in range(data.shape[0] - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        
        return torch.stack(sequences)

    def save_model(self, filepath: str) -> None:
        """Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            AdapterError: If model is not fitted or save fails
        """
        if not self.is_fitted or self._model is None:
            raise AdapterError("Model must be fitted before saving")
        
        try:
            import pickle
            from pathlib import Path
            
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state and metadata
            save_data = {
                'algorithm_name': self.algorithm_name,
                'model_state_dict': self._model.state_dict(),
                'model_class': self._model.__class__.__name__,
                'parameters': self.parameters,
                'contamination_rate': self.contamination_rate.value,
                'device': str(self._device),
                'threshold': getattr(self, 'threshold', None),
            }
            
            # Add model-specific parameters for reconstruction
            if hasattr(self._model, 'input_dim'):
                save_data['input_dim'] = self._model.input_dim
            if hasattr(self._model, 'hidden_dim'):
                save_data['hidden_dim'] = self._model.hidden_dim
            if hasattr(self._model, 'num_layers'):
                save_data['num_layers'] = self._model.num_layers
            if hasattr(self._model, 'sequence_length'):
                save_data['sequence_length'] = self._model.sequence_length
                
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
                
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            raise AdapterError(f"Failed to save model: {e}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Raises:
            AdapterError: If loading fails
        """
        try:
            import pickle
            from pathlib import Path
            
            load_path = Path(filepath)
            if not load_path.exists():
                raise AdapterError(f"Model file not found: {filepath}")
            
            with open(load_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore basic properties
            self._algorithm_name = save_data['algorithm_name']
            self._parameters = save_data['parameters']
            self._contamination_rate = ContaminationRate(save_data['contamination_rate'])
            self._device = torch.device(save_data.get('device', 'cpu'))
            
            # Recreate model
            model_class = self._algorithm_map[self._algorithm_name]
            
            if self._algorithm_name == "DAGMM":
                self._model = model_class(
                    save_data['input_dim'],
                    save_data['parameters'].get('hidden_dims', [128, 64, 32]),
                    save_data['parameters'].get('latent_dim', 16),
                    save_data['parameters'].get('n_gmm', 4)
                )
            elif self._algorithm_name == "LSTMAutoEncoder":
                self._model = model_class(
                    save_data['input_dim'],
                    save_data.get('hidden_dim', 64),
                    save_data.get('num_layers', 2),
                    save_data.get('sequence_length', 10),
                    save_data['parameters'].get('dropout', 0.2)
                )
            else:
                self._model = model_class(
                    save_data['input_dim'],
                    save_data['parameters'].get('hidden_dims', [128, 64, 32]),
                    save_data['parameters'].get('latent_dim', 16)
                )
            
            # Load model state
            self._model.load_state_dict(save_data['model_state_dict'])
            self._model.to(self._device)
            self._model.eval()
            
            # Restore training state
            self._is_fitted = True
            if 'threshold' in save_data:
                setattr(self, 'threshold', save_data['threshold'])
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            raise AdapterError(f"Failed to load model: {e}")

    @classmethod
    def get_supported_algorithms(cls) -> list[str]:
        """Get list of supported PyTorch algorithms.

        Returns:
            List of algorithm names
        """
        return list(cls._algorithm_map.keys())

    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> dict[str, Any]:
        """Get information about a specific algorithm.

        Args:
            algorithm: Algorithm name

        Returns:
            Algorithm metadata and parameters
        """
        if algorithm not in cls._algorithm_map:
            raise AlgorithmNotFoundError(f"Algorithm '{algorithm}' not found")

        info = {
            "AutoEncoder": {
                "name": "AutoEncoder",
                "type": "Deep Learning",
                "description": "Vanilla autoencoder that learns to reconstruct normal data",
                "parameters": {
                    "hidden_dims": {
                        "type": "list",
                        "default": [128, 64, 32],
                        "description": "Hidden layer dimensions",
                    },
                    "latent_dim": {
                        "type": "int",
                        "default": 16,
                        "description": "Latent space dimension",
                    },
                    "epochs": {
                        "type": "int",
                        "default": 100,
                        "description": "Training epochs",
                    },
                    "batch_size": {
                        "type": "int",
                        "default": 256,
                        "description": "Batch size",
                    },
                    "learning_rate": {
                        "type": "float",
                        "default": 0.001,
                        "description": "Learning rate",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Expected anomaly rate",
                    },
                },
                "suitable_for": [
                    "tabular_data",
                    "high_dimensional",
                    "nonlinear_patterns",
                ],
                "pros": [
                    "Simple architecture",
                    "Fast training",
                    "Interpretable reconstruction",
                ],
                "cons": ["May overfit", "Sensitive to architecture choices"],
            },
            "VAE": {
                "name": "Variational AutoEncoder",
                "type": "Deep Learning",
                "description": "Probabilistic autoencoder with latent variable modeling",
                "parameters": {
                    "hidden_dims": {
                        "type": "list",
                        "default": [128, 64, 32],
                        "description": "Hidden dimensions",
                    },
                    "latent_dim": {
                        "type": "int",
                        "default": 16,
                        "description": "Latent dimension",
                    },
                    "beta": {
                        "type": "float",
                        "default": 1.0,
                        "description": "KL divergence weight",
                    },
                    "epochs": {
                        "type": "int",
                        "default": 100,
                        "description": "Training epochs",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Anomaly rate",
                    },
                },
                "suitable_for": ["complex_distributions", "uncertainty_quantification"],
                "pros": [
                    "Principled probabilistic approach",
                    "Can generate new samples",
                ],
                "cons": ["More complex than AE", "Requires tuning beta parameter"],
            },
            "DeepSVDD": {
                "name": "Deep Support Vector Data Description",
                "type": "Deep Learning",
                "description": "Deep learning extension of SVDD for one-class classification",
                "parameters": {
                    "hidden_dims": {
                        "type": "list",
                        "default": [128, 64, 32],
                        "description": "Network dimensions",
                    },
                    "latent_dim": {
                        "type": "int",
                        "default": 32,
                        "description": "Output dimension",
                    },
                    "epochs": {
                        "type": "int",
                        "default": 100,
                        "description": "Training epochs",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Anomaly rate",
                    },
                },
                "suitable_for": ["one_class_classification", "compact_representations"],
                "pros": ["Theoretically motivated", "Compact decision boundary"],
                "cons": [
                    "Sensitive to center initialization",
                    "May collapse representations",
                ],
            },
            "DAGMM": {
                "name": "Deep Autoencoding Gaussian Mixture Model",
                "type": "Deep Learning",
                "description": "Combines deep autoencoder with Gaussian mixture model",
                "parameters": {
                    "hidden_dims": {
                        "type": "list",
                        "default": [60, 30, 10],
                        "description": "Compression network",
                    },
                    "latent_dim": {
                        "type": "int",
                        "default": 5,
                        "description": "Encoding dimension",
                    },
                    "n_gmm": {
                        "type": "int",
                        "default": 4,
                        "description": "Number of GMM components",
                    },
                    "epochs": {
                        "type": "int",
                        "default": 100,
                        "description": "Training epochs",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Anomaly rate",
                    },
                },
                "suitable_for": ["multi_modal_data", "complex_anomaly_patterns"],
                "pros": ["Captures multiple normal modes", "End-to-end learning"],
                "cons": ["Complex architecture", "Computationally intensive"],
            },
            "LSTMAutoEncoder": {
                "name": "LSTM AutoEncoder",
                "type": "Deep Learning",
                "description": "LSTM-based autoencoder for time series anomaly detection",
                "parameters": {
                    "hidden_dim": {
                        "type": "int",
                        "default": 64,
                        "description": "LSTM hidden dimension",
                    },
                    "num_layers": {
                        "type": "int",
                        "default": 2,
                        "description": "Number of LSTM layers",
                    },
                    "sequence_length": {
                        "type": "int",
                        "default": 10,
                        "description": "Length of input sequences",
                    },
                    "dropout": {
                        "type": "float",
                        "default": 0.2,
                        "description": "Dropout rate",
                    },
                    "epochs": {
                        "type": "int",
                        "default": 100,
                        "description": "Training epochs",
                    },
                    "contamination": {
                        "type": "float",
                        "default": 0.1,
                        "description": "Anomaly rate",
                    },
                },
                "suitable_for": ["time_series", "sequential_data", "temporal_patterns"],
                "pros": ["Captures temporal dependencies", "Good for sequential data"],
                "cons": ["Requires sequence preparation", "More complex than feedforward"],
            },
        }

        return info.get(algorithm, {})

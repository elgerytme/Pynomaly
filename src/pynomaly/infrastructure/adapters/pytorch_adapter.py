"""PyTorch adapter for deep learning anomaly detection models.

This module provides PyTorch-based deep learning models for anomaly detection,
including autoencoders, VAEs, GANs, and custom architectures.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pynomaly.domain.entities import Dataset, DetectionResult, Detector
from pynomaly.domain.exceptions import AdapterError, AlgorithmNotFoundError
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.shared.protocols import DetectorProtocol

logger = logging.getLogger(__name__)


class BaseAnomalyModel(nn.Module):
    """Base class for PyTorch anomaly detection models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss_function(
        self, x: torch.Tensor, recon: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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
        return nn.functional.mse_loss(recon, x, reduction="mean")

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
        recon_loss = nn.functional.mse_loss(recon, x, reduction="sum") / x.size(0)

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
        recon_cosine = nn.functional.cosine_similarity(x, x_recon, dim=1).unsqueeze(1)

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


class PyTorchAdapter(DetectorProtocol):
    """Adapter for PyTorch-based deep learning anomaly detection models."""

    _algorithm_map = {
        "AutoEncoder": AutoEncoder,
        "VAE": VariationalAutoEncoder,
        "DeepSVDD": DeepSVDD,
        "DAGMM": DAGMM,
    }

    def __init__(self, detector: Detector):
        """Initialize PyTorch adapter with detector configuration.

        Args:
            detector: Detector entity with algorithm configuration
        """
        self.detector = detector
        self._model: BaseAnomalyModel | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_algorithm()

    def _init_algorithm(self) -> None:
        """Initialize the PyTorch model."""
        if self.detector.algorithm not in self._algorithm_map:
            available = ", ".join(self._algorithm_map.keys())
            raise AlgorithmNotFoundError(
                f"Algorithm '{self.detector.algorithm}' not found. "
                f"Available algorithms: {available}"
            )

        # Model will be created when we know input dimensions
        self._model_class = self._algorithm_map[self.detector.algorithm]

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
            params = self.detector.parameters.copy()
            hidden_dims = params.get("hidden_dims", [128, 64, 32])
            latent_dim = params.get("latent_dim", 16)

            if self.detector.algorithm == "DAGMM":
                n_gmm = params.get("n_gmm", 4)
                self._model = self._model_class(
                    input_dim, hidden_dims, latent_dim, n_gmm
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

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / len(train_loader)
                    logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

            self.is_fitted = True
            logger.info(f"Successfully trained PyTorch {self.algorithm_name}")

        except Exception as e:
            raise AdapterError(f"Failed to train PyTorch model: {e}")

    def predict(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies using the trained model.

        Args:
            dataset: Dataset to analyze

        Returns:
            Detection results with anomaly scores and labels
        """
        if not self.detector.is_fitted or self._model is None:
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
            contamination = self.detector.parameters.get("contamination", 0.1)
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
                detector_id=self.detector.id,
                dataset_id=dataset.id,
                scores=anomaly_scores,
                labels=labels.tolist(),
                metadata={
                    "algorithm": self.detector.algorithm,
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
        """
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

        return X

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
        }

        return info.get(algorithm, {})

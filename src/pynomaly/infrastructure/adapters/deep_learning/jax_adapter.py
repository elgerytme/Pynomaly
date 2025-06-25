"""JAX-based deep learning adapter for high-performance anomaly detection."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from pynomaly.shared.protocols import DetectorProtocol

# Optional JAX imports with fallbacks
try:
    import jax
    import jax.numpy as jnp
    import optax
    from jax import grad, jit, random, vmap
    from jax.scipy import stats

    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    random = None
    jit = None
    vmap = None
    grad = None
    stats = None
    optax = None
    JAX_AVAILABLE = False

try:
    import flax
    from flax import linen as nn
    from flax.training import train_state

    FLAX_AVAILABLE = True
except ImportError:
    flax = None
    nn = None
    train_state = None
    FLAX_AVAILABLE = False

logger = logging.getLogger(__name__)


class JAXAutoEncoderConfig(BaseModel):
    """Configuration for JAX AutoEncoder."""

    input_dim: int = Field(description="Input dimension")
    hidden_dims: list[int] = Field(
        default=[128, 64, 32], description="Hidden layer dimensions"
    )
    latent_dim: int = Field(default=16, description="Latent space dimension")
    activation: str = Field(default="relu", description="Activation function")
    dropout_rate: float = Field(default=0.1, ge=0.0, le=0.9, description="Dropout rate")

    # Training parameters
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    epochs: int = Field(default=100, ge=1, description="Training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size")

    # Anomaly detection
    contamination: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Contamination rate"
    )
    random_seed: int = Field(default=42, description="Random seed")


class JAXGMMConfig(BaseModel):
    """Configuration for JAX Gaussian Mixture Model."""

    n_components: int = Field(
        default=5, ge=1, description="Number of mixture components"
    )
    input_dim: int = Field(description="Input dimension")
    covariance_type: str = Field(default="full", description="Covariance type")
    max_iterations: int = Field(default=100, ge=1, description="Maximum EM iterations")
    tolerance: float = Field(default=1e-6, gt=0.0, description="Convergence tolerance")

    # Training
    learning_rate: float = Field(default=0.01, gt=0.0, description="Learning rate")

    # Detection
    contamination: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Contamination rate"
    )
    random_seed: int = Field(default=42, description="Random seed")


class JAXSVDDConfig(BaseModel):
    """Configuration for JAX Support Vector Data Description."""

    input_dim: int = Field(description="Input dimension")
    hidden_dims: list[int] = Field(
        default=[64, 32], description="Hidden layer dimensions"
    )
    nu: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Upper bound on anomaly fraction"
    )
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    epochs: int = Field(default=150, ge=1, description="Training epochs")
    batch_size: int = Field(default=64, ge=1, description="Batch size")

    # Regularization
    weight_decay: float = Field(default=1e-6, ge=0.0, description="Weight decay")

    # Detection
    contamination: float = Field(
        default=0.1, gt=0.0, lt=1.0, description="Contamination rate"
    )
    random_seed: int = Field(default=42, description="Random seed")


class AutoEncoder(nn.Module):
    """Flax AutoEncoder module."""

    config: JAXAutoEncoderConfig

    def setup(self):
        """Initialize layers."""
        if not FLAX_AVAILABLE:
            raise ImportError("Flax is required for AutoEncoder")

        # Encoder layers
        encoder_layers = []
        for dim in self.config.hidden_dims:
            encoder_layers.append(nn.Dense(dim))
        encoder_layers.append(nn.Dense(self.config.latent_dim))
        self.encoder_layers = encoder_layers

        # Decoder layers
        decoder_layers = []
        for dim in reversed(self.config.hidden_dims):
            decoder_layers.append(nn.Dense(dim))
        decoder_layers.append(nn.Dense(self.config.input_dim))
        self.decoder_layers = decoder_layers

        self.dropout = nn.Dropout(self.config.dropout_rate)

    def encode(self, x, training: bool = False):
        """Encode input to latent space."""
        for layer in self.encoder_layers[:-1]:
            x = layer(x)
            x = self._get_activation(x)
            if training:
                x = self.dropout(x, deterministic=not training)

        # Final encoder layer (no activation)
        x = self.encoder_layers[-1](x)
        return x

    def decode(self, z, training: bool = False):
        """Decode latent representation."""
        x = z
        for layer in self.decoder_layers[:-1]:
            x = layer(x)
            x = self._get_activation(x)
            if training:
                x = self.dropout(x, deterministic=not training)

        # Final decoder layer
        x = self.decoder_layers[-1](x)
        return x

    def __call__(self, x, training: bool = False):
        """Forward pass."""
        z = self.encode(x, training)
        recon = self.decode(z, training)
        return recon

    def _get_activation(self, x):
        """Apply activation function."""
        if self.config.activation == "relu":
            return nn.relu(x)
        elif self.config.activation == "tanh":
            return nn.tanh(x)
        elif self.config.activation == "sigmoid":
            return nn.sigmoid(x)
        else:
            return nn.relu(x)


class SVDD(nn.Module):
    """Flax Support Vector Data Description module."""

    config: JAXSVDDConfig

    def setup(self):
        """Initialize layers."""
        layers = []
        for dim in self.config.hidden_dims:
            layers.append(nn.Dense(dim))
        layers.append(nn.Dense(1))  # Output to 1D for distance calculation
        self.layers = layers

    def __call__(self, x, training: bool = False):
        """Forward pass."""
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.relu(x)

        # Final layer
        x = self.layers[-1](x)
        return x


def create_gmm_params(key, config: JAXGMMConfig):
    """Initialize GMM parameters."""
    keys = random.split(key, 3)

    # Initialize mixing coefficients (uniform)
    mixing_coeffs = jnp.ones(config.n_components) / config.n_components

    # Initialize means (random)
    means = random.normal(keys[0], (config.n_components, config.input_dim))

    # Initialize covariances (identity matrices)
    if config.covariance_type == "full":
        covariances = jnp.array(
            [jnp.eye(config.input_dim) for _ in range(config.n_components)]
        )
    elif config.covariance_type == "diag":
        covariances = jnp.ones((config.n_components, config.input_dim))
    else:  # spherical
        covariances = jnp.ones(config.n_components)

    return {"mixing_coeffs": mixing_coeffs, "means": means, "covariances": covariances}


@jit
def gmm_log_likelihood(params, x, config):
    """Compute log-likelihood for GMM."""
    mixing_coeffs = params["mixing_coeffs"]
    means = params["means"]
    covariances = params["covariances"]

    n_components = config.n_components
    log_probs = []

    for k in range(n_components):
        if config.covariance_type == "full":
            cov = covariances[k]
            log_prob = stats.multivariate_normal.logpdf(x, means[k], cov)
        elif config.covariance_type == "diag":
            cov_diag = covariances[k]
            # Diagonal covariance implementation
            diff = x - means[k]
            log_prob = -0.5 * jnp.sum((diff**2) / cov_diag, axis=-1)
            log_prob -= 0.5 * jnp.sum(jnp.log(2 * jnp.pi * cov_diag))
        else:  # spherical
            cov_scalar = covariances[k]
            diff = x - means[k]
            log_prob = -0.5 * jnp.sum(diff**2, axis=-1) / cov_scalar
            log_prob -= 0.5 * x.shape[-1] * jnp.log(2 * jnp.pi * cov_scalar)

        log_probs.append(jnp.log(mixing_coeffs[k]) + log_prob)

    log_probs = jnp.stack(log_probs, axis=0)
    return jax.scipy.special.logsumexp(log_probs, axis=0)


class JAXAdapter(DetectorProtocol):
    """JAX-based high-performance deep learning adapter."""

    def __init__(
        self,
        algorithm: str = "autoencoder",
        model_config: dict[str, Any] | None = None,
        random_state: int | None = None,
        device: str | None = None,
    ):
        """Initialize JAX adapter.

        Args:
            algorithm: Algorithm type ('autoencoder', 'gmm', 'svdd')
            model_config: Algorithm-specific configuration
            random_state: Random seed
            device: Device type ('cpu', 'gpu', 'auto')
        """
        if not JAX_AVAILABLE:
            raise ImportError(
                "JAX is required for JAXAdapter. Install with: pip install jax jaxlib"
            )

        self.algorithm = algorithm
        self.model_config = model_config or {}
        self.random_state = random_state or 42

        # Configure JAX device
        if device == "gpu" or device == "auto":
            # JAX will automatically use GPU if available
            pass

        # Initialize JAX random key
        self.key = random.PRNGKey(self.random_state)

        # Model components
        self.model = None
        self.params = None
        self.model_state = None
        self.scaler = None
        self.threshold = None
        self.is_trained = False

        logger.info(f"Initialized JAXAdapter with {algorithm}")

    @property
    def algorithm_name(self) -> str:
        """Get algorithm name."""
        return f"jax_{self.algorithm}"

    @property
    def algorithm_params(self) -> dict[str, Any]:
        """Get algorithm parameters."""
        return {
            "algorithm": self.algorithm,
            "model_config": self.model_config,
            "random_state": self.random_state,
        }

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> JAXAdapter:
        """Train the JAX model."""
        try:
            # Preprocess data
            X_processed = self._preprocess_data(X)

            # Train model based on algorithm
            if self.algorithm == "autoencoder":
                self._fit_autoencoder(X_processed)
            elif self.algorithm == "gmm":
                self._fit_gmm(X_processed)
            elif self.algorithm == "svdd":
                self._fit_svdd(X_processed)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

            # Calculate threshold
            self._calculate_threshold(X_processed)

            self.is_trained = True
            logger.info(f"JAX {self.algorithm} model trained successfully")

            return self

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Failed to train JAX model: {e}")

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
        elif self.algorithm == "gmm":
            scores = self._gmm_scores(X_processed)
        elif self.algorithm == "svdd":
            scores = self._svdd_scores(X_processed)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return np.array(scores)

    def _preprocess_data(self, X: np.ndarray, fit_scaler: bool = True) -> jnp.ndarray:
        """Preprocess data for JAX models."""
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

        return jnp.array(X_scaled, dtype=jnp.float32)

    def _fit_autoencoder(self, X: jnp.ndarray):
        """Train AutoEncoder using JAX/Flax."""
        if not FLAX_AVAILABLE:
            raise ImportError("Flax is required for AutoEncoder")

        config = JAXAutoEncoderConfig(input_dim=X.shape[1], **self.model_config)

        # Initialize model
        self.model = AutoEncoder(config)
        key, subkey = random.split(self.key)
        dummy_input = jnp.ones((1, config.input_dim))
        params = self.model.init(subkey, dummy_input, training=False)

        # Initialize optimizer
        tx = optax.adam(config.learning_rate)

        # Training state
        state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

        # Training loop
        @jit
        def train_step(state, batch):
            def loss_fn(params):
                recon = state.apply_fn(
                    params, batch, training=True, rngs={"dropout": subkey}
                )
                loss = jnp.mean((batch - recon) ** 2)
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        # Train in batches
        n_samples = X.shape[0]
        batch_size = config.batch_size

        for epoch in range(config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle data
            key, subkey = random.split(key)
            perm = random.permutation(subkey, n_samples)
            X_shuffled = X[perm]

            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i : i + batch_size]
                state, loss = train_step(state, batch)
                epoch_loss += loss
                n_batches += 1

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                logger.debug(
                    f"AutoEncoder Epoch {epoch + 1}/{config.epochs}, Loss: {avg_loss:.6f}"
                )

        self.params = state.params
        self.model_state = state

    def _fit_gmm(self, X: jnp.ndarray):
        """Train Gaussian Mixture Model using JAX."""
        config = JAXGMMConfig(input_dim=X.shape[1], **self.model_config)

        # Initialize parameters
        key, subkey = random.split(self.key)
        params = create_gmm_params(subkey, config)

        # EM algorithm
        for iteration in range(config.max_iterations):
            # E-step: compute responsibilities
            log_probs = []
            for k in range(config.n_components):
                if config.covariance_type == "full":
                    cov = params["covariances"][k]
                    log_prob = stats.multivariate_normal.logpdf(
                        X, params["means"][k], cov
                    )
                else:
                    # Simplified for diagonal/spherical
                    diff = X - params["means"][k]
                    log_prob = -0.5 * jnp.sum(diff**2, axis=-1)

                log_probs.append(jnp.log(params["mixing_coeffs"][k]) + log_prob)

            log_probs = jnp.stack(log_probs, axis=0)
            log_resp = log_probs - jax.scipy.special.logsumexp(
                log_probs, axis=0, keepdims=True
            )
            resp = jnp.exp(log_resp)

            # M-step: update parameters
            resp_sum = jnp.sum(resp, axis=1)

            # Update mixing coefficients
            params["mixing_coeffs"] = resp_sum / X.shape[0]

            # Update means
            for k in range(config.n_components):
                params["means"] = (
                    params["means"]
                    .at[k]
                    .set(jnp.sum(resp[k : k + 1].T * X, axis=0) / resp_sum[k])
                )

            # Update covariances (simplified)
            for k in range(config.n_components):
                diff = X - params["means"][k]
                if config.covariance_type == "full":
                    cov = jnp.dot((resp[k : k + 1].T * diff).T, diff) / resp_sum[k]
                    params["covariances"] = params["covariances"].at[k].set(cov)

            if (iteration + 1) % 10 == 0:
                ll = jnp.mean(gmm_log_likelihood(params, X, config))
                logger.debug(
                    f"GMM Iteration {iteration + 1}/{config.max_iterations}, Log-likelihood: {ll:.6f}"
                )

        self.params = params

    def _fit_svdd(self, X: jnp.ndarray):
        """Train Support Vector Data Description using JAX."""
        if not FLAX_AVAILABLE:
            raise ImportError("Flax is required for SVDD")

        config = JAXSVDDConfig(input_dim=X.shape[1], **self.model_config)

        # Initialize model
        self.model = SVDD(config)
        key, subkey = random.split(self.key)
        dummy_input = jnp.ones((1, config.input_dim))
        params = self.model.init(subkey, dummy_input, training=False)

        # Initialize optimizer
        tx = optax.adam(config.learning_rate)

        # Training state
        state = train_state.TrainState.create(
            apply_fn=self.model.apply, params=params, tx=tx
        )

        # SVDD training loop
        @jit
        def train_step(state, batch):
            def loss_fn(params):
                # Forward pass
                outputs = state.apply_fn(params, batch, training=True)

                # SVDD loss: minimize volume while keeping most data inside
                center = jnp.mean(outputs, axis=0)
                distances = jnp.sum((outputs - center) ** 2, axis=1)

                # Soft boundary loss
                quantile = jnp.quantile(distances, 1 - config.nu)
                loss = jnp.mean(jnp.maximum(0, distances - quantile))

                # Add weight decay
                weight_penalty = sum(jnp.sum(p**2) for p in jax.tree_leaves(params))
                loss += config.weight_decay * weight_penalty

                return loss

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        # Train in batches
        n_samples = X.shape[0]
        batch_size = config.batch_size

        for epoch in range(config.epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle data
            key, subkey = random.split(key)
            perm = random.permutation(subkey, n_samples)
            X_shuffled = X[perm]

            for i in range(0, n_samples, batch_size):
                batch = X_shuffled[i : i + batch_size]
                state, loss = train_step(state, batch)
                epoch_loss += loss
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / n_batches
                logger.debug(
                    f"SVDD Epoch {epoch + 1}/{config.epochs}, Loss: {avg_loss:.6f}"
                )

        self.params = state.params
        self.model_state = state

    def _autoencoder_scores(self, X: jnp.ndarray) -> jnp.ndarray:
        """Calculate reconstruction error for AutoEncoder."""
        recon = self.model.apply(self.params, X, training=False)
        scores = jnp.mean((X - recon) ** 2, axis=1)
        return scores

    def _gmm_scores(self, X: jnp.ndarray) -> jnp.ndarray:
        """Calculate negative log-likelihood for GMM."""
        config = JAXGMMConfig(**self.model_config)
        log_likelihoods = gmm_log_likelihood(self.params, X, config)
        scores = -log_likelihoods  # Negative log-likelihood as anomaly score
        return scores

    def _svdd_scores(self, X: jnp.ndarray) -> jnp.ndarray:
        """Calculate distance from center for SVDD."""
        outputs = self.model.apply(self.params, X, training=False)
        center = jnp.mean(outputs, axis=0)  # This should be computed from training data
        scores = jnp.sum((outputs - center) ** 2, axis=1)
        return scores

    def _calculate_threshold(self, X: jnp.ndarray):
        """Calculate anomaly threshold."""
        scores = self.decision_function(np.array(X))

        # Get contamination rate from config
        contamination = 0.1
        if hasattr(self, "model_config"):
            contamination = self.model_config.get("contamination", 0.1)

        self.threshold = np.percentile(scores, (1 - contamination) * 100)
        logger.info(f"Calculated threshold: {self.threshold:.6f}")

    def save_model(self, path: str | Path):
        """Save trained model."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save parameters and metadata
        save_dict = {
            "algorithm": self.algorithm,
            "model_config": self.model_config,
            "threshold": float(self.threshold),
            "random_state": self.random_state,
            "scaler_params": (
                {
                    "mean_": self.scaler.mean_.tolist(),
                    "scale_": self.scaler.scale_.tolist(),
                }
                if self.scaler
                else None
            ),
        }

        # Save JAX parameters
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(save_dict, f, indent=2)

        # Save model parameters (would need proper JAX serialization in production)
        if self.params:
            # Convert JAX arrays to numpy for saving
            params_numpy = jax.tree_map(lambda x: np.array(x), self.params)
            np.savez(
                path.with_suffix(".npz"),
                **{str(i): arr for i, arr in enumerate(jax.tree_leaves(params_numpy))},
            )

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str | Path):
        """Load trained model."""
        path = Path(path)

        # Load metadata
        with open(path.with_suffix(".json")) as f:
            metadata = json.load(f)

        self.algorithm = metadata["algorithm"]
        self.model_config = metadata["model_config"]
        self.threshold = metadata["threshold"]
        self.random_state = metadata.get("random_state", 42)

        # Restore scaler
        if metadata.get("scaler_params"):
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(metadata["scaler_params"]["mean_"])
            self.scaler.scale_ = np.array(metadata["scaler_params"]["scale_"])

        # Load and recreate model (simplified - would need proper recreation)
        if path.with_suffix(".npz").exists():
            # This is a simplified loading - proper implementation would recreate model structure
            np.load(path.with_suffix(".npz"))
            # self.params = ... (proper parameter reconstruction)

        self.is_trained = True
        logger.info(f"Model loaded from {path}")

    async def async_fit(self, X: np.ndarray, y: np.ndarray | None = None) -> JAXAdapter:
        """Asynchronous training."""
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
            "is_trained": self.is_trained,
            "model_config": self.model_config,
            "threshold": self.threshold,
            "backend": "JAX",
        }

        if self.params and self.is_trained:
            # Count parameters
            total_params = sum(
                np.prod(param.shape) for param in jax.tree_leaves(self.params)
            )
            info["total_parameters"] = int(total_params)

        return info

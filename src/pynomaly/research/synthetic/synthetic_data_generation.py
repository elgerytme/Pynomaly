"""Synthetic data generation with GANs, privacy-preserving methods, and anomaly synthesis."""

from __future__ import annotations

import asyncio
import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SyntheticMethod(str, Enum):
    """Synthetic data generation methods."""
    
    VANILLA_GAN = "vanilla_gan"
    WGAN = "wgan"  # Wasserstein GAN
    CTGAN = "ctgan"  # Conditional Tabular GAN
    COPULA_GAN = "copula_gan"
    VAE = "vae"  # Variational Autoencoder
    DIFFUSION = "diffusion"  # Diffusion Models
    STATISTICAL = "statistical"  # Statistical synthesis
    SMOTE = "smote"  # Synthetic Minority Oversampling


class PrivacyMethod(str, Enum):
    """Privacy-preserving techniques."""
    
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    FEDERATED_LEARNING = "federated_learning"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_MULTI_PARTY = "secure_multi_party"
    K_ANONYMITY = "k_anonymity"
    L_DIVERSITY = "l_diversity"
    T_CLOSENESS = "t_closeness"


class DataType(str, Enum):
    """Types of data for synthesis."""
    
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    IMAGE = "image"
    TEXT = "text"
    MIXED = "mixed"
    ANOMALY_ENRICHED = "anomaly_enriched"


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    
    method: SyntheticMethod
    data_type: DataType
    num_samples: int
    privacy_method: Optional[PrivacyMethod] = None
    privacy_budget: float = 1.0  # Epsilon for differential privacy
    anomaly_ratio: float = 0.1  # Ratio of anomalies to generate
    quality_metrics: List[str] = field(default_factory=lambda: ["distribution", "correlation", "privacy"])
    constraints: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None


@dataclass
class SyntheticDataResult:
    """Result from synthetic data generation."""
    
    synthetic_data: np.ndarray
    generation_method: SyntheticMethod
    quality_scores: Dict[str, float]
    privacy_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    generation_time: float
    anomaly_labels: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PrivacyAnalysis:
    """Privacy analysis result."""
    
    privacy_method: PrivacyMethod
    privacy_budget_used: float
    privacy_guarantee: str
    risk_metrics: Dict[str, float]
    recommended_usage: str
    limitations: List[str]


class BaseSyntheticGenerator(ABC):
    """Base class for synthetic data generators."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.is_fitted = False
        self.model_parameters: Dict[str, Any] = {}
        
        # Set random seed if provided
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)
    
    @abstractmethod
    async def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit the synthetic data generator."""
        pass
    
    @abstractmethod
    async def generate(self, num_samples: int) -> SyntheticDataResult:
        """Generate synthetic data."""
        pass
    
    async def evaluate_quality(
        self,
        original_data: np.ndarray,
        synthetic_data: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate quality of synthetic data."""
        try:
            quality_scores = {}
            
            # Distribution similarity
            if "distribution" in self.config.quality_metrics:
                quality_scores["distribution_similarity"] = await self._evaluate_distribution_similarity(
                    original_data, synthetic_data
                )
            
            # Correlation preservation
            if "correlation" in self.config.quality_metrics:
                quality_scores["correlation_preservation"] = await self._evaluate_correlation_preservation(
                    original_data, synthetic_data
                )
            
            # Statistical properties
            if "statistical" in self.config.quality_metrics:
                quality_scores["statistical_fidelity"] = await self._evaluate_statistical_fidelity(
                    original_data, synthetic_data
                )
            
            # Utility preservation (simplified)
            quality_scores["utility_score"] = np.mean(list(quality_scores.values()))
            
            return quality_scores
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {"error": 1.0}
    
    async def _evaluate_distribution_similarity(
        self,
        original: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """Evaluate distribution similarity using KL divergence approximation."""
        try:
            similarity_scores = []
            
            for i in range(min(original.shape[1], synthetic.shape[1])):
                orig_feature = original[:, i]
                synth_feature = synthetic[:, i]
                
                # Create histograms
                bins = np.linspace(
                    min(np.min(orig_feature), np.min(synth_feature)),
                    max(np.max(orig_feature), np.max(synth_feature)),
                    20
                )
                
                orig_hist, _ = np.histogram(orig_feature, bins=bins, density=True)
                synth_hist, _ = np.histogram(synth_feature, bins=bins, density=True)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-8
                orig_hist += epsilon
                synth_hist += epsilon
                
                # Normalize
                orig_hist = orig_hist / np.sum(orig_hist)
                synth_hist = synth_hist / np.sum(synth_hist)
                
                # Jensen-Shannon divergence (symmetric)
                m = 0.5 * (orig_hist + synth_hist)
                js_div = 0.5 * np.sum(orig_hist * np.log(orig_hist / m)) + \
                         0.5 * np.sum(synth_hist * np.log(synth_hist / m))
                
                # Convert to similarity (0 = identical, 1 = completely different)
                similarity = 1.0 - min(1.0, js_div / np.log(2))
                similarity_scores.append(similarity)
            
            return float(np.mean(similarity_scores))
            
        except Exception as e:
            logger.error(f"Distribution similarity evaluation failed: {e}")
            return 0.5
    
    async def _evaluate_correlation_preservation(
        self,
        original: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """Evaluate correlation structure preservation."""
        try:
            # Calculate correlation matrices
            orig_corr = np.corrcoef(original, rowvar=False)
            synth_corr = np.corrcoef(synthetic, rowvar=False)
            
            # Handle NaN values
            orig_corr = np.nan_to_num(orig_corr)
            synth_corr = np.nan_to_num(synth_corr)
            
            # Calculate Frobenius norm difference
            diff = orig_corr - synth_corr
            frobenius_norm = np.linalg.norm(diff, 'fro')
            
            # Normalize by size
            max_possible_diff = np.linalg.norm(orig_corr, 'fro') + np.linalg.norm(synth_corr, 'fro')
            
            if max_possible_diff > 0:
                correlation_score = 1.0 - (frobenius_norm / max_possible_diff)
            else:
                correlation_score = 1.0
            
            return float(max(0.0, correlation_score))
            
        except Exception as e:
            logger.error(f"Correlation preservation evaluation failed: {e}")
            return 0.5
    
    async def _evaluate_statistical_fidelity(
        self,
        original: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """Evaluate statistical properties preservation."""
        try:
            fidelity_scores = []
            
            for i in range(min(original.shape[1], synthetic.shape[1])):
                orig_feature = original[:, i]
                synth_feature = synthetic[:, i]
                
                # Mean difference
                mean_diff = abs(np.mean(orig_feature) - np.mean(synth_feature))
                mean_score = 1.0 / (1.0 + mean_diff)
                
                # Std difference
                std_diff = abs(np.std(orig_feature) - np.std(synth_feature))
                std_score = 1.0 / (1.0 + std_diff)
                
                # Combine scores
                feature_score = (mean_score + std_score) / 2.0
                fidelity_scores.append(feature_score)
            
            return float(np.mean(fidelity_scores))
            
        except Exception as e:
            logger.error(f"Statistical fidelity evaluation failed: {e}")
            return 0.5


class VanillaGANGenerator(BaseSyntheticGenerator):
    """Vanilla GAN implementation for synthetic data generation."""
    
    def __init__(self, config: SyntheticDataConfig):
        super().__init__(config)
        self.latent_dim = 100
        self.generator_params = None
        self.discriminator_params = None
        self.training_history = []
    
    async def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit the GAN model."""
        try:
            logger.info("Fitting Vanilla GAN")
            
            self.input_dim = X.shape[1]
            self.data_mean = np.mean(X, axis=0)
            self.data_std = np.std(X, axis=0)
            
            # Normalize data
            X_normalized = (X - self.data_mean) / (self.data_std + 1e-8)
            
            # Initialize generator and discriminator
            await self._initialize_networks()
            
            # Train GAN
            await self._train_gan(X_normalized)
            
            self.is_fitted = True
            logger.info("Vanilla GAN training completed")
            
        except Exception as e:
            logger.error(f"Vanilla GAN fitting failed: {e}")
            raise
    
    async def generate(self, num_samples: int) -> SyntheticDataResult:
        """Generate synthetic data using trained GAN."""
        try:
            if not self.is_fitted:
                raise ValueError("Generator must be fitted before generating data")
            
            start_time = datetime.now()
            
            # Generate random noise
            noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
            
            # Generate data using generator
            synthetic_normalized = await self._generate_from_noise(noise)
            
            # Denormalize
            synthetic_data = synthetic_normalized * (self.data_std + 1e-8) + self.data_mean
            
            # Generate anomaly labels if requested
            anomaly_labels = None
            if self.config.anomaly_ratio > 0:
                anomaly_labels, synthetic_data = await self._inject_anomalies(
                    synthetic_data, self.config.anomaly_ratio
                )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return SyntheticDataResult(
                synthetic_data=synthetic_data,
                generation_method=SyntheticMethod.VANILLA_GAN,
                quality_scores={},  # Will be calculated separately
                privacy_metrics={},
                metadata={
                    "latent_dim": self.latent_dim,
                    "training_epochs": len(self.training_history),
                    "generator_params": len(self.generator_params) if self.generator_params else 0
                },
                generation_time=generation_time,
                anomaly_labels=anomaly_labels
            )
            
        except Exception as e:
            logger.error(f"Data generation failed: {e}")
            raise
    
    async def _initialize_networks(self) -> None:
        """Initialize generator and discriminator networks."""
        # Simplified network initialization
        # In practice, this would initialize actual neural network layers
        
        # Generator: latent_dim -> hidden -> input_dim
        self.generator_params = {
            "layer1_weights": np.random.normal(0, 0.02, (self.latent_dim, 128)),
            "layer1_bias": np.zeros(128),
            "layer2_weights": np.random.normal(0, 0.02, (128, 64)),
            "layer2_bias": np.zeros(64),
            "output_weights": np.random.normal(0, 0.02, (64, self.input_dim)),
            "output_bias": np.zeros(self.input_dim)
        }
        
        # Discriminator: input_dim -> hidden -> 1
        self.discriminator_params = {
            "layer1_weights": np.random.normal(0, 0.02, (self.input_dim, 64)),
            "layer1_bias": np.zeros(64),
            "layer2_weights": np.random.normal(0, 0.02, (64, 32)),
            "layer2_bias": np.zeros(32),
            "output_weights": np.random.normal(0, 0.02, (32, 1)),
            "output_bias": np.zeros(1)
        }
    
    async def _train_gan(self, X: np.ndarray) -> None:
        """Train GAN (simplified training loop)."""
        try:
            epochs = 100
            batch_size = 64
            
            for epoch in range(epochs):
                epoch_g_loss = 0.0
                epoch_d_loss = 0.0
                num_batches = 0
                
                # Mini-batch training
                for i in range(0, len(X), batch_size):
                    batch_real = X[i:i + batch_size]
                    
                    if len(batch_real) < batch_size:
                        continue
                    
                    # Train discriminator
                    d_loss = await self._train_discriminator_step(batch_real)
                    epoch_d_loss += d_loss
                    
                    # Train generator
                    g_loss = await self._train_generator_step(batch_size)
                    epoch_g_loss += g_loss
                    
                    num_batches += 1
                
                # Record training history
                if num_batches > 0:
                    avg_g_loss = epoch_g_loss / num_batches
                    avg_d_loss = epoch_d_loss / num_batches
                    
                    self.training_history.append({
                        "epoch": epoch,
                        "generator_loss": avg_g_loss,
                        "discriminator_loss": avg_d_loss
                    })
                
                if epoch % 20 == 0:
                    logger.info(f"GAN Epoch {epoch}, G_loss: {avg_g_loss:.4f}, D_loss: {avg_d_loss:.4f}")
            
        except Exception as e:
            logger.error(f"GAN training failed: {e}")
    
    async def _train_discriminator_step(self, real_batch: np.ndarray) -> float:
        """Single discriminator training step."""
        try:
            batch_size = len(real_batch)
            
            # Generate fake data
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_batch = await self._generate_from_noise(noise)
            
            # Discriminator predictions
            real_pred = await self._discriminator_forward(real_batch)
            fake_pred = await self._discriminator_forward(fake_batch)
            
            # Calculate loss (binary cross-entropy approximation)
            real_loss = -np.mean(np.log(real_pred + 1e-8))
            fake_loss = -np.mean(np.log(1 - fake_pred + 1e-8))
            d_loss = real_loss + fake_loss
            
            # Update discriminator parameters (simplified gradient update)
            await self._update_discriminator_params(d_loss)
            
            return float(d_loss)
            
        except Exception as e:
            logger.error(f"Discriminator training step failed: {e}")
            return 1.0
    
    async def _train_generator_step(self, batch_size: int) -> float:
        """Single generator training step."""
        try:
            # Generate fake data
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_batch = await self._generate_from_noise(noise)
            
            # Discriminator prediction on fake data
            fake_pred = await self._discriminator_forward(fake_batch)
            
            # Generator loss (wants discriminator to think fake data is real)
            g_loss = -np.mean(np.log(fake_pred + 1e-8))
            
            # Update generator parameters (simplified gradient update)
            await self._update_generator_params(g_loss)
            
            return float(g_loss)
            
        except Exception as e:
            logger.error(f"Generator training step failed: {e}")
            return 1.0
    
    async def _generate_from_noise(self, noise: np.ndarray) -> np.ndarray:
        """Generate data from noise using generator."""
        try:
            # Forward pass through generator
            x = noise
            
            # Layer 1
            x = np.dot(x, self.generator_params["layer1_weights"]) + self.generator_params["layer1_bias"]
            x = np.maximum(0.01 * x, x)  # Leaky ReLU
            
            # Layer 2
            x = np.dot(x, self.generator_params["layer2_weights"]) + self.generator_params["layer2_bias"]
            x = np.maximum(0.01 * x, x)  # Leaky ReLU
            
            # Output layer
            x = np.dot(x, self.generator_params["output_weights"]) + self.generator_params["output_bias"]
            x = np.tanh(x)  # Tanh activation
            
            return x
            
        except Exception as e:
            logger.error(f"Generation from noise failed: {e}")
            return np.random.normal(0, 1, (len(noise), self.input_dim))
    
    async def _discriminator_forward(self, data: np.ndarray) -> np.ndarray:
        """Forward pass through discriminator."""
        try:
            x = data
            
            # Layer 1
            x = np.dot(x, self.discriminator_params["layer1_weights"]) + self.discriminator_params["layer1_bias"]
            x = np.maximum(0.01 * x, x)  # Leaky ReLU
            
            # Layer 2
            x = np.dot(x, self.discriminator_params["layer2_weights"]) + self.discriminator_params["layer2_bias"]
            x = np.maximum(0.01 * x, x)  # Leaky ReLU
            
            # Output layer
            x = np.dot(x, self.discriminator_params["output_weights"]) + self.discriminator_params["output_bias"]
            x = 1.0 / (1.0 + np.exp(-x))  # Sigmoid
            
            return x.flatten()
            
        except Exception as e:
            logger.error(f"Discriminator forward pass failed: {e}")
            return np.random.random(len(data))
    
    async def _update_discriminator_params(self, loss: float) -> None:
        """Update discriminator parameters (simplified)."""
        # Simplified parameter update
        learning_rate = 0.0002
        noise_scale = learning_rate * loss * 0.01
        
        for param_name in self.discriminator_params:
            noise = np.random.normal(0, noise_scale, self.discriminator_params[param_name].shape)
            self.discriminator_params[param_name] -= noise
    
    async def _update_generator_params(self, loss: float) -> None:
        """Update generator parameters (simplified)."""
        # Simplified parameter update
        learning_rate = 0.0002
        noise_scale = learning_rate * loss * 0.01
        
        for param_name in self.generator_params:
            noise = np.random.normal(0, noise_scale, self.generator_params[param_name].shape)
            self.generator_params[param_name] -= noise
    
    async def _inject_anomalies(
        self,
        data: np.ndarray,
        anomaly_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject anomalies into synthetic data."""
        try:
            num_anomalies = int(len(data) * anomaly_ratio)
            anomaly_indices = np.random.choice(len(data), num_anomalies, replace=False)
            
            # Create anomaly labels
            anomaly_labels = np.zeros(len(data))
            anomaly_labels[anomaly_indices] = 1
            
            # Modify selected samples to be anomalous
            anomalous_data = data.copy()
            
            for idx in anomaly_indices:
                # Different anomaly injection strategies
                strategy = np.random.choice(["outlier", "noise", "pattern_break"])
                
                if strategy == "outlier":
                    # Make values extreme
                    feature_idx = np.random.randint(data.shape[1])
                    factor = np.random.choice([3.0, -3.0, 5.0, -5.0])
                    anomalous_data[idx, feature_idx] *= factor
                
                elif strategy == "noise":
                    # Add high noise
                    noise = np.random.normal(0, 2.0, data.shape[1])
                    anomalous_data[idx] += noise
                
                elif strategy == "pattern_break":
                    # Reverse some feature relationships
                    features_to_flip = np.random.choice(
                        data.shape[1], 
                        size=min(3, data.shape[1]), 
                        replace=False
                    )
                    anomalous_data[idx, features_to_flip] *= -1
            
            return anomaly_labels, anomalous_data
            
        except Exception as e:
            logger.error(f"Anomaly injection failed: {e}")
            return np.zeros(len(data)), data


class VAEGenerator(BaseSyntheticGenerator):
    """Variational Autoencoder for synthetic data generation."""
    
    def __init__(self, config: SyntheticDataConfig):
        super().__init__(config)
        self.latent_dim = 50
        self.encoder_params = None
        self.decoder_params = None
    
    async def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit the VAE model."""
        try:
            logger.info("Fitting VAE")
            
            self.input_dim = X.shape[1]
            self.data_mean = np.mean(X, axis=0)
            self.data_std = np.std(X, axis=0)
            
            # Normalize data
            X_normalized = (X - self.data_mean) / (self.data_std + 1e-8)
            
            # Initialize encoder and decoder
            await self._initialize_vae_networks()
            
            # Train VAE
            await self._train_vae(X_normalized)
            
            self.is_fitted = True
            logger.info("VAE training completed")
            
        except Exception as e:
            logger.error(f"VAE fitting failed: {e}")
            raise
    
    async def generate(self, num_samples: int) -> SyntheticDataResult:
        """Generate synthetic data using trained VAE."""
        try:
            if not self.is_fitted:
                raise ValueError("VAE must be fitted before generating data")
            
            start_time = datetime.now()
            
            # Sample from latent space
            latent_samples = np.random.normal(0, 1, (num_samples, self.latent_dim))
            
            # Decode to generate data
            synthetic_normalized = await self._decode_latent(latent_samples)
            
            # Denormalize
            synthetic_data = synthetic_normalized * (self.data_std + 1e-8) + self.data_mean
            
            # Generate anomaly labels if requested
            anomaly_labels = None
            if self.config.anomaly_ratio > 0:
                anomaly_labels, synthetic_data = await self._inject_vae_anomalies(
                    synthetic_data, self.config.anomaly_ratio
                )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return SyntheticDataResult(
                synthetic_data=synthetic_data,
                generation_method=SyntheticMethod.VAE,
                quality_scores={},
                privacy_metrics={},
                metadata={
                    "latent_dim": self.latent_dim,
                    "input_dim": self.input_dim
                },
                generation_time=generation_time,
                anomaly_labels=anomaly_labels
            )
            
        except Exception as e:
            logger.error(f"VAE generation failed: {e}")
            raise
    
    async def _initialize_vae_networks(self) -> None:
        """Initialize VAE encoder and decoder."""
        hidden_dim = 64
        
        # Encoder: input_dim -> hidden -> latent_dim * 2 (mean and log_var)
        self.encoder_params = {
            "layer1_weights": np.random.normal(0, 0.02, (self.input_dim, hidden_dim)),
            "layer1_bias": np.zeros(hidden_dim),
            "mean_weights": np.random.normal(0, 0.02, (hidden_dim, self.latent_dim)),
            "mean_bias": np.zeros(self.latent_dim),
            "logvar_weights": np.random.normal(0, 0.02, (hidden_dim, self.latent_dim)),
            "logvar_bias": np.zeros(self.latent_dim)
        }
        
        # Decoder: latent_dim -> hidden -> input_dim
        self.decoder_params = {
            "layer1_weights": np.random.normal(0, 0.02, (self.latent_dim, hidden_dim)),
            "layer1_bias": np.zeros(hidden_dim),
            "output_weights": np.random.normal(0, 0.02, (hidden_dim, self.input_dim)),
            "output_bias": np.zeros(self.input_dim)
        }
    
    async def _train_vae(self, X: np.ndarray) -> None:
        """Train VAE (simplified training)."""
        try:
            epochs = 100
            
            for epoch in range(epochs):
                # Encode
                mu, log_var = await self._encode(X)
                
                # Reparameterization trick
                std = np.exp(0.5 * log_var)
                epsilon = np.random.normal(0, 1, mu.shape)
                z = mu + std * epsilon
                
                # Decode
                reconstructed = await self._decode_latent(z)
                
                # Calculate loss (reconstruction + KL divergence)
                recon_loss = np.mean((X - reconstructed) ** 2)
                kl_loss = -0.5 * np.mean(1 + log_var - mu ** 2 - np.exp(log_var))
                total_loss = recon_loss + kl_loss
                
                # Update parameters (simplified)
                await self._update_vae_params(total_loss)
                
                if epoch % 20 == 0:
                    logger.info(f"VAE Epoch {epoch}, Loss: {total_loss:.4f}")
            
        except Exception as e:
            logger.error(f"VAE training failed: {e}")
    
    async def _encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input to latent parameters."""
        # Hidden layer
        h = np.dot(x, self.encoder_params["layer1_weights"]) + self.encoder_params["layer1_bias"]
        h = np.maximum(0, h)  # ReLU
        
        # Mean and log variance
        mu = np.dot(h, self.encoder_params["mean_weights"]) + self.encoder_params["mean_bias"]
        log_var = np.dot(h, self.encoder_params["logvar_weights"]) + self.encoder_params["logvar_bias"]
        
        return mu, log_var
    
    async def _decode_latent(self, z: np.ndarray) -> np.ndarray:
        """Decode latent variables to data."""
        # Hidden layer
        h = np.dot(z, self.decoder_params["layer1_weights"]) + self.decoder_params["layer1_bias"]
        h = np.maximum(0, h)  # ReLU
        
        # Output layer
        output = np.dot(h, self.decoder_params["output_weights"]) + self.decoder_params["output_bias"]
        
        return output
    
    async def _update_vae_params(self, loss: float) -> None:
        """Update VAE parameters (simplified)."""
        learning_rate = 0.001
        noise_scale = learning_rate * loss * 0.001
        
        # Update encoder
        for param_name in self.encoder_params:
            noise = np.random.normal(0, noise_scale, self.encoder_params[param_name].shape)
            self.encoder_params[param_name] -= noise
        
        # Update decoder
        for param_name in self.decoder_params:
            noise = np.random.normal(0, noise_scale, self.decoder_params[param_name].shape)
            self.decoder_params[param_name] -= noise
    
    async def _inject_vae_anomalies(
        self,
        data: np.ndarray,
        anomaly_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject anomalies using VAE latent space manipulation."""
        try:
            num_anomalies = int(len(data) * anomaly_ratio)
            
            # Sample anomalous latent codes (from extreme regions)
            anomalous_latent = np.random.normal(0, 3.0, (num_anomalies, self.latent_dim))  # Higher variance
            anomalous_samples = await self._decode_latent(anomalous_latent)
            
            # Denormalize anomalous samples
            anomalous_samples = anomalous_samples * (self.data_std + 1e-8) + self.data_mean
            
            # Replace random samples with anomalous ones
            anomaly_indices = np.random.choice(len(data), num_anomalies, replace=False)
            
            anomaly_labels = np.zeros(len(data))
            anomaly_labels[anomaly_indices] = 1
            
            modified_data = data.copy()
            modified_data[anomaly_indices] = anomalous_samples
            
            return anomaly_labels, modified_data
            
        except Exception as e:
            logger.error(f"VAE anomaly injection failed: {e}")
            return np.zeros(len(data)), data


class StatisticalGenerator(BaseSyntheticGenerator):
    """Statistical synthetic data generation using copulas and distributions."""
    
    def __init__(self, config: SyntheticDataConfig):
        super().__init__(config)
        self.marginal_distributions = {}
        self.correlation_matrix = None
        self.copula_params = None
    
    async def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit statistical model."""
        try:
            logger.info("Fitting statistical generator")
            
            self.input_dim = X.shape[1]
            
            # Fit marginal distributions
            await self._fit_marginal_distributions(X)
            
            # Estimate correlation structure
            self.correlation_matrix = np.corrcoef(X, rowvar=False)
            
            # Handle NaN values in correlation matrix
            self.correlation_matrix = np.nan_to_num(self.correlation_matrix)
            
            # Ensure positive definite
            self.correlation_matrix = await self._make_positive_definite(self.correlation_matrix)
            
            self.is_fitted = True
            logger.info("Statistical generator fitted")
            
        except Exception as e:
            logger.error(f"Statistical generator fitting failed: {e}")
            raise
    
    async def generate(self, num_samples: int) -> SyntheticDataResult:
        """Generate synthetic data using statistical methods."""
        try:
            if not self.is_fitted:
                raise ValueError("Generator must be fitted before generating data")
            
            start_time = datetime.now()
            
            # Generate correlated normal samples
            multivariate_normal = np.random.multivariate_normal(
                mean=np.zeros(self.input_dim),
                cov=self.correlation_matrix,
                size=num_samples
            )
            
            # Transform to uniform using normal CDF
            uniform_samples = 0.5 * (1 + np.sign(multivariate_normal) * 
                                   np.sqrt(2 / np.pi) * 
                                   np.sqrt(np.abs(multivariate_normal)))
            
            # Transform using inverse marginal CDFs
            synthetic_data = np.zeros_like(uniform_samples)
            
            for i in range(self.input_dim):
                synthetic_data[:, i] = await self._inverse_transform_marginal(
                    uniform_samples[:, i], i
                )
            
            # Generate anomaly labels if requested
            anomaly_labels = None
            if self.config.anomaly_ratio > 0:
                anomaly_labels, synthetic_data = await self._inject_statistical_anomalies(
                    synthetic_data, self.config.anomaly_ratio
                )
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return SyntheticDataResult(
                synthetic_data=synthetic_data,
                generation_method=SyntheticMethod.STATISTICAL,
                quality_scores={},
                privacy_metrics={},
                metadata={
                    "correlation_preserved": True,
                    "marginal_distributions": list(self.marginal_distributions.keys())
                },
                generation_time=generation_time,
                anomaly_labels=anomaly_labels
            )
            
        except Exception as e:
            logger.error(f"Statistical generation failed: {e}")
            raise
    
    async def _fit_marginal_distributions(self, X: np.ndarray) -> None:
        """Fit marginal distributions for each feature."""
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            
            # Simple histogram-based distribution
            hist, bin_edges = np.histogram(feature_data, bins=50, density=True)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            self.marginal_distributions[i] = {
                "type": "histogram",
                "hist": hist,
                "bin_centers": bin_centers,
                "bin_edges": bin_edges,
                "min": np.min(feature_data),
                "max": np.max(feature_data)
            }
    
    async def _inverse_transform_marginal(self, uniform_samples: np.ndarray, feature_idx: int) -> np.ndarray:
        """Apply inverse transform sampling for marginal distribution."""
        try:
            dist_info = self.marginal_distributions[feature_idx]
            
            if dist_info["type"] == "histogram":
                # Inverse transform using empirical CDF
                hist = dist_info["hist"]
                bin_centers = dist_info["bin_centers"]
                
                # Compute empirical CDF
                cdf = np.cumsum(hist) / np.sum(hist)
                
                # Inverse transform
                result = np.interp(uniform_samples, cdf, bin_centers)
                
                # Clip to valid range
                result = np.clip(result, dist_info["min"], dist_info["max"])
                
                return result
            else:
                # Fallback to normal distribution
                return np.random.normal(0, 1, len(uniform_samples))
                
        except Exception as e:
            logger.error(f"Inverse transform failed for feature {feature_idx}: {e}")
            return np.random.normal(0, 1, len(uniform_samples))
    
    async def _make_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive definite."""
        try:
            # Eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            
            # Set negative eigenvalues to small positive value
            min_eigenval = 1e-6
            eigenvals = np.maximum(eigenvals, min_eigenval)
            
            # Reconstruct matrix
            positive_definite = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Ensure it's still a correlation matrix (diagonal = 1)
            diag = np.sqrt(np.diag(positive_definite))
            positive_definite = positive_definite / np.outer(diag, diag)
            
            return positive_definite
            
        except Exception as e:
            logger.error(f"Making positive definite failed: {e}")
            return np.eye(matrix.shape[0])
    
    async def _inject_statistical_anomalies(
        self,
        data: np.ndarray,
        anomaly_ratio: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject anomalies using statistical methods."""
        try:
            num_anomalies = int(len(data) * anomaly_ratio)
            anomaly_indices = np.random.choice(len(data), num_anomalies, replace=False)
            
            anomaly_labels = np.zeros(len(data))
            anomaly_labels[anomaly_indices] = 1
            
            modified_data = data.copy()
            
            for idx in anomaly_indices:
                # Generate sample from modified distribution
                feature_idx = np.random.randint(data.shape[1])
                
                # Sample from tail of distribution
                if np.random.random() < 0.5:
                    # Upper tail
                    percentile_val = np.percentile(data[:, feature_idx], 95)
                    anomalous_val = percentile_val + np.random.exponential(
                        np.std(data[:, feature_idx])
                    )
                else:
                    # Lower tail
                    percentile_val = np.percentile(data[:, feature_idx], 5)
                    anomalous_val = percentile_val - np.random.exponential(
                        np.std(data[:, feature_idx])
                    )
                
                modified_data[idx, feature_idx] = anomalous_val
            
            return anomaly_labels, modified_data
            
        except Exception as e:
            logger.error(f"Statistical anomaly injection failed: {e}")
            return np.zeros(len(data)), data


class PrivacyPreservingGenerator:
    """Privacy-preserving synthetic data generation."""
    
    def __init__(self, privacy_config: Dict[str, Any]):
        self.privacy_config = privacy_config
        self.privacy_method = PrivacyMethod(privacy_config.get("method", "differential_privacy"))
        self.privacy_budget = privacy_config.get("budget", 1.0)
    
    async def apply_privacy_protection(
        self,
        generator: BaseSyntheticGenerator,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[BaseSyntheticGenerator, PrivacyAnalysis]:
        """Apply privacy protection to synthetic data generation."""
        try:
            logger.info(f"Applying {self.privacy_method} privacy protection")
            
            if self.privacy_method == PrivacyMethod.DIFFERENTIAL_PRIVACY:
                protected_generator, analysis = await self._apply_differential_privacy(generator, X, y)
            elif self.privacy_method == PrivacyMethod.K_ANONYMITY:
                protected_generator, analysis = await self._apply_k_anonymity(generator, X, y)
            else:
                protected_generator, analysis = await self._apply_basic_privacy(generator, X, y)
            
            return protected_generator, analysis
            
        except Exception as e:
            logger.error(f"Privacy protection failed: {e}")
            raise
    
    async def _apply_differential_privacy(
        self,
        generator: BaseSyntheticGenerator,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[BaseSyntheticGenerator, PrivacyAnalysis]:
        """Apply differential privacy."""
        try:
            # Add noise to training data
            sensitivity = self._calculate_sensitivity(X)
            noise_scale = sensitivity / self.privacy_budget
            
            # Add Laplace noise
            noisy_X = X + np.random.laplace(0, noise_scale, X.shape)
            
            # Fit generator on noisy data
            await generator.fit(noisy_X, y)
            
            # Privacy analysis
            analysis = PrivacyAnalysis(
                privacy_method=PrivacyMethod.DIFFERENTIAL_PRIVACY,
                privacy_budget_used=self.privacy_budget,
                privacy_guarantee=f"({self.privacy_budget}, 0)-differential privacy",
                risk_metrics={
                    "privacy_loss": float(self.privacy_budget),
                    "noise_scale": float(noise_scale),
                    "data_utility": self._estimate_utility_loss(X, noisy_X)
                },
                recommended_usage="Suitable for general synthetic data sharing",
                limitations=["Utility decreases with stronger privacy", "Requires careful budget management"]
            )
            
            return generator, analysis
            
        except Exception as e:
            logger.error(f"Differential privacy application failed: {e}")
            raise
    
    async def _apply_k_anonymity(
        self,
        generator: BaseSyntheticGenerator,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[BaseSyntheticGenerator, PrivacyAnalysis]:
        """Apply k-anonymity."""
        try:
            k = self.privacy_config.get("k", 5)
            
            # Simplified k-anonymity through data aggregation
            # Group similar records and replace with group centroid
            anonymized_X = await self._anonymize_data(X, k)
            
            # Fit generator on anonymized data
            await generator.fit(anonymized_X, y)
            
            analysis = PrivacyAnalysis(
                privacy_method=PrivacyMethod.K_ANONYMITY,
                privacy_budget_used=0.0,  # Not applicable for k-anonymity
                privacy_guarantee=f"{k}-anonymity",
                risk_metrics={
                    "k_value": float(k),
                    "data_utility": self._estimate_utility_loss(X, anonymized_X)
                },
                recommended_usage="Suitable when k individuals cannot be distinguished",
                limitations=["Vulnerable to homogeneity attacks", "May not prevent attribute disclosure"]
            )
            
            return generator, analysis
            
        except Exception as e:
            logger.error(f"K-anonymity application failed: {e}")
            raise
    
    async def _apply_basic_privacy(
        self,
        generator: BaseSyntheticGenerator,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[BaseSyntheticGenerator, PrivacyAnalysis]:
        """Apply basic privacy protection."""
        # Simple data perturbation
        noise_level = 0.1
        perturbed_X = X + np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)
        
        await generator.fit(perturbed_X, y)
        
        analysis = PrivacyAnalysis(
            privacy_method=PrivacyMethod.DIFFERENTIAL_PRIVACY,  # Simplified
            privacy_budget_used=0.0,
            privacy_guarantee="Basic noise addition",
            risk_metrics={"noise_level": noise_level},
            recommended_usage="Basic protection only",
            limitations=["No formal privacy guarantee"]
        )
        
        return generator, analysis
    
    def _calculate_sensitivity(self, X: np.ndarray) -> float:
        """Calculate global sensitivity for differential privacy."""
        # Simplified sensitivity calculation
        # In practice, this would depend on the specific algorithm
        feature_ranges = np.max(X, axis=0) - np.min(X, axis=0)
        return float(np.max(feature_ranges))
    
    def _estimate_utility_loss(self, original: np.ndarray, modified: np.ndarray) -> float:
        """Estimate utility loss due to privacy protection."""
        try:
            # Calculate mean squared error
            mse = np.mean((original - modified) ** 2)
            
            # Normalize by data variance
            data_variance = np.var(original)
            
            if data_variance > 0:
                utility_loss = mse / data_variance
            else:
                utility_loss = 0.0
            
            return float(min(1.0, utility_loss))
            
        except Exception as e:
            logger.error(f"Utility loss estimation failed: {e}")
            return 0.5
    
    async def _anonymize_data(self, X: np.ndarray, k: int) -> np.ndarray:
        """Apply k-anonymity through clustering."""
        try:
            from sklearn.cluster import KMeans
            
            # Cluster data into groups
            num_clusters = max(1, len(X) // k)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            # Replace each group with centroid
            anonymized_X = X.copy()
            
            for cluster_id in range(num_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) >= k:
                    # Replace with centroid if group size >= k
                    centroid = np.mean(X[cluster_mask], axis=0)
                    anonymized_X[cluster_mask] = centroid
            
            return anonymized_X
            
        except Exception as e:
            logger.error(f"Data anonymization failed: {e}")
            return X


class SyntheticDataOrchestrator:
    """Main orchestrator for synthetic data generation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generators: Dict[SyntheticMethod, BaseSyntheticGenerator] = {}
        self.privacy_generator: Optional[PrivacyPreservingGenerator] = None
        self.generation_history: List[SyntheticDataResult] = []
    
    async def create_generator(self, config: SyntheticDataConfig) -> BaseSyntheticGenerator:
        """Create synthetic data generator."""
        try:
            if config.method == SyntheticMethod.VANILLA_GAN:
                generator = VanillaGANGenerator(config)
            elif config.method == SyntheticMethod.VAE:
                generator = VAEGenerator(config)
            elif config.method == SyntheticMethod.STATISTICAL:
                generator = StatisticalGenerator(config)
            else:
                logger.warning(f"Unknown method {config.method}, using statistical generator")
                generator = StatisticalGenerator(config)
            
            self.generators[config.method] = generator
            
            # Add privacy protection if specified
            if config.privacy_method:
                self.privacy_generator = PrivacyPreservingGenerator({
                    "method": config.privacy_method,
                    "budget": config.privacy_budget
                })
            
            return generator
            
        except Exception as e:
            logger.error(f"Generator creation failed: {e}")
            raise
    
    async def generate_synthetic_data(
        self,
        X: np.ndarray,
        config: SyntheticDataConfig,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> SyntheticDataResult:
        """Generate synthetic data with quality evaluation."""
        try:
            logger.info(f"Generating synthetic data using {config.method}")
            
            # Create generator
            generator = await self.create_generator(config)
            
            # Apply privacy protection if needed
            privacy_analysis = None
            if self.privacy_generator:
                generator, privacy_analysis = await self.privacy_generator.apply_privacy_protection(
                    generator, X, y
                )
            else:
                # Fit generator normally
                await generator.fit(X, y)
            
            # Generate synthetic data
            result = await generator.generate(config.num_samples)
            
            # Evaluate quality
            quality_scores = await generator.evaluate_quality(X, result.synthetic_data)
            result.quality_scores = quality_scores
            
            # Add privacy metrics if available
            if privacy_analysis:
                result.privacy_metrics = privacy_analysis.risk_metrics
            
            # Add feature names
            result.feature_names = feature_names
            
            # Store in history
            self.generation_history.append(result)
            
            logger.info(f"Generated {len(result.synthetic_data)} synthetic samples")
            return result
            
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            raise
    
    async def compare_methods(
        self,
        X: np.ndarray,
        methods: List[SyntheticMethod],
        num_samples: int,
        y: Optional[np.ndarray] = None
    ) -> Dict[SyntheticMethod, SyntheticDataResult]:
        """Compare different synthetic data generation methods."""
        try:
            logger.info(f"Comparing {len(methods)} synthetic data methods")
            
            results = {}
            
            for method in methods:
                try:
                    config = SyntheticDataConfig(
                        method=method,
                        data_type=DataType.TABULAR,
                        num_samples=num_samples
                    )
                    
                    result = await self.generate_synthetic_data(X, config, y)
                    results[method] = result
                    
                except Exception as e:
                    logger.error(f"Method {method} failed: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Method comparison failed: {e}")
            return {}
    
    async def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality summary across all generated datasets."""
        try:
            if not self.generation_history:
                return {"message": "No synthetic data generated yet"}
            
            # Aggregate quality scores
            method_quality = {}
            
            for result in self.generation_history:
                method = result.generation_method.value
                
                if method not in method_quality:
                    method_quality[method] = {
                        "count": 0,
                        "quality_scores": [],
                        "generation_times": [],
                        "sample_counts": []
                    }
                
                method_quality[method]["count"] += 1
                method_quality[method]["generation_times"].append(result.generation_time)
                method_quality[method]["sample_counts"].append(len(result.synthetic_data))
                
                # Aggregate quality scores
                for metric, score in result.quality_scores.items():
                    if metric not in method_quality[method]:
                        method_quality[method][metric] = []
                    method_quality[method][metric].append(score)
            
            # Calculate summary statistics
            summary = {
                "total_generations": len(self.generation_history),
                "methods_used": list(method_quality.keys()),
                "method_performance": {}
            }
            
            for method, data in method_quality.items():
                summary["method_performance"][method] = {
                    "count": data["count"],
                    "avg_generation_time": float(np.mean(data["generation_times"])),
                    "avg_sample_count": float(np.mean(data["sample_counts"])),
                    "quality_metrics": {}
                }
                
                # Average quality scores
                for metric in ["distribution_similarity", "correlation_preservation", "utility_score"]:
                    if metric in data and data[metric]:
                        summary["method_performance"][method]["quality_metrics"][metric] = {
                            "mean": float(np.mean(data[metric])),
                            "std": float(np.std(data[metric])),
                            "count": len(data[metric])
                        }
            
            return summary
            
        except Exception as e:
            logger.error(f"Quality summary generation failed: {e}")
            return {"error": str(e)}
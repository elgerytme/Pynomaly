#!/usr/bin/env python3
"""
Advanced Algorithms Example for Anomaly Detection Package

This example demonstrates advanced anomaly detection techniques including:
- Deep learning-based approaches (Autoencoders, VAE)
- PyOD library integration (40+ algorithms)
- Custom algorithm implementation
- Time series specific algorithms
- Hybrid ensemble approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some examples will be skipped.")

# Deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Deep learning examples will be skipped.")

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Some deep learning examples will be skipped.")

# PyOD library
try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA as PyOD_PCA
    from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS
    from pyod.models.abod import ABOD
    from pyod.models.feature_bagging import FeatureBagging
    from pyod.models.combination import aom, moa, average, maximization
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    print("Warning: PyOD not available. Install with: pip install pyod")

# Time series libraries
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Time series examples will be limited.")

# Import anomaly detection components
try:
    from anomaly_detection import DetectionService, EnsembleService, PyODAdapter, DeepLearningAdapter
    from anomaly_detection.domain.entities.detection_result import DetectionResult
except ImportError:
    print("Please install the anomaly_detection package first:")
    print("pip install -e .")
    exit(1)


class AutoencoderPyTorch(nn.Module):
    """PyTorch Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16]):
        super(AutoencoderPyTorch, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i-1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, latent_dim: int = 10, hidden_dim: int = 64):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


class CustomAnomalyDetector:
    """Custom anomaly detection algorithm implementation."""
    
    def __init__(self, method: str = 'distance_based', contamination: float = 0.1):
        self.method = method
        self.contamination = contamination
        self.threshold_ = None
        self.centroids_ = None
        
    def _distance_based_fit(self, X: np.ndarray):
        """Distance-based anomaly detection."""
        # Calculate centroid
        self.centroids_ = np.mean(X, axis=0)
        
        # Calculate distances to centroid
        distances = np.sqrt(np.sum((X - self.centroids_) ** 2, axis=1))
        
        # Set threshold based on contamination
        self.threshold_ = np.percentile(distances, (1 - self.contamination) * 100)
        
    def _density_based_fit(self, X: np.ndarray):
        """Density-based anomaly detection using k-nearest neighbors."""
        from sklearn.neighbors import NearestNeighbors
        
        k = min(10, len(X) // 10)  # Adaptive k
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(X)
        
        # Calculate average distance to k nearest neighbors
        distances, _ = nbrs.kneighbors(X)
        avg_distances = np.mean(distances, axis=1)
        
        self.threshold_ = np.percentile(avg_distances, (1 - self.contamination) * 100)
        self.model_ = nbrs
        
    def fit(self, X: np.ndarray):
        """Fit the custom anomaly detector."""
        if self.method == 'distance_based':
            self._distance_based_fit(X)
        elif self.method == 'density_based':
            self._density_based_fit(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if self.method == 'distance_based':
            distances = np.sqrt(np.sum((X - self.centroids_) ** 2, axis=1))
            return (distances > self.threshold_).astype(int) * 2 - 1  # Convert to -1/1
            
        elif self.method == 'density_based':
            distances, _ = self.model_.kneighbors(X)
            avg_distances = np.mean(distances, axis=1)
            return (avg_distances > self.threshold_).astype(int) * 2 - 1
            
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores."""
        if self.method == 'distance_based':
            distances = np.sqrt(np.sum((X - self.centroids_) ** 2, axis=1))
            return distances
            
        elif self.method == 'density_based':
            distances, _ = self.model_.kneighbors(X)
            return np.mean(distances, axis=1)


class AdvancedAnomalyDetector:
    """Main class for advanced anomaly detection examples."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def generate_complex_data(self, 
                            n_samples: int = 1000, 
                            n_features: int = 10,
                            anomaly_type: str = 'mixed') -> Tuple[np.ndarray, np.ndarray]:
        """Generate complex synthetic data with various anomaly types."""
        np.random.seed(42)
        
        # Generate normal data
        normal_samples = int(n_samples * 0.9)
        
        if anomaly_type == 'mixed':
            # Mix of different distributions
            X_normal = np.random.multivariate_normal(
                mean=np.zeros(n_features),
                cov=np.eye(n_features),
                size=normal_samples
            )
            
            # Different types of anomalies
            anomaly_samples = n_samples - normal_samples
            
            # Global anomalies (far from normal data)
            global_anomalies = np.random.multivariate_normal(
                mean=np.full(n_features, 5),
                cov=np.eye(n_features) * 0.5,
                size=anomaly_samples // 3
            )
            
            # Contextual anomalies (unusual in context)
            contextual_anomalies = np.random.multivariate_normal(
                mean=np.zeros(n_features),
                cov=np.eye(n_features) * 4,  # Higher variance
                size=anomaly_samples // 3
            )
            
            # Collective anomalies (unusual patterns)
            collective_anomalies = np.random.multivariate_normal(
                mean=np.full(n_features, -2),
                cov=np.eye(n_features) * 0.1,  # Very tight cluster
                size=anomaly_samples - (anomaly_samples // 3) * 2
            )
            
            X_anomaly = np.vstack([global_anomalies, contextual_anomalies, collective_anomalies])
            
        elif anomaly_type == 'time_series':
            # Time series with seasonal and trend components
            t = np.linspace(0, 4 * np.pi, n_samples)
            
            # Base signal with trend and seasonality
            trend = 0.1 * t
            seasonal = 2 * np.sin(t) + np.sin(2 * t)
            noise = np.random.normal(0, 0.5, n_samples)
            
            base_signal = trend + seasonal + noise
            
            # Add additional features
            X_normal = np.column_stack([
                base_signal,
                np.roll(base_signal, 1),  # Lagged feature
                np.gradient(base_signal),  # Derivative
                pd.Series(base_signal).rolling(window=10).mean().fillna(0),  # Moving average
            ])
            
            # Extend to desired feature count
            if n_features > 4:
                additional_features = np.random.randn(n_samples, n_features - 4) * 0.5
                X_normal = np.hstack([X_normal, additional_features])
            
            # Add anomalies (sudden spikes, level shifts)
            X_anomaly = X_normal.copy()
            anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
            
            for idx in anomaly_indices:
                if np.random.random() < 0.5:
                    # Spike anomaly
                    X_anomaly[idx] += np.random.normal(0, 3, n_features)
                else:
                    # Level shift
                    shift_duration = min(20, n_samples - idx)
                    X_anomaly[idx:idx+shift_duration] += np.random.normal(2, 0.5, n_features)
            
            return X_anomaly, (np.isin(np.arange(n_samples), anomaly_indices)).astype(int)
        
        # Combine normal and anomalous data
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]
    
    def train_autoencoder_pytorch(self, 
                                X_train: np.ndarray, 
                                X_val: np.ndarray,
                                epochs: int = 100,
                                learning_rate: float = 0.001) -> AutoencoderPyTorch:
        """Train PyTorch autoencoder for anomaly detection."""
        if not PYTORCH_AVAILABLE:
            print("PyTorch not available. Skipping autoencoder training.")
            return None
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        model = AutoencoderPyTorch(X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        model.train()
        for epoch in range(epochs):
            epoch_train_loss = 0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            # Validation loss
            model.eval()
            with torch.no_grad():
                val_reconstructed = model(X_val_tensor)
                val_loss = criterion(val_reconstructed, X_val_tensor).item()
            model.train()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        model.eval()
        return model
    
    def train_vae_pytorch(self,
                         X_train: np.ndarray,
                         X_val: np.ndarray,
                         latent_dim: int = 10,
                         epochs: int = 100,
                         learning_rate: float = 0.001) -> VariationalAutoencoder:
        """Train Variational Autoencoder for anomaly detection."""
        if not PYTORCH_AVAILABLE:
            print("PyTorch not available. Skipping VAE training.")
            return None
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare data
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize VAE
        vae = VariationalAutoencoder(X_train.shape[1], latent_dim).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
        
        def vae_loss(recon_x, x, mu, logvar):
            # Reconstruction loss
            recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            return recon_loss + kl_loss
        
        # Training loop
        train_losses = []
        val_losses = []
        
        vae.train()
        for epoch in range(epochs):
            epoch_train_loss = 0
            for batch_x, _ in train_loader:
                optimizer.zero_grad()
                recon_batch, mu, logvar = vae(batch_x)
                loss = vae_loss(recon_batch, batch_x, mu, logvar)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            # Validation loss
            vae.eval()
            with torch.no_grad():
                val_recon, val_mu, val_logvar = vae(X_val_tensor)
                val_loss = vae_loss(val_recon, X_val_tensor, val_mu, val_logvar).item()
            vae.train()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        vae.eval()
        return vae
    
    def detect_with_autoencoder(self, model, X: np.ndarray, contamination: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies using trained autoencoder."""
        if model is None:
            return np.ones(len(X)), np.zeros(len(X))
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            if isinstance(model, VariationalAutoencoder):
                reconstructed, _, _ = model(X_tensor)
            else:
                reconstructed = model(X_tensor)
            
            # Calculate reconstruction errors
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            scores = reconstruction_errors.cpu().numpy()
        
        # Determine threshold
        threshold = np.percentile(scores, (1 - contamination) * 100)
        predictions = (scores > threshold).astype(int) * 2 - 1  # Convert to -1/1
        
        return predictions, scores


def example_1_pyod_algorithms():
    """Example 1: PyOD algorithm showcase."""
    print("\n" + "="*60)
    print("Example 1: PyOD Algorithms Showcase")
    print("="*60)
    
    if not PYOD_AVAILABLE:
        print("PyOD not available. Install with: pip install pyod")
        return
    
    detector = AdvancedAnomalyDetector()
    
    # Generate complex data
    X, y = detector.generate_complex_data(n_samples=500, n_features=8, anomaly_type='mixed')
    X_scaled = detector.scaler.fit_transform(X)
    
    # PyOD algorithms to test
    algorithms = {
        'Isolation Forest': IForest(contamination=0.1, random_state=42),
        'Local Outlier Factor': LOF(contamination=0.1),
        'One-Class SVM': OCSVM(contamination=0.1),
        'PCA': PyOD_PCA(contamination=0.1),
        'k-NN': KNN(contamination=0.1),
        'HBOS': HBOS(contamination=0.1),
        'ABOD': ABOD(contamination=0.1),
        'Feature Bagging': FeatureBagging(contamination=0.1, random_state=42)
    }
    
    results = {}
    
    print("Testing PyOD algorithms...")
    for name, algorithm in algorithms.items():
        try:
            print(f"\nTraining {name}...")
            
            # Fit and predict
            algorithm.fit(X_scaled)
            y_pred = algorithm.predict(X_scaled)
            scores = algorithm.decision_function(X_scaled)
            
            # Calculate metrics
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, scores)
            
            results[name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'scores': scores
            }
            
            print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
            
        except Exception as e:
            print(f"  Error with {name}: {e}")
    
    # Visualize results
    if results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        # Performance comparison
        metrics_df = pd.DataFrame({name: [res['precision'], res['recall'], res['f1'], res['auc']] 
                                 for name, res in results.items()}, 
                                index=['Precision', 'Recall', 'F1', 'AUC'])
        
        metrics_df.T.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Algorithm Performance Comparison')
        axes[0].set_ylabel('Score')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Score distributions for top 3 algorithms
        top_algorithms = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
        
        for i, (name, result) in enumerate(top_algorithms):
            if i < 3:
                ax = axes[i+1]
                normal_scores = result['scores'][y == 0]
                anomaly_scores = result['scores'][y == 1]
                
                ax.hist(normal_scores, bins=30, alpha=0.7, label='Normal', density=True)
                ax.hist(anomaly_scores, bins=30, alpha=0.7, label='Anomaly', density=True)
                ax.set_title(f'{name} - Score Distribution')
                ax.set_xlabel('Anomaly Score')
                ax.set_ylabel('Density')
                ax.legend()
        
        plt.tight_layout()
        plt.show()


def example_2_deep_learning_autoencoders():
    """Example 2: Deep learning autoencoders for anomaly detection."""
    print("\n" + "="*60)
    print("Example 2: Deep Learning Autoencoders")
    print("="*60)
    
    if not PYTORCH_AVAILABLE:
        print("PyTorch not available. Skipping deep learning examples.")
        return
    
    detector = AdvancedAnomalyDetector()
    
    # Generate high-dimensional data
    X, y = detector.generate_complex_data(n_samples=1000, n_features=20, anomaly_type='mixed')
    X_scaled = detector.scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Use only normal data for training
    X_train_normal = X_train[y_train == 0]
    X_val_normal = X_test[y_test == 0][:50]  # Small validation set
    
    print(f"Training data shape: {X_train_normal.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train regular autoencoder
    print("\n2.1 Training Regular Autoencoder...")
    autoencoder = detector.train_autoencoder_pytorch(
        X_train_normal, X_val_normal, epochs=50, learning_rate=0.001
    )
    
    if autoencoder is not None:
        # Detect anomalies with autoencoder
        ae_predictions, ae_scores = detector.detect_with_autoencoder(
            autoencoder, X_test, contamination=0.1
        )
        
        # Calculate metrics
        from sklearn.metrics import classification_report
        ae_predictions_binary = (ae_predictions == -1).astype(int)
        
        print("\nAutoencoder Results:")
        print(classification_report(y_test, ae_predictions_binary, 
                                  target_names=['Normal', 'Anomaly']))
    
    # Train VAE
    print("\n2.2 Training Variational Autoencoder...")
    vae = detector.train_vae_pytorch(
        X_train_normal, X_val_normal, latent_dim=10, epochs=50, learning_rate=0.001
    )
    
    if vae is not None:
        # Detect anomalies with VAE
        vae_predictions, vae_scores = detector.detect_with_autoencoder(
            vae, X_test, contamination=0.1
        )
        
        vae_predictions_binary = (vae_predictions == -1).astype(int)
        
        print("\nVAE Results:")
        print(classification_report(y_test, vae_predictions_binary,
                                  target_names=['Normal', 'Anomaly']))
        
        # Compare autoencoder vs VAE
        if autoencoder is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # AE scores
            normal_ae = ae_scores[y_test == 0]
            anomaly_ae = ae_scores[y_test == 1]
            
            axes[0].hist(normal_ae, bins=30, alpha=0.7, label='Normal', density=True)
            axes[0].hist(anomaly_ae, bins=30, alpha=0.7, label='Anomaly', density=True)
            axes[0].set_title('Autoencoder Score Distribution')
            axes[0].set_xlabel('Reconstruction Error')
            axes[0].legend()
            
            # VAE scores
            normal_vae = vae_scores[y_test == 0]
            anomaly_vae = vae_scores[y_test == 1]
            
            axes[1].hist(normal_vae, bins=30, alpha=0.7, label='Normal', density=True)
            axes[1].hist(anomaly_vae, bins=30, alpha=0.7, label='Anomaly', density=True)
            axes[1].set_title('VAE Score Distribution')
            axes[1].set_xlabel('Reconstruction Error')
            axes[1].legend()
            
            plt.tight_layout()
            plt.show()


def example_3_custom_algorithms():
    """Example 3: Custom algorithm implementation."""
    print("\n" + "="*60)
    print("Example 3: Custom Algorithm Implementation")
    print("="*60)
    
    detector = AdvancedAnomalyDetector()
    
    # Generate data
    X, y = detector.generate_complex_data(n_samples=800, n_features=5, anomaly_type='mixed')
    X_scaled = detector.scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Test custom algorithms
    custom_algorithms = {
        'Distance-based': CustomAnomalyDetector('distance_based', contamination=0.1),
        'Density-based': CustomAnomalyDetector('density_based', contamination=0.1)
    }
    
    results = {}
    
    print("Testing custom algorithms...")
    for name, algorithm in custom_algorithms.items():
        print(f"\nTesting {name} algorithm...")
        
        # Fit on training data
        algorithm.fit(X_train)
        
        # Predict on test data
        y_pred = algorithm.predict(X_test)
        scores = algorithm.decision_function(X_test)
        
        # Convert predictions to binary
        y_pred_binary = (y_pred == -1).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        auc = roc_auc_score(y_test, scores)
        
        results[name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': y_pred_binary,
            'scores': scores
        }
        
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  AUC: {auc:.3f}")
    
    # Compare with standard algorithms if PyOD is available
    if PYOD_AVAILABLE:
        print("\nComparing with standard algorithms...")
        
        standard_algorithms = {
            'Isolation Forest': IForest(contamination=0.1, random_state=42),
            'LOF': LOF(contamination=0.1)
        }
        
        for name, algorithm in standard_algorithms.items():
            algorithm.fit(X_train)
            y_pred = algorithm.predict(X_test)
            scores = algorithm.decision_function(X_test)
            
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, scores)
            
            results[name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'scores': scores
            }
    
    # Visualization
    if results:
        # Performance comparison
        metrics_df = pd.DataFrame({name: [res['precision'], res['recall'], res['f1'], res['auc']] 
                                 for name, res in results.items()}, 
                                index=['Precision', 'Recall', 'F1', 'AUC'])
        
        plt.figure(figsize=(12, 6))
        metrics_df.T.plot(kind='bar', width=0.8)
        plt.title('Custom vs Standard Algorithms Performance')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


def example_4_time_series_algorithms():
    """Example 4: Time series specific algorithms."""
    print("\n" + "="*60)
    print("Example 4: Time Series Anomaly Detection")
    print("="*60)
    
    detector = AdvancedAnomalyDetector()
    
    # Generate time series data
    X, y = detector.generate_complex_data(n_samples=1000, n_features=6, anomaly_type='time_series')
    
    # Create time index
    dates = pd.date_range('2024-01-01', periods=len(X), freq='h')
    
    print(f"Generated time series with {len(X)} points and {X.shape[1]} features")
    print(f"Anomaly rate: {np.mean(y):.2%}")
    
    # Method 1: Statistical approach with seasonal decomposition
    if STATSMODELS_AVAILABLE:
        print("\n4.1 Statistical Approach with Seasonal Decomposition...")
        
        # Use first feature as main signal
        ts_data = pd.Series(X[:, 0], index=dates)
        
        # Seasonal decomposition
        decomposition = seasonal_decompose(ts_data, model='additive', period=24)
        
        # Calculate residuals
        residuals = decomposition.resid.dropna()
        
        # Anomaly detection on residuals
        residual_std = residuals.std()
        residual_mean = residuals.mean()
        threshold = 3 * residual_std  # 3-sigma rule
        
        anomalies_stat = np.abs(residuals - residual_mean) > threshold
        
        print(f"Statistical method detected {anomalies_stat.sum()} anomalies")
        
        # Visualize decomposition
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        decomposition.observed.plot(ax=axes[0], title='Original Time Series')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residuals')
        
        # Highlight anomalies on residuals
        anomaly_points = residuals[anomalies_stat]
        axes[3].scatter(anomaly_points.index, anomaly_points.values, 
                       color='red', s=50, alpha=0.8, label='Anomalies')
        axes[3].legend()
        
        plt.tight_layout()
        plt.show()
    
    # Method 2: Rolling window approach
    print("\n4.2 Rolling Window Approach...")
    
    # Create rolling features
    window_size = 24
    rolling_features = []
    
    for i in range(X.shape[1]):
        feature_series = pd.Series(X[:, i])
        rolling_mean = feature_series.rolling(window=window_size).mean()
        rolling_std = feature_series.rolling(window=window_size).std()
        rolling_diff = feature_series.diff()
        
        rolling_features.extend([rolling_mean, rolling_std, rolling_diff])
    
    # Combine features
    rolling_df = pd.concat(rolling_features, axis=1).dropna()
    X_rolling = rolling_df.values
    y_rolling = y[window_size:]  # Adjust labels for dropped rows
    
    # Apply anomaly detection
    if PYOD_AVAILABLE:
        detector_rolling = IForest(contamination=0.1, random_state=42)
        detector_rolling.fit(X_rolling)
        predictions_rolling = detector_rolling.predict(X_rolling)
        scores_rolling = detector_rolling.decision_function(X_rolling)
        
        # Calculate metrics
        from sklearn.metrics import classification_report
        print("\nRolling Window + Isolation Forest Results:")
        print(classification_report(y_rolling, predictions_rolling,
                                  target_names=['Normal', 'Anomaly']))
        
        # Visualize results
        plt.figure(figsize=(14, 8))
        
        # Plot original signal
        plt.subplot(2, 1, 1)
        plt.plot(dates, X[:, 0], 'b-', alpha=0.7, label='Original Signal')
        anomaly_indices = np.where(y == 1)[0]
        plt.scatter(dates[anomaly_indices], X[anomaly_indices, 0], 
                   color='red', s=50, alpha=0.8, label='True Anomalies')
        plt.title('Original Time Series with True Anomalies')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot detection results
        plt.subplot(2, 1, 2)
        adjusted_dates = dates[window_size:]
        plt.plot(adjusted_dates, X[window_size:, 0], 'b-', alpha=0.7, label='Signal')
        
        detected_indices = np.where(predictions_rolling == 1)[0]
        plt.scatter(adjusted_dates[detected_indices], X[window_size:][detected_indices, 0], 
                   color='orange', s=50, alpha=0.8, label='Detected Anomalies')
        
        plt.title('Anomaly Detection Results (Rolling Window Approach)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def example_5_hybrid_ensemble():
    """Example 5: Hybrid ensemble approaches."""
    print("\n" + "="*60)
    print("Example 5: Hybrid Ensemble Approaches")
    print("="*60)
    
    if not PYOD_AVAILABLE:
        print("PyOD not available. Skipping ensemble examples.")
        return
    
    detector = AdvancedAnomalyDetector()
    
    # Generate data
    X, y = detector.generate_complex_data(n_samples=800, n_features=10, anomaly_type='mixed')
    X_scaled = detector.scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Individual algorithms
    algorithms = {
        'IForest': IForest(contamination=0.1, random_state=42),
        'LOF': LOF(contamination=0.1),
        'OCSVM': OCSVM(contamination=0.1),
        'PCA': PyOD_PCA(contamination=0.1),
        'HBOS': HBOS(contamination=0.1)
    }
    
    # Train individual algorithms
    print("Training individual algorithms...")
    individual_scores = {}
    individual_predictions = {}
    
    for name, algorithm in algorithms.items():
        print(f"  Training {name}...")
        algorithm.fit(X_train)
        scores = algorithm.decision_function(X_test)
        predictions = algorithm.predict(X_test)
        
        individual_scores[name] = scores
        individual_predictions[name] = predictions
        
        # Calculate individual performance
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, predictions)
        print(f"    F1-Score: {f1:.3f}")
    
    # Ensemble methods
    print("\nApplying ensemble methods...")
    
    # Combine scores
    scores_matrix = np.column_stack(list(individual_scores.values()))
    
    # Method 1: Average combination
    avg_scores = average(scores_matrix)
    avg_threshold = np.percentile(avg_scores, 90)
    avg_predictions = (avg_scores > avg_threshold).astype(int)
    
    # Method 2: Maximum combination
    max_scores = maximization(scores_matrix)
    max_threshold = np.percentile(max_scores, 90)
    max_predictions = (max_scores > max_threshold).astype(int)
    
    # Method 3: Average of Maximum (AOM)
    aom_scores = aom(scores_matrix, n_buckets=5)
    aom_threshold = np.percentile(aom_scores, 90)
    aom_predictions = (aom_scores > aom_threshold).astype(int)
    
    # Method 4: Maximum of Average (MOA)
    moa_scores = moa(scores_matrix, n_buckets=5)
    moa_threshold = np.percentile(moa_scores, 90)
    moa_predictions = (moa_scores > moa_threshold).astype(int)
    
    # Evaluate ensemble methods
    ensemble_methods = {
        'Average': avg_predictions,
        'Maximum': max_predictions,
        'AOM': aom_predictions,
        'MOA': moa_predictions
    }
    
    print("\nEnsemble method performance:")
    ensemble_results = {}
    
    for method, predictions in ensemble_methods.items():
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        ensemble_results[method] = {'precision': precision, 'recall': recall, 'f1': f1}
        
        print(f"  {method}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # Compare individual vs ensemble performance
    individual_f1 = {}
    for name, predictions in individual_predictions.items():
        individual_f1[name] = f1_score(y_test, predictions)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual algorithm performance
    names = list(individual_f1.keys())
    f1_scores = list(individual_f1.values())
    
    ax1.bar(names, f1_scores, color='skyblue', alpha=0.7)
    ax1.set_title('Individual Algorithm Performance')
    ax1.set_ylabel('F1-Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Ensemble method performance
    ensemble_names = list(ensemble_results.keys())
    ensemble_f1 = [res['f1'] for res in ensemble_results.values()]
    
    ax2.bar(ensemble_names, ensemble_f1, color='lightcoral', alpha=0.7)
    ax2.set_title('Ensemble Method Performance')
    ax2.set_ylabel('F1-Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show best performing method
    best_individual = max(individual_f1, key=individual_f1.get)
    best_ensemble = max(ensemble_results, key=lambda x: ensemble_results[x]['f1'])
    
    print(f"\nBest individual algorithm: {best_individual} (F1={individual_f1[best_individual]:.3f})")
    print(f"Best ensemble method: {best_ensemble} (F1={ensemble_results[best_ensemble]['f1']:.3f})")


def main():
    """Run all advanced algorithm examples."""
    print("\n" + "="*60)
    print("ADVANCED ALGORITHMS FOR ANOMALY DETECTION")
    print("="*60)
    
    examples = [
        ("PyOD Algorithms Showcase", example_1_pyod_algorithms),
        ("Deep Learning Autoencoders", example_2_deep_learning_autoencoders),
        ("Custom Algorithm Implementation", example_3_custom_algorithms),
        ("Time Series Algorithms", example_4_time_series_algorithms),
        ("Hybrid Ensemble Approaches", example_5_hybrid_ensemble)
    ]
    
    while True:
        print("\nAvailable Examples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"{i}. {name}")
        print("0. Exit")
        
        try:
            choice = int(input("\nSelect an example (0-5): "))
            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= len(examples):
                examples[choice-1][1]()
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error running example: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
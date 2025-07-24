"""Deep learning adapter for anomaly detection using autoencoders."""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Try to import deep learning frameworks
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


class DeepLearningAdapter:
    """Adapter for deep learning-based anomaly detection.
    
    Supports autoencoder-based anomaly detection using TensorFlow or PyTorch.
    Anomalies are detected based on reconstruction error.
    """
    
    def __init__(
        self,
        framework: Literal["tensorflow", "pytorch"] = "tensorflow",
        hidden_dims: list[int] | None = None,
        contamination: float = 0.1,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        **kwargs: Any
    ):
        """Initialize deep learning adapter.
        
        Args:
            framework: Deep learning framework to use
            hidden_dims: Hidden layer dimensions for autoencoder
            contamination: Expected proportion of anomalies
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            **kwargs: Additional framework-specific parameters
        """
        self.framework = framework
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.contamination = contamination
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        
        self.model = None
        self._threshold = None
        self._fitted = False
        
        # Check framework availability
        if framework == "tensorflow" and not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for tensorflow framework")
        elif framework == "pytorch" and not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for pytorch framework")
    
    def fit(self, data: npt.NDArray[np.floating]) -> DeepLearningAdapter:
        """Fit autoencoder on training data.
        
        Args:
            data: Training data of shape (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        if self.framework == "tensorflow":
            self._fit_tensorflow(data)
        elif self.framework == "pytorch":
            self._fit_pytorch(data)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
            
        # Calculate threshold based on training reconstruction errors
        self._calculate_threshold(data)
        self._fitted = True
        
        return self
    
    def predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Predict anomalies based on reconstruction error.
        
        Args:
            data: Data to predict on
            
        Returns:
            Binary predictions where 1 indicates anomaly
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
            
        scores = self.decision_function(data)
        return (scores > self._threshold).astype(int)
    
    def decision_function(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Get reconstruction errors as anomaly scores.
        
        Args:
            data: Data to score
            
        Returns:
            Reconstruction errors (higher = more anomalous)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before scoring")
            
        if self.framework == "tensorflow":
            return self._score_tensorflow(data)
        elif self.framework == "pytorch":
            return self._score_pytorch(data)
        else:
            raise ValueError(f"Unknown framework: {self.framework}")
    
    def fit_predict(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]:
        """Fit and predict in one step."""
        self.fit(data)
        return self.predict(data)
    
    def _fit_tensorflow(self, data: npt.NDArray[np.floating]) -> None:
        """Fit using TensorFlow/Keras."""
        input_dim = data.shape[1]
        
        # Build autoencoder architecture
        encoder_layers = []
        decoder_layers = []
        
        # Encoder
        current_dim = input_dim
        for hidden_dim in self.hidden_dims:
            encoder_layers.append(tf.keras.layers.Dense(hidden_dim, activation='relu'))
            current_dim = hidden_dim
        
        # Decoder (reverse of encoder)
        for hidden_dim in reversed(self.hidden_dims[:-1]):
            decoder_layers.append(tf.keras.layers.Dense(hidden_dim, activation='relu'))
        decoder_layers.append(tf.keras.layers.Dense(input_dim, activation='linear'))
        
        # Create model
        inputs = tf.keras.layers.Input(shape=(input_dim,))
        x = inputs
        
        # Encoder
        for layer in encoder_layers:
            x = layer(x)
        
        # Decoder
        for layer in decoder_layers:
            x = layer(x)
        
        self.model = tf.keras.Model(inputs, x)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        # Train model
        self.model.fit(
            data, data,  # Autoencoder: input = target
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
    
    def _fit_pytorch(self, data: npt.NDArray[np.floating]) -> None:
        """Fit using PyTorch."""
        input_dim = data.shape[1]
        
        # Create autoencoder model
        class Autoencoder(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: list[int]):
                super().__init__()
                
                # Encoder
                encoder_layers = []
                current_dim = input_dim
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU()
                    ])
                    current_dim = hidden_dim
                self.encoder = nn.Sequential(*encoder_layers)
                
                # Decoder
                decoder_layers = []
                for hidden_dim in reversed(hidden_dims[:-1]):
                    decoder_layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU()
                    ])
                    current_dim = hidden_dim
                decoder_layers.append(nn.Linear(current_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        self.model = Autoencoder(input_dim, self.hidden_dims)
        
        # Convert data to tensor
        data_tensor = torch.FloatTensor(data)
        dataset = torch.utils.data.TensorDataset(data_tensor, data_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_data, batch_target in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_data)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.6f}")
    
    def _score_tensorflow(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Calculate reconstruction errors using TensorFlow."""
        reconstructed = self.model.predict(data, verbose=0)
        errors = np.mean(np.square(data - reconstructed), axis=1)
        return errors
    
    def _score_pytorch(self, data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Calculate reconstruction errors using PyTorch."""
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)
            reconstructed = self.model(data_tensor)
            errors = torch.mean(torch.square(data_tensor - reconstructed), dim=1)
            return errors.numpy()
    
    def _calculate_threshold(self, data: npt.NDArray[np.floating]) -> None:
        """Calculate threshold based on contamination rate."""
        scores = self.decision_function(data)
        threshold_percentile = (1 - self.contamination) * 100
        self._threshold = np.percentile(scores, threshold_percentile)
        
        logger.info(f"Set threshold to {self._threshold:.6f} "
                   f"(contamination={self.contamination})")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if not self._fitted:
            return "Model not fitted"
            
        if self.framework == "tensorflow":
            from io import StringIO
            import sys
            
            # Capture model summary
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            return buffer.getvalue()
        elif self.framework == "pytorch":
            return str(self.model)
        else:
            return "Unknown framework"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {
            "framework": self.framework,
            "hidden_dims": self.hidden_dims,
            "contamination": self.contamination,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "threshold": self._threshold
        }
    
    @staticmethod
    def get_available_frameworks() -> list[str]:
        """Get list of available deep learning frameworks."""
        frameworks = []
        if TF_AVAILABLE:
            frameworks.append("tensorflow")
        if TORCH_AVAILABLE:
            frameworks.append("pytorch")
        return frameworks
    
    @staticmethod
    def create_default_architecture(input_dim: int) -> list[int]:
        """Create default autoencoder architecture based on input dimension."""
        # Simple heuristic for autoencoder architecture
        if input_dim <= 10:
            return [8, 4]
        elif input_dim <= 50:
            return [32, 16, 8]
        elif input_dim <= 100:
            return [64, 32, 16]
        else:
            return [128, 64, 32, 16]
    
    def __str__(self) -> str:
        """String representation."""
        return (f"DeepLearningAdapter(framework='{self.framework}', "
                f"hidden_dims={self.hidden_dims}, fitted={self._fitted})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"DeepLearningAdapter(framework='{self.framework}', "
                f"hidden_dims={self.hidden_dims}, contamination={self.contamination}, "
                f"epochs={self.epochs}, batch_size={self.batch_size})")
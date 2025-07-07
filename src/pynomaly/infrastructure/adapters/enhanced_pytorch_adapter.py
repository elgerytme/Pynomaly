"""
Enhanced PyTorch adapter with advanced deep learning models for anomaly detection.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from pynomaly.domain.entities.anomaly import Anomaly, Score
from pynomaly.domain.entities.dataset import Dataset
from pynomaly.domain.entities.detector import Detector, DetectorProtocol
from pynomaly.domain.entities.detection_result import DetectionResult
from pynomaly.shared.exceptions import ModelTrainingError, ModelPredictionError
from pynomaly.shared.types import DetectorId


class TransformerAnomalyDetector(nn.Module):
    """Transformer-based anomaly detection for time series and sequential data."""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super(TransformerAnomalyDetector, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)
        )
        
        # Anomaly score head
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim)
        seq_len, batch_size, _ = x.transpose(0, 1).shape
        
        # Project input to d_model
        x = self.input_projection(x.transpose(0, 1))  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        
        # Apply transformer
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Reconstruction
        reconstruction = self.reconstruction_head(transformer_out)
        
        # Anomaly scores
        anomaly_scores = self.anomaly_head(transformer_out)
        
        return reconstruction.transpose(0, 1), anomaly_scores.transpose(0, 1)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MemoryAugmentedAutoEncoder(nn.Module):
    """Memory-augmented autoencoder with attention mechanism."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], memory_size: int = 100, shrink_threshold: float = 0.1):
        super(MemoryAugmentedAutoEncoder, self).__init__()
        
        self.memory_size = memory_size
        self.shrink_threshold = shrink_threshold
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Memory module
        self.memory = nn.Parameter(torch.randn(memory_size, prev_dim), requires_grad=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=prev_dim, num_heads=8, batch_first=True)
        
        # Decoder
        decoder_layers = []
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Memory attention
        batch_size = encoded.size(0)
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        encoded_expanded = encoded.unsqueeze(1)
        
        attended_memory, attention_weights = self.attention(
            encoded_expanded, memory_expanded, memory_expanded
        )
        
        # Combine with original encoding
        enhanced_encoding = encoded + attended_memory.squeeze(1)
        
        # Decode
        reconstructed = self.decoder(enhanced_encoding)
        
        return reconstructed, attention_weights, encoded
    
    def update_memory(self, encoded_samples, attention_weights, threshold=0.1):
        """Update memory based on attention weights and encoding similarity."""
        with torch.no_grad():
            # Find the most attended memory slots
            max_attention, max_indices = torch.max(attention_weights.squeeze(1), dim=1)
            
            # Update memory for high-attention samples
            for i, (sample, idx, attention) in enumerate(zip(encoded_samples, max_indices, max_attention)):
                if attention > threshold:
                    # Exponential moving average update
                    self.memory[idx] = 0.9 * self.memory[idx] + 0.1 * sample


class GraphNeuralAnomalyDetector(nn.Module):
    """Graph neural network for anomaly detection in graph-structured data."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super(GraphNeuralAnomalyDetector, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphConvLayer(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.conv_layers.append(GraphConvLayer(hidden_dim, hidden_dim))
        
        self.conv_layers.append(GraphConvLayer(hidden_dim, hidden_dim))
        
        # Anomaly score predictor
        self.anomaly_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_weight=None):
        h = x
        
        for i, conv_layer in enumerate(self.conv_layers):
            h = conv_layer(h, edge_index, edge_weight)
            if i < len(self.conv_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Global pooling (mean pooling across nodes)
        batch_size = h.size(0)
        h_pooled = torch.mean(h, dim=0, keepdim=True).expand(batch_size, -1)
        
        # Anomaly scores
        anomaly_scores = self.anomaly_predictor(h_pooled)
        
        return anomaly_scores, h


class GraphConvLayer(nn.Module):
    """Simple graph convolution layer."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, edge_index, edge_weight=None):
        # Simple message passing
        row, col = edge_index
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        
        # Aggregate messages
        out = torch.zeros_like(x[:, :self.linear.out_features])
        for i in range(edge_index.size(1)):
            src, dst = row[i], col[i]
            out[dst] += edge_weight[i] * x[src]
        
        # Apply linear transformation
        out = self.linear(out)
        
        return out


class CapsuleAnomalyDetector(nn.Module):
    """Capsule network for anomaly detection."""
    
    def __init__(self, input_dim: int, num_capsules: int = 8, capsule_dim: int = 16, num_iterations: int = 3):
        super(CapsuleAnomalyDetector, self).__init__()
        
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_iterations = num_iterations
        
        # Primary capsules
        self.primary_capsules = nn.Linear(input_dim, num_capsules * capsule_dim)
        
        # Routing weights
        self.routing_weights = nn.Parameter(
            torch.randn(num_capsules, num_capsules, capsule_dim, capsule_dim)
        )
        
        # Reconstruction network
        self.reconstruction = nn.Sequential(
            nn.Linear(num_capsules * capsule_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Primary capsules
        primary = self.primary_capsules(x)
        primary = primary.view(batch_size, self.num_capsules, self.capsule_dim)
        
        # Dynamic routing
        capsules = self.dynamic_routing(primary)
        
        # Reconstruction
        capsules_flat = capsules.view(batch_size, -1)
        reconstructed = self.reconstruction(capsules_flat)
        
        # Anomaly score based on capsule lengths
        capsule_lengths = torch.norm(capsules, dim=2)
        anomaly_score = 1.0 - torch.mean(capsule_lengths, dim=1, keepdim=True)
        
        return reconstructed, anomaly_score, capsules
    
    def dynamic_routing(self, primary_capsules):
        """Dynamic routing algorithm for capsules."""
        batch_size = primary_capsules.size(0)
        
        # Initialize routing logits
        b = torch.zeros(batch_size, self.num_capsules, self.num_capsules, device=primary_capsules.device)
        
        for iteration in range(self.num_iterations):
            # Softmax routing coefficients
            c = F.softmax(b, dim=2)
            
            # Weighted sum of prediction vectors
            s = torch.sum(c.unsqueeze(-1) * primary_capsules.unsqueeze(1), dim=2)
            
            # Squash function
            v = self.squash(s)
            
            if iteration < self.num_iterations - 1:
                # Update routing logits
                agreement = torch.sum(primary_capsules.unsqueeze(1) * v.unsqueeze(2), dim=-1)
                b = b + agreement
        
        return v
    
    def squash(self, s):
        """Squash function for capsules."""
        s_norm = torch.norm(s, dim=-1, keepdim=True)
        return (s_norm ** 2 / (1 + s_norm ** 2)) * (s / (s_norm + 1e-8))


class EnhancedPyTorchAdapter(DetectorProtocol):
    """Enhanced PyTorch adapter with advanced deep learning models."""
    
    AVAILABLE_ALGORITHMS = {
        "transformer": TransformerAnomalyDetector,
        "memory_ae": MemoryAugmentedAutoEncoder,
        "graph_nn": GraphNeuralAnomalyDetector,
        "capsule_net": CapsuleAnomalyDetector,
    }
    
    def __init__(
        self,
        algorithm: str = "transformer",
        parameters: Optional[Dict[str, Any]] = None,
        device: str = "auto"
    ):
        self.algorithm = algorithm.lower()
        self.parameters = parameters or {}
        self.device = self._get_device(device)
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_history = []
        
        # Set random seeds for reproducibility
        if "random_state" in self.parameters:
            torch.manual_seed(self.parameters["random_state"])
            np.random.seed(self.parameters["random_state"])
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate PyTorch device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    async def train(self, dataset: Dataset) -> Detector:
        """Train the enhanced deep learning model."""
        try:
            # Prepare data
            X_train, X_val = await self._prepare_training_data(dataset)
            
            # Create model
            input_dim = X_train.shape[-1]  # Last dimension for potential sequence data
            self.model = self._create_model(input_dim)
            self.model.to(self.device)
            
            # Train model
            training_stats = await self._train_model(X_train, X_val)
            
            self.is_trained = True
            
            # Create detector entity
            detector = Detector(
                id=DetectorId(str(uuid.uuid4())),
                name=f"Enhanced_PyTorch_{self.algorithm.title()}",
                algorithm=f"enhanced_pytorch_{self.algorithm}",
                parameters=self.parameters,
                is_trained=True,
                training_metadata={
                    "input_features": input_dim,
                    "device": str(self.device),
                    "training_samples": len(X_train),
                    "validation_samples": len(X_val),
                    **training_stats
                },
                created_at=datetime.utcnow()
            )
            
            return detector
            
        except Exception as e:
            raise ModelTrainingError(f"Enhanced PyTorch training failed: {str(e)}")
    
    def _create_model(self, input_dim: int) -> nn.Module:
        """Create the appropriate advanced model."""
        if self.algorithm == "transformer":
            d_model = self.parameters.get("d_model", 128)
            nhead = self.parameters.get("nhead", 8)
            num_layers = self.parameters.get("num_layers", 3)
            dropout = self.parameters.get("dropout", 0.1)
            return TransformerAnomalyDetector(input_dim, d_model, nhead, num_layers, dropout)
        
        elif self.algorithm == "memory_ae":
            hidden_dims = self.parameters.get("hidden_dims", [256, 128, 64])
            memory_size = self.parameters.get("memory_size", 100)
            shrink_threshold = self.parameters.get("shrink_threshold", 0.1)
            return MemoryAugmentedAutoEncoder(input_dim, hidden_dims, memory_size, shrink_threshold)
        
        elif self.algorithm == "graph_nn":
            hidden_dim = self.parameters.get("hidden_dim", 64)
            num_layers = self.parameters.get("num_layers", 3)
            dropout = self.parameters.get("dropout", 0.1)
            return GraphNeuralAnomalyDetector(input_dim, hidden_dim, num_layers, dropout)
        
        elif self.algorithm == "capsule_net":
            num_capsules = self.parameters.get("num_capsules", 8)
            capsule_dim = self.parameters.get("capsule_dim", 16)
            num_iterations = self.parameters.get("num_iterations", 3)
            return CapsuleAnomalyDetector(input_dim, num_capsules, capsule_dim, num_iterations)
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    async def _prepare_training_data(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and preprocess training data for advanced models."""
        # Convert to numpy array
        data = dataset.data.select_dtypes(include=[np.number]).values
        
        # Handle missing values
        data = np.nan_to_num(data, nan=0.0)
        
        # Scale data
        scaler_type = self.parameters.get("scaler", "standard")
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        
        data_scaled = self.scaler.fit_transform(data)
        
        # For transformer, reshape for sequence data if needed
        if self.algorithm == "transformer":
            seq_len = self.parameters.get("sequence_length", 10)
            if len(data_scaled) >= seq_len:
                sequences = []
                for i in range(len(data_scaled) - seq_len + 1):
                    sequences.append(data_scaled[i:i+seq_len])
                data_scaled = np.array(sequences)
        
        # Split data
        test_size = self.parameters.get("validation_split", 0.2)
        X_train, X_val = train_test_split(
            data_scaled, 
            test_size=test_size, 
            random_state=self.parameters.get("random_state", 42)
        )
        
        return X_train, X_val
    
    async def _train_model(self, X_train: np.ndarray, X_val: np.ndarray) -> Dict[str, Any]:
        """Train the advanced model with algorithm-specific logic."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        
        # Create data loaders
        batch_size = self.parameters.get("batch_size", 64)
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and loss
        learning_rate = self.parameters.get("learning_rate", 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        epochs = self.parameters.get("epochs", 100)
        patience = self.parameters.get("early_stopping_patience", 15)
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                loss = self._compute_loss(batch_X, batch_y)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    loss = self._compute_loss(batch_X, batch_y)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Restore best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return {
            "epochs_trained": epoch + 1,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": best_val_loss,
            "training_history": {
                "train_losses": train_losses,
                "val_losses": val_losses
            }
        }
    
    def _compute_loss(self, batch_X, batch_y):
        """Compute loss based on the algorithm."""
        if self.algorithm == "transformer":
            reconstructed, anomaly_scores = self.model(batch_X)
            recon_loss = F.mse_loss(reconstructed, batch_y)
            
            # Add anomaly score regularization (encourage normal data to have low scores)
            anomaly_reg = torch.mean(anomaly_scores)
            return recon_loss + 0.1 * anomaly_reg
            
        elif self.algorithm == "memory_ae":
            reconstructed, attention_weights, encoded = self.model(batch_X)
            recon_loss = F.mse_loss(reconstructed, batch_y)
            
            # Update memory
            if self.model.training:
                self.model.update_memory(encoded, attention_weights)
            
            # Add memory diversity loss
            memory_diversity = -torch.mean(torch.var(self.model.memory, dim=0))
            return recon_loss + 0.01 * memory_diversity
            
        elif self.algorithm == "graph_nn":
            # For simplified graph case, treat each sample as a node
            # Create a simple adjacency matrix (fully connected for now)
            batch_size = batch_X.size(0)
            edge_index = torch.combinations(torch.arange(batch_size), r=2).t().to(batch_X.device)
            
            anomaly_scores, embeddings = self.model(batch_X, edge_index)
            
            # Loss based on anomaly scores (we want normal data to have low scores)
            anomaly_loss = torch.mean(anomaly_scores)
            
            # Add embedding smoothness constraint
            src, dst = edge_index
            smoothness_loss = torch.mean((embeddings[src] - embeddings[dst]) ** 2)
            
            return anomaly_loss + 0.1 * smoothness_loss
            
        elif self.algorithm == "capsule_net":
            reconstructed, anomaly_scores, capsules = self.model(batch_X)
            
            # Reconstruction loss
            recon_loss = F.mse_loss(reconstructed, batch_y)
            
            # Margin loss for capsules (encourage diverse representations)
            capsule_lengths = torch.norm(capsules, dim=2)
            margin_loss = torch.mean(torch.max(torch.zeros_like(capsule_lengths), 
                                             0.9 - capsule_lengths) ** 2)
            
            return recon_loss + 0.1 * margin_loss
        
        else:
            return F.mse_loss(batch_X, batch_y)
    
    async def detect(self, dataset: Dataset) -> DetectionResult:
        """Detect anomalies using the trained advanced model."""
        if not self.is_trained or self.model is None:
            raise ModelPredictionError("Model must be trained before detection")
        
        try:
            # Prepare data (similar to training preparation)
            data = dataset.data.select_dtypes(include=[np.number]).values
            data = np.nan_to_num(data, nan=0.0)
            
            if self.scaler is None:
                raise ModelPredictionError("Scaler not available - model not properly trained")
            
            data_scaled = self.scaler.transform(data)
            
            # Handle sequence data for transformer
            if self.algorithm == "transformer":
                seq_len = self.parameters.get("sequence_length", 10)
                if len(data_scaled) >= seq_len:
                    sequences = []
                    for i in range(len(data_scaled) - seq_len + 1):
                        sequences.append(data_scaled[i:i+seq_len])
                    data_scaled = np.array(sequences)
            
            # Convert to tensor
            data_tensor = torch.FloatTensor(data_scaled).to(self.device)
            
            # Get anomaly scores
            self.model.eval()
            with torch.no_grad():
                anomaly_scores = self._compute_anomaly_scores(data_tensor)
            
            # Convert to numpy
            scores_np = anomaly_scores.cpu().numpy()
            
            # Determine threshold
            threshold = self._calculate_threshold(scores_np)
            
            # Create anomaly objects
            anomalies = []
            for i, score in enumerate(scores_np):
                if score > threshold:
                    anomaly = Anomaly(
                        index=i,
                        score=Score(float(score)),
                        features=dataset.data.iloc[i].to_dict() if i < len(dataset.data) else {},
                        metadata={
                            "algorithm": f"enhanced_pytorch_{self.algorithm}",
                            "threshold": threshold,
                            "anomaly_score": float(score)
                        }
                    )
                    anomalies.append(anomaly)
            
            # Create detection result
            result = DetectionResult(
                anomalies=anomalies,
                algorithm=f"enhanced_pytorch_{self.algorithm}",
                threshold=threshold,
                metadata={
                    "total_samples": len(data),
                    "anomaly_count": len(anomalies),
                    "anomaly_rate": len(anomalies) / len(data) * 100,
                    "mean_score": float(np.mean(scores_np)),
                    "std_score": float(np.std(scores_np)),
                    "device": str(self.device),
                    "model_type": "enhanced_deep_learning"
                },
                execution_time=0.0  # Will be set by caller
            )
            
            return result
            
        except Exception as e:
            raise ModelPredictionError(f"Enhanced PyTorch detection failed: {str(e)}")
    
    def _compute_anomaly_scores(self, data_tensor):
        """Compute anomaly scores based on the algorithm."""
        if self.algorithm == "transformer":
            reconstructed, anomaly_scores = self.model(data_tensor)
            # Combine reconstruction error with direct anomaly scores
            recon_error = torch.mean((data_tensor - reconstructed) ** 2, dim=(1, 2))
            return 0.7 * recon_error + 0.3 * anomaly_scores.squeeze()
            
        elif self.algorithm == "memory_ae":
            reconstructed, attention_weights, encoded = self.model(data_tensor)
            # Use reconstruction error and memory attention as anomaly indicators
            recon_error = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
            memory_distance = torch.min(torch.cdist(encoded.unsqueeze(1), 
                                                  self.model.memory.unsqueeze(0)), dim=2)[0]
            return recon_error + 0.1 * memory_distance
            
        elif self.algorithm == "graph_nn":
            batch_size = data_tensor.size(0)
            edge_index = torch.combinations(torch.arange(batch_size), r=2).t().to(data_tensor.device)
            
            anomaly_scores, _ = self.model(data_tensor, edge_index)
            return anomaly_scores.squeeze()
            
        elif self.algorithm == "capsule_net":
            reconstructed, anomaly_scores, capsules = self.model(data_tensor)
            # Combine reconstruction error with capsule-based anomaly scores
            recon_error = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
            return 0.6 * recon_error + 0.4 * anomaly_scores.squeeze()
        
        else:
            # Fallback to reconstruction error
            reconstructed = self.model(data_tensor)
            return torch.mean((data_tensor - reconstructed) ** 2, dim=1)
    
    def _calculate_threshold(self, scores: np.ndarray) -> float:
        """Calculate anomaly threshold based on scores."""
        contamination = self.parameters.get("contamination", 0.1)
        
        if contamination > 0:
            # Use percentile-based threshold
            threshold = np.percentile(scores, (1 - contamination) * 100)
        else:
            # Use statistical threshold (mean + 2*std)
            threshold = np.mean(scores) + 2 * np.std(scores)
        
        return float(threshold)
    
    @staticmethod
    def get_available_algorithms() -> List[str]:
        """Get list of available enhanced PyTorch algorithms."""
        return list(EnhancedPyTorchAdapter.AVAILABLE_ALGORITHMS.keys())
    
    @staticmethod
    def get_algorithm_parameters(algorithm: str) -> Dict[str, Any]:
        """Get default parameters for an algorithm."""
        base_params = {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "early_stopping_patience": 15,
            "contamination": 0.1,
            "scaler": "standard",
            "random_state": 42
        }
        
        algorithm_specific = {
            "transformer": {
                "d_model": 128,
                "nhead": 8,
                "num_layers": 3,
                "dropout": 0.1,
                "sequence_length": 10
            },
            "memory_ae": {
                "hidden_dims": [256, 128, 64],
                "memory_size": 100,
                "shrink_threshold": 0.1
            },
            "graph_nn": {
                "hidden_dim": 64,
                "num_layers": 3,
                "dropout": 0.1
            },
            "capsule_net": {
                "num_capsules": 8,
                "capsule_dim": 16,
                "num_iterations": 3
            }
        }
        
        return {**base_params, **algorithm_specific.get(algorithm, {})}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the trained model."""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "algorithm": self.algorithm,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "training_history": self.training_history
        }
"""Neural network adapters for neuro-symbolic models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from numpy.typing import NDArray

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("Transformers not available. Transformer models will be disabled.")

from ...domain.value_objects.reasoning_result import ReasoningResult


class NeuralBackbone(ABC):
    """Abstract base class for neural network backbones."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network."""
        pass
    
    @abstractmethod
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate embeddings from the neural network."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        pass


class TransformerBackbone(NeuralBackbone):
    """Transformer-based neural backbone for sequence processing."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_dim: int = 256,
        num_classes: int = 2,
        device: Optional[str] = None
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library required for TransformerBackbone")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Load pre-trained transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Add classification head
        self.classifier = nn.Linear(
            self.transformer.config.hidden_size, 
            hidden_dim
        )
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.1)
        
        # Move to device
        self.transformer.to(self.device)
        self.classifier.to(self.device)
        self.output_layer.to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(
            list(self.transformer.parameters()) + 
            list(self.classifier.parameters()) + 
            list(self.output_layer.parameters()),
            lr=2e-5
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: Union[torch.Tensor, List[str]]) -> torch.Tensor:
        """Forward pass through transformer."""
        if isinstance(x, list):  # Text input
            inputs = self.tokenizer(
                x, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            outputs = self.transformer(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        else:
            embeddings = x.to(self.device)
        
        hidden = self.dropout(torch.relu(self.classifier(embeddings)))
        output = self.output_layer(hidden)
        return output
    
    def get_embeddings(self, x: Union[torch.Tensor, List[str]]) -> torch.Tensor:
        """Get transformer embeddings."""
        if isinstance(x, list):  # Text input
            inputs = self.tokenizer(
                x, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.transformer(**inputs)
                return outputs.last_hidden_state[:, 0]  # CLS token
        else:
            return x.to(self.device)
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y).float().mean()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item()
        }


class CNNBackbone(NeuralBackbone):
    """Convolutional neural network backbone for tabular/image data."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64, 32],
        num_classes: int = 2,
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build CNN layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.network = nn.Sequential(*layers).to(self.device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN."""
        return self.network(x.to(self.device))
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate embeddings."""
        x = x.to(self.device)
        # Return activations from second-to-last layer
        for i, layer in enumerate(self.network[:-1]):
            x = layer(x)
        return x
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y).float().mean()
        
        return {
            "loss": loss.item(),
            "accuracy": accuracy.item()
        }


class AutoencoderBackbone(NeuralBackbone):
    """Autoencoder backbone for unsupervised representation learning."""
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: List[int] = [128, 64, 32],
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        
        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers).to(self.device)
        
        # Decoder
        decoder_layers = []
        current_dim = encoding_dims[-1]
        for dim in reversed(encoding_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(current_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = dim
        
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers).to(self.device)
        
        # Classifier head (for reasoning tasks)
        self.classifier = nn.Linear(encoding_dims[-1], 2).to(self.device)
        
        # Optimizers and loss
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) +
            list(self.classifier.parameters()),
            lr=0.001
        )
        self.mse_criterion = nn.MSELoss()
        self.ce_criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder + classifier."""
        embeddings = self.get_embeddings(x)
        return self.classifier(embeddings)
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder embeddings."""
        return self.encoder(x.to(self.device))
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input through autoencoder."""
        embeddings = self.get_embeddings(x)
        return self.decoder(embeddings)
    
    def train_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        reconstruction_weight: float = 1.0,
        classification_weight: float = 1.0
    ) -> Dict[str, float]:
        """Perform one training step with combined loss."""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        embeddings = self.get_embeddings(x)
        reconstructed = self.decoder(embeddings)
        logits = self.classifier(embeddings)
        
        # Combined loss
        reconstruction_loss = self.mse_criterion(reconstructed, x)
        classification_loss = self.ce_criterion(logits, y)
        total_loss = (
            reconstruction_weight * reconstruction_loss + 
            classification_weight * classification_loss
        )
        
        total_loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y).float().mean()
        
        return {
            "total_loss": total_loss.item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "classification_loss": classification_loss.item(),
            "accuracy": accuracy.item()
        }


class NeuralAdapter:
    """Adapter for managing neural network backbones."""
    
    BACKBONE_REGISTRY = {
        "transformer": TransformerBackbone,
        "cnn": CNNBackbone, 
        "autoencoder": AutoencoderBackbone
    }
    
    def __init__(self):
        self.backbones: Dict[str, NeuralBackbone] = {}
    
    def create_backbone(
        self,
        backbone_type: str,
        backbone_id: str,
        **kwargs
    ) -> NeuralBackbone:
        """Create and register a neural backbone."""
        if backbone_type not in self.BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone type: {backbone_type}. "
                f"Available: {list(self.BACKBONE_REGISTRY.keys())}"
            )
        
        backbone_class = self.BACKBONE_REGISTRY[backbone_type]
        backbone = backbone_class(**kwargs)
        self.backbones[backbone_id] = backbone
        
        return backbone
    
    def get_backbone(self, backbone_id: str) -> NeuralBackbone:
        """Get a registered backbone."""
        if backbone_id not in self.backbones:
            raise ValueError(f"Backbone {backbone_id} not found")
        return self.backbones[backbone_id]
    
    def train_backbone(
        self,
        backbone_id: str,
        train_data: Union[DataLoader, Tuple[NDArray, NDArray]],
        epochs: int = 10,
        validation_data: Optional[Union[DataLoader, Tuple[NDArray, NDArray]]] = None
    ) -> Dict[str, List[float]]:
        """Train a neural backbone."""
        backbone = self.get_backbone(backbone_id)
        
        # Convert numpy arrays to DataLoader if needed
        if isinstance(train_data, tuple):
            X_train, y_train = train_data
            X_train = torch.FloatTensor(X_train)
            y_train = torch.LongTensor(y_train)
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        else:
            train_loader = train_data
        
        if validation_data and isinstance(validation_data, tuple):
            X_val, y_val = validation_data
            X_val = torch.FloatTensor(X_val)
            y_val = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=32)
        else:
            val_loader = validation_data
        
        # Training history
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        for epoch in range(epochs):
            # Training
            train_losses, train_accuracies = [], []
            for batch in train_loader:
                metrics = backbone.train_step(batch)
                train_losses.append(metrics["loss"])
                train_accuracies.append(metrics["accuracy"])
            
            avg_train_loss = np.mean(train_losses)
            avg_train_accuracy = np.mean(train_accuracies)
            
            history["train_loss"].append(avg_train_loss)
            history["train_accuracy"].append(avg_train_accuracy)
            
            # Validation
            if val_loader:
                val_losses, val_accuracies = [], []
                with torch.no_grad():
                    for batch in val_loader:
                        x, y = batch
                        logits = backbone.forward(x)
                        loss = backbone.criterion(logits, y.to(backbone.device))
                        predictions = torch.argmax(logits, dim=1)
                        accuracy = (predictions == y.to(backbone.device)).float().mean()
                        
                        val_losses.append(loss.item())
                        val_accuracies.append(accuracy.item())
                
                avg_val_loss = np.mean(val_losses)
                avg_val_accuracy = np.mean(val_accuracies)
                
                history["val_loss"].append(avg_val_loss)
                history["val_accuracy"].append(avg_val_accuracy)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Train Acc: {avg_train_accuracy:.4f}")
            
            if val_loader:
                print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}")
        
        return history
    
    def predict(
        self,
        backbone_id: str,
        data: Union[torch.Tensor, NDArray, List[str]]
    ) -> torch.Tensor:
        """Make predictions using a trained backbone."""
        backbone = self.get_backbone(backbone_id)
        
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
        
        with torch.no_grad():
            return backbone.forward(data)
    
    def get_embeddings(
        self,
        backbone_id: str,
        data: Union[torch.Tensor, NDArray, List[str]]
    ) -> torch.Tensor:
        """Get embeddings from a trained backbone."""
        backbone = self.get_backbone(backbone_id)
        
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
        
        with torch.no_grad():
            return backbone.get_embeddings(data)
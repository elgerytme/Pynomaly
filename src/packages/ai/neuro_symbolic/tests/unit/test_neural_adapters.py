"""Unit tests for neural adapter backbone classes."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from neuro_symbolic.infrastructure.neural_adapters import (
    TransformerBackbone,
    CNNBackbone,
    AutoencoderBackbone,
    GNNBackbone
)


class TestTransformerBackbone:
    """Test cases for TransformerBackbone adapter."""
    
    def test_create_transformer_backbone(self):
        """Test creating a valid transformer backbone."""
        backbone = TransformerBackbone(
            input_dim=512,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            dropout=0.1
        )
        
        assert backbone.input_dim == 512
        assert backbone.hidden_dim == 256
        assert backbone.num_layers == 6
        assert backbone.num_heads == 8
        assert backbone.dropout == 0.1
        assert isinstance(backbone.transformer, nn.TransformerEncoder)
        assert isinstance(backbone.embedding, nn.Linear)
        assert isinstance(backbone.layer_norm, nn.LayerNorm)
    
    def test_transformer_forward_pass(self):
        """Test transformer forward pass."""
        backbone = TransformerBackbone(
            input_dim=128,
            hidden_dim=64,
            num_layers=2,
            num_heads=4
        )
        
        # Create sample input (batch_size=2, seq_length=10, input_dim=128)
        batch_size, seq_length = 2, 10
        input_tensor = torch.randn(batch_size, seq_length, 128)
        
        output = backbone.forward(input_tensor)
        
        # Output should maintain sequence structure
        assert output.shape == (batch_size, seq_length, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_transformer_encoding_with_mask(self):
        """Test transformer with attention mask."""
        backbone = TransformerBackbone(
            input_dim=64,
            hidden_dim=32,
            num_layers=2,
            num_heads=2
        )
        
        batch_size, seq_length = 2, 8
        input_tensor = torch.randn(batch_size, seq_length, 64)
        
        # Create attention mask (True means ignore)
        attention_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
        attention_mask[0, 6:] = True  # Mask last 2 positions for first sequence
        
        output = backbone.encode(input_tensor, attention_mask=attention_mask)
        
        assert output.shape == (batch_size, seq_length, 32)
        # Masked positions should have different values than unmasked
        assert not torch.equal(output[0, 5], output[0, 7])
    
    def test_transformer_get_attention_weights(self):
        """Test getting attention weights from transformer."""
        backbone = TransformerBackbone(
            input_dim=32,
            hidden_dim=16,
            num_layers=1,
            num_heads=2
        )
        
        input_tensor = torch.randn(1, 5, 32)
        output, attention_weights = backbone.forward_with_attention(input_tensor)
        
        assert output.shape == (1, 5, 16)
        assert len(attention_weights) == 1  # num_layers
        assert attention_weights[0].shape == (1, 2, 5, 5)  # (batch, heads, seq, seq)
    
    def test_invalid_transformer_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="Input dimension must be positive"):
            TransformerBackbone(input_dim=0, hidden_dim=64)
        
        with pytest.raises(ValueError, match="Hidden dimension must be positive"):
            TransformerBackbone(input_dim=64, hidden_dim=0)
        
        with pytest.raises(ValueError, match="Number of layers must be positive"):
            TransformerBackbone(input_dim=64, hidden_dim=32, num_layers=0)
        
        with pytest.raises(ValueError, match="Number of heads must be positive"):
            TransformerBackbone(input_dim=64, hidden_dim=32, num_heads=0)


class TestCNNBackbone:
    """Test cases for CNN backbone adapter."""
    
    def test_create_cnn_backbone(self):
        """Test creating a valid CNN backbone."""
        backbone = CNNBackbone(
            input_channels=3,
            hidden_channels=[32, 64, 128],
            kernel_sizes=[3, 3, 3],
            strides=[1, 2, 2],
            dropout=0.2
        )
        
        assert backbone.input_channels == 3
        assert backbone.hidden_channels == [32, 64, 128]
        assert len(backbone.conv_layers) == 3
        assert isinstance(backbone.adaptive_pool, nn.AdaptiveAvgPool2d)
    
    def test_cnn_forward_pass(self):
        """Test CNN forward pass."""
        backbone = CNNBackbone(
            input_channels=1,
            hidden_channels=[16, 32],
            kernel_sizes=[5, 3],
            strides=[1, 2]
        )
        
        # Create sample input (batch_size=2, channels=1, height=32, width=32)
        input_tensor = torch.randn(2, 1, 32, 32)
        
        output = backbone.forward(input_tensor)
        
        # Output should be flattened feature vector
        assert output.dim() == 2
        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 32  # final channel count
    
    def test_cnn_feature_extraction(self):
        """Test CNN feature extraction at different layers."""
        backbone = CNNBackbone(
            input_channels=3,
            hidden_channels=[8, 16, 32],
            kernel_sizes=[3, 3, 3]
        )
        
        input_tensor = torch.randn(1, 3, 64, 64)
        features = backbone.extract_features(input_tensor)
        
        # Should return features from each layer
        assert len(features) == 3
        assert features[0].shape[1] == 8   # first layer channels
        assert features[1].shape[1] == 16  # second layer channels
        assert features[2].shape[1] == 32  # third layer channels
    
    def test_cnn_with_batch_normalization(self):
        """Test CNN with batch normalization enabled."""
        backbone = CNNBackbone(
            input_channels=1,
            hidden_channels=[16],
            kernel_sizes=[3],
            use_batch_norm=True
        )
        
        input_tensor = torch.randn(4, 1, 28, 28)
        output = backbone.forward(input_tensor)
        
        assert output.shape == (4, 16)
        assert not torch.isnan(output).any()
    
    def test_invalid_cnn_parameters(self):
        """Test that invalid CNN parameters raise errors."""
        with pytest.raises(ValueError, match="Input channels must be positive"):
            CNNBackbone(input_channels=0, hidden_channels=[16])
        
        with pytest.raises(ValueError, match="Hidden channels list cannot be empty"):
            CNNBackbone(input_channels=3, hidden_channels=[])
        
        with pytest.raises(ValueError, match="Kernel sizes must match hidden channels"):
            CNNBackbone(
                input_channels=3,
                hidden_channels=[16, 32],
                kernel_sizes=[3]  # Mismatch
            )


class TestAutoencoderBackbone:
    """Test cases for Autoencoder backbone adapter."""
    
    def test_create_autoencoder_backbone(self):
        """Test creating a valid autoencoder backbone."""
        backbone = AutoencoderBackbone(
            input_dim=784,
            hidden_dims=[256, 128, 64],
            latent_dim=32,
            dropout=0.1
        )
        
        assert backbone.input_dim == 784
        assert backbone.hidden_dims == [256, 128, 64]
        assert backbone.latent_dim == 32
        assert isinstance(backbone.encoder, nn.Sequential)
        assert isinstance(backbone.decoder, nn.Sequential)
    
    def test_autoencoder_encode_decode(self):
        """Test autoencoder encoding and decoding."""
        backbone = AutoencoderBackbone(
            input_dim=100,
            hidden_dims=[64, 32],
            latent_dim=16
        )
        
        input_tensor = torch.randn(4, 100)
        
        # Test encoding
        encoded = backbone.encode(input_tensor)
        assert encoded.shape == (4, 16)
        
        # Test decoding
        decoded = backbone.decode(encoded)
        assert decoded.shape == (4, 100)
        
        # Test full forward pass
        reconstructed = backbone.forward(input_tensor)
        assert reconstructed.shape == input_tensor.shape
    
    def test_autoencoder_bottleneck_features(self):
        """Test extracting bottleneck features."""
        backbone = AutoencoderBackbone(
            input_dim=50,
            hidden_dims=[32, 16],
            latent_dim=8
        )
        
        input_tensor = torch.randn(2, 50)
        features = backbone.get_bottleneck_features(input_tensor)
        
        assert features.shape == (2, 8)
        # Features should be meaningful (not all zeros)
        assert not torch.all(features == 0)
    
    def test_autoencoder_reconstruction_loss(self):
        """Test reconstruction loss calculation."""
        backbone = AutoencoderBackbone(
            input_dim=20,
            hidden_dims=[10],
            latent_dim=5
        )
        
        input_tensor = torch.randn(3, 20)
        loss = backbone.reconstruction_loss(input_tensor)
        
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_invalid_autoencoder_parameters(self):
        """Test that invalid autoencoder parameters raise errors."""
        with pytest.raises(ValueError, match="Input dimension must be positive"):
            AutoencoderBackbone(input_dim=0, latent_dim=10)
        
        with pytest.raises(ValueError, match="Latent dimension must be positive"):
            AutoencoderBackbone(input_dim=100, latent_dim=0)
        
        with pytest.raises(ValueError, match="Hidden dimensions must be decreasing"):
            AutoencoderBackbone(
                input_dim=100,
                hidden_dims=[32, 64],  # Not decreasing
                latent_dim=16
            )


class TestGNNBackbone:
    """Test cases for Graph Neural Network backbone adapter."""
    
    def test_create_gnn_backbone(self):
        """Test creating a valid GNN backbone."""
        backbone = GNNBackbone(
            input_dim=64,
            hidden_dims=[32, 16],
            num_layers=2,
            aggregation="mean",
            dropout=0.1
        )
        
        assert backbone.input_dim == 64
        assert backbone.hidden_dims == [32, 16]
        assert backbone.num_layers == 2
        assert backbone.aggregation == "mean"
        assert len(backbone.conv_layers) == 2
    
    @patch('torch_geometric.nn.GCNConv')
    def test_gnn_forward_pass(self, mock_gcn):
        """Test GNN forward pass with mock graph data."""
        # Mock GCN layer
        mock_conv = Mock()
        mock_conv.return_value = torch.randn(4, 32)
        mock_gcn.return_value = mock_conv
        
        backbone = GNNBackbone(
            input_dim=64,
            hidden_dims=[32],
            num_layers=1
        )
        
        # Create mock graph data
        node_features = torch.randn(4, 64)  # 4 nodes, 64 features each
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # Circular graph
        
        output = backbone.forward(node_features, edge_index)
        
        assert output.shape == (4, 32)
        mock_conv.assert_called_once()
    
    def test_gnn_node_embeddings(self):
        """Test extracting node embeddings."""
        backbone = GNNBackbone(
            input_dim=32,
            hidden_dims=[16],
            num_layers=1
        )
        
        with patch.object(backbone, 'forward') as mock_forward:
            mock_forward.return_value = torch.randn(3, 16)
            
            node_features = torch.randn(3, 32)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
            
            embeddings = backbone.get_node_embeddings(node_features, edge_index)
            
            assert embeddings.shape == (3, 16)
            mock_forward.assert_called_once_with(node_features, edge_index)
    
    def test_gnn_graph_level_pooling(self):
        """Test graph-level pooling operations."""
        backbone = GNNBackbone(
            input_dim=16,
            hidden_dims=[8],
            num_layers=1,
            aggregation="max"
        )
        
        node_embeddings = torch.randn(5, 8)
        batch_index = torch.tensor([0, 0, 1, 1, 1])  # 2 graphs in batch
        
        graph_embedding = backbone.pool_graph(node_embeddings, batch_index)
        
        assert graph_embedding.shape == (2, 8)  # 2 graphs, 8 features each
    
    def test_invalid_gnn_parameters(self):
        """Test that invalid GNN parameters raise errors."""
        with pytest.raises(ValueError, match="Input dimension must be positive"):
            GNNBackbone(input_dim=0, hidden_dims=[16])
        
        with pytest.raises(ValueError, match="Number of layers must be positive"):
            GNNBackbone(input_dim=32, hidden_dims=[16], num_layers=0)
        
        with pytest.raises(ValueError, match="Invalid aggregation method"):
            GNNBackbone(
                input_dim=32,
                hidden_dims=[16],
                aggregation="invalid"
            )


class TestBackboneIntegration:
    """Test integration between different backbone adapters."""
    
    def test_backbone_output_compatibility(self):
        """Test that backbone outputs are compatible for fusion."""
        # Create different backbones with same output dimension
        transformer = TransformerBackbone(
            input_dim=64, hidden_dim=32, num_layers=1, num_heads=2
        )
        
        cnn = CNNBackbone(
            input_channels=1,
            hidden_channels=[32],
            kernel_sizes=[3]
        )
        
        autoencoder = AutoencoderBackbone(
            input_dim=100,
            hidden_dims=[64],
            latent_dim=32
        )
        
        # Test that outputs have compatible shapes for fusion
        transformer_input = torch.randn(2, 10, 64)
        cnn_input = torch.randn(2, 1, 28, 28)
        ae_input = torch.randn(2, 100)
        
        transformer_out = transformer.forward(transformer_input)
        cnn_out = cnn.forward(cnn_input)
        ae_out = autoencoder.encode(ae_input)
        
        # All should have same feature dimension for fusion
        assert transformer_out.shape[-1] == 32
        assert cnn_out.shape[-1] == 32
        assert ae_out.shape[-1] == 32
    
    def test_backbone_gradients_flow(self):
        """Test that gradients flow properly through backbones."""
        backbone = TransformerBackbone(
            input_dim=32, hidden_dim=16, num_layers=1, num_heads=2
        )
        
        input_tensor = torch.randn(1, 5, 32, requires_grad=True)
        output = backbone.forward(input_tensor)
        loss = output.sum()
        
        loss.backward()
        
        # Check gradients exist
        assert input_tensor.grad is not None
        assert not torch.all(input_tensor.grad == 0)
        
        # Check model parameters have gradients
        for param in backbone.parameters():
            if param.requires_grad:
                assert param.grad is not None
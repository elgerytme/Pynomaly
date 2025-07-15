# Graph Anomaly Detection with PyGOD

This guide covers the implementation and usage of graph-based anomaly detection in Pynomaly using the PyGOD (Python Graph Outlier Detection) library integration.

## Overview

Pynomaly provides comprehensive support for graph anomaly detection through the PyGOD adapter, enabling detection of anomalies in:

- **Social Networks**: Identify suspicious users, bot accounts, or unusual interaction patterns
- **Citation Networks**: Detect fraudulent papers or unusual citation patterns  
- **Financial Networks**: Identify money laundering, fraud, or unusual transaction patterns
- **Computer Networks**: Detect intrusions, malware, or unusual traffic patterns
- **Biological Networks**: Identify anomalous proteins, genes, or interaction patterns

## Installation

### Prerequisites

Graph anomaly detection requires additional dependencies that are not installed by default:

```bash
# Install PyGOD and dependencies
pip install "pynomaly[graph]"

# Or install manually
pip install pygod torch torch-geometric
```

### Verification

Verify your installation:

```python
from pynomaly.infrastructure.adapters.pygod_adapter import PyGODAdapter

# Check available algorithms
algorithms = PyGODAdapter.get_supported_algorithms()
print(f"Available algorithms: {algorithms}")
```

## Supported Algorithms

### Deep Learning Methods

#### DOMINANT

- **Type**: Deep Graph Convolutional Autoencoder
- **Best For**: Attributed graphs with node features
- **Strengths**: State-of-the-art performance, handles complex patterns
- **Requirements**: GPU recommended for large graphs

```python
adapter = PyGODAdapter(
    algorithm_name='DOMINANT',
    hidden_dim=64,
    num_layers=2,
    epoch=100,
    dropout=0.3
)
```

#### GCNAE (Graph Convolutional Network Autoencoder)

- **Type**: Graph autoencoder with GCN layers
- **Best For**: Semi-supervised anomaly detection
- **Strengths**: Good balance of performance and interpretability

#### ANOMALOUS

- **Type**: Graph autoencoder approach
- **Best For**: Large-scale graphs
- **Strengths**: Scalable, memory efficient

### Statistical Methods

#### SCAN

- **Type**: Density-based clustering
- **Best For**: Graphs with clear community structure
- **Strengths**: Fast, interpretable, no GPU required

```python
adapter = PyGODAdapter(
    algorithm_name='SCAN',
    eps=0.5,
    mu=2
)
```

#### RADAR

- **Type**: Residual analysis
- **Best For**: Structural anomaly detection
- **Strengths**: Good for detecting structural outliers

## Data Formats

The PyGOD adapter supports multiple graph data formats:

### 1. Edge List Format

```python
import pandas as pd
from pynomaly.domain.entities import Dataset

# Create graph data with edges
data = pd.DataFrame({
    'source': [0, 1, 1, 2, 2, 3],
    'target': [1, 0, 2, 1, 3, 2],
    'weight': [1.0, 1.0, 0.8, 0.9, 0.7, 0.6],  # Optional
    'feature_0': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # Node features
    'feature_1': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
})

dataset = Dataset(
    id="social_network",
    name="Social Network Graph",
    data=data,
    metadata={
        'is_graph': True,
        'edge_columns': ['source', 'target'],
        'weight_column': 'weight',  # Optional
        'feature_columns': ['feature_0', 'feature_1']
    }
)
```

### 2. Adjacency Matrix Format

```python
# Store adjacency matrix in metadata
dataset = Dataset(
    id="network_graph",
    name="Network Graph",
    data=node_features_df,
    metadata={
        'adjacency_matrix': adjacency_matrix,  # numpy array
        'feature_columns': ['feature_0', 'feature_1', 'feature_2']
    }
)
```

### 3. Feature-Only Format (k-NN Graph Construction)

```python
# When no explicit graph structure is provided,
# the adapter will construct a k-NN graph from features
data = pd.DataFrame({
    'node_id': range(100),
    'feature_0': np.random.normal(0, 1, 100),
    'feature_1': np.random.normal(0, 1, 100),
    'feature_2': np.random.normal(0, 1, 100)
})

dataset = Dataset(
    id="feature_graph",
    name="Feature-based Graph",
    data=data,
    metadata={
        'feature_columns': ['feature_0', 'feature_1', 'feature_2'],
        'knn_k': 5  # Number of neighbors for k-NN graph
    }
)
```

## Usage Examples

### Basic Usage

```python
from pynomaly.infrastructure.adapters.pygod_adapter import PyGODAdapter
from pynomaly.domain.value_objects import ContaminationRate

# Create adapter
adapter = PyGODAdapter(
    algorithm_name='DOMINANT',
    contamination_rate=ContaminationRate(0.1),
    hidden_dim=64,
    num_layers=2
)

# Train the model
adapter.fit(dataset)

# Detect anomalies
result = adapter.predict(dataset)

# Access results
anomaly_scores = [score.value for score in result.scores]
anomaly_labels = result.labels
confidence_scores = [score.confidence for score in result.scores]

print(f"Detected {sum(anomaly_labels)} anomalies")
print(f"Average anomaly score: {np.mean(anomaly_scores):.3f}")
```

### Advanced Configuration

```python
# Configure algorithm with custom parameters
adapter = PyGODAdapter(
    algorithm_name='GCNAE',
    name='CustomGraphDetector',
    contamination_rate=ContaminationRate(0.05),
    # Algorithm-specific parameters
    hidden_dim=128,
    num_layers=3,
    dropout=0.2,
    lr=0.001,
    epoch=200,
    weight_decay=0.0001,
    # GPU configuration
    gpu=0,  # Use GPU 0
    verbose=True
)
```

### Social Network Example

```python
# Detect suspicious users in a social network
social_data = pd.DataFrame({
    'user_id': range(1000),
    'follower_from': edge_data['source'],
    'follower_to': edge_data['target'],
    'account_age': user_features['age'],
    'post_frequency': user_features['posts_per_day'],
    'engagement_rate': user_features['likes_per_post'],
    'network_centrality': user_features['centrality']
})

dataset = Dataset(
    id="social_network",
    data=social_data,
    metadata={
        'is_graph': True,
        'edge_columns': ['follower_from', 'follower_to'],
        'feature_columns': ['account_age', 'post_frequency', 'engagement_rate', 'network_centrality']
    }
)

# Use DOMINANT for complex pattern detection
adapter = PyGODAdapter(
    algorithm_name='DOMINANT',
    contamination_rate=ContaminationRate(0.05),  # Expect 5% suspicious users
    hidden_dim=64,
    epoch=150
)

adapter.fit(dataset)
result = adapter.predict(dataset)

# Identify suspicious users
suspicious_users = [
    user_id for user_id, label in zip(social_data['user_id'], result.labels)
    if label == 1
]
```

### Financial Network Example

```python
# Detect money laundering patterns
financial_data = pd.DataFrame({
    'from_account': transaction_data['sender'],
    'to_account': transaction_data['receiver'],
    'amount': transaction_data['amount'],
    'frequency': account_features['transaction_frequency'],
    'balance_volatility': account_features['balance_std'],
    'cross_border': account_features['international_ratio']
})

dataset = Dataset(
    id="financial_network",
    data=financial_data,
    metadata={
        'is_graph': True,
        'edge_columns': ['from_account', 'to_account'],
        'weight_column': 'amount',
        'feature_columns': ['frequency', 'balance_volatility', 'cross_border']
    }
)

# Use SCAN for community-based detection
adapter = PyGODAdapter(
    algorithm_name='SCAN',
    contamination_rate=ContaminationRate(0.02),  # Expect 2% suspicious accounts
    eps=0.3,
    mu=3
)

adapter.fit(dataset)
result = adapter.predict(dataset)
```

## Algorithm Selection Guide

### When to Use Deep Learning Methods

- **Large graphs** (>1000 nodes) with rich node features
- **Complex patterns** that statistical methods might miss
- **GPU resources** are available
- **High accuracy** is more important than interpretability

### When to Use Statistical Methods

- **Small to medium graphs** (<1000 nodes)
- **Limited computational resources** (no GPU)
- **Interpretability** is important
- **Fast results** are needed
- **Clear structural patterns** are expected

## Performance Optimization

### GPU Configuration

```python
# Enable GPU acceleration for deep learning methods
adapter = PyGODAdapter(
    algorithm_name='DOMINANT',
    gpu=0,  # Use first GPU
    batch_size=256,  # Larger batch for better GPU utilization
    num_workers=4  # Parallel data loading
)
```

### Memory Management

```python
# For large graphs, use memory-efficient settings
adapter = PyGODAdapter(
    algorithm_name='GCNAE',
    hidden_dim=32,  # Smaller hidden dimension
    batch_size=64,  # Smaller batch size
    gradient_accumulation_steps=4  # Accumulate gradients
)
```

### Distributed Processing

```python
# Use with distributed anomaly detection
from pynomaly.infrastructure.distributed.distributed_detector import DistributedDetector

distributed_detector = DistributedDetector(
    adapter=PyGODAdapter(algorithm_name='SCAN'),
    n_workers=4
)
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'pygod'

```bash
# Install PyGOD dependencies
pip install "pynomaly[graph]"
```

#### CUDA out of memory

```python
# Reduce memory usage
adapter = PyGODAdapter(
    algorithm_name='DOMINANT',
    hidden_dim=32,  # Reduce from default 64
    batch_size=32,  # Reduce batch size
    num_layers=1    # Reduce model complexity
)
```

#### Poor performance on small graphs

```python
# Use statistical methods for small graphs
adapter = PyGODAdapter(
    algorithm_name='SCAN',  # Instead of deep learning methods
    eps=0.5,
    mu=2
)
```

### Performance Monitoring

```python
# Monitor training progress
adapter = PyGODAdapter(
    algorithm_name='DOMINANT',
    verbose=True,  # Enable progress logging
    early_stopping=True,  # Stop early if no improvement
    patience=10
)

# Access training metrics
result = adapter.predict(dataset)
training_metrics = result.metadata.get('training_metrics', {})
print(f"Training loss: {training_metrics.get('final_loss', 'N/A')}")
print(f"Training time: {training_metrics.get('training_time', 'N/A')}")
```

## Integration with Pynomaly Pipeline

### Using with Ensemble Methods

```python
from pynomaly.application.services import EnsembleService

# Create multiple graph detectors
detectors = [
    PyGODAdapter(algorithm_name='DOMINANT', name='graph_dominant'),
    PyGODAdapter(algorithm_name='SCAN', name='graph_scan'),
    PyGODAdapter(algorithm_name='GCNAE', name='graph_gcnae')
]

# Use in ensemble
ensemble_service = EnsembleService()
ensemble_result = ensemble_service.detect_anomalies_ensemble(
    detectors=detectors,
    dataset=dataset,
    aggregation_method='weighted_average'
)
```

### Caching Graph Computations

```python
from pynomaly.infrastructure.cache import CacheManager

# Cache expensive graph preprocessing
cache_manager = CacheManager()

@cache_manager.cache(ttl=3600)  # Cache for 1 hour
def preprocess_graph(dataset):
    adapter = PyGODAdapter(algorithm_name='DOMINANT')
    return adapter._prepare_graph_data(dataset)
```

## Best Practices

1. **Algorithm Selection**: Start with statistical methods (SCAN) for exploration, then move to deep learning methods (DOMINANT) for production
2. **Data Preparation**: Ensure node features are normalized and edge weights are meaningful
3. **Hyperparameter Tuning**: Use grid search or Bayesian optimization for critical applications
4. **Validation**: Always validate results with domain experts, especially for high-stakes applications
5. **Monitoring**: Track model performance over time as graph structure evolves
6. **Scalability**: Consider distributed processing for very large graphs (>10k nodes)

## References

- [PyGOD Documentation](https://pygod.readthedocs.io/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Graph Neural Networks for Anomaly Detection](https://arxiv.org/abs/2003.00773)
- [DOMINANT Paper](https://arxiv.org/abs/1901.03474)

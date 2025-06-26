# Specialized Algorithms

## Overview

This guide covers domain-specific anomaly detection algorithms designed for specialized data types and use cases. These algorithms are optimized for specific domains such as time series, graphs, text, and images.

## Quick Navigation

### By Data Type
- **[Time Series](#time-series-algorithms)** - Temporal data, seasonal patterns
- **[Graph Data](#graph-neural-networks)** - Networks, relationships, social data
- **[Text Data](#text-anomaly-detection)** - Documents, logs, natural language
- **[Image Data](#image-anomaly-detection)** - Computer vision, medical imaging

### By Use Case
- **[Real-Time Monitoring](#streaming-algorithms)** - Live data streams
- **[Sequential Patterns](#sequence-algorithms)** - Ordered data, logs
- **[Spatial Data](#spatial-algorithms)** - Geographic, location-based

---

## Time Series Algorithms

### 1. Matrix Profile

**Type**: Pattern matching  
**Library**: TODS/Custom  
**Complexity**: O(n²)  
**Best for**: Motif discovery, pattern anomalies, univariate time series

#### Description
Matrix Profile computes the distance profile between all subsequences in a time series, enabling fast discovery of anomalous patterns and motifs.

#### Algorithm Details
- Sliding window approach over time series
- Computes z-normalized Euclidean distance
- Identifies discord (anomalous subsequences)
- Efficient for pattern-based anomalies

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `window_size` | int | 50 | 10-500 | Subsequence length |
| `normalize` | bool | True | - | Z-normalize subsequences |
| `distance_profile` | str | "euclidean" | "euclidean", "manhattan" | Distance metric |
| `contamination` | float | 0.1 | 0.0-0.5 | Expected anomaly proportion |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Time Series Pattern Detector",
    algorithm="MatrixProfile",
    library="tods",
    parameters={
        "window_size": 100,
        "normalize": True,
        "contamination": 0.05
    }
)
```

#### Strengths
- ✅ Excellent for pattern-based anomalies
- ✅ Parameter-light (mainly window size)
- ✅ Fast motif and discord discovery
- ✅ Robust to noise

#### Limitations
- ❌ Limited to univariate time series
- ❌ Requires selecting appropriate window size
- ❌ Memory intensive for long series
- ❌ Not suitable for point anomalies

#### When to Use
- Recurring pattern detection
- Time series motif analysis
- Sensor data monitoring
- Manufacturing quality control

---

### 2. LSTM AutoEncoder

**Type**: Deep learning/RNN  
**Library**: TensorFlow/PyTorch  
**Complexity**: O(n×T)  
**Best for**: Sequential patterns, multivariate time series, long-term dependencies

#### Description
Recurrent neural network autoencoder that learns to encode and decode time series sequences. Anomalies are detected based on reconstruction error.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `sequence_length` | int | 50 | 10-200 | Input sequence length |
| `hidden_neurons` | list | [64, 32] | Various | LSTM layer sizes |
| `dropout_rate` | float | 0.2 | 0.0-0.5 | Dropout regularization |
| `epochs` | int | 100 | 50-500 | Training epochs |
| `batch_size` | int | 32 | 16-128 | Training batch size |
| `learning_rate` | float | 0.001 | 0.0001-0.01 | Learning rate |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="LSTM Time Series Detector",
    algorithm="LSTMAutoEncoder",
    library="tensorflow",
    parameters={
        "sequence_length": 50,
        "hidden_neurons": [128, 64, 32, 64, 128],
        "epochs": 200,
        "dropout_rate": 0.3
    }
)
```

#### Strengths
- ✅ Handles multivariate time series
- ✅ Captures long-term dependencies
- ✅ Learns complex temporal patterns
- ✅ State-of-the-art for sequence data

#### Limitations
- ❌ Requires large datasets
- ❌ Computationally expensive
- ❌ Many hyperparameters
- ❌ Black box nature

---

### 3. Prophet

**Type**: Statistical forecasting  
**Library**: Facebook Prophet  
**Complexity**: O(n)  
**Best for**: Business time series, seasonal patterns, trend analysis

#### Description
Facebook's time series forecasting tool adapted for anomaly detection by identifying points that deviate significantly from predicted values.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `growth` | str | "linear" | "linear", "logistic" | Growth model |
| `yearly_seasonality` | bool/str | "auto" | True, False, "auto" | Yearly seasonal component |
| `weekly_seasonality` | bool/str | "auto" | True, False, "auto" | Weekly seasonal component |
| `daily_seasonality` | bool/str | "auto" | True, False, "auto" | Daily seasonal component |
| `changepoint_prior_scale` | float | 0.05 | 0.001-0.5 | Trend flexibility |
| `interval_width` | float | 0.80 | 0.5-0.99 | Prediction interval |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Business Metrics Detector", 
    algorithm="Prophet",
    library="prophet",
    parameters={
        "growth": "linear",
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "changepoint_prior_scale": 0.1,
        "interval_width": 0.95
    }
)
```

#### Strengths
- ✅ Handles seasonality automatically
- ✅ Robust to missing data
- ✅ Interpretable components
- ✅ Works well with business metrics

#### Limitations
- ❌ Primarily for univariate series
- ❌ Assumes additive or multiplicative seasonality
- ❌ Not suitable for high-frequency data
- ❌ Limited to trend-seasonal anomalies

---

### 4. ARIMA-based Detection

**Type**: Statistical modeling  
**Library**: statsmodels  
**Complexity**: O(n log n)  
**Best for**: Stationary time series, short-term dependencies

#### Description
Uses ARIMA models to forecast time series values and identifies anomalies as points with high residuals.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `order` | tuple | (1,1,1) | (0-5, 0-2, 0-5) | (p,d,q) ARIMA parameters |
| `seasonal_order` | tuple | (0,0,0,0) | Various | Seasonal ARIMA parameters |
| `trend` | str | "c" | "n", "c", "t", "ct" | Trend component |
| `method` | str | "lbfgs" | "newton", "nm", "bfgs", "lbfgs" | Optimization method |
| `contamination` | float | 0.1 | 0.0-0.5 | Expected anomaly proportion |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="ARIMA Anomaly Detector",
    algorithm="ARIMA",
    library="statsmodels", 
    parameters={
        "order": (2, 1, 2),
        "seasonal_order": (1, 1, 1, 12),
        "trend": "c"
    }
)
```

---

## Graph Neural Networks

### 1. DOMINANT

**Type**: Graph autoencoder  
**Library**: PyGOD  
**Complexity**: O(|E|)  
**Best for**: Attributed graphs, node anomalies, social networks

#### Description
Deep anomaly detection on attributed networks using graph neural network autoencoder to reconstruct both structure and attributes.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `hid_dim` | int | 64 | 32-256 | Hidden dimension |
| `num_layers` | int | 4 | 2-8 | Number of GCN layers |
| `dropout` | float | 0.3 | 0.0-0.5 | Dropout rate |
| `weight_decay` | float | 0.0 | 0.0-0.01 | L2 regularization |
| `lr` | float | 0.004 | 0.001-0.01 | Learning rate |
| `epoch` | int | 5 | 5-100 | Training epochs |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Graph Anomaly Detector",
    algorithm="DOMINANT", 
    library="pygod",
    parameters={
        "hid_dim": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "lr": 0.005,
        "epoch": 50
    }
)
```

#### Strengths
- ✅ Handles both structure and attributes
- ✅ State-of-the-art for graph anomalies
- ✅ Scalable to large graphs
- ✅ Unsupervised learning

#### Limitations
- ❌ Requires graph structure data
- ❌ Complex implementation
- ❌ GPU recommended for large graphs
- ❌ Limited interpretability

#### When to Use
- Social network anomaly detection
- Fraud detection in transaction networks
- Citation network analysis
- Knowledge graph validation

---

### 2. Graph Convolutional AutoEncoder (GCNAE)

**Type**: Convolutional autoencoder  
**Library**: PyGOD  
**Complexity**: O(|E|)  
**Best for**: Node reconstruction, community detection

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `hid_dim` | int | 64 | 32-256 | Hidden dimension |
| `num_layers` | int | 2 | 2-6 | Number of layers |
| `dropout` | float | 0.2 | 0.0-0.5 | Dropout rate |
| `act` | str | "relu" | "relu", "tanh", "sigmoid" | Activation function |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Graph Convolution Detector",
    algorithm="GCNAE",
    library="pygod",
    parameters={
        "hid_dim": 64,
        "num_layers": 3,
        "dropout": 0.3,
        "act": "relu"
    }
)
```

---

## Text Anomaly Detection

### 1. TF-IDF with Clustering

**Type**: Text vectorization + clustering  
**Complexity**: O(n log n)  
**Best for**: Document anomalies, topic drift, spam detection

#### Description
Converts text to TF-IDF vectors and uses clustering algorithms to identify documents that don't fit well into any cluster.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_features` | int | 10000 | 1000-50000 | Maximum features in vocabulary |
| `ngram_range` | tuple | (1,2) | (1,1) to (1,3) | N-gram range |
| `stop_words` | str | "english" | "english", None, list | Stop words to remove |
| `min_df` | int/float | 1 | 1 to 0.1 | Minimum document frequency |
| `max_df` | float | 0.95 | 0.5-1.0 | Maximum document frequency |
| `clustering_algorithm` | str | "kmeans" | "kmeans", "dbscan" | Clustering method |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Document Anomaly Detector",
    algorithm="TFIDFClustering",
    library="sklearn",
    parameters={
        "max_features": 5000,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "clustering_algorithm": "kmeans",
        "n_clusters": 20
    }
)
```

#### Strengths
- ✅ Handles large text corpora
- ✅ Interpretable features
- ✅ Fast computation
- ✅ Language agnostic

#### Limitations
- ❌ Bag-of-words limitations
- ❌ Loses word order information
- ❌ Requires preprocessing
- ❌ Sensitive to vocabulary size

---

### 2. Word Embeddings Anomaly Detection

**Type**: Dense vector representation  
**Complexity**: O(n)  
**Best for**: Semantic anomalies, context-aware detection

#### Description
Uses pre-trained word embeddings (Word2Vec, GloVe, FastText) to represent text as dense vectors and detects semantic anomalies.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `embedding_model` | str | "word2vec" | "word2vec", "glove", "fasttext" | Embedding type |
| `vector_size` | int | 300 | 100-500 | Vector dimension |
| `window` | int | 5 | 3-10 | Context window |
| `min_count` | int | 1 | 1-10 | Minimum word count |
| `aggregation` | str | "mean" | "mean", "max", "tfidf_weighted" | Document aggregation |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Semantic Anomaly Detector",
    algorithm="WordEmbeddings",
    library="gensim",
    parameters={
        "embedding_model": "fasttext",
        "vector_size": 300,
        "window": 5,
        "aggregation": "tfidf_weighted"
    }
)
```

---

## Image Anomaly Detection

### 1. CNN AutoEncoder

**Type**: Convolutional neural network  
**Complexity**: O(n)  
**Best for**: Image anomalies, visual inspection, medical imaging

#### Description
Convolutional autoencoder that learns to compress and reconstruct images. Anomalies are detected based on reconstruction error.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `input_shape` | tuple | (224,224,3) | Various | Input image dimensions |
| `conv_layers` | list | [(32,3), (64,3)] | Various | (filters, kernel_size) pairs |
| `pooling` | str | "max" | "max", "avg" | Pooling type |
| `dropout_rate` | float | 0.25 | 0.0-0.5 | Dropout rate |
| `epochs` | int | 100 | 50-500 | Training epochs |
| `batch_size` | int | 32 | 16-128 | Batch size |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Image Anomaly Detector",
    algorithm="CNNAutoEncoder",
    library="tensorflow",
    parameters={
        "input_shape": (128, 128, 3),
        "conv_layers": [
            {"filters": 32, "kernel_size": 3},
            {"filters": 64, "kernel_size": 3},
            {"filters": 128, "kernel_size": 3}
        ],
        "epochs": 200,
        "dropout_rate": 0.3
    }
)
```

#### Strengths
- ✅ Excellent for image data
- ✅ Preserves spatial structure
- ✅ Hierarchical feature learning
- ✅ Transfer learning capable

#### Limitations
- ❌ Requires large image datasets
- ❌ GPU intensive
- ❌ Many hyperparameters
- ❌ Black box nature

---

## Streaming Algorithms

### 1. Incremental Statistics

**Type**: Online statistical monitoring  
**Complexity**: O(1) per sample  
**Best for**: Real-time monitoring, concept drift detection

#### Description
Maintains running statistics and detects anomalies in streaming data using adaptive thresholds.

#### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `window_size` | int | 1000 | 100-10000 | Sliding window size |
| `threshold_factor` | float | 3.0 | 2.0-5.0 | Standard deviation threshold |
| `adaptation_rate` | float | 0.01 | 0.001-0.1 | Adaptation speed |
| `min_samples` | int | 30 | 10-100 | Minimum samples for statistics |

#### Usage Example
```python
detector = await detection_service.create_detector(
    name="Real-time Monitor",
    algorithm="IncrementalStats",
    library="custom",
    parameters={
        "window_size": 500,
        "threshold_factor": 2.5,
        "adaptation_rate": 0.05
    }
)
```

#### Strengths
- ✅ Real-time processing
- ✅ Constant memory usage
- ✅ Adaptive to concept drift
- ✅ Low computational overhead

#### Limitations
- ❌ Limited to simple patterns
- ❌ Assumes stationarity within windows
- ❌ May miss complex anomalies
- ❌ Parameter sensitive

---

## Ensemble Specialized Methods

### 1. Time Series Ensemble

**Type**: Multi-algorithm ensemble  
**Best for**: Maximum accuracy on temporal data

#### Description
Combines multiple time series algorithms (LSTM, Prophet, ARIMA) for robust temporal anomaly detection.

#### Usage Example
```python
ts_ensemble_config = {
    "algorithms": [
        {"name": "LSTM", "weight": 0.4},
        {"name": "Prophet", "weight": 0.3}, 
        {"name": "MatrixProfile", "weight": 0.3}
    ],
    "voting_strategy": "weighted",
    "contamination": 0.1
}

detector = await detection_service.create_ensemble_detector(
    name="Time Series Ensemble",
    config=ts_ensemble_config
)
```

### 2. Multi-Modal Ensemble

**Type**: Cross-domain ensemble  
**Best for**: Mixed data types

#### Description
Combines algorithms from different domains (text, images, tabular) for comprehensive anomaly detection.

#### Usage Example
```python
multimodal_config = {
    "tabular_algorithm": "IsolationForest",
    "text_algorithm": "TFIDFClustering", 
    "image_algorithm": "CNNAutoEncoder",
    "fusion_strategy": "late_fusion",
    "weights": [0.4, 0.3, 0.3]
}
```

---

## Performance Guidelines

### Time Series Algorithms

| Algorithm | Data Size | Seasonality | Real-time | Multivariate |
|-----------|-----------|-------------|-----------|--------------|
| Matrix Profile | Small-Medium | ❌ | ✅ | ❌ |
| LSTM | Medium-Large | ✅ | ❌ | ✅ |
| Prophet | Small-Medium | ✅ | ✅ | ❌ |
| ARIMA | Small | ✅ | ✅ | ❌ |

### Graph Algorithms

| Algorithm | Graph Size | Attributes | Directed | Performance |
|-----------|------------|------------|----------|-------------|
| DOMINANT | Medium-Large | ✅ | ✅ | ⭐⭐⭐⭐ |
| GCNAE | Small-Medium | ✅ | ❌ | ⭐⭐⭐ |

### Text Algorithms

| Algorithm | Corpus Size | Semantic | Real-time | Languages |
|-----------|-------------|----------|-----------|-----------|
| TF-IDF | Large | ❌ | ✅ | Multiple |
| Word Embeddings | Medium | ✅ | ❌ | Multiple |

## Best Practices

### 1. Data Preprocessing
```python
# Time series
def preprocess_timeseries(data):
    # Handle missing values
    data = data.interpolate(method='time')
    # Remove seasonality if needed
    data = seasonal_decompose(data).resid
    return data

# Text
def preprocess_text(documents):
    # Clean and tokenize
    cleaned = [clean_text(doc) for doc in documents]
    return cleaned

# Graphs
def preprocess_graph(graph):
    # Normalize node features
    features = normalize(graph.node_features)
    return graph
```

### 2. Domain-Specific Validation
```python
# Time series: Use temporal splits
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Text: Stratified by topic/class if available
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Graphs: Use graph-specific cross-validation
# Avoid data leakage through graph connections
```

### 3. Hyperparameter Tuning
```python
# Time series specific
time_series_params = {
    "sequence_length": [20, 50, 100],
    "window_size": [10, 50, 100],
    "seasonality": [True, False]
}

# Graph specific  
graph_params = {
    "hidden_dim": [32, 64, 128],
    "num_layers": [2, 3, 4],
    "dropout": [0.1, 0.3, 0.5]
}
```

---

## Related Documentation

- **[Core Algorithms](core-algorithms.md)** - General-purpose algorithms
- **[Experimental Algorithms](experimental-algorithms.md)** - Advanced research methods  
- **[Algorithm Comparison](algorithm-comparison.md)** - Performance comparisons
- **[Time Series Guide](../../guides/time-series-anomaly-detection.md)** - Time series specific guidance
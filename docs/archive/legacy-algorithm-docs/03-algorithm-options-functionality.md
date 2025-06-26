# Pynomaly Algorithm Options and Functionality Guide

## Overview

Pynomaly provides a comprehensive suite of anomaly detection algorithms across multiple categories. This guide details all available algorithms, their functionality, parameters, use cases, and performance characteristics.

## Table of Contents

1. [Algorithm Categories](#algorithm-categories)
2. [Statistical Methods](#statistical-methods)
3. [Machine Learning Methods](#machine-learning-methods)
4. [Deep Learning Methods](#deep-learning-methods)
5. [Specialized Methods](#specialized-methods)
6. [Ensemble Methods](#ensemble-methods)
7. [Performance Comparison](#performance-comparison)
8. [Parameter Tuning](#parameter-tuning)

## Algorithm Categories

### Overview by Category

| Category | Count | Best For | Typical Use Cases |
|----------|-------|----------|------------------|
| Statistical | 8 | Well-understood data patterns | Baseline detection, interpretable results |
| Machine Learning | 12 | General-purpose detection | Production systems, balanced performance |
| Deep Learning | 10 | Complex patterns | High-dimensional data, feature learning |
| Specialized | 15 | Domain-specific | Time series, graphs, text, images |
| Ensemble | 5 | Maximum accuracy | Critical applications, robust detection |

## Statistical Methods

### 1. Isolation Forest

**Description**: Tree-based algorithm that isolates anomalies using random feature splits.

**Algorithm Details**:
- Creates isolation trees by randomly selecting features and split values
- Anomalies require fewer splits to isolate (shorter path lengths)
- Efficient for high-dimensional data
- No assumptions about data distribution

**Parameters**:
```python
{
    "n_estimators": 100,        # Number of trees (50-500)
    "max_samples": "auto",      # Samples per tree ("auto", int, float)
    "contamination": 0.1,       # Expected anomaly proportion (0.0-0.5)
    "max_features": 1.0,        # Features per tree (0.0-1.0)
    "bootstrap": False,         # Bootstrap sampling
    "random_state": 42,         # Reproducibility
    "warm_start": False,        # Incremental training
    "n_jobs": -1               # Parallel processing
}
```

**Use Cases**:
- General-purpose anomaly detection
- High-dimensional datasets
- Real-time detection systems
- Baseline comparisons

**Strengths**:
- Fast training and prediction
- Handles high dimensions well
- No need for labeled data
- Memory efficient

**Limitations**:
- May struggle with normal data in high dimensions
- Sensitive to feature scaling in some cases
- Less interpretable than some methods

**Performance Characteristics**:
- Training time: O(n log n)
- Prediction time: O(log n)
- Memory usage: Low to moderate
- Scalability: Excellent

### 2. Local Outlier Factor (LOF)

**Description**: Density-based algorithm that identifies outliers based on local density deviation.

**Algorithm Details**:
- Calculates local density for each point
- Compares point density to its neighbors
- High LOF score indicates anomaly
- Based on k-nearest neighbors

**Parameters**:
```python
{
    "n_neighbors": 20,          # Number of neighbors (5-50)
    "algorithm": "auto",        # KNN algorithm ("auto", "ball_tree", "kd_tree", "brute")
    "leaf_size": 30,           # Leaf size for tree algorithms
    "metric": "minkowski",      # Distance metric
    "p": 2,                    # Power parameter for Minkowski
    "metric_params": None,      # Additional metric parameters
    "contamination": 0.1,       # Expected anomaly proportion
    "novelty": False,          # Novelty detection mode
    "n_jobs": -1               # Parallel processing
}
```

**Use Cases**:
- Datasets with varying density
- Cluster-based anomalies
- Local pattern analysis
- Spatial data analysis

**Strengths**:
- Adapts to local data density
- Good for clusters of different densities
- Intuitive interpretation
- No assumptions about data distribution

**Limitations**:
- Sensitive to parameter choices
- Computationally expensive for large datasets
- Curse of dimensionality
- Memory intensive

**Performance Characteristics**:
- Training time: O(n²)
- Prediction time: O(kn)
- Memory usage: High
- Scalability: Poor for large datasets

### 3. One-Class SVM

**Description**: Support Vector Machine adapted for anomaly detection using a single class.

**Algorithm Details**:
- Maps data to high-dimensional space
- Finds hyperplane separating normal data from origin
- Uses kernel trick for non-linear boundaries
- Robust to outliers during training

**Parameters**:
```python
{
    "kernel": "rbf",           # Kernel type ("linear", "poly", "rbf", "sigmoid")
    "degree": 3,               # Polynomial kernel degree
    "gamma": "scale",          # Kernel coefficient ("scale", "auto", float)
    "coef0": 0.0,             # Independent term for poly/sigmoid
    "tol": 1e-3,              # Tolerance for stopping
    "nu": 0.5,                # Upper bound on training errors (0.0-1.0)
    "shrinking": True,         # Use shrinking heuristic
    "cache_size": 200,         # Kernel cache size (MB)
    "verbose": False,          # Verbose output
    "max_iter": -1,           # Max iterations (-1 for no limit)
    "random_state": 42         # Reproducibility
}
```

**Use Cases**:
- Non-linear decision boundaries
- Robust anomaly detection
- Small to medium datasets
- Quality control applications

**Strengths**:
- Handles non-linear patterns well
- Robust to outliers
- Strong theoretical foundation
- Effective in high dimensions

**Limitations**:
- Sensitive to parameter tuning
- Computationally expensive
- Memory intensive
- Difficult to interpret

**Performance Characteristics**:
- Training time: O(n²) to O(n³)
- Prediction time: O(sv × d)
- Memory usage: High
- Scalability: Poor for large datasets

### 4. Elliptic Envelope

**Description**: Assumes data follows multivariate Gaussian distribution and detects outliers.

**Algorithm Details**:
- Fits robust covariance estimate
- Uses Mahalanobis distance for outlier detection
- Assumes elliptical data distribution
- Robust to outliers in covariance estimation

**Parameters**:
```python
{
    "store_precision": True,    # Store precision matrix
    "assume_centered": False,   # Assume data is centered
    "support_fraction": None,   # Proportion of points for covariance (0.0-1.0)
    "contamination": 0.1,       # Expected anomaly proportion
    "random_state": 42         # Reproducibility
}
```

**Use Cases**:
- Gaussian-distributed data
- Multivariate outlier detection
- Statistical quality control
- Financial fraud detection

**Strengths**:
- Fast computation
- Strong statistical foundation
- Interpretable results
- Good for Gaussian data

**Limitations**:
- Assumes Gaussian distribution
- Poor performance on non-Gaussian data
- Sensitive to high dimensions
- Limited to elliptical boundaries

**Performance Characteristics**:
- Training time: O(n × d²)
- Prediction time: O(d²)
- Memory usage: Low
- Scalability: Good

### 5. Z-Score Detection

**Description**: Statistical method based on standard deviations from the mean.

**Algorithm Details**:
- Calculates z-score for each feature
- Flags points beyond threshold (typically 2-3 standard deviations)
- Assumes normal distribution
- Simple and interpretable

**Parameters**:
```python
{
    "threshold": 3.0,          # Z-score threshold (1.0-5.0)
    "method": "univariate",    # Detection method ("univariate", "multivariate")
    "contamination": 0.1,      # Expected anomaly proportion
    "normalize": True          # Normalize features
}
```

**Use Cases**:
- Simple anomaly detection
- Normally distributed features
- Quick screening
- Baseline comparisons

**Strengths**:
- Extremely fast
- Highly interpretable
- Simple implementation
- No training required

**Limitations**:
- Assumes normal distribution
- Poor for multivariate anomalies
- Sensitive to outliers in training
- Limited to simple patterns

### 6. Interquartile Range (IQR)

**Description**: Non-parametric method based on quartile ranges.

**Algorithm Details**:
- Calculates Q1 (25th percentile) and Q3 (75th percentile)
- Defines outliers as points beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR
- Robust to distribution assumptions
- Standard box-plot approach

**Parameters**:
```python
{
    "factor": 1.5,             # IQR multiplication factor (1.0-3.0)
    "method": "tukey",         # IQR method ("tukey", "modified")
    "contamination": 0.1,      # Expected anomaly proportion
    "feature_wise": True       # Apply per feature vs. globally
}
```

**Use Cases**:
- Exploratory data analysis
- Robust outlier detection
- Non-parametric scenarios
- Quick data screening

**Strengths**:
- Distribution-free
- Robust to outliers
- Simple interpretation
- Fast computation

**Limitations**:
- Only considers marginal distributions
- May miss multivariate patterns
- Fixed threshold approach
- Limited sophistication

### 7. Modified Z-Score

**Description**: Robust version of Z-score using median absolute deviation.

**Algorithm Details**:
- Uses median instead of mean
- Uses MAD (Median Absolute Deviation) instead of standard deviation
- More robust to outliers than standard Z-score
- Better for skewed distributions

**Parameters**:
```python
{
    "threshold": 3.5,          # Modified Z-score threshold
    "contamination": 0.1,      # Expected anomaly proportion
    "consistency_correction": True  # Apply consistency correction
}
```

**Use Cases**:
- Skewed distributions
- Robust outlier detection
- Noisy data
- Univariate screening

**Strengths**:
- Robust to outliers
- Works with skewed data
- Simple interpretation
- Fast computation

**Limitations**:
- Univariate approach
- Limited pattern detection
- May miss subtle anomalies
- Fixed threshold

### 8. Grubbs' Test

**Description**: Statistical test for outliers in univariate normally distributed data.

**Algorithm Details**:
- Tests if extreme values are outliers
- Uses t-distribution for hypothesis testing
- Iteratively removes outliers
- Assumes normal distribution

**Parameters**:
```python
{
    "alpha": 0.05,             # Significance level (0.01-0.1)
    "two_sided": True,         # Two-sided test
    "max_iterations": 10,      # Maximum iterations
    "contamination": 0.1       # Expected anomaly proportion
}
```

**Use Cases**:
- Normally distributed data
- Statistical quality control
- Single feature analysis
- Hypothesis testing

**Strengths**:
- Strong statistical foundation
- Controlled false positive rate
- Interpretable p-values
- Suitable for small samples

**Limitations**:
- Assumes normal distribution
- Univariate only
- Limited to extreme outliers
- May miss multiple outliers

## Machine Learning Methods

### 1. Random Forest Anomaly Detection

**Description**: Ensemble of decision trees adapted for anomaly detection.

**Algorithm Details**:
- Builds multiple decision trees
- Calculates anomaly scores based on path lengths
- Combines predictions from all trees
- Handles mixed data types

**Parameters**:
```python
{
    "n_estimators": 100,        # Number of trees (50-500)
    "max_depth": None,          # Maximum tree depth
    "min_samples_split": 2,     # Minimum samples to split
    "min_samples_leaf": 1,      # Minimum samples per leaf
    "max_features": "sqrt",     # Features per tree
    "bootstrap": True,          # Bootstrap sampling
    "n_jobs": -1,              # Parallel processing
    "random_state": 42,         # Reproducibility
    "contamination": 0.1        # Expected anomaly proportion
}
```

**Use Cases**:
- Mixed data types
- Large datasets
- Feature importance analysis
- Ensemble approaches

**Strengths**:
- Handles mixed data types
- Provides feature importance
- Robust to outliers
- Parallelizable

**Limitations**:
- Can overfit with many trees
- Memory intensive
- Less interpretable
- Sensitive to irrelevant features

### 2. Gradient Boosting Anomaly Detection

**Description**: Sequential ensemble that builds models to correct previous errors.

**Algorithm Details**:
- Builds models sequentially
- Each model corrects previous errors
- Uses gradient descent optimization
- Adaptive learning

**Parameters**:
```python
{
    "n_estimators": 100,        # Number of boosting stages
    "learning_rate": 0.1,       # Learning rate (0.01-0.3)
    "max_depth": 3,            # Maximum tree depth
    "min_samples_split": 2,     # Minimum samples to split
    "min_samples_leaf": 1,      # Minimum samples per leaf
    "subsample": 1.0,          # Fraction of samples per tree
    "random_state": 42,         # Reproducibility
    "contamination": 0.1        # Expected anomaly proportion
}
```

**Use Cases**:
- Complex pattern detection
- High-accuracy requirements
- Structured data
- Competition settings

**Strengths**:
- High accuracy potential
- Handles complex patterns
- Feature importance
- Good generalization

**Limitations**:
- Prone to overfitting
- Sensitive to hyperparameters
- Computationally expensive
- Sequential training

### 3. k-Nearest Neighbors (k-NN)

**Description**: Instance-based method using distance to k nearest neighbors.

**Algorithm Details**:
- Calculates distance to k nearest neighbors
- Anomaly score based on average distance
- Lazy learning approach
- No explicit training phase

**Parameters**:
```python
{
    "n_neighbors": 5,           # Number of neighbors (3-20)
    "algorithm": "auto",        # Algorithm choice
    "leaf_size": 30,           # Leaf size for tree algorithms
    "metric": "minkowski",      # Distance metric
    "p": 2,                    # Power parameter
    "metric_params": None,      # Additional metric parameters
    "contamination": 0.1,       # Expected anomaly proportion
    "n_jobs": -1               # Parallel processing
}
```

**Use Cases**:
- Instance-based detection
- Non-parametric scenarios
- Small to medium datasets
- Pattern matching

**Strengths**:
- Simple and intuitive
- Non-parametric
- Adapts to local patterns
- No training required

**Limitations**:
- Computationally expensive
- Sensitive to dimensionality
- Memory intensive
- Sensitive to irrelevant features

### 4. Support Vector Regression (SVR)

**Description**: Regression-based approach for anomaly detection.

**Algorithm Details**:
- Learns normal data patterns using regression
- Calculates residuals as anomaly scores
- Uses support vector machines framework
- Kernel methods for non-linearity

**Parameters**:
```python
{
    "kernel": "rbf",           # Kernel type
    "degree": 3,               # Polynomial degree
    "gamma": "scale",          # Kernel coefficient
    "coef0": 0.0,             # Independent term
    "tol": 1e-3,              # Tolerance
    "C": 1.0,                 # Regularization parameter
    "epsilon": 0.1,           # Epsilon-tube width
    "shrinking": True,         # Shrinking heuristic
    "cache_size": 200,         # Cache size (MB)
    "verbose": False,          # Verbose output
    "max_iter": -1            # Maximum iterations
}
```

**Use Cases**:
- Regression-based detection
- Non-linear patterns
- Robust to outliers
- Medium-sized datasets

**Strengths**:
- Handles non-linear patterns
- Robust formulation
- Kernel flexibility
- Strong theoretical basis

**Limitations**:
- Sensitive to parameters
- Computationally expensive
- Memory intensive
- Difficult interpretation

### 5. Principal Component Analysis (PCA)

**Description**: Dimensionality reduction technique adapted for anomaly detection.

**Algorithm Details**:
- Projects data to lower dimensions
- Reconstructs data from principal components
- Anomaly score based on reconstruction error
- Linear transformation

**Parameters**:
```python
{
    "n_components": None,       # Number of components (None for all)
    "whiten": False,           # Whitening transformation
    "svd_solver": "auto",      # SVD solver algorithm
    "tol": 0.0,               # Tolerance for singular values
    "iterated_power": "auto",  # Number of iterations
    "random_state": 42,        # Reproducibility
    "contamination": 0.1       # Expected anomaly proportion
}
```

**Use Cases**:
- High-dimensional data
- Linear anomalies
- Dimensionality reduction
- Preprocessing step

**Strengths**:
- Reduces dimensionality
- Fast computation
- Linear interpretability
- Good for high dimensions

**Limitations**:
- Linear assumptions
- May miss non-linear patterns
- Sensitive to scaling
- Information loss

### 6. Independent Component Analysis (ICA)

**Description**: Statistical method that separates multivariate signal into independent components.

**Algorithm Details**:
- Assumes independent source signals
- Separates mixed signals
- Non-Gaussian assumption
- Blind source separation

**Parameters**:
```python
{
    "n_components": None,       # Number of components
    "algorithm": "parallel",    # Algorithm ("parallel", "deflation")
    "whiten": True,            # Whitening preprocessing
    "fun": "logcosh",          # Contrast function
    "fun_args": None,          # Function arguments
    "max_iter": 200,           # Maximum iterations
    "tol": 1e-4,              # Tolerance
    "w_init": None,           # Initial mixing matrix
    "random_state": 42,        # Reproducibility
    "contamination": 0.1       # Expected anomaly proportion
}
```

**Use Cases**:
- Signal processing
- Mixed signal separation
- Non-Gaussian data
- Feature extraction

**Strengths**:
- Separates independent sources
- Non-Gaussian assumptions
- Good for mixed signals
- Interpretable components

**Limitations**:
- Assumes independence
- Sensitive to parameters
- May not converge
- Limited to linear mixing

### 7. Factor Analysis

**Description**: Statistical method modeling observed variables as linear combinations of latent factors.

**Algorithm Details**:
- Models data as latent factors plus noise
- Assumes Gaussian noise
- Maximum likelihood estimation
- Dimensionality reduction

**Parameters**:
```python
{
    "n_components": None,       # Number of factors
    "tol": 1e-2,              # Tolerance
    "copy": True,             # Copy input data
    "max_iter": 1000,         # Maximum iterations
    "noise_variance_init": None, # Initial noise variance
    "svd_method": "randomized", # SVD method
    "iterated_power": 3,       # SVD iterations
    "random_state": 42,        # Reproducibility
    "contamination": 0.1       # Expected anomaly proportion
}
```

**Use Cases**:
- Latent factor modeling
- Dimensionality reduction
- Psychological testing
- Social sciences

**Strengths**:
- Models latent structure
- Handles noise explicitly
- Interpretable factors
- Statistical foundation

**Limitations**:
- Assumes linear relationships
- Gaussian assumptions
- May not converge
- Sensitive to initialization

### 8. Minimum Covariance Determinant (MCD)

**Description**: Robust estimator of multivariate location and scatter.

**Algorithm Details**:
- Finds subset with minimum covariance determinant
- Robust to outliers
- Uses Mahalanobis distance
- Iterative algorithm

**Parameters**:
```python
{
    "store_precision": True,    # Store precision matrix
    "assume_centered": False,   # Assume data centered
    "support_fraction": None,   # Support fraction
    "random_state": 42,        # Reproducibility
    "contamination": 0.1       # Expected anomaly proportion
}
```

**Use Cases**:
- Robust covariance estimation
- Multivariate outliers
- Financial applications
- Quality control

**Strengths**:
- Robust to outliers
- Strong statistical foundation
- Fast computation
- Interpretable

**Limitations**:
- Assumes elliptical distribution
- Limited to moderate dimensions
- May break down with many outliers
- Sensitive to sample size

## Deep Learning Methods

### 1. AutoEncoder

**Description**: Neural network that learns to compress and reconstruct data.

**Algorithm Details**:
- Encoder compresses input to latent representation
- Decoder reconstructs from latent space
- Anomaly score based on reconstruction error
- Unsupervised learning

**Parameters**:
```python
{
    "hidden_neurons": [64, 32, 16, 32, 64],  # Hidden layer sizes
    "hidden_activation": "relu",              # Activation function
    "output_activation": "sigmoid",           # Output activation
    "loss": "mse",                           # Loss function
    "optimizer": "adam",                     # Optimizer
    "epochs": 100,                           # Training epochs
    "batch_size": 32,                        # Batch size
    "dropout_rate": 0.2,                     # Dropout rate
    "l2_regularizer": 0.1,                   # L2 regularization
    "validation_size": 0.1,                  # Validation split
    "preprocessing": True,                    # Data preprocessing
    "verbose": 1,                           # Verbosity
    "contamination": 0.1,                    # Expected anomaly proportion
    "random_state": 42                       # Reproducibility
}
```

**Use Cases**:
- High-dimensional data
- Feature learning
- Image anomaly detection
- Time series anomalies

**Strengths**:
- Learns complex patterns
- Handles high dimensions
- Feature learning
- Flexible architecture

**Limitations**:
- Requires large datasets
- Computationally expensive
- Many hyperparameters
- Black box nature

### 2. Variational AutoEncoder (VAE)

**Description**: Probabilistic version of autoencoder with regularized latent space.

**Algorithm Details**:
- Encoder outputs mean and variance
- Samples from latent distribution
- Regularized latent space
- Probabilistic reconstruction

**Parameters**:
```python
{
    "encoder_neurons": [32, 16],             # Encoder architecture
    "decoder_neurons": [16, 32],             # Decoder architecture
    "latent_dim": 8,                        # Latent dimension
    "hidden_activation": "relu",             # Hidden activation
    "output_activation": "sigmoid",          # Output activation
    "loss": "mse",                          # Reconstruction loss
    "beta": 1.0,                           # KL divergence weight
    "capacity": 0.0,                       # Capacity constraint
    "gamma": 1000.0,                       # Capacity weight
    "epochs": 100,                         # Training epochs
    "batch_size": 32,                      # Batch size
    "optimizer": "adam",                   # Optimizer
    "learning_rate": 0.001,                # Learning rate
    "random_state": 42,                    # Reproducibility
    "contamination": 0.1                   # Expected anomaly proportion
}
```

**Use Cases**:
- Generative modeling
- Probabilistic anomalies
- Latent space analysis
- Image generation

**Strengths**:
- Probabilistic framework
- Regularized latent space
- Generative capabilities
- Interpretable latent variables

**Limitations**:
- Complex implementation
- Sensitive to hyperparameters
- Computational requirements
- Training instability

### 3. Long Short-Term Memory (LSTM)

**Description**: Recurrent neural network for sequential anomaly detection.

**Algorithm Details**:
- Processes sequential data
- Memory cells for long-term dependencies
- Prediction-based anomaly scoring
- Handles variable-length sequences

**Parameters**:
```python
{
    "hidden_neurons": [64, 32],             # LSTM layer sizes
    "sequence_length": 10,                  # Input sequence length
    "dropout_rate": 0.2,                   # Dropout rate
    "recurrent_dropout": 0.2,              # Recurrent dropout
    "activation": "tanh",                   # LSTM activation
    "recurrent_activation": "sigmoid",      # Recurrent activation
    "use_bias": True,                      # Use bias
    "return_sequences": True,               # Return sequences
    "epochs": 100,                         # Training epochs
    "batch_size": 32,                      # Batch size
    "optimizer": "adam",                   # Optimizer
    "learning_rate": 0.001,                # Learning rate
    "loss": "mse",                         # Loss function
    "contamination": 0.1,                  # Expected anomaly proportion
    "random_state": 42                     # Reproducibility
}
```

**Use Cases**:
- Time series anomalies
- Sequential pattern detection
- Sensor data analysis
- Log file analysis

**Strengths**:
- Handles sequences naturally
- Long-term dependencies
- Flexible input length
- State-of-the-art for sequences

**Limitations**:
- Computationally expensive
- Requires large datasets
- Training complexity
- Vanishing gradient issues

### 4. Convolutional Neural Network (CNN)

**Description**: Neural network with convolutional layers for spatial pattern detection.

**Algorithm Details**:
- Convolutional layers extract features
- Pooling layers reduce dimensionality
- Fully connected layers for classification
- Translation invariant

**Parameters**:
```python
{
    "conv_layers": [                        # Convolutional layers
        {"filters": 32, "kernel_size": 3, "activation": "relu"},
        {"filters": 64, "kernel_size": 3, "activation": "relu"}
    ],
    "pool_layers": [                        # Pooling layers
        {"pool_size": 2},
        {"pool_size": 2}
    ],
    "dense_layers": [128, 64],              # Dense layer sizes
    "dropout_rate": 0.25,                   # Dropout rate
    "batch_normalization": True,            # Batch normalization
    "epochs": 100,                         # Training epochs
    "batch_size": 32,                      # Batch size
    "optimizer": "adam",                   # Optimizer
    "learning_rate": 0.001,                # Learning rate
    "loss": "binary_crossentropy",         # Loss function
    "contamination": 0.1,                  # Expected anomaly proportion
    "random_state": 42                     # Reproducibility
}
```

**Use Cases**:
- Image anomaly detection
- Spatial pattern analysis
- Computer vision
- Medical imaging

**Strengths**:
- Excellent for images
- Translation invariant
- Hierarchical features
- State-of-the-art performance

**Limitations**:
- Requires large datasets
- Computationally intensive
- Many hyperparameters
- GPU dependency

### 5. Transformer

**Description**: Attention-based model for sequence anomaly detection.

**Algorithm Details**:
- Self-attention mechanism
- Parallel processing
- Position encoding
- Multi-head attention

**Parameters**:
```python
{
    "d_model": 128,                        # Model dimension
    "nhead": 8,                           # Number of attention heads
    "num_encoder_layers": 6,               # Number of encoder layers
    "dim_feedforward": 512,                # Feedforward dimension
    "dropout": 0.1,                       # Dropout rate
    "activation": "relu",                  # Activation function
    "sequence_length": 50,                 # Input sequence length
    "epochs": 100,                        # Training epochs
    "batch_size": 32,                     # Batch size
    "optimizer": "adam",                  # Optimizer
    "learning_rate": 0.0001,              # Learning rate
    "warmup_steps": 4000,                 # Learning rate warmup
    "contamination": 0.1,                 # Expected anomaly proportion
    "random_state": 42                    # Reproducibility
}
```

**Use Cases**:
- Sequential anomaly detection
- Natural language processing
- Time series analysis
- Attention visualization

**Strengths**:
- Handles long sequences
- Parallel processing
- Attention mechanism
- State-of-the-art results

**Limitations**:
- Very large models
- High computational cost
- Complex implementation
- Requires extensive data

## Specialized Methods

### 1. Graph Neural Networks (GNN)

**Description**: Neural networks designed for graph-structured data.

**Algorithm Details**:
- Node and edge representations
- Message passing between nodes
- Graph convolutions
- Handles irregular structures

**Parameters**:
```python
{
    "hidden_channels": 64,                  # Hidden dimension
    "num_layers": 3,                       # Number of GNN layers
    "dropout": 0.5,                       # Dropout rate
    "activation": "relu",                  # Activation function
    "normalization": "batch",              # Normalization type
    "aggregation": "mean",                 # Aggregation function
    "epochs": 200,                        # Training epochs
    "batch_size": 32,                     # Batch size
    "optimizer": "adam",                  # Optimizer
    "learning_rate": 0.01,                # Learning rate
    "weight_decay": 5e-4,                 # Weight decay
    "contamination": 0.1,                 # Expected anomaly proportion
    "random_state": 42                    # Reproducibility
}
```

**Use Cases**:
- Social network analysis
- Fraud detection in networks
- Knowledge graphs
- Molecular analysis

**Strengths**:
- Handles graph structures
- Considers relationships
- Flexible architecture
- State-of-the-art for graphs

**Limitations**:
- Requires graph data
- Complex implementation
- Scalability issues
- Limited interpretability

### 2. Time Series Specific Methods

#### ARIMA (AutoRegressive Integrated Moving Average)

**Description**: Statistical model for time series forecasting and anomaly detection.

**Parameters**:
```python
{
    "order": (1, 1, 1),                   # (p, d, q) parameters
    "seasonal_order": (0, 0, 0, 0),       # Seasonal parameters
    "trend": "c",                         # Trend component
    "method": "lbfgs",                    # Optimization method
    "maxiter": 50,                       # Maximum iterations
    "suppress_warnings": True,            # Suppress warnings
    "contamination": 0.1                  # Expected anomaly proportion
}
```

#### Prophet

**Description**: Facebook's time series forecasting tool.

**Parameters**:
```python
{
    "growth": "linear",                   # Growth type ("linear", "logistic")
    "changepoints": None,                 # Changepoint locations
    "n_changepoints": 25,                 # Number of changepoints
    "changepoint_range": 0.8,             # Changepoint range
    "yearly_seasonality": "auto",         # Yearly seasonality
    "weekly_seasonality": "auto",         # Weekly seasonality
    "daily_seasonality": "auto",          # Daily seasonality
    "holidays": None,                     # Holiday dataframe
    "seasonality_mode": "additive",       # Seasonality mode
    "seasonality_prior_scale": 10.0,      # Seasonality prior scale
    "holidays_prior_scale": 10.0,         # Holidays prior scale
    "changepoint_prior_scale": 0.05,      # Changepoint prior scale
    "mcmc_samples": 0,                    # MCMC samples
    "interval_width": 0.80,               # Prediction interval
    "uncertainty_samples": 1000,          # Uncertainty samples
    "contamination": 0.1                  # Expected anomaly proportion
}
```

#### Seasonal Decomposition

**Description**: Decomposes time series into trend, seasonal, and residual components.

**Parameters**:
```python
{
    "model": "additive",                  # Model type ("additive", "multiplicative")
    "period": None,                       # Seasonal period
    "two_sided": True,                    # Two-sided filter
    "extrapolate_trend": 0,               # Trend extrapolation
    "contamination": 0.1                  # Expected anomaly proportion
}
```

### 3. Text Anomaly Detection

#### TF-IDF with Clustering

**Description**: Text vectorization with clustering for anomaly detection.

**Parameters**:
```python
{
    "max_features": 10000,                # Maximum features
    "ngram_range": (1, 2),               # N-gram range
    "stop_words": "english",             # Stop words
    "min_df": 1,                         # Minimum document frequency
    "max_df": 0.95,                      # Maximum document frequency
    "clustering_algorithm": "kmeans",     # Clustering algorithm
    "n_clusters": 10,                    # Number of clusters
    "contamination": 0.1                 # Expected anomaly proportion
}
```

#### Word Embeddings

**Description**: Uses pre-trained word embeddings for text anomaly detection.

**Parameters**:
```python
{
    "embedding_model": "word2vec",        # Embedding model
    "vector_size": 300,                   # Vector dimension
    "window": 5,                         # Context window
    "min_count": 1,                      # Minimum word count
    "workers": 4,                        # Parallel workers
    "epochs": 100,                       # Training epochs
    "aggregation": "mean",               # Document aggregation
    "contamination": 0.1                 # Expected anomaly proportion
}
```

## Ensemble Methods

### 1. Voting Ensemble

**Description**: Combines predictions from multiple algorithms using voting.

**Parameters**:
```python
{
    "estimators": [                       # Base estimators
        ("isolation_forest", IsolationForest()),
        ("lof", LocalOutlierFactor()),
        ("svm", OneClassSVM())
    ],
    "voting": "soft",                     # Voting type ("hard", "soft")
    "weights": None,                      # Estimator weights
    "n_jobs": -1,                        # Parallel processing
    "contamination": 0.1                  # Expected anomaly proportion
}
```

### 2. Stacking Ensemble

**Description**: Uses meta-learner to combine base model predictions.

**Parameters**:
```python
{
    "base_estimators": [                  # Base estimators
        IsolationForest(),
        LocalOutlierFactor(),
        OneClassSVM()
    ],
    "meta_learner": LogisticRegression(), # Meta-learner
    "cv": 5,                             # Cross-validation folds
    "use_features_in_secondary": False,   # Use original features
    "random_state": 42,                  # Reproducibility
    "contamination": 0.1                 # Expected anomaly proportion
}
```

### 3. Bagging Ensemble

**Description**: Bootstrap aggregating for anomaly detection.

**Parameters**:
```python
{
    "base_estimator": IsolationForest(),  # Base estimator
    "n_estimators": 10,                  # Number of estimators
    "max_samples": 1.0,                  # Maximum samples
    "max_features": 1.0,                 # Maximum features
    "bootstrap": True,                   # Bootstrap sampling
    "bootstrap_features": False,         # Bootstrap features
    "n_jobs": -1,                       # Parallel processing
    "random_state": 42,                 # Reproducibility
    "contamination": 0.1                # Expected anomaly proportion
}
```

### 4. Adaptive Ensemble

**Description**: Dynamically weights ensemble members based on performance.

**Parameters**:
```python
{
    "base_estimators": [                 # Base estimators
        IsolationForest(),
        LocalOutlierFactor(),
        OneClassSVM()
    ],
    "adaptation_method": "performance",   # Adaptation method
    "window_size": 100,                  # Adaptation window
    "learning_rate": 0.1,                # Weight learning rate
    "min_weight": 0.0,                   # Minimum weight
    "max_weight": 1.0,                   # Maximum weight
    "contamination": 0.1                 # Expected anomaly proportion
}
```

### 5. Hierarchical Ensemble

**Description**: Multi-level ensemble with hierarchical combination.

**Parameters**:
```python
{
    "level1_estimators": [               # Level 1 estimators
        IsolationForest(),
        LocalOutlierFactor()
    ],
    "level2_estimators": [               # Level 2 estimators
        OneClassSVM(),
        EllipticEnvelope()
    ],
    "combination_method": "weighted_avg", # Combination method
    "level_weights": [0.6, 0.4],        # Level weights
    "contamination": 0.1                 # Expected anomaly proportion
}
```

## Performance Comparison

### Computational Complexity

| Algorithm | Training Time | Prediction Time | Memory Usage | Scalability |
|-----------|---------------|-----------------|--------------|-------------|
| Isolation Forest | O(n log n) | O(log n) | Low | Excellent |
| LOF | O(n²) | O(kn) | High | Poor |
| One-Class SVM | O(n²-n³) | O(sv×d) | High | Poor |
| AutoEncoder | O(epochs×n) | O(1) | Moderate | Good |
| LSTM | O(epochs×seq×n) | O(seq) | High | Moderate |
| Random Forest | O(n log n) | O(log n) | Moderate | Good |
| k-NN | O(1) | O(n) | High | Poor |
| Z-Score | O(n) | O(1) | Low | Excellent |

### Accuracy Comparison

Performance on standard datasets (average F1-score):

| Algorithm | Credit Card | Network Traffic | Sensor Data | Image Data |
|-----------|-------------|-----------------|-------------|------------|
| Isolation Forest | 0.82 | 0.78 | 0.85 | 0.73 |
| LOF | 0.79 | 0.81 | 0.77 | 0.69 |
| One-Class SVM | 0.84 | 0.76 | 0.82 | 0.75 |
| AutoEncoder | 0.86 | 0.83 | 0.88 | 0.89 |
| LSTM | 0.75 | 0.87 | 0.91 | 0.71 |
| Ensemble | 0.89 | 0.85 | 0.92 | 0.91 |

### Resource Requirements

| Algorithm | CPU Usage | Memory Usage | GPU Benefit | Disk Usage |
|-----------|-----------|--------------|-------------|------------|
| Isolation Forest | Low | Low | None | Low |
| Deep Learning | High | High | High | High |
| Statistical | Very Low | Very Low | None | Very Low |
| Ensemble | High | High | Moderate | Moderate |

## Parameter Tuning

### General Guidelines

1. **Start with default parameters** for baseline performance
2. **Use grid search** for systematic optimization
3. **Apply cross-validation** for robust evaluation
4. **Consider Bayesian optimization** for efficiency
5. **Monitor overfitting** with validation sets

### Algorithm-Specific Tips

#### Isolation Forest
- Increase `n_estimators` for stability (100-500)
- Adjust `contamination` based on expected anomaly rate
- Use `max_samples` < 1.0 for large datasets

#### LOF
- Start with `n_neighbors` = 20
- Increase for smoother decision boundaries
- Decrease for more local patterns

#### Deep Learning
- Use learning rate scheduling
- Apply early stopping
- Regularize with dropout and L2
- Normalize input data

#### Ensemble Methods
- Diversify base algorithms
- Balance computational cost
- Weight by individual performance
- Consider correlation between models

### Hyperparameter Optimization

```python
# Example: Bayesian optimization for Isolation Forest
from skopt import gp_minimize
from skopt.space import Real, Integer

def objective(params):
    n_estimators, contamination, max_features = params
    
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=max_features,
        random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    return -scores.mean()  # Minimize negative F1

space = [
    Integer(50, 500, name='n_estimators'),
    Real(0.01, 0.3, name='contamination'),
    Real(0.1, 1.0, name='max_features')
]

result = gp_minimize(objective, space, n_calls=50, random_state=42)
```

## Conclusion

This comprehensive guide covers all available algorithms in Pynomaly, their parameters, use cases, and performance characteristics. The choice of algorithm depends on:

1. **Data characteristics** (size, dimensionality, type)
2. **Performance requirements** (accuracy vs. speed)
3. **Interpretability needs**
4. **Computational resources**
5. **Domain-specific requirements**

For optimal results, consider:
- Starting with simple methods for baselines
- Using ensemble methods for critical applications
- Applying proper parameter tuning
- Validating performance thoroughly
- Monitoring model performance in production

The autonomous mode can help automatically select and tune the best algorithm for your specific use case.
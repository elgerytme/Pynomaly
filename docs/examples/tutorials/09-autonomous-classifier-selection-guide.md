# Autonomous Classifier Selection Guide

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 💡 [Examples](README.md) > 📁 Tutorials > 📄 09 Autonomous Classifier Selection Guide

---


## Overview

Pynomaly's autonomous mode features sophisticated classifier selection based on comprehensive data profiling and algorithm characteristics. This guide explains how the system intelligently chooses the optimal algorithms for your data.

## How Autonomous Mode Decides Which Classifiers to Use

### 1. Data Profiling Stage

The autonomous service (`AutonomousDetectionService`) performs comprehensive data analysis:

```python
# Key data characteristics analyzed:
- Sample count (n_samples)
- Feature count (n_features) 
- Feature types (numeric, categorical, temporal)
- Missing values ratio
- Data sparsity ratio
- Correlation patterns
- Complexity score (0-1 scale)
- Outlier ratio estimates
- Seasonal/trend patterns
```

### 2. Algorithm Recommendation Engine

The system uses a scoring algorithm to match datasets with optimal classifiers:

#### Algorithm Families Available:

**Statistical Algorithms:**
- ECOD (Empirical Cumulative Distribution)
- COPOD (Copula-based Outlier Detection)

**Distance-Based Algorithms:**
- KNN (K-Nearest Neighbors)
- LOF (Local Outlier Factor)
- One-Class SVM

**Isolation-Based Algorithms:**
- Isolation Forest (primary recommendation)

**Density-Based Algorithms:**
- LOF (Local Outlier Factor)

**Neural Networks:**
- AutoEncoder
- VAE (Variational AutoEncoder)

### 3. Intelligent Selection Logic

#### Primary Recommendations:

**Isolation Forest** (Default Choice):
- **When**: Always recommended as baseline
- **Confidence**: 0.8 base, +0.1 for high-dimensional data (>20 features)
- **Reasoning**: "General purpose algorithm, works well with mixed data types"
- **Best for**: Mixed data types, high-dimensional data, general anomaly detection

**Local Outlier Factor**:
- **When**: ≥70% numeric features
- **Confidence**: 0.75 base, +0.1 for small datasets (<10K samples)
- **Reasoning**: "Good for density-based anomalies in numeric data"
- **Best for**: Density-based outliers, numeric datasets, smaller datasets

**One-Class SVM**:
- **When**: <50K samples AND complexity score >0.5
- **Confidence**: 0.7
- **Reasoning**: "Handles complex decision boundaries well"
- **Best for**: Complex decision boundaries, medium-sized datasets

**Elliptic Envelope**:
- **When**: Low correlation (<0.8) AND >2 numeric features
- **Confidence**: 0.65
- **Reasoning**: "Good for Gaussian-distributed data with low correlation"
- **Best for**: Gaussian-distributed data, low feature correlation

**AutoEncoder**:
- **When**: >10K samples AND complexity score >0.6
- **Confidence**: 0.75
- **Reasoning**: "Deep learning approach for complex, large datasets"
- **Best for**: Large, complex datasets, non-linear patterns

### 4. Scoring Algorithm Details

Each algorithm receives a suitability score based on:

```python
def _calculate_algorithm_score(config: AlgorithmConfig, profile: DataProfile) -> float:
    score = 1.0
    
    # Sample size suitability
    if profile.n_samples < config.recommended_min_samples:
        score *= 0.5  # Penalty for insufficient data
    elif profile.n_samples > config.recommended_max_samples:
        score *= 0.8  # Slight penalty for very large datasets
    
    # Complexity matching
    complexity_diff = abs(config.complexity_score - profile.complexity_score)
    score *= (1.0 - complexity_diff * 0.3)
    
    # Feature type compatibility
    if profile.categorical_features and not config.supports_categorical:
        score *= 0.7
    
    # Temporal structure considerations
    if profile.has_temporal_structure and not config.supports_streaming:
        score *= 0.8
    
    # Memory and performance considerations
    if profile.dataset_size_mb > 1000:  # >1GB
        score *= (1.0 - config.memory_factor * 0.3)
    
    # Family-specific bonuses
    # ... (see implementation for full logic)
    
    return max(score, 0.1)  # Minimum viable score
```

## AutoML Integration

### Hyperparameter Optimization

The AutoML service (`AutoMLService`) provides:

1. **Algorithm Configuration Management**: 15+ pre-configured algorithms with parameter spaces
2. **Optuna-based Optimization**: TPE sampling for efficient hyperparameter search
3. **Performance-based Ranking**: Cross-validation and scoring metrics
4. **Ensemble Creation**: Automatic ensemble generation from top performers

#### Key Features:

```python
# Available optimization objectives:
- AUC (Area Under Curve)
- Precision
- Recall
- F1-Score
- Balanced Accuracy
- Detection Rate

# Algorithm families supported:
- Statistical, Distance-based, Density-based
- Isolation-based, Neural Networks, Ensemble
- Graph-based (PyGOD integration)
```

### AutoML Across Interfaces

**CLI Autonomous Mode**: ✅ Full AutoML integration
- `pynomaly auto detect data.csv --auto-tune`
- Automatic algorithm selection and optimization

**Web API**: ❌ Not yet implemented
- Missing autonomous detection endpoints
- No AutoML endpoints exposed

**Web UI**: ❌ Not yet implemented  
- No AutoML interface
- No autonomous mode UI

## Ensemble Methods Available

### 1. Weighted Voting Ensemble

```python
# EnsembleService creates ensembles with:
ensemble_config = {
    "method": "weighted_voting",
    "algorithms": [
        {"name": "IsolationForest", "params": {...}, "weight": 0.4},
        {"name": "LOF", "params": {...}, "weight": 0.3},
        {"name": "OneClassSVM", "params": {...}, "weight": 0.3}
    ],
    "voting_strategy": "soft",
    "normalize_scores": True
}
```

### 2. Available Ensemble Strategies

**Aggregation Methods:**
- Average (default)
- Weighted voting  
- Maximum
- Minimum
- Majority voting

**Ensemble Types:**
- Multi-algorithm ensembles
- Cross-validation ensembles
- Bagging-style ensembles

### 3. Family-Based Ensembles

**Current State**: ❌ Not implemented

**Proposed Implementation**:
```python
# By algorithm family
families = {
    "statistical": ["ECOD", "COPOD"],
    "distance_based": ["KNN", "LOF", "OneClassSVM"], 
    "isolation_based": ["IsolationForest"],
    "neural_networks": ["AutoEncoder", "VAE"]
}

# Family ensemble then meta-ensemble
family_ensembles = create_family_ensembles(families)
meta_ensemble = create_meta_ensemble(family_ensembles)
```

## Current Functionality Status

### ✅ Working Features

**Autonomous Mode (CLI)**:
- ✅ Data format auto-detection (CSV, JSON, Parquet, Excel)
- ✅ Intelligent preprocessing with quality assessment  
- ✅ Data profiling and algorithm recommendation
- ✅ Hyperparameter auto-tuning
- ✅ Results export (CSV, Excel, Parquet)

**AutoML Service**:
- ✅ Dataset profiling
- ✅ Algorithm recommendation (5 algorithms)
- ✅ Hyperparameter optimization with Optuna
- ✅ Ensemble creation

**Ensemble Service**:
- ✅ Multi-detector ensembles
- ✅ Weighted voting
- ✅ Diversity analysis
- ✅ Performance optimization

### ❌ Missing Features

**CLI Enhancements**:
- ❌ `--use-all-classifiers` option
- ❌ `--ensemble-by-family` option
- ❌ `--explain-choices` option
- ❌ Results analysis and explanation

**Web API**:
- ❌ `/api/autonomous/detect` endpoint
- ❌ `/api/automl/recommend` endpoint
- ❌ `/api/ensemble/create` endpoint
- ❌ `/api/explain/choices` endpoint

**Web UI**:
- ❌ Autonomous detection interface
- ❌ AutoML configuration UI
- ❌ Ensemble builder UI
- ❌ Results explanation dashboard

## Classifier Selection Examples

### Example 1: Small Tabular Dataset

```python
# Dataset: 1,000 samples, 10 features, mostly numeric
profile = DataProfile(
    n_samples=1000,
    n_features=10,
    numeric_features=8,
    categorical_features=2,
    complexity_score=0.3
)

# Recommended algorithms (in order):
1. IsolationForest (confidence: 0.8)
2. LOF (confidence: 0.85) # +0.1 for small dataset
3. EllipticEnvelope (confidence: 0.65)
```

### Example 2: Large Complex Dataset

```python
# Dataset: 100,000 samples, 50 features, mixed types
profile = DataProfile(
    n_samples=100000,
    n_features=50,
    numeric_features=35,
    categorical_features=15,
    complexity_score=0.7
)

# Recommended algorithms (in order):
1. IsolationForest (confidence: 0.9) # +0.1 for high dimensions
2. AutoEncoder (confidence: 0.75) # Complex, large dataset
3. LOF (confidence: 0.75)
4. OneClassSVM (confidence: 0.7) # Complex boundaries
```

### Example 3: Time Series Data

```python
# Dataset: 50,000 samples with temporal features
profile = DataProfile(
    n_samples=50000,
    n_features=20,
    temporal_features=3,
    has_temporal_structure=True,
    complexity_score=0.6
)

# Recommended algorithms (adjusted for temporal):
1. IsolationForest (confidence: 0.8) # Supports streaming
2. AutoEncoder (confidence: 0.75)
3. LOF (confidence: 0.6) # -0.2 for temporal structure
```

## Implementation Recommendations

### 1. Add Missing CLI Options

```bash
# Proposed new CLI options:
pynomaly auto detect data.csv --use-all-classifiers
pynomaly auto detect data.csv --ensemble-by-family 
pynomaly auto detect data.csv --explain-choices
pynomaly auto detect data.csv --analyze-results
```

### 2. Extend Web API

```python
# Proposed new endpoints:
POST /api/autonomous/detect
POST /api/automl/profile
POST /api/automl/recommend  
POST /api/ensemble/create-by-family
GET  /api/explain/algorithm-choices
```

### 3. Build Web UI Features

- Autonomous detection wizard
- Interactive algorithm selector
- Ensemble builder interface
- Results explanation dashboard
- Performance comparison views

### 4. Add Explainability Features

```python
# Algorithm choice explanations:
{
    "chosen_algorithm": "IsolationForest",
    "confidence": 0.85,
    "reasoning": {
        "primary_factors": [
            "High-dimensional data (50 features)",
            "Mixed data types supported", 
            "Efficient for large datasets"
        ],
        "data_characteristics": {
            "sample_size": "Large (100K samples)",
            "feature_complexity": "High (0.7/1.0)",
            "data_quality": "Good (0.85/1.0)"
        },
        "alternatives_considered": [
            {"algorithm": "LOF", "score": 0.72, "reason": "Good, but slower for large data"},
            {"algorithm": "OneClassSVM", "score": 0.68, "reason": "Complex boundaries, but memory intensive"}
        ]
    }
}
```

This guide provides a comprehensive understanding of how Pynomaly's autonomous mode intelligently selects classifiers and identifies areas for enhancement across all interfaces.

---

## 🔗 **Related Documentation**

### **Getting Started**
- **[Installation Guide](../../getting-started/installation.md)** - Setup and installation
- **[Quick Start](../../getting-started/quickstart.md)** - Your first detection
- **[Platform Setup](../../getting-started/platform-specific/)** - Platform-specific guides

### **User Guides**
- **[Basic Usage](../basic-usage/README.md)** - Essential functionality
- **[Advanced Features](../advanced-features/README.md)** - Sophisticated capabilities  
- **[Troubleshooting](../troubleshooting/README.md)** - Problem solving

### **Reference**
- **[Algorithm Reference](../../reference/algorithms/README.md)** - Algorithm documentation
- **[API Documentation](../../developer-guides/api-integration/README.md)** - Programming interfaces
- **[Configuration](../../reference/configuration/)** - System configuration

### **Examples**
- **[Examples & Tutorials](../../examples/README.md)** - Real-world use cases
- **[Banking Examples](../../examples/banking/)** - Financial fraud detection
- **[Notebooks](../../examples/notebooks/)** - Interactive examples

---

## 🆘 **Getting Help**

- **[Troubleshooting Guide](../troubleshooting/troubleshooting.md)** - Common issues and solutions
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs and request features
- **[GitHub Discussions](https://github.com/your-org/pynomaly/discussions)** - Ask questions and share ideas
- **[Security Issues](mailto:security@pynomaly.org)** - Report security vulnerabilities

# Pynomaly Classifier Selection Guide

## Overview
This guide explains how Pynomaly's autonomous mode selects classifiers and provides recommendations for optimal algorithm choices based on your data characteristics.

## Autonomous Mode Classifier Selection

### How Selection Works
Autonomous mode uses a **data-driven algorithm recommendation system** that analyzes your dataset characteristics and matches them with the most suitable anomaly detection algorithms.

#### Selection Process:
1. **Dataset Profiling**: Analyzes data characteristics (size, features, types, complexity)
2. **Algorithm Scoring**: Rates each algorithm's suitability based on dataset profile
3. **Confidence Assessment**: Assigns confidence scores based on expected performance
4. **Multi-Algorithm Recommendation**: Selects 3-5 best-matched algorithms

### Core Algorithms Used

#### Primary Algorithms (Always Considered):
1. **Isolation Forest** (`sklearn_IsolationForest`)
   - **Confidence**: 0.8 (High)
   - **Best for**: General-purpose, mixed data types, high-dimensional data
   - **Reasoning**: "General purpose algorithm, works well with mixed data types"

2. **Local Outlier Factor** (`sklearn_LocalOutlierFactor`)
   - **Best for**: Dense numeric data, local density analysis
   - **Condition**: `numeric_ratio >= 0.7` (70%+ numeric features)

3. **One-Class SVM** (`sklearn_OneClassSVM`)
   - **Best for**: Complex decision boundaries, non-linear patterns
   - **Condition**: `sample_size <= 10000` (computational efficiency)

4. **Elliptic Envelope** (`sklearn_EllipticEnvelope`)
   - **Best for**: Gaussian-distributed data, outlier detection
   - **Condition**: `n_features <= 50` (covariance matrix stability)

5. **AutoEncoder** (`neural_AutoEncoder`)
   - **Best for**: Large, complex datasets, deep learning approach
   - **Condition**: `sample_size >= 1000` and high complexity

### Selection Criteria and Logic

#### Data Characteristics Analyzed:
- **Sample Size**: Number of records in dataset
- **Feature Count**: Number of columns/features
- **Numeric Ratio**: Percentage of numeric vs categorical features
- **Missing Value Ratio**: Percentage of missing data
- **Complexity Score**: Based on correlations, sparsity, variance

#### Algorithm-Specific Selection Rules:

**Isolation Forest Enhancement**:
```python
if profile.n_features > 20:
    confidence += 0.1  # Better for high-dimensional data
if profile.sample_size >= 1000:
    confidence += 0.05  # More reliable with larger samples
```

**LOF (Local Outlier Factor)**:
```python
if profile.numeric_ratio >= 0.7:
    confidence = 0.75  # Strong for numeric data
    reasoning = "Excellent for dense numeric data and local outlier detection"
```

**One-Class SVM**:
```python
if profile.sample_size <= 10000:
    confidence = 0.7  # Computationally feasible
else:
    confidence = 0.4  # May be slow for large datasets
```

**AutoEncoder**:
```python
if profile.sample_size >= 1000 and complexity_score > 0.6:
    confidence = 0.8  # Excellent for complex, large datasets
    reasoning = "Deep learning approach for complex patterns"
```

## Algorithm Families and Characteristics

### Statistical Family
- **ECOD** (Empirical Cumulative Distribution Outliers)
- **COPOD** (Copula-Based Outlier Detection)
- **Best for**: Well-understood statistical distributions
- **Strengths**: Interpretable, fast, parameter-light

### Distance-Based Family
- **KNN** (K-Nearest Neighbors)
- **LOF** (Local Outlier Factor)
- **Best for**: Local density analysis, neighborhood-based patterns
- **Strengths**: Intuitive, works with various data types

### Isolation-Based Family
- **Isolation Forest**
- **Best for**: High-dimensional data, mixed data types
- **Strengths**: No assumptions about data distribution, efficient

### Density-Based Family
- **LOF** (overlaps with distance-based)
- **Best for**: Varying density regions
- **Strengths**: Handles clusters of different densities

### Neural Network Family
- **AutoEncoder**
- **VAE** (Variational AutoEncoder)
- **Best for**: Complex patterns, large datasets, non-linear relationships
- **Strengths**: Can learn complex representations

## Ensemble Methods Available

### Basic Ensemble Types
1. **Voting Ensemble**: Hard/soft voting across multiple algorithms
2. **Stacking Ensemble**: Meta-learner approach with base detectors
3. **Adaptive Ensemble**: Dynamic weight learning based on performance
4. **Average Ensemble**: Simple score averaging
5. **Max/Median Ensemble**: Conservative aggregation methods

### Advanced Ensemble Features
- **Family-Based Ensembles**: Hierarchical ensembles within algorithm families
- **Meta-Learning**: Cross-dataset knowledge transfer
- **Diversity Optimization**: Balances performance and algorithm diversity
- **Dynamic Selection**: Per-sample algorithm selection

## Usage Recommendations

### Data Size Guidelines
- **Small Datasets** (< 1,000 samples): Statistical methods, LOF, Elliptic Envelope
- **Medium Datasets** (1,000 - 10,000): Isolation Forest, One-Class SVM, LOF
- **Large Datasets** (> 10,000): Isolation Forest, AutoEncoder, ensemble methods

### Feature Count Guidelines
- **Low Dimensional** (< 10 features): All algorithms suitable
- **Medium Dimensional** (10-50 features): Isolation Forest, AutoEncoder, statistical methods
- **High Dimensional** (> 50 features): Isolation Forest, AutoEncoder (avoid Elliptic Envelope)

### Data Type Guidelines
- **Primarily Numeric**: LOF, One-Class SVM, statistical methods
- **Mixed Types**: Isolation Forest, ensemble methods
- **Categorical Heavy**: Custom preprocessing + Isolation Forest

### Performance vs Interpretability Trade-offs
- **High Interpretability**: Statistical methods (ECOD, COPOD), LOF
- **Balanced**: Isolation Forest, One-Class SVM
- **High Performance**: AutoEncoder, ensemble methods

## Interface-Specific Usage

### CLI Usage
```bash
# Autonomous mode with all classifiers
pynomaly auto detect-all data.csv --ensemble

# Family-based detection
pynomaly auto detect-by-family data.csv --family statistical isolation_based

# AutoML optimization (when enabled)
pynomaly automl optimize data.csv IsolationForest --max-trials 100
```

### API Usage
```python
# Autonomous detection
POST /api/autonomous/detect
{
  "dataset_id": "dataset-123",
  "config": {
    "max_algorithms": 5,
    "auto_tune_hyperparams": true,
    "enable_ensemble": true
  }
}

# Family-based ensemble
POST /api/autonomous/ensemble/create-by-family
{
  "families": ["statistical", "isolation_based"],
  "aggregation_method": "weighted_voting"
}
```

### Programmatic Usage
```python
from pynomaly.application.services.autonomous_service import AutonomousDetectionService

service = AutonomousDetectionService()
result = await service.detect_anomalies_autonomous(
    dataset_id="dataset-123",
    config=AutonomousConfig(
        max_algorithms=5,
        enable_ensemble=True,
        auto_tune_hyperparams=True
    )
)
```

## Performance Expectations

### Algorithm Performance Characteristics
- **Isolation Forest**: Fast training, moderate memory, good scalability
- **LOF**: Moderate training, high memory for large datasets, excellent accuracy
- **One-Class SVM**: Slow training for large datasets, low memory, good accuracy
- **AutoEncoder**: Slow training, high memory, excellent for complex patterns
- **Statistical Methods**: Very fast training, low memory, good for simple patterns

### Ensemble Performance
- **Training Time**: 3-5x longer than single algorithms
- **Memory Usage**: Proportional to number of base algorithms
- **Accuracy**: Typically 10-15% improvement over single algorithms
- **Interpretability**: Reduced but can be enhanced with explanation features

## Troubleshooting Common Issues

### Algorithm Selection Problems
1. **Too few algorithms recommended**: Check data quality and preprocessing
2. **Poor performance**: Consider ensemble methods or AutoML optimization
3. **Slow execution**: Reduce max_algorithms or use faster algorithm families

### Ensemble Issues
1. **Memory errors**: Reduce number of base algorithms or use simpler methods
2. **Poor ensemble performance**: Check base algorithm diversity
3. **Inconsistent results**: Ensure proper cross-validation and stable data splits

## Future Enhancements

### Planned Features
1. **Enhanced Explainability**: SHAP/LIME integration for algorithm choice explanations
2. **Meta-Learning**: Learning from historical optimization results
3. **Adaptive Selection**: Real-time algorithm switching based on data drift
4. **Resource-Aware Selection**: GPU/CPU resource optimization
5. **Interactive Selection**: Web UI for manual algorithm selection and tuning

### Contributing
The classifier selection logic is implemented in:
- `src/pynomaly/application/services/autonomous_service.py:482-591`
- Algorithm families in `src/pynomaly/infrastructure/adapters/`
- AutoML services in `src/pynomaly/application/services/automl_service.py`

For improvements or custom selection logic, refer to the clean architecture patterns in the codebase.
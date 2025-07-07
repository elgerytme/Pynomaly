# Comprehensive Dataset Analysis Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👤 [User Guides](../README.md) > 🔶 [Advanced Features](README.md) > 📄 Dataset Analysis Guide

---


This guide provides detailed information on analyzing different types of anomaly detection datasets using Pynomaly, including specific approaches, algorithms, and techniques for each scenario.

## Overview

Pynomaly includes a comprehensive collection of tabular datasets covering various real-world anomaly detection scenarios. Each dataset type requires specific analysis approaches and algorithm selections to achieve optimal results.

## Available Datasets

### 1. Financial Fraud Detection
**Dataset**: `financial_fraud.csv`  
**Samples**: 10,000 | **Features**: 9 | **Anomaly Rate**: 2.0%

#### Characteristics
- **Transaction amounts**: Log-normal distribution with extreme outliers
- **Temporal patterns**: Time-based fraud patterns (unusual hours)
- **Categorical features**: Merchant categories with varying risk levels
- **Velocity patterns**: Transaction frequency and amount relationships

#### Key Fraud Indicators
- **Amount anomalies**: Micro-transactions (<$1) and large transactions (>95th percentile)
- **Time anomalies**: Night transactions (11pm-5am) have higher fraud rates
- **Velocity anomalies**: High transaction frequency or large amounts per frequency
- **Merchant risk**: Certain categories (ATM, casino, luxury) show higher fraud rates

#### Recommended Algorithms
1. **IsolationForest** (Primary) - Excellent for mixed numerical/categorical features
2. **LocalOutlierFactor** - Good for density-based fraud detection
3. **OneClassSVM** - Effective for non-linear fraud boundaries

#### Analysis Approach
```python
# Feature Engineering
df['amount_log'] = np.log1p(df['transaction_amount'])
df['is_night_transaction'] = ((df['hour_of_day'] >= 23) | (df['hour_of_day'] <= 5)).astype(int)
df['velocity_score'] = df['transaction_amount'] / (df['daily_frequency'] + 1)

# Algorithm Configuration
from pynomaly.algorithms import IsolationForest
detector = IsolationForest(contamination=0.02, random_state=42)
```

#### Implementation Considerations
- **Precision focus**: Minimize false positives to avoid customer friction
- **Real-time scoring**: Sub-second response times required
- **Concept drift**: Monitor for changing fraud patterns over time
- **Cost sensitivity**: False negatives are more costly than false positives

---

### 2. Network Intrusion Detection
**Dataset**: `network_intrusion.csv`  
**Samples**: 8,000 | **Features**: 11 | **Anomaly Rate**: 5.0%

#### Characteristics
- **Traffic volume**: Highly variable packet counts and byte transfers
- **Protocol diversity**: TCP, UDP, ICMP with different risk profiles
- **Port patterns**: Standard ports vs. high ports usage
- **Timing patterns**: Connection duration and packet rate relationships

#### Key Attack Patterns
- **DDoS attacks**: High packet rate + short duration
- **Port scanning**: Many ports + small packet sizes
- **Data exfiltration**: Large transfers + long duration
- **Protocol anomalies**: Unusual protocol usage patterns

#### Recommended Algorithms
1. **IsolationForest** (Primary) - Handles network traffic characteristics well
2. **LocalOutlierFactor** - Effective for port scanning detection
3. **EllipticEnvelope** - Good baseline for clean environments

#### Analysis Approach
```python
# Feature Engineering
df['traffic_intensity'] = df['packet_count'] * df['packets_per_second']
df['is_well_known_port'] = (df['destination_port'] <= 1023).astype(int)
df['bandwidth_usage'] = df['bytes_transferred'] / (df['duration_seconds'] + 0.001)

# Multi-tier Detection
quick_detector = IsolationForest(contamination=0.05, max_features=0.8)
detailed_detector = LocalOutlierFactor(contamination=0.03)
```

#### Implementation Strategy
- **Real-time processing**: Stream-based detection with sliding windows
- **Multi-tier approach**: Fast screening + detailed analysis
- **Protocol-specific models**: Different algorithms for different protocols
- **Adaptive thresholds**: Dynamic contamination rates based on traffic patterns

---

### 3. IoT Sensor Monitoring
**Dataset**: `iot_sensors.csv`  
**Samples**: 12,000 | **Features**: 10 | **Anomaly Rate**: 3.0%

#### Characteristics
- **Temporal dependencies**: Strong daily and seasonal patterns
- **Sensor correlations**: Temperature, humidity, pressure relationships
- **Environmental factors**: External conditions affect readings
- **Drift patterns**: Gradual sensor degradation over time

#### Key Anomaly Types
- **Point anomalies**: Single bad sensor readings
- **Contextual anomalies**: Values unusual for time/environmental conditions
- **Collective anomalies**: Patterns indicating system failures
- **Sensor failures**: Stuck values or impossible readings

#### Recommended Algorithms
1. **LocalOutlierFactor** (Primary) - Excellent for contextual anomalies
2. **EllipticEnvelope** - Good for environmental correlation analysis
3. **IsolationForest** - Effective for sensor failure detection

#### Analysis Approach
```python
# Time Series Features
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
df['temp_humidity_ratio'] = df['temperature_celsius'] / (df['humidity_percent'] + 1)

# Rolling Statistics
df['temp_rolling_mean'] = df['temperature_celsius'].rolling(window=24).mean()
df['temp_deviation'] = df['temperature_celsius'] - df['temp_rolling_mean']
```

#### Implementation Considerations
- **Seasonal adjustment**: Account for daily/weekly patterns
- **Cross-sensor validation**: Use correlations for validation
- **Predictive maintenance**: Early warning for sensor failures
- **Environmental context**: Include weather and operational data

---

### 4. Manufacturing Quality Control
**Dataset**: `manufacturing_quality.csv`  
**Samples**: 6,000 | **Features**: 11 | **Anomaly Rate**: 8.0%

#### Characteristics
- **Specification limits**: Clear boundaries for acceptable quality
- **Process parameters**: Machine settings affect product quality
- **Dimensional relationships**: Correlated measurement features
- **Control charts**: Statistical process control applications

#### Key Quality Issues
- **Dimensional defects**: Out-of-specification measurements
- **Process variations**: Machine parameter deviations
- **Material defects**: Surface roughness and hardness issues
- **Systematic errors**: Consistent bias in measurements

#### Recommended Algorithms
1. **IsolationForest** (Primary) - Handles process complexity well
2. **EllipticEnvelope** - Good for specification boundary detection
3. **OneClassSVM** - Effective for complex quality boundaries

#### Analysis Approach
```python
# Specification Deviation
df['spec_deviation'] = np.sqrt(
    ((df['dimension_1_mm'] - 100) / 2) ** 2 +
    ((df['dimension_2_mm'] - 50) / 1) ** 2 +
    ((df['weight_grams'] - 500) / 10) ** 2
)

# Process Capability
df['process_stability'] = (
    df[['machine_speed_rpm', 'temperature_celsius', 'pressure_bar']]
    .rolling(window=50).std().mean(axis=1)
)
```

#### Implementation Strategy
- **Real-time quality assessment**: Inline measurement integration
- **Predictive quality**: Predict defects before they occur
- **Root cause analysis**: Link process parameters to quality issues
- **Cost optimization**: Balance quality control costs with defect costs

---

### 5. E-commerce Behavior Analysis
**Dataset**: `ecommerce_behavior.csv`  
**Samples**: 15,000 | **Features**: 12 | **Anomaly Rate**: 4.0%

#### Characteristics
- **Session patterns**: Duration, page views, interaction rates
- **Conversion funnels**: Click-through and purchase rates
- **Behavioral velocity**: Pages per minute, rapid actions
- **Purchase patterns**: Amount distributions and frequencies

#### Key Anomaly Types
- **Bot behavior**: High speed, many pages, no purchases
- **Fraud patterns**: Quick high-value purchases
- **Account takeover**: Sudden behavior changes
- **Scraping activity**: High page views, minimal interaction

#### Recommended Algorithms
1. **LocalOutlierFactor** (Primary) - Excellent for behavioral patterns
2. **IsolationForest** - Good for mixed behavioral features
3. **PyOD.COPOD** - Effective for behavioral correlation analysis

#### Analysis Approach
```python
# Behavioral Ratios
df['click_through_rate'] = df['items_clicked'] / (df['pages_viewed'] + 1)
df['conversion_rate'] = df['made_purchase'] / (df['pages_viewed'] + 1)
df['pages_per_minute'] = df['pages_viewed'] / (df['session_duration_minutes'] + 0.1)

# Anomaly Scoring
df['behavior_score'] = (
    df['pages_per_minute'] * 0.3 +
    df['click_through_rate'] * 0.2 +
    (1 - df['conversion_rate']) * 0.5
)
```

#### Implementation Strategy
- **Real-time scoring**: Session-based anomaly detection
- **Adaptive thresholds**: Dynamic scoring based on user segments
- **Multi-signal fusion**: Combine behavioral, temporal, and transactional signals
- **Risk-based actions**: Progressive response based on anomaly severity

---

### 6. Time Series Anomaly Detection
**Dataset**: `time_series_anomalies.csv`  
**Samples**: 5,000 | **Features**: 10 | **Anomaly Rate**: 6.0%

#### Characteristics
- **Trend components**: Long-term directional changes
- **Seasonal patterns**: Daily and weekly cyclical behavior
- **Noise levels**: Random variations around the trend
- **Multiple anomaly types**: Spikes, drops, level shifts, trend changes

#### Key Anomaly Types
- **Point anomalies**: Single value spikes or drops
- **Contextual anomalies**: Values unusual for the time context
- **Collective anomalies**: Sequences indicating pattern changes
- **Trend anomalies**: Sudden changes in long-term direction

#### Recommended Algorithms
1. **LocalOutlierFactor** (Primary) - Good for contextual anomalies
2. **IsolationForest** - Effective for point anomalies
3. **EllipticEnvelope** - Good for trend-adjusted detection

#### Analysis Approach
```python
# Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['value'], period=24)
df['trend'] = decomposition.trend
df['seasonal'] = decomposition.seasonal
df['residual'] = decomposition.resid

# Moving Statistics
df['moving_avg_5'] = df['value'].rolling(window=5).mean()
df['moving_std_5'] = df['value'].rolling(window=5).std()
df['deviation_score'] = abs(df['value'] - df['moving_avg_5']) / (df['moving_std_5'] + 0.001)
```

#### Implementation Strategy
- **Streaming detection**: Real-time anomaly detection with buffering
- **Context awareness**: Consider seasonal and trend context
- **Multi-scale analysis**: Different window sizes for different anomaly types
- **Change point detection**: Identify structural breaks in time series

---

### 7. High-Dimensional Data
**Dataset**: `high_dimensional.csv`  
**Samples**: 3,000 | **Features**: 54 | **Anomaly Rate**: 10.0%

#### Characteristics
- **Curse of dimensionality**: Distance concentration in high dimensions
- **Feature correlations**: Complex correlation structure
- **Sparse anomalies**: Outliers in high-dimensional subspaces
- **Computational challenges**: Algorithm scalability issues

#### Key Challenges
- **Distance metrics**: Euclidean distance becomes less meaningful
- **Feature selection**: Identifying relevant features for anomaly detection
- **Visualization**: Difficult to interpret high-dimensional results
- **Overfitting**: Risk of memorizing training data

#### Recommended Algorithms
1. **IsolationForest** (Primary) - Robust to high dimensionality
2. **PyOD.PCA** - Leverages correlation structure
3. **PyOD.ABOD** - Angle-based, less affected by dimensionality
4. **LocalOutlierFactor** - May struggle but can work with proper parameters

#### Analysis Approach
```python
# Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
df_reduced = pca.fit_transform(df[feature_cols])

# Aggregate Features
df['feature_sum'] = df[feature_cols].sum(axis=1)
df['feature_std'] = df[feature_cols].std(axis=1)
df['feature_range'] = df[feature_cols].max(axis=1) - df[feature_cols].min(axis=1)

# Distance-based Features
from sklearn.metrics.pairwise import euclidean_distances
distances = euclidean_distances(df[feature_cols])
df['avg_distance'] = distances.mean(axis=1)
```

#### Implementation Strategy
- **Feature selection**: Use variance and correlation analysis
- **Ensemble methods**: Combine multiple algorithms
- **Dimensionality reduction**: PCA/ICA preprocessing
- **Subspace analysis**: Detection in lower-dimensional projections

---

## General Implementation Guidelines

### Algorithm Selection Matrix

| Dataset Type | Primary Algorithm | Secondary | Use Case |
|--------------|------------------|-----------|----------|
| Financial Fraud | IsolationForest | LOF | Real-time fraud scoring |
| Network Intrusion | IsolationForest | LOF | Multi-tier detection |
| IoT Sensors | LOF | EllipticEnvelope | Contextual anomalies |
| Manufacturing | IsolationForest | EllipticEnvelope | Quality control |
| E-commerce | LOF | IsolationForest | Behavioral analysis |
| Time Series | LOF | IsolationForest | Temporal patterns |
| High-Dimensional | IsolationForest | PCA-based | Curse of dimensionality |

### Feature Engineering Best Practices

1. **Domain-specific features**: Create features that capture domain knowledge
2. **Ratio and interaction terms**: Combine features meaningfully
3. **Temporal features**: For time-aware datasets, add time-based features
4. **Normalization**: Scale features appropriately for the algorithm
5. **Outlier handling**: Pre-process extreme outliers if needed

### Evaluation Strategies

1. **Metrics selection**: Choose appropriate metrics for each use case
2. **Cross-validation**: Use stratified K-fold for imbalanced data
3. **Time-based splits**: For temporal data, use time-based validation
4. **Cost-sensitive evaluation**: Weight false positives vs false negatives
5. **Stability analysis**: Test algorithm stability across different runs

### Production Deployment

1. **Real-time requirements**: Consider latency and throughput needs
2. **Model monitoring**: Track performance drift over time
3. **Retraining strategy**: Periodic model updates with new data
4. **Explainability**: Provide interpretable results for stakeholders
5. **Scalability**: Design for increasing data volumes

## Example Usage Scripts

Each dataset type includes a complete analysis script in the `examples/` directory:

- `analyze_financial_fraud.py` - Complete fraud detection workflow
- `analyze_network_intrusion.py` - Network security analysis
- `analyze_iot_sensors.py` - IoT monitoring system
- `analyze_manufacturing_quality.py` - Quality control system
- `analyze_ecommerce_behavior.py` - User behavior analysis
- `analyze_time_series.py` - Time series anomaly detection
- `analyze_high_dimensional.py` - High-dimensional data analysis

Run the comprehensive analysis script to analyze all datasets:

```bash
python scripts/analyze_dataset_comprehensive.py
```

## Additional Resources

- **Dataset Generation**: `scripts/generate_comprehensive_datasets.py`
- **Autonomous Mode**: Use Pynomaly's autonomous mode for automatic algorithm selection
- **Documentation**: Detailed API documentation in `docs/api/`
- **Examples**: More examples in `examples/` directory
- **Benchmarks**: Performance benchmarks in `benchmarks/` directory

This comprehensive guide provides the foundation for effective anomaly detection across various domains using Pynomaly's algorithms and techniques.

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

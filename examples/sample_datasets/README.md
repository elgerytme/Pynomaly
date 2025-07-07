# Pynomaly Sample Datasets

This directory contains a comprehensive collection of tabular datasets for testing and demonstrating Pynomaly's anomaly detection capabilities.

## Dataset Overview

- **Total Datasets**: 8
- **Synthetic Datasets**: 7
- **Real-world Datasets**: 1
- **Generated**: 2025-06-24 16:35:02

## Synthetic Datasets

### Financial Fraud
- **File**: `synthetic/financial_fraud.csv`
- **Description**: Financial transaction fraud detection with unusual amounts, timing, and patterns
- **Samples**: 10,000
- **Features**: 9
- **Anomaly Rate**: 2.0%
- **Anomaly Types**: Transaction fraud, Unusual amounts, Timing anomalies, Location anomalies
- **Recommended Algorithms**: IsolationForest, LocalOutlierFactor, OneClassSVM

### Network Intrusion
- **File**: `synthetic/network_intrusion.csv`
- **Description**: Network traffic anomaly detection with DDoS, port scanning, and malicious patterns
- **Samples**: 8,000
- **Features**: 11
- **Anomaly Rate**: 5.0%
- **Anomaly Types**: DDoS attacks, Port scanning, Traffic volume anomalies, Protocol anomalies
- **Recommended Algorithms**: IsolationForest, EllipticEnvelope, PyOD.ABOD

### Iot Sensors
- **File**: `synthetic/iot_sensors.csv`
- **Description**: IoT sensor monitoring with failures, environmental anomalies, and drift
- **Samples**: 12,000
- **Features**: 10
- **Anomaly Rate**: 3.0%
- **Anomaly Types**: Sensor failures, Environmental anomalies, Measurement drift, Temporal anomalies
- **Recommended Algorithms**: LocalOutlierFactor, EllipticEnvelope, PyOD.KNN

### Manufacturing Quality
- **File**: `synthetic/manufacturing_quality.csv`
- **Description**: Manufacturing quality control with defective products and process variations
- **Samples**: 6,000
- **Features**: 11
- **Anomaly Rate**: 8.0%
- **Anomaly Types**: Product defects, Process variations, Machine malfunctions, Specification violations
- **Recommended Algorithms**: IsolationForest, PyOD.OCSVM, EllipticEnvelope

### Ecommerce Behavior
- **File**: `synthetic/ecommerce_behavior.csv`
- **Description**: E-commerce user behavior with bot detection and fraud patterns
- **Samples**: 15,000
- **Features**: 12
- **Anomaly Rate**: 4.0%
- **Anomaly Types**: Bot behavior, Purchase fraud, Unusual browsing patterns, Session anomalies
- **Recommended Algorithms**: LocalOutlierFactor, IsolationForest, PyOD.COPOD

### Time Series Anomalies
- **File**: `synthetic/time_series_anomalies.csv`
- **Description**: Time series with spikes, drops, trend changes, and seasonality breaks
- **Samples**: 5,000
- **Features**: 10
- **Anomaly Rate**: 6.0%
- **Anomaly Types**: Value spikes, Value drops, Trend changes, Level shifts, Seasonality breaks
- **Recommended Algorithms**: PyOD.KNN, LocalOutlierFactor, EllipticEnvelope

### High Dimensional
- **File**: `synthetic/high_dimensional.csv`
- **Description**: High-dimensional dataset with correlated features and outliers
- **Samples**: 3,000
- **Features**: 54
- **Anomaly Rate**: 10.0%
- **Anomaly Types**: High-dimensional outliers, Correlation anomalies, Feature space anomalies
- **Recommended Algorithms**: IsolationForest, PyOD.PCA, PyOD.ABOD, LocalOutlierFactor


## Real-world Datasets

### Kdd Cup 1999
- **File**: `real_world/kdd_cup_1999.csv`
- **Description**: Network intrusion detection dataset from KDD Cup 1999
- **Samples**: 10,000
- **Features**: 41
- **Anomaly Rate**: 3.3%


## Usage Examples

See the `scripts/` directory for analysis examples and the `docs/` directory for detailed guides on how to analyze each dataset type.

## Dataset Characteristics

Each dataset is designed to test different aspects of anomaly detection:

1. **Financial Fraud**: Tests detection of transaction anomalies
2. **Network Intrusion**: Tests traffic pattern anomalies  
3. **IoT Sensors**: Tests time-series and environmental anomalies
4. **Manufacturing Quality**: Tests process control anomalies
5. **E-commerce Behavior**: Tests behavioral pattern anomalies
6. **Time Series**: Tests temporal anomalies and trend changes
7. **High Dimensional**: Tests curse of dimensionality handling

## File Format

All datasets are saved as CSV files with:
- One row per sample
- Features as columns
- `is_anomaly` column (0 = normal, 1 = anomaly)
- Consistent naming conventions

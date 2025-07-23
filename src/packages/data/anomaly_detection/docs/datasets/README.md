# Example Datasets for Anomaly Detection

This directory contains various example datasets that can be used with the anomaly detection package tutorials and documentation.

## Available Datasets

### 1. credit_card_transactions.csv
- **Use case**: Financial fraud detection
- **Features**: transaction amount, time, merchant category, location risk
- **Anomalies**: Fraudulent transactions with unusual patterns

### 2. network_traffic.csv  
- **Use case**: Network intrusion detection
- **Features**: packet size, duration, protocol, byte counts
- **Anomalies**: Attack traffic with suspicious patterns

### 3. sensor_readings.csv
- **Use case**: IoT device monitoring
- **Features**: temperature, humidity, pressure, vibration, power
- **Anomalies**: Device malfunctions with extreme readings

### 4. server_metrics.csv
- **Use case**: IT infrastructure monitoring  
- **Features**: CPU, memory, disk I/O, network, response time
- **Anomalies**: Performance issues and system problems

### 5. manufacturing_quality.csv
- **Use case**: Quality control monitoring
- **Features**: part dimensions, process parameters, surface finish
- **Anomalies**: Defective parts outside specifications

### 6. user_behavior.csv
- **Use case**: User activity analysis
- **Features**: session duration, page views, clicks, login patterns
- **Anomalies**: Suspicious user behavior (bots, malicious activity)

### 7. time_series_anomalies.csv
- **Use case**: Temporal anomaly detection
- **Features**: time-indexed values with trend and seasonality
- **Anomalies**: Spikes, drops, level shifts, noise bursts

### 8. mixed_features.csv
- **Use case**: General purpose anomaly detection
- **Features**: Mix of continuous, discrete, categorical, binary features
- **Anomalies**: Points with unusual feature combinations

## Usage

These datasets can be used directly with the quickstart templates:

```python
import pandas as pd
from anomaly_detection import DetectionService

# Load any dataset
data = pd.read_csv('datasets/credit_card_transactions.csv')

# Use numeric features for detection
numeric_features = data.select_dtypes(include=['number']).values

# Detect anomalies
service = DetectionService()
result = service.detect(numeric_features, algorithm='isolation_forest')
```

## Generation

To regenerate these datasets with different parameters:

```bash
python generate_example_data.py
python generate_example_data.py --small  # For smaller test datasets
```

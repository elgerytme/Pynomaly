# Practical Anomaly Detection Examples

This directory contains real-world examples demonstrating how to use Pynomaly for various anomaly detection scenarios. Each example includes complete code, sample data, and step-by-step explanations.

## 🚀 Quick Start Examples

### [Financial Fraud Detection](./financial-fraud/)
Detect fraudulent transactions in financial data using multiple algorithms and ensemble methods.

**Use Cases:**
- Credit card fraud detection
- Insurance claims anomaly detection
- Money laundering detection

**Techniques Covered:**
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Ensemble methods

### [Network Security Monitoring](./network-security/)
Monitor network traffic for security threats and unusual patterns.

**Use Cases:**
- Intrusion detection
- DDoS attack identification
- Unusual network behavior monitoring

**Techniques Covered:**
- Autoencoder neural networks
- DBSCAN clustering
- Statistical outlier detection

### [Manufacturing Quality Control](./manufacturing/)
Detect defects and anomalies in manufacturing processes and product quality.

**Use Cases:**
- Product defect detection
- Process parameter monitoring
- Predictive maintenance

**Techniques Covered:**
- Time series anomaly detection
- Multivariate analysis
- Real-time monitoring

### [IoT Sensor Monitoring](./iot-sensors/)
Monitor IoT device data for malfunctions, sensor drift, and environmental anomalies.

**Use Cases:**
- Smart home automation
- Industrial IoT monitoring
- Environmental sensor networks

**Techniques Covered:**
- Streaming anomaly detection
- Time series analysis
- Multivariate monitoring

### [Healthcare Patient Monitoring](./healthcare/)
Detect anomalous patterns in patient data for early intervention.

**Use Cases:**
- Vital signs monitoring
- Treatment response analysis
- Epidemic outbreak detection

**Techniques Covered:**
- Statistical process control
- Change point detection
- Ensemble methods

## 📊 Data Types and Scenarios

### Tabular Data
- **Credit Card Transactions**: Detect fraudulent transactions
- **Customer Behavior**: Identify unusual purchasing patterns
- **Employee Data**: Monitor access patterns and behaviors

### Time Series Data
- **Server Metrics**: Monitor CPU, memory, and network usage
- **Stock Prices**: Detect market anomalies
- **Sensor Readings**: Monitor environmental conditions

### Image Data
- **Medical Imaging**: Detect abnormalities in X-rays, MRIs
- **Manufacturing**: Visual inspection for defects
- **Security**: Anomaly detection in surveillance footage

### Text Data
- **Log Analysis**: Detect unusual patterns in system logs
- **Social Media**: Identify spam or malicious content
- **Document Analysis**: Detect unusual document patterns

## 🛠 Implementation Approaches

### 1. Statistical Methods
```python
from pynomaly.detectors import StatisticalDetector

# Z-score based detection
detector = StatisticalDetector(method="zscore", threshold=3.0)
anomalies = detector.detect(data)
```

### 2. Machine Learning Methods
```python
from pynomaly.detectors import IsolationForest, LOF

# Isolation Forest
if_detector = IsolationForest(contamination=0.1)
if_anomalies = if_detector.fit_detect(data)

# Local Outlier Factor
lof_detector = LOF(n_neighbors=20)
lof_anomalies = lof_detector.fit_detect(data)
```

### 3. Deep Learning Methods
```python
from pynomaly.detectors import Autoencoder

# Autoencoder for complex patterns
autoencoder = Autoencoder(
    encoding_dim=32,
    contamination=0.05
)
dl_anomalies = autoencoder.fit_detect(data)
```

### 4. Ensemble Methods
```python
from pynomaly.ensemble import AnomalyEnsemble

# Combine multiple detectors
ensemble = AnomalyEnsemble([
    IsolationForest(contamination=0.1),
    LOF(n_neighbors=20),
    StatisticalDetector(method="zscore")
])
ensemble_anomalies = ensemble.fit_detect(data)
```

## 🎯 Performance Considerations

### Data Size Guidelines
- **Small datasets** (< 10K rows): Any algorithm
- **Medium datasets** (10K - 1M rows): Isolation Forest, LOF, Statistical
- **Large datasets** (> 1M rows): Streaming algorithms, distributed processing

### Algorithm Selection
- **Fast training**: Isolation Forest, Statistical methods
- **High accuracy**: Ensemble methods, Deep learning
- **Interpretability**: Statistical methods, Local Outlier Factor
- **Real-time**: Streaming algorithms, Simple statistical methods

### Memory and CPU Optimization
```python
# For large datasets, use batch processing
from pynomaly.utils import batch_detect

results = batch_detect(
    detector=detector,
    data=large_dataset,
    batch_size=10000,
    n_jobs=4  # Parallel processing
)
```

## 📈 Evaluation and Validation

### Metrics for Anomaly Detection
```python
from pynomaly.evaluation import AnomalyMetrics

metrics = AnomalyMetrics(y_true, y_pred)
print(f"Precision: {metrics.precision():.3f}")
print(f"Recall: {metrics.recall():.3f}")
print(f"F1-Score: {metrics.f1_score():.3f}")
print(f"AUC-ROC: {metrics.auc_roc():.3f}")
```

### Cross-Validation for Anomaly Detection
```python
from pynomaly.validation import anomaly_cross_validate

scores = anomaly_cross_validate(
    detector=detector,
    data=data,
    contamination=0.1,
    cv=5,
    scoring=['precision', 'recall', 'f1']
)
```

## 🔧 Deployment Patterns

### 1. Batch Processing
```python
# Process data in batches (e.g., daily, hourly)
def batch_anomaly_detection(data_path):
    data = load_data(data_path)
    detector = IsolationForest()
    anomalies = detector.fit_detect(data)
    save_results(anomalies)
    return anomalies
```

### 2. Real-time Processing
```python
# Process data as it arrives
from pynomaly.streaming import StreamingDetector

streaming_detector = StreamingDetector(
    base_detector=IsolationForest(),
    window_size=1000,
    update_frequency=100
)

for data_point in data_stream:
    is_anomaly = streaming_detector.detect_online(data_point)
    if is_anomaly:
        handle_anomaly(data_point)
```

### 3. API Integration
```python
# Deploy as a web service
from fastapi import FastAPI
from pynomaly.api import PynomalyAPI

app = FastAPI()
pynomaly_api = PynomalyAPI()

@app.post("/detect")
async def detect_anomalies(data: dict):
    result = await pynomaly_api.detect(data)
    return result
```

## 📚 Learning Path

### Beginner Level
1. Start with [Financial Fraud Detection](./financial-fraud/) example
2. Learn basic statistical methods
3. Understand evaluation metrics
4. Practice with small datasets

### Intermediate Level
1. Explore [Network Security](./network-security/) example
2. Learn machine learning methods (Isolation Forest, LOF)
3. Understand ensemble techniques
4. Work with time series data

### Advanced Level
1. Study [Manufacturing Quality Control](./manufacturing/) example
2. Implement deep learning methods
3. Build streaming detection systems
4. Deploy production solutions

## 🤝 Contributing Examples

We welcome contributions of new examples! Please follow these guidelines:

1. **Clear Use Case**: Define a specific, real-world problem
2. **Complete Code**: Provide working, executable examples
3. **Documentation**: Include detailed explanations and comments
4. **Sample Data**: Provide or generate realistic sample data
5. **Evaluation**: Show how to evaluate and interpret results

### Example Structure
```
example-name/
├── README.md                 # Overview and instructions
├── data/                     # Sample data files
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Python scripts
├── requirements.txt          # Dependencies
└── results/                  # Expected outputs
```

## 📞 Support

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Real-world use cases and implementations

---

**Note**: All examples include synthetic data to protect privacy. For production use, always validate with your specific data and requirements.

# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is Pynomaly and how does it differ from other anomaly detection libraries?

**A:** Pynomaly is a production-ready Python anomaly detection platform built with Clean Architecture principles. Key differentiators include:

- **Clean Architecture**: Modular, testable, and maintainable codebase
- **Enterprise-Ready**: Built-in data quality, monitoring, and governance features
- **Multi-Modal**: Supports tabular, time series, graph, and text data
- **AutoML Integration**: Automatic algorithm selection and hyperparameter tuning
- **Production Features**: Model versioning, drift detection, and automated retraining
- **Comprehensive**: Includes data profiling, quality assessment, and transformation

### Q: What Python versions are supported?

**A:** Pynomaly supports Python 3.11 and higher. We recommend using Python 3.11 or 3.12 for optimal performance and feature compatibility.

### Q: Can I use Pynomaly for real-time anomaly detection?

**A:** Yes! Pynomaly supports real-time anomaly detection through:

- Streaming data processing capabilities
- Low-latency prediction APIs
- Background processing with message queues
- WebSocket integration for real-time alerts
- Configurable batch and micro-batch processing

## Installation and Setup

### Q: I'm getting dependency conflicts during installation. How do I resolve them?

**A:** Try these approaches in order:

1. **Use minimal installation first:**
   ```bash
   pip install pynomaly[minimal]
   ```

2. **Create a fresh virtual environment:**
   ```bash
   python -m venv pynomaly_env
   source pynomaly_env/bin/activate  # Linux/Mac
   # or
   pynomaly_env\Scripts\activate  # Windows
   pip install pynomaly[standard]
   ```

3. **Use conda for dependency management:**
   ```bash
   conda create -n pynomaly python=3.11
   conda activate pynomaly
   pip install pynomaly[standard]
   ```

4. **Install specific extras based on your needs:**
   ```bash
   pip install pynomaly[ml,api,monitoring]
   ```

### Q: Which installation option should I choose?

**A:** Choose based on your use case:

- **`pynomaly[minimal]`**: Basic anomaly detection only
- **`pynomaly[standard]`**: Most common use cases (recommended)
- **`pynomaly[ml-all]`**: All ML algorithms including deep learning
- **`pynomaly[production]`**: Production deployment with monitoring
- **`pynomaly[all]`**: Everything included (largest installation)

### Q: How do I verify my installation is working correctly?

**A:** Run this verification script:

```python
import pynomaly
from pynomaly import AnomalyDetector
import pandas as pd
import numpy as np

# Check version
print(f"Pynomaly version: {pynomaly.__version__}")

# Test basic functionality
np.random.seed(42)
data = np.random.normal(0, 1, (100, 2))
df = pd.DataFrame(data, columns=['feature1', 'feature2'])

detector = AnomalyDetector(algorithm='IsolationForest')
detector.fit(df)
results = detector.detect(df)

print(f"Test successful! Detected {results.n_anomalies} anomalies")
```

## Data and Preprocessing

### Q: What data formats does Pynomaly support?

**A:** Pynomaly supports multiple data formats:

- **Tabular Data**: CSV, Excel, Parquet, JSON, SQL databases
- **Time Series**: Temporal data with datetime indices
- **Streaming Data**: Real-time data streams via APIs or message queues
- **Graph Data**: Network/graph structures (with pygod integration)
- **Text Data**: Document anomaly detection
- **Images**: Basic image anomaly detection capabilities

### Q: How should I handle missing values before anomaly detection?

**A:** The best approach depends on your data:

```python
from pynomaly.packages.data_transformation import DataTransformationEngine

# Assess missing value patterns first
transformer = DataTransformationEngine()
missing_analysis = transformer.analyze_missing_patterns(data)

# Choose strategy based on analysis
if missing_analysis.missing_rate < 0.05:
    # Low missing rate - simple imputation
    strategy = 'median_imputation'
elif missing_analysis.is_missing_at_random:
    # Missing at random - advanced imputation
    strategy = 'knn_imputation'
else:
    # Missing not at random - flag missing values
    strategy = 'flag_and_impute'

# Apply chosen strategy
cleaned_data = transformer.handle_missing_values(data, strategy=strategy)
```

### Q: Should I remove outliers before anomaly detection?

**A:** Generally, no! Outliers might be the anomalies you want to detect. However:

- **Do remove data errors**: Fix obvious data quality issues
- **Consider domain knowledge**: Remove known noise or measurement errors
- **Use outlier-robust preprocessing**: Use robust scaling instead of standard scaling
- **Document your decisions**: Keep track of what you remove and why

```python
# ❌ Don't do this
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # Sensitive to outliers

# ✅ Do this instead
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # Robust to outliers
```

### Q: How do I handle categorical variables?

**A:** Choose encoding method based on cardinality:

```python
from pynomaly.packages.data_transformation import CategoricalFeatureProcessor

processor = CategoricalFeatureProcessor()

# Configure based on cardinality
encoding_config = {
    'low_cardinality': 'one_hot',      # < 10 categories
    'medium_cardinality': 'target_encoding',  # 10-100 categories
    'high_cardinality': 'feature_hashing'     # > 100 categories
}

# Process categorical features
processed_data = processor.smart_encode(data, encoding_config)
```

## Algorithm Selection and Configuration

### Q: Which algorithm should I use for my data?

**A:** Use this decision tree:

```python
def recommend_algorithm(data_characteristics):
    if data_characteristics['is_high_dimensional']:
        return 'IsolationForest'  # Scales well with dimensions
    
    elif data_characteristics['has_clear_clusters']:
        return 'LocalOutlierFactor'  # Good for cluster-based outliers
    
    elif data_characteristics['is_time_series']:
        return 'TimeSeriesOutlier'  # Temporal patterns
    
    elif data_characteristics['is_linear_separable']:
        return 'OneClassSVM'  # Linear boundaries
    
    else:
        return 'AutoEnsemble'  # Combines multiple algorithms
```

**Quick recommendations:**
- **Start with**: `IsolationForest` (good general performance)
- **High dimensions**: `IsolationForest` or `AutoEncoder`
- **Dense clusters**: `LocalOutlierFactor` or `DBSCAN`
- **Time series**: `TimeSeriesOutlier` or `Prophet`
- **Uncertain**: `AutoEnsemble` (combines multiple methods)

### Q: How do I determine the right contamination rate?

**A:** Use multiple approaches:

```python
from pynomaly.estimation import ContaminationEstimator

# 1. Statistical estimation
estimator = ContaminationEstimator()
estimated_rate = estimator.estimate_statistical(data)

# 2. Domain knowledge
domain_rate = 0.01  # e.g., 1% fraud rate in finance

# 3. Elbow method
elbow_rate = estimator.estimate_elbow_method(data)

# 4. Use the most conservative (smallest) estimate
contamination = min(estimated_rate, domain_rate, elbow_rate)
print(f"Recommended contamination: {contamination:.4f}")
```

### Q: My algorithm is taking too long to train. How can I speed it up?

**A:** Try these optimization strategies:

```python
# 1. Enable parallel processing
detector = AnomalyDetector(
    algorithm='IsolationForest',
    n_jobs=-1  # Use all CPU cores
)

# 2. Use sampling for large datasets
sample_size = min(50000, len(data))
sample_data = data.sample(n=sample_size, random_state=42)
detector.fit(sample_data)

# 3. Reduce algorithm complexity
detector = AnomalyDetector(
    algorithm='IsolationForest',
    n_estimators=100,    # Reduce from default 200
    max_samples=0.5      # Use smaller sample size
)

# 4. Use approximate algorithms
detector = AnomalyDetector(
    algorithm='FastIsolationForest',
    approximate=True
)
```

## Results and Interpretation

### Q: How do I interpret anomaly scores?

**A:** Anomaly scores interpretation depends on the algorithm:

```python
# Get detailed results with interpretation
results = detector.detect_with_interpretation(data)

print(f"Anomaly scores range: {results.scores.min():.3f} to {results.scores.max():.3f}")
print(f"Threshold used: {results.threshold:.3f}")
print(f"Score interpretation: {results.score_interpretation}")

# For IsolationForest: Lower scores = more anomalous
# For LOF: Higher scores = more anomalous
# For OneClassSVM: Negative scores = anomalous

# Get percentile-based interpretation
percentiles = np.percentile(results.scores, [1, 5, 10, 90, 95, 99])
print(f"Score percentiles: {percentiles}")
```

### Q: How can I explain why a point was detected as an anomaly?

**A:** Use the explainability features:

```python
from pynomaly.explainability import AnomalyExplainer

# Create explainer
explainer = AnomalyExplainer(detector)

# Explain specific anomaly
anomaly_index = results.anomaly_indices[0]
explanation = explainer.explain_anomaly(
    data=data,
    anomaly_index=anomaly_index,
    explanation_type='feature_importance'
)

print(f"Feature contributions:")
for feature, contribution in explanation.feature_contributions.items():
    print(f"  {feature}: {contribution:.3f}")

# Visualize explanation
explanation.plot_feature_importance()
explanation.plot_local_explanation()
```

### Q: How do I know if my results are reliable?

**A:** Validate your results using multiple methods:

```python
from pynomaly.validation import ResultValidator

validator = ResultValidator()

# 1. Check result stability
stability_score = validator.check_stability(detector, data, n_runs=10)
print(f"Stability score: {stability_score:.3f}")

# 2. Validate against domain knowledge
domain_validation = validator.validate_domain_knowledge(
    results=results,
    domain_rules=['credit_score > 300', 'age > 0', 'income > 0']
)

# 3. Cross-validation for unsupervised learning
cv_scores = validator.cross_validate_unsupervised(detector, data)
print(f"CV silhouette score: {cv_scores.silhouette.mean():.3f}")

# 4. Check for data leakage
leakage_check = validator.check_data_leakage(data, results)
if leakage_check.suspicious:
    print("Warning: Possible data leakage detected")
```

## Performance and Scalability

### Q: How do I handle large datasets that don't fit in memory?

**A:** Use these strategies:

```python
# 1. Streaming processing
from pynomaly.streaming import StreamingAnomalyDetector

streaming_detector = StreamingAnomalyDetector(
    base_detector=detector,
    chunk_size=10000
)

# Process in chunks
for chunk in data_chunks:
    chunk_results = streaming_detector.process_chunk(chunk)

# 2. Distributed processing with Dask
from pynomaly.distributed import DistributedDetector

distributed_detector = DistributedDetector(
    detector=detector,
    n_workers=4,
    memory_per_worker='4GB'
)

# 3. Use memory-efficient algorithms
detector = AnomalyDetector(
    algorithm='IncrementalIsolationForest',
    memory_efficient=True,
    batch_size=5000
)
```

### Q: My predictions are too slow for real-time applications. How can I optimize?

**A:** Optimize for prediction speed:

```python
# 1. Pre-compile the model
detector.compile_for_inference()

# 2. Use batch predictions
batch_results = detector.predict_batch(data_batch)

# 3. Enable caching for repeated queries
from pynomaly.caching import PredictionCache

cache = PredictionCache(backend='redis', ttl=300)
detector.enable_caching(cache)

# 4. Use lighter algorithms for real-time
fast_detector = AnomalyDetector(
    algorithm='FastLOF',
    n_neighbors=10,  # Reduce from default 20
    approximate=True
)

# 5. Feature reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
reduced_data = pca.fit_transform(data)
```

## Production Deployment

### Q: How do I deploy Pynomaly models in production?

**A:** Multiple deployment options are available:

```python
# 1. REST API deployment
from pynomaly.deployment import APIServer

api_server = APIServer(detector)
api_server.deploy(host='0.0.0.0', port=8000)

# 2. Batch processing deployment
from pynomaly.deployment import BatchProcessor

batch_processor = BatchProcessor(
    detector=detector,
    input_source='s3://data-bucket/input/',
    output_sink='s3://data-bucket/output/',
    schedule='0 */6 * * *'  # Every 6 hours
)

# 3. Streaming deployment
from pynomaly.deployment import StreamProcessor

stream_processor = StreamProcessor(
    detector=detector,
    input_stream='kafka://topic-name',
    output_stream='kafka://alerts-topic'
)
```

### Q: How do I monitor model performance in production?

**A:** Set up comprehensive monitoring:

```python
from pynomaly.monitoring import ProductionMonitor

monitor = ProductionMonitor(detector)

# Configure monitoring metrics
monitor.configure({
    'prediction_latency': {'threshold': 100, 'unit': 'ms'},
    'throughput': {'threshold': 1000, 'unit': 'predictions/sec'},
    'error_rate': {'threshold': 0.01, 'unit': 'percentage'},
    'data_drift': {'threshold': 0.05, 'method': 'ks_test'},
    'model_degradation': {'threshold': 0.1, 'method': 'silhouette'}
})

# Set up alerts
monitor.configure_alerts(
    email=['ops@company.com'],
    slack_webhook='webhook_url',
    pagerduty_key='pd_key'
)

# Start monitoring
monitor.start()
```

### Q: How do I handle model versioning and rollbacks?

**A:** Use the MLOps integration:

```python
from pynomaly.mlops import ModelRegistry

registry = ModelRegistry(backend='mlflow')

# Register new model version
registry.register_model(
    model=new_detector,
    name='fraud_detection_model',
    version='v2.1',
    stage='staging'
)

# Promote to production with validation
validation_passed = registry.validate_model('fraud_detection_model', 'v2.1')
if validation_passed:
    registry.promote_model('fraud_detection_model', 'v2.1', 'production')

# Rollback if needed
registry.rollback_model('fraud_detection_model', 'v2.0')
```

## Troubleshooting Common Issues

### Q: I'm getting "NaN" values in my anomaly scores. What's wrong?

**A:** This usually indicates data preprocessing issues:

```python
# Debug NaN issues
print(f"Data has NaN: {data.isnull().any().any()}")
print(f"Data has inf: {np.isinf(data.select_dtypes(include=[np.number])).any().any()}")

# Fix common issues
# 1. Handle missing values
data_cleaned = data.fillna(data.median(numeric_only=True))

# 2. Handle infinite values
data_cleaned = data_cleaned.replace([np.inf, -np.inf], np.nan)
data_cleaned = data_cleaned.fillna(data_cleaned.median(numeric_only=True))

# 3. Check for constant features
constant_features = data_cleaned.columns[data_cleaned.nunique() <= 1]
if len(constant_features) > 0:
    print(f"Removing constant features: {constant_features}")
    data_cleaned = data_cleaned.drop(columns=constant_features)
```

### Q: My model performance degrades over time. How do I handle this?

**A:** Implement drift detection and automated retraining:

```python
from pynomaly.drift import DriftDetector
from pynomaly.retraining import AutoRetrainer

# Set up drift detection
drift_detector = DriftDetector(
    reference_data=training_data,
    method='ks_test',
    threshold=0.05
)

# Set up automated retraining
retrainer = AutoRetrainer(
    detector=detector,
    drift_detector=drift_detector,
    retrain_threshold=0.1,
    validation_split=0.2
)

# Monitor and retrain automatically
def production_pipeline(new_data):
    # Check for drift
    drift_detected = drift_detector.detect_drift(new_data)
    
    if drift_detected:
        print("Drift detected - triggering retraining")
        new_detector = retrainer.retrain(new_data)
        if retrainer.validate_new_model(new_detector):
            return new_detector
    
    return detector
```

### Q: How do I debug poor anomaly detection performance?

**A:** Use systematic debugging approach:

```python
from pynomaly.debugging import PerformanceDebugger

debugger = PerformanceDebugger(detector, data)

# 1. Check data quality
data_quality = debugger.analyze_data_quality()
print(f"Data quality issues: {data_quality.issues}")

# 2. Analyze feature distributions
feature_analysis = debugger.analyze_features()
print(f"Problematic features: {feature_analysis.problematic_features}")

# 3. Check algorithm suitability
algorithm_analysis = debugger.analyze_algorithm_fit()
print(f"Algorithm recommendations: {algorithm_analysis.recommendations}")

# 4. Hyperparameter sensitivity analysis
param_sensitivity = debugger.analyze_parameter_sensitivity()
print(f"Sensitive parameters: {param_sensitivity.sensitive_params}")

# 5. Generate improvement suggestions
suggestions = debugger.generate_improvement_suggestions()
print(f"Improvement suggestions: {suggestions}")
```

## Getting Help

### Q: Where can I find more examples and tutorials?

**A:** Check these resources:

- **Documentation**: [https://docs.pynomaly.io](../README.md)
- **Tutorials**: [Getting Started Tutorial](../tutorials/getting-started-tutorial.md)
- **Examples**: [GitHub Examples](https://github.com/elgerytme/Pynomaly/tree/main/examples)
- **API Reference**: [API Documentation](../api-reference/)
- **Best Practices**: [Best Practices Guide](../tutorials/best-practices.md)

### Q: How do I report bugs or request features?

**A:** 
- **Bug Reports**: [GitHub Issues](https://github.com/elgerytme/Pynomaly/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/elgerytme/Pynomaly/discussions)
- **Security Issues**: Email security@pynomaly.io
- **General Questions**: [Community Forum](https://community.pynomaly.io)

### Q: Is there commercial support available?

**A:** Yes, commercial support options include:

- **Enterprise Support**: Priority bug fixes and feature requests
- **Consulting Services**: Custom algorithm development and optimization
- **Training**: On-site training and workshops
- **Deployment Support**: Production deployment assistance

Contact enterprise@pynomaly.io for more information.

### Q: How can I contribute to the project?

**A:** We welcome contributions! See our [Contributing Guide](https://github.com/elgerytme/Pynomaly/blob/main/CONTRIBUTING.md) for:

- Code contributions
- Documentation improvements
- Bug reports and testing
- Feature suggestions
- Community support

### Q: What's the project roadmap?

**A:** Our current roadmap includes:

- **Q1 2025**: Enhanced AutoML capabilities
- **Q2 2025**: Graph neural network algorithms
- **Q3 2025**: Federated learning support
- **Q4 2025**: Advanced explainability features

See our [Roadmap](https://github.com/elgerytme/Pynomaly/blob/main/ROADMAP.md) for detailed plans.

---

**Still have questions?** Check our [comprehensive documentation](../README.md) or reach out to the community!
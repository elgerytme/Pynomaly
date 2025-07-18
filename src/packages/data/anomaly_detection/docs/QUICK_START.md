# Quick Start Tutorial

Get up and running with Pynomaly Detection in 5 minutes!

## Installation

```bash
pip install pynomaly-detection
```

## 1. Basic Anomaly Detection

```python
from pynomaly_detection import AnomalyDetector
import numpy as np

# Create sample data (normal + anomalies)
np.random.seed(42)
normal_data = np.random.normal(0, 1, (950, 5))
anomaly_data = np.random.normal(5, 1, (50, 5))
data = np.vstack([normal_data, anomaly_data])

# Create detector
detector = AnomalyDetector()

# Detect anomalies
predictions = detector.detect(data, contamination=0.05)

print(f"✅ Detected {np.sum(predictions)} anomalies out of {len(data)} samples")
```

## 2. Phase 2 Services (Recommended)

```python
from pynomaly_detection import CoreDetectionService

# Use the simplified service
service = CoreDetectionService()
result = service.detect_anomalies(data, algorithm="iforest", contamination=0.05)

print(f"🎯 Algorithm: {result.algorithm}")
print(f"🔍 Found {result.n_anomalies} anomalies")
print(f"📊 Anomaly rate: {result.n_anomalies/result.n_samples:.2%}")
```

## 3. AutoML Detection

```python
from pynomaly_detection import AutoMLService

# Let AI choose the best algorithm
automl = AutoMLService()
result = automl.auto_detect(data)

print(f"🤖 Best algorithm: {result.algorithm}")
print(f"🎯 Performance score: {result.metadata.get('performance_score', 'N/A')}")
```

## 4. Ensemble Detection

```python
from pynomaly_detection import EnsembleService

# Combine multiple algorithms
ensemble = EnsembleService()
result = ensemble.smart_ensemble(data, n_algorithms=3)

print(f"🎭 Ensemble result: {result.n_anomalies} anomalies")
print(f"📈 Algorithms used: {result.metadata.get('algorithms', 'N/A')}")
```

## 5. Explainability

```python
from pynomaly_detection import AdvancedExplainability

# Understand why predictions were made
explainer = AdvancedExplainability(
    feature_names=[f"feature_{i}" for i in range(5)]
)

# Explain the first anomaly
if result.n_anomalies > 0:
    anomaly_indices = np.where(result.predictions == 1)[0]
    explanation = explainer.explain_prediction(
        sample=data[anomaly_indices[0]],
        sample_index=anomaly_indices[0],
        detection_result=result,
        training_data=data
    )
    
    print(f"🔬 Explanation: {explanation.explanation_text}")
    print(f"📊 Confidence: {explanation.confidence:.2f}")
```

## 6. Model Persistence

```python
from pynomaly_detection import ModelPersistence

# Save your model
persistence = ModelPersistence()
model_id = persistence.save_model(
    model_data={"algorithm": result.algorithm},
    training_data=data,
    algorithm=result.algorithm,
    performance_metrics={"accuracy": 0.95}
)

print(f"💾 Model saved with ID: {model_id}")

# Load it back
loaded_model = persistence.load_model(model_id)
print(f"📂 Model loaded successfully")
```

## 7. Production Monitoring

```python
from pynomaly_detection import MonitoringAlertingSystem

# Set up monitoring
monitoring = MonitoringAlertingSystem()

# Record detection results
monitoring.record_detection_result(result, processing_time=0.5, source="tutorial")

# Get metrics
metrics = monitoring.get_current_metrics()
print(f"📊 Total samples processed: {metrics.total_samples_processed}")
print(f"🚨 Active alerts: {metrics.active_alerts}")
```

## 8. Streaming Detection

```python
from pynomaly_detection import StreamingDetector

# Set up real-time detection
stream_detector = StreamingDetector(
    algorithm="lof",
    window_size=100,
    drift_detection=True
)

# Simulate streaming data
print("🌊 Processing streaming data...")
for i in range(5):
    # Generate batch
    batch = np.random.normal(0, 1, (20, 5))
    if i == 3:  # Add anomalies in batch 3
        batch = np.random.normal(3, 1, (20, 5))
    
    # Process batch
    stream_result = stream_detector.process_batch(batch)
    
    print(f"   Batch {i+1}: {stream_result.n_anomalies} anomalies")
    
    if stream_result.drift_detected:
        print("   🚨 Drift detected!")
```

## 9. Integration Pipeline

```python
from pynomaly_detection import IntegrationManager, create_adapter

# Create integration pipeline
manager = IntegrationManager()

# Add mock data source
import tempfile
import pandas as pd

# Create temporary CSV file
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(5)])
    df.to_csv(f.name, index=False)
    csv_path = f.name

# Set up file adapter
fs_adapter = create_adapter("filesystem", {
    "source_type": "filesystem",
    "connection_params": {"base_path": tempfile.gettempdir()},
    "data_format": "csv"
})

manager.register_adapter("csv_source", fs_adapter)

# Run pipeline
print("🔄 Running integration pipeline...")
pipeline_results = manager.run_anomaly_detection_pipeline(
    source_adapters=["csv_source"],
    output_adapters=[],
    algorithm="iforest"
)

print(f"✅ Pipeline completed: {len(pipeline_results)} results")
```

## 10. Check Everything Works

```python
from pynomaly_detection import check_phase2_availability, get_version_info

# Check system status
availability = check_phase2_availability()
version_info = get_version_info()

print("\n🎉 System Status:")
print(f"   Version: {version_info['version']}")
print(f"   Phase 2 Available: {all(availability.values())}")
print(f"   Available Services: {sum(availability.values())}/{len(availability)}")

if all(availability.values()):
    print("✅ All Phase 2 features are available!")
else:
    print("⚠️  Some features may be limited")
```

## Next Steps

Now that you've completed the quick start:

1. **📚 Read the [User Guide](USER_GUIDE.md)** for detailed documentation
2. **🔧 Try the [Advanced Examples](../examples/)** for real-world scenarios
3. **🚀 Deploy to production** with monitoring and alerting
4. **🤝 Join the community** on [GitHub Discussions](https://github.com/pynomaly/pynomaly-detection/discussions)

## Complete Example

Here's everything in one script:

```python
#!/usr/bin/env python3
"""
Pynomaly Detection Quick Start - Complete Example
"""

import numpy as np
from pynomaly_detection import (
    AnomalyDetector,
    CoreDetectionService,
    AutoMLService,
    EnsembleService,
    AdvancedExplainability,
    ModelPersistence,
    MonitoringAlertingSystem,
    check_phase2_availability
)

def main():
    print("🚀 Pynomaly Detection Quick Start")
    print("=" * 40)
    
    # Check Phase 2 availability
    availability = check_phase2_availability()
    print(f"Phase 2 Available: {all(availability.values())}")
    
    # Generate sample data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (950, 5))
    anomaly_data = np.random.normal(5, 1, (50, 5))
    data = np.vstack([normal_data, anomaly_data])
    
    print(f"Data shape: {data.shape}")
    
    # 1. Basic detection
    print("\n1. Basic Detection:")
    detector = AnomalyDetector()
    predictions = detector.detect(data, contamination=0.05)
    print(f"   Detected {np.sum(predictions)} anomalies")
    
    # 2. Phase 2 services
    print("\n2. Phase 2 Services:")
    service = CoreDetectionService()
    result = service.detect_anomalies(data, algorithm="iforest", contamination=0.05)
    print(f"   Algorithm: {result.algorithm}")
    print(f"   Anomalies: {result.n_anomalies}")
    
    # 3. AutoML
    print("\n3. AutoML Detection:")
    automl = AutoMLService()
    auto_result = automl.auto_detect(data)
    print(f"   Best algorithm: {auto_result.algorithm}")
    print(f"   Anomalies: {auto_result.n_anomalies}")
    
    # 4. Ensemble
    print("\n4. Ensemble Detection:")
    ensemble = EnsembleService()
    ensemble_result = ensemble.smart_ensemble(data, n_algorithms=3)
    print(f"   Ensemble anomalies: {ensemble_result.n_anomalies}")
    
    # 5. Explainability
    print("\n5. Explainability:")
    explainer = AdvancedExplainability(feature_names=[f"feature_{i}" for i in range(5)])
    if result.n_anomalies > 0:
        anomaly_indices = np.where(result.predictions == 1)[0]
        explanation = explainer.explain_prediction(
            sample=data[anomaly_indices[0]],
            sample_index=anomaly_indices[0],
            detection_result=result,
            training_data=data
        )
        print(f"   Confidence: {explanation.confidence:.2f}")
    
    # 6. Model persistence
    print("\n6. Model Persistence:")
    persistence = ModelPersistence()
    model_id = persistence.save_model(
        model_data={"algorithm": result.algorithm},
        training_data=data,
        algorithm=result.algorithm,
        performance_metrics={"accuracy": 0.95}
    )
    print(f"   Model saved: {model_id}")
    
    # 7. Monitoring
    print("\n7. Monitoring:")
    monitoring = MonitoringAlertingSystem()
    monitoring.record_detection_result(result, processing_time=0.5, source="quickstart")
    metrics = monitoring.get_current_metrics()
    print(f"   Total samples: {metrics.total_samples_processed}")
    print(f"   Anomaly rate: {metrics.anomaly_rate:.2%}")
    
    print("\n✅ Quick start completed successfully!")
    print("Next: Check out the User Guide for more advanced features.")

if __name__ == "__main__":
    main()
```

Save this as `quickstart.py` and run:

```bash
python quickstart.py
```

You're now ready to build production-ready anomaly detection systems! 🎉
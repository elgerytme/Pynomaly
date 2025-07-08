# Anomaly Classification Taxonomy

🍞 **Breadcrumb:** 🏠 [Home](../../README.md) > 📚 [Reference](../README.md) > 🏷️ [Taxonomy](README.md) > 🔍 [Anomaly Classification](anomaly-classification-taxonomy.md)

---

## Overview

The Pynomaly anomaly classification taxonomy provides a structured approach to categorizing anomalies discovered during detection processes. This system enables consistent anomaly characterization across different domains, algorithms, and use cases.

## Core Taxonomy Dimensions

### 1. Severity Classification

The severity dimension categorizes anomalies based on their potential impact and urgency for investigation.

#### Severity Levels

| Level | Score Range | Description | Response Time | Business Impact |
|-------|-------------|-------------|---------------|-----------------|
| **Critical** | ≥0.9 | Immediate attention required | < 15 minutes | High business impact, potential security threats |
| **High** | ≥0.7 | Significant deviation detected | < 1 hour | Moderate business impact, requires investigation |
| **Medium** | ≥0.5 | Moderate anomaly observed | < 4 hours | Low business impact, should be reviewed |
| **Low** | ≥0.0 | Minor deviation detected | < 24 hours | Informational, background monitoring |

#### Severity Classification Implementation

```python
from pynomaly.domain.services.anomaly_classifiers import DefaultSeverityClassifier

# Initialize with custom thresholds
classifier = DefaultSeverityClassifier({
    "critical": 0.95,  # Custom critical threshold
    "high": 0.8,
    "medium": 0.6,
    "low": 0.0
})

# Classify anomaly
severity = classifier.classify_severity(anomaly)
```

### 2. Type Classification

The type dimension categorizes anomalies based on their structural characteristics and temporal patterns.

#### Type Categories

| Type | Description | Characteristics | Detection Methods |
|------|-------------|-----------------|-------------------|
| **Point Anomaly** | Individual data points that deviate from normal patterns | • Single observation outliers<br/>• Isolated deviations<br/>• Independent anomalies | • Statistical outlier detection<br/>• Distance-based methods<br/>• Density-based approaches |
| **Collective Anomaly** | Groups of data points that together form anomalous patterns | • Pattern-based anomalies<br/>• Sequential dependencies<br/>• Group behavior deviations | • Sequence analysis<br/>• Clustering methods<br/>• Pattern recognition |
| **Contextual Anomaly** | Points that are anomalous in specific contexts or time periods | • Context-dependent deviations<br/>• Temporal anomalies<br/>• Conditional outliers | • Time series analysis<br/>• Seasonal decomposition<br/>• Conditional modeling |

#### Type Classification Implementation

```python
from pynomaly.domain.services.anomaly_classifiers import DefaultTypeClassifier

classifier = DefaultTypeClassifier()
anomaly_type = classifier.classify_type(anomaly)

# Type-specific handling
if anomaly_type == "collective":
    # Handle pattern-based anomalies
    analyze_sequence_patterns(anomaly)
elif anomaly_type == "contextual":
    # Handle context-dependent anomalies
    analyze_temporal_context(anomaly)
```

## Classification Service Architecture

### Service Components

#### AnomalyClassificationService

The main service orchestrates the classification process:

```python
from pynomaly.application.services.anomaly_classification_service import AnomalyClassificationService

# Initialize service
service = AnomalyClassificationService()

# Classify anomaly (updates metadata in-place)
service.classify(anomaly)

# Access results
severity = anomaly.metadata.get('severity')
type_category = anomaly.metadata.get('type')
```

#### Specialized Classifiers

##### ML-Enhanced Severity Classification

```python
from pynomaly.domain.services.anomaly_classifiers import MLSeverityClassifier

# Initialize with custom ML classifier
ml_classifier = MLSeverityClassifier(ml_classifier=custom_model)
service.set_severity_classifier(ml_classifier)
```

##### Batch Processing Optimization

```python
# Enable batch processing optimization
service.use_batch_processing_classifiers()

# Process multiple anomalies efficiently
for anomaly in anomaly_batch:
    service.classify(anomaly)

# Clear cache after batch processing
service.clear_classifier_cache()
```

##### Dashboard-Friendly Classification

```python
# Enable dashboard-optimized classification
service.use_dashboard_classifiers()

# Returns human-readable categories:
# "Point Anomaly", "Pattern Anomaly", "Context Anomaly"
```

## Advanced Classification Features

### Custom Classifier Implementation

#### Severity Classifier Protocol

```python
from pynomaly.domain.services.anomaly_classifiers import SeverityClassifier
from pynomaly.domain.entities.anomaly import Anomaly

class CustomSeverityClassifier:
    def classify_severity(self, anomaly: Anomaly) -> str:
        # Custom severity logic
        score = anomaly.score.value
        feature_count = len(anomaly.data_point)
        
        if score >= 0.9 and feature_count > 5:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"
```

#### Type Classifier Protocol

```python
from pynomaly.domain.services.anomaly_classifiers import TypeClassifier

class CustomTypeClassifier:
    def classify_type(self, anomaly: Anomaly) -> str:
        # Custom type logic
        metadata = anomaly.metadata
        
        if metadata.get('sequence_length', 0) > 1:
            return "collective"
        elif metadata.get('temporal_context'):
            return "contextual"
        else:
            return "point"
```

### Classification Metadata Enhancement

The classification system enriches anomaly metadata with additional context:

```python
# Classification adds these metadata fields:
{
    "severity": "high",
    "type": "contextual",
    "classification_timestamp": "2025-01-08T10:30:00Z",
    "classifier_version": "1.0",
    "confidence_score": 0.85,
    "reasoning": "High score with temporal context indicators"
}
```

## Domain-Specific Taxonomies

### Financial Services

```python
# Financial-specific severity thresholds
financial_classifier = DefaultSeverityClassifier({
    "critical": 0.95,    # Fraud indicators
    "high": 0.85,        # Suspicious transactions
    "medium": 0.7,       # Unusual patterns
    "low": 0.5           # Minor deviations
})
```

### Industrial IoT

```python
# IoT-specific type classification
class IoTTypeClassifier:
    def classify_type(self, anomaly: Anomaly) -> str:
        sensor_data = anomaly.data_point
        
        # Check for sensor failure patterns
        if self._is_sensor_failure(sensor_data):
            return "point"
        
        # Check for cascade failures
        if self._is_cascade_failure(sensor_data):
            return "collective"
        
        # Check for environmental anomalies
        if self._is_environmental_anomaly(sensor_data):
            return "contextual"
        
        return "point"
```

### Cybersecurity

```python
# Security-specific classification
class SecuritySeverityClassifier:
    def classify_severity(self, anomaly: Anomaly) -> str:
        threat_indicators = anomaly.metadata.get('threat_indicators', [])
        
        if 'malware' in threat_indicators:
            return "critical"
        elif 'intrusion_attempt' in threat_indicators:
            return "high"
        elif 'policy_violation' in threat_indicators:
            return "medium"
        else:
            return "low"
```

## Performance Considerations

### Batch Processing

```python
# Enable batch processing for high-throughput scenarios
service.use_batch_processing_classifiers()

# Process large batches efficiently
batch_size = 1000
for i in range(0, len(anomalies), batch_size):
    batch = anomalies[i:i+batch_size]
    for anomaly in batch:
        service.classify(anomaly)
    
    # Clear cache periodically
    if i % (batch_size * 10) == 0:
        service.clear_classifier_cache()
```

### Memory Management

```python
# Monitor cache size for memory management
if isinstance(service.severity_classifier, BatchProcessingSeverityClassifier):
    cache_size = len(service.severity_classifier._batch_cache)
    if cache_size > 10000:  # Threshold
        service.clear_classifier_cache()
```

## Integration Examples

### Real-Time Classification

```python
from pynomaly.application.services.anomaly_classification_service import AnomalyClassificationService
from pynomaly.domain.services.anomaly_classifiers import DefaultSeverityClassifier

# Initialize service
service = AnomalyClassificationService()

# Real-time anomaly processing
def process_real_time_anomaly(anomaly):
    # Classify anomaly
    service.classify(anomaly)
    
    # Route based on severity
    severity = anomaly.metadata.get('severity')
    if severity == 'critical':
        send_immediate_alert(anomaly)
    elif severity == 'high':
        queue_for_investigation(anomaly)
    else:
        log_for_analysis(anomaly)
```

### Dashboard Integration

```python
# Enable dashboard-friendly classification
service.use_dashboard_classifiers()

# Generate dashboard data
def get_anomaly_dashboard_data(anomalies):
    classified_anomalies = []
    for anomaly in anomalies:
        service.classify(anomaly)
        classified_anomalies.append({
            'id': anomaly.id,
            'severity': anomaly.metadata.get('severity'),
            'type': anomaly.metadata.get('type'),
            'score': anomaly.score.value,
            'timestamp': anomaly.timestamp
        })
    return classified_anomalies
```

## Best Practices

### 1. Classifier Selection

- **Use DefaultSeverityClassifier** for general-purpose severity assessment
- **Use MLSeverityClassifier** when you have domain-specific training data
- **Use BatchProcessingSeverityClassifier** for high-throughput scenarios
- **Use DashboardTypeClassifier** for user-facing applications

### 2. Threshold Tuning

```python
# Start with default thresholds
default_thresholds = {
    "critical": 0.9,
    "high": 0.7,
    "medium": 0.5,
    "low": 0.0
}

# Adjust based on domain requirements
financial_thresholds = {
    "critical": 0.95,  # Higher threshold for financial data
    "high": 0.85,
    "medium": 0.7,
    "low": 0.5
}
```

### 3. Performance Optimization

- Clear classifier caches regularly in batch processing
- Use appropriate classifier types for your use case
- Monitor memory usage with large anomaly volumes
- Consider async classification for real-time systems

### 4. Testing Classification Logic

```python
import pytest
from pynomaly.domain.entities.anomaly import Anomaly
from pynomaly.domain.value_objects.anomaly_score import AnomalyScore

def test_severity_classification():
    # Create test anomaly
    anomaly = Anomaly(
        id="test-001",
        score=AnomalyScore(0.85),
        data_point={"feature1": 1.0, "feature2": 2.0}
    )
    
    # Test classification
    classifier = DefaultSeverityClassifier()
    severity = classifier.classify_severity(anomaly)
    
    assert severity == "high"
```

## Related Documentation

- [Anomaly Detection Service](../services/anomaly-detection-service.md)
- [Domain Entities](../entities/README.md)
- [Value Objects](../value-objects/README.md)
- [Classification Examples](../../examples/classification/README.md)

---

**Maintained by:** Architecture Team  
**Last Updated:** 2025-01-08  
**Review Frequency:** Monthly

# Getting Started with Core Package

This guide will help you get started with the Core package, which provides the fundamental domain logic for the Pynomaly anomaly detection platform.

## Prerequisites

Before you begin, ensure you have:
- Python 3.11 or higher
- Basic understanding of anomaly detection concepts
- Familiarity with Python type hints

## Installation

### Basic Installation

```bash
# Install the core package
pip install pynomaly-core

# Or install from source
cd src/packages/software/core
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

## Core Concepts

### 1. Domain-Driven Design

The Core package follows Domain-Driven Design principles:

- **Entities**: Objects with identity (Dataset, Detector, Anomaly)
- **Value Objects**: Immutable objects without identity (AnomalyScore, ContaminationRate)
- **Domain Services**: Business logic that doesn't belong to entities
- **Use Cases**: Application workflows that orchestrate domain objects

### 2. Clean Architecture

The package is organized in layers:

```
┌─────────────────────┐
│  Application Layer  │ ← Use Cases, Application Services
├─────────────────────┤
│    Domain Layer     │ ← Entities, Value Objects, Domain Services
├─────────────────────┤
│    Shared Layer     │ ← Types, Utilities, Configuration
└─────────────────────┘
```

## Quick Start

### Basic Anomaly Detection

```python
from pynomaly.core.domain.entities import Dataset, Detector
from pynomaly.core.domain.value_objects import ContaminationRate
from pynomaly.core.application.use_cases import DetectAnomaliesUseCase
import numpy as np

# Create a dataset
data = np.random.randn(1000, 5)
data[0:10] += 5  # Add some outliers
dataset = Dataset.from_array("sensor_data", data)

# Create a detector
detector = Detector(
    name="isolation_forest",
    algorithm="IsolationForest",
    contamination_rate=ContaminationRate(0.1)
)

# Train the detector
detector.fit(dataset)

# Create use case and detect anomalies
use_case = DetectAnomaliesUseCase()
result = use_case.execute(dataset, detector)

print(f"Detected {len(result.anomalies)} anomalies")
print(f"Anomaly rate: {result.anomaly_rate:.2%}")
```

### Working with Entities

#### Dataset Entity

```python
from pynomaly.core.domain.entities import Dataset
import pandas as pd

# Create from different sources
numpy_dataset = Dataset.from_array("numpy_data", np.random.randn(100, 5))
pandas_dataset = Dataset.from_dataframe("pandas_data", pd.DataFrame(data))
csv_dataset = Dataset.from_csv("csv_data", "data.csv")

# Access dataset properties
print(f"Dataset size: {dataset.size}")
print(f"Dataset shape: {dataset.shape}")
print(f"Feature names: {dataset.feature_names}")

# Add and remove samples
dataset.add_sample([1.0, 2.0, 3.0, 4.0, 5.0])
dataset.remove_sample(0)

# Split dataset
train_dataset, test_dataset = dataset.split(test_size=0.2)

# Normalize dataset
normalized_dataset = dataset.normalize(method="standard")
```

#### Detector Entity

```python
from pynomaly.core.domain.entities import Detector
from pynomaly.core.domain.value_objects import ContaminationRate

# Create detector
detector = Detector(
    name="lof_detector",
    algorithm="LOF",
    contamination_rate=ContaminationRate(0.05),
    hyperparameters={"n_neighbors": 20, "metric": "euclidean"}
)

# Check detector state
print(f"Is fitted: {detector.is_fitted}")
print(f"Algorithm: {detector.algorithm}")
print(f"Contamination rate: {detector.contamination_rate.percentage}%")

# Fit detector
detector.fit(train_dataset)

# Make predictions
result = detector.predict(test_dataset)
probabilities = detector.predict_proba(test_dataset)
scores = detector.decision_function(test_dataset)
```

### Working with Value Objects

#### AnomalyScore

```python
from pynomaly.core.domain.value_objects import AnomalyScore

# Create anomaly score
score = AnomalyScore(0.85)

print(f"Score value: {score.value}")
print(f"Severity: {score.severity}")
print(f"Is high: {score.is_high}")
print(f"Percentage: {score.to_percentage()}%")

# Compare scores
score1 = AnomalyScore(0.8)
score2 = AnomalyScore(0.9)
print(f"Comparison: {score1.compare(score2)}")  # -1 (score1 < score2)
```

#### ContaminationRate

```python
from pynomaly.core.domain.value_objects import ContaminationRate

# Create contamination rate
rate = ContaminationRate(0.1)

print(f"Rate value: {rate.value}")
print(f"Percentage: {rate.percentage}%")
print(f"Is high: {rate.is_high}")

# Calculate expected anomaly count
expected_anomalies = rate.to_sample_count(1000)
print(f"Expected anomalies in 1000 samples: {expected_anomalies}")
```

#### Performance Metrics

```python
from pynomaly.core.domain.value_objects import PerformanceMetrics

# Create performance metrics
metrics = PerformanceMetrics(
    precision=0.92,
    recall=0.88,
    f1_score=0.90,
    accuracy=0.95,
    auc_roc=0.93
)

print(f"Overall performance: {metrics.overall_score}")
print(f"Is good performance: {metrics.is_good}")

# Compare with other metrics
other_metrics = PerformanceMetrics(0.85, 0.82, 0.83, 0.90)
comparison = metrics.compare(other_metrics)
print(f"Performance comparison: {comparison}")
```

## Advanced Usage

### Domain Services

#### AnomalyScorer Service

```python
from pynomaly.core.domain.services import AnomalyScorer

scorer = AnomalyScorer()

# Calculate anomaly scores
scores = scorer.calculate_scores(test_data, fitted_detector)

# Normalize scores
normalized_scores = scorer.normalize_scores(raw_scores, method="min_max")

# Apply threshold
binary_predictions = scorer.threshold_scores(scores, threshold=0.8)
```

#### Statistical Analyzer

```python
from pynomaly.core.domain.services import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Analyze dataset
stats = analyzer.analyze_dataset(dataset)
print(f"Mean: {stats.mean}")
print(f"Standard deviation: {stats.std}")
print(f"Correlation matrix shape: {stats.correlation_matrix.shape}")

# Detect statistical outliers
outlier_analysis = analyzer.detect_outliers(data, method="iqr")
print(f"Outlier count: {np.sum(outlier_analysis.outliers)}")

# Analyze distribution
distribution = analyzer.analyze_distribution(data)
print(f"Distribution type: {distribution.type}")
print(f"Parameters: {distribution.parameters}")
```

### Application Services

#### Detection Orchestrator

```python
from pynomaly.core.application.services import DetectionOrchestrator
from pynomaly.core.shared.preprocessing import StandardScalingStep

orchestrator = DetectionOrchestrator()

# Run complete detection pipeline
preprocessing_steps = [StandardScalingStep()]
result = orchestrator.run_detection_pipeline(
    dataset=dataset,
    detector=detector,
    preprocessing_steps=preprocessing_steps
)

# Batch detection on multiple datasets
datasets = [dataset1, dataset2, dataset3]
results = orchestrator.batch_detection(datasets, detector, parallel=True)

# Streaming detection
async def stream_detection():
    async for result in orchestrator.streaming_detection(data_stream, detector):
        print(f"Processed batch: {result.anomaly_count} anomalies")
```

### Use Cases

#### Complete Detection Workflow

```python
from pynomaly.core.application.use_cases import (
    DetectAnomaliesUseCase,
    TrainDetectorUseCase,
    EvaluateModelUseCase
)

# Train detector
train_use_case = TrainDetectorUseCase()
detector_config = DetectorConfig(
    algorithm="isolation_forest",
    contamination_rate=0.1,
    hyperparameters={"n_estimators": 100}
)
trained_detector = train_use_case.execute(train_dataset, detector_config)

# Detect anomalies
detect_use_case = DetectAnomaliesUseCase()
detection_result = detect_use_case.execute(test_dataset, trained_detector)

# Evaluate performance
evaluate_use_case = EvaluateModelUseCase()
performance = evaluate_use_case.execute(
    detector=trained_detector,
    test_dataset=test_dataset,
    ground_truth=ground_truth_labels
)

print(f"F1 Score: {performance.f1_score}")
print(f"Precision: {performance.precision}")
print(f"Recall: {performance.recall}")
```

#### Cross-Validation

```python
from pynomaly.core.application.use_cases import EvaluateModelUseCase

evaluate_use_case = EvaluateModelUseCase()

# Perform k-fold cross-validation
cv_result = evaluate_use_case.cross_validate(
    detector=detector,
    dataset=dataset,
    k_folds=5,
    stratified=True
)

print(f"Mean F1 Score: {cv_result.mean_f1_score}")
print(f"Std F1 Score: {cv_result.std_f1_score}")
print(f"Fold scores: {cv_result.fold_scores}")
```

## Configuration and Dependency Injection

### Using the Container

```python
from pynomaly.core.shared.container import CoreContainer

# Create and configure container
container = CoreContainer()
container.config.detection.threshold.from_value(0.8)
container.config.preprocessing.normalize.from_value(True)

# Get configured services
detection_service = container.detection_service()
preprocessing_service = container.preprocessing_service()
```

### Custom Configuration

```python
from pynomaly.core.shared.config import DetectionConfig, PreprocessingConfig

# Create custom configuration
detection_config = DetectionConfig(
    threshold=0.85,
    normalize_scores=True,
    batch_size=1000,
    parallel_processing=True
)

preprocessing_config = PreprocessingConfig(
    normalize=True,
    handle_missing=True,
    remove_duplicates=True,
    scaling_method="standard"
)

# Use configuration
detector = Detector.from_config(detection_config)
preprocessor = Preprocessor.from_config(preprocessing_config)
```

## Event Handling

### Domain Events

```python
from pynomaly.core.domain.events import DetectionCompletedEvent, AnomalyDetectedEvent
from pynomaly.core.shared.events import EventPublisher

# Create event publisher
event_publisher = EventPublisher()

# Subscribe to events
def handle_detection_completed(event: DetectionCompletedEvent):
    print(f"Detection completed: {event.result.anomaly_count} anomalies")

def handle_anomaly_detected(event: AnomalyDetectedEvent):
    print(f"High-severity anomaly detected: {event.anomaly.score.value}")

event_publisher.subscribe(DetectionCompletedEvent, handle_detection_completed)
event_publisher.subscribe(AnomalyDetectedEvent, handle_anomaly_detected)

# Events are automatically published during detection
result = detector.predict(dataset)
```

### Custom Event Handlers

```python
from pynomaly.core.shared.events import EventHandler

class CustomEventHandler(EventHandler):
    def handle_detection_completed(self, event: DetectionCompletedEvent):
        # Custom handling logic
        self.log_metrics(event.result)
        self.update_dashboard(event.result)
        self.send_notifications(event.result)
    
    def handle_anomaly_detected(self, event: AnomalyDetectedEvent):
        # Custom anomaly handling
        if event.anomaly.score.value > 0.9:
            self.send_alert(event.anomaly)

# Register custom handler
event_publisher.register_handler(CustomEventHandler())
```

## Data Preprocessing

### Built-in Preprocessing Steps

```python
from pynomaly.core.shared.preprocessing import (
    StandardScalingStep,
    MinMaxScalingStep,
    MissingValueHandlerStep,
    OutlierRemovalStep
)

# Create preprocessing pipeline
preprocessing_steps = [
    MissingValueHandlerStep(strategy="mean"),
    OutlierRemovalStep(method="iqr", threshold=1.5),
    StandardScalingStep()
]

# Apply preprocessing
preprocessed_dataset = dataset.apply_preprocessing(preprocessing_steps)
```

### Custom Preprocessing

```python
from pynomaly.core.shared.preprocessing import PreprocessingStep

class CustomNormalizationStep(PreprocessingStep):
    def __init__(self, target_range=(0, 1)):
        self.target_range = target_range
    
    def process(self, data: np.ndarray) -> np.ndarray:
        min_val, max_val = self.target_range
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        
        normalized = (data - data_min) / (data_max - data_min)
        return normalized * (max_val - min_val) + min_val
    
    def reverse(self, data: np.ndarray) -> np.ndarray:
        # Implement reverse transformation
        pass

# Use custom preprocessing
custom_step = CustomNormalizationStep(target_range=(-1, 1))
preprocessed_data = custom_step.process(dataset.data)
```

## Validation and Error Handling

### Input Validation

```python
from pynomaly.core.shared.validation import (
    validate_dataset,
    validate_detector,
    validate_detection_result
)

# Validate dataset
validation_result = validate_dataset(dataset)
if not validation_result.is_valid:
    print(f"Dataset validation errors: {validation_result.errors}")

# Validate detector
validation_result = validate_detector(detector)
if not validation_result.is_valid:
    print(f"Detector validation errors: {validation_result.errors}")

# Validate detection result
validation_result = validate_detection_result(result)
if not validation_result.is_valid:
    print(f"Result validation errors: {validation_result.errors}")
```

### Exception Handling

```python
from pynomaly.core.shared.exceptions import (
    DomainError,
    InvalidDatasetError,
    DetectionError,
    ConfigurationError
)

try:
    # Perform operations
    detector.fit(dataset)
    result = detector.predict(test_dataset)
    
except InvalidDatasetError as e:
    print(f"Dataset error: {e}")
    # Handle dataset issues
    
except DetectionError as e:
    print(f"Detection error: {e}")
    # Handle detection issues
    
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle configuration issues
    
except DomainError as e:
    print(f"Domain error: {e}")
    # Handle general domain issues
```

## Testing

### Unit Testing

```python
import pytest
from pynomaly.core.domain.entities import Dataset, Detector
from pynomaly.core.domain.value_objects import ContaminationRate

class TestAnomalyDetection:
    def test_dataset_creation(self):
        data = np.random.randn(100, 5)
        dataset = Dataset.from_array("test", data)
        
        assert dataset.size == 100
        assert dataset.shape == (100, 5)
        assert dataset.name == "test"
    
    def test_detector_training(self):
        dataset = Dataset.from_array("train", np.random.randn(100, 5))
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination_rate=ContaminationRate(0.1)
        )
        
        detector.fit(dataset)
        assert detector.is_fitted
    
    def test_anomaly_detection(self):
        train_data = np.random.randn(100, 5)
        test_data = np.random.randn(50, 5)
        
        train_dataset = Dataset.from_array("train", train_data)
        test_dataset = Dataset.from_array("test", test_data)
        
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination_rate=ContaminationRate(0.1)
        )
        
        detector.fit(train_dataset)
        result = detector.predict(test_dataset)
        
        assert len(result.anomalies) > 0
        assert result.anomaly_rate > 0
```

### Integration Testing

```python
from pynomaly.core.application.use_cases import DetectAnomaliesUseCase

class TestDetectionWorkflow:
    def test_complete_workflow(self):
        # Create test data
        data = np.random.randn(1000, 5)
        data[0:10] += 3  # Add outliers
        
        dataset = Dataset.from_array("test_data", data)
        detector = Detector(
            name="test_detector",
            algorithm="isolation_forest",
            contamination_rate=ContaminationRate(0.1)
        )
        
        # Execute workflow
        detector.fit(dataset)
        use_case = DetectAnomaliesUseCase()
        result = use_case.execute(dataset, detector)
        
        # Verify results
        assert len(result.anomalies) > 0
        assert result.anomaly_rate > 0
        assert all(isinstance(anomaly.score, AnomalyScore) for anomaly in result.anomalies)
```

## Performance Optimization

### Memory Management

```python
from pynomaly.core.shared.memory import MemoryManager

memory_manager = MemoryManager()

# Monitor memory usage
with memory_manager.monitor():
    result = detector.predict(large_dataset)
    
print(f"Peak memory usage: {memory_manager.peak_usage}")
print(f"Memory efficiency: {memory_manager.efficiency_score}")
```

### Batch Processing

```python
from pynomaly.core.shared.batch import BatchProcessor

# Process large datasets in batches
batch_processor = BatchProcessor(batch_size=1000)

anomalies = []
for batch in batch_processor.process(large_dataset):
    batch_result = detector.predict(batch)
    anomalies.extend(batch_result.anomalies)

print(f"Total anomalies found: {len(anomalies)}")
```

## Best Practices

### 1. Use Type Hints

```python
from typing import List, Optional
from pynomaly.core.domain.entities import Dataset, Anomaly

def process_anomalies(anomalies: List[Anomaly]) -> Optional[Anomaly]:
    """Find the most severe anomaly."""
    if not anomalies:
        return None
    
    return max(anomalies, key=lambda a: a.score.value)
```

### 2. Handle Exceptions Gracefully

```python
def safe_detection(dataset: Dataset, detector: Detector) -> Optional[DetectionResult]:
    """Safely perform anomaly detection."""
    try:
        if not detector.is_fitted:
            detector.fit(dataset)
        return detector.predict(dataset)
    except DomainError as e:
        logger.error(f"Detection failed: {e}")
        return None
```

### 3. Use Dependency Injection

```python
from dependency_injector.wiring import inject, Provide
from pynomaly.core.shared.container import CoreContainer

@inject
def create_detection_service(
    orchestrator: DetectionOrchestrator = Provide[CoreContainer.detection_orchestrator]
) -> DetectionService:
    return DetectionService(orchestrator)
```

### 4. Validate Inputs

```python
def create_detector(name: str, algorithm: str, contamination: float) -> Detector:
    """Create detector with validation."""
    # Validate inputs
    if not name:
        raise ValueError("Detector name cannot be empty")
    
    if contamination <= 0 or contamination >= 1:
        raise ValueError("Contamination rate must be between 0 and 1")
    
    return Detector(
        name=name,
        algorithm=algorithm,
        contamination_rate=ContaminationRate(contamination)
    )
```

## Common Patterns

### 1. Factory Pattern for Detectors

```python
class DetectorFactory:
    @staticmethod
    def create_isolation_forest(contamination: float = 0.1) -> Detector:
        return Detector(
            name="isolation_forest",
            algorithm="IsolationForest",
            contamination_rate=ContaminationRate(contamination),
            hyperparameters={"n_estimators": 100}
        )
    
    @staticmethod
    def create_lof(n_neighbors: int = 20) -> Detector:
        return Detector(
            name="lof",
            algorithm="LOF",
            contamination_rate=ContaminationRate(0.1),
            hyperparameters={"n_neighbors": n_neighbors}
        )
```

### 2. Builder Pattern for Complex Datasets

```python
class DatasetBuilder:
    def __init__(self):
        self._name = None
        self._data = None
        self._metadata = {}
        self._preprocessing_steps = []
    
    def name(self, name: str) -> 'DatasetBuilder':
        self._name = name
        return self
    
    def data(self, data: np.ndarray) -> 'DatasetBuilder':
        self._data = data
        return self
    
    def metadata(self, key: str, value: Any) -> 'DatasetBuilder':
        self._metadata[key] = value
        return self
    
    def preprocessing(self, step: PreprocessingStep) -> 'DatasetBuilder':
        self._preprocessing_steps.append(step)
        return self
    
    def build(self) -> Dataset:
        dataset = Dataset.from_array(self._name, self._data, metadata=self._metadata)
        
        for step in self._preprocessing_steps:
            dataset = step.process(dataset)
        
        return dataset

# Usage
dataset = (DatasetBuilder()
    .name("sensor_data")
    .data(sensor_readings)
    .metadata("source", "production")
    .preprocessing(StandardScalingStep())
    .build())
```

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**: Use batch processing
2. **Validation errors**: Check input data types and ranges
3. **Detection failures**: Ensure detector is fitted before prediction
4. **Performance issues**: Enable parallel processing and optimize preprocessing

### Debug Mode

```python
from pynomaly.core.shared.logging import setup_debug_logging

# Enable debug logging
setup_debug_logging()

# Use debug-enabled components
detector = Detector.create_debug("isolation_forest")
result = detector.predict(dataset, debug=True)
```

## Next Steps

1. **Explore Examples**: Check the [examples](../examples/) directory
2. **Read API Documentation**: Review the [API reference](API.md)
3. **Study Architecture**: Understand the [architecture](architecture.md)
4. **Integration**: Learn how to integrate with other packages
5. **Contributing**: See the [contributing guidelines](../CONTRIBUTING.md)

## Support

- **Documentation**: [Full documentation](../docs/)
- **Examples**: [Code examples](../examples/)
- **Issues**: [GitHub Issues](https://github.com/your-org/repo/issues)
- **Community**: [Discussions](https://github.com/your-org/repo/discussions)

This getting started guide provides a solid foundation for using the Core package. The clean architecture and domain-driven design make it easy to build robust anomaly detection applications.
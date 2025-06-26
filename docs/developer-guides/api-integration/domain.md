# Domain API Reference

This document provides detailed information about the domain layer entities, value objects, and services in Pynomaly.

## Overview

The domain layer contains the core business logic of Pynomaly, implemented as pure Python without external dependencies. It follows Domain-Driven Design (DDD) principles.

## Entities

### Detector

The `Detector` entity represents an anomaly detection algorithm instance.

```python
from pynomaly.domain.entities import Detector

detector = Detector(
    name="Fraud Detector",
    algorithm="IsolationForest",
    parameters={
        "contamination": 0.1,
        "n_estimators": 100
    },
    metadata={
        "created_by": "data_team",
        "use_case": "fraud_detection"
    }
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique identifier (auto-generated) |
| `name` | `str` | Human-readable name |
| `algorithm` | `str` | Algorithm type (e.g., "IsolationForest") |
| `parameters` | `Dict[str, Any]` | Algorithm-specific parameters |
| `metadata` | `Dict[str, Any]` | Additional metadata |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last update timestamp |
| `is_trained` | `bool` | Training status |

#### Methods

##### `update_parameters(parameters: Dict[str, Any]) -> None`
Updates the detector's parameters.

```python
detector.update_parameters({
    "contamination": 0.15,
    "random_state": 42
})
```

##### `mark_as_trained() -> None`
Marks the detector as trained.

##### `mark_as_untrained() -> None`
Marks the detector as untrained.

### Dataset

The `Dataset` entity represents a collection of data for training or detection.

```python
from pynomaly.domain.entities import Dataset

dataset = Dataset(
    name="Credit Transactions",
    description="Historical credit card transactions",
    data_source="s3://bucket/transactions.csv",
    schema={
        "amount": "float",
        "merchant_id": "string",
        "timestamp": "datetime"
    }
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique identifier |
| `name` | `str` | Dataset name |
| `description` | `str` | Dataset description |
| `data_source` | `str` | Source location or identifier |
| `schema` | `Dict[str, str]` | Data schema definition |
| `metadata` | `Dict[str, Any]` | Additional metadata |
| `created_at` | `datetime` | Creation timestamp |
| `sample_count` | `int` | Number of samples |
| `feature_count` | `int` | Number of features |

#### Methods

##### `add_metadata(key: str, value: Any) -> None`
Adds metadata to the dataset.

##### `validate_schema(data: pd.DataFrame) -> bool`
Validates data against the dataset schema.

### Anomaly

The `Anomaly` entity represents a detected anomalous data point.

```python
from pynomaly.domain.entities import Anomaly

anomaly = Anomaly(
    data_point={"amount": 5000, "merchant_id": "suspicious_store"},
    detector_id="detector_123",
    anomaly_score=0.95,
    explanation="High transaction amount for this merchant",
    severity="high"
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique identifier |
| `data_point` | `Dict[str, Any]` | The anomalous data |
| `detector_id` | `str` | ID of detecting algorithm |
| `anomaly_score` | `float` | Anomaly confidence score (0-1) |
| `explanation` | `str` | Human-readable explanation |
| `severity` | `str` | Severity level (low/medium/high) |
| `detected_at` | `datetime` | Detection timestamp |
| `reviewed` | `bool` | Human review status |

### DetectionResult

The `DetectionResult` entity encapsulates the results of anomaly detection.

```python
from pynomaly.domain.entities import DetectionResult

result = DetectionResult(
    detector_id="detector_123",
    dataset_id="dataset_456",
    anomalies_detected=15,
    total_samples=1000,
    anomaly_rate=0.015,
    execution_time_ms=2500,
    metadata={
        "algorithm": "IsolationForest",
        "parameters_used": {"contamination": 0.1}
    }
)
```

## Value Objects

### AnomalyScore

Represents an anomaly confidence score with validation.

```python
from pynomaly.domain.value_objects import AnomalyScore

score = AnomalyScore(0.85)
print(score.value)  # 0.85
print(score.is_high_confidence())  # True (if > 0.8)
print(score.severity_level())  # "high"
```

#### Methods

- `is_anomaly(threshold: float = 0.5) -> bool`
- `is_high_confidence(threshold: float = 0.8) -> bool`
- `severity_level() -> str`

### ContaminationRate

Represents the expected contamination rate for training.

```python
from pynomaly.domain.value_objects import ContaminationRate

rate = ContaminationRate(0.1)  # 10% contamination
print(rate.as_percentage())  # "10.0%"
print(rate.validate())  # True if 0 < rate <= 0.5
```

### ConfidenceInterval

Represents statistical confidence intervals for anomaly scores.

```python
from pynomaly.domain.value_objects import ConfidenceInterval

interval = ConfidenceInterval(
    lower_bound=0.2,
    upper_bound=0.8,
    confidence_level=0.95
)
```

## Domain Services

### AnomalyAnalysisService

Provides business logic for analyzing anomalies.

```python
from pynomaly.domain.services import AnomalyAnalysisService

service = AnomalyAnalysisService()

# Calculate anomaly statistics
stats = service.calculate_statistics(anomalies_list)

# Determine severity
severity = service.determine_severity(anomaly_score, historical_scores)

# Generate explanation
explanation = service.generate_explanation(
    anomaly, 
    feature_contributions
)
```

#### Methods

##### `calculate_statistics(anomalies: List[Anomaly]) -> Dict[str, Any]`
Calculates statistical summaries of detected anomalies.

##### `determine_severity(score: AnomalyScore, context: Dict) -> str`
Determines anomaly severity based on score and context.

##### `generate_explanation(anomaly: Anomaly, features: Dict) -> str`
Generates human-readable explanations for anomalies.

### DetectorValidationService

Validates detector configurations and parameters.

```python
from pynomaly.domain.services import DetectorValidationService

service = DetectorValidationService()

# Validate parameters
is_valid = service.validate_parameters("IsolationForest", {
    "contamination": 0.1,
    "n_estimators": 100
})

# Get parameter schema
schema = service.get_parameter_schema("LOF")
```

### EnsembleService

Manages ensemble detection logic.

```python
from pynomaly.domain.services import EnsembleService

service = EnsembleService()

# Combine predictions
combined = service.combine_predictions(
    predictions=[True, False, True, True],
    scores=[0.9, 0.3, 0.8, 0.7],
    strategy="majority"
)
```

## Domain Events

### AnomalyDetectedEvent

Fired when an anomaly is detected.

```python
from pynomaly.domain.events import AnomalyDetectedEvent

event = AnomalyDetectedEvent(
    anomaly_id="anomaly_123",
    detector_id="detector_456",
    severity="high",
    occurred_at=datetime.now()
)
```

### DetectorTrainedEvent

Fired when a detector completes training.

```python
from pynomaly.domain.events import DetectorTrainedEvent

event = DetectorTrainedEvent(
    detector_id="detector_123",
    training_samples=1000,
    training_duration_ms=5000
)
```

## Repository Interfaces

### DetectorRepository

Interface for detector persistence.

```python
from pynomaly.domain.repositories import DetectorRepository

class DetectorRepository(ABC):
    @abstractmethod
    async def save(self, detector: Detector) -> None:
        pass
    
    @abstractmethod
    async def get(self, detector_id: str) -> Optional[Detector]:
        pass
    
    @abstractmethod
    async def list_all(self) -> List[Detector]:
        pass
    
    @abstractmethod
    async def delete(self, detector_id: str) -> None:
        pass
```

### DatasetRepository

Interface for dataset persistence.

```python
from pynomaly.domain.repositories import DatasetRepository

class DatasetRepository(ABC):
    @abstractmethod
    async def save(self, dataset: Dataset) -> None:
        pass
    
    @abstractmethod
    async def get(self, dataset_id: str) -> Optional[Dataset]:
        pass
    
    @abstractmethod
    async def get_data(self, dataset_id: str) -> pd.DataFrame:
        pass
```

## Exception Handling

### Domain Exceptions

```python
from pynomaly.domain.exceptions import (
    DomainError,
    DetectorNotFoundError,
    InvalidParametersError,
    DatasetValidationError,
    TrainingError
)

try:
    detector = repository.get("invalid_id")
except DetectorNotFoundError as e:
    print(f"Detector not found: {e}")
```

### Exception Hierarchy

```
DomainError
├── DetectorError
│   ├── DetectorNotFoundError
│   ├── DetectorNotTrainedError
│   └── InvalidParametersError
├── DatasetError
│   ├── DatasetNotFoundError
│   ├── DatasetValidationError
│   └── SchemaError
└── AnomalyError
    ├── InvalidScoreError
    └── ExplanationError
```

## Best Practices

### 1. Entity Creation
Always use factory methods or builders for complex entities:

```python
from pynomaly.domain.factories import DetectorFactory

detector = DetectorFactory.create_isolation_forest(
    name="My Detector",
    contamination=0.1
)
```

### 2. Value Object Usage
Use value objects for validated data:

```python
# Good
score = AnomalyScore(0.85)
if score.is_high_confidence():
    send_alert()

# Avoid
score = 0.85  # No validation
if score > 0.8:  # Magic number
    send_alert()
```

### 3. Domain Service Usage
Keep business logic in domain services:

```python
# Good
explanation = analysis_service.generate_explanation(anomaly, features)

# Avoid
explanation = f"Score {anomaly.score} is high"  # Business logic in application
```

### 4. Event Handling
Use domain events for cross-cutting concerns:

```python
@event_handler(AnomalyDetectedEvent)
async def handle_high_severity_anomaly(event: AnomalyDetectedEvent):
    if event.severity == "high":
        await send_alert(event)
```

## Testing

### Unit Testing Domain Objects

```python
import pytest
from pynomaly.domain.entities import Detector

def test_detector_creation():
    detector = Detector(
        name="Test Detector",
        algorithm="IsolationForest",
        parameters={"contamination": 0.1}
    )
    
    assert detector.name == "Test Detector"
    assert not detector.is_trained
    
def test_detector_training():
    detector = Detector(name="Test", algorithm="LOF")
    detector.mark_as_trained()
    
    assert detector.is_trained
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=0.0, max_value=1.0))
def test_anomaly_score_validation(score):
    anomaly_score = AnomalyScore(score)
    assert 0.0 <= anomaly_score.value <= 1.0
```
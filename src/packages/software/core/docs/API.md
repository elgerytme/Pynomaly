# Core Package API Reference

## Overview

The Core package provides the fundamental domain logic and business rules for the Pynomaly platform. This API reference covers all public classes, methods, and functions available in the package.

## Domain Entities

### Dataset

The main entity representing data for anomaly detection operations.

```python
from pynomaly.core.domain.entities import Dataset

dataset = Dataset(
    name="sensor_data",
    data=data_array,
    metadata={"source": "production"}
)
```

#### Constructor

```python
Dataset(
    name: str,
    data: Union[np.ndarray, pd.DataFrame, List],
    metadata: Optional[Dict[str, Any]] = None,
    features: Optional[List[str]] = None,
    target: Optional[str] = None
)
```

**Parameters:**
- `name`: Unique identifier for the dataset
- `data`: The actual data (numpy array, pandas DataFrame, or list)
- `metadata`: Optional metadata dictionary
- `features`: Optional list of feature names
- `target`: Optional target column name

#### Properties

```python
@property
def size(self) -> int:
    """Number of samples in the dataset."""
    
@property
def shape(self) -> Tuple[int, int]:
    """Shape of the dataset (samples, features)."""
    
@property
def feature_names(self) -> List[str]:
    """Names of the features."""
    
@property
def is_empty(self) -> bool:
    """Whether the dataset is empty."""
```

#### Methods

```python
def add_sample(self, sample: Union[np.ndarray, List]) -> None:
    """Add a new sample to the dataset."""
    
def remove_sample(self, index: int) -> None:
    """Remove a sample by index."""
    
def get_sample(self, index: int) -> np.ndarray:
    """Get a sample by index."""
    
def get_samples(self, indices: List[int]) -> np.ndarray:
    """Get multiple samples by indices."""
    
def split(self, test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple['Dataset', 'Dataset']:
    """Split dataset into train and test sets."""
    
def normalize(self, method: str = "standard") -> 'Dataset':
    """Normalize the dataset using specified method."""
    
def validate(self) -> ValidationResult:
    """Validate the dataset structure and content."""
```

#### Class Methods

```python
@classmethod
def from_array(cls, name: str, data: np.ndarray, **kwargs) -> 'Dataset':
    """Create dataset from numpy array."""
    
@classmethod
def from_dataframe(cls, name: str, df: pd.DataFrame, **kwargs) -> 'Dataset':
    """Create dataset from pandas DataFrame."""
    
@classmethod
def from_csv(cls, name: str, file_path: str, **kwargs) -> 'Dataset':
    """Create dataset from CSV file."""
    
@classmethod
def from_json(cls, name: str, file_path: str, **kwargs) -> 'Dataset':
    """Create dataset from JSON file."""
```

### Detector

Configuration and state for anomaly detection algorithms.

```python
from pynomaly.core.domain.entities import Detector

detector = Detector(
    name="isolation_forest",
    algorithm="IsolationForest",
    contamination_rate=ContaminationRate(0.1)
)
```

#### Constructor

```python
Detector(
    name: str,
    algorithm: str,
    contamination_rate: ContaminationRate,
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None
)
```

**Parameters:**
- `name`: Unique identifier for the detector
- `algorithm`: Algorithm name (e.g., "IsolationForest", "LOF")
- `contamination_rate`: Expected contamination rate
- `hyperparameters`: Algorithm-specific parameters
- `random_state`: Random seed for reproducibility

#### Properties

```python
@property
def is_fitted(self) -> bool:
    """Whether the detector has been fitted."""
    
@property
def algorithm_type(self) -> str:
    """Type of the underlying algorithm."""
    
@property
def feature_count(self) -> Optional[int]:
    """Number of features the detector expects."""
```

#### Methods

```python
def fit(self, dataset: Dataset) -> None:
    """Fit the detector to the dataset."""
    
def predict(self, dataset: Dataset) -> DetectionResult:
    """Predict anomalies in the dataset."""
    
def predict_proba(self, dataset: Dataset) -> np.ndarray:
    """Predict anomaly probabilities."""
    
def decision_function(self, dataset: Dataset) -> np.ndarray:
    """Compute anomaly scores."""
    
def validate_configuration(self) -> ValidationResult:
    """Validate detector configuration."""
    
def get_feature_importance(self) -> Optional[np.ndarray]:
    """Get feature importance scores if available."""
```

### Anomaly

Represents a detected anomaly with score and metadata.

```python
from pynomaly.core.domain.entities import Anomaly

anomaly = Anomaly(
    id="anomaly_001",
    dataset_id="dataset_001",
    score=AnomalyScore(0.95),
    timestamp=Timestamp.now(),
    features={"temperature": 120.5}
)
```

#### Constructor

```python
Anomaly(
    id: str,
    dataset_id: str,
    score: AnomalyScore,
    timestamp: Timestamp,
    features: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
)
```

#### Properties

```python
@property
def is_severe(self) -> bool:
    """Whether the anomaly is considered severe."""
    
@property
def confidence_level(self) -> float:
    """Confidence level of the anomaly detection."""
    
@property
def feature_contributions(self) -> Dict[str, float]:
    """Feature contributions to the anomaly score."""
```

#### Methods

```python
def explain(self, explainer: Optional[Explainer] = None) -> Explanation:
    """Generate explanation for the anomaly."""
    
def to_dict(self) -> Dict[str, Any]:
    """Convert anomaly to dictionary."""
    
def to_json(self) -> str:
    """Convert anomaly to JSON string."""
```

### DetectionResult

Container for anomaly detection results.

```python
from pynomaly.core.domain.entities import DetectionResult

result = DetectionResult(
    dataset_id="dataset_001",
    detector_id="detector_001",
    anomalies=anomalies_list,
    statistics=detection_stats
)
```

#### Constructor

```python
DetectionResult(
    dataset_id: str,
    detector_id: str,
    anomalies: List[Anomaly],
    statistics: DetectionStatistics,
    timestamp: Optional[Timestamp] = None
)
```

#### Properties

```python
@property
def anomaly_count(self) -> int:
    """Number of detected anomalies."""
    
@property
def anomaly_rate(self) -> float:
    """Percentage of anomalies detected."""
    
@property
def scores(self) -> np.ndarray:
    """Array of all anomaly scores."""
    
@property
def predictions(self) -> np.ndarray:
    """Binary predictions (1 for anomaly, 0 for normal)."""
```

#### Methods

```python
def get_top_anomalies(self, n: int = 10) -> List[Anomaly]:
    """Get top N anomalies by score."""
    
def get_anomalies_by_threshold(self, threshold: float) -> List[Anomaly]:
    """Get anomalies above threshold."""
    
def filter_by_score(self, min_score: float, max_score: float = 1.0) -> List[Anomaly]:
    """Filter anomalies by score range."""
    
def to_dataframe(self) -> pd.DataFrame:
    """Convert results to pandas DataFrame."""
    
def save(self, file_path: str, format: str = "json") -> None:
    """Save results to file."""
```

## Value Objects

### AnomalyScore

Immutable value object representing an anomaly score.

```python
from pynomaly.core.domain.value_objects import AnomalyScore

score = AnomalyScore(0.95)
```

#### Constructor

```python
AnomalyScore(value: float)
```

**Parameters:**
- `value`: Score value between 0.0 and 1.0

#### Properties

```python
@property
def value(self) -> float:
    """The score value."""
    
@property
def severity(self) -> str:
    """Severity level based on score."""
    
@property
def is_high(self) -> bool:
    """Whether score is considered high."""
```

#### Methods

```python
def compare(self, other: 'AnomalyScore') -> int:
    """Compare with another score."""
    
def to_percentage(self) -> float:
    """Convert to percentage."""
    
def normalize(self, min_val: float, max_val: float) -> 'AnomalyScore':
    """Normalize score to range."""
```

### ContaminationRate

Represents the expected proportion of anomalies in a dataset.

```python
from pynomaly.core.domain.value_objects import ContaminationRate

rate = ContaminationRate(0.1)  # 10% contamination
```

#### Constructor

```python
ContaminationRate(value: float)
```

**Parameters:**
- `value`: Contamination rate between 0.0 and 1.0

#### Properties

```python
@property
def value(self) -> float:
    """The contamination rate value."""
    
@property
def percentage(self) -> float:
    """Rate as percentage."""
    
@property
def is_high(self) -> bool:
    """Whether contamination is considered high."""
```

### ConfidenceInterval

Statistical confidence interval for anomaly detection.

```python
from pynomaly.core.domain.value_objects import ConfidenceInterval

interval = ConfidenceInterval(
    lower_bound=0.85,
    upper_bound=0.95,
    confidence_level=0.95
)
```

#### Constructor

```python
ConfidenceInterval(
    lower_bound: float,
    upper_bound: float,
    confidence_level: float = 0.95
)
```

#### Properties

```python
@property
def width(self) -> float:
    """Width of the confidence interval."""
    
@property
def midpoint(self) -> float:
    """Midpoint of the interval."""
    
@property
def margin_of_error(self) -> float:
    """Margin of error."""
```

#### Methods

```python
def contains(self, value: float) -> bool:
    """Check if value is within interval."""
    
def overlaps(self, other: 'ConfidenceInterval') -> bool:
    """Check if intervals overlap."""
```

### PerformanceMetrics

Collection of model performance metrics.

```python
from pynomaly.core.domain.value_objects import PerformanceMetrics

metrics = PerformanceMetrics(
    precision=0.92,
    recall=0.88,
    f1_score=0.90,
    accuracy=0.95
)
```

#### Constructor

```python
PerformanceMetrics(
    precision: float,
    recall: float,
    f1_score: float,
    accuracy: float,
    auc_roc: Optional[float] = None,
    auc_pr: Optional[float] = None
)
```

#### Properties

```python
@property
def is_good(self) -> bool:
    """Whether metrics indicate good performance."""
    
@property
def overall_score(self) -> float:
    """Overall performance score."""
```

#### Methods

```python
def compare(self, other: 'PerformanceMetrics') -> Dict[str, float]:
    """Compare with other metrics."""
    
def to_dict(self) -> Dict[str, float]:
    """Convert to dictionary."""
```

## Application Services

### DetectionOrchestrator

Main service for orchestrating anomaly detection workflows.

```python
from pynomaly.core.application.services import DetectionOrchestrator

orchestrator = DetectionOrchestrator()
```

#### Methods

```python
def run_detection_pipeline(
    self,
    dataset: Dataset,
    detector: Detector,
    preprocessing_steps: Optional[List[PreprocessingStep]] = None,
    validation_enabled: bool = True
) -> DetectionResult:
    """Run complete detection pipeline."""
    
def batch_detection(
    self,
    datasets: List[Dataset],
    detector: Detector,
    parallel: bool = True
) -> List[DetectionResult]:
    """Run detection on multiple datasets."""
    
def streaming_detection(
    self,
    data_stream: AsyncIterator[Dataset],
    detector: Detector
) -> AsyncIterator[DetectionResult]:
    """Run detection on streaming data."""
```

### AnomalyScorer

Service for calculating and normalizing anomaly scores.

```python
from pynomaly.core.application.services import AnomalyScorer

scorer = AnomalyScorer()
```

#### Methods

```python
def calculate_scores(
    self,
    data: np.ndarray,
    detector: Detector
) -> np.ndarray:
    """Calculate anomaly scores."""
    
def normalize_scores(
    self,
    scores: np.ndarray,
    method: str = "min_max"
) -> np.ndarray:
    """Normalize scores to [0, 1] range."""
    
def threshold_scores(
    self,
    scores: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Apply threshold to scores."""
```

### StatisticalAnalyzer

Service for statistical analysis of datasets and results.

```python
from pynomaly.core.application.services import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
```

#### Methods

```python
def analyze_dataset(self, dataset: Dataset) -> DatasetStatistics:
    """Analyze dataset statistics."""
    
def analyze_distribution(self, data: np.ndarray) -> DistributionAnalysis:
    """Analyze data distribution."""
    
def detect_outliers(self, data: np.ndarray) -> OutlierAnalysis:
    """Detect statistical outliers."""
    
def correlation_analysis(self, data: np.ndarray) -> CorrelationMatrix:
    """Perform correlation analysis."""
```

## Use Cases

### DetectAnomaliesUseCase

Main use case for detecting anomalies in datasets.

```python
from pynomaly.core.application.use_cases import DetectAnomaliesUseCase

use_case = DetectAnomaliesUseCase()
result = use_case.execute(dataset, detector)
```

#### Methods

```python
def execute(
    self,
    dataset: Dataset,
    detector: Detector,
    options: Optional[DetectionOptions] = None
) -> DetectionResult:
    """Execute anomaly detection."""
    
def validate_inputs(
    self,
    dataset: Dataset,
    detector: Detector
) -> ValidationResult:
    """Validate inputs before detection."""
```

### TrainDetectorUseCase

Use case for training anomaly detectors.

```python
from pynomaly.core.application.use_cases import TrainDetectorUseCase

use_case = TrainDetectorUseCase()
trained_detector = use_case.execute(dataset, detector_config)
```

#### Methods

```python
def execute(
    self,
    dataset: Dataset,
    detector_config: DetectorConfig,
    training_options: Optional[TrainingOptions] = None
) -> Detector:
    """Train detector on dataset."""
    
def validate_training_data(self, dataset: Dataset) -> ValidationResult:
    """Validate training data."""
```

### EvaluateModelUseCase

Use case for evaluating detector performance.

```python
from pynomaly.core.application.use_cases import EvaluateModelUseCase

use_case = EvaluateModelUseCase()
metrics = use_case.execute(detector, test_dataset, ground_truth)
```

#### Methods

```python
def execute(
    self,
    detector: Detector,
    test_dataset: Dataset,
    ground_truth: np.ndarray,
    evaluation_options: Optional[EvaluationOptions] = None
) -> PerformanceMetrics:
    """Evaluate detector performance."""
    
def cross_validate(
    self,
    detector: Detector,
    dataset: Dataset,
    k_folds: int = 5
) -> CrossValidationResult:
    """Perform cross-validation."""
```

## Utility Functions

### Data Validation

```python
from pynomaly.core.shared.validation import (
    validate_dataset,
    validate_detector,
    validate_anomaly_scores
)

# Validate dataset
validation_result = validate_dataset(dataset)
if not validation_result.is_valid:
    print(f"Validation errors: {validation_result.errors}")

# Validate detector
validation_result = validate_detector(detector)

# Validate anomaly scores
validation_result = validate_anomaly_scores(scores)
```

### Data Preprocessing

```python
from pynomaly.core.shared.preprocessing import (
    normalize_data,
    handle_missing_values,
    remove_duplicates
)

# Normalize data
normalized_data = normalize_data(data, method="standard")

# Handle missing values
cleaned_data = handle_missing_values(data, strategy="mean")

# Remove duplicates
unique_data = remove_duplicates(data)
```

### Feature Engineering

```python
from pynomaly.core.shared.features import (
    extract_features,
    select_features,
    transform_features
)

# Extract features
features = extract_features(raw_data, feature_extractors)

# Select features
selected_features = select_features(features, selector="variance")

# Transform features
transformed_features = transform_features(features, transformers)
```

## Exception Handling

### Core Exceptions

```python
from pynomaly.core.shared.exceptions import (
    DomainError,
    InvalidDatasetError,
    DetectionError,
    ConfigurationError
)

try:
    result = detector.predict(dataset)
except InvalidDatasetError as e:
    print(f"Dataset validation failed: {e}")
except DetectionError as e:
    print(f"Detection failed: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Exception Hierarchy

```
DomainError
├── InvalidDatasetError
│   ├── EmptyDatasetError
│   ├── InvalidDataTypeError
│   └── MissingFeaturesError
├── DetectionError
│   ├── ModelNotFittedError
│   ├── IncompatibleDataError
│   └── DetectionTimeoutError
└── ConfigurationError
    ├── InvalidParameterError
    ├── MissingParameterError
    └── IncompatibleParameterError
```

## Type Hints

The package provides comprehensive type hints. Import common types:

```python
from pynomaly.core.shared.types import (
    DataArray,
    ScoreArray,
    PredictionArray,
    FeatureArray,
    MetadataDict,
    ParameterDict
)
```

## Configuration

### Dependency Injection

```python
from pynomaly.core.shared.container import CoreContainer

# Configure container
container = CoreContainer()
container.config.detection.threshold.from_value(0.8)
container.config.preprocessing.normalize.from_value(True)

# Get configured services
detection_service = container.detection_service()
preprocessing_service = container.preprocessing_service()
```

### Configuration Classes

```python
from pynomaly.core.shared.config import (
    DetectionConfig,
    PreprocessingConfig,
    ValidationConfig
)

# Detection configuration
detection_config = DetectionConfig(
    threshold=0.8,
    normalize_scores=True,
    batch_size=1000
)

# Preprocessing configuration
preprocessing_config = PreprocessingConfig(
    normalize=True,
    handle_missing=True,
    remove_duplicates=True
)
```

## Logging

### Structured Logging

```python
import structlog
from pynomaly.core.shared.logging import get_logger

logger = get_logger(__name__)

# Log with context
logger.info(
    "Detection completed",
    dataset_size=len(dataset),
    anomaly_count=len(result.anomalies),
    processing_time=elapsed_time
)
```

## Testing Utilities

### Test Fixtures

```python
from pynomaly.core.testing.fixtures import (
    create_test_dataset,
    create_test_detector,
    create_test_anomalies
)

# Create test data
dataset = create_test_dataset(samples=1000, features=10, contamination=0.1)

# Create test detector
detector = create_test_detector("isolation_forest")

# Create test anomalies
anomalies = create_test_anomalies(count=10, score_range=(0.8, 1.0))
```

### Test Utilities

```python
from pynomaly.core.testing.utils import (
    assert_valid_dataset,
    assert_valid_detector,
    assert_valid_result
)

# Assert valid dataset
assert_valid_dataset(dataset)

# Assert valid detector
assert_valid_detector(detector)

# Assert valid result
assert_valid_result(result)
```

## Performance Considerations

### Memory Management

```python
from pynomaly.core.shared.memory import MemoryManager

memory_manager = MemoryManager()

# Monitor memory usage
with memory_manager.monitor():
    result = detector.predict(large_dataset)

# Optimize memory usage
optimized_dataset = memory_manager.optimize_dataset(dataset)
```

### Batch Processing

```python
from pynomaly.core.shared.batch import BatchProcessor

processor = BatchProcessor(batch_size=1000)

# Process in batches
for batch_result in processor.process(large_dataset, detector):
    # Handle batch result
    handle_batch_result(batch_result)
```

## Integration Examples

### Complete Workflow

```python
from pynomaly.core.domain.entities import Dataset, Detector
from pynomaly.core.application.use_cases import DetectAnomaliesUseCase
from pynomaly.core.domain.value_objects import ContaminationRate
import numpy as np

# Create dataset
data = np.random.randn(1000, 10)
dataset = Dataset.from_array("test_data", data)

# Create detector
detector = Detector(
    name="isolation_forest",
    algorithm="IsolationForest",
    contamination_rate=ContaminationRate(0.1),
    hyperparameters={"n_estimators": 100, "random_state": 42}
)

# Train detector
detector.fit(dataset)

# Detect anomalies
use_case = DetectAnomaliesUseCase()
result = use_case.execute(dataset, detector)

# Analyze results
print(f"Detected {result.anomaly_count} anomalies")
print(f"Anomaly rate: {result.anomaly_rate:.2%}")

# Get top anomalies
top_anomalies = result.get_top_anomalies(n=5)
for anomaly in top_anomalies:
    print(f"Anomaly {anomaly.id}: score={anomaly.score.value:.3f}")
```

This API reference provides comprehensive coverage of the Core package's public interface. For additional examples and advanced usage patterns, refer to the [examples](../examples/) directory.
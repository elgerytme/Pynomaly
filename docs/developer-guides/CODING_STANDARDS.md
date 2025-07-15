# Pynomaly Coding Standards

This document defines the coding standards and best practices for the Pynomaly project. Following these standards ensures code consistency, maintainability, and quality across the entire codebase.

## 📋 Table of Contents

- [General Principles](#general-principles)
- [Python Code Style](#python-code-style)
- [Architecture Standards](#architecture-standards)
- [Documentation Standards](#documentation-standards)
- [Testing Standards](#testing-standards)
- [Security Standards](#security-standards)
- [Performance Standards](#performance-standards)
- [Error Handling](#error-handling)
- [Tool Configuration](#tool-configuration)

## 🎯 General Principles

### Clean Code Principles
- **Clarity over cleverness**: Write code that's easy to understand
- **Single Responsibility**: Each function/class has one reason to change
- **DRY (Don't Repeat Yourself)**: Extract common functionality
- **YAGNI (You Aren't Gonna Need It)**: Don't add unnecessary complexity
- **SOLID Principles**: Follow object-oriented design principles

### Code Organization
- **Domain-Driven Design**: Organize code around business concepts
- **Clean Architecture**: Dependencies point inward toward the domain
- **Separation of Concerns**: Each layer has distinct responsibilities
- **Explicit Dependencies**: Use dependency injection over hidden dependencies

## 🐍 Python Code Style

### Formatting (Automated with Ruff)

#### Import Organization
```python
# Standard library imports
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

# Third-party imports
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Local imports
from pynomaly.domain.entities import Anomaly, Detector
from pynomaly.domain.value_objects import AnomalyScore
from pynomaly.domain.protocols import DetectorProtocol
```

#### Line Length and Formatting
- **Maximum line length**: 88 characters (Black standard)
- **Indentation**: 4 spaces (no tabs)
- **String quotes**: Double quotes preferred
- **Trailing commas**: Always use in multi-line structures

```python
# Good
detector_config = DetectorConfig(
    name="isolation_forest",
    contamination_rate=0.1,
    random_state=42,
    n_estimators=100,
)

# Bad
detector_config = DetectorConfig(name="isolation_forest", contamination_rate=0.1, random_state=42, n_estimators=100)
```

### Type Annotations (Required)

#### Function Signatures
```python
# Good - Complete type annotations
def detect_anomalies(
    detector: DetectorProtocol,
    dataset: Dataset,
    threshold: Optional[float] = None
) -> DetectionResult:
    """Detect anomalies in the given dataset."""
    pass

# Bad - Missing type annotations
def detect_anomalies(detector, dataset, threshold=None):
    pass
```

#### Generic Types
```python
from typing import Dict, List, Optional, Protocol, TypeVar, Generic

T = TypeVar('T')

class Repository(Protocol, Generic[T]):
    """Generic repository protocol."""
    
    def save(self, entity: T) -> None: ...
    def find_by_id(self, entity_id: str) -> Optional[T]: ...
    def find_all(self) -> List[T]: ...
```

#### Union Types (Python 3.11+ Style)
```python
# Good - Modern union syntax
def process_data(data: pd.DataFrame | np.ndarray) -> List[float]:
    pass

# Acceptable - Traditional typing
from typing import Union
def process_data(data: Union[pd.DataFrame, np.ndarray]) -> List[float]:
    pass
```

### Naming Conventions

#### Variables and Functions
```python
# Variables: snake_case
anomaly_score = 0.85
detection_results = []
contamination_rate = 0.1

# Functions: snake_case with verbs
def calculate_anomaly_score(data: np.ndarray) -> float:
    pass

def is_anomaly(score: float, threshold: float) -> bool:
    pass

def get_detector_by_name(name: str) -> Optional[Detector]:
    pass
```

#### Classes and Protocols
```python
# Classes: PascalCase
class AnomalyDetector:
    pass

class IsolationForestAdapter:
    pass

# Protocols: PascalCase with 'Protocol' suffix
class DetectorProtocol(Protocol):
    pass

class RepositoryProtocol(Protocol):
    pass
```

#### Constants and Enums
```python
# Constants: SCREAMING_SNAKE_CASE
DEFAULT_CONTAMINATION_RATE = 0.1
MAX_RETRY_ATTEMPTS = 3
API_VERSION = "v1"

# Enums: PascalCase
class AlgorithmType(Enum):
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
```

#### Files and Directories
```python
# Files: snake_case
anomaly_detector.py
detection_service.py
sklearn_adapter.py

# Directories: snake_case
domain/entities/
application/use_cases/
infrastructure/adapters/
```

### Class Design

#### Domain Entities
```python
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

@dataclass(frozen=True)
class Anomaly:
    """Domain entity representing an anomaly."""
    
    id: UUID = Field(default_factory=uuid4)
    score: AnomalyScore
    detected_at: datetime
    data_point: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Validate entity invariants."""
        if self.score.value < 0 or self.score.value > 1:
            raise ValueError("Anomaly score must be between 0 and 1")
```

#### Value Objects
```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ContaminationRate:
    """Value object for contamination rate."""
    
    value: float
    
    def __post_init__(self) -> None:
        if not 0 < self.value < 1:
            raise ValueError("Contamination rate must be between 0 and 1")
    
    def __str__(self) -> str:
        return f"{self.value:.1%}"
```

#### Services
```python
class DetectionService:
    """Application service for anomaly detection."""
    
    def __init__(
        self,
        detector_repository: DetectorRepositoryProtocol,
        dataset_repository: DatasetRepositoryProtocol,
        logger: LoggerProtocol,
    ) -> None:
        self._detector_repository = detector_repository
        self._dataset_repository = dataset_repository
        self._logger = logger
    
    async def detect_anomalies(
        self,
        request: DetectionRequest
    ) -> DetectionResponse:
        """Detect anomalies using the specified detector."""
        # Implementation here
        pass
```

## 🏗️ Architecture Standards

### Clean Architecture Layers

#### Domain Layer (Core Business Logic)
```python
# domain/entities/detector.py
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

@dataclass
class Detector:
    """Core detector domain entity."""
    
    id: UUID
    name: str
    algorithm_name: str
    hyperparameters: Dict[str, Any]
    created_at: datetime
    is_fitted: bool = False
    
    def mark_as_fitted(self) -> None:
        """Mark detector as trained."""
        self.is_fitted = True
```

#### Application Layer (Use Cases)
```python
# application/use_cases/detect_anomalies.py
from pynomaly.domain.protocols import DetectorProtocol
from pynomaly.application.dto import DetectionRequest, DetectionResponse

class DetectAnomaliesUseCase:
    """Use case for detecting anomalies."""
    
    def __init__(self, detector_service: DetectorProtocol) -> None:
        self._detector_service = detector_service
    
    async def execute(self, request: DetectionRequest) -> DetectionResponse:
        """Execute anomaly detection."""
        # Business logic here
        pass
```

#### Infrastructure Layer (External Dependencies)
```python
# infrastructure/adapters/sklearn_adapter.py
from sklearn.ensemble import IsolationForest
from pynomaly.domain.protocols import DetectorProtocol

class SklearnIsolationForestAdapter(DetectorProtocol):
    """Scikit-learn Isolation Forest adapter."""
    
    def __init__(self, **kwargs) -> None:
        self._model = IsolationForest(**kwargs)
        self._is_fitted = False
    
    def fit(self, data: np.ndarray) -> None:
        """Train the model."""
        self._model.fit(data)
        self._is_fitted = True
```

### Dependency Injection

#### Protocol Definition
```python
# domain/protocols/detector_protocols.py
from typing import Protocol
import numpy as np

class DetectorProtocol(Protocol):
    """Protocol for anomaly detectors."""
    
    def fit(self, data: np.ndarray) -> None:
        """Train the detector."""
        ...
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        ...
    
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores."""
        ...
```

#### Container Configuration
```python
# infrastructure/config/container.py
from dependency_injector import containers, providers
from pynomaly.infrastructure.adapters import SklearnAdapter

class Container(containers.DeclarativeContainer):
    """Dependency injection container."""
    
    # Configuration
    config = providers.Configuration()
    
    # Detectors
    isolation_forest = providers.Factory(
        SklearnAdapter,
        algorithm_name="IsolationForest",
        contamination=config.contamination_rate,
    )
```

## 📚 Documentation Standards

### Docstring Format (Google Style)

#### Module Docstrings
```python
"""Anomaly detection domain entities.

This module contains the core domain entities for the anomaly detection system,
including Anomaly, Detector, and Dataset entities. These entities encapsulate
the business logic and maintain data integrity through validation.

Example:
    Basic usage of domain entities:
    
    >>> detector = Detector(
    ...     name="my_detector",
    ...     algorithm_name="IsolationForest"
    ... )
    >>> dataset = Dataset.from_dataframe(df)
    >>> anomalies = detector.detect(dataset)
"""
```

#### Class Docstrings
```python
class AnomalyDetector:
    """High-level interface for anomaly detection.
    
    This class provides a simplified interface for anomaly detection tasks,
    encapsulating the complexity of different algorithms and providing
    consistent behavior across different detection methods.
    
    Attributes:
        name: Human-readable name for the detector.
        algorithm: The underlying detection algorithm.
        is_fitted: Whether the detector has been trained.
    
    Example:
        Basic usage:
        
        >>> detector = AnomalyDetector("IsolationForest")
        >>> detector.fit(training_data)
        >>> anomalies = detector.detect(test_data)
    """
```

#### Function Docstrings
```python
def calculate_anomaly_score(
    data_point: np.ndarray,
    detector: DetectorProtocol,
    normalization_method: str = "min_max"
) -> AnomalyScore:
    """Calculate anomaly score for a single data point.
    
    Args:
        data_point: The data point to score as a numpy array.
        detector: The trained detector to use for scoring.
        normalization_method: Method to normalize the score ("min_max" or "z_score").
    
    Returns:
        AnomalyScore object containing the calculated score and metadata.
    
    Raises:
        ValueError: If the detector is not fitted or data_point is invalid.
        NotImplementedError: If normalization_method is not supported.
    
    Example:
        Calculate score for a data point:
        
        >>> score = calculate_anomaly_score(
        ...     data_point=np.array([1.0, 2.0, 3.0]),
        ...     detector=fitted_detector
        ... )
        >>> print(score.value)
        0.85
    """
```

### Code Comments

#### Inline Comments
```python
# Calculate the contamination threshold based on the specified rate
threshold = np.percentile(scores, (1 - contamination_rate) * 100)

# Apply temporal smoothing to reduce false positives
smoothed_scores = apply_temporal_smoothing(scores, window_size=5)

# TODO: Implement adaptive threshold based on data distribution
# See issue #123 for requirements
```

#### Complex Logic Comments
```python
def _calculate_ensemble_score(self, individual_scores: List[np.ndarray]) -> np.ndarray:
    """Calculate ensemble score using weighted voting."""
    
    # Normalize individual scores to [0, 1] range to ensure fair weighting
    normalized_scores = []
    for scores in individual_scores:
        min_score, max_score = scores.min(), scores.max()
        normalized = (scores - min_score) / (max_score - min_score)
        normalized_scores.append(normalized)
    
    # Apply detector-specific weights based on historical performance
    # Weights are calculated using cross-validation AUC scores
    weights = self._get_detector_weights()
    
    # Calculate weighted average, handling edge case of single detector
    if len(normalized_scores) == 1:
        return normalized_scores[0]
    
    weighted_sum = np.zeros_like(normalized_scores[0])
    for score, weight in zip(normalized_scores, weights):
        weighted_sum += score * weight
    
    return weighted_sum / sum(weights)
```

## 🧪 Testing Standards

### Test Organization

#### Test Structure
```python
# tests/unit/domain/entities/test_detector.py
import pytest
from datetime import datetime
from uuid import uuid4

from pynomaly.domain.entities import Detector
from pynomaly.domain.exceptions import ValidationError


class TestDetector:
    """Test suite for Detector entity."""
    
    def test_detector_creation_with_valid_data(self):
        """Test that detector can be created with valid data."""
        # Arrange
        detector_id = uuid4()
        name = "test_detector"
        algorithm_name = "IsolationForest"
        
        # Act
        detector = Detector(
            id=detector_id,
            name=name,
            algorithm_name=algorithm_name,
            hyperparameters={},
            created_at=datetime.utcnow()
        )
        
        # Assert
        assert detector.id == detector_id
        assert detector.name == name
        assert detector.algorithm_name == algorithm_name
        assert not detector.is_fitted
    
    def test_detector_mark_as_fitted(self):
        """Test that detector can be marked as fitted."""
        # Arrange
        detector = self._create_valid_detector()
        
        # Act
        detector.mark_as_fitted()
        
        # Assert
        assert detector.is_fitted
    
    def test_detector_creation_with_empty_name_raises_error(self):
        """Test that creating detector with empty name raises ValidationError."""
        # Arrange & Act & Assert
        with pytest.raises(ValidationError, match="Name cannot be empty"):
            Detector(
                id=uuid4(),
                name="",
                algorithm_name="IsolationForest",
                hyperparameters={},
                created_at=datetime.utcnow()
            )
    
    def _create_valid_detector(self) -> Detector:
        """Helper method to create a valid detector."""
        return Detector(
            id=uuid4(),
            name="test_detector",
            algorithm_name="IsolationForest",
            hyperparameters={"contamination": 0.1},
            created_at=datetime.utcnow()
        )
```

### Test Naming Conventions
- **Test classes**: `TestClassName`
- **Test methods**: `test_method_name_with_scenario`
- **Test fixtures**: `fixture_name` (descriptive, no `test_` prefix)

### Assertions and Test Data
```python
# Use descriptive assertions
assert len(anomalies) == 5, "Should detect exactly 5 anomalies"
assert detector.is_fitted, "Detector should be fitted after training"

# Use factories for test data
@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return Dataset.from_array(
        data=np.random.randn(100, 3),
        name="test_dataset"
    )

# Parametrize tests for multiple scenarios
@pytest.mark.parametrize("contamination_rate,expected_anomalies", [
    (0.1, 10),
    (0.05, 5),
    (0.2, 20),
])
def test_contamination_rate_affects_anomaly_count(
    contamination_rate, expected_anomalies, sample_dataset
):
    # Test implementation
    pass
```

## 🔒 Security Standards

### Input Validation
```python
def validate_contamination_rate(rate: float) -> None:
    """Validate contamination rate parameter."""
    if not isinstance(rate, (int, float)):
        raise TypeError("Contamination rate must be numeric")
    
    if not 0 < rate < 1:
        raise ValueError("Contamination rate must be between 0 and 1")
    
    if rate > 0.5:
        warnings.warn("High contamination rate (>50%) may indicate data quality issues")

def sanitize_detector_name(name: str) -> str:
    """Sanitize detector name to prevent injection attacks."""
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[^\w\-_.]', '', name.strip())
    
    if not sanitized:
        raise ValueError("Detector name cannot be empty after sanitization")
    
    return sanitized[:50]  # Limit length
```

### Secrets Management
```python
# Good - Use environment variables
import os
from pathlib import Path

def get_api_key() -> str:
    """Get API key from environment or file."""
    api_key = os.getenv("PYNOMALY_API_KEY")
    if not api_key:
        key_file = Path.home() / ".pynomaly" / "api_key"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    
    if not api_key:
        raise ValueError("API key not found in environment or config file")
    
    return api_key

# Bad - Hardcoded secrets
API_KEY = "sk-1234567890abcdef"  # Never do this!
```

## ⚡ Performance Standards

### Efficient Data Processing
```python
# Good - Use vectorized operations
def calculate_scores_vectorized(data: np.ndarray) -> np.ndarray:
    """Calculate anomaly scores using vectorized operations."""
    # Vectorized computation is much faster than loops
    distances = np.linalg.norm(data - data.mean(axis=0), axis=1)
    scores = (distances - distances.min()) / (distances.max() - distances.min())
    return scores

# Bad - Use loops for large data
def calculate_scores_loop(data: np.ndarray) -> np.ndarray:
    """Don't do this for large datasets."""
    scores = []
    mean = data.mean(axis=0)
    for row in data:
        distance = np.linalg.norm(row - mean)
        scores.append(distance)
    return np.array(scores)
```

### Memory Management
```python
def process_large_dataset(file_path: Path, chunk_size: int = 10000) -> Iterator[np.ndarray]:
    """Process large dataset in chunks to manage memory."""
    
    # Use generators for large data processing
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        processed_chunk = preprocess_data(chunk.values)
        yield processed_chunk
        
        # Explicitly clean up if needed
        del chunk

# Use context managers for resource cleanup
def detect_anomalies_from_file(file_path: Path) -> List[Anomaly]:
    """Detect anomalies from large file."""
    anomalies = []
    
    with open(file_path, 'r') as file:
        # Process file content
        for line in file:
            # Process line
            pass
    
    # File automatically closed
    return anomalies
```

### Caching and Memoization
```python
from functools import lru_cache
from typing import Tuple

@lru_cache(maxsize=128)
def get_algorithm_hyperparameters(algorithm_name: str) -> Dict[str, Any]:
    """Get default hyperparameters for algorithm (cached)."""
    # Expensive computation cached automatically
    return load_hyperparameters_from_config(algorithm_name)

class DetectorCache:
    """Simple detector result caching."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self._max_size = max_size
    
    def get(self, key: str, max_age: timedelta) -> Optional[np.ndarray]:
        """Get cached result if still valid."""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.utcnow() - timestamp < max_age:
                return result
        return None
    
    def set(self, key: str, value: np.ndarray) -> None:
        """Cache result with timestamp."""
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        
        self._cache[key] = (value, datetime.utcnow())
```

## 🚨 Error Handling

### Exception Hierarchy
```python
# domain/exceptions.py
class PynormalyError(Exception):
    """Base exception for all Pynomaly errors."""
    pass

class DomainError(PynormalyError):
    """Base class for domain-related errors."""
    pass

class ValidationError(DomainError):
    """Raised when validation fails."""
    pass

class DetectorNotFittedError(DomainError):
    """Raised when trying to use an unfitted detector."""
    pass

class InfrastructureError(PynormalyError):
    """Base class for infrastructure-related errors."""
    pass

class AdapterError(InfrastructureError):
    """Raised when adapter operations fail."""
    pass
```

### Error Handling Patterns
```python
def detect_anomalies(detector: DetectorProtocol, data: np.ndarray) -> List[Anomaly]:
    """Detect anomalies with proper error handling."""
    
    # Validate inputs
    if data is None or data.size == 0:
        raise ValidationError("Data cannot be None or empty")
    
    if not detector.is_fitted:
        raise DetectorNotFittedError("Detector must be fitted before detection")
    
    try:
        # Perform detection
        scores = detector.score_samples(data)
        anomalies = _create_anomalies_from_scores(scores, data)
        return anomalies
        
    except Exception as e:
        # Log error with context
        logger.error(
            "Anomaly detection failed",
            extra={
                "detector_type": type(detector).__name__,
                "data_shape": data.shape,
                "error": str(e)
            }
        )
        
        # Re-raise with more context
        raise AdapterError(f"Detection failed: {e}") from e

def _create_anomalies_from_scores(scores: np.ndarray, data: np.ndarray) -> List[Anomaly]:
    """Helper function with specific error handling."""
    if len(scores) != len(data):
        raise ValidationError(
            f"Score count ({len(scores)}) doesn't match data count ({len(data)})"
        )
    
    anomalies = []
    for i, (score, point) in enumerate(zip(scores, data)):
        try:
            anomaly = Anomaly(
                score=AnomalyScore(score),
                data_point=point.tolist(),
                detected_at=datetime.utcnow()
            )
            anomalies.append(anomaly)
        except Exception as e:
            logger.warning(f"Failed to create anomaly for point {i}: {e}")
            continue  # Skip invalid points
    
    return anomalies
```

## 🔧 Tool Configuration

### Ruff Configuration (.ruff.toml)
```toml
target-version = "py311"
line-length = 88
exclude = [
    ".git",
    "__pycache__",
    ".venv",
    "build",
    "dist",
]

[lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[lint.per-file-ignores]
"tests/**/*.py" = ["S101"]  # Allow assert in tests

[format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
```

### MyPy Configuration (pyproject.toml)
```toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

### Pytest Configuration (pyproject.toml)
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--tb=short",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

---

## 📝 Summary

These coding standards ensure:

1. **Consistency**: Uniform code style across the project
2. **Quality**: High-quality, maintainable code
3. **Security**: Secure coding practices
4. **Performance**: Efficient and scalable solutions
5. **Testability**: Comprehensive test coverage
6. **Documentation**: Clear and helpful documentation

Remember: These standards are enforced by automated tools (ruff, mypy, pytest) and should be followed consistently. When in doubt, refer to this document or ask in code reviews.

---

*Last updated: 2025-01-14*
# Contributing to Core Package

Thank you for your interest in contributing to the Core package! This package contains the pure business logic and domain models that drive the Pynomaly platform, following Clean Architecture principles with strict separation of concerns.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Architecture Guidelines](#architecture-guidelines)
- [Domain-Driven Design](#domain-driven-design)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Performance Considerations](#performance-considerations)
- [Community](#community)

## Code of Conduct

This project adheres to our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.11+
- Understanding of Clean Architecture principles
- Knowledge of Domain-Driven Design concepts
- Familiarity with type hints and mypy

### Repository Setup

```bash
# Clone the repository
git clone https://github.com/your-org/monorepo.git
cd monorepo/src/packages/software/core

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test,benchmark]"

# Install pre-commit hooks
pre-commit install
```

### Core Principles Validation

```bash
# Validate clean architecture dependencies
python scripts/validate_dependencies.py

# Check domain purity (no external dependencies)
python scripts/check_domain_purity.py

# Verify type coverage
mypy core/ --strict --show-coverage
```

## Development Environment

### IDE Configuration

Recommended VS Code settings:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.mypyEnabled": true,
    "python.linting.enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.analysis.typeCheckingMode": "strict"
}
```

Recommended extensions:
- Python
- Pylance
- Python Type Hint
- mypy

### Environment Variables

Create a `.env` file for local development:

```bash
# Core Configuration
CORE_LOG_LEVEL=DEBUG
CORE_STRICT_MODE=true
CORE_VALIDATE_INVARIANTS=true

# Testing Configuration
PYTEST_ADDOPTS="--strict-markers --strict-config"
HYPOTHESIS_MAX_EXAMPLES=1000
HYPOTHESIS_DEADLINE=None

# Type Checking
MYPY_STRICT=true
MYPY_WARN_UNUSED_IGNORES=true
MYPY_WARN_REDUNDANT_CASTS=true
```

## Architecture Guidelines

### Clean Architecture Principles

This package follows strict Clean Architecture principles:

1. **Dependency Inversion**: Core depends only on abstractions, never on concrete implementations
2. **Independence**: Domain logic is independent of external frameworks, databases, or UI
3. **Testability**: All business logic is easily unit testable
4. **Separation of Concerns**: Clear boundaries between domain, application, and infrastructure

### Package Structure

```
core/
├── domain/                 # Pure business logic (no external dependencies)
│   ├── entities/          # Business objects with identity
│   ├── value_objects/     # Immutable values with equality semantics
│   ├── services/          # Domain services for business logic
│   ├── events/           # Domain events for loose coupling
│   ├── exceptions/       # Domain-specific errors
│   └── repositories/     # Repository interfaces (abstractions)
├── application/          # Use cases and application services
│   ├── use_cases/       # Business operations and workflows
│   ├── services/        # Application services
│   ├── dto/            # Data Transfer Objects
│   └── interfaces/     # Application layer interfaces
└── shared/              # Shared utilities and types
    ├── types/          # Common type definitions
    ├── utils/          # Pure utility functions
    └── exceptions/     # Base exceptions
```

### Dependency Rules

**Allowed Dependencies by Layer:**

```python
# Domain Layer (core/domain/)
# ✅ Allowed: Only Python standard library + typing
# ❌ Forbidden: External frameworks, infrastructure, persistence

# Application Layer (core/application/)
# ✅ Allowed: Domain layer, Python standard library
# ❌ Forbidden: Infrastructure concerns, frameworks

# Shared (core/shared/)
# ✅ Allowed: Python standard library, typing extensions
# ❌ Forbidden: Business logic, external frameworks
```

### Layer Implementation Patterns

**Domain Entities:**
```python
from typing import Optional, List
from datetime import datetime
from uuid import UUID, uuid4
from dataclasses import dataclass, field

from core.domain.value_objects import AnomalyScore, ContaminationRate
from core.domain.events import DomainEvent
from core.shared.types import EntityId

@dataclass
class Detector:
    """Domain entity representing an anomaly detector.
    
    Encapsulates business rules and invariants for anomaly detection
    configuration. Maintains consistency through domain validation.
    """
    
    id: EntityId = field(default_factory=lambda: EntityId(uuid4()))
    name: str = field()
    algorithm: str = field()
    contamination_rate: ContaminationRate = field()
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_trained: bool = field(default=False)
    
    # Domain events for loose coupling
    _events: List[DomainEvent] = field(default_factory=list, init=False)
    
    def __post_init__(self) -> None:
        """Validate entity invariants after creation."""
        self._validate_invariants()
    
    def train(self, dataset: 'Dataset') -> None:
        """Train the detector with business rules validation."""
        if self.is_trained:
            raise DetectorAlreadyTrainedError(self.id)
        
        if not dataset.is_valid_for_training():
            raise InvalidTrainingDatasetError(dataset.id)
        
        # Domain business logic
        self.is_trained = True
        self._add_event(DetectorTrainedEvent(self.id, dataset.id))
    
    def _validate_invariants(self) -> None:
        """Validate domain invariants."""
        if not self.name.strip():
            raise ValueError("Detector name cannot be empty")
        
        if not self.algorithm.strip():
            raise ValueError("Algorithm name cannot be empty")
    
    def _add_event(self, event: DomainEvent) -> None:
        """Add domain event for publishing."""
        self._events.append(event)
    
    def clear_events(self) -> List[DomainEvent]:
        """Clear and return domain events."""
        events = self._events.copy()
        self._events.clear()
        return events
```

**Value Objects:**
```python
from typing import Union
from dataclasses import dataclass

@dataclass(frozen=True)
class AnomalyScore:
    """Value object representing an anomaly score.
    
    Immutable value with business validation and equality semantics.
    Scores are normalized between 0.0 (normal) and 1.0 (anomalous).
    """
    
    value: float
    
    def __post_init__(self) -> None:
        """Validate value object invariants."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Anomaly score must be between 0.0 and 1.0, got {self.value}")
    
    def is_anomalous(self, threshold: float = 0.5) -> bool:
        """Business logic: determine if score indicates anomaly."""
        return self.value >= threshold
    
    def confidence_level(self) -> str:
        """Business logic: categorize confidence level."""
        if self.value >= 0.9:
            return "very_high"
        elif self.value >= 0.7:
            return "high"
        elif self.value >= 0.5:
            return "medium"
        elif self.value >= 0.3:
            return "low"
        else:
            return "very_low"
    
    def __str__(self) -> str:
        return f"{self.value:.3f}"
```

**Use Cases:**
```python
from typing import Protocol
from abc import abstractmethod

from core.domain.entities import Dataset, Detector, DetectionResult
from core.domain.repositories import DatasetRepository, DetectorRepository
from core.domain.services import AnomalyDetectionService

class DetectAnomaliesUseCase:
    """Use case for detecting anomalies in a dataset.
    
    Orchestrates domain services and repositories to implement
    the business workflow for anomaly detection.
    """
    
    def __init__(
        self,
        dataset_repository: DatasetRepository,
        detector_repository: DetectorRepository,
        detection_service: AnomalyDetectionService
    ) -> None:
        self._dataset_repository = dataset_repository
        self._detector_repository = detector_repository
        self._detection_service = detection_service
    
    async def execute(
        self,
        dataset_id: EntityId,
        detector_id: EntityId
    ) -> DetectionResult:
        """Execute anomaly detection use case."""
        # Load domain objects
        dataset = await self._dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise DatasetNotFoundError(dataset_id)
        
        detector = await self._detector_repository.get_by_id(detector_id)
        if not detector:
            raise DetectorNotFoundError(detector_id)
        
        # Validate business rules
        if not detector.is_trained:
            raise DetectorNotTrainedError(detector_id)
        
        if not dataset.is_valid_for_detection():
            raise InvalidDetectionDatasetError(dataset_id)
        
        # Execute domain service
        result = await self._detection_service.detect_anomalies(
            dataset=dataset,
            detector=detector
        )
        
        # Publish domain events
        events = detector.clear_events()
        for event in events:
            await self._event_publisher.publish(event)
        
        return result
```

## Domain-Driven Design

### Ubiquitous Language

Use domain terms consistently throughout the codebase:

**Core Domain Terms:**
- **Anomaly**: Unusual data point that deviates from normal patterns
- **Detector**: Configuration for anomaly detection algorithms
- **Dataset**: Collection of data points for analysis
- **Score**: Numerical measure of how anomalous a data point is
- **Contamination Rate**: Expected proportion of anomalies in data
- **Model**: Trained representation of normal behavior patterns

### Bounded Contexts

The core package defines these bounded contexts:

1. **Detection Context**: Anomaly detection algorithms and scoring
2. **Data Context**: Dataset management and validation
3. **Model Context**: Model training and lifecycle management
4. **Analysis Context**: Statistical analysis and evaluation

### Aggregates and Aggregate Roots

```python
from typing import List, Optional
from core.domain.value_objects import EntityId

class DetectionSession:
    """Aggregate root for anomaly detection session.
    
    Maintains consistency boundaries and business invariants
    across related entities within the detection context.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        detector: Detector,
        configuration: DetectionConfiguration
    ) -> None:
        self._dataset = dataset
        self._detector = detector
        self._configuration = configuration
        self._results: List[DetectionResult] = []
        self._status = SessionStatus.INITIALIZED
    
    def execute_detection(self) -> DetectionResult:
        """Execute detection maintaining aggregate consistency."""
        self._validate_session_state()
        
        # Business rule: detector must be trained
        if not self._detector.is_trained:
            raise BusinessRuleViolationError("Detector must be trained")
        
        # Execute detection
        self._status = SessionStatus.RUNNING
        result = self._perform_detection()
        self._results.append(result)
        self._status = SessionStatus.COMPLETED
        
        return result
    
    def _validate_session_state(self) -> None:
        """Validate aggregate invariants."""
        if self._status != SessionStatus.INITIALIZED:
            raise InvalidSessionStateError(self._status)
```

## Testing Requirements

### Test Categories

1. **Unit Tests**: Test individual domain components in isolation
2. **Integration Tests**: Test interaction between domain components
3. **Property-Based Tests**: Test domain invariants with generated inputs
4. **Performance Tests**: Ensure acceptable performance characteristics

### Test Structure

```bash
tests/
├── unit/                    # Unit tests for each layer
│   ├── domain/             # Domain entity and service tests
│   │   ├── entities/       # Entity behavior tests
│   │   ├── value_objects/  # Value object validation tests
│   │   └── services/       # Domain service tests
│   ├── application/        # Use case and application service tests
│   └── shared/            # Shared utility tests
├── integration/           # Integration tests
│   ├── use_cases/        # End-to-end use case tests
│   └── workflows/        # Complex workflow tests
├── property/             # Property-based tests
└── performance/          # Performance and benchmark tests
```

### Test Requirements

- **Coverage**: Minimum 95% code coverage for domain layer
- **Isolation**: All tests must be completely isolated
- **Deterministic**: No random or time-dependent behavior
- **Fast**: Unit tests under 10ms each
- **Comprehensive**: Test all business rules and edge cases

### Property-Based Testing

```python
from hypothesis import given, strategies as st
from hypothesis.strategies import composite

from core.domain.value_objects import AnomalyScore, ContaminationRate

@composite
def anomaly_scores(draw):
    """Generate valid anomaly scores."""
    return AnomalyScore(draw(st.floats(min_value=0.0, max_value=1.0)))

@composite
def contamination_rates(draw):
    """Generate valid contamination rates."""
    return ContaminationRate(draw(st.floats(min_value=0.0, max_value=0.5)))

@given(score=anomaly_scores())
def test_anomaly_score_invariants(score: AnomalyScore):
    """Test anomaly score maintains invariants."""
    assert 0.0 <= score.value <= 1.0
    assert isinstance(score.is_anomalous(), bool)
    assert score.confidence_level() in {
        "very_low", "low", "medium", "high", "very_high"
    }

@given(
    score1=anomaly_scores(),
    score2=anomaly_scores()
)
def test_anomaly_score_equality(score1: AnomalyScore, score2: AnomalyScore):
    """Test value object equality semantics."""
    if score1.value == score2.value:
        assert score1 == score2
        assert hash(score1) == hash(score2)
    else:
        assert score1 != score2
```

### Domain Test Patterns

```python
import pytest
from unittest.mock import Mock

from core.domain.entities import Detector, Dataset
from core.domain.value_objects import ContaminationRate
from core.domain.exceptions import DetectorAlreadyTrainedError

class TestDetectorEntity:
    """Test detector entity business rules."""
    
    def test_detector_creation_with_valid_data(self):
        """Test detector creates successfully with valid data."""
        detector = Detector(
            name="test_detector",
            algorithm="IsolationForest",
            contamination_rate=ContaminationRate(0.1)
        )
        
        assert detector.name == "test_detector"
        assert detector.algorithm == "IsolationForest"
        assert not detector.is_trained
        assert len(detector._events) == 0
    
    def test_detector_training_business_rules(self):
        """Test detector training follows business rules."""
        detector = Detector(
            name="test_detector",
            algorithm="IsolationForest",
            contamination_rate=ContaminationRate(0.1)
        )
        
        # Mock dataset that's valid for training
        dataset = Mock(spec=Dataset)
        dataset.is_valid_for_training.return_value = True
        dataset.id = "dataset_001"
        
        # Should train successfully
        detector.train(dataset)
        assert detector.is_trained
        
        # Should not allow training again
        with pytest.raises(DetectorAlreadyTrainedError):
            detector.train(dataset)
    
    @pytest.mark.parametrize("invalid_name", ["", "   ", "\t\n"])
    def test_detector_name_validation(self, invalid_name):
        """Test detector name validation."""
        with pytest.raises(ValueError, match="Detector name cannot be empty"):
            Detector(
                name=invalid_name,
                algorithm="IsolationForest",
                contamination_rate=ContaminationRate(0.1)
            )
```

## Documentation Standards

### Code Documentation

All public APIs must have comprehensive docstrings:

```python
from typing import List, Optional
from core.domain.value_objects import AnomalyScore

class AnomalyDetectionService:
    """Domain service for anomaly detection operations.
    
    Implements core business logic for detecting anomalies in datasets
    using various algorithms. Maintains separation between algorithm
    implementation details and business rules.
    
    This service coordinates between detectors, datasets, and scoring
    mechanisms while enforcing domain invariants and business rules.
    
    Examples:
        Basic anomaly detection:
        
        >>> service = AnomalyDetectionService()
        >>> result = await service.detect_anomalies(dataset, detector)
        >>> print(f"Found {len(result.anomalies)} anomalies")
        
        Batch processing:
        
        >>> results = await service.detect_anomalies_batch(
        ...     datasets=[dataset1, dataset2],
        ...     detector=detector
        ... )
    """
    
    async def detect_anomalies(
        self,
        dataset: Dataset,
        detector: Detector,
        threshold: Optional[float] = None
    ) -> DetectionResult:
        """Detect anomalies in a dataset using the specified detector.
        
        Applies business rules for anomaly detection including validation
        of detector training state, dataset compatibility, and threshold
        constraints.
        
        Args:
            dataset: The dataset to analyze for anomalies. Must be valid
                    for detection and compatible with the detector.
            detector: The trained detector to use for analysis. Must be
                     in trained state and compatible with dataset format.
            threshold: Optional custom threshold for anomaly classification.
                      If not provided, uses detector's default threshold.
                      Must be between 0.0 and 1.0.
        
        Returns:
            DetectionResult containing identified anomalies, scores, and
            metadata about the detection process.
        
        Raises:
            DetectorNotTrainedError: If detector is not in trained state
            IncompatibleDatasetError: If dataset format doesn't match detector
            InvalidThresholdError: If threshold is outside valid range
            DetectionError: If detection process fails for any reason
            
        Business Rules:
            - Detector must be trained before use
            - Dataset must pass validation checks
            - Threshold must be within [0.0, 1.0] range
            - Results must include confidence scores
        """
        # Implementation here
        pass
```

### Domain Documentation

- **Business Rules**: Document all business rules and invariants
- **Domain Model**: Maintain up-to-date domain model diagrams
- **Use Cases**: Document all supported use cases and workflows
- **Examples**: Provide comprehensive usage examples

## Pull Request Process

### Before Submitting

1. **Run Tests**: Ensure all tests pass with high coverage
2. **Type Checking**: Verify strict mypy compliance
3. **Architecture Validation**: Check clean architecture compliance
4. **Performance**: Run benchmarks for performance-sensitive changes
5. **Documentation**: Update relevant documentation

### Pull Request Template

```markdown
## Description
Brief description of changes and business motivation.

## Type of Change
- [ ] New domain entity or value object
- [ ] New use case or application service
- [ ] Domain service enhancement
- [ ] Bug fix in business logic
- [ ] Performance improvement
- [ ] Refactoring (no behavior change)

## Domain Impact
- [ ] New business rules added
- [ ] Existing business rules modified
- [ ] Domain model changes
- [ ] Use case workflow changes
- [ ] Breaking changes to domain APIs

## Architecture Compliance
- [ ] Follows Clean Architecture principles
- [ ] Domain layer remains pure (no external dependencies)
- [ ] Dependency inversion maintained
- [ ] Separation of concerns preserved

## Testing
- [ ] Unit tests added/updated (95%+ coverage)
- [ ] Integration tests for use cases
- [ ] Property-based tests for invariants
- [ ] Performance tests for critical paths
- [ ] All existing tests pass

## Business Rules Validation
- [ ] All domain invariants enforced
- [ ] Business rules documented
- [ ] Edge cases handled
- [ ] Error conditions properly managed

## Type Safety
- [ ] Full type hint coverage
- [ ] mypy strict mode compliance
- [ ] No type: ignore comments added
- [ ] Generic types properly constrained

## Documentation
- [ ] Docstrings for all public APIs
- [ ] Business rules documented
- [ ] Examples provided
- [ ] Domain model updated if needed
```

### Review Process

1. **Automated Checks**: Architecture and type checking validation
2. **Domain Expert Review**: Business logic and domain model review
3. **Technical Review**: Code quality and performance review
4. **Architecture Review**: Clean architecture compliance review

## Performance Considerations

### Memory Efficiency

- Use `__slots__` for frequently created objects
- Implement lazy loading for expensive operations
- Use generators for large dataset processing
- Minimize object copying and mutation

```python
from typing import Iterator
from dataclasses import dataclass

@dataclass
class Dataset:
    """Memory-efficient dataset implementation."""
    __slots__ = ('_data', '_metadata', '_size')
    
    def __init__(self, data: Iterator, metadata: dict):
        self._data = data
        self._metadata = metadata
        self._size = None
    
    def stream_batches(self, batch_size: int = 1000) -> Iterator[List]:
        """Stream data in batches to minimize memory usage."""
        batch = []
        for item in self._data:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
```

### CPU Optimization

- Use appropriate data structures for access patterns
- Implement caching for expensive computations
- Minimize algorithmic complexity
- Use vectorized operations where possible

### Benchmarking

```python
import pytest
from timeit import timeit

@pytest.mark.benchmark
def test_anomaly_score_creation_performance(benchmark):
    """Benchmark anomaly score creation performance."""
    
    def create_scores():
        return [AnomalyScore(i / 1000) for i in range(1000)]
    
    result = benchmark(create_scores)
    assert len(result) == 1000

@pytest.mark.benchmark
def test_detector_training_performance(benchmark, large_dataset):
    """Benchmark detector training performance."""
    detector = Detector(
        name="perf_test",
        algorithm="IsolationForest",
        contamination_rate=ContaminationRate(0.1)
    )
    
    benchmark(detector.train, large_dataset)
    assert detector.is_trained
```

## Community

### Communication Channels

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for architecture and design questions
- **Slack**: #core-development channel for real-time discussion
- **Email**: core-team@yourorg.com for sensitive architectural discussions

### Domain Expertise Areas

- **Domain Modeling**: Entity and value object design
- **Clean Architecture**: Dependency management and layer separation  
- **Business Rules**: Domain logic and invariant enforcement
- **Performance**: Optimization and scalability
- **Type Safety**: Advanced typing and mypy usage

### Getting Help

1. **Architecture Questions**: Post in GitHub Discussions
2. **Bug Reports**: Create detailed GitHub Issues
3. **Performance Issues**: Include benchmarks and profiling data
4. **Design Discussions**: Schedule architecture review sessions

### Core Team

- **Domain Architect**: Overall domain design and business rules
- **Clean Architecture Expert**: Architecture compliance and patterns
- **Performance Engineer**: Optimization and scalability
- **Type Safety Expert**: Advanced typing and static analysis

Thank you for contributing to the Core package! Your contributions help maintain the solid foundation that powers the entire Pynomaly platform.
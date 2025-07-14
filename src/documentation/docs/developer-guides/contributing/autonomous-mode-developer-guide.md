# Autonomous Mode Developer Guide

🍞 **Breadcrumb:** 🏠 [Home](../../index.md) > 👨‍💻 [Developer Guides](../README.md) > 🤝 [Contributing](README.md) > 📄 Autonomous Mode Developer Guide

---


This guide provides comprehensive information for developers working with or extending Pynomaly's Autonomous Mode functionality.

## Architecture Overview

Autonomous Mode follows the Clean Architecture principles with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ CLI Commands│  │ REST API    │  │ Python SDK         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │          AutonomousDetectionService                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │ │
│  │  │Data Profiler│  │ Algorithm   │  │ Result         │ │ │
│  │  │            │  │ Recommender │  │ Aggregator     │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Domain Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ DataProfile │  │ Algorithm   │  │ DetectionResult    │ │
│  │            │  │ Recommendation│  │                    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │Data Loaders │  │ ML Adapters │  │ Repositories       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. AutonomousDetectionService

The main orchestrator for autonomous detection workflows.

```python
from pynomaly.application.services.autonomous_service import (
    AutonomousDetectionService,
    AutonomousConfig
)

class AutonomousDetectionService:
    """
    Main service for autonomous anomaly detection.

    Orchestrates the entire autonomous detection pipeline:
    1. Data loading and validation
    2. Data profiling and analysis
    3. Algorithm recommendation
    4. Detection execution
    5. Result aggregation and ranking
    """

    async def detect_autonomous(
        self,
        data_source: str,
        config: AutonomousConfig
    ) -> AutonomousDetectionResult:
        """
        Execute complete autonomous detection pipeline.

        Args:
            data_source: Path to data file or connection string
            config: Configuration for autonomous detection

        Returns:
            Comprehensive detection results with recommendations
        """
```

### 2. Data Profiling System

Analyzes data characteristics to inform algorithm selection.

```python
from pynomaly.application.services.autonomous_service import DataProfiler

class DataProfile:
    """Domain entity representing data characteristics."""

    n_samples: int
    n_features: int
    numeric_features: int
    categorical_features: int
    temporal_features: int
    missing_values_ratio: float
    correlation_score: float
    sparsity_ratio: float
    complexity_score: float
    recommended_contamination: float
    feature_importance: Dict[str, float]
    statistical_summary: Dict[str, Any]

async def _profile_data(
    self,
    dataset: Dataset,
    config: AutonomousConfig
) -> DataProfile:
    """
    Generate comprehensive data profile.

    Analyzes:
    - Basic statistics (mean, std, min, max)
    - Data types and distributions
    - Missing value patterns
    - Feature correlations
    - Outlier indicators
    - Complexity metrics
    """
```

### 3. Algorithm Recommendation Engine

Intelligently selects optimal algorithms based on data characteristics.

```python
class AlgorithmRecommendation:
    """Domain entity for algorithm recommendations."""

    algorithm: str
    confidence: float
    reasoning: str
    expected_performance: float
    hyperparameters: Dict[str, Any]
    computational_complexity: str
    memory_requirements: str

class AlgorithmRecommender:
    """Service for generating algorithm recommendations."""

    async def recommend_algorithms(
        self,
        profile: DataProfile,
        config: AutonomousConfig
    ) -> List[AlgorithmRecommendation]:
        """
        Generate ranked algorithm recommendations.

        Considers:
        - Data size and dimensionality
        - Feature types and distributions
        - Computational constraints
        - Performance requirements
        - Domain-specific patterns
        """
```

## Configuration System

### AutonomousConfig

Comprehensive configuration for autonomous detection:

```python
@dataclass
class AutonomousConfig:
    """Configuration for autonomous detection."""

    # Algorithm Selection
    max_algorithms: int = 3
    preferred_algorithms: Optional[List[str]] = None
    excluded_algorithms: Optional[List[str]] = None

    # Performance Settings
    auto_tune_hyperparams: bool = True
    confidence_threshold: float = 0.7
    max_execution_time: Optional[int] = None

    # Data Processing
    sample_size: Optional[int] = None
    contamination_override: Optional[float] = None
    feature_selection: bool = True

    # Output Options
    save_results: bool = False
    export_results: bool = False
    export_format: str = "json"
    save_models: bool = False

    # Monitoring
    verbose: bool = False
    progress_callback: Optional[Callable] = None

    # Advanced Options
    ensemble_mode: bool = False
    cross_validation: bool = False
    uncertainty_quantification: bool = False
```

## Extending the System

### Adding New Algorithms

1. **Create Algorithm Adapter**

```python
from pynomaly.shared.protocols import DetectorProtocol

class CustomAlgorithmAdapter(DetectorProtocol):
    """Adapter for custom anomaly detection algorithm."""

    def __init__(self, **hyperparams):
        self.hyperparams = hyperparams
        self.model = None

    async def fit(self, data: np.ndarray) -> None:
        """Train the algorithm on data."""
        self.model = CustomAlgorithm(**self.hyperparams)
        self.model.fit(data)

    async def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        return self.model.decision_function(data)

    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """Define hyperparameter search space."""
        return {
            'param1': {'type': 'float', 'range': [0.1, 1.0]},
            'param2': {'type': 'int', 'range': [10, 100]}
        }
```

2. **Register Algorithm**

```python
from pynomaly.infrastructure.registries import AlgorithmRegistry

# Register the new algorithm
AlgorithmRegistry.register(
    name="custom_algorithm",
    adapter_class=CustomAlgorithmAdapter,
    metadata={
        "description": "Custom anomaly detection algorithm",
        "suitable_for": ["tabular", "high_dimensional"],
        "complexity": "medium",
        "memory_usage": "low"
    }
)
```

3. **Add Recommendation Logic**

```python
class CustomAlgorithmRecommender:
    """Custom recommendation logic."""

    def should_recommend(self, profile: DataProfile) -> bool:
        """Determine if algorithm is suitable for data."""
        return (
            profile.n_features > 10 and
            profile.complexity_score > 0.5 and
            profile.categorical_features == 0
        )

    def get_confidence(self, profile: DataProfile) -> float:
        """Calculate confidence score."""
        base_confidence = 0.7
        if profile.n_samples > 1000:
            base_confidence += 0.1
        if profile.missing_values_ratio < 0.05:
            base_confidence += 0.1
        return min(base_confidence, 0.95)
```

### Custom Data Loaders

Add support for new data formats:

```python
from pynomaly.shared.protocols import DataLoaderProtocol

class CustomDataLoader(DataLoaderProtocol):
    """Loader for custom data format."""

    def can_load(self, source: str) -> bool:
        """Check if loader can handle the data source."""
        return source.endswith('.custom')

    async def load(self, source: str) -> Dataset:
        """Load data from custom format."""
        # Custom loading logic
        data = load_custom_format(source)

        return Dataset(
            name=Path(source).name,
            data=data,
            metadata={
                'source': source,
                'format': 'custom',
                'loaded_at': datetime.utcnow()
            }
        )
```

### Custom Profilers

Extend data profiling capabilities:

```python
class CustomDataProfiler:
    """Custom data profiling logic."""

    async def profile_domain_specific(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Add domain-specific profiling."""

        profile = {}

        # Time series specific analysis
        if self._is_time_series(data):
            profile.update(self._analyze_temporal_patterns(data))

        # Text data analysis
        if self._has_text_data(data):
            profile.update(self._analyze_text_features(data))

        # Graph data analysis
        if self._is_graph_data(data):
            profile.update(self._analyze_graph_structure(data))

        return profile
```

## Testing Autonomous Mode

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, Mock
from pynomaly.application.services.autonomous_service import AutonomousDetectionService

class TestAutonomousDetectionService:
    """Test suite for autonomous detection service."""

    @pytest.fixture
    def service(self):
        """Create service instance with mocked dependencies."""
        detector_repo = Mock()
        result_repo = Mock()
        data_loaders = {"csv": Mock()}

        return AutonomousDetectionService(
            detector_repository=detector_repo,
            result_repository=result_repo,
            data_loaders=data_loaders
        )

    @pytest.mark.asyncio
    async def test_data_profiling(self, service):
        """Test data profiling functionality."""
        # Create test dataset
        dataset = create_test_dataset()
        config = AutonomousConfig()

        # Profile data
        profile = await service._profile_data(dataset, config)

        # Verify profile
        assert profile.n_samples > 0
        assert profile.n_features > 0
        assert 0 <= profile.complexity_score <= 1
        assert 0 <= profile.recommended_contamination <= 1

    @pytest.mark.asyncio
    async def test_algorithm_recommendation(self, service):
        """Test algorithm recommendation engine."""
        # Create test profile
        profile = create_test_profile()
        config = AutonomousConfig(max_algorithms=3)

        # Get recommendations
        recommendations = await service._recommend_algorithms(profile, config)

        # Verify recommendations
        assert len(recommendations) <= 3
        assert all(0 <= rec.confidence <= 1 for rec in recommendations)
        assert all(rec.algorithm for rec in recommendations)
```

### Integration Testing

```python
@pytest.mark.integration
class TestAutonomousIntegration:
    """Integration tests for autonomous mode."""

    @pytest.mark.asyncio
    async def test_end_to_end_detection(self, temp_csv_file):
        """Test complete autonomous detection pipeline."""
        service = create_autonomous_service()
        config = AutonomousConfig(
            max_algorithms=2,
            auto_tune_hyperparams=False
        )

        # Run autonomous detection
        results = await service.detect_autonomous(str(temp_csv_file), config)

        # Verify results
        assert results.success
        assert results.recommendations
        assert results.best_result
        assert results.data_profile

    @pytest.mark.parametrize("data_type", ["tabular", "high_dimensional", "mixed"])
    async def test_different_data_types(self, data_type):
        """Test autonomous mode with different data types."""
        dataset = create_dataset_by_type(data_type)
        # Test logic here
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, columns

@given(
    data=data_frames([
        columns('feature1', dtype=float),
        columns('feature2', dtype=float),
        columns('feature3', dtype=int)
    ])
)
def test_profiling_properties(data):
    """Property-based tests for data profiling."""
    if len(data) < 10:  # Skip very small datasets
        return

    profiler = DataProfiler()
    profile = profiler.profile(data)

    # Properties that should always hold
    assert profile.n_samples == len(data)
    assert profile.n_features == len(data.columns)
    assert 0 <= profile.missing_values_ratio <= 1
    assert profile.complexity_score >= 0
```

## Performance Optimization

### Memory Management

```python
class MemoryOptimizedProfiler:
    """Memory-efficient data profiling."""

    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size

    async def profile_large_dataset(
        self,
        dataset: Dataset
    ) -> DataProfile:
        """Profile large datasets using chunking."""

        profiles = []
        for chunk in self._chunk_data(dataset.data):
            chunk_profile = await self._profile_chunk(chunk)
            profiles.append(chunk_profile)

        return self._aggregate_profiles(profiles)
```

### Parallel Processing

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

class ParallelAutonomousService:
    """Parallel execution for autonomous detection."""

    def __init__(self, max_workers: int = None):
        self.executor = ProcessPoolExecutor(max_workers=max_workers)

    async def parallel_algorithm_evaluation(
        self,
        algorithms: List[str],
        dataset: Dataset
    ) -> List[DetectionResult]:
        """Evaluate multiple algorithms in parallel."""

        loop = asyncio.get_event_loop()
        tasks = []

        for algorithm in algorithms:
            task = loop.run_in_executor(
                self.executor,
                self._run_algorithm,
                algorithm,
                dataset
            )
            tasks.append(task)

        return await asyncio.gather(*tasks)
```

### Caching Strategy

```python
from functools import lru_cache
import hashlib

class CachedProfiler:
    """Caching for expensive profiling operations."""

    @lru_cache(maxsize=100)
    def _cache_key(self, data_hash: str, config: str) -> str:
        """Generate cache key for profiling results."""
        return f"profile_{data_hash}_{config}"

    async def profile_with_cache(
        self,
        dataset: Dataset,
        config: AutonomousConfig
    ) -> DataProfile:
        """Profile data with caching."""

        # Generate cache key
        data_hash = hashlib.md5(
            dataset.data.to_string().encode()
        ).hexdigest()

        cache_key = self._cache_key(data_hash, str(config))

        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate profile
        profile = await self._profile_data(dataset, config)

        # Cache result
        self.cache[cache_key] = profile
        return profile
```

## Monitoring and Observability

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
autonomous_detections = Counter(
    'pynomaly_autonomous_detections_total',
    'Total autonomous detections',
    ['algorithm', 'status']
)

detection_duration = Histogram(
    'pynomaly_detection_duration_seconds',
    'Detection execution time',
    ['algorithm']
)

active_detections = Gauge(
    'pynomaly_active_detections',
    'Currently running detections'
)

class MonitoredAutonomousService:
    """Autonomous service with monitoring."""

    async def detect_autonomous(self, *args, **kwargs):
        """Monitored autonomous detection."""
        active_detections.inc()
        start_time = time.time()

        try:
            result = await super().detect_autonomous(*args, **kwargs)

            # Record metrics
            for rec in result.recommendations:
                autonomous_detections.labels(
                    algorithm=rec.algorithm,
                    status='success'
                ).inc()

                detection_duration.labels(
                    algorithm=rec.algorithm
                ).observe(time.time() - start_time)

            return result

        except Exception as e:
            autonomous_detections.labels(
                algorithm='unknown',
                status='error'
            ).inc()
            raise
        finally:
            active_detections.dec()
```

### Logging Strategy

```python
import structlog

logger = structlog.get_logger()

class LoggedAutonomousService:
    """Autonomous service with structured logging."""

    async def detect_autonomous(
        self,
        data_source: str,
        config: AutonomousConfig
    ) -> AutonomousDetectionResult:
        """Logged autonomous detection."""

        log = logger.bind(
            operation="autonomous_detection",
            data_source=data_source,
            config=config.dict()
        )

        log.info("Starting autonomous detection")

        try:
            # Load data
            log.info("Loading data")
            dataset = await self._auto_load_data(data_source, config)
            log.info("Data loaded", samples=dataset.data.shape[0])

            # Profile data
            log.info("Profiling data")
            profile = await self._profile_data(dataset, config)
            log.info("Data profiled", complexity=profile.complexity_score)

            # Get recommendations
            log.info("Generating recommendations")
            recommendations = await self._recommend_algorithms(profile, config)
            log.info("Recommendations generated", count=len(recommendations))

            # Execute detection
            log.info("Executing detection")
            results = await self._execute_detection(dataset, recommendations, config)
            log.info("Detection completed", best_algorithm=results.best_result.algorithm)

            return results

        except Exception as e:
            log.error("Detection failed", error=str(e))
            raise
```

## Best Practices

### Code Organization

1. **Separation of Concerns**: Keep profiling, recommendation, and execution logic separate
2. **Dependency Injection**: Use dependency injection for testability
3. **Protocol-Based Design**: Define clear interfaces for extensibility
4. **Error Handling**: Implement comprehensive error handling and recovery

### Performance Guidelines

1. **Lazy Loading**: Load algorithms only when needed
2. **Memory Management**: Use streaming for large datasets
3. **Caching**: Cache expensive computations
4. **Parallel Processing**: Leverage multiprocessing for CPU-bound tasks

### Testing Strategy

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Property-Based Tests**: Verify invariants across input space
4. **Performance Tests**: Monitor execution time and memory usage

### Documentation

1. **Code Documentation**: Comprehensive docstrings with examples
2. **Architecture Docs**: Clear architectural decision records
3. **API Documentation**: Complete API reference with examples
4. **User Guides**: Step-by-step usage guides

## Troubleshooting

### Common Development Issues

**Import Errors**
```python
# Circular import resolution
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pynomaly.application.services import AutonomousDetectionService
```

**Memory Leaks**
```python
# Proper cleanup in async code
async def detect_with_cleanup(self, *args, **kwargs):
    resources = []
    try:
        # Detection logic
        pass
    finally:
        # Cleanup resources
        for resource in resources:
            await resource.cleanup()
```

**Performance Bottlenecks**
```python
# Profiling autonomous mode
import cProfile

def profile_autonomous_detection():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run detection
    result = autonomous_service.detect(...)

    profiler.disable()
    profiler.dump_stats('autonomous_profile.stats')
```

For additional support, see the [API Reference](../api/autonomous-mode-api.md) and [Troubleshooting Guide](../troubleshooting/autonomous-mode-troubleshooting.md).

---

## 🔗 **Related Documentation**

### **Development**
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - How to contribute
- **[Development Setup](../contributing/README.md)** - Local development environment
- **[Architecture Overview](../architecture/overview.md)** - System design
- **[Implementation Guide](../contributing/IMPLEMENTATION_GUIDE.md)** - Coding standards

### **API Integration**
- **[REST API](../api-integration/rest-api.md)** - HTTP API reference
- **[Python SDK](../api-integration/python-sdk.md)** - Python client library
- **[CLI Reference](../api-integration/cli.md)** - Command-line interface
- **[Authentication](../api-integration/authentication.md)** - Security and auth

### **User Documentation**
- **[User Guides](../../user-guides/README.md)** - Feature usage guides
- **[Getting Started](../../getting-started/README.md)** - Installation and setup
- **[Examples](../../examples/README.md)** - Real-world use cases

### **Deployment**
- **[Production Deployment](../../deployment/README.md)** - Production deployment
- **[Security Setup](../../deployment/SECURITY.md)** - Security configuration
- **[Monitoring](../../user-guides/basic-usage/monitoring.md)** - System observability

---

## 🆘 **Getting Help**

- **[Development Troubleshooting](../contributing/troubleshooting/)** - Development issues
- **[GitHub Issues](https://github.com/your-org/pynomaly/issues)** - Report bugs
- **[Contributing Guidelines](../contributing/CONTRIBUTING.md)** - Contribution process

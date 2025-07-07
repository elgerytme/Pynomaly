# Pynomaly Testing Guide

This comprehensive guide covers all aspects of testing in the Pynomaly project, from basic unit tests to advanced testing strategies.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Organization](#test-organization)
3. [Testing Tools and Frameworks](#testing-tools-and-frameworks)
4. [Running Tests](#running-tests)
5. [Writing Tests](#writing-tests)
6. [Advanced Testing Strategies](#advanced-testing-strategies)
7. [Continuous Integration](#continuous-integration)
8. [Performance Testing](#performance-testing)
9. [Test Quality Assessment](#test-quality-assessment)
10. [Troubleshooting](#troubleshooting)

## Testing Philosophy

Pynomaly follows a comprehensive testing strategy based on these principles:

### Test-Driven Development (TDD)
- **Domain-first approach**: Start with domain entity tests
- **Red-Green-Refactor cycle**: Write failing tests, make them pass, refactor
- **Test requirements before implementation**: Define behavior through tests

### Clean Architecture Testing
- **Layer isolation**: Test each architectural layer independently
- **Dependency inversion**: Mock external dependencies
- **Interface testing**: Test protocols and abstractions

### Quality Metrics
- **Coverage target**: 85%+ for critical paths (domain/application layers)
- **Mutation score**: 60%+ for test quality assessment
- **Performance benchmarks**: Maintain regression testing

## Test Organization

```
tests/
├── fixtures/                    # Test data and shared fixtures
│   ├── conftest.py             # Pytest configuration and fixtures
│   └── test_data_generator.py  # Test data generation utilities
├── unit/                       # Unit tests
│   ├── domain/                 # Domain layer tests
│   ├── application/            # Application layer tests
│   ├── infrastructure/         # Infrastructure layer tests
│   └── presentation/           # Presentation layer tests
├── integration/                # Integration tests
│   ├── framework.py           # Integration testing framework
│   └── test_end_to_end_workflows.py
├── property_based/            # Property-based tests with Hypothesis
│   ├── test_domain_entities_properties.py
│   └── test_algorithm_properties.py
├── benchmarks/               # Performance benchmarks
│   ├── conftest.py
│   └── test_algorithm_performance.py
├── contract/                 # Contract and API tests
├── e2e/                     # End-to-end tests
└── branch_coverage/         # Branch coverage enhancement tests
```

## Testing Tools and Frameworks

### Core Testing Stack
- **pytest**: Primary testing framework
- **pytest-cov**: Coverage reporting
- **pytest-asyncio**: Async test support
- **pytest-xdist**: Parallel test execution
- **pytest-benchmark**: Performance benchmarking

### Advanced Testing Tools
- **Hypothesis**: Property-based testing
- **mutmut**: Mutation testing
- **Factory Boy**: Test data factories
- **Faker**: Fake data generation

### Quality Assurance
- **Coverage monitoring**: Automated coverage tracking
- **Mutation testing**: Test quality assessment
- **Performance regression**: Benchmark tracking

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/domain/              # Domain tests only
python -m pytest tests/integration/         # Integration tests only
python -m pytest -m "not slow"             # Skip slow tests

# Run with coverage
python -m pytest tests/ --cov=src/pynomaly --cov-report=html
```

### Using Hatch (Recommended)

```bash
# Run tests in isolated environment
hatch run test:pytest tests/

# Run specific test suites
hatch run test:pytest tests/domain/ -v
hatch run test:pytest tests/integration/ -v --tb=short

# Run with coverage
hatch run test:pytest tests/ --cov=src/pynomaly --cov-report=term --cov-report=html
```

### Test Markers

Tests are organized using pytest markers:

```bash
# Run tests by marker
pytest -m unit                  # Unit tests only
pytest -m integration          # Integration tests only
pytest -m benchmark           # Performance benchmarks
pytest -m "not slow"          # Skip slow tests
pytest -m property_based      # Property-based tests
```

### Parallel Execution

```bash
# Run tests in parallel
pytest -n auto                 # Auto-detect CPU cores
pytest -n 4                   # Use 4 processes
```

## Writing Tests

### Unit Test Example

```python
import pytest
from pynomaly.domain.entities import Dataset
from pynomaly.domain.exceptions import ValidationError

class TestDataset:
    def test_create_valid_dataset(self):
        """Test creating a valid dataset."""
        data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        dataset = Dataset(name="test", data=data)
        
        assert dataset.name == "test"
        assert dataset.n_samples == 3
        assert dataset.n_features == 2
    
    def test_empty_dataset_raises_error(self):
        """Test that empty dataset raises ValidationError."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValidationError):
            Dataset(name="empty", data=empty_data)
```

### Integration Test Example

```python
import pytest
from tests.integration.framework import IntegrationTestBuilder

@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_anomaly_detection_workflow():
    """Test complete anomaly detection pipeline."""
    
    async def load_data():
        # Load test data
        return test_data
    
    async def train_model():
        # Train anomaly detector
        return trained_model
    
    # Build integration test
    suite = (IntegrationTestBuilder("anomaly_detection", "Complete workflow")
             .add_step("load", "Load data", load_data)
             .add_step("train", "Train model", train_model, dependencies=["load"])
             .build())
    
    runner = IntegrationTestRunner()
    result = await runner.run_suite(suite)
    
    assert result.failed_steps == 0
```

### Property-Based Test Example

```python
from hypothesis import given, strategies as st
from pynomaly.domain.value_objects import ContaminationRate

class TestContaminationRateProperties:
    @given(st.floats(min_value=0.001, max_value=0.499))
    def test_valid_contamination_rates(self, rate):
        """Property: Valid rates should create valid objects."""
        contamination = ContaminationRate(rate)
        assert contamination.value == rate
        assert 0 < contamination.value < 0.5
```

### Performance Benchmark Example

```python
import pytest

@pytest.mark.benchmark
def test_isolation_forest_performance(benchmark):
    """Benchmark Isolation Forest algorithm."""
    
    def run_isolation_forest():
        detector = IsolationForest(n_estimators=100)
        detector.fit(X_train)
        return detector.predict(X_test)
    
    result = benchmark(run_isolation_forest)
    assert len(result) == len(X_test)
```

## Advanced Testing Strategies

### Test Data Management

Use the built-in test data generation system:

```python
from tests.fixtures.test_data_generator import TestDataManager

def test_with_generated_data():
    """Test using generated data."""
    manager = TestDataManager()
    
    # Get standardized test dataset
    df, labels = manager.get_dataset(
        'simple',
        n_samples=1000,
        n_features=10,
        contamination=0.1
    )
    
    # Create domain entities
    dataset, anomalies, result = manager.create_domain_entities('simple')
```

### Parametrized Testing

```python
@pytest.mark.parametrize("algorithm,params", [
    ("IsolationForest", {"n_estimators": 100}),
    ("LocalOutlierFactor", {"n_neighbors": 20}),
    ("OneClassSVM", {"nu": 0.1})
])
def test_algorithm_performance(algorithm, params):
    """Test multiple algorithms with different parameters."""
    detector = create_detector(algorithm, params)
    result = detector.fit_predict(X)
    assert len(result) == len(X)
```

### Fixture Usage

```python
@pytest.fixture
def sample_dataset():
    """Provide a sample dataset for tests."""
    return create_test_dataset(n_samples=100)

def test_using_fixture(sample_dataset):
    """Test that uses the fixture."""
    assert len(sample_dataset) == 100
```

## Continuous Integration

### GitHub Actions Workflows

The project uses automated CI/CD pipelines:

- **test.yml**: Main testing workflow
  - Matrix testing across Python versions (3.11, 3.12, 3.13)
  - Multiple OS support (Ubuntu, Windows, macOS)
  - Dependency level testing (minimal, standard, full)
  - Parallel test execution

- **quality.yml**: Code quality checks
  - Linting with ruff
  - Type checking with mypy
  - Complexity analysis
  - Mutation testing (on main branch)

### Running CI Locally

```bash
# Simulate CI environment
python scripts/testing/test_environment_manager.py matrix \
    --config test_matrix_config.yml \
    --command "python -m pytest tests/domain/ tests/application/"

# Generate test matrix report
python scripts/testing/test_environment_manager.py matrix \
    --config test_matrix_config.yml \
    --report local_test_matrix.html
```

## Performance Testing

### Benchmarking

```bash
# Run performance benchmarks
pytest tests/benchmarks/ --benchmark-only

# Generate benchmark report
pytest tests/benchmarks/ --benchmark-only --benchmark-json=benchmark.json

# Compare benchmarks
pytest-benchmark compare benchmark.json
```

### Performance Regression Detection

```bash
# Run performance regression check
python scripts/testing/performance_monitor.py check \
    --baseline benchmark_baseline.json \
    --current benchmark_current.json
```

### Memory Profiling

```bash
# Profile memory usage
pytest tests/benchmarks/ --benchmark-only --memray
```

## Test Quality Assessment

### Coverage Analysis

```bash
# Generate coverage report
python scripts/testing/coverage_monitor.py run

# Check coverage trends
python scripts/testing/coverage_monitor.py trends --days 30

# Generate HTML report
python scripts/testing/coverage_monitor.py report
```

### Mutation Testing

```bash
# Run mutation testing on domain layer
python scripts/testing/mutation_testing.py run \
    --paths src/pynomaly/domain/ \
    --test-command "python -m pytest tests/domain/ -x"

# Quick mutation testing
python scripts/testing/mutation_testing.py quick --domain-only
```

### Test Quality Metrics

Monitor these key metrics:

- **Code Coverage**: Target 85%+ for critical paths
- **Mutation Score**: Target 60%+ for test effectiveness
- **Test Execution Time**: Monitor for performance regressions
- **Test Reliability**: Track flaky test occurrences

## Environment Management

### Automated Environment Provisioning

```bash
# Create test environment
python scripts/testing/test_environment_manager.py create \
    --name test_env \
    --python-version 3.11 \
    --requirements requirements.txt

# Run test matrix
python scripts/testing/test_environment_manager.py matrix \
    --config test_matrix_config.yml

# Clean up environments
python scripts/testing/test_environment_manager.py clean --all
```

### Docker Testing

```bash
# Build test image
docker build -t pynomaly-test -f docker/Dockerfile.test .

# Run tests in container
docker run --rm pynomaly-test pytest tests/
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or use -e flag with pip
pip install -e .
```

#### Slow Tests
```bash
# Skip slow tests
pytest -m "not slow"

# Run specific test categories
pytest tests/domain/  # Fast unit tests only
```

#### Memory Issues
```bash
# Run tests with limited memory
pytest --maxfail=1  # Stop on first failure
pytest -x           # Stop on first failure (short form)
```

#### Flaky Tests
```bash
# Run with retries
pytest --lf         # Run last failed tests
pytest --ff         # Run failures first
```

### Debugging Tests

```python
# Add debugging to tests
import pytest

def test_with_debugging():
    result = function_under_test()
    
    # Add breakpoint for debugging
    breakpoint()  # Python 3.7+
    # or
    import pdb; pdb.set_trace()
    
    assert result.is_valid
```

### Test Data Issues

```bash
# Clear test data cache
python -c "from tests.fixtures.test_data_generator import TestDataManager; TestDataManager().clear_cache()"

# Regenerate test data
pytest tests/ --cache-clear
```

## Best Practices

### Writing Effective Tests

1. **Follow AAA Pattern**: Arrange, Act, Assert
2. **One assertion per test**: Focus on single behavior
3. **Use descriptive names**: Test names should describe behavior
4. **Test edge cases**: Include boundary conditions
5. **Mock external dependencies**: Isolate units under test

### Test Organization

1. **Group related tests**: Use test classes for organization
2. **Use fixtures for setup**: Avoid repetitive setup code
3. **Keep tests independent**: Tests should not depend on each other
4. **Test both positive and negative cases**: Happy path and error conditions

### Performance Considerations

1. **Use appropriate test sizes**: Unit < Integration < E2E
2. **Parallel execution**: Leverage pytest-xdist for speed
3. **Cache test data**: Use TestDataManager for consistent data
4. **Profile slow tests**: Identify and optimize bottlenecks

### Maintenance

1. **Regular test cleanup**: Remove obsolete tests
2. **Update test data**: Keep test datasets relevant
3. **Monitor test metrics**: Track coverage and mutation scores
4. **Review test failures**: Investigate and fix flaky tests

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Property-Based Testing Guide](https://increment.com/testing/in-praise-of-property-based-testing/)
- [Test-Driven Development](https://testdriven.io/)

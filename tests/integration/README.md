# Integration Testing Framework

Comprehensive integration testing suite for the Pynomaly anomaly detection platform, providing production-grade test coverage for all system components.

## Overview

This integration testing framework validates the complete anomaly detection platform through realistic workflows, performance benchmarks, and security testing.

### Test Categories

- **End-to-End Workflows** (`test_end_to_end_workflows.py`): Complete user workflows from data ingestion to results
- **Performance Benchmarks** (`test_performance_benchmarks.py`): Performance, scalability, and resource utilization tests
- **Security Integration** (`test_security_integration.py`): Authentication, authorization, encryption, and audit trails

### Architecture

The integration tests follow the same clean architecture principles as the main application:

```
tests/integration/
├── conftest.py                     # Test configuration and fixtures
├── test_end_to_end_workflows.py    # Complete workflow tests
├── test_performance_benchmarks.py  # Performance and benchmark tests
├── test_security_integration.py    # Security and compliance tests
├── pytest.ini                      # Pytest configuration
└── README.md                       # This file
```

## Getting Started

### Prerequisites

1. **Python 3.11+** with all dependencies installed
2. **Redis** (optional, for cache testing)
3. **PostgreSQL** (optional, for database testing)

### Installation

```bash
# Install development dependencies
poetry install --with dev,test

# Install integration test dependencies
pip install pytest pytest-asyncio pytest-benchmark pytest-mock httpx
```

### Running Tests

#### Basic Test Execution

```bash
# Run all integration tests
pytest tests/integration/

# Run specific test categories
pytest tests/integration/ -m "not slow"           # Exclude slow tests
pytest tests/integration/ -m "performance"        # Performance tests only
pytest tests/integration/ -m "security"           # Security tests only
pytest tests/integration/ -m "end_to_end"         # E2E tests only
```

#### Advanced Options

```bash
# Run with coverage reporting
pytest tests/integration/ --cov=pynomaly --cov-report=html

# Run with performance profiling
pytest tests/integration/ --benchmark-only

# Run with detailed logging
pytest tests/integration/ --log-cli-level=DEBUG

# Run specific test file
pytest tests/integration/test_end_to_end_workflows.py

# Run specific test method
pytest tests/integration/test_performance_benchmarks.py::TestPerformanceBenchmarks::test_detection_latency_benchmark
```

#### Test Markers

Use pytest markers to run specific test subsets:

```bash
# Run tests that require Redis
pytest -m "requires_redis"

# Run tests that require database
pytest -m "requires_database"

# Run quick tests only (exclude slow tests)
pytest -m "not slow"

# Run stress tests
pytest -m "stress"
```

## Test Configuration

### Environment Variables

Configure tests using environment variables:

```bash
# Test database
export PYNOMALY_TEST_DATABASE_URL="sqlite:///test.db"

# Test Redis
export PYNOMALY_TEST_REDIS_URL="redis://localhost:6379/15"

# Test data directory
export PYNOMALY_TEST_DATA_DIR="/tmp/pynomaly_test"

# Performance benchmarks
export PYNOMALY_TEST_MAX_LATENCY_MS="100"
export PYNOMALY_TEST_MIN_THROUGHPUT="1000"
```

### Configuration Files

- `pytest.ini`: Main pytest configuration
- `conftest.py`: Test fixtures and shared configuration
- Performance benchmarks defined in fixtures

## Test Structure

### Fixtures

Key fixtures provided in `conftest.py`:

- **`test_settings`**: Test application settings
- **`test_container`**: Dependency injection container
- **`test_cache`**: In-memory cache service
- **`test_security_service`**: Security service with test configuration
- **`sample_datasets`**: Generated test datasets
- **`performance_benchmarks`**: Performance thresholds
- **`integration_helper`**: Helper for API operations
- **`integration_assertions`**: Custom test assertions

### Test Data

The framework generates various types of test data:

```python
# Normal dataset (no anomalies)
normal_dataset = Dataset(
    name="normal_test_data",
    data=np.random.randn(1000, 5),
    metadata=DatasetMetadata(...)
)

# Anomalous dataset (10% anomalies)
anomalous_dataset = Dataset(
    name="anomalous_test_data",
    data=combined_data,
    metadata=DatasetMetadata(anomaly_labels=labels)
)

# Time series dataset
time_series_dataset = Dataset(
    name="time_series_test_data",
    data=time_series_data,
    metadata=DatasetMetadata(is_time_series=True)
)
```

## Writing Integration Tests

### Basic Test Structure

```python
@pytest.mark.end_to_end
@pytest.mark.asyncio
async def test_my_workflow(
    integration_helper: IntegrationTestHelper,
    integration_assertions: IntegrationTestAssertions,
    sample_dataset_csv: str,
):
    \"\"\"Test description.\"\"\"
    # 1. Setup
    dataset = await integration_helper.upload_dataset(
        sample_dataset_csv, "my_test_dataset"
    )

    # 2. Execute workflow
    detector = await integration_helper.create_detector(
        dataset["id"], "isolation_forest"
    )

    # 3. Validate results
    integration_assertions.assert_api_response_structure(
        {"status": "success", "data": detector},
        ["id", "name", "algorithm"]
    )
```

### Performance Testing

```python
@pytest.mark.performance
async def test_performance_benchmark(
    detection_service,
    performance_benchmarks: Dict,
):
    \"\"\"Test performance benchmark.\"\"\"
    start_time = time.perf_counter()

    # Execute operation
    result = await detection_service.detect(detector, data)

    execution_time = time.perf_counter() - start_time

    # Validate against benchmark
    assert execution_time <= performance_benchmarks["detection_latency_ms"] / 1000
```

### Security Testing

```python
@pytest.mark.security
async def test_security_workflow(
    test_security_service,
    integration_assertions: IntegrationTestAssertions,
):
    \"\"\"Test security workflow.\"\"\"
    # Test authentication
    session = test_security_service.auth_service.authenticate_user(
        username="test_user",
        password="TestPassword123!",
        ip_address="127.0.0.1",
        user_agent="test-client",
    )

    assert session is not None

    # Test authorization
    has_permission = test_security_service.authz_service.check_permission(
        "test_user", "data:read", SecurityLevel.INTERNAL
    )

    assert has_permission == True
```

## Test Helpers

### IntegrationTestHelper

Provides common operations for integration tests:

```python
# Upload dataset
dataset = await integration_helper.upload_dataset(file_path, "dataset_name")

# Create detector
detector = await integration_helper.create_detector(dataset_id, algorithm)

# Train detector
result = await integration_helper.train_detector(detector_id)

# Wait for async operation
operation = await integration_helper.wait_for_operation(operation_id)

# Cleanup resources
await integration_helper.cleanup_resources()
```

### IntegrationTestAssertions

Custom assertions for integration testing:

```python
# Assert API response structure
integration_assertions.assert_api_response_structure(
    response, ["field1", "field2", "field3"]
)

# Assert detection quality
integration_assertions.assert_detection_quality(result, min_accuracy=0.8)

# Assert performance benchmarks
integration_assertions.assert_performance_benchmarks(metrics, benchmarks)

# Assert resource cleanup
integration_assertions.assert_resource_cleanup(response, "detector")
```

## Performance Benchmarks

### Default Benchmarks

```python
performance_benchmarks = {
    "detection_latency_ms": 100,           # Max detection latency
    "training_time_seconds": 30,           # Max training time
    "memory_usage_mb": 512,                # Max memory usage
    "throughput_samples_per_second": 1000, # Min throughput
    "cache_hit_rate": 0.8,                 # Min cache hit rate
    "accuracy_threshold": 0.7,             # Min accuracy
    "precision_threshold": 0.6,            # Min precision
    "recall_threshold": 0.6,               # Min recall
}
```

### Custom Benchmarks

Override benchmarks for specific tests:

```python
@pytest.fixture
def custom_benchmarks():
    return {
        "detection_latency_ms": 50,  # Stricter requirement
        "accuracy_threshold": 0.9,   # Higher accuracy requirement
    }

async def test_with_custom_benchmarks(custom_benchmarks):
    # Test with custom performance requirements
    pass
```

## Continuous Integration

### GitHub Actions

```yaml
name: Integration Tests
on: [push, pull_request]

jobs:
  integration:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install poetry
        poetry install --with dev,test

    - name: Run integration tests
      run: |
        poetry run pytest tests/integration/ \
          --cov=pynomaly \
          --cov-report=xml \
          -m "not slow"

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Local CI Simulation

```bash
# Run the same tests as CI
./scripts/testing/test-integration-ci.sh

# Run with coverage reporting
./scripts/testing/test-integration-coverage.sh

# Run performance regression tests
./scripts/testing/test-performance-regression.sh
```

## Troubleshooting

### Common Issues

1. **Redis Connection Errors**
   ```bash
   # Start Redis for testing
   docker run -d -p 6379:6379 redis:7-alpine

   # Or skip Redis tests
   pytest -m "not requires_redis"
   ```

2. **Database Connection Errors**
   ```bash
   # Use SQLite for testing (default)
   export PYNOMALY_TEST_DATABASE_URL="sqlite:///test.db"

   # Or skip database tests
   pytest -m "not requires_database"
   ```

3. **Performance Test Failures**
   ```bash
   # Run with relaxed benchmarks
   export PYNOMALY_TEST_MAX_LATENCY_MS="200"

   # Or skip performance tests
   pytest -m "not performance"
   ```

4. **Memory Issues**
   ```bash
   # Run tests with memory monitoring
   pytest --tb=short --maxfail=1 -v

   # Run single test at a time
   pytest tests/integration/test_performance_benchmarks.py::test_memory_usage_benchmark
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
pytest tests/integration/ \
  --log-cli-level=DEBUG \
  --log-cli-format="%(asctime)s [%(levelname)8s] %(name)s: %(message)s" \
  -v -s
```

### Test Isolation

Ensure test isolation by running tests individually:

```bash
# Run each test file separately
pytest tests/integration/test_end_to_end_workflows.py
pytest tests/integration/test_performance_benchmarks.py  
pytest tests/integration/test_security_integration.py

# Run with fresh environment each time
pytest --forked tests/integration/
```

## Contributing

### Adding New Tests

1. **Create test file** following naming convention `test_*.py`
2. **Add appropriate markers** (`@pytest.mark.category`)
3. **Use fixtures** from `conftest.py` for consistency
4. **Add documentation** explaining test purpose and setup
5. **Update benchmarks** if needed for new performance tests

### Test Guidelines

- **Use descriptive test names** that explain what is being tested
- **Follow AAA pattern**: Arrange, Act, Assert
- **Test realistic scenarios** that match production usage
- **Include negative tests** for error conditions
- **Validate both functionality and performance**
- **Clean up resources** in fixtures or helper methods
- **Add meaningful assertions** with clear error messages

### Performance Testing Guidelines

- **Set realistic benchmarks** based on production requirements
- **Test with representative data sizes**
- **Include memory and CPU monitoring**
- **Test concurrent operations**
- **Validate scalability characteristics**
- **Monitor for memory leaks and resource cleanup**

## Related Documentation

- [Testing Strategy](../../docs/testing_strategy.md)
- [Performance Monitoring](../../docs/performance_monitoring.md)
- [Security Architecture](../../docs/security_architecture.md)
- [API Documentation](../../docs/api_documentation.md)

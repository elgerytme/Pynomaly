# SDK Client Tests

This directory contains comprehensive tests for the platform SDK client libraries.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_sdk_core.py    # Core SDK functionality
│   └── ...
├── integration/             # Integration tests with mocked services
│   ├── test_anomaly_detection_integration.py
│   └── ...
├── e2e/                     # End-to-end tests against real APIs
│   ├── test_full_workflows.py
│   └── ...
├── performance/             # Performance and load tests
│   ├── test_client_performance.py
│   └── ...
└── fixtures/                # Test data and fixtures
    ├── sample_data.py
    └── ...
```

## Running Tests

### Prerequisites

Install test dependencies:

```bash
# Python tests
pip install -e ../shared/sdk_core[dev]
pip install -e ../clients/anomaly_detection_client[dev]

# TypeScript tests (if applicable)
cd ../clients/platform_client_ts
npm install
```

### Unit Tests

Test individual components in isolation:

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_sdk_core.py -v

# Run with coverage
pytest tests/unit/ --cov=sdk_core --cov-report=html
```

### Integration Tests

Test client integration with mocked services:

```bash
# Run integration tests
pytest tests/integration/ -v

# Run specific integration test
pytest tests/integration/test_anomaly_detection_integration.py -v
```

### End-to-End Tests

Test against real API services (requires API keys):

```bash
# Set up environment
export ANOMALY_DETECTION_API_KEY=your-api-key
export INTEGRATION_TEST=true

# Run e2e tests
pytest tests/e2e/ -v -m "not slow"

# Run all e2e tests including slow ones
pytest tests/e2e/ -v
```

### Performance Tests

Test client performance and resource usage:

```bash
# Run performance tests
pytest tests/performance/ -v

# Run with performance profiling
pytest tests/performance/ -v --profile
```

## Test Categories

### Unit Tests (`tests/unit/`)

- **test_sdk_core.py**: Core SDK functionality
  - Configuration management
  - Authentication handlers
  - HTTP client behavior
  - Error handling
  - Response models
  - Utility functions

- **test_models.py**: Data model validation
  - Pydantic model validation
  - Serialization/deserialization
  - Type checking

- **test_auth.py**: Authentication components
  - JWT token handling
  - Token refresh logic
  - Different auth methods

### Integration Tests (`tests/integration/`)

- **test_anomaly_detection_integration.py**: Anomaly detection client
  - Basic detection workflows
  - Model training and prediction
  - Batch processing
  - Error scenarios
  - Data format handling

- **test_mlops_integration.py**: MLOps client integration
  - Pipeline management
  - Model deployment
  - Monitoring integration

### End-to-End Tests (`tests/e2e/`)

- **test_full_workflows.py**: Complete user workflows
  - Account setup to anomaly detection
  - Model training to production deployment
  - Multi-service interactions

- **test_real_api.py**: Tests against live API
  - Health checks
  - Authentication flows
  - Rate limiting behavior
  - Error response handling

### Performance Tests (`tests/performance/`)

- **test_client_performance.py**: Client performance
  - Request/response times
  - Memory usage
  - Connection pooling efficiency
  - Concurrent request handling

- **test_load_scenarios.py**: Load testing
  - High-volume detection requests
  - Batch processing performance
  - Resource consumption under load

## Test Configuration

### pytest Configuration

The tests use pytest with these key configurations:

```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=sdk_core",
    "--cov=anomaly_detection_client",
    "--cov-report=term-missing",
    "--cov-report=xml",
    "--cov-report=html",
    "--strict-markers",
    "--asyncio-mode=auto",
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "slow: Slow running tests",
    "performance: Performance tests",
]
```

### Environment Variables

Tests can be configured via environment variables:

```bash
# API Configuration
ANOMALY_DETECTION_API_KEY=your-api-key
PLATFORM_BASE_URL=https://api.platform.com

# Test Control
INTEGRATION_TEST=true          # Enable integration tests
E2E_TEST=true                 # Enable e2e tests
PERFORMANCE_TEST=true         # Enable performance tests
SLOW_TESTS=true              # Enable slow tests

# Test Environment
TEST_ENVIRONMENT=development  # Test environment to use
TEST_TIMEOUT=30              # Test timeout in seconds
```

### Fixtures and Test Data

Common test fixtures are available:

```python
# Sample data fixtures
@pytest.fixture
def sample_anomaly_data():
    """Generate sample data with known anomalies."""
    return create_test_data_with_anomalies()

@pytest.fixture  
def mock_api_responses():
    """Mock API responses for testing."""
    return load_mock_responses()

# Configuration fixtures
@pytest.fixture
def test_config():
    """Test client configuration."""
    return ClientConfig.for_environment(Environment.LOCAL)
```

## Mock Services

For integration testing, we provide mock services that simulate the API behavior:

### Mock API Server

```python
# Start mock server for testing
from tests.mocks import MockAPIServer

async with MockAPIServer() as server:
    # Run tests against mock server
    client = AnomalyDetectionClient(base_url=server.url)
    # ... test code
```

### Response Mocking

```python
# Mock specific API responses
with patch('sdk_core.client.BaseClient._make_request') as mock_request:
    mock_request.return_value = {
        "success": True,
        "anomalies": [0, 1],
        "scores": [0.9, 0.8, 0.1]
    }
    
    result = await client.detect(data=test_data)
    assert result.anomalies == [0, 1]
```

## Continuous Integration

Tests are automatically run in CI/CD pipelines:

### GitHub Actions

```yaml
# .github/workflows/test-sdk.yml
name: SDK Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e src/packages/shared/sdk_core[dev]
        pip install -e src/packages/clients/anomaly_detection_client[dev]
    
    - name: Run unit tests
      run: pytest tests/unit/ -v
    
    - name: Run integration tests
      run: pytest tests/integration/ -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Test Quality Standards

### Coverage Requirements

- **Unit tests**: > 90% code coverage
- **Integration tests**: > 80% API coverage
- **E2E tests**: Cover all major user workflows

### Performance Benchmarks

- **Response time**: < 100ms for simple detection
- **Memory usage**: < 50MB for typical workloads
- **Concurrent requests**: Handle 100+ concurrent requests

### Test Documentation

Each test should include:

- Clear docstring describing what is being tested
- Expected behavior and edge cases
- Setup and teardown requirements
- Performance expectations (for performance tests)

## Debugging Tests

### Running Individual Tests

```bash
# Run single test with verbose output
pytest tests/unit/test_sdk_core.py::TestClientConfig::test_default_config -v -s

# Run with pdb debugging
pytest tests/unit/test_sdk_core.py::TestClientConfig::test_default_config --pdb

# Run with custom markers
pytest -m "unit and not slow" -v
```

### Test Output and Logging

```bash
# Enable debug logging
pytest tests/ -v -s --log-cli-level=DEBUG

# Capture print statements
pytest tests/ -v -s --capture=no
```

### Common Issues

1. **Import Errors**: Make sure SDK packages are installed in development mode
2. **Async Test Issues**: Ensure `pytest-asyncio` is installed and configured
3. **Mock Failures**: Check that mocked objects match the expected interface
4. **Environment Variables**: Verify test environment variables are set correctly

## Contributing to Tests

When adding new features:

1. **Write unit tests** for new functionality
2. **Add integration tests** for new API endpoints
3. **Update e2e tests** for new user workflows
4. **Include performance tests** for performance-critical features
5. **Update test documentation** for new test categories

### Test Review Checklist

- [ ] Tests cover happy path and edge cases
- [ ] Error conditions are tested
- [ ] Performance impact is measured
- [ ] Tests are deterministic (no flaky tests)
- [ ] Test data is realistic but not sensitive
- [ ] Documentation is updated
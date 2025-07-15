# Pynomaly Integration Testing Suite

🍞 **Breadcrumb:** 🏠 [Home](../index.md) > 📁 Testing

---


This document describes the comprehensive integration testing suite for Pynomaly, providing end-to-end testing capabilities for all major system components and workflows.

## Overview

The integration testing suite validates:

- ✅ **Complete API workflows** - End-to-end user scenarios
- ✅ **Database operations** - Data persistence and CRUD operations  
- ✅ **Real-time streaming** - WebSocket connections and data processing
- ✅ **Performance testing** - Load testing and scalability validation
- ✅ **Security features** - Authentication, authorization, and vulnerability testing
- ✅ **End-to-end scenarios** - Real-world user workflows
- ✅ **Regression testing** - Backward compatibility and breaking change detection

## Quick Start

### 1. Health Check

First, verify your testing environment is properly configured:

```bash
python scripts/test_health_check.py
```

This will check:
- Dependencies installation
- Test infrastructure setup
- API functionality
- Sample data creation

### 2. Run All Integration Tests

Execute the complete integration test suite:

```bash
python scripts/run_integration_tests.py
```

### 3. Run Specific Test Suites

Run individual test categories:

```bash
# API workflows
python scripts/run_integration_tests.py --suite test_api_workflows.py

# Database operations
python scripts/run_integration_tests.py --suite test_database_integration.py

# Streaming functionality
python scripts/run_integration_tests.py --suite test_streaming_integration.py

# Performance testing
python scripts/run_integration_tests.py --suite test_performance_integration.py

# Security testing
python scripts/run_integration_tests.py --suite test_security_integration.py

# End-to-end scenarios
python scripts/run_integration_tests.py --suite test_end_to_end_scenarios.py

# Regression testing
python scripts/run_integration_tests.py --suite test_regression_suite.py
```

### 4. Advanced Options

```bash
# Run with coverage reporting
python scripts/run_integration_tests.py --coverage

# Enable authentication testing
python scripts/run_integration_tests.py --auth

# Fail fast on first error
python scripts/run_integration_tests.py --fail-fast

# Filter by test markers
python scripts/run_integration_tests.py --markers api workflows

# Set custom timeout
python scripts/run_integration_tests.py --timeout 3600

# Increase max failures per suite
python scripts/run_integration_tests.py --max-failures 10
```

## Test Suite Structure

### 📁 `tests/integration/`

```
tests/integration/
├── conftest.py                      # Test configuration and fixtures
├── test_api_workflows.py            # Complete API workflow testing
├── test_database_integration.py     # Database operations and persistence
├── test_streaming_integration.py    # Real-time streaming and WebSocket
├── test_performance_integration.py  # Load testing and performance
├── test_security_integration.py     # Security and vulnerability testing
├── test_end_to_end_scenarios.py     # Real-world user scenarios
└── test_regression_suite.py         # Backward compatibility testing
```

### 🔧 Configuration Files

- **`pytest.ini`** - Pytest configuration with markers and settings
- **`scripts/run_integration_tests.py`** - Test runner with reporting
- **`scripts/test_health_check.py`** - Environment validation script

## Test Categories

### 1. API Workflows (`test_api_workflows.py`)

Tests complete API workflows from data upload to predictions:

```python
# Example: Complete anomaly detection workflow
async def test_complete_anomaly_detection_workflow():
    # 1. Upload dataset
    # 2. Create detector  
    # 3. Train model
    # 4. Validate performance
    # 5. Make predictions
    # 6. Verify results
```

**Key Test Scenarios:**
- Complete anomaly detection workflow
- Streaming anomaly detection
- Experiment management and comparison
- Model lifecycle management
- AutoML workflows
- Event processing and pattern detection
- Performance monitoring

### 2. Database Integration (`test_database_integration.py`)

Validates data persistence and database operations:

```python
# Example: Dataset CRUD operations
async def test_dataset_crud_operations():
    # Create, Read, Update, Delete operations
    # Data integrity validation
    # Concurrent access testing
```

**Key Test Scenarios:**
- CRUD operations for all entities
- Model versioning and persistence
- Experiment result storage
- Streaming session state management
- Transaction integrity
- Concurrent access handling

### 3. Streaming Integration (`test_streaming_integration.py`)

Tests real-time data processing capabilities:

```python
# Example: WebSocket streaming
async def test_websocket_streaming_monitoring():
    # WebSocket connection establishment
    # Real-time data processing
    # Metrics collection
    # Error handling
```

**Key Test Scenarios:**
- WebSocket connections
- Session lifecycle management
- Concurrent streaming sessions
- Error handling and recovery
- Data sink integration
- Performance under load

### 4. Performance Integration (`test_performance_integration.py`)

Validates system performance and scalability:

```python
# Example: API load testing
async def test_api_performance_under_load():
    # Concurrent request handling
    # Throughput measurement
    # Response time analysis
    # Resource utilization
```

**Key Test Scenarios:**
- API performance under load
- Streaming scalability
- Database performance
- Memory usage patterns
- Rate limiting effectiveness

### 5. Security Integration (`test_security_integration.py`)

Tests security features and vulnerability resistance:

```python
# Example: Input validation
async def test_input_validation_and_sanitization():
    # SQL injection prevention
    # XSS protection
    # Path traversal prevention
    # Command injection protection
```

**Key Test Scenarios:**
- Input validation and sanitization
- Authentication bypass attempts
- Authorization and access control
- Data privacy and encryption
- Rate limiting and DDoS protection
- Secure file upload handling
- Security headers validation

### 6. End-to-End Scenarios (`test_end_to_end_scenarios.py`)

Real-world user workflows and complete scenarios:

```python
# Example: Data scientist research workflow
async def test_data_scientist_research_workflow():
    # Data exploration
    # Algorithm comparison
    # Hyperparameter optimization
    # Model evaluation
    # Results export
```

**Key Test Scenarios:**
- Data scientist research workflow
- Production deployment workflow
- Security analyst monitoring workflow
- Complete user journeys
- Cross-system integration

### 7. Regression Suite (`test_regression_suite.py`)

Ensures backward compatibility and prevents breaking changes:

```python
# Example: API compatibility
async def test_api_backward_compatibility():
    # Response format validation
    # Field presence verification
    # Data type consistency
    # Endpoint behavior
```

**Key Test Scenarios:**
- API backward compatibility
- Model serialization compatibility
- Streaming API stability
- Error response consistency
- Configuration stability
- Performance regression detection

## Test Configuration

### Environment Variables

Tests automatically configure the environment:

```bash
PYNOMALY_ENVIRONMENT=testing
PYNOMALY_LOG_LEVEL=INFO
PYNOMALY_CACHE_ENABLED=false
PYNOMALY_AUTH_ENABLED=false  # Configurable
PYNOMALY_DOCS_ENABLED=true
```

### Test Markers

Use pytest markers to run specific test categories:

```bash
# Run only API tests
pytest -m api

# Run performance tests
pytest -m performance

# Run security tests  
pytest -m security

# Run slow tests
pytest -m slow

# Exclude slow tests
pytest -m "not slow"
```

Available markers:
- `api` - API endpoint tests
- `database` - Database operation tests
- `streaming` - Real-time streaming tests
- `performance` - Performance and load tests
- `security` - Security and vulnerability tests
- `regression` - Regression and compatibility tests
- `slow` - Long-running tests
- `integration` - All integration tests

### Fixtures and Test Data

The test suite provides comprehensive fixtures:

```python
# Async client for API testing
async def test_example(async_test_client: AsyncClient):
    response = await async_test_client.get("/api/health")

# Integration helper for resource management
async def test_example(integration_helper: IntegrationTestHelper):
    dataset = await integration_helper.upload_dataset(csv_path, "test_dataset")
    detector = await integration_helper.create_detector(dataset["id"], "isolation_forest")

# Sample datasets
def test_example(sample_dataset_csv: str, sample_time_series_csv: str):
    # Pre-generated test datasets
```

## Test Reports

### Automatic Report Generation

The test runner generates comprehensive reports:

- **HTML Report**: `test_reports/integration_test_summary.html`
- **Text Report**: `test_reports/integration_test_summary.txt`
- **JUnit XML**: `test_reports/junit_*.xml`
- **Coverage Report**: `test_reports/coverage/` (when `--coverage` is used)

### Example HTML Report

```html
<!DOCTYPE html>
<html>
<head>
    <title>Pynomaly Integration Test Report</title>
</head>
<body>
    <h1>Integration Test Summary</h1>
    <div class="summary">
        <p>Total Suites: 7</p>
        <p>Passed: 6</p>
        <p>Failed: 1</p>
        <p>Success Rate: 85.7%</p>
        <p>Total Duration: 245.3 seconds</p>
    </div>
    <!-- Detailed results for each test suite -->
</body>
</html>
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests
on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install poetry
        poetry install

    - name: Health check
      run: poetry run python scripts/test_health_check.py

    - name: Run integration tests
      run: |
        poetry run python scripts/run_integration_tests.py \
          --coverage \
          --max-failures 3 \
          --timeout 1800

    - name: Upload test reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-test-reports
        path: test_reports/

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      if: always()
      with:
        file: test_reports/coverage/coverage.xml
```

### Docker Integration

```dockerfile
# Dockerfile.integration-tests
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install

# Health check
RUN python scripts/test_health_check.py

# Run integration tests
CMD ["python", "scripts/run_integration_tests.py", "--coverage"]
```

## Best Practices

### 1. Test Data Management

- Use temporary directories for test data
- Clean up resources after each test
- Provide realistic sample datasets
- Handle concurrent test execution

### 2. Error Handling

- Test both success and failure scenarios
- Validate error messages and status codes
- Test timeout and retry behavior
- Verify graceful degradation

### 3. Performance Considerations

- Set reasonable timeouts
- Use concurrent testing where appropriate
- Monitor resource usage
- Test scalability limits

### 4. Security Testing

- Test input validation thoroughly
- Verify authentication and authorization
- Check for common vulnerabilities
- Validate security headers

### 5. Maintenance

- Update tests when APIs change
- Maintain backward compatibility tests
- Regular performance baseline updates
- Keep test documentation current

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Ensure dependencies are installed
poetry install

# Verify Python path
python scripts/test_health_check.py
```

**2. Database Connection Issues**
```bash
# Solution: Check database configuration
export PYNOMALY_DATABASE_URL="sqlite:///test.db"
```

**3. Timeout Errors**
```bash
# Solution: Increase timeout for slow tests
python scripts/run_integration_tests.py --timeout 3600
```

**4. Resource Cleanup Issues**
```bash
# Solution: Run individual test suites
python scripts/run_integration_tests.py --suite test_api_workflows.py
```

**5. Authentication Errors**
```bash
# Solution: Disable auth for testing
python scripts/run_integration_tests.py  # Auth disabled by default
```

### Debug Mode

Enable detailed logging for debugging:

```bash
# Set debug log level
python scripts/run_integration_tests.py --log-level DEBUG

# Run specific test with verbose output
pytest tests/integration/test_api_workflows.py::TestCompleteAPIWorkflows::test_complete_anomaly_detection_workflow -v -s
```

## Contributing

### Adding New Integration Tests

1. **Choose the appropriate test file** based on functionality
2. **Follow naming conventions**: `test_*` functions  
3. **Use provided fixtures** for setup and cleanup
4. **Add appropriate markers** for categorization
5. **Include comprehensive assertions**
6. **Add documentation** for complex test scenarios

### Example New Test

```python
@pytest.mark.asyncio
@pytest.mark.api
@pytest.mark.workflows  
async def test_new_feature_workflow(
    async_test_client: AsyncClient,
    integration_helper: IntegrationTestHelper,
    sample_dataset_csv: str,
    disable_auth
):
    """Test new feature end-to-end workflow."""

    # Setup
    dataset = await integration_helper.upload_dataset(
        sample_dataset_csv,
        "new_feature_test"
    )

    # Test new feature
    response = await async_test_client.post(
        "/api/new-feature",
        json={"dataset_id": dataset["id"]}
    )
    response.raise_for_status()

    # Validate result
    result = response.json()["data"]
    assert "feature_result" in result
    assert result["status"] == "completed"
```

### Test Review Checklist

- [ ] Test covers both success and failure cases
- [ ] Appropriate fixtures and markers used
- [ ] Resources properly cleaned up
- [ ] Realistic test data and scenarios
- [ ] Clear test documentation
- [ ] Performance considerations addressed
- [ ] Security implications tested

---

For more information, see the main [README.md](README.md) and [CLAUDE.md](CLAUDE.md) files.

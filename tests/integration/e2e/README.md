# Comprehensive End-to-End Integration Testing

This directory contains the comprehensive end-to-end integration testing framework for Pynomaly, implementing Issue #164: Phase 6.1 Integration Testing - End-to-End Validation.

## 🎯 Overview

The integration testing framework provides thorough validation of:

- **Cross-Architectural Layer Integration**: Tests interaction between API, Application, Domain, and Infrastructure layers
- **End-to-End Workflows**: Complete user journey validation from data ingestion to result visualization
- **Performance and Load Testing**: System behavior under various load conditions with stress testing
- **Security and Compliance**: Authentication, authorization, data protection, and regulatory compliance
- **Multi-Tenant Isolation**: Proper tenant separation and resource isolation
- **Resilience and Recovery**: System behavior under failure conditions

## 📁 Test Structure

```
tests/integration/e2e/
├── conftest.py                                    # Test configuration and fixtures
├── test_enhanced_cross_layer_integration.py       # Cross-layer integration tests
├── test_end_to_end_workflows.py                   # Complete workflow tests
├── test_performance_load.py                       # Performance and stress tests
├── test_security_compliance.py                    # Security and compliance tests
├── test_sdk_integration.py                        # SDK integration tests
├── test_integration_runner.py                     # Test orchestration and reporting
└── README.md                                      # This documentation
```

## 🚀 Quick Start

### Running Individual Test Suites

```bash
# Cross-layer integration tests
pytest tests/integration/e2e/test_enhanced_cross_layer_integration.py -v

# Performance and load tests
pytest tests/integration/e2e/test_performance_load.py -v -m "not stress"

# Security tests
pytest tests/integration/e2e/test_security_compliance.py -v

# All integration tests (excluding stress tests)
pytest tests/integration/e2e/ -v -m "integration and not stress"
```

### Running Comprehensive Test Suite

```bash
# Basic comprehensive test run
python tests/integration/e2e/test_integration_runner.py

# Include stress tests
python tests/integration/e2e/test_integration_runner.py --include-stress

# Parallel execution for faster runs
python tests/integration/e2e/test_integration_runner.py --parallel

# Exclude slow tests for quick validation
python tests/integration/e2e/test_integration_runner.py --exclude-slow
```

### Using with Hatch (Recommended)

```bash
# Run integration tests with proper environment
hatch run test:run-integration

# Run with coverage
hatch run test:run-cov-integration

# Run parallel integration tests
hatch run test:run-parallel-integration
```

## 📋 Test Categories and Markers

### Pytest Markers

- `@pytest.mark.integration` - All integration tests
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.performance` - Performance baseline tests
- `@pytest.mark.load` - Load testing scenarios
- `@pytest.mark.stress` - Stress testing (resource intensive)
- `@pytest.mark.security` - Security validation tests
- `@pytest.mark.compliance` - Regulatory compliance tests
- `@pytest.mark.slow` - Tests that take significant time
- `@pytest.mark.resource` - Resource utilization tests

### Running Specific Categories

```bash
# Only performance tests
pytest tests/integration/e2e/ -m "performance"

# Security and compliance only
pytest tests/integration/e2e/ -m "security or compliance"

# Fast tests only (exclude slow and stress)
pytest tests/integration/e2e/ -m "integration and not slow and not stress"

# Load tests without stress tests
pytest tests/integration/e2e/ -m "load and not stress"
```

## 🏗️ Test Configuration

### Environment Variables

```bash
# Test database configuration
export PYNOMALY_TEST_DATABASE_URL="sqlite:///:memory:"

# API configuration
export PYNOMALY_TEST_API_BASE_URL="http://testserver"
export PYNOMALY_TEST_API_TIMEOUT="30.0"

# Load testing configuration
export PYNOMALY_TEST_LOAD_DURATION="10"
export PYNOMALY_TEST_CONCURRENT_REQUESTS="5"

# Feature flags
export PYNOMALY_TEST_ENABLE_STRESS_TESTS="false"
export PYNOMALY_TEST_ENABLE_SECURITY_TESTS="true"
```

### Test Data Configuration

The test framework automatically generates synthetic datasets:

- **Sample Dataset**: 1000 samples, 5 features, 5% anomaly rate
- **Multimodal Dataset**: Multiple clusters with scattered anomalies
- **Time Series Dataset**: Temporal data with trend and seasonality
- **Large Dataset**: Configurable size for stress testing

## 🔬 Detailed Test Descriptions

### Cross-Architectural Layer Integration

**File**: `test_enhanced_cross_layer_integration.py`

- **Complete Data Flow**: API → Application → Domain → Infrastructure
- **Event-Driven Workflows**: Cross-layer event propagation
- **Resilience Testing**: Failure recovery across layers
- **Multi-Tenant Isolation**: Tenant separation validation
- **Distributed Processing**: Multi-worker coordination

### End-to-End Workflows

**File**: `test_end_to_end_workflows.py`

- **Basic Anomaly Detection**: Create → Train → Detect workflow
- **Multimodal Detection**: Advanced data type handling
- **Time Series Analysis**: Temporal anomaly detection
- **Batch Processing**: Large-scale data processing
- **Model Lifecycle**: Complete ML model management

### Performance and Load Testing

**File**: `test_performance_load.py`

Enhanced features:
- **Performance Baselines**: Individual operation benchmarks
- **Scaling Tests**: Performance with varying data sizes
- **Concurrent Operations**: Multi-user simulation
- **Rapid Fire Requests**: High-frequency request handling
- **Resource Monitoring**: CPU, memory, and system utilization
- **Stress Testing**: System behavior under extreme load
- **Sustained Load**: Endurance testing over time
- **Memory Pressure**: Large dataset handling

### Security and Compliance

**File**: `test_security_compliance.py`

- **Authentication**: API key validation and format checking
- **Authorization**: Resource isolation and permissions
- **Data Privacy**: Sensitive data protection and sanitization
- **Input Validation**: Protection against injection attacks
- **Audit Logging**: Compliance tracking requirements
- **Data Retention**: GDPR/regulatory compliance
- **Data Minimization**: Privacy-by-design principles

## 📊 Test Reporting

### Automated Reports

The integration runner generates comprehensive reports:

```json
{
  "start_time": 1647856800.0,
  "end_time": 1647857100.0,
  "total_duration": 300.0,
  "overall_success_rate": 0.95,
  "suite_results": [
    {
      "suite_name": "Cross-Layer Integration",
      "passed": 15,
      "failed": 1,
      "skipped": 2,
      "success_rate": 0.83,
      "duration": 45.2,
      "errors": []
    }
  ],
  "system_info": {
    "platform": "Linux-5.15.0",
    "python_version": "3.11.0",
    "cpu_count": 8,
    "memory_total_gb": 16.0
  }
}
```

### Report Locations

- **JSON Reports**: `test-results/integration_test_report_YYYYMMDD_HHMMSS.json`
- **HTML Reports**: `test-results/integration_report.html` (with `--html` flag)
- **Coverage Reports**: `htmlcov/index.html` (with coverage enabled)

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Use in-memory database for tests
   export PYNOMALY_TEST_DATABASE_URL="sqlite:///:memory:"
   ```

2. **Port Conflicts**
   ```bash
   # Check for running services
   netstat -tulpn | grep :8000
   # Kill conflicting processes
   pkill -f "uvicorn.*8000"
   ```

3. **Memory Issues with Stress Tests**
   ```bash
   # Exclude stress tests
   pytest tests/integration/e2e/ -m "not stress"
   # Or increase system memory/swap
   ```

4. **Timeout Issues**
   ```bash
   # Increase test timeouts
   export PYNOMALY_TEST_API_TIMEOUT="60.0"
   pytest tests/integration/e2e/ --timeout=300
   ```

### Debug Mode

```bash
# Enable verbose logging
pytest tests/integration/e2e/ -v -s --log-cli-level=DEBUG

# Capture output
pytest tests/integration/e2e/ --capture=no

# Fail fast on first error
pytest tests/integration/e2e/ -x
```

### Performance Tuning

```bash
# Parallel test execution
pytest tests/integration/e2e/ -n auto

# Rerun failed tests only
pytest tests/integration/e2e/ --lf

# Skip slow tests for development
pytest tests/integration/e2e/ -m "not slow"
```

## 🔧 Development and Extension

### Adding New Test Scenarios

1. **Create Test Class**:
   ```python
   @pytest.mark.asyncio
   @pytest.mark.integration
   class TestNewScenario:
       async def test_new_functionality(self, async_client, api_headers):
           # Test implementation
           pass
   ```

2. **Add to Test Runner**:
   ```python
   # In test_integration_runner.py
   self.test_suites.append(
       ("New Scenario", "test_new_scenario.py")
   )
   ```

3. **Update Documentation**:
   - Add to this README
   - Update test markers
   - Document configuration

### Custom Fixtures

```python
# In conftest.py
@pytest.fixture
def custom_test_data():
    """Generate custom test data for specific scenarios."""
    return generate_custom_data()
```

### Performance Benchmarks

```python
def test_performance_benchmark(performance_monitor):
    performance_monitor.start_timer("operation")
    # Perform operation
    performance_monitor.end_timer("operation")
    
    duration = performance_monitor.get_duration("operation")
    assert_performance_within_limits(duration, 2.0)  # 2 second limit
```

## 📈 Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests
on: [push, pull_request]

jobs:
  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Integration Tests
        run: |
          python tests/integration/e2e/test_integration_runner.py \
            --exclude-slow --parallel
```

### Test Metrics and Monitoring

- **Success Rate Tracking**: Monitor test stability over time
- **Performance Regression**: Alert on performance degradation
- **Coverage Tracking**: Ensure comprehensive test coverage
- **Flaky Test Detection**: Identify and fix unstable tests

## 🎯 Success Criteria

The integration testing framework validates Issue #164 requirements:

- ✅ **Cross-Package Integration**: Tests interaction between all packages
- ✅ **Production Workflow Validation**: Complete user journeys tested
- ✅ **Performance Regression Detection**: Baseline and stress testing
- ✅ **Security Workflow Testing**: Authentication, authorization, compliance
- ✅ **End-to-End Validation**: Full system integration coverage
- ✅ **Multi-Tenant Isolation**: Tenant separation validation
- ✅ **Disaster Recovery**: System resilience under failure conditions

### Quality Gates

- **Minimum Success Rate**: 90% for integration tests
- **Performance Baselines**: No more than 20% degradation
- **Security Compliance**: 100% pass rate for security tests
- **Test Coverage**: 95% coverage for integration scenarios

## 📚 Related Documentation

- [Testing Strategy](../../../docs/development/testing-strategy.md)
- [API Documentation](../../../docs/api/README.md)
- [Architecture Decision Records](../../../docs/architecture/adr/)
- [Development Setup](../../../docs/development/setup.md)

## 🤝 Contributing

When adding new integration tests:

1. Follow existing patterns and conventions
2. Add appropriate pytest markers
3. Include performance assertions where relevant
4. Update this documentation
5. Ensure tests are deterministic and reliable
6. Add cleanup procedures to prevent test pollution

For questions or issues with the integration testing framework, please refer to the main project issues or create new ones with the `testing` label.
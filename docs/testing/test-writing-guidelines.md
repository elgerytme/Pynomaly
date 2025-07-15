# 🧪 Test Writing Guidelines for Pynomaly

## Overview

This document provides comprehensive guidelines for writing high-quality tests in the Pynomaly project to achieve and maintain 100% test coverage.

## Table of Contents

1. [Test Structure and Organization](#test-structure-and-organization)
2. [Test Naming Conventions](#test-naming-conventions)
3. [Test Types and Categories](#test-types-and-categories)
4. [Writing Effective Tests](#writing-effective-tests)
5. [Coverage Requirements](#coverage-requirements)
6. [Testing Best Practices](#testing-best-practices)
7. [Tools and Frameworks](#tools-and-frameworks)
8. [Examples and Templates](#examples-and-templates)

## Test Structure and Organization

### Directory Structure

```
tests/
├── unit/                   # Unit tests
│   ├── domain/            # Domain layer tests
│   ├── application/       # Application layer tests
│   └── infrastructure/    # Infrastructure layer tests
├── integration/           # Integration tests
│   ├── api/              # API integration tests
│   ├── database/         # Database integration tests
│   └── external/         # External service tests
├── e2e/                  # End-to-end tests
├── performance/          # Performance tests
├── security/             # Security tests
├── contract/             # Contract tests
└── fixtures/             # Test fixtures and data
```

### File Organization

- **One test file per source file**: `src/module/service.py` → `tests/unit/module/test_service.py`
- **Mirror source structure**: Test directory structure should mirror source code structure
- **Group related tests**: Use test classes to group related test functions

## Test Naming Conventions

### Test Files
- **Pattern**: `test_<module_name>.py`
- **Examples**: `test_anomaly_detector.py`, `test_data_processor.py`

### Test Functions
- **Pattern**: `test_<what_is_being_tested>_<expected_outcome>`
- **Examples**:
  ```python
  def test_detector_fit_with_valid_data_succeeds():
  def test_anomaly_detection_with_empty_data_raises_error():
  def test_score_calculation_returns_expected_range():
  ```

### Test Classes
- **Pattern**: `Test<ClassName>` or `Test<Functionality>`
- **Examples**:
  ```python
  class TestAnomalyDetector:
  class TestDataValidation:
  class TestAPIEndpoints:
  ```

## Test Types and Categories

### Unit Tests
- **Purpose**: Test individual functions, methods, or classes in isolation
- **Scope**: Single unit of code
- **Dependencies**: Mocked or stubbed
- **Speed**: Very fast (< 100ms per test)

```python
@pytest.mark.unit
def test_anomaly_score_calculation():
    """Test anomaly score calculation with known input."""
    detector = AnomalyDetector()
    data = np.array([[1, 2], [3, 4]])
    scores = detector.calculate_scores(data)
    assert len(scores) == 2
    assert all(0 <= score <= 1 for score in scores)
```

### Integration Tests
- **Purpose**: Test interaction between components
- **Scope**: Multiple units working together
- **Dependencies**: Real or test doubles
- **Speed**: Medium (< 5s per test)

```python
@pytest.mark.integration
def test_api_anomaly_detection_pipeline():
    """Test complete anomaly detection through API."""
    client = TestClient(app)
    data = {"features": [[1, 2], [3, 4]]}
    response = client.post("/api/detect", json=data)
    assert response.status_code == 200
    assert "anomalies" in response.json()
```

### End-to-End Tests
- **Purpose**: Test complete user workflows
- **Scope**: Full system functionality
- **Dependencies**: Production-like environment
- **Speed**: Slow (< 30s per test)

```python
@pytest.mark.e2e
def test_complete_anomaly_detection_workflow():
    """Test complete anomaly detection from data upload to results."""
    # Upload data, train model, detect anomalies, export results
    pass
```

## Writing Effective Tests

### Test Structure (AAA Pattern)

```python
def test_function_name():
    """Clear description of what is being tested."""
    # Arrange - Set up test data and dependencies
    detector = AnomalyDetector(contamination=0.1)
    test_data = np.array([[1, 2], [3, 4], [5, 6]])
    
    # Act - Execute the code being tested
    result = detector.fit_predict(test_data)
    
    # Assert - Verify the expected outcome
    assert len(result) == 3
    assert all(r in [-1, 1] for r in result)
```

### Comprehensive Assertions

```python
def test_anomaly_detection_result():
    """Test anomaly detection returns valid results."""
    detector = AnomalyDetector()
    data = generate_test_data()
    
    result = detector.detect(data)
    
    # Test multiple aspects
    assert isinstance(result, AnomalyResult)
    assert len(result.scores) == len(data)
    assert all(0 <= score <= 1 for score in result.scores)
    assert result.threshold > 0
    assert len(result.anomalies) <= len(data)
    assert all(idx < len(data) for idx in result.anomaly_indices)
```

### Error Testing

```python
def test_detector_with_invalid_data_raises_error():
    """Test that detector raises appropriate error for invalid data."""
    detector = AnomalyDetector()
    
    # Test various invalid inputs
    with pytest.raises(ValueError, match="Data cannot be empty"):
        detector.fit(np.array([]))
    
    with pytest.raises(TypeError, match="Data must be numeric"):
        detector.fit([["not", "numeric"]])
    
    with pytest.raises(ValueError, match="Contamination must be between 0 and 1"):
        AnomalyDetector(contamination=1.5)
```

### Parametrized Tests

```python
@pytest.mark.parametrize("contamination,expected_anomalies", [
    (0.1, 10),
    (0.2, 20),
    (0.05, 5),
])
def test_contamination_rate_affects_anomaly_count(contamination, expected_anomalies):
    """Test that contamination rate affects number of detected anomalies."""
    data = generate_normal_data(100)
    detector = AnomalyDetector(contamination=contamination)
    
    result = detector.fit_predict(data)
    anomaly_count = sum(1 for r in result if r == -1)
    
    assert abs(anomaly_count - expected_anomalies) <= 2  # Allow small variance
```

## Coverage Requirements

### Coverage Targets
- **Overall Project**: 90% minimum, 100% goal
- **Domain Layer**: 100% (critical business logic)
- **Application Layer**: 95% minimum
- **Infrastructure Layer**: 85% minimum
- **New Code**: 100% (all new code must be fully tested)

### Coverage Types
- **Line Coverage**: Every line of code executed
- **Branch Coverage**: Every branch (if/else) path taken
- **Function Coverage**: Every function called
- **Condition Coverage**: Every boolean condition tested

### Exclusions
Acceptable coverage exclusions (use `# pragma: no cover`):
- Abstract methods
- Platform-specific code
- Debug/logging statements
- `__repr__` and `__str__` methods
- Exception handling for unexpected errors

```python
def abstract_method(self):  # pragma: no cover
    raise NotImplementedError

if sys.platform == "win32":  # pragma: no cover
    # Windows-specific code
    pass
```

## Testing Best Practices

### 1. Test Independence
- Each test should be independent and can run in any order
- Use fixtures for setup and teardown
- Don't rely on test execution order

### 2. Test Data Management
```python
@pytest.fixture
def sample_data():
    """Provide consistent test data."""
    return np.random.rand(100, 5)

@pytest.fixture
def trained_detector():
    """Provide a trained detector for testing."""
    detector = AnomalyDetector()
    detector.fit(generate_training_data())
    return detector
```

### 3. Mock External Dependencies
```python
@patch('external_service.api_call')
def test_service_with_external_dependency(mock_api):
    """Test service behavior with mocked external dependency."""
    mock_api.return_value = {"status": "success"}
    
    service = ExternalService()
    result = service.process_data(test_data)
    
    assert result.success
    mock_api.assert_called_once()
```

### 4. Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(
    data=st.lists(st.lists(st.floats(min_value=0, max_value=100), min_size=1), min_size=1),
    contamination=st.floats(min_value=0.01, max_value=0.5)
)
def test_detector_handles_random_data(data, contamination):
    """Test detector with randomly generated valid data."""
    detector = AnomalyDetector(contamination=contamination)
    np_data = np.array(data)
    
    # Should not raise exceptions with valid data
    result = detector.fit_predict(np_data)
    assert len(result) == len(data)
```

### 5. Performance Testing
```python
@pytest.mark.benchmark
def test_detection_performance(benchmark):
    """Test detection performance with benchmark."""
    detector = AnomalyDetector()
    data = generate_large_dataset(10000)
    
    result = benchmark(detector.fit_predict, data)
    
    # Performance assertions
    assert len(result) == 10000
```

## Tools and Frameworks

### Core Testing Tools
- **pytest**: Primary testing framework
- **pytest-cov**: Coverage measurement
- **pytest-asyncio**: Async testing support
- **pytest-benchmark**: Performance testing
- **pytest-xdist**: Parallel test execution

### Additional Tools
- **hypothesis**: Property-based testing
- **factory-boy**: Test data generation
- **responses**: HTTP request mocking
- **freezegun**: Time/date mocking
- **mutmut**: Mutation testing

### Configuration
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
addopts = 
    --cov=src/pynomaly
    --cov-branch
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=90
    --strict-markers
```

## Examples and Templates

### Basic Test Template
```python
"""Tests for module_name functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from pynomaly.domain.entities import EntityName
from pynomaly.application.services import ServiceName


class TestEntityName:
    """Test suite for EntityName."""
    
    @pytest.fixture
    def valid_entity(self):
        """Provide a valid entity instance."""
        return EntityName(param1="value1", param2="value2")
    
    def test_entity_creation_with_valid_params_succeeds(self):
        """Test entity creation with valid parameters."""
        # Arrange
        params = {"param1": "value1", "param2": "value2"}
        
        # Act
        entity = EntityName(**params)
        
        # Assert
        assert entity.param1 == "value1"
        assert entity.param2 == "value2"
        assert entity.is_valid()
    
    def test_entity_creation_with_invalid_params_raises_error(self):
        """Test entity creation with invalid parameters raises appropriate error."""
        with pytest.raises(ValueError, match="param1 cannot be empty"):
            EntityName(param1="", param2="value2")
    
    @pytest.mark.parametrize("param1,param2,expected", [
        ("value1", "value2", True),
        ("", "value2", False),
        ("value1", "", False),
    ])
    def test_entity_validation(self, param1, param2, expected):
        """Test entity validation with various parameter combinations."""
        try:
            entity = EntityName(param1=param1, param2=param2)
            result = entity.is_valid()
            assert result == expected
        except ValueError:
            assert not expected
```

### Integration Test Template
```python
"""Integration tests for service layer."""

import pytest
from unittest.mock import Mock

from pynomaly.application.services import ServiceName
from pynomaly.infrastructure.repositories import RepositoryName


@pytest.mark.integration
class TestServiceIntegration:
    """Test service integration with repository layer."""
    
    @pytest.fixture
    def mock_repository(self):
        """Provide mock repository."""
        return Mock(spec=RepositoryName)
    
    @pytest.fixture
    def service(self, mock_repository):
        """Provide service with mocked repository."""
        return ServiceName(repository=mock_repository)
    
    def test_service_operation_with_repository_interaction(self, service, mock_repository):
        """Test service operation that interacts with repository."""
        # Arrange
        mock_repository.find_by_id.return_value = Mock(id=1, name="test")
        
        # Act
        result = service.process_item(item_id=1)
        
        # Assert
        assert result.success
        mock_repository.find_by_id.assert_called_once_with(1)
        mock_repository.save.assert_called_once()
```

## Continuous Improvement

### Coverage Monitoring
- Monitor coverage trends over time
- Set up alerts for coverage decreases
- Regular review of uncovered code

### Test Quality Reviews
- Regular review of test code quality
- Identify and refactor complex tests
- Ensure tests remain maintainable

### Performance Monitoring
- Monitor test execution time
- Optimize slow tests
- Maintain fast feedback loops

### Documentation
- Keep test documentation up to date
- Document complex test scenarios
- Share testing knowledge across team

---

**Remember**: Good tests are not just about coverage percentage—they should be readable, maintainable, and provide confidence in the system's behavior.
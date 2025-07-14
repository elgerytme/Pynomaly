# Pynomaly Test Configuration

This document describes the consolidated test configuration and setup for the Pynomaly project.

## Overview

The test configuration has been consolidated to eliminate duplication, improve maintainability, and provide consistent testing patterns across all test modules.

## Structure

```
tests/
├── conftest.py              # Root configuration - essential fixtures only
├── pytest.ini              # DEPRECATED - moved to project root
├── README.md               # This file
├── shared/                 # Centralized test utilities
│   ├── __init__.py
│   ├── fixtures.py         # Common fixtures for all tests
│   ├── factories.py        # Mock factories and data generators
│   └── utilities.py        # Test helper functions and utilities
├── cli/                    # CLI-specific tests
│   ├── conftest.py         # CLI-specific fixtures only
│   └── test_*.py
├── ui/                     # UI/Frontend tests
│   ├── conftest.py         # UI-specific fixtures only
│   └── test_*.py
└── [other domains]/        # Domain-specific test directories
    ├── conftest.py         # Domain-specific fixtures only
    └── test_*.py
```

## Key Features

### 1. Unified Mock Factory

```python
def test_example(mock_factory):
    # Create various types of mocks
    detector_mock = mock_factory.create_detector_mock()
    database_mock = mock_factory.create_database_mock()
    api_client_mock = mock_factory.create_api_client_mock()
```

### 2. Unified Data Factory

```python
def test_data_processing(data_factory):
    # Generate test data
    df = data_factory.create_sample_dataframe(rows=100, with_anomalies=True)
    time_series = data_factory.create_time_series_data(length=1000)
    labeled_data, labels = data_factory.create_anomaly_detection_dataset()
```

### 3. Resource Management

```python
def test_file_operations(resource_manager):
    # Create temporary files and directories
    temp_file = resource_manager.create_temp_file(suffix=".csv")
    temp_dir = resource_manager.create_temp_dir()
    
    # Automatic cleanup after test
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit                    # Unit tests only
pytest -m "integration and not slow"  # Fast integration tests

# Run with coverage
pytest --cov=pynomaly
```

## Writing Tests

### Basic Test Structure

```python
def test_anomaly_detection(mock_factory, data_factory):
    """Test basic anomaly detection functionality."""
    # Arrange
    detector_mock = mock_factory.create_detector_mock()
    test_data = data_factory.create_sample_dataframe()
    
    # Act
    results = detector_mock.predict(test_data)
    
    # Assert
    assert len(results) == len(test_data)
```

For complete documentation, see the full README in the repository.
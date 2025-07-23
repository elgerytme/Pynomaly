# Test Utilities

Shared test utilities and fixtures for consistent testing across all packages in the monorepo.

## Overview

This package provides:

- **Common Fixtures**: Database sessions, HTTP clients, temporary files, mock services
- **Test Factories**: Data generation using Factory Boy for consistent test data
- **Test Helpers**: Assertion helpers, data generation, external service mocking
- **Standard Markers**: Consistent test categorization across packages

## Usage

```python
from test_utilities import (
    async_client,
    temp_directory,
    UserFactory,
    assert_response_valid,
    unit_test,
    integration_test
)

@unit_test
async def test_user_creation(async_client, temp_directory):
    """Test user creation endpoint."""
    user_data = UserFactory()
    
    response = await async_client.post("/users", json=user_data)
    data = assert_response_valid(response, 201, ["id", "username"])
    
    assert data["username"] == user_data["username"]

@integration_test
def test_data_processing(temp_directory):
    """Test data processing workflow."""
    input_file = temp_directory / "input.csv" 
    # Test implementation...
```

## Available Fixtures

- `async_client`: Async HTTP client for API testing
- `sync_client`: Synchronous HTTP client for API testing  
- `db_session`: Mock database session
- `temp_directory`: Temporary directory for file operations
- `mock_logger`: Mock structured logger
- `mock_redis`: Mock Redis client
- `mock_s3_client`: Mock S3 client
- `mock_external_service`: Mock external service

## Test Factories

- `UserFactory`: Generate test user data
- `DataFactory`: Generate generic test data objects
- `ModelFactory`: Generate ML model test data
- `ExperimentFactory`: Generate experiment test data
- `DatasetFactory`: Generate dataset test data

## Test Helpers

- `assert_response_valid()`: Validate HTTP responses
- `assert_model_trained()`: Validate trained ML models
- `generate_test_data()`: Create synthetic data for ML
- `create_temp_file()`: Create temporary files
- `mock_external_service()`: Mock external services

## Standard Test Markers

### Test Types
- `@unit_test`: Unit tests
- `@integration_test`: Integration tests  
- `@e2e_test`: End-to-end tests
- `@performance_test`: Performance tests
- `@security_test`: Security tests
- `@slow_test`: Slow running tests

### Domain Specific
- `@ml_test`: Machine learning tests
- `@data_test`: Data processing tests
- `@api_test`: API endpoint tests
- `@cli_test`: CLI tests
- `@database_test`: Database tests

### Environment
- `@requires_docker`: Tests requiring Docker
- `@requires_gpu`: Tests requiring GPU
- `@local_only`: Local environment only
- `@ci_only`: CI environment only

## Integration in Package Tests

Add to your package's `pyproject.toml`:

```toml
[project.optional-dependencies]
test = [
    "test_utilities @ file:///src/shared/test_utilities",
    # other test dependencies...
]
```

Add to your `conftest.py`:

```python
from test_utilities.markers import pytest_configure
from test_utilities.fixtures import *
```

## Contributing

When adding new utilities:

1. Follow the established patterns
2. Add comprehensive docstrings
3. Include type hints
4. Add tests for the utilities themselves
5. Update this README
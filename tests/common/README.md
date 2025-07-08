# Common Test Utilities

This directory contains shared utilities, fixtures, and helper functions to reduce code duplication across test files.

## Overview

The `tests.common` package provides:

- **Data Generation**: Standardized test data generators for consistent testing across different scenarios
- **Mock Objects**: Factory for creating common mock objects (repositories, services, entities)
- **Storage Management**: Temporary storage management with automatic cleanup
- **Async Helpers**: Utilities for async testing
- **Assertions**: Common assertion helpers with detailed error messages
- **Configuration**: Test configuration helpers

## Usage

### Basic Import

```python
from tests.common import (
    test_data_generator,
    mock_factory,
    sample_data,  # pytest fixture
    mock_repository,  # pytest fixture
)
```

### Data Generation

```python
from tests.common import test_data_generator

# Generate simple dataset with anomalies
df, labels = test_data_generator.generate_simple_dataset(
    n_samples=1000, 
    n_features=5, 
    contamination=0.1
)

# Generate time series data
ts_df, ts_labels = test_data_generator.generate_time_series_dataset(
    n_timestamps=1000,
    n_features=3,
    anomaly_periods=[(200, 250), (700, 750)]
)

# Generate mixed-type data
mixed_df, mixed_labels = test_data_generator.generate_mixed_type_dataset(
    n_samples=500,
    contamination=0.15
)
```

### Mock Objects

```python
from tests.common import mock_factory

# Create mock repository
repo = mock_factory.create_mock_repository()
repo.get.return_value = some_object

# Create mock detector
detector = mock_factory.create_mock_detector(detector_id="test-123")

# Create mock dataset
dataset = mock_factory.create_mock_dataset(data=some_dataframe)

# Create mock service
service = mock_factory.create_mock_service()
```

### Using Fixtures

```python
import pytest
from tests.common import sample_data, mock_repository, temp_storage

@pytest.mark.asyncio
async def test_my_feature(sample_data, mock_repository, temp_storage):
    """Test using common fixtures."""
    df, labels = sample_data
    
    # Use the mock repository
    mock_repository.get.return_value = some_value
    
    # Use temporary storage
    test_file = temp_storage / "test_data.json"
    # ... write to file
    
    # Cleanup happens automatically
```

### Assertions

```python
from tests.common import assertions
import pandas as pd
import numpy as np

# Assert DataFrames are equal
assertions.assert_dataframe_equal(df1, df2)

# Assert arrays are almost equal
assertions.assert_array_almost_equal(arr1, arr2, decimal=5)

# Assert score is in valid range
assertions.assert_score_in_range(anomaly_score, min_val=0.0, max_val=1.0)

# Assert async mock was called correctly
assertions.assert_async_mock_called_with(mock_service.process, arg1, arg2)
```

### Configuration

```python
from tests.common import config_helper

# Create test settings
settings = config_helper.create_test_settings(
    debug=True,
    storage_path="/custom/path"
)

# Create algorithm configuration
algo_config = config_helper.create_algorithm_config(
    algorithm_name="IsolationForest",
    contamination=0.15,
    n_estimators=200
)
```

### Async Testing

```python
from tests.common import async_helper

# Run async function in sync context
result = async_helper.run_async(some_async_function())

# Wait for a condition with timeout
async def test_async_condition():
    success = await async_helper.wait_for_condition(
        lambda: some_condition_is_true(),
        timeout=5.0,
        check_interval=0.1
    )
    assert success
```

### Storage Management

```python
from tests.common import storage_manager

# Create temporary directory
temp_dir = storage_manager.create_temp_directory()

# Use the directory for testing
test_file = temp_dir / "test.txt"
test_file.write_text("test data")

# Cleanup happens automatically at the end of test session
```

## Benefits

1. **Reduced Duplication**: Common patterns are centralized and reusable
2. **Consistency**: Standardized data generation and mock objects across tests
3. **Maintainability**: Changes to common patterns only need to be made in one place
4. **Reliability**: Well-tested utilities reduce the chance of test-specific bugs
5. **Documentation**: Clear patterns for new test development

## Migration Guide

### Before (Duplicated Code)

```python
# In multiple test files
@pytest.fixture
def sample_data():
    np.random.seed(42)
    normal_data = np.random.randn(1000, 5)
    # ... duplicate implementation
    
@pytest.fixture  
def mock_repository():
    repo = Mock()
    repo.get = AsyncMock()
    # ... duplicate implementation
```

### After (Using Common Utilities)

```python
# In test files
from tests.common import sample_data, mock_repository

def test_my_feature(sample_data, mock_repository):
    # Test implementation using shared fixtures
    pass
```

## Adding New Utilities

When adding new common utilities:

1. Add the utility class or function to `utils.py`
2. Export it in `__init__.py`
3. Document it in this README
4. Add tests for the utility itself
5. Update existing tests to use the new utility where applicable

## File Structure

```
tests/common/
├── __init__.py          # Package exports
├── utils.py             # Main utilities implementation
├── README.md            # This documentation
└── data/               # Directory for test data files (if needed)
```

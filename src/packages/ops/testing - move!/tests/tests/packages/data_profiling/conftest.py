"""
Pytest configuration and fixtures for data profiling package tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="data_profiling_tests_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_dataframe():
    """Create a standard sample DataFrame for testing."""
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'name': [f'User_{i}' for i in range(1, 101)],
        'email': [f'user{i}@example.com' for i in range(1, 101)],
        'age': np.random.randint(18, 80, 100),
        'salary': np.random.normal(50000, 15000, 100),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 100),
        'is_active': np.random.choice([True, False], 100),
        'created_at': pd.date_range('2020-01-01', periods=100, freq='D'),
        'score': np.random.uniform(0, 100, 100)
    }
    return pd.DataFrame(data)


@pytest.fixture
def large_dataframe():
    """Create a large DataFrame for testing performance features."""
    np.random.seed(42)
    size = 10000
    data = {
        'id': range(1, size + 1),
        'value1': np.random.normal(0, 1, size),
        'value2': np.random.uniform(0, 100, size),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size),
        'timestamp': pd.date_range('2020-01-01', periods=size, freq='H')
    }
    return pd.DataFrame(data)


@pytest.fixture
def mixed_types_dataframe():
    """Create DataFrame with mixed data types for testing."""
    data = {
        'string_col': ['text1', 'text2', 'text3'] * 10,
        'int_col': list(range(30)),
        'float_col': np.random.normal(0, 1, 30),
        'bool_col': [True, False, True] * 10,
        'datetime_col': pd.date_range('2023-01-01', periods=30),
        'null_col': [None] * 30,
        'mixed_col': ['123', 'abc', '456'] * 10
    }
    return pd.DataFrame(data)


@pytest.fixture
def pattern_rich_dataframe():
    """Create DataFrame rich in patterns for testing pattern discovery."""
    data = {
        'emails': [f'user{i}@example.com' for i in range(50)],
        'phones': [f'+1-555-{1000+i:04d}' for i in range(50)],
        'urls': [f'https://site{i}.com/path' for i in range(50)],
        'dates': pd.date_range('2023-01-01', periods=50).astype(str),
        'credit_cards': [f'4532-1234-5678-{9000+i:04d}' for i in range(50)],
        'product_codes': [f'PROD-{i:06d}' for i in range(50)]
    }
    return pd.DataFrame(data)


@pytest.fixture
def quality_issues_dataframe():
    """Create DataFrame with various quality issues for testing."""
    data = {
        'incomplete_col': [1, 2, None, 4, None, 6, 7, None, 9, 10],
        'inconsistent_col': ['A', 'B', 'a', 'B', 'A', 'b', 'A', 'B', 'a', 'B'],
        'outlier_col': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10],  # 100 is outlier
        'duplicate_col': [1, 2, 3, 2, 1, 3, 4, 5, 4, 5],
        'format_col': ['2023-01-01', '2023/01/02', '01-03-2023', '2023-01-04', 'invalid_date'] * 2
    }
    return pd.DataFrame(data)


@pytest.fixture
def time_series_dataframe():
    """Create time series DataFrame for testing temporal analysis."""
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    np.random.seed(42)
    
    # Create trending data with seasonality and noise
    trend = np.linspace(100, 200, 365)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # Quarterly seasonality
    noise = np.random.normal(0, 5, 365)
    values = trend + seasonal + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values,
        'category': np.random.choice(['X', 'Y', 'Z'], 365)
    })


@pytest.fixture
def numerical_dataframe():
    """Create DataFrame with various numerical distributions for testing."""
    np.random.seed(42)
    size = 1000
    
    data = {
        'normal_dist': np.random.normal(100, 15, size),
        'uniform_dist': np.random.uniform(0, 100, size),
        'exponential_dist': np.random.exponential(2, size),
        'skewed_data': np.random.lognormal(0, 1, size),
        'constant_col': [42.0] * size,
        'high_variance': np.random.normal(0, 100, size),
        'low_variance': np.random.normal(50, 1, size)
    }
    return pd.DataFrame(data)


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests
        if "performance" in item.nodeid or "test_performance" in item.name:
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if "large_dataframe" in item.fixturenames or "slow" in item.name:
            item.add_marker(pytest.mark.slow)


# Skip markers for missing dependencies
def pytest_runtest_setup(item):
    """Skip tests based on missing dependencies."""
    # Skip tests requiring specific libraries if not available
    try:
        import sklearn  # noqa
    except ImportError:
        if "sklearn" in getattr(item, "fixturenames", []) or "isolation_forest" in item.name.lower():
            pytest.skip("scikit-learn not available")
    
    try:
        import statsmodels  # noqa
    except ImportError:
        if "statsmodels" in getattr(item, "fixturenames", []) or "stationarity" in item.name.lower():
            pytest.skip("statsmodels not available")
    
    try:
        import fastavro  # noqa
    except ImportError:
        if "avro" in item.name.lower():
            pytest.skip("fastavro not available")
    
    try:
        import pyorc  # noqa
    except ImportError:
        if "orc" in item.name.lower():
            pytest.skip("pyorc not available")
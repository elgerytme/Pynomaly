"""
Pytest configuration for data quality package testing.
Provides fixtures for quality assessment, data cleansing, and validation testing.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import time


@pytest.fixture
def sample_quality_data() -> pd.DataFrame:
    """Sample dataset for quality testing."""
    np.random.seed(42)
    
    data = {
        'numeric_clean': np.random.randn(1000),
        'numeric_with_nulls': np.random.randn(1000),
        'categorical_clean': np.random.choice(['A', 'B', 'C'], 1000),
        'categorical_with_nulls': np.random.choice(['X', 'Y', 'Z'], 1000),
        'boolean_col': np.random.choice([True, False], 1000),
        'datetime_col': pd.date_range('2024-01-01', periods=1000, freq='H')
    }
    
    df = pd.DataFrame(data)
    
    # Introduce nulls
    null_indices = np.random.choice(1000, 50, replace=False)
    df.loc[null_indices, 'numeric_with_nulls'] = np.nan
    
    null_indices_cat = np.random.choice(1000, 30, replace=False)  
    df.loc[null_indices_cat, 'categorical_with_nulls'] = np.nan
    
    return df


@pytest.fixture
def problematic_data() -> pd.DataFrame:
    """Dataset with various quality issues for testing."""
    np.random.seed(42)
    
    data = {
        'high_nulls': [np.nan] * 200 + list(range(800)),
        'duplicated_values': [1, 2, 3] * 333 + [4],
        'outliers': np.random.randn(1000),
        'inconsistent_format': ['2024-01-01', '01/02/2024', '2024.03.01'] * 333 + ['2024-04-01']
    }
    
    df = pd.DataFrame(data)
    
    # Add extreme outliers
    df.loc[990:995, 'outliers'] = 100  # Extreme values
    df.loc[995:999, 'outliers'] = -100
    
    return df


@pytest.fixture
def quality_rules() -> List[Dict[str, Any]]:
    """Standard quality validation rules."""
    return [
        {
            'id': 'completeness_check',
            'type': 'completeness',
            'threshold': 0.95,
            'description': 'Check for minimum data completeness'
        },
        {
            'id': 'range_validation', 
            'type': 'validity',
            'column': 'numeric_clean',
            'min_value': -5,
            'max_value': 5,
            'description': 'Validate numeric values are within expected range'
        },
        {
            'id': 'uniqueness_check',
            'type': 'uniqueness',
            'column': 'categorical_clean',
            'min_unique_ratio': 0.01,
            'description': 'Check minimum uniqueness in categorical data'
        }
    ]


@pytest.fixture
def cleansing_config() -> Dict[str, Any]:
    """Standard data cleansing configuration."""
    return {
        'remove_nulls': True,
        'remove_duplicates': True,
        'remove_outliers': True,
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'standardize_formats': True,
        'fill_missing_strategy': 'median'
    }


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.perf_counter()
            
        def stop(self):
            self.end_time = time.perf_counter()
            
        @property
        def elapsed(self) -> float:
            if self.start_time is None or self.end_time is None:
                return 0.0
            return self.end_time - self.start_time
    
    return Timer()


@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Large dataset for performance testing."""
    np.random.seed(42)
    n_rows = 10000
    
    return pd.DataFrame({
        'col_1': np.random.randn(n_rows),
        'col_2': np.random.exponential(2, n_rows),
        'col_3': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'col_4': np.random.uniform(0, 100, n_rows),
        'col_5': pd.date_range('2024-01-01', periods=n_rows, freq='min')
    })


def pytest_configure(config):
    """Configure pytest markers for quality testing."""
    markers = [
        "quality: Data quality assessment tests",
        "cleansing: Data cleansing tests",
        "validation: Data validation tests", 
        "performance: Quality performance tests",
        "edge_cases: Edge case handling tests"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)
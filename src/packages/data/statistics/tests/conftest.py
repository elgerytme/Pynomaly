"""
Pytest configuration for Statistics package testing.
Provides fixtures for statistical analysis, distributions, and hypothesis testing.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
from scipy import stats
from sklearn.datasets import make_regression, make_classification


@pytest.fixture
def sample_normal_data() -> np.ndarray:
    """Generate sample normal distribution data."""
    np.random.seed(42)
    return np.random.normal(100, 15, 1000)


@pytest.fixture
def sample_skewed_data() -> np.ndarray:
    """Generate sample skewed distribution data."""
    np.random.seed(42)
    return np.random.exponential(2, 1000)


@pytest.fixture
def sample_multivariate_data() -> pd.DataFrame:
    """Generate sample multivariate data for testing."""
    np.random.seed(42)
    
    # Generate correlated variables
    mean = [50, 100, 25]
    cov = [[10, 5, 2],
           [5, 20, 8],
           [2, 8, 15]]
    
    data = np.random.multivariate_normal(mean, cov, 1000)
    
    df = pd.DataFrame(data, columns=['var_A', 'var_B', 'var_C'])
    df['category'] = np.random.choice(['group1', 'group2', 'group3'], 1000)
    df['timestamp'] = pd.date_range('2024-01-01', periods=1000, freq='h')
    
    return df


@pytest.fixture
def regression_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Generate regression dataset for testing."""
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        noise=0.1,
        random_state=42
    )
    return X, y


@pytest.fixture
def classification_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Generate classification dataset for testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    return X, y


@pytest.fixture
def time_series_data() -> pd.DataFrame:
    """Generate time series data with trend and seasonality."""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    n_days = len(dates)
    
    # Generate trend component
    trend = np.linspace(100, 150, n_days)
    
    # Generate seasonal component (yearly)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
    
    # Generate noise
    noise = np.random.normal(0, 5, n_days)
    
    # Combine components
    values = trend + seasonal + noise
    
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'trend': trend,
        'seasonal': seasonal,
        'noise': noise
    })
    
    return df


@pytest.fixture
def distribution_test_cases() -> List[Dict[str, Any]]:
    """Test cases for distribution fitting and testing."""
    np.random.seed(42)
    
    return [
        {
            'name': 'normal_distribution',
            'data': np.random.normal(50, 10, 1000),
            'expected_distribution': 'norm',
            'expected_params': (50, 10)
        },
        {
            'name': 'exponential_distribution', 
            'data': np.random.exponential(2, 1000),
            'expected_distribution': 'expon',
            'expected_params': (0, 2)
        },
        {
            'name': 'uniform_distribution',
            'data': np.random.uniform(10, 90, 1000),
            'expected_distribution': 'uniform',
            'expected_params': (10, 80)
        },
        {
            'name': 'gamma_distribution',
            'data': np.random.gamma(2, 2, 1000),
            'expected_distribution': 'gamma',
            'expected_params': (2, 0, 2)
        }
    ]


@pytest.fixture
def hypothesis_test_scenarios() -> List[Dict[str, Any]]:
    """Hypothesis testing scenarios."""
    np.random.seed(42)
    
    return [
        {
            'test_name': 'one_sample_t_test',
            'data': np.random.normal(100, 15, 50),
            'null_hypothesis': 'mean equals 100',
            'expected_p_value_range': (0.05, 1.0),  # Should not reject null
            'test_statistic_expected': 'close_to_zero'
        },
        {
            'test_name': 'two_sample_t_test',
            'data1': np.random.normal(100, 15, 50),
            'data2': np.random.normal(105, 15, 50),
            'null_hypothesis': 'means are equal',
            'expected_p_value_range': (0.01, 0.5),  # Might reject null
            'test_statistic_expected': 'negative'
        },
        {
            'test_name': 'chi_square_goodness_of_fit',
            'observed': np.random.poisson(5, 100),
            'expected_distribution': 'poisson',
            'null_hypothesis': 'follows poisson distribution',
            'expected_p_value_range': (0.05, 1.0)
        },
        {
            'test_name': 'anova_test',
            'groups': [
                np.random.normal(100, 10, 30),
                np.random.normal(102, 10, 30),
                np.random.normal(98, 10, 30)
            ],
            'null_hypothesis': 'all group means are equal',
            'expected_p_value_range': (0.1, 1.0)
        }
    ]


@pytest.fixture
def correlation_test_data() -> Dict[str, np.ndarray]:
    """Generate data with known correlation patterns."""
    np.random.seed(42)
    n = 1000
    
    # Strong positive correlation
    x1 = np.random.normal(0, 1, n)
    y1 = 2 * x1 + np.random.normal(0, 0.5, n)
    
    # Strong negative correlation
    x2 = np.random.normal(0, 1, n)
    y2 = -1.5 * x2 + np.random.normal(0, 0.3, n)
    
    # No correlation
    x3 = np.random.normal(0, 1, n)
    y3 = np.random.normal(0, 1, n)
    
    # Non-linear relationship
    x4 = np.random.uniform(-3, 3, n)
    y4 = x4**2 + np.random.normal(0, 1, n)
    
    return {
        'strong_positive': (x1, y1),
        'strong_negative': (x2, y2),
        'no_correlation': (x3, y3),
        'non_linear': (x4, y4)
    }


@pytest.fixture
def outlier_detection_data() -> Dict[str, np.ndarray]:
    """Generate datasets with known outliers."""
    np.random.seed(42)
    
    # Normal data with outliers
    normal_data = np.random.normal(50, 10, 950)
    outliers = np.array([100, 120, -20, 0, 150])  # 5 outliers
    data_with_outliers = np.concatenate([normal_data, outliers])
    
    # Multivariate data with outliers
    normal_mv = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 980)
    outlier_mv = np.array([[5, 5], [-4, 6], [3, -5], [-6, -4], [7, -2]])  # 5 outliers
    mv_data_with_outliers = np.vstack([normal_mv, outlier_mv])
    
    return {
        'univariate_with_outliers': data_with_outliers,
        'univariate_clean': normal_data,
        'multivariate_with_outliers': mv_data_with_outliers,
        'multivariate_clean': normal_mv,
        'known_outlier_indices': list(range(950, 955))  # Last 5 indices
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
def large_statistical_dataset() -> pd.DataFrame:
    """Generate large dataset for performance testing."""
    np.random.seed(42)
    n_samples = 50000
    
    # Generate multiple variables with different distributions
    data = {
        'normal_var': np.random.normal(100, 15, n_samples),
        'exponential_var': np.random.exponential(2, n_samples),
        'uniform_var': np.random.uniform(0, 100, n_samples),
        'gamma_var': np.random.gamma(2, 2, n_samples),
        'binary_var': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'categorical_var': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'ordinal_var': np.random.choice(range(1, 6), n_samples)
    }
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.date_range('2023-01-01', periods=n_samples, freq='min')
    
    # Add some correlated variables
    df['correlated_var'] = 0.7 * df['normal_var'] + np.random.normal(0, 5, n_samples)
    df['interaction_var'] = df['normal_var'] * df['uniform_var'] / 100
    
    return df


def pytest_configure(config):
    """Configure pytest markers for statistics testing."""
    markers = [
        "statistics: Statistical analysis tests",
        "distributions: Distribution fitting and testing",
        "hypothesis_testing: Statistical hypothesis tests",
        "correlation: Correlation and association tests", 
        "outliers: Outlier detection tests",
        "regression: Regression analysis tests",
        "performance: Statistics performance tests"
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)

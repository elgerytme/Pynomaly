import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from typing import Any, Dict
from unittest.mock import Mock
from uuid import uuid4
from datetime import datetime

from data_science.domain.entities.statistical_analysis import (
    StatisticalAnalysis, StatisticalAnalysisId, DatasetId, UserId,
    AnalysisType, StatisticalTest, StatisticalMetrics
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for statistical testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric_col': np.random.normal(100, 15, 1000),
        'categorical_col': np.random.choice(['A', 'B', 'C'], 1000),
        'binary_col': np.random.choice([0, 1], 1000),
        'target_col': np.random.normal(50, 10, 1000),
        'outlier_col': np.concatenate([
            np.random.normal(10, 2, 950),
            np.random.normal(100, 5, 50)  # Outliers
        ]),
        'datetime_col': pd.date_range('2023-01-01', periods=1000, freq='H')
    })


@pytest.fixture
def time_series_dataframe():
    """Create a time series DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    np.random.seed(42)
    trend = np.linspace(100, 120, 365)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 30)
    noise = np.random.normal(0, 2, 365)
    
    return pd.DataFrame({
        'date': dates,
        'value': trend + seasonal + noise,
        'category': np.random.choice(['X', 'Y', 'Z'], 365)
    })


@pytest.fixture
def sample_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def statistical_analysis_id():
    """Create a statistical analysis ID."""
    return StatisticalAnalysisId(value=uuid4())


@pytest.fixture
def dataset_id():
    """Create a dataset ID."""
    return DatasetId(value=uuid4())


@pytest.fixture
def user_id():
    """Create a user ID."""
    return UserId(value=uuid4())


@pytest.fixture
def descriptive_analysis_type():
    """Create a descriptive analysis type."""
    return AnalysisType(
        name="descriptive_statistics",
        description="Basic descriptive statistical analysis",
        requires_target=False
    )


@pytest.fixture
def hypothesis_analysis_type():
    """Create a hypothesis testing analysis type."""
    return AnalysisType(
        name="hypothesis_testing",
        description="Statistical hypothesis testing",
        requires_target=True
    )


@pytest.fixture
def sample_statistical_test():
    """Create a sample statistical test result."""
    return StatisticalTest(
        test_name="t_test",
        statistic=2.45,
        p_value=0.014,
        critical_value=1.96,
        confidence_level=0.95,
        interpretation="Reject null hypothesis at 95% confidence level"
    )


@pytest.fixture
def sample_statistical_metrics():
    """Create sample statistical metrics."""
    return StatisticalMetrics(
        descriptive_stats={
            "mean": 100.0,
            "std": 15.0,
            "min": 50.0,
            "max": 150.0,
            "median": 98.5,
            "skewness": 0.1,
            "kurtosis": -0.2
        },
        correlation_matrix={
            "numeric_col": {"target_col": 0.75, "outlier_col": 0.15},
            "target_col": {"numeric_col": 0.75, "outlier_col": 0.10}
        },
        distribution_params={
            "distribution_type": "normal",
            "parameters": {"mu": 100.0, "sigma": 15.0}
        },
        outlier_scores=[0.1, 0.05, 0.95, 0.02, 0.98]
    )


@pytest.fixture
def sample_statistical_analysis(
    statistical_analysis_id,
    dataset_id, 
    user_id,
    descriptive_analysis_type
):
    """Create a sample statistical analysis."""
    return StatisticalAnalysis(
        analysis_id=statistical_analysis_id,
        dataset_id=dataset_id,
        user_id=user_id,
        analysis_type=descriptive_analysis_type,
        status="pending",
        feature_columns=["numeric_col", "categorical_col"],
        analysis_params={"confidence_level": 0.95}
    )


@pytest.fixture
def completed_statistical_analysis(
    sample_statistical_analysis,
    sample_statistical_metrics,
    sample_statistical_test
):
    """Create a completed statistical analysis."""
    analysis = sample_statistical_analysis
    analysis.complete_analysis(
        metrics=sample_statistical_metrics,
        tests=[sample_statistical_test],
        insights=["Strong correlation between numeric_col and target_col"]
    )
    return analysis


@pytest.fixture
def mock_statistical_analysis_repository():
    """Create a mock statistical analysis repository."""
    return Mock()


@pytest.fixture
def mock_dataset_repository():
    """Create a mock dataset repository."""
    return Mock()


@pytest.fixture
def mock_visualization_service():
    """Create a mock visualization service."""
    return Mock()


@pytest.fixture
def mock_statistical_engine():
    """Create a mock statistical engine."""
    return Mock()


@pytest.fixture
def analysis_config():
    """Create analysis configuration."""
    return {
        "confidence_level": 0.95,
        "outlier_detection": True,
        "correlation_threshold": 0.5,
        "hypothesis_tests": ["t_test", "chi_square"],
        "visualization": True
    }
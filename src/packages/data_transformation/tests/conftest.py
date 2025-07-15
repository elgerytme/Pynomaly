import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from typing import Any, Dict

from data_transformation.domain.value_objects.pipeline_config import (
    PipelineConfig, SourceType, CleaningStrategy, ScalingMethod, EncodingStrategy
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
        'categorical_col': ['A', 'B', 'A', 'C', 'B'],
        'mixed_col': [1, 'text', 3, 4, 'another'],
        'datetime_col': pd.date_range('2023-01-01', periods=5),
        'binary_col': [True, False, True, False, True]
    })


@pytest.fixture
def sample_csv_file(sample_dataframe):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_json_file(sample_dataframe):
    """Create a temporary JSON file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        sample_dataframe.to_json(f.name, orient='records')
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def basic_pipeline_config():
    """Create a basic pipeline configuration."""
    return PipelineConfig(
        source_type=SourceType.CSV,
        cleaning_strategy=CleaningStrategy.AUTO,
        scaling_method=ScalingMethod.STANDARD,
        encoding_strategy=EncodingStrategy.ONEHOT,
        feature_engineering=True,
        output_format='pandas'
    )


@pytest.fixture
def advanced_pipeline_config():
    """Create an advanced pipeline configuration."""
    return PipelineConfig(
        source_type=SourceType.PARQUET,
        cleaning_strategy=CleaningStrategy.STATISTICAL,
        scaling_method=ScalingMethod.ROBUST,
        encoding_strategy=EncodingStrategy.TARGET,
        feature_engineering=True,
        output_format='polars',
        parallel_processing=True,
        cache_enabled=True,
        validation_enabled=True
    )


@pytest.fixture
def mock_config_dict():
    """Create a mock configuration dictionary."""
    return {
        'source_type': 'csv',
        'cleaning_strategy': 'auto',
        'scaling_method': 'standard',
        'encoding_strategy': 'one_hot',
        'feature_engineering': True,
        'output_format': 'pandas'
    }
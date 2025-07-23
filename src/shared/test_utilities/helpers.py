"""Test helper functions and utilities."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from fastapi import Response
from httpx import Response as HTTPXResponse


def assert_response_valid(
    response: Union[Response, HTTPXResponse],
    expected_status: int = 200,
    expected_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Assert that an HTTP response is valid and contains expected data."""
    assert response.status_code == expected_status, f"Expected {expected_status}, got {response.status_code}"
    
    if hasattr(response, 'json'):
        data = response.json()
    else:
        data = json.loads(response.content)
    
    if expected_keys:
        for key in expected_keys:
            assert key in data, f"Expected key '{key}' not found in response"
    
    return data


def assert_model_trained(model: Any, required_attributes: Optional[List[str]] = None) -> None:
    """Assert that a model is properly trained."""
    if required_attributes is None:
        required_attributes = ['fit', 'predict']
    
    for attr in required_attributes:
        assert hasattr(model, attr), f"Model missing required attribute: {attr}"
    
    # Check if model has been fitted (common pattern in scikit-learn)
    if hasattr(model, 'feature_importances_'):
        assert model.feature_importances_ is not None
    elif hasattr(model, 'coef_'):
        assert model.coef_ is not None


def generate_test_data(
    n_samples: int = 100,
    n_features: int = 10,
    n_classes: int = 2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic test data for ML models."""
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    if n_classes == 2:
        y = np.random.choice([0, 1], size=n_samples)
    else:
        y = np.random.choice(range(n_classes), size=n_samples)
    
    return X, y


def generate_test_dataframe(
    n_rows: int = 100,
    columns: Optional[List[str]] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """Generate a test pandas DataFrame."""
    np.random.seed(random_state)
    
    if columns is None:
        columns = ['feature_1', 'feature_2', 'feature_3', 'target']
    
    data = {}
    for col in columns:
        if 'target' in col.lower() or 'label' in col.lower():
            data[col] = np.random.choice([0, 1], size=n_rows)
        elif 'category' in col.lower():
            data[col] = np.random.choice(['A', 'B', 'C'], size=n_rows)
        else:
            data[col] = np.random.randn(n_rows)
    
    return pd.DataFrame(data)


def create_temp_file(
    content: str = "",
    suffix: str = ".txt",
    prefix: str = "test_"
) -> Path:
    """Create a temporary file with specified content."""
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix=suffix,
        prefix=prefix,
        delete=False
    )
    temp_file.write(content)
    temp_file.close()
    return Path(temp_file.name)


def create_temp_csv(
    data: Optional[pd.DataFrame] = None,
    **kwargs
) -> Path:
    """Create a temporary CSV file."""
    if data is None:
        data = generate_test_dataframe(**kwargs)
    
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.csv',
        prefix='test_data_',
        delete=False
    )
    data.to_csv(temp_file.name, index=False)
    return Path(temp_file.name)


def mock_external_service(
    service_name: str,
    responses: Optional[Dict[str, Any]] = None
) -> Mock:
    """Create a mock for an external service."""
    if responses is None:
        responses = {
            'get': {'status': 'success', 'data': []},
            'post': {'status': 'created', 'id': 1},
            'put': {'status': 'updated'},
            'delete': {'status': 'deleted'}
        }
    
    service_mock = Mock()
    for method, response in responses.items():
        getattr(service_mock, method).return_value = response
    
    return service_mock


def patch_external_service(
    service_path: str,
    responses: Optional[Dict[str, Any]] = None
):
    """Context manager to patch an external service."""
    mock_service = mock_external_service(service_path, responses)
    return patch(service_path, mock_service)


def assert_dataframe_equal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    check_dtype: bool = True,
    check_index_type: bool = True,
    check_column_type: bool = True
) -> None:
    """Assert that two DataFrames are equal with helpful error messages."""
    try:
        pd.testing.assert_frame_equal(
            df1, df2,
            check_dtype=check_dtype,
            check_index_type=check_index_type,
            check_column_type=check_column_type
        )
    except AssertionError as e:
        print(f"DataFrames are not equal:\n{e}")
        print(f"DataFrame 1 shape: {df1.shape}")
        print(f"DataFrame 2 shape: {df2.shape}")
        print(f"DataFrame 1 columns: {list(df1.columns)}")
        print(f"DataFrame 2 columns: {list(df2.columns)}")
        raise


def assert_array_close(
    array1: np.ndarray,
    array2: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 0
) -> None:
    """Assert that two numpy arrays are close."""
    np.testing.assert_allclose(array1, array2, rtol=rtol, atol=atol)


def capture_logs(logger_name: str = ""):
    """Context manager to capture log messages for testing."""
    from io import StringIO
    import logging
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    try:
        yield log_capture
    finally:
        logger.removeHandler(handler)


def time_function(func, *args, **kwargs) -> tuple[Any, float]:
    """Time a function execution and return result and execution time."""
    import time
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time


def memory_usage(func, *args, **kwargs) -> tuple[Any, float]:
    """Measure memory usage of a function and return result and peak memory."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    result = func(*args, **kwargs)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    peak_memory = final_memory - initial_memory
    
    return result, peak_memory
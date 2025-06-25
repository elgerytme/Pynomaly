"""
Basic tests for demo functions.
"""

import pytest
import numpy as np
from src.pynomaly.demo_functions import (
    add_numbers, multiply_positive, normalize_array,
    calculate_distance, detect_anomaly_simple, clean_data, score_prediction
)


def test_add_numbers():
    """Test addition function."""
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0


def test_multiply_positive():
    """Test multiplication of positive numbers."""
    assert multiply_positive(2.0, 3.0) == 6.0
    assert multiply_positive(1.5, 2.0) == 3.0
    
    with pytest.raises(ValueError):
        multiply_positive(-1.0, 2.0)
    
    with pytest.raises(ValueError):
        multiply_positive(2.0, 0.0)


def test_normalize_array():
    """Test array normalization."""
    arr = np.array([1, 2, 3, 4, 5])
    normalized = normalize_array(arr)
    assert np.allclose(normalized, [0, 0.25, 0.5, 0.75, 1.0])
    
    # Test edge case: all same values
    same_vals = np.array([5, 5, 5])
    normalized_same = normalize_array(same_vals)
    assert np.allclose(normalized_same, [0, 0, 0])
    
    # Test empty array
    empty = np.array([])
    assert len(normalize_array(empty)) == 0


def test_calculate_distance():
    """Test distance calculation."""
    p1 = [0, 0]
    p2 = [3, 4]
    assert calculate_distance(p1, p2) == 5.0
    
    p3 = [1, 1, 1]
    p4 = [1, 1, 1]
    assert calculate_distance(p3, p4) == 0.0
    
    with pytest.raises(ValueError):
        calculate_distance([1, 2], [1, 2, 3])


def test_detect_anomaly_simple():
    """Test simple anomaly detection."""
    normal_values = [1, 2, 3, 4, 5]
    anomalies = detect_anomaly_simple(normal_values + [100])
    assert anomalies[-1] == True  # 100 should be detected as anomaly
    
    # Test empty list
    assert detect_anomaly_simple([]) == []
    
    # Test all same values
    same_vals = [5, 5, 5, 5]
    assert all(not x for x in detect_anomaly_simple(same_vals))


def test_clean_data():
    """Test data cleaning."""
    dirty_data = [1.0, None, 2.0, None, 3.0]
    clean = clean_data(dirty_data)
    assert clean == [1.0, 2.0, 3.0]
    
    # Test all None
    all_none = [None, None, None]
    assert clean_data(all_none) == []
    
    # Test no None
    no_none = [1.0, 2.0, 3.0]
    assert clean_data(no_none) == [1.0, 2.0, 3.0]


def test_score_prediction():
    """Test prediction scoring."""
    # Perfect prediction
    assert score_prediction(5.0, 5.0) == 1.0
    
    # Some error
    score = score_prediction(5.0, 6.0)
    assert 0.0 <= score <= 1.0
    
    # Large error
    score_large = score_prediction(1.0, 100.0)
    assert score_large < score
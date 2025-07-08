#!/usr/bin/env python3
"""Test script to verify mocking infrastructure."""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from tests.mocks.data_mocks import MockDatasetRegistry, MockDetectorRegistry


def test_mock_data_registry():
    """Test that mock data registry works correctly."""
    registry = MockDatasetRegistry()
    
    # Test create_dataset
    dataset = registry.create_dataset("test_dataset", size=100)
    assert dataset is not None
    assert dataset.shape[0] == 100
    print("✓ Mock dataset creation works")
    
    # Test cached datasets
    cached_dataset = registry.get_cached_dataset("small")
    assert cached_dataset is not None
    print("✓ Cached dataset retrieval works")


def test_mock_detector_registry():
    """Test that mock detector registry works correctly."""
    registry = MockDetectorRegistry()
    
    # Test create_detector
    detector = registry.create_detector("IsolationForest", contamination=0.1)
    assert detector is not None
    assert detector.algorithm_name == "IsolationForest"
    print("✓ Mock detector creation works")
    
    # Test cached detectors
    cached_detector = registry.get_cached_detector("lof")
    assert cached_detector is not None
    print("✓ Cached detector retrieval works")


def test_sklearn_mocks():
    """Test that sklearn mocks work correctly."""
    # Mock sklearn imports
    with patch('sklearn.ensemble.IsolationForest') as mock_iforest:
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([1, -1, 1, -1])
        mock_model.decision_function.return_value = np.array([0.1, -0.9, 0.2, -0.8])
        mock_iforest.return_value = mock_model
        
        # Test using the mocked model
        from sklearn.ensemble import IsolationForest
        model = IsolationForest()
        model.fit(np.random.random((100, 5)))
        predictions = model.predict(np.random.random((4, 5)))
        
        assert predictions.tolist() == [1, -1, 1, -1]
        print("✓ Sklearn mocks work correctly")


def test_file_io_mocks():
    """Test that file I/O mocks work correctly."""
    mock_data = pd.DataFrame({
        'feature_0': [1.0, 2.0, 3.0, 4.0],
        'feature_1': [0.5, 1.5, 2.5, 3.5],
        'feature_2': [0.1, 0.2, 0.3, 0.4],
        'target': [0, 0, 1, 1]
    })
    
    with patch('pandas.read_csv', return_value=mock_data):
        import pandas as pd
        result = pd.read_csv('fake_file.csv')
        
        assert result.shape == (4, 4)
        assert 'feature_0' in result.columns
        print("✓ File I/O mocks work correctly")


def test_network_mocks():
    """Test that network mocks work correctly."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'status': 'success', 'data': []}
    
    with patch('requests.get', return_value=mock_response):
        import requests
        response = requests.get('http://fake-api.com/data')
        
        assert response.status_code == 200
        assert response.json()['status'] == 'success'
        print("✓ Network mocks work correctly")


if __name__ == "__main__":
    print("Testing mocking infrastructure...")
    
    test_mock_data_registry()
    test_mock_detector_registry()
    test_sklearn_mocks()
    test_file_io_mocks()
    test_network_mocks()
    
    print("\n✓ All mocking tests passed!")

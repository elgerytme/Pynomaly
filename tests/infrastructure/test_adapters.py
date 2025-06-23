"""Comprehensive tests for infrastructure adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from pynomaly.domain.entities import Dataset, Detector
from pynomaly.domain.exceptions import InvalidAlgorithmError, AdapterError
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.infrastructure.adapters import (
    PyODAdapter, 
    SklearnAdapter, 
    PyTorchAdapter, 
    PyGODAdapter, 
    TODSAdapter
)


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    features = np.random.RandomState(42).normal(0, 1, (100, 5))
    targets = np.random.RandomState(42).choice([0, 1], size=100, p=[0.9, 0.1])
    return Dataset(name="test_dataset", features=features, targets=targets)


@pytest.fixture
def sample_detector():
    """Create a sample detector for testing."""
    return Detector(
        name="test_detector",
        algorithm="isolation_forest",
        contamination=ContaminationRate(0.1),
        hyperparameters={"n_estimators": 100, "random_state": 42}
    )


@pytest.fixture
def large_dataset():
    """Create a larger dataset for performance testing."""
    features = np.random.RandomState(42).normal(0, 1, (1000, 10))
    return Dataset(name="large_dataset", features=features)


@pytest.fixture
def anomalous_dataset():
    """Create a dataset with clear anomalies."""
    # Normal data
    normal = np.random.RandomState(42).normal(0, 1, (95, 3))
    # Clear anomalies
    anomalies = np.random.RandomState(42).normal(5, 0.5, (5, 3))
    features = np.vstack([normal, anomalies])
    targets = np.array([0] * 95 + [1] * 5)
    return Dataset(name="anomalous_dataset", features=features, targets=targets)


class TestPyODAdapter:
    """Test PyODAdapter."""
    
    def test_list_algorithms(self):
        """Test listing available algorithms."""
        # Test the class-level algorithm mapping
        algorithms = list(PyODAdapter.ALGORITHM_MAPPING.keys())
        
        assert len(algorithms) > 0
        assert "IsolationForest" in algorithms
        assert "LOF" in algorithms
        assert "OCSVM" in algorithms
    
    def test_create_model(self):
        """Test creating adapter instance."""
        adapter = PyODAdapter("IsolationForest")
        
        # Test that adapter is properly initialized
        assert adapter.algorithm_name == "IsolationForest"
        assert adapter.name == "PyOD_IsolationForest"
        assert hasattr(adapter, "fit")
        assert hasattr(adapter, "detect")
    
    def test_fit_predict(self):
        """Test fitting and predicting."""
        adapter = PyODAdapter("IsolationForest")
        
        # Generate data
        X = pd.DataFrame(
            np.random.randn(100, 3),
            columns=["a", "b", "c"]
        )
        dataset = Dataset(name="test", data=X)
        
        # Test fitting
        adapter.fit(dataset)
        assert adapter.is_fitted
        
        # Test detection
        result = adapter.detect(dataset)
        assert len(result.scores) == 100
        assert len(result.labels) == 100
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert all(label in [0, 1] for label in result.labels)
        assert len(result.anomalies) == sum(result.labels)  # Anomalies match labels
    
    def test_score_function(self):
        """Test getting anomaly scores."""
        adapter = PyODAdapter("IsolationForest")
        
        # Generate data
        X = np.random.rand(100, 3)
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        dataset = Dataset(name="test", data=df)
        
        # Fit and get scores
        adapter.fit(dataset)
        scores = adapter.score(dataset)
        
        assert len(scores) == len(df)
        assert all(isinstance(s, AnomalyScore) for s in scores)
    
    def test_unsupported_algorithm(self):
        """Test creating unsupported algorithm."""
        with pytest.raises(InvalidAlgorithmError):
            PyODAdapter("UnsupportedAlgorithm")


class TestSklearnAdapter:
    """Test SklearnAdapter."""
    
    def test_list_algorithms(self):
        """Test listing available algorithms."""
        # Test the class-level algorithm mapping
        algorithms = list(SklearnAdapter.ALGORITHM_MAPPING.keys())
        
        assert len(algorithms) > 0
        assert "OneClassSVM" in algorithms
        assert "IsolationForest" in algorithms
        assert "EllipticEnvelope" in algorithms
        assert "LocalOutlierFactor" in algorithms
    
    def test_create_model(self):
        """Test creating adapter instance."""
        adapter = SklearnAdapter("IsolationForest")
        
        # Test that adapter is properly initialized
        assert adapter.algorithm_name == "IsolationForest"
        assert adapter.name == "Sklearn_IsolationForest"
        assert hasattr(adapter, "fit")
        assert hasattr(adapter, "detect")
    
    def test_fit_predict(self):
        """Test fitting and predicting."""
        adapter = SklearnAdapter("IsolationForest")
        
        # Generate data
        X = np.random.rand(100, 3)
        df = pd.DataFrame(X, columns=["a", "b", "c"])
        dataset = Dataset(name="test", data=df)
        
        # Test fitting
        adapter.fit(dataset)
        assert adapter.is_fitted
        
        # Test detection
        result = adapter.detect(dataset)
        assert len(result.scores) == 100
        assert len(result.labels) == 100
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert all(label in [0, 1] for label in result.labels)
        assert len(result.anomalies) == sum(result.labels)  # Anomalies match labels
    
    def test_local_outlier_factor(self):
        """Test LocalOutlierFactor specifically."""
        adapter = SklearnAdapter("LocalOutlierFactor")
        
        # Generate data with clear outliers
        X = np.random.randn(100, 2)
        X[-5:] = X[-5:] + 3  # Add clear outliers
        df = pd.DataFrame(X, columns=["feature_0", "feature_1"])
        dataset = Dataset(name="test", data=df)
        
        # Test fitting and detection
        adapter.fit(dataset)
        result = adapter.detect(dataset)
        
        assert len(result.scores) == 100
        assert len(result.labels) == 100
        assert all(isinstance(score, AnomalyScore) for score in result.scores)
        assert all(label in [0, 1] for label in result.labels)
        assert sum(result.labels) > 0  # Should detect some anomalies
        assert len(result.anomalies) == sum(result.labels)  # Anomalies match labels
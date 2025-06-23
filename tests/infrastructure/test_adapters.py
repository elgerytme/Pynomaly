"""Tests for infrastructure adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pynomaly.infrastructure.adapters import PyODAdapter, SklearnAdapter


class TestPyODAdapter:
    """Test PyODAdapter."""
    
    def test_list_algorithms(self):
        """Test listing available algorithms."""
        adapter = PyODAdapter()
        algorithms = adapter.list_algorithms()
        
        assert len(algorithms) > 0
        assert "IsolationForest" in algorithms
        assert "LOF" in algorithms
        assert "OCSVM" in algorithms
    
    def test_create_model(self):
        """Test creating models."""
        adapter = PyODAdapter()
        
        # Test IsolationForest
        model = adapter.create_model("IsolationForest", {"contamination": 0.1})
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
    
    def test_fit_predict(self):
        """Test fitting and predicting."""
        adapter = PyODAdapter()
        
        # Generate data
        X = pd.DataFrame(
            np.random.randn(100, 3),
            columns=["a", "b", "c"]
        )
        
        # Create and fit model
        model = adapter.create_model("IsolationForest", {"contamination": 0.1})
        adapter.fit(model, X)
        
        # Predict
        predictions = adapter.predict(model, X)
        
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)
        assert sum(predictions) > 0  # Should have some anomalies
    
    def test_decision_function(self):
        """Test getting anomaly scores."""
        adapter = PyODAdapter()
        
        # Generate data
        X = pd.DataFrame(
            np.random.randn(100, 3),
            columns=["a", "b", "c"]
        )
        
        # Create and fit model
        model = adapter.create_model("LOF", {"contamination": 0.05})
        adapter.fit(model, X)
        
        # Get scores
        scores = adapter.decision_function(model, X)
        
        assert len(scores) == len(X)
        assert all(isinstance(s, (int, float)) for s in scores)
    
    def test_unsupported_algorithm(self):
        """Test creating unsupported algorithm."""
        adapter = PyODAdapter()
        
        with pytest.raises(ValueError, match="not supported"):
            adapter.create_model("UnsupportedAlgorithm", {})


class TestSklearnAdapter:
    """Test SklearnAdapter."""
    
    def test_list_algorithms(self):
        """Test listing available algorithms."""
        adapter = SklearnAdapter()
        algorithms = adapter.list_algorithms()
        
        assert len(algorithms) > 0
        assert "OneClassSVM" in algorithms
        assert "IsolationForest" in algorithms
        assert "EllipticEnvelope" in algorithms
        assert "LocalOutlierFactor" in algorithms
    
    def test_create_model(self):
        """Test creating models."""
        adapter = SklearnAdapter()
        
        # Test OneClassSVM
        model = adapter.create_model("OneClassSVM", {"nu": 0.1})
        assert model is not None
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
    
    def test_fit_predict(self):
        """Test fitting and predicting."""
        adapter = SklearnAdapter()
        
        # Generate data
        X = pd.DataFrame(
            np.random.randn(100, 3),
            columns=["a", "b", "c"]
        )
        
        # Create and fit model
        model = adapter.create_model("IsolationForest", {"contamination": 0.1})
        adapter.fit(model, X)
        
        # Predict
        predictions = adapter.predict(model, X)
        
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)
    
    def test_local_outlier_factor(self):
        """Test LOF which requires special handling."""
        adapter = SklearnAdapter()
        
        # Generate data
        X = pd.DataFrame(
            np.random.randn(100, 3),
            columns=["a", "b", "c"]
        )
        
        # Create and fit model
        model = adapter.create_model("LocalOutlierFactor", {"contamination": 0.1})
        adapter.fit(model, X)
        
        # LOF can only predict on training data
        predictions = adapter.predict(model, X)
        
        assert len(predictions) == len(X)
        assert sum(predictions) > 0  # Should have some anomalies
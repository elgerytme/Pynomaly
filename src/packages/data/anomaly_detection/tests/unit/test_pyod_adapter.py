"""Unit tests for PyODAdapter."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Dict

from anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter import (
    PyODAdapter,
    PyODEnsemble
)


class MockPyODModel:
    """Mock PyOD model for testing."""
    
    def __init__(self, contamination=0.1, **kwargs):
        self.contamination = contamination
        self.threshold_ = None
        self.decision_scores_ = None
        self._fitted = False
        self.parameters = kwargs
    
    def fit(self, data):
        """Mock fit method."""
        self._fitted = True
        # Mock training scores
        np.random.seed(42)
        self.decision_scores_ = np.random.rand(len(data))
        self.threshold_ = np.percentile(self.decision_scores_, (1 - self.contamination) * 100)
        return self
    
    def predict(self, data):
        """Mock predict method.""" 
        if not self._fitted:
            raise ValueError("Model not fitted")
        # Return mix of 0s and 1s
        np.random.seed(42)
        return np.random.choice([0, 1], size=len(data), p=[0.9, 0.1])
    
    def decision_function(self, data):
        """Mock decision function."""
        if not self._fitted:
            raise ValueError("Model not fitted")
        np.random.seed(42)
        return np.random.rand(len(data))
    
    def predict_proba(self, data):
        """Mock predict proba method."""
        if not self._fitted:
            raise ValueError("Model not fitted")
        np.random.seed(42)
        # Return probabilities for [normal, anomaly]
        proba_anomaly = np.random.rand(len(data)) * 0.3  # Low anomaly probabilities
        proba_normal = 1 - proba_anomaly
        return np.column_stack([proba_normal, proba_anomaly])


class TestPyODAdapter:
    """Test suite for PyODAdapter."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.randn(100, 5).astype(np.float64)
    
    @pytest.fixture
    def small_data(self):
        """Create small dataset for testing."""
        np.random.seed(42)
        return np.random.randn(10, 3).astype(np.float64)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', False)
    def test_initialization_pyod_not_available(self):
        """Test initialization when PyOD is not available."""
        with pytest.raises(ImportError) as exc_info:
            PyODAdapter()
        
        assert "PyOD is required for PyODAdapter" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_initialization_with_defaults(self):
        """Test adapter initialization with default parameters."""
        adapter = PyODAdapter()
        
        assert adapter.algorithm == "iforest"
        assert adapter.model is None
        assert adapter._fitted is False
        
        # Check default parameters were set
        assert adapter.parameters["n_estimators"] == 100
        assert adapter.parameters["contamination"] == 0.1
        assert adapter.parameters["random_state"] == 42
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_initialization_with_custom_algorithm(self):
        """Test adapter initialization with custom algorithm."""
        adapter = PyODAdapter(algorithm="lof", n_neighbors=15, contamination=0.05)
        
        assert adapter.algorithm == "lof"
        assert adapter.parameters["n_neighbors"] == 15
        assert adapter.parameters["contamination"] == 0.05
        # Default should be merged
        assert adapter.parameters["algorithm"] == "auto"
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_set_defaults_all_algorithms(self):
        """Test default parameter setting for all supported algorithms."""
        algorithms = ["iforest", "lof", "ocsvm", "pca", "knn", "hbos", "abod", "feature_bagging"]
        
        for algo in algorithms:
            adapter = PyODAdapter(algorithm=algo)
            assert "contamination" in adapter.parameters
            assert adapter.parameters["contamination"] == 0.1
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_set_defaults_unknown_algorithm(self):
        """Test default parameter setting for unknown algorithm."""
        adapter = PyODAdapter(algorithm="unknown_algo")
        # Should still initialize but without defaults
        assert adapter.algorithm == "unknown_algo"
        assert len(adapter.parameters) == 0
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', False)
    def test_fit_pyod_not_available(self, sample_data):
        """Test fit method when PyOD is not available."""
        # Create adapter bypassing init check
        adapter = PyODAdapter.__new__(PyODAdapter)
        adapter.algorithm = "iforest"
        adapter.parameters = {}
        adapter.model = None
        adapter._fitted = False
        
        with pytest.raises(ImportError) as exc_info:
            adapter.fit(sample_data)
        
        assert "PyOD is required" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.IForest')
    def test_fit_success(self, mock_iforest_class, sample_data):
        """Test successful model fitting."""
        # Setup mock
        mock_model = MockPyODModel()
        mock_iforest_class.return_value = mock_model
        
        adapter = PyODAdapter(algorithm="iforest")
        result = adapter.fit(sample_data)
        
        assert result is adapter  # Method chaining
        assert adapter._fitted is True
        assert adapter.model is mock_model
        
        # Verify model was created and fitted
        mock_iforest_class.assert_called_once()
        assert mock_model._fitted is True
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.LOF')
    def test_fit_lof_algorithm(self, mock_lof_class, sample_data):
        """Test fitting with LOF algorithm."""
        mock_model = MockPyODModel()
        mock_lof_class.return_value = mock_model
        
        adapter = PyODAdapter(algorithm="lof", n_neighbors=25)
        adapter.fit(sample_data)
        
        mock_lof_class.assert_called_once()
        call_kwargs = mock_lof_class.call_args[1]
        assert call_kwargs["n_neighbors"] == 25
        assert call_kwargs["contamination"] == 0.1  # Default
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_create_model_unknown_algorithm(self):
        """Test model creation with unknown algorithm."""
        adapter = PyODAdapter.__new__(PyODAdapter)
        adapter.algorithm = "unknown_algo"
        adapter.parameters = {}
        
        with pytest.raises(ValueError) as exc_info:
            adapter._create_model()
        
        assert "Unknown algorithm: unknown_algo" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.IForest', None)
    def test_create_model_unavailable_algorithm(self):
        """Test model creation when algorithm class is None."""
        adapter = PyODAdapter.__new__(PyODAdapter)
        adapter.algorithm = "iforest"
        adapter.parameters = {}
        
        with pytest.raises(ImportError) as exc_info:
            adapter._create_model()
        
        assert "PyOD algorithm iforest not available" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.IForest')
    def test_predict_success(self, mock_iforest_class, sample_data):
        """Test successful prediction."""
        mock_model = MockPyODModel()
        mock_iforest_class.return_value = mock_model
        
        adapter = PyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)
        
        predictions = adapter.predict(sample_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_data)
        assert all(pred in [0, 1] for pred in predictions)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_predict_not_fitted(self, sample_data):
        """Test prediction when model is not fitted."""
        adapter = PyODAdapter.__new__(PyODAdapter)
        adapter._fitted = False
        
        with pytest.raises(ValueError) as exc_info:
            adapter.predict(sample_data)
        
        assert "Model must be fitted before prediction" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.IForest')
    def test_fit_predict(self, mock_iforest_class, sample_data):
        """Test fit_predict method."""
        mock_model = MockPyODModel()
        mock_iforest_class.return_value = mock_model
        
        adapter = PyODAdapter(algorithm="iforest")
        predictions = adapter.fit_predict(sample_data)
        
        assert adapter._fitted is True
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_data)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.IForest')
    def test_decision_function(self, mock_iforest_class, sample_data):
        """Test decision function."""
        mock_model = MockPyODModel()
        mock_iforest_class.return_value = mock_model
        
        adapter = PyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)
        
        scores = adapter.decision_function(sample_data)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_data)
        assert all(isinstance(score, (int, float)) for score in scores)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_decision_function_not_fitted(self, sample_data):
        """Test decision function when model is not fitted."""
        adapter = PyODAdapter.__new__(PyODAdapter)
        adapter._fitted = False
        
        with pytest.raises(ValueError) as exc_info:
            adapter.decision_function(sample_data)
        
        assert "Model must be fitted before scoring" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.IForest')
    def test_predict_proba(self, mock_iforest_class, sample_data):
        """Test predict_proba method."""
        mock_model = MockPyODModel()
        mock_iforest_class.return_value = mock_model
        
        adapter = PyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)
        
        probas = adapter.predict_proba(sample_data)
        
        assert isinstance(probas, np.ndarray)
        assert len(probas) == len(sample_data)
        assert all(0 <= prob <= 1 for prob in probas)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_predict_proba_not_fitted(self, sample_data):
        """Test predict_proba when model is not fitted."""
        adapter = PyODAdapter.__new__(PyODAdapter)
        adapter._fitted = False
        
        with pytest.raises(ValueError) as exc_info:
            adapter.predict_proba(sample_data)
        
        assert "Model must be fitted before prediction" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_get_feature_importances_not_fitted(self):
        """Test get_feature_importances when model is not fitted."""
        adapter = PyODAdapter.__new__(PyODAdapter)
        adapter._fitted = False
        
        result = adapter.get_feature_importances()
        assert result is None
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.IForest')
    def test_get_feature_importances_fitted(self, mock_iforest_class, sample_data):
        """Test get_feature_importances when model is fitted."""
        mock_model = MockPyODModel()
        mock_iforest_class.return_value = mock_model
        
        adapter = PyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)
        
        # PyOD doesn't provide feature importances for most models
        result = adapter.get_feature_importances()
        assert result is None
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_get_parameters(self):
        """Test get_parameters method."""
        adapter = PyODAdapter(algorithm="lof", n_neighbors=15, contamination=0.05)
        
        params = adapter.get_parameters()
        
        assert isinstance(params, dict)
        assert params["n_neighbors"] == 15
        assert params["contamination"] == 0.05
        # Should be a copy, not the original
        params["n_neighbors"] = 25
        assert adapter.parameters["n_neighbors"] == 15
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.IForest')
    def test_set_parameters(self, mock_iforest_class, sample_data):
        """Test set_parameters method."""
        mock_model = MockPyODModel()
        mock_iforest_class.return_value = mock_model
        
        adapter = PyODAdapter(algorithm="iforest")
        adapter.fit(sample_data)  # Fit first
        
        assert adapter._fitted is True
        
        # Set new parameters
        result = adapter.set_parameters(n_estimators=200, contamination=0.05)
        
        assert result is adapter  # Method chaining
        assert adapter.parameters["n_estimators"] == 200
        assert adapter.parameters["contamination"] == 0.05
        # Should reset fitted state
        assert adapter._fitted is False
        assert adapter.model is None
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_get_model_info_not_fitted(self):
        """Test get_model_info when model is not fitted."""
        adapter = PyODAdapter.__new__(PyODAdapter)
        adapter._fitted = False
        
        info = adapter.get_model_info()
        
        assert info == {"fitted": False}
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.IForest')
    def test_get_model_info_fitted(self, mock_iforest_class, sample_data):
        """Test get_model_info when model is fitted."""
        mock_model = MockPyODModel(contamination=0.15)
        mock_iforest_class.return_value = mock_model
        
        adapter = PyODAdapter(algorithm="iforest", contamination=0.15)
        adapter.fit(sample_data)
        
        info = adapter.get_model_info()
        
        assert info["fitted"] is True
        assert info["algorithm"] == "iforest"
        assert info["contamination"] == 0.15
        assert "threshold" in info
        assert "training_scores_stats" in info
        assert "mean" in info["training_scores_stats"]
        assert "std" in info["training_scores_stats"]
        assert "min" in info["training_scores_stats"]
        assert "max" in info["training_scores_stats"]
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', False)
    def test_get_available_algorithms_pyod_not_available(self):
        """Test get_available_algorithms when PyOD is not available."""
        algorithms = PyODAdapter.get_available_algorithms()
        assert algorithms == []
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_get_available_algorithms(self):
        """Test get_available_algorithms when PyOD is available."""
        algorithms = PyODAdapter.get_available_algorithms()
        
        expected = ["iforest", "lof", "ocsvm", "pca", "knn", "hbos", "abod", "feature_bagging"]
        assert algorithms == expected
    
    def test_get_algorithm_info_known_algorithm(self):
        """Test get_algorithm_info for known algorithm."""
        info = PyODAdapter.get_algorithm_info("iforest")
        
        assert info["name"] == "Isolation Forest (PyOD)"
        assert info["type"] == "ensemble"
        assert info["complexity"] == "medium"
        assert info["scalability"] == "high"
        assert "parameters" in info
        assert "n_estimators" in info["parameters"]
    
    def test_get_algorithm_info_unknown_algorithm(self):
        """Test get_algorithm_info for unknown algorithm."""
        info = PyODAdapter.get_algorithm_info("unknown_algo")
        
        assert info["name"] == "Unknown Algorithm"
        assert info["description"] == "Algorithm not found"
        assert info["type"] == "unknown"
        assert info["complexity"] == "unknown"
        assert info["scalability"] == "unknown"
    
    @pytest.mark.parametrize("algorithm", [
        "iforest", "lof", "ocsvm", "pca", "knn", "hbos", "abod", "feature_bagging"
    ])
    def test_get_algorithm_info_all_algorithms(self, algorithm):
        """Test get_algorithm_info for all supported algorithms."""
        info = PyODAdapter.get_algorithm_info(algorithm)
        
        assert "name" in info
        assert "description" in info
        assert "type" in info
        assert "complexity" in info
        assert "scalability" in info
        assert "parameters" in info
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_create_ensemble(self):
        """Test create_ensemble static method."""
        ensemble = PyODAdapter.create_ensemble(
            algorithms=["iforest", "lof"],
            combination_method="average"
        )
        
        assert isinstance(ensemble, PyODEnsemble)
        assert ensemble.algorithms == ["iforest", "lof"]
        assert ensemble.combination_method == "average"
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_str_representation(self):
        """Test string representation."""
        adapter = PyODAdapter(algorithm="lof")
        
        str_repr = str(adapter)
        assert "PyODAdapter" in str_repr
        assert "algorithm='lof'" in str_repr
        assert "fitted=False" in str_repr
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_repr_representation(self):
        """Test detailed string representation."""
        adapter = PyODAdapter(algorithm="lof", n_neighbors=15)
        
        repr_str = repr(adapter)
        assert "PyODAdapter" in repr_str
        assert "algorithm='lof'" in repr_str
        assert "parameters=" in repr_str
        assert "n_neighbors" in repr_str


class TestPyODEnsemble:
    """Test suite for PyODEnsemble."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.randn(50, 4).astype(np.float64)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', False)
    def test_initialization_pyod_not_available(self):
        """Test initialization when PyOD is not available."""
        with pytest.raises(ImportError) as exc_info:
            PyODEnsemble(["iforest", "lof"])
        
        assert "PyOD is required for ensemble" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_initialization_success(self):
        """Test successful ensemble initialization."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            ensemble = PyODEnsemble(
                algorithms=["iforest", "lof"],
                combination_method="average"
            )
            
            assert ensemble.algorithms == ["iforest", "lof"]
            assert ensemble.combination_method == "average"
            assert len(ensemble.adapters) == 2
            assert ensemble._fitted is False
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_initialization_with_algorithm_params(self):
        """Test initialization with algorithm-specific parameters."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            ensemble = PyODEnsemble(
                algorithms=["iforest", "lof"],
                iforest_params={"n_estimators": 200},
                lof_params={"n_neighbors": 25}
            )
            
            # Check that adapters were created with correct parameters
            assert mock_adapter_class.call_count == 2
            
            # First call should be for iforest
            first_call = mock_adapter_class.call_args_list[0]
            assert first_call[0][0] == "iforest"
            assert first_call[1]["n_estimators"] == 200
            
            # Second call should be for lof
            second_call = mock_adapter_class.call_args_list[1]
            assert second_call[0][0] == "lof"
            assert second_call[1]["n_neighbors"] == 25
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_fit_success(self, sample_data):
        """Test successful ensemble fitting."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            # Create mock adapters
            mock_adapter1 = Mock()
            mock_adapter2 = Mock()
            mock_adapter_class.side_effect = [mock_adapter1, mock_adapter2]
            
            ensemble = PyODEnsemble(["iforest", "lof"])
            result = ensemble.fit(sample_data)
            
            assert result is ensemble  # Method chaining
            assert ensemble._fitted is True
            
            # Check that both adapters were fitted
            mock_adapter1.fit.assert_called_once_with(sample_data)
            mock_adapter2.fit.assert_called_once_with(sample_data)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_predict_not_fitted(self, sample_data):
        """Test prediction when ensemble is not fitted."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter'):
            ensemble = PyODEnsemble(["iforest", "lof"])
            
            with pytest.raises(ValueError) as exc_info:
                ensemble.predict(sample_data)
            
            assert "Ensemble must be fitted before prediction" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_predict_average_combination(self, sample_data):
        """Test prediction with average combination method."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            # Create mock adapters with different predictions
            mock_adapter1 = Mock()
            mock_adapter2 = Mock()
            mock_adapter1.predict.return_value = np.array([0, 1, 0, 1, 0])
            mock_adapter2.predict.return_value = np.array([1, 1, 0, 0, 0])
            mock_adapter_class.side_effect = [mock_adapter1, mock_adapter2]
            
            ensemble = PyODEnsemble(["iforest", "lof"], combination_method="average")
            ensemble._fitted = True
            ensemble.adapters = [mock_adapter1, mock_adapter2]
            
            predictions = ensemble.predict(np.random.randn(5, 4))
            
            # Average: [0.5, 1.0, 0.0, 0.5, 0.0] > 0.5 -> [0, 1, 0, 0, 0]
            expected = np.array([0, 1, 0, 0, 0])
            np.testing.assert_array_equal(predictions, expected)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_predict_max_combination(self, sample_data):
        """Test prediction with max combination method."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            # Create mock adapters
            mock_adapter1 = Mock()
            mock_adapter2 = Mock()
            mock_adapter1.predict.return_value = np.array([0, 1, 0, 1])
            mock_adapter2.predict.return_value = np.array([1, 0, 0, 0])
            mock_adapter_class.side_effect = [mock_adapter1, mock_adapter2]
            
            ensemble = PyODEnsemble(["iforest", "lof"], combination_method="max")
            ensemble._fitted = True
            ensemble.adapters = [mock_adapter1, mock_adapter2]
            
            predictions = ensemble.predict(np.random.randn(4, 4))
            
            # Max: [max(0,1), max(1,0), max(0,0), max(1,0)] -> [1, 1, 0, 1]
            expected = np.array([1, 1, 0, 1])
            np.testing.assert_array_equal(predictions, expected)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_predict_majority_vote_combination(self, sample_data):
        """Test prediction with majority vote (default) combination method."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            # Create mock adapters
            mock_adapter1 = Mock()
            mock_adapter2 = Mock()
            mock_adapter3 = Mock()
            mock_adapter1.predict.return_value = np.array([0, 1, 1])
            mock_adapter2.predict.return_value = np.array([1, 1, 0])
            mock_adapter3.predict.return_value = np.array([0, 0, 1])
            mock_adapter_class.side_effect = [mock_adapter1, mock_adapter2, mock_adapter3]
            
            ensemble = PyODEnsemble(["iforest", "lof", "ocsvm"], combination_method="majority")
            ensemble._fitted = True
            ensemble.adapters = [mock_adapter1, mock_adapter2, mock_adapter3]
            
            predictions = ensemble.predict(np.random.randn(3, 4))
            
            # Majority: [1>1.5?, 2>1.5?, 2>1.5?] -> [0, 1, 1]
            expected = np.array([0, 1, 1])
            np.testing.assert_array_equal(predictions, expected)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_decision_function_not_fitted(self, sample_data):
        """Test decision function when ensemble is not fitted."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter'):
            ensemble = PyODEnsemble(["iforest", "lof"])
            
            with pytest.raises(ValueError) as exc_info:
                ensemble.decision_function(sample_data)
            
            assert "Ensemble must be fitted before scoring" in str(exc_info.value)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.average')
    def test_decision_function_average_method(self, mock_average, sample_data):
        """Test decision function with average combination method."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            # Setup mocks
            mock_adapter1 = Mock()
            mock_adapter2 = Mock()
            mock_adapter1.decision_function.return_value = np.array([0.1, 0.8, 0.3])
            mock_adapter2.decision_function.return_value = np.array([0.5, 0.2, 0.9])
            mock_adapter_class.side_effect = [mock_adapter1, mock_adapter2]
            
            expected_scores = np.array([0.3, 0.5, 0.6])  # Mock average result
            mock_average.return_value = expected_scores
            
            ensemble = PyODEnsemble(["iforest", "lof"], combination_method="average")
            ensemble._fitted = True
            ensemble.adapters = [mock_adapter1, mock_adapter2]
            
            scores = ensemble.decision_function(np.random.randn(3, 4))
            
            np.testing.assert_array_equal(scores, expected_scores)
            mock_average.assert_called_once()
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.maximization')
    def test_decision_function_max_method(self, mock_maximization, sample_data):
        """Test decision function with max combination method."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            # Setup mocks
            mock_adapter1 = Mock()
            mock_adapter2 = Mock()
            mock_adapter1.decision_function.return_value = np.array([0.1, 0.8])
            mock_adapter2.decision_function.return_value = np.array([0.5, 0.2])
            mock_adapter_class.side_effect = [mock_adapter1, mock_adapter2]
            
            expected_scores = np.array([0.5, 0.8])  # Mock maximization result
            mock_maximization.return_value = expected_scores
            
            ensemble = PyODEnsemble(["iforest", "lof"], combination_method="max")
            ensemble._fitted = True
            ensemble.adapters = [mock_adapter1, mock_adapter2]
            
            scores = ensemble.decision_function(np.random.randn(2, 4))
            
            np.testing.assert_array_equal(scores, expected_scores)
            mock_maximization.assert_called_once()
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.aom')
    def test_decision_function_aom_method(self, mock_aom, sample_data):
        """Test decision function with AOM combination method."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            # Setup mocks
            mock_adapter1 = Mock()
            mock_adapter2 = Mock()
            mock_adapter1.decision_function.return_value = np.array([0.1, 0.8])
            mock_adapter2.decision_function.return_value = np.array([0.5, 0.2])
            mock_adapter_class.side_effect = [mock_adapter1, mock_adapter2]
            
            expected_scores = np.array([0.35, 0.6])  # Mock AOM result
            mock_aom.return_value = expected_scores
            
            ensemble = PyODEnsemble(["iforest", "lof"], combination_method="aom")
            ensemble._fitted = True
            ensemble.adapters = [mock_adapter1, mock_adapter2]
            
            scores = ensemble.decision_function(np.random.randn(2, 4))
            
            np.testing.assert_array_equal(scores, expected_scores)
            mock_aom.assert_called_once_with(mock.ANY, n_buckets=5)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.moa')
    def test_decision_function_moa_method(self, mock_moa, sample_data):
        """Test decision function with MOA combination method."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            # Setup mocks
            mock_adapter1 = Mock()
            mock_adapter2 = Mock()
            mock_adapter1.decision_function.return_value = np.array([0.1, 0.8])
            mock_adapter2.decision_function.return_value = np.array([0.5, 0.2])
            mock_adapter_class.side_effect = [mock_adapter1, mock_adapter2]
            
            expected_scores = np.array([0.25, 0.7])  # Mock MOA result
            mock_moa.return_value = expected_scores
            
            ensemble = PyODEnsemble(["iforest", "lof"], combination_method="moa")
            ensemble._fitted = True
            ensemble.adapters = [mock_adapter1, mock_adapter2]
            
            scores = ensemble.decision_function(np.random.randn(2, 4))
            
            np.testing.assert_array_equal(scores, expected_scores)
            mock_moa.assert_called_once_with(mock.ANY, n_buckets=5)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_decision_function_fallback_average(self, sample_data):
        """Test decision function fallback to simple average."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            # Setup mocks
            mock_adapter1 = Mock()
            mock_adapter2 = Mock()
            mock_adapter1.decision_function.return_value = np.array([0.2, 0.8])
            mock_adapter2.decision_function.return_value = np.array([0.6, 0.4])
            mock_adapter_class.side_effect = [mock_adapter1, mock_adapter2]
            
            ensemble = PyODEnsemble(["iforest", "lof"], combination_method="unknown_method")
            ensemble._fitted = True
            ensemble.adapters = [mock_adapter1, mock_adapter2]
            
            scores = ensemble.decision_function(np.random.randn(2, 4))
            
            # Should fallback to simple numpy average: (0.2+0.6)/2, (0.8+0.4)/2
            expected = np.array([0.4, 0.6])
            np.testing.assert_array_equal(scores, expected)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PYOD_AVAILABLE', True)
    def test_ensemble_workflow(self, sample_data):
        """Test complete ensemble workflow."""
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.pyod_adapter.PyODAdapter') as mock_adapter_class:
            # Create mock adapters
            mock_adapter1 = Mock()
            mock_adapter2 = Mock()
            mock_adapter1.predict.return_value = np.array([0, 1, 0])
            mock_adapter2.predict.return_value = np.array([1, 1, 0])
            mock_adapter1.decision_function.return_value = np.array([0.1, 0.9, 0.2])
            mock_adapter2.decision_function.return_value = np.array([0.8, 0.7, 0.1])
            mock_adapter_class.side_effect = [mock_adapter1, mock_adapter2]
            
            # Create and fit ensemble
            ensemble = PyODEnsemble(["iforest", "lof"])
            ensemble.fit(sample_data)
            
            # Test predictions
            test_data = np.random.randn(3, 4)
            predictions = ensemble.predict(test_data)
            scores = ensemble.decision_function(test_data)
            
            assert isinstance(predictions, np.ndarray)
            assert isinstance(scores, np.ndarray)
            assert len(predictions) == 3
            assert len(scores) == 3
"""Unit tests for Scikit-learn adapter."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter import SklearnAdapter


class TestSklearnAdapter:
    """Test suite for Scikit-learn algorithm adapter."""
    
    def test_adapter_initialization_iforest(self):
        """Test initialization with Isolation Forest."""
        adapter = SklearnAdapter('iforest', n_estimators=100, contamination=0.1)
        
        assert adapter.algorithm_name == 'iforest'
        assert adapter.params['n_estimators'] == 100
        assert adapter.params['contamination'] == 0.1
        assert adapter._model is None
        assert not adapter._is_fitted
    
    def test_adapter_initialization_lof(self):
        """Test initialization with Local Outlier Factor."""
        adapter = SklearnAdapter('lof', n_neighbors=20, novelty=True)
        
        assert adapter.algorithm_name == 'lof'
        assert adapter.params['n_neighbors'] == 20
        assert adapter.params['novelty'] is True
    
    def test_adapter_initialization_ocsvm(self):
        """Test initialization with One-Class SVM."""
        adapter = SklearnAdapter('ocsvm', kernel='rbf', nu=0.05)
        
        assert adapter.algorithm_name == 'ocsvm'
        assert adapter.params['kernel'] == 'rbf'
        assert adapter.params['nu'] == 0.05
    
    def test_adapter_initialization_elliptic(self):
        """Test initialization with Elliptic Envelope."""
        adapter = SklearnAdapter('elliptic', contamination=0.15)
        
        assert adapter.algorithm_name == 'elliptic'
        assert adapter.params['contamination'] == 0.15
    
    def test_adapter_initialization_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            SklearnAdapter('invalid_algo')
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter.IsolationForest')
    def test_fit_iforest(self, mock_iforest_class):
        """Test fitting Isolation Forest."""
        # Setup mock
        mock_model = Mock()
        mock_iforest_class.return_value = mock_model
        
        # Create adapter and fit
        adapter = SklearnAdapter('iforest', n_estimators=50)
        X = np.random.randn(100, 5)
        
        adapter.fit(X)
        
        # Verify
        mock_iforest_class.assert_called_once_with(n_estimators=50)
        mock_model.fit.assert_called_once()
        assert adapter._is_fitted
        
        # Check input data was passed correctly
        fit_call_args = mock_model.fit.call_args[0]
        assert np.array_equal(fit_call_args[0], X)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter.LocalOutlierFactor')
    def test_fit_lof_novelty(self, mock_lof_class):
        """Test fitting LOF with novelty detection."""
        mock_model = Mock()
        mock_lof_class.return_value = mock_model
        
        adapter = SklearnAdapter('lof', n_neighbors=10, novelty=True)
        X = np.random.randn(100, 3)
        
        adapter.fit(X)
        
        mock_lof_class.assert_called_once_with(n_neighbors=10, novelty=True)
        mock_model.fit.assert_called_once()
    
    def test_fit_empty_data(self):
        """Test fitting with empty data."""
        adapter = SklearnAdapter('iforest')
        
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            adapter.fit(np.array([]))
    
    def test_fit_invalid_shape(self):
        """Test fitting with invalid data shape."""
        adapter = SklearnAdapter('iforest')
        
        # 1D array
        with pytest.raises(ValueError, match="Expected 2D array"):
            adapter.fit(np.array([1, 2, 3, 4]))
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter.IsolationForest')
    def test_predict_fitted(self, mock_iforest_class):
        """Test prediction with fitted model."""
        # Setup mock
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 1, -1, 1])
        mock_iforest_class.return_value = mock_model
        
        # Fit and predict
        adapter = SklearnAdapter('iforest')
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(4, 5)
        
        adapter.fit(X_train)
        predictions = adapter.predict(X_test)
        
        assert np.array_equal(predictions, [1, 1, -1, 1])
        mock_model.predict.assert_called_once()
    
    def test_predict_not_fitted(self):
        """Test prediction without fitting."""
        adapter = SklearnAdapter('iforest')
        X = np.random.randn(10, 5)
        
        with pytest.raises(RuntimeError, match="Model must be fitted"):
            adapter.predict(X)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter.IsolationForest')
    def test_fit_predict(self, mock_iforest_class):
        """Test combined fit and predict."""
        mock_model = Mock()
        mock_model.fit_predict.return_value = np.array([1, 1, -1, 1, -1])
        mock_iforest_class.return_value = mock_model
        
        adapter = SklearnAdapter('iforest')
        X = np.random.randn(5, 3)
        
        predictions = adapter.fit_predict(X)
        
        assert np.array_equal(predictions, [1, 1, -1, 1, -1])
        mock_model.fit_predict.assert_called_once()
        assert adapter._is_fitted
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter.LocalOutlierFactor')
    def test_lof_without_novelty(self, mock_lof_class):
        """Test LOF without novelty (only fit_predict)."""
        mock_model = Mock()
        mock_model.fit_predict.return_value = np.array([1, -1, 1])
        mock_lof_class.return_value = mock_model
        
        adapter = SklearnAdapter('lof', novelty=False)
        X = np.random.randn(3, 2)
        
        # Should use fit_predict
        predictions = adapter.fit_predict(X)
        assert np.array_equal(predictions, [1, -1, 1])
        
        # Regular predict should fail
        with pytest.raises(RuntimeError, match="LOF with novelty=False"):
            adapter.predict(X)
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter.IsolationForest')
    def test_decision_function(self, mock_iforest_class):
        """Test getting anomaly scores."""
        mock_model = Mock()
        mock_model.decision_function.return_value = np.array([0.1, -0.5, 0.3, -0.8])
        mock_model.score_samples.return_value = np.array([0.1, -0.5, 0.3, -0.8])
        mock_iforest_class.return_value = mock_model
        
        adapter = SklearnAdapter('iforest')
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(4, 5)
        
        adapter.fit(X_train)
        scores = adapter.decision_function(X_test)
        
        # Scores should be normalized to [0, 1]
        assert len(scores) == 4
        assert np.all(scores >= 0)
        assert np.all(scores <= 1)
    
    def test_get_params(self):
        """Test getting adapter parameters."""
        params = {'n_estimators': 100, 'max_samples': 256}
        adapter = SklearnAdapter('iforest', **params)
        
        retrieved_params = adapter.get_params()
        
        assert retrieved_params['n_estimators'] == 100
        assert retrieved_params['max_samples'] == 256
        assert retrieved_params['algorithm'] == 'iforest'
    
    def test_set_params(self):
        """Test setting adapter parameters."""
        adapter = SklearnAdapter('iforest', n_estimators=50)
        
        adapter.set_params(n_estimators=200, max_samples='auto')
        
        assert adapter.params['n_estimators'] == 200
        assert adapter.params['max_samples'] == 'auto'
        
        # Model should be reset
        assert adapter._model is None
        assert not adapter._is_fitted
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter.OneClassSVM')
    def test_ocsvm_parameters(self, mock_svm_class):
        """Test One-Class SVM specific parameters."""
        mock_model = Mock()
        mock_svm_class.return_value = mock_model
        
        adapter = SklearnAdapter(
            'ocsvm',
            kernel='poly',
            degree=3,
            gamma='scale',
            nu=0.1
        )
        
        X = np.random.randn(50, 4)
        adapter.fit(X)
        
        mock_svm_class.assert_called_once_with(
            kernel='poly',
            degree=3,
            gamma='scale',
            nu=0.1
        )
    
    @patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter.EllipticEnvelope')
    def test_elliptic_envelope_parameters(self, mock_ee_class):
        """Test Elliptic Envelope specific parameters."""
        mock_model = Mock()
        mock_ee_class.return_value = mock_model
        
        adapter = SklearnAdapter(
            'elliptic',
            contamination=0.2,
            support_fraction=0.9
        )
        
        X = np.random.randn(100, 3)
        adapter.fit(X)
        
        mock_ee_class.assert_called_once_with(
            contamination=0.2,
            support_fraction=0.9
        )
    
    def test_save_load_model(self, tmp_path):
        """Test saving and loading fitted model."""
        import joblib
        
        # Create and fit a simple model
        adapter = SklearnAdapter('iforest', n_estimators=10)
        X = np.random.randn(50, 3)
        
        # Mock the fit to avoid actual sklearn dependency in tests
        adapter._model = Mock()
        adapter._model.predict = Mock(return_value=np.array([1, -1, 1]))
        adapter._is_fitted = True
        
        # Save model
        model_path = tmp_path / "model.pkl"
        adapter.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        new_adapter = SklearnAdapter('iforest')
        new_adapter.load_model(str(model_path))
        
        assert new_adapter._is_fitted
        assert new_adapter.algorithm_name == 'iforest'
    
    def test_validation_contamination_parameter(self):
        """Test validation of contamination parameter."""
        # Valid contamination
        adapter = SklearnAdapter('iforest', contamination=0.1)
        assert adapter.params['contamination'] == 0.1
        
        # Invalid contamination
        with pytest.raises(ValueError, match="Contamination must be in"):
            SklearnAdapter('iforest', contamination=0.6)
        
        with pytest.raises(ValueError, match="Contamination must be in"):
            SklearnAdapter('iforest', contamination=-0.1)
    
    def test_handle_different_input_types(self):
        """Test handling different input data types."""
        adapter = SklearnAdapter('iforest')
        
        # List input
        X_list = [[1, 2], [3, 4], [5, 6]]
        X_array = adapter._validate_input(X_list)
        assert isinstance(X_array, np.ndarray)
        assert X_array.shape == (3, 2)
        
        # Pandas DataFrame
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        X_df = adapter._validate_input(df)
        assert isinstance(X_df, np.ndarray)
        assert X_df.shape == (3, 2)
    
    def test_reproducibility_with_random_state(self):
        """Test reproducibility with random_state parameter."""
        X = np.random.randn(100, 5)
        
        # First run
        adapter1 = SklearnAdapter('iforest', random_state=42)
        predictions1 = adapter1.fit_predict(X)
        
        # Second run with same random state
        adapter2 = SklearnAdapter('iforest', random_state=42)
        predictions2 = adapter2.fit_predict(X)
        
        # Should be identical
        assert np.array_equal(predictions1, predictions2)
    
    def test_adapter_properties(self):
        """Test adapter properties and metadata."""
        adapter = SklearnAdapter('lof', n_neighbors=15)
        
        assert adapter.name == 'sklearn_lof'
        assert adapter.requires_fit is True
        assert adapter.supports_streaming is False
        
        info = adapter.get_info()
        assert info['algorithm'] == 'lof'
        assert info['library'] == 'scikit-learn'
        assert 'parameters' in info
    
    def test_error_handling_during_fit(self):
        """Test error handling during model fitting."""
        adapter = SklearnAdapter('iforest')
        
        # Simulate sklearn not installed
        with patch('anomaly_detection.infrastructure.adapters.algorithms.adapters.sklearn_adapter.IsolationForest',
                  side_effect=ImportError("sklearn not found")):
            with pytest.raises(ImportError):
                adapter.fit(np.random.randn(10, 2))
    
    def test_parameter_validation_in_algorithms(self):
        """Test algorithm-specific parameter validation."""
        # LOF specific
        with pytest.raises(ValueError, match="n_neighbors must be positive"):
            SklearnAdapter('lof', n_neighbors=-5)
        
        # OCSVM specific  
        with pytest.raises(ValueError, match="nu must be in"):
            SklearnAdapter('ocsvm', nu=1.5)
        
        # Isolation Forest specific
        with pytest.raises(ValueError, match="n_estimators must be positive"):
            SklearnAdapter('iforest', n_estimators=0)
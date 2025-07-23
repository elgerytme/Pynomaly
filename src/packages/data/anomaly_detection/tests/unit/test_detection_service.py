"""Unit tests for DetectionService."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict

from anomaly_detection.domain.services.detection_service import DetectionService, AlgorithmAdapter
from anomaly_detection.domain.entities.detection_result import DetectionResult
from anomaly_detection.infrastructure.logging.error_handler import (
    InputValidationError, 
    AlgorithmError
)


class MockAlgorithmAdapter:
    """Mock algorithm adapter for testing."""
    
    def __init__(self, fail_fit=False, fail_predict=False):
        self.fitted = False
        self.fail_fit = fail_fit
        self.fail_predict = fail_predict
        self.fit_data = None
        self.predict_data = None
    
    def fit(self, data: np.ndarray) -> None:
        """Mock fit method."""
        if self.fail_fit:
            raise ValueError("Mock fit failure")
        self.fitted = True
        self.fit_data = data
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Mock predict method."""
        if self.fail_predict:
            raise ValueError("Mock predict failure")
        if not self.fitted:
            raise ValueError("Model not fitted")
        self.predict_data = data
        # Return mock predictions (alternating pattern)
        return np.array([1 if i % 2 == 0 else -1 for i in range(len(data))])
    
    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """Mock fit_predict method."""
        self.fit(data)
        return self.predict(data)


class TestDetectionService:
    """Test suite for DetectionService."""
    
    @pytest.fixture
    def detection_service(self):
        """Create detection service instance."""
        return DetectionService()
    
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
    
    def test_initialization(self, detection_service):
        """Test detection service initialization."""
        assert detection_service._adapters == {}
        assert detection_service._fitted_models == {}
    
    def test_register_adapter(self, detection_service):
        """Test adapter registration."""
        adapter = MockAlgorithmAdapter()
        detection_service.register_adapter("mock_algo", adapter)
        
        assert "mock_algo" in detection_service._adapters
        assert detection_service._adapters["mock_algo"] == adapter
    
    def test_detect_anomalies_with_registered_adapter(self, detection_service, sample_data):
        """Test anomaly detection using registered adapter."""
        adapter = MockAlgorithmAdapter()
        detection_service.register_adapter("mock_algo", adapter)
        
        result = detection_service.detect_anomalies(
            data=sample_data,
            algorithm="mock_algo",
            contamination=0.1
        )
        
        assert isinstance(result, DetectionResult)
        assert result.algorithm == "mock_algo"
        assert len(result.predictions) == len(sample_data)
        assert result.metadata["contamination"] == 0.1
        assert result.metadata["data_shape"] == sample_data.shape
        assert adapter.fit_data is not None
        np.testing.assert_array_equal(adapter.fit_data, sample_data)
    
    @patch('anomaly_detection.domain.services.detection_service.IsolationForest')
    def test_detect_anomalies_builtin_iforest(self, mock_iforest_class, detection_service, sample_data):
        """Test anomaly detection using built-in Isolation Forest."""
        # Setup mock
        mock_model = Mock()
        mock_predictions = np.array([1, -1, 1, -1] * 25)  # 100 predictions
        mock_model.fit_predict.return_value = mock_predictions
        mock_iforest_class.return_value = mock_model
        
        result = detection_service.detect_anomalies(
            data=sample_data,
            algorithm="iforest",
            contamination=0.15
        )
        
        assert isinstance(result, DetectionResult)
        assert result.algorithm == "iforest"
        assert len(result.predictions) == len(sample_data)
        np.testing.assert_array_equal(result.predictions, mock_predictions)
        
        # Verify model was called correctly
        mock_iforest_class.assert_called_once_with(
            contamination=0.15,
            random_state=42
        )
        mock_model.fit_predict.assert_called_once()
    
    @patch('anomaly_detection.domain.services.detection_service.LocalOutlierFactor')
    def test_detect_anomalies_builtin_lof(self, mock_lof_class, detection_service, sample_data):
        """Test anomaly detection using built-in Local Outlier Factor."""
        # Setup mock
        mock_model = Mock()
        mock_predictions = np.array([-1, 1, -1, 1] * 25)  # 100 predictions
        mock_model.fit_predict.return_value = mock_predictions
        mock_lof_class.return_value = mock_model
        
        result = detection_service.detect_anomalies(
            data=sample_data,
            algorithm="lof",
            contamination=0.05
        )
        
        assert isinstance(result, DetectionResult)
        assert result.algorithm == "lof"
        np.testing.assert_array_equal(result.predictions, mock_predictions)
        
        # Verify model was called correctly
        mock_lof_class.assert_called_once_with(contamination=0.05)
        mock_model.fit_predict.assert_called_once()
    
    def test_detect_anomalies_unknown_algorithm(self, detection_service, sample_data):
        """Test detection with unknown algorithm raises error."""
        with pytest.raises(AlgorithmError) as exc_info:
            detection_service.detect_anomalies(
                data=sample_data,
                algorithm="unknown_algo",
                contamination=0.1
            )
        
        assert "Unknown algorithm: unknown_algo" in str(exc_info.value)
        assert "available_algorithms" in exc_info.value.details
    
    def test_detect_anomalies_with_kwargs(self, detection_service, sample_data):
        """Test detection with additional kwargs."""
        adapter = MockAlgorithmAdapter()
        detection_service.register_adapter("mock_algo", adapter)
        
        result = detection_service.detect_anomalies(
            data=sample_data,
            algorithm="mock_algo",
            contamination=0.1,
            n_estimators=200,
            max_features=0.8
        )
        
        assert result.metadata["algorithm_params"]["n_estimators"] == 200
        assert result.metadata["algorithm_params"]["max_features"] == 0.8
    
    def test_fit_with_registered_adapter(self, detection_service, sample_data):
        """Test fitting with registered adapter."""
        adapter = MockAlgorithmAdapter()
        detection_service.register_adapter("mock_algo", adapter)
        
        result = detection_service.fit(
            data=sample_data,
            algorithm="mock_algo"
        )
        
        assert result is detection_service  # Method chaining
        assert "mock_algo" in detection_service._fitted_models
        assert detection_service._fitted_models["mock_algo"] == adapter
        assert adapter.fitted is True
        np.testing.assert_array_equal(adapter.fit_data, sample_data)
    
    @patch('anomaly_detection.domain.services.detection_service.IsolationForest')
    def test_fit_builtin_iforest(self, mock_iforest_class, detection_service, sample_data):
        """Test fitting built-in Isolation Forest."""
        mock_model = Mock()
        mock_iforest_class.return_value = mock_model
        
        result = detection_service.fit(
            data=sample_data,
            algorithm="iforest",
            n_estimators=150
        )
        
        assert result is detection_service
        assert "iforest" in detection_service._fitted_models
        mock_iforest_class.assert_called_once_with(n_estimators=150)
        mock_model.fit.assert_called_once_with(sample_data)
    
    def test_fit_unknown_algorithm(self, detection_service, sample_data):
        """Test fitting unknown algorithm raises error."""
        with pytest.raises(AlgorithmError) as exc_info:
            detection_service.fit(
                data=sample_data,
                algorithm="unknown_algo"
            )
        
        assert "Unknown algorithm for fitting: unknown_algo" in str(exc_info.value)
    
    def test_fit_adapter_failure(self, detection_service, sample_data):
        """Test fitting when adapter fails."""
        adapter = MockAlgorithmAdapter(fail_fit=True)
        detection_service.register_adapter("failing_algo", adapter)
        
        with pytest.raises(AlgorithmError):
            detection_service.fit(
                data=sample_data,
                algorithm="failing_algo"
            )
    
    def test_predict_with_fitted_adapter(self, detection_service, sample_data):
        """Test prediction with fitted adapter."""
        adapter = MockAlgorithmAdapter()
        detection_service.register_adapter("mock_algo", adapter)
        detection_service.fit(data=sample_data, algorithm="mock_algo")
        
        # Create test data for prediction
        test_data = np.random.randn(20, 5).astype(np.float64)
        
        result = detection_service.predict(
            data=test_data,
            algorithm="mock_algo"
        )
        
        assert isinstance(result, DetectionResult)
        assert result.algorithm == "mock_algo"
        assert len(result.predictions) == len(test_data)
        np.testing.assert_array_equal(adapter.predict_data, test_data)
    
    def test_predict_builtin_model(self, detection_service, sample_data):
        """Test prediction with fitted built-in model."""
        # Mock a fitted sklearn model
        mock_model = Mock()
        mock_predictions = np.array([1, -1, 1, -1, 1])
        mock_model.predict.return_value = mock_predictions
        
        # Manually add fitted model
        detection_service._fitted_models["iforest"] = mock_model
        
        test_data = np.random.randn(5, 5).astype(np.float64)
        
        result = detection_service.predict(
            data=test_data,
            algorithm="iforest"
        )
        
        assert isinstance(result, DetectionResult)
        assert result.algorithm == "iforest"
        np.testing.assert_array_equal(result.predictions, mock_predictions)
        mock_model.predict.assert_called_once_with(test_data)
    
    def test_predict_without_fitting(self, detection_service, sample_data):
        """Test prediction without fitting raises error."""
        with pytest.raises(AlgorithmError) as exc_info:
            detection_service.predict(
                data=sample_data,
                algorithm="iforest"
            )
        
        assert "Algorithm iforest not fitted" in str(exc_info.value)
        assert "available_models" in exc_info.value.details
    
    def test_predict_empty_data(self, detection_service):
        """Test prediction with empty data raises error."""
        empty_data = np.array([]).reshape(0, 5)
        
        with pytest.raises(InputValidationError) as exc_info:
            detection_service.predict(
                data=empty_data,
                algorithm="iforest"
            )
        
        assert "Input data cannot be empty" in str(exc_info.value)
    
    def test_predict_adapter_failure(self, detection_service, sample_data):
        """Test prediction when adapter fails."""
        adapter = MockAlgorithmAdapter(fail_predict=True)
        detection_service.register_adapter("failing_algo", adapter)
        detection_service.fit(data=sample_data, algorithm="failing_algo")
        
        with pytest.raises(AlgorithmError):
            detection_service.predict(
                data=sample_data,
                algorithm="failing_algo"
            )
    
    def test_list_available_algorithms(self, detection_service):
        """Test listing available algorithms."""
        # Test with no registered adapters
        algorithms = detection_service.list_available_algorithms()
        assert "iforest" in algorithms
        assert "lof" in algorithms
        # Time series algorithms
        assert "lstm_autoencoder" in algorithms
        assert "prophet" in algorithms
        assert "statistical_ts" in algorithms
        assert "isolation_forest_ts" in algorithms
        # Graph algorithms
        assert "gcn" in algorithms
        assert "gaan" in algorithms
        assert "anomalydae" in algorithms
        assert "radar" in algorithms
        assert "dominant" in algorithms
        assert "simple_graph" in algorithms
        assert len(algorithms) == 12  # 2 builtin + 4 time series + 6 graph
        
        # Test with registered adapters
        adapter = MockAlgorithmAdapter()
        detection_service.register_adapter("custom_algo", adapter)
        
        algorithms = detection_service.list_available_algorithms()
        assert "iforest" in algorithms
        assert "lof" in algorithms
        assert "custom_algo" in algorithms
        assert len(algorithms) == 13  # 12 + 1 custom
    
    def test_get_algorithm_info_builtin(self, detection_service):
        """Test getting info for built-in algorithms."""
        info_iforest = detection_service.get_algorithm_info("iforest")
        assert info_iforest["name"] == "iforest"
        assert info_iforest["type"] == "builtin"
        assert "scikit-learn" in info_iforest["requires"]
        
        info_lof = detection_service.get_algorithm_info("lof")
        assert info_lof["name"] == "lof"
        assert info_lof["type"] == "builtin"
        assert "scikit-learn" in info_lof["requires"]
    
    def test_get_algorithm_info_registered(self, detection_service):
        """Test getting info for registered adapters."""
        adapter = MockAlgorithmAdapter()
        detection_service.register_adapter("custom_algo", adapter)
        
        info = detection_service.get_algorithm_info("custom_algo")
        assert info["name"] == "custom_algo"
        assert info["type"] == "registered_adapter"
    
    def test_get_algorithm_info_graph(self, detection_service):
        """Test getting info for graph algorithms."""
        info_gcn = detection_service.get_algorithm_info("gcn")
        assert info_gcn["name"] == "gcn"
        assert info_gcn["type"] == "graph"
        assert "pygod" in info_gcn["requires"]
        assert "torch" in info_gcn["requires"]
        
        info_simple = detection_service.get_algorithm_info("simple_graph")
        assert info_simple["name"] == "simple_graph"
        assert info_simple["type"] == "graph"
        assert info_simple["requires"] == []
    
    def test_get_algorithm_info_unknown(self, detection_service):
        """Test getting info for unknown algorithm."""
        info = detection_service.get_algorithm_info("unknown")
        assert info["name"] == "unknown"
        assert info["type"] == "unknown"
    
    def test_validate_detection_inputs_valid(self, detection_service, sample_data):
        """Test input validation with valid inputs."""
        # Should not raise any exception
        detection_service._validate_detection_inputs(
            data=sample_data,
            algorithm="iforest",
            contamination=0.1
        )
    
    def test_validate_detection_inputs_empty_data(self, detection_service):
        """Test input validation with empty data."""
        empty_data = np.array([]).reshape(0, 5)
        
        with pytest.raises(InputValidationError) as exc_info:
            detection_service._validate_detection_inputs(
                data=empty_data,
                algorithm="iforest",
                contamination=0.1
            )
        
        assert "Input data cannot be empty" in str(exc_info.value)
    
    def test_validate_detection_inputs_wrong_dimensions(self, detection_service):
        """Test input validation with wrong dimensions."""
        wrong_dim_data = np.array([1, 2, 3, 4, 5])  # 1D array
        
        with pytest.raises(InputValidationError) as exc_info:
            detection_service._validate_detection_inputs(
                data=wrong_dim_data,
                algorithm="iforest",
                contamination=0.1
            )
        
        assert "Input data must be 2-dimensional" in str(exc_info.value)
    
    def test_validate_detection_inputs_too_few_samples(self, detection_service):
        """Test input validation with too few samples."""
        few_samples = np.array([[1, 2]]).astype(np.float64)  # Only 1 sample
        
        with pytest.raises(InputValidationError) as exc_info:
            detection_service._validate_detection_inputs(
                data=few_samples,
                algorithm="iforest",
                contamination=0.1
            )
        
        assert "Need at least 2 samples" in str(exc_info.value)
    
    def test_validate_detection_inputs_invalid_algorithm(self, detection_service, sample_data):
        """Test input validation with invalid algorithm."""
        with pytest.raises(InputValidationError) as exc_info:
            detection_service._validate_detection_inputs(
                data=sample_data,
                algorithm="",
                contamination=0.1
            )
        
        assert "Algorithm name must be a non-empty string" in str(exc_info.value)
        
        with pytest.raises(InputValidationError):
            detection_service._validate_detection_inputs(
                data=sample_data,
                algorithm=None,
                contamination=0.1
            )
    
    def test_validate_detection_inputs_invalid_contamination(self, detection_service, sample_data):
        """Test input validation with invalid contamination."""
        # Too low
        with pytest.raises(InputValidationError) as exc_info:
            detection_service._validate_detection_inputs(
                data=sample_data,
                algorithm="iforest",
                contamination=0.0005
            )
        
        assert "Contamination must be between 0.001 and 0.5" in str(exc_info.value)
        
        # Too high
        with pytest.raises(InputValidationError) as exc_info:
            detection_service._validate_detection_inputs(
                data=sample_data,
                algorithm="iforest",
                contamination=0.6
            )
        
        assert "Contamination must be between 0.001 and 0.5" in str(exc_info.value)
    
    def test_validate_detection_inputs_non_finite_values(self, detection_service):
        """Test input validation with non-finite values."""
        # Data with NaN
        nan_data = np.array([[1.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
        
        with pytest.raises(InputValidationError) as exc_info:
            detection_service._validate_detection_inputs(
                data=nan_data,
                algorithm="iforest",
                contamination=0.1
            )
        
        assert "Input data contains non-finite values" in str(exc_info.value)
        
        # Data with infinity
        inf_data = np.array([[1.0, 2.0], [np.inf, 4.0], [5.0, 6.0]])
        
        with pytest.raises(InputValidationError):
            detection_service._validate_detection_inputs(
                data=inf_data,
                algorithm="iforest",
                contamination=0.1
            )
    
    @patch('anomaly_detection.domain.services.detection_service.IsolationForest')
    def test_isolation_forest_sklearn_import_error(self, mock_iforest_class, detection_service, sample_data):
        """Test handling of sklearn import error for Isolation Forest."""
        mock_iforest_class.side_effect = ImportError("No module named 'sklearn'")
        
        with pytest.raises(AlgorithmError) as exc_info:
            detection_service._isolation_forest(sample_data, 0.1)
        
        assert "scikit-learn required for IsolationForest" in str(exc_info.value)
        assert exc_info.value.details["missing_dependency"] == "scikit-learn"
    
    @patch('anomaly_detection.domain.services.detection_service.LocalOutlierFactor')
    def test_local_outlier_factor_sklearn_import_error(self, mock_lof_class, detection_service, sample_data):
        """Test handling of sklearn import error for Local Outlier Factor."""
        mock_lof_class.side_effect = ImportError("No module named 'sklearn'")
        
        with pytest.raises(AlgorithmError) as exc_info:
            detection_service._local_outlier_factor(sample_data, 0.1)
        
        assert "scikit-learn required for LocalOutlierFactor" in str(exc_info.value)
        assert exc_info.value.details["missing_dependency"] == "scikit-learn"
    
    @patch('anomaly_detection.domain.services.detection_service.IsolationForest')
    def test_isolation_forest_execution_error(self, mock_iforest_class, detection_service, sample_data):
        """Test handling of execution errors in Isolation Forest."""
        mock_model = Mock()
        mock_model.fit_predict.side_effect = ValueError("Model execution failed")
        mock_iforest_class.return_value = mock_model
        
        with pytest.raises(AlgorithmError) as exc_info:
            detection_service._isolation_forest(sample_data, 0.1)
        
        assert "Isolation Forest execution failed" in str(exc_info.value)
        assert exc_info.value.details["contamination"] == 0.1
        assert exc_info.value.details["data_shape"] == sample_data.shape
    
    def test_method_chaining_fit_predict(self, detection_service, sample_data):
        """Test method chaining: fit then predict."""
        adapter = MockAlgorithmAdapter()
        detection_service.register_adapter("mock_algo", adapter)
        
        # Test method chaining
        result = (detection_service
                 .fit(data=sample_data, algorithm="mock_algo")
                 .predict(data=sample_data[:20], algorithm="mock_algo"))
        
        assert isinstance(result, DetectionResult)
        assert result.algorithm == "mock_algo"
        assert len(result.predictions) == 20
        assert adapter.fitted is True
    
    @pytest.mark.parametrize("algorithm,contamination", [
        ("iforest", 0.05),
        ("iforest", 0.1),
        ("iforest", 0.2),
        ("lof", 0.05),
        ("lof", 0.15),
    ])
    @patch('anomaly_detection.domain.services.detection_service.IsolationForest')
    @patch('anomaly_detection.domain.services.detection_service.LocalOutlierFactor')
    def test_detect_anomalies_parameterized(self, mock_lof, mock_iforest, 
                                          detection_service, sample_data, 
                                          algorithm, contamination):
        """Test detection with various algorithm/contamination combinations."""
        # Setup mocks
        mock_predictions = np.array([1, -1] * 50)  # 100 predictions
        
        if algorithm == "iforest":
            mock_model = Mock()
            mock_model.fit_predict.return_value = mock_predictions
            mock_iforest.return_value = mock_model
        else:  # lof
            mock_model = Mock()
            mock_model.fit_predict.return_value = mock_predictions
            mock_lof.return_value = mock_model
        
        result = detection_service.detect_anomalies(
            data=sample_data,
            algorithm=algorithm,
            contamination=contamination
        )
        
        assert isinstance(result, DetectionResult)
        assert result.algorithm == algorithm
        assert result.metadata["contamination"] == contamination
        assert len(result.predictions) == len(sample_data)
        np.testing.assert_array_equal(result.predictions, mock_predictions)
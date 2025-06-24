"""
Enhanced Scikit-learn Adapter Testing Suite
Comprehensive tests for scikit-learn based anomaly detection adapter.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import joblib
import pickle

from pynomaly.infrastructure.adapters.sklearn_adapter import SklearnAdapter
from pynomaly.domain.entities import Dataset, Detector, DetectionResult
from pynomaly.domain.value_objects import AnomalyScore, ContaminationRate
from pynomaly.domain.exceptions import DetectorError, AdapterError


class TestSklearnAdapter:
    """Enhanced test suite for Scikit-learn adapter functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (800, 6))
        anomalous_data = np.random.normal(3, 1, (200, 6))
        data = np.vstack([normal_data, anomalous_data])
        labels = np.hstack([np.zeros(800), np.ones(200)])
        
        return {
            "X_train": data[:600],
            "X_test": data[600:],
            "y_train": labels[:600],
            "y_test": labels[600:],
            "features": [f"feature_{i}" for i in range(6)]
        }

    @pytest.fixture
    def sklearn_adapter(self):
        """Create Scikit-learn adapter instance."""
        return SklearnAdapter(
            algorithm="IsolationForest",
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )

    @pytest.fixture
    def mock_dataset(self, sample_data):
        """Create mock dataset."""
        dataset = Mock(spec=Dataset)
        dataset.id = "test_dataset"
        dataset.data = sample_data["X_train"]
        dataset.features = sample_data["features"]
        dataset.target_column = None
        return dataset

    # Adapter Initialization Tests

    def test_sklearn_adapter_initialization_default(self):
        """Test Scikit-learn adapter initialization with default parameters."""
        adapter = SklearnAdapter()
        
        assert adapter.algorithm == "IsolationForest"
        assert adapter.contamination == "auto"
        assert adapter.random_state is None

    def test_sklearn_adapter_initialization_custom(self):
        """Test Scikit-learn adapter initialization with custom parameters."""
        adapter = SklearnAdapter(
            algorithm="LocalOutlierFactor",
            n_neighbors=25,
            contamination=0.15,
            algorithm_params={"leaf_size": 40},
            n_jobs=-1
        )
        
        assert adapter.algorithm == "LocalOutlierFactor"
        assert adapter.n_neighbors == 25
        assert adapter.contamination == 0.15
        assert adapter.algorithm_params["leaf_size"] == 40
        assert adapter.n_jobs == -1

    def test_sklearn_adapter_supported_algorithms(self):
        """Test initialization with all supported algorithms."""
        supported_algorithms = [
            "IsolationForest",
            "LocalOutlierFactor", 
            "OneClassSVM",
            "EllipticEnvelope",
            "SGDOneClassSVM"
        ]
        
        for algorithm in supported_algorithms:
            adapter = SklearnAdapter(algorithm=algorithm)
            assert adapter.algorithm == algorithm

    def test_sklearn_adapter_invalid_algorithm(self):
        """Test adapter initialization with invalid algorithm."""
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            SklearnAdapter(algorithm="InvalidAlgorithm")

    def test_sklearn_adapter_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid contamination
        with pytest.raises(ValueError):
            SklearnAdapter(contamination=1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            SklearnAdapter(contamination=-0.1)  # < 0
        
        # Test invalid n_estimators for IsolationForest
        with pytest.raises(ValueError):
            SklearnAdapter(algorithm="IsolationForest", n_estimators=0)

    # IsolationForest Tests

    def test_isolation_forest_fit_basic(self, sklearn_adapter, mock_dataset):
        """Test basic IsolationForest training."""
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_model
            mock_model.decision_function.return_value = np.random.randn(600)
            mock_iso.return_value = mock_model
            
            detector = sklearn_adapter.fit(mock_dataset)
            
            assert isinstance(detector, Detector)
            assert detector.algorithm == "sklearn_isolation_forest"
            assert "model" in detector.parameters
            mock_model.fit.assert_called_once()

    def test_isolation_forest_predict(self, sklearn_adapter, sample_data):
        """Test IsolationForest prediction."""
        detector = Detector(
            id="test_detector",
            algorithm="sklearn_isolation_forest",
            parameters={
                "model": "serialized_model",
                "contamination": 0.1,
                "threshold": -0.1
            }
        )
        
        with patch.object(sklearn_adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.decision_function.return_value = np.random.randn(400)
            mock_model.predict.return_value = np.random.choice([-1, 1], 400)
            mock_load.return_value = mock_model
            
            result = sklearn_adapter.predict(detector, sample_data["X_test"])
            
            assert isinstance(result, DetectionResult)
            assert len(result.anomaly_scores) == len(sample_data["X_test"])
            assert all(isinstance(score, AnomalyScore) for score in result.anomaly_scores)

    def test_isolation_forest_contamination_auto(self, mock_dataset):
        """Test IsolationForest with automatic contamination detection."""
        adapter = SklearnAdapter(algorithm="IsolationForest", contamination="auto")
        
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_model
            mock_iso.return_value = mock_model
            
            detector = adapter.fit(mock_dataset)
            
            # Should pass 'auto' to IsolationForest
            mock_iso.assert_called_with(
                contamination="auto",
                random_state=None,
                n_estimators=100
            )

    def test_isolation_forest_feature_importance(self, sklearn_adapter, sample_data):
        """Test feature importance calculation for IsolationForest."""
        detector = Detector(
            id="test_detector",
            algorithm="sklearn_isolation_forest",
            parameters={"model": "serialized_model"}
        )
        
        with patch.object(sklearn_adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            # Mock feature importances (not directly available in sklearn IF)
            mock_model.estimators_ = [MagicMock() for _ in range(5)]
            for estimator in mock_model.estimators_:
                estimator.tree_.feature = np.random.randint(0, 6, 20)
            
            mock_load.return_value = mock_model
            
            importance = sklearn_adapter.get_feature_importance(detector, sample_data["X_test"][:10])
            
            assert importance is not None
            assert len(importance) == sample_data["X_test"].shape[1]

    # LocalOutlierFactor Tests

    def test_local_outlier_factor_fit(self, mock_dataset):
        """Test LocalOutlierFactor training."""
        adapter = SklearnAdapter(algorithm="LocalOutlierFactor", n_neighbors=20)
        
        with patch('sklearn.neighbors.LocalOutlierFactor') as mock_lof:
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_model
            mock_model.negative_outlier_factor_ = -np.random.random(600) - 1
            mock_lof.return_value = mock_model
            
            detector = adapter.fit(mock_dataset)
            
            assert detector.algorithm == "sklearn_local_outlier_factor"
            mock_lof.assert_called_with(
                n_neighbors=20,
                contamination="auto"
            )

    def test_local_outlier_factor_predict(self, sample_data):
        """Test LocalOutlierFactor prediction."""
        adapter = SklearnAdapter(algorithm="LocalOutlierFactor")
        detector = Detector(
            id="test_detector",
            algorithm="sklearn_local_outlier_factor",
            parameters={"model": "serialized_model"}
        )
        
        with patch.object(adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.random.choice([-1, 1], 400)
            mock_model.negative_outlier_factor_ = -np.random.random(400) - 1
            mock_load.return_value = mock_model
            
            result = adapter.predict(detector, sample_data["X_test"])
            
            assert isinstance(result, DetectionResult)
            assert len(result.anomaly_scores) == len(sample_data["X_test"])

    def test_local_outlier_factor_novelty_detection(self, mock_dataset, sample_data):
        """Test LocalOutlierFactor in novelty detection mode."""
        adapter = SklearnAdapter(
            algorithm="LocalOutlierFactor",
            novelty=True,
            n_neighbors=20
        )
        
        with patch('sklearn.neighbors.LocalOutlierFactor') as mock_lof:
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_model
            mock_model.decision_function.return_value = np.random.randn(400)
            mock_lof.return_value = mock_model
            
            detector = adapter.fit(mock_dataset)
            
            # Should enable novelty detection
            mock_lof.assert_called_with(
                n_neighbors=20,
                contamination="auto",
                novelty=True
            )

    # OneClassSVM Tests

    def test_one_class_svm_fit(self, mock_dataset):
        """Test OneClassSVM training."""
        adapter = SklearnAdapter(
            algorithm="OneClassSVM",
            nu=0.05,
            kernel="rbf",
            gamma="scale"
        )
        
        with patch('sklearn.svm.OneClassSVM') as mock_svm:
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_model
            mock_svm.return_value = mock_model
            
            detector = adapter.fit(mock_dataset)
            
            assert detector.algorithm == "sklearn_one_class_svm"
            mock_svm.assert_called_with(
                nu=0.05,
                kernel="rbf",
                gamma="scale"
            )

    def test_one_class_svm_predict(self, sample_data):
        """Test OneClassSVM prediction."""
        adapter = SklearnAdapter(algorithm="OneClassSVM")
        detector = Detector(
            id="test_detector",
            algorithm="sklearn_one_class_svm",
            parameters={"model": "serialized_model"}
        )
        
        with patch.object(adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.decision_function.return_value = np.random.randn(400)
            mock_model.predict.return_value = np.random.choice([-1, 1], 400)
            mock_load.return_value = mock_model
            
            result = adapter.predict(detector, sample_data["X_test"])
            
            assert isinstance(result, DetectionResult)
            assert len(result.anomaly_scores) == len(sample_data["X_test"])

    def test_one_class_svm_kernel_variants(self, mock_dataset):
        """Test OneClassSVM with different kernels."""
        kernels = ["linear", "poly", "rbf", "sigmoid"]
        
        for kernel in kernels:
            adapter = SklearnAdapter(algorithm="OneClassSVM", kernel=kernel)
            
            with patch('sklearn.svm.OneClassSVM') as mock_svm:
                mock_model = MagicMock()
                mock_svm.return_value = mock_model
                
                detector = adapter.fit(mock_dataset)
                
                assert detector is not None
                mock_svm.assert_called_with(kernel=kernel)

    # EllipticEnvelope Tests

    def test_elliptic_envelope_fit(self, mock_dataset):
        """Test EllipticEnvelope training."""
        adapter = SklearnAdapter(
            algorithm="EllipticEnvelope",
            contamination=0.1,
            support_fraction=0.8
        )
        
        with patch('sklearn.covariance.EllipticEnvelope') as mock_elliptic:
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_model
            mock_elliptic.return_value = mock_model
            
            detector = adapter.fit(mock_dataset)
            
            assert detector.algorithm == "sklearn_elliptic_envelope"
            mock_elliptic.assert_called_with(
                contamination=0.1,
                support_fraction=0.8
            )

    def test_elliptic_envelope_predict(self, sample_data):
        """Test EllipticEnvelope prediction."""
        adapter = SklearnAdapter(algorithm="EllipticEnvelope")
        detector = Detector(
            id="test_detector",
            algorithm="sklearn_elliptic_envelope",
            parameters={"model": "serialized_model"}
        )
        
        with patch.object(adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.decision_function.return_value = np.random.randn(400)
            mock_model.predict.return_value = np.random.choice([-1, 1], 400)
            mock_load.return_value = mock_model
            
            result = adapter.predict(detector, sample_data["X_test"])
            
            assert isinstance(result, DetectionResult)
            assert len(result.anomaly_scores) == len(sample_data["X_test"])

    def test_elliptic_envelope_robust_covariance(self, mock_dataset):
        """Test EllipticEnvelope with robust covariance estimation."""
        adapter = SklearnAdapter(
            algorithm="EllipticEnvelope",
            store_precision=True,
            assume_centered=False
        )
        
        with patch('sklearn.covariance.EllipticEnvelope') as mock_elliptic:
            mock_model = MagicMock()
            mock_elliptic.return_value = mock_model
            
            detector = adapter.fit(mock_dataset)
            
            mock_elliptic.assert_called_with(
                store_precision=True,
                assume_centered=False,
                contamination="auto"
            )

    # SGDOneClassSVM Tests (if available in newer sklearn versions)

    def test_sgd_one_class_svm_fit(self, mock_dataset):
        """Test SGDOneClassSVM training."""
        adapter = SklearnAdapter(
            algorithm="SGDOneClassSVM",
            nu=0.05,
            learning_rate="constant",
            eta0=0.01
        )
        
        try:
            with patch('sklearn.linear_model.SGDOneClassSVM') as mock_sgd:
                mock_model = MagicMock()
                mock_model.fit.return_value = mock_model
                mock_sgd.return_value = mock_model
                
                detector = adapter.fit(mock_dataset)
                
                assert detector.algorithm == "sklearn_sgd_one_class_svm"
                mock_sgd.assert_called_with(
                    nu=0.05,
                    learning_rate="constant",
                    eta0=0.01
                )
        except ImportError:
            pytest.skip("SGDOneClassSVM not available in this sklearn version")

    # Data Preprocessing Tests

    def test_data_preprocessing_standardization(self, sklearn_adapter, mock_dataset):
        """Test data standardization preprocessing."""
        sklearn_adapter.standardize = True
        
        with patch('sklearn.preprocessing.StandardScaler') as mock_scaler, \
             patch('sklearn.ensemble.IsolationForest') as mock_iso:
            
            mock_scaler_instance = MagicMock()
            mock_scaler_instance.fit_transform.return_value = mock_dataset.data
            mock_scaler.return_value = mock_scaler_instance
            
            mock_model = MagicMock()
            mock_iso.return_value = mock_model
            
            detector = sklearn_adapter.fit(mock_dataset)
            
            assert "scaler" in detector.parameters
            mock_scaler_instance.fit_transform.assert_called_once()

    def test_data_preprocessing_normalization(self, sklearn_adapter, mock_dataset):
        """Test data normalization preprocessing."""
        sklearn_adapter.normalize = True
        
        with patch('sklearn.preprocessing.MinMaxScaler') as mock_normalizer, \
             patch('sklearn.ensemble.IsolationForest') as mock_iso:
            
            mock_normalizer_instance = MagicMock()
            mock_normalizer_instance.fit_transform.return_value = mock_dataset.data
            mock_normalizer.return_value = mock_normalizer_instance
            
            mock_model = MagicMock()
            mock_iso.return_value = mock_model
            
            detector = sklearn_adapter.fit(mock_dataset)
            
            assert "normalizer" in detector.parameters
            mock_normalizer_instance.fit_transform.assert_called_once()

    def test_data_preprocessing_pca(self, sklearn_adapter, mock_dataset):
        """Test PCA dimensionality reduction preprocessing."""
        sklearn_adapter.use_pca = True
        sklearn_adapter.pca_components = 4
        
        with patch('sklearn.decomposition.PCA') as mock_pca, \
             patch('sklearn.ensemble.IsolationForest') as mock_iso:
            
            mock_pca_instance = MagicMock()
            mock_pca_instance.fit_transform.return_value = np.random.randn(600, 4)
            mock_pca.return_value = mock_pca_instance
            
            mock_model = MagicMock()
            mock_iso.return_value = mock_model
            
            detector = sklearn_adapter.fit(mock_dataset)
            
            assert "pca" in detector.parameters
            mock_pca.assert_called_with(n_components=4)

    # Model Persistence Tests

    def test_model_serialization_joblib(self, sklearn_adapter, mock_dataset):
        """Test model serialization using joblib."""
        sklearn_adapter.serialization_method = "joblib"
        
        with patch('sklearn.ensemble.IsolationForest') as mock_iso, \
             patch('joblib.dumps') as mock_dumps:
            
            mock_model = MagicMock()
            mock_iso.return_value = mock_model
            mock_dumps.return_value = b"serialized_model"
            
            detector = sklearn_adapter.fit(mock_dataset)
            
            assert "model" in detector.parameters
            mock_dumps.assert_called()

    def test_model_serialization_pickle(self, sklearn_adapter, mock_dataset):
        """Test model serialization using pickle."""
        sklearn_adapter.serialization_method = "pickle"
        
        with patch('sklearn.ensemble.IsolationForest') as mock_iso, \
             patch('pickle.dumps') as mock_dumps:
            
            mock_model = MagicMock()
            mock_iso.return_value = mock_model
            mock_dumps.return_value = b"serialized_model"
            
            detector = sklearn_adapter.fit(mock_dataset)
            
            assert "model" in detector.parameters
            mock_dumps.assert_called()

    def test_model_deserialization_joblib(self, sklearn_adapter, sample_data):
        """Test model deserialization using joblib."""
        detector = Detector(
            id="test_detector",
            algorithm="sklearn_isolation_forest",
            parameters={
                "model": b"serialized_model",
                "serialization_method": "joblib"
            }
        )
        
        with patch('joblib.loads') as mock_loads:
            mock_model = MagicMock()
            mock_model.decision_function.return_value = np.random.randn(400)
            mock_loads.return_value = mock_model
            
            result = sklearn_adapter.predict(detector, sample_data["X_test"])
            
            assert result is not None
            mock_loads.assert_called_with(b"serialized_model")

    # Hyperparameter Optimization Tests

    def test_hyperparameter_grid_search(self, sklearn_adapter, mock_dataset):
        """Test hyperparameter optimization with grid search."""
        sklearn_adapter.use_grid_search = True
        sklearn_adapter.param_grid = {
            "n_estimators": [50, 100],
            "contamination": [0.05, 0.1]
        }
        
        with patch('sklearn.model_selection.GridSearchCV') as mock_grid, \
             patch('sklearn.ensemble.IsolationForest') as mock_iso:
            
            mock_grid_instance = MagicMock()
            mock_grid_instance.fit.return_value = mock_grid_instance
            mock_grid_instance.best_estimator_ = MagicMock()
            mock_grid.return_value = mock_grid_instance
            
            detector = sklearn_adapter.fit(mock_dataset)
            
            assert "best_params" in detector.parameters
            mock_grid.assert_called()

    def test_hyperparameter_random_search(self, sklearn_adapter, mock_dataset):
        """Test hyperparameter optimization with random search."""
        sklearn_adapter.use_random_search = True
        sklearn_adapter.param_distributions = {
            "n_estimators": [50, 100, 150],
            "contamination": [0.05, 0.1, 0.15]
        }
        sklearn_adapter.n_iter = 10
        
        with patch('sklearn.model_selection.RandomizedSearchCV') as mock_random, \
             patch('sklearn.ensemble.IsolationForest') as mock_iso:
            
            mock_random_instance = MagicMock()
            mock_random_instance.fit.return_value = mock_random_instance
            mock_random_instance.best_estimator_ = MagicMock()
            mock_random.return_value = mock_random_instance
            
            detector = sklearn_adapter.fit(mock_dataset)
            
            assert "best_params" in detector.parameters
            mock_random.assert_called()

    # Cross-Validation Tests

    def test_cross_validation_scoring(self, sklearn_adapter, sample_data):
        """Test cross-validation performance scoring."""
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.85, 0.75, 0.9, 0.82])
            
            scores = sklearn_adapter.cross_validate(
                sample_data["X_train"],
                cv=5,
                scoring="roc_auc",
                contamination_rate=ContaminationRate(0.1)
            )
            
            assert len(scores) == 5
            assert all(0 <= score <= 1 for score in scores)
            mock_cv.assert_called()

    def test_stratified_cross_validation(self, sklearn_adapter, sample_data):
        """Test stratified cross-validation."""
        with patch('sklearn.model_selection.StratifiedKFold') as mock_skf, \
             patch('sklearn.model_selection.cross_val_score') as mock_cv:
            
            mock_skf_instance = MagicMock()
            mock_skf.return_value = mock_skf_instance
            mock_cv.return_value = np.array([0.8, 0.85, 0.75])
            
            scores = sklearn_adapter.cross_validate(
                sample_data["X_train"],
                labels=sample_data["y_train"],
                cv="stratified",
                n_splits=3
            )
            
            assert len(scores) == 3
            mock_skf.assert_called()

    # Performance Tests

    def test_batch_prediction(self, sklearn_adapter, sample_data):
        """Test batch prediction for large datasets."""
        detector = Detector(
            id="test_detector",
            algorithm="sklearn_isolation_forest",
            parameters={"model": "serialized_model"}
        )
        
        # Large dataset
        large_data = np.random.randn(10000, 6)
        
        with patch.object(sklearn_adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.decision_function.return_value = np.random.randn(10000)
            mock_load.return_value = mock_model
            
            result = sklearn_adapter.predict(detector, large_data, batch_size=1000)
            
            assert len(result.anomaly_scores) == len(large_data)

    def test_parallel_processing(self, sklearn_adapter, mock_dataset):
        """Test parallel processing with n_jobs."""
        sklearn_adapter.n_jobs = -1  # Use all cores
        
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            mock_model = MagicMock()
            mock_iso.return_value = mock_model
            
            detector = sklearn_adapter.fit(mock_dataset)
            
            # Should pass n_jobs to sklearn
            mock_iso.assert_called_with(
                contamination="auto",
                random_state=42,
                n_estimators=100,
                n_jobs=-1
            )

    def test_memory_efficient_training(self, sklearn_adapter, mock_dataset):
        """Test memory-efficient training for large datasets."""
        sklearn_adapter.memory_efficient = True
        sklearn_adapter.chunk_size = 1000
        
        # Simulate large dataset
        large_data = np.random.randn(5000, 6)
        large_dataset = Mock()
        large_dataset.data = large_data
        
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            mock_model = MagicMock()
            mock_iso.return_value = mock_model
            
            detector = sklearn_adapter.fit(large_dataset)
            
            assert detector is not None
            # Should use partial_fit or chunked training
            assert mock_model.fit.call_count >= 1

    # Error Handling Tests

    def test_fit_invalid_data_type(self, sklearn_adapter):
        """Test training with invalid data type."""
        invalid_dataset = Mock()
        invalid_dataset.data = "invalid_data_type"
        
        with pytest.raises(AdapterError):
            sklearn_adapter.fit(invalid_dataset)

    def test_fit_empty_dataset(self, sklearn_adapter):
        """Test training with empty dataset."""
        empty_dataset = Mock()
        empty_dataset.data = np.array([]).reshape(0, 6)
        
        with pytest.raises(AdapterError):
            sklearn_adapter.fit(empty_dataset)

    def test_predict_model_not_found(self, sklearn_adapter, sample_data):
        """Test prediction when model is not found."""
        detector = Detector(
            id="test_detector",
            algorithm="sklearn_isolation_forest",
            parameters={}  # No model
        )
        
        with pytest.raises(DetectorError):
            sklearn_adapter.predict(detector, sample_data["X_test"])

    def test_predict_incompatible_data_shape(self, sklearn_adapter, sample_data):
        """Test prediction with incompatible data shape."""
        detector = Detector(
            id="test_detector",
            algorithm="sklearn_isolation_forest",
            parameters={"model": "serialized_model"}
        )
        
        # Wrong number of features
        wrong_data = np.random.randn(100, 10)  # 10 features instead of 6
        
        with patch.object(sklearn_adapter, '_load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.decision_function.side_effect = ValueError("Feature mismatch")
            mock_load.return_value = mock_model
            
            with pytest.raises(AdapterError):
                sklearn_adapter.predict(detector, wrong_data)

    def test_sklearn_version_compatibility(self, sklearn_adapter):
        """Test sklearn version compatibility checking."""
        with patch('sklearn.__version__', '0.20.0'):  # Old version
            sklearn_adapter._check_sklearn_version()
            
            # Should warn about old version but not fail
            assert True  # If we get here, no exception was raised


class TestSklearnAdapterIntegration:
    """Integration tests for Scikit-learn adapter."""

    def test_complete_anomaly_detection_pipeline(self, sample_data):
        """Test complete anomaly detection pipeline."""
        adapter = SklearnAdapter(
            algorithm="IsolationForest",
            n_estimators=50,  # Faster for testing
            contamination=0.1
        )
        
        # Create dataset
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        dataset.features = sample_data["features"]
        
        with patch('sklearn.ensemble.IsolationForest') as mock_iso:
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_model
            mock_model.decision_function.return_value = np.random.randn(400)
            mock_model.predict.return_value = np.random.choice([-1, 1], 400)
            mock_iso.return_value = mock_model
            
            # Train detector
            detector = adapter.fit(dataset)
            assert detector is not None
            
            # Make predictions
            with patch.object(adapter, '_load_model', return_value=mock_model):
                result = adapter.predict(detector, sample_data["X_test"])
                
                assert isinstance(result, DetectionResult)
                assert len(result.anomaly_scores) == len(sample_data["X_test"])

    def test_ensemble_detection_workflow(self, sample_data):
        """Test ensemble detection with multiple algorithms."""
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        adapters = [SklearnAdapter(algorithm=alg) for alg in algorithms]
        
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        
        detectors = []
        
        for i, adapter in enumerate(adapters):
            with patch(f'sklearn.{["ensemble", "neighbors", "svm"][i]}.{algorithms[i]}') as mock_alg:
                mock_model = MagicMock()
                mock_model.fit.return_value = mock_model
                mock_alg.return_value = mock_model
                
                detector = adapter.fit(dataset)
                detectors.append(detector)
        
        assert len(detectors) == 3
        assert all(det is not None for det in detectors)

    def test_preprocessing_pipeline_integration(self, sample_data):
        """Test integration with preprocessing pipeline."""
        adapter = SklearnAdapter(
            algorithm="IsolationForest",
            standardize=True,
            use_pca=True,
            pca_components=4
        )
        
        dataset = Mock()
        dataset.data = sample_data["X_train"]
        
        with patch('sklearn.preprocessing.StandardScaler') as mock_scaler, \
             patch('sklearn.decomposition.PCA') as mock_pca, \
             patch('sklearn.ensemble.IsolationForest') as mock_iso:
            
            # Setup mocks
            mock_scaler_instance = MagicMock()
            mock_scaler_instance.fit_transform.return_value = sample_data["X_train"]
            mock_scaler.return_value = mock_scaler_instance
            
            mock_pca_instance = MagicMock()
            mock_pca_instance.fit_transform.return_value = np.random.randn(600, 4)
            mock_pca.return_value = mock_pca_instance
            
            mock_model = MagicMock()
            mock_iso.return_value = mock_model
            
            detector = adapter.fit(dataset)
            
            assert "scaler" in detector.parameters
            assert "pca" in detector.parameters
            assert "model" in detector.parameters

    def test_model_comparison_workflow(self, sample_data):
        """Test model comparison across different algorithms."""
        algorithms = ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]
        performance_scores = {}
        
        for algorithm in algorithms:
            adapter = SklearnAdapter(algorithm=algorithm)
            
            with patch('sklearn.model_selection.cross_val_score') as mock_cv:
                mock_cv.return_value = np.array([0.8, 0.85, 0.75, 0.9, 0.82])
                
                scores = adapter.cross_validate(
                    sample_data["X_train"],
                    cv=5,
                    contamination_rate=ContaminationRate(0.1)
                )
                
                performance_scores[algorithm] = np.mean(scores)
        
        assert len(performance_scores) == 3
        assert all(0 <= score <= 1 for score in performance_scores.values())
        
        # Find best performing algorithm
        best_algorithm = max(performance_scores, key=performance_scores.get)
        assert best_algorithm in algorithms
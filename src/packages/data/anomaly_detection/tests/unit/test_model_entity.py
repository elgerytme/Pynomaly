"""Unit tests for Model entity."""

import pytest
import json
import pickle
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import numpy as np

from anomaly_detection.domain.entities.model import (
    Model, ModelMetadata, ModelStatus, SerializationFormat, ModelType, SerializableModel
)


class MockAlgorithm:
    """Mock algorithm for testing."""
    
    def __init__(self, should_fail_predict=False, should_fail_score=False):
        self.fitted = False
        self.should_fail_predict = should_fail_predict
        self.should_fail_score = should_fail_score
    
    def fit(self, X):
        self.fitted = True
        
    def predict(self, X):
        if self.should_fail_predict:
            raise ValueError("Prediction failed")
        return np.array([0, 1, 0, 1])
    
    def decision_function(self, X):
        if self.should_fail_score:
            raise ValueError("Scoring failed")
        return np.array([0.1, 0.9, 0.2, 0.8])


class MockPreprocessor:
    """Mock preprocessing pipeline."""
    
    def transform(self, X):
        return X * 2  # Simple transformation


class TestModelMetadata:
    """Test suite for ModelMetadata."""
    
    def test_metadata_creation_minimal(self):
        """Test minimal metadata creation."""
        metadata = ModelMetadata(
            model_id="test-model-1",
            name="Test Model",
            algorithm="isolation_forest"
        )
        
        assert metadata.model_id == "test-model-1"
        assert metadata.name == "Test Model"
        assert metadata.algorithm == "isolation_forest"
        assert metadata.version == "1.0.0"
        assert metadata.status == ModelStatus.TRAINING
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)
        assert metadata.hyperparameters == {}
        assert metadata.tags == []
        assert metadata.description == ""
        assert metadata.author == "system"
    
    def test_metadata_creation_full(self):
        """Test full metadata creation with all fields."""
        created_time = datetime(2024, 1, 1, 12, 0, 0)
        updated_time = datetime(2024, 1, 2, 12, 0, 0)
        
        metadata = ModelMetadata(
            model_id="full-model",
            name="Full Test Model",
            algorithm="autoencoder",
            version="2.1.0",
            created_at=created_time,
            updated_at=updated_time,
            status=ModelStatus.DEPLOYED,
            training_samples=10000,
            training_features=50,
            contamination_rate=0.05,
            training_duration_seconds=300.5,
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            feature_names=["feature_1", "feature_2", "feature_3"],
            deployment_environment="production",
            api_endpoint="https://api.example.com/v1/detect",
            tags=["production", "high-accuracy"],
            description="Production anomaly detection model",
            author="data-scientist"
        )
        
        assert metadata.model_id == "full-model"
        assert metadata.version == "2.1.0"
        assert metadata.status == ModelStatus.DEPLOYED
        assert metadata.training_samples == 10000
        assert metadata.contamination_rate == 0.05
        assert metadata.accuracy == 0.95
        assert metadata.hyperparameters == {"n_estimators": 100, "max_depth": 10}
        assert metadata.tags == ["production", "high-accuracy"]
        assert metadata.author == "data-scientist"
    
    def test_metadata_enum_validation(self):
        """Test enum field validation."""
        # Valid status
        metadata = ModelMetadata(
            model_id="test",
            name="Test",
            algorithm="test",
            status=ModelStatus.TRAINED
        )
        assert metadata.status == ModelStatus.TRAINED
        
        # Test all status values
        for status in ModelStatus:
            metadata = ModelMetadata(
                model_id="test",
                name="Test",
                algorithm="test",
                status=status
            )
            assert metadata.status == status


class TestSerializationFormat:
    """Test suite for SerializationFormat enum."""
    
    def test_serialization_format_values(self):
        """Test serialization format enum values."""
        assert SerializationFormat.PICKLE.value == "pickle"
        assert SerializationFormat.JOBLIB.value == "joblib"  
        assert SerializationFormat.ONNX.value == "onnx"
        assert SerializationFormat.JSON.value == "json"


class TestModelType:
    """Test suite for ModelType enum."""
    
    def test_model_type_values(self):
        """Test model type enum values."""
        assert ModelType.ISOLATION_FOREST.value == "isolation_forest"
        assert ModelType.LOCAL_OUTLIER_FACTOR.value == "lof"
        assert ModelType.ONE_CLASS_SVM.value == "one_class_svm"
        assert ModelType.AUTOENCODER.value == "autoencoder"
        assert ModelType.ENSEMBLE.value == "ensemble"
        assert ModelType.CUSTOM.value == "custom"


class TestModel:
    """Test suite for Model entity."""
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata for testing."""
        return ModelMetadata(
            model_id="test-model",
            name="Test Model",
            algorithm="isolation_forest",
            training_samples=1000,
            contamination_rate=0.1
        )
    
    @pytest.fixture
    def mock_algorithm(self):
        """Create mock algorithm."""
        return MockAlgorithm()
    
    @pytest.fixture
    def mock_preprocessor(self):
        """Create mock preprocessor."""
        return MockPreprocessor()
    
    def test_model_creation_minimal(self, sample_metadata):
        """Test minimal model creation."""
        model = Model(metadata=sample_metadata)
        
        assert model.metadata == sample_metadata
        assert model.model_object is None
        assert model.preprocessing_pipeline is None
    
    def test_model_creation_full(self, sample_metadata, mock_algorithm, mock_preprocessor):
        """Test full model creation."""
        model = Model(
            metadata=sample_metadata,
            model_object=mock_algorithm,
            preprocessing_pipeline=mock_preprocessor
        )
        
        assert model.metadata == sample_metadata
        assert model.model_object == mock_algorithm
        assert model.preprocessing_pipeline == mock_preprocessor
    
    def test_model_post_init_updates_time(self):
        """Test that post_init updates time if needed."""
        old_time = datetime(2020, 1, 1)
        new_time = datetime(2024, 1, 1)
        
        metadata = ModelMetadata(
            model_id="test",
            name="Test",
            algorithm="test",
            created_at=new_time,
            updated_at=old_time  # Updated time is before created time
        )
        
        model = Model(metadata=metadata)
        
        # Should update updated_at to match created_at
        assert model.metadata.updated_at >= model.metadata.created_at
    
    def test_model_predict_success(self, sample_metadata, mock_algorithm):
        """Test successful prediction."""
        model = Model(
            metadata=sample_metadata,
            model_object=mock_algorithm
        )
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        predictions = model.predict(X)
        
        np.testing.assert_array_equal(predictions, [0, 1, 0, 1])
    
    def test_model_predict_with_preprocessing(self, sample_metadata, mock_algorithm, mock_preprocessor):
        """Test prediction with preprocessing."""
        model = Model(
            metadata=sample_metadata,
            model_object=mock_algorithm,
            preprocessing_pipeline=mock_preprocessor
        )
        
        X = np.array([[1, 2], [3, 4]])
        predictions = model.predict(X)
        
        # Should still work with preprocessing applied
        np.testing.assert_array_equal(predictions, [0, 1, 0, 1])
    
    def test_model_predict_no_model_object(self, sample_metadata):
        """Test prediction without model object."""
        model = Model(metadata=sample_metadata)
        
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="No model object loaded"):
            model.predict(X)
    
    def test_model_predict_no_predict_method(self, sample_metadata):
        """Test prediction with object that doesn't support prediction."""
        mock_obj = Mock()
        # Don't add predict method
        
        model = Model(
            metadata=sample_metadata,
            model_object=mock_obj
        )
        
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="does not support prediction"):
            model.predict(X)
    
    def test_model_predict_algorithm_failure(self, sample_metadata):
        """Test prediction when algorithm fails."""
        mock_algorithm = MockAlgorithm(should_fail_predict=True)
        
        model = Model(
            metadata=sample_metadata,
            model_object=mock_algorithm
        )
        
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="Prediction failed"):
            model.predict(X)
    
    def test_model_get_anomaly_scores_success(self, sample_metadata, mock_algorithm):
        """Test successful anomaly scoring."""
        model = Model(
            metadata=sample_metadata,
            model_object=mock_algorithm
        )
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        scores = model.get_anomaly_scores(X)
        
        np.testing.assert_array_equal(scores, [0.1, 0.9, 0.2, 0.8])
    
    def test_model_get_anomaly_scores_score_samples(self, sample_metadata):
        """Test scoring with score_samples method."""
        mock_obj = Mock()
        mock_obj.score_samples.return_value = np.array([0.2, 0.8, 0.3, 0.7])
        # No decision_function method, should fall back to score_samples
        
        model = Model(
            metadata=sample_metadata,
            model_object=mock_obj
        )
        
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        scores = model.get_anomaly_scores(X)
        
        np.testing.assert_array_equal(scores, [0.2, 0.8, 0.3, 0.7])
    
    def test_model_get_anomaly_scores_no_model_object(self, sample_metadata):
        """Test scoring without model object."""
        model = Model(metadata=sample_metadata)
        
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="No model object loaded"):
            model.get_anomaly_scores(X)
    
    def test_model_get_anomaly_scores_no_scoring_method(self, sample_metadata):
        """Test scoring with object that doesn't support scoring."""
        mock_obj = Mock()
        # Don't add decision_function or score_samples
        
        model = Model(
            metadata=sample_metadata,
            model_object=mock_obj
        )
        
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="does not support anomaly scoring"):
            model.get_anomaly_scores(X)
    
    def test_update_performance_metrics(self, sample_metadata):
        """Test updating performance metrics."""
        model = Model(metadata=sample_metadata)
        
        original_updated_at = model.metadata.updated_at
        
        model.update_performance_metrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90
        )
        
        assert model.metadata.accuracy == 0.95
        assert model.metadata.precision == 0.92
        assert model.metadata.recall == 0.88
        assert model.metadata.f1_score == 0.90
        assert model.metadata.updated_at > original_updated_at
    
    def test_add_tag(self, sample_metadata):
        """Test adding tags."""
        model = Model(metadata=sample_metadata)
        
        assert model.metadata.tags == []
        
        model.add_tag("production")
        assert "production" in model.metadata.tags
        
        model.add_tag("high-accuracy")
        assert len(model.metadata.tags) == 2
        
        # Adding duplicate tag should not duplicate
        model.add_tag("production")
        assert len(model.metadata.tags) == 2
    
    def test_remove_tag(self, sample_metadata):
        """Test removing tags."""
        sample_metadata.tags = ["production", "high-accuracy", "tested"]
        model = Model(metadata=sample_metadata)
        
        model.remove_tag("high-accuracy")
        assert "high-accuracy" not in model.metadata.tags
        assert len(model.metadata.tags) == 2
        
        # Removing non-existent tag should not fail
        model.remove_tag("non-existent")
        assert len(model.metadata.tags) == 2
    
    def test_get_summary(self, sample_metadata, mock_algorithm, mock_preprocessor):
        """Test getting model summary."""
        model = Model(
            metadata=sample_metadata,
            model_object=mock_algorithm,
            preprocessing_pipeline=mock_preprocessor
        )
        
        summary = model.get_summary()
        
        assert isinstance(summary, dict)
        assert summary["model_id"] == "test-model"
        assert summary["name"] == "Test Model"
        assert summary["algorithm"] == "isolation_forest"
        assert summary["has_model_object"] is True
        assert summary["has_preprocessing"] is True
    
    def test_get_summary_without_objects(self, sample_metadata):
        """Test getting summary without model objects."""
        model = Model(metadata=sample_metadata)
        
        summary = model.get_summary()
        
        assert summary["has_model_object"] is False
        assert summary["has_preprocessing"] is False
    
    def test_model_str_representation(self, sample_metadata):
        """Test string representation."""
        model = Model(metadata=sample_metadata)
        
        str_repr = str(model)
        assert "Model" in str_repr
        assert "test-model" in str_repr
        assert "Test Model" in str_repr
        assert "isolation_forest" in str_repr
        assert "training" in str_repr
    
    def test_model_repr_representation(self, sample_metadata):
        """Test detailed representation."""
        model = Model(metadata=sample_metadata)
        
        repr_str = repr(model)
        assert repr_str == str(model)


class TestModelSerialization:
    """Test suite for model serialization and deserialization."""
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for serialization tests."""
        metadata = ModelMetadata(
            model_id="serialize-test",
            name="Serialization Test Model",
            algorithm="isolation_forest",
            training_samples=500,
            accuracy=0.92
        )
        
        mock_algo = MockAlgorithm()
        mock_prep = MockPreprocessor()
        
        return Model(
            metadata=metadata,
            model_object=mock_algo,
            preprocessing_pipeline=mock_prep
        )
    
    def test_save_pickle_format(self, sample_model):
        """Test saving in pickle format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model.pkl"
            
            sample_model.save(file_path, SerializationFormat.PICKLE)
            
            assert file_path.exists()
            assert file_path.suffix == ".pkl"
            
            # Verify file contains expected data
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            assert 'metadata' in data
            assert 'model_object' in data
            assert 'preprocessing_pipeline' in data
            assert data['serialization_format'] == 'pickle'
    
    def test_save_pickle_auto_extension(self, sample_model):
        """Test pickle save adds extension automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model"  # No extension
            
            sample_model.save(file_path, SerializationFormat.PICKLE)
            
            expected_path = file_path.with_suffix('.pkl')
            assert expected_path.exists()
    
    @patch('joblib.dump')
    @patch('anomaly_detection.domain.entities.model.joblib', create=True)
    def test_save_joblib_format(self, mock_joblib_module, mock_dump, sample_model):
        """Test saving in joblib format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model.joblib"
            
            sample_model.save(file_path, SerializationFormat.JOBLIB)
            
            mock_dump.assert_called_once()
            call_args = mock_dump.call_args
            assert call_args[0][1] == file_path  # Second argument is file path
            assert call_args[1]['compress'] == 3
    
    def test_save_joblib_no_library(self, sample_model):
        """Test joblib save without library installed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model.joblib"
            
            with patch('anomaly_detection.domain.entities.model.joblib', None):
                with pytest.raises(ImportError, match="joblib is required"):
                    sample_model.save(file_path, SerializationFormat.JOBLIB)
    
    def test_save_json_format(self, sample_model):
        """Test saving in JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model.json"
            
            sample_model.save(file_path, SerializationFormat.JSON)
            
            assert file_path.exists()
            assert file_path.suffix == ".json"
            
            # Verify JSON structure
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            assert data['model_id'] == "serialize-test"
            assert data['name'] == "Serialization Test Model"
            assert data['algorithm'] == "isolation_forest"
            assert data['serialization_format'] == 'json'
            assert 'created_at' in data
            assert 'updated_at' in data
    
    def test_save_unsupported_format(self, sample_model):
        """Test saving with unsupported format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model"
            
            with pytest.raises(ValueError, match="Unsupported serialization format"):
                sample_model.save(file_path, SerializationFormat.ONNX)
    
    def test_save_creates_directory(self, sample_model):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "deep" / "test_model.pkl"
            
            sample_model.save(nested_path, SerializationFormat.PICKLE)
            
            assert nested_path.exists()
            assert nested_path.parent.exists()
    
    def test_load_pickle_format(self, sample_model):
        """Test loading from pickle format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model.pkl"
            
            # Save first
            sample_model.save(file_path, SerializationFormat.PICKLE)
            
            # Load and verify
            loaded_model = Model.load(file_path)
            
            assert loaded_model.metadata.model_id == sample_model.metadata.model_id
            assert loaded_model.metadata.name == sample_model.metadata.name
            assert loaded_model.model_object is not None
            assert loaded_model.preprocessing_pipeline is not None
    
    @patch('joblib.load')
    @patch('anomaly_detection.domain.entities.model.joblib', create=True)
    def test_load_joblib_format(self, mock_joblib_module, mock_load, sample_model):
        """Test loading from joblib format."""
        # Mock joblib.load to return expected data structure
        mock_data = {
            'metadata': sample_model.metadata,
            'model_object': sample_model.model_object,
            'preprocessing_pipeline': sample_model.preprocessing_pipeline
        }
        mock_load.return_value = mock_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model.joblib"
            file_path.touch()  # Create empty file
            
            loaded_model = Model.load(file_path)
            
            mock_load.assert_called_once_with(file_path)
            assert loaded_model.metadata == sample_model.metadata
    
    def test_load_joblib_no_library(self):
        """Test loading joblib without library installed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model.joblib"
            file_path.touch()
            
            with patch('anomaly_detection.domain.entities.model.joblib', None):
                with pytest.raises(ImportError, match="joblib is required"):
                    Model.load(file_path)
    
    def test_load_json_format(self, sample_model):
        """Test loading from JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model.json"
            
            # Save first
            sample_model.save(file_path, SerializationFormat.JSON)
            
            # Load and verify
            loaded_model = Model.load(file_path)
            
            assert loaded_model.metadata.model_id == sample_model.metadata.model_id
            assert loaded_model.metadata.name == sample_model.metadata.name
            assert loaded_model.model_object is None  # JSON doesn't store objects
            assert loaded_model.preprocessing_pipeline is None
    
    def test_load_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            Model.load("/non/existent/path.pkl")
    
    def test_load_auto_detect_format(self, sample_model):
        """Test automatic format detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save without extension
            file_path = Path(temp_dir) / "test_model"
            sample_model.save(file_path, SerializationFormat.PICKLE)
            
            # Should detect pickle format
            loaded_model = Model.load(file_path.with_suffix('.pkl'))
            assert loaded_model.metadata.model_id == sample_model.metadata.model_id
    
    def test_load_format_detection_fallback(self):
        """Test format detection with fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model.unknown"
            
            # Create a file that's not valid pickle/joblib but valid JSON
            json_data = {
                "model_id": "test",
                "name": "Test",
                "algorithm": "test",
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "status": "training"
            }
            
            with open(file_path, 'w') as f:
                json.dump(json_data, f)
            
            # Should fall back to JSON
            loaded_model = Model.load(file_path)
            assert loaded_model.metadata.model_id == "test"
    
    def test_load_corrupt_file(self):
        """Test loading corrupt file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "corrupt.pkl"
            
            # Write invalid data
            with open(file_path, 'wb') as f:
                f.write(b"not a valid pickle file")
            
            # Should raise an exception during format detection
            with pytest.raises(Exception):
                Model.load(file_path)
    
    def test_save_updates_metadata_timestamp(self, sample_model):
        """Test that save updates metadata timestamp."""
        original_updated_at = sample_model.metadata.updated_at
        
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_model.pkl"
            
            sample_model.save(file_path, SerializationFormat.PICKLE)
            
            assert sample_model.metadata.updated_at > original_updated_at


class TestSerializableModelProtocol:
    """Test suite for SerializableModel protocol."""
    
    def test_protocol_implementation(self):
        """Test that mock algorithm implements protocol."""
        mock_algo = MockAlgorithm()
        
        # Check that it has required methods
        assert hasattr(mock_algo, 'fit')
        assert hasattr(mock_algo, 'predict')
        assert hasattr(mock_algo, 'decision_function')
        
        # Test protocol usage
        X = np.array([[1, 2], [3, 4]])
        mock_algo.fit(X)
        predictions = mock_algo.predict(X)
        scores = mock_algo.decision_function(X)
        
        assert len(predictions) == 4
        assert len(scores) == 4


class TestModelIntegration:
    """Integration tests for Model entity."""
    
    def test_full_workflow_pickle(self):
        """Test complete workflow with pickle serialization."""
        # Create model
        metadata = ModelMetadata(
            model_id="integration-test",
            name="Integration Test Model",
            algorithm="test_algorithm",
            contamination_rate=0.1
        )
        
        algorithm = MockAlgorithm()
        preprocessor = MockPreprocessor()
        
        model = Model(
            metadata=metadata,
            model_object=algorithm,
            preprocessing_pipeline=preprocessor
        )
        
        # Update performance metrics
        model.update_performance_metrics(0.95, 0.92, 0.88, 0.90)
        
        # Add tags
        model.add_tag("integration-test")
        model.add_tag("validated")
        
        # Make predictions
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        predictions = model.predict(X)
        scores = model.get_anomaly_scores(X)
        
        assert len(predictions) == 4
        assert len(scores) == 4
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "integration_model.pkl"
            model.save(file_path)
            
            # Load model
            loaded_model = Model.load(file_path)
            
            # Verify loaded model
            assert loaded_model.metadata.model_id == metadata.model_id
            assert loaded_model.metadata.accuracy == 0.95
            assert "integration-test" in loaded_model.metadata.tags
            
            # Test predictions with loaded model
            loaded_predictions = loaded_model.predict(X)
            loaded_scores = loaded_model.get_anomaly_scores(X)
            
            np.testing.assert_array_equal(predictions, loaded_predictions)
            np.testing.assert_array_equal(scores, loaded_scores)
    
    def test_model_lifecycle_status_transitions(self):
        """Test model status transitions through lifecycle."""
        metadata = ModelMetadata(
            model_id="lifecycle-test",
            name="Lifecycle Test",
            algorithm="test",
            status=ModelStatus.TRAINING
        )
        
        model = Model(metadata=metadata)
        
        # Training -> Trained
        model.metadata.status = ModelStatus.TRAINED
        assert model.metadata.status == ModelStatus.TRAINED
        
        # Trained -> Deployed
        model.metadata.status = ModelStatus.DEPLOYED
        assert model.metadata.status == ModelStatus.DEPLOYED
        
        # Deployed -> Deprecated
        model.metadata.status = ModelStatus.DEPRECATED
        assert model.metadata.status == ModelStatus.DEPRECATED
    
    def test_model_with_real_sklearn_like_object(self):
        """Test model with sklearn-like object."""
        from unittest.mock import Mock
        
        # Create sklearn-like mock
        sklearn_mock = Mock()
        sklearn_mock.predict.return_value = np.array([0, 1, 0])
        sklearn_mock.decision_function.return_value = np.array([0.2, 0.8, 0.3])
        
        metadata = ModelMetadata(
            model_id="sklearn-test",
            name="Sklearn Test",
            algorithm="isolation_forest"
        )
        
        model = Model(metadata=metadata, model_object=sklearn_mock)
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        predictions = model.predict(X)
        scores = model.get_anomaly_scores(X)
        
        np.testing.assert_array_equal(predictions, [0, 1, 0])
        np.testing.assert_array_equal(scores, [0.2, 0.8, 0.3])
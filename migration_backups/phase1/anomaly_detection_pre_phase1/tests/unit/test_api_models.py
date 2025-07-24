"""Unit tests for model management API endpoints."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from anomaly_detection.api.v1.models import (
    router, ModelPredictionRequest, ModelListResponse, PredictionResponse,
    TrainingRequest, TrainingResponse, get_model_repository
)
from anomaly_detection.domain.entities.model import Model, ModelMetadata, ModelStatus
from anomaly_detection.domain.entities.detection_result import DetectionResult


@pytest.fixture
def app():
    """Create FastAPI app with models router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/models")
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_repository():
    """Create mock model repository."""
    repo = Mock()
    
    # Mock model list
    repo.list_models.return_value = [
        {
            "model_id": "model-1",
            "name": "Test Model 1",
            "algorithm": "isolation_forest",
            "status": "trained",
            "created_at": "2024-01-01T12:00:00",
            "accuracy": 0.95
        },
        {
            "model_id": "model-2", 
            "name": "Test Model 2",
            "algorithm": "one_class_svm",
            "status": "trained",
            "created_at": "2024-01-02T12:00:00",
            "accuracy": 0.92
        }
    ]
    
    # Mock model metadata
    repo.get_model_metadata.return_value = {
        "model_id": "model-1",
        "name": "Test Model 1",
        "algorithm": "isolation_forest",
        "status": "trained",
        "training_samples": 1000,
        "accuracy": 0.95,
        "created_at": "2024-01-01T12:00:00"
    }
    
    # Mock model loading
    mock_model = Mock()
    mock_model.metadata = ModelMetadata(
        model_id="model-1",
        name="Test Model 1",
        algorithm="isolation_forest"
    )
    mock_model.predict.return_value = np.array([1, -1, 1, -1])
    mock_model.get_anomaly_scores.return_value = np.array([0.2, 0.8, 0.3, 0.9])
    repo.load.return_value = mock_model
    
    # Mock deletion
    repo.delete.return_value = True
    
    # Mock repository stats
    repo.get_repository_stats.return_value = {
        "total_models": 5,
        "models_by_status": {"trained": 3, "deployed": 2},
        "models_by_algorithm": {"isolation_forest": 2, "one_class_svm": 3},
        "storage_size_mb": 125.5
    }
    
    # Mock saving
    repo.save.return_value = "saved-model-id"
    
    return repo


@pytest.fixture
def mock_detection_service():
    """Create mock detection service."""
    service = Mock()
    
    # Mock fitted models
    service._fitted_models = {"iforest": Mock()}
    
    # Mock fit method
    service.fit.return_value = None
    
    # Mock detection result
    result = DetectionResult(
        success=True,
        predictions=np.array([1, -1, 1, -1]),
        confidence_scores=np.array([0.2, 0.8, 0.3, 0.9]),
        anomalies=[1, 3],
        algorithm="iforest",
        total_samples=4,
        anomaly_count=2,
        anomaly_rate=0.5,
        execution_time_ms=150.0
    )
    service.detect_anomalies.return_value = result
    
    return service


class TestModelListEndpoint:
    """Test model listing endpoint."""
    
    def test_list_models_success(self, client, mock_model_repository):
        """Test successful model listing."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            response = client.get("/api/v1/models")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "models" in data
            assert "total_count" in data
            assert data["total_count"] == 2
            assert len(data["models"]) == 2
            
            # Check first model
            model1 = data["models"][0]
            assert model1["model_id"] == "model-1"
            assert model1["name"] == "Test Model 1"
            assert model1["algorithm"] == "isolation_forest"
            assert model1["accuracy"] == 0.95
    
    def test_list_models_with_algorithm_filter(self, client, mock_model_repository):
        """Test model listing with algorithm filter."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            response = client.get("/api/v1/models?algorithm=isolation_forest")
            
            assert response.status_code == 200
            
            # Verify repository was called with filter
            mock_model_repository.list_models.assert_called_once_with(
                status=None,
                algorithm="isolation_forest"
            )
    
    def test_list_models_with_status_filter(self, client, mock_model_repository):
        """Test model listing with status filter."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            response = client.get("/api/v1/models?status=trained")
            
            assert response.status_code == 200
            
            # Verify repository was called with correct status enum
            call_args = mock_model_repository.list_models.call_args
            assert call_args[1]["status"] == ModelStatus.TRAINED
    
    def test_list_models_invalid_status(self, client, mock_model_repository):
        """Test model listing with invalid status."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            response = client.get("/api/v1/models?status=invalid_status")
            
            assert response.status_code == 400
            assert "Invalid status" in response.json()["detail"]
    
    def test_list_models_repository_error(self, client):
        """Test model listing with repository error."""
        mock_repo = Mock()
        mock_repo.list_models.side_effect = Exception("Database error")
        
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_repo):
            
            response = client.get("/api/v1/models")
            
            assert response.status_code == 500
            assert "Failed to list models" in response.json()["detail"]


class TestModelInfoEndpoint:
    """Test model info endpoint."""
    
    def test_get_model_info_success(self, client, mock_model_repository):
        """Test successful model info retrieval."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            response = client.get("/api/v1/models/model-1")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["model_id"] == "model-1"
            assert data["name"] == "Test Model 1"
            assert data["algorithm"] == "isolation_forest"
            assert data["accuracy"] == 0.95
            
            # Verify repository method was called
            mock_model_repository.get_model_metadata.assert_called_once_with("model-1")
    
    def test_get_model_info_not_found(self, client):
        """Test model info for non-existent model."""
        mock_repo = Mock()
        mock_repo.get_model_metadata.side_effect = FileNotFoundError("Model not found")
        
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_repo):
            
            response = client.get("/api/v1/models/non-existent")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
    
    def test_get_model_info_repository_error(self, client):
        """Test model info with repository error."""
        mock_repo = Mock()
        mock_repo.get_model_metadata.side_effect = Exception("Database error")
        
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_repo):
            
            response = client.get("/api/v1/models/model-1")
            
            assert response.status_code == 500
            assert "Failed to get model info" in response.json()["detail"]


class TestModelPredictionEndpoint:
    """Test model prediction endpoint."""
    
    def test_predict_with_model_success(self, client, mock_model_repository):
        """Test successful prediction with saved model."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            request_data = {
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "model_id": "model-1"
            }
            
            response = client.post("/api/v1/models/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["model_id"] == "model-1"
            assert data["algorithm"] == "isolation_forest"
            assert data["total_samples"] == 4
            assert data["anomalies_detected"] == 2
            assert data["anomaly_rate"] == 0.5
            assert data["anomalies"] == [1, 3]
            assert data["scores"] == [0.2, 0.8, 0.3, 0.9]
            assert "timestamp" in data
            assert "processing_time_ms" in data
            
            # Verify model was loaded and used
            mock_model_repository.load.assert_called_once_with("model-1")
    
    def test_predict_with_model_empty_data(self, client, mock_model_repository):
        """Test prediction with empty data."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            request_data = {
                "data": [],
                "model_id": "model-1"
            }
            
            response = client.post("/api/v1/models/predict", json=request_data)
            
            assert response.status_code == 400
            assert "empty" in response.json()["detail"].lower()
    
    def test_predict_with_model_not_found(self, client):
        """Test prediction with non-existent model."""
        mock_repo = Mock()
        mock_repo.load.side_effect = FileNotFoundError("Model not found")
        
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_repo):
            
            request_data = {
                "data": [[1, 2], [3, 4]],
                "model_id": "non-existent"
            }
            
            response = client.post("/api/v1/models/predict", json=request_data)
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
    
    def test_predict_with_model_no_scores(self, client, mock_model_repository):
        """Test prediction when model doesn't support scoring."""
        # Create mock model that fails on scoring
        mock_model = Mock()
        mock_model.metadata.algorithm = "test_algorithm"
        mock_model.predict.return_value = np.array([1, -1, 1, -1])
        mock_model.get_anomaly_scores.side_effect = Exception("Scoring not supported")
        
        mock_repo = Mock()
        mock_repo.load.return_value = mock_model
        
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_repo):
            
            request_data = {
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "model_id": "model-1"
            }
            
            response = client.post("/api/v1/models/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["scores"] is None  # Should be None when scoring fails
    
    def test_predict_with_model_prediction_error(self, client):
        """Test prediction with model prediction error."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Prediction failed")
        
        mock_repo = Mock()
        mock_repo.load.return_value = mock_model
        
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_repo):
            
            request_data = {
                "data": [[1, 2], [3, 4]],
                "model_id": "model-1"
            }
            
            response = client.post("/api/v1/models/predict", json=request_data)
            
            assert response.status_code == 500
            assert "Prediction failed" in response.json()["detail"]


class TestModelDeletionEndpoint:
    """Test model deletion endpoint."""
    
    def test_delete_model_success(self, client, mock_model_repository):
        """Test successful model deletion."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            response = client.delete("/api/v1/models/model-1")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "deleted successfully" in data["message"]
            
            # Verify repository delete was called
            mock_model_repository.delete.assert_called_once_with("model-1")
    
    def test_delete_model_not_found(self, client):
        """Test deletion of non-existent model."""
        mock_repo = Mock()
        mock_repo.delete.return_value = False  # Model not found
        
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_repo):
            
            response = client.delete("/api/v1/models/non-existent")
            
            assert response.status_code == 404
            assert "not found" in response.json()["detail"]
    
    def test_delete_model_repository_error(self, client):
        """Test model deletion with repository error."""
        mock_repo = Mock()
        mock_repo.delete.side_effect = Exception("Database error")
        
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_repo):
            
            response = client.delete("/api/v1/models/model-1")
            
            assert response.status_code == 500
            assert "Failed to delete model" in response.json()["detail"]


class TestModelStatsEndpoint:
    """Test model repository stats endpoint."""
    
    def test_get_repository_stats_success(self, client, mock_model_repository):
        """Test successful repository stats retrieval."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            response = client.get("/api/v1/models/stats/repository")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "repository_stats" in data
            assert "timestamp" in data
            
            stats = data["repository_stats"]
            assert stats["total_models"] == 5
            assert "models_by_status" in stats
            assert "models_by_algorithm" in stats
            assert stats["storage_size_mb"] == 125.5
    
    def test_get_repository_stats_error(self, client):
        """Test repository stats with error."""
        mock_repo = Mock()
        mock_repo.get_repository_stats.side_effect = Exception("Stats error")
        
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_repo):
            
            response = client.get("/api/v1/models/stats/repository")
            
            assert response.status_code == 500
            assert "Failed to get repository stats" in response.json()["detail"]


class TestModelTrainingEndpoint:
    """Test model training endpoint."""
    
    def test_train_model_success(self, client, mock_model_repository, mock_detection_service):
        """Test successful model training."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository), \
             patch('anomaly_detection.api.v1.models.DetectionService',
                  return_value=mock_detection_service), \
             patch('anomaly_detection.api.v1.models.uuid.uuid4',
                  return_value=Mock(hex="test-model-id")):
            
            request_data = {
                "model_name": "Test Trained Model",
                "algorithm": "isolation_forest",
                "contamination": 0.1,
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "labels": [1, -1, 1, -1],
                "feature_names": ["feature_1", "feature_2"],
                "description": "Test model description",
                "hyperparameters": {"n_estimators": 100}
            }
            
            response = client.post("/api/v1/models/train", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["model_id"] == "saved-model-id"  # From mock
            assert data["model_name"] == "Test Trained Model"
            assert data["algorithm"] == "isolation_forest"
            assert data["training_samples"] == 4
            assert data["training_features"] == 2
            assert data["contamination_rate"] == 0.1
            assert data["accuracy"] is not None
            assert data["precision"] is not None
            assert data["recall"] is not None
            assert data["f1_score"] is not None
            assert "timestamp" in data
            assert "training_duration_seconds" in data
            
            # Verify service methods were called
            mock_detection_service.fit.assert_called_once()
            mock_detection_service.detect_anomalies.assert_called_once()
            mock_model_repository.save.assert_called_once()
    
    def test_train_model_without_labels(self, client, mock_model_repository, mock_detection_service):
        """Test model training without labels."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository), \
             patch('anomaly_detection.api.v1.models.DetectionService',
                  return_value=mock_detection_service), \
             patch('anomaly_detection.api.v1.models.uuid.uuid4',
                  return_value=Mock(hex="test-model-id")):
            
            request_data = {
                "model_name": "Test Trained Model",
                "algorithm": "isolation_forest",
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]]
                # No labels provided
            }
            
            response = client.post("/api/v1/models/train", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            # Metrics should be None when no labels provided
            assert data["accuracy"] is None
            assert data["precision"] is None
            assert data["recall"] is None
            assert data["f1_score"] is None
    
    def test_train_model_empty_data(self, client, mock_model_repository):
        """Test training with empty data."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            request_data = {
                "model_name": "Test Model",
                "algorithm": "isolation_forest",
                "data": []
            }
            
            response = client.post("/api/v1/models/train", json=request_data)
            
            assert response.status_code == 400
            assert "empty" in response.json()["detail"].lower()
    
    def test_train_model_invalid_data_shape(self, client, mock_model_repository):
        """Test training with invalid data shape."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            request_data = {
                "model_name": "Test Model",
                "algorithm": "isolation_forest",
                "data": [1, 2, 3, 4]  # 1D array instead of 2D
            }
            
            response = client.post("/api/v1/models/train", json=request_data)
            
            assert response.status_code == 400
            assert "2D array" in response.json()["detail"]
    
    def test_train_model_labels_length_mismatch(self, client, mock_model_repository):
        """Test training with mismatched labels length."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository):
            
            request_data = {
                "model_name": "Test Model",
                "algorithm": "isolation_forest",
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "labels": [1, -1]  # Only 2 labels for 4 samples
            }
            
            response = client.post("/api/v1/models/train", json=request_data)
            
            assert response.status_code == 400
            assert "Labels length must match data length" in response.json()["detail"]
    
    def test_train_model_algorithm_mapping(self, client, mock_model_repository, mock_detection_service):
        """Test algorithm name mapping during training."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository), \
             patch('anomaly_detection.api.v1.models.DetectionService',
                  return_value=mock_detection_service), \
             patch('anomaly_detection.api.v1.models.uuid.uuid4',
                  return_value=Mock(hex="test-model-id")):
            
            test_cases = [
                ("isolation_forest", "iforest"),
                ("one_class_svm", "ocsvm"),
                ("lof", "lof")
            ]
            
            for input_algo, expected_algo in test_cases:
                request_data = {
                    "model_name": "Test Model",
                    "algorithm": input_algo,
                    "data": [[1, 2], [3, 4]]
                }
                
                response = client.post("/api/v1/models/train", json=request_data)
                
                assert response.status_code == 200
                
                # Verify service was called with mapped algorithm
                fit_call_args = mock_detection_service.fit.call_args
                assert fit_call_args[0][1] == expected_algo  # Second argument is algorithm
    
    def test_train_model_training_error(self, client, mock_model_repository):
        """Test training with service error."""
        mock_service = Mock()
        mock_service.fit.side_effect = Exception("Training failed")
        
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository), \
             patch('anomaly_detection.api.v1.models.DetectionService',
                  return_value=mock_service):
            
            request_data = {
                "model_name": "Test Model",
                "algorithm": "isolation_forest",
                "data": [[1, 2], [3, 4]]
            }
            
            response = client.post("/api/v1/models/train", json=request_data)
            
            assert response.status_code == 500
            assert "Training failed" in response.json()["detail"]


class TestModelPydanticModels:
    """Test Pydantic models for model endpoints."""
    
    def test_model_prediction_request(self):
        """Test ModelPredictionRequest model."""
        request = ModelPredictionRequest(
            data=[[1, 2], [3, 4]],
            model_id="test-model"
        )
        
        assert request.data == [[1, 2], [3, 4]]
        assert request.model_id == "test-model"
    
    def test_training_request_defaults(self):
        """Test TrainingRequest model defaults."""
        request = TrainingRequest(
            model_name="Test Model",
            data=[[1, 2], [3, 4]]
        )
        
        assert request.algorithm == "isolation_forest"
        assert request.contamination == 0.1
        assert request.labels is None
        assert request.feature_names is None
        assert request.description is None
        assert request.hyperparameters is None
    
    def test_training_request_validation(self):
        """Test TrainingRequest model validation."""
        # Invalid contamination
        with pytest.raises(ValueError):
            TrainingRequest(
                model_name="Test",
                data=[[1, 2]],
                contamination=0.6  # Too high
            )
        
        with pytest.raises(ValueError):
            TrainingRequest(
                model_name="Test",
                data=[[1, 2]],
                contamination=-0.1  # Negative
            )
    
    def test_model_list_response(self):
        """Test ModelListResponse model."""
        models = [{"id": "1", "name": "Model 1"}]
        response = ModelListResponse(
            models=models,
            total_count=1
        )
        
        assert response.models == models
        assert response.total_count == 1
    
    def test_prediction_response(self):
        """Test PredictionResponse model."""
        response = PredictionResponse(
            success=True,
            anomalies=[1, 3],
            scores=[0.8, 0.9],
            algorithm="isolation_forest",
            model_id="test-model",
            total_samples=4,
            anomalies_detected=2,
            anomaly_rate=0.5,
            timestamp="2024-01-01T12:00:00",
            processing_time_ms=150.5
        )
        
        assert response.success is True
        assert response.model_id == "test-model"
        assert response.anomalies == [1, 3]
        assert response.algorithm == "isolation_forest"
    
    def test_training_response(self):
        """Test TrainingResponse model."""
        response = TrainingResponse(
            success=True,
            model_id="trained-model",
            model_name="Test Model",
            algorithm="isolation_forest",
            training_samples=1000,
            training_features=10,
            contamination_rate=0.1,
            training_duration_seconds=45.5,
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            timestamp="2024-01-01T12:00:00"
        )
        
        assert response.success is True
        assert response.model_id == "trained-model"
        assert response.training_samples == 1000
        assert response.accuracy == 0.95


class TestModelDependencies:
    """Test dependency injection for model endpoints."""
    
    def test_get_model_repository_singleton(self):
        """Test that model repository is singleton."""
        # Clear any existing instance
        import anomaly_detection.api.v1.models as models_module
        models_module._model_repository = None
        
        repo1 = get_model_repository()
        repo2 = get_model_repository()
        
        assert repo1 is repo2


class TestModelIntegration:
    """Integration tests for model endpoints."""
    
    def test_full_model_lifecycle(self, client, mock_model_repository, mock_detection_service):
        """Test complete model lifecycle: train -> list -> predict -> delete."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository), \
             patch('anomaly_detection.api.v1.models.DetectionService',
                  return_value=mock_detection_service), \
             patch('anomaly_detection.api.v1.models.uuid.uuid4',
                  return_value=Mock(hex="lifecycle-test-id")):
            
            # 1. Train model
            train_request = {
                "model_name": "Lifecycle Test Model",
                "algorithm": "isolation_forest",
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "labels": [1, -1, 1, -1]
            }
            
            train_response = client.post("/api/v1/models/train", json=train_request)
            assert train_response.status_code == 200
            
            # 2. List models
            list_response = client.get("/api/v1/models")
            assert list_response.status_code == 200
            assert len(list_response.json()["models"]) == 2
            
            # 3. Make prediction
            predict_request = {
                "data": [[1, 2], [3, 4]],
                "model_id": "model-1"
            }
            
            predict_response = client.post("/api/v1/models/predict", json=predict_request)
            assert predict_response.status_code == 200
            
            # 4. Get model info
            info_response = client.get("/api/v1/models/model-1")
            assert info_response.status_code == 200
            
            # 5. Delete model
            delete_response = client.delete("/api/v1/models/model-1")
            assert delete_response.status_code == 200
    
    def test_dataset_validation_integration(self, client, mock_model_repository, mock_detection_service):
        """Test integration with dataset validation during training."""
        with patch('anomaly_detection.api.v1.models.get_model_repository',
                  return_value=mock_model_repository), \
             patch('anomaly_detection.api.v1.models.DetectionService',
                  return_value=mock_detection_service), \
             patch('anomaly_detection.api.v1.models.uuid.uuid4',
                  return_value=Mock(hex="validation-test-id")):
            
            request_data = {
                "model_name": "Validation Test Model",
                "algorithm": "isolation_forest",
                "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                "feature_names": ["temperature", "pressure"],
                "description": "Test model with validation"
            }
            
            response = client.post("/api/v1/models/train", json=request_data)
            
            assert response.status_code == 200
            
            # Should have created dataset with proper metadata
            # This tests the integration between the API and domain entities
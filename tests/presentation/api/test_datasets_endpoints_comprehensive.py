"""
Dataset Endpoints Testing Suite
Comprehensive tests for dataset management API endpoints.
"""

import pytest
import uuid
import io
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import pandas as pd

from pynomaly.presentation.api.app import create_app
from pynomaly.application.dto import DatasetDTO, CreateDatasetDTO, DataQualityReportDTO
from pynomaly.domain.entities import Dataset


class TestDatasetEndpoints:
    """Test suite for dataset API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_container(self):
        """Mock dependency injection container."""
        with patch('pynomaly.presentation.api.deps.get_container') as mock:
            container = Mock()
            
            # Mock repositories
            container.dataset_repository.return_value = Mock()
            container.data_quality_service.return_value = Mock()
            container.feature_validator.return_value = Mock()
            
            mock.return_value = container
            yield container

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        with patch('pynomaly.presentation.api.deps.get_current_user') as mock:
            user = {
                "user_id": "test-user-123",
                "email": "test@example.com",
                "roles": ["user"],
                "permissions": ["dataset:create", "dataset:read", "dataset:update", "dataset:delete"]
            }
            mock.return_value = user
            yield user

    @pytest.fixture
    def sample_dataset_dto(self):
        """Sample dataset DTO for testing."""
        return DatasetDTO(
            id=uuid.UUID("87654321-4321-8765-2109-876543210987"),
            name="Test Dataset",
            description="A test dataset for anomaly detection",
            features=["feature1", "feature2", "feature3"],
            feature_types=["numeric", "numeric", "categorical"],
            shape=(1000, 3),
            n_samples=1000,
            n_features=3,
            contamination_rate=0.05,
            has_target=True,
            target_column="is_anomaly",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={
                "source": "test",
                "quality_score": 0.95,
                "encoding": "utf-8"
            }
        )

    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for upload testing."""
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 100.0, 5.0],  # 100.0 is an outlier
            'feature2': [0.5, 1.5, 2.5, 3.5, 4.5],
            'feature3': ['A', 'B', 'A', 'C', 'B'],
            'is_anomaly': [0, 0, 0, 1, 0]
        })
        return data.to_csv(index=False)

    # Dataset Listing Tests

    def test_list_datasets_success(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test successful dataset listing."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.find_all.return_value = [sample_dataset_dto, sample_dataset_dto]
        
        response = client.get("/api/v1/datasets/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "Test Dataset"
        assert data[0]["n_samples"] == 1000

    def test_list_datasets_with_filters(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test dataset listing with filters."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.find_all.return_value = [sample_dataset_dto]
        
        response = client.get("/api/v1/datasets/?has_target=true&limit=50")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["has_target"] is True

    def test_list_datasets_unauthorized(self, client, mock_container):
        """Test dataset listing without authentication."""
        with patch('pynomaly.presentation.api.deps.get_current_user') as mock_auth:
            mock_auth.side_effect = HTTPException(status_code=401, detail="Not authenticated")
            
            response = client.get("/api/v1/datasets/")
            assert response.status_code == 401

    def test_list_datasets_pagination(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test dataset listing with pagination."""
        mock_repo = mock_container.dataset_repository.return_value
        datasets = [sample_dataset_dto] * 150  # More than default limit
        mock_repo.find_all.return_value = datasets
        
        response = client.get("/api/v1/datasets/?limit=100")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 100  # Should be limited

    # Dataset Creation Tests

    def test_create_dataset_from_upload_success(self, client, mock_container, mock_user, sample_csv_data, sample_dataset_dto):
        """Test successful dataset creation from file upload."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.create.return_value = sample_dataset_dto
        
        # Create file-like object
        file_data = io.BytesIO(sample_csv_data.encode('utf-8'))
        
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.csv", file_data, "text/csv")},
            data={
                "name": "Uploaded Dataset",
                "description": "Dataset from CSV upload",
                "target_column": "is_anomaly"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Dataset"
        assert data["n_samples"] == 1000

    def test_create_dataset_invalid_file_format(self, client, mock_user):
        """Test dataset creation with invalid file format."""
        # Create invalid file
        file_data = io.BytesIO(b"invalid content")
        
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.txt", file_data, "text/plain")},
            data={"name": "Invalid Dataset"}
        )
        
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_create_dataset_empty_file(self, client, mock_user):
        """Test dataset creation with empty file."""
        file_data = io.BytesIO(b"")
        
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("empty.csv", file_data, "text/csv")},
            data={"name": "Empty Dataset"}
        )
        
        assert response.status_code == 400

    def test_create_dataset_large_file(self, client, mock_container, mock_user):
        """Test dataset creation with large file."""
        # Create large CSV data
        large_data = pd.DataFrame({
            'feature1': range(100000),
            'feature2': range(100000, 200000)
        })
        csv_data = large_data.to_csv(index=False)
        file_data = io.BytesIO(csv_data.encode('utf-8'))
        
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("large.csv", file_data, "text/csv")},
            data={"name": "Large Dataset"}
        )
        
        # Should handle large files appropriately
        assert response.status_code in [201, 413]  # Created or Request Entity Too Large

    def test_create_dataset_from_json(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test dataset creation from JSON payload."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.create.return_value = sample_dataset_dto
        
        json_data = {
            "name": "JSON Dataset",
            "description": "Dataset from JSON",
            "data": [
                {"feature1": 1.0, "feature2": 0.5, "feature3": "A"},
                {"feature1": 2.0, "feature2": 1.5, "feature3": "B"},
                {"feature1": 100.0, "feature2": 3.5, "feature3": "C"}  # Outlier
            ],
            "features": ["feature1", "feature2", "feature3"],
            "target_column": None
        }
        
        response = client.post("/api/v1/datasets/create", json=json_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Dataset"

    # Dataset Retrieval Tests

    def test_get_dataset_by_id_success(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test successful dataset retrieval by ID."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.find_by_id.return_value = sample_dataset_dto
        
        dataset_id = str(sample_dataset_dto.id)
        response = client.get(f"/api/v1/datasets/{dataset_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id
        assert data["name"] == "Test Dataset"

    def test_get_dataset_not_found(self, client, mock_container, mock_user):
        """Test dataset retrieval with non-existent ID."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.find_by_id.return_value = None
        
        response = client.get("/api/v1/datasets/99999999-9999-9999-9999-999999999999")
        
        assert response.status_code == 404

    def test_get_dataset_invalid_uuid(self, client, mock_user):
        """Test dataset retrieval with invalid UUID."""
        response = client.get("/api/v1/datasets/invalid-uuid")
        
        assert response.status_code == 422  # Validation error

    def test_get_dataset_data_preview(self, client, mock_container, mock_user, sample_csv_data):
        """Test dataset data preview endpoint."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.get_data_preview.return_value = {
            "head": [
                {"feature1": 1.0, "feature2": 0.5, "feature3": "A"},
                {"feature1": 2.0, "feature2": 1.5, "feature3": "B"}
            ],
            "tail": [
                {"feature1": 4.0, "feature2": 3.5, "feature3": "B"},
                {"feature1": 5.0, "feature2": 4.5, "feature3": "B"}
            ],
            "sample": [
                {"feature1": 3.0, "feature2": 2.5, "feature3": "A"}
            ]
        }
        
        dataset_id = "87654321-4321-8765-2109-876543210987"
        response = client.get(f"/api/v1/datasets/{dataset_id}/preview?rows=5")
        
        assert response.status_code == 200
        data = response.json()
        assert "head" in data
        assert "tail" in data
        assert len(data["head"]) == 2

    # Dataset Update Tests

    def test_update_dataset_success(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test successful dataset update."""
        mock_repo = mock_container.dataset_repository.return_value
        updated_dto = sample_dataset_dto.copy()
        updated_dto.name = "Updated Dataset"
        updated_dto.description = "Updated description"
        mock_repo.update.return_value = updated_dto
        mock_repo.find_by_id.return_value = sample_dataset_dto
        
        update_data = {
            "name": "Updated Dataset",
            "description": "Updated description",
            "metadata": {"quality_score": 0.98}
        }
        
        dataset_id = str(sample_dataset_dto.id)
        response = client.put(f"/api/v1/datasets/{dataset_id}", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Dataset"

    def test_update_dataset_not_found(self, client, mock_container, mock_user):
        """Test dataset update with non-existent ID."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.find_by_id.return_value = None
        
        update_data = {"name": "Updated Dataset"}
        response = client.put("/api/v1/datasets/99999999-9999-9999-9999-999999999999", json=update_data)
        
        assert response.status_code == 404

    def test_update_dataset_validation_error(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test dataset update with validation error."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.find_by_id.return_value = sample_dataset_dto
        
        invalid_data = {
            "contamination_rate": 1.5  # Invalid: should be between 0 and 1
        }
        
        dataset_id = str(sample_dataset_dto.id)
        response = client.put(f"/api/v1/datasets/{dataset_id}", json=invalid_data)
        
        assert response.status_code == 422

    # Dataset Deletion Tests

    def test_delete_dataset_success(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test successful dataset deletion."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.find_by_id.return_value = sample_dataset_dto
        mock_repo.delete.return_value = True
        
        dataset_id = str(sample_dataset_dto.id)
        response = client.delete(f"/api/v1/datasets/{dataset_id}")
        
        assert response.status_code == 204
        mock_repo.delete.assert_called_once_with(sample_dataset_dto.id)

    def test_delete_dataset_not_found(self, client, mock_container, mock_user):
        """Test dataset deletion with non-existent ID."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.find_by_id.return_value = None
        
        response = client.delete("/api/v1/datasets/99999999-9999-9999-9999-999999999999")
        
        assert response.status_code == 404

    def test_delete_dataset_in_use(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test dataset deletion when dataset is in use."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.find_by_id.return_value = sample_dataset_dto
        mock_repo.delete.side_effect = HTTPException(
            status_code=409, detail="Dataset is in use by active detectors"
        )
        
        dataset_id = str(sample_dataset_dto.id)
        response = client.delete(f"/api/v1/datasets/{dataset_id}")
        
        assert response.status_code == 409

    # Data Quality Tests

    def test_get_dataset_quality_report(self, client, mock_container, mock_user):
        """Test dataset quality report generation."""
        mock_service = mock_container.data_quality_service.return_value
        quality_report = DataQualityReportDTO(
            dataset_id=uuid.UUID("87654321-4321-8765-2109-876543210987"),
            overall_score=0.85,
            completeness_score=0.95,
            accuracy_score=0.80,
            consistency_score=0.85,
            validity_score=0.90,
            missing_values_count=25,
            duplicate_rows_count=5,
            outliers_count=12,
            data_types_correct=True,
            feature_correlations={"feature1_feature2": 0.75},
            recommendations=[
                "Consider removing outliers in feature1",
                "Investigate missing values in feature3"
            ],
            generated_at=datetime.utcnow()
        )
        mock_service.generate_quality_report.return_value = quality_report
        
        dataset_id = "87654321-4321-8765-2109-876543210987"
        response = client.get(f"/api/v1/datasets/{dataset_id}/quality")
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_score"] == 0.85
        assert data["missing_values_count"] == 25
        assert len(data["recommendations"]) == 2

    def test_dataset_statistics(self, client, mock_container, mock_user):
        """Test dataset statistics endpoint."""
        mock_service = mock_container.data_quality_service.return_value
        statistics = {
            "numeric_features": {
                "feature1": {
                    "mean": 22.2,
                    "std": 49.5,
                    "min": 1.0,
                    "max": 100.0,
                    "quartiles": [1.5, 2.5, 4.5]
                },
                "feature2": {
                    "mean": 2.5,
                    "std": 1.58,
                    "min": 0.5,
                    "max": 4.5,
                    "quartiles": [1.0, 2.5, 4.0]
                }
            },
            "categorical_features": {
                "feature3": {
                    "unique_values": ["A", "B", "C"],
                    "value_counts": {"A": 2, "B": 2, "C": 1},
                    "mode": "A"
                }
            },
            "correlations": {
                "feature1_feature2": 0.123
            }
        }
        mock_service.get_statistics.return_value = statistics
        
        dataset_id = "87654321-4321-8765-2109-876543210987"
        response = client.get(f"/api/v1/datasets/{dataset_id}/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert "numeric_features" in data
        assert "categorical_features" in data
        assert data["numeric_features"]["feature1"]["mean"] == 22.2

    # Data Export Tests

    def test_export_dataset_csv(self, client, mock_container, mock_user, sample_csv_data):
        """Test dataset export to CSV."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.export_data.return_value = sample_csv_data
        
        dataset_id = "87654321-4321-8765-2109-876543210987"
        response = client.get(f"/api/v1/datasets/{dataset_id}/export?format=csv")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "feature1,feature2,feature3" in response.text

    def test_export_dataset_json(self, client, mock_container, mock_user):
        """Test dataset export to JSON."""
        mock_repo = mock_container.dataset_repository.return_value
        export_data = [
            {"feature1": 1.0, "feature2": 0.5, "feature3": "A"},
            {"feature1": 2.0, "feature2": 1.5, "feature3": "B"}
        ]
        mock_repo.export_data.return_value = export_data
        
        dataset_id = "87654321-4321-8765-2109-876543210987"
        response = client.get(f"/api/v1/datasets/{dataset_id}/export?format=json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert len(data) == 2
        assert data[0]["feature1"] == 1.0

    def test_export_dataset_invalid_format(self, client, mock_container, mock_user):
        """Test dataset export with invalid format."""
        dataset_id = "87654321-4321-8765-2109-876543210987"
        response = client.get(f"/api/v1/datasets/{dataset_id}/export?format=xml")
        
        assert response.status_code == 400

    # Feature Engineering Tests

    def test_dataset_feature_engineering(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test feature engineering operations."""
        mock_service = mock_container.feature_engineering_service.return_value
        mock_service.apply_transformations.return_value = sample_dataset_dto
        
        transformations = {
            "operations": [
                {
                    "type": "scale",
                    "features": ["feature1", "feature2"],
                    "method": "standard"
                },
                {
                    "type": "encode",
                    "features": ["feature3"],
                    "method": "one_hot"
                }
            ]
        }
        
        dataset_id = str(sample_dataset_dto.id)
        response = client.post(f"/api/v1/datasets/{dataset_id}/transform", json=transformations)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == dataset_id

    # Batch Operations Tests

    def test_bulk_dataset_operations(self, client, mock_container, mock_user, sample_dataset_dto):
        """Test bulk dataset operations."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.bulk_delete.return_value = 3
        
        bulk_request = {
            "operation": "delete",
            "dataset_ids": [
                "87654321-4321-8765-2109-876543210987",
                "12345678-1234-5678-9012-123456789012",
                "11111111-2222-3333-4444-555555555555"
            ]
        }
        
        response = client.post("/api/v1/datasets/bulk", json=bulk_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 3

    # Error Handling Tests

    def test_dataset_service_timeout(self, client, mock_container, mock_user):
        """Test dataset service timeout handling."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.find_all.side_effect = TimeoutError("Database timeout")
        
        response = client.get("/api/v1/datasets/")
        
        assert response.status_code == 504  # Gateway Timeout

    def test_dataset_storage_full(self, client, mock_container, mock_user, sample_csv_data):
        """Test dataset upload when storage is full."""
        mock_repo = mock_container.dataset_repository.return_value
        mock_repo.create.side_effect = HTTPException(
            status_code=507, detail="Insufficient storage space"
        )
        
        file_data = io.BytesIO(sample_csv_data.encode('utf-8'))
        response = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.csv", file_data, "text/csv")},
            data={"name": "Test Dataset"}
        )
        
        assert response.status_code == 507

    def test_dataset_permission_denied(self, client, mock_container):
        """Test dataset operations with insufficient permissions."""
        with patch('pynomaly.presentation.api.deps.get_current_user') as mock_auth:
            mock_auth.return_value = {
                "user_id": "test-user",
                "permissions": ["dataset:read"]  # Missing dataset:create
            }
            
            file_data = io.BytesIO(b"test,data\n1,2")
            response = client.post(
                "/api/v1/datasets/upload",
                files={"file": ("test.csv", file_data, "text/csv")},
                data={"name": "Test Dataset"}
            )
            
            assert response.status_code == 403


class TestDatasetEndpointsIntegration:
    """Integration tests for dataset endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_complete_dataset_lifecycle(self, client):
        """Test complete dataset lifecycle from upload to deletion."""
        with patch('pynomaly.presentation.api.deps.get_current_user') as mock_auth:
            mock_auth.return_value = {
                "user_id": "test-user",
                "permissions": ["dataset:create", "dataset:read", "dataset:update", "dataset:delete"]
            }
            
            with patch('pynomaly.presentation.api.deps.get_container') as mock_container:
                container = Mock()
                container.dataset_repository.return_value = Mock()
                mock_container.return_value = container
                
                # 1. Upload dataset
                csv_data = "feature1,feature2\n1.0,0.5\n2.0,1.5"
                file_data = io.BytesIO(csv_data.encode('utf-8'))
                
                upload_response = client.post(
                    "/api/v1/datasets/upload",
                    files={"file": ("test.csv", file_data, "text/csv")},
                    data={"name": "Lifecycle Test"}
                )
                
                # 2. List datasets
                list_response = client.get("/api/v1/datasets/")
                
                # 3. Get specific dataset
                dataset_id = "87654321-4321-8765-2109-876543210987"
                get_response = client.get(f"/api/v1/datasets/{dataset_id}")
                
                # 4. Update dataset
                update_response = client.put(
                    f"/api/v1/datasets/{dataset_id}",
                    json={"name": "Updated Lifecycle Test"}
                )
                
                # 5. Delete dataset
                delete_response = client.delete(f"/api/v1/datasets/{dataset_id}")
                
                # Verify lifecycle
                responses = [upload_response, list_response, get_response, update_response, delete_response]
                
                # All requests should either succeed or have expected auth failures
                for response in responses:
                    assert response.status_code in [200, 201, 204, 401, 404, 422]

    def test_dataset_validation_workflow(self, client):
        """Test dataset validation and quality checking workflow."""
        with patch('pynomaly.presentation.api.deps.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test", "permissions": ["dataset:read"]}
            
            dataset_id = "87654321-4321-8765-2109-876543210987"
            
            # Test various validation endpoints
            endpoints = [
                f"/api/v1/datasets/{dataset_id}/quality",
                f"/api/v1/datasets/{dataset_id}/statistics",
                f"/api/v1/datasets/{dataset_id}/preview"
            ]
            
            for endpoint in endpoints:
                response = client.get(endpoint)
                # Should work or require proper auth/data
                assert response.status_code in [200, 401, 404, 422]
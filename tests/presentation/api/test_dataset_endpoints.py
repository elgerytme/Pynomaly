"""
Dataset Endpoints Testing Suite
Tests for dataset CRUD operations, upload, validation, and management endpoints.
"""

import io
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from pynomaly.application.dto.dataset_dto import (
    DatasetResponseDTO,
    DatasetUploadResponseDTO,
)
from pynomaly.presentation.api.app import create_app


class TestDatasetEndpoints:
    """Test suite for dataset management endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_auth(self):
        """Mock authentication."""
        with patch("pynomaly.infrastructure.auth.jwt_auth.JWTAuthHandler") as mock:
            handler = Mock()
            handler.get_current_user.return_value = {
                "id": "user123",
                "email": "test@example.com",
                "role": "user",
                "permissions": ["read:datasets", "write:datasets"],
            }
            mock.return_value = handler
            yield handler

    @pytest.fixture
    def auth_headers(self):
        """Authentication headers."""
        return {"Authorization": "Bearer test-jwt-token"}

    @pytest.fixture
    def sample_dataset_request(self):
        """Sample dataset request data."""
        return {
            "name": "test_dataset",
            "description": "Test dataset for anomaly detection",
            "file_format": "csv",
            "separator": ",",
            "has_header": True,
            "target_column": "is_anomaly",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "metadata": {
                "source": "test_system",
                "collection_date": "2024-01-01",
                "domain": "network_security",
            },
        }

    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data for upload testing."""
        return """feature1,feature2,feature3,is_anomaly
1.0,2.0,3.0,0
1.5,2.5,3.5,0
10.0,20.0,30.0,1
2.0,3.0,4.0,0
15.0,25.0,35.0,1"""

    # Dataset CRUD Operations

    def test_create_dataset_successful(
        self, client, mock_auth, auth_headers, sample_dataset_request
    ):
        """Test successful dataset creation."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.CreateDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = DatasetResponseDTO(
                id="dataset123",
                name="test_dataset",
                description="Test dataset for anomaly detection",
                file_format="csv",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                owner_id="user123",
                size_bytes=1024,
                row_count=100,
                column_count=4,
                status="active",
            )
            mock_use_case.return_value = mock_instance

            response = client.post(
                "/api/datasets", json=sample_dataset_request, headers=auth_headers
            )

            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "dataset123"
            assert data["name"] == "test_dataset"
            assert data["status"] == "active"

    def test_create_dataset_validation_error(self, client, mock_auth, auth_headers):
        """Test dataset creation with validation errors."""
        invalid_request = {
            "name": "",  # Empty name should fail validation
            "file_format": "invalid_format",
        }

        response = client.post(
            "/api/datasets", json=invalid_request, headers=auth_headers
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_create_dataset_unauthorized(self, client):
        """Test dataset creation without authentication."""
        response = client.post("/api/datasets", json={})

        assert response.status_code == 401

    def test_get_dataset_by_id(self, client, mock_auth, auth_headers):
        """Test retrieving dataset by ID."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.GetDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = DatasetResponseDTO(
                id="dataset123",
                name="test_dataset",
                description="Test dataset",
                file_format="csv",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                owner_id="user123",
                size_bytes=1024,
                row_count=100,
                column_count=4,
                status="active",
            )
            mock_use_case.return_value = mock_instance

            response = client.get("/api/datasets/dataset123", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "dataset123"
            assert data["name"] == "test_dataset"

    def test_get_dataset_not_found(self, client, mock_auth, auth_headers):
        """Test retrieving non-existent dataset."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.GetDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.side_effect = ValueError("Dataset not found")
            mock_use_case.return_value = mock_instance

            response = client.get("/api/datasets/nonexistent", headers=auth_headers)

            assert response.status_code == 404

    def test_list_datasets(self, client, mock_auth, auth_headers):
        """Test listing user's datasets."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.ListDatasetsUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = [
                DatasetResponseDTO(
                    id="dataset1",
                    name="dataset_one",
                    description="First dataset",
                    file_format="csv",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    owner_id="user123",
                    size_bytes=1024,
                    row_count=100,
                    column_count=4,
                    status="active",
                ),
                DatasetResponseDTO(
                    id="dataset2",
                    name="dataset_two",
                    description="Second dataset",
                    file_format="parquet",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    owner_id="user123",
                    size_bytes=2048,
                    row_count=200,
                    column_count=5,
                    status="active",
                ),
            ]
            mock_use_case.return_value = mock_instance

            response = client.get("/api/datasets", headers=auth_headers)

            assert response.status_code == 200
            data = response.json()
            assert len(data["datasets"]) == 2
            assert data["datasets"][0]["name"] == "dataset_one"
            assert data["datasets"][1]["name"] == "dataset_two"

    def test_list_datasets_with_pagination(self, client, mock_auth, auth_headers):
        """Test listing datasets with pagination."""
        response = client.get("/api/datasets?page=1&size=10", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "pagination" in data
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["size"] == 10

    def test_update_dataset(self, client, mock_auth, auth_headers):
        """Test updating dataset metadata."""
        update_data = {
            "description": "Updated dataset description",
            "metadata": {"version": "2.0", "last_modified": "2024-01-15"},
        }

        with patch(
            "pynomaly.application.use_cases.dataset_use_case.UpdateDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = DatasetResponseDTO(
                id="dataset123",
                name="test_dataset",
                description="Updated dataset description",
                file_format="csv",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                owner_id="user123",
                size_bytes=1024,
                row_count=100,
                column_count=4,
                status="active",
            )
            mock_use_case.return_value = mock_instance

            response = client.put(
                "/api/datasets/dataset123", json=update_data, headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["description"] == "Updated dataset description"

    def test_delete_dataset(self, client, mock_auth, auth_headers):
        """Test deleting a dataset."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.DeleteDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = True
            mock_use_case.return_value = mock_instance

            response = client.delete("/api/datasets/dataset123", headers=auth_headers)

            assert response.status_code == 204

    def test_delete_dataset_not_found(self, client, mock_auth, auth_headers):
        """Test deleting non-existent dataset."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.DeleteDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.side_effect = ValueError("Dataset not found")
            mock_use_case.return_value = mock_instance

            response = client.delete("/api/datasets/nonexistent", headers=auth_headers)

            assert response.status_code == 404

    # Dataset Upload Operations

    def test_upload_dataset_csv(self, client, mock_auth, auth_headers, sample_csv_data):
        """Test uploading CSV dataset."""
        files = {"file": ("test_data.csv", io.StringIO(sample_csv_data), "text/csv")}
        data = {
            "name": "uploaded_dataset",
            "description": "Dataset uploaded via API",
            "has_header": "true",
        }

        with patch(
            "pynomaly.application.use_cases.dataset_use_case.UploadDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = DatasetUploadResponseDTO(
                id="upload123",
                name="uploaded_dataset",
                status="processing",
                file_size=len(sample_csv_data),
                upload_url="http://example.com/upload/123",
            )
            mock_use_case.return_value = mock_instance

            response = client.post(
                "/api/datasets/upload", files=files, data=data, headers=auth_headers
            )

            assert response.status_code == 202
            result = response.json()
            assert result["id"] == "upload123"
            assert result["status"] == "processing"

    def test_upload_dataset_invalid_format(self, client, mock_auth, auth_headers):
        """Test uploading dataset with invalid format."""
        files = {"file": ("test_data.txt", io.StringIO("invalid data"), "text/plain")}

        response = client.post(
            "/api/datasets/upload", files=files, headers=auth_headers
        )

        assert response.status_code == 400
        data = response.json()
        assert "Unsupported file format" in data["detail"]

    def test_upload_dataset_too_large(self, client, mock_auth, auth_headers):
        """Test uploading dataset that exceeds size limit."""
        large_data = "x" * (100 * 1024 * 1024)  # 100MB
        files = {"file": ("large_data.csv", io.StringIO(large_data), "text/csv")}

        response = client.post(
            "/api/datasets/upload", files=files, headers=auth_headers
        )

        assert response.status_code == 413
        data = response.json()
        assert "File too large" in data["detail"]

    def test_upload_status_check(self, client, mock_auth, auth_headers):
        """Test checking upload status."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.GetUploadStatusUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = {
                "id": "upload123",
                "status": "completed",
                "progress": 100,
                "dataset_id": "dataset456",
                "error_message": None,
            }
            mock_use_case.return_value = mock_instance

            response = client.get(
                "/api/datasets/upload/upload123/status", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["dataset_id"] == "dataset456"

    # Dataset Validation Operations

    def test_validate_dataset(self, client, mock_auth, auth_headers):
        """Test dataset validation."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.ValidateDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = {
                "is_valid": True,
                "validation_results": {
                    "schema_valid": True,
                    "data_quality_score": 0.95,
                    "missing_values": 0.02,
                    "outlier_percentage": 0.05,
                    "column_types": {
                        "feature1": "float64",
                        "feature2": "float64",
                        "feature3": "float64",
                        "is_anomaly": "int64",
                    },
                },
                "recommendations": [
                    "Consider handling missing values in feature1",
                    "Review outliers in feature3",
                ],
            }
            mock_use_case.return_value = mock_instance

            response = client.post(
                "/api/datasets/dataset123/validate", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["is_valid"] is True
            assert data["validation_results"]["data_quality_score"] == 0.95

    def test_validate_dataset_with_errors(self, client, mock_auth, auth_headers):
        """Test dataset validation with errors."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.ValidateDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = {
                "is_valid": False,
                "validation_results": {
                    "schema_valid": False,
                    "errors": [
                        "Missing required column: target",
                        "Invalid data type in feature1",
                    ],
                },
            }
            mock_use_case.return_value = mock_instance

            response = client.post(
                "/api/datasets/dataset123/validate", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["is_valid"] is False
            assert len(data["validation_results"]["errors"]) == 2

    # Dataset Preview and Statistics

    def test_dataset_preview(self, client, mock_auth, auth_headers):
        """Test getting dataset preview."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.GetDatasetPreviewUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = {
                "columns": ["feature1", "feature2", "feature3", "is_anomaly"],
                "data": [[1.0, 2.0, 3.0, 0], [1.5, 2.5, 3.5, 0], [10.0, 20.0, 30.0, 1]],
                "total_rows": 100,
                "preview_rows": 3,
            }
            mock_use_case.return_value = mock_instance

            response = client.get(
                "/api/datasets/dataset123/preview", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["columns"]) == 4
            assert len(data["data"]) == 3
            assert data["total_rows"] == 100

    def test_dataset_statistics(self, client, mock_auth, auth_headers):
        """Test getting dataset statistics."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.GetDatasetStatisticsUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = {
                "row_count": 1000,
                "column_count": 4,
                "missing_values": 15,
                "data_types": {
                    "feature1": "float64",
                    "feature2": "float64",
                    "feature3": "float64",
                    "is_anomaly": "int64",
                },
                "summary_statistics": {
                    "feature1": {"mean": 2.5, "std": 1.2, "min": 0.1, "max": 15.0},
                    "feature2": {"mean": 3.5, "std": 1.8, "min": 0.2, "max": 25.0},
                },
                "anomaly_distribution": {"normal": 950, "anomalous": 50},
            }
            mock_use_case.return_value = mock_instance

            response = client.get(
                "/api/datasets/dataset123/statistics", headers=auth_headers
            )

            assert response.status_code == 200
            data = response.json()
            assert data["row_count"] == 1000
            assert data["column_count"] == 4
            assert "summary_statistics" in data

    # Dataset Export Operations

    def test_export_dataset(self, client, mock_auth, auth_headers):
        """Test exporting dataset."""
        export_params = {
            "format": "csv",
            "include_headers": True,
            "filter_anomalies": False,
        }

        response = client.post(
            "/api/datasets/dataset123/export", json=export_params, headers=auth_headers
        )

        assert response.status_code == 202
        data = response.json()
        assert "export_id" in data
        assert "download_url" in data

    def test_download_dataset(self, client, mock_auth, auth_headers):
        """Test downloading exported dataset."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.DownloadDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = {
                "file_path": "/tmp/dataset123.csv",
                "content_type": "text/csv",
                "filename": "test_dataset.csv",
            }
            mock_use_case.return_value = mock_instance

            response = client.get(
                "/api/datasets/dataset123/download", headers=auth_headers
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/csv"

    # Dataset Sharing and Permissions

    def test_share_dataset(self, client, mock_auth, auth_headers):
        """Test sharing dataset with other users."""
        share_data = {
            "user_emails": ["colleague@example.com"],
            "permissions": ["read"],
            "expiry_date": "2024-12-31",
        }

        response = client.post(
            "/api/datasets/dataset123/share", json=share_data, headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "share_id" in data

    def test_get_dataset_permissions(self, client, mock_auth, auth_headers):
        """Test getting dataset permissions."""
        response = client.get(
            "/api/datasets/dataset123/permissions", headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "owner" in data
        assert "shared_with" in data

    # Error Handling Tests

    def test_dataset_access_forbidden(self, client, mock_auth, auth_headers):
        """Test accessing dataset without proper permissions."""
        mock_auth.get_current_user.return_value = {
            "id": "otheruser",
            "email": "other@example.com",
            "role": "user",
            "permissions": ["read:own_datasets"],
        }

        response = client.get("/api/datasets/dataset123", headers=auth_headers)

        assert response.status_code == 403

    def test_dataset_operations_with_processing_status(
        self, client, mock_auth, auth_headers
    ):
        """Test operations on dataset that is still processing."""
        with patch(
            "pynomaly.application.use_cases.dataset_use_case.GetDatasetUseCase"
        ) as mock_use_case:
            mock_instance = Mock()
            mock_instance.execute.return_value = DatasetResponseDTO(
                id="dataset123",
                name="processing_dataset",
                description="Dataset still processing",
                file_format="csv",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                owner_id="user123",
                size_bytes=1024,
                row_count=0,
                column_count=0,
                status="processing",
            )
            mock_use_case.return_value = mock_instance

            response = client.get(
                "/api/datasets/dataset123/preview", headers=auth_headers
            )

            assert response.status_code == 409  # Conflict - dataset not ready


class TestDatasetEndpointsIntegration:
    """Integration tests for dataset endpoints with realistic workflows."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def authenticated_client(self, client):
        """Client with authentication setup."""
        with patch("pynomaly.infrastructure.auth.jwt_auth.JWTAuthHandler") as mock:
            handler = Mock()
            handler.get_current_user.return_value = {
                "id": "user123",
                "email": "test@example.com",
                "role": "user",
                "permissions": ["read:datasets", "write:datasets", "delete:datasets"],
            }
            mock.return_value = handler

            client.headers.update({"Authorization": "Bearer test-token"})
            yield client

    def test_complete_dataset_lifecycle(self, authenticated_client, sample_csv_data):
        """Test complete dataset lifecycle from upload to deletion."""
        # 1. Upload dataset
        files = {"file": ("test.csv", io.StringIO(sample_csv_data), "text/csv")}
        data = {"name": "lifecycle_test", "description": "Lifecycle test dataset"}

        upload_response = authenticated_client.post(
            "/api/datasets/upload", files=files, data=data
        )
        assert upload_response.status_code == 202

        # 2. Check upload status (simulate polling)
        upload_id = upload_response.json()["id"]
        status_response = authenticated_client.get(
            f"/api/datasets/upload/{upload_id}/status"
        )
        assert status_response.status_code == 200

        # 3. Get dataset (assume upload completed)
        dataset_id = "dataset123"  # Simulated completed dataset ID
        get_response = authenticated_client.get(f"/api/datasets/{dataset_id}")
        assert get_response.status_code in [200, 404]

        # 4. Update dataset metadata
        update_data = {"description": "Updated description"}
        update_response = authenticated_client.put(
            f"/api/datasets/{dataset_id}", json=update_data
        )
        assert update_response.status_code in [200, 404]

        # 5. Get statistics and preview
        authenticated_client.get(f"/api/datasets/{dataset_id}/statistics")
        authenticated_client.get(f"/api/datasets/{dataset_id}/preview")

        # 6. Export dataset
        export_data = {"format": "csv"}
        authenticated_client.post(
            f"/api/datasets/{dataset_id}/export", json=export_data
        )

        # 7. Delete dataset
        delete_response = authenticated_client.delete(f"/api/datasets/{dataset_id}")
        assert delete_response.status_code in [204, 404]

    def test_dataset_permission_workflow(self, authenticated_client):
        """Test dataset sharing and permission management workflow."""
        dataset_id = "dataset123"

        # 1. Check initial permissions
        permissions_response = authenticated_client.get(
            f"/api/datasets/{dataset_id}/permissions"
        )
        assert permissions_response.status_code in [200, 404]

        # 2. Share dataset
        share_data = {
            "user_emails": ["colleague@example.com"],
            "permissions": ["read"],
            "expiry_date": "2024-12-31",
        }
        share_response = authenticated_client.post(
            f"/api/datasets/{dataset_id}/share", json=share_data
        )
        assert share_response.status_code in [200, 404]

        # 3. Check updated permissions
        updated_permissions = authenticated_client.get(
            f"/api/datasets/{dataset_id}/permissions"
        )
        assert updated_permissions.status_code in [200, 404]
